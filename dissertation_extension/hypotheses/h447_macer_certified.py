"""
H447 - MACER attack-free certified training (gap G2).

Anchor: Zhai et al., "MACER: Attack-free and Scalable Robust Training via
Maximizing Certified Radius", ICLR 2020 (arXiv:2001.02378).

Critique (orientation, not a verdict)
-------------------------------------
MACER trains a randomized-smoothing classifier without any inner-PGD inner
loop. Instead of adversarial examples, K Gaussian noise samples per input
are pushed through the base classifier; the loss has two parts:

  L = CE_loss(soft_logits_under_noise, y)
    + lambda * RobustnessLoss(R_cert, gamma)

where R_cert = (sigma/2)*(Phi^-1(p_top) - Phi^-1(p_runner_up)) is the
soft certified L2 radius and RobustnessLoss is a hinge-style penalty on
samples whose R_cert < gamma (Eq. 6 of Zhai et al.):

  RobustLoss = (1/n) sum_{correct} max(gamma - R_cert, 0)

Cohen et al. RS (h268, h204) provided certification but only trained with
Gaussian augmentation. SmoothAdv (h446 sibling, Salman 2019) injects PGD
into the smoothed classifier — strong, but slow. MACER's selling point is
"same robust acc cheaper" because there is no inner PGD; just a hinge on
the certified-radius surrogate. Risk: at small N_train=6000 and only 10
epochs, the hinge may be dominated by CE (under-trained smoothed
classifier); also Phi^-1 saturates at p=1 so the radius signal goes silent
once samples are easy under noise.

Additional prior art consulted via WebSearch
-------------------------------------------
- Jeong & Shin (NeurIPS 2020) — Consistency regularization for smoothed
  classifiers; alternative attack-free trainer matching/beating MACER.
- Yang, Duan, Hu, Salman, Razenshteyn, Li (2020) — "Randomized Smoothing
  of All Shapes and Sizes" — generalises smoothing distributions.
- Sukenik, Kuvshinov, Gunnemann (2021/22) — Curse-of-dimensionality /
  truncation effect; certified radius shrinks fast with dim and trivial
  classifier threat at high sigma. Tempers expectations even if MACER
  beats Cohen.

What this script tests on Fashion-MNIST (28x28, low dim, fits campaign)
---------------------------------------------------------------------
Compute-matched comparison at fixed train budget:

  A. Cohen RS baseline:        Gaussian-aug training, sigma=0.25.
  B. SmoothAdv (compute-matched): PGD-on-smoothed inner-max, sigma=0.25.
  C. MACER, lambda in {1, 6, 12}: CE_under_noise + lambda * hinge(gamma - R).

For each model:
  - clean acc (raw classifier),
  - smoothed accuracy at sigma=0.25 (Monte-Carlo, K_eval samples),
  - certified L2 radius (Cohen Lemma 2 abstain rule, n=K_eval samples,
    alpha=0.001 one-sided Clopper-Pearson lower bound on p_top),
  - certified accuracy at L2 thresholds r in {0.0, 0.25, 0.5, 0.75, 1.0},
  - empirical PGD-Linf ASR at EPS=0.1 (against the *base* classifier;
    standard campaign attack — checks if certified L2 buys empirical Linf).

VERDICT rule (set before run, falsifiable):
  MACER (best lambda) BEATS Cohen on certified acc @ r=0.5  AND  matches
  or beats SmoothAdv at strictly lower wall-clock training cost.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD mom=0.9 wd=5e-4,
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01. ASCII only.
"""
import os
import sys
import time
import math

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

try:
    from scipy.stats import norm as scipy_norm
    from scipy.stats import beta as scipy_beta
except Exception:
    scipy_norm = None
    scipy_beta = None

# ---- config --------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 500              # certification is K_eval-heavy; keep modest
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1                 # campaign Linf eps for empirical ASR
PGD_STEPS = 10
PGD_ALPHA = 0.01

# smoothing
SIGMA = 0.25              # Cohen / MACER standard mid-range sigma
K_TRAIN = 2               # noise samples per input during training (memory-light)
K_EVAL = 100              # MC samples for certification at eval
CERT_ALPHA = 0.001        # one-sided confidence for p_lower
CERT_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0]

# MACER-specific
MACER_LAMBDAS = [1.0, 6.0, 12.0]
MACER_GAMMA = 8.0 * SIGMA     # hinge threshold on radius (paper uses ~2*sigma * scale)
MACER_BETA = 16.0             # temperature for soft prediction
                              # (paper Sec. 4.1: "scaling factor beta")

# SmoothAdv compute-matched
SMOOTHADV_STEPS = 2       # cheap inner-PGD on smoothed classifier
SMOOTHADV_EPS_L2 = 0.5    # L2 attack radius (matches sigma * 2)

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h447_macer_certified_output.txt")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- logger --------------------------------------------------------------
_lines = []
def log(s=""):
    s = str(s)
    print(s, flush=True)
    _lines.append(s)
    # flush to disk after every line so partial output survives a crash
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(_lines) + "\n")


# ---- helpers -------------------------------------------------------------
def _opt_sgd(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def _new_model():
    return C.build_model("cnn", META, width=32, act="relu", bn=True)


# ---- training: Cohen RS (Gaussian augmentation) --------------------------
def train_cohen(Xtr, Ytr, sigma, epochs, batch, lr):
    model = _new_model()
    opt = _opt_sgd(model, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            xn = (xb + sigma * torch.randn_like(xb))  # NOTE: not clamped; matches RS protocol
            opt.zero_grad()
            loss = F.cross_entropy(model(xn), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- training: SmoothAdv (compute-matched PGD on smoothed classifier) ----
def _pgd_l2_on_smoothed(model, x, y, sigma, K, eps_l2, steps, alpha=None):
    """PGD with L2 ball constraint against the soft smoothed classifier."""
    if alpha is None:
        alpha = 2.0 * eps_l2 / max(1, steps)
    x0 = x.clone().detach()
    delta = torch.zeros_like(x0, requires_grad=True)
    for _ in range(steps):
        # soft smoothed logits via K noise samples
        xs = (x0 + delta).unsqueeze(1) + sigma * torch.randn(
            x0.size(0), K, *x0.shape[1:], device=x0.device)
        xs = xs.view(-1, *x0.shape[1:])
        logits = model(xs).view(x0.size(0), K, -1).mean(1)
        loss = F.cross_entropy(logits, y)
        g, = torch.autograd.grad(loss, delta)
        # normalise per-sample
        flat = g.view(g.size(0), -1)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        step = (flat / norms).view_as(g) * alpha
        delta = (delta + step).detach()
        # L2 project
        flat = delta.view(delta.size(0), -1)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        factor = torch.clamp(eps_l2 / norms, max=1.0)
        delta = (flat * factor).view_as(delta)
        # clamp to image domain
        delta = ((x0 + delta).clamp(0, 1) - x0).detach().requires_grad_(True)
    return (x0 + delta).detach()


def train_smoothadv(Xtr, Ytr, sigma, K, eps_l2, steps, epochs, batch, lr):
    model = _new_model()
    opt = _opt_sgd(model, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = _pgd_l2_on_smoothed(model, xb, yb, sigma, K, eps_l2, steps)
            xn = (xa + sigma * torch.randn_like(xa))
            opt.zero_grad()
            loss = F.cross_entropy(model(xn), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- training: MACER ------------------------------------------------------
def train_macer(Xtr, Ytr, sigma, K, lam, gamma, beta, epochs, batch, lr, ncls=10):
    """MACER: CE under noise + lambda * hinge(gamma - R_cert).

    R_cert(soft) = (sigma/2) * (Phi^-1(p_top) - Phi^-1(p_runner_up)),
    where p_c is soft-max probability of class c averaged over K noise samples.
    Hinge applied only to samples whose smoothed prediction is correct
    (per Zhai et al. Eq. 6).
    """
    model = _new_model()
    opt = _opt_sgd(model, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    inv = (math.sqrt(2.0))  # sigma/2 outside, beta inside
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            B = xb.size(0)
            # K noise copies, processed in a single big batch
            xs = xb.unsqueeze(1) + sigma * torch.randn(
                B, K, *xb.shape[1:], device=xb.device)
            xs = xs.view(B * K, *xb.shape[1:])
            logits = model(xs).view(B, K, ncls)
            # softmax over classes; average over noise samples => soft p
            probs = F.softmax(beta * logits, dim=-1).mean(1)   # (B, ncls)
            # standard CE on log of averaged probs (numerically safer than mean of log)
            ce = F.nll_loss(torch.log(probs.clamp_min(1e-12)), yb)

            # robustness loss only on smoothed-correct samples
            pred = probs.argmax(1)
            correct = (pred == yb)
            if correct.any():
                p_correct = probs[correct]
                yb_c = yb[correct]
                p_top = p_correct.gather(1, yb_c.view(-1, 1)).squeeze(1)
                # runner-up: max prob over wrong classes
                mask = torch.ones_like(p_correct)
                mask.scatter_(1, yb_c.view(-1, 1), 0.0)
                p_run = (p_correct * mask).max(1).values
                # avoid Phi^-1 saturation; clamp to (eps, 1-eps)
                eps_p = 1e-3
                p_top_c = p_top.clamp(eps_p, 1.0 - eps_p)
                p_run_c = p_run.clamp(eps_p, 1.0 - eps_p)
                # Phi^-1 implemented via torch.special.ndtri (or erfinv)
                # Phi^-1(p) = sqrt(2) * erfinv(2p - 1)
                ip_top = inv * torch.erfinv(2 * p_top_c - 1)
                ip_run = inv * torch.erfinv(2 * p_run_c - 1)
                radius = (sigma / 2.0) * (ip_top - ip_run)
                hinge = F.relu(gamma - radius)
                rloss = hinge.mean()
            else:
                rloss = torch.zeros((), device=xb.device)
            loss = ce + lam * rloss
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- evaluation: smoothed classifier + Cohen certification ---------------
def _clopper_pearson_lower(k, n, alpha):
    """One-sided lower bound on a binomial p at confidence 1-alpha."""
    if k == 0:
        return 0.0
    if scipy_beta is None:
        # fallback: Hoeffding lower bound
        return max(0.0, k / n - math.sqrt(math.log(1 / alpha) / (2 * n)))
    return float(scipy_beta.ppf(alpha, k, n - k + 1))


def _phi_inv(p):
    if scipy_norm is None:
        return math.sqrt(2.0) * math.erf(2 * p - 1)  # approx
    return float(scipy_norm.ppf(p))


@torch.no_grad()
def smoothed_eval_and_certify(model, X, Y, sigma, K, alpha, batch=64):
    """For each test sample, draw K noise samples, return (smoothed_pred,
    certified_radius). Cohen abstain rule: if p_lower<=0.5, radius=0 with
    smoothed_pred = top counted class but flagged as uncertified.
    """
    model.eval()
    N = X.size(0)
    counts = np.zeros((N, META["n_classes"]), dtype=int)
    # process by sample groups to control memory; each sample fires K noise draws
    sub = max(1, batch // max(1, K))
    for i in range(0, N, sub):
        x = X[i:i + sub]
        b = x.size(0)
        xs = x.unsqueeze(1) + sigma * torch.randn(b, K, *x.shape[1:], device=x.device)
        xs = xs.view(b * K, *x.shape[1:])
        preds = model(xs).argmax(1).view(b, K).cpu().numpy()
        for j in range(b):
            for p in preds[j]:
                counts[i + j, p] += 1
    smooth_pred = counts.argmax(1)
    top_counts = counts.max(1)
    radii = np.zeros(N, dtype=np.float64)
    certified = np.zeros(N, dtype=bool)
    for i in range(N):
        p_lower = _clopper_pearson_lower(int(top_counts[i]), K, alpha)
        if p_lower > 0.5:
            radii[i] = sigma * _phi_inv(p_lower)
            certified[i] = True
        else:
            radii[i] = 0.0
            certified[i] = False
    return smooth_pred, radii, certified


def certified_acc_at(smooth_pred, radii, certified, Y, threshold):
    Y = Y.cpu().numpy() if hasattr(Y, "cpu") else np.asarray(Y)
    ok = certified & (smooth_pred == Y) & (radii >= threshold)
    return float(ok.mean())


# ---- main ----------------------------------------------------------------
log("=" * 72)
log("H447 - MACER attack-free certified training")
log("=" * 72)
log(f"DS={DS}  N_TRAIN={N_TRAIN}  EPOCHS={EPOCHS}  LR={LR}  BATCH={BATCH}")
log(f"SIGMA={SIGMA}  K_TRAIN={K_TRAIN}  K_EVAL={K_EVAL}  cert_alpha={CERT_ALPHA}")
log(f"MACER lambdas={MACER_LAMBDAS}  gamma={MACER_GAMMA:.3f}  beta={MACER_BETA}")
log(f"SmoothAdv steps={SMOOTHADV_STEPS}  eps_L2={SMOOTHADV_EPS_L2}")
log(f"PGD-Linf empirical: EPS={EPS} steps={PGD_STEPS}")
log("")

C.set_seed(SEED)
Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
log(f"Loaded: Xtr={tuple(Xtr.shape)} Xte={tuple(Xte.shape)}")
log("")

results = {}  # name -> dict
order = []

def evaluate(name, model, train_secs):
    log(f"--- evaluating {name} ---")
    t0 = time.time()
    # clean acc of base classifier
    with torch.no_grad():
        clean_acc = float((model(Xte).argmax(1) == Yte).float().mean())
    # smoothed prediction + certification
    smooth_pred, radii, certd = smoothed_eval_and_certify(
        model, Xte, Yte, sigma=SIGMA, K=K_EVAL, alpha=CERT_ALPHA)
    Yte_np = Yte.cpu().numpy()
    smooth_acc = float((smooth_pred == Yte_np).mean())
    cert_at = {t: certified_acc_at(smooth_pred, radii, certd, Yte, t)
               for t in CERT_THRESHOLDS}
    # PGD-Linf empirical ASR on base classifier (campaign metric)
    pgd_res = C.attack_success(model, Xte, Yte, attack="pgd",
                               eps=EPS, steps=PGD_STEPS)
    pgd_asr = pgd_res["asr"]
    eval_secs = time.time() - t0
    rec = {
        "train_secs": train_secs,
        "eval_secs": eval_secs,
        "clean_acc": clean_acc,
        "smooth_acc": smooth_acc,
        "cert_at": cert_at,
        "pgd_linf_asr": pgd_asr,
        "n_certified": int(certd.sum()),
        "median_radius": float(np.median(radii[certd])) if certd.any() else 0.0,
    }
    log(f"  train_secs={train_secs:.1f}  eval_secs={eval_secs:.1f}")
    log(f"  clean_acc={clean_acc:.4f}  smooth_acc(sigma={SIGMA})={smooth_acc:.4f}")
    log(f"  n_certified={int(certd.sum())}/{N_EVAL}  "
        f"median_certified_radius={rec['median_radius']:.3f}")
    for t in CERT_THRESHOLDS:
        log(f"    certified_acc @ r>={t:.2f} : {cert_at[t]:.4f}")
    log(f"  PGD-Linf ASR (eps={EPS}) on base clf: {pgd_asr:.4f}")
    log("")
    results[name] = rec
    order.append(name)


# --- A. Cohen RS baseline -----------------------------------------------
log("[A] Training Cohen RS baseline (Gaussian-aug, sigma=0.25) ...")
C.set_seed(SEED)
t0 = time.time()
m_cohen = train_cohen(Xtr, Ytr, sigma=SIGMA, epochs=EPOCHS, batch=BATCH, lr=LR)
secs_cohen = time.time() - t0
log(f"  done in {secs_cohen:.1f}s")
evaluate("cohen_rs", m_cohen, secs_cohen)


# --- B. SmoothAdv compute-matched ---------------------------------------
log("[B] Training SmoothAdv (sigma=0.25, PGD-L2 inner) ...")
C.set_seed(SEED)
t0 = time.time()
m_smoothadv = train_smoothadv(
    Xtr, Ytr, sigma=SIGMA, K=K_TRAIN, eps_l2=SMOOTHADV_EPS_L2,
    steps=SMOOTHADV_STEPS, epochs=EPOCHS, batch=BATCH, lr=LR)
secs_smoothadv = time.time() - t0
log(f"  done in {secs_smoothadv:.1f}s")
evaluate("smoothadv", m_smoothadv, secs_smoothadv)


# --- C. MACER lambda sweep ----------------------------------------------
for lam in MACER_LAMBDAS:
    name = f"macer_lam{lam:g}"
    log(f"[C] Training MACER lambda={lam} ...")
    C.set_seed(SEED)
    t0 = time.time()
    m_macer = train_macer(
        Xtr, Ytr, sigma=SIGMA, K=K_TRAIN, lam=lam, gamma=MACER_GAMMA,
        beta=MACER_BETA, epochs=EPOCHS, batch=BATCH, lr=LR)
    secs = time.time() - t0
    log(f"  done in {secs:.1f}s")
    evaluate(name, m_macer, secs)


# ---- summary -------------------------------------------------------------
log("=" * 72)
log("SUMMARY")
log("=" * 72)
header = ("name", "tr_s", "clean", "smooth", "c@0", "c@.25", "c@.5", "c@.75",
          "c@1", "pgdLinf")
log("  " + " ".join(f"{h:>10s}" for h in header))
for name in order:
    r = results[name]
    row = (name, f"{r['train_secs']:.1f}", f"{r['clean_acc']:.3f}",
           f"{r['smooth_acc']:.3f}",
           f"{r['cert_at'][0.0]:.3f}",
           f"{r['cert_at'][0.25]:.3f}",
           f"{r['cert_at'][0.5]:.3f}",
           f"{r['cert_at'][0.75]:.3f}",
           f"{r['cert_at'][1.0]:.3f}",
           f"{r['pgd_linf_asr']:.3f}")
    log("  " + " ".join(f"{v:>10s}" for v in row))

# ---- verdict -------------------------------------------------------------
log("")
log("VERDICT RULE (set pre-run, falsifiable):")
log("  MACER(best lambda) certified_acc @ r=0.5 > Cohen certified_acc @ r=0.5")
log("  AND MACER(best lambda) train_secs < SmoothAdv train_secs")
log("  AND MACER(best lambda) certified_acc @ r=0.5 >= SmoothAdv certified_acc @ r=0.5 - 0.02")
log("")

cohen_c5 = results["cohen_rs"]["cert_at"][0.5]
sa_c5 = results["smoothadv"]["cert_at"][0.5]
sa_secs = results["smoothadv"]["train_secs"]
best_macer = max(MACER_LAMBDAS, key=lambda l: results[f"macer_lam{l:g}"]["cert_at"][0.5])
mb = results[f"macer_lam{best_macer:g}"]
mb_c5 = mb["cert_at"][0.5]
mb_secs = mb["train_secs"]

log(f"  best MACER lambda           = {best_macer:g}")
log(f"  Cohen     certified@0.5     = {cohen_c5:.4f}")
log(f"  SmoothAdv certified@0.5     = {sa_c5:.4f}  (train {sa_secs:.1f}s)")
log(f"  MACER*    certified@0.5     = {mb_c5:.4f}  (train {mb_secs:.1f}s)")

cond1 = mb_c5 > cohen_c5
cond2 = mb_secs < sa_secs
cond3 = mb_c5 >= sa_c5 - 0.02

log(f"  beats Cohen           : {cond1}")
log(f"  cheaper than SmoothAdv: {cond2}")
log(f"  matches SmoothAdv-0.02: {cond3}")

if cond1 and cond2 and cond3:
    verdict = "SUPPORTED: MACER delivers attack-free certified training " \
              "competitive with SmoothAdv at lower training cost on Fashion-MNIST."
elif cond1 and cond2:
    verdict = "PARTIAL: MACER beats Cohen and is cheaper than SmoothAdv, " \
              "but does not match SmoothAdv's certified accuracy."
elif cond1:
    verdict = "PARTIAL: MACER beats Cohen but is not compute-cheaper than SmoothAdv here."
else:
    verdict = "NOT SUPPORTED: MACER does not improve over Cohen RS at sigma=0.25, " \
              "lambda in {1,6,12}, on Fashion-MNIST at this budget."

log("")
log(f"VERDICT: {verdict}")
log("")
log(f"Output saved: {OUT_FILE}")
log("=" * 72)

if __name__ == "__main__":
    pass
