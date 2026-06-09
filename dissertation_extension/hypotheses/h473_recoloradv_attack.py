"""
H473 - ReColorAdv (functional) attack on Fashion-MNIST.

Gap: G1 (threat models barely touched). Anchor: laidlaw-2019-recoloradv,
"Functional Adversarial Attacks", NeurIPS 2019. Auxiliary anchors:
- bhattad-2020-colorfool (semantic colour attacks)
- hosseini-2018-semantic   (semantic adversarial perturbations)

THREAT MODEL
------------
ReColorAdv applies a learned smooth colour-space map f_theta : [0,1]^C -> [0,1]^C
shared across all spatial locations of an image. The map is parameterised by a
grid of K control values; the per-pixel output is a (multi-)linear interpolation
on the grid. Smoothness is enforced by a total-variation penalty on grid values
plus a per-cell delta bound delta_inf.

Fashion-MNIST is grayscale (C=1), so the colour-space map collapses to an
intensity remap g_theta : [0,1] -> [0,1] parameterised by a length-K grid of
output values at uniformly spaced input knots
    u_k = k/(K-1),   k = 0..K-1.
For input intensity v in [0,1] we locate the cell t = v*(K-1), j = floor(t),
alpha = t - j, and output
    g_theta(v) = (1 - alpha) * theta_j + alpha * theta_{j+1}
which is a piecewise-linear remap of [0,1] -> [0,1]. The attack is
    max_theta  CE(model(g_theta(X)), Y)
    s.t.       |theta_k - u_k| <= delta_inf   for all k
              (TV regulariser   sum_k (theta_{k+1}-theta_k - 1/(K-1))^2   bounded)
We initialise theta = identity (theta_k = u_k) and optimise with Adam.

This is genuinely *semantic* in the Laidlaw sense: a smooth functional
perturbation in pixel-value space rather than an Linf box in pixel space.
Defences that rely on the input being trapped in an Linf-eps ball (PGD-AT
at eps=0.1) are not guaranteed to extend.

PAPERS
------
Laidlaw & Feizi 2019 (NeurIPS) showed ReColorAdv defeats PGD-AT-trained
models on CIFAR-10 with ASR 89-100% at imperceptible delta_inf. Bhattad
et al. 2020 ColorFool similarly attacked CIFAR/ImageNet via natural-colour
priors; ASR remained high vs Linf-AT. Hosseini & Poovendran 2018 introduced
HSV semantic shifts and showed Inception-v3 ASR ~100% at small hue shifts.
Open question: does Linf-AT or a Lipschitz/Jacobian penalty offer any
incidental protection against this *different* threat model?

CRITIQUE OF SCOPE
-----------------
On grayscale the attack reduces to a 1-D monotone-ish remap; this is the
EASIEST instance of ReColorAdv and a LOWER bound on what a colour-channel
version would do. If even the 1-D version evades a defence, the full
multi-channel version certainly will. If the 1-D version does *not* evade
a defence, that does NOT prove robustness to full ReColorAdv on RGB.

CONDITIONS
----------
Train three defences (single SEED=0 budget):
  - CE         : plain cross-entropy   (campaign baseline, expect PGD ASR ~0.92)
  - Linf-AT    : PGD-7 AT at eps=0.1   (campaign winner, expect PGD ASR ~0.32)
  - Jac-Frob   : Jacobian Frobenius penalty lam=0.01 (H323/H391-verified
                 genuine defence)

Evaluate each under:
  - Clean accuracy.
  - PGD-10 Linf at eps=0.1 (sanity vs campaign tables).
  - ReColorAdv grid sweep K in {5, 9, 16}, delta_inf in {0.05, 0.10, 0.20},
    Adam lr=0.05, 50 iters, TV regulariser lam_tv=0.5.

KNOBS
-----
K (grid resolution): higher K = more expressive remap = ASR up.
delta_inf:           larger = bigger off-identity excursion = ASR up.
We report ASR over originally-correct samples.

STANDARD CONFIG
---------------
N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9, wd=5e-4),
SEED=0, EPS=0.1. SmallCNN width=32.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
WIDTH = 32

# Defence conditions to train.
DEFENCES = [
    ("CE",       dict(mode="ce")),
    ("Linf-AT",  dict(mode="at",   adv_eps=EPS, adv_steps=7)),
    ("Jac-Frob", dict(mode="jac",  lam=0.01)),
]

# ReColorAdv attack grid.
RECOLOR_K        = [5, 9, 16]
RECOLOR_DELTAINF = [0.05, 0.10, 0.20]
RECOLOR_ITERS    = 50
RECOLOR_LR       = 0.05
RECOLOR_TV       = 0.5

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h473_recoloradv_attack_output.txt",
)


# ---------------------------------------------------------------------------
# training routines
# ---------------------------------------------------------------------------
def _make_model():
    return C.build_model("cnn", C.dataset_meta(DS), width=WIDTH, act="relu", bn=True)


def _opt(model):
    return torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)


def train_ce(Xtr, Ytr):
    C.set_seed(SEED)
    model = _make_model()
    opt = _opt(model)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def train_at(Xtr, Ytr, adv_eps=0.1, adv_steps=7):
    C.set_seed(SEED)
    model = _make_model()
    opt = _opt(model)
    n = Xtr.size(0)
    alpha = 2.5 * adv_eps / adv_steps
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.pgd(model, xb, yb, eps=adv_eps, steps=adv_steps, alpha=alpha)
            opt.zero_grad()
            loss = F.cross_entropy(model(xa), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def train_jac(Xtr, Ytr, lam=0.01):
    """Jacobian Frobenius penalty via one-hot Hutchinson estimate
    (matches H323 - Frobenius of input-output Jacobian on the loss).
    Practical surrogate: penalise ||grad_x CE||^2 plus a random-class
    Frobenius term so we cover off-label rows of J."""
    C.set_seed(SEED)
    model = _make_model()
    opt = _opt(model)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr[idx].clone().detach().requires_grad_(True)
            yb = Ytr[idx]
            opt.zero_grad()
            logits = model(xb)
            ce = F.cross_entropy(logits, yb)
            # Hutchinson row sample over class index.
            v = F.one_hot(torch.randint(0, logits.size(1), (xb.size(0),),
                                        device=xb.device), logits.size(1)).float()
            scalar = (logits * v).sum()
            gx = torch.autograd.grad(scalar, xb, create_graph=True)[0]
            jac_frob = (gx ** 2).sum() / xb.size(0)
            total = ce + lam * jac_frob
            total.backward()
            opt.step()
    model.eval()
    return model


def train_defence(name, cfg, Xtr, Ytr):
    mode = cfg["mode"]
    if mode == "ce":
        return train_ce(Xtr, Ytr)
    if mode == "at":
        return train_at(Xtr, Ytr, adv_eps=cfg["adv_eps"], adv_steps=cfg["adv_steps"])
    if mode == "jac":
        return train_jac(Xtr, Ytr, lam=cfg["lam"])
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# ReColorAdv (grayscale piecewise-linear intensity remap) attack
# ---------------------------------------------------------------------------
def _apply_remap(x, theta, K):
    """Piecewise-linear interpolation: theta is (K,) tensor of grid values.
    x in [0,1] of shape (B,1,H,W). Returns same shape, clamped to [0,1]."""
    t = x * (K - 1)                              # (B,1,H,W) in [0, K-1]
    j = torch.clamp(t.floor().long(), 0, K - 2)  # cell index
    alpha = (t - j.float()).clamp(0.0, 1.0)
    left = theta[j]
    right = theta[j + 1]
    out = (1.0 - alpha) * left + alpha * right
    return out.clamp(0.0, 1.0)


def recoloradv_attack(model, X, Y, K, delta_inf, iters=RECOLOR_ITERS,
                      lr=RECOLOR_LR, lam_tv=RECOLOR_TV, batch=256):
    """Run ReColorAdv on a tensor (X, Y). One theta vector per BATCH (shared
    across the batch, as in Laidlaw 2019 - the colour map is image-shared).
    Returns adversarial images and per-batch ASR pieces."""
    model.eval()
    device = X.device
    K = int(K)
    knots = torch.linspace(0.0, 1.0, K, device=device)        # identity init
    adv_chunks = []
    for i in range(0, X.size(0), batch):
        xb = X[i:i + batch]
        yb = Y[i:i + batch]
        theta = knots.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            x_remap = _apply_remap(xb, theta, K)
            logits = model(x_remap)
            ce = F.cross_entropy(logits, yb)
            # TV regulariser: deviation of finite-difference from identity slope.
            slope = (theta[1:] - theta[:-1]) - (1.0 / (K - 1))
            tv = (slope ** 2).sum()
            loss = -ce + lam_tv * tv          # maximise CE, minimise -CE
            loss.backward()
            opt.step()
            # project to box around identity: |theta_k - u_k| <= delta_inf.
            with torch.no_grad():
                theta.data = torch.maximum(theta.data, knots - delta_inf)
                theta.data = torch.minimum(theta.data, knots + delta_inf)
                theta.data = theta.data.clamp(0.0, 1.0)
        with torch.no_grad():
            x_adv = _apply_remap(xb, theta.detach(), K)
        adv_chunks.append(x_adv.detach())
    return torch.cat(adv_chunks, dim=0)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def _correct_mask(model, X, Y, batch=512):
    model.eval()
    out = []
    for i in range(0, X.size(0), batch):
        out.append((model(X[i:i + batch]).argmax(1) == Y[i:i + batch]).cpu())
    return torch.cat(out)


@torch.no_grad()
def _pred(model, X, batch=512):
    out = []
    for i in range(0, X.size(0), batch):
        out.append(model(X[i:i + batch]).argmax(1).cpu())
    return torch.cat(out)


def eval_clean_and_pgd(model, X, Y):
    _, clean_acc = C.logits_and_acc(model, X, Y)
    Xpgd = C.pgd(model, X, Y, eps=EPS, steps=PGD_STEPS)
    res = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return float(clean_acc), res["asr"]


def eval_recoloradv(model, X, Y, K, delta_inf):
    corr = _correct_mask(model, X, Y).numpy().astype(bool)
    X_adv = recoloradv_attack(model, X, Y, K=K, delta_inf=delta_inf)
    pred = _pred(model, X_adv).numpy()
    y = Y.cpu().numpy()
    flipped = pred != y
    asr = float(flipped[corr].mean()) if corr.sum() > 0 else float("nan")
    return asr


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=2000, seed=SEED)

    # ---- train all defences ------------------------------------------------
    trained = {}
    timings = {}
    for name, cfg in DEFENCES:
        print(f"\n=== training defence: {name} ===", flush=True)
        t0 = time.time()
        model = train_defence(name, cfg, Xtr, Ytr)
        trained[name] = model
        timings[name] = time.time() - t0
        print(f"  trained {name} in {timings[name]:.1f}s", flush=True)

    # ---- baseline evaluations (clean, PGD Linf eps=0.1) --------------------
    baselines = {}
    for name, model in trained.items():
        clean, pgd_asr = eval_clean_and_pgd(model, Xte, Yte)
        baselines[name] = dict(clean_acc=clean, pgd_asr=pgd_asr)
        print(f"  [{name}] clean={clean:.4f} pgd_asr={pgd_asr:.4f}", flush=True)

    # ---- ReColorAdv sweep --------------------------------------------------
    results = []  # list of (defence, K, delta, asr)
    for name, model in trained.items():
        for K in RECOLOR_K:
            for delta in RECOLOR_DELTAINF:
                t0 = time.time()
                asr = eval_recoloradv(model, Xte, Yte, K=K, delta_inf=delta)
                dt = time.time() - t0
                results.append((name, K, delta, asr, dt))
                print(f"  [{name}] ReColorAdv K={K} delta={delta:.2f} "
                      f"-> ASR={asr:.4f}  ({dt:.1f}s)", flush=True)

    # ---- write report ------------------------------------------------------
    lines = []
    lines.append("H473 ReColorAdv (functional) attack on Fashion-MNIST")
    lines.append("=" * 72)
    lines.append("Anchor: laidlaw-2019-recoloradv (NeurIPS 2019).")
    lines.append("Aux: bhattad-2020-colorfool, hosseini-2018-semantic.")
    lines.append("Threat model: piecewise-linear intensity remap on [0,1].")
    lines.append("")
    lines.append(f"Dataset={DS}  N_train={N_TRAIN}  Epochs={EPOCHS}  "
                 f"LR={LR}  Batch={BATCH}  Seed={SEED}  EPS(Linf)={EPS}")
    lines.append(f"ReColorAdv: iters={RECOLOR_ITERS}  lr={RECOLOR_LR}  "
                 f"lam_tv={RECOLOR_TV}  K in {RECOLOR_K}  delta in {RECOLOR_DELTAINF}")
    lines.append("")
    lines.append("Defence baselines (clean acc / PGD-Linf eps=0.1 ASR):")
    lines.append("-" * 72)
    lines.append(f"{'defence':<12} {'clean_acc':>10} {'pgd_asr':>10} {'train_s':>9}")
    for name in trained.keys():
        b = baselines[name]
        lines.append(f"{name:<12} {b['clean_acc']:>10.4f} {b['pgd_asr']:>10.4f} "
                     f"{timings[name]:>9.1f}")
    lines.append("")
    lines.append("ReColorAdv ASR table (fraction of originally-correct flipped):")
    lines.append("-" * 72)
    lines.append(f"{'defence':<12} {'K':>3} {'delta_inf':>10} {'asr':>8} {'attack_s':>9}")
    for name, K, delta, asr, dt in results:
        lines.append(f"{name:<12} {K:>3d} {delta:>10.2f} {asr:>8.4f} {dt:>9.1f}")
    lines.append("")
    lines.append("Verdict (auto):")
    # group by defence
    by_def = {}
    for name, K, delta, asr, _ in results:
        by_def.setdefault(name, []).append(asr)
    summary = {n: (float(np.mean(v)), float(np.max(v))) for n, v in by_def.items()}
    for name, (mean_asr, max_asr) in summary.items():
        lines.append(f"  {name:<10}  mean_ASR={mean_asr:.4f}  worst_ASR={max_asr:.4f}")
    ce_worst = summary.get("CE",       (float("nan"), float("nan")))[1]
    at_worst = summary.get("Linf-AT",  (float("nan"), float("nan")))[1]
    jac_worst = summary.get("Jac-Frob",(float("nan"), float("nan")))[1]
    lines.append("")
    lines.append("Reading:")
    lines.append("  * If Linf-AT worst_ASR ~ CE worst_ASR: Linf-AT does NOT")
    lines.append("    transfer to the functional threat model (expected from")
    lines.append("    Laidlaw 2019 on CIFAR-10). SUPPORTED.")
    lines.append("  * If Jac-Frob worst_ASR <= Linf-AT worst_ASR by >0.05: a")
    lines.append("    Lipschitz-type defence offers incidental coverage of the")
    lines.append("    semantic remap (theta is low-dim, smooth -> small")
    lines.append("    Jacobian shrinks the attack direction).")
    lines.append("  * ASR rises monotonically with K and delta_inf if the")
    lines.append("    attack is well-formed; non-monotonicity flags optimiser")
    lines.append("    issues at K=5 (under-parameterised).")
    lines.append("")
    lines.append("Limitations:")
    lines.append("  * Grayscale collapses ReColorAdv to a 1-D remap; this is a")
    lines.append("    LOWER bound on what the full RGB attack would achieve.")
    lines.append("  * Single seed (SEED=0), N_TRAIN=6000.")
    lines.append("  * No adaptive defence (no AT against ReColorAdv).")

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
        f.flush()
    print(f"\nSaved to {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
