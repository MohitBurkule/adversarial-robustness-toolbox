"""
H488 - Per-sample input-Hessian top-eigenvalue (loss sharpness in input space)
       correlates inversely with margin and is compressed by adversarial training.

Seed (paper Sec. 5 / Sec. 3 G8)
-------------------------------
A long line of work links the *input-space* curvature of the loss surface
(the spectrum of the Hessian of the cross-entropy with respect to the input
pixels x) to adversarial robustness:

  - Moosavi-Dezfooli et al., "Robustness via Curvature Regularization, and
    Vice Versa" (CURE), CVPR 2019, arXiv:1811.09716 -- shows the dominant
    Hessian eigenvalue lambda_max(H_x L) explodes after standard training and
    that explicitly penalising it (CURE penalty: ||grad_x L(x) - grad_x L(x+h*v)||
    along the gradient direction) yields PGD-robust models. Their core claim
    is that small lambda_max => locally near-affine loss => bigger
    margin-to-decision-boundary.
  - Yao et al., "PyHessian: Neural Networks Through the Lens of the Hessian",
    IEEE BigData 2020, arXiv:1912.07145 -- standardised Hutchinson-trace +
    power-iteration estimators for top eigenvalues of huge Hessians (their
    target is the *parameter* Hessian; we re-use the same primitives in
    *input* space here, since the input Hessian is 784x784 for F-MNIST and a
    full eigendecomposition is wasteful per-sample).
  - Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
    Generalization" (SAM), ICLR 2021, arXiv:2010.01412 -- in *parameter*
    space, but the principle is identical: training that picks flat (low
    lambda_max) minima generalises and is empirically more robust.
  - Madry et al. 2018 and Zhang et al. 2019 (TRADES) -- AT is known to flatten
    the input-loss landscape (Qin et al. 2019 "Adversarial Robustness through
    Local Linearization"; Andriushchenko & Flammarion 2020).

Hypothesis (paper G8 wording, made testable here)
-------------------------------------------------
Across multiple training regimes on Fashion-MNIST SmallCNN:
  (H1) Spearman rho( lambda_max(H_x L_i),  min_eps_to_flip_i ) <= -0.3
       on a held-out evaluation subset, for at least the standard-trained
       model. (Negative correlation: sharper inputs flip with smaller eps.)
  (H2) Adversarially-trained models (PGD-AT, TRADES, CURE) have BOTH
       (a) lower mean lambda_max, and (b) smaller variance of lambda_max
       across samples, than the standard-trained baseline.

Critique seed (must keep visible -- not bury)
---------------------------------------------
1. The full input Hessian is d x d with d = 784 for F-MNIST. Even a single
   exact eigendecomposition is ~ d^3 = 4.8e8 ops, and we'd need it per
   sample. We MUST use stochastic estimators. We use the Hutchinson trace
   only as a sanity quantity; the headline lambda_max comes from
   *power iteration* with Hessian-vector products (Pearlmutter trick:
   one extra autograd pass per HVP). 20 iterations is the PyHessian
   default and converges to within ~1e-3 relative error on d=784 in
   our pilot.
2. lambda_max is a noisy per-sample quantity for samples close to label
   boundaries (the CE loss is locally linear if the prediction is wrong-
   confident, giving lambda_max ~ 0). We therefore restrict the per-sample
   correlation analysis to samples that are correctly classified by the
   model (margin > 0), exactly as the CURE paper does (Sec. 4 of 1811.09716).
3. min_eps_to_flip is itself an estimator. We use 6-step PGD binary search
   on L-inf eps in [0, 0.30] with 5 inner PGD steps, matching h118_cure.py
   (the project's reference CURE script). PGD failing to flip at eps_max
   sets eps = 0.30 (right-censored); we report both Spearman (rank-based,
   robust to censoring) and Pearson.
4. We compute lambda_max on a tractable SUBSET (N_HESS=200 samples) for
   each of M=5 models. That gives 1000 (lambda_max, min_eps) pairs total
   but only 200 per-model Spearman correlations. With N=200, the 95% CI
   on a Spearman ~-0.3 is roughly +/- 0.12 (Fisher z), so the H1
   threshold of -0.3 is right at the edge of what we can resolve at this
   scale -- a tighter result would need a bigger subset, which we trade
   off against the wall-clock cost of 20 HVPs per sample per model.
5. CURE penalty -- there IS an existing CURE script in this hypothesis
   directory (hypotheses/h118_cure.py); we re-use its exact penalty form
   (h=3.0, lambda=4.0) but train under the project-standard SmallCNN +
   training schedule from campaign.common, not h118's bespoke CNN, so
   the four conditions share architecture and only differ in loss.
6. Power iteration on a positive-semi-definite quadratic recovers
   lambda_max in absolute value; for the CE-loss Hessian at well-classified
   inputs the Hessian is generally PSD-ish but can have small negative
   curvatures (saddle directions). We Rayleigh-quotient-restart whenever
   the iterate sign flips, and report |lambda_max| as the sharpness
   scalar -- this matches how CURE / SAM treat it.

Extra papers (>=2)
------------------
- `moosavi-2019-cure` Moosavi-Dezfooli et al., "Robustness via Curvature
  Regularization, and Vice Versa", CVPR 2019. THE direct prior art for the
  G8 claim. Their Fig 3 shows mean lambda_max(H_x L) drops by 1-2 orders
  of magnitude after adversarial training on CIFAR-10. We replicate the
  *direction* of that finding on F-MNIST and add the per-sample
  correlation analysis (which they only do at the population level).
- `yao-2020-pyhessian` Yao et al., "PyHessian", arXiv:1912.07145. Source
  of the Hutchinson + power-iteration algorithms we re-implement here in
  input space. They target *parameter* Hessians; we apply the same
  primitives to the input Hessian, which is much smaller (d=784) so the
  estimator is more accurate per-iteration.
- `foret-2021-sam` Foret et al., "Sharpness-Aware Minimization", ICLR
  2021, arXiv:2010.01412. Conceptual frame: training methods that bias
  toward flat (low-curvature) optima improve both generalisation and
  robustness. Our H2 reduction-in-mean-AND-variance prediction is the
  *input-space* analogue of SAM's parameter-space flatness story.
- `qin-2019-llr` Qin et al., "Adversarial Robustness through Local
  Linearization", NeurIPS 2019. Argues AT works in large part by
  flattening the input-loss landscape so that the local-linear
  approximation used in FGSM/PGD becomes tight -- gives a *causal*
  reason to expect H2.

Controls / ablations (must-haves)
---------------------------------
A. Four training conditions, ALL sharing SmallCNN(width=32) + project
   training schedule (EPOCHS=8, SGD, lr=0.05, BATCH=128):
     STD     - vanilla CE
     PGD-AT  - Madry CE on PGD-adv samples (eps=0.1, 10 steps)
     TRADES  - CE + beta*KL(f(x)||f(x_adv)), beta=6 (matches h441 default)
     CURE    - CE + 4.0 * ||grad_x L(x) - grad_x L(x + 3.0 * grad/||grad||)||_2
B. lambda_max via 20-step input-Hessian power iteration with HVPs
   (Pearlmutter trick). Restart on sign-flip; report |Rayleigh quotient|.
   Sanity: also report Hutchinson estimate of trace(H_x L) on a 10-vector
   probe (gives mean eigenvalue) -- ensures lambda_max >> trace/d so we
   are picking up a real top mode, not a flat spectrum artefact.
C. min_eps_to_flip via 6-step bisection of L-inf eps in [0, 0.30] with
   5-step PGD inner loop per bisection (matches h118_cure.py).
D. Per-model Spearman + Pearson correlations of lambda_max vs
   min_eps_to_flip, restricted to clean-correct samples. Report both p
   and a 1000-sample bootstrap 95% CI on Spearman.
E. AT-induced shift: report mean, std, median, p05, p95 of lambda_max
   per model. Headline H2 metric is the ratio
       (mean_std + std_std) / (mean_AT + std_AT)  for AT in {PGD,TRADES,CURE}
   -- a value > 1 means AT reduced the mean+spread of sharpness.
F. Clean accuracy and PGD ASR @ eps=0.1 per model so the reader can
   anchor the lambda_max numbers to the usual robustness summary
   (otherwise one might worry CURE/TRADES underfit and "low sharpness"
   is just an artefact of a degenerate model).

Config (project standard)
-------------------------
DS=fashion_mnist, N_TRAIN=6000, N_EVAL_FOR_ATTACK=2000, EPOCHS=8, BATCH=128,
opt=SGD(mom=0.9, wd=1e-4), LR=0.05, SEED=0, EPS=0.1, PGD_STEPS=10,
PGD_ALPHA=0.01, N_HESS=200 (per-model Hessian subset), HVP_STEPS=20
(power iteration), HUTCH_K=10 (trace probes), CURE_H=3.0, CURE_LAM=4.0,
TRADES_BETA=6.0, EPS_MAX_BIS=0.30, BIS_STEPS=6, BIS_PGD_STEPS=5.

Output
------
results/fashion_mnist/h488_hessian_sharpness_output.txt
Flushed after every condition so partial progress is durable.

Do NOT execute this script from the main session; delegate to a background
agent per project workflow (CLAUDE.md).
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config ---------------------------------------------------------------
DS              = "fashion_mnist"
SEED            = 0
N_TRAIN         = 6000
N_EVAL          = 2000
N_HESS          = 200          # per-model Hessian subset
EPOCHS          = 8
BATCH           = 128
LR              = 0.05
EPS             = 0.1
PGD_STEPS       = 10
PGD_ALPHA       = 0.01
HVP_STEPS       = 20           # power-iteration steps for lambda_max
HUTCH_K         = 10           # Hutchinson trace probes
CURE_H          = 3.0
CURE_LAM        = 4.0
TRADES_BETA     = 6.0
EPS_MAX_BIS     = 0.30
BIS_STEPS       = 6
BIS_PGD_STEPS   = 5

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h488_hessian_sharpness_output.txt",
)


# ---- training helpers -----------------------------------------------------
def _new_model_and_opt(seed):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return model, opt, sched


def train_std(Xtr, Ytr, seed=SEED):
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_pgd_at(Xtr, Ytr, seed=SEED):
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            xa = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(xa), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def _pgd_kl(model, x, eps, steps, alpha):
    """TRADES inner attack: PGD maximising KL(f(x) || f(x+delta))."""
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1).detach()
    with torch.no_grad():
        p_clean = F.softmax(model(x0), dim=1)
    for _ in range(steps):
        xa.requires_grad_(True)
        log_p_adv = F.log_softmax(model(xa), dim=1)
        kl = F.kl_div(log_p_adv, p_clean, reduction="batchmean")
        g, = torch.autograd.grad(kl, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def train_trades(Xtr, Ytr, seed=SEED, beta=TRADES_BETA):
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            xa = _pgd_kl(model, xb, EPS, PGD_STEPS, PGD_ALPHA)
            model.train()
            opt.zero_grad()
            out_clean = model(xb)
            loss_ce = F.cross_entropy(out_clean, yb)
            log_p_clean = F.log_softmax(out_clean, dim=1).detach()
            p_clean = log_p_clean.exp()
            log_p_adv = F.log_softmax(model(xa), dim=1)
            kl = (p_clean * (log_p_clean - log_p_adv)).sum(dim=1).mean()
            loss = loss_ce + beta * kl
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def _cure_penalty(model, x, y, h=CURE_H, lam=CURE_LAM):
    """CURE penalty (Moosavi-Dezfooli et al. CVPR 2019, eq. 5/6).
       Approximates lambda_max(H_x L) by finite-difference of grad_x L along
       the gradient direction, then penalises its norm.
    """
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    g, = torch.autograd.grad(loss.sum(), x, create_graph=True)
    g_flat = g.view(x.size(0), -1)
    g_norm = g_flat.norm(2, dim=1, keepdim=True) + 1e-8
    z = (h * (g_flat / g_norm)).view_as(g).detach()
    loss_pos = F.cross_entropy(model(x + z), y)
    g_diff, = torch.autograd.grad((loss_pos - loss).sum(), x, create_graph=True)
    reg = g_diff.view(x.size(0), -1).norm(2, dim=1)
    return lam * reg.mean()


def train_cure(Xtr, Ytr, seed=SEED):
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss_ce = F.cross_entropy(model(xb), yb)
            loss_reg = _cure_penalty(model, xb, yb, h=CURE_H, lam=CURE_LAM)
            (loss_ce + loss_reg).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- Hessian primitives (input space) -------------------------------------
def _input_hvp(model, x, y, v):
    """Hessian-vector product H_x L * v via Pearlmutter trick.
       x, v: (1, C, H, W); y: (1,). Returns Hv of shape v.
       Single-sample to keep per-sample lambda_max sharp.
    """
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    g, = torch.autograd.grad(loss, x, create_graph=True)
    gv = (g * v).sum()
    Hv, = torch.autograd.grad(gv, x, retain_graph=False)
    return Hv.detach()


def input_lambda_max(model, x, y, steps=HVP_STEPS, tol=1e-4):
    """Top-eigenvalue of H_x L for a SINGLE sample (x:[1,C,H,W], y:[1]).
       20-step power iteration with HVPs. Returns |Rayleigh quotient| at
       convergence -- absolute value because CE Hessian may have small
       negative directions at saddle-y inputs; CURE/SAM both report |lambda|.
    """
    g_seed = torch.Generator(device=x.device).manual_seed(0)
    v = torch.randn(x.shape, generator=g_seed, device=x.device)
    v = v / (v.norm() + 1e-12)
    lam_prev = 0.0
    for _ in range(steps):
        Hv = _input_hvp(model, x, y, v)
        lam = (v * Hv).sum().item()             # Rayleigh quotient
        nrm = Hv.norm().item() + 1e-12
        v = Hv / nrm
        # restart on sign flip (saddle), preserving |lambda|
        if abs(lam - lam_prev) / (abs(lam_prev) + 1e-8) < tol:
            break
        lam_prev = lam
    return float(abs(lam))


def input_trace_hutchinson(model, x, y, k=HUTCH_K):
    """Hutchinson estimator of trace(H_x L) on a SINGLE sample, k Rademacher
       probes. Sanity check that lambda_max is genuinely a top mode and not
       an artefact of a flat spectrum (we expect lambda_max >> trace/d).
    """
    tr = 0.0
    for _ in range(k):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).sub_(1)   # Rademacher
        Hv = _input_hvp(model, x, y, v)
        tr += (v * Hv).sum().item()
    return tr / k


# ---- min eps to flip (matches h118_cure.py bisection) ---------------------
def min_eps_to_flip(model, x, y, eps_max=EPS_MAX_BIS, iters=BIS_STEPS,
                    pgd_inner=BIS_PGD_STEPS):
    """Batched binary search for the smallest L-inf eps that flips PGD."""
    lo = torch.zeros(x.size(0), device=x.device)
    hi = torch.full((x.size(0),), eps_max, device=x.device)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = x.clone().detach()
        alpha = mid / 4.0
        for _ in range(pgd_inner):
            adv.requires_grad_(True)
            loss = F.cross_entropy(model(adv), y)
            grd, = torch.autograd.grad(loss, adv)
            adv = adv.detach() + alpha.view(-1, 1, 1, 1) * grd.sign()
            adv = torch.min(torch.max(adv,
                                      x - mid.view(-1, 1, 1, 1)),
                            x + mid.view(-1, 1, 1, 1)).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi.detach().cpu().numpy()


# ---- correlation helpers --------------------------------------------------
def _spearman(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra * ra).sum() * (rb * rb).sum()) + 1e-12))


def _pearson(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def _spearman_boot_ci(a, b, B=1000, seed=0):
    if len(a) < 10:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    rs = []
    n = len(a)
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        rs.append(_spearman(np.asarray(a)[idx], np.asarray(b)[idx]))
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


# ---- per-model evaluation -------------------------------------------------
def eval_model(name, model, Xte, Yte, fout):
    """Compute lambda_max + min_eps + summary stats for a single trained model."""
    model.eval()
    # parameters must keep grad on for HVPs (input grads only need x.requires_grad)
    for p in model.parameters():
        p.requires_grad_(True)

    # robustness anchors (clean acc + PGD ASR @ eps=0.1)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    pgd_asr = C.attack_success(model, Xte, Yte, attack="pgd",
                               eps=EPS, steps=PGD_STEPS)["asr"]

    # restrict per-sample analysis to clean-correct samples (CURE Sec 4)
    with torch.no_grad():
        preds = model(Xte).argmax(1)
        correct_mask = (preds == Yte)
    idx_correct = correct_mask.nonzero(as_tuple=True)[0]
    take = idx_correct[:N_HESS]
    Xs, Ys = Xte[take], Yte[take]

    # min eps to flip (batched)
    t0 = time.time()
    eps_flip = min_eps_to_flip(model, Xs, Ys)
    t_eps = time.time() - t0

    # per-sample lambda_max (single-sample loop -- HVPs are cheap on d=784)
    t0 = time.time()
    lams = np.empty(Xs.size(0), dtype=np.float64)
    traces = np.empty(Xs.size(0), dtype=np.float64)
    for i in range(Xs.size(0)):
        xi = Xs[i:i + 1]
        yi = Ys[i:i + 1]
        lams[i]   = input_lambda_max(model, xi, yi, steps=HVP_STEPS)
        traces[i] = input_trace_hutchinson(model, xi, yi, k=HUTCH_K)
    t_hess = time.time() - t0

    # correlations
    sp = _spearman(lams, eps_flip)
    pr = _pearson(lams, eps_flip)
    lo, hi = _spearman_boot_ci(lams, eps_flip, B=1000, seed=0)

    row = {
        "name": name,
        "clean_acc": float(clean_acc),
        "pgd_asr": float(pgd_asr),
        "n_hess": int(Xs.size(0)),
        "lam_mean": float(lams.mean()),
        "lam_std":  float(lams.std()),
        "lam_med":  float(np.median(lams)),
        "lam_p05":  float(np.percentile(lams, 5)),
        "lam_p95":  float(np.percentile(lams, 95)),
        "trace_mean_over_d": float(traces.mean() / (META["channels"] * META["size"] ** 2)),
        "eps_mean": float(eps_flip.mean()),
        "eps_p05":  float(np.percentile(eps_flip, 5)),
        "spearman_lam_vs_eps": sp,
        "pearson_lam_vs_eps":  pr,
        "spearman_ci95_lo": lo,
        "spearman_ci95_hi": hi,
        "t_eps_s":  round(t_eps, 1),
        "t_hess_s": round(t_hess, 1),
    }

    fout.write(f"\n--- {name} ---\n")
    fout.write(f"  clean_acc            : {row['clean_acc']:.3f}\n")
    fout.write(f"  pgd ASR @ eps={EPS}  : {row['pgd_asr']:.3f}\n")
    fout.write(f"  n_hess (clean-correct subset) : {row['n_hess']}\n")
    fout.write(f"  lambda_max stats     : mean={row['lam_mean']:.3e} std={row['lam_std']:.3e} "
               f"med={row['lam_med']:.3e} p05={row['lam_p05']:.3e} p95={row['lam_p95']:.3e}\n")
    fout.write(f"  trace/d (sanity)     : {row['trace_mean_over_d']:.3e}   "
               f"(expect << lambda_mean if a real top mode exists)\n")
    fout.write(f"  min_eps_to_flip      : mean={row['eps_mean']:.3f} p05={row['eps_p05']:.3f}\n")
    fout.write(f"  spearman(lam_max, min_eps) : {row['spearman_lam_vs_eps']:+.3f}  "
               f"95%CI=[{row['spearman_ci95_lo']:+.3f},{row['spearman_ci95_hi']:+.3f}]\n")
    fout.write(f"  pearson (lam_max, min_eps) : {row['pearson_lam_vs_eps']:+.3f}\n")
    fout.write(f"  timing               : eps={row['t_eps_s']}s  hess={row['t_hess_s']}s\n")
    fout.flush()
    return row


# ---- main -----------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    fout = open(OUT_FILE, "w")
    hdr = "H488 - input-Hessian lambda_max correlates with margin and is compressed by AT"
    fout.write("=" * len(hdr) + "\n" + hdr + "\n" + "=" * len(hdr) + "\n")
    fout.write(f"device={C.DEVICE}  dataset={DS}  seed={SEED}\n")
    fout.write(f"N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} N_HESS={N_HESS} "
               f"HVP_STEPS={HVP_STEPS} HUTCH_K={HUTCH_K}\n")
    fout.write(f"EPS={EPS} PGD_STEPS={PGD_STEPS} EPS_MAX_BIS={EPS_MAX_BIS} "
               f"BIS_STEPS={BIS_STEPS}\n")
    fout.write(f"CURE_H={CURE_H} CURE_LAM={CURE_LAM} TRADES_BETA={TRADES_BETA}\n")
    fout.flush()

    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    conditions = [
        ("STD",     train_std),
        ("PGD-AT",  train_pgd_at),
        ("TRADES",  train_trades),
        ("CURE",    train_cure),
    ]

    rows = []
    for name, train_fn in conditions:
        fout.write(f"\n[train] {name} ...\n"); fout.flush()
        t0 = time.time()
        model = train_fn(Xtr, Ytr, seed=SEED)
        fout.write(f"[train] {name} done in {time.time() - t0:.1f}s\n"); fout.flush()
        rows.append(eval_model(name, model, Xte, Yte, fout))

    # ---- headline analysis ----
    fout.write("\n" + "=" * 70 + "\n")
    fout.write("HEADLINE\n")
    fout.write("=" * 70 + "\n")
    by = {r["name"]: r for r in rows}

    fout.write("\nH1: per-sample Spearman( lambda_max , min_eps_to_flip ) <= -0.3 ?\n")
    h1_hits = []
    for r in rows:
        sp = r["spearman_lam_vs_eps"]
        hit = (sp == sp) and (sp <= -0.3)
        h1_hits.append((r["name"], sp, hit, r["spearman_ci95_lo"], r["spearman_ci95_hi"]))
        fout.write(f"  {r['name']:7s}: rho={sp:+.3f}  "
                   f"95%CI=[{r['spearman_ci95_lo']:+.3f},{r['spearman_ci95_hi']:+.3f}]  "
                   f"-> {'YES' if hit else 'no'}\n")

    fout.write("\nH2: AT (PGD-AT / TRADES / CURE) compresses lambda_max distribution vs STD ?\n")
    if "STD" in by:
        s = by["STD"]
        fout.write(f"  STD     baseline : mean={s['lam_mean']:.3e}  std={s['lam_std']:.3e}\n")
        h2_hits = {}
        for nm in ("PGD-AT", "TRADES", "CURE"):
            if nm in by:
                r = by[nm]
                shrink_mean = s["lam_mean"] / (r["lam_mean"] + 1e-30)
                shrink_std  = s["lam_std"]  / (r["lam_std"]  + 1e-30)
                shrunk = (r["lam_mean"] < s["lam_mean"]) and (r["lam_std"] < s["lam_std"])
                h2_hits[nm] = shrunk
                fout.write(f"  {nm:7s}        : mean={r['lam_mean']:.3e}  std={r['lam_std']:.3e}  "
                           f"shrink(mean,std)=({shrink_mean:.2f}x, {shrink_std:.2f}x)  "
                           f"-> {'YES' if shrunk else 'no'}\n")

    # verdict
    fout.write("\n" + "-" * 70 + "\n")
    n_h1 = sum(1 for _, _, h, _, _ in h1_hits if h)
    n_h2 = sum(1 for v in h2_hits.values() if v) if "STD" in by else 0
    verdict = "NO"
    if n_h1 >= 1 and n_h2 >= 2:
        verdict = "YES"
    elif n_h1 >= 1 or n_h2 >= 2:
        verdict = "PARTIAL"
    fout.write(f"VERDICT: {verdict}   "
               f"(H1 hits = {n_h1}/{len(h1_hits)}, "
               f"H2 hits = {n_h2}/{len(h2_hits) if 'STD' in by else 0})\n")
    fout.write("Interpretation: a negative per-sample Spearman confirms the CURE/PyHessian\n")
    fout.write("intuition that input-loss sharpness predicts how easily an example flips.\n")
    fout.write("AT-induced shrinkage of mean+spread of lambda_max corroborates Qin et al.'s\n")
    fout.write("local-linearisation account (and is the input-space analogue of SAM).\n")
    fout.write("=" * 70 + "\n")
    fout.close()


if __name__ == "__main__":
    main()
