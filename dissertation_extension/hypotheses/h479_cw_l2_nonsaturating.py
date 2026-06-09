"""
H479 - Carlini-Wagner L2 (no eps budget) as a NON-SATURATING strength metric.

Seed (CAMPAIGN_GAP_MAP S5 / S2 M8):
    The campaign has been measuring defences with PGD at L_inf eps=0.1.  At
    that eps, the metric SATURATES on both ends of the ladder:
      * undefended / lightly defended models hit ASR ~= 1.0  (ceiling = 1),
      * strong AT defences (PGD-AT eps=0.1, TRADES) hit ASR floors that
        differ from each other by less than seed noise.
    So PGD-ASR @ eps=0.1 cannot DISCRIMINATE between defences in either
    extreme.  We need a metric whose scale stretches with the defence.

    The natural choice is the *median L2 distance to the nearest adversarial
    example*, found by an UNCONSTRAINED minimum-norm attack.  Carlini & Wagner
    2017 introduced exactly this attack (the "CW-L2" attack); it has no eps
    budget, only an Adam-optimised trade-off between L2 distortion and an
    adversarial loss with a tanh change-of-variable to keep x_adv in [0,1].

    Hypothesis H479 (formal):
       median CW-L2 distance, taken over correctly classified test samples,
       discriminates strong vs weak Fashion-MNIST defences MORE finely than
       PGD-ASR @ eps=0.1 does.  Operationally: the Spearman rank correlation
       between the CW-L2-median ranking of defences and the PGD-eps=0.1 ASR
       ranking will be LOW (|rho| < 0.7) - the two metrics will disagree on
       at least one defence at the saturated ends - while the CW-L2 metric
       will produce a well-separated total order with no ties.

Critique seed (prior art - >= 2 papers, plus the original CW)
-------------------------------------------------------------
1. carlini-wagner-2017  Carlini & Wagner, "Towards Evaluating the Robustness
   of Neural Networks", IEEE S&P 2017.  Introduces CW-L2 with the tanh
   change-of-variable x_adv = 0.5*(tanh(w)+1) (so [0,1] is exact, not
   projected), Adam on `w`, and the f-loss
        f(x) = max( Z_y(x) - max_{j != y} Z_j(x), -kappa )
   optimised as  ||x - x0||_2^2 + c * f(x).  c is chosen by a binary search.
   We re-implement in ~80 LOC of pure torch (no ART per project rules).
2. croce-hein-2020       Croce & Hein, "Reliable evaluation of adversarial
   robustness with an ensemble of diverse parameter-free attacks", ICML 2020
   (AutoAttack).  AutoAttack's APGD-DLR component is the modern strong-attack
   successor to CW-DLR: the DLR loss is exactly CW's f-loss rescaled so it is
   shift-invariant w.r.t. logit calibration, and APGD adapts the step size.
   AutoAttack reports min-norm L2 numbers as one of its standard signals;
   our CW-L2 median is the cheap-but-honest proxy of the same thing.
3. brendel-rauber-bethge-2019  Brendel et al., "Accurate, reliable and fast
   robustness evaluation", NeurIPS 2019, a.k.a. the "Decoupling Direction
   and Norm" (DDN) attack.  DDN is a direct minimum-L2 attack designed to
   match CW-L2's accuracy at ~10x lower compute by decoupling the L2 norm
   (line search) from the perturbation direction (one gradient step).  We
   keep classical CW-L2 because it is the canonical literature reference
   and our 80-LOC budget allows it; DDN would be the natural follow-up if
   we hit a compute ceiling.

Why this is not already in the campaign
---------------------------------------
 * H02 ran CW-L2 on ONE vanilla CNN as a feature-vulnerability label; it
   did not use CW-L2 as a *defence-ranking* metric across the AT ladder.
 * H173-H433 only report PGD-Linf ASR.  The saturation problem (CAMPAIGN
   GAP S2 M8) is the actual research gap.
 * No prior hypothesis has computed CW-L2 median + per-class CW-L2 +
   compared its defence-ranking to PGD-ASR via Spearman.

Plan
----
Train 6 defences on Fashion-MNIST (single seed, N_train=6000, 10 epochs).
The 6 are chosen to span the campaign's defence ladder so the saturation
is visible at BOTH ends:

  D1  STD          standard CE-trained SmallCNN  (undefended -> PGD-ASR 1.0)
  D2  PGD-AT_e0.1  Madry PGD-AT at eps=0.1       (strong AT, the "winner")
  D3  TRADES_b6    Zhang 2019 TRADES, beta=6     (alternative strong AT)
  D4  MART         Wang 2020 MART (boosted CE +  (third strong AT family)
                                   KL on misclassified)
  D5  FGSM-AT      single-step FGSM-AT           (masking-suspect:
                                                  catastrophic overfit at
                                                  this eps; Wong 2020)
  D6  RFNN         frozen random conv features + (robust-feature / non-GD
                   trained linear readout         defence; Theme-B control)

For each defence, run CW-L2 (60 steps, Adam lr=0.01, c sweep {0.1, 1, 10})
on N_EVAL=500 correctly-classified test samples (CW-L2 is per-sample
optimisation - 500 samples * 6 models is a realistic budget at ~5 min
each).  Take best-c per sample.

Report (controls):
  (1) Per-defence: clean acc, PGD-Linf-ASR @ eps=0.1 (the saturated metric),
      PGD-L2-ASR @ eps_L2=2.0 (a cheaper L2 baseline), CW-success-rate,
      median CW-L2, mean CW-L2, CW-ASR @ L2 <= 2.0, CW-ASR @ L2 <= 3.0.
  (2) Per-class CW-L2 median (10 classes x 6 defences).
  (3) Spearman rho between
        ranking by CW-L2-median (descending = more robust)
        ranking by PGD-Linf-ASR (ascending  = more robust)
      If |rho| < 0.7 the two metrics disagree -> CW-L2 is the better
      discriminator (validates the seed).  If |rho| >= 0.9 the metrics
      agree and the saturation worry is overstated.

Verdict ladder (HEADLINE)
-------------------------
 YES        : CW-L2 median spreads the 6 defences over a range > 1.0 in L2,
              AND Spearman |rho| with PGD-Linf-ASR < 0.7
              -> CW-L2 is the right campaign-replacement metric.
 PARTIAL    : CW-L2 spreads the defences (range > 0.5) but agrees with
              PGD-Linf ranking (|rho| >= 0.7); useful as a secondary check
              but not strictly necessary.
 NO         : CW-L2 median range < 0.5 across the ladder.  Either the attack
              is too weak or the ladder is genuinely flat.
 SUSPICIOUS : any defence has CW-L2 median > 3.0 (close to the mean
              MNIST-class L2 distance ~7.0); on Fashion-MNIST that level
              of L2 is implausible without gradient masking, flag it.

Config
------
DS=fashion_mnist, N_TRAIN=6000, N_EVAL=500 (CW samples; clean/PGD use 2000),
EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9, wd=5e-4), SEED=0.
CW_STEPS=60, CW_LR=0.01, CW_C_SWEEP={0.1, 1.0, 10.0}, KAPPA=0.
PGD_LINF: eps=0.1, steps=10.   PGD-L2: eps_L2=2.0, steps=20, step=0.25.

Output
------
results/fashion_mnist/h479_cw_l2_nonsaturating_output.txt
Flushed after every model.

DO NOT execute this script from the main session; delegate to a background
agent per dissertation_extension/CLAUDE.md.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


# ---- config (project standard) --------------------------------------------
DS         = "fashion_mnist"
N_TRAIN    = 6000
N_EVAL_CW  = 500            # CW-L2 is per-sample optimisation; keep this small
N_EVAL_PGD = 2000           # PGD eval can use the full 2k cohort
EPOCHS     = 10
LR         = 0.05
BATCH      = 128
SEED       = 0

# Linf threat-model (saturated metric we want to beat)
EPS_LINF   = 0.1
PGD_STEPS  = 10

# L2 threat-model (cheap baseline)
EPS_L2     = 2.0
PGD_L2_STEPS = 20
PGD_L2_ALPHA = 0.25

# CW-L2 attack
CW_STEPS   = 60
CW_LR      = 1e-2
CW_C_SWEEP = [0.1, 1.0, 10.0]
CW_KAPPA   = 0.0

# TRADES / MART hyperparams
BETA_TRADES = 6.0
BETA_MART   = 5.0

META = {"channels": 1, "size": 28, "n_classes": 10}
NCLS = META["n_classes"]

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h479_cw_l2_nonsaturating_output.txt",
)


# ===========================================================================
# CW-L2 attack (Carlini & Wagner 2017) - pure torch, ~80 LOC
# ===========================================================================
def cw_l2_attack(model, x0, y, c, steps=CW_STEPS, lr=CW_LR, kappa=CW_KAPPA):
    """Untargeted CW-L2 with tanh change-of-variable + Adam.

        x_adv = 0.5*(tanh(w) + 1)               (so x_adv in (0,1) exactly)
        f(x') = max(Z_y - max_{j != y} Z_j, -kappa)
        min_w  ||x_adv - x0||_2^2  +  c * f(x_adv)

    Returns (x_adv, l2_dist, flipped) where l2_dist is per-sample L2 and
    flipped is bool argmax(x_adv) != y.
    """
    model.eval()
    n = x0.size(0)
    # invert tanh; clamp to avoid atanh(+/-1) blowup
    x_clamped = x0.clamp(1e-6, 1.0 - 1e-6)
    w = torch.atanh(2.0 * x_clamped - 1.0).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    one_hot = F.one_hot(y, NCLS).float()

    best_l2 = torch.full((n,), float("inf"), device=x0.device)
    best_x  = x0.clone()
    best_flipped = torch.zeros(n, dtype=torch.bool, device=x0.device)

    for _ in range(steps):
        x_adv = 0.5 * (torch.tanh(w) + 1.0)
        delta = x_adv - x0
        l2sq = delta.flatten(1).pow(2).sum(dim=1)
        logits = model(x_adv)
        real  = (one_hot * logits).sum(dim=1)
        other = ((1.0 - one_hot) * logits - one_hot * 1e4).max(dim=1).values
        f = torch.clamp(real - other, min=-kappa)
        loss = (l2sq + c * f).sum()

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            l2 = l2sq.sqrt()
            pred = logits.argmax(dim=1)
            flipped = (pred != y)
            improve = flipped & (l2 < best_l2)
            best_l2 = torch.where(improve, l2, best_l2)
            if improve.any():
                best_x[improve] = x_adv[improve].detach()
            best_flipped |= flipped

    return best_x.detach(), best_l2.detach(), best_flipped.detach()


def cw_l2_best_over_c(model, x0, y, c_sweep=CW_C_SWEEP):
    """Run CW-L2 across c_sweep; per sample keep the smallest L2 that flipped."""
    n = x0.size(0)
    best_l2 = torch.full((n,), float("inf"), device=x0.device)
    best_x  = x0.clone()
    best_flipped = torch.zeros(n, dtype=torch.bool, device=x0.device)
    for c in c_sweep:
        x_adv, l2, flipped = cw_l2_attack(model, x0, y, c=c)
        # CW often "succeeds" at non-flipping points too (the loss decreased
        # but argmax did not change); we must AND with flipped.
        better = flipped & (l2 < best_l2)
        best_x[better] = x_adv[better]
        best_l2[better] = l2[better]
        best_flipped |= flipped
    return best_x.detach(), best_l2.detach(), best_flipped.detach()


# ===========================================================================
# PGD-L2 baseline attack
# ===========================================================================
def pgd_l2(model, x, y, eps=EPS_L2, steps=PGD_L2_STEPS, alpha=PGD_L2_ALPHA):
    """L2-projected PGD, untargeted."""
    model.eval()
    x0 = x.clone().detach()
    # random init inside the L2 ball
    noise = torch.randn_like(x0)
    noise_flat = noise.flatten(1)
    noise_norm = noise_flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    noise = (noise_flat / noise_norm).view_as(x0) * eps * torch.rand(
        x0.size(0), 1, 1, 1, device=x0.device
    )
    xa = (x0 + noise).clamp(0, 1).detach()
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        gflat = g.flatten(1)
        gnorm = gflat.norm(dim=1, keepdim=True).clamp(min=1e-12)
        step = (gflat / gnorm).view_as(xa) * alpha
        xa = xa.detach() + step
        # project back into L2 ball around x0
        delta = xa - x0
        dflat = delta.flatten(1)
        dnorm = dflat.norm(dim=1, keepdim=True)
        factor = torch.clamp(eps / dnorm.clamp(min=1e-12), max=1.0)
        delta = (dflat * factor).view_as(delta)
        xa = (x0 + delta).clamp(0, 1)
    return xa.detach()


# ===========================================================================
# Training recipes for the 6 defences
# ===========================================================================
def _new_model_and_opt(seed):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=5e-4,
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


def train_pgd_at(Xtr, Ytr, seed=SEED, eps=EPS_LINF, steps=7):
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            xa = C.pgd(model, xb, yb, eps=eps, steps=steps,
                       alpha=2.5 * eps / steps)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(xa), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def _trades_inner(model, x, eps, steps, alpha):
    """PGD-Linf maximising KL(p(x_clean) || p(x_adv))."""
    x0 = x.clone().detach()
    xa = x0 + 0.001 * torch.randn_like(x0)
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


def train_trades(Xtr, Ytr, seed=SEED):
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    alpha = 2.5 * EPS_LINF / PGD_STEPS
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            x_adv = _trades_inner(model, xb, EPS_LINF, PGD_STEPS, alpha)
            model.train()
            opt.zero_grad()
            out_clean = model(xb)
            loss_ce = F.cross_entropy(out_clean, yb)
            log_p_clean = F.log_softmax(out_clean, dim=1).detach()
            p_clean = log_p_clean.exp()
            log_p_adv = F.log_softmax(model(x_adv), dim=1)
            kl = (p_clean * (log_p_clean - log_p_adv)).sum(dim=1).mean()
            (loss_ce + BETA_TRADES * kl).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_mart(Xtr, Ytr, seed=SEED):
    """MART (Wang et al. 2020): boosted CE on adv + KL weighted by
    (1 - p_y) on the clean prediction.

        loss = BCE_boosted(adv, y) + beta * KL(adv || clean) * (1 - p_y_clean)
    """
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    alpha = 2.5 * EPS_LINF / PGD_STEPS
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            xa = C.pgd(model, xb, yb, eps=EPS_LINF, steps=PGD_STEPS, alpha=alpha)
            model.train()
            opt.zero_grad()
            logits_adv = model(xa)
            logits_clean = model(xb)
            p_adv = F.softmax(logits_adv, dim=1)
            # boosted CE: CE + log(1 - max_{j != y} p_j(adv))
            one_hot = F.one_hot(yb, NCLS).float()
            wrong = (1.0 - one_hot) * p_adv
            top_wrong = wrong.max(dim=1).values
            ce_adv = F.cross_entropy(logits_adv, yb)
            boost = -torch.log((1.0 - top_wrong).clamp(min=1e-12)).mean()
            # KL on advs weighted per-sample by (1 - p_y_clean)
            p_clean = F.softmax(logits_clean, dim=1)
            log_p_adv = F.log_softmax(logits_adv, dim=1)
            log_p_clean = F.log_softmax(logits_clean, dim=1)
            per_sample_kl = (p_clean * (log_p_clean - log_p_adv)).sum(dim=1)
            wt = (1.0 - p_clean.gather(1, yb[:, None]).squeeze(1)).detach()
            kl_term = (per_sample_kl * wt).mean()
            loss = ce_adv + boost + BETA_MART * kl_term
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_fgsm_at(Xtr, Ytr, seed=SEED):
    """Single-step FGSM-AT (Goodfellow 2015 / Wong 2020 fast-AT).  Known to
    suffer catastrophic overfitting at this eps - included specifically as
    the masking-suspect defence whose PGD-eps=0.1 ASR is often anomalously
    low for the wrong reason (gradient masking via overfit)."""
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            xa = C.fgsm(model, xb, yb, eps=EPS_LINF)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(xa), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_rfnn(Xtr, Ytr, seed=SEED):
    """Frozen random conv features + trained linear readout.  No GD through
    features at all - the Theme-B "is vulnerability a GD artefact" control.
    """
    C.set_seed(seed)
    model = C.build_model("rfnn", META, width=64, seed=seed)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )
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
    model.eval()
    return model


DEFENCES = [
    ("STD",         train_std),
    ("PGD-AT_e0.1", lambda X, Y, seed=SEED: train_pgd_at(X, Y, seed, EPS_LINF, 7)),
    ("TRADES_b6",   train_trades),
    ("MART",        train_mart),
    ("FGSM-AT",     train_fgsm_at),
    ("RFNN",        train_rfnn),
]


# ===========================================================================
# Evaluation
# ===========================================================================
def eval_clean(model, X, Y):
    _, acc = C.logits_and_acc(model, X, Y)
    return float(acc)


def eval_pgd_linf_asr(model, X, Y):
    out = C.attack_success(model, X, Y, attack="pgd",
                           eps=EPS_LINF, steps=PGD_STEPS)
    return float(out["asr"])


def eval_pgd_l2_asr(model, X, Y, batch=256):
    """ASR (over correctly-classified samples) of L2-PGD at eps_L2=2.0."""
    model.eval()
    flips, corr = [], []
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        with torch.no_grad():
            c = model(xb).argmax(1) == yb
        xa = pgd_l2(model, xb, yb)
        with torch.no_grad():
            fl = model(xa).argmax(1) != yb
        flips.append(fl.cpu()); corr.append(c.cpu())
    flips = torch.cat(flips).numpy()
    corr  = torch.cat(corr).numpy().astype(bool)
    if corr.sum() == 0:
        return float("nan")
    return float(flips[corr].mean())


def eval_cw_l2(model, X, Y, n_max, batch=64):
    """Run CW-L2 on (up to) the first n_max correctly classified samples.

    Returns dict with per-sample arrays:
        l2          (numpy, NaN for non-flipped)
        flipped     (numpy bool)
        y           (numpy int)  - true labels for per-class breakdown
        n_used      int          - number of correctly-classified samples eval'd
    """
    model.eval()
    # find correctly-classified samples first (CW only defined for those)
    with torch.no_grad():
        preds = []
        for i in range(0, X.size(0), 512):
            preds.append(model(X[i:i + 512]).argmax(1))
        preds = torch.cat(preds)
    correct = (preds == Y).nonzero(as_tuple=True)[0]
    if correct.numel() == 0:
        return {"l2": np.array([]), "flipped": np.array([]),
                "y": np.array([]), "n_used": 0}
    idx = correct[: n_max]
    Xc = X[idx]; Yc = Y[idx]

    out_l2 = []
    out_fl = []
    for i in range(0, Xc.size(0), batch):
        xb = Xc[i:i + batch]
        yb = Yc[i:i + batch]
        _, l2, flipped = cw_l2_best_over_c(model, xb, yb)
        out_l2.append(l2.cpu().numpy())
        out_fl.append(flipped.cpu().numpy())
    l2 = np.concatenate(out_l2)
    fl = np.concatenate(out_fl).astype(bool)
    # samples that never flipped get NaN L2 (their inf would poison medians)
    l2 = np.where(fl, l2, np.nan)
    return {"l2": l2, "flipped": fl, "y": Yc.cpu().numpy(),
            "n_used": int(Xc.size(0))}


def per_class_median_l2(l2, y, ncls=NCLS):
    """Returns list of length ncls; NaN for classes with no flipped samples."""
    out = []
    for c in range(ncls):
        m = (y == c) & np.isfinite(l2)
        if m.sum() == 0:
            out.append(float("nan"))
        else:
            out.append(float(np.median(l2[m])))
    return out


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H479  CW-L2 (no eps) as a non-saturating defence-ranking metric")
    out("      Fashion-MNIST, 6-defence ladder")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN}  N_EVAL_PGD={N_EVAL_PGD}  "
        f"N_EVAL_CW={N_EVAL_CW}  EPOCHS={EPOCHS}  LR={LR}  BATCH={BATCH}")
    out(f"        Linf eps={EPS_LINF}  PGD_steps={PGD_STEPS}")
    out(f"        PGD-L2 eps={EPS_L2}  steps={PGD_L2_STEPS}  "
        f"alpha={PGD_L2_ALPHA}")
    out(f"        CW-L2 steps={CW_STEPS}  lr={CW_LR}  "
        f"c_sweep={CW_C_SWEEP}  kappa={CW_KAPPA}")
    out(f"        SEED={SEED}  device={C.DEVICE}")
    out("")
    out("Hypothesis: median CW-L2 (unconstrained min-norm) discriminates the")
    out("ladder MORE finely than PGD-Linf-ASR @ eps=0.1 (which saturates).")
    out("Prior art: Carlini-Wagner 2017 (CW-L2), Croce-Hein 2020 (AutoAttack")
    out("/ APGD-DLR), Brendel-Rauber-Bethge 2019 (DDN decoupling).")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN,
                                        n_eval=N_EVAL_PGD, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    rows = []
    for name, fn in DEFENCES:
        out("-" * 80)
        out(f"[train] {name}")
        out("-" * 80)
        ts = time.time()
        C.set_seed(SEED)
        model = fn(Xtr, Ytr)
        train_s = time.time() - ts

        clean_acc = eval_clean(model, Xte, Yte)
        linf_asr  = eval_pgd_linf_asr(model, Xte, Yte)
        l2_pgd    = eval_pgd_l2_asr(model, Xte, Yte)
        out(f"    clean_acc            = {clean_acc:.4f}   "
            f"train={train_s:.1f}s")
        out(f"    PGD-Linf ASR (e=0.1) = {linf_asr:.4f}   <- the saturated metric")
        out(f"    PGD-L2   ASR (e=2.0) = {l2_pgd:.4f}     <- cheap L2 baseline")

        out(f"    running CW-L2 (steps={CW_STEPS}, c sweep={CW_C_SWEEP}) on "
            f"<= {N_EVAL_CW} correct samples...")
        ts = time.time()
        cw = eval_cw_l2(model, Xte, Yte, n_max=N_EVAL_CW)
        cw_s = time.time() - ts
        n_used = cw["n_used"]
        if n_used == 0:
            out("    [skip] no correctly-classified samples available")
            rows.append({"name": name, "clean": clean_acc, "linf_asr": linf_asr,
                         "l2_pgd_asr": l2_pgd,
                         "cw_succ": float("nan"), "cw_med": float("nan"),
                         "cw_mean": float("nan"),
                         "asr_l2_2": float("nan"), "asr_l2_3": float("nan"),
                         "per_class": [float("nan")] * NCLS,
                         "train_s": train_s, "cw_s": cw_s})
            flush_file()
            continue

        l2 = cw["l2"]; flipped = cw["flipped"]; ylab = cw["y"]
        cw_succ = float(flipped.mean())
        finite = np.isfinite(l2)
        if finite.any():
            cw_med  = float(np.median(l2[finite]))
            cw_mean = float(np.mean(l2[finite]))
        else:
            cw_med = float("nan"); cw_mean = float("nan")
        # CW-ASR at a budget B = fraction of evaluated samples whose CW
        # adv example flipped AND has L2 <= B.
        def asr_at(B):
            ok = flipped & (np.nan_to_num(l2, nan=np.inf) <= B)
            return float(ok.mean())
        asr_l2_2 = asr_at(2.0)
        asr_l2_3 = asr_at(3.0)
        per_class = per_class_median_l2(l2, ylab)

        out(f"    CW samples used      = {n_used}      "
            f"CW time = {cw_s:.1f}s")
        out(f"    CW success rate      = {cw_succ:.4f}  "
            f"(any L2, just flipped)")
        out(f"    CW median L2         = {cw_med:.4f}   "
            f"(NaN if no flips)")
        out(f"    CW mean   L2         = {cw_mean:.4f}")
        out(f"    CW-ASR @ L2 <= 2.0   = {asr_l2_2:.4f}")
        out(f"    CW-ASR @ L2 <= 3.0   = {asr_l2_3:.4f}")
        per_class_str = "  ".join(
            "c{}={:.2f}".format(c, v) if np.isfinite(v) else "c{}=  -- ".format(c)
            for c, v in enumerate(per_class)
        )
        out("    per-class CW-L2 median:")
        out("      " + per_class_str)

        rows.append({
            "name": name, "clean": clean_acc, "linf_asr": linf_asr,
            "l2_pgd_asr": l2_pgd,
            "cw_succ": cw_succ, "cw_med": cw_med, "cw_mean": cw_mean,
            "asr_l2_2": asr_l2_2, "asr_l2_3": asr_l2_3,
            "per_class": per_class, "train_s": train_s, "cw_s": cw_s,
        })
        flush_file()

    # ---- summary matrix ------------------------------------------------
    out("")
    out("=" * 80)
    out("[Summary] per-defence headline metrics")
    out("=" * 80)
    hdr = "{:<14} {:>7} {:>9} {:>9} {:>8} {:>9} {:>9} {:>9} {:>9}".format(
        "model", "clean",
        "Linf_ASR", "L2_PGD_ASR",
        "CW_succ", "CW_medL2", "CW_meanL2",
        "ASR_L2_2", "ASR_L2_3")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<14} {:>7.4f} {:>9.4f} {:>9.4f} {:>8.4f} {:>9.4f} {:>9.4f} "
            "{:>9.4f} {:>9.4f}".format(
                r["name"], r["clean"], r["linf_asr"], r["l2_pgd_asr"],
                r["cw_succ"], r["cw_med"], r["cw_mean"],
                r["asr_l2_2"], r["asr_l2_3"]))

    out("")
    out("=" * 80)
    out("[Per-class CW-L2 median]  rows = defences, cols = Fashion-MNIST class")
    out("=" * 80)
    hdr = "{:<14}".format("model") + " ".join("{:>7}".format("c%d" % c)
                                              for c in range(NCLS))
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        cells = []
        for v in r["per_class"]:
            cells.append("{:>7.3f}".format(v) if np.isfinite(v) else "{:>7}".format("nan"))
        out("{:<14}".format(r["name"]) + " ".join(cells))

    # ---- ranking comparison: CW-L2-median vs PGD-Linf-ASR --------------
    out("")
    out("=" * 80)
    out("[Ranking] CW-L2-median ranking vs PGD-Linf-ASR ranking")
    out("=" * 80)
    names = [r["name"] for r in rows]
    cw_meds  = np.array([r["cw_med"] for r in rows], dtype=float)
    linf_asr = np.array([r["linf_asr"] for r in rows], dtype=float)

    # for CW-L2 high = more robust; for Linf-ASR low = more robust.
    # Spearman on (cw_meds, -linf_asr) so both increase with robustness.
    if spearmanr is not None and np.isfinite(cw_meds).all() and np.isfinite(linf_asr).all():
        rho, p = spearmanr(cw_meds, -linf_asr)
    else:
        rho, p = float("nan"), float("nan")

    # print the two rankings side by side
    order_cw   = np.argsort(-cw_meds)   # descending CW-L2 = most robust first
    order_linf = np.argsort(linf_asr)   # ascending  Linf-ASR = most robust first
    out("rank   by CW-L2 median (most -> least robust)        by PGD-Linf-ASR")
    for k in range(len(names)):
        a = names[order_cw[k]]   if k < len(order_cw)   else "-"
        b = names[order_linf[k]] if k < len(order_linf) else "-"
        av = cw_meds[order_cw[k]]
        bv = linf_asr[order_linf[k]]
        out(f"  {k+1}    {a:<12} (CW-L2 med={av:.3f})   "
            f"{b:<12} (Linf-ASR={bv:.4f})")
    out("")
    out(f"  Spearman rho( CW-L2-median ranking,  -PGD-Linf-ASR ranking ) "
        f"= {rho:+.4f}   p = {p:.3e}")

    # ---- verdict --------------------------------------------------------
    out("")
    out("=" * 80)
    out("[Verdict] does CW-L2 non-saturate where PGD-Linf-ASR saturates?")
    out("=" * 80)
    finite_meds = cw_meds[np.isfinite(cw_meds)]
    if finite_meds.size >= 2:
        cw_range = float(finite_meds.max() - finite_meds.min())
    else:
        cw_range = float("nan")
    out(f"  CW-L2-median range across the 6 defences = {cw_range:.3f}")
    out(f"  Spearman |rho| with PGD-Linf-ASR ranking = {abs(rho):.3f}"
        if rho == rho else "  Spearman |rho| with PGD-Linf-ASR ranking = nan")

    suspicious = any(np.isfinite(r["cw_med"]) and r["cw_med"] > 3.0 for r in rows)

    verdict = None
    if suspicious:
        sus_models = [r["name"] for r in rows
                      if np.isfinite(r["cw_med"]) and r["cw_med"] > 3.0]
        verdict = ("SUSPICIOUS: at least one defence has CW-L2 median > 3.0 "
                   "(" + ", ".join(sus_models) + "). On 28x28 Fashion-MNIST "
                   "that level of L2 is implausible without gradient masking; "
                   "rerun with a non-gradient or transfer attack to confirm.")
    elif np.isfinite(cw_range) and cw_range > 1.0 and (rho == rho) and abs(rho) < 0.7:
        verdict = ("YES: CW-L2 median spans > 1.0 across the ladder AND "
                   "disagrees with PGD-Linf-ASR ranking (|rho| < 0.7). The "
                   "CW-L2 median is the right non-saturating defence-ranking "
                   "metric for the campaign.")
    elif np.isfinite(cw_range) and cw_range > 0.5:
        verdict = ("PARTIAL: CW-L2 median spreads the ladder (range > 0.5) "
                   "but tracks PGD-Linf-ASR ranking (|rho| >= 0.7); useful as "
                   "a secondary check, not strictly necessary.")
    else:
        verdict = ("NO: CW-L2 median range < 0.5 across the ladder. Either "
                   "the CW attack is too weak at these settings (try more "
                   "steps / wider c sweep / DDN) or the ladder is genuinely "
                   "flat in L2.")
    out("")
    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out("  Caveats: single seed (SEED=0); N_EVAL_CW=500; CW_STEPS=60 "
        "(canonical is 1000 but our budget is fixed). A SUSPICIOUS verdict "
        "is just a masking flag; H391 5-signal battery / DDN follow-up "
        "needed to confirm.")
    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
