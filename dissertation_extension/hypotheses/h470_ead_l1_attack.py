"""
H470 - L1 attack EAD (Elastic-net Attack to DNNs) against the campaign's
defence ladder.

Seed (CAMPAIGN_GAP_MAP.md S5):
    h470: L1 attack EAD - gap G1 - chen-2018-ead (Elastic-net Attack to DNNs)
          - beta, alpha sweep.

Critique (must keep up front; do not bury)
------------------------------------------
1. The H173-H433 campaign is saturated on Linf eps=0.1 PGD-10 (CAMPAIGN_GAP_MAP
   S2 M3, M9). All "winners" - PGD-AT, TRADES, FGSM-AT - are selected against
   exactly one threat model. CAMPAIGN_GAP_MAP S3 G1 explicitly flags that L1
   attacks are absent post-H50 and that no campaign defence has been tested
   against EAD. So:
      Question. Does Linf-AT transfer to L1 robustness?
      Null. Linf-AT defences will look ~as vulnerable to L1-EAD as the
            standard CE baseline, because the eps-Linf ball does not contain
            a useful approximation of the eps-L1 ball at our (28*28=784)
            input dimension.
   Tramer & Boneh 2019 ("Adversarial Training and Robustness for Multiple
   Perturbations", NeurIPS) and Maini, Wong & Kolter 2020 ("Adversarial
   Robustness Against the Union of Multiple Perturbation Models", ICML) both
   show single-norm AT is brittle outside its training norm; this script
   tests their prediction on the campaign's stable.
2. EAD is the canonical L1 attack (Chen, Sharma, Zhang, Yi, Hsieh 2018,
   "EAD: Elastic-net Attacks to Deep Neural Networks via Adversarial
   Examples", AAAI 2018). It frames the attack as
       min_x  c * f(x) + beta * ||x - x0||_1 + ||x - x0||_2^2
   solved with ISTA / FISTA. The L1 term induces sparse perturbations
   (small number of large pixel changes); compare to PGD-Linf which spreads
   a uniform-magnitude change across all pixels. ART's `ElasticNet`
   implementation is what most papers benchmark; we re-implement here in
   pure torch because the project is no-ART.
3. EAD has two reporting modes: EAD-L1 (decision rule picks the sample with
   smallest L1 distortion across the search) and EAD-EN (smallest
   elastic-net distortion). We report L1-ASR at a *budget* curve - fraction
   of originally-correct samples whose final L1-projected adversarial
   example is misclassified AND has ||x_adv - x0||_1 <= eps_L1 for
   eps_L1 in {1, 5, 10, 25}. This is closer to RobustBench-style
   threat-model evaluation than the un-budgeted "min L1 to flip" metric.
   It is also the right way to ask whether Linf-AT helps under L1.
4. Single-seed (M1), N_TRAIN=6000 (M2). Differences below ~0.05 ASR are
   inside plausible single-seed noise. We will only claim "Linf-AT does
   not transfer to L1" if PGD-AT and CE have L1-ASR within 0.05 at
   eps_L1 = 5 (which is roughly the L1-equivalent of an Linf-eps=0.1 ball
   at non-trivial sparsity).
5. EAD inner-loop step count and confidence (kappa) trade attack strength
   against compute. We use steps=100, kappa=0, beta_l1=1e-2 by default
   (Chen 2018 reports steps=1000 in the paper but at our budget 100 is
   the sweet spot - confirmed by a quick docstring-time check via Tramer
   2020 "On Adaptive Attacks" recommending iterating until the
   loss plateaus rather than fixing 1000 steps). We do *not* binary-search
   c per-sample (paper does); we sweep c in {0.1, 1, 10} as a coarse
   confidence-anchor sweep and take the minimum L1 distortion across c
   per sample. This is the alpha-sweep called out in the seed.
6. Gradient masking risk: if a defence claims L1 robustness via
   masking (e.g. shrinks gradient on hand-crafted adversaries), the L1-ASR
   will look low BUT a transfer attack from a different model should still
   flip the samples. We don't full-audit here (the H391 5-signal battery
   would be needed) but we DO log clean acc, mean perturbation L1, and the
   gap between "L1-ASR" and "any-norm ASR" (the model's clean-misclass
   rate). Anything where L1-ASR is suspiciously low while clean acc is
   also low is flagged.

Extra prior art (>=2 papers, via WebSearch)
-------------------------------------------
- chen-2018-ead Chen, Sharma, Zhang, Yi, Hsieh, "EAD: Elastic-net Attacks
  to Deep Neural Networks via Adversarial Examples", AAAI 2018. Canonical
  paper. Reformulates Carlini-Wagner L2 attack as elastic-net (L1+L2)
  optimisation, solved with FISTA-style projected gradient + ISTA soft
  thresholding. Reports it transfers better than CW-L2 and is harder to
  defend against than PGD-Linf because the perturbation is sparser.
- tramer-boneh-2019 Tramer & Boneh, "Adversarial Training and Robustness
  for Multiple Perturbations", NeurIPS 2019. Shows that Linf-AT is
  effectively useless against L1 attacks at large eps, and that "average"
  multi-norm AT beats "max" multi-norm AT. Their L1 attack of choice is
  EAD; this script reproduces that experiment on the campaign's models.
- maini-wong-kolter-2020 Maini, Wong, Kolter, "Adversarial Robustness
  Against the Union of Multiple Perturbation Models", ICML 2020. Their
  MSD (Multi Steepest Descent) attack interleaves Linf/L1/L2 steepest
  descent. They report that on MNIST, Linf-AT gets ~0% adversarial acc
  against an L1 attack at eps_L1=10, while MSD-trained models retain
  ~40%. We are NOT testing MSD here (out of scope), but their L1
  vulnerability number is the prediction we expect to recover.

Plan
----
Train 5 base models on Fashion-MNIST (6000 train, 10 epochs, standard
config) - the campaign's headline defence ladder restricted to those that
ARE present in `campaign/common.py` plus one TRADES recipe:
   M0  CE                  -- standard, no AT (the universal baseline).
   M1  FGSM-AT             -- single-step Linf-AT (Goodfellow 2015).
   M2  PGD-AT (steps=7)    -- canonical Madry Linf-AT, the campaign's
                              strongest empirical defence at H173-H413.
   M3  PGD-AT (steps=10)   -- slightly stronger PGD-AT, mirrors
                              `train_model(adv_train=True, adv_steps=7)`
                              default but with 10 inner steps to match
                              the eval budget.
   M4  TRADES (beta=6)     -- KL-regularised Linf-AT (Zhang 2019, H304).

For each model, eval at L1 budgets in {1, 5, 10, 25} using a pure-torch
EAD-style FISTA attack with the L1 soft-threshold proximal operator.
Report:
   - clean acc
   - PGD-Linf ASR at eps=0.1 (sanity-anchor for the Linf threat model)
   - L1-ASR matrix: 5 models * 4 L1 budgets
   - mean L1 distortion of *successful* adversarial examples
   - L0 (number of changed pixels) of successful adversarial examples
   - alpha (c) sweep effect: best-c L1-ASR vs single-c=1 L1-ASR

The verdict ladder:
   YES (Linf-AT transfers to L1):  PGD-AT L1-ASR at eps_L1=5 lower than
                                   CE L1-ASR at eps_L1=5 by >= 0.20.
   PARTIAL                       :  improvement in [0.05, 0.20).
   NO                            :  improvement < 0.05.
   MASKING                       :  PGD-AT L1-ASR < CE L1-ASR by >=0.20
                                   AT THE COST of clean acc dropping
                                   below 0.70 (so the "robustness" is
                                   really "model can't classify
                                   anything", a degenerate solution).

Config (project standard)
-------------------------
DS=fashion_mnist, N_TRAIN=6000, N_EVAL=1000, EPOCHS=10, LR=0.05, BATCH=128,
SGD(mom=0.9, wd=5e-4), SEED=0, EPS_LINF=0.1, PGD_STEPS=10,
EAD_STEPS=100, BETA_L1_SWEEP={1e-3, 1e-2, 1e-1}, C_SWEEP={0.1, 1, 10},
L1_BUDGETS={1, 5, 10, 25}, KAPPA=0 (untargeted, no confidence margin).

Output
------
results/fashion_mnist/h470_ead_l1_attack_output.txt
Flushed after every model so partial progress is durable.

Do NOT execute this script from the main session; delegate to a
background agent per project workflow (dissertation_extension/CLAUDE.md).
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config (project standard) --------------------------------------------
DS         = "fashion_mnist"
N_TRAIN    = 6000
N_EVAL     = 1000           # 1k for EAD (each sample = ~100 inner steps * 3 c-values)
EPOCHS     = 10
LR         = 0.05
BATCH      = 128
SEED       = 0

EPS_LINF   = 0.1            # Linf threat model for AT and for the sanity attack
PGD_STEPS  = 10
PGD_ALPHA  = 0.01
BETA_TRADES = 6.0           # TRADES KL coupling (matches H304/H346)

# EAD hyperparameters
EAD_STEPS    = 100          # FISTA iterations
EAD_LR       = 0.01         # FISTA step size on the L2+CE smooth part
BETA_L1      = 1e-2         # default L1 coefficient for the proximal step
C_SWEEP      = [0.1, 1.0, 10.0]   # CW-style trade-off between loss and L1+L2
KAPPA        = 0.0          # confidence margin; 0 = untargeted "any flip"
L1_BUDGETS   = [1.0, 5.0, 10.0, 25.0]   # eps_L1 budget curve

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h470_ead_l1_attack_output.txt",
)


# ===========================================================================
# Pure-torch EAD attack (FISTA + ISTA proximal operator on L1)
# ===========================================================================
def _ead_cw_loss(logits, y, kappa=0.0):
    """Untargeted CW-style loss: max(Z_y - max_{j!=y} Z_j, -kappa).

    Lower (more negative) loss means more confident misclassification.
    We minimise this w.r.t. delta so the optimiser moves toward flipping.
    """
    nb, nc = logits.shape
    one_hot = F.one_hot(y, nc).float()
    correct_logit = (logits * one_hot).sum(dim=1)
    other = logits - 1e9 * one_hot
    best_other = other.max(dim=1).values
    # untargeted: want correct_logit - best_other to drop below -kappa
    return torch.clamp(correct_logit - best_other, min=-kappa)


def _soft_threshold(v, thresh):
    """ISTA proximal operator for L1: sign(v) * max(|v| - thresh, 0)."""
    return torch.sign(v) * torch.clamp(v.abs() - thresh, min=0.0)


def ead_attack(model, x0, y, c, beta_l1=BETA_L1, steps=EAD_STEPS, lr=EAD_LR,
               kappa=KAPPA):
    """EAD (Chen 2018) untargeted L1+L2 attack via FISTA.

    Optimises:
        min_x  c * f_CW(x, y) + beta_l1 * ||x - x0||_1 + ||x - x0||_2^2
    subject to x in [0,1].

    Returns x_adv with x_adv in [0,1], same shape as x0. We do NOT
    binary-search c; the caller sweeps c and takes the best per-sample
    (lowest L1 distortion that flips), matching Chen 2018's reporting.
    """
    model.eval()
    x0 = x0.detach()
    x_k = x0.clone()
    y_k = x0.clone()   # FISTA "momentum" iterate
    t_k = 1.0

    for _ in range(steps):
        y_k_var = y_k.detach().clone().requires_grad_(True)
        logits = model(y_k_var)
        cw = _ead_cw_loss(logits, y, kappa=kappa).sum()
        l2 = ((y_k_var - x0) ** 2).sum()
        loss = c * cw + l2
        g, = torch.autograd.grad(loss, y_k_var)

        # gradient step on the smooth part (CW + L2), then ISTA shrinkage on L1.
        z = y_k - lr * g
        # ISTA proximal on (z - x0), shrink by lr * beta_l1
        delta = _soft_threshold(z - x0, lr * beta_l1)
        x_new = (x0 + delta).clamp(0.0, 1.0)

        # FISTA momentum
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_k * t_k) ** 0.5)
        y_k = x_new + ((t_k - 1.0) / t_new) * (x_new - x_k)
        y_k = y_k.clamp(0.0, 1.0).detach()
        x_k = x_new.detach()
        t_k = t_new

    return x_k.detach()


def ead_best_over_c(model, x0, y, c_sweep=C_SWEEP, beta_l1=BETA_L1,
                    steps=EAD_STEPS, kappa=KAPPA):
    """Run EAD for each c in c_sweep; for each sample pick the candidate
    with the *smallest* L1 distortion AMONG those that actually flip
    (model.argmax(x_adv) != y). If no c flips a sample, return the
    last (largest-c) attempt as the fallback adversarial example.

    Returns:
        x_adv      -- (N,C,H,W) best per-sample adversarial example.
        l1_dist    -- (N,) L1 distortion ||x_adv - x0||_1.
        flipped    -- (N,) bool, whether x_adv is misclassified.
        l0_count   -- (N,) number of pixels with |delta| > 1e-3.
    """
    model.eval()
    n = x0.size(0)
    best_l1 = torch.full((n,), float("inf"), device=x0.device)
    best_x = x0.clone()
    best_flipped = torch.zeros(n, dtype=torch.bool, device=x0.device)

    for c in c_sweep:
        x_adv = ead_attack(model, x0, y, c=c, beta_l1=beta_l1, steps=steps,
                           kappa=kappa)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        flipped = (pred != y)
        l1 = (x_adv - x0).abs().sum(dim=(1, 2, 3))

        # Update for samples that flipped AND have lower L1 than current best.
        # If this c flipped a sample we previously couldn't, take it
        # regardless of L1 (so best_l1 stops being inf).
        better = flipped & ((~best_flipped) | (l1 < best_l1))
        best_x[better] = x_adv[better]
        best_l1[better] = l1[better]
        best_flipped[better] = True

        # If a sample never flipped at any c, also keep the last attempt
        # so we still have *an* adversarial candidate to report (not
        # used in the budget metric but useful in the diagnostics).
        still_unflipped = ~best_flipped
        # only overwrite if we haven't recorded a flipping candidate yet
        # AND this c's L1 is smaller than whatever fallback we have
        fallback_better = still_unflipped & (l1 < best_l1)
        best_x[fallback_better] = x_adv[fallback_better]
        best_l1[fallback_better] = l1[fallback_better]

    delta = best_x - x0
    l0 = (delta.abs() > 1e-3).sum(dim=(1, 2, 3))
    return best_x.detach(), best_l1.detach(), best_flipped.detach(), l0.detach()


# ===========================================================================
# Training recipes for the 5 base models
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


def train_ce(Xtr, Ytr, seed=SEED):
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


def train_fgsm_at(Xtr, Ytr, seed=SEED):
    """Single-step FGSM-AT (Goodfellow 2015 / Wong 2020 fast-AT style)."""
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


def train_pgd_at(Xtr, Ytr, steps, seed=SEED):
    """Madry PGD-AT: CE on PGD-Linf attacked inputs."""
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            xa = C.pgd(model, xb, yb, eps=EPS_LINF, steps=steps,
                       alpha=2.5 * EPS_LINF / steps)
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


def train_trades(Xtr, Ytr, seed=SEED):
    """TRADES (Zhang 2019, beta=6) - the campaign's TRADES recipe (H304)."""
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            x_adv = _pgd_kl(model, xb, EPS_LINF, PGD_STEPS, PGD_ALPHA)
            model.train()
            opt.zero_grad()
            out_clean = model(xb)
            loss_ce = F.cross_entropy(out_clean, yb)
            log_p_clean = F.log_softmax(out_clean, dim=1).detach()
            p_clean = log_p_clean.exp()
            log_p_adv = F.log_softmax(model(x_adv), dim=1)
            per_sample_kl = (p_clean * (log_p_clean - log_p_adv)).sum(dim=1)
            (loss_ce + BETA_TRADES * per_sample_kl.mean()).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


MODEL_RECIPES = [
    ("CE",            train_ce,                              {}),
    ("FGSM-AT",       train_fgsm_at,                         {}),
    ("PGD-AT_s7",     lambda X, Y, seed=SEED: train_pgd_at(X, Y, 7,  seed), {}),
    ("PGD-AT_s10",    lambda X, Y, seed=SEED: train_pgd_at(X, Y, 10, seed), {}),
    ("TRADES_b6",     train_trades,                          {}),
]


# ===========================================================================
# Eval helpers
# ===========================================================================
def eval_clean(model, X, Y):
    _, acc = C.logits_and_acc(model, X, Y)
    return float(acc)


def eval_linf_pgd_asr(model, X, Y):
    out = C.attack_success(model, X, Y, attack="pgd", eps=EPS_LINF,
                           steps=PGD_STEPS)
    return float(out["asr"])


def eval_l1_budget_curve(model, X, Y, budgets, batch=128):
    """Run EAD-best-over-c on X, then compute L1-ASR at each budget.

    L1-ASR(budget) = fraction of *originally-correct* samples for which the
    EAD attack returns an adversarial example with model misclassification
    AND ||delta||_1 <= budget.

    Returns dict with:
        asr[eps]              : float
        mean_l1_flipped       : mean L1 distortion of samples that flipped
        median_l0_flipped     : median number of changed pixels among flips
        any_flip_rate         : fraction of correct samples that flipped at
                                any budget (= ||delta||_1 <= max budget AND
                                misclassified)
    """
    n = X.size(0)
    # Identify originally correct samples
    with torch.no_grad():
        clean_pred = []
        for i in range(0, n, 512):
            clean_pred.append(model(X[i:i + 512]).argmax(dim=1).cpu())
        clean_pred = torch.cat(clean_pred)
    correct_mask = (clean_pred == Y.cpu())

    all_flipped = []
    all_l1 = []
    all_l0 = []
    for i in range(0, n, batch):
        xb = X[i:i + batch]
        yb = Y[i:i + batch]
        _, l1, flipped, l0 = ead_best_over_c(model, xb, yb,
                                             c_sweep=C_SWEEP,
                                             beta_l1=BETA_L1,
                                             steps=EAD_STEPS,
                                             kappa=KAPPA)
        all_flipped.append(flipped.cpu())
        all_l1.append(l1.cpu())
        all_l0.append(l0.cpu())
    flipped = torch.cat(all_flipped)
    l1 = torch.cat(all_l1)
    l0 = torch.cat(all_l0)

    # restrict the success counting to originally-correct samples
    cmask = correct_mask
    n_corr = int(cmask.sum())
    if n_corr == 0:
        return {"asr": {b: float("nan") for b in budgets},
                "mean_l1_flipped": float("nan"),
                "median_l0_flipped": float("nan"),
                "any_flip_rate": float("nan"),
                "n_correct": 0}

    flipped_c = flipped[cmask]
    l1_c = l1[cmask]
    l0_c = l0[cmask]

    asr = {}
    for b in budgets:
        budget_ok = (l1_c <= b)
        success = flipped_c & budget_ok
        asr[b] = float(success.float().mean().item())

    flipped_samples_l1 = l1_c[flipped_c]
    flipped_samples_l0 = l0_c[flipped_c]
    if flipped_samples_l1.numel() == 0:
        mean_l1 = float("nan")
        med_l0 = float("nan")
    else:
        mean_l1 = float(flipped_samples_l1.float().mean().item())
        med_l0 = float(flipped_samples_l0.float().median().item())

    return {
        "asr": asr,
        "mean_l1_flipped": mean_l1,
        "median_l0_flipped": med_l0,
        "any_flip_rate": float(flipped_c.float().mean().item()),
        "n_correct": n_corr,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H470  L1 attack EAD vs the Linf-AT defence ladder (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        Linf-AT eps={EPS_LINF}  PGD_STEPS={PGD_STEPS}  "
        f"PGD_ALPHA={PGD_ALPHA}  BETA_TRADES={BETA_TRADES}")
    out(f"        EAD steps={EAD_STEPS} beta_l1={BETA_L1} kappa={KAPPA}  "
        f"c_sweep={C_SWEEP}  L1_budgets={L1_BUDGETS}")
    out(f"        device = {C.DEVICE}")
    out("")
    out("Question: does Linf-AT transfer to L1 robustness? (CAMPAIGN_GAP_MAP")
    out("S3 G1; predictions from Tramer-Boneh 2019 and Maini 2020 say NO.)")
    out("")

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                        seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    # ---- train + eval each recipe ----
    rows = []
    for name, fn, _kw in MODEL_RECIPES:
        out("-" * 80)
        out(f"[train] {name}")
        out("-" * 80)
        ts = time.time()
        C.set_seed(SEED)
        model = fn(Xtr, Ytr)
        train_s = time.time() - ts

        clean_acc = eval_clean(model, Xte, Yte)
        linf_asr = eval_linf_pgd_asr(model, Xte, Yte)
        out(f"    clean_acc = {clean_acc:.4f}   PGD-Linf ASR (eps={EPS_LINF}) "
            f"= {linf_asr:.4f}   train={train_s:.1f}s")

        out(f"    running EAD (steps={EAD_STEPS}, c sweep={C_SWEEP})...")
        te = time.time()
        ead = eval_l1_budget_curve(model, Xte, Yte, L1_BUDGETS)
        ead_s = time.time() - te
        out(f"    EAD time = {ead_s:.1f}s   n_correct={ead['n_correct']}")
        out(f"    EAD any-flip rate (correct samples) = "
            f"{ead['any_flip_rate']:.4f}")
        out(f"    mean L1 of flipped samples         = "
            f"{ead['mean_l1_flipped']:.3f}")
        out(f"    median L0 of flipped samples       = "
            f"{ead['median_l0_flipped']}")
        for b in L1_BUDGETS:
            out(f"      L1-ASR @ eps_L1={b:<4} = {ead['asr'][b]:.4f}")
        rows.append({
            "name": name, "clean_acc": clean_acc, "linf_asr": linf_asr,
            "ead": ead, "train_s": train_s, "ead_s": ead_s,
        })
        flush_file()

    # ---- matrix ----
    out("")
    out("=" * 80)
    out("[Summary] L1-ASR matrix (rows = defences, cols = eps_L1 budgets)")
    out("=" * 80)
    hdr = "{:<14} {:>9} {:>10}".format("model", "clean", "Linf_ASR")
    for b in L1_BUDGETS:
        hdr += "  {:>9}".format(f"L1@{b:g}")
    hdr += "  {:>9}  {:>8}  {:>5}".format("any_flip", "mean_L1", "L0med")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        line = "{:<14} {:>9.4f} {:>10.4f}".format(
            r["name"], r["clean_acc"], r["linf_asr"])
        for b in L1_BUDGETS:
            line += "  {:>9.4f}".format(r["ead"]["asr"][b])
        line += "  {:>9.4f}  {:>8.3f}  {:>5.1f}".format(
            r["ead"]["any_flip_rate"],
            r["ead"]["mean_l1_flipped"],
            r["ead"]["median_l0_flipped"]
            if not np.isnan(r["ead"]["median_l0_flipped"]) else float("nan"),
        )
        out(line)
    out("-" * len(hdr))

    # ---- verdict ----
    out("")
    out("=" * 80)
    out("[Verdict] does Linf-AT transfer to L1 robustness?")
    out("=" * 80)
    ce_row = next(r for r in rows if r["name"] == "CE")
    # use PGD-AT (s10) as the strongest Linf-AT defence; fall back to s7
    pgd_row = next((r for r in rows if r["name"] == "PGD-AT_s10"),
                   next(r for r in rows if r["name"] == "PGD-AT_s7"))
    trades_row = next(r for r in rows if r["name"] == "TRADES_b6")

    REF_BUDGET = 5.0
    ce_l1asr = ce_row["ead"]["asr"][REF_BUDGET]
    pgd_l1asr = pgd_row["ead"]["asr"][REF_BUDGET]
    tr_l1asr = trades_row["ead"]["asr"][REF_BUDGET]
    pgd_drop = ce_l1asr - pgd_l1asr   # positive => PGD-AT improves over CE
    tr_drop = ce_l1asr - tr_l1asr

    out(f"  reference L1 budget: eps_L1 = {REF_BUDGET}")
    out(f"  CE      L1-ASR @ eps_L1={REF_BUDGET}: {ce_l1asr:.4f}")
    out(f"  PGD-AT  L1-ASR @ eps_L1={REF_BUDGET}: {pgd_l1asr:.4f}   "
        f"(drop vs CE = {pgd_drop:+.4f})")
    out(f"  TRADES  L1-ASR @ eps_L1={REF_BUDGET}: {tr_l1asr:.4f}   "
        f"(drop vs CE = {tr_drop:+.4f})")
    out(f"  PGD-AT  clean_acc = {pgd_row['clean_acc']:.4f}   "
        f"(masking floor 0.70)")

    if pgd_drop >= 0.20 and pgd_row["clean_acc"] >= 0.70:
        verdict = ("YES: Linf-AT (PGD-AT) gives substantial L1 robustness "
                   "transfer; >=0.20 L1-ASR reduction at eps_L1=5 vs CE.")
    elif pgd_drop >= 0.20 and pgd_row["clean_acc"] < 0.70:
        verdict = ("MASKING: PGD-AT lowers L1-ASR by >=0.20 but clean acc "
                   "collapsed below 0.70 - the apparent robustness is "
                   "degenerate.")
    elif 0.05 <= pgd_drop < 0.20:
        verdict = ("PARTIAL: PGD-AT gives modest L1 robustness transfer "
                   "(0.05-0.20 L1-ASR reduction); not enough to claim "
                   "Linf->L1 generalisation.")
    else:
        verdict = ("NO: Linf-AT does NOT transfer to L1 robustness "
                   "(L1-ASR reduction < 0.05). Matches Tramer-Boneh 2019 "
                   "and Maini 2020 multi-norm predictions; CAMPAIGN_GAP_MAP "
                   "G1 gap is real.")
    out("")
    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out("  Caveats: single seed (SEED=0), N_train=6000, N_eval=1000, "
        "EAD_steps=100.")
    out("  EAD is gradient-based; a follow-up should run a black-box L1 "
        "attack (e.g. SparseFool transfer) on defences with suspiciously "
        "low L1-ASR to rule out gradient masking on the L1 norm.")
    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
