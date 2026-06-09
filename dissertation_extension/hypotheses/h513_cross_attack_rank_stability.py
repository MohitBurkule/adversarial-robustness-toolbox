"""
H513 - Cross-attack defense-ranking stability.

Seed (M12 / §5 / §2): the literature consensus on "which defense is best" depends
heavily on how robustness is measured. RobustBench (Croce et al. 2021) freezes
the threat model to AutoAttack at a single eps to make rankings comparable
exactly because rankings change between attacks. Carlini et al. ("On Evaluating
Adversarial Robustness", arXiv:1902.06705) warn that any robustness number
quoted under a single (attack, eps) is uninformative without a sweep. Tramer
et al. ("On Adaptive Attacks to Adversarial Example Defenses", NeurIPS 2020)
shows defense rankings collapse under attack-specific adaptive evaluation.

Hypothesis: the ordering of 8 defenses by robust accuracy SWAPS significantly
between the six evaluation conditions
    (PGD@eps=0.05, PGD@eps=0.1, PGD@eps=0.2, AutoAttack-lite, Square, CW-L2).
Concretely: Spearman rho between ranking pairs is below 0.7 for at least 3 of
the 15 pairwise comparisons, and the top-ranked defense at PGD@eps=0.1 is NOT
the top-ranked defense at PGD@eps=0.2 or under CW-L2.

If true: any paper quoting "defense X is the best" under a single (attack, eps)
is not credible -- ranking is a function of the eval threat model, not of a
defense's intrinsic robustness. This is exactly the RobustBench critique.

Defenses (8, all small SmallCNN, same data / seed / epochs to keep them
on the same footing):

  D1  STD             -- standard cross-entropy
  D2  FGSM-AT         -- Goodfellow 2015 single-step adversarial training
  D3  PGD-AT (mid)    -- Madry et al. 2018, eps=0.1, 7 steps
  D4  TRADES          -- Zhang et al. ICML 2019, beta=6, KL on KL-PGD
  D5  MART-ls         -- MART-style label-smoothed AT (Wang et al. ICLR 2020)
  D6  PGD-AT (high)   -- Madry-style AT trained at eps=0.2 instead of 0.1
  D7  PGD-AT (low)    -- Madry-style AT trained at eps=0.05
  D8  Jac-Frob-AT     -- Hoffman et al. 2019 Jacobian-Frobenius regularised CE

Attacks (6):

  A1  PGD@eps=0.05  (Madry 2018; 20 steps; random restart)
  A2  PGD@eps=0.10  (campaign reference budget)
  A3  PGD@eps=0.20  (above all defense training budgets except D6)
  A4  AutoAttack-lite = max-over-{PGD-CE-20, APGD-DLR-20}  (Croce 2020 spirit)
  A5  Square (gradient-free, Andriushchenko 2020) at eps=0.1, 400 queries
  A6  CW-L2  (Carlini-Wagner L2, 30 steps, c=1.0)

The 8x6 robust-accuracy matrix is the headline artefact; we then:

  (1) report per-attack robust accuracy for each defense,
  (2) compute Spearman rho between every pair of attack columns (15 pairs),
  (3) report per-attack top-3 defenses,
  (4) flag any defense whose rank changes by > 3 between any two attacks,
  (5) issue a HEADLINE verdict: SUPPORTED / PARTIAL / REJECTED.

Extra papers (cited):
  * Croce et al. "RobustBench: a standardized adversarial robustness
    benchmark" (NeurIPS Datasets & Benchmarks 2021; arXiv:2010.09670) -- the
    project exists because ranking changes between attacks; AA is canonical.
  * Carlini, Athalye et al. "On Evaluating Adversarial Robustness"
    (arXiv:1902.06705) -- explicit warning: single-(attack, eps) numbers are
    misleading; always sweep.
  * Tramer, Carlini, Brendel, Madry "On Adaptive Attacks to Adversarial
    Example Defenses" (NeurIPS 2020; arXiv:2002.08347) -- rankings change
    completely once attacks are tailored per defense.

Critique handled: re-training 8 defenses on a single small CNN within the
same budget (10 epochs, 6000 train samples). We deliberately use *fast*
versions of every defense -- the goal is RANK STABILITY, not absolute SOTA
robust accuracy.

This file ONLY defines the experiment; no execution.
Pure torch + numpy + scipy.stats.spearmanr. ASCII only. Flushes per defense.
"""
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 1000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0

EPS_LOW = 0.05
EPS_MID = 0.10
EPS_HIGH = 0.20

PGD_STEPS_EVAL = 20
PGD_STEPS_TRAIN = 7
APGD_ITERS = 20             # APGD-DLR for AA-lite
SQUARE_QUERIES = 400        # gradient-free budget
CW_STEPS = 30
CW_C = 1.0
CW_LR = 0.01

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
OUT_FILE = os.path.join(OUT_DIR, "h513_cross_attack_rank_stability_output.txt")

_FH = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if _FH is not None:
        _FH.write(s + "\n")
        _FH.flush()


# ---------------------------------------------------------------------------
# Defenses (8). Each takes (meta, Xtr, Ytr) and returns a trained SmallCNN.
# All use the same shared opt/sched recipe so differences come from the loss.
# ---------------------------------------------------------------------------
def _opt_sched(model, epochs=EPOCHS):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    return opt, sched


def _iter_batches(Xtr, Ytr, batch=BATCH):
    n = Xtr.size(0)
    perm = torch.randperm(n, device=Xtr.device)
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        yield Xtr[idx], Ytr[idx]


# D1 -- standard CE
def train_std(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt, sched = _opt_sched(model)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in _iter_batches(Xtr, Ytr):
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# D2 -- FGSM-AT (Goodfellow 2015)
def train_fgsm_at(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt, sched = _opt_sched(model)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in _iter_batches(Xtr, Ytr):
            xb_adv = C.fgsm(model, xb, yb, eps=EPS_MID)
            opt.zero_grad()
            F.cross_entropy(model(xb_adv), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# D3 -- PGD-AT mid (eps=0.10)
def train_pgd_at_mid(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                         opt="sgd", lr=LR, adv_train=True,
                         adv_eps=EPS_MID, adv_steps=PGD_STEPS_TRAIN)


# D4 -- TRADES (beta=6) -- KL-PGD inner loop, beta * KL clean||adv penalty
def _pgd_on_kl(model, x, steps=10, eps=EPS_MID, alpha=0.01):
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)
    x_adv = (x.detach() + 0.001 * torch.randn_like(x)).clamp(0, 1)
    x0 = x.detach()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        kl = F.kl_div(F.log_softmax(model(x_adv), dim=1), p_clean,
                      reduction='batchmean')
        g, = torch.autograd.grad(kl, x_adv)
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1)
    return x_adv.detach()


def train_trades(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt, sched = _opt_sched(model)
    beta = 6.0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in _iter_batches(Xtr, Ytr):
            x_adv = _pgd_on_kl(model, xb, steps=10, eps=EPS_MID, alpha=0.01)
            model.train()
            opt.zero_grad()
            out_clean = model(xb)
            out_adv = model(x_adv)
            loss_ce = F.cross_entropy(out_clean, yb)
            loss_kl = F.kl_div(F.log_softmax(out_adv, dim=1),
                               F.softmax(out_clean, dim=1),
                               reduction='batchmean')
            (loss_ce + beta * loss_kl).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# D5 -- MART-style label-smoothed AT
# Following Wang et al. ICLR 2020 (MART), we use a BCE-like loss on the adv
# example that up-weights misclassified samples; here we use the simpler
# label-smoothed variant of PGD-AT which captures MART's "softer label"
# spirit while staying within budget.
def train_mart_ls(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                         opt="sgd", lr=LR,
                         adv_train=True, adv_eps=EPS_MID,
                         adv_steps=PGD_STEPS_TRAIN,
                         label_smooth=0.1)


# D6 -- PGD-AT high (eps=0.20)
def train_pgd_at_high(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                         opt="sgd", lr=LR, adv_train=True,
                         adv_eps=EPS_HIGH, adv_steps=PGD_STEPS_TRAIN)


# D7 -- PGD-AT low (eps=0.05)
def train_pgd_at_low(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                         opt="sgd", lr=LR, adv_train=True,
                         adv_eps=EPS_LOW, adv_steps=PGD_STEPS_TRAIN)


# D8 -- Jacobian-Frobenius regularised standard training (Hoffman 2019)
def _jac_frob_pen(model, xb, k=4):
    out = model(xb)
    pen = 0.0
    for _ in range(k):
        v = torch.randn(xb.size(0), out.size(1), device=xb.device)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        xb_g = xb.clone().detach().requires_grad_(True)
        out_g = model(xb_g)
        proj = (out_g * v).sum()
        g, = torch.autograd.grad(proj, xb_g, create_graph=True)
        pen = pen + (g ** 2).sum(dim=(1, 2, 3)).mean()
    return pen / k


def train_jac_frob(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt, sched = _opt_sched(model)
    lam = 0.01
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in _iter_batches(Xtr, Ytr):
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb)
            jp = _jac_frob_pen(model, xb, k=4)
            (ce + lam * jp).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


DEFENSES = [
    ("STD",          train_std),
    ("FGSM-AT",      train_fgsm_at),
    ("PGD-AT@0.10",  train_pgd_at_mid),
    ("TRADES",       train_trades),
    ("MART-ls",      train_mart_ls),
    ("PGD-AT@0.20",  train_pgd_at_high),
    ("PGD-AT@0.05",  train_pgd_at_low),
    ("JacFrob-AT",   train_jac_frob),
]


# ---------------------------------------------------------------------------
# Attacks.  Each returns ROBUST ACCURACY = fraction of N_EVAL samples whose
# label survives the attack (NOT the ASR; this is the natural quantity for
# *ranking* defenses).  All operate in [0,1] and Linf-clamp.
# ---------------------------------------------------------------------------
@torch.no_grad()
def clean_acc(model, X, Y, batch=512):
    correct = 0
    for i in range(0, X.size(0), batch):
        out = model(X[i:i + batch])
        correct += int((out.argmax(1) == Y[i:i + batch]).sum().item())
    return correct / X.size(0)


def robust_acc_pgd(model, X, Y, eps, steps=PGD_STEPS_EVAL, batch=256):
    model.eval()
    correct = 0
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        xa = C.pgd(model, x, y, eps=eps, steps=steps,
                   alpha=2.5 * eps / steps, random_start=True)
        with torch.no_grad():
            correct += int((model(xa).argmax(1) == y).sum().item())
    return correct / X.size(0)


# ---- APGD-DLR (Croce 2020) simplified, single restart -----------------------
def _dlr_loss(logits, y):
    z_sorted, _ = logits.sort(dim=1, descending=True)
    z_clone = logits.clone()
    z_clone[torch.arange(z_clone.size(0)), y] = -1e9
    z_other = z_clone.max(1).values
    z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
    pi1 = z_sorted[:, 0]
    pi3 = z_sorted[:, 2] if logits.size(1) > 2 else z_sorted[:, -1]
    return -(z_y - z_other) / (pi1 - pi3 + 1e-12)


def apgd_dlr_attack(model, X, Y, eps=EPS_MID, n_iter=APGD_ITERS, batch=256):
    """One-restart APGD-DLR. Returns adv tensor."""
    model.eval()
    out_advs = []
    for i in range(0, X.size(0), batch):
        x_orig = X[i:i + batch]
        y = Y[i:i + batch]
        x = (x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)).clamp(0, 1)
        x_prev = x.clone()
        alpha = 2.0 * eps
        x_best = x.clone()
        loss_best = None
        for k in range(n_iter):
            x_in = x.clone().detach().requires_grad_(True)
            logits = model(x_in)
            L_per = _dlr_loss(logits, y)
            g, = torch.autograd.grad(L_per.sum(), x_in)
            with torch.no_grad():
                z = (x + alpha * g.sign())
                z = torch.min(torch.max(z, x_orig - eps), x_orig + eps).clamp(0, 1)
                x_new = x + 0.75 * (z - x) + 0.25 * (x - x_prev)
                x_new = torch.min(torch.max(x_new, x_orig - eps),
                                  x_orig + eps).clamp(0, 1)
                x_prev = x.clone()
                x = x_new
                if loss_best is None:
                    loss_best = L_per.detach().clone()
                    x_best = x.clone()
                else:
                    better = L_per.detach() > loss_best
                    if better.any():
                        loss_best = torch.where(better, L_per.detach(), loss_best)
                        x_best[better] = x[better]
                # halve step at half-way (very lightweight schedule)
                if k == n_iter // 2:
                    alpha = alpha / 2.0
                    x = x_best.clone()
                    x_prev = x_best.clone()
        out_advs.append(x_best.detach())
    return torch.cat(out_advs, 0)


def robust_acc_aa_lite(model, X, Y, eps=EPS_MID):
    """AutoAttack-lite: a defense must survive BOTH PGD-CE-20 AND APGD-DLR-20.

    We treat AA-lite as max-over-attacks (any successful attack flips the
    sample); the surviving set is the intersection.
    """
    model.eval()
    # 1) PGD-CE
    flips_pgd = []
    for i in range(0, X.size(0), 256):
        x, y = X[i:i + 256], Y[i:i + 256]
        xa = C.pgd(model, x, y, eps=eps, steps=PGD_STEPS_EVAL,
                   alpha=2.5 * eps / PGD_STEPS_EVAL, random_start=True)
        with torch.no_grad():
            flips_pgd.append((model(xa).argmax(1) != y).cpu())
    flips_pgd = torch.cat(flips_pgd).numpy().astype(bool)
    # 2) APGD-DLR
    adv = apgd_dlr_attack(model, X, Y, eps=eps)
    with torch.no_grad():
        flips_apgd = []
        for i in range(0, adv.size(0), 256):
            flips_apgd.append(
                (model(adv[i:i + 256]).argmax(1) != Y[i:i + 256]).cpu())
    flips_apgd = torch.cat(flips_apgd).numpy().astype(bool)
    union_flip = flips_pgd | flips_apgd
    return float((~union_flip).mean())


# ---- Square attack (gradient-free; Andriushchenko 2020) ---------------------
def _p_selection(p_init, it, n_iters):
    it = int(it / max(1, n_iters) * 10000)
    if 10 < it <= 50:        return p_init / 2
    if 50 < it <= 200:       return p_init / 4
    if 200 < it <= 500:      return p_init / 8
    if 500 < it <= 1000:     return p_init / 16
    if 1000 < it <= 2000:    return p_init / 32
    if 2000 < it <= 4000:    return p_init / 64
    if 4000 < it <= 6000:    return p_init / 128
    return max(p_init, 1e-6)


def robust_acc_square(model, X, Y, eps=EPS_MID, n_queries=SQUARE_QUERIES,
                      p_init=0.05):
    """Square attack (Linf, untargeted). Returns robust acc over X."""
    model.eval()
    device = X.device
    n, Cch, H, W = X.shape
    # init with vertical stripes of random sign (paper section 4.1)
    delta = torch.zeros_like(X)
    init_signs = torch.empty(n, Cch, 1, W, device=device).uniform_(-1, 1).sign()
    delta = delta + eps * init_signs
    x_curr = (X + delta).clamp(0, 1)

    with torch.no_grad():
        logits = model(x_curr)
        z_y = logits.gather(1, Y.view(-1, 1)).squeeze(1)
        tmp = logits.clone()
        tmp[torch.arange(n), Y] = -1e9
        z_other = tmp.max(1).values
        loss_curr = z_y - z_other     # want to minimise (then becomes negative)
        succ = (logits.argmax(1) != Y)

    for q in range(n_queries):
        if succ.all():
            break
        p = _p_selection(p_init, q, n_queries)
        s = max(1, int(round(math.sqrt(p * H * W))))
        s = min(s, H - 1, W - 1)
        h = torch.randint(0, H - s + 1, (n,), device=device)
        w = torch.randint(0, W - s + 1, (n,), device=device)
        signs = torch.empty(n, Cch, 1, 1, device=device).uniform_(-1, 1).sign()
        delta_new = delta.clone()
        # vectorised would need scatter; this loop is fine for n<=1000
        for j in range(n):
            if succ[j]:
                continue
            delta_new[j, :, h[j]:h[j] + s, w[j]:w[j] + s] = eps * signs[j]
        x_new = (X + delta_new).clamp(0, 1)
        x_new = torch.min(torch.max(x_new, X - eps), X + eps).clamp(0, 1)
        with torch.no_grad():
            logits_new = model(x_new)
            z_y_new = logits_new.gather(1, Y.view(-1, 1)).squeeze(1)
            tmp = logits_new.clone()
            tmp[torch.arange(n), Y] = -1e9
            z_other_new = tmp.max(1).values
            loss_new = z_y_new - z_other_new
            improved = (loss_new < loss_curr) & (~succ)
            if improved.any():
                delta = torch.where(improved[:, None, None, None],
                                    delta_new, delta)
                loss_curr = torch.where(improved, loss_new, loss_curr)
                succ = succ | (logits_new.argmax(1) != Y)
    return float((~succ).cpu().numpy().mean())


# ---- Carlini-Wagner L2 (Carlini & Wagner 2017) -----------------------------
def cw_l2_attack(model, X, Y, c=CW_C, steps=CW_STEPS, lr=CW_LR, batch=128):
    """Untargeted CW-L2 with tanh box constraint + margin loss (kappa=0)."""
    model.eval()
    out_advs = []
    for i in range(0, X.size(0), batch):
        x = X[i:i + batch].clone().detach()
        y = Y[i:i + batch]
        # tanh reparameterisation:  x = 0.5*(tanh(w)+1)
        x_clamped = x.clamp(1e-6, 1 - 1e-6)
        w = torch.atanh(2 * x_clamped - 1).detach().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        best_adv = x.clone()
        best_l2 = torch.full((x.size(0),), float("inf"), device=x.device)
        for _ in range(steps):
            adv = 0.5 * (torch.tanh(w) + 1)
            logits = model(adv)
            z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
            tmp = logits.clone()
            tmp[torch.arange(tmp.size(0)), y] = -1e9
            z_other = tmp.max(1).values
            # untargeted margin: f(x) = max(z_y - z_other, -kappa); kappa=0
            f = torch.clamp(z_y - z_other, min=0.0)
            l2 = ((adv - x) ** 2).flatten(1).sum(1)
            loss = (l2 + c * f).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                preds = logits.argmax(1)
                flipped = preds != y
                better = flipped & (l2 < best_l2)
                if better.any():
                    best_l2 = torch.where(better, l2, best_l2)
                    best_adv[better] = adv[better].detach()
        out_advs.append(best_adv.detach())
    return torch.cat(out_advs, 0)


def robust_acc_cw_l2(model, X, Y, c=CW_C, steps=CW_STEPS):
    """CW-L2 robust accuracy. We DO NOT cap L2 -- CW is the "did we find ANY
    adversarial?" attack. This means for STD-ish models the robust acc will
    be ~0; the interesting signal is whether AT defenses also collapse,
    because CW-L2 is a fundamentally different threat model from Linf-PGD.
    """
    model.eval()
    adv = cw_l2_attack(model, X, Y, c=c, steps=steps)
    with torch.no_grad():
        correct = 0
        for i in range(0, adv.size(0), 256):
            correct += int((model(adv[i:i + 256]).argmax(1)
                            == Y[i:i + 256]).sum().item())
    return correct / adv.size(0)


# Attacks list: (label, callable(model, X, Y) -> robust_acc)
ATTACKS = [
    ("PGD@0.05",     lambda m, X, Y: robust_acc_pgd(m, X, Y, eps=EPS_LOW)),
    ("PGD@0.10",     lambda m, X, Y: robust_acc_pgd(m, X, Y, eps=EPS_MID)),
    ("PGD@0.20",     lambda m, X, Y: robust_acc_pgd(m, X, Y, eps=EPS_HIGH)),
    ("AA-lite@0.10", lambda m, X, Y: robust_acc_aa_lite(m, X, Y, eps=EPS_MID)),
    ("Square@0.10",  lambda m, X, Y: robust_acc_square(m, X, Y, eps=EPS_MID)),
    ("CW-L2",        lambda m, X, Y: robust_acc_cw_l2(m, X, Y)),
]


# ---------------------------------------------------------------------------
# Spearman rho (no scipy dependency required, but use scipy if present).
# ---------------------------------------------------------------------------
def _spearman_rho(a, b):
    """Spearman rank correlation between equal-length 1D arrays a and b.

    We rank descending (rank 1 = best defense) but the correlation is
    invariant to the direction.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(a, b)
        if np.isnan(rho):
            return float("nan")
        return float(rho)
    except Exception:
        # fallback: Pearson on ranks
        ra = np.argsort(np.argsort(-a))
        rb = np.argsort(np.argsort(-b))
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        if denom < 1e-12:
            return float("nan")
        return float((ra * rb).sum() / denom)


def _ranks_desc(col):
    """Return integer ranks (1 = highest robust acc) for a 1D array, ties
    broken by stable original index ordering."""
    col = np.asarray(col, dtype=float)
    order = np.argsort(-col, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(col) + 1)
    return ranks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _FH
    os.makedirs(OUT_DIR, exist_ok=True)
    _FH = open(OUT_FILE, "w")

    log("=" * 78)
    log("H513 - Cross-attack defense-ranking stability")
    log("=" * 78)
    log(f"Device={C.DEVICE}  dataset={DS}  n_train={N_TRAIN}  n_eval={N_EVAL}")
    log(f"Defenses (8): {[d[0] for d in DEFENSES]}")
    log(f"Attacks  (6): {[a[0] for a in ATTACKS]}")
    log("Refs: Croce 2021 RobustBench (arXiv:2010.09670);")
    log("      Carlini 2019 'On Evaluating Adversarial Robustness' "
        "(arXiv:1902.06705);")
    log("      Tramer 2020 'On Adaptive Attacks' (NeurIPS; arXiv:2002.08347);")
    log("      Croce & Hein 2020 AutoAttack (ICML);")
    log("      Andriushchenko 2020 Square Attack (ECCV);")
    log("      Carlini & Wagner 2017 (S&P).")
    log()

    # ----- data -----
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                        seed=SEED)
    log(f"Train: {tuple(Xtr.shape)}  Eval: {tuple(Xte.shape)}")
    log()

    # ----- train each defense -----
    models = {}
    clean_accs = {}
    for name, fn in DEFENSES:
        t0 = time.time()
        log(f"[train] {name} ...")
        m = fn(meta, Xtr, Ytr)
        clean_accs[name] = clean_acc(m, Xte, Yte)
        models[name] = m
        log(f"  trained in {time.time()-t0:.1f}s  cleanAcc={clean_accs[name]:.3f}")
    log()

    # ----- eval each defense against each attack -----
    log("=" * 78)
    log("Robust accuracy matrix  (rows=defenses, columns=attacks)")
    log("=" * 78)

    matrix = np.zeros((len(DEFENSES), len(ATTACKS)), dtype=float)
    for i, (dname, _) in enumerate(DEFENSES):
        m = models[dname]
        for j, (aname, afn) in enumerate(ATTACKS):
            t0 = time.time()
            ra = afn(m, Xte, Yte)
            matrix[i, j] = ra
            log(f"  {dname:<14} {aname:<14} robust_acc={ra:.4f}  "
                f"({time.time()-t0:.1f}s)")
    log()

    # ----- pretty table -----
    hdr = f"{'Defense':<14} " + " ".join(f"{a[0]:>13}" for a in ATTACKS) \
        + f" {'clean':>8}"
    log(hdr)
    log("-" * len(hdr))
    for i, (dname, _) in enumerate(DEFENSES):
        row = f"{dname:<14} " + " ".join(f"{matrix[i, j]:>13.4f}"
                                          for j in range(len(ATTACKS))) \
            + f" {clean_accs[dname]:>8.4f}"
        log(row)
    log()

    # ----- per-attack rankings (1 = best) -----
    log("=" * 78)
    log("Per-attack defense rankings (1 = best)")
    log("=" * 78)
    rank_matrix = np.zeros_like(matrix, dtype=int)
    for j in range(len(ATTACKS)):
        rank_matrix[:, j] = _ranks_desc(matrix[:, j])

    hdr = f"{'Defense':<14} " + " ".join(f"{a[0]:>13}" for a in ATTACKS)
    log(hdr)
    log("-" * len(hdr))
    for i, (dname, _) in enumerate(DEFENSES):
        row = f"{dname:<14} " + " ".join(f"{rank_matrix[i, j]:>13d}"
                                         for j in range(len(ATTACKS)))
        log(row)
    log()

    # ----- per-attack top-3 winners -----
    log("=" * 78)
    log("Top-3 winners per attack")
    log("=" * 78)
    for j, (aname, _) in enumerate(ATTACKS):
        order = np.argsort(-matrix[:, j], kind="stable")[:3]
        winners = [(DEFENSES[i][0], matrix[i, j]) for i in order]
        s = "  ".join(f"{n} ({v:.3f})" for n, v in winners)
        log(f"  {aname:<14} -> {s}")
    log()

    # ----- Spearman rho between every attack pair -----
    log("=" * 78)
    log("Spearman rho between attack-column rankings")
    log("=" * 78)
    pairs = []
    n_a = len(ATTACKS)
    rho_mat = np.full((n_a, n_a), np.nan)
    for j1 in range(n_a):
        for j2 in range(j1 + 1, n_a):
            rho = _spearman_rho(matrix[:, j1], matrix[:, j2])
            rho_mat[j1, j2] = rho
            rho_mat[j2, j1] = rho
            pairs.append((ATTACKS[j1][0], ATTACKS[j2][0], rho))
            log(f"  rho({ATTACKS[j1][0]:<14}, {ATTACKS[j2][0]:<14}) = "
                f"{rho:+.3f}")
    log()

    # ----- rank-shift flags: any defense whose rank changes by > 3 -----
    log("=" * 78)
    log("Defenses with rank shifts > 3 between any attack pair")
    log("=" * 78)
    flagged = []
    for i, (dname, _) in enumerate(DEFENSES):
        rng = int(rank_matrix[i].max() - rank_matrix[i].min())
        if rng > 3:
            best_j = int(rank_matrix[i].argmin())
            worst_j = int(rank_matrix[i].argmax())
            log(f"  {dname:<14} range={rng}  best @ {ATTACKS[best_j][0]} "
                f"(rank {rank_matrix[i, best_j]}), worst @ "
                f"{ATTACKS[worst_j][0]} (rank {rank_matrix[i, worst_j]})")
            flagged.append(dname)
    if not flagged:
        log("  (none)")
    log()

    # ----- HEADLINE verdict -----
    log("=" * 78)
    log("HEADLINE")
    log("=" * 78)
    n_unstable = sum(1 for *_, r in pairs if (not np.isnan(r)) and r < 0.7)
    top_pgd010 = DEFENSES[int(np.argmax(matrix[:, 1]))][0]   # col 1 = PGD@0.10
    top_pgd020 = DEFENSES[int(np.argmax(matrix[:, 2]))][0]   # col 2 = PGD@0.20
    top_cw = DEFENSES[int(np.argmax(matrix[:, 5]))][0]       # col 5 = CW-L2
    top_changes = (top_pgd010 != top_pgd020) or (top_pgd010 != top_cw)

    if n_unstable >= 3 and top_changes:
        verdict = "SUPPORTED"
    elif n_unstable >= 1 or top_changes:
        verdict = "PARTIAL"
    else:
        verdict = "REJECTED"

    log(f"  Pairs with Spearman rho < 0.7: {n_unstable} / {len(pairs)}")
    log(f"  Top defense @ PGD@0.10:  {top_pgd010}")
    log(f"  Top defense @ PGD@0.20:  {top_pgd020}  "
        f"(differs: {top_pgd010 != top_pgd020})")
    log(f"  Top defense @ CW-L2:     {top_cw}     "
        f"(differs: {top_pgd010 != top_cw})")
    log(f"  Flagged (rank range > 3): {flagged or 'none'}")
    log()
    log(f"  HEADLINE  ({DS}, SmallCNN, 8 defenses x 6 attacks): {verdict}.")
    log("  Interpretation: if SUPPORTED, the RobustBench / Carlini / Tramer")
    log("  warning holds on F-MNIST -- 'best' defense is a function of the")
    log("  eval threat model, not an intrinsic property of the defense. Any")
    log("  paper quoting a winner under a single (attack, eps) is misleading.")
    log("=" * 78)

    _FH.close()


if __name__ == "__main__":
    main()
