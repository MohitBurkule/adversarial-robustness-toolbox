"""
H464 - AutoAttack ladder on top-10 campaign winners.

Gaps closed: M3 (eval threat model collapsed to PGD-10), G9 (no AutoAttack
              ensemble as default eval), M4 (winners never adapt-attacked).

Paper anchors:
  * croce-2020-autoattack  - Croce & Hein, ICML 2020. APGD-CE, APGD-DLR,
    parameter-free step-size adaptation (Algorithm 1), DLR loss, FAB,
    Square ensemble. AutoAttack = max over four diverse attacks; treat the
    union of successes as the verdict.
  * andriushchenko-2020-square - Square Attack (ECCV 2020). Gradient-FREE
    random-search black-box attack; *critical* for detecting gradient
    masking because no gradient through the defended model is used.
  * croce-2021-robustbench - RobustBench evaluation protocol; AutoAttack
    is the canonical empirical eval.
  * tramer-2020-adaptive - "On Adaptive Attacks to Adversarial Example
    Defenses". A defence that survives PGD-10 but falls to AutoAttack OR
    is broken by Square (gradient-free) is gradient-masking, not robust.
  * athalye-2018-obfuscated - obfuscated-gradients signals; large gap
    between white-box (APGD) and gradient-free (Square) ASR => masking.

Critique of the campaign eval up to H413:
  - PGD-10 with fixed step (alpha = 2.5*eps/steps) and a single random
    start is the WEAKEST member of the AutoAttack ensemble. Defences
    rated "robust" at PGD-ASR < 0.7 (H323, H344, H365, H372, H376, H381,
    H290-block0, H304 TRADES, H316 AWP) have never been evaluated under
    APGD with adaptive step, DLR loss, or any gradient-free attack.
  - H391 ran a limited masking battery (transfer, restarts, step curve,
    FGSM-PGD gap) but did NOT run gradient-free Square. Square is the
    decisive test: a true robust model degrades smoothly under Square;
    a gradient-masking model takes essentially no extra hits from Square
    relative to APGD (or, conversely, Square beats APGD - the smoking gun).

This script:
  1. Retrains each of 10 winner-configurations from scratch (10 epochs,
     N_TRAIN=6000) to keep models on the same footing.
  2. Runs APGD-CE, APGD-DLR, and Square against each model on a 1000-image
     test subset, at eps=0.1 (Linf), reporting per-attack ASR and the
     AutoAttack-ladder ASR = max-over-attacks union of successes.
  3. Issues a per-model verdict:
        GENUINE       - AA_ASR within +0.05 of PGD-10 (no leakage)
        WEAKLY_MASKED - AA_ASR exceeds PGD-10 by 0.05-0.15
        MASKED        - AA_ASR exceeds PGD-10 by >0.15, OR Square > APGD-CE
                        (gradient-free beats gradient-based => obfuscation).

Pure torch. ASCII only. Flushes per model. No checkpoints reused.
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
# Standard campaign config (from CLAUDE.md / CAMPAIGN_GAP_MAP)
# ---------------------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 1000          # subset for AA (per task brief)
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1              # Linf budget

# APGD config (Croce 2020): n_iter typically 100, but we use 50 to keep
# 10-model x 3-attack budget feasible on a single 4090.
APGD_ITERS = 50
APGD_RHO = 0.75        # step-size halving condition fraction
APGD_RESTARTS = 1      # single restart per attack to fit time budget
APGD_CHECKPOINTS_FRAC = (0.22, 0.06)  # p0, p1 per paper (p_{j+1}=max(p_j - p_1, 0.06))

# Square config (Andriushchenko 2020): query budget 1000 per sample.
SQUARE_QUERIES = 1000
SQUARE_P_INIT = 0.05   # initial fraction of image area for square side ratio

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
OUT_FILE = os.path.join(OUT_DIR, "h464_autoattack_top10_output.txt")

_LINES = []
_FH = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LINES.append(s)
    if _FH is not None:
        _FH.write(s + "\n")
        _FH.flush()


# ===========================================================================
# Trainers - thin reuses of the published winner-hypothesis logic.
# Each takes (meta, Xtr, Ytr) and returns a trained model.
# Settings standardised: 10 epochs, LR=0.05, SGD(mom=0.9, wd=5e-4), cosine.
# ===========================================================================

def _opt_sched(model):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return opt, sched


def _iter_batches(Xtr, Ytr):
    n = Xtr.size(0)
    perm = torch.randperm(n, device=Xtr.device)
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        yield Xtr[idx], Ytr[idx]


# --- (1) standard baseline ------------------------------------------------
def train_standard(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt, sched = _opt_sched(model)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# --- (2) PGD-AT (Madry 2018) ---------------------------------------------
def train_pgd_at(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                         opt="sgd", lr=LR, adv_train=True,
                         adv_eps=EPS, adv_steps=7)


# --- (3) H290 block-0 activation-space FGSM-AT ---------------------------
def train_h290_block0(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    EPS_BLOCK0 = 0.5
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=1e-4)
    model.train()
    feat = list(model.features.children())
    for _ in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            opt.zero_grad()
            # forward prefix (block 0 only) without grad
            h = xb
            prefix = feat[0:4]
            suffix = feat[4:]
            with torch.no_grad():
                for m in prefix:
                    h = m(h)
            # 1-step FGSM in activation space
            h_in = h.detach().requires_grad_(True)
            suffix_net = nn.Sequential(*suffix, model.head).to(C.DEVICE)
            ce = F.cross_entropy(suffix_net(h_in), yb)
            ce.backward()
            with torch.no_grad():
                h_adv = h_in + EPS_BLOCK0 * h_in.grad.sign()
            opt.zero_grad()
            logits = suffix_net(h_adv.detach())
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


# --- (4) H304 TRADES (beta=6) --------------------------------------------
def _pgd_on_kl(model, x, steps=10, eps=EPS, alpha=0.01):
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)
    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(0, 1)
    x0 = x.detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        kl = F.kl_div(F.log_softmax(model(x_adv), dim=1), p_clean,
                      reduction='batchmean')
        g, = torch.autograd.grad(kl, x_adv)
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1)
    return x_adv.detach()


def train_h304_trades(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt, sched = _opt_sched(model)
    beta = 6.0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in _iter_batches(Xtr, Ytr):
            opt.zero_grad()
            x_adv = _pgd_on_kl(model, xb, steps=10, eps=EPS, alpha=0.01)
            model.train()
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


# --- (5) H316 AWP (gamma=0.01) -------------------------------------------
def train_h316_awp(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt, sched = _opt_sched(model)
    gamma = 0.01
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in _iter_batches(Xtr, Ytr):
            xb_adv = C.fgsm(model, xb, yb, eps=EPS)
            model.zero_grad()
            F.cross_entropy(model(xb_adv), yb).backward()
            perturbations = {}
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        delta = gamma * p.grad.sign()
                        perturbations[name] = delta
                        p.add_(delta)
            opt.zero_grad()
            F.cross_entropy(model(xb_adv), yb).backward()
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in perturbations:
                        p.sub_(perturbations[name])
            opt.step()
        sched.step()
    model.eval()
    return model


# --- (6) H323 Jacobian-Frobenius (lam=0.01) ------------------------------
def _jacobian_frob_pen(model, xb, k=5):
    out = model(xb)
    penalties = []
    for _ in range(k):
        v = torch.randn(xb.size(0), out.size(1), device=xb.device)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        xb_g = xb.clone().detach().requires_grad_(True)
        out_g = model(xb_g)
        proj = (out_g * v).sum()
        g, = torch.autograd.grad(proj, xb_g, create_graph=True)
        penalties.append((g ** 2).sum(dim=(1, 2, 3)))
    return torch.stack(penalties, 0).mean(0).mean()


def train_h323_jac_frob(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt, sched = _opt_sched(model)
    lam = 0.01
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb)
            jp = _jacobian_frob_pen(model, xb, k=5)
            (ce + lam * jp).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# --- (7) H344 spectral Jacobian (lam=0.01) -------------------------------
def train_h344_jac_spectral(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt, sched = _opt_sched(model)
    lam = 0.01
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            opt.zero_grad()
            xb_req = xb.clone().detach().requires_grad_(True)
            logits = model(xb_req)
            ce = F.cross_entropy(logits, yb)
            u = torch.randn(xb.size(0), logits.size(1), device=xb.device)
            u = u / (u.norm(dim=1, keepdim=True) + 1e-8)
            Jtu, = torch.autograd.grad((logits * u.detach()).sum(),
                                       xb_req, create_graph=True)
            sigma_sq = (Jtu.flatten(1) ** 2).sum(dim=1).mean()
            (ce + lam * sigma_sq).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# --- (8) H365 confidence-weighted gradient penalty (lam=0.1) -------------
def train_h365_cwgp(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt, sched = _opt_sched(model)
    lam = 0.1
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            opt.zero_grad()
            xb_req = xb.detach().requires_grad_(True)
            out = model(xb_req)
            ce_per = F.cross_entropy(out, yb, reduction='none')
            g, = torch.autograd.grad(ce_per.sum(), xb_req, create_graph=True)
            with torch.no_grad():
                p_correct = F.softmax(out.detach(), 1).gather(
                    1, yb.view(-1, 1)).squeeze(1)
            w = (1 - p_correct).detach()
            gnsq = (g.view(g.size(0), -1) ** 2).sum(1)
            pen = (w * gnsq).mean()
            ce = F.cross_entropy(model(xb), yb)
            (ce + lam * pen).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# --- (9) H372 per-layer Jacobian penalty (lambdas=[1e-4,1e-4,1e-4]) ------
class _SmallCNNBlocks(nn.Module):
    def __init__(self, in_ch=1, size=28, n_classes=10, width=32):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.block0 = block(in_ch, width)
        self.block1 = block(width, width * 2)
        self.block2 = block(width * 2, width * 4)
        feat = size // 8
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(width * 4 * feat * feat, 256),
                                  nn.ReLU(), nn.Linear(256, n_classes))

    def forward(self, x):
        h = self.block0(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.head(h)


def _est_jac_frob_block(block, h_in, k=5):
    h_out = block(h_in)
    pen = 0.0
    for _ in range(k):
        v = torch.randn_like(h_out)
        JTv, = torch.autograd.grad((h_out * v).sum(), h_in,
                                   create_graph=True, retain_graph=True)
        pen = pen + (JTv ** 2).sum() / h_in.size(0)
    return pen / k


def train_h372_per_layer_jac(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = _SmallCNNBlocks(meta["channels"], meta["size"],
                            meta["n_classes"]).to(C.DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lambdas = [1e-4, 1e-4, 1e-4]
    blocks = [model.block0, model.block1, model.block2]
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            h = xb
            h_inputs = []
            for blk in blocks:
                h_det = h.detach().requires_grad_(True)
                h_inputs.append(h_det)
                h = blk(h_det)
            logits = model.head(h)
            ce = F.cross_entropy(logits, yb)
            jac_loss = 0.0
            for lam, blk, h_in in zip(lambdas, blocks, h_inputs):
                if lam > 0:
                    jac_loss = jac_loss + lam * _est_jac_frob_block(blk, h_in)
            opt.zero_grad()
            (ce + jac_loss).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# --- (10) H376 anti-SAM (rho=0.05) ---------------------------------------
def train_h376_anti_sam(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt, sched = _opt_sched(model)
    rho = 0.05
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            gn = torch.sqrt(sum((p.grad ** 2).sum()
                                for p in model.parameters() if p.grad is not None))
            if gn > 1e-12:
                old = [p.data.clone() for p in model.parameters()]
                for p in model.parameters():
                    if p.grad is not None:
                        p.data.add_(rho * p.grad / gn)
                opt.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                for p, o in zip(model.parameters(), old):
                    p.data.copy_(o)
            opt.step()
        sched.step()
    model.eval()
    return model


# --- (11) H381 activation-stats matching (lam=0.01; one-pass with AT
#          stats produced inline since we need a self-contained trainer)
def train_h381_act_stats(meta, Xtr, Ytr):
    C.set_seed(SEED)
    # Phase 1: small AT model to obtain block-norm targets
    at_model = C.build_model("cnn", meta)
    at_model = C.train_model(at_model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                             opt="sgd", lr=LR, adv_train=True,
                             adv_eps=EPS, adv_steps=7)
    # measure mean block-norms on training subset
    at_model.eval()
    accum = [0.0, 0.0, 0.0]
    cnt = 0
    with torch.no_grad():
        for i in range(0, Xtr.size(0), 256):
            xb = Xtr[i:i + 256]
            h = xb
            for bidx in range(3):
                start = bidx * 4
                for layer in at_model.features[start:start + 4]:
                    h = layer(h)
                accum[bidx] += h.flatten(1).norm(dim=1).sum().item()
            cnt += xb.size(0)
    targets = [a / cnt for a in accum]
    # Phase 2: stats-matched standard model
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt, sched = _opt_sched(model)
    lam = 0.01
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            opt.zero_grad()
            h = xb
            penalty = torch.tensor(0.0, device=xb.device)
            for bidx in range(3):
                start = bidx * 4
                for layer in model.features[start:start + 4]:
                    h = layer(h)
                penalty = penalty + (h.flatten(1).norm(dim=1).mean()
                                     - targets[bidx]) ** 2
            logits = model.head(h)
            (F.cross_entropy(logits, yb) + lam * penalty).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ===========================================================================
# Attacks
# ===========================================================================

def pgd10_asr(model, X, Y, eps=EPS, steps=10, alpha=0.01, batch=256):
    """Plain PGD-10 (campaign reference). ASR over originally-correct."""
    model.eval()
    flips, corr = [], []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        with torch.no_grad():
            c = (model(x).argmax(1) == y).cpu()
        xa = C.pgd(model, x, y, eps=eps, steps=steps, alpha=alpha,
                   random_start=True)
        with torch.no_grad():
            f = (model(xa).argmax(1) != y).cpu()
        flips.append(f)
        corr.append(c)
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr).numpy().astype(bool)
    return (float(flips[corr].mean()) if corr.sum() else float("nan"),
            corr, flips)


# ---- APGD (Croce 2020 Algorithm 1, simplified) --------------------------
def _dlr_loss(logits, y):
    """Croce 2020 DLR: -(z_y - z_pi1) / (z_pi1 - z_pi3 + 1e-12).

    Where pi sorts logits descending and pi1, pi3 are the highest indices
    excluding y for the numerator's pi1 (the max-wrong logit). The original
    paper writes it as:
        L_DLR(x, y) = -( z_y - max_{i != y} z_i ) / ( z_pi1 - z_pi3 + 1e-12 )
    We minimise -DLR (i.e. maximise the loss for an untargeted attack).
    """
    z_sorted, ind_sorted = logits.sort(dim=1, descending=True)
    # max-wrong logit:
    z_clone = logits.clone()
    z_clone[torch.arange(z_clone.size(0)), y] = -1e9
    z_other = z_clone.max(1).values
    z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
    # pi1, pi3 from full sort (with y included)
    pi1 = z_sorted[:, 0]
    pi3 = z_sorted[:, 2] if logits.size(1) > 2 else z_sorted[:, -1]
    return -(z_y - z_other) / (pi1 - pi3 + 1e-12)


def apgd_attack(model, X, Y, eps=EPS, n_iter=APGD_ITERS, loss="ce",
                rho=APGD_RHO, batch=256):
    """Auto-PGD (Croce & Hein 2020), Linf, single restart.

    Step 1: init x0 = x + uniform[-eps,eps], project to [0,1] ball
    Step 2: compute fixed-step schedule with checkpoints W = {w_j}.
            At each w_j, halve step size if (a) <rho fraction of last
            interval iterations improved the loss OR (b) step size and
            best-loss unchanged since previous checkpoint.
    Step 3: maintain x_best = argmax loss seen so far; restart from x_best
            on step-size halving (warm restart).
    Step 4: update x_{k+1} = x_k + alpha_k * sign(grad L(x_k, y))
            with Nesterov-style momentum (paper Eq. 5):
              z = x_k + alpha * sign(grad)
              x_{k+1} = x_k + a * (z - x_k) + (1 - a) * (x_k - x_{k-1})
            We use a = 0.75 per paper recommendation.
    Loss in {ce, dlr}.
    """
    model.eval()
    out_advs = []
    for i in range(0, X.size(0), batch):
        x_orig = X[i:i + batch]
        y = Y[i:i + batch]
        x0 = x_orig.clone()
        # init: random within eps-ball
        x = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        x = x.clamp(0, 1)
        x_prev = x.clone()
        # step-size init: 2*eps (paper)
        alpha = 2.0 * eps

        # build checkpoint schedule W
        p0, p1 = APGD_CHECKPOINTS_FRAC
        W = []
        p = p0
        ck = int(round(p * n_iter))
        while ck < n_iter:
            W.append(ck)
            p = max(p - p1, 0.06)
            new_ck = ck + max(int(round(p * n_iter)), 1)
            if new_ck <= ck:
                break
            ck = new_ck
        if not W or W[-1] != n_iter - 1:
            W.append(n_iter - 1)

        # tracking
        x_best = x.clone()
        loss_best = None
        loss_at_last_ck = None
        alpha_at_last_ck = alpha
        improved_count = 0
        last_ck = 0

        for k in range(n_iter):
            x_in = x.clone().detach().requires_grad_(True)
            logits = model(x_in)
            if loss == "ce":
                L_per = F.cross_entropy(logits, y, reduction='none')
            elif loss == "dlr":
                L_per = _dlr_loss(logits, y)
            else:
                raise ValueError(loss)
            L = L_per.sum()
            g, = torch.autograd.grad(L, x_in)
            with torch.no_grad():
                # momentum update (a=0.75)
                z = x + alpha * g.sign()
                # project
                z = torch.min(torch.max(z, x_orig - eps), x_orig + eps).clamp(0, 1)
                x_new = x + 0.75 * (z - x) + 0.25 * (x - x_prev)
                x_new = torch.min(torch.max(x_new, x_orig - eps),
                                  x_orig + eps).clamp(0, 1)
                x_prev = x.clone()
                x = x_new

                # bookkeeping: per-sample best (use scalar mean improvement
                # signal -- simplifies vs Croce's per-sample tracking but is
                # sufficient for a batch-level adaptive step).
                if loss_best is None:
                    loss_best = L_per.detach().clone()
                    x_best = x.clone()
                else:
                    better = L_per.detach() > loss_best
                    if better.any():
                        loss_best = torch.where(better, L_per.detach(), loss_best)
                        # update x_best per-sample
                        x_best[better] = x[better]
                        improved_count += int(better.sum().item())

                # checkpoint condition
                if k in W:
                    interval = max(1, k - last_ck)
                    cond1 = improved_count < rho * interval * x.size(0)
                    cond2 = (loss_at_last_ck is not None
                             and abs(alpha - alpha_at_last_ck) < 1e-12
                             and torch.allclose(loss_best.mean().detach(),
                                                loss_at_last_ck))
                    if cond1 or cond2:
                        alpha = alpha / 2.0
                        # warm restart from x_best
                        x = x_best.clone()
                        x_prev = x_best.clone()
                    loss_at_last_ck = loss_best.mean().detach().clone()
                    alpha_at_last_ck = alpha
                    improved_count = 0
                    last_ck = k

        out_advs.append(x_best.detach())
    return torch.cat(out_advs, 0)


def apgd_asr(model, X, Y, loss, corr, eps=EPS):
    """Run APGD on whole set, return ASR over correct + per-sample flips."""
    adv = apgd_attack(model, X, Y, eps=eps, loss=loss)
    with torch.no_grad():
        flips = []
        for i in range(0, adv.size(0), 256):
            flips.append((model(adv[i:i + 256]).argmax(1) != Y[i:i + 256]).cpu())
        flips = torch.cat(flips).numpy()
    asr = float(flips[corr].mean()) if corr.sum() else float("nan")
    return asr, flips


# ---- Square attack (Andriushchenko 2020) --------------------------------
def _p_selection(p_init, it, n_iters):
    """Piecewise-constant schedule for the squared-area fraction p."""
    it = int(it / n_iters * 10000)
    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init
    return max(p, 1e-6)


def square_attack_linf(model, X, Y, eps=EPS, n_queries=SQUARE_QUERIES,
                       p_init=SQUARE_P_INIT, batch_first=True):
    """Square attack (Linf), Algorithm 1 of Andriushchenko 2020.

    Gradient-FREE: only forward calls to the model. At each step, sample a
    random square location (h, w) of side s = round(sqrt(p * H * W)), set
    delta in that square to a random {-eps, +eps}, clamp, evaluate the
    margin loss, accept if it decreases (untargeted: we want correct-class
    margin to drop, i.e. loss = z_y - max_{i!=y} z_i to be minimised).
    """
    model.eval()
    device = X.device
    n, C_, H, W = X.shape
    # init with vertical-stripe random sign perturbation (paper Sec 4.1)
    delta = torch.zeros_like(X)
    init_signs = torch.empty(n, C_, 1, W, device=device).uniform_(-1, 1).sign()
    delta = delta + eps * init_signs
    x_curr = (X + delta).clamp(0, 1)

    with torch.no_grad():
        logits = model(x_curr)
        # margin loss: z_y - max_{i!=y} z_i  -> minimise to flip
        z_y = logits.gather(1, Y.view(-1, 1)).squeeze(1)
        tmp = logits.clone()
        tmp[torch.arange(n), Y] = -1e9
        z_other = tmp.max(1).values
        loss_curr = z_y - z_other  # we want to minimise this
        # init success
        succ = logits.argmax(1) != Y

    for q in range(n_queries):
        p = _p_selection(p_init, q, n_queries)
        s = max(1, int(round(math.sqrt(p * H * W))))
        s = min(s, H - 1, W - 1)
        # batched proposal -> sample one square location per sample
        h = torch.randint(0, H - s + 1, (n,), device=device)
        w = torch.randint(0, W - s + 1, (n,), device=device)
        # per-sample per-channel sign
        signs = torch.empty(n, C_, 1, 1, device=device).uniform_(-1, 1).sign()
        delta_new = delta.clone()
        for j in range(n):
            if succ[j]:
                continue
            delta_new[j, :, h[j]:h[j] + s, w[j]:w[j] + s] = eps * signs[j]
        x_new = (X + delta_new).clamp(0, 1)
        # enforce Linf constraint (since we set absolute, redundant for {-eps,+eps})
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
                x_curr = (X + delta).clamp(0, 1)
                loss_curr = torch.where(improved, loss_new, loss_curr)
                new_succ = logits_new.argmax(1) != Y
                succ = succ | (new_succ & improved)
        if succ.all():
            break
    return (X + delta).clamp(0, 1), succ.cpu().numpy()


def square_asr(model, X, Y, corr, eps=EPS, n_queries=SQUARE_QUERIES,
               batch=200):
    """Wrap Square in batches; ASR over originally-correct samples."""
    flips_all = np.zeros(X.size(0), dtype=bool)
    for i in range(0, X.size(0), batch):
        xa, succ = square_attack_linf(model, X[i:i + batch], Y[i:i + batch],
                                      eps=eps, n_queries=n_queries)
        with torch.no_grad():
            preds = []
            for j in range(0, xa.size(0), 256):
                preds.append(model(xa[j:j + 256]).argmax(1).cpu())
            preds = torch.cat(preds).numpy()
        flips_all[i:i + xa.size(0)] = preds != Y[i:i + xa.size(0)].cpu().numpy()
    asr = float(flips_all[corr].mean()) if corr.sum() else float("nan")
    return asr, flips_all


# ===========================================================================
# Per-model verdict
# ===========================================================================

def verdict(pgd, aa, sq, apgd_ce):
    """Issue genuine/weakly-masked/masked label.

    Heuristics (combine Croce 2020 and Athalye 2018):
      - if AA - PGD-10 > 0.15: MASKED (PGD-10 grossly under-estimated).
      - elif Square > APGD-CE + 0.02: MASKED (gradient-free beat gradient =
        gradient was unreliable).
      - elif AA - PGD-10 > 0.05: WEAKLY_MASKED.
      - else: GENUINE.

    Always reported for any model claiming non-trivial robustness
    (PGD-10 ASR < 0.85). Wide-open models receive verdict N/A.
    """
    if pgd >= 0.85:
        return "N/A_wide_open"
    if (aa - pgd) > 0.15:
        return "MASKED"
    if sq > apgd_ce + 0.02:
        return "MASKED_grad_free_beats_grad"
    if (aa - pgd) > 0.05:
        return "WEAKLY_MASKED"
    return "GENUINE"


# ===========================================================================
# Main
# ===========================================================================

TRAINERS = [
    ("standard",          train_standard),
    ("pgd_at",            train_pgd_at),
    ("h290_block0",       train_h290_block0),
    ("h304_trades_b6",    train_h304_trades),
    ("h316_awp_g0p01",    train_h316_awp),
    ("h323_jac_frob",     train_h323_jac_frob),
    ("h344_jac_spectral", train_h344_jac_spectral),
    ("h365_cwgp_lam0p1",  train_h365_cwgp),
    ("h372_per_layer_jac",train_h372_per_layer_jac),
    ("h376_anti_sam_r05", train_h376_anti_sam),
    ("h381_act_stats",    train_h381_act_stats),
]
# Note: spec said "top-10 winners"; we include 11 entries (standard as
# reference baseline + 10 winner configs). Standard's verdict will be N/A.


def main():
    global _FH
    os.makedirs(OUT_DIR, exist_ok=True)
    _FH = open(OUT_FILE, "w")

    t_global = time.time()
    log("=" * 78)
    log("H464  AutoAttack ladder on top-10 campaign winners (Fashion-MNIST)")
    log(f"  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}  EPOCHS={EPOCHS}  EPS={EPS}")
    log(f"  APGD iters={APGD_ITERS}  Square queries={SQUARE_QUERIES}")
    log(f"  device={C.DEVICE}  seed={SEED}")
    log("  attacks: PGD-10 (ref) | APGD-CE | APGD-DLR | Square (grad-free)")
    log("  ladder ASR = max-over-attacks union of per-sample flips")
    log("=" * 78)

    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                        seed=SEED)
    log(f"\ndata: Xtr={tuple(Xtr.shape)} Xte={tuple(Xte.shape)}")

    log("\n" + "-" * 78)
    log(f"{'model':22s} {'clean':>6s} {'PGD10':>6s} {'APGDce':>7s} "
        f"{'APGDdlr':>8s} {'Square':>7s} {'AA':>6s} {'verdict':>26s}  t(s)")
    log("-" * 78)

    summary_rows = []

    for name, trainer in TRAINERS:
        t0 = time.time()
        log(f"\n[train] {name} ...")
        model = trainer(meta, Xtr, Ytr)
        t_train = time.time() - t0

        # clean acc + originally-correct mask
        _, clean_acc = C.logits_and_acc(model, Xte, Yte)
        clean_acc = float(clean_acc)
        with torch.no_grad():
            corr_mask = []
            for i in range(0, Xte.size(0), 256):
                corr_mask.append((model(Xte[i:i + 256]).argmax(1)
                                  == Yte[i:i + 256]).cpu())
            corr_mask = torch.cat(corr_mask).numpy().astype(bool)

        # ---- attacks
        t1 = time.time()
        pgd_a, _, pgd_flips = pgd10_asr(model, Xte, Yte)
        t2 = time.time()
        ce_asr, ce_flips = apgd_asr(model, Xte, Yte, "ce", corr_mask)
        t3 = time.time()
        dlr_asr, dlr_flips = apgd_asr(model, Xte, Yte, "dlr", corr_mask)
        t4 = time.time()
        sq_asr, sq_flips = square_asr(model, Xte, Yte, corr_mask)
        t5 = time.time()

        # ladder ASR: union of successes across PGD/APGD-CE/APGD-DLR/Square
        union = pgd_flips | ce_flips | dlr_flips | sq_flips
        aa_asr = float(union[corr_mask].mean()) if corr_mask.sum() else float("nan")

        v = verdict(pgd_a, aa_asr, sq_asr, ce_asr)

        row = dict(name=name, clean=clean_acc, pgd=pgd_a, ce=ce_asr,
                   dlr=dlr_asr, sq=sq_asr, aa=aa_asr,
                   verdict=v,
                   t_train=t_train,
                   t_pgd=t2 - t1, t_ce=t3 - t2, t_dlr=t4 - t3, t_sq=t5 - t4)
        summary_rows.append(row)

        log(f"{name:22s} {clean_acc:6.3f} {pgd_a:6.3f} {ce_asr:7.3f} "
            f"{dlr_asr:8.3f} {sq_asr:7.3f} {aa_asr:6.3f} {v:>26s}  "
            f"{(t5 - t0):.0f}")

        # free model GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log("\n" + "=" * 78)
    log("FINAL TABLE  (PGD10 = campaign reference, AA = ladder max)")
    log("=" * 78)
    log(f"{'model':22s} {'clean':>6s} {'PGD10':>6s} {'APGDce':>7s} "
        f"{'APGDdlr':>8s} {'Square':>7s} {'AA':>6s}  AA-PGD10  verdict")
    for r in summary_rows:
        gap = r["aa"] - r["pgd"]
        log(f"{r['name']:22s} {r['clean']:6.3f} {r['pgd']:6.3f} {r['ce']:7.3f} "
            f"{r['dlr']:8.3f} {r['sq']:7.3f} {r['aa']:6.3f}  {gap:+8.3f}  "
            f"{r['verdict']}")

    log("\nverdict legend:")
    log("  GENUINE                       - AA within +0.05 of PGD-10 AND Square ~ APGD-CE")
    log("  WEAKLY_MASKED                 - AA exceeds PGD-10 by 0.05-0.15")
    log("  MASKED                        - AA exceeds PGD-10 by > 0.15")
    log("  MASKED_grad_free_beats_grad   - Square > APGD-CE by > 0.02 (Athalye 2018 signal)")
    log("  N/A_wide_open                 - PGD-10 ASR >= 0.85 already; masking N/A")

    log("\npaper anchors used:")
    log("  - croce-2020-autoattack  : APGD-CE + APGD-DLR with adaptive step")
    log("  - andriushchenko-2020-square : gradient-free random search; key masking probe")
    log("  - croce-2021-robustbench : AA is the canonical eval protocol")
    log("  - tramer-2020-adaptive   : verdict heuristics on PGD-vs-AA gap")
    log("  - athalye-2018-obfuscated: Square > APGD => obfuscated gradients")

    log(f"\ntotal runtime {time.time() - t_global:.0f}s")
    _FH.close()
    print(f"saved -> {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
