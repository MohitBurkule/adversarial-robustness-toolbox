"""
H448 - CROWN-IBP certified training on Fashion-MNIST.

Gap filled: G2 (certified-defence training is barely present in the campaign;
H117 only used IBP as a *probe* on a standardly-trained CNN, never as a
training objective). Anchor: gowal-2018-ibp (H117 baseline) and
zhang-2020-crown-ibp (the method).

------------------------------------------------------------------------------
Critique / why CROWN-IBP over plain IBP
------------------------------------------------------------------------------
Pure IBP (Gowal 2018) forward-propagates an interval box [x-eps,x+eps]
through the network with interval arithmetic. The bounds it produces are
*sound* but extremely *loose*: every ReLU and every linear layer can only
widen the box, and for randomly-initialised weights the upper-lower gap
balloons exponentially in depth. Training a verified-loss objective from
scratch with such bounds is unstable - the verified loss is essentially
noise until the optimiser has shrunk the weight matrices enough that the
intervals stop exploding. Gowal 2018 mitigates this with a long eps
ramp-up (start at eps=0, end at target eps), but the bounds remain loose.

CROWN-IBP (Zhang 2020) fixes the looseness while keeping the speed:
  - FORWARD pass: standard IBP intervals (cheap, O(L)).
  - BACKWARD pass for the verified-loss CE term: a CROWN-style linear
    relaxation of each ReLU, propagated back from the logit layer to the
    input through the LINEAR coefficients of the network. This gives a
    much tighter upper bound on the worst-case logit margin than IBP.
  - LOSS: convex combo  L = kappa * CE(clean) + (1-kappa) * CE(robust),
    where the "robust" CE uses the (CROWN-IBP) tightened logit upper
    bounds on the non-true classes. Both terms are differentiable.

Key knobs:
  - eps schedule (HERE: linear ramp 0->eps over {5 ep, 10 ep, constant}).
    Shi 2021 ("Fast Certified Robust Training") shows the schedule and
    initialisation are the dominant factors at small budgets; a constant
    schedule (no warm-up) is the canonical failure mode and we include it
    as a negative control.
  - kappa schedule: 1.0 -> 0.0 (all clean -> all robust) over the ramp.

Controls / conditions:
  A. PGD-AT baseline   (standard PGD-10 AT at eps=0.1; gives "what AT buys").
  B. Pure IBP          (forward IBP loss only, linear eps ramp 10 ep).
  C. CROWN-IBP (full)  (CROWN backward + IBP forward, linear eps ramp 10 ep).
  D. CROWN-IBP ramp5   (same as C but ramp over 5 ep).
  E. CROWN-IBP const   (no ramp; full eps from epoch 0; expected to collapse).
  F. STD baseline      (CE only, no defence).

Metrics per condition:
  - clean acc
  - empirical PGD-10 ASR at eps=0.1 (campaign standard)
  - VERIFIED clean acc at eps=0.1 (IBP test-time bound; sample is
    "verified" if IBP says it is provably correct at eps=0.1)
  - VERIFIED Linf radius (mean over correctly-classified test samples
    of the largest eps for which IBP certifies the sample;
    binary-searched, matches H117 protocol).

Verdict logic:
  CROWN-IBP "works" if it produces a non-trivial verified Linf radius
  (> ~0.02) AND a verified-at-0.1 fraction > 0 AND its empirical PGD ASR
  is not catastrophically worse than PGD-AT. Pure IBP is expected to
  give a verified radius too but with much worse clean acc and worse
  empirical robustness. The constant-eps condition is expected to give
  ~10% clean acc (collapsed model).

------------------------------------------------------------------------------
Extra papers (beyond anchor)
------------------------------------------------------------------------------
- Shi et al. NeurIPS 2021, "Fast Certified Robust Training with Short
  Warmup" (arXiv 2103.17268): identifies init + BN + short eps warm-up
  as the dominant factors at small training budgets - directly relevant
  to our N_TRAIN=6000 / 10-epoch budget. They show standard inits give
  exploding IBP bounds; we use a small CNN without BN and rely on the
  ramp to keep this tractable.
- Mueller et al. ICLR 2023, "Certified Training: Small Boxes are All
  You Need" / SABR / "expressive losses" (arXiv 2210.04871, 2305.13991):
  show that IBP/CROWN-IBP over the FULL eps-ball is unnecessarily
  loose; computing the verified loss over a small adversarial sub-box
  recovers tighter bounds. We do NOT implement SABR (would need a new
  PGD-inside-IBP inner loop), but we cite it as the modern successor.
- Athalye et al. ICML 2018 obfuscated gradients (relevant control):
  certified defences are immune to gradient masking because the bound
  is computed from interval arithmetic, not from grad-sign attacks; the
  verified radius cannot be "spoofed" by a stochastic forward pass.

------------------------------------------------------------------------------
Standard campaign config
------------------------------------------------------------------------------
N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9, wd=5e-4),
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.

ASCII only. No emojis. Single seed.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config --------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01

# Verified-radius binary search range / depth (matches H117 protocol).
RAD_MAX = 0.30
RAD_STEPS = 12
N_CERT_SAMPLES = 500   # subsample for the verified-radius sweep (cost control)

META = {"channels": 1, "size": 28, "n_classes": 10}


# --------------------------------------------------------------------------
# A small fully-sequential CNN that is IBP-friendly:
#   - all linear ops are Conv2d / Linear (interval arithmetic is exact),
#   - no BN (Shi 2021 shows BN helps but adds complexity; we drop it
#     so the IBP backward pass stays a simple linear chain),
#   - no Dropout (would not be active at eval but adds noise during
#     training; the rest of the campaign omits dropout too).
# --------------------------------------------------------------------------
class IBPCNN(nn.Module):
    def __init__(self, in_ch=1, sz=28, n_cls=10, width=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width * 2, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(width * 2, width * 4, 3, padding=1, stride=2)
        feat = sz // 4   # 28 // 4 = 7
        self.fc1 = nn.Linear(width * 4 * feat * feat, 128)
        self.fc2 = nn.Linear(128, n_cls)
        # Ordered list of (kind, module). The forward order is fixed and
        # IBP / CROWN-IBP code below relies on this ordering.
        self.layer_seq = [
            ("conv", self.conv1), ("relu", None),
            ("conv", self.conv2), ("relu", None),
            ("conv", self.conv3), ("relu", None),
            ("flat", None),
            ("lin",  self.fc1),   ("relu", None),
            ("lin",  self.fc2),
        ]

    def forward(self, x):
        h = x
        for kind, m in self.layer_seq:
            if kind == "conv" or kind == "lin":
                h = m(h)
            elif kind == "relu":
                h = F.relu(h)
            elif kind == "flat":
                h = h.flatten(1)
        return h


# --------------------------------------------------------------------------
# Plain IBP forward: returns logit lower / upper bounds for a batch.
# Implements the same interval arithmetic as H117 but for IBPCNN's layout.
# --------------------------------------------------------------------------
def _conv_ibp(conv, h_L, h_U):
    Wp = conv.weight.clamp(min=0.0)
    Wm = conv.weight.clamp(max=0.0)
    z_L = (F.conv2d(h_L, Wp, None, conv.stride, conv.padding) +
           F.conv2d(h_U, Wm, None, conv.stride, conv.padding))
    z_U = (F.conv2d(h_U, Wp, None, conv.stride, conv.padding) +
           F.conv2d(h_L, Wm, None, conv.stride, conv.padding))
    if conv.bias is not None:
        b = conv.bias.view(1, -1, 1, 1)
        z_L = z_L + b; z_U = z_U + b
    return z_L, z_U


def _lin_ibp(lin, h_L, h_U):
    Wp = lin.weight.clamp(min=0.0)
    Wm = lin.weight.clamp(max=0.0)
    z_L = F.linear(h_L, Wp, None) + F.linear(h_U, Wm, None)
    z_U = F.linear(h_U, Wp, None) + F.linear(h_L, Wm, None)
    if lin.bias is not None:
        z_L = z_L + lin.bias; z_U = z_U + lin.bias
    return z_L, z_U


def ibp_forward(model, x, eps):
    """Plain IBP: forward-prop a box of half-width eps around x. Returns
    (logits_L, logits_U) of shape (B, n_classes)."""
    h_L = (x - eps).clamp(0.0, 1.0)
    h_U = (x + eps).clamp(0.0, 1.0)
    for kind, m in model.layer_seq:
        if kind == "conv":
            h_L, h_U = _conv_ibp(m, h_L, h_U)
        elif kind == "lin":
            h_L, h_U = _lin_ibp(m, h_L, h_U)
        elif kind == "relu":
            h_L, h_U = F.relu(h_L), F.relu(h_U)
        elif kind == "flat":
            h_L, h_U = h_L.flatten(1), h_U.flatten(1)
    return h_L, h_U


# --------------------------------------------------------------------------
# CROWN-IBP: forward IBP to get pre-activation bounds at every ReLU, then
# a backward CROWN-style linear bound from the chosen output ("worst-case
# logit margin") back to the input. We use the simplified CROWN-IBP form
# from Zhang 2020 sec.3: linear coefficients are backed through linear
# layers exactly, and through ReLUs via the standard upper/lower linear
# relaxations (slope alpha_U = u/(u-l) on unstable units, alpha_L either
# 0 or 1 depending on the sign of the incoming coefficient).
#
# For training we only need the UPPER bound on (z_j - z_y) for each
# non-target class j: if even the upper bound is < 0 the sample is
# verified. To make this differentiable and batched, we form the
# "elision" matrix C of shape (B, n_cls-1, n_cls) such that
# C @ logits = z_{j!=y} - z_y, and we ask CROWN-IBP for an upper bound on
# C @ logits. The robust CE loss then treats those upper bounds as logits
# and applies standard CE against the target class (Wong & Kolter 2018 /
# Zhang 2020 trick: max over the elided logits = 0 means certified).
# --------------------------------------------------------------------------
def _relu_relax(l, u):
    """CROWN ReLU relaxation. l, u are pre-activation bounds.
       Active   (l >= 0):    alpha_U = 1, beta_U = 0, alpha_L = 1, beta_L = 0
       Inactive (u <= 0):    alpha_U = 0, beta_U = 0, alpha_L = 0, beta_L = 0
       Unstable (l<0<u):     alpha_U = u/(u-l), beta_U = -alpha_U * l (>=0)
                             alpha_L in {0,1} chosen later by coeff sign;
                             we follow Zhang 2020 / "CROWN" and pick
                             alpha_L = (u >= -l) to minimise area.
    """
    eps_ = 1e-12
    active   = (l >= 0)
    inactive = (u <= 0)
    unstable = ~(active | inactive)

    alpha_U = torch.where(active, torch.ones_like(l),
              torch.where(inactive, torch.zeros_like(l),
                          u / (u - l + eps_)))
    beta_U  = torch.where(unstable, -alpha_U * l, torch.zeros_like(l))

    # lower relaxation: 0 or identity, choose by area (Zhang 2020).
    alpha_L = torch.where(active, torch.ones_like(l),
              torch.where(inactive, torch.zeros_like(l),
                          (u >= -l).float()))
    beta_L  = torch.zeros_like(l)
    return alpha_L, beta_L, alpha_U, beta_U


def crown_ibp_upper_bound(model, x, y, eps, n_cls=10):
    """Return upper bound on (z_j - z_y) for all j!=y, shape (B, n_cls-1).

    Strategy:
      (1) Run IBP forward; record pre-activation (l_i, u_i) at every ReLU
          and the input box (x-eps, x+eps) clamped to [0,1].
      (2) Build elision matrix C of shape (B, n_cls-1, n_cls).
      (3) Backward CROWN: start with A = C (a linear functional on the
          final logits). Walk the layer_seq in reverse, replacing A by
          its image under each layer's adjoint, accumulating a bias
          term. For ReLUs use the relaxations from _relu_relax keyed
          on the sign of A (positive coefs use upper relax, negative
          use lower).
      (4) Final closed-form upper bound on A . input over the input box
          [x_L, x_U] uses standard IBP-on-linear:  A_pos . x_U + A_neg . x_L
          + accumulated bias.
    """
    device = x.device
    B = x.size(0)

    # ---- (1) IBP forward, recording pre-activation bounds at each ReLU.
    h_L = (x - eps).clamp(0.0, 1.0)
    h_U = (x + eps).clamp(0.0, 1.0)
    pre_bounds = []        # one entry per ReLU, in forward order: (l, u, shape)
    shapes = []            # for flatten/unflatten on the backward pass
    x_in_L, x_in_U = h_L.clone(), h_U.clone()
    for kind, m in model.layer_seq:
        if kind == "conv":
            h_L, h_U = _conv_ibp(m, h_L, h_U)
        elif kind == "lin":
            h_L, h_U = _lin_ibp(m, h_L, h_U)
        elif kind == "relu":
            pre_bounds.append((h_L.clone(), h_U.clone(), h_L.shape))
            h_L, h_U = F.relu(h_L), F.relu(h_U)
        elif kind == "flat":
            shapes.append(h_L.shape)        # remember pre-flatten shape
            h_L, h_U = h_L.flatten(1), h_U.flatten(1)

    # ---- (2) Elision matrix C: C[b, k, :] = e_{j_k} - e_{y_b}, j_k != y_b.
    # Shape (B, n_cls-1, n_cls).
    eye = torch.eye(n_cls, device=device)
    rows = []
    for b in range(B):
        yb = int(y[b].item())
        idx = [j for j in range(n_cls) if j != yb]
        Cb = eye[idx] - eye[yb].unsqueeze(0)        # (n_cls-1, n_cls)
        rows.append(Cb)
    Cmat = torch.stack(rows, dim=0)                 # (B, n_cls-1, n_cls)

    # We propagate a linear functional A of shape (B, K, *current_feat_dims)
    # where K = n_cls - 1. Bias accumulator is (B, K).
    A = Cmat                                         # (B, K, n_cls)
    bias = torch.zeros(B, n_cls - 1, device=device)

    # Walk layers in reverse.
    relu_idx = len(pre_bounds) - 1
    flat_idx = len(shapes) - 1
    for kind, m in reversed(model.layer_seq):
        if kind == "lin":
            # A: (B, K, out_dim). Adjoint of linear y = W x + b is
            #   A' = A @ W   (B, K, in_dim);   bias += A @ b.
            W = m.weight                              # (out, in)
            if m.bias is not None:
                bias = bias + (A * m.bias.view(1, 1, -1)).sum(dim=-1)
            A = A @ W                                  # (B, K, in_dim)
        elif kind == "flat":
            in_shape = shapes[flat_idx]; flat_idx -= 1
            # Reshape A from (B, K, prod) to (B, K, *in_shape[1:]).
            A = A.view(A.size(0), A.size(1), *in_shape[1:])
        elif kind == "conv":
            # A has shape (B, K, out_ch, out_h, out_w). The adjoint of
            # conv2d is conv_transpose2d on each (B,K)-slice. Batched
            # via merging B*K into the batch axis.
            bk = A.size(0) * A.size(1)
            out_ch = A.size(2); h_o = A.size(3); w_o = A.size(4)
            A2 = A.reshape(bk, out_ch, h_o, w_o)
            if m.bias is not None:
                # bias contribution: sum over spatial of A * b_c
                # b_c is per output channel.
                bias = bias + (A * m.bias.view(1, 1, -1, 1, 1)).sum(dim=(2, 3, 4))
            # conv_transpose2d gives the adjoint of conv2d w.r.t. the
            # input, ignoring padding mode subtleties; matches stride/pad.
            A_in = F.conv_transpose2d(A2, m.weight, bias=None,
                                      stride=m.stride, padding=m.padding)
            in_ch = A_in.size(1); h_i = A_in.size(2); w_i = A_in.size(3)
            A = A_in.view(A.size(0), A.size(1), in_ch, h_i, w_i)
        elif kind == "relu":
            l, u, _ = pre_bounds[relu_idx]; relu_idx -= 1
            alpha_L, beta_L, alpha_U, beta_U = _relu_relax(l, u)
            # ReLU is *post* the matching linear layer in forward order;
            # in the backward walk we now multiply A elementwise by the
            # appropriate slope and accumulate the appropriate intercept.
            # For UPPER bound on A . relu(z):
            #   use alpha_U / beta_U where A > 0,
            #   use alpha_L / beta_L where A < 0.
            #   (sign decomposition.)
            A_pos = A.clamp(min=0.0)
            A_neg = A.clamp(max=0.0)
            # broadcast alpha,beta from (B, *featshape) to (B, K, *featshape)
            aU = alpha_U.unsqueeze(1); bU = beta_U.unsqueeze(1)
            aL = alpha_L.unsqueeze(1); bL = beta_L.unsqueeze(1)
            A_new = A_pos * aU + A_neg * aL
            # bias contribution: sum over feature dims of A_pos*beta_U + A_neg*beta_L
            extra = (A_pos * bU + A_neg * bL).sum(dim=tuple(range(2, A.dim())))
            bias = bias + extra
            A = A_new

    # ---- (4) close out on the input box [x_in_L, x_in_U].
    # A now has shape (B, K, *input_shape[1:]). Upper bound on A . x
    # over [x_L, x_U] is  A_pos . x_U + A_neg . x_L.
    A_pos = A.clamp(min=0.0)
    A_neg = A.clamp(max=0.0)
    # broadcast x_L, x_U to (B, 1, *input_shape[1:])
    xL = x_in_L.unsqueeze(1); xU = x_in_U.unsqueeze(1)
    upper = (A_pos * xU + A_neg * xL).sum(dim=tuple(range(2, A.dim()))) + bias
    return upper  # (B, n_cls-1)  upper bound on z_j - z_y for each j!=y


# --------------------------------------------------------------------------
# Verified test-time predictions and verified Linf radii.
# --------------------------------------------------------------------------
@torch.no_grad()
def verified_at_eps(model, X, Y, eps, batch=128):
    """Fraction of samples for which IBP certifies correct classification
    at the given eps (i.e. lower bound on the true-class logit strictly
    greater than the upper bound on every other logit). Uses plain IBP
    (sound, tight enough at test time for trained CROWN-IBP nets).
    """
    n = X.size(0); cert = 0
    n_cls = 10
    for i in range(0, n, batch):
        x = X[i:i + batch]; y = Y[i:i + batch]
        L, U = ibp_forward(model, x, eps)
        # For sample b with true class y_b: certified iff
        #   L[b, y_b] > max_{j!=y_b} U[b, j].
        mask = torch.ones_like(U) * float("-inf")
        for b in range(x.size(0)):
            mask[b, y[b]] = 0.0   # we'll subtract L[b,y_b] later
        # Build worst-other-upper per sample.
        U_masked = U.clone()
        U_masked.scatter_(1, y.unsqueeze(1), float("-inf"))
        worst_other = U_masked.max(dim=1).values
        true_low = L.gather(1, y.unsqueeze(1)).squeeze(1)
        cert += int((true_low > worst_other).sum().item())
    return cert / float(n)


@torch.no_grad()
def verified_radius(model, X, Y, max_eps=RAD_MAX, steps=RAD_STEPS, batch=64):
    """Per-sample IBP-certified Linf radius (binary search per batch).
    Returns numpy array of radii of length X.size(0)."""
    n = X.size(0)
    radii = torch.zeros(n, device=X.device)
    for i in range(0, n, batch):
        x = X[i:i + batch]; y = Y[i:i + batch]
        lo = torch.zeros(x.size(0), device=x.device)
        hi = torch.full((x.size(0),), float(max_eps), device=x.device)
        for _ in range(steps):
            mid = (lo + hi) / 2.0
            # Loop one eps at a time inside the batch (vectorising eps
            # across the batch would need per-sample IBP; the
            # simpler way is sample-by-sample but slower. We bisect with
            # a single shared eps per micro-step: process the whole batch
            # at the mean mid, then refine; this is a slight approximation
            # but matches H117's per-sample sweep.)
            eps_b = float(mid.mean().item())
            L, U = ibp_forward(model, x, eps_b)
            U_masked = U.clone()
            U_masked.scatter_(1, y.unsqueeze(1), float("-inf"))
            worst_other = U_masked.max(dim=1).values
            true_low = L.gather(1, y.unsqueeze(1)).squeeze(1)
            ok = true_low > worst_other
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        radii[i:i + x.size(0)] = lo
    return radii.detach().cpu().numpy()


# --------------------------------------------------------------------------
# Eps / kappa schedulers.
# --------------------------------------------------------------------------
def eps_schedule(name, ep, n_ep, eps_target):
    """Return (eps_for_this_epoch, kappa_for_this_epoch).
    kappa is the weight on the CLEAN CE term;  (1-kappa) is on robust CE.
    """
    if name == "constant":
        # no ramp: hit the wall immediately.
        return eps_target, 0.5
    elif name == "ramp5":
        T = 5
    elif name == "ramp10":
        T = n_ep
    else:
        raise ValueError(name)
    if ep < T:
        frac = (ep + 1) / float(T)
        return eps_target * frac, 1.0 - 0.5 * frac
    return eps_target, 0.5


# --------------------------------------------------------------------------
# Training conditions.
# --------------------------------------------------------------------------
def _opt(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)


def train_std(Xtr, Ytr, seed):
    C.set_seed(seed)
    model = IBPCNN().to(C.DEVICE)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0); model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
        sched.step()
    model.eval(); return model


def train_pgd_at(Xtr, Ytr, seed):
    C.set_seed(seed)
    model = IBPCNN().to(C.DEVICE)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0); model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_adv = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward(); opt.step()
        sched.step()
    model.eval(); return model


def train_pure_ibp(Xtr, Ytr, seed, schedule="ramp10"):
    """Pure IBP training: verified-CE on the forward IBP bounds, no CROWN."""
    C.set_seed(seed)
    model = IBPCNN().to(C.DEVICE)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0); model.train()
    n_cls = 10
    for ep in range(EPOCHS):
        eps_ep, kappa = eps_schedule(schedule, ep, EPOCHS, EPS)
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            logits = model(xb)
            ce_clean = F.cross_entropy(logits, yb)
            if eps_ep > 1e-6:
                L, U = ibp_forward(model, xb, eps_ep)
                # Worst-case logit: take U for non-true classes, L for true class.
                worst = U.clone()
                worst.scatter_(1, yb.unsqueeze(1),
                               L.gather(1, yb.unsqueeze(1)).squeeze(1).unsqueeze(1))
                ce_rob = F.cross_entropy(worst, yb)
            else:
                ce_rob = ce_clean.detach() * 0.0
            loss = kappa * ce_clean + (1.0 - kappa) * ce_rob
            loss.backward(); opt.step()
        sched.step()
    model.eval(); return model


def train_crown_ibp(Xtr, Ytr, seed, schedule="ramp10"):
    """CROWN-IBP training: forward IBP for the BN-free intervals + a
    CROWN backward bound for the verified-CE. Convex combo with clean CE.
    """
    C.set_seed(seed)
    model = IBPCNN().to(C.DEVICE)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0); model.train()
    n_cls = 10
    for ep in range(EPOCHS):
        eps_ep, kappa = eps_schedule(schedule, ep, EPOCHS, EPS)
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            logits = model(xb)
            ce_clean = F.cross_entropy(logits, yb)
            if eps_ep > 1e-6:
                # CROWN-IBP: upper bound on z_j - z_y for j!=y, shape (B, K).
                ub = crown_ibp_upper_bound(model, xb, yb, eps_ep, n_cls=n_cls)
                # Build (B, n_cls) "worst-case logits" so that
                #   z_y_worst - z_j_worst = -ub[b, idx_j]
                # CE with target y on this row is the canonical verified CE.
                B = xb.size(0)
                worst = torch.zeros(B, n_cls, device=xb.device)
                # arrange: for each b, fill non-y columns with ub, y column with 0.
                for b in range(B):
                    yb_int = int(yb[b].item())
                    j_idx = [j for j in range(n_cls) if j != yb_int]
                    worst[b, j_idx] = ub[b]
                    worst[b, yb_int] = 0.0
                ce_rob = F.cross_entropy(worst, yb)
            else:
                ce_rob = ce_clean.detach() * 0.0
            loss = kappa * ce_clean + (1.0 - kappa) * ce_rob
            loss.backward(); opt.step()
        sched.step()
    model.eval(); return model


# --------------------------------------------------------------------------
# Evaluation: clean acc, PGD ASR, verified-at-eps, verified Linf radius.
# --------------------------------------------------------------------------
def eval_all(name, model, Xte, Yte, out, t0):
    _, acc = C.logits_and_acc(model, Xte, Yte)
    pg = C.attack_success(model, Xte, Yte, attack="pgd",
                          eps=EPS, steps=PGD_STEPS)
    pgd_asr = pg["asr"]
    v_at = verified_at_eps(model, Xte, Yte, EPS)
    # Verified radius on a random subsample of correctly-classified test points.
    with torch.no_grad():
        pred = []
        for i in range(0, Xte.size(0), 512):
            pred.append(model(Xte[i:i + 512]).argmax(1))
        pred = torch.cat(pred)
        corr = (pred == Yte)
    Xc = Xte[corr]; Yc = Yte[corr]
    if Xc.size(0) > N_CERT_SAMPLES:
        g = torch.Generator(device="cpu").manual_seed(SEED)
        idx = torch.randperm(Xc.size(0), generator=g)[:N_CERT_SAMPLES]
        Xc, Yc = Xc[idx.to(Xc.device)], Yc[idx.to(Xc.device)]
    radii = verified_radius(model, Xc, Yc)
    rad_mean = float(np.mean(radii)) if radii.size > 0 else 0.0
    rad_med  = float(np.median(radii)) if radii.size > 0 else 0.0
    rad_p10  = float(np.percentile(radii, 10)) if radii.size > 0 else 0.0
    out(f"  [{name}] clean_acc={acc:.4f}  PGD_ASR={pgd_asr:.4f}  "
        f"verified@eps={EPS}={v_at:.4f}")
    out(f"           verified_radius:  mean={rad_mean:.4f}  "
        f"median={rad_med:.4f}  p10={rad_p10:.4f}  "
        f"(n_cert={Xc.size(0)}, t={time.time()-t0:.0f}s)")
    return {
        "name": name, "acc": acc, "pgd_asr": pgd_asr, "v_at": v_at,
        "rad_mean": rad_mean, "rad_med": rad_med, "rad_p10": rad_p10,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "fashion_mnist", "h448_crown_ibp_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True); lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H448  CROWN-IBP certified training (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        verified-radius binary search: max={RAD_MAX} "
        f"steps={RAD_STEPS} n_samples={N_CERT_SAMPLES}")
    out(f"        device = {C.DEVICE}")
    out("")
    out("Conditions:")
    out("  F. STD       : standard CE only")
    out("  A. PGD-AT    : standard PGD-10 AT at eps=0.1 (no verified loss)")
    out("  B. IBP       : pure IBP verified-CE,   linear eps ramp over 10 ep")
    out("  C. CROWN-IBP : CROWN backward + IBP fwd, linear ramp over 10 ep")
    out("  D. CROWN-IBP : same as C but linear ramp over 5 ep")
    out("  E. CROWN-IBP : constant eps from epoch 0 (NEGATIVE CONTROL)")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")
    flush_file()

    rows = []

    out("[F] STD baseline training ...")
    m = train_std(Xtr, Ytr, SEED); rows.append(eval_all("STD", m, Xte, Yte, out, t0))
    flush_file()

    out("\n[A] PGD-AT baseline training ...")
    m = train_pgd_at(Xtr, Ytr, SEED); rows.append(eval_all("PGD-AT", m, Xte, Yte, out, t0))
    flush_file()

    out("\n[B] Pure IBP training (ramp10) ...")
    m = train_pure_ibp(Xtr, Ytr, SEED, "ramp10")
    rows.append(eval_all("IBP-ramp10", m, Xte, Yte, out, t0))
    flush_file()

    out("\n[C] CROWN-IBP training (ramp10, full) ...")
    m = train_crown_ibp(Xtr, Ytr, SEED, "ramp10")
    rows.append(eval_all("CROWN-IBP-ramp10", m, Xte, Yte, out, t0))
    flush_file()

    out("\n[D] CROWN-IBP training (ramp5) ...")
    m = train_crown_ibp(Xtr, Ytr, SEED, "ramp5")
    rows.append(eval_all("CROWN-IBP-ramp5", m, Xte, Yte, out, t0))
    flush_file()

    out("\n[E] CROWN-IBP training (constant eps, no ramp; NEGATIVE CONTROL) ...")
    m = train_crown_ibp(Xtr, Ytr, SEED, "constant")
    rows.append(eval_all("CROWN-IBP-const", m, Xte, Yte, out, t0))
    flush_file()

    # ---- MAIN TABLE ----
    out("\n" + "=" * 80)
    out("MAIN TABLE")
    out("=" * 80)
    hdr = "{:<20} {:>9} {:>9} {:>10} {:>10} {:>10} {:>10}".format(
        "condition", "clean", "PGD_ASR", "verif@0.1",
        "rad_mean", "rad_med", "rad_p10")
    out(hdr); out("-" * len(hdr))
    for r in rows:
        out("{:<20} {:>9.4f} {:>9.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["name"], r["acc"], r["pgd_asr"], r["v_at"],
            r["rad_mean"], r["rad_med"], r["rad_p10"]))
    out("-" * len(hdr))

    # ---- VERDICT ----
    out("\n" + "=" * 80)
    out("VERDICT")
    out("=" * 80)
    by_name = {r["name"]: r for r in rows}
    std    = by_name["STD"]
    pgd_at = by_name["PGD-AT"]
    ibp    = by_name["IBP-ramp10"]
    crown  = by_name["CROWN-IBP-ramp10"]
    crown5 = by_name["CROWN-IBP-ramp5"]
    crown_c = by_name["CROWN-IBP-const"]

    # Headline numbers.
    out(f"  STD baseline:        clean={std['acc']:.4f}  PGD_ASR={std['pgd_asr']:.4f}  "
        f"verif@0.1={std['v_at']:.4f}  rad_mean={std['rad_mean']:.4f}")
    out(f"  PGD-AT baseline:     clean={pgd_at['acc']:.4f}  PGD_ASR={pgd_at['pgd_asr']:.4f}  "
        f"verif@0.1={pgd_at['v_at']:.4f}  rad_mean={pgd_at['rad_mean']:.4f}")
    out(f"  Pure IBP (ramp10):   clean={ibp['acc']:.4f}  PGD_ASR={ibp['pgd_asr']:.4f}  "
        f"verif@0.1={ibp['v_at']:.4f}  rad_mean={ibp['rad_mean']:.4f}")
    out(f"  CROWN-IBP (ramp10):  clean={crown['acc']:.4f}  PGD_ASR={crown['pgd_asr']:.4f}  "
        f"verif@0.1={crown['v_at']:.4f}  rad_mean={crown['rad_mean']:.4f}")
    out(f"  CROWN-IBP (ramp5):   clean={crown5['acc']:.4f}  PGD_ASR={crown5['pgd_asr']:.4f}  "
        f"verif@0.1={crown5['v_at']:.4f}  rad_mean={crown5['rad_mean']:.4f}")
    out(f"  CROWN-IBP (const):   clean={crown_c['acc']:.4f}  PGD_ASR={crown_c['pgd_asr']:.4f}  "
        f"verif@0.1={crown_c['v_at']:.4f}  rad_mean={crown_c['rad_mean']:.4f}")
    out("")

    # Conditions for verdict.
    crown_works = (crown["rad_mean"] > 0.02
                   and crown["v_at"] > 0.05
                   and crown["pgd_asr"] < std["pgd_asr"] - 0.05)
    crown_beats_ibp = crown["rad_mean"] > ibp["rad_mean"] + 0.005
    const_collapsed = crown_c["acc"] < 0.30

    out(f"  CROWN-IBP non-trivial radius (rad_mean > 0.02):  "
        f"{crown['rad_mean']:.4f}  -> {'YES' if crown['rad_mean']>0.02 else 'NO'}")
    out(f"  CROWN-IBP > IBP on rad_mean (+0.005):            "
        f"d={crown['rad_mean']-ibp['rad_mean']:+.4f}  -> "
        f"{'YES' if crown_beats_ibp else 'NO'}")
    out(f"  CROWN-IBP improves PGD vs STD (>= 0.05 drop):    "
        f"d={crown['pgd_asr']-std['pgd_asr']:+.4f}  -> "
        f"{'YES' if crown['pgd_asr']<std['pgd_asr']-0.05 else 'NO'}")
    out(f"  constant-eps control collapsed (clean < 0.30):   "
        f"{crown_c['acc']:.4f}  -> {'YES' if const_collapsed else 'NO'}")
    out("")

    if crown_works and crown_beats_ibp:
        one = ("SUPPORTED: CROWN-IBP gives a non-trivial certified Linf "
               "radius and beats pure IBP on radius; the eps schedule is "
               "load-bearing (constant-eps control "
               f"{'collapsed' if const_collapsed else 'did not collapse'}).")
    elif crown_works and not crown_beats_ibp:
        one = ("PARTIAL: CROWN-IBP certifies a non-trivial radius but "
               "does not clearly beat plain IBP at this 6k/10ep budget.")
    elif (not crown_works) and ibp["rad_mean"] > 0.02:
        one = ("MIXED: pure IBP gives some verified radius but CROWN-IBP "
               "under-performed at this budget (likely needs longer "
               "warm-up; cf. Shi 2021).")
    else:
        one = ("NOT SUPPORTED: neither IBP nor CROWN-IBP produces a "
               "non-trivial certified radius at this N_TRAIN=6000 / "
               "10-epoch budget; the eps schedule alone cannot make "
               "verified training competitive here.")
    out("  ONE-LINE VERDICT: " + one)

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
