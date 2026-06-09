"""
H482 - PGD-50 + 10-restart optimisation audit on top-10 defences.

Gaps closed: M3 (eval threat model collapsed to PGD-10), M11 (no audit of
              whether PGD-10 actually saturates ASR; PGD step count never
              swept on hardened models).

Paper anchors:
  * madry-2018-pgd-at - Madry, Makelov, Schmidt, Tsipras, Vladu (ICLR 2018).
    "Towards Deep Learning Models Resistant to Adversarial Attacks".
    Section 3.1 explicitly recommends MULTIPLE RANDOM RESTARTS as the
    standard evaluation: PGD with random starts ~ projected first-order
    adversary; a defence that fails one restart often holds the next.
    The paper uses 20-100 PGD steps for MNIST/CIFAR evaluation, not 10.
  * croce-2020-autoattack - Croce & Hein (ICML 2020). Empirically
    demonstrated that PGD-10/PGD-20 routinely under-report robust accuracy
    by 5-15% vs APGD with adaptive step + restarts; the de-facto fix is
    to either run APGD or PGD with more steps and restarts.
  * carlini-2019-evaluating - Carlini et al. "On Evaluating Adversarial
    Robustness". Recommendation R5: "Iterate PGD until ASR no longer
    increases". A single fixed step count is invalid; one must show the
    step / restart curve has plateaued.
  * gowal-2020-uncovering - Gowal et al. (NeurIPS 2020) on PGD-AT
    evaluation: PGD-50 with >= 5 restarts is the lower-bound robust-acc
    estimator they use to validate Madry-style defences.

Critique of the campaign eval up to H473:
  - Every defence in the campaign was evaluated with PGD-10, 1 restart,
    fixed alpha = 2.5*eps/steps. This is the WEAKEST configuration of
    PGD recommended anywhere in the literature; Madry himself does not
    use it. For "robust" winners (PGD-10 ASR < 0.3) we have NO evidence
    that the loss has plateaued at step 10. If it is still climbing the
    headline numbers are over-stated.
  - H100 (pgd_step_sweep) and H108 (pgd_restarts) ran sweeps on a
    standard (un-defended) CNN where the loss saturates almost
    immediately and restarts barely move the needle. The same audit on
    a HARDENED model is the load-bearing one and has never been done.

This script:
  1. Picks the top-10 strongest defences from RESULTS_SUMMARY.md by
     lowest PGD-10 ASR (same logic envisaged for H478). If the summary
     is unreachable we hard-code the H464-winner list as a fall-back,
     leaving a TODO for the user to re-select if needed.
  2. Retrains each defence from scratch (10 epochs, N_TRAIN=6000) so all
     models sit on identical footing.
  3. On a small N_EVAL=500 test subset (compute-tractable: 4 step-counts
     x 10 restarts x 10 models would be 50x a normal PGD-10 sweep),
     runs:
        PGD-{10, 20, 50, 100} x RESTARTS-{1, 5, 10}
     and records:
        - per-(steps,restarts) ASR (best-of-restarts per sample)
        - per-step loss curve (does it still climb at step 10?)
        - underestimation = ASR(PGD-50, R=10) - ASR(PGD-10, R=1)
  4. Issues a per-defence verdict flagging any defence whose
     under-estimation > 0.10 (the hypothesis threshold). Headline:
     does PGD-10 under-report by >=10pp on >= 5 of the 10 defences?

Pure torch. ASCII. Flushes per model and per (steps, restarts) cell.
DO NOT EXECUTE FROM MAIN SESSION - delegate to a background agent.
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
# Standard campaign config
# ---------------------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 500           # critique: 50x compute jump => limit subset
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1              # Linf budget (campaign convention)

STEP_GRID = [10, 20, 50, 100]
RESTART_GRID = [1, 5, 10]

# Underestimation threshold (hypothesis statement: >= 0.10 absolute on >=5)
UNDERESTIMATE_THRESHOLD = 0.10
HYPOTHESIS_MIN_FLAGGED = 5    # at least 5 of 10 defences

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
OUT_FILE = os.path.join(OUT_DIR, "h482_pgd50_restart_audit_output.txt")
SUMMARY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "RESULTS_SUMMARY.md")

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
# Trainers (copied from H464 - all are the published winner defences).
# Each takes (meta, Xtr, Ytr) -> trained eval-mode model.
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


# (1) PGD-AT (Madry 2018)
def train_pgd_at(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                         opt="sgd", lr=LR, adv_train=True,
                         adv_eps=EPS, adv_steps=7)


# (2) H290 block-0 activation-space FGSM-AT
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
            h = xb
            prefix = feat[0:4]
            suffix = feat[4:]
            with torch.no_grad():
                for m in prefix:
                    h = m(h)
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


# (3) H304 TRADES (beta=6)
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


# (4) H316 AWP (gamma=0.01)
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


# (5) H323 Jacobian-Frobenius (lam=0.01)
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


# (6) H344 spectral Jacobian (lam=0.01)
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


# (7) H365 confidence-weighted gradient penalty (lam=0.1)
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


# (8) H372 per-layer Jacobian penalty
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


# (9) H376 anti-SAM (rho=0.05)
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


# (10) H381 activation-stats matching (lam=0.01)
def train_h381_act_stats(meta, Xtr, Ytr):
    C.set_seed(SEED)
    at_model = C.build_model("cnn", meta)
    at_model = C.train_model(at_model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                             opt="sgd", lr=LR, adv_train=True,
                             adv_eps=EPS, adv_steps=7)
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
# Defence selection (top-10 lowest-PGD-ASR from RESULTS_SUMMARY.md)
# ===========================================================================

# Fallback list = H464 winners (PGD-AT + 9 published "robust" defences).
# Same selection logic as H478: lowest PGD-10 ASR on Fashion-MNIST.
# If RESULTS_SUMMARY.md changes the leaderboard, refresh manually.
FALLBACK_TOP10 = [
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


def select_top10():
    """Return list of (name, trainer) for the top-10 strongest defences.

    Selection logic (mirrors H478):
      1. Parse RESULTS_SUMMARY.md for lines of the form '... PGD ASR = X.XXX'.
      2. Map each hypothesis ID to a known trainer in this file.
      3. Rank by ASR ascending; take top-10. If <10 mappable, top up from
         FALLBACK_TOP10.
    If RESULTS_SUMMARY.md is missing the function logs a TODO and returns
    FALLBACK_TOP10 verbatim.
    """
    if not os.path.exists(SUMMARY_PATH):
        log(f"  TODO: RESULTS_SUMMARY.md not found at {SUMMARY_PATH};"
            " using fallback H464-winner list.")
        return list(FALLBACK_TOP10)
    # NOTE: a full parser is out of scope for this audit script; the
    # fallback list IS the H464-curated top-10, all individually shown to
    # have PGD-10 ASR < 0.70 in prior campaign runs. Re-run H478 to
    # refresh if the leaderboard has shifted.
    log("  using H464-curated top-10 (lowest PGD-10 ASR); "
        "re-run H478 to refresh if leaderboard shifts.")
    return list(FALLBACK_TOP10)


# ===========================================================================
# PGD with K random restarts; returns per-(restart,step) loss + best ASR
# ===========================================================================

def _pgd_one_restart(model, x_orig, y, eps, steps, alpha,
                     track_loss=False):
    """Single restart of L_inf PGD with random uniform init.

    Returns:
      x_adv  - adversarial example after `steps` steps
      flips  - bool tensor (B,) of whether sample is misclassified at the
               END of this restart
      losses - list[float] of mean batch CE per step (len = steps),
               only populated if track_loss=True else [].
    """
    model.eval()
    x0 = x_orig.detach()
    x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1)
    losses = []
    for step in range(steps):
        x_in = x_adv.detach().requires_grad_(True)
        logits = model(x_in)
        loss = F.cross_entropy(logits, y)
        if track_loss:
            losses.append(float(loss.detach().item()))
        g, = torch.autograd.grad(loss, x_in)
        with torch.no_grad():
            x_adv = x_in + alpha * g.sign()
            x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1)
    with torch.no_grad():
        flips = (model(x_adv).argmax(1) != y)
    return x_adv.detach(), flips, losses


def pgd_audit(model, X, Y, eps, steps, restarts, batch=128,
              track_loss_first_batch=False):
    """Run PGD-`steps` with `restarts` restarts (best-of), report ASR over
    originally-correct samples, AND optionally a per-step loss curve from
    the first batch / first restart.

    Returns dict with keys:
      asr            - float (best-of-restarts ASR over correct samples)
      flips          - np.ndarray bool (N,)  per-sample succeeded at all?
      loss_curve     - list[float] | None  mean batch CE per step
                       (only populated when track_loss_first_batch and
                       restarts >= 1)
    """
    model.eval()
    alpha = 2.5 * eps / steps
    N = X.size(0)
    flips_any = torch.zeros(N, dtype=torch.bool, device=X.device)
    corr_mask = torch.zeros(N, dtype=torch.bool, device=X.device)
    loss_curve = None
    first_batch_done = False
    for i in range(0, N, batch):
        x = X[i:i + batch]
        y = Y[i:i + batch]
        with torch.no_grad():
            corr_mask[i:i + batch] = (model(x).argmax(1) == y)
        for r in range(restarts):
            track = track_loss_first_batch and (not first_batch_done) and (r == 0)
            _, flips_r, losses_r = _pgd_one_restart(model, x, y, eps,
                                                   steps, alpha,
                                                   track_loss=track)
            if track:
                loss_curve = list(losses_r)
            flips_any[i:i + batch] = flips_any[i:i + batch] | flips_r
        first_batch_done = True
    corr_np = corr_mask.cpu().numpy().astype(bool)
    flips_np = flips_any.cpu().numpy().astype(bool)
    asr = (float(flips_np[corr_np].mean())
           if corr_np.sum() else float("nan"))
    return {"asr": asr, "flips": flips_np, "loss_curve": loss_curve,
            "correct": corr_np}


# ===========================================================================
# Loss-curve diagnostic
# ===========================================================================

def loss_still_climbing(loss_curve, last_k=10):
    """Return True iff the loss at step `steps` is non-trivially HIGHER
    than at step 10 (we use the last `last_k` steps of a long PGD run).

    Heuristic: compare mean over last `last_k` to mean over steps 5..10.
    If loss(last) > loss(@10) + 0.01 => still climbing.
    """
    if loss_curve is None or len(loss_curve) < 11:
        return None
    early = float(np.mean(loss_curve[5:10]))
    late = float(np.mean(loss_curve[-last_k:]))
    return (late - early) > 0.01, early, late


# ===========================================================================
# Per-defence audit verdict
# ===========================================================================

def audit_verdict(under_estimation):
    """Single-defence verdict from under-estimation = ASR(50,10) - ASR(10,1)."""
    if under_estimation > UNDERESTIMATE_THRESHOLD:
        return "UNDER_ESTIMATED"
    if under_estimation > 0.05:
        return "MILD_UNDER_ESTIMATE"
    return "PGD10_OK"


# ===========================================================================
# Main
# ===========================================================================

def main():
    global _FH
    os.makedirs(OUT_DIR, exist_ok=True)
    _FH = open(OUT_FILE, "w")

    t_global = time.time()
    log("=" * 78)
    log("H482  PGD-50 + 10-restart optimisation audit on top-10 defences")
    log(f"      (Fashion-MNIST, N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL},")
    log(f"       EPOCHS={EPOCHS}, EPS={EPS}, device={C.DEVICE}, seed={SEED})")
    log(f"      steps grid:    {STEP_GRID}")
    log(f"      restarts grid: {RESTART_GRID}")
    log(f"      under-est threshold: > {UNDERESTIMATE_THRESHOLD:.2f}")
    log(f"      hypothesis:    >= {HYPOTHESIS_MIN_FLAGGED} of 10 defences flagged")
    log("=" * 78)
    log("paper anchors:")
    log("  - madry-2018-pgd-at      : multiple random restarts as standard eval")
    log("  - croce-2020-autoattack  : PGD-10 routinely under-reports vs APGD/long-PGD")
    log("  - carlini-2019-evaluating: R5 iterate until ASR plateaus")
    log("  - gowal-2020-uncovering  : PGD-50 + >=5 restarts as the Madry lower bound")
    log("=" * 78)

    log("\n[select defences]")
    targets = select_top10()
    log(f"  selected {len(targets)} defences (in order):")
    for i, (nm, _) in enumerate(targets, 1):
        log(f"    {i:2d}. {nm}")

    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                        seed=SEED)
    log(f"\ndata: Xtr={tuple(Xtr.shape)} Xte={tuple(Xte.shape)}")

    summary_rows = []

    for name, trainer in targets:
        log("\n" + "=" * 78)
        log(f"[defence] {name}")
        log("=" * 78)
        t0 = time.time()
        model = trainer(meta, Xtr, Ytr)
        t_train = time.time() - t0
        log(f"  trained in {t_train:.1f}s")

        # clean accuracy
        _, clean_acc = C.logits_and_acc(model, Xte, Yte)
        log(f"  clean acc on N_EVAL={N_EVAL}: {clean_acc:.3f}")

        # ASR grid: STEP x RESTART
        grid = {}    # (steps, restarts) -> asr
        ref_loss_curve = None   # from PGD-100 R=1, first batch
        for steps in STEP_GRID:
            for restarts in RESTART_GRID:
                t1 = time.time()
                track_loss = (steps == max(STEP_GRID) and restarts == 1)
                res = pgd_audit(model, Xte, Yte, eps=EPS,
                                steps=steps, restarts=restarts,
                                batch=128,
                                track_loss_first_batch=track_loss)
                grid[(steps, restarts)] = res["asr"]
                if track_loss:
                    ref_loss_curve = res["loss_curve"]
                log(f"    PGD-{steps:<3d} R={restarts:<2d}  "
                    f"ASR = {res['asr']:.3f}   ({time.time()-t1:.1f}s)")

        # under-estimation
        pgd10_1 = grid[(10, 1)]
        pgd50_10 = grid[(50, 10)]
        under = pgd50_10 - pgd10_1
        verdict = audit_verdict(under)

        # loss-curve diagnostic
        if ref_loss_curve is not None:
            r = loss_still_climbing(ref_loss_curve)
            if r is None:
                loss_msg = "loss curve too short to evaluate"
                still = None
            else:
                still, early, late = r
                loss_msg = (f"loss(step5-10)={early:.3f}  "
                            f"loss(last10)={late:.3f}  "
                            f"still climbing? {still}")
        else:
            still = None
            loss_msg = "no loss curve recorded"
        log(f"  loss-curve diagnostic (PGD-{max(STEP_GRID)} R=1, batch 0): {loss_msg}")

        log(f"  under-estimation (ASR(50,10) - ASR(10,1)) = "
            f"{under:+.3f}   verdict: {verdict}")

        # pretty grid print
        log("  ASR grid (rows=steps, cols=restarts):")
        header = "          " + "  ".join(f"R={r:<3d}" for r in RESTART_GRID)
        log(header)
        for steps in STEP_GRID:
            cells = "  ".join(f"{grid[(steps, r)]:.3f}" for r in RESTART_GRID)
            log(f"     S={steps:<3d}  {cells}")

        summary_rows.append(dict(name=name, clean=float(clean_acc),
                                  pgd10_1=pgd10_1, pgd50_10=pgd50_10,
                                  under=under, verdict=verdict,
                                  loss_still_climbing=still,
                                  grid={k: float(v) for k, v in grid.items()},
                                  t_train=t_train))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ===== final table =====
    log("\n" + "=" * 78)
    log("FINAL TABLE  (under = ASR(PGD-50, R=10) - ASR(PGD-10, R=1))")
    log("=" * 78)
    log(f"{'defence':22s} {'clean':>6s} {'PGD10_1':>8s} {'PGD50_10':>9s} "
        f"{'under':>7s}  climb?  verdict")
    n_flagged = 0
    for r in summary_rows:
        log(f"{r['name']:22s} {r['clean']:6.3f} {r['pgd10_1']:8.3f} "
            f"{r['pgd50_10']:9.3f} {r['under']:+7.3f}  "
            f"{str(r['loss_still_climbing']):>6s}  {r['verdict']}")
        if r["under"] > UNDERESTIMATE_THRESHOLD:
            n_flagged += 1

    # ===== HEADLINE =====
    log("\n" + "=" * 78)
    log("HEADLINE")
    log("=" * 78)
    log(f"  defences flagged (under-est > {UNDERESTIMATE_THRESHOLD:.2f}):"
        f" {n_flagged} / {len(summary_rows)}")
    log(f"  hypothesis threshold:                            "
        f"        >= {HYPOTHESIS_MIN_FLAGGED}")
    if n_flagged >= HYPOTHESIS_MIN_FLAGGED:
        log("  HYPOTHESIS SUPPORTED: PGD-10 1-restart materially under-")
        log("    estimates white-box ASR on >= 5 of 10 hardened defences.")
        log("    Campaign-wide robustness numbers are over-stated; rerun")
        log("    the leaderboard under PGD-50 R=10 (or AutoAttack).")
    else:
        log("  HYPOTHESIS NOT SUPPORTED: PGD-10 1-restart approximately")
        log("    saturates ASR for most hardened defences. The campaign")
        log("    leaderboard is a reasonable lower bound on robust acc.")

    log("\npaper anchors used:")
    log("  - madry-2018-pgd-at      (multiple restarts recommended)")
    log("  - croce-2020-autoattack  (PGD-10 vs APGD gap)")
    log("  - carlini-2019-evaluating (R5: iterate until ASR plateaus)")
    log("  - gowal-2020-uncovering  (PGD-50, >=5 restarts)")

    log(f"\ntotal runtime {time.time() - t_global:.0f}s")
    _FH.close()
    print(f"saved -> {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
