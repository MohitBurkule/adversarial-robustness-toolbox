"""
H468 - Adaptive attack on H411 VOneNet (gap M4 + M12).

H411 finding (RESULTS_SUMMARY): a fixed Gabor V1 front-end alone (sigma=0) gives
no PGD robustness; adding stochastic neuronal noise (sigma=0.25) drops PGD ASR
from 0.962 -> 0.902 (gain 0.060). H411 already ran EOT-PGD with K=8 noise draws
and reported the gain was "mostly real". This is the M4 gap: the H411 audit was
not strong enough. The Athalye-2018 / Carlini-2023 protocol for stochastic
defenses is:
  (a) sweep K in EOT-PGD until ASR saturates (K=20 minimum, K=50 recommended);
  (b) attack THROUGH the fixed Gabor front-end (the front-end is a known fixed
      linear operator -> we can construct adversarial inputs that target its
      OUTPUT directly via a per-feature eps-budget = eps * Lip(Gabor));
  (c) BPDA-style feature-space attack: average the stochastic-neuron output and
      attack the deterministic surrogate.

This script:
  1. Retrains H411 VOneNet at sigma=0.25 (the H411 winner) at the standard config.
  2. Measures ASR(K) for K in {1, 8, 20, 50} via EOT-PGD in input space.
  3. Computes the analytic Lipschitz constant of the Gabor front-end
     (Lip = max singular value of the conv weights, plus contribution from
     the magnitude branch which is 1-Lipschitz, plus ReLU which is 1-Lip).
  4. Runs a feature-space PGD attack: optimise delta in input space but bound
     the attack budget in the FEATURE space at eps * L_gabor; equivalent to
     attacking the post-Gabor representation while honouring the input-Linf
     budget. This is the "inverse-Gabor / through-Gabor" probe.
  5. Reports a verdict: is the H411 stochastic gain real under K=50 EOT?

Refs:
  - Dapello et al. 2020 (VOneNet, NeurIPS) - original defense.
  - Athalye, Carlini & Wagner 2018 (ICML, "Obfuscated Gradients...") - EOT-PGD,
    BPDA, the K-sweep saturation test for stochastic defenses.
  - Carlini 2023 ("A LLM Can Defeat Itself...") and Carlini-Wagner family on
    "evaluating defenses": demand ASR-vs-K saturation curves, not single-K.
  - Tramer et al. 2020 ("On Adaptive Attacks") - adaptive attack methodology.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9,wd=5e-4),
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01, width=32.
No execution here - this script is written but not run.
"""
import os
import sys
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C
from hypotheses.h411_vonenet_gabor_frontend import (
    VOneBlock, VOneNet, build_gabor_bank,
    KSIZE, N_ORIENT, FREQS, STRIDE, PADDING, WIDTH,
)

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
SIGMA = 0.25                          # H411 winner
EOT_KS = [1, 8, 20, 50]               # ASR-vs-K saturation sweep
EOT_KS_FEATURE = 20                   # EOT samples used inside feature-space PGD
N_FEATURE_BATCH = 256

META = {"channels": 1, "size": 28, "n_classes": 10}


# ---------------------------------------------------------------------------
# training (identical to H411)
# ---------------------------------------------------------------------------
def train_net(model, Xtr, Ytr, seed):
    C.set_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# EOT-PGD with configurable K (n_samples)
# ---------------------------------------------------------------------------
def eot_pgd(model, x, y, eps, steps, alpha, n_samples, random_start=True):
    """EOT-PGD: at each step, average the input-space gradient over n_samples
    fresh noise draws through the stochastic model. K=1 reduces to vanilla PGD
    on a single noise sample per step."""
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        xa = (xa + torch.empty_like(xa).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        grad = torch.zeros_like(xa)
        for _s in range(n_samples):
            xa_ = xa.detach().clone().requires_grad_(True)
            loss = F.cross_entropy(model(xa_), y)
            g, = torch.autograd.grad(loss, xa_)
            grad = grad + g
        grad = grad / float(n_samples)
        xa = xa.detach() + alpha * grad.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


@torch.no_grad()
def _flip_eval_eot(model, X, Y, Xadv, n_eval_draws=5, batch=256):
    """Score adversarial examples by averaging the logits over n_eval_draws
    noise samples (so a one-shot lucky noise draw cannot flip the verdict)."""
    flips, corr = [], []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        xa = Xadv[i:i + batch]
        # majority over n_eval_draws to defang inference-time lottery
        votes_c = torch.zeros(x.size(0), META["n_classes"], device=x.device)
        votes_a = torch.zeros(x.size(0), META["n_classes"], device=x.device)
        for _ in range(n_eval_draws):
            votes_c += F.softmax(model(x), dim=1)
            votes_a += F.softmax(model(xa), dim=1)
        corr.append((votes_c.argmax(1) == y).cpu())
        flips.append((votes_a.argmax(1) != y).cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() else float("nan")


def eot_pgd_asr(model, X, Y, K, batch=N_FEATURE_BATCH):
    model.eval()
    advs = []
    for i in range(0, X.size(0), batch):
        advs.append(eot_pgd(model, X[i:i + batch], Y[i:i + batch],
                            EPS, PGD_STEPS, PGD_ALPHA, K))
    Xadv = torch.cat(advs, dim=0)
    return _flip_eval_eot(model, X, Y, Xadv)


# ---------------------------------------------------------------------------
# Analytic Lipschitz constant of the Gabor front-end.
#
# Front-end = [simple_cells; complex_cells] applied to conv(Wgabor, x):
#   simple   = ReLU(g),                 op-norm wrt g  = 1
#   complex  = sqrt(g0^2 + g90^2 + e),  it is 1-Lip wrt g (subgradient norm <=1)
# Stacking two 1-Lip maps on disjoint subsets of g gives a sqrt(2)-Lip map (L2).
# The Linf->L2 op-norm of the Gabor conv is bounded by the spectral norm of the
# unfolded matrix (we compute it via power iteration on conv weights).
# We report L = sqrt(2) * spectral_norm(Wgabor_conv).
# ---------------------------------------------------------------------------
@torch.no_grad()
def gabor_spectral_norm(conv_layer, input_shape, n_iter=40, device=None):
    """Power-iterate to estimate the operator 2-norm of a fixed conv layer.
    Returns sigma_max of the linear map x -> conv(x)."""
    device = device or next(conv_layer.parameters()).device
    u = torch.randn(1, *input_shape, device=device)
    u = u / (u.norm() + 1e-12)
    for _ in range(n_iter):
        v = conv_layer(u)                         # forward
        vn = v.norm() + 1e-12
        v = v / vn
        # adjoint via conv_transpose with same weights
        u_new = F.conv_transpose2d(v, conv_layer.weight,
                                   stride=conv_layer.stride,
                                   padding=conv_layer.padding)
        # crop / pad u_new back to u's shape if needed
        if u_new.shape != u.shape:
            # central crop or pad
            target = u.shape
            diff = [t - s for t, s in zip(target[-2:], u_new.shape[-2:])]
            if diff[0] >= 0 and diff[1] >= 0:
                pad = (diff[1] // 2, diff[1] - diff[1] // 2,
                       diff[0] // 2, diff[0] - diff[0] // 2)
                u_new = F.pad(u_new, pad)
            else:
                # crop
                h, w = u_new.shape[-2:]
                th, tw = target[-2:]
                sy = (h - th) // 2
                sx = (w - tw) // 2
                u_new = u_new[..., sy:sy + th, sx:sx + tw]
        un = u_new.norm() + 1e-12
        u = u_new / un
    # final estimate
    v = conv_layer(u)
    return float(v.norm() / (u.norm() + 1e-12))


def gabor_lipschitz(vone_block, input_shape, device):
    """Linf->L2 Lipschitz upper bound for the (deterministic part of the)
    VOneBlock: simple-ReLU || complex-magnitude branches stacked.
    Returns the L2-op-norm bound and the per-pixel input dim factor."""
    sigma_max = gabor_spectral_norm(vone_block.conv, input_shape, device=device)
    # branches: simple (1-Lip) + complex (1-Lip), stacked -> sqrt(2)
    L_branches = math.sqrt(2.0)
    L = L_branches * sigma_max
    # convert L2 op-norm to Linf->Linf via the input/output dim factors:
    # ||A x||_inf <= ||A||_inf,inf which is row-sum; we bound it crudely by
    # ||A||_2 * sqrt(out_dim). We report both numbers.
    return {"sigma_max_gabor": sigma_max,
            "L_op_l2": L,
            "L_op_linf_upper": L * math.sqrt(np.prod(input_shape))}


# ---------------------------------------------------------------------------
# Feature-space (BPDA-style) PGD attack.
#
# We treat the stochastic-neuron output as the "feature" and attack a
# DETERMINISTIC surrogate where the noise is replaced by its expectation (=0).
# Gradient is taken through (a) the fixed Gabor conv (we have its exact
# gradient), (b) the deterministic ReLU+magnitude (1-Lip; honest gradient),
# (c) the trainable backbone. At forward eval we still pass the perturbed input
# through the FULL stochastic model and average predictions.
# This is the BPDA recipe: replace a non-differentiable / stochastic component
# by a differentiable surrogate at gradient time, evaluate with the real model.
# ---------------------------------------------------------------------------
class DeterministicSurrogate(nn.Module):
    """Surrogate of a VOneNet model with sigma_neuron forced to 0 at gradient
    time. Shares all parameters with the real model (so it sees the actual
    learned weights), differing only in the noise draw."""
    def __init__(self, real_model):
        super().__init__()
        self.real = real_model
        # snapshot of saved sigma; we will toggle it during forward
        self._sigma_save = real_model.vone.sigma_neuron

    def forward(self, x):
        self.real.vone.sigma_neuron = 0.0
        try:
            out = self.real(x)
        finally:
            self.real.vone.sigma_neuron = self._sigma_save
        return out


def feature_space_pgd(real_model, x, y, eps, steps, alpha, random_start=True):
    """PGD using the DETERMINISTIC surrogate for gradients; eval against the
    actual stochastic model. This is the strongest "single-noise-draw"
    adaptive attack: it removes any masking that stochasticity could buy."""
    surrogate = DeterministicSurrogate(real_model)
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        xa = (xa + torch.empty_like(xa).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa_ = xa.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(surrogate(xa_), y)
        g, = torch.autograd.grad(loss, xa_)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def feature_pgd_asr(model, X, Y, batch=N_FEATURE_BATCH):
    model.eval()
    advs = []
    for i in range(0, X.size(0), batch):
        advs.append(feature_space_pgd(model, X[i:i + batch], Y[i:i + batch],
                                      EPS, PGD_STEPS, PGD_ALPHA))
    Xadv = torch.cat(advs, dim=0)
    return _flip_eval_eot(model, X, Y, Xadv)


# ---------------------------------------------------------------------------
# Clean acc (averaged over noise draws)
# ---------------------------------------------------------------------------
@torch.no_grad()
def clean_acc(model, X, Y, n_draws=5, batch=512):
    correct = 0
    total = 0
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        votes = torch.zeros(x.size(0), META["n_classes"], device=x.device)
        for _ in range(n_draws):
            votes += F.softmax(model(x), dim=1)
        correct += int((votes.argmax(1) == y).sum())
        total += x.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h468_adaptive_attack_vonenet_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H468  Adaptive attack on H411 VOneNet (M4 + M12 audit, Fashion-MNIST)")
    out("=" * 80)
    out("Premise: H411 reported PGD 0.962 -> 0.902 with sigma=0.25 (gain 0.060)")
    out("         and an EOT-PGD K=8 audit that 'mostly held'. We re-audit with")
    out("         (a) K in {1,8,20,50} EOT-PGD ASR sweep,")
    out("         (b) analytic Lipschitz of the Gabor front-end, and")
    out("         (c) BPDA feature-space PGD through the deterministic surrogate.")
    out("")
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} "
        f"width={WIDTH} sigma_neuron={SIGMA}")
    out(f"        EOT K sweep = {EOT_KS}; feature-PGD batch = {N_FEATURE_BATCH}")
    out(f"        device = {C.DEVICE}")
    out("")

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    # ---- retrain H411 VOneNet at sigma=0.25 ----
    out(f"[1] training VOneNet (sigma_neuron={SIGMA}) ...")
    C.set_seed(SEED)
    model = VOneNet(META, width=WIDTH, sigma_neuron=SIGMA).to(C.DEVICE)
    model = train_net(model, Xtr, Ytr, SEED)
    acc = clean_acc(model, Xte, Yte)
    out(f"    clean_acc (avg over 5 noise draws) = {acc:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- Lipschitz of the Gabor front-end ----
    out("")
    out("[2] analytic Lipschitz of the Gabor front-end")
    lip = gabor_lipschitz(model.vone, input_shape=(1, 28, 28), device=C.DEVICE)
    out(f"    spectral norm of Gabor conv (power-iter, 40 steps) = "
        f"{lip['sigma_max_gabor']:.4f}")
    out(f"    front-end L2 op-norm bound (sqrt(2) * sigma_max)    = "
        f"{lip['L_op_l2']:.4f}")
    out(f"    front-end Linf-Linf upper bound (L2 * sqrt(dim))    = "
        f"{lip['L_op_linf_upper']:.4f}")
    out(f"    => any input Linf eps={EPS} maps to at most "
        f"|feat-delta|_2 <= {EPS * lip['L_op_l2']:.4f} in feature space.")
    flush_file()

    # ---- ASR-vs-K saturation sweep ----
    out("")
    out("[3] EOT-PGD ASR vs K (saturation sweep)")
    asr_by_k = {}
    for K in EOT_KS:
        out(f"    running EOT-PGD K={K} ...")
        asr = eot_pgd_asr(model, Xte, Yte, K)
        asr_by_k[K] = asr
        out(f"      ASR(K={K}) = {asr:.4f}  ({time.time()-t0:.0f}s)")
        flush_file()

    # ---- BPDA feature-space PGD ----
    out("")
    out("[4] BPDA feature-space PGD (deterministic surrogate for gradients,")
    out("    stochastic real model for evaluation)")
    feat_asr = feature_pgd_asr(model, Xte, Yte)
    out(f"    feature-space PGD ASR = {feat_asr:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- main table ----
    out("")
    out("=" * 80)
    out("[TABLE] attack | ASR")
    out("=" * 80)
    out("{:<40} {:>10}".format("attack", "ASR"))
    out("-" * 52)
    for K in EOT_KS:
        out("{:<40} {:>10.4f}".format(f"EOT-PGD K={K}", asr_by_k[K]))
    out("{:<40} {:>10.4f}".format("BPDA feature-space PGD", feat_asr))
    out("-" * 52)

    # ---- verdict ----
    out("")
    out("=" * 80)
    out("[VERDICT]")
    out("=" * 80)
    h411_pgd = 0.902             # H411 reported PGD ASR at sigma=0.25
    h411_base_pgd = 0.962        # H411 baseline plain SmallCNN PGD ASR
    asr_k1 = asr_by_k[1]
    asr_k8 = asr_by_k[8]
    asr_k20 = asr_by_k[20]
    asr_k50 = asr_by_k[50]
    out(f"  H411 reported: baseline PGD = {h411_base_pgd:.3f}, "
        f"VOneNet sigma=0.25 PGD = {h411_pgd:.3f}  (gain {h411_base_pgd-h411_pgd:+.3f})")
    out(f"  H468 EOT-PGD: K=1 {asr_k1:.4f} | K=8 {asr_k8:.4f} | "
        f"K=20 {asr_k20:.4f} | K=50 {asr_k50:.4f}")
    saturated = abs(asr_k50 - asr_k20) < 0.01
    out(f"  K=20 -> K=50 delta = {asr_k50-asr_k20:+.4f}  "
        f"=> ASR(K) {'SATURATED' if saturated else 'NOT yet saturated'} by K=50.")
    rise_vs_k1 = asr_k50 - asr_k1
    out(f"  EOT rise vs K=1: {rise_vs_k1:+.4f}  "
        f"({'masking suspected' if rise_vs_k1 > 0.05 else 'no significant masking'})")
    bpda_rise = feat_asr - asr_k50
    out(f"  BPDA feature-PGD vs K=50 EOT: {bpda_rise:+.4f}  "
        f"({'BPDA strictly stronger' if bpda_rise > 0.01 else 'BPDA on par with EOT'})")
    out("")
    # one-line verdict
    if asr_k50 >= h411_base_pgd - 0.02 or feat_asr >= h411_base_pgd - 0.02:
        verdict = ("H411 stochastic gain is BROKEN: a sufficiently strong adaptive "
                   "attack (EOT K=50 or BPDA) closes the gap back to the baseline.")
    elif rise_vs_k1 > 0.05 or bpda_rise > 0.05:
        verdict = ("H411 stochastic gain is PARTIALLY masking: ASR rises "
                   f"{rise_vs_k1:+.3f} from K=1 to K=50 (and BPDA {bpda_rise:+.3f}); "
                   "the residual gap is the only honest robustness claim.")
    else:
        verdict = ("H411 stochastic gain is REAL under adaptive attack: ASR is flat "
                   "across K=1..50 and BPDA does not exceed EOT; the small residual "
                   "robustness from sigma=0.25 is not gradient masking.")
    out(f"  ONE-LINE VERDICT: {verdict}")

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
