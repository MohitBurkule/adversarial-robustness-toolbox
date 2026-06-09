"""
H210 - Stochastic resonance test-time ensemble as training-free defense.

Hypothesis: averaging logits from K=8 pixel-shift translations at test time
reduces PGD-10 ASR by 10-20pp on a frozen standard-trained model -- a completely
training-free defense.

Grounded in: arXiv:2510.03224 (stochastic resonance latent ensembles, Oct 2025).

Protocol:
  - Train one standard CNN on Fashion-MNIST (n_train=6000, 15 epochs).
  - Generate PGD-10 adversarial examples (eps=0.1) on 500 test samples.
  - Three inference modes:
    1. No defense: standard forward pass.
    2. Shift ensemble: average logits over K=8 pixel shifts + original.
    3. Noise ensemble: average logits over K=8 Gaussian noise (sigma=0.05) + original.
  - Also test adaptive attack (PGD-10 optimizing against mean logits).
  - Report: clean_acc, oblivious_asr, adaptive_asr for each defense mode.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign.common import (
    DEVICE, set_seed, load_dataset, dataset_meta, build_model,
    train_model, pgd, logits_and_acc,
)

DATASET = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 500
EPOCHS = 15
ADV_EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 2.5 * ADV_EPS / PGD_STEPS
BATCH = 128
K_ENSEMBLE = 8
NOISE_SIGMA = 0.05
SEED = 42


# ---------------------------------------------------------------------------
# Pixel-shift transforms
# ---------------------------------------------------------------------------
def get_shifts():
    """Return list of (dy, dx) shifts excluding (0,0)."""
    shifts = []
    for dy in [-2, -1, 0, 1]:
        for dx in [-2, -1, 0, 1]:
            if (dy, dx) != (0, 0):
                shifts.append((dy, dx))
    # take first K
    return shifts[:K_ENSEMBLE]


def shift_ensemble_logits(model, x):
    """Average logits over original + K pixel-shifted versions."""
    logits = model(x)
    for dy, dx in get_shifts():
        x_shifted = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        logits = logits + model(x_shifted)
    return logits / (K_ENSEMBLE + 1)


def noise_ensemble_logits(model, x):
    """Average logits over original + K Gaussian-noise versions."""
    logits = model(x)
    for _ in range(K_ENSEMBLE):
        x_noisy = (x + torch.randn_like(x) * NOISE_SIGMA).clamp(0, 1)
        logits = logits + model(x_noisy)
    return logits / (K_ENSEMBLE + 1)


# ---------------------------------------------------------------------------
# Ensemble-aware wrappers (for attack_success / adaptive attacks)
# ---------------------------------------------------------------------------
class ShiftEnsembleModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        return shift_ensemble_logits(self.base, x)


class NoiseEnsembleModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        return noise_ensemble_logits(self.base, x)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def eval_asr(model, X, Y, Xadv):
    """Compute ASR given pre-crafted adversarial examples, restricted to originally-correct."""
    model.eval()
    correct_all, flip_all = [], []
    for i in range(0, X.size(0), BATCH):
        x, y, xa = X[i:i + BATCH], Y[i:i + BATCH], Xadv[i:i + BATCH]
        with torch.no_grad():
            correct = (model(x).argmax(1) == y)
            flipped = (model(xa).argmax(1) != y)
        correct_all.append(correct.cpu())
        flip_all.append(flipped.cpu())
    correct_all = torch.cat(correct_all).numpy().astype(bool)
    flip_all = torch.cat(flip_all).numpy()
    asr = float(flip_all[correct_all].mean()) if correct_all.sum() > 0 else float("nan")
    return asr


def adaptive_pgd(model, x, y, eps=ADV_EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """PGD that optimizes against the model's forward (which may be an ensemble)."""
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def generate_adaptive_adv(model, X, Y):
    """Generate adversarial examples with PGD aware of the ensemble."""
    parts = []
    for i in range(0, X.size(0), BATCH):
        x, y = X[i:i + BATCH], Y[i:i + BATCH]
        parts.append(adaptive_pgd(model, x, y))
    return torch.cat(parts)


def main():
    print("=" * 74)
    print("H210 - Stochastic resonance test-time ensemble (training-free defense)")
    print("=" * 74)
    t0 = time.time()
    set_seed(SEED)

    meta = dataset_meta(DATASET)
    Xtr, Ytr, Xte, Yte = load_dataset(DATASET, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    print(f"Dataset: {DATASET}  n_train={Xtr.size(0)}  n_eval={Xte.size(0)}  device={DEVICE}")

    # --- Train standard model ---
    print("\n--- Training standard CNN (15 epochs) ---")
    model = build_model("cnn", meta, width=32)
    train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, opt="adam", lr=1e-3,
                ncls=meta["n_classes"])
    _, clean_base = logits_and_acc(model, Xte, Yte)
    print(f"  Base clean acc: {clean_base:.4f}")

    # --- Build ensemble wrappers ---
    shift_model = ShiftEnsembleModel(model)
    noise_model = NoiseEnsembleModel(model)

    # --- Clean accuracy for each defense ---
    print("\n--- Clean accuracy per defense ---")
    _, clean_shift = logits_and_acc(shift_model, Xte, Yte)
    _, clean_noise = logits_and_acc(noise_model, Xte, Yte)
    print(f"  No defense:      {clean_base:.4f}")
    print(f"  Shift ensemble:  {clean_shift:.4f}")
    print(f"  Noise ensemble:  {clean_noise:.4f}")

    # --- Oblivious attack: PGD-10 crafted on base model ---
    print("\n--- Oblivious PGD-10 (crafted on base model) ---")
    Xadv_obliv = []
    for i in range(0, Xte.size(0), BATCH):
        x, y = Xte[i:i + BATCH], Yte[i:i + BATCH]
        Xadv_obliv.append(pgd(model, x, y, eps=ADV_EPS, steps=PGD_STEPS))
    Xadv_obliv = torch.cat(Xadv_obliv)

    obliv_base = eval_asr(model, Xte, Yte, Xadv_obliv)
    obliv_shift = eval_asr(shift_model, Xte, Yte, Xadv_obliv)
    obliv_noise = eval_asr(noise_model, Xte, Yte, Xadv_obliv)
    print(f"  No defense ASR:      {obliv_base:.4f}")
    print(f"  Shift ensemble ASR:  {obliv_shift:.4f}")
    print(f"  Noise ensemble ASR:  {obliv_noise:.4f}")

    # --- Adaptive attack: PGD-10 crafted against each ensemble ---
    print("\n--- Adaptive PGD-10 (crafted against each ensemble) ---")
    Xadv_shift = generate_adaptive_adv(shift_model, Xte, Yte)
    adapt_shift = eval_asr(shift_model, Xte, Yte, Xadv_shift)
    print(f"  Shift ensemble adaptive ASR: {adapt_shift:.4f}")

    Xadv_noise = generate_adaptive_adv(noise_model, Xte, Yte)
    adapt_noise = eval_asr(noise_model, Xte, Yte, Xadv_noise)
    print(f"  Noise ensemble adaptive ASR: {adapt_noise:.4f}")

    # --- Summary ---
    elapsed = time.time() - t0
    print(f"\n{'=' * 74}")
    print("RESULTS")
    print(f"{'=' * 74}")
    print(f"  {'Defense':<18} {'Clean':>7} {'Obliv ASR':>11} {'Adapt ASR':>11} {'Obliv Reduct':>14}")
    print(f"  {'-'*18} {'-'*7} {'-'*11} {'-'*11} {'-'*14}")

    for name, ca, oa, aa in [
        ("No defense", clean_base, obliv_base, obliv_base),
        ("Shift ensemble", clean_shift, obliv_shift, adapt_shift),
        ("Noise ensemble", clean_noise, obliv_noise, adapt_noise),
    ]:
        reduct = obliv_base - oa
        print(f"  {name:<18} {ca:>7.4f} {oa:>11.4f} {aa:>11.4f} {reduct:>+14.4f}")

    obliv_reduct_shift = obliv_base - obliv_shift
    obliv_reduct_noise = obliv_base - obliv_noise
    print(f"\n  Shift ensemble oblivious ASR reduction: {obliv_reduct_shift:+.4f} ({obliv_reduct_shift*100:+.1f}pp)")
    print(f"  Noise ensemble oblivious ASR reduction: {obliv_reduct_noise:+.4f} ({obliv_reduct_noise*100:+.1f}pp)")
    print(f"  Shift > Noise (oblivious)?  {'YES' if obliv_reduct_shift > obliv_reduct_noise else 'NO'}")

    adapt_collapse_shift = adapt_shift - obliv_shift
    adapt_collapse_noise = adapt_noise - obliv_noise
    print(f"\n  Adaptive collapse (shift): {adapt_collapse_shift:+.4f} ({adapt_collapse_shift*100:+.1f}pp)")
    print(f"  Adaptive collapse (noise): {adapt_collapse_noise:+.4f} ({adapt_collapse_noise*100:+.1f}pp)")

    h_supported = obliv_reduct_shift > 0.10
    print(f"\n  Hypothesis (>10pp oblivious reduction via shift)? {'YES' if h_supported else 'NO'}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
