"""
H226 — Spatial Mixed-Precision Quantisation
Variable bit-depth per image region: more bits where local structure is complex.

For each 28x28 image, compute variance in 4x4 non-overlapping patches (49 patches).
Bit-depth map: patch_var > median_threshold → 8 bits, else → 3 bits.
Train CNN on variable-precision images; test on original 8-bit images.
Compare to baselines: 8-bit, 3-bit, 5-bit.
Also measure: fraction of adversarial perturbation energy in 3-bit vs 8-bit patches.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign import common as C

os.makedirs("results/fashion_mnist", exist_ok=True)

SEED   = 0
N_EVAL = 300
EPS    = 0.1
PATCH  = 4      # patch size
IMG_SZ = 28
C.set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def quantise(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Uniform quantise tensor in [0,1] to `bits` levels."""
    levels = 2 ** bits - 1
    return (x * levels).round() / levels


def patch_variances(X: torch.Tensor) -> torch.Tensor:
    """
    X: (N,1,28,28) in [0,1].
    Returns (N, n_patches) tensor of per-patch variances.
    n_patches = (28//4)^2 = 49.
    """
    N = X.size(0)
    # unfold: (N,1,28,28) -> patches of shape 4x4
    patches = X.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
    # patches: (N, 1, 7, 7, 4, 4)
    patches = patches.contiguous().view(N, -1, PATCH * PATCH)   # (N, 49, 16)
    var = patches.var(dim=2)                                      # (N, 49)
    return var


def mixed_precision_quantise(X: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Apply 8-bit quant where patch_var > threshold, 3-bit elsewhere.
    Returns quantised images of same shape.
    """
    N, C_ch, H, W = X.shape
    n_ph = H // PATCH   # 7
    n_pw = W // PATCH   # 7
    var = patch_variances(X)  # (N, 49)
    high_var = var > threshold  # (N, 49) bool

    out = X.clone()
    for pi in range(n_ph):
        for pj in range(n_pw):
            idx = pi * n_pw + pj
            y0, y1 = pi * PATCH, (pi + 1) * PATCH
            x0, x1 = pj * PATCH, (pj + 1) * PATCH
            patch_slice = out[:, :, y0:y1, x0:x1]   # (N,1,4,4)
            mask8 = high_var[:, idx]                  # (N,)
            mask3 = ~mask8
            if mask8.any():
                patch_slice[mask8] = quantise(patch_slice[mask8], 8)
            if mask3.any():
                patch_slice[mask3] = quantise(patch_slice[mask3], 3)
    return out


def uniform_quantise_dataset(X: torch.Tensor, bits: int) -> torch.Tensor:
    return quantise(X, bits)


@torch.no_grad()
def accuracy(model, X, Y, batch=256):
    corr = 0
    for i in range(0, X.size(0), batch):
        corr += (model(X[i:i+batch]).argmax(1) == Y[i:i+batch]).sum().item()
    return corr / X.size(0)


def asr(model, X, Y, attack, eps=EPS, steps=10, batch=128):
    model.eval()
    flips, corr = [], []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i+batch], Y[i:i+batch]
        with torch.no_grad():
            c = model(x).argmax(1) == y
        if attack == "fgsm":
            xa = C.fgsm(model, x, y, eps)
        else:
            xa = C.pgd(model, x, y, eps, steps)
        with torch.no_grad():
            f = model(xa).argmax(1) != y
        flips.append(f.cpu()); corr.append(c.cpu())
    flips = torch.cat(flips).numpy()
    corr  = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() > 0 else float("nan")


def perturbation_energy_by_precision(model, X, Y, threshold, eps=EPS, batch=64):
    """
    Generate PGD adversarials; measure fraction of ||delta||^2 in 3-bit vs 8-bit patches.
    """
    model.eval()
    N = X.size(0)
    n_ph = IMG_SZ // PATCH
    n_pw = IMG_SZ // PATCH
    energy_3bit = 0.0
    energy_8bit = 0.0

    for i in range(0, N, batch):
        x = X[i:i+batch]
        y = Y[i:i+batch]
        xa = C.pgd(model, x, y, eps=eps, steps=10)
        delta = (xa - x).abs()   # (B,1,28,28)

        var = patch_variances(x)  # (B, 49)
        high_var = var > threshold

        for pi in range(n_ph):
            for pj in range(n_pw):
                idx = pi * n_pw + pj
                y0, y1 = pi * PATCH, (pi + 1) * PATCH
                x0, x1 = pj * PATCH, (pj + 1) * PATCH
                patch_d = delta[:, :, y0:y1, x0:x1].pow(2).sum((1,2,3))  # (B,)
                mask8 = high_var[:, idx]
                energy_8bit += patch_d[mask8].sum().item()
                energy_3bit += patch_d[~mask8].sum().item()

    total = energy_3bit + energy_8bit + 1e-12
    return energy_3bit / total, energy_8bit / total


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
print("Loading Fashion-MNIST …")
Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_eval=N_EVAL, seed=SEED)
Xte = Xte[:N_EVAL].to(device)
Yte = Yte[:N_EVAL].to(device)

meta = {"channels": 1, "size": 28, "n_classes": 10}

# Compute threshold on training set
print("Computing patch variance threshold on training set …")
# Use a CPU sample to be memory-safe (full 60k)
Xtr_cpu = Xtr.cpu()
var_all  = patch_variances(Xtr_cpu)   # (60000, 49)
threshold = float(var_all.median())
print(f"Threshold (median patch var): {threshold:.6f}")

results = {}

# --- Baseline: 8-bit (original) ---
print("\n[Baseline 8-bit] Training …")
model8 = C.build_model("cnn", meta, width=32, seed=SEED)
C.train_model(model8, Xtr, Ytr, epochs=10)
acc8    = accuracy(model8, Xte, Yte)
fgsm8   = asr(model8, Xte, Yte, "fgsm")
pgd8    = asr(model8, Xte, Yte, "pgd")
results["8bit_baseline"] = {"clean_acc": acc8, "fgsm_asr": fgsm8, "pgd_asr": pgd8}
print(f"  clean={acc8:.3f}  fgsm={fgsm8:.3f}  pgd={pgd8:.3f}")

# --- Baseline: 3-bit ---
print("\n[Baseline 3-bit] Quantising training data …")
C.set_seed(SEED)
Xtr3 = uniform_quantise_dataset(Xtr, 3)
model3 = C.build_model("cnn", meta, width=32, seed=SEED)
C.train_model(model3, Xtr3, Ytr, epochs=10)
acc3  = accuracy(model3, Xte, Yte)
fgsm3 = asr(model3, Xte, Yte, "fgsm")
pgd3  = asr(model3, Xte, Yte, "pgd")
results["3bit_uniform"] = {"clean_acc": acc3, "fgsm_asr": fgsm3, "pgd_asr": pgd3}
print(f"  clean={acc3:.3f}  fgsm={fgsm3:.3f}  pgd={pgd3:.3f}")

# --- Baseline: 5-bit ---
print("\n[Baseline 5-bit] Quantising training data …")
C.set_seed(SEED)
Xtr5 = uniform_quantise_dataset(Xtr, 5)
model5 = C.build_model("cnn", meta, width=32, seed=SEED)
C.train_model(model5, Xtr5, Ytr, epochs=10)
acc5  = accuracy(model5, Xte, Yte)
fgsm5 = asr(model5, Xte, Yte, "fgsm")
pgd5  = asr(model5, Xte, Yte, "pgd")
results["5bit_uniform"] = {"clean_acc": acc5, "fgsm_asr": fgsm5, "pgd_asr": pgd5}
print(f"  clean={acc5:.3f}  fgsm={fgsm5:.3f}  pgd={pgd5:.3f}")

# --- Mixed precision ---
print("\n[Mixed precision 3/8-bit] Quantising training data …")
C.set_seed(SEED)
XtrM = mixed_precision_quantise(Xtr.cpu(), threshold).to(device)
modelM = C.build_model("cnn", meta, width=32, seed=SEED)
C.train_model(modelM, XtrM, Ytr, epochs=10)
accM  = accuracy(modelM, Xte, Yte)
fgsmM = asr(modelM, Xte, Yte, "fgsm")
pgdM  = asr(modelM, Xte, Yte, "pgd")
results["mixed_3_8bit"] = {"clean_acc": accM, "fgsm_asr": fgsmM, "pgd_asr": pgdM}
print(f"  clean={accM:.3f}  fgsm={fgsmM:.3f}  pgd={pgdM:.3f}")

# --- Perturbation energy analysis ---
print("\nAnalysing perturbation energy distribution (on 8-bit model, first 100 test samples) …")
e3, e8 = perturbation_energy_by_precision(model8, Xte[:100], Yte[:100], threshold)
print(f"  PGD energy in 3-bit (low-var) patches: {e3:.3f}")
print(f"  PGD energy in 8-bit (high-var) patches: {e8:.3f}")

# --- Summary table ---
print("\n" + "="*70)
print(f"{'Condition':<25} {'Clean Acc':>10} {'FGSM ASR':>10} {'PGD ASR':>10}")
print("-"*70)
for name, m in results.items():
    print(f"{name:<25} {m['clean_acc']:>10.3f} {m['fgsm_asr']:>10.3f} {m['pgd_asr']:>10.3f}")
print("="*70)
print(f"\nPerturbation energy fractions (PGD vs 8-bit model, N=100):")
print(f"  3-bit (low-var) patches : {e3:.3f}")
print(f"  8-bit (high-var) patches: {e8:.3f}")
print(f"  Hypothesis: attacker forced to high-var patches → {'SUPPORTED' if e8 > e3 else 'NOT SUPPORTED'}")
print("\nDone.")
