"""
H227 — JPEG Training Domain
Training on JPEG-compressed images: does DCT quantisation confer robustness?

For jpeg_quality in [10, 30, 50, 70, 90, 100]:
  - Train CNN on JPEG-compressed training images.
  - Test on q=100 (near-original) test images.
Also compare: test-time JPEG only (no training compression).
Key hypothesis: training at q=30-50 forces model to use low-frequency features
→ adversarial perturbations (high-frequency) less effective.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from campaign import common as C

os.makedirs("results/fashion_mnist", exist_ok=True)

SEED   = 0
N_EVAL = 300
EPS    = 0.1
QUALITIES = [10, 30, 50, 70, 90, 100]
C.set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# JPEG helpers (operate on single [0,1] float tensors, grayscale C=1)
# ---------------------------------------------------------------------------

def jpeg_compress(tensor: torch.Tensor, q: int) -> torch.Tensor:
    """
    tensor: (1, H, W) float in [0,1].
    Returns (1, H, W) float in [0,1] after JPEG round-trip.
    """
    arr = (tensor.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    arr2 = np.array(Image.open(buf))
    return torch.tensor(arr2 / 255.0, dtype=torch.float32).unsqueeze(0)


def jpeg_compress_batch(X: torch.Tensor, q: int) -> torch.Tensor:
    """X: (N,1,H,W) float in [0,1]. Returns JPEG-compressed version."""
    out = torch.empty_like(X)
    for i in range(X.size(0)):
        out[i] = jpeg_compress(X[i].cpu(), q)
    return out


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy(model, X, Y, batch=256):
    corr = 0
    for i in range(0, X.size(0), batch):
        corr += (model(X[i:i+batch]).argmax(1) == Y[i:i+batch]).sum().item()
    return corr / X.size(0)


def asr(model, X, Y, attack, eps=EPS, steps=10, batch=64):
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
meta = {"channels": 1, "size": 28, "n_classes": 10}

print("Loading Fashion-MNIST …")
Xtr_orig, Ytr, Xte_orig, Yte = C.load_dataset("fashion_mnist", n_eval=N_EVAL, seed=SEED)
Xte_orig = Xte_orig[:N_EVAL].to(device)
Yte      = Yte[:N_EVAL].to(device)

# Test images at q=100 (effectively lossless JPEG)
print("Preparing q=100 test images (reference) …")
Xte_ref = jpeg_compress_batch(Xte_orig.cpu(), 100).to(device)

results = {}

# ---- 1. Training-compression experiments ----
for q in QUALITIES:
    print(f"\n[Train q={q}] Compressing training set …")
    C.set_seed(SEED)
    Xtr_q = jpeg_compress_batch(Xtr_orig.cpu(), q).to(device)
    model_q = C.build_model("cnn", meta, width=32, seed=SEED)
    C.train_model(model_q, Xtr_q, Ytr, epochs=10)
    acc   = accuracy(model_q, Xte_ref, Yte)
    fgsm_ = asr(model_q, Xte_ref, Yte, "fgsm")
    pgd_  = asr(model_q, Xte_ref, Yte, "pgd")
    results[f"train_q{q}"] = {"mode": "train_compress", "q": q,
                               "clean_acc": acc, "fgsm_asr": fgsm_, "pgd_asr": pgd_}
    print(f"  clean={acc:.3f}  fgsm={fgsm_:.3f}  pgd={pgd_:.3f}")

# ---- 2. Test-time compression only (no training compression) ----
print("\n[No-train-compress baseline] Training on original images …")
C.set_seed(SEED)
model_orig = C.build_model("cnn", meta, width=32, seed=SEED)
C.train_model(model_orig, Xtr_orig, Ytr, epochs=10)

for q in QUALITIES:
    print(f"  [Test-time q={q}]")
    Xte_q = jpeg_compress_batch(Xte_orig.cpu(), q).to(device)
    acc   = accuracy(model_orig, Xte_q, Yte)
    fgsm_ = asr(model_orig, Xte_q, Yte, "fgsm")
    pgd_  = asr(model_orig, Xte_q, Yte, "pgd")
    results[f"test_only_q{q}"] = {"mode": "test_only", "q": q,
                                   "clean_acc": acc, "fgsm_asr": fgsm_, "pgd_asr": pgd_}
    print(f"    clean={acc:.3f}  fgsm={fgsm_:.3f}  pgd={pgd_:.3f}")

# ---- Summary table ----
print("\n" + "="*75)
print(f"{'Condition':<22} {'Mode':<16} {'Quality':>7} {'Clean':>8} {'FGSM ASR':>10} {'PGD ASR':>9}")
print("-"*75)
for name, m in results.items():
    print(f"{name:<22} {m['mode']:<16} {m['q']:>7} {m['clean_acc']:>8.3f} {m['fgsm_asr']:>10.3f} {m['pgd_asr']:>9.3f}")
print("="*75)

# Identify best robustness training quality
train_rows = {k: v for k, v in results.items() if v["mode"] == "train_compress"}
best_pgd_q = min(train_rows, key=lambda k: train_rows[k]["pgd_asr"])
best_fgsm_q = min(train_rows, key=lambda k: train_rows[k]["fgsm_asr"])
print(f"\nBest PGD robustness: {best_pgd_q} (pgd_asr={train_rows[best_pgd_q]['pgd_asr']:.3f})")
print(f"Best FGSM robustness: {best_fgsm_q} (fgsm_asr={train_rows[best_fgsm_q]['fgsm_asr']:.3f})")

q30 = results.get("train_q30", {})
q50 = results.get("train_q50", {})
q100 = results.get("train_q100", {})
if q30 and q50 and q100:
    print(f"\nHypothesis check (q30-50 more robust than q100):")
    print(f"  train_q30  PGD ASR: {q30['pgd_asr']:.3f}")
    print(f"  train_q50  PGD ASR: {q50['pgd_asr']:.3f}")
    print(f"  train_q100 PGD ASR: {q100['pgd_asr']:.3f}")
    low_q_avg = (q30['pgd_asr'] + q50['pgd_asr']) / 2
    supported = low_q_avg < q100['pgd_asr']
    print(f"  → Hypothesis {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

print("\nDone.")
