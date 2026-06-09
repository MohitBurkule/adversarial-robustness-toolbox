"""
Hypothesis H46: Input-transformation defences (bit-depth reduction, JPEG
compression, median filtering, total variance minimisation) recover accuracy
on adversarial inputs differently per sample. Different defences benefit
different samples; per-sample features may predict which defence works.

Context:
  - Guo et al. 2018 "Countering Adversarial Images Using Input Transformations"
    (bit-depth, JPEG, total variance minimisation, image quilting).
  - Xu et al. 2017 "Feature Squeezing: Detecting Adversarial Examples in Deep
    Neural Networks" (bit-depth + spatial smoothing).
  - Aggregate recovery rates are well known; per-sample analysis (who benefits
    from which defence) is less studied.

Pipeline:
  1. Train small CNN victim on Fashion-MNIST (10 epochs Adam, architecture
     matching diagnostic_test.py).
  2. Generate FGSM adversarial samples at eps = 15/255 on the test set.
     Restrict downstream analysis to samples that were originally correctly
     classified AND successfully flipped by FGSM (these are the samples for
     which "recovery" is meaningful).
  3. Apply four defences to each adversarial sample:
       - bit_depth: quantise to 3 bits per pixel (8 levels)
       - jpeg_q50: encode JPEG q=50 then re-decode
       - median_3x3: 3x3 median filter
       - tvm_proxy: simple Gaussian smoothing (sigma=1.0) as a
                    cheap proxy for total variance minimisation
  4. For each defence record per-sample recovery flag (defended prediction
     matches the true label).
  5. Per-sample features computed from the original clean image:
       - victim_margin (top1 - top2 logit, clean image)
       - mean_pix
       - std_pix
       - sobel_mean (mean Sobel-gradient magnitude)
  6. Univariate AUROC: does each feature predict, among flipped samples,
     which ones each defence recovers?
  7. Cross-defence overlap: confusion / Jaccard matrix of recovery sets,
     plus marginal-recovery contributions.

Tools: PyTorch + CUDA, PIL, scipy.ndimage. Data root /tmp/data.

Caveats:
  - tvm_proxy is Gaussian smoothing, not the iterative TV minimisation in
    Guo et al.; it is a fast stand-in capturing the "smooth out noise"
    spirit but not the exact operator.
  - Only one attack (FGSM eps=15/255) is studied; behaviour under PGD or
    other attacks may differ.
  - Fashion-MNIST is 28x28 grayscale; JPEG and median filters are quite
    aggressive at this resolution relative to natural images.
  - Recovery is conditioned on "originally correct AND flipped by FGSM",
    so absolute counts depend on the victim's clean accuracy and FGSM
    success rate.
"""

import os
import sys
import time
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter, sobel
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
SEED = 0


class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


# ----------------------------- training ------------------------------------

def train_victim():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tfm = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)

    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS}  {time.time()-t0:.1f}s")
    model.eval()
    return model, test_set


# ----------------------------- attacks -------------------------------------

def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    adv = (x + eps * x.grad.sign()).clamp(0, 1).detach()
    return adv


# ----------------------------- defences ------------------------------------

def defence_bit_depth(arr, bits=3):
    """arr: (N,1,28,28) float in [0,1]. Quantise to `bits` bits per pixel."""
    levels = (1 << bits) - 1  # e.g. 7 for 3 bits -> 8 levels (0..7)
    q = np.round(arr * levels) / levels
    return q.astype(np.float32)


def defence_jpeg(arr, q=50):
    """Encode JPEG q=50 then re-decode. arr: (N,1,28,28) float in [0,1]."""
    N = arr.shape[0]
    out = np.empty_like(arr)
    for i in range(N):
        u8 = (arr[i, 0] * 255.0).round().clip(0, 255).astype(np.uint8)
        im = Image.fromarray(u8, mode="L")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        dec = np.asarray(Image.open(buf).convert("L"), dtype=np.float32) / 255.0
        out[i, 0] = dec
    return out


def defence_median(arr, size=3):
    """3x3 median filter, applied per sample."""
    N = arr.shape[0]
    out = np.empty_like(arr)
    for i in range(N):
        out[i, 0] = median_filter(arr[i, 0], size=size, mode="reflect")
    return out


def defence_tvm_proxy(arr, sigma=1.0):
    """Gaussian smoothing as a cheap proxy for total variance minimisation."""
    N = arr.shape[0]
    out = np.empty_like(arr)
    for i in range(N):
        out[i, 0] = gaussian_filter(arr[i, 0], sigma=sigma, mode="reflect")
    return out.astype(np.float32)


# ----------------------------- features ------------------------------------

def sobel_mean_per_image(arr):
    """arr: (N,1,28,28). Return mean Sobel magnitude per sample."""
    N = arr.shape[0]
    out = np.zeros(N, dtype=np.float32)
    for i in range(N):
        gx = sobel(arr[i, 0], axis=0, mode="reflect")
        gy = sobel(arr[i, 0], axis=1, mode="reflect")
        out[i] = float(np.sqrt(gx * gx + gy * gy).mean())
    return out


# ----------------------------- helpers -------------------------------------

def batched_argmax(model, x_np, bs=512):
    """x_np: (N,1,28,28) float numpy. Returns (N,) int64 numpy predictions."""
    preds = []
    with torch.no_grad():
        for i in range(0, x_np.shape[0], bs):
            xb = torch.from_numpy(x_np[i:i+bs]).to(DEVICE)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def safe_auroc(y, score):
    y = np.asarray(y); score = np.asarray(score)
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, score)


# ----------------------------- main ----------------------------------------

def main():
    print(f"device: {DEVICE}")
    print("training victim ...")
    model, test_set = train_victim()

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)
    print(f"test set: {N}")

    # clean predictions, margins
    with torch.no_grad():
        logits = []
        for i in range(0, N, 512):
            logits.append(model(test_x[i:i+512]))
        logits = torch.cat(logits, 0)
    clean_pred = logits.argmax(1)
    sorted_l, _ = logits.sort(1, descending=True)
    margin = (sorted_l[:, 0] - sorted_l[:, 1]).detach().cpu().numpy()
    correct_mask = (clean_pred == test_y).detach().cpu().numpy().astype(bool)
    print(f"clean accuracy: {correct_mask.mean():.4f}")

    # FGSM at eps=15/255
    print(f"running FGSM (eps={EPS_TEST:.4f}) ...")
    adv_np = np.zeros((N, 1, 28, 28), dtype=np.float32)
    flipped_fgsm = np.zeros(N, dtype=bool)
    for i in range(0, N, 512):
        xb = test_x[i:i+512]; yb = test_y[i:i+512]
        adv = fgsm(model, xb, yb, EPS_TEST)
        with torch.no_grad():
            pred = model(adv).argmax(1)
        flipped_fgsm[i:i+512] = (pred != yb).detach().cpu().numpy()
        adv_np[i:i+512] = adv.detach().cpu().numpy()

    # Eligible samples: clean-correct AND fgsm-flipped.
    elig = correct_mask & flipped_fgsm
    n_elig = int(elig.sum())
    print(f"clean-correct & FGSM-flipped: {n_elig}  "
          f"(of {int(correct_mask.sum())} clean-correct)")
    if n_elig == 0:
        print("no eligible samples; aborting.")
        return

    idx = np.where(elig)[0]
    adv_sub = adv_np[idx]                 # (M,1,28,28)
    y_sub = test_y.detach().cpu().numpy()[idx]
    clean_sub_np = test_x.detach().cpu().numpy()[idx]

    # Per-sample features (computed on the CLEAN image).
    print("computing per-sample features ...")
    margin_sub = margin[idx]
    mean_pix_sub = clean_sub_np.mean(axis=(1, 2, 3))
    std_pix_sub = clean_sub_np.std(axis=(1, 2, 3))
    sobel_mean_sub = sobel_mean_per_image(clean_sub_np)

    # Apply each defence to the adversarial samples and record recovery.
    print("applying defences ...")
    defences = {
        "bit_depth_3":  lambda a: defence_bit_depth(a, bits=3),
        "jpeg_q50":     lambda a: defence_jpeg(a, q=50),
        "median_3x3":   lambda a: defence_median(a, size=3),
        "tvm_proxy":    lambda a: defence_tvm_proxy(a, sigma=1.0),
    }
    recovered = {}
    for dname, dfn in defences.items():
        t0 = time.time()
        defended = dfn(adv_sub)
        preds = batched_argmax(model, defended)
        rec = (preds == y_sub)
        recovered[dname] = rec
        print(f"  {dname:>14s}  recovery={rec.mean():.4f}  "
              f"({rec.sum()}/{len(rec)})  {time.time()-t0:.1f}s")

    # ---------- Univariate AUROC: feature predicts per-defence recovery -----
    print("\n=== Univariate AUROC: features -> defence recovery (on flipped set) ===")
    feats = {
        "victim_margin": margin_sub,
        "mean_pix":      mean_pix_sub,
        "std_pix":       std_pix_sub,
        "sobel_mean":    sobel_mean_sub,
    }
    for dname, rec in recovered.items():
        pos = rec.astype(np.int64)
        print(f"\nDefence {dname}   recovery_rate={pos.mean():.3f}  "
              f"n_pos={int(pos.sum())}  n_neg={int((1-pos).sum())}")
        if len(np.unique(pos)) < 2:
            print("  (degenerate target, skipping AUROC)")
            continue
        for fname, fv in feats.items():
            a_pos = safe_auroc(pos, fv)
            a_neg = safe_auroc(pos, -fv)
            a = max(a_pos, a_neg)
            sign = "+" if a_pos >= a_neg else "-"
            print(f"  {fname:>16s}  AUROC={a:.4f}  (sign={sign})")

    # ---------- Cross-defence recovery overlap ------------------------------
    print("\n=== Cross-defence recovery overlap ===")
    names = list(recovered.keys())
    n_def = len(names)
    M = len(idx)
    mat_count = np.zeros((n_def, n_def), dtype=np.int64)
    mat_jacc = np.zeros((n_def, n_def), dtype=np.float64)
    mat_cond = np.zeros((n_def, n_def), dtype=np.float64)  # P(recovered by j | recovered by i)
    for i_, ni in enumerate(names):
        ri = recovered[ni]
        for j_, nj in enumerate(names):
            rj = recovered[nj]
            inter = int(np.logical_and(ri, rj).sum())
            union = int(np.logical_or(ri, rj).sum())
            mat_count[i_, j_] = inter
            mat_jacc[i_, j_] = (inter / union) if union > 0 else float("nan")
            mat_cond[i_, j_] = (inter / ri.sum()) if ri.sum() > 0 else float("nan")

    def _print_matrix(title, mat, fmt):
        print(f"\n{title}")
        header = "                " + "  ".join(f"{n:>12s}" for n in names)
        print(header)
        for i_, n in enumerate(names):
            row = "  ".join(fmt.format(mat[i_, j_]) for j_ in range(n_def))
            print(f"  {n:>14s}  {row}")

    _print_matrix("Recovered-by-BOTH counts (intersection):", mat_count, "{:>12d}")
    _print_matrix("Jaccard(recovered_i, recovered_j):", mat_jacc, "{:>12.4f}")
    _print_matrix("P(recovered_j | recovered_i):", mat_cond, "{:>12.4f}")

    # Union & marginal contributions (greedy add-one analysis).
    any_rec = np.zeros(M, dtype=bool)
    for n in names:
        any_rec |= recovered[n]
    print(f"\nUnion recovery (any defence works): {any_rec.mean():.4f}  "
          f"({int(any_rec.sum())}/{M})")

    print("\nMarginal recovery added by each defence beyond the others:")
    for n in names:
        others = np.zeros(M, dtype=bool)
        for m in names:
            if m == n:
                continue
            others |= recovered[m]
        only_this = recovered[n] & (~others)
        print(f"  {n:>14s}  recovers_alone={int(only_this.sum())}  "
              f"share_of_union={(only_this.sum()/max(1,any_rec.sum())):.4f}")

    # Samples recovered by exactly k defences.
    stack = np.stack([recovered[n] for n in names], axis=1)  # (M, n_def)
    k_counts = stack.sum(axis=1)
    print("\nDistribution: # defences that recover each sample")
    for k in range(n_def + 1):
        c = int((k_counts == k).sum())
        print(f"  k={k}  count={c}  frac={c/M:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
