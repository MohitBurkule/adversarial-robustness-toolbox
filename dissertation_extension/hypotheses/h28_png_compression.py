"""
Hypothesis H28: PNG (lossless DEFLATE) byte size is a better
Kolmogorov-complexity proxy than JPEG byte size for predicting adversarial
vulnerability. PNG is lossless, so its byte size reflects raw image
complexity / entropy more directly without conflating image content with the
JPEG quality-knob (quantisation table, chroma subsampling, etc.).

Motivation: in our existing work JPEG byte size at q=75 gave model-free
AUROC ~0.55-0.65 on Fashion-MNIST. We want to test if a stricter,
content-only complexity proxy (lossless compressors) gives a stronger or at
least different signal.

Pipeline:
  1. Train small CNN victim on Fashion-MNIST (10 epochs Adam, same
     architecture as diagnostic_test.py).
  2. For each correctly-classified test sample compute four compressor sizes
     on the raw uint8 pixels:
       - PNG byte size           (PIL save, DEFLATE, lossless)
       - WebP byte size q=100    (PIL save, lossless WebP)
       - bz2 byte size           (bz2.compress on raw pixel bytes)
       - lzma byte size          (lzma.compress on raw pixel bytes,
                                  stricter Kolmogorov proxy)
       - JPEG byte size q=75     (baseline used in earlier hypotheses)
  3. Pixel/model baselines:
       - victim_margin (top1 - top2 logit)
       - mean_pix
       - std_pix
  4. Vulnerability targets (binary):
       - flipped_FGSM    eps = 15/255
       - flipped_PGD     eps = 15/255, 20 steps, alpha = 2/255
       - FGSM_min_eps    binary-searched smallest eps to flip
                         (binarised at <= q25)
  5. Univariate AUROC per feature.
     Multivariate logistic regression for nested sets:
        margin / pix(2) / jpeg(1) / png(1) / webp(1) / bz2(1) / lzma(1) /
        all_compressors(5) / margin+all_compressors
     Asks: does PNG / lzma beat JPEG as a vulnerability predictor?
     Does adding a lossless compressor add signal on top of margin?

Tools: PyTorch + CUDA, PIL, bz2, lzma (stdlib). Data root /tmp/data.

Caveats:
  - Fashion-MNIST is 28x28 single-channel; PNG/WebP overhead may dominate
    informational content. The differences between compressors at this scale
    are likely small.
  - bz2/lzma operate on the raw pixel byte stream and so do not exploit any
    2-D structure, unlike PNG's DEFLATE-after-filter pipeline.
  - Univariate AUROC values are model-free predictors of a target that
    depends on the trained victim, so all numbers couple sample difficulty
    with this particular victim's decision surface.
"""

import os
import sys
import time
import io
import bz2
import lzma

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0
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


def pgd(model, x, y, eps, steps=PGD_STEPS, alpha=PGD_ALPHA):
    x_orig = x.clone().detach()
    adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    adv = adv.clamp(0, 1)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        F.cross_entropy(model(adv), y).backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    return adv.detach()


def batched_predict(model, x, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i+bs]).argmax(1))
    return torch.cat(out)


def fgsm_min_eps(model, x, y, candidates):
    """Smallest eps in candidates that flips prediction (or last+1 if none)."""
    N = x.size(0)
    result = torch.full((N,), len(candidates), dtype=torch.long, device=DEVICE)
    remaining_mask = torch.ones(N, dtype=torch.bool, device=DEVICE)
    for k, eps in enumerate(candidates):
        idx = remaining_mask.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            break
        xs = x[idx]
        ys = y[idx]
        # batched FGSM
        flipped_local = torch.zeros(idx.numel(), dtype=torch.bool, device=DEVICE)
        for i in range(0, idx.numel(), 512):
            xb = xs[i:i+512]; yb = ys[i:i+512]
            adv = fgsm(model, xb, yb, eps)
            with torch.no_grad():
                pred = model(adv).argmax(1)
            flipped_local[i:i+512] = (pred != yb)
        flipped_global_idx = idx[flipped_local]
        result[flipped_global_idx] = k
        remaining_mask[flipped_global_idx] = False
    return result  # index into candidates; len(candidates) means "never flipped"


# --------------------------- compressors -----------------------------------

def png_bytes(arr_u8):
    """arr_u8: (H,W) uint8 numpy array."""
    img = Image.fromarray(arr_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return len(buf.getvalue())


def webp_lossless_bytes(arr_u8):
    img = Image.fromarray(arr_u8, mode="L")
    buf = io.BytesIO()
    # WebP requires RGB/RGBA; convert.
    img.convert("RGB").save(buf, format="WebP", lossless=True, quality=100)
    return len(buf.getvalue())


def jpeg_bytes(arr_u8, q=75):
    img = Image.fromarray(arr_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    return len(buf.getvalue())


def bz2_bytes(arr_u8):
    return len(bz2.compress(arr_u8.tobytes(), compresslevel=9))


def lzma_bytes(arr_u8):
    return len(lzma.compress(arr_u8.tobytes(), preset=9 | lzma.PRESET_EXTREME))


def compute_compressor_features(test_x):
    """test_x: (N,1,28,28) float in [0,1] on DEVICE. Returns dict of np arrays."""
    arr = (test_x.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    arr = arr[:, 0]  # (N,28,28)
    N = arr.shape[0]
    out = {k: np.zeros(N, dtype=np.float32) for k in
           ["png", "webp", "jpeg", "bz2", "lzma"]}
    t0 = time.time()
    for i in range(N):
        im = arr[i]
        out["png"][i] = png_bytes(im)
        out["webp"][i] = webp_lossless_bytes(im)
        out["jpeg"][i] = jpeg_bytes(im, 75)
        out["bz2"][i] = bz2_bytes(im)
        out["lzma"][i] = lzma_bytes(im)
        if (i + 1) % 1000 == 0:
            print(f"  compressed {i+1}/{N}  {time.time()-t0:.1f}s")
    return out


# ----------------------------- analysis ------------------------------------

def safe_auroc(y, score):
    y = np.asarray(y); score = np.asarray(score)
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, score)


def cv_auroc(X, y, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    X = np.asarray(X); y = np.asarray(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    preds = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        preds[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return safe_auroc(y, preds)


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
    correct_mask = (clean_pred == test_y)
    print(f"clean accuracy: {correct_mask.float().mean().item():.4f}")

    # attacks (only meaningful for correctly classified)
    print("running FGSM (eps=15/255) ...")
    flipped_fgsm = np.zeros(N, dtype=np.int64)
    flipped_pgd = np.zeros(N, dtype=np.int64)
    for i in range(0, N, 512):
        xb = test_x[i:i+512]; yb = test_y[i:i+512]
        adv = fgsm(model, xb, yb, EPS_TEST)
        with torch.no_grad():
            pred = model(adv).argmax(1)
        flipped_fgsm[i:i+512] = (pred != yb).detach().cpu().numpy()

    print("running PGD (eps=15/255, 20 steps) ...")
    for i in range(0, N, 512):
        xb = test_x[i:i+512]; yb = test_y[i:i+512]
        adv = pgd(model, xb, yb, EPS_TEST)
        with torch.no_grad():
            pred = model(adv).argmax(1)
        flipped_pgd[i:i+512] = (pred != yb).detach().cpu().numpy()

    print("running FGSM min-eps sweep ...")
    eps_grid = [1/255, 2/255, 4/255, 6/255, 8/255, 10/255, 12/255, 15/255,
                20/255, 25/255, 32/255, 48/255, 64/255]
    min_eps_idx = fgsm_min_eps(model, test_x, test_y, eps_grid).cpu().numpy()
    # cap value -> grid length
    min_eps_val = np.array([eps_grid[i] if i < len(eps_grid) else eps_grid[-1] + 1/255
                            for i in min_eps_idx])

    # pixel stats
    mean_pix = test_x.mean(dim=(1, 2, 3)).detach().cpu().numpy()
    std_pix = test_x.std(dim=(1, 2, 3)).detach().cpu().numpy()

    # compressors
    print("computing compressor sizes ...")
    comp = compute_compressor_features(test_x)

    # restrict to correctly-classified samples for vulnerability framing
    keep = correct_mask.detach().cpu().numpy().astype(bool)
    print(f"correctly classified kept: {keep.sum()}")

    margin_k = margin[keep]
    mean_pix_k = mean_pix[keep]
    std_pix_k = std_pix[keep]
    comp_k = {k: v[keep] for k, v in comp.items()}
    flipped_fgsm_k = flipped_fgsm[keep]
    flipped_pgd_k = flipped_pgd[keep]
    min_eps_val_k = min_eps_val[keep]

    # FGSM_min_eps as a binary target: vulnerable == min_eps <= 25th percentile
    q25 = np.quantile(min_eps_val_k, 0.25)
    fgsm_low_eps = (min_eps_val_k <= q25).astype(np.int64)

    targets = {
        "flipped_FGSM": flipped_fgsm_k,
        "flipped_PGD": flipped_pgd_k,
        "FGSM_min_eps_low(<=q25)": fgsm_low_eps,
    }

    feats_uni = {
        "victim_margin": margin_k,
        "mean_pix": mean_pix_k,
        "std_pix": std_pix_k,
        "jpeg_q75": comp_k["jpeg"],
        "png": comp_k["png"],
        "webp_lossless": comp_k["webp"],
        "bz2": comp_k["bz2"],
        "lzma": comp_k["lzma"],
    }

    print("\n=== Univariate AUROC ===")
    for tname, ty in targets.items():
        base = ty.mean()
        print(f"\nTarget {tname}   positive_rate={base:.3f}")
        for fname, fv in feats_uni.items():
            # margin is inversely related to vulnerability -> try both
            a_pos = safe_auroc(ty, fv)
            a_neg = safe_auroc(ty, -fv)
            a = max(a_pos, a_neg)
            sign = "+" if a_pos >= a_neg else "-"
            print(f"  {fname:>22s}  AUROC={a:.4f}  (sign={sign})")

    print("\n=== Multivariate logistic-regression CV AUROC ===")
    feature_sets = {
        "margin":              ["victim_margin"],
        "pix(2)":              ["mean_pix", "std_pix"],
        "jpeg(1)":             ["jpeg_q75"],
        "png(1)":              ["png"],
        "webp(1)":             ["webp_lossless"],
        "bz2(1)":              ["bz2"],
        "lzma(1)":             ["lzma"],
        "all_compressors(5)":  ["jpeg_q75", "png", "webp_lossless", "bz2", "lzma"],
        "margin+all_comp(6)":  ["victim_margin", "jpeg_q75", "png",
                                "webp_lossless", "bz2", "lzma"],
        "margin+png(2)":       ["victim_margin", "png"],
        "margin+lzma(2)":      ["victim_margin", "lzma"],
        "margin+jpeg(2)":      ["victim_margin", "jpeg_q75"],
    }
    for tname, ty in targets.items():
        print(f"\nTarget {tname}")
        for set_name, cols in feature_sets.items():
            X = np.stack([feats_uni[c] for c in cols], axis=1)
            a = cv_auroc(X, ty, n_splits=5)
            print(f"  {set_name:>22s}  CV-AUROC={a:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
