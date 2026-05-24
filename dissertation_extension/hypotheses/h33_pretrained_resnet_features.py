"""
Hypothesis H33: Distance in ImageNet-pretrained ResNet18 feature space
(cross-domain transfer, model-agnostic to the victim) predicts adversarial
vulnerability on Fashion-MNIST -- even though the pretrained model was
trained on natural RGB photos, not on 28x28 greyscale clothing.

Motivation:
  Pretrained ImageNet feature extractors are essentially "free" -- they
  involve no fine-tuning, no labels on the target domain, and no white-box
  access to the victim. If their feature geometry still captures per-sample
  atypicality in Fashion-MNIST well enough to predict which test points the
  victim will misclassify under FGSM/PGD, that is a strong practical
  signal. H22 (DINOv2) tests an analogous hypothesis with a self-supervised
  ViT; H10 (CLIP) tests an image-text variant. H33 isolates the contribution
  of a *plain supervised ImageNet CNN* (ResNet18).

Setup:
  * Train a small CNN victim on Fashion-MNIST (10 epochs, Adam) -- same
    architecture as ``diagnostic_test.py`` and the other H* scripts.
  * Load ``torchvision.models.resnet18(weights=IMAGENET1K_V1)``, drop the
    final classifier, freeze the rest. The penultimate layer (output of the
    global average pool) is 512-dimensional.
  * For *every* Fashion-MNIST image (train + test):
        - bicubic-upsample 28x28 -> 224x224
        - replicate the single channel to 3
        - apply standard ImageNet normalisation
        - extract the 512-d penultimate feature
  * From the *training* features, compute one 512-d centroid per class
    (mean of L2-normalised features). For each *test* sample:
        - resnet_true_cos   = cos sim to its true-class centroid
        - resnet_nearest_other_cos = max cos sim across the 9 other class centroids
        - resnet_margin     = resnet_true_cos - resnet_nearest_other_cos
  * Baseline features from the victim: victim_margin, mean_pix, std_pix.
  * Vulnerability targets:
        - flipped_FGSM       eps = 15/255
        - flipped_PGD        eps = 15/255 (20 steps, step = 2/255)
        - low_min_eps_FGSM   below-median min FGSM eps (binary search)
  * Univariate AUROC for every (feature, target) pair, scoring -feat so
    that higher AUROC => feature is small/negative on vulnerable samples.
  * Multivariate logistic regression: does ``resnet_margin`` add predictive
    power on top of ``victim_margin``?
  * If ``h22_results.json`` (DINOv2, the closest neighbour hypothesis) is
    found in the cwd or in ``dissertation_extension/``, its multivariate
    deltas are loaded and reported side-by-side for direct comparison.

Run from repo root (do not run inside this commit -- code only):
    /mnt/1tbone/trashy/adversarial-robustness-toolbox/.venv/bin/python \\
        dissertation_extension/hypotheses/h33_pretrained_resnet_features.py

Caveats:
  * ResNet18 was trained on natural RGB photographs at ~224x224. Fashion-MNIST
    is 28x28 greyscale clothing on black backgrounds. Bicubic upsampling
    cannot manufacture detail the original images do not have, so the
    extracted features live in an out-of-distribution region of ResNet18's
    feature manifold. We only need *relative* ordering across samples to be
    meaningful, not absolute fidelity.
  * Centroids are computed from L2-normalised features (cosine geometry on
    the unit sphere). Mean of unit vectors is not itself unit-norm; we
    L2-normalise the centroid as well to keep the cosine semantics clean.
  * "min_eps_FGSM" is binary-searched along the single FGSM gradient sign,
    not the optimum adversarial direction; it lower-bounds the true minimum
    L-inf radius. We dichotomise at the median to keep all targets binary.
  * The H22 side-by-side comparison only happens if h22_results.json exists
    in a known location -- otherwise that block is silently skipped.
  * No fine-tuning of ResNet18, by design: the question is whether the
    *raw* pretrained representation is good enough.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_STEP_SIZE = 2.0 / 255.0
EPOCHS = 10
BATCH = 128

# ImageNet normalisation -- standard torchvision values used to train ResNet18.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------------------
# Victim model (matches diagnostic_test.py)
# ---------------------------------------------------------------------------
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


def train_victim(train_set):
    torch.manual_seed(0)
    np.random.seed(0)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep + 1}/{EPOCHS}  ({time.time() - t0:.1f}s)", flush=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    g = x.grad.sign().detach()
    return (x.detach() + eps * g).clamp(0.0, 1.0)


def pgd(model, x, y, eps, steps=PGD_STEPS, step_size=PGD_STEP_SIZE):
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0.0, 1.0)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        F.cross_entropy(model(xa), y).backward()
        with torch.no_grad():
            xa = xa + step_size * xa.grad.sign()
            xa = torch.max(torch.min(xa, x0 + eps), x0 - eps).clamp(0.0, 1.0)
    return xa.detach()


def flipped_mask(model, x_adv, y_true, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, x_adv.size(0), batch):
            pred = model(x_adv[i:i + batch]).argmax(1)
            out.append(pred != y_true[i:i + batch])
    return torch.cat(out).cpu().numpy().astype(np.int32)


def min_eps_fgsm(model, x, y, max_eps=64 / 255, n_steps=8):
    """Per-sample binary search for the smallest FGSM epsilon that flips."""
    N = x.size(0)
    lo = torch.zeros(N, device=x.device)
    hi = torch.full((N,), max_eps, device=x.device)
    xg = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xg), y).backward()
    sign = xg.grad.sign().detach()
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        x_adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0.0, 1.0)
        with torch.no_grad():
            flipped = torch.zeros(N, dtype=torch.bool, device=x.device)
            for i in range(0, N, 512):
                pred = model(x_adv[i:i + 512]).argmax(1)
                flipped[i:i + 512] = pred != y[i:i + 512]
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi.cpu().numpy()


# ---------------------------------------------------------------------------
# Pretrained ResNet18 feature extractor
# ---------------------------------------------------------------------------
def build_resnet18_extractor():
    """Return (extractor, feature_dim). extractor(x_3xHxW) -> (N, 512)."""
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
    except Exception:
        # Older torchvision API fallback.
        net = models.resnet18(pretrained=True)
    net.fc = nn.Identity()                # penultimate = output of avgpool, 512-d
    net = net.to(DEVICE).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, 512


def extract_features(extractor, x_28, batch=128, tag=""):
    """
    x_28 : (N, 1, 28, 28) float tensor in [0, 1] on CPU.
    Returns (N, 512) numpy array of L2-normalised features.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
    N = x_28.size(0)
    feats = np.zeros((N, 512), dtype=np.float32)
    t0 = time.time()
    for i in range(0, N, batch):
        chunk = x_28[i:i + batch].to(DEVICE)
        chunk = chunk.repeat(1, 3, 1, 1)
        chunk = F.interpolate(chunk, size=(224, 224), mode="bicubic", align_corners=False)
        chunk = chunk.clamp(0.0, 1.0)
        chunk = (chunk - mean) / std
        with torch.no_grad():
            f = extractor(chunk)
            f = F.normalize(f, dim=-1)
        feats[i:i + batch] = f.cpu().numpy()
        if (i // batch) % 20 == 0:
            print(f"  [{tag}] {i + chunk.size(0)}/{N}  ({time.time() - t0:.1f}s)", flush=True)
    return feats


def class_centroids(feats, labels, n_classes=10):
    """Per-class mean of L2-normalised features, then L2-normalise the means."""
    cents = np.zeros((n_classes, feats.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        cents[c] = feats[mask].mean(axis=0)
    norms = np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12
    return cents / norms


# ---------------------------------------------------------------------------
# Optional H22 comparison
# ---------------------------------------------------------------------------
def try_load_h22():
    """Look in a few sensible places for h22_results.json. Returns dict or None."""
    candidates = [
        "h22_results.json",
        os.path.join(os.getcwd(), "h22_results.json"),
        os.path.join(os.path.dirname(__file__), "h22_results.json"),
        os.path.join(os.path.dirname(__file__), "..", "h22_results.json"),
        os.path.join(os.path.dirname(__file__), "..", "results", "h22_results.json"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(path):
            try:
                with open(path, "r") as fh:
                    return path, json.load(fh)
            except Exception as e:
                print(f"  (could not parse {path}: {e})", flush=True)
    return None, None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=None,
                    help="Use only this many test samples (smoke test).")
    ap.add_argument("--data-root", default="/tmp/data",
                    help="FashionMNIST root (default: /tmp/data).")
    ap.add_argument("--out", default="h33_results.json")
    args = ap.parse_args()

    print(f"Device: {DEVICE}", flush=True)
    tfm = transforms.ToTensor()
    train_set = datasets.FashionMNIST(args.data_root, train=True,  download=True, transform=tfm)
    test_set  = datasets.FashionMNIST(args.data_root, train=False, download=True, transform=tfm)

    print("Training victim CNN ...", flush=True)
    victim = train_victim(train_set)

    # Stack train + test into dense tensors.
    print("Stacking datasets ...", flush=True)
    train_x = torch.stack([train_set[i][0] for i in range(len(train_set))])
    train_y = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    test_x  = torch.stack([test_set[i][0]  for i in range(len(test_set))])
    test_y  = torch.tensor([test_set[i][1] for i in range(len(test_set))])
    if args.subset is not None:
        test_x = test_x[:args.subset]
        test_y = test_y[:args.subset]
    N = test_x.size(0)
    print(f"Train size: {train_x.size(0)}  Test size: {N}", flush=True)

    test_x_dev = test_x.to(DEVICE)
    test_y_dev = test_y.to(DEVICE)

    # --- Baseline features from victim ---
    print("Computing victim logit margins ...", flush=True)
    with torch.no_grad():
        logits_all = []
        for i in range(0, N, 512):
            logits_all.append(victim(test_x_dev[i:i + 512]))
        logits = torch.cat(logits_all)
    true_logit = logits[torch.arange(N), test_y_dev]
    logits_masked = logits.clone()
    logits_masked[torch.arange(N), test_y_dev] = -1e9
    second_logit = logits_masked.max(1).values
    victim_margin = (true_logit - second_logit).cpu().numpy()

    flat = test_x.view(N, -1).numpy()
    mean_pix = flat.mean(axis=1)
    std_pix  = flat.std(axis=1)

    # --- Pretrained ResNet18 features ---
    print("Building pretrained ResNet18 ...", flush=True)
    extractor, _ = build_resnet18_extractor()

    print("Extracting ResNet18 features for TRAIN ...", flush=True)
    train_feats = extract_features(extractor, train_x, tag="train")
    print("Extracting ResNet18 features for TEST ...", flush=True)
    test_feats  = extract_features(extractor, test_x,  tag="test")

    # Free the extractor + cuda cache.
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Computing per-class centroids from TRAIN features ...", flush=True)
    centroids = class_centroids(train_feats, train_y.numpy(), n_classes=10)  # (10, 512)

    # Cosine similarities: test_feats and centroids are unit-norm, so dot product = cos.
    cos_all = test_feats @ centroids.T                      # (N, 10)
    resnet_true_cos = cos_all[np.arange(N), test_y.numpy()]
    cos_masked = cos_all.copy()
    cos_masked[np.arange(N), test_y.numpy()] = -np.inf
    resnet_nearest_other_cos = cos_masked.max(axis=1)
    resnet_margin = resnet_true_cos - resnet_nearest_other_cos

    # Sanity: nearest-centroid zero-shot accuracy (does the feature space at
    # least carve up Fashion-MNIST somewhat?)
    nc_pred = cos_all.argmax(axis=1)
    nc_acc = float((nc_pred == test_y.numpy()).mean())
    print(f"Pretrained-ResNet18 nearest-centroid accuracy: {nc_acc:.3f}", flush=True)

    # --- Attacks / vulnerability targets ---
    print("FGSM eps=15/255 ...", flush=True)
    x_fgsm = fgsm(victim, test_x_dev, test_y_dev, EPS_TEST)
    flipped_fgsm = flipped_mask(victim, x_fgsm, test_y_dev)

    print("PGD eps=15/255 ...", flush=True)
    x_pgd_chunks = []
    for i in range(0, N, 256):
        x_pgd_chunks.append(pgd(victim, test_x_dev[i:i + 256], test_y_dev[i:i + 256], EPS_TEST))
    x_pgd = torch.cat(x_pgd_chunks)
    flipped_pgd = flipped_mask(victim, x_pgd, test_y_dev)

    print("FGSM min-eps binary search ...", flush=True)
    min_eps = min_eps_fgsm(victim, test_x_dev, test_y_dev)

    # --- Feature & target tables ---
    feats = {
        "victim_margin":            victim_margin,
        "mean_pix":                 mean_pix,
        "std_pix":                  std_pix,
        "resnet_true_cos":          resnet_true_cos,
        "resnet_nearest_other_cos": resnet_nearest_other_cos,
        "resnet_margin":            resnet_margin,
    }
    targets_binary = {
        "flipped_FGSM":     flipped_fgsm,
        "flipped_PGD":      flipped_pgd,
        "low_min_eps_FGSM": (min_eps < np.median(min_eps)).astype(np.int32),
    }

    # --- Univariate AUROC (vulnerability = positive class) ---
    print("\nUnivariate AUROC (positive = vulnerable; scoring -feat):", flush=True)
    auroc = {}
    for tname, tvec in targets_binary.items():
        auroc[tname] = {}
        if tvec.sum() == 0 or tvec.sum() == len(tvec):
            print(f"  [{tname}] degenerate target ({tvec.sum()}/{len(tvec)} pos)")
            continue
        for fname, fvec in feats.items():
            try:
                a = roc_auc_score(tvec, -fvec)
            except ValueError:
                a = float("nan")
            auroc[tname][fname] = a
            print(f"  {tname:>18s}  {fname:>25s}  AUROC(-feat)={a:.3f}")

    # --- Multivariate: does resnet_margin add over victim_margin? ---
    print("\nLogistic regression: does resnet_margin add over victim_margin?", flush=True)
    multivar = {}
    for tname, tvec in targets_binary.items():
        if tvec.sum() == 0 or tvec.sum() == len(tvec):
            continue
        Xb = np.stack([victim_margin], axis=1)
        Xf = np.stack([victim_margin, resnet_margin], axis=1)
        Xb_s = StandardScaler().fit_transform(Xb)
        Xf_s = StandardScaler().fit_transform(Xf)
        lr_b = LogisticRegression(max_iter=1000).fit(Xb_s, tvec)
        lr_f = LogisticRegression(max_iter=1000).fit(Xf_s, tvec)
        a_b = roc_auc_score(tvec, lr_b.predict_proba(Xb_s)[:, 1])
        a_f = roc_auc_score(tvec, lr_f.predict_proba(Xf_s)[:, 1])
        coefs = dict(zip(["victim_margin", "resnet_margin"], lr_f.coef_[0].tolist()))
        multivar[tname] = {
            "auroc_victim_only":          a_b,
            "auroc_victim_plus_resnet":   a_f,
            "delta":                      a_f - a_b,
            "stdised_coefs":              coefs,
        }
        print(f"  [{tname}] victim-only AUROC={a_b:.3f}  +resnet_margin AUROC={a_f:.3f} "
              f"(delta={a_f - a_b:+.3f})  coefs={coefs}")

    # --- Side-by-side with H22 (DINOv2) if results are around ---
    print("\nLooking for H22 (DINOv2) results for comparison ...", flush=True)
    h22_path, h22 = try_load_h22()
    h22_comparison = None
    if h22 is not None:
        print(f"  found H22 results at {h22_path}", flush=True)
        h22_mv = h22.get("multivariate", {})
        h22_comparison = {}
        for tname in targets_binary.keys():
            row_h33 = multivar.get(tname, {})
            row_h22 = h22_mv.get(tname, {})
            # H22 may name its column differently; try a few likely keys.
            h22_full_auroc = (row_h22.get("auroc_victim_plus_dino")
                              or row_h22.get("auroc_victim_plus_dinov2")
                              or row_h22.get("auroc_full")
                              or None)
            h22_delta = row_h22.get("delta")
            entry = {
                "h33_delta": row_h33.get("delta"),
                "h33_auroc_full": row_h33.get("auroc_victim_plus_resnet"),
                "h22_delta": h22_delta,
                "h22_auroc_full": h22_full_auroc,
            }
            h22_comparison[tname] = entry
            print(f"  [{tname}] H33 delta={entry['h33_delta']}  H22 delta={entry['h22_delta']}")
    else:
        print("  no h22_results.json found in known locations -- skipping comparison.",
              flush=True)

    # --- Save ---
    out = {
        "n_test":                            int(N),
        "epochs":                            EPOCHS,
        "eps_test":                          EPS_TEST,
        "resnet_nearest_centroid_accuracy":  nc_acc,
        "univariate_auroc":                  auroc,
        "multivariate":                      multivar,
        "min_eps_quartiles":                 np.quantile(min_eps, [0.25, 0.5, 0.75]).tolist(),
        "h22_comparison":                    h22_comparison,
        "h22_results_path":                  h22_path,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
