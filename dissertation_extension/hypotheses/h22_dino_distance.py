"""
Hypothesis H22: Distance in DINOv2 self-supervised feature space to the
per-class centroid predicts adversarial vulnerability on Fashion-MNIST.

Idea:
  DINOv2 (Oquab et al., 2023) provides strong self-supervised image features.
  We hypothesise that the cosine distance from a test sample's DINOv2 embedding
  to the centroid of its OWN class (computed from a subsample of the training
  set) captures "typicality" -- atypical samples sit far from their class
  centroid and should be easier to attack adversarially. We further look at the
  "DINO margin" = (own-class cosine distance) - (nearest-other-class cosine
  distance) which encodes a confidence-like signal in self-sup feature space.

Setup:
  * Train a small CNN victim on Fashion-MNIST (10 epochs, Adam) -- matches
    diagnostic_test.py / h10_clip_distance.py.
  * Load DINOv2 ViT-S/14 via torch.hub:
        torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    Use eval mode. Output features are 384-d.
  * Upsample Fashion-MNIST 28x28 -> 224x224 (bicubic), replicate the single
    channel to RGB, normalise with ImageNet stats.
  * Encode a 5000-sample stratified subsample of the training set and the full
    test set.
  * Per class: centroid = mean of train embeddings (L2-normalised mean).
  * Per test sample compute:
        - dino_own_cos       : cos sim to OWN-class centroid (true label)
        - dino_own_dist      : 1 - dino_own_cos
        - dino_other_cos     : max cos sim across the 9 other-class centroids
        - dino_other_dist    : 1 - dino_other_cos
        - dino_margin        : dino_own_cos - dino_other_cos
                               (positive => sample is closer to own class)
  * Baseline features:
        - victim_margin   : top - 2nd logit on true class
        - mean_pix, std_pix
  * Vulnerability targets:
        - flipped_FGSM       eps = 15/255
        - flipped_PGD        eps = 15/255
        - low_min_eps_FGSM   binary search FGSM, thresholded at median
  * Univariate AUROC and multivariate logistic regression (does dino_margin add
    over victim_margin?).

Run:
    /mnt/1tbone/trashy/adversarial-robustness-toolbox/.venv/bin/python \
        dissertation_extension/hypotheses/h22_dino_distance.py

Caveats:
  * torch.hub.load requires internet on first run (clones the dinov2 repo and
    pulls weights from dl.fbaipublicfiles.com). In offline environments, set
    TORCH_HOME to a cache dir already populated, or this will fail. We trap the
    error and print a clear message; there is no automatic ResNet fallback in
    this script (it would change the hypothesis identity).
  * DINOv2 expects RGB natural images; Fashion-MNIST is 28x28 greyscale.
    Embeddings are upsampled + channel-replicated; absolute cosine values are
    not directly comparable to ImageNet pretraining literature. Within-dataset
    ranking is what matters.
  * Patch size is 14, so 224x224 (=16x16 patches) is a clean input size.
  * Train-set subsample (5000) is stratified to give ~500 per class for stable
    centroids; pass --train-sub to change.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_STEP_SIZE = 2.0 / 255.0
EPOCHS = 10
BATCH = 128
DATA_ROOT = "/tmp/data"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ImageNet normalisation (DINOv2 uses ImageNet stats).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Victim model (matches diagnostic_test.py / h10)
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
        print(f"  epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)", flush=True)
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
            pred = model(x_adv[i:i+batch]).argmax(1)
            out.append(pred != y_true[i:i+batch])
    return torch.cat(out).cpu().numpy().astype(np.int32)


def min_eps_fgsm(model, x, y, max_eps=64/255, n_steps=8):
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
                pred = model(x_adv[i:i+512]).argmax(1)
                flipped[i:i+512] = pred != y[i:i+512]
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi.cpu().numpy()


# ---------------------------------------------------------------------------
# DINOv2 embeddings
# ---------------------------------------------------------------------------
def load_dinov2():
    """Load DINOv2 ViT-S/14 via torch.hub. Raises with a clear message on failure."""
    print("Loading DINOv2 ViT-S/14 via torch.hub ...", flush=True)
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                               trust_repo=True)
    except Exception as exc:
        print("ERROR: failed to load DINOv2 via torch.hub.", file=sys.stderr)
        print("  This usually means torch.hub cannot reach github or",
              "dl.fbaipublicfiles.com.", file=sys.stderr)
        print("  Pre-populate TORCH_HOME or run once online to cache.",
              file=sys.stderr)
        traceback.print_exc()
        raise
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def dino_embed(model, x_28, batch=64):
    """
    x_28 : (N, 1, 28, 28) tensor in [0, 1] on CPU
    Returns: (N, 384) numpy float32, L2-normalised.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=DEVICE).view(1, 3, 1, 1)
    N = x_28.size(0)
    out = np.zeros((N, 384), dtype=np.float32)
    print(f"  embedding {N} images through DINOv2 ...", flush=True)
    t0 = time.time()
    for i in range(0, N, batch):
        chunk = x_28[i:i+batch].to(DEVICE)               # (b, 1, 28, 28)
        chunk = chunk.repeat(1, 3, 1, 1)                 # (b, 3, 28, 28)
        chunk = F.interpolate(chunk, size=(224, 224),
                              mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        chunk = (chunk - mean) / std
        with torch.no_grad():
            emb = model(chunk)                           # (b, 384) CLS token
            emb = F.normalize(emb, dim=-1)
        out[i:i+batch] = emb.cpu().numpy()
        if (i // batch) % 20 == 0:
            print(f"    {i+chunk.size(0)}/{N}  ({time.time()-t0:.1f}s)",
                  flush=True)
    return out


def stratified_subsample(targets_np, per_class, seed=0):
    rng = np.random.RandomState(seed)
    idxs = []
    for c in range(10):
        cidx = np.where(targets_np == c)[0]
        rng.shuffle(cidx)
        idxs.append(cidx[:per_class])
    return np.concatenate(idxs)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=None,
                    help="Use only this many test samples (smoke test).")
    ap.add_argument("--train-sub", type=int, default=5000,
                    help="Total training samples to embed for centroids "
                         "(stratified across 10 classes).")
    ap.add_argument("--out", default="h22_results.json")
    args = ap.parse_args()

    print(f"Device: {DEVICE}", flush=True)
    tfm = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True,  download=True,
                                      transform=tfm)
    test_set  = datasets.FashionMNIST(DATA_ROOT, train=False, download=True,
                                      transform=tfm)

    print("Training victim CNN ...", flush=True)
    victim = train_victim(train_set)

    # Stack tensors
    train_x_all = torch.stack([train_set[i][0] for i in range(len(train_set))])
    train_y_all = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))])

    if args.subset is not None:
        test_x = test_x[:args.subset]
        test_y = test_y[:args.subset]
    N = test_x.size(0)
    print(f"Test set size: {N}", flush=True)

    test_x_dev = test_x.to(DEVICE)
    test_y_dev = test_y.to(DEVICE)

    # --- Baseline victim features ---
    print("Computing victim logit margins ...", flush=True)
    with torch.no_grad():
        logits_all = []
        for i in range(0, N, 512):
            logits_all.append(victim(test_x_dev[i:i+512]))
        logits = torch.cat(logits_all)
    true_logit = logits[torch.arange(N), test_y_dev]
    logits_masked = logits.clone()
    logits_masked[torch.arange(N), test_y_dev] = -1e9
    second_logit = logits_masked.max(1).values
    victim_margin = (true_logit - second_logit).cpu().numpy()

    flat = test_x.view(N, -1).numpy()
    mean_pix = flat.mean(axis=1)
    std_pix  = flat.std(axis=1)

    # --- DINOv2 embeddings ---
    dino = load_dinov2()

    per_class = max(1, args.train_sub // 10)
    sub_idx = stratified_subsample(train_y_all.numpy(), per_class)
    train_x_sub = train_x_all[sub_idx]
    train_y_sub = train_y_all[sub_idx].numpy()
    print(f"Training subsample for centroids: {len(sub_idx)} "
          f"({per_class}/class)", flush=True)

    print("Encoding training subsample ...", flush=True)
    train_emb = dino_embed(dino, train_x_sub)              # (M, 384), L2-norm
    print("Encoding test set ...", flush=True)
    test_emb = dino_embed(dino, test_x)                    # (N, 384), L2-norm

    # Free DINO once embeddings are computed
    del dino
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Per-class centroids (mean of L2-normed embeddings, then re-normalise) ---
    centroids = np.zeros((10, 384), dtype=np.float32)
    for c in range(10):
        m = train_y_sub == c
        if m.sum() == 0:
            print(f"  WARNING: class {c} has zero training samples in subsample.",
                  flush=True)
            continue
        v = train_emb[m].mean(axis=0)
        v /= (np.linalg.norm(v) + 1e-12)
        centroids[c] = v

    # Cosine sim between each test embedding and each centroid: (N, 10)
    cos_all = test_emb @ centroids.T                        # both unit-norm

    test_y_np = test_y.numpy()
    dino_own_cos = cos_all[np.arange(N), test_y_np]
    cos_masked = cos_all.copy()
    cos_masked[np.arange(N), test_y_np] = -np.inf
    dino_other_cos = cos_masked.max(axis=1)
    dino_own_dist   = 1.0 - dino_own_cos
    dino_other_dist = 1.0 - dino_other_cos
    dino_margin    = dino_own_cos - dino_other_cos

    # Nearest-centroid classification accuracy as a sanity check
    nn_pred = cos_all.argmax(axis=1)
    nn_acc = float((nn_pred == test_y_np).mean())
    print(f"DINOv2 nearest-centroid accuracy on Fashion-MNIST: {nn_acc:.3f}",
          flush=True)

    # --- Attacks / targets ---
    print("FGSM eps=15/255 ...", flush=True)
    x_fgsm = fgsm(victim, test_x_dev, test_y_dev, EPS_TEST)
    flipped_fgsm = flipped_mask(victim, x_fgsm, test_y_dev)

    print("PGD eps=15/255 ...", flush=True)
    x_pgd_chunks = []
    for i in range(0, N, 256):
        x_pgd_chunks.append(
            pgd(victim, test_x_dev[i:i+256], test_y_dev[i:i+256], EPS_TEST))
    x_pgd = torch.cat(x_pgd_chunks)
    flipped_pgd = flipped_mask(victim, x_pgd, test_y_dev)

    print("FGSM min-eps binary search ...", flush=True)
    min_eps = min_eps_fgsm(victim, test_x_dev, test_y_dev)

    # --- Feature table ---
    feats = {
        "victim_margin":   victim_margin,
        "mean_pix":        mean_pix,
        "std_pix":         std_pix,
        "dino_own_cos":    dino_own_cos,
        "dino_own_dist":   dino_own_dist,
        "dino_other_cos":  dino_other_cos,
        "dino_other_dist": dino_other_dist,
        "dino_margin":     dino_margin,
    }
    targets_binary = {
        "flipped_FGSM":     flipped_fgsm,
        "flipped_PGD":      flipped_pgd,
        "low_min_eps_FGSM": (min_eps < np.median(min_eps)).astype(np.int32),
    }

    # --- Univariate AUROC (positive class = vulnerable) ---
    # For "closeness-to-own-class" features (cos, margin) higher = more typical
    # = more robust, so AUROC(-feat). For "distance" features higher = more
    # vulnerable already, so AUROC(+feat). To keep one consistent reading we
    # report AUROC(-feat) for every feature (so values > 0.5 always mean
    # "larger feature predicts robustness").
    print("\nUnivariate AUROC(-feat) (positive = vulnerable, "
          ">0.5 means LARGER feature => MORE robust):", flush=True)
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
            print(f"  {tname:>18s}  {fname:>18s}  AUROC(-feat)={a:.3f}")

    # --- Multivariate: does dino_margin add over victim_margin? ---
    print("\nLogistic regression: does dino_margin add over victim_margin?",
          flush=True)
    multivar = {}
    for tname, tvec in targets_binary.items():
        if tvec.sum() == 0 or tvec.sum() == len(tvec):
            continue
        Xb = np.stack([victim_margin], axis=1)
        Xf = np.stack([victim_margin, dino_margin], axis=1)
        Xb_s = StandardScaler().fit_transform(Xb)
        Xf_s = StandardScaler().fit_transform(Xf)
        lr_b = LogisticRegression(max_iter=1000).fit(Xb_s, tvec)
        lr_f = LogisticRegression(max_iter=1000).fit(Xf_s, tvec)
        a_b = roc_auc_score(tvec, lr_b.predict_proba(Xb_s)[:, 1])
        a_f = roc_auc_score(tvec, lr_f.predict_proba(Xf_s)[:, 1])
        coefs = dict(zip(["victim_margin", "dino_margin"], lr_f.coef_[0].tolist()))
        multivar[tname] = {
            "auroc_victim_only": a_b,
            "auroc_victim_plus_dino": a_f,
            "delta": a_f - a_b,
            "stdised_coefs": coefs,
        }
        print(f"  [{tname}] victim-only AUROC={a_b:.3f}  "
              f"+dino_margin AUROC={a_f:.3f} (delta={a_f-a_b:+.3f})  "
              f"coefs={coefs}")

    # --- Save ---
    out = {
        "n_test": int(N),
        "n_train_subsample": int(len(sub_idx)),
        "epochs": EPOCHS,
        "eps_test": EPS_TEST,
        "dino_nearest_centroid_accuracy": nn_acc,
        "univariate_auroc": auroc,
        "multivariate": multivar,
        "min_eps_quartiles": np.quantile(min_eps, [0.25, 0.5, 0.75]).tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
