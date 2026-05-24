"""
H25: Per-sample consistency under *benign natural augmentations* (small rotation,
translation/crop shift, brightness/contrast jitter) predicts adversarial
vulnerability.  Samples whose prediction wobbles under tiny natural transforms
sit near a decision boundary and should be more attackable.

This is the natural-augmentation analog of H17 (Gaussian-noise consistency,
randomized-smoothing style).  Where H17 perturbs in pixel-noise space, H25
perturbs along the manifold of label-preserving image transforms.

Pipeline:
  1. Train small CNN on Fashion-MNIST (10 epochs, Adam) -- architecture matches
     diagnostic_test.py / h17_noise_consistency.py.
  2. For each test sample apply K=16 random augmentations using
     torchvision.transforms.functional:
        - rotation              ~ U(-10 deg, +10 deg)
        - translation           ~ U(-2 px , +2 px) on each axis
        - brightness  factor    ~ U(0.9, 1.1)
        - contrast    factor    ~ U(0.9, 1.1)
     Horizontal flip is intentionally *skipped* on Fashion-MNIST -- several
     classes (Sandal, Sneaker, Ankle boot, Bag) are not flip-invariant, so a
     flipped image is genuinely a different example and would dominate the
     consistency signal as label noise.  We expose a USE_HFLIP flag (False by
     default) for completeness.
  3. Per sample features:
        - aug_vote_agreement   = fraction of K augmented preds matching clean argmax
        - aug_softmax_l2       = mean L2 distance from augmented softmax to clean
  4. Baseline features: victim_margin, mean_pix, std_pix.
  5. Targets:
        - flipped_FGSM (eps = 15/255)
        - flipped_PGD  (eps = 15/255, 20 steps, alpha = 2/255)
        - FGSM_min_eps_binary_search (continuous; ranked with Spearman)
  6. Univariate AUROC + Spearman.
  7. Multivariate logistic regression: does aug-consistency add over margin?

Refs (no internet calls):
  - Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing",
    ICML 2019 (the noise-consistency analog).
  - Bahat & Shakhnarovich, "Classification confidence estimation with test-time
    data-augmentation", 2018 (TTA consistency as confidence estimator).
  - Engstrom et al., "Exploring the Landscape of Spatial Robustness", ICML 2019
    (rotation/translation robustness).

Run: python h25_augmentation_consistency.py
Data cached under /tmp/data.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0
K_AUG = 16
ROT_DEG = 10.0
TRANS_PX = 2
BRIGHT_DELTA = 0.10
CONTRAST_DELTA = 0.10
USE_HFLIP = False  # Fashion-MNIST classes are not all flip-invariant; default off
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


def train_model(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
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
        print(f"  epoch {ep+1}/{EPOCHS} done in {time.time()-t0:.1f}s")
    model.eval()
    return model


def batched(fn, x, bs=256, **kw):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(fn(x[i:i+bs], **kw))
    return torch.cat(out, 0)


@torch.no_grad()
def softmax_logits(model, x):
    return F.softmax(model(x), dim=1)


@torch.no_grad()
def clean_predictions(model, x):
    p = batched(lambda b: softmax_logits(model, b), x)
    s = p.sort(1, descending=True)[0]
    margin = s[:, 0] - s[:, 1]
    return p, p.argmax(1), margin


def fgsm_grad_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS_TEST):
    s = fgsm_grad_sign(model, x, y)
    return (x + eps * s).clamp(0, 1)


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        g = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * g.sign()
        adv = torch.min(torch.max(adv, x0 - eps), x0 + eps).clamp(0, 1)
    return adv.detach()


def min_eps_fgsm_bsearch(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad_sign(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def _augment_batch(xb, rng):
    """Apply one random natural augmentation independently per-sample in xb.

    xb: [b, 1, 28, 28] in [0, 1].  Returns same shape, in [0, 1].
    rng: numpy.random.Generator for sampling transform params.
    """
    b = xb.size(0)
    out = torch.empty_like(xb)
    # sample params for the whole batch on CPU; transforms.functional accepts
    # per-image scalars, so we loop (b is small, K=16 outer also batched).
    angles = rng.uniform(-ROT_DEG, ROT_DEG, size=b)
    tx = rng.integers(-TRANS_PX, TRANS_PX + 1, size=b)
    ty = rng.integers(-TRANS_PX, TRANS_PX + 1, size=b)
    bright = rng.uniform(1.0 - BRIGHT_DELTA, 1.0 + BRIGHT_DELTA, size=b)
    contrast = rng.uniform(1.0 - CONTRAST_DELTA, 1.0 + CONTRAST_DELTA, size=b)
    flips = rng.integers(0, 2, size=b) if USE_HFLIP else np.zeros(b, dtype=int)

    for i in range(b):
        img = xb[i:i+1]  # [1,1,28,28]
        # affine: rotation + translation, no scale, no shear
        img = TF.affine(
            img,
            angle=float(angles[i]),
            translate=[int(tx[i]), int(ty[i])],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0.0,
        )
        # brightness/contrast (these expect [0,1] tensors and work on grayscale)
        img = TF.adjust_brightness(img, float(bright[i]))
        img = TF.adjust_contrast(img, float(contrast[i]))
        if USE_HFLIP and flips[i] == 1:
            img = TF.hflip(img)
        out[i] = img[0].clamp(0, 1)
    return out


@torch.no_grad()
def aug_consistency(model, x, clean_softmax, clean_argmax, K=K_AUG, bs=64):
    """Return (vote_agreement, softmax_l2_dist) tensors of shape [N]."""
    N = x.size(0)
    agree = torch.zeros(N, device=DEVICE)
    l2 = torch.zeros(N, device=DEVICE)
    rng = np.random.default_rng(SEED + 1)
    for i in range(0, N, bs):
        xb = x[i:i+bs]
        cs = clean_softmax[i:i+bs]
        ca = clean_argmax[i:i+bs]
        b = xb.size(0)
        votes = torch.zeros(b, device=DEVICE)
        l2sum = torch.zeros(b, device=DEVICE)
        for _ in range(K):
            xa = _augment_batch(xb, rng)
            p = F.softmax(model(xa), dim=1)
            votes += (p.argmax(1) == ca).float()
            l2sum += (p - cs).norm(dim=1)
        agree[i:i+b] = votes / K
        l2[i:i+b] = l2sum / K
    return agree, l2


def auroc_safe(y, score):
    """AUROC, taking max(direction, 1-direction)."""
    if y.std() == 0:
        return float("nan")
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def main():
    print("device:", DEVICE)
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("Training victim CNN ...")
    model = train_model(train_set)

    print("Clean predictions ...")
    soft, pred, margin = clean_predictions(model, test_x)

    correct = (pred == test_y)
    print(f"  clean accuracy = {correct.float().mean().item():.4f}")
    x = test_x[correct]
    y = test_y[correct]
    soft_c = soft[correct]
    pred_c = pred[correct]
    margin_c = margin[correct]

    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))

    print(f"Augmentation consistency (K={K_AUG}, rot=+/-{ROT_DEG}deg, "
          f"trans=+/-{TRANS_PX}px, bright=+/-{BRIGHT_DELTA}, "
          f"contrast=+/-{CONTRAST_DELTA}, hflip={USE_HFLIP}) ...")
    t0 = time.time()
    torch.manual_seed(SEED + 7)
    agree, l2 = aug_consistency(model, x, soft_c, pred_c)
    print(f"  done in {time.time()-t0:.1f}s  "
          f"agree mean={agree.mean().item():.3f}  "
          f"l2 mean={l2.mean().item():.3f}")

    # ---- targets ----
    print("FGSM @ eps=15/255 ...")
    flip_fgsm = []
    for i in range(0, x.size(0), 256):
        xb, yb = x[i:i+256], y[i:i+256]
        adv = fgsm_attack(model, xb, yb)
        with torch.no_grad():
            flip_fgsm.append(model(adv).argmax(1) != yb)
    flip_fgsm = torch.cat(flip_fgsm)

    print("PGD ...")
    flip_pgd = []
    for i in range(0, x.size(0), 256):
        xb, yb = x[i:i+256], y[i:i+256]
        adv = pgd_attack(model, xb, yb)
        with torch.no_grad():
            flip_pgd.append(model(adv).argmax(1) != yb)
    flip_pgd = torch.cat(flip_pgd)

    print("FGSM min-eps binary search ...")
    me = []
    for i in range(0, x.size(0), 256):
        me.append(min_eps_fgsm_bsearch(model, x[i:i+256], y[i:i+256]))
    min_eps = torch.cat(me)

    margin_np = margin_c.cpu().numpy()
    mean_pix_np = mean_pix.cpu().numpy()
    std_pix_np = std_pix.cpu().numpy()
    agree_np = agree.cpu().numpy()
    l2_np = l2.cpu().numpy()
    flip_fgsm_np = flip_fgsm.cpu().numpy().astype(int)
    flip_pgd_np = flip_pgd.cpu().numpy().astype(int)
    min_eps_np = min_eps.cpu().numpy()

    print(f"\nN={len(margin_np)}  "
          f"FGSM flip rate={flip_fgsm_np.mean():.3f}  "
          f"PGD flip rate={flip_pgd_np.mean():.3f}  "
          f"mean min_eps={min_eps_np.mean():.4f}")

    targets = {
        "flipped_FGSM": flip_fgsm_np,
        "flipped_PGD": flip_pgd_np,
    }
    feats = {
        "victim_margin": margin_np,
        "mean_pix": mean_pix_np,
        "std_pix": std_pix_np,
        "aug_vote_agreement": agree_np,
        "aug_softmax_l2": l2_np,
    }

    # ---- univariate ----
    print("\n========== UNIVARIATE AUROC ==========")
    for tname, t in targets.items():
        print(f" target: {tname}  (pos rate = {t.mean():.3f})")
        for fname, fv in feats.items():
            a = auroc_safe(t, fv)
            print(f"   {fname:<22} AUROC = {a:.4f}")

    print(" target: FGSM_min_eps  (Spearman)")
    for fname, fv in feats.items():
        rho, _ = spearmanr(fv, min_eps_np)
        print(f"   {fname:<22} Spearman = {rho:+.4f}")

    # ---- multivariate ----
    print("\n========== MULTIVARIATE: aug consistency over margin ==========")
    X_margin = margin_np.reshape(-1, 1)
    X_full = np.stack([margin_np, agree_np, l2_np], axis=1)
    Xs_m = StandardScaler().fit_transform(X_margin)
    Xs_f = StandardScaler().fit_transform(X_full)
    summary = []
    for tname, t in targets.items():
        if t.std() == 0:
            continue
        lr_m = LogisticRegression(max_iter=2000).fit(Xs_m, t)
        lr_f = LogisticRegression(max_iter=2000).fit(Xs_f, t)
        auc_m = roc_auc_score(t, lr_m.predict_proba(Xs_m)[:, 1])
        auc_f = roc_auc_score(t, lr_f.predict_proba(Xs_f)[:, 1])
        print(f" {tname}: margin-only AUROC = {auc_m:.4f}  "
              f"margin+aug AUROC = {auc_f:.4f}  delta = {auc_f-auc_m:+.4f}")
        print(f"   coefs (margin, vote_agree, softmax_l2): "
              f"{lr_f.coef_.flatten().tolist()}")
        summary.append((tname, auc_m, auc_f, auc_f - auc_m))

    print("\n===== SUMMARY (multivariate) =====")
    print(f"{'target':<16} {'margin':>8} {'+aug':>8} {'delta':>8}")
    for t, am, af, d in summary:
        print(f"{t:<16} {am:>8.4f} {af:>8.4f} {d:>+8.4f}")


if __name__ == "__main__":
    main()
