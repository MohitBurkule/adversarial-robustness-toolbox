"""
Hypothesis H27: Per-sample occlusion sensitivity predicts adversarial vulnerability.

The maximum (or mean) drop in true-class softmax probability when sliding a small
grey patch over the image is hypothesised to correlate with adversarial
vulnerability.  Occlusion sensitivity is the classic interpretability technique of
Zeiler & Fergus (2014, "Visualizing and Understanding Convolutional Networks").
Samples whose prediction depends on a small, concentrated image region (high
spatial saliency) are plausibly more attackable, because an adversary only needs
to perturb a few critical pixels to flip the decision.

Per-sample features (5x5 grid of patch positions, 6x6 patch filled with the
image's own mean pixel value):
    - occ_max_drop : max over positions of (p_clean - p_occluded) for true class
    - occ_mean_drop: mean over positions of (p_clean - p_occluded) for true class
    - occ_std_drop : std  over positions
    - occ_max_drop_other: max DROP among the 25 positions taking the absolute,
                           captures sensitivity to occlusion regardless of sign

Baselines (forced to match the rest of the hypothesis suite):
    - victim_margin (eval-mode top1 - top2 logit gap)
    - mean_pix
    - std_pix
    - sobel_mean (mean Sobel-gradient magnitude, simple edge-energy proxy)

Targets (defined ONLY on samples the victim classifies correctly):
    - flipped_FGSM   binary,  L_inf eps = 15/255
    - flipped_PGD    binary,  20 steps, alpha = 2/255, eps = 15/255
    - FGSM_min_eps   continuous, per-sample L_inf binary search

Evaluation:
    1. Univariate AUROC (binary targets) and Spearman (continuous).
    2. Multivariate logistic regression:
         margin-only        vs.  margin + occlusion block
         all-features       vs.  all-features minus occlusion block
       Reports delta AUROC: does occlusion sensitivity ADD over the margin
       baseline (which is essentially what causes a sample to flip in the first
       place)?

Self-contained: trains the victim CNN from scratch (Adam, 10 epochs).  Downloads
Fashion-MNIST to /tmp/data.  Writes nothing else to disk.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0
SEED = 0

# Occlusion params
PATCH_SIZE = 6        # 6x6 grey patch
GRID = 5              # 5x5 grid of positions  --> 25 occlusions per sample
IMG_SIZE = 28


# ---------------------------------------------------------------------------
class CNN(nn.Module):
    """Matches diagnostic_test.py: two conv layers, two FC, dropout 0.25 / 0.5."""
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


# ---------------------------------------------------------------------------
def train_victim(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done")
    model.eval()
    return model


# ---------------------------------------------------------------------------
@torch.no_grad()
def batched_logits(model, x, bs=512):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(model(x[i:i+bs]))
    return torch.cat(out, 0)


def fgsm_attack(model, x, y, eps=EPS_TEST):
    was_training = model.training
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_adv), y).backward()
    sign = x_adv.grad.sign().detach()
    if was_training:
        model.train()
    return (x + eps * sign).clamp(0, 1).detach()


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    was_training = model.training
    model.eval()
    x0 = x.clone().detach()
    delta = torch.empty_like(x0).uniform_(-eps, eps)
    x_adv = (x0 + delta).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)
    if was_training:
        model.train()
    return x_adv.detach()


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for the smallest L_inf eps that flips FGSM."""
    was_training = model.training
    model.eval()
    x_grad = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_grad), y).backward()
    sign = x_grad.grad.sign().detach()
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    if was_training:
        model.train()
    return hi


def batched_attack_flag(model, x, y, attack_fn, bs=256):
    flags = []
    model.eval()
    for i in range(0, x.size(0), bs):
        adv = attack_fn(model, x[i:i+bs], y[i:i+bs])
        with torch.no_grad():
            flags.append(model(adv).argmax(1) != y[i:i+bs])
    return torch.cat(flags, 0)


# ---------------------------------------------------------------------------
def patch_positions():
    """Return list of (top, left) coords for a GRID x GRID grid of patches of
    size PATCH_SIZE inside an IMG_SIZE x IMG_SIZE image, evenly spaced."""
    max_top = IMG_SIZE - PATCH_SIZE
    if GRID == 1:
        coords = [max_top // 2]
    else:
        coords = np.linspace(0, max_top, GRID).round().astype(int).tolist()
    pos = []
    for t in coords:
        for l in coords:
            pos.append((int(t), int(l)))
    return pos


@torch.no_grad()
def occlusion_features(model, x, y, bs=128):
    """For each test sample compute true-class softmax under 25 occlusions.

    Each occluded image is a clone with a PATCH_SIZE x PATCH_SIZE square set to
    the sample's own mean pixel value (i.e. a uniform grey patch).  Returns dict
    of (N,) tensors: occ_max_drop, occ_mean_drop, occ_std_drop, occ_max_abs_drop.
    """
    model.eval()
    N = x.size(0)
    positions = patch_positions()
    P = len(positions)

    # clean true-class probabilities
    clean_logits = batched_logits(model, x)
    clean_sm = F.softmax(clean_logits, dim=1)
    p_clean = clean_sm.gather(1, y.view(-1, 1)).squeeze(1)        # (N,)

    # per-sample mean pix (used as grey fill, broadcast to patch)
    mean_pix = x.mean(dim=(1, 2, 3))                              # (N,)

    drops = torch.zeros(N, P, device=DEVICE)                      # (N, P)

    for i in range(0, N, bs):
        xb = x[i:i+bs]
        yb = y[i:i+bs]
        mb = mean_pix[i:i+bs]
        Bsz = xb.size(0)
        # build P occluded versions, evaluate.
        for j, (t, l) in enumerate(positions):
            occ = xb.clone()
            # fill patch with mean pixel (broadcast across H,W of the patch)
            occ[:, :, t:t+PATCH_SIZE, l:l+PATCH_SIZE] = mb.view(Bsz, 1, 1, 1)
            logits = model(occ)
            sm = F.softmax(logits, dim=1)
            p_occ = sm.gather(1, yb.view(-1, 1)).squeeze(1)       # (B,)
            drops[i:i+Bsz, j] = p_clean[i:i+Bsz] - p_occ          # +ve = occlusion hurt true class

    occ_max_drop = drops.max(dim=1).values
    occ_mean_drop = drops.mean(dim=1)
    occ_std_drop = drops.std(dim=1)
    occ_max_abs_drop = drops.abs().max(dim=1).values

    return {
        "occ_max_drop": occ_max_drop,
        "occ_mean_drop": occ_mean_drop,
        "occ_std_drop": occ_std_drop,
        "occ_max_abs_drop": occ_max_abs_drop,
    }


# ---------------------------------------------------------------------------
def victim_margin(model, x):
    model.eval()
    with torch.no_grad():
        logits = batched_logits(model, x)
    s, _ = logits.sort(1, descending=True)
    return (s[:, 0] - s[:, 1])


def sobel_mean(x):
    """Mean magnitude of Sobel gradients per image. x: (N,1,28,28)."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    with torch.no_grad():
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        mag = (gx * gx + gy * gy).sqrt()
    return mag.mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
def auroc_both_dirs(y, score):
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    tf = transforms.ToTensor()
    print("Loading Fashion-MNIST ...")
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print(f"Training victim CNN ({EPOCHS} epochs) on {DEVICE} ...")
    model = train_victim(train_set)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # restrict to correctly classified
    model.eval()
    with torch.no_grad():
        pred = batched_logits(model, test_x).argmax(1)
    correct = pred == test_y
    print(f"Test acc (eval mode): {correct.float().mean().item():.4f}")
    x = test_x[correct]
    y = test_y[correct]
    N = x.size(0)
    print(f"Using {N} correctly-classified samples.")

    # --- features -------------------------------------------------------
    print("Computing victim_margin (eval mode) ...")
    margin = victim_margin(model, x)

    print(f"Computing occlusion-sensitivity features  "
          f"(patch={PATCH_SIZE}x{PATCH_SIZE}, {GRID}x{GRID}={GRID*GRID} positions) ...")
    occ = occlusion_features(model, x, y)

    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))
    sob = sobel_mean(x)

    feat_tensors = [
        margin,
        occ["occ_max_drop"],
        occ["occ_mean_drop"],
        occ["occ_std_drop"],
        occ["occ_max_abs_drop"],
        mean_pix,
        std_pix,
        sob,
    ]
    names = [
        "victim_margin",
        "occ_max_drop",
        "occ_mean_drop",
        "occ_std_drop",
        "occ_max_abs_drop",
        "mean_pix",
        "std_pix",
        "sobel_mean",
    ]
    occ_block_names = [
        "occ_max_drop",
        "occ_mean_drop",
        "occ_std_drop",
        "occ_max_abs_drop",
    ]
    feats = torch.stack(feat_tensors, 1).detach().cpu().numpy()

    # --- targets --------------------------------------------------------
    print("Running FGSM attack ...")
    flip_fgsm = batched_attack_flag(model, x, y, fgsm_attack).cpu().numpy().astype(int)

    print("Running PGD attack ...")
    flip_pgd = batched_attack_flag(model, x, y, pgd_attack).cpu().numpy().astype(int)

    print("Running min-eps FGSM binary search ...")
    me = []
    for i in range(0, N, 256):
        me.append(min_eps_fgsm(model, x[i:i+256], y[i:i+256]))
    min_eps = torch.cat(me, 0).cpu().numpy()

    # --- evaluation -----------------------------------------------------
    print("\n========== H27: Occlusion sensitivity vs adversarial vulnerability ==========")
    print(f"N = {N}  FGSM-flip rate = {flip_fgsm.mean():.3f}  "
          f"PGD-flip rate = {flip_pgd.mean():.3f}")
    print(f"min_eps FGSM:  mean={min_eps.mean():.4f}  median={np.median(min_eps):.4f}")

    bin_targets = [("flipped_FGSM", flip_fgsm), ("flipped_PGD", flip_pgd)]

    Xs = StandardScaler().fit_transform(feats)
    margin_idx = names.index("victim_margin")
    occ_idxs = [names.index(n) for n in occ_block_names]
    non_occ_idxs = [i for i in range(len(names)) if i not in occ_idxs]

    for t_name, yv in bin_targets:
        if yv.std() == 0:
            print(f"\n--- target {t_name}: no variance, skipping ---")
            continue
        print(f"\n--- target: {t_name}  (positive rate = {yv.mean():.3f}) ---")

        per_feat_auc = {}
        for i, n in enumerate(names):
            a = auroc_both_dirs(yv, feats[:, i])
            per_feat_auc[n] = a
            print(f"   univariate AUROC  {n:<22} {a:.4f}")

        best_occ = max(occ_block_names, key=lambda n: per_feat_auc[n])
        print(f"   strongest single occlusion feature: {best_occ}  "
              f"(AUROC = {per_feat_auc[best_occ]:.4f})")

        lr_full = LogisticRegression(max_iter=4000).fit(Xs, yv)
        auc_full = roc_auc_score(yv, lr_full.predict_proba(Xs)[:, 1])

        Xs_m = Xs[:, [margin_idx]]
        lr_m = LogisticRegression(max_iter=4000).fit(Xs_m, yv)
        auc_m = roc_auc_score(yv, lr_m.predict_proba(Xs_m)[:, 1])

        Xs_m_occ = Xs[:, [margin_idx] + occ_idxs]
        lr_m_occ = LogisticRegression(max_iter=4000).fit(Xs_m_occ, yv)
        auc_m_occ = roc_auc_score(yv, lr_m_occ.predict_proba(Xs_m_occ)[:, 1])

        Xs_no_occ = Xs[:, non_occ_idxs]
        lr_no_occ = LogisticRegression(max_iter=4000).fit(Xs_no_occ, yv)
        auc_no_occ = roc_auc_score(yv, lr_no_occ.predict_proba(Xs_no_occ)[:, 1])

        print(f"   multivariate AUROC (margin only):              {auc_m:.4f}")
        print(f"   multivariate AUROC (margin + occlusion block): {auc_m_occ:.4f}")
        print(f"     Delta over margin-alone:                       {auc_m_occ - auc_m:+.4f}")
        print(f"   multivariate AUROC (all features):             {auc_full:.4f}")
        print(f"   multivariate AUROC (all minus occlusion):      {auc_no_occ:.4f}")
        print(f"     Delta from removing occlusion block:           {auc_full - auc_no_occ:+.4f}")
        print("   standardised coefficients (full model):")
        for n, c in zip(names, lr_full.coef_.flatten()):
            print(f"     {n:<22} {c:+.4f}")

    # --- continuous target ----------------------------------------------
    print(f"\n--- continuous target: FGSM_min_eps (lower = more vulnerable) ---")
    from scipy.stats import spearmanr
    for i, n in enumerate(names):
        r, p = spearmanr(feats[:, i], min_eps)
        print(f"   spearman  {n:<22} rho={r:+.4f}  p={p:.2e}")

    ols_full = LinearRegression().fit(Xs, min_eps)
    r2_full = ols_full.score(Xs, min_eps)
    ols_m = LinearRegression().fit(Xs[:, [margin_idx]], min_eps)
    r2_m = ols_m.score(Xs[:, [margin_idx]], min_eps)
    ols_m_occ = LinearRegression().fit(Xs[:, [margin_idx] + occ_idxs], min_eps)
    r2_m_occ = ols_m_occ.score(Xs[:, [margin_idx] + occ_idxs], min_eps)
    ols_no_occ = LinearRegression().fit(Xs[:, non_occ_idxs], min_eps)
    r2_no_occ = ols_no_occ.score(Xs[:, non_occ_idxs], min_eps)
    print(f"   OLS R^2 (margin only):                 {r2_m:.4f}")
    print(f"   OLS R^2 (margin + occlusion block):    {r2_m_occ:.4f}")
    print(f"     Delta:                                 {r2_m_occ - r2_m:+.4f}")
    print(f"   OLS R^2 (all features):                {r2_full:.4f}")
    print(f"   OLS R^2 (all minus occlusion):         {r2_no_occ:.4f}")
    print(f"     Delta:                                 {r2_full - r2_no_occ:+.4f}")
    print("   OLS coefficients (full model):")
    for n, c in zip(names, ols_full.coef_.flatten()):
        print(f"     {n:<22} {c:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
