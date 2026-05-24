"""
H63: Color-shift adversarial vulnerability.

Hypothesis:
  Small uniform brightness/contrast offsets — a "color-shift" adversarial
  perturbation that simply re-scales pixel intensities globally — can flip
  predictions on a non-trivial subset of samples. We measure per-sample
  vulnerability to this family and ask whether margin / mean_pix / std_pix
  features predict it, and whether it correlates with FGSM L_inf flipping.

Pipeline:
  1. Train a small CNN (matching diagnostic_test.CNN) on Fashion-MNIST for
     10 epochs.
  2. For each correctly-classified test sample, grid-search a global
     brightness offset b in [-0.2, 0.2] and contrast multiplier c in
     [0.7, 1.3] applied as  x' = clamp(c * x + b, 0, 1).
        - color_shift_min_magnitude: smallest "magnitude" of (b, c) that
          flips the prediction, where magnitude := sqrt(b^2 + (c-1)^2).
        - flipped_at_default: 1 iff ANY grid point in the search region
          flips the prediction (vulnerable at all).
  3. Per-sample features:  final_margin, mean_pix, std_pix.
  4. Univariate AUROC of each feature for predicting flipped_at_default,
     and Pearson correlation with color_shift_min_magnitude.
  5. Cross with FGSM: compute min-eps-FGSM and report Pearson/Spearman
     correlation between color-shift min magnitude and FGSM min eps,
     plus contingency of flipped_at_default vs FGSM-self-flip at eps=15/255.

This file is self-contained — it duplicates the small CNN and FGSM utilities
from diagnostic_test.py so it can run independently.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0

# color-shift grid
BRIGHT_LO, BRIGHT_HI, BRIGHT_N = -0.2, 0.2, 21   # step 0.02
CONTRAST_LO, CONTRAST_HI, CONTRAST_N = 0.7, 1.3, 13  # step 0.05


class CNN(nn.Module):
    """Matches diagnostic_test.CNN exactly."""
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2); x = self.do1(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.do2(x); return self.fc2(x)


def train_model(seed, train_set):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def batched_predict(model, x, bs=512):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(model(x[i:i + bs]).argmax(1))
    return torch.cat(out)


@torch.no_grad()
def batched_logits(model, x, bs=512):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(model(x[i:i + bs]))
    return torch.cat(out, 0)


def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def fgsm_flip_at(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def color_shift_search(model, x, y, bs=256):
    """For each sample, grid-search (brightness, contrast) and return
    - flipped_at_default : bool, True iff ANY (b, c) in grid flips it
    - min_magnitude      : float, min sqrt(b^2 + (c-1)^2) over flipping
                           grid points; +inf if none flip.
    """
    N = x.size(0)
    brights = torch.linspace(BRIGHT_LO, BRIGHT_HI, BRIGHT_N, device=DEVICE)
    contrasts = torch.linspace(CONTRAST_LO, CONTRAST_HI, CONTRAST_N, device=DEVICE)

    # precompute magnitudes for each (b, c)
    bg, cg = torch.meshgrid(brights, contrasts, indexing="ij")
    mags = torch.sqrt(bg ** 2 + (cg - 1.0) ** 2)  # [BN, CN]
    flat_b = bg.reshape(-1)        # [G]
    flat_c = cg.reshape(-1)        # [G]
    flat_m = mags.reshape(-1)      # [G]
    G = flat_b.numel()

    flipped_any = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    min_mag = torch.full((N,), float("inf"), device=DEVICE)

    # iterate over grid points (outer); batch over samples (inner)
    for g in range(G):
        b = float(flat_b[g]); c = float(flat_c[g]); m = float(flat_m[g])
        # skip the identity (b=0, c=1) — it cannot flip a correct sample
        if b == 0.0 and c == 1.0:
            continue
        for i in range(0, N, bs):
            xb = x[i:i + bs]
            yb = y[i:i + bs]
            adv = (c * xb + b).clamp(0.0, 1.0)
            with torch.no_grad():
                pred = model(adv).argmax(1)
            flip = pred != yb
            sl = slice(i, i + bs)
            flipped_any[sl] = flipped_any[sl] | flip
            # update min magnitude where this grid point flipped
            better = flip & (m < min_mag[sl])
            cur = min_mag[sl].clone()
            cur[better] = m
            min_mag[sl] = cur
    return flipped_any, min_mag


def main():
    tf = transforms.ToTensor()
    print("loading Fashion-MNIST...")
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(f"training CNN ({EPOCHS} epochs) on {DEVICE}...")
    t0 = time.time()
    model = train_model(0, train_set)
    print(f"  done ({time.time() - t0:.1f}s)")

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # restrict to samples model predicts correctly
    pred = batched_predict(model, test_x)
    correct = pred == test_y
    x_c = test_x[correct]; y_c = test_y[correct]
    N = x_c.size(0)
    print(f"using {N} correctly-classified samples (acc={correct.float().mean():.4f})")

    # ------------- per-sample features -------------
    logits = batched_logits(model, x_c)
    sorted_logits, _ = logits.sort(1, descending=True)
    final_margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).detach().cpu().numpy()
    mean_pix = x_c.view(N, -1).mean(1).detach().cpu().numpy()
    std_pix = x_c.view(N, -1).std(1).detach().cpu().numpy()

    # ------------- color-shift targets -------------
    print("running color-shift grid search...")
    t0 = time.time()
    flipped_any, min_mag = color_shift_search(model, x_c, y_c)
    print(f"  done ({time.time() - t0:.1f}s)")
    flipped_at_default = flipped_any.detach().cpu().numpy().astype(int)
    min_mag_np = min_mag.detach().cpu().numpy()
    # for samples that never flip, set magnitude to a sentinel just above the
    # maximum reachable magnitude in the grid for correlation purposes
    max_reach = float(np.sqrt(max(abs(BRIGHT_LO), BRIGHT_HI) ** 2
                              + max(1 - CONTRAST_LO, CONTRAST_HI - 1) ** 2))
    min_mag_for_corr = np.where(np.isinf(min_mag_np), max_reach * 1.5, min_mag_np)

    print(f"  flipped_at_default rate = {flipped_at_default.mean():.4f}")
    finite = np.isfinite(min_mag_np)
    if finite.any():
        print(f"  among flipped: mean min magnitude = {min_mag_np[finite].mean():.4f}"
              f"  median = {np.median(min_mag_np[finite]):.4f}")

    # ------------- univariate AUROC for flipped_at_default -------------
    print("\n=== Univariate AUROC vs flipped_at_default ===")
    if flipped_at_default.std() == 0:
        print("  degenerate target — skipping")
    else:
        for name, feat in [("final_margin", final_margin),
                           ("mean_pix", mean_pix),
                           ("std_pix", std_pix)]:
            a = roc_auc_score(flipped_at_default, feat)
            a = max(a, 1 - a)
            print(f"  {name:<14} AUROC = {a:.4f}")

    # ------------- correlations with color_shift_min_magnitude -------------
    print("\n=== Correlations with color_shift_min_magnitude ===")
    for name, feat in [("final_margin", final_margin),
                       ("mean_pix", mean_pix),
                       ("std_pix", std_pix)]:
        pr = pearsonr(feat, min_mag_for_corr)[0]
        sr = spearmanr(feat, min_mag_for_corr)[0]
        print(f"  {name:<14} Pearson = {pr:+.4f}   Spearman = {sr:+.4f}")

    # ------------- cross with FGSM -------------
    print("\n=== Cross with FGSM ===")
    print("  computing min_eps_FGSM (binary search)...")
    t0 = time.time()
    me = []
    for i in range(0, N, 512):
        me.append(min_eps_fgsm(model, x_c[i:i + 512], y_c[i:i + 512]))
    fgsm_min_eps = torch.cat(me).detach().cpu().numpy()
    print(f"    done ({time.time() - t0:.1f}s)  mean = {fgsm_min_eps.mean():.4f}")

    print("  computing FGSM self-flip at eps=15/255...")
    fs = []
    for i in range(0, N, 512):
        fs.append(fgsm_flip_at(model, x_c[i:i + 512], y_c[i:i + 512]))
    fgsm_flip = torch.cat(fs).detach().cpu().numpy().astype(int)
    print(f"    fgsm_self_flip rate = {fgsm_flip.mean():.4f}")

    # correlations color-shift vs FGSM
    pr_min = pearsonr(min_mag_for_corr, fgsm_min_eps)[0]
    sr_min = spearmanr(min_mag_for_corr, fgsm_min_eps)[0]
    print(f"\n  Pearson (color_shift_min_mag, fgsm_min_eps) = {pr_min:+.4f}")
    print(f"  Spearman(color_shift_min_mag, fgsm_min_eps) = {sr_min:+.4f}")

    # contingency / agreement of binary flips
    if flipped_at_default.std() > 0 and fgsm_flip.std() > 0:
        both = ((flipped_at_default == 1) & (fgsm_flip == 1)).mean()
        only_cs = ((flipped_at_default == 1) & (fgsm_flip == 0)).mean()
        only_fg = ((flipped_at_default == 0) & (fgsm_flip == 1)).mean()
        neither = ((flipped_at_default == 0) & (fgsm_flip == 0)).mean()
        print("\n  contingency (rows = color-shift flip, cols = FGSM flip):")
        print(f"    both flipped:        {both:.4f}")
        print(f"    only color-shift:    {only_cs:.4f}")
        print(f"    only FGSM:           {only_fg:.4f}")
        print(f"    neither:             {neither:.4f}")

        # AUROC of color_shift_min_magnitude as a predictor of FGSM self-flip
        try:
            a = roc_auc_score(fgsm_flip, -min_mag_for_corr)  # smaller mag => more vulnerable
            a = max(a, 1 - a)
            print(f"\n  AUROC color_shift_min_mag predicting FGSM self-flip: {a:.4f}")
        except Exception as e:
            print(f"  AUROC failed: {e}")

        # AUROC of flipped_at_default predicting FGSM self-flip
        try:
            a = roc_auc_score(fgsm_flip, flipped_at_default)
            a = max(a, 1 - a)
            print(f"  AUROC flipped_at_default predicting FGSM self-flip:  {a:.4f}")
        except Exception as e:
            print(f"  AUROC failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
