"""
H34: Per-sample local-patch entropy statistics (mean / max / std / 90th-percentile
of patch entropy over a sliding 5x5 window) predict adversarial vulnerability.

Motivation: global image entropy collapses the whole picture to one scalar and
loses *where* complexity sits.  A local-entropy map (Shannon entropy of an
intensity histogram inside a small patch around each pixel) captures spatial
heterogeneity -- a sample with a few very high-entropy hot spots is qualitatively
different from one with uniformly moderate texture, but both can hit the same
global entropy.  We test whether summary statistics of that local map outperform
the single global-entropy scalar as predictors of adversarial vulnerability.

Pipeline:
  1. Train small CNN victim on Fashion-MNIST (10 epochs Adam) -- architecture
     matches diagnostic_test.py / sibling hypothesis files.
  2. For each (correctly-classified) test sample compute a per-pixel local
     entropy map: at each pixel, Shannon entropy of the intensity histogram
     of a 5x5 patch around it (8 bins on [0,1], reflect padding).
     Implementation: torch.nn.functional.unfold -> bincount over 8 bins.
  3. Reduce the map to scalar features:
        - le_mean   : mean of local entropy map
        - le_max    : max  of local entropy map
        - le_std    : std  of local entropy map
        - le_p90    : 90th percentile of local entropy map
  4. Baseline features:
        - victim_margin (clean softmax top1 - top2)
        - mean_pix, std_pix
        - global_entropy : single-scalar Shannon entropy of the whole image,
                           same 8 bins, computed identically -> apples to apples.
  5. Attack targets:
        - flipped_FGSM (eps = 15/255)
        - flipped_PGD  (eps = 15/255, 20 steps, alpha = 2/255)
        - FGSM_min_eps_binary_search (continuous; ranked with Spearman)
  6. Analysis:
        - Univariate AUROC for every (feature, binary target) pair, Spearman
          against min-eps.
        - Multivariate logistic regression compares
              {margin}                    (baseline floor)
              {margin, global_entropy}    (global-only)
              {margin, le_mean, le_max, le_std, le_p90}   (local-only)
              {margin, global_entropy, le_mean, le_max, le_std, le_p90}  (full)
          The interesting question is whether (local-only) beats (global-only)
          and whether (full) beats (global-only) by more than (local-only) alone.

Caveats:
  - Local entropy on 28x28 grayscale with a 5x5 window with only 25 samples per
    histogram and 8 bins is noisy -- a fully-on patch and a fully-off patch
    both have entropy 0, while a half-on/half-off patch peaks at ~1 bit.  The
    statistic is best read as "edge/texture density map", not classical entropy.
  - 8 bins is an arbitrary choice; results will move with bin count.
  - Local-entropy stats are *attribute-style* features computed on the clean
    image -- they say nothing about gradient geometry, so we expect them to be
    correlated with std_pix and (weaker) with victim_margin, but not to dominate
    margin-based predictors.  The point of the experiment is to compare them
    head-to-head against the single global-entropy scalar.
  - Results are reported on a single random seed; no resampling / CIs.

Run:  python h34_local_entropy.py
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
PATCH = 5         # local window size
N_BINS = 8        # histogram bins on [0,1]
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


# ----- local entropy -----

@torch.no_grad()
def local_entropy_map(x, patch=PATCH, n_bins=N_BINS, bs=256):
    """Compute per-pixel local Shannon entropy (bits) over a `patch`x`patch`
    window using `n_bins` bins on [0, 1].  Uses reflect padding so the map
    has the same H,W as the input.

    x: [N, 1, H, W] float in [0,1] on DEVICE.
    Returns: [N, H, W] entropy map on DEVICE.
    """
    N, C, H, W = x.shape
    assert C == 1
    pad = patch // 2
    out_chunks = []
    log2 = float(np.log(2.0))
    for i in range(0, N, bs):
        xb = x[i:i+bs]
        b = xb.size(0)
        xp = F.pad(xb, (pad, pad, pad, pad), mode="reflect")
        # unfold -> [b, patch*patch, H*W]
        patches = F.unfold(xp, kernel_size=patch)
        # bin indices in [0, n_bins-1]
        idx = torch.clamp((patches * n_bins).long(), 0, n_bins - 1)
        # one-hot count -> histogram per (sample, location)
        # shape [b, patch*patch, H*W, n_bins] is too big; instead use scatter_add
        # along a new last dim of size n_bins.
        bcount = torch.zeros(b, idx.size(2), n_bins, device=x.device)
        # reshape idx to [b, H*W, patch*patch] for scatter on dim=2
        idx_t = idx.transpose(1, 2)  # [b, H*W, p*p]
        ones = torch.ones_like(idx_t, dtype=torch.float32)
        bcount.scatter_add_(2, idx_t, ones)
        total = bcount.sum(dim=2, keepdim=True).clamp_min(1.0)
        prob = bcount / total
        # Shannon entropy in bits; 0 * log 0 := 0
        safe = prob.clamp_min(1e-12)
        ent = -(prob * torch.log(safe)).sum(dim=2) / log2  # [b, H*W]
        ent = ent.view(b, H, W)
        out_chunks.append(ent)
    return torch.cat(out_chunks, 0)


@torch.no_grad()
def global_entropy(x, n_bins=N_BINS):
    """Single-scalar Shannon entropy (bits) of the whole image, same n_bins."""
    N = x.size(0)
    flat = x.view(N, -1)
    idx = torch.clamp((flat * n_bins).long(), 0, n_bins - 1)
    bcount = torch.zeros(N, n_bins, device=x.device)
    ones = torch.ones_like(idx, dtype=torch.float32)
    bcount.scatter_add_(1, idx, ones)
    prob = bcount / bcount.sum(dim=1, keepdim=True).clamp_min(1.0)
    safe = prob.clamp_min(1e-12)
    return -(prob * torch.log(safe)).sum(dim=1) / float(np.log(2.0))


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
    margin_c = margin[correct]

    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))

    print(f"Local entropy map (patch={PATCH}x{PATCH}, bins={N_BINS}) ...")
    t0 = time.time()
    le = local_entropy_map(x)  # [N, H, W]
    print(f"  done in {time.time()-t0:.1f}s")

    le_flat = le.view(le.size(0), -1)
    le_mean = le_flat.mean(dim=1)
    le_max = le_flat.max(dim=1).values
    le_std = le_flat.std(dim=1)
    le_p90 = torch.quantile(le_flat, 0.90, dim=1)
    ge = global_entropy(x)

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

    to_np = lambda t: t.detach().cpu().numpy()
    margin_np = to_np(margin_c)
    mean_pix_np = to_np(mean_pix)
    std_pix_np = to_np(std_pix)
    ge_np = to_np(ge)
    le_mean_np = to_np(le_mean)
    le_max_np = to_np(le_max)
    le_std_np = to_np(le_std)
    le_p90_np = to_np(le_p90)
    flip_fgsm_np = to_np(flip_fgsm).astype(int)
    flip_pgd_np = to_np(flip_pgd).astype(int)
    min_eps_np = to_np(min_eps)

    print(f"\nN={len(margin_np)}  "
          f"FGSM flip rate={flip_fgsm_np.mean():.3f}  "
          f"PGD flip rate={flip_pgd_np.mean():.3f}  "
          f"mean min_eps={min_eps_np.mean():.4f}")
    print(f"global_entropy: mean={ge_np.mean():.3f} std={ge_np.std():.3f}")
    print(f"le_mean:        mean={le_mean_np.mean():.3f} std={le_mean_np.std():.3f}")
    print(f"le_max:         mean={le_max_np.mean():.3f} std={le_max_np.std():.3f}")
    print(f"le_std:         mean={le_std_np.mean():.3f} std={le_std_np.std():.3f}")
    print(f"le_p90:         mean={le_p90_np.mean():.3f} std={le_p90_np.std():.3f}")

    feats = {
        "victim_margin":   margin_np,
        "mean_pix":        mean_pix_np,
        "std_pix":         std_pix_np,
        "global_entropy":  ge_np,
        "le_mean":         le_mean_np,
        "le_max":          le_max_np,
        "le_std":          le_std_np,
        "le_p90":          le_p90_np,
    }
    targets = {
        "flipped_FGSM": flip_fgsm_np,
        "flipped_PGD":  flip_pgd_np,
    }

    # ---- univariate ----
    print("\n========== UNIVARIATE AUROC ==========")
    for tname, t in targets.items():
        print(f" target: {tname}  (pos rate = {t.mean():.3f})")
        for fname, fv in feats.items():
            a = auroc_safe(t, fv)
            print(f"   {fname:<18} AUROC = {a:.4f}")

    print(" target: FGSM_min_eps  (Spearman)")
    for fname, fv in feats.items():
        rho, _ = spearmanr(fv, min_eps_np)
        print(f"   {fname:<18} Spearman = {rho:+.4f}")

    # ---- multivariate: local vs global entropy on top of margin ----
    print("\n========== MULTIVARIATE: local- vs global-entropy over margin ==========")
    cfgs = {
        "margin":           ["victim_margin"],
        "+global":          ["victim_margin", "global_entropy"],
        "+local":           ["victim_margin", "le_mean", "le_max", "le_std", "le_p90"],
        "+global+local":    ["victim_margin", "global_entropy",
                             "le_mean", "le_max", "le_std", "le_p90"],
    }
    summary = {tname: {} for tname in targets}
    for cname, cols in cfgs.items():
        X = np.stack([feats[c] for c in cols], axis=1)
        Xs = StandardScaler().fit_transform(X)
        for tname, t in targets.items():
            if t.std() == 0:
                continue
            lr = LogisticRegression(max_iter=2000).fit(Xs, t)
            auc = roc_auc_score(t, lr.predict_proba(Xs)[:, 1])
            summary[tname][cname] = auc
            print(f" {tname:<14} cfg={cname:<14} cols={cols}  AUROC={auc:.4f}")

    print("\n===== SUMMARY (multivariate, train-set AUROC) =====")
    print(f"{'target':<14} {'margin':>8} {'+global':>9} {'+local':>9} "
          f"{'+g+l':>9}  {'local-global':>14}")
    for tname, d in summary.items():
        m   = d.get("margin", float("nan"))
        g   = d.get("+global", float("nan"))
        l   = d.get("+local", float("nan"))
        gl  = d.get("+global+local", float("nan"))
        print(f"{tname:<14} {m:>8.4f} {g:>9.4f} {l:>9.4f} {gl:>9.4f}  "
              f"{(l - g):>+14.4f}")

    print("\nInterpretation: 'local-global' > 0 means the 4 local-entropy stats,"
          " by themselves on top of margin, outperform the single global-entropy"
          " scalar on top of margin -- i.e. spatial heterogeneity adds signal"
          " that the global scalar misses.  This is in-sample (train==test) AUROC"
          " on the logistic fit and so is mildly optimistic; the univariate"
          " AUROCs above are the cleaner per-feature read.")


if __name__ == "__main__":
    main()
