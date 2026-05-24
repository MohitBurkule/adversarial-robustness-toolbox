"""
Hypothesis H14: Image statistics computed at multiple spatial scales
(Gaussian pyramid levels) are stronger model-free vulnerability predictors
than single-scale stats.

Motivation: single-scale pixel stats (mean, std, sobel) gave AUROC ~0.70 on
Fashion-MNIST in earlier hypotheses. Adversarial attacks may exploit different
spatial frequencies, so a multi-scale (image pyramid) descriptor might pick up
signal that the single-scale stats miss.

Pipeline:
  1. Train small CNN victim on Fashion-MNIST (10 epochs Adam, same architecture
     as diagnostic_test.py / h04).
  2. For each correctly-classified test sample build a 4-level Gaussian pyramid
     (sigma=1, decimate by 2 between levels). At each level compute:
       - mean
       - std
       - Sobel-magnitude mean
     -> 12 multi-scale features (level0..level3 x {mean,std,sobel}).
  3. Baselines: victim_margin, single-scale stats {mean_pix, std_pix, sobel_mean}
     (which are identical to the level-0 pyramid features but kept under the
     "single-scale" names for clarity in feature-set comparisons).
  4. Targets:
       - flipped_FGSM   (eps=15/255)
       - flipped_PGD    (eps=15/255, 20 steps)
       - min_eps_FGSM <= q25  (binary search smallest-eps that flips with FGSM)
  5. Univariate AUROC for every pyramid feature.
     Multivariate logistic regression for nested sets:
       margin / stats(3) / pyramid(12) / margin+stats / margin+pyramid /
       margin+stats+pyramid
     Asks: does pyramid (12) beat single-scale (3)?
            does pyramid+margin beat margin alone?

Tools: PyTorch + CUDA. Gaussian blur implemented as a separable conv with a
sigma=1 kernel; downsampling via F.avg_pool2d(2) for clean decimation
(equivalent to standard pyrDown when paired with the preceding blur).
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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

PYRAMID_LEVELS = 4
GAUSS_SIGMA = 1.0


# --- model (mirrors diagnostic_test.py) -------------------------------------
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


# --- training ---------------------------------------------------------------
def train_victim(seed, train_set):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# --- attacks ----------------------------------------------------------------
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x_orig = x.clone().detach()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    adv = (x_orig + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0].sign()
        adv = adv.detach() + alpha * grad
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    sign = fgsm_grad(model, x, y)
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# --- features ---------------------------------------------------------------
def victim_margin(model, x):
    with torch.no_grad():
        out = []
        for i in range(0, x.size(0), 512):
            logits = model(x[i:i+512])
            s, _ = logits.sort(1, descending=True)
            out.append((s[:, 0] - s[:, 1]).cpu())
        return torch.cat(out).numpy()


def _gaussian_kernel_1d(sigma, device):
    radius = max(1, int(round(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    return k


def gaussian_blur(x, sigma=GAUSS_SIGMA):
    """Separable Gaussian blur on (N,1,H,W) tensors, reflect-padded."""
    k = _gaussian_kernel_1d(sigma, x.device)
    r = (k.numel() - 1) // 2
    kh = k.view(1, 1, 1, -1)
    kv = k.view(1, 1, -1, 1)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, kh)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, kv)
    return x


# Sobel kernels for batched magnitude.
_SOBEL_X = torch.tensor([[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1., -2., -1.],
                         [ 0.,  0.,  0.],
                         [ 1.,  2.,  1.]]).view(1, 1, 3, 3)


def sobel_mean(x):
    """Mean Sobel magnitude over each (1,H,W) image. x: (N,1,H,W) -> (N,)."""
    sx = _SOBEL_X.to(x.device)
    sy = _SOBEL_Y.to(x.device)
    xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(xp, sx)
    gy = F.conv2d(xp, sy)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag.mean(dim=(1, 2, 3))


def pyramid_features(x, levels=PYRAMID_LEVELS, sigma=GAUSS_SIGMA):
    """
    Build a Gaussian pyramid and return (N, 3*levels) features:
    for each level l in [0..levels-1]: mean_l, std_l, sobel_l.

    Level 0 is the original image. Level l+1 = avg_pool2d(blur(level_l), 2).
    """
    feats = []
    names = []
    cur = x
    for l in range(levels):
        if l > 0:
            blurred = gaussian_blur(cur, sigma=sigma)
            # decimate by 2
            cur = F.avg_pool2d(blurred, kernel_size=2, stride=2)
        flat = cur.view(cur.size(0), -1)
        m = flat.mean(dim=1)
        s = flat.std(dim=1, unbiased=False)
        sb = sobel_mean(cur)
        feats.append(torch.stack([m, s, sb], dim=1))  # (N,3)
        names += [f"pyr_L{l}_mean", f"pyr_L{l}_std", f"pyr_L{l}_sobel"]
    out = torch.cat(feats, dim=1).cpu().numpy()
    return out, names


# --- analysis ---------------------------------------------------------------
def univariate_auroc(X, y, names):
    print("\n  univariate AUROC (max(auc, 1-auc)):")
    for i, n in enumerate(names):
        try:
            a = roc_auc_score(y, X[:, i])
            a = max(a, 1 - a)
            print(f"    {n:<22} {a:.4f}")
        except Exception as e:
            print(f"    {n:<22} ERROR {e}")


def multivariate_auroc(X_full, y, names, sets):
    print("\n  multivariate logistic-regression AUROC (in-sample):")
    name_to_idx = {n: i for i, n in enumerate(names)}
    for set_name, set_names in sets:
        cols = [name_to_idx[n] for n in set_names if n in name_to_idx]
        X = X_full[:, cols]
        Xs = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=2000).fit(Xs, y)
        a = roc_auc_score(y, clf.predict_proba(Xs)[:, 1])
        print(f"    {set_name:<32} ({len(cols):>2} feats)  AUROC = {a:.4f}")


# --- main -------------------------------------------------------------------
def main():
    print(f"device = {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print("training victim CNN on Fashion-MNIST...")
    t0 = time.time()
    model = train_victim(0, train_set)
    print(f"trained in {time.time()-t0:.1f}s")

    # materialise full test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)
    print(f"test set: N={N}")

    with torch.no_grad():
        preds = []
        for i in range(0, N, 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct = preds == test_y
    print(f"clean accuracy = {correct.float().mean().item():.4f}")
    x = test_x[correct]
    y = test_y[correct]
    Nc = x.size(0)
    print(f"using {Nc} correctly-classified samples")

    # ---- features ----
    print("\ncomputing victim margin...")
    margin = victim_margin(model, x)

    print("computing single-scale (level-0) stats...")
    # Single-scale baseline: mean, std, sobel on the original image only.
    with torch.no_grad():
        flat0 = x.view(Nc, -1)
        mean_pix = flat0.mean(dim=1).cpu().numpy()
        std_pix = flat0.std(dim=1, unbiased=False).cpu().numpy()
        sobel_pix = sobel_mean(x).cpu().numpy()

    print("computing 4-level Gaussian pyramid features...")
    t0 = time.time()
    # Batch through GPU to keep memory low.
    pyr_chunks = []
    pyr_names = None
    with torch.no_grad():
        for i in range(0, Nc, 512):
            feats, pyr_names = pyramid_features(x[i:i+512])
            pyr_chunks.append(feats)
    pyr_feats = np.concatenate(pyr_chunks, axis=0)
    print(f"  pyramid done in {time.time()-t0:.1f}s  shape={pyr_feats.shape}")

    baseline_names = ["victim_margin", "mean_pix", "std_pix", "sobel_mean"]
    feat_names = baseline_names + pyr_names
    X = np.column_stack([margin, mean_pix, std_pix, sobel_pix, pyr_feats])
    assert X.shape[1] == len(feat_names)
    print(f"feature matrix: {X.shape}, columns: {feat_names}")

    # ---- targets ----
    print("\ncomputing flipped_FGSM ...")
    fgsm_flip = []
    for i in range(0, Nc, 512):
        fgsm_flip.append(fgsm_attack(model, x[i:i+512], y[i:i+512]))
    fgsm_flip = torch.cat(fgsm_flip).cpu().numpy().astype(int)

    print("computing flipped_PGD ...")
    pgd_flip = []
    for i in range(0, Nc, 512):
        pgd_flip.append(pgd_attack(model, x[i:i+512], y[i:i+512]))
    pgd_flip = torch.cat(pgd_flip).cpu().numpy().astype(int)

    print("computing min_eps_FGSM (binary search) ...")
    me = []
    for i in range(0, Nc, 512):
        me.append(min_eps_fgsm(model, x[i:i+512], y[i:i+512]))
    min_eps = torch.cat(me).cpu().numpy()
    me_thresh = np.quantile(min_eps, 0.25)
    min_eps_bin = (min_eps <= me_thresh).astype(int)

    targets = {
        "flipped_FGSM (eps=15/255)": fgsm_flip,
        "flipped_PGD (eps=15/255)": pgd_flip,
        f"min_eps_FGSM<=q25 ({me_thresh:.4f})": min_eps_bin,
    }

    stats_names = ["mean_pix", "std_pix", "sobel_mean"]
    sets = [
        ("margin",                       ["victim_margin"]),
        ("single-scale stats",           stats_names),
        ("pyramid (12)",                 pyr_names),
        ("margin + single-scale stats",  ["victim_margin"] + stats_names),
        ("margin + pyramid",             ["victim_margin"] + pyr_names),
        ("margin + stats + pyramid",     ["victim_margin"] + stats_names + pyr_names),
    ]

    for tname, yt in targets.items():
        print(f"\n========== target: {tname}  (pos rate = {yt.mean():.3f}) ==========")
        if yt.std() == 0:
            print("  degenerate target, skipping")
            continue
        univariate_auroc(X, yt, feat_names)
        multivariate_auroc(X, yt, feat_names, sets)

    print("\n========== Pearson corr with min_eps_FGSM (continuous) ==========")
    for i, n in enumerate(feat_names):
        c = np.corrcoef(X[:, i], min_eps)[0, 1]
        print(f"   corr({n:<22}, min_eps) = {c:+.4f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
