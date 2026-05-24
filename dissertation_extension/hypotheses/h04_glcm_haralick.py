"""
Hypothesis H04: GLCM (Gray-Level Co-occurrence Matrix) Haralick texture features
predict adversarial vulnerability better than the simple Sobel/std features.

Classical Haralick descriptors (contrast, dissimilarity, homogeneity, energy,
correlation, ASM, entropy) computed from the GLCM at distance=1, four angles
[0, pi/4, pi/2, 3pi/4], averaged across angles.

Pipeline:
  1. Train small CNN on Fashion-MNIST (10 epochs, Adam, mirrors diagnostic_test.py).
  2. For each test sample, compute:
       - baseline features: victim_margin, mean_pix, std_pix, sobel_mean
       - GLCM features:     contrast, dissimilarity, homogeneity, energy,
                            correlation, ASM, haralick_entropy
  3. Targets: flipped_FGSM (eps=15/255), flipped_PGD, min_eps_FGSM (binary search).
  4. Univariate AUROC per feature.
  5. Multivariate logistic regression for four nested feature sets:
       margin / margin+glcm / margin+stats / margin+glcm+stats
"""
import os
import sys
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- ensure scikit-image is available ---------------------------------------
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    print("scikit-image not found, attempting pip install into current venv...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
    from skimage.feature import graycomatrix, graycoprops

from scipy.ndimage import sobel  # comes with scipy, used for sobel_mean baseline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0

GLCM_DISTANCES = [1]
GLCM_ANGLES = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
GLCM_LEVELS = 32  # quantise 8-bit images down to 32 levels for GLCM


class CNN(nn.Module):
    """Same architecture as diagnostic_test.py."""
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
    # random start within eps-ball
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
    """Binary-search smallest L_inf eps that flips FGSM."""
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


def baseline_pixel_features(x_np):
    """mean_pix, std_pix, sobel_mean over (N, 28, 28) float32 in [0,1]."""
    N = x_np.shape[0]
    mean_pix = x_np.reshape(N, -1).mean(axis=1)
    std_pix = x_np.reshape(N, -1).std(axis=1)
    sobel_mean = np.zeros(N, dtype=np.float32)
    for i in range(N):
        sx = sobel(x_np[i], axis=0)
        sy = sobel(x_np[i], axis=1)
        sobel_mean[i] = np.sqrt(sx * sx + sy * sy).mean()
    return mean_pix, std_pix, sobel_mean


def glcm_features(x_np, levels=GLCM_LEVELS):
    """
    Compute Haralick GLCM features per sample.
    Returns array of shape (N, 7): contrast, dissimilarity, homogeneity,
    energy, correlation, ASM, haralick_entropy. Each is averaged over the
    four angles at distance=1.
    """
    N = x_np.shape[0]
    feat_names = ["glcm_contrast", "glcm_dissim", "glcm_homog",
                  "glcm_energy", "glcm_corr", "glcm_asm", "glcm_entropy"]
    out = np.zeros((N, 7), dtype=np.float32)
    # quantise floats in [0,1] to integer levels in [0, levels-1]
    q = np.clip((x_np * levels).astype(np.int32), 0, levels - 1).astype(np.uint8)
    eps = 1e-12
    for i in range(N):
        g = graycomatrix(q[i], distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                         levels=levels, symmetric=True, normed=True)
        # graycoprops returns shape (n_distances, n_angles)
        out[i, 0] = graycoprops(g, "contrast").mean()
        out[i, 1] = graycoprops(g, "dissimilarity").mean()
        out[i, 2] = graycoprops(g, "homogeneity").mean()
        out[i, 3] = graycoprops(g, "energy").mean()
        out[i, 4] = graycoprops(g, "correlation").mean()
        out[i, 5] = graycoprops(g, "ASM").mean()
        # Haralick entropy: -sum(P * log(P)), averaged across angles
        P = g[:, :, 0, :]  # (levels, levels, n_angles)
        ent = -(P * np.log(P + eps)).sum(axis=(0, 1))
        out[i, 6] = ent.mean()
        if (i + 1) % 1000 == 0:
            print(f"   glcm: {i+1}/{N}")
    return out, feat_names


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
        print(f"    {set_name:<28} ({len(cols)} feats)  AUROC = {a:.4f}")


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

    # restrict to correctly classified samples (vulnerability defined on them)
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

    x_np = x.squeeze(1).cpu().numpy().astype(np.float32)

    print("computing baseline pixel/sobel features...")
    mean_pix, std_pix, sobel_mean = baseline_pixel_features(x_np)

    print("computing GLCM Haralick features...")
    t0 = time.time()
    glcm_feats, glcm_names = glcm_features(x_np)
    print(f"  glcm done in {time.time()-t0:.1f}s")

    baseline_names = ["victim_margin", "mean_pix", "std_pix", "sobel_mean"]
    feat_names = baseline_names + glcm_names
    X = np.column_stack([margin, mean_pix, std_pix, sobel_mean, glcm_feats])
    assert X.shape[1] == len(feat_names)

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
    # Binary target: smallest-eps samples = bottom 25% (most vulnerable)
    me_thresh = np.quantile(min_eps, 0.25)
    min_eps_bin = (min_eps <= me_thresh).astype(int)

    targets = {
        "flipped_FGSM (eps=15/255)": fgsm_flip,
        "flipped_PGD (eps=15/255)": pgd_flip,
        f"min_eps_FGSM<=q25 ({me_thresh:.4f})": min_eps_bin,
    }

    # ---- feature sets for multivariate ----
    sets = [
        ("margin",             ["victim_margin"]),
        ("margin + glcm",      ["victim_margin"] + glcm_names),
        ("margin + stats",     ["victim_margin", "mean_pix", "std_pix", "sobel_mean"]),
        ("margin + glcm + stats", feat_names),
    ]

    for tname, yt in targets.items():
        print(f"\n========== target: {tname}  (pos rate = {yt.mean():.3f}) ==========")
        if yt.std() == 0:
            print("  degenerate target, skipping")
            continue
        univariate_auroc(X, yt, feat_names)
        multivariate_auroc(X, yt, feat_names, sets)

    # ---- continuous correlation with min_eps_FGSM ----
    print("\n========== Pearson corr with min_eps_FGSM (continuous) ==========")
    for i, n in enumerate(feat_names):
        c = np.corrcoef(X[:, i], min_eps)[0, 1]
        print(f"   corr({n:<22}, min_eps) = {c:+.4f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
