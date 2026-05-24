"""
h93_kde_density.py

Hypothesis: Per-class Kernel Density Estimation on penultimate-layer features
(Feinman et al. 2017, "Detecting Adversarial Samples from Artifacts") predicts
adversarial vulnerability of clean samples.

For each test sample we compute:
  - log p(phi(x) | y) under a Gaussian KDE fit on training penultimate features
    of the sample's own class
  - log p(phi(x) | y') under the KDE of the nearest other class (in mean feature
    distance from the sample)
  - density margin: own - nearest-other

Per-sample features:
  - own_log_density
  - nearest_other_log_density
  - density_margin
  - mean_pix
  - std_pix

Vulnerability targets (computed on samples Model A classifies correctly):
  - flipped_by_FGSM_self  @ eps=15/255
  - flipped_by_PGD_self   @ eps=15/255, 10 steps
  - min_eps_FGSM          (binary-search smallest L_inf eps to flip)

We report univariate AUROC for each feature against the two binary targets
and Pearson correlation against min_eps, plus a multivariate logistic
regression / OLS using all features.

This is a CODE-ONLY hypothesis script; do not run.

Architecture / training match dissertation_extension/diagnostic_test.py
(CNN: conv 1->32, conv 32->64, maxpool 2, dropout 0.25, fc 64*12*12->128,
dropout 0.5, fc 128->10), Adam lr=1e-3, batch 128, Fashion-MNIST, 10 epochs.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0
N_CLASSES = 10
KDE_FIT_PER_CLASS = 2000  # subsample training features per class to keep KDE tractable


class CNN(nn.Module):
    """Matches diagnostic_test.py CNN; exposes penultimate features."""
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def features(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        h = self.features(x)
        h = self.do2(h)
        return self.fc2(h)


def train_model(seed, train_set, n_classes=10, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, x, batch=512):
    feats = []
    for i in range(0, x.size(0), batch):
        feats.append(model.features(x[i:i + batch]).cpu())
    return torch.cat(feats, 0).numpy()


def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack_success(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack_success(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0].sign()
        adv = (adv.detach() + alpha * grad).clamp(min=x0 - eps, max=x0 + eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
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


def auto_bandwidth(feats_2d):
    """Silverman-style bandwidth on standardised features."""
    n, d = feats_2d.shape
    sigma = float(np.median(feats_2d.std(0)) + 1e-12)
    bw = sigma * (n ** (-1.0 / (d + 4)))
    return max(bw, 1e-3)


def fit_per_class_kdes(train_feats, train_labels, n_classes=N_CLASSES,
                       subsample=KDE_FIT_PER_CLASS, seed=0):
    rng = np.random.RandomState(seed)
    kdes = {}
    class_means = np.zeros((n_classes, train_feats.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        idx = np.where(train_labels == c)[0]
        if len(idx) > subsample:
            idx = rng.choice(idx, subsample, replace=False)
        feats_c = train_feats[idx]
        bw = auto_bandwidth(feats_c)
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(feats_c)
        kdes[c] = kde
        class_means[c] = feats_c.mean(0)
    return kdes, class_means


def compute_kde_features(test_feats, test_labels, kdes, class_means, n_classes=N_CLASSES):
    """For each test sample: own-class log-density and nearest-other-class log-density."""
    N = test_feats.shape[0]
    own_logp = np.zeros(N, dtype=np.float32)
    other_logp = np.zeros(N, dtype=np.float32)

    # nearest other class per sample (by L2 distance to class means, excluding own class)
    dists = np.linalg.norm(test_feats[:, None, :] - class_means[None, :, :], axis=2)
    mask_own = np.zeros_like(dists, dtype=bool)
    mask_own[np.arange(N), test_labels] = True
    dists_masked = np.where(mask_own, np.inf, dists)
    nearest_other = dists_masked.argmin(1)

    # batched scoring per class to amortise sklearn overhead
    for c in range(n_classes):
        own_mask = test_labels == c
        if own_mask.any():
            own_logp[own_mask] = kdes[c].score_samples(test_feats[own_mask])
        oth_mask = nearest_other == c
        if oth_mask.any():
            other_logp[oth_mask] = kdes[c].score_samples(test_feats[oth_mask])
    return own_logp, other_logp, nearest_other


def evaluate(feats_np, feat_names, targets_bin, target_names, min_eps_np):
    Xs = StandardScaler().fit_transform(feats_np)
    print("\n--- univariate AUROC ---")
    for t_name, y in zip(target_names, targets_bin):
        y = y.astype(int)
        if y.std() == 0:
            print(f"  {t_name}: degenerate (pos rate = {y.mean():.3f})")
            continue
        print(f"\n  target: {t_name}  (pos rate = {y.mean():.3f})")
        for i, n in enumerate(feat_names):
            a = roc_auc_score(y, feats_np[:, i])
            a = max(a, 1 - a)
            print(f"    {n:<26} AUROC = {a:.4f}")
        lr = LogisticRegression(max_iter=4000).fit(Xs, y)
        full_auc = roc_auc_score(y, lr.predict_proba(Xs)[:, 1])
        print(f"    [multivariate logistic] AUROC = {full_auc:.4f}")
        for n, c in zip(feat_names, lr.coef_.flatten()):
            print(f"      coef {n:<26} {c:+.4f}")

    print("\n--- continuous target: min_eps_FGSM ---")
    for i, n in enumerate(feat_names):
        cor = np.corrcoef(feats_np[:, i], min_eps_np)[0, 1]
        print(f"    corr(min_eps, {n:<26}) = {cor:+.4f}")
    ols = LinearRegression().fit(Xs, min_eps_np)
    print(f"    [multivariate OLS] R^2 = {ols.score(Xs, min_eps_np):.4f}")
    for n, c in zip(feat_names, ols.coef_):
        print(f"      coef {n:<26} {c:+.6f}")


def run():
    print("##### h93: per-class KDE density on penultimate features #####")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(" training CNN (seed 0, 10 epochs) ...")
    t0 = time.time()
    model = train_model(0, train_set, N_CLASSES, EPOCHS)
    print(f"  done ({time.time() - t0:.1f}s)")

    # tensors
    train_x = torch.stack([train_set[i][0] for i in range(len(train_set))]).to(DEVICE)
    train_y = torch.tensor([train_set[i][1] for i in range(len(train_set))]).to(DEVICE)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print(" extracting penultimate features ...")
    t0 = time.time()
    train_feats = extract_features(model, train_x)
    test_feats = extract_features(model, test_x)
    print(f"  done ({time.time() - t0:.1f}s)  train_feats={train_feats.shape}")

    print(" fitting per-class KDEs ...")
    t0 = time.time()
    kdes, class_means = fit_per_class_kdes(train_feats, train_y.cpu().numpy(),
                                           N_CLASSES, KDE_FIT_PER_CLASS, seed=0)
    print(f"  done ({time.time() - t0:.1f}s)")

    print(" scoring test samples under KDEs ...")
    t0 = time.time()
    own_logp, other_logp, _ = compute_kde_features(test_feats, test_y.cpu().numpy(),
                                                   kdes, class_means, N_CLASSES)
    print(f"  done ({time.time() - t0:.1f}s)")
    density_margin = own_logp - other_logp

    # pixel-stat baselines
    mean_pix = test_x.mean(dim=(1, 2, 3)).cpu().numpy()
    std_pix = test_x.std(dim=(1, 2, 3)).cpu().numpy()

    # restrict to correctly classified
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i + 512]).argmax(1))
        preds = torch.cat(preds)
    correct = (preds == test_y).cpu().numpy()
    print(f" using {int(correct.sum())} correctly-classified test samples")

    x_c = test_x[torch.as_tensor(correct, device=DEVICE)]
    y_c = test_y[torch.as_tensor(correct, device=DEVICE)]

    print(" computing FGSM attack success ...")
    fgsm_flip = []
    for i in range(0, x_c.size(0), 512):
        fgsm_flip.append(fgsm_attack_success(model, x_c[i:i + 512], y_c[i:i + 512]))
    fgsm_flip = torch.cat(fgsm_flip).cpu().numpy()

    print(" computing PGD attack success ...")
    pgd_flip = []
    for i in range(0, x_c.size(0), 512):
        pgd_flip.append(pgd_attack_success(model, x_c[i:i + 512], y_c[i:i + 512]))
    pgd_flip = torch.cat(pgd_flip).cpu().numpy()

    print(" computing min_eps_to_flip (FGSM) ...")
    me = []
    for i in range(0, x_c.size(0), 512):
        me.append(min_eps_to_flip(model, x_c[i:i + 512], y_c[i:i + 512]))
    min_eps = torch.cat(me).cpu().numpy()

    # assemble features restricted to correctly-classified samples
    feats_np = np.stack([
        own_logp[correct],
        other_logp[correct],
        density_margin[correct],
        mean_pix[correct],
        std_pix[correct],
    ], axis=1).astype(np.float64)
    feat_names = ["own_log_density",
                  "nearest_other_log_density",
                  "density_margin",
                  "mean_pix",
                  "std_pix"]

    target_names = ["FGSM_flip", "PGD_flip"]
    targets_bin = [fgsm_flip, pgd_flip]
    evaluate(feats_np, feat_names, targets_bin, target_names, min_eps)


if __name__ == "__main__":
    run()
