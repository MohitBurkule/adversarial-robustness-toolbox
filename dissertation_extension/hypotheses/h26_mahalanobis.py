"""
H26: Mahalanobis distance in penultimate-feature space (Lee et al., NeurIPS 2018)
predicts adversarial vulnerability.

Background
----------
Lee, Lee, Lee & Shin (NeurIPS 2018), "A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks" (arXiv:1807.03888), fit
class-conditional Gaussians with a shared (tied / LDA-style) covariance on the
penultimate features of a trained classifier and use the resulting Mahalanobis
distance as a confidence score for OOD / adversarial detection.

Here we use the *clean* sample's Mahalanobis geometry, computed using training
features, to predict whether that sample will be flipped by a white-box attack.

Pipeline
--------
1. Train a small CNN victim (matching diagnostic_test.py) on Fashion-MNIST for
   10 epochs with Adam.
2. Extract penultimate features (128-d activations of fc1 after relu) for all
   training samples. Per class fit mu_c and a single pooled covariance Sigma
   (LDA-style, as in Lee et al.).
3. Per test sample compute:
        M_true   = Mahalanobis dist^2 to its OWN class' Gaussian
        M_other  = min Mahalanobis dist^2 to ANY OTHER class
        M_margin = M_other - M_true  (smaller = more confusable)
   We also expose -min_c M_c (the Lee et al. detection score; here negated so
   that "higher = more in-distribution" is preserved as a feature direction).
4. Baseline features:
        victim_margin       (logit gap, top1 - top2 of the victim)
        mean_pix, std_pix   (raw-pixel summaries)
5. Targets:
        flipped_FGSM @ eps = 15/255
        flipped_PGD  @ eps = 15/255 (40 steps, alpha = eps/10)
        FGSM_min_eps        (binary-search smallest eps that FGSM-flips)
6. Univariate AUROC for each feature. Multivariate logistic regression:
        baseline (victim_margin, mean_pix, std_pix)
        baseline + M_margin
   to test whether the Mahalanobis margin adds over victim_margin.
   For FGSM_min_eps (continuous) we use OLS R^2 with the same nested models
   and Spearman correlation.

Self-contained: trains, evaluates, prints. Data cached under /tmp/data.

Run:  python h26_mahalanobis.py
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 40
PGD_ALPHA = EPS_TEST / 10.0
SEED = 0


# ----------------------------- model ----------------------------------------
class CNN(nn.Module):
    """Same architecture as diagnostic_test.py. fc1 (128-d, post-ReLU) is the
    penultimate feature layer used for the Mahalanobis fit."""

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
        x = F.relu(self.fc1(x))   # penultimate features
        return x

    def forward(self, x):
        z = self.features(x)
        z = self.do2(z)
        return self.fc2(z)


# ----------------------------- training --------------------------------------
def train_victim(train_set, n_classes=10):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(n_classes).to(DEVICE)
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


# ------------------------- mahalanobis fit -----------------------------------
@torch.no_grad()
def extract_features(model, dataset, batch=512):
    """Return (features [N,d] float32 on CPU, labels [N] long on CPU)."""
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2)
    feats, labs = [], []
    model.eval()
    for x, y in loader:
        x = x.to(DEVICE)
        feats.append(model.features(x).cpu())
        labs.append(y)
    return torch.cat(feats), torch.cat(labs)


def fit_gaussian(train_feats, train_labels, n_classes=10, ridge=1e-3):
    """Class-conditional means and one tied (pooled) covariance.

    Follows Lee et al. 2018, Sec. 2.1: mu_c = mean over class-c training
    samples, Sigma = (1/N) sum_c sum_{i in c} (x_i - mu_c)(x_i - mu_c)^T.
    Returns mus [C,d], precision [d,d] (Sigma^{-1}).
    """
    d = train_feats.size(1)
    mus = torch.zeros(n_classes, d, dtype=torch.float64)
    centred_chunks = []
    for c in range(n_classes):
        mask = train_labels == c
        Xc = train_feats[mask].to(torch.float64)
        mus[c] = Xc.mean(0)
        centred_chunks.append(Xc - mus[c])
    centred = torch.cat(centred_chunks, 0)
    N = centred.size(0)
    Sigma = (centred.t() @ centred) / N
    Sigma += ridge * torch.eye(d, dtype=torch.float64)  # numerical stability
    precision = torch.linalg.inv(Sigma)
    return mus, precision


def mahalanobis_all(feats, mus, precision):
    """Per-sample Mahalanobis^2 to every class mean.

    feats [N,d], mus [C,d], precision [d,d]; returns [N,C] float64."""
    feats = feats.to(torch.float64)
    # M_c(x) = (x - mu_c)^T P (x - mu_c)
    # = x P x^T - 2 x P mu_c^T + mu_c P mu_c^T
    P = precision
    xP = feats @ P                               # [N,d]
    xPx = (xP * feats).sum(1, keepdim=True)      # [N,1]
    muP = mus @ P                                # [C,d]
    muPmu = (muP * mus).sum(1)                   # [C]
    cross = xP @ mus.t()                         # [N,C]
    return xPx - 2 * cross + muPmu.unsqueeze(0)


def mahalanobis_features(feats, labels, mus, precision):
    """Return dict of per-sample features derived from Mahalanobis distances."""
    D = mahalanobis_all(feats, mus, precision)               # [N,C]
    N, C = D.shape
    idx = torch.arange(N)
    M_true = D[idx, labels]
    D_other = D.clone()
    D_other[idx, labels] = float("inf")
    M_other, _ = D_other.min(1)
    M_margin = M_other - M_true
    # Lee et al. detection score: max_c -M_c(x)  =>  -min_c M_c.
    M_min, _ = D.min(1)
    Lee_score = -M_min
    return {
        "M_true": M_true.float(),
        "M_other": M_other.float(),
        "M_margin": M_margin.float(),
        "Lee_score": Lee_score.float(),
    }


# ------------------------- attacks -------------------------------------------
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    return (x + eps * sign).clamp(0, 1)


def pgd_attack(model, x, y, eps=EPS_TEST, steps=PGD_STEPS, alpha=PGD_ALPHA):
    x0 = x.clone().detach()
    delta = (torch.rand_like(x0) * 2 - 1) * eps
    adv = (x0 + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
    return adv


@torch.no_grad()
def predict(model, x, batch=512):
    out = []
    for i in range(0, x.size(0), batch):
        out.append(model(x[i:i+batch]).argmax(1))
    return torch.cat(out)


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=15, batch=512):
    """Per-sample binary search for smallest L_inf eps that FGSM-flips."""
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        sign = fgsm_grad(model, xb, yb)
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xb + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
            with torch.no_grad():
                flipped = model(adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        out.append(hi)
    return torch.cat(out)


def run_attack(model, x, y, attack_fn, batch=256, **kw):
    flips = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        adv = attack_fn(model, xb, yb, **kw)
        with torch.no_grad():
            flips.append((model(adv).argmax(1) != yb).cpu())
    return torch.cat(flips)


# ------------------------- evaluation ----------------------------------------
def univariate_auroc(feat_np, y):
    a = roc_auc_score(y, feat_np)
    return max(a, 1 - a)


def evaluate(features_dict, targets_binary, target_continuous, target_names_bin):
    feat_names = list(features_dict.keys())
    F_mat = np.stack([features_dict[k] for k in feat_names], 1)

    print("\n--- univariate AUROC (binary targets) ---")
    print(f"{'feature':<16}" + "".join(f"{t:>22}" for t in target_names_bin))
    for i, name in enumerate(feat_names):
        row = f"{name:<16}"
        for y in targets_binary:
            if y.std() == 0:
                row += f"{'n/a':>22}"
            else:
                row += f"{univariate_auroc(F_mat[:, i], y):>22.4f}"
        print(row)

    print("\n--- univariate correlation with FGSM_min_eps (Pearson, Spearman) ---")
    for i, name in enumerate(feat_names):
        r = np.corrcoef(F_mat[:, i], target_continuous)[0, 1]
        rho, _ = spearmanr(F_mat[:, i], target_continuous)
        print(f"  {name:<16} pearson={r:+.4f}  spearman={rho:+.4f}")

    # nested multivariate: does M_margin add over baseline?
    baseline_keys = ["victim_margin", "mean_pix", "std_pix"]
    augment_keys = baseline_keys + ["M_margin"]

    def Xmat(keys):
        return np.stack([features_dict[k] for k in keys], 1)

    Xb = StandardScaler().fit_transform(Xmat(baseline_keys))
    Xa = StandardScaler().fit_transform(Xmat(augment_keys))

    print("\n--- multivariate AUROC: baseline vs baseline+M_margin ---")
    for y, tname in zip(targets_binary, target_names_bin):
        if y.std() == 0:
            print(f"  {tname}: degenerate (mean={y.mean():.3f})")
            continue
        lr_b = LogisticRegression(max_iter=2000).fit(Xb, y)
        lr_a = LogisticRegression(max_iter=2000).fit(Xa, y)
        auc_b = roc_auc_score(y, lr_b.predict_proba(Xb)[:, 1])
        auc_a = roc_auc_score(y, lr_a.predict_proba(Xa)[:, 1])
        print(f"  {tname:<22} baseline={auc_b:.4f}  +M_margin={auc_a:.4f}  "
              f"delta={auc_a-auc_b:+.4f}  (positive rate={y.mean():.3f})")
        print(f"     standardised coefs (augmented): " +
              ", ".join(f"{k}={c:+.3f}" for k, c in
                        zip(augment_keys, lr_a.coef_.flatten())))

    print("\n--- OLS R^2 on FGSM_min_eps: baseline vs baseline+M_margin ---")
    yc = target_continuous
    ols_b = LinearRegression().fit(Xb, yc)
    ols_a = LinearRegression().fit(Xa, yc)
    r2_b = ols_b.score(Xb, yc)
    r2_a = ols_a.score(Xa, yc)
    print(f"  baseline R^2 = {r2_b:.4f}   baseline+M_margin R^2 = {r2_a:.4f}   "
          f"delta = {r2_a-r2_b:+.4f}")
    print(f"  standardised coefs (augmented): " +
          ", ".join(f"{k}={c:+.4f}" for k, c in zip(augment_keys, ols_a.coef_)))


# ------------------------- main ----------------------------------------------
def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print(f"device={DEVICE}  data={DATA_ROOT}")
    print("[1/5] training victim CNN on Fashion-MNIST...")
    t0 = time.time()
    model = train_victim(train_set, n_classes=10)
    print(f"  trained in {time.time()-t0:.1f}s")

    print("[2/5] extracting penultimate features for train set...")
    train_feats, train_labels = extract_features(model, train_set)
    print(f"  train feats shape={tuple(train_feats.shape)}")

    print("[3/5] fitting class-conditional Gaussians with tied covariance...")
    mus, precision = fit_gaussian(train_feats, train_labels, n_classes=10)

    print("[4/5] extracting features and computing test targets...")
    test_feats, test_labels = extract_features(model, test_set)
    mah = mahalanobis_features(test_feats, test_labels, mus, precision)

    # build full test tensor for attacks
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # restrict to samples the victim classifies correctly
    final_pred = predict(model, test_x)
    correct = (final_pred == test_y).cpu()
    print(f"  victim test accuracy = {correct.float().mean():.4f}")
    keep = correct.nonzero(as_tuple=False).flatten()
    x_c = test_x[keep.to(DEVICE)]
    y_c = test_y[keep.to(DEVICE)]
    feats_c = {k: v[keep].numpy() for k, v in mah.items()}

    # baseline features
    with torch.no_grad():
        logits = []
        for i in range(0, x_c.size(0), 512):
            logits.append(model(x_c[i:i+512]))
        logits = torch.cat(logits, 0)
    sorted_logits, _ = logits.sort(1, descending=True)
    victim_margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).cpu().numpy()
    flat = x_c.view(x_c.size(0), -1).cpu().numpy()
    mean_pix = flat.mean(1)
    std_pix = flat.std(1)

    features_dict = {
        "victim_margin": victim_margin,
        "mean_pix": mean_pix,
        "std_pix": std_pix,
        "M_true": feats_c["M_true"],
        "M_other": feats_c["M_other"],
        "M_margin": feats_c["M_margin"],
        "Lee_score": feats_c["Lee_score"],
    }

    print(f"  evaluating on {x_c.size(0)} correctly-classified test samples")
    print("[5/5] running attacks...")
    t0 = time.time()
    fgsm_flip = run_attack(model, x_c, y_c, fgsm_attack, eps=EPS_TEST).numpy().astype(int)
    print(f"  FGSM done ({time.time()-t0:.1f}s)  flip rate={fgsm_flip.mean():.4f}")
    t0 = time.time()
    pgd_flip = run_attack(model, x_c, y_c, pgd_attack,
                          eps=EPS_TEST, steps=PGD_STEPS, alpha=PGD_ALPHA).numpy().astype(int)
    print(f"  PGD  done ({time.time()-t0:.1f}s)  flip rate={pgd_flip.mean():.4f}")
    t0 = time.time()
    min_eps = fgsm_min_eps(model, x_c, y_c).cpu().numpy()
    print(f"  min_eps done ({time.time()-t0:.1f}s)  mean={min_eps.mean():.4f}")

    targets_binary = [fgsm_flip, pgd_flip]
    target_names_bin = ["flipped_FGSM@15/255", "flipped_PGD@15/255"]

    evaluate(features_dict, targets_binary, min_eps, target_names_bin)
    print("\nDone.")


if __name__ == "__main__":
    main()
