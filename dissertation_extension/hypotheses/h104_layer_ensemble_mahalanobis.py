"""
H104: Layer-ensemble Mahalanobis (Lee et al. 2018 multi-layer variant)

Hypothesis: the SUM of Mahalanobis OOD scores across multiple intermediate
layers is a stronger predictor of adversarial vulnerability than a single-layer
Mahalanobis score.

Pipeline:
  1. Train a small CNN (matching diagnostic_test.py architecture) on
     Fashion-MNIST for 10 epochs.
  2. Extract features at three layers for the entire training set:
        L1 = relu(conv2)              (after second conv + ReLU)
        L2 = after maxpool (flattened)
        L3 = relu(fc1)                (penultimate)
  3. For each layer fit per-class Gaussians with a SHARED (tied) covariance
     using training-set features (Lee et al. 2018, eq. 1-2).
  4. For each test sample compute the Mahalanobis "confidence" at every layer:
        M_l(x) = max_c  -(f_l(x) - mu_{l,c})^T Sigma_l^{-1} (f_l(x) - mu_{l,c})
     The combined ensemble score is the SUM across layers:
        M_sum(x) = sum_l M_l(x)
  5. Per-layer features and the combined score are evaluated against three
     adversarial-vulnerability targets:
        - FGSM_flip   (eps = 15/255)
        - PGD_flip    (eps = 15/255, 10 steps)
        - min_eps     (binary-search smallest FGSM eps that flips; continuous)
  6. Univariate AUROC per feature + multivariate logistic regression using all
     per-layer scores (and the combined score) as inputs.

Run:
    python h104_layer_ensemble_mahalanobis.py
"""
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
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0
N_CLASSES = 10
LAYER_NAMES = ["relu_conv2", "after_pool", "fc1"]


# ------------------------------------------------------------------ model

class CNN(nn.Module):
    """Same architecture as diagnostic_test.py CNN."""
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x, return_features=False):
        x = F.relu(self.c1(x))
        h1 = F.relu(self.c2(x))                  # L1: relu_conv2
        p = F.max_pool2d(h1, 2)
        h2 = p.flatten(1)                        # L2: after_pool (flattened)
        d = self.do1(p).flatten(1)
        h3 = F.relu(self.fc1(d))                 # L3: fc1
        out = self.fc2(self.do2(h3))
        if return_features:
            # use global average pooling on conv features to keep dims small
            f1 = F.adaptive_avg_pool2d(h1, 1).flatten(1)   # (B, 32)
            f2 = F.adaptive_avg_pool2d(p, 1).flatten(1)    # (B, 64)
            f3 = h3                                        # (B, 128)
            return out, [f1, f2, f3]
        return out


# ------------------------------------------------------------------ train

def train_model(train_set, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"   epoch {ep+1}/{EPOCHS} done")
    model.eval()
    return model


# ----------------------------------------------------- feature extraction

@torch.no_grad()
def extract_features(model, x, batch=512):
    """Run model in batches, collect feature tensors per layer."""
    model.eval()
    feats = [[] for _ in LAYER_NAMES]
    logits_all = []
    for i in range(0, x.size(0), batch):
        out, fs = model(x[i:i+batch].to(DEVICE), return_features=True)
        logits_all.append(out.cpu())
        for k, f in enumerate(fs):
            feats[k].append(f.cpu())
    return [torch.cat(f, 0) for f in feats], torch.cat(logits_all, 0)


# ----------------------------------------------- fit per-layer Gaussians

def fit_class_gaussians(feats, labels, n_classes=N_CLASSES):
    """
    Lee et al. 2018: per-class means + a single SHARED (tied) covariance.
    feats: (N, D) torch tensor on CPU
    labels: (N,) torch tensor on CPU
    returns: means (C, D), precision (D, D)  -- both numpy float64
    """
    feats_np = feats.numpy().astype(np.float64)
    labels_np = labels.numpy()
    D = feats_np.shape[1]
    means = np.zeros((n_classes, D), dtype=np.float64)
    centered = np.zeros_like(feats_np)
    for c in range(n_classes):
        m = labels_np == c
        means[c] = feats_np[m].mean(0)
        centered[m] = feats_np[m] - means[c]
    # tied (shared) covariance, ML estimator
    cov = (centered.T @ centered) / feats_np.shape[0]
    # regularise for numerical stability
    cov += np.eye(D) * 1e-4
    precision = np.linalg.inv(cov)
    return means, precision


def mahalanobis_score(feats, means, precision):
    """
    Per-sample Mahalanobis confidence:
        M(x) = max_c  -(f - mu_c)^T Sigma^{-1} (f - mu_c)
    Larger = more in-distribution. Returns (N,) numpy array.
    """
    f = feats.numpy().astype(np.float64)
    # compute squared mahalanobis to each class mean
    # d_c = (f - mu_c) P (f - mu_c)^T   for each c
    N = f.shape[0]
    C = means.shape[0]
    dists = np.zeros((N, C), dtype=np.float64)
    for c in range(C):
        diff = f - means[c]                         # (N, D)
        # (N, D) @ (D, D) -> (N, D); then row-wise dot with diff
        tmp = diff @ precision
        dists[:, c] = np.einsum("nd,nd->n", tmp, diff)
    return -dists.min(1)


# ---------------------------------------------------- adversarial attacks

def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x_orig = x.clone().detach()
    adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps)
        adv = adv.clamp(0, 1).detach()
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf FGSM eps that flips."""
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


def batched_attack(fn, model, x, y, batch=512):
    out = []
    for i in range(0, x.size(0), batch):
        out.append(fn(model, x[i:i+batch].to(DEVICE), y[i:i+batch].to(DEVICE)))
    return torch.cat(out).cpu()


# ----------------------------------------------------------- evaluation

def evaluate(feats_np, names, targets, target_names):
    print("\n========== H104: layer-ensemble Mahalanobis ==========")
    Xs = StandardScaler().fit_transform(feats_np)
    for t_idx, t_name in enumerate(target_names):
        y = targets[t_idx]
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        print(f"\n--- target: {t_name} ---")
        if t_name == "min_eps":
            # continuous target: report Pearson correlation, plus
            # binarise at median for an AUROC-style readout
            for i, n in enumerate(names):
                cor = np.corrcoef(feats_np[:, i], y)[0, 1]
                print(f"   corr({n:<28}, min_eps) = {cor:+.4f}")
            y_bin = (y < np.median(y)).astype(int)        # 1 = more vulnerable
            print(f"   (binarised at median, pos rate = {y_bin.mean():.3f})")
            for i, n in enumerate(names):
                a = roc_auc_score(y_bin, feats_np[:, i])
                a = max(a, 1 - a)
                print(f"   univariate AUROC  {n:<28} {a:.4f}")
            try:
                lr = LogisticRegression(max_iter=2000).fit(Xs, y_bin)
                full_auc = roc_auc_score(y_bin, lr.predict_proba(Xs)[:, 1])
                print(f"   multivariate AUROC (all features):  {full_auc:.4f}")
            except Exception as e:
                print(f"   multivariate failed: {e}")
            continue

        y = y.astype(int)
        if y.std() == 0:
            print(f"   target degenerate (pos rate = {y.mean():.3f})")
            continue
        print(f"   positive rate = {y.mean():.3f}")
        for i, n in enumerate(names):
            a = roc_auc_score(y, feats_np[:, i])
            a = max(a, 1 - a)
            print(f"   univariate AUROC  {n:<28} {a:.4f}")
        try:
            lr = LogisticRegression(max_iter=2000).fit(Xs, y)
            full_auc = roc_auc_score(y, lr.predict_proba(Xs)[:, 1])
            print(f"   multivariate AUROC (all features):  {full_auc:.4f}")
            for n, c in zip(names, lr.coef_.flatten()):
                print(f"     {n:<28} coef = {c:+.4f}")
        except Exception as e:
            print(f"   multivariate failed: {e}")


# ------------------------------------------------------------------ main

def main():
    tf = transforms.ToTensor()
    print("Loading Fashion-MNIST...")
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("Training CNN (10 epochs)...")
    t0 = time.time()
    model = train_model(train_set, seed=0)
    print(f"  trained in {time.time()-t0:.1f}s")

    # ---- gather training features for Gaussian fit ----
    print("Extracting training features for Gaussian fit...")
    # use a fixed-order DataLoader so labels align with feature batches
    train_x = torch.stack([train_set[i][0] for i in range(len(train_set))])
    train_y = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    train_feats, _ = extract_features(model, train_x)
    print("  feature dims per layer:",
          [tuple(f.shape) for f in train_feats])

    print("Fitting per-class Gaussians (shared covariance) per layer...")
    gauss_params = []
    for k, name in enumerate(LAYER_NAMES):
        means, prec = fit_class_gaussians(train_feats[k], train_y)
        gauss_params.append((means, prec))
        print(f"  layer {name}: means {means.shape}  precision {prec.shape}")

    # free training-feature memory
    del train_feats, train_x

    # ---- test features + Mahalanobis scores ----
    print("Extracting test features...")
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))])
    test_feats, _ = extract_features(model, test_x)

    print("Computing per-layer Mahalanobis scores on test set...")
    per_layer = []
    for k, name in enumerate(LAYER_NAMES):
        means, prec = gauss_params[k]
        s = mahalanobis_score(test_feats[k], means, prec)
        per_layer.append(s)
        print(f"  layer {name}: score mean={s.mean():.3f}  std={s.std():.3f}")
    combined = np.sum(np.stack(per_layer, 1), axis=1)
    print(f"  combined (sum): mean={combined.mean():.3f}  std={combined.std():.3f}")

    # restrict to test samples model classifies correctly
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512].to(DEVICE)).argmax(1).cpu())
        preds = torch.cat(preds)
    correct_mask = (preds == test_y).numpy()
    print(f"Model accuracy on test: {correct_mask.mean():.4f}; "
          f"keeping {int(correct_mask.sum())} correctly-classified samples")

    x_c = test_x[correct_mask]
    y_c = test_y[correct_mask]
    per_layer_c = [s[correct_mask] for s in per_layer]
    combined_c = combined[correct_mask]

    # ---- adversarial-vulnerability targets ----
    print("Computing FGSM_flip targets...")
    t0 = time.time()
    fgsm_t = batched_attack(fgsm_flip, model, x_c, y_c).numpy()
    print(f"  fgsm done ({time.time()-t0:.1f}s)  flip rate = {fgsm_t.mean():.3f}")

    print("Computing PGD_flip targets...")
    t0 = time.time()
    pgd_t = batched_attack(pgd_flip, model, x_c, y_c).numpy()
    print(f"  pgd done  ({time.time()-t0:.1f}s)  flip rate = {pgd_t.mean():.3f}")

    print("Computing min_eps_to_flip (FGSM binary search)...")
    t0 = time.time()
    me = []
    for i in range(0, x_c.size(0), 512):
        me.append(min_eps_to_flip(model,
                                  x_c[i:i+512].to(DEVICE),
                                  y_c[i:i+512].to(DEVICE)))
    min_eps = torch.cat(me).cpu().numpy()
    print(f"  done ({time.time()-t0:.1f}s)  mean min_eps={min_eps.mean():.4f}")

    # ---- assemble feature matrix ----
    feat_names = [f"mahalanobis_{n}" for n in LAYER_NAMES] + ["combined_mahalanobis"]
    feats_np = np.stack(per_layer_c + [combined_c], axis=1)

    targets = [fgsm_t, pgd_t, min_eps]
    target_names = ["FGSM_flip", "PGD_flip", "min_eps"]
    evaluate(feats_np, feat_names, targets, target_names)

    # ---- focused comparison: combined vs best single layer ----
    print("\n========== Combined vs best single-layer (margin signal) ==========")
    for t_idx, t_name in enumerate(target_names):
        y = targets[t_idx]
        if t_name == "min_eps":
            y = (y < np.median(y)).astype(int)
        else:
            y = y.astype(int)
        if y.std() == 0:
            continue
        aucs = []
        for i, n in enumerate(feat_names):
            a = roc_auc_score(y, feats_np[:, i])
            a = max(a, 1 - a)
            aucs.append((n, a))
        single = aucs[:len(LAYER_NAMES)]
        combined_auc = aucs[-1][1]
        best_single = max(single, key=lambda r: r[1])
        delta = combined_auc - best_single[1]
        print(f"  {t_name:<10}  best_single={best_single[0]} ({best_single[1]:.4f})  "
              f"combined={combined_auc:.4f}  delta={delta:+.4f}")


if __name__ == "__main__":
    main()
