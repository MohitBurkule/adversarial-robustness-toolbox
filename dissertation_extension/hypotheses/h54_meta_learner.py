"""
Hypothesis H54: A small meta-learner (XGBoost / gradient-boosting / random forest)
trained on all known per-sample features predicts adversarial vulnerability much
better than any single feature. Feature-importance rankings reveal which features
matter most.

Pipeline:
  1. Train small CNN on Fashion-MNIST (10 epochs).
  2. Compute ~20 per-sample features.
  3. Compute three vulnerability targets:
        flipped_FGSM  (eps = 15/255)
        flipped_PGD   (eps = 15/255, 10 steps)
        FGSM_min_eps  (regression target, also binarised at median)
  4. 5-fold CV with XGBoost (fallback: sklearn GradientBoostingClassifier),
     compare to logistic regression on the same features.
  5. Rank features via SHAP -> built-in importance -> permutation importance.

Self-contained. Run with the venv at
/mnt/1tbone/trashy/adversarial-robustness-toolbox/.venv .
"""

import os
import io
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

from PIL import Image
from scipy import stats as sstats
from scipy import ndimage as ndi

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS / 4.0
SEED = 0


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
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


def train_cnn(train_set):
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
        print(f"  epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# --------------------------------------------------------------------------
# Attacks
# --------------------------------------------------------------------------
def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    g = x.grad.sign().detach()
    return (x.detach() + eps * g).clamp(0, 1)


def pgd(model, x, y, eps, alpha, steps):
    x0 = x.clone().detach()
    delta = torch.empty_like(x0).uniform_(-eps, eps)
    xa = (x0 + delta).clamp(0, 1).detach()
    for _ in range(steps):
        xa.requires_grad_(True)
        F.cross_entropy(model(xa), y).backward()
        g = xa.grad.sign().detach()
        xa = (xa.detach() + alpha * g).clamp(x0 - eps, x0 + eps).clamp(0, 1)
    return xa


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=12):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
    N = x.size(0)
    lo = torch.zeros(N, device=x.device)
    hi = torch.full((N,), eps_max, device=x.device)
    # ensure eps_max flips
    with torch.no_grad():
        x_adv = fgsm(model, x, y, eps_max)
        flips_at_max = (model(x_adv).argmax(1) != y)
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        x_adv = fgsm_per_sample(model, x, y, mid)
        with torch.no_grad():
            flipped = (model(x_adv).argmax(1) != y)
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    out = hi.clone()
    out[~flips_at_max] = eps_max  # mark non-flippable as eps_max
    return out


def fgsm_per_sample(model, x, y, eps_vec):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    g = x.grad.sign().detach()
    return (x.detach() + eps_vec.view(-1, 1, 1, 1) * g).clamp(0, 1)


# --------------------------------------------------------------------------
# Per-sample features
# --------------------------------------------------------------------------
SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
SOBEL_Y = SOBEL_X.t().clone()
LAPLACE = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)


def jpeg_recon_err(arr_uint8, q):
    """arr_uint8: HxW uint8 numpy array."""
    img = Image.fromarray(arr_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    rec = np.array(Image.open(buf), dtype=np.float32)
    return float(np.mean(np.abs(rec - arr_uint8.astype(np.float32))) / 255.0)


def compute_features(model, x_all, y_all):
    """x_all: (N,1,28,28) on DEVICE in [0,1]. Returns numpy (N, n_feats) and names."""
    N = x_all.size(0)
    names = [
        "victim_margin", "mean_pix", "std_pix", "skew_pix", "kurt_pix",
        "MAD_pix", "entropy_pix", "range_pix", "IQR_pix",
        "sobel_mean", "sobel_std", "laplacian_abs_mean",
        "OTI_simple", "OAR_simple",
        "fourier_hf_25", "fourier_hf_50",
        "jpeg_q75", "jpeg_q50", "input_grad_norm",
    ]
    F_feat = np.zeros((N, len(names)), dtype=np.float32)

    # -- victim margin (top1 - top2) ---------------------------------------
    margins = []
    with torch.no_grad():
        for i in range(0, N, 512):
            logits = model(x_all[i:i + 512])
            s, _ = logits.sort(1, descending=True)
            margins.append((s[:, 0] - s[:, 1]).cpu())
    F_feat[:, 0] = torch.cat(margins).numpy()

    # -- input grad norm ---------------------------------------------------
    grad_norms = np.zeros(N, dtype=np.float32)
    for i in range(0, N, 256):
        xb = x_all[i:i + 256].clone().detach().requires_grad_(True)
        yb = y_all[i:i + 256]
        F.cross_entropy(model(xb), yb).backward()
        g = xb.grad.detach()
        grad_norms[i:i + g.size(0)] = g.flatten(1).norm(dim=1).cpu().numpy()
    F_feat[:, 18] = grad_norms

    # -- sobel / laplacian via conv2d --------------------------------------
    sx = SOBEL_X.to(DEVICE).view(1, 1, 3, 3)
    sy = SOBEL_Y.to(DEVICE).view(1, 1, 3, 3)
    lap = LAPLACE.to(DEVICE).view(1, 1, 3, 3)
    sob_mean = np.zeros(N, dtype=np.float32)
    sob_std = np.zeros(N, dtype=np.float32)
    lap_mean = np.zeros(N, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, 512):
            xb = x_all[i:i + 512]
            gx = F.conv2d(xb, sx, padding=1)
            gy = F.conv2d(xb, sy, padding=1)
            mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
            sob_mean[i:i + xb.size(0)] = mag.flatten(1).mean(1).cpu().numpy()
            sob_std[i:i + xb.size(0)] = mag.flatten(1).std(1).cpu().numpy()
            lp = F.conv2d(xb, lap, padding=1).abs()
            lap_mean[i:i + xb.size(0)] = lp.flatten(1).mean(1).cpu().numpy()
    F_feat[:, 9] = sob_mean
    F_feat[:, 10] = sob_std
    F_feat[:, 11] = lap_mean

    # -- per-image stats (CPU loop, vectorised where possible) -------------
    x_np = x_all.cpu().numpy().reshape(N, -1)  # in [0,1]
    F_feat[:, 1] = x_np.mean(1)
    F_feat[:, 2] = x_np.std(1)
    F_feat[:, 3] = sstats.skew(x_np, axis=1)
    F_feat[:, 4] = sstats.kurtosis(x_np, axis=1)
    F_feat[:, 5] = np.mean(np.abs(x_np - x_np.mean(1, keepdims=True)), axis=1)
    # entropy from 32-bin histogram
    ent = np.zeros(N, dtype=np.float32)
    for i in range(N):
        h, _ = np.histogram(x_np[i], bins=32, range=(0.0, 1.0), density=False)
        p = h.astype(np.float64) / max(h.sum(), 1)
        p = p[p > 0]
        ent[i] = float(-(p * np.log2(p)).sum())
    F_feat[:, 6] = ent
    F_feat[:, 7] = x_np.max(1) - x_np.min(1)
    q75 = np.quantile(x_np, 0.75, axis=1)
    q25 = np.quantile(x_np, 0.25, axis=1)
    F_feat[:, 8] = q75 - q25

    # -- OTI / OAR (simple proxies) ----------------------------------------
    # OTI_simple: object/background contrast = |mean(fg) - mean(bg)| with Otsu-ish
    #   threshold at 0.5 (Fashion-MNIST is centered). OAR_simple: fraction of
    #   pixels > 0.1 (object area ratio).
    fg_mask = (x_np > 0.5)
    bg_mask = ~fg_mask
    oti = np.zeros(N, dtype=np.float32)
    for i in range(N):
        if fg_mask[i].any() and bg_mask[i].any():
            oti[i] = float(abs(x_np[i][fg_mask[i]].mean() - x_np[i][bg_mask[i]].mean()))
    F_feat[:, 12] = oti
    F_feat[:, 13] = (x_np > 0.1).mean(1)

    # -- Fourier high-frequency energy fractions ---------------------------
    x_img = x_np.reshape(N, 28, 28)
    fft = np.fft.fftshift(np.fft.fft2(x_img), axes=(-2, -1))
    power = np.abs(fft) ** 2
    cy, cx = 14, 14
    yy, xx = np.mgrid[0:28, 0:28]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    total = power.sum((1, 2)) + 1e-12
    m25 = (rr > (28 * 0.25)).astype(np.float32)
    m50 = (rr > (28 * 0.50)).astype(np.float32)
    F_feat[:, 14] = (power * m25).sum((1, 2)) / total
    F_feat[:, 15] = (power * m50).sum((1, 2)) / total

    # -- JPEG reconstruction error (q=75, q=50) ----------------------------
    x_u8 = (x_img * 255.0).clip(0, 255).astype(np.uint8)
    jq75 = np.zeros(N, dtype=np.float32)
    jq50 = np.zeros(N, dtype=np.float32)
    for i in range(N):
        jq75[i] = jpeg_recon_err(x_u8[i], 75)
        jq50[i] = jpeg_recon_err(x_u8[i], 50)
    F_feat[:, 16] = jq75
    F_feat[:, 17] = jq50

    return F_feat, names


# --------------------------------------------------------------------------
# Vulnerability targets
# --------------------------------------------------------------------------
def compute_targets(model, x_all, y_all):
    N = x_all.size(0)
    flipped_fgsm = np.zeros(N, dtype=np.int64)
    flipped_pgd = np.zeros(N, dtype=np.int64)
    min_eps = np.zeros(N, dtype=np.float32)
    for i in range(0, N, 256):
        xb = x_all[i:i + 256]
        yb = y_all[i:i + 256]
        x_fgsm = fgsm(model, xb, yb, EPS)
        with torch.no_grad():
            flipped_fgsm[i:i + xb.size(0)] = (model(x_fgsm).argmax(1) != yb).cpu().numpy()
        x_pgd = pgd(model, xb, yb, EPS, PGD_ALPHA, PGD_STEPS)
        with torch.no_grad():
            flipped_pgd[i:i + xb.size(0)] = (model(x_pgd).argmax(1) != yb).cpu().numpy()
        eps_b = fgsm_min_eps(model, xb, yb)
        min_eps[i:i + xb.size(0)] = eps_b.cpu().numpy()
    return flipped_fgsm, flipped_pgd, min_eps


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def cv_auroc_xgb_or_gbm(X, y, n_splits=5):
    """Returns (mean_auc, std_auc, fitted_model_on_full_data, used_lib)."""
    try:
        import xgboost as xgb
        used = "xgboost"
    except Exception:
        xgb = None
        used = "sklearn-GBM"

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        if xgb is not None:
            clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                use_label_encoder=False, eval_metric="logloss",
                tree_method="hist", n_jobs=4, random_state=SEED,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                random_state=SEED,
            )
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))

    # final model trained on all data for importance extraction
    if xgb is not None:
        final_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            use_label_encoder=False, eval_metric="logloss",
            tree_method="hist", n_jobs=4, random_state=SEED,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        final_clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=SEED,
        )
    final_clf.fit(X, y)
    return float(np.mean(aucs)), float(np.std(aucs)), final_clf, used


def cv_auroc_logreg(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def single_feature_auroc(X, y, names):
    """Best per-feature AUROC (treat each feature as its own score / -score)."""
    best = []
    for j, nm in enumerate(names):
        s = X[:, j]
        a = roc_auc_score(y, s)
        a = max(a, 1.0 - a)  # allow inverted
        best.append((nm, a))
    best.sort(key=lambda t: -t[1])
    return best


def feature_importances(clf, X, y, names):
    # Try SHAP first
    try:
        import shap
        try:
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X)
            if isinstance(sv, list):  # multiclass list
                sv = sv[1] if len(sv) > 1 else sv[0]
            imp = np.abs(sv).mean(0)
            return list(zip(names, imp.tolist())), "shap"
        except Exception:
            pass
    except Exception:
        pass
    # Built-in
    if hasattr(clf, "feature_importances_"):
        return list(zip(names, clf.feature_importances_.tolist())), "builtin"
    # Permutation
    r = permutation_importance(clf, X, y, n_repeats=5, random_state=SEED, n_jobs=2)
    return list(zip(names, r.importances_mean.tolist())), "permutation"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("[H54] Loading Fashion-MNIST ...")
    tf = T.Compose([T.ToTensor()])
    train_set = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=True, download=True, transform=tf)
    test_set = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=False, download=True, transform=tf)

    print(f"[H54] Training CNN on {DEVICE} for {EPOCHS} epochs ...")
    model = train_cnn(train_set)

    print("[H54] Stacking test set ...")
    x_all = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y_all = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("[H54] Computing per-sample features ...")
    X, names = compute_features(model, x_all, y_all)
    print(f"  features: {len(names)} -> X.shape={X.shape}")

    print("[H54] Computing vulnerability targets ...")
    y_fgsm, y_pgd, eps_min = compute_targets(model, x_all, y_all)
    # binarise min_eps at median
    eps_med = float(np.median(eps_min))
    y_minfgsm = (eps_min <= eps_med).astype(np.int64)
    print(f"  base rate flipped_FGSM  = {y_fgsm.mean():.3f}")
    print(f"  base rate flipped_PGD   = {y_pgd.mean():.3f}")
    print(f"  median min-eps (FGSM)   = {eps_med:.4f}; y_minfgsm rate = {y_minfgsm.mean():.3f}")

    # restrict to samples the victim classifies correctly (vulnerability is
    # only well-defined there)
    with torch.no_grad():
        clean_pred = []
        for i in range(0, x_all.size(0), 512):
            clean_pred.append(model(x_all[i:i + 512]).argmax(1).cpu())
        clean_pred = torch.cat(clean_pred).numpy()
    keep = (clean_pred == y_all.cpu().numpy())
    print(f"  keeping {keep.sum()}/{len(keep)} correctly-classified samples")
    X = X[keep]
    y_fgsm = y_fgsm[keep]
    y_pgd = y_pgd[keep]
    y_minfgsm = y_minfgsm[keep]

    # replace NaN/Inf defensively
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    targets = {
        "flipped_FGSM": y_fgsm,
        "flipped_PGD": y_pgd,
        "FGSM_min_eps (binarised at median)": y_minfgsm,
    }

    print("\n========== H54 RESULTS ==========")
    summary = {}
    for tname, y in targets.items():
        if y.sum() == 0 or y.sum() == len(y):
            print(f"\n[{tname}] degenerate target (all {y.mean():.2f}); skipping.")
            continue
        print(f"\n--- Target: {tname}  (rate={y.mean():.3f}) ---")

        mean_lr, std_lr = cv_auroc_logreg(X, y)
        print(f"  Logistic regression CV AUROC = {mean_lr:.4f} +- {std_lr:.4f}")

        mean_g, std_g, clf, used = cv_auroc_xgb_or_gbm(X, y)
        print(f"  {used} CV AUROC            = {mean_g:.4f} +- {std_g:.4f}")

        single = single_feature_auroc(X, y, names)
        print(f"  Best single-feature AUROC   = {single[0][1]:.4f} ({single[0][0]})")
        print(f"  Top-5 single-feature AUROCs:")
        for nm, a in single[:5]:
            print(f"    {nm:22s}  {a:.4f}")

        imp, kind = feature_importances(clf, X, y, names)
        imp.sort(key=lambda t: -t[1])
        print(f"  Feature importance ({kind}), top-10:")
        for nm, v in imp[:10]:
            print(f"    {nm:22s}  {v:.5f}")

        summary[tname] = {
            "logreg_auc": mean_lr,
            "meta_auc": mean_g,
            "best_single_auc": single[0][1],
            "best_single_name": single[0][0],
            "uplift_vs_single": mean_g - single[0][1],
            "uplift_vs_logreg": mean_g - mean_lr,
        }

    print("\n========== H54 SUMMARY ==========")
    for tname, s in summary.items():
        print(f"[{tname}]  meta={s['meta_auc']:.4f}  "
              f"logreg={s['logreg_auc']:.4f}  "
              f"best_single={s['best_single_auc']:.4f} ({s['best_single_name']})  "
              f"Δ_meta-single={s['uplift_vs_single']:+.4f}  "
              f"Δ_meta-logreg={s['uplift_vs_logreg']:+.4f}")
    print("\nVerdict on H54: meta-learner is supported if Δ_meta-single > 0 "
          "consistently across targets, indicating non-linear feature combinations.")


if __name__ == "__main__":
    main()
