"""
H259 — Perturbation Detectability via Pixel Statistics
Hypothesis: Adversarial examples leave detectable statistical fingerprints.
A simple binary classifier trained on pixel statistics (mean, std, min, max,
L2 norm of delta, gradient norm) achieves high AUROC on clean vs adversarial detection.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0; EPS = 0.1; N_EVAL = 300
META = {"channels": 1, "size": 28, "n_classes": 10}
torch.manual_seed(SEED); np.random.seed(SEED)

Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = C.build_model("cnn", META, width=32, seed=SEED)
C.train_model(model, Xtr, Ytr, epochs=10)
model.eval()


def compute_gradient_norm(mdl: nn.Module, X: torch.Tensor, Y: torch.Tensor,
                           batch: int = 64) -> np.ndarray:
    """Compute input-gradient L2 norm per sample."""
    grad_norms = []
    for start in range(0, len(X), batch):
        xb = X[start:start + batch].clone().requires_grad_(True)
        yb = Y[start:start + batch]
        logits = mdl(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        gn = xb.grad.data.view(len(xb), -1).norm(dim=1)
        grad_norms.append(gn.detach().cpu())
    return torch.cat(grad_norms).cpu().numpy()


def pixel_features(X: torch.Tensor) -> np.ndarray:
    """
    Compute pixel-level statistics per sample.
    Returns (N, 5): [mean, std, min, max, l2_norm]
    """
    Xf = X.view(len(X), -1)  # (N, D)
    feats = torch.stack([
        Xf.mean(dim=1),
        Xf.std(dim=1),
        Xf.min(dim=1).values,
        Xf.max(dim=1).values,
        Xf.norm(dim=1),
    ], dim=1)
    return feats.cpu().numpy()


def main():
    t0 = time.time()
    print("=" * 60)
    print("H259 — Perturbation Detectability via Pixel Statistics")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Section 1: generate adversarial examples (FGSM and PGD)
    # ------------------------------------------------------------------ #
    print("\n[1] Generating adversarial examples ...")
    for p in model.parameters():
        p.requires_grad_(True)
    Xadv_fgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    for p in model.parameters():
        p.requires_grad_(True)
    Xadv_pgd  = C.pgd(model,  Xte, Yte, eps=EPS, steps=10, alpha=0.01)
    model.eval()

    with torch.no_grad():
        logits_c, acc_c = C.logits_and_acc(model, Xte, Yte)
        _, acc_f = C.logits_and_acc(model, Xadv_fgsm, Yte)
        _, acc_p = C.logits_and_acc(model, Xadv_pgd, Yte)
    print(f"  Clean acc:    {acc_c:.4f}")
    print(f"  FGSM acc:     {acc_f:.4f}  (ASR ~ {1 - acc_f:.4f})")
    print(f"  PGD  acc:     {acc_p:.4f}  (ASR ~ {1 - acc_p:.4f})")

    # ------------------------------------------------------------------ #
    # Section 2: compute pixel features
    # ------------------------------------------------------------------ #
    print("\n[2] Computing pixel statistics features ...")
    feat_clean     = pixel_features(Xte)          # (N, 5)
    feat_adv_fgsm  = pixel_features(Xadv_fgsm)    # (N, 5)
    feat_adv_pgd   = pixel_features(Xadv_pgd)     # (N, 5)

    feat_names = ["pixel_mean", "pixel_std", "pixel_min", "pixel_max", "l2_norm"]
    print(f"  Feature matrix shape: {feat_clean.shape}")
    for j, name in enumerate(feat_names):
        delta_fgsm = feat_adv_fgsm[:, j].mean() - feat_clean[:, j].mean()
        delta_pgd  = feat_adv_pgd[:, j].mean()  - feat_clean[:, j].mean()
        print(f"    {name:<14s}: clean_mean={feat_clean[:,j].mean():.5f}  "
              f"fgsm_delta={delta_fgsm:+.5f}  pgd_delta={delta_pgd:+.5f}")

    # ------------------------------------------------------------------ #
    # Section 3: compute delta and gradient norm features
    # ------------------------------------------------------------------ #
    print("\n[3] Computing perturbation delta and gradient norm features ...")
    delta_fgsm = (Xadv_fgsm - Xte).view(N_EVAL, -1)
    delta_pgd  = (Xadv_pgd  - Xte).view(N_EVAL, -1)
    l2_delta_fgsm = delta_fgsm.norm(dim=1).cpu().numpy()
    l2_delta_pgd  = delta_pgd.norm(dim=1).cpu().numpy()
    print(f"  FGSM delta L2: mean={l2_delta_fgsm.mean():.5f}  std={l2_delta_fgsm.std():.5f}")
    print(f"  PGD  delta L2: mean={l2_delta_pgd.mean():.5f}  std={l2_delta_pgd.std():.5f}")

    grad_norms = compute_gradient_norm(model, Xte, Yte)
    print(f"  Gradient norm: mean={grad_norms.mean():.5f}  std={grad_norms.std():.5f}")

    # ------------------------------------------------------------------ #
    # Section 4: build feature matrices for detection
    # ------------------------------------------------------------------ #
    print("\n[4] Building detection datasets ...")
    # Full feature set for adversarial: pixel stats + delta L2 + grad norm
    extra_clean_fgsm = np.stack([np.zeros(N_EVAL), grad_norms], axis=1)  # delta=0 for clean
    extra_adv_fgsm   = np.stack([l2_delta_fgsm,   grad_norms], axis=1)

    extra_clean_pgd  = np.stack([np.zeros(N_EVAL), grad_norms], axis=1)
    extra_adv_pgd    = np.stack([l2_delta_pgd,    grad_norms], axis=1)

    X_fgsm = np.concatenate([
        np.vstack([feat_clean, feat_adv_fgsm]),
        np.vstack([extra_clean_fgsm, extra_adv_fgsm])
    ], axis=1)
    y_fgsm = np.array([0] * N_EVAL + [1] * N_EVAL)

    X_pgd  = np.concatenate([
        np.vstack([feat_clean, feat_adv_pgd]),
        np.vstack([extra_clean_pgd, extra_adv_pgd])
    ], axis=1)
    y_pgd  = np.array([0] * N_EVAL + [1] * N_EVAL)

    print(f"  FGSM detection dataset: X={X_fgsm.shape}, y={y_fgsm.shape}")
    print(f"  PGD  detection dataset: X={X_pgd.shape},  y={y_pgd.shape}")

    # ------------------------------------------------------------------ #
    # Section 5: train logistic regression detector — 5-fold CV AUROC
    # ------------------------------------------------------------------ #
    print("\n[5] Training logistic regression detector (5-fold CV) ...")
    scaler = StandardScaler()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=500, random_state=SEED, C=1.0)

    for attack_name, Xdet, ydet in [("FGSM", X_fgsm, y_fgsm), ("PGD", X_pgd, y_pgd)]:
        Xdet_sc = scaler.fit_transform(Xdet)
        aucs = cross_val_score(clf, Xdet_sc, ydet, cv=cv, scoring="roc_auc")
        print(f"  {attack_name} detector AUROC (5-fold CV): "
              f"{aucs.mean():.4f}  ± {aucs.std():.4f}  per-fold={np.round(aucs, 4)}")

    # ------------------------------------------------------------------ #
    # Section 6: per-feature AUROC (univariate)
    # ------------------------------------------------------------------ #
    print("\n[6] Univariate AUROC per feature (PGD vs clean)")
    print("-" * 55)
    all_feat_names = feat_names + ["delta_l2", "grad_norm"]
    X_all_clean = np.hstack([feat_clean, np.zeros((N_EVAL, 1)), grad_norms[:, None]])
    X_all_adv   = np.hstack([feat_adv_pgd, l2_delta_pgd[:, None], grad_norms[:, None]])
    y_bin       = np.array([0] * N_EVAL + [1] * N_EVAL)

    for j, fname in enumerate(all_feat_names):
        vals_clean = X_all_clean[:, j]
        vals_adv   = X_all_adv[:, j]
        combined   = np.concatenate([vals_clean, vals_adv])
        auc = roc_auc_score(y_bin, combined)
        rho, pval = spearmanr(combined, y_bin)
        print(f"  {fname:<14s}: AUROC={auc:.4f}  rho={rho:+.4f}  p={pval:.4f}")

    print(f"\nDone in {time.time() - t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
