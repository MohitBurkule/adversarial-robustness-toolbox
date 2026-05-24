"""
H100: Per-sample first PGD step at which prediction flips, varied across step-size
choices, reveals per-sample geometric structure.

Setup
-----
- Train a small CNN (matching diagnostic_test.py) on Fashion-MNIST for 10 epochs.
- For each test sample, run PGD with K=20 steps at four step sizes
  alpha in {eps/40, eps/20, eps/10, eps/5}, eps = 15/255.
- Record the *first* step at which the prediction flips away from the clean
  label (per-sample, per-alpha). If no flip occurs within K steps, record K+1
  (right-censored).
- Features (per sample): margin (logit_true - max_other) on clean input,
  mean_pix, std_pix.
- Targets: binary "flipped within K steps" at each alpha.
- Cross-alpha analysis: Spearman rank correlation of step-to-flip across alpha
  pairs; do coarse-alpha step-to-flip values predict fine-alpha step-to-flip
  values (linear regression on the censored integer values, plus AUROC for the
  binary flipped/not flipped target predicted by coarse-alpha step-to-flip)?

This script is self-contained: it trains the model, runs the PGD sweep, and
prints/saves the analysis. DO NOT run from here; this file is code only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
BATCH = 128
EPOCHS = 10
EPS = 15.0 / 255.0
K_STEPS = 20
ALPHAS = {
    "eps_over_40": EPS / 40.0,
    "eps_over_20": EPS / 20.0,
    "eps_over_10": EPS / 10.0,
    "eps_over_5":  EPS / 5.0,
}
ALPHA_ORDER = ["eps_over_40", "eps_over_20", "eps_over_10", "eps_over_5"]

OUT_DIR = Path(__file__).resolve().parent / "h100_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data"))


# ----------------------------------------------------------------------------
# Model: matches diagnostic_test.py
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_cnn(train_set, n_classes=10):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"[train] epoch {ep + 1}/{EPOCHS} done")
    model.eval()
    return model


# ----------------------------------------------------------------------------
# PGD with per-step flip tracking
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(model, x):
    return model(x).argmax(dim=1)


def pgd_step_to_flip(model, x, y_clean, alpha, eps, K):
    """
    Run K steps of L-inf PGD with sign-of-gradient updates, projected to the
    L-inf eps-ball around x and clipped to [0, 1]. Return a tensor of shape
    (N,) with the first step index k (1..K) at which the prediction flipped
    away from y_clean. Samples that never flip receive K + 1 (right-censored).

    No random start: deterministic comparison across alphas.
    """
    model.eval()
    x_orig = x.detach()
    x_adv = x.detach().clone()
    N = x.size(0)
    step_to_flip = torch.full((N,), K + 1, dtype=torch.long, device=x.device)
    flipped = torch.zeros(N, dtype=torch.bool, device=x.device)

    for k in range(1, K + 1):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y_clean)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            x_adv = x_adv.clamp(0.0, 1.0)

            preds = model(x_adv).argmax(dim=1)
            newly_flipped = (preds != y_clean) & (~flipped)
            step_to_flip[newly_flipped] = k
            flipped = flipped | newly_flipped

    return step_to_flip.detach().cpu()


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------
@torch.no_grad()
def compute_features(model, x, y):
    """Return (margin, mean_pix, std_pix) per sample on the clean inputs."""
    model.eval()
    logits = model(x)
    true_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    logits_masked = logits.clone()
    logits_masked.scatter_(1, y.unsqueeze(1), float("-inf"))
    max_other = logits_masked.max(dim=1).values
    margin = (true_logit - max_other).detach().cpu().numpy()

    x_flat = x.view(x.size(0), -1)
    mean_pix = x_flat.mean(dim=1).detach().cpu().numpy()
    std_pix = x_flat.std(dim=1).detach().cpu().numpy()
    return margin, mean_pix, std_pix


# ----------------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------------
def run_pgd_sweep(model, test_x, test_y, batch=256):
    """Run PGD step-to-flip per alpha over the full test set in batches."""
    N = test_x.size(0)
    results = {name: torch.empty(N, dtype=torch.long) for name in ALPHA_ORDER}
    for name in ALPHA_ORDER:
        alpha = ALPHAS[name]
        print(f"[pgd] alpha={name} ({alpha:.6f})")
        for i in range(0, N, batch):
            xb = test_x[i:i + batch].to(DEVICE)
            yb = test_y[i:i + batch].to(DEVICE)
            stf = pgd_step_to_flip(model, xb, yb, alpha=alpha, eps=EPS, K=K_STEPS)
            results[name][i:i + batch] = stf
    return {name: results[name].numpy() for name in ALPHA_ORDER}


def analyze(stf_by_alpha, features):
    """
    Cross-alpha analyses:
      - Spearman rank correlation of step-to-flip across each alpha pair.
      - Binary 'flipped within K' target per alpha, fraction flipped.
      - Predict fine-alpha step-to-flip from coarse-alpha step-to-flip
        (LinearRegression on integer step values, R^2).
      - Predict fine-alpha binary flip from coarse-alpha step-to-flip
        (LogisticRegression, AUROC).
      - Predict binary flip per alpha from clean-input features
        (margin, mean_pix, std_pix) via LogisticRegression, AUROC.
    """
    margin, mean_pix, std_pix = features
    feat_mat = np.stack([margin, mean_pix, std_pix], axis=1)

    out = {"alphas": {k: float(v) for k, v in ALPHAS.items()},
           "eps": EPS, "K": K_STEPS, "n_samples": int(feat_mat.shape[0])}

    # Per-alpha fraction flipped and feature-based AUROC
    out["per_alpha"] = {}
    for name in ALPHA_ORDER:
        stf = stf_by_alpha[name]
        flipped = (stf <= K_STEPS).astype(int)
        entry = {
            "fraction_flipped": float(flipped.mean()),
            "step_to_flip_mean_censored": float(stf.mean()),
            "step_to_flip_median_censored": float(np.median(stf)),
        }
        if 0 < flipped.sum() < flipped.size:
            try:
                lr = LogisticRegression(max_iter=1000).fit(feat_mat, flipped)
                p = lr.predict_proba(feat_mat)[:, 1]
                entry["features_auroc"] = float(roc_auc_score(flipped, p))
                entry["features_coef"] = {
                    "margin": float(lr.coef_[0, 0]),
                    "mean_pix": float(lr.coef_[0, 1]),
                    "std_pix": float(lr.coef_[0, 2]),
                    "intercept": float(lr.intercept_[0]),
                }
            except Exception as e:
                entry["features_auroc_error"] = str(e)
        out["per_alpha"][name] = entry

    # Cross-alpha Spearman rank correlation matrix
    out["spearman"] = {}
    for i, a in enumerate(ALPHA_ORDER):
        for b in ALPHA_ORDER[i + 1:]:
            rho, pval = spearmanr(stf_by_alpha[a], stf_by_alpha[b])
            out["spearman"][f"{a}__vs__{b}"] = {
                "rho": float(rho), "pvalue": float(pval),
            }

    # Coarse -> fine prediction (coarse = larger alpha, fewer steps to flip).
    # ALPHA_ORDER is sorted finest-to-coarsest by alpha magnitude (eps/40 is
    # the finest step). Coarser alpha = larger step.
    # We sweep all (coarse, fine) pairs where coarse alpha > fine alpha.
    out["coarse_to_fine"] = {}
    for i_coarse, coarse in enumerate(ALPHA_ORDER):
        for fine in ALPHA_ORDER[:i_coarse]:
            x_coarse = stf_by_alpha[coarse].reshape(-1, 1).astype(float)
            y_fine = stf_by_alpha[fine].astype(float)
            entry = {}
            try:
                reg = LinearRegression().fit(x_coarse, y_fine)
                pred = reg.predict(x_coarse)
                entry["linreg_r2"] = float(r2_score(y_fine, pred))
                entry["linreg_slope"] = float(reg.coef_[0])
                entry["linreg_intercept"] = float(reg.intercept_)
            except Exception as e:
                entry["linreg_error"] = str(e)

            y_fine_bin = (stf_by_alpha[fine] <= K_STEPS).astype(int)
            if 0 < y_fine_bin.sum() < y_fine_bin.size:
                try:
                    clf = LogisticRegression(max_iter=1000).fit(
                        x_coarse, y_fine_bin
                    )
                    p = clf.predict_proba(x_coarse)[:, 1]
                    entry["binary_auroc"] = float(roc_auc_score(y_fine_bin, p))
                except Exception as e:
                    entry["binary_auroc_error"] = str(e)
            out["coarse_to_fine"][f"coarse={coarse}__fine={fine}"] = entry

    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tfm = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True,  download=True, transform=tfm)
    test_set  = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tfm)

    model = train_cnn(train_set)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))])

    # Restrict to samples the model classifies correctly on clean input, so
    # "step-to-flip" is well-defined as flipping away from the *correct* class.
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i + 512].to(DEVICE)).argmax(dim=1).cpu())
        preds = torch.cat(preds)
    correct_mask = (preds == test_y)
    print(f"[info] clean accuracy: {correct_mask.float().mean().item():.4f}")
    test_x = test_x[correct_mask]
    test_y = test_y[correct_mask]
    print(f"[info] using {test_x.size(0)} correctly-classified test samples")

    # PGD sweep
    stf_by_alpha = run_pgd_sweep(model, test_x, test_y)

    # Features on clean inputs (batched)
    margins, means, stds = [], [], []
    for i in range(0, test_x.size(0), 512):
        xb = test_x[i:i + 512].to(DEVICE)
        yb = test_y[i:i + 512].to(DEVICE)
        m, mu, sd = compute_features(model, xb, yb)
        margins.append(m); means.append(mu); stds.append(sd)
    features = (
        np.concatenate(margins),
        np.concatenate(means),
        np.concatenate(stds),
    )

    # Analysis
    summary = analyze(stf_by_alpha, features)

    # Persist raw arrays for later inspection
    np.savez(
        OUT_DIR / "h100_raw.npz",
        margin=features[0], mean_pix=features[1], std_pix=features[2],
        **{f"stf_{name}": stf_by_alpha[name] for name in ALPHA_ORDER},
        labels=test_y.numpy(),
    )

    with open(OUT_DIR / "h100_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[done] outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
