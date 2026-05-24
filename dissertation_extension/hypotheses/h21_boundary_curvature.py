"""
H21: Per-sample decision-boundary curvature estimate predicts adversarial vulnerability.

Test: does the input-Hessian trace (Hutchinson estimator, K=10 Rademacher probes)
on clean test samples predict whether the sample flips under FGSM/PGD?

Background: Moosavi-Dezfooli et al. (CVPR 2019), "Robustness via Curvature
Regularization", show adversarial training reduces decision-boundary curvature.
Here we ask whether *clean-sample* curvature is a per-sample vulnerability proxy.

Outputs: stdout-printed univariate AUROCs + Spearman, and a small multivariate
ablation testing whether trace(H_xx) adds over ||grad_x L||_2 and victim margin.

Self-contained: torchvision Fashion-MNIST in /tmp/data, single CUDA device.
"""

from __future__ import annotations

import os
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0
HUTCH_K = 10
N_SAMPLES_CURV = 2000  # sub-sample test set for Hessian-trace estimation (cost)
SEED = 0


# ---------------------------------------------------------------------------
# Model (matches diagnostic_test.CNN)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_victim(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        running = 0.0
        n = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  epoch {ep+1}/{EPOCHS}  loss={running/n:.4f}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = (x + eps * grad.sign()).clamp(0.0, 1.0).detach()
    return x_adv


def pgd(model, x, y, eps, alpha, steps):
    x0 = x.clone().detach()
    # random start in eps ball
    x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
        x_adv = x_adv.clamp(0.0, 1.0).detach()
    return x_adv


def attack_flipped(model, x, y, attack_fn, batch=256):
    """Return bool tensor: prediction != y on adversarial example."""
    flips = []
    for i in range(0, x.size(0), batch):
        xb = x[i:i + batch].to(DEVICE)
        yb = y[i:i + batch].to(DEVICE)
        x_adv = attack_fn(model, xb, yb)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        flips.append((pred != yb).cpu())
    return torch.cat(flips)


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=15, batch=256):
    """Per-sample binary search: minimal eps under which FGSM flips."""
    out = torch.zeros(x.size(0))
    for i in range(0, x.size(0), batch):
        xb = x[i:i + batch].to(DEVICE)
        yb = y[i:i + batch].to(DEVICE)
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        # cache sign gradient at clean x (FGSM uses sign of grad at x_clean)
        xc = xb.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(xc), yb)
        g = torch.autograd.grad(loss, xc)[0].sign().detach()
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            x_adv = (xb + mid.view(-1, 1, 1, 1) * g).clamp(0.0, 1.0)
            with torch.no_grad():
                flipped = model(x_adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        out[i:i + xb.size(0)] = hi.cpu()
    return out


# ---------------------------------------------------------------------------
# Curvature: Hutchinson trace of input Hessian of CE loss
# ---------------------------------------------------------------------------
def hutchinson_input_hessian_trace(model, x, y, K=HUTCH_K, batch=32):
    """
    For each sample i, estimate trace(H_i) where H_i = d^2 L / d x_i d x_i^T,
    L = CE loss(model(x_i), y_i). Hutchinson: E[v^T H v] for v ~ Rademacher.

    Implementation: per-sample sum-of-losses, take grad of (g . v) wrt x to get
    Hv, then sum elementwise with v to get v^T H v -- per sample because each
    sample's loss only depends on its own input.
    """
    model.eval()
    N = x.size(0)
    trace_est = torch.zeros(N)
    grad_norm = torch.zeros(N)

    for i in range(0, N, batch):
        xb = x[i:i + batch].to(DEVICE).clone().detach().requires_grad_(True)
        yb = y[i:i + batch].to(DEVICE)

        logits = model(xb)
        # per-sample losses; summing is fine for grad since samples are independent
        losses = F.cross_entropy(logits, yb, reduction="sum")
        grads = torch.autograd.grad(losses, xb, create_graph=True)[0]  # [B,1,28,28]

        # input gradient L2 norm (per sample)
        gn = grads.detach().flatten(1).norm(dim=1).cpu()
        grad_norm[i:i + xb.size(0)] = gn

        acc = torch.zeros(xb.size(0), device=DEVICE)
        for k in range(K):
            v = torch.randint(0, 2, xb.shape, device=DEVICE, dtype=xb.dtype)
            v = v.mul_(2).sub_(1)  # Rademacher {-1,+1}
            gv = (grads * v).flatten(1).sum(1).sum()
            Hv = torch.autograd.grad(gv, xb, retain_graph=(k < K - 1))[0]
            acc = acc + (Hv * v).flatten(1).sum(1).detach()
        trace_est[i:i + xb.size(0)] = (acc / K).cpu()

        # free graph
        del grads, logits, losses
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    return trace_est, grad_norm


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def report_univariate(features: dict, targets: dict):
    print("\n=== Univariate AUROC (higher score -> more vulnerable expected) ===")
    print(f"{'feature':>20s} | " + " | ".join(f"{t:>18s}" for t in targets))
    for fname, fvals in features.items():
        row = [fname]
        for tname, tvals in targets.items():
            if tvals.dtype == bool or set(np.unique(tvals)).issubset({0, 1}):
                a = auroc(fvals, tvals.astype(int))
                row.append(f"AUROC={a:.3f}")
            else:
                # continuous target -> Spearman as primary, also negate-AUROC undefined
                rho, _ = spearmanr(fvals, tvals)
                row.append(f"rho={rho:+.3f}")
        print(f"{row[0]:>20s} | " + " | ".join(f"{c:>18s}" for c in row[1:]))


def multivariate_ablation(features: dict, target: np.ndarray, name: str):
    """Logistic regression: does Hessian trace add over [grad_norm, margin]?"""
    print(f"\n=== Multivariate ablation on target: {name} ===")
    keys_base = ["grad_norm", "victim_margin"]
    keys_full = keys_base + ["hess_trace"]

    def fit_auc(keys):
        X = np.stack([features[k] for k in keys], axis=1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, target)
        return auroc(clf.predict_proba(X)[:, 1], target), dict(zip(keys, clf.coef_[0].tolist()))

    a_base, c_base = fit_auc(keys_base)
    a_full, c_full = fit_auc(keys_full)
    print(f"  baseline (grad_norm + margin): AUROC={a_base:.3f} coefs={c_base}")
    print(f"  + hess_trace                  : AUROC={a_full:.3f} coefs={c_full}")
    print(f"  delta AUROC                   : {a_full - a_base:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print("Training victim CNN on Fashion-MNIST...")
    model = train_victim(train_set)

    # Build test tensors
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))])

    # Clean accuracy + restrict to correctly-classified samples (standard practice
    # for flip-probability analysis: flip is undefined if model already wrong).
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i + 512].to(DEVICE)).argmax(1).cpu())
        preds = torch.cat(preds)
    correct_mask = (preds == test_y)
    print(f"Clean accuracy: {correct_mask.float().mean().item():.4f}")

    idx_correct = torch.where(correct_mask)[0]
    g = torch.Generator().manual_seed(SEED)
    perm = idx_correct[torch.randperm(idx_correct.numel(), generator=g)]
    sub_idx = perm[:N_SAMPLES_CURV]

    x_sub = test_x[sub_idx]
    y_sub = test_y[sub_idx]
    print(f"Curvature-analysis sub-sample: {x_sub.size(0)} correctly-classified examples.")

    # --- features ---
    print("Estimating input-Hessian trace (Hutchinson K=%d)..." % HUTCH_K)
    hess_trace, grad_norm = hutchinson_input_hessian_trace(model, x_sub, y_sub)

    # victim margin (logit_top - logit_2nd) on clean samples
    with torch.no_grad():
        margins = []
        for i in range(0, x_sub.size(0), 512):
            lg = model(x_sub[i:i + 512].to(DEVICE))
            srt, _ = lg.sort(dim=1, descending=True)
            margins.append((srt[:, 0] - srt[:, 1]).cpu())
        margin = torch.cat(margins)

    mean_pix = x_sub.flatten(1).mean(1)
    std_pix = x_sub.flatten(1).std(1)

    # --- targets ---
    print("Running FGSM attack...")
    flipped_fgsm = attack_flipped(
        model, x_sub, y_sub, lambda m, x, y: fgsm(m, x, y, EPS_TEST)
    )
    print("Running PGD attack...")
    flipped_pgd = attack_flipped(
        model, x_sub, y_sub,
        lambda m, x, y: pgd(m, x, y, EPS_TEST, PGD_ALPHA, PGD_STEPS),
    )
    print("Binary-searching FGSM min-eps...")
    fgsm_meps = fgsm_min_eps(model, x_sub, y_sub)

    features = {
        "hess_trace":    hess_trace.numpy(),
        "abs_hess_trace": np.abs(hess_trace.numpy()),
        "grad_norm":     grad_norm.numpy(),
        "victim_margin": margin.numpy(),
        "mean_pix":      mean_pix.numpy(),
        "std_pix":       std_pix.numpy(),
    }
    targets_binary = {
        "flipped_FGSM": flipped_fgsm.numpy().astype(int),
        "flipped_PGD":  flipped_pgd.numpy().astype(int),
    }
    targets_cont = {
        "FGSM_min_eps": fgsm_meps.numpy(),
    }

    # --- report ---
    print(f"\nAttack success rates: FGSM={flipped_fgsm.float().mean():.3f}  "
          f"PGD={flipped_pgd.float().mean():.3f}")
    print(f"FGSM min-eps  mean={fgsm_meps.mean():.4f}  median={fgsm_meps.median():.4f}")

    report_univariate(features, {**targets_binary, **targets_cont})

    # multivariate
    for tname, tvals in targets_binary.items():
        multivariate_ablation(features, tvals, tname)

    # Save raw arrays for downstream plots
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h21_boundary_curvature.npz"
    np.savez(
        out_path,
        **features,
        **{f"y_{k}": v for k, v in targets_binary.items()},
        **{f"y_{k}": v for k, v in targets_cont.items()},
        sub_idx=sub_idx.numpy(),
    )
    print(f"\nSaved raw arrays -> {out_path}")

    # Final headline
    print("\n=== Headline (H21) ===")
    print("If trace(H_xx) is a per-sample vulnerability proxy, expect:")
    print("  - positive Spearman with FGSM/PGD flip indicator")
    print("  - AUROC > 0.5 for predicting flip")
    print("  - negative correlation with FGSM_min_eps (lower-eps flip <=> higher curvature)")
    print("  - non-negligible multivariate delta over grad_norm + margin baseline")


if __name__ == "__main__":
    main()
