"""
H18: Cross-architecture margin transfer for vulnerability prediction.

We have shown that the victim's own margin dominates per-sample vulnerability
prediction. That is an "oracle" feature: it presumes white-box access to the
victim's logits. The realistic, model-agnostic regime is: the attacker has
some surrogate / proxy model (possibly a very different architecture) and
wants to rank victim test samples by vulnerability.

This script trains three Fashion-MNIST classifiers:
    - VictimSmall : 2-conv + 2-FC CNN (mirrors diagnostic_test.py)
    - ProxyWide   : wider 5-layer CNN (64 -> 128 -> 256 channels)
    - ProxyMLP    : 3-layer MLP (784 -> 256 -> 128 -> 10), qualitatively different

For every Fashion-MNIST test sample we compute each model's margin
(logit_true - max_other). Vulnerability targets are *defined on the
VictimSmall*:
    - flipped_FGSM   at eps = 15/255
    - flipped_PGD    at eps = 15/255, 10 steps
    - min_eps_FGSM   via binary search (continuous target; ranked low-to-high
                     => more vulnerable = smaller eps, so we use -min_eps as
                     "vulnerability score" for AUROC against binarised label)

Univariate AUROC is reported for:
    - victim_margin           (oracle baseline)
    - proxy_wide_margin       (similar architecture family)
    - proxy_mlp_margin        (very different architecture)
    - image_mean, image_std, sobel_energy (model-free image stats baseline)

The headline questions:
    H18a: does proxy_wide_margin's AUROC approach victim_margin's?
    H18b: does proxy_mlp_margin transfer at all, or collapse to baseline?

Run:  python h18_cross_arch_margin.py
Outputs: prints a results table and writes h18_results.json next to this file.
"""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "h18_results.json")


# ----------------------------------------------------------------------------
# Architectures
# ----------------------------------------------------------------------------
class VictimSmall(nn.Module):
    """Mirrors diagnostic_test.py's CNN: 2 conv + 2 FC."""
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


class ProxyWide(nn.Module):
    """Wider 5-layer CNN: 64 -> 128 -> 256 channels, then 2 FC."""
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 128, 3, padding=1)
        self.c3 = nn.Conv2d(128, 256, 3, padding=1)
        # After 3 conv-pool stages: 28 -> 14 -> 7 -> 3
        self.fc1 = nn.Linear(256 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, n)
        self.do = nn.Dropout(0.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)   # -> 14
        x = F.max_pool2d(F.relu(self.c2(x)), 2)   # -> 7
        x = F.max_pool2d(F.relu(self.c3(x)), 2)   # -> 3
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do(x)
        return self.fc2(x)


class ProxyMLP(nn.Module):
    """Pure MLP: flatten -> 256 -> 128 -> 10. No convolution at all."""
    def __init__(self, n=10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n)
        self.do = nn.Dropout(0.3)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_model(model, train_loader, epochs=EPOCHS, lr=1e-3, tag=""):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        print(f"[{tag}] epoch {ep + 1}/{epochs}  loss={running / seen:.4f}  "
              f"({time.time() - t0:.1f}s)")
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Per-sample margin
# ----------------------------------------------------------------------------
@torch.no_grad()
def compute_margin(model, X, Y, batch=512):
    """margin(x) = logit_true(x) - max_{k != true} logit_k(x)."""
    model.eval()
    N = X.size(0)
    out = torch.zeros(N, device=DEVICE)
    for i in range(0, N, batch):
        xb = X[i:i + batch]
        yb = Y[i:i + batch]
        logits = model(xb)
        true_logit = logits[torch.arange(logits.size(0)), yb]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(logits.size(0)), yb] = False
        other_max = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
        out[i:i + batch] = true_logit - other_max
    return out.cpu().numpy()


# ----------------------------------------------------------------------------
# Attacks (defined on VictimSmall)
# ----------------------------------------------------------------------------
def fgsm_perturb(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    return (x + eps * grad.sign()).clamp(0, 1).detach()


def pgd_perturb(model, x, y, eps, alpha, steps):
    x_orig = x.clone().detach()
    x_adv = x_orig + (torch.rand_like(x_orig) * 2 - 1) * eps
    x_adv = x_adv.clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = x_adv.clamp(0, 1).detach()
    return x_adv


@torch.no_grad()
def predict(model, x, batch=512):
    preds = []
    for i in range(0, x.size(0), batch):
        preds.append(model(x[i:i + batch]).argmax(1))
    return torch.cat(preds)


def flipped_by_attack(model, X, Y, attack_fn, batch=256):
    """Return bool array: True if attack flipped the prediction."""
    flips = []
    for i in range(0, X.size(0), batch):
        xb = X[i:i + batch]
        yb = Y[i:i + batch]
        x_adv = attack_fn(model, xb, yb)
        with torch.no_grad():
            new_pred = model(x_adv).argmax(1)
        flips.append((new_pred != yb).cpu().numpy())
    return np.concatenate(flips).astype(np.int32)


def min_eps_fgsm_binary_search(model, X, Y, eps_lo=0.0, eps_hi=0.3,
                               n_iters=10, batch=256):
    """Per-sample smallest FGSM eps that flips. Returns float array."""
    N = X.size(0)
    lo = np.full(N, eps_lo, dtype=np.float32)
    hi = np.full(N, eps_hi, dtype=np.float32)
    # First check whether eps_hi flips at all; if not we cap at eps_hi.
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        flipped = np.zeros(N, dtype=bool)
        for i in range(0, N, batch):
            xb = X[i:i + batch]
            yb = Y[i:i + batch]
            mb = torch.tensor(mid[i:i + batch], device=DEVICE).view(-1, 1, 1, 1)
            x_adv = (xb + mb * _signed_grad(model, xb, yb)).clamp(0, 1)
            with torch.no_grad():
                flipped[i:i + batch] = (model(x_adv).argmax(1) != yb).cpu().numpy()
        # If flipped at mid, smaller eps may also work -> shrink hi.
        hi = np.where(flipped, mid, hi)
        lo = np.where(flipped, lo, mid)
    return hi  # upper bound = smallest known eps that flips (or eps_hi cap)


def _signed_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    return grad.sign().detach()


# ----------------------------------------------------------------------------
# Image stat baselines (model-free)
# ----------------------------------------------------------------------------
def image_stats(X):
    """Return dict of per-sample model-free features."""
    Xc = X.detach().cpu().numpy()  # (N, 1, 28, 28)
    flat = Xc.reshape(Xc.shape[0], -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    # Sobel energy (no scipy dependency: manual conv).
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = kx.T
    sobel = np.zeros(Xc.shape[0], dtype=np.float32)
    img = Xc[:, 0]
    # Vectorised valid-convolution via stride tricks.
    H, W = img.shape[1], img.shape[2]
    patches = np.lib.stride_tricks.sliding_window_view(img, (3, 3), axis=(1, 2))
    # patches shape: (N, H-2, W-2, 3, 3)
    gx = (patches * kx).sum(axis=(-1, -2))
    gy = (patches * ky).sum(axis=(-1, -2))
    sobel = np.sqrt(gx * gx + gy * gy).mean(axis=(1, 2))
    return {"image_mean": mean, "image_std": std, "sobel_energy": sobel}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    tfm = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True,
                                      transform=tfm)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True,
                                     transform=tfm)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)

    # Materialise full test set on device (it's small).
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    print(f"Test set: {test_x.shape}")

    # Train all three. Different seeds so they aren't accidentally aligned.
    torch.manual_seed(0)
    victim = train_model(VictimSmall(), train_loader, tag="VictimSmall")
    torch.manual_seed(1)
    proxy_wide = train_model(ProxyWide(), train_loader, tag="ProxyWide")
    torch.manual_seed(2)
    proxy_mlp = train_model(ProxyMLP(), train_loader, tag="ProxyMLP")

    # Clean accuracy sanity.
    for name, m in [("VictimSmall", victim), ("ProxyWide", proxy_wide),
                    ("ProxyMLP", proxy_mlp)]:
        preds = predict(m, test_x)
        acc = (preds == test_y).float().mean().item()
        print(f"  {name} clean acc = {acc:.4f}")

    # Per-sample margins.
    print("Computing margins...")
    victim_margin = compute_margin(victim, test_x, test_y)
    pw_margin = compute_margin(proxy_wide, test_x, test_y)
    pm_margin = compute_margin(proxy_mlp, test_x, test_y)

    # Image stats baseline.
    print("Computing image stats baseline...")
    stats = image_stats(test_x)

    # Vulnerability targets on VictimSmall.
    print("Computing FGSM flip targets...")
    flipped_fgsm = flipped_by_attack(
        victim, test_x, test_y,
        lambda m, x, y: fgsm_perturb(m, x, y, EPS_TEST))
    print(f"  FGSM flip rate = {flipped_fgsm.mean():.4f}")

    print("Computing PGD flip targets...")
    flipped_pgd = flipped_by_attack(
        victim, test_x, test_y,
        lambda m, x, y: pgd_perturb(m, x, y, EPS_TEST, PGD_ALPHA, PGD_STEPS))
    print(f"  PGD flip rate = {flipped_pgd.mean():.4f}")

    print("Computing min-eps FGSM binary search...")
    min_eps = min_eps_fgsm_binary_search(victim, test_x, test_y,
                                         eps_lo=0.0, eps_hi=0.3, n_iters=10)
    # For AUROC against a binary label we use the *median* split:
    # samples below the median min_eps are the more vulnerable half.
    median_eps = float(np.median(min_eps))
    flipped_min_eps = (min_eps < median_eps).astype(np.int32)
    print(f"  median min_eps = {median_eps:.4f}")

    # AUROC: higher score should mean more vulnerable.
    # Margin: smaller margin = more vulnerable, so we use NEGATIVE margin.
    # Image mean/std/sobel: direction unknown a priori, report AUROC and
    # also report max(AUROC, 1 - AUROC) as an "orientation-free" score.
    feats = {
        "victim_margin":    -victim_margin,
        "proxy_wide_margin": -pw_margin,
        "proxy_mlp_margin":  -pm_margin,
        "image_mean":        stats["image_mean"],
        "image_std":         stats["image_std"],
        "sobel_energy":      stats["sobel_energy"],
    }
    targets = {
        "flipped_FGSM":   flipped_fgsm,
        "flipped_PGD":    flipped_pgd,
        "flipped_minEps": flipped_min_eps,
    }

    results = {"auroc": {}, "auroc_orientation_free": {}}
    print("\n=== Univariate AUROC (higher score => more vulnerable) ===")
    header = f"{'feature':<22s}" + "".join(f"{t:>16s}" for t in targets)
    print(header)
    print("-" * len(header))
    for fname, fvals in feats.items():
        row_auc = {}
        row_orient = {}
        for tname, tvals in targets.items():
            if len(np.unique(tvals)) < 2:
                a = float("nan")
            else:
                a = roc_auc_score(tvals, fvals)
            row_auc[tname] = a
            row_orient[tname] = max(a, 1 - a) if not np.isnan(a) else a
        results["auroc"][fname] = row_auc
        results["auroc_orientation_free"][fname] = row_orient
        cells = "".join(f"{row_auc[t]:>16.4f}" for t in targets)
        print(f"{fname:<22s}{cells}")

    # Quick H18 verdict against flipped_FGSM.
    a_oracle = results["auroc"]["victim_margin"]["flipped_FGSM"]
    a_wide = results["auroc"]["proxy_wide_margin"]["flipped_FGSM"]
    a_mlp = results["auroc"]["proxy_mlp_margin"]["flipped_FGSM"]
    a_base = max(results["auroc_orientation_free"][s]["flipped_FGSM"]
                 for s in ("image_mean", "image_std", "sobel_energy"))
    print("\n=== H18 verdict (target = flipped_FGSM @ eps=15/255) ===")
    print(f"  oracle (victim_margin)        AUROC = {a_oracle:.4f}")
    print(f"  proxy_wide (similar arch)     AUROC = {a_wide:.4f}  "
          f"gap to oracle = {a_oracle - a_wide:+.4f}")
    print(f"  proxy_mlp  (different arch)   AUROC = {a_mlp:.4f}  "
          f"gap to oracle = {a_oracle - a_mlp:+.4f}")
    print(f"  best image-stat baseline      AUROC = {a_base:.4f}")

    results["verdict_flipped_FGSM"] = {
        "oracle": a_oracle,
        "proxy_wide": a_wide,
        "proxy_mlp": a_mlp,
        "best_image_stat_baseline": a_base,
        "wide_recovers_fraction_of_oracle":
            (a_wide - 0.5) / (a_oracle - 0.5) if a_oracle > 0.5 else None,
        "mlp_recovers_fraction_of_oracle":
            (a_mlp - 0.5) / (a_oracle - 0.5) if a_oracle > 0.5 else None,
    }
    results["meta"] = {
        "epochs": EPOCHS, "batch": BATCH, "eps_test": EPS_TEST,
        "pgd_steps": PGD_STEPS, "pgd_alpha": PGD_ALPHA,
        "n_test": int(test_x.size(0)),
        "fgsm_flip_rate": float(flipped_fgsm.mean()),
        "pgd_flip_rate": float(flipped_pgd.mean()),
        "median_min_eps": median_eps,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
