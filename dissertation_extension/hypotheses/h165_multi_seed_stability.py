"""
H165: Multi-seed stability of top vulnerability feature AUROC rankings.

Reviewer concern: AUROC differences < 0.02 may be seed artefacts.

Procedure:
  1. Train 3 CNNs with seeds 0, 1, 2 (each trained independently for 10 epochs).
     During training, save checkpoints at epochs 5, 7, 10 (for softmax_variance).
  2. Fix the evaluation set: 1000 correctly-classified test samples from seed=0's
     model perspective (correctly classified by model_seed0).
  3. For each seed s, use model_seed_s as the primary model and the other two as
     auxiliary models. Evaluate all 8 top vulnerability features:
       1. margin                 — top1 minus top2 logit
       2. smoothgrad_l2_norm     — SmoothGrad K=10, sigma=0.1
       3. predictive_entropy     — MC-dropout K=10 (model in train mode)
       4. input_grad_l2_norm     — plain L2 norm of input gradient
       5. memorization_proxy     — softmax-max variance across the 3 seed models
       6. softmax_variance       — snapshot ensemble (checkpoints at epochs 5,7,10)
       7. bnn_predictive_variance — SWAG-lite weight perturbation variance K=5
       8. pixel_sign_agreement   — gradient-sign consensus across all 3 seed models
     Record AUROC for flipped_PGD and min_eps targets at EPS=15/255.
  4. Print table: feature | seed0 | seed1 | seed2 | mean | std
     Label each row as "Stable" (std < 0.01), "Moderate" (std < 0.03),
     or "Unstable" (std >= 0.03).

Dataset: Fashion-MNIST (monkey-patched at runtime via patch_dataset.py if needed).
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10
SEEDS = [0, 1, 2]

# Snapshot checkpoints (epoch indices, 1-based)
SNAPSHOT_EPOCHS = {5, 7, 10}

# SWAG-lite: number of weight samples for bnn_predictive_variance
SWAG_K = 5
# SWAG weight-noise scale (fraction of parameter std)
SWAG_NOISE_SCALE = 0.01

# SmoothGrad settings
SG_K = 10
SG_SIGMA = 0.1

# MC-dropout passes
MC_K = 10


# ---------------------------------------------------------------------------
# Architecture
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
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auroc_both_directions(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    a = roc_auc_score(y_true, y_score)
    return float(max(a, 1.0 - a))


# ---------------------------------------------------------------------------
# Attack / target helpers
# ---------------------------------------------------------------------------

def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               eps: float = EPS, alpha: float = 2.0 / 255.0, steps: int = 10) -> torch.Tensor:
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = (x_adv + alpha * x_adv.grad.sign())
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0.0, 1.0)
    return x_adv.detach()


def min_eps_to_flip(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                    eps_max: float = 0.3, iters: int = 15) -> torch.Tensor:
    """Binary search for minimum L-inf epsilon to flip the prediction."""
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    # Use FGSM sign direction for speed
    x_var = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_var), y)
    loss.backward()
    sign = x_var.grad.sign().detach()
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0.0, 1.0)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(train_set, seed: int):
    """Train a CNN and return (model, snapshots_dict).

    snapshots_dict maps epoch number (1-based) -> state_dict copy,
    for the epochs in SNAPSHOT_EPOCHS.
    """
    set_seed(seed)
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    snapshots: dict = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
        if epoch in SNAPSHOT_EPOCHS:
            snapshots[epoch] = copy.deepcopy(model.state_dict())
        print(f"  seed={seed}  epoch {epoch}/{EPOCHS} done")

    model.eval()
    return model, snapshots


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def feat_margin(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Top-1 minus top-2 logit (higher = more confident, potentially less vulnerable)."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        top2, _ = logits.topk(2, dim=1)
        return top2[:, 0] - top2[:, 1]


def feat_smoothgrad_l2(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SmoothGrad L2 norm (K=10, sigma=0.1)."""
    model.eval()
    grad_sum = torch.zeros_like(x)
    for _ in range(SG_K):
        noise = torch.randn_like(x) * SG_SIGMA
        x_n = (x + noise).detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_n), y)
        loss.backward()
        with torch.no_grad():
            if x_n.grad is not None:
                grad_sum += x_n.grad
    sg = grad_sum / SG_K
    return sg.view(x.size(0), -1).norm(2, dim=1)


def feat_predictive_entropy(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """MC-dropout predictive entropy (K=10 passes, model in train mode)."""
    model.train()
    probs_sum = torch.zeros(x.size(0), N_CLASSES, device=DEVICE)
    with torch.no_grad():
        for _ in range(MC_K):
            probs_sum += F.softmax(model(x), dim=1)
    mean_probs = probs_sum / MC_K
    entropy = -(mean_probs * (mean_probs + 1e-12).log()).sum(dim=1)
    model.eval()
    return entropy


def feat_input_grad_l2(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Plain L2 norm of input gradient."""
    model.eval()
    x_var = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_var), y)
    loss.backward()
    grad = x_var.grad.detach()
    return grad.view(x.size(0), -1).norm(2, dim=1)


def feat_memorization_proxy(models: list, x: torch.Tensor) -> torch.Tensor:
    """Variance of softmax-max across the 3 seed models."""
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(F.softmax(m(x), dim=1).max(dim=1).values)
    stacked = torch.stack(preds, dim=1)  # (N, 3)
    return stacked.var(dim=1)


def feat_softmax_variance(snapshots: dict, x: torch.Tensor) -> torch.Tensor:
    """Snapshot ensemble softmax variance using checkpoints at epochs 5, 7, 10."""
    base = CNN(N_CLASSES).to(DEVICE)
    preds = []
    for ep in sorted(snapshots.keys()):
        base.load_state_dict(snapshots[ep])
        base.eval()
        with torch.no_grad():
            preds.append(F.softmax(base(x), dim=1))
    stacked = torch.stack(preds, dim=1)  # (N, 3, C)
    return stacked.var(dim=1).mean(dim=1)  # mean class-wise variance


def feat_bnn_predictive_variance(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """SWAG-lite: perturb weights with N(0, noise_scale * |w|) K=5 times,
    compute variance of softmax max predictions."""
    model.eval()
    preds = []
    orig_state = copy.deepcopy(model.state_dict())
    for _ in range(SWAG_K):
        perturbed = copy.deepcopy(orig_state)
        for k, v in perturbed.items():
            if v.dtype.is_floating_point:
                perturbed[k] = v + torch.randn_like(v) * SWAG_NOISE_SCALE * v.abs()
        model.load_state_dict(perturbed)
        model.eval()
        with torch.no_grad():
            preds.append(F.softmax(model(x), dim=1).max(dim=1).values)
    # Restore original weights
    model.load_state_dict(orig_state)
    stacked = torch.stack(preds, dim=1)  # (N, K)
    return stacked.var(dim=1)


def feat_pixel_sign_agreement(models: list, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean fraction of pixels where all 3 models agree on gradient sign."""
    signs = []
    for m in models:
        m.eval()
        x_var = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(m(x_var), y)
        loss.backward()
        signs.append(x_var.grad.sign().detach())  # (N, 1, 28, 28)
    # Agreement where all three have the same sign (all +1 or all -1)
    s0, s1, s2 = signs[0], signs[1], signs[2]
    agree = ((s0 == s1) & (s1 == s2)).float()
    return agree.view(x.size(0), -1).mean(dim=1)


# ---------------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------------

def evaluate_seed(primary_model: nn.Module,
                  snapshots: dict,
                  all_models: list,
                  x_eval: torch.Tensor,
                  y_eval: torch.Tensor,
                  seed_label: int) -> dict:
    """Compute 8 features + PGD/min_eps targets. Return AUROC dict."""
    print(f"\n  [seed={seed_label}] Computing features...")

    print(f"    margin")
    f_margin = feat_margin(primary_model, x_eval).cpu().numpy()

    print(f"    smoothgrad_l2_norm")
    f_sg = feat_smoothgrad_l2(primary_model, x_eval, y_eval).cpu().numpy()

    print(f"    predictive_entropy")
    f_ent = feat_predictive_entropy(primary_model, x_eval).cpu().numpy()

    print(f"    input_grad_l2_norm")
    f_ig = feat_input_grad_l2(primary_model, x_eval, y_eval).cpu().numpy()

    print(f"    memorization_proxy")
    f_mem = feat_memorization_proxy(all_models, x_eval).cpu().numpy()

    print(f"    softmax_variance (snapshot ensemble)")
    f_snap = feat_softmax_variance(snapshots, x_eval).cpu().numpy()

    print(f"    bnn_predictive_variance (SWAG-lite)")
    f_bnn = feat_bnn_predictive_variance(primary_model, x_eval).cpu().numpy()

    print(f"    pixel_sign_agreement")
    f_psa = feat_pixel_sign_agreement(all_models, x_eval, y_eval).cpu().numpy()

    print(f"  [seed={seed_label}] Running PGD attack...")
    primary_model.eval()
    x_pgd = pgd_attack(primary_model, x_eval, y_eval)
    with torch.no_grad():
        flipped_pgd = (primary_model(x_pgd).argmax(1) != y_eval).long().cpu().numpy()

    print(f"  [seed={seed_label}] Computing min_eps...")
    min_eps_vals = min_eps_to_flip(primary_model, x_eval, y_eval).cpu().numpy()

    features = {
        "margin": f_margin,
        "smoothgrad_l2_norm": f_sg,
        "predictive_entropy": f_ent,
        "input_grad_l2_norm": f_ig,
        "memorization_proxy": f_mem,
        "softmax_variance": f_snap,
        "bnn_predictive_variance": f_bnn,
        "pixel_sign_agreement": f_psa,
    }

    results = {}
    for feat_name, feat_vals in features.items():
        auroc_pgd = auroc_both_directions(flipped_pgd, feat_vals)
        # min_eps is lower = more vulnerable, so we negate for AUROC direction
        auroc_mineps = auroc_both_directions((min_eps_vals < np.median(min_eps_vals)).astype(int),
                                             -feat_vals)
        results[feat_name] = {"flipped_PGD": auroc_pgd, "min_eps": auroc_mineps}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def stability_label(std: float) -> str:
    if std < 0.01:
        return "Stable"
    elif std < 0.03:
        return "Moderate"
    else:
        return "Unstable"


def main():
    print("=" * 70)
    print("H165: Multi-seed AUROC stability of top vulnerability features")
    print("=" * 70)

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Load full test set once
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=2)
    x_test_all, y_test_all = next(iter(test_loader))
    x_test_all = x_test_all.to(DEVICE)
    y_test_all = y_test_all.to(DEVICE)

    # -----------------------------------------------------------------------
    # Train 3 models
    # -----------------------------------------------------------------------
    models = []
    all_snapshots = []

    for s in SEEDS:
        print(f"\nTraining model for seed={s}...")
        m, snaps = train_model(train_set, seed=s)
        models.append(m)
        all_snapshots.append(snaps)
        print(f"  Snapshots at epochs: {sorted(snaps.keys())}")

    # -----------------------------------------------------------------------
    # Fix evaluation set: correctly classified by model_seed0 (first 1000)
    # -----------------------------------------------------------------------
    models[0].eval()
    with torch.no_grad():
        preds0 = models[0](x_test_all).argmax(1)
        correct_mask = (preds0 == y_test_all)

    correct_indices = correct_mask.nonzero(as_tuple=True)[0]
    # Take the first 1000 correctly classified samples
    eval_idx = correct_indices[:1000]
    x_eval = x_test_all[eval_idx]
    y_eval = y_test_all[eval_idx]
    print(f"\nFixed evaluation set: {x_eval.size(0)} samples "
          f"(correctly classified by model_seed0)")

    # -----------------------------------------------------------------------
    # Evaluate each seed
    # -----------------------------------------------------------------------
    # results_per_seed[seed_idx][feat_name] = {"flipped_PGD": ..., "min_eps": ...}
    results_per_seed = []

    for si, s in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"Evaluating seed={s} (model {si+1}/3)")
        print(f"{'='*60}")
        primary = models[si]
        snaps = all_snapshots[si]
        # all 3 models available as auxiliary
        res = evaluate_seed(primary, snaps, models, x_eval, y_eval, seed_label=s)
        results_per_seed.append(res)

    # -----------------------------------------------------------------------
    # Print summary tables
    # -----------------------------------------------------------------------
    feature_names = [
        "margin",
        "smoothgrad_l2_norm",
        "predictive_entropy",
        "input_grad_l2_norm",
        "memorization_proxy",
        "softmax_variance",
        "bnn_predictive_variance",
        "pixel_sign_agreement",
    ]

    for target_key in ("flipped_PGD", "min_eps"):
        print(f"\n{'='*90}")
        print(f"AUROC TABLE — target: {target_key}")
        print(f"{'='*90}")
        header = f"{'Feature':<26} {'seed0':>7} {'seed1':>7} {'seed2':>7} {'mean':>7} {'std':>7}  stability"
        print(header)
        print("-" * 90)

        for fn in feature_names:
            vals = [results_per_seed[i][fn][target_key] for i in range(len(SEEDS))]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            label = stability_label(std_v)
            row = (f"{fn:<26} "
                   f"{vals[0]:>7.4f} "
                   f"{vals[1]:>7.4f} "
                   f"{vals[2]:>7.4f} "
                   f"{mean_v:>7.4f} "
                   f"{std_v:>7.4f}  {label}")
            print(row)

    # -----------------------------------------------------------------------
    # Overall stability summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STABILITY SUMMARY")
    print(f"{'='*70}")
    print("Criteria: Stable = std < 0.01, Moderate = std < 0.03, Unstable = std >= 0.03")
    print()

    for target_key in ("flipped_PGD", "min_eps"):
        stable_count = 0
        moderate_count = 0
        unstable_count = 0
        for fn in feature_names:
            vals = [results_per_seed[i][fn][target_key] for i in range(len(SEEDS))]
            std_v = float(np.std(vals))
            lbl = stability_label(std_v)
            if lbl == "Stable":
                stable_count += 1
            elif lbl == "Moderate":
                moderate_count += 1
            else:
                unstable_count += 1
        print(f"Target={target_key}: "
              f"Stable={stable_count}/8  Moderate={moderate_count}/8  Unstable={unstable_count}/8")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
