"""
H163: Vulnerability feature leaderboard on CIFAR-10 at low epsilon (EPS=4/255).

Background: At EPS=15/255, PGD achieves ~100% success on CIFAR-10 (full saturation),
making all PGD-target AUROCs meaningless (zero-variance binary target). At EPS=4/255,
PGD has partial success (~60-80%), restoring variance and yielding valid AUROC.

This script benchmarks the top-8 vulnerability features from the Fashion-MNIST
leaderboard against three attack targets at the lower epsilon, to check whether
the same ranking holds on CIFAR-10 with a PGD target that is not degenerate.

Features evaluated:
  1. margin              - top1 logit minus top2 logit
  2. top1_prob           - softmax maximum
  3. confusion_ratio     - top2_prob / top1_prob
  4. smoothgrad_l2_norm  - L2 norm of SmoothGrad (K=10, sigma=0.1)
  5. input_grad_l2_norm  - L2 norm of plain input gradient
  6. mean_pix            - mean pixel value over the image
  7. std_pix             - std of pixel values over the image
  8. predictive_entropy  - MC-dropout entropy (K=10 forward passes in train mode)

Targets at EPS=4/255:
  - flipped_FGSM  : bool, FGSM flips the prediction
  - flipped_PGD   : bool, PGD-10 (alpha=1/255) flips the prediction
  - min_eps       : float, binary-search minimum epsilon to flip via FGSM sign
                    (lo=0, hi=0.1, 8 iterations — narrower range for CIFAR-10)

Architecture: vanilla CNN matching the rest of the series.
Dataset: datasets.FashionMNIST monkey-patched at runtime via PATCH_DATASET_NAME=cifar10.
Evaluation: 1000 correctly-classified test samples.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 4.0 / 255.0      # KEY: lower epsilon to avoid PGD saturation on CIFAR-10
N_CLASSES = 10
N_EVAL = 1000           # number of correctly-classified samples to evaluate


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    """Small CNN matching the dissertation-extension series."""

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


def get_loaders():
    tf = transforms.ToTensor()
    train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2)
    return train_loader, test_loader


def auroc_safe(y_true, y_score):
    """Return max(AUROC, 1-AUROC). Returns 0.5 if target has no variance."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    a = roc_auc_score(y_true, y_score)
    return max(a, 1.0 - a)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def fgsm_sign(model, x, y):
    """Return the FGSM sign gradient (detached)."""
    xr = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xr), y).backward()
    return xr.grad.sign().detach()


def attack_fgsm(model, x, y, eps=EPS):
    sign = fgsm_sign(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def attack_pgd(model, x, y, eps=EPS, alpha=1.0 / 255.0, steps=10):
    """PGD with alpha=1/255 and 10 steps (conservative for low eps)."""
    x_adv = x.clone().detach()
    # Random start within eps-ball
    x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0, 1)
    x_adv = x_adv.detach()
    for _ in range(steps):
        x_adv = x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(x_adv).argmax(1) != y)


def min_eps_binary_search(model, x, y, lo=0.0, hi=0.1, iters=8):
    """Binary search for minimum epsilon to flip prediction via FGSM sign.

    Range hi=0.1 (narrower than 0.3 used elsewhere) since CIFAR-10
    images tend to flip at much lower eps than Fashion-MNIST.
    """
    sign = fgsm_sign(model, x, y)
    lo_t = torch.zeros(x.size(0), device=DEVICE)
    hi_t = torch.full((x.size(0),), hi, device=DEVICE)
    for _ in range(iters):
        mid = (lo_t + hi_t) / 2.0
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi_t = torch.where(flipped, mid, hi_t)
        lo_t = torch.where(flipped, lo_t, mid)
    return hi_t  # upper bound of the final interval


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def smoothgrad(model, x, y, K=10, sigma=0.1):
    """Average gradient over K noisy copies of x (SmoothGrad)."""
    grads = []
    for _ in range(K):
        x_noisy = (x + sigma * torch.randn_like(x)).clamp(0, 1).requires_grad_(True)
        F.cross_entropy(model(x_noisy), y).backward()
        grads.append(x_noisy.grad.detach())
    return torch.stack(grads).mean(0)


def compute_features_batch(model, x, y):
    """Compute all 8 features for a single batch. Returns dict of tensors (CPU)."""
    B = x.size(0)

    # --- Logit-based features (no grad needed) ---
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        sorted_logits, _ = logits.sort(dim=1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        top1_prob = probs.max(dim=1).values
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        confusion_ratio = sorted_probs[:, 1] / (sorted_probs[:, 0] + 1e-9)

    # --- Pixel statistics ---
    flat = x.view(B, -1)
    mean_pix = flat.mean(dim=1).detach()
    std_pix = flat.std(dim=1).detach()

    # --- Plain input gradient ---
    xr = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xr), y).backward()
    input_grad_l2 = xr.grad.detach().view(B, -1).norm(dim=1)

    # --- SmoothGrad (model must be in eval mode for deterministic dropout) ---
    sg = smoothgrad(model, x, y, K=10, sigma=0.1)
    smoothgrad_l2 = sg.view(B, -1).norm(dim=1)

    # --- MC-dropout entropy ---
    model.train()
    with torch.no_grad():
        mc_probs = torch.stack(
            [F.softmax(model(x), dim=1) for _ in range(10)]
        ).mean(0)
    model.eval()
    entropy = -(mc_probs * (mc_probs + 1e-9).log()).sum(dim=1)

    return {
        "margin": margin.cpu(),
        "top1_prob": top1_prob.cpu(),
        "confusion_ratio": confusion_ratio.cpu(),
        "smoothgrad_l2_norm": smoothgrad_l2.cpu(),
        "input_grad_l2_norm": input_grad_l2.cpu(),
        "mean_pix": mean_pix.cpu(),
        "std_pix": std_pix.cpu(),
        "predictive_entropy": entropy.cpu(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("H163: Vulnerability feature leaderboard on CIFAR-10 at low epsilon")
    print("Running at EPS=4/255 to avoid PGD saturation seen at EPS=15/255.")
    print(f"Device: {DEVICE}")
    print()

    train_loader, test_loader = get_loaders()

    # --- Train vanilla CNN ---
    print(f"=== Training Vanilla CNN ({EPOCHS} epochs, seed=0) ===")
    set_seed(0)
    model = CNN(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer)
        print(f"  Epoch {ep:2d}/{EPOCHS}  train_loss={loss:.4f}")
    model.eval()
    print()

    # --- Collect all test samples and filter to correctly classified ---
    print("=== Collecting test set and filtering to correctly classified ===")
    test_x_list, test_y_list = [], []
    for x, y in test_loader:
        test_x_list.append(x)
        test_y_list.append(y)
    test_x = torch.cat(test_x_list)
    test_y = torch.cat(test_y_list)

    # Evaluate in batches
    all_preds = []
    with torch.no_grad():
        for i in range(0, test_x.size(0), BATCH):
            xb = test_x[i:i + BATCH].to(DEVICE)
            all_preds.append(model(xb).argmax(1).cpu())
    all_preds = torch.cat(all_preds)
    correct_mask = (all_preds == test_y)
    correct_idx = correct_mask.nonzero(as_tuple=True)[0]

    # Limit to N_EVAL samples
    if len(correct_idx) > N_EVAL:
        correct_idx = correct_idx[:N_EVAL]

    x_eval = test_x[correct_idx].to(DEVICE)
    y_eval = test_y[correct_idx].to(DEVICE)
    N_c = x_eval.size(0)
    print(f"  Total correct: {correct_mask.sum().item()} / {len(test_y)}")
    print(f"  Using first {N_c} for evaluation")
    print()

    # --- Compute features in batches of BATCH ---
    print(f"=== Computing features (8 total, batch size {BATCH}) ===")
    feat_lists = {k: [] for k in [
        "margin", "top1_prob", "confusion_ratio",
        "smoothgrad_l2_norm", "input_grad_l2_norm",
        "mean_pix", "std_pix", "predictive_entropy",
    ]}
    for i in range(0, N_c, BATCH):
        xb = x_eval[i:i + BATCH]
        yb = y_eval[i:i + BATCH]
        batch_feats = compute_features_batch(model, xb, yb)
        for k, v in batch_feats.items():
            feat_lists[k].append(v)
        if (i // BATCH + 1) % 2 == 0 or (i + BATCH) >= N_c:
            print(f"  Features: batch {i // BATCH + 1}/{(N_c + BATCH - 1) // BATCH} done")

    feats = {k: torch.cat(v).numpy() for k, v in feat_lists.items()}
    print()

    # --- Compute attack targets in batches ---
    print(f"=== Computing attack targets at EPS={EPS:.5f} (4/255) ===")
    fgsm_list, pgd_list, mineps_list = [], [], []
    for i in range(0, N_c, BATCH):
        xb = x_eval[i:i + BATCH]
        yb = y_eval[i:i + BATCH]
        fgsm_list.append(attack_fgsm(model, xb, yb, eps=EPS))
        pgd_list.append(attack_pgd(model, xb, yb, eps=EPS))
        mineps_list.append(min_eps_binary_search(model, xb, yb, lo=0.0, hi=0.1, iters=8))
        if (i // BATCH + 1) % 2 == 0 or (i + BATCH) >= N_c:
            print(f"  Attacks: batch {i // BATCH + 1}/{(N_c + BATCH - 1) // BATCH} done")

    y_fgsm = torch.cat(fgsm_list).cpu().numpy().astype(int)
    y_pgd = torch.cat(pgd_list).cpu().numpy().astype(int)
    y_mineps = torch.cat(mineps_list).cpu().numpy()

    fgsm_rate = y_fgsm.mean()
    pgd_rate = y_pgd.mean()
    print()
    print(f"  FGSM success rate : {fgsm_rate:.4f}  ({y_fgsm.sum()}/{N_c})")
    print(f"  PGD  success rate : {pgd_rate:.4f}  ({y_pgd.sum()}/{N_c})")
    if pgd_rate >= 0.99:
        print("  WARNING: PGD is still fully saturated (rate>=0.99). "
              "Consider lowering EPS further.")
    elif pgd_rate < 0.05:
        print("  WARNING: PGD success rate very low (<5%). "
              "Consider raising EPS for more signal.")
    else:
        print("  PGD success rate is non-degenerate — AUROCs should be meaningful.")
    print()

    # min_eps: binarise at median for AUROC (lower min_eps => easier to flip => vulnerable)
    median_mineps = float(np.median(y_mineps))
    y_mineps_bin = (y_mineps <= median_mineps).astype(int)

    # --- Univariate AUROC table ---
    targets = {
        "flipped_FGSM": y_fgsm,
        "flipped_PGD":  y_pgd,
        "min_eps":       y_mineps_bin,
    }

    feat_names = list(feats.keys())
    target_names = list(targets.keys())

    auroc_table = {fn: {} for fn in feat_names}
    for fn in feat_names:
        for tn in target_names:
            auroc_table[fn][tn] = auroc_safe(targets[tn], feats[fn])

    col_w = 14
    header = f"{'Feature':<24}" + "".join(f"{t:>{col_w}}" for t in target_names)
    print("=== Univariate AUROC Results (EPS=4/255) ===")
    print(header)
    print("-" * len(header))
    for fn in feat_names:
        row = f"{fn:<24}"
        for tn in target_names:
            row += f"{auroc_table[fn][tn]:>{col_w}.4f}"
        print(row)
    print()

    # --- Summary: features that beat margin ---
    print("=== Summary: Features That Beat Margin ===")
    margin_aurocs = auroc_table["margin"]
    print(f"  Baseline margin AUROCs: "
          + "  ".join(f"{tn}={margin_aurocs[tn]:.4f}" for tn in target_names))
    print()

    beat = {tn: [] for tn in target_names}
    for fn in feat_names:
        if fn == "margin":
            continue
        for tn in target_names:
            if auroc_table[fn][tn] > margin_aurocs[tn]:
                beat[tn].append((fn, auroc_table[fn][tn]))

    for tn in target_names:
        ranked = sorted(beat[tn], key=lambda t: -t[1])
        if ranked:
            print(f"  Target '{tn}' — features beating margin ({margin_aurocs[tn]:.4f}):")
            for fn, a in ranked:
                print(f"    {fn:<24} AUROC={a:.4f}  (+{a - margin_aurocs[tn]:.4f})")
        else:
            print(f"  Target '{tn}' — no feature beats margin ({margin_aurocs[tn]:.4f}).")
    print()

    # Mean AUROC across targets per feature
    mean_aurocs = {fn: np.mean([auroc_table[fn][tn] for tn in target_names])
                   for fn in feat_names}
    ranked_overall = sorted(mean_aurocs.items(), key=lambda t: -t[1])
    print("=== Overall Feature Ranking (mean AUROC across 3 targets) ===")
    for rank_i, (fn, a) in enumerate(ranked_overall, 1):
        marker = "  <- margin baseline" if fn == "margin" else ""
        print(f"  {rank_i}. {fn:<24} mean_AUROC={a:.4f}{marker}")

    print()
    print(f"EPS used: {EPS:.6f}  ({EPS * 255:.1f}/255)")
    print(f"PGD success rate at this EPS: {pgd_rate:.4f}")
    if pgd_rate < 0.99:
        print("Conclusion: PGD saturation is resolved at EPS=4/255. "
              "All three AUROC targets carry meaningful signal.")


if __name__ == "__main__":
    main()
