"""
H157: Adversarial Vulnerability Over Training Time (Training Trajectory Analysis)

Hypothesis: Adversarial robustness does not emerge uniformly during training.
Instead, it may follow one of several patterns:
  - Gradual monotonic increase alongside clean accuracy
  - A delayed "grokking-like" onset where robustness lags behind accuracy
  - A spike followed by regression
  - No meaningful trend (robustness is largely independent of training duration)

This script trains two CNN models for 30 epochs on Fashion-MNIST:
  1. Standard (vanilla) training with cross-entropy loss
  2. PGD-AT (adversarial training) using PGD-generated adversarial examples

At every 3 epochs (10 checkpoints) we evaluate on 500 test samples:
  - Clean accuracy
  - FGSM attack success rate
  - PGD-10 attack success rate
  - Mean min_eps_to_flip (binary search, 8 iterations)

We then compute the Spearman rank correlation between epoch and min_eps_to_flip
for each model to quantify whether robustness grows monotonically.

Finally we note whether the vanilla model shows a "delayed robustness" onset
(i.e., robustness does not begin to increase until well into training).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10
CHECKPOINT_EVERY = 3          # evaluate every this many epochs
N_EVAL = 500                  # test samples per checkpoint evaluation
FGSM_EPS = EPS
PGD_EPS = EPS
PGD_ALPHA = EPS / 4.0
PGD_STEPS = 10
BINARY_SEARCH_ITERS = 8
BINARY_HI = 0.3


# ---------------------------------------------------------------------------
# Model
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
# Attacks
# ---------------------------------------------------------------------------
def fgsm(model, x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    return (x_adv + eps * x_adv.grad.sign()).clamp(0.0, 1.0).detach()


def pgd(model, x, y, eps, alpha, steps):
    x_adv = x.clone().detach() + torch.zeros_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0.0, 1.0)
    return x_adv.detach()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_epoch_standard(model, loader, optimizer):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        optimizer.step()


def train_epoch_pgdat(model, loader, optimizer):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        model.eval()
        xb_adv = pgd(model, xb, yb, PGD_EPS, PGD_ALPHA, PGD_STEPS)
        model.train()
        optimizer.zero_grad()
        F.cross_entropy(model(xb_adv), yb).backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_checkpoint(model, x_eval, y_eval):
    """Return (clean_acc, fgsm_sr, pgd_sr, mean_min_eps)."""
    model.eval()

    # Clean accuracy
    with torch.no_grad():
        logits = model(x_eval)
        preds = logits.argmax(1)
        clean_acc = (preds == y_eval).float().mean().item()

    # FGSM success rate (on correctly classified samples only)
    correct_mask = preds == y_eval
    if correct_mask.sum() == 0:
        fgsm_sr = 0.0
        pgd_sr = 0.0
        mean_min_eps = 0.0
        return clean_acc, fgsm_sr, pgd_sr, mean_min_eps

    xc = x_eval[correct_mask]
    yc = y_eval[correct_mask]

    x_fgsm = fgsm(model, xc, yc, FGSM_EPS)
    with torch.no_grad():
        fgsm_preds = model(x_fgsm).argmax(1)
        fgsm_sr = (fgsm_preds != yc).float().mean().item()

    x_pgd = pgd(model, xc, yc, PGD_EPS, PGD_ALPHA, PGD_STEPS)
    with torch.no_grad():
        pgd_preds = model(x_pgd).argmax(1)
        pgd_sr = (pgd_preds != yc).float().mean().item()

    # Mean min_eps_to_flip via binary search
    min_eps_vals = []
    for i in range(len(xc)):
        xi = xc[i:i+1]
        yi = yc[i:i+1]
        lo, hi = 0.0, BINARY_HI
        with torch.no_grad():
            orig_pred = model(xi).argmax(1)
        if orig_pred.item() != yi.item():
            min_eps_vals.append(0.0)
            continue
        for _ in range(BINARY_SEARCH_ITERS):
            mid = (lo + hi) / 2.0
            x_p = (xi + mid * xi.grad.sign() if False else
                   (xi + mid * torch.sign(torch.randn_like(xi))).clamp(0, 1))
            # Use FGSM-sign direction: need gradient
            xi_g = xi.clone().detach().requires_grad_(True)
            F.cross_entropy(model(xi_g), yi).backward()
            direction = xi_g.grad.sign().detach()
            x_pert = (xi + mid * direction).clamp(0.0, 1.0)
            with torch.no_grad():
                flipped = model(x_pert).argmax(1).item() != yi.item()
            if flipped:
                hi = mid
            else:
                lo = mid
        min_eps_vals.append(hi)

    mean_min_eps = sum(min_eps_vals) / len(min_eps_vals) if min_eps_vals else 0.0
    return clean_acc, fgsm_sr, pgd_sr, mean_min_eps


# ---------------------------------------------------------------------------
# Spearman correlation (no scipy dependency)
# ---------------------------------------------------------------------------
def spearman_corr(xs, ys):
    """Compute Spearman rank correlation between two lists."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(vals):
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        for rank, idx in enumerate(sorted_idx, 1):
            r[idx] = float(rank)
        return r

    rx = ranks(xs)
    ry = ranks(ys)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("H157: Training Trajectory — Adversarial Vulnerability Over Time")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}, Checkpoints every {CHECKPOINT_EVERY} epochs")
    print(f"Eval samples per checkpoint: {N_EVAL}")
    print()

    tf = transforms.ToTensor()

    train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=2, pin_memory=True)

    # Fixed eval subset (same 500 samples for all checkpoints)
    eval_indices = list(range(N_EVAL))
    eval_subset = Subset(test_ds, eval_indices)
    eval_loader = DataLoader(eval_subset, batch_size=N_EVAL, shuffle=False)
    x_eval, y_eval = next(iter(eval_loader))
    x_eval, y_eval = x_eval.to(DEVICE), y_eval.to(DEVICE)

    # -----------------------------------------------------------------------
    # Train both models, collecting checkpoint data
    # -----------------------------------------------------------------------
    results = {}

    for mode in ("standard", "pgdat"):
        print(f"--- Training: {mode.upper()} ---")
        model = CNN(N_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        rows = []  # list of (epoch, clean_acc, fgsm_sr, pgd_sr, mean_min_eps)

        for epoch in range(1, EPOCHS + 1):
            if mode == "standard":
                train_epoch_standard(model, train_loader, optimizer)
            else:
                train_epoch_pgdat(model, train_loader, optimizer)
            scheduler.step()

            if epoch % CHECKPOINT_EVERY == 0:
                print(f"  Evaluating epoch {epoch:2d}...", end=" ", flush=True)
                clean_acc, fgsm_sr, pgd_sr, mean_min_eps = evaluate_checkpoint(
                    model, x_eval, y_eval
                )
                rows.append((epoch, clean_acc, fgsm_sr, pgd_sr, mean_min_eps))
                print(f"clean={clean_acc:.3f} fgsm_sr={fgsm_sr:.3f} "
                      f"pgd_sr={pgd_sr:.3f} min_eps={mean_min_eps:.4f}")

        results[mode] = rows
        print()

    # -----------------------------------------------------------------------
    # Print trajectory tables
    # -----------------------------------------------------------------------
    header = (f"{'Epoch':>5}  {'CleanAcc':>8}  {'FGSM_SR':>7}  "
              f"{'PGD_SR':>6}  {'MinEps':>7}")
    sep = "-" * len(header)

    for mode in ("standard", "pgdat"):
        label = "STANDARD" if mode == "standard" else "PGD-AT"
        print(f"Trajectory Table — {label} Model")
        print(sep)
        print(header)
        print(sep)
        for epoch, ca, fs, ps, me in results[mode]:
            print(f"{epoch:>5}  {ca:>8.3f}  {fs:>7.3f}  {ps:>6.3f}  {me:>7.4f}")
        print(sep)
        print()

    # -----------------------------------------------------------------------
    # Spearman correlations
    # -----------------------------------------------------------------------
    print("Spearman Rank Correlation: Epoch vs Mean Min-Eps-to-Flip")
    print("-" * 55)
    for mode in ("standard", "pgdat"):
        rows = results[mode]
        epochs = [r[0] for r in rows]
        min_eps = [r[4] for r in rows]
        rho = spearman_corr(epochs, min_eps)
        label = "Standard" if mode == "standard" else "PGD-AT  "
        print(f"  {label}: rho = {rho:+.4f}")
    print()

    # -----------------------------------------------------------------------
    # Delayed robustness onset analysis (vanilla model)
    # -----------------------------------------------------------------------
    std_rows = results["standard"]
    min_eps_vals = [r[4] for r in std_rows]
    epochs_list = [r[0] for r in std_rows]

    # Find first epoch where min_eps exceeds 10% of max
    max_me = max(min_eps_vals) if min_eps_vals else 0.0
    threshold = 0.10 * max_me
    onset_epoch = None
    for epoch, me in zip(epochs_list, min_eps_vals):
        if me >= threshold:
            onset_epoch = epoch
            break

    print("Delayed Robustness Onset Analysis (Standard Model)")
    print("-" * 55)
    print(f"  Max min_eps observed : {max_me:.4f}")
    print(f"  10% threshold        : {threshold:.4f}")
    if onset_epoch is not None:
        onset_frac = onset_epoch / EPOCHS
        print(f"  Robustness onset at  : epoch {onset_epoch} "
              f"({100*onset_frac:.0f}% through training)")
        if onset_frac > 0.5:
            print("  => DELAYED onset: robustness does not begin until the "
                  "second half of training.")
        elif onset_frac > 0.3:
            print("  => MODERATE delay: robustness begins in the middle third "
                  "of training.")
        else:
            print("  => EARLY onset: robustness begins to emerge early in training.")
    else:
        print("  => No meaningful robustness onset detected.")

    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    std_rho = spearman_corr(
        [r[0] for r in results["standard"]],
        [r[4] for r in results["standard"]]
    )
    at_rho = spearman_corr(
        [r[0] for r in results["pgdat"]],
        [r[4] for r in results["pgdat"]]
    )

    print("Summary")
    print("=" * 70)
    print(f"  Standard model Spearman rho (epoch vs min_eps): {std_rho:+.4f}")
    print(f"  PGD-AT   model Spearman rho (epoch vs min_eps): {at_rho:+.4f}")
    print()
    if std_rho > 0.7:
        print("  Standard model shows a STRONG positive trend: robustness grows "
              "monotonically with training time.")
    elif std_rho > 0.3:
        print("  Standard model shows a MODERATE positive trend: robustness "
              "tends to improve but not monotonically.")
    elif std_rho < -0.3:
        print("  Standard model shows a NEGATIVE trend: longer training may "
              "reduce robustness (overfitting?).")
    else:
        print("  Standard model shows NO clear trend: robustness is largely "
              "independent of training duration.")

    if at_rho > std_rho + 0.2:
        print("  PGD-AT yields a STRONGER monotonic robustness trend than "
              "standard training.")
    elif at_rho < std_rho - 0.2:
        print("  PGD-AT shows WEAKER monotonic trend: adversarial training "
              "dynamics differ from standard training.")
    else:
        print("  Both models show SIMILAR robustness trends over training time.")

    # Final AUROC-style metric: mean min_eps of final checkpoint for each model
    final_std = results["standard"][-1][4]
    final_at = results["pgdat"][-1][4]
    print()
    print(f"  Final mean min_eps (standard): {final_std:.4f}")
    print(f"  Final mean min_eps (PGD-AT)  : {final_at:.4f}")
    print()
    print("AUROC proxy (mean min_eps at final epoch):")
    print(f"  standard={final_std:.4f}  pgdat={final_at:.4f}")


if __name__ == "__main__":
    main()
