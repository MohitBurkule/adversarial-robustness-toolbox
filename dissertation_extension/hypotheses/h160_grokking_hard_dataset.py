"""
H160: Grokking-Like Delayed Robustness on a Harder Dataset (CIFAR-100)

Hypothesis: On a harder classification problem (CIFAR-100, 100 classes),
adversarial robustness may follow a qualitatively different trajectory from
clean accuracy — specifically, robustness could emerge *after* clean accuracy
has already plateaued, analogous to the "grokking" phenomenon where
generalisation is delayed relative to training loss convergence.

This script trains a wider CNN on CIFAR-100 for 60 epochs. Every 5 epochs
(12 checkpoints) we evaluate on 300 test samples:
  - Training accuracy (1000 random training samples)
  - Test clean accuracy
  - FGSM attack success rate (eps=8/255)
  - PGD-10 attack success rate (eps=8/255)
  - Mean min_eps_to_flip (binary search, 8 iters, hi=0.15)

We then identify:
  1. The epoch at which clean accuracy plateaus (less than 1% gain over 10 ep)
  2. The epoch at which adversarial robustness (mean min_eps) peaks or starts
     growing meaningfully (exceeds 20% of its eventual maximum)
  3. The ratio of mean min_eps at epoch 60 vs epoch 5 — how much does
     robustness grow relative to the early period?

A large gap between the clean-accuracy plateau epoch and the robustness
onset epoch is the "grokking-like delayed robustness" signature.

Note: This script uses datasets.CIFAR100 directly — it does NOT use
datasets.FashionMNIST (which is monkey-patched elsewhere in this project).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
BATCH = 128
EPS = 8.0 / 255.0          # CIFAR standard
PGD_ALPHA = EPS / 4.0
PGD_STEPS = 10
CHECKPOINT_EVERY = 5        # 12 checkpoints total
N_EVAL = 300                # test samples per checkpoint
N_TRAIN_EVAL = 1000         # training samples for train-acc estimate
BINARY_SEARCH_ITERS = 8
BINARY_HI = 0.15


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class CNN100(nn.Module):
    """Wider CNN for CIFAR-100 (3-channel input, 100 classes)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),   # 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),   # 4x4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 100),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def fgsm(model, x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_adv), y).backward()
    return (x_adv + eps * x_adv.grad.sign()).clamp(0.0, 1.0).detach()


def pgd(model, x, y, eps, alpha, steps):
    x_adv = x.clone().detach() + torch.zeros_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        F.cross_entropy(model(x_adv), y).backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0.0, 1.0)
    return x_adv.detach()


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_checkpoint(model, x_eval, y_eval):
    """Return (clean_acc, fgsm_sr, pgd_sr, mean_min_eps)."""
    model.eval()

    with torch.no_grad():
        logits = model(x_eval)
        preds = logits.argmax(1)
        clean_acc = (preds == y_eval).float().mean().item()

    correct_mask = preds == y_eval
    if correct_mask.sum() == 0:
        return clean_acc, 0.0, 0.0, 0.0

    xc = x_eval[correct_mask]
    yc = y_eval[correct_mask]

    # FGSM success rate
    x_fgsm = fgsm(model, xc, yc, EPS)
    with torch.no_grad():
        fgsm_sr = (model(x_fgsm).argmax(1) != yc).float().mean().item()

    # PGD success rate
    x_pgd = pgd(model, xc, yc, EPS, PGD_ALPHA, PGD_STEPS)
    with torch.no_grad():
        pgd_sr = (model(x_pgd).argmax(1) != yc).float().mean().item()

    # Mean min_eps_to_flip via binary search along FGSM gradient direction
    min_eps_vals = []
    for i in range(len(xc)):
        xi = xc[i:i+1]
        yi = yc[i:i+1]
        lo, hi = 0.0, BINARY_HI
        for _ in range(BINARY_SEARCH_ITERS):
            mid = (lo + hi) / 2.0
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

    mean_min_eps = sum(min_eps_vals) / len(min_eps_vals)
    return clean_acc, fgsm_sr, pgd_sr, mean_min_eps


def train_accuracy(model, x_train_sample, y_train_sample):
    model.eval()
    with torch.no_grad():
        preds = model(x_train_sample).argmax(1)
        return (preds == y_train_sample).float().mean().item()


# ---------------------------------------------------------------------------
# Plateau detection helpers
# ---------------------------------------------------------------------------
def find_plateau_epoch(rows, window=2, threshold=0.01):
    """Return the first epoch where clean acc gain over next `window` checkpoints
    is below `threshold` (i.e., it has plateaued)."""
    for i in range(len(rows) - window):
        gain = rows[i + window][2] - rows[i][2]  # col 2 = clean_acc
        if gain < threshold:
            return rows[i][0]  # epoch number
    return rows[-1][0]


def find_robustness_onset_epoch(rows, fraction=0.20):
    """Return the first epoch where mean_min_eps >= fraction * max(mean_min_eps)."""
    max_me = max(r[4] for r in rows)
    threshold = fraction * max_me
    for r in rows:
        if r[4] >= threshold:
            return r[0]
    return rows[-1][0]


def spearman_corr(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(vals):
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        for rank, idx in enumerate(sorted_idx, 1):
            r[idx] = float(rank)
        return r

    rx, ry = ranks(xs), ranks(ys)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("H160: Grokking-Like Delayed Robustness on CIFAR-100")
    print("=" * 70)
    print(f"Device       : {DEVICE}")
    print(f"Epochs       : {EPOCHS}  (checkpoints every {CHECKPOINT_EVERY})")
    print(f"Eval samples : {N_EVAL} test + {N_TRAIN_EVAL} train per checkpoint")
    print(f"EPS          : {EPS:.5f}  ({round(EPS*255)}/255)")
    print(f"Binary hi    : {BINARY_HI}  iters={BINARY_SEARCH_ITERS}")
    print()

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100("./data", train=True, download=True,
                                  transform=train_tf)
    test_set = datasets.CIFAR100("./data", train=False, download=True,
                                 transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              num_workers=2, pin_memory=True)

    # Fixed test eval subset (same 300 samples at every checkpoint)
    eval_subset = Subset(test_set, list(range(N_EVAL)))
    eval_loader = DataLoader(eval_subset, batch_size=N_EVAL, shuffle=False)
    x_eval, y_eval = next(iter(eval_loader))
    x_eval, y_eval = x_eval.to(DEVICE), y_eval.to(DEVICE)

    # Fixed training eval subset (1000 samples, no augmentation — use test_tf
    # on the training set via a separate loader with deterministic subset)
    train_eval_set = datasets.CIFAR100("./data", train=True, download=False,
                                       transform=test_tf)
    train_eval_subset = Subset(train_eval_set, list(range(N_TRAIN_EVAL)))
    train_eval_loader = DataLoader(train_eval_subset, batch_size=N_TRAIN_EVAL,
                                   shuffle=False)
    x_train_eval, y_train_eval = next(iter(train_eval_loader))
    x_train_eval = x_train_eval.to(DEVICE)
    y_train_eval = y_train_eval.to(DEVICE)

    model = CNN100().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # rows: (epoch, train_acc, clean_acc, fgsm_sr, pgd_sr, mean_min_eps)
    rows = []

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer)
        scheduler.step()

        if epoch % CHECKPOINT_EVERY == 0:
            print(f"  Evaluating epoch {epoch:2d}...", end=" ", flush=True)
            tr_acc = train_accuracy(model, x_train_eval, y_train_eval)
            clean_acc, fgsm_sr, pgd_sr, mean_min_eps = evaluate_checkpoint(
                model, x_eval, y_eval
            )
            rows.append((epoch, tr_acc, clean_acc, fgsm_sr, pgd_sr, mean_min_eps))
            print(
                f"train={tr_acc:.3f}  clean={clean_acc:.3f}  "
                f"fgsm_sr={fgsm_sr:.3f}  pgd_sr={pgd_sr:.3f}  "
                f"min_eps={mean_min_eps:.4f}"
            )

    # -----------------------------------------------------------------------
    # Trajectory table
    # -----------------------------------------------------------------------
    print()
    print("Full Trajectory Table — CIFAR-100 Standard Training")
    header = (f"{'Epoch':>5}  {'TrainAcc':>8}  {'CleanAcc':>8}  "
              f"{'FGSM_SR':>7}  {'PGD_SR':>6}  {'MinEps':>7}")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for epoch, tr, ca, fs, ps, me in rows:
        print(f"{epoch:>5}  {tr:>8.3f}  {ca:>8.3f}  "
              f"{fs:>7.3f}  {ps:>6.3f}  {me:>7.4f}")
    print(sep)
    print()

    # -----------------------------------------------------------------------
    # Clean accuracy plateau detection
    # -----------------------------------------------------------------------
    plateau_epoch = find_plateau_epoch(rows, window=2, threshold=0.01)
    robustness_onset_epoch = find_robustness_onset_epoch(rows, fraction=0.20)

    max_clean = max(r[2] for r in rows)
    max_min_eps = max(r[5] for r in rows)

    print("Grokking-Like Delayed Robustness Analysis")
    print("-" * 55)
    print(f"  Clean accuracy plateau at epoch  : {plateau_epoch}")
    print(f"  Robustness onset (20% max) epoch : {robustness_onset_epoch}")
    delay = robustness_onset_epoch - plateau_epoch
    print(f"  Delay (onset - plateau)          : {delay} epochs")
    if delay > 10:
        print("  => STRONG delayed robustness: robustness lags clean accuracy "
              "by more than 10 epochs — grokking-like signature detected.")
    elif delay > 0:
        print("  => MILD delay: robustness emerges slightly after clean "
              "accuracy plateau.")
    elif delay == 0:
        print("  => NO delay: robustness onset coincides with clean accuracy "
              "plateau.")
    else:
        print("  => ROBUSTNESS LEADS: robustness onset precedes clean accuracy "
              "plateau (unusual).")
    print()

    # -----------------------------------------------------------------------
    # Robustness growth ratio: epoch 60 vs epoch 5
    # -----------------------------------------------------------------------
    first_row = rows[0]   # epoch 5
    last_row = rows[-1]   # epoch 60
    eps_early = first_row[5]
    eps_final = last_row[5]
    if eps_early > 0:
        growth_ratio = eps_final / eps_early
    else:
        growth_ratio = float("inf")

    print("Robustness Growth Ratio (epoch 60 vs epoch 5)")
    print("-" * 55)
    print(f"  Mean min_eps at epoch  5 : {eps_early:.4f}")
    print(f"  Mean min_eps at epoch 60 : {eps_final:.4f}")
    print(f"  Ratio (final / early)    : {growth_ratio:.3f}x")
    if growth_ratio > 2.0:
        print("  => Large robustness growth over training (>2x). "
              "Late-phase dynamics contribute substantially.")
    elif growth_ratio > 1.2:
        print("  => Moderate robustness growth over training (1.2x–2x).")
    else:
        print("  => Robustness is relatively stable across training; "
              "no strong late-phase growth.")
    print()

    # -----------------------------------------------------------------------
    # Spearman correlations
    # -----------------------------------------------------------------------
    epoch_list = [r[0] for r in rows]
    clean_list = [r[2] for r in rows]
    min_eps_list = [r[5] for r in rows]

    rho_clean = spearman_corr(epoch_list, clean_list)
    rho_rob = spearman_corr(epoch_list, min_eps_list)

    print("Spearman Rank Correlations (Epoch vs Metric)")
    print("-" * 55)
    print(f"  Epoch vs clean accuracy   : rho = {rho_clean:+.4f}")
    print(f"  Epoch vs mean min_eps     : rho = {rho_rob:+.4f}")
    if rho_rob < rho_clean - 0.2:
        print("  => Robustness grows LESS monotonically than clean accuracy — "
              "consistent with delayed / non-uniform robustness emergence.")
    elif rho_rob > 0.7:
        print("  => Robustness grows strongly and monotonically with training.")
    else:
        print("  => No strong monotonic trend in robustness across epochs.")
    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("Summary")
    print("=" * 70)
    print(f"  Dataset               : CIFAR-100 (100 classes)")
    print(f"  Epochs trained        : {EPOCHS}")
    print(f"  Max clean accuracy    : {max_clean:.3f}")
    print(f"  Max mean min_eps      : {max_min_eps:.4f}")
    print(f"  Clean plateau epoch   : {plateau_epoch}")
    print(f"  Robustness onset epoch: {robustness_onset_epoch}")
    print(f"  Delay                 : {delay} epochs")
    print(f"  Robustness growth ratio (ep60/ep5): {growth_ratio:.3f}x")
    print(f"  Spearman rho (epoch vs min_eps)   : {rho_rob:+.4f}")
    print()
    if delay > 10 and growth_ratio > 1.5:
        print("  CONCLUSION: Strong evidence of grokking-like delayed robustness "
              "on CIFAR-100. Clean accuracy plateaus well before adversarial "
              "robustness meaningfully emerges.")
    elif delay > 0 and growth_ratio > 1.2:
        print("  CONCLUSION: Moderate evidence of delayed robustness. "
              "Robustness continues to improve after clean accuracy stops growing.")
    else:
        print("  CONCLUSION: No clear grokking-like delayed robustness observed. "
              "Robustness and clean accuracy evolve similarly over training.")


if __name__ == "__main__":
    main()
