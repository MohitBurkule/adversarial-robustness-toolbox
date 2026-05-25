"""
H162: Class-level margin shift under adversarial training (AT).

Hypothesis: Adversarial training is known to reduce clean accuracy. Does this
clean accuracy loss fall disproportionately on structurally harder classes —
those with lower average vanilla margin (closer to the decision boundary)?
Specifically: is the per-class accuracy drop after PGD-AT predictable from the
per-class margin distribution of the vanilla model?

Procedure:
  1. Train a vanilla CNN for 10 epochs (seed=0).
  2. Train a PGD-AT CNN for 10 epochs using PGD-7 during training (seed=0).
  3. Evaluate both models on the full 10,000-sample test set:
       - Per-class accuracy (vanilla and AT).
       - Per-class mean margin (vanilla and AT).
         Margin = logit of correct class minus highest competing logit.
  4. Compute accuracy drop per class: vanilla_acc[c] - at_acc[c].
  5. Compute Spearman and Pearson correlations:
       - vanilla per-class mean margin vs. accuracy drop.
       - vanilla per-class accuracy vs. accuracy drop.
  6. Report top-3 most-hurt classes with their vanilla margin rank.
  7. Report confusion matrix changes: class pairs newly or more-confused after AT.

Dataset: Fashion-MNIST (patched in via patch_dataset.py when needed).
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import spearmanr, pearsonr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


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


def pgd_attack(model, x, y, eps=EPS, alpha=2 / 255, steps=7):
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        x_adv = (x_adv + alpha * x_adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return x_adv.detach()


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


def train_vanilla(model, loader, optimizer):
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


def train_at(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = pgd_attack(model, x, y)
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def compute_per_class_stats(model, loader):
    """Return per-class accuracy, per-class mean margin, and confusion matrix."""
    model.eval()
    correct = np.zeros(N_CLASSES, dtype=int)
    total = np.zeros(N_CLASSES, dtype=int)
    margin_sum = np.zeros(N_CLASSES, dtype=float)
    margin_count = np.zeros(N_CLASSES, dtype=int)
    conf_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)  # [true, pred]

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)

            y_np = y.cpu().numpy()
            preds_np = preds.cpu().numpy()
            logits_np = logits.cpu().numpy()

            for i in range(len(y_np)):
                true_c = y_np[i]
                pred_c = preds_np[i]
                total[true_c] += 1
                if pred_c == true_c:
                    correct[true_c] += 1
                conf_matrix[true_c, pred_c] += 1

                # Margin: logit of correct class minus max competing logit
                row = logits_np[i].copy()
                correct_logit = row[true_c]
                row[true_c] = -np.inf
                best_other = row.max()
                margin = correct_logit - best_other
                margin_sum[true_c] += margin
                margin_count[true_c] += 1

    acc = correct / total
    mean_margin = margin_sum / np.maximum(margin_count, 1)
    return acc, mean_margin, conf_matrix


def confusion_changes(conf_v, conf_at, top_n=5):
    """Return top class pairs where AT increased confusion vs vanilla."""
    # Normalise each row to per-true-class rate
    def row_norm(cm):
        row_sums = cm.sum(axis=1, keepdims=True).clip(1, None)
        return cm / row_sums

    rate_v = row_norm(conf_v)
    rate_at = row_norm(conf_at)
    delta = rate_at - rate_v  # positive = AT confused more

    changes = []
    for true_c in range(N_CLASSES):
        for pred_c in range(N_CLASSES):
            if true_c == pred_c:
                continue
            d = delta[true_c, pred_c]
            if d > 0:
                changes.append((true_c, pred_c, d, rate_v[true_c, pred_c], rate_at[true_c, pred_c]))

    changes.sort(key=lambda t: -t[2])
    return changes[:top_n]


def main():
    print(f"H162: Class-level margin shift under adversarial training")
    print(f"Device: {DEVICE}")
    print()

    train_loader, test_loader = get_loaders()

    # --- Train vanilla model ---
    print("=== Training Vanilla CNN ===")
    set_seed(0)
    vanilla = CNN(N_CLASSES).to(DEVICE)
    opt_v = torch.optim.Adam(vanilla.parameters(), lr=1e-3)
    for ep in range(1, EPOCHS + 1):
        loss = train_vanilla(vanilla, train_loader, opt_v)
        print(f"  Epoch {ep:2d}/{EPOCHS}  train_loss={loss:.4f}")
    print()

    # --- Train PGD-AT model ---
    print("=== Training PGD-AT CNN (PGD-7, eps=15/255, alpha=2/255) ===")
    set_seed(0)
    at_model = CNN(N_CLASSES).to(DEVICE)
    opt_at = torch.optim.Adam(at_model.parameters(), lr=1e-3)
    for ep in range(1, EPOCHS + 1):
        loss = train_at(at_model, train_loader, opt_at)
        print(f"  Epoch {ep:2d}/{EPOCHS}  adv_train_loss={loss:.4f}")
    print()

    # --- Evaluate ---
    print("=== Evaluating on full test set (10,000 samples) ===")
    vanilla.eval()
    at_model.eval()
    acc_v, margin_v, conf_v = compute_per_class_stats(vanilla, test_loader)
    acc_at, margin_at, conf_at = compute_per_class_stats(at_model, test_loader)
    print()

    acc_drop = acc_v - acc_at

    # --- Per-class table ---
    print("=== Per-Class Results ===")
    header = (
        f"{'Class':<12} {'Vanilla Acc':>11} {'AT Acc':>8} {'Acc Drop':>10} "
        f"{'Van Margin':>11} {'AT Margin':>10}"
    )
    print(header)
    print("-" * len(header))
    for c in range(N_CLASSES):
        print(
            f"{CLASS_NAMES[c]:<12} {acc_v[c]:>11.4f} {acc_at[c]:>8.4f} "
            f"{acc_drop[c]:>10.4f} {margin_v[c]:>11.4f} {margin_at[c]:>10.4f}"
        )
    print()

    # Overall accuracy
    overall_v = acc_v.mean()
    overall_at = acc_at.mean()
    print(f"Overall mean class accuracy — Vanilla: {overall_v:.4f}  AT: {overall_at:.4f}  "
          f"Drop: {overall_v - overall_at:.4f}")
    print()

    # --- Top-3 most-hurt classes ---
    hurt_order = np.argsort(-acc_drop)  # descending by drop
    margin_rank_v = np.argsort(np.argsort(margin_v))  # rank 0 = lowest margin

    print("=== Top-3 Most-Hurt Classes (largest accuracy drop) ===")
    for rank_i, c in enumerate(hurt_order[:3], 1):
        print(
            f"  {rank_i}. {CLASS_NAMES[c]:<12}  acc_drop={acc_drop[c]:.4f}  "
            f"vanilla_margin={margin_v[c]:.4f}  "
            f"vanilla_margin_rank={margin_rank_v[c]} (0=lowest margin)"
        )
    print()

    # --- Correlations ---
    # 1. Vanilla per-class mean margin vs accuracy drop
    sp1, sp1_p = spearmanr(margin_v, acc_drop)
    pe1, pe1_p = pearsonr(margin_v, acc_drop)
    print("=== Correlation: Vanilla Per-Class Mean Margin vs Accuracy Drop ===")
    print(f"  Spearman rho = {sp1:.4f}  (p={sp1_p:.4f})")
    print(f"  Pearson  r   = {pe1:.4f}  (p={pe1_p:.4f})")
    if sp1 < -0.3:
        print("  Interpretation: SUPPORTED — lower vanilla margin -> larger accuracy drop after AT.")
    elif sp1 > 0.3:
        print("  Interpretation: INVERSE — higher vanilla margin -> larger accuracy drop after AT.")
    else:
        print("  Interpretation: NO CLEAR relationship between vanilla margin and accuracy drop.")
    print()

    # 2. Vanilla per-class accuracy vs accuracy drop
    sp2, sp2_p = spearmanr(acc_v, acc_drop)
    pe2, pe2_p = pearsonr(acc_v, acc_drop)
    print("=== Correlation: Vanilla Per-Class Accuracy vs Accuracy Drop ===")
    print(f"  Spearman rho = {sp2:.4f}  (p={sp2_p:.4f})")
    print(f"  Pearson  r   = {pe2:.4f}  (p={pe2_p:.4f})")
    if sp2 < -0.3:
        print("  Interpretation: SUPPORTED — lower vanilla accuracy -> larger accuracy drop after AT.")
    elif sp2 > 0.3:
        print("  Interpretation: Higher vanilla accuracy -> larger accuracy drop after AT.")
    else:
        print("  Interpretation: NO CLEAR relationship between vanilla accuracy and accuracy drop.")
    print()

    # --- Confusion matrix changes ---
    top_changes = confusion_changes(conf_v, conf_at, top_n=5)
    print("=== Top-5 Class Pairs Newly/More Confused After AT ===")
    print(f"  (true -> predicted, delta=rate_AT - rate_vanilla)")
    for true_c, pred_c, delta, rate_v_val, rate_at_val in top_changes:
        print(
            f"  {CLASS_NAMES[true_c]:<12} -> {CLASS_NAMES[pred_c]:<12}  "
            f"vanilla={rate_v_val:.4f}  AT={rate_at_val:.4f}  delta=+{delta:.4f}"
        )
    print()

    # --- Summary ---
    print("=== Summary ===")
    print(f"  AT reduces mean class accuracy by {overall_v - overall_at:.4f}.")
    print(
        f"  Spearman(vanilla_margin, acc_drop) = {sp1:.4f}: "
        + ("lower-margin classes are disproportionately hurt."
           if sp1 < -0.3 else
           "no strong margin-based pattern." if abs(sp1) <= 0.3 else
           "higher-margin classes are paradoxically more hurt.")
    )
    print(
        f"  Spearman(vanilla_accuracy, acc_drop) = {sp2:.4f}: "
        + ("harder classes (lower vanilla acc) suffer more."
           if sp2 < -0.3 else
           "no strong accuracy-based pattern." if abs(sp2) <= 0.3 else
           "easier classes (higher vanilla acc) suffer more.")
    )


if __name__ == "__main__":
    main()
