"""
Hypothesis H147: Margin-weighted Adversarial Training on Fashion-MNIST.

Train two victims on Fashion-MNIST for 10 epochs each:
  1. Standard PGD-AT: adversarial training where all samples in the batch have equal weight.
  2. Margin-weighted PGD-AT: adversarial training where the adversarial loss for each sample
     is weighted by 1 / (clean_margin + 0.1).

Per victim features:
  - margin
  - mean_pix
  - std_pix

Per victim targets:
  - flipped_FGSM
  - flipped_PGD
  - min_eps (min epsilon to flip via binary search)

Analysis:
  - Compare accuracy and robustness of Standard PGD-AT vs Margin-weighted PGD-AT.
  - Compare per-feature AUROC for predicting vulnerability in both models.
"""
import time
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


class CNN(nn.Module):
    """Small CNN matching diagnostic_test.py architecture."""
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


# ---- Attack Functions ----
def pgd_attack(model, x, y, eps=EPS, alpha=2.0/255.0, steps=10, random_start=True):
    model.eval()
    adv = x.clone().detach().requires_grad_(True)
    if random_start:
        adv = adv + torch.FloatTensor(*adv.shape).uniform_(-eps, eps).to(DEVICE)
        adv = adv.clamp(0, 1)

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        adv = (adv + alpha * adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return adv.detach()


def fgsm_attack(model, x, y, eps=EPS):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    sign = x_adv.grad.sign().detach()
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    model.eval()
    N = x.size(0)
    lo = torch.zeros(N, device=DEVICE)
    hi = torch.full((N,), eps_max, device=DEVICE)

    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    sign = x_adv.grad.sign().detach()

    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = (model(adv).argmax(1) != y)
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# ---- Training Functions ----
def train_standard_at(train_set, seed=0):
    """Standard PGD-AT (equal weighting for all samples)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Generate adversarial examples
            x_adv = pgd_attack(model, x, y, eps=EPS, alpha=5.0/255.0, steps=2, random_start=True)

            model.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x_adv), y)
            loss.backward()
            optimizer.step()
    return model


def train_margin_weighted_at(train_set, seed=0):
    """Margin-weighted PGD-AT: loss weighted by 1 / (clean_margin + 0.1)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Compute clean margins for sample-weighting
            with torch.no_grad():
                model.eval()
                clean_logits = model(x)
                N = x.size(0)
                correct_logits = clean_logits[torch.arange(N), y]
                # mask correct logit to find max other logit
                mask = torch.ones_like(clean_logits, dtype=torch.bool)
                mask[torch.arange(N), y] = False
                max_other_logits = clean_logits[mask].view(N, -1).max(dim=1)[0]
                margin = correct_logits - max_other_logits

                # Compute weights
                w = 1.0 / (margin + 0.1)
                # Normalize weights to have mean 1.0 in the batch for scale stability
                w = w / w.mean()

            # Generate adversarial examples
            x_adv = pgd_attack(model, x, y, eps=EPS, alpha=5.0/255.0, steps=2, random_start=True)

            model.train()
            optimizer.zero_grad()
            logits_adv = model(x_adv)
            loss_vec = F.cross_entropy(logits_adv, y, reduction='none')
            # Apply normalized weights
            loss = (loss_vec * w).mean()
            loss.backward()
            optimizer.step()
    return model


def evaluate_victim(model, x_test, y_test, name):
    print(f"\nEvaluating Model: {name}")
    model.eval()
    with torch.no_grad():
        preds = model(x_test).argmax(1)
        correct = (preds == y_test)

    x_c = x_test[correct]
    y_c = y_test[correct]
    N_correct = correct.sum().item()
    print(f"  Test Accuracy: {N_correct}/{x_test.size(0)} ({N_correct/x_test.size(0)*100:.2f}%)")

    # Compute Features
    with torch.no_grad():
        logits_c = model(x_c)
        sorted_logits, _ = logits_c.sort(1, descending=True)
        margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).cpu()

    flat_x = x_c.view(N_correct, -1)
    mean_pix = flat_x.mean(1).cpu()
    std_pix = flat_x.std(1).cpu()
    features = torch.stack([margin, mean_pix, std_pix], dim=1).numpy()

    # Compute Targets
    flipped_fgsm = []
    flipped_pgd = []
    min_eps = []

    for i in range(0, N_correct, 512):
        bx = x_c[i:i+512]
        by = y_c[i:i+512]
        flipped_fgsm.append(fgsm_attack(model, bx, by))
        flipped_pgd.append(pgd_attack(model, bx, by, steps=10))
        min_eps.append(min_eps_to_flip(model, bx, by))

    flipped_fgsm = torch.cat(flipped_fgsm).cpu().numpy().astype(int)
    flipped_pgd = torch.cat(flipped_pgd).cpu().numpy().astype(int)
    min_eps = torch.cat(min_eps).cpu().numpy()

    print(f"  FGSM flip rate = {flipped_fgsm.mean():.4f}")
    print(f"  PGD-10 flip rate = {flipped_pgd.mean():.4f}")
    print(f"  Mean min_eps = {min_eps.mean():.4f}")

    # Univariate AUROC Analysis
    feature_names = ["margin", "mean_pix", "std_pix"]
    for target_name, y in [("flipped_FGSM", flipped_fgsm), ("flipped_PGD", flipped_pgd)]:
        print(f"  Target: {target_name} | Univariate AUROC")
        if y.std() == 0:
            print("    Constant target, skipping.")
            continue
        for i, fname in enumerate(feature_names):
            x_i = features[:, i]
            a = roc_auc_score(y, x_i)
            a = max(a, 1 - a)
            print(f"    {fname:<10} AUROC = {a:.4f}")


def main():
    print("=" * 60)
    print("Hypothesis H147: Margin-weighted Adversarial Training")
    print("=" * 60)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # Train Standard PGD-AT Model
    print("\nTraining Standard PGD-AT victim...")
    t0 = time.time()
    std_at_model = train_standard_at(train_set)
    print(f"Standard PGD-AT trained in {time.time() - t0:.1f}s")

    # Train Margin-weighted PGD-AT Model
    print("\nTraining Margin-weighted PGD-AT victim...")
    t0 = time.time()
    weighted_at_model = train_margin_weighted_at(train_set)
    print(f"Margin-weighted PGD-AT trained in {time.time() - t0:.1f}s")

    # Evaluate both
    evaluate_victim(std_at_model, test_x, test_y, "Standard PGD-AT")
    evaluate_victim(weighted_at_model, test_x, test_y, "Margin-weighted PGD-AT")


if __name__ == "__main__":
    main()
