"""
Hypothesis H153: BatchNorm Running Stats Sensitivity.

Train a small CNN on Fashion-MNIST for 10 epochs WITH batchnorm added to the convolutional layers.
For each test sample:
- Compute predictions in eval mode (using BN running stats).
- Compute predictions in train mode (using batch size 1 stats, which makes BN estimate mean/var from that single sample).
- Feature 'bn_drift': L2 norm of the difference between eval logits and train logits.

Features:
- bn_drift (||eval_logits - train_logits||_2)
- margin (logits margin in eval mode)
- mean_pix (mean pixel value)
- std_pix (std pixel value)

Targets:
- flipped_FGSM
- flipped_PGD
- min_eps (binarized at median)
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


class CNN_BN(nn.Module):
    """CNN with BatchNorm layers added to Conv layers."""
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


def train_model(train_set, test_set, seed=0):
    """Train CNN WITH BatchNorm on Fashion-MNIST for 10 epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    model = CNN_BN(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch+1}/{EPOCHS} completed")

    return model, test_x, test_y


def compute_bn_drift_features(x, model):
    """Compute BN drift feature, eval margin, and pixel statistics."""
    N = x.size(0)

    # 1. Eval mode logits
    model.eval()
    with torch.no_grad():
        eval_logits = model(x)
        sorted_logits, _ = eval_logits.sort(dim=1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]

    # 2. Train mode logits (batch_size = 1)
    model.train()
    train_logits_list = []
    with torch.no_grad():
        for idx in range(N):
            x_i = x[idx].unsqueeze(0)  # Shape (1, 1, 28, 28)
            logits_i = model(x_i)      # Uses single-sample stats
            train_logits_list.append(logits_i)

    train_logits = torch.cat(train_logits_list, dim=0)

    # BN Drift L2 norm
    bn_drift = torch.norm(eval_logits - train_logits, p=2, dim=1)

    # Image statistics
    flat_x = x.view(N, -1)
    mean_pix = flat_x.mean(1)
    std_pix = flat_x.std(1)

    return (
        bn_drift.cpu().numpy(),
        margin.cpu().numpy(),
        mean_pix.cpu().numpy(),
        std_pix.cpu().numpy()
    )


# --- Attack Implementations ---

def fgsm_grad_sign(model, x, y, eps=EPS):
    """FGSM gradient computed in eval mode."""
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    return x_adv.grad.sign().detach()


def attack_fgsm(model, x, y, eps=EPS):
    model.eval()
    sign = fgsm_grad_sign(model, x, y, eps)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return model(adv).argmax(1) != y


def attack_pgd(model, x, y, eps=EPS, steps=10):
    model.eval()
    alpha = eps / 5.0
    x_adv = x.clone().detach() + (torch.rand_like(x) * 2 - 1) * eps
    x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.sign().detach()
        x_adv = torch.clamp(x_adv + alpha * grad, x - eps, x + eps).clamp(0, 1).detach()
    with torch.no_grad():
        return model(x_adv).argmax(1) != y


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Binary search for minimum FGSM perturbation to flip classification."""
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)

    sign = fgsm_grad_sign(model, x, y, eps_max)

    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def main():
    print("=" * 70)
    print("Hypothesis H153: BatchNorm Running Stats Sensitivity")
    print("=" * 70)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("\nTraining CNN model with BatchNorm...")
    t0 = time.time()
    model, test_x, test_y = train_model(train_set, test_set)
    print(f"Training completed in {time.time() - t0:.1f}s")

    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        pred = logits.argmax(1)
        correct = (pred == test_y)

    x_c = test_x[correct][:1000]
    y_c = test_y[correct][:1000]
    N = x_c.size(0)
    print(f"\nUsing {N} correctly classified samples for evaluation.")

    # Compute BN drift and other features
    print("\nComputing BN drift features (eval vs train mode at batch=1)...")
    bn_drift, margin, mean_pix, std_pix = compute_bn_drift_features(x_c, model)

    # Run attacks
    print("\nEvaluating attack outcomes...")
    batch_size = 256
    fgsm_success = []
    pgd_success = []
    min_eps_list = []

    for i in range(0, N, batch_size):
        xb = x_c[i:i+batch_size]
        yb = y_c[i:i+batch_size]

        fgsm_success.append(attack_fgsm(model, xb, yb))
        pgd_success.append(attack_pgd(model, xb, yb))
        min_eps_list.append(min_eps_to_flip(model, xb, yb))

    fgsm_success = torch.cat(fgsm_success)
    pgd_success = torch.cat(pgd_success)
    min_eps = torch.cat(min_eps_list)

    # Convert targets to numpy
    flipped_FGSM = fgsm_success.cpu().numpy().astype(int)
    flipped_PGD = pgd_success.cpu().numpy().astype(int)

    min_eps_np = min_eps.cpu().numpy()
    median_eps = np.median(min_eps_np)
    vulnerable_min_eps = (min_eps_np < median_eps).astype(int)

    targets = {
        "flipped_FGSM": flipped_FGSM,
        "flipped_PGD": flipped_PGD,
        "min_eps_below_median": vulnerable_min_eps
    }

    features = {
        "bn_drift": bn_drift,
        "margin": margin,
        "mean_pix": mean_pix,
        "std_pix": std_pix
    }

    # Print stats
    print("\nTarget rates:")
    for tname, tval in targets.items():
        print(f"  {tname:<25}: positive rate = {tval.mean():.3f}")

    # Univariate AUROC analysis
    print("\n" + "=" * 60)
    print("Univariate AUROC Evaluation")
    print("=" * 60)

    for tname, y in targets.items():
        if y.std() == 0:
            print(f"\nTarget {tname} has no variance. Skipping AUROC.")
            continue
        print(f"\n--- Target: {tname} ---")
        for fname, x_i in features.items():
            a = roc_auc_score(y, x_i)
            a_best = max(a, 1 - a)
            direction = "+" if a >= 0.5 else "-"
            print(f"  {fname:<25} AUROC = {a_best:.4f} (direction: {direction})")


if __name__ == "__main__":
    main()
