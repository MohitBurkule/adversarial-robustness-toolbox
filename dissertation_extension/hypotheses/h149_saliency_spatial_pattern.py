"""
Hypothesis H149: Saliency Spatial Pattern.

Train a small CNN on Fashion-MNIST for 10 epochs.
For each test sample, compute the saliency map |grad_x L|.
Compute spatial pattern features from this saliency map:
- Center-of-mass distance from the image center (28x28 grid, center = 13.5).
- Moment of inertia of the saliency distribution.
- Principal-axis eccentricity of the saliency map.

Features:
- saliency_cm_dist
- saliency_moment_inertia
- saliency_eccentricity
- margin (logits margin)
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


def train_model(train_set, test_set, seed=0):
    """Train CNN on Fashion-MNIST for 10 epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    model = CNN(N_CLASSES).to(DEVICE)
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


def compute_saliency_features(x, y, model):
    """
    Compute input gradient saliency features:
    - center-of-mass distance from (13.5, 13.5)
    - moment of inertia
    - principal-axis eccentricity
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    loss.backward()
    grad = x.grad.abs().detach().squeeze(1)  # Shape (N, 28, 28)

    N = grad.size(0)

    # Add a small epsilon to avoid division by zero
    saliency_sum = grad.sum(dim=(1, 2), keepdim=True) + 1e-9
    P = grad / saliency_sum  # Shape (N, 28, 28) Normalized spatial probability map

    # Coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(28, dtype=torch.float, device=DEVICE),
        torch.arange(28, dtype=torch.float, device=DEVICE),
        indexing="ij"
    )

    # Expand grids for batch matching
    x_coords = x_coords.unsqueeze(0)  # Shape (1, 28, 28)
    y_coords = y_coords.unsqueeze(0)  # Shape (1, 28, 28)

    # 1. Center of mass
    x_c = (P * x_coords).sum(dim=(1, 2))  # Shape (N,)
    y_c = (P * y_coords).sum(dim=(1, 2))  # Shape (N,)
    cm_dist = torch.sqrt((x_c - 13.5) ** 2 + (y_c - 13.5) ** 2)

    # 2. Central moments
    # Reshape coordinates to (N, 28, 28) for correct broadcast with x_c/y_c
    x_diff = x_coords - x_c.view(-1, 1, 1)
    y_diff = y_coords - y_c.view(-1, 1, 1)

    mu_20 = (P * (x_diff ** 2)).sum(dim=(1, 2))  # Variance in X
    mu_02 = (P * (y_diff ** 2)).sum(dim=(1, 2))  # Variance in Y
    mu_11 = (P * (x_diff * y_diff)).sum(dim=(1, 2))  # Covariance

    # Moment of inertia: trace of the covariance tensor
    moment_inertia = mu_20 + mu_02

    # 3. Eccentricity from eigenvalues of the 2D spatial covariance matrix
    # Trace = mu_20 + mu_02
    # Det = mu_20 * mu_02 - mu_11^2
    # lambda_1, lambda_2 = (Trace +- sqrt(Trace^2 - 4*Det)) / 2
    # which simplifies to: (mu_20 + mu_02 +- sqrt((mu_20 - mu_02)^2 + 4 * mu_11^2)) / 2
    eig_diff = torch.sqrt((mu_20 - mu_02) ** 2 + 4 * (mu_11 ** 2) + 1e-9)
    lambda_max = 0.5 * (mu_20 + mu_02 + eig_diff)
    lambda_min = 0.5 * (mu_20 + mu_02 - eig_diff)

    # Eccentricity: sqrt(1 - lambda_min / lambda_max)
    eccentricity = torch.sqrt(1.0 - (lambda_min / (lambda_max + 1e-9)) + 1e-9)

    # Baseline features
    with torch.no_grad():
        sorted_logits, _ = outputs.detach().sort(1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]

    flat_x = x.view(N, -1).detach()
    mean_pix = flat_x.mean(1)
    std_pix = flat_x.std(1)

    return (
        cm_dist.detach(),
        moment_inertia.detach(),
        eccentricity.detach(),
        margin.detach(),
        mean_pix,
        std_pix
    )


# --- Attack Implementations ---

def attack_fgsm(model, x, y, eps=EPS):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    grad = x_adv.grad.sign().detach()
    adv = (x + eps * grad).clamp(0, 1)
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

    x_grad = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_grad), y).backward()
    sign = x_grad.grad.sign().detach()

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
    print("Hypothesis H149: Saliency Spatial Pattern Analysis")
    print("=" * 70)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("\nTraining model...")
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

    # Compute saliency spatial features
    print("\nComputing saliency and spatial features...")
    cm_dist, moment_inertia, eccentricity, margin, mean_pix, std_pix = compute_saliency_features(x_c, y_c, model)

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

    # Compute binarized targets
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
        "saliency_cm_dist": cm_dist.cpu().numpy(),
        "saliency_moment_inertia": moment_inertia.cpu().numpy(),
        "saliency_eccentricity": eccentricity.cpu().numpy(),
        "margin": margin.cpu().numpy(),
        "mean_pix": mean_pix.cpu().numpy(),
        "std_pix": std_pix.cpu().numpy()
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
