"""
Hypothesis H154: Representation Distance from Origin (Feature Norm Growth).

Train a small CNN on Fashion-MNIST for 10 epochs.
For each test sample:
- Compute the L2 norm of the representation (activations) at each of the 4 layers:
  1. Conv1 activation (after ReLU)
  2. Conv2 activation (after ReLU)
  3. FC1 activation (after ReLU)
  4. FC2 activation (logits)
- Compute "feature-norm growth rate" = norm_layer4 / (norm_layer1 + 1e-9).

Features:
- norm_layer1
- norm_layer2
- norm_layer3
- norm_layer4
- growth_rate
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
    """Small CNN matching diagnostic_test.py architecture with internal activation access."""
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

    def forward_with_internals(self, x):
        """Returns logits and activations from the 4 primary layers during eval mode."""
        a1 = F.relu(self.c1(x))
        a2 = F.relu(self.c2(a1))
        x_pool = F.max_pool2d(a2, 2)
        # In eval mode, dropout is identity, but we still flatten
        x_flat = x_pool.flatten(1)
        a3 = F.relu(self.fc1(x_flat))
        a4 = self.fc2(a3)
        return a4, (a1, a2, a3, a4)


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


def compute_representation_features(x, model):
    """Compute activation L2 norms across layers, growth rate, and margin/pixel features."""
    model.eval()
    with torch.no_grad():
        logits, (a1, a2, a3, a4) = model.forward_with_internals(x)

        # L2 norm for each layer per sample
        norm1 = torch.norm(a1.flatten(1), p=2, dim=1)
        norm2 = torch.norm(a2.flatten(1), p=2, dim=1)
        norm3 = torch.norm(a3.flatten(1), p=2, dim=1)
        norm4 = torch.norm(a4.flatten(1), p=2, dim=1)

        growth_rate = norm4 / (norm1 + 1e-9)

        # Margin
        sorted_logits, _ = logits.sort(dim=1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]

    N = x.size(0)
    flat_x = x.view(N, -1)
    mean_pix = flat_x.mean(1)
    std_pix = flat_x.std(1)

    return (
        norm1.cpu().numpy(),
        norm2.cpu().numpy(),
        norm3.cpu().numpy(),
        norm4.cpu().numpy(),
        growth_rate.cpu().numpy(),
        margin.cpu().numpy(),
        mean_pix.cpu().numpy(),
        std_pix.cpu().numpy()
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
    print("Hypothesis H154: Representation Distance / Feature Norm Growth Analysis")
    print("=" * 70)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("\nTraining CNN model...")
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

    # Compute features
    print("\nComputing layer activation norms and growth features...")
    (
        norm_layer1,
        norm_layer2,
        norm_layer3,
        norm_layer4,
        growth_rate,
        margin,
        mean_pix,
        std_pix
    ) = compute_representation_features(x_c, model)

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
        "norm_layer1": norm_layer1,
        "norm_layer2": norm_layer2,
        "norm_layer3": norm_layer3,
        "norm_layer4": norm_layer4,
        "growth_rate": growth_rate,
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
