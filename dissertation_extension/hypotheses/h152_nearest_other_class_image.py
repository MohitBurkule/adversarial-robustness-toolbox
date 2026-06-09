"""
Hypothesis H152: Nearest Adversarial Existence Radius via Inter-Class Interpolation.

Train a small CNN on Fashion-MNIST for 10 epochs.
For each test sample:
- Find the nearest sample of a different class (in pixel space L2) from a reference pool.
- Linearly interpolate from the clean sample (t=0) to the nearest other-class sample (t=1).
- Find the minimum t in [0, 1] (using 21 steps of 0.05) such that the model's prediction flips.
  This represents a model-aware boundary distance along image manifolds.

Features:
- t_flip (minimum interpolation parameter to change prediction)
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


def find_nearest_other_class_and_interpolate(x_eval, y_eval, x_pool, y_pool, model):
    """
    For each eval sample, find nearest other-class image in pixel space,
    interpolate, and find the minimum t in [0, 1] that flips the model's prediction.
    """
    N = x_eval.size(0)
    M = x_pool.size(0)

    # Flatten images to compute L2 distances
    x_eval_flat = x_eval.view(N, -1)
    x_pool_flat = x_pool.view(M, -1)

    # Compute pairwise L2 distances: shape (N, M)
    print("  Calculating pixel L2 distances...")
    dists = torch.cdist(x_eval_flat, x_pool_flat, p=2)

    # Mask out same-class pool items
    # Same-class items get a huge distance penalty
    same_class_mask = y_eval.view(N, 1) == y_pool.view(1, M)
    dists = dists + same_class_mask.float() * 1e9

    # Find closest different-class index for each eval sample
    closest_indices = dists.argmin(dim=1)  # Shape (N,)

    t_flips = torch.zeros(N, device=DEVICE)
    model.eval()

    # Pre-define the interpolation values to evaluate (21 steps)
    t_vals = torch.linspace(0.0, 1.0, 21, device=DEVICE)

    print("  Running interpolation predictions...")
    with torch.no_grad():
        for idx in range(N):
            x_start = x_eval[idx]  # (1, 28, 28)
            y_start = y_eval[idx]
            x_end = x_pool[closest_indices[idx]]  # (1, 28, 28)

            # Interpolate for all t-values: shape (21, 1, 28, 28)
            x_interp = torch.stack([(1.0 - t) * x_start + t * x_end for t in t_vals])

            # Classify all 21 points
            preds = model(x_interp).argmax(dim=1)

            # Find the first index where prediction is not the true label
            flips = (preds != y_start)
            if flips.any():
                first_flip_idx = flips.nonzero(as_tuple=False)[0].item()
                t_flips[idx] = t_vals[first_flip_idx]
            else:
                t_flips[idx] = 1.0  # Fails to flip within the range

            if (idx + 1) % 200 == 0:
                print(f"    processed {idx + 1}/{N} samples")

    # Compute margin and image statistics
    with torch.no_grad():
        logits = model(x_eval)
        sorted_logits, _ = logits.sort(dim=1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]

    mean_pix = x_eval_flat.mean(1)
    std_pix = x_eval_flat.std(1)

    return (
        t_flips.cpu().numpy(),
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
    print("Hypothesis H152: Nearest Other-Class Interpolation Boundary Distance")
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

    # We select 1000 correct test samples for evaluation, and the remaining correct samples as the search pool
    correct_indices = correct.nonzero(as_tuple=False).squeeze()
    eval_indices = correct_indices[:1000]
    pool_indices = correct_indices[1000:3000]  # Reference pool of 2000 samples

    x_eval = test_x[eval_indices].to(DEVICE)
    y_eval = test_y[eval_indices].to(DEVICE)

    x_pool = test_x[pool_indices].to(DEVICE)
    y_pool = test_y[pool_indices].to(DEVICE)

    N = x_eval.size(0)
    print(f"\nUsing {N} evaluation samples and {x_pool.size(0)} pool samples.")

    # Compute interpolation distance and other features
    print("\nComputing interpolation flip boundaries...")
    t_flip, margin, mean_pix, std_pix = find_nearest_other_class_and_interpolate(
        x_eval, y_eval, x_pool, y_pool, model
    )

    # Run attacks
    print("\nEvaluating attack outcomes...")
    batch_size = 256
    fgsm_success = []
    pgd_success = []
    min_eps_list = []

    for i in range(0, N, batch_size):
        xb = x_eval[i:i+batch_size]
        yb = y_eval[i:i+batch_size]

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
        "t_flip": t_flip,
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
