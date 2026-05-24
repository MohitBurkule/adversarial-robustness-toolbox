"""
Hypothesis H151: Input Gradient Phase Angle.

Train a small CNN on Fashion-MNIST for 10 epochs.
For each test sample:
- Compute the input loss gradient.
- Take its 2D FFT to convert the gradient into frequency space.
- Compute the zero-shifted magnitude of the spectrum.
- Calculate:
  1. Low-frequency energy fraction (radius < 4 pixels in frequency space).
  2. Dominant orientation angle (via second-order spatial moments of the frequency power spectrum).
  3. High-frequency to low-frequency energy ratio (high: radius > 8 pixels).

Features:
- low_freq_fraction
- dominant_orientation
- high_to_low_ratio
- input_grad_l2_norm
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


def compute_gradient_fft_features(x, y, model):
    """Compute 2D FFT gradient features, loss gradient L2 norm, and baseline margin."""
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    loss.backward()

    grad = x.grad.detach()  # Shape (N, 1, 28, 28)
    N = grad.size(0)

    # 1. Gradient L2 norm
    grad_l2 = torch.norm(grad.flatten(1), p=2, dim=1)  # Shape (N,)

    # 2. 2D FFT of input gradients
    # Remove channel dim
    grad_2d = grad.squeeze(1)  # Shape (N, 28, 28)
    # Perform 2D real-to-complex FFT, or standard FFT2 on complexified tensor
    G = torch.fft.fft2(grad_2d.to(torch.complex64))
    G_mag = torch.abs(G)  # Magnitude spectrum
    G_shift = torch.fft.fftshift(G_mag, dim=(-2, -1))  # Shift zero frequency to center

    # Frequency grid setup (center at 14, 14 in 28x28 grid)
    # y coordinates: -14 to 13, x coordinates: -14 to 13
    y_coords, x_coords = torch.meshgrid(
        torch.arange(-14, 14, dtype=torch.float, device=DEVICE),
        torch.arange(-14, 14, dtype=torch.float, device=DEVICE),
        indexing="ij"
    )

    # Compute radius for each frequency bin
    r2 = x_coords**2 + y_coords**2
    r = torch.sqrt(r2)

    # Masks for Low and High frequencies
    low_mask = (r < 4.0).float().unsqueeze(0)  # Shape (1, 28, 28)
    high_mask = (r > 8.0).float().unsqueeze(0)  # Shape (1, 28, 28)

    # Energy in frequency bins
    total_energy = G_shift.sum(dim=(1, 2)) + 1e-9
    low_energy = (G_shift * low_mask).sum(dim=(1, 2))
    high_energy = (G_shift * high_mask).sum(dim=(1, 2))

    low_freq_fraction = low_energy / total_energy
    high_to_low_ratio = (high_energy + 1e-9) / (low_energy + 1e-9)

    # 3. Dominant orientation via 2D spatial covariance moments of G_shift
    # Shift G_shift to form a probability-like distribution
    G_prob = G_shift / total_energy.view(-1, 1, 1)

    # Compute moments
    x_grid = x_coords.unsqueeze(0)  # (1, 28, 28)
    y_grid = y_coords.unsqueeze(0)  # (1, 28, 28)

    # Mean center in frequency space (usually (0,0) by definition after fftshift, but let's compute to be general)
    x_c = (G_prob * x_grid).sum(dim=(1, 2))  # (N,)
    y_c = (G_prob * y_grid).sum(dim=(1, 2))  # (N,)

    x_diff = x_grid - x_c.view(-1, 1, 1)
    y_diff = y_grid - y_c.view(-1, 1, 1)

    mu_20 = (G_prob * (x_diff**2)).sum(dim=(1, 2))
    mu_02 = (G_prob * (y_diff**2)).sum(dim=(1, 2))
    mu_11 = (G_prob * (x_diff * y_diff)).sum(dim=(1, 2))

    # Angle of dominant orientation: 0.5 * atan2(2 * mu_11, mu_20 - mu_02)
    dominant_orientation = 0.5 * torch.atan2(2 * mu_11, mu_20 - mu_02 + 1e-9)

    # Baseline features
    with torch.no_grad():
        sorted_logits, _ = outputs.detach().sort(1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]

    flat_x = x.view(N, -1).detach()
    mean_pix = flat_x.mean(1)
    std_pix = flat_x.std(1)

    return (
        low_freq_fraction.cpu().numpy(),
        dominant_orientation.cpu().numpy(),
        high_to_low_ratio.cpu().numpy(),
        grad_l2.cpu().numpy(),
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
    print("Hypothesis H151: Input Gradient FFT Phase and Frequency Analysis")
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

    # Compute features
    print("\nComputing gradient FFT and baseline features...")
    (
        low_freq_fraction,
        dominant_orientation,
        high_to_low_ratio,
        grad_l2,
        margin,
        mean_pix,
        std_pix
    ) = compute_gradient_fft_features(x_c, y_c, model)

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
        "low_freq_fraction": low_freq_fraction,
        "dominant_orientation": dominant_orientation,
        "high_to_low_ratio": high_to_low_ratio,
        "input_grad_l2_norm": grad_l2,
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
