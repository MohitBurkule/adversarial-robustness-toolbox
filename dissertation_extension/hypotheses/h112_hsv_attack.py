"""
Hypothesis H112: Saturation/Value Gamma Attack on Greyscale Images.

Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
For each correctly classified test sample, modulate the pixel intensities (representing Value in HSV)
using gamma corrections in the range [0.5, 2.0] across 13 uniform steps.

Identify the minimum |log(gamma)| required to flip the prediction.

Features:
  - margin: logit margin
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixels

Targets:
  - flipped_any_gamma: binary indicator of whether any gamma correction in [0.5, 2.0] flips the prediction
  - gamma_flip_magnitude: the minimum |log(gamma)| that flips the prediction (sentinel value of 1.0 if never flips)

Compute univariate AUROC for each feature-target pair.
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
N_CLASSES = 10
N_SUB = 1000  # Subsample size to keep execution times reasonable


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


def compute_features(test_x, model):
    """Compute features for test samples: margin, mean_pix, std_pix."""
    N = test_x.size(0)
    model.eval()

    with torch.no_grad():
        logits = []
        for i in range(0, N, 512):
            logits.append(model(test_x[i:i+512]))
        logits = torch.cat(logits, 0)

    sorted_logits, _ = logits.sort(1, descending=True)
    margin = (sorted_logits[:, 0] - sorted_logits[:, 1])

    flat_x = test_x.view(N, -1)
    mean_pix = flat_x.mean(1)
    std_pix = flat_x.std(1)

    features = torch.stack([margin, mean_pix, std_pix], 1)
    return features


def run_gamma_attack(model, x, y):
    """
    Apply 13 steps of gamma correction in [0.5, 2.0] to check if they flip.
    Returns:
      - flipped_any: binary mask
      - min_log_gamma: float array of minimum |log(gamma)| that flips
    """
    model.eval()
    B = x.size(0)
    gammas = np.linspace(0.5, 2.0, 13)
    
    flipped_any = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    min_log_gamma = torch.full((B,), 1.0, device=DEVICE) # default sentinel

    # Exclude gamma=1.0 from search for efficiency and safety since we only check flips
    for gamma in gammas:
        if abs(gamma - 1.0) < 1e-5:
            continue
            
        # Apply gamma correction. Adding small epsilon to prevent NaNs when pixel is exactly 0.0
        x_gamma = torch.pow(x + 1e-8, gamma).clamp(0.0, 1.0)
        
        with torch.no_grad():
            preds = model(x_gamma).argmax(1)
            flipped = (preds != y)
            
        # For samples that flipped, record the magnitude if it's smaller than current recorded
        magnitude = abs(np.log(gamma))
        update_mask = flipped & (~flipped_any | (torch.tensor(magnitude, device=DEVICE) < min_log_gamma))
        min_log_gamma[update_mask] = magnitude
        flipped_any = flipped_any | flipped

    return flipped_any, min_log_gamma


def main():
    print("=" * 60)
    print("Hypothesis H112: Saturation/Value Gamma Attack Analysis")
    print("=" * 60)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Train model
    print("\nTraining model for 10 epochs...")
    t0 = time.time()
    model, test_x, test_y = train_model(train_set, test_set)
    print(f"Training completed in {time.time() - t0:.1f}s")

    # Filter to correctly classified samples
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds, 0)
        correct = (preds == test_y)

    x_c = test_x[correct]
    y_c = test_y[correct]

    # Subsample correctly classified instances to keep execution fast
    torch.manual_seed(0)
    indices = torch.randperm(x_c.size(0))[:N_SUB]
    x_sub = x_c[indices]
    y_sub = y_c[indices]

    print(f"Evaluating gamma attacks on {x_sub.size(0)} correctly classified test samples...")

    # Compute features for subsample
    all_features = compute_features(x_sub, model)
    feature_names = ["margin", "mean_pix", "std_pix"]

    t0 = time.time()
    flipped_any, min_log_gamma = run_gamma_attack(model, x_sub, y_sub)
    print(f"Gamma attack evaluations completed in {time.time() - t0:.1f}s")

    targets = {
        "flipped_any_gamma": flipped_any.long(),
    }

    # Print vulnerability statistics
    print(f"\nGamma Attack Success Rate: {flipped_any.float().mean().item():.4f}")
    
    # Analyze min log gamma for flipped samples
    flipped_mask = flipped_any.cpu().numpy().astype(bool)
    if flipped_mask.any():
        mean_flipped_mag = min_log_gamma[flipped_any].mean().item()
        print(f"Mean |log(gamma)| to flip (for flipped samples): {mean_flipped_mag:.4f}")
    else:
        print("No samples were flipped by any gamma correction.")

    # Univariate AUROC analysis
    print("\n" + "=" * 60)
    print("UNIVARIATE AUROC ANALYSIS")
    print("=" * 60)
    
    for t_name, t_val in targets.items():
        y = t_val.cpu().numpy()
        if y.std() == 0:
            print(f"\nTarget {t_name} has no variance (all 0 or all 1), skipping.")
            continue
        print(f"\n--- Target: {t_name} (Positive Rate = {y.mean():.4f}) ---")
        print(f"  {'Feature':<15} {'AUROC':>8} {'Direction':<10}")
        for i, fname in enumerate(feature_names):
            x_i = all_features[:, i].cpu().numpy()
            a = roc_auc_score(y, x_i)
            direction = "+" if a >= 0.5 else "-"
            a = max(a, 1 - a)
            print(f"  {fname:<15} {a:>8.4f}  {direction}")

    # Analyze feature differences between flipped and non-flipped samples
    print("\n" + "=" * 60)
    print("FEATURE COMPARISON BY FLIP STATUS")
    print("=" * 60)
    
    for i, fname in enumerate(feature_names):
        val_flipped = all_features[flipped_any, i].mean().item() if flipped_any.any() else 0.0
        val_non_flipped = all_features[~flipped_any, i].mean().item() if (~flipped_any).any() else 0.0
        print(f"  {fname:<15} -> Flipped Mean: {val_flipped:.4f} | Non-Flipped Mean: {val_non_flipped:.4f}")


if __name__ == "__main__":
    main()
