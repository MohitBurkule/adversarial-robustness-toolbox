"""
Hypothesis H138: Layer ablation vulnerability analysis on Fashion-MNIST.

Train small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
Per test sample: for each conv layer, zero out the output of that layer for
that sample's forward pass, and record the prediction drop (change in true class probability).

Per-sample features:
  - max_layer_ablation_drop
  - mean_layer_ablation_drop
  - std_layer_ablation_drop
  - margin
  - mean_pix
  - std_pix

Targets:
  - flipped_FGSM
  - flipped_PGD
  - min_eps (min epsilon to flip via binary search)

Univariate AUROC for each feature-target pair.
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
    """Small CNN matching diagnostic_test.py architecture with ablation support."""
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x, ablate_layer=None):
        # Layer 1
        x = self.c1(x)
        if ablate_layer == 1:
            x = torch.zeros_like(x)
        x = F.relu(x)

        # Layer 2
        x = self.c2(x)
        if ablate_layer == 2:
            x = torch.zeros_like(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


def train_model(train_set, test_set, seed=0):
    """Train CNN on Fashion-MNIST."""
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


def compute_layer_ablation_features(model, x, y):
    """Compute layer ablation drops for conv layers (1 and 2)."""
    model.eval()
    N = x.size(0)

    # Base logits and probabilities
    with torch.no_grad():
        logits_clean = model(x)
        probs_clean = F.softmax(logits_clean, dim=1)
        prob_true_clean = probs_clean[torch.arange(N), y]

        # Ablate layer 1
        logits_abl1 = model(x, ablate_layer=1)
        prob_true_abl1 = F.softmax(logits_abl1, dim=1)[torch.arange(N), y]
        drop1 = (prob_true_clean - prob_true_abl1).cpu()

        # Ablate layer 2
        logits_abl2 = model(x, ablate_layer=2)
        prob_true_abl2 = F.softmax(logits_abl2, dim=1)[torch.arange(N), y]
        drop2 = (prob_true_clean - prob_true_abl2).cpu()

    max_drop = torch.max(drop1, drop2)
    mean_drop = (drop1 + drop2) / 2.0
    # Standard deviation of 2 numbers is simply absolute difference / 2
    std_drop = torch.abs(drop1 - drop2) / 2.0

    sorted_logits, _ = logits_clean.sort(1, descending=True)
    margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).cpu()

    flat_x = x.view(N, -1)
    mean_pix = flat_x.mean(1).cpu()
    std_pix = flat_x.std(1).cpu()

    features = torch.stack([max_drop, mean_drop, std_drop, margin, mean_pix, std_pix], 1)
    return features


def fgsm_attack(model, x, y, eps=EPS):
    """FGSM attack."""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    sign = x_adv.grad.sign().detach()
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS, alpha=2.0/255.0, steps=10):
    """PGD-10 attack."""
    adv = x.clone().detach().requires_grad_(True)
    # Random start
    adv = adv + torch.FloatTensor(*adv.shape).uniform_(-eps, eps).to(DEVICE)
    adv = adv.clamp(0, 1)

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        adv = (adv + alpha * adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Compute min epsilon to flip using binary search."""
    N = x.size(0)
    lo = torch.zeros(N, device=DEVICE)
    hi = torch.full((N,), eps_max, device=DEVICE)

    # Base FGSM direction
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


def main():
    print("=" * 60)
    print("Hypothesis H138: Layer Ablation Vulnerability Analysis")
    print("=" * 60)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Train model
    print("\nTraining model...")
    t0 = time.time()
    model, test_x, test_y = train_model(train_set, test_set)
    print(f"Training completed in {time.time() - t0:.1f}s")

    # Restrict to correctly classified samples
    model.eval()
    with torch.no_grad():
        pred = model(test_x).argmax(1)
        correct = (pred == test_y)
    x_c, y_c = test_x[correct], test_y[correct]
    N_correct = correct.sum().item()
    print(f"Using {N_correct}/{test_x.size(0)} correctly classified samples")

    # Compute features
    print("\nComputing layer ablation features...")
    features = compute_layer_ablation_features(model, x_c, y_c)

    # Run attacks
    print("\nRunning attacks...")
    flipped_fgsm = []
    flipped_pgd = []
    min_eps = []

    for i in range(0, N_correct, 512):
        bx = x_c[i:i+512]
        by = y_c[i:i+512]
        flipped_fgsm.append(fgsm_attack(model, bx, by))
        flipped_pgd.append(pgd_attack(model, bx, by))
        min_eps.append(min_eps_to_flip(model, bx, by))

    flipped_fgsm = torch.cat(flipped_fgsm).cpu().numpy().astype(int)
    flipped_pgd = torch.cat(flipped_pgd).cpu().numpy().astype(int)
    min_eps = torch.cat(min_eps).cpu().numpy()

    print(f"  FGSM flip rate = {flipped_fgsm.mean():.4f}")
    print(f"  PGD-10 flip rate = {flipped_pgd.mean():.4f}")
    print(f"  Mean min_eps = {min_eps.mean():.4f}")

    # Analyze
    feature_names = [
        "max_ablation_drop",
        "mean_ablation_drop",
        "std_ablation_drop",
        "margin",
        "mean_pix",
        "std_pix"
    ]

    for target_name, y in [("flipped_FGSM", flipped_fgsm), ("flipped_PGD", flipped_pgd)]:
        print("\n" + "=" * 60)
        print(f"TARGET: {target_name} | Univariate AUROC")
        print("=" * 60)
        if y.std() == 0:
            print("  Constant target, skipping.")
            continue
        for i, fname in enumerate(feature_names):
            x_i = features[:, i].numpy()
            a = roc_auc_score(y, x_i)
            a = max(a, 1 - a)
            print(f"  {fname:<25} AUROC = {a:.4f}")

    print("\n" + "=" * 60)
    print("TARGET: min_eps (continuous) | Univariate AUROC (above median)")
    print("=" * 60)
    y_bin = (min_eps > np.median(min_eps)).astype(int)
    if y_bin.std() == 0:
        print("  Constant target, skipping.")
    else:
        for i, fname in enumerate(feature_names):
            x_i = features[:, i].numpy()
            a = roc_auc_score(y_bin, x_i)
            a = max(a, 1 - a)
            print(f"  {fname:<25} AUROC = {a:.4f}")


if __name__ == "__main__":
    main()
