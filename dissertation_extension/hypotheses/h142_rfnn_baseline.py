"""
Hypothesis H142: Random Features Neural Network (RFNN) baseline on Fashion-MNIST.

Train small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs as the victim.
Also train an RFNN: same Conv layers (randomly initialized and frozen) and a final linear layer
trained on Fashion-MNIST for 10 epochs.

Per test sample:
  - rfnn_margin: margin of logits from RFNN
  - rfnn_confidence: max softmax probability from RFNN

Features:
  - rfnn_margin
  - rfnn_confidence
  - cnn_margin (from victim)
  - mean_pix
  - std_pix

Targets (on CNN victim):
  - flipped_FGSM
  - flipped_PGD
  - min_eps (min epsilon to flip via binary search)

Analysis:
  - Univariate AUROC for all features against CNN victim targets.
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


# ---- CNN victim architecture ----
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


# ---- RFNN architecture ----
class RFNN(nn.Module):
    """Random features network: frozen random Conv layers + trainable Linear head."""
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc = nn.Linear(64 * 12 * 12, n)

        # Freeze the convolution layers
        for p in self.c1.parameters():
            p.requires_grad = False
        for p in self.c2.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = F.relu(self.c1(x))
            x = F.relu(self.c2(x))
            x = F.max_pool2d(x, 2)
            x = x.flatten(1)
        # linear head is trained
        return self.fc(x)


def train_cnn(train_set, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
    return model


def train_rfnn(train_set, seed=0):
    torch.manual_seed(seed + 100)
    np.random.seed(seed + 100)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = RFNN(N_CLASSES).to(DEVICE)
    # Only self.fc parameters will be updated
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
    return model


def fgsm_attack(model, x, y, eps=EPS):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    sign = x_adv.grad.sign().detach()
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS, alpha=2.0/255.0, steps=10):
    adv = x.clone().detach().requires_grad_(True)
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


def main():
    print("=" * 60)
    print("Hypothesis H142: Random Features Neural Network Baseline")
    print("=" * 60)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Train CNN victim
    print("\nTraining CNN victim...")
    t0 = time.time()
    cnn_model = train_cnn(train_set)
    print(f"CNN trained in {time.time() - t0:.1f}s")

    # Train RFNN
    print("\nTraining RFNN baseline...")
    t0 = time.time()
    rfnn_model = train_rfnn(train_set)
    print(f"RFNN trained in {time.time() - t0:.1f}s")

    # Stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    # Restrict to correctly classified samples of CNN victim
    cnn_model.eval()
    rfnn_model.eval()
    with torch.no_grad():
        cnn_pred = cnn_model(test_x).argmax(1)
        correct = (cnn_pred == test_y)

    x_c = test_x[correct]
    y_c = test_y[correct]
    N_correct = correct.sum().item()
    print(f"\nUsing {N_correct}/{N} correctly classified samples from victim (CNN)")

    # Compute features
    print("\nComputing features...")
    with torch.no_grad():
        # CNN Margin
        cnn_logits = cnn_model(x_c)
        sorted_cnn, _ = cnn_logits.sort(1, descending=True)
        cnn_margin = (sorted_cnn[:, 0] - sorted_cnn[:, 1]).cpu()

        # RFNN Margin & Confidence
        rfnn_logits = rfnn_model(x_c)
        sorted_rfnn, _ = rfnn_logits.sort(1, descending=True)
        rfnn_margin = (sorted_rfnn[:, 0] - sorted_rfnn[:, 1]).cpu()
        rfnn_confidence = F.softmax(rfnn_logits, dim=1).max(dim=1)[0].cpu()

    flat_x = x_c.view(N_correct, -1)
    mean_pix = flat_x.mean(1).cpu()
    std_pix = flat_x.std(1).cpu()

    features = torch.stack([rfnn_margin, rfnn_confidence, cnn_margin, mean_pix, std_pix], dim=1).numpy()

    # Run attacks on CNN victim
    print("\nRunning attacks on CNN victim...")
    flipped_fgsm = []
    flipped_pgd = []
    min_eps = []

    for i in range(0, N_correct, 512):
        bx = x_c[i:i+512]
        by = y_c[i:i+512]
        flipped_fgsm.append(fgsm_attack(cnn_model, bx, by))
        flipped_pgd.append(pgd_attack(cnn_model, bx, by))
        min_eps.append(min_eps_to_flip(cnn_model, bx, by))

    flipped_fgsm = torch.cat(flipped_fgsm).cpu().numpy().astype(int)
    flipped_pgd = torch.cat(flipped_pgd).cpu().numpy().astype(int)
    min_eps = torch.cat(min_eps).cpu().numpy()

    print(f"  FGSM flip rate = {flipped_fgsm.mean():.4f}")
    print(f"  PGD-10 flip rate = {flipped_pgd.mean():.4f}")
    print(f"  Mean min_eps = {min_eps.mean():.4f}")

    # Analyze
    feature_names = [
        "rfnn_margin",
        "rfnn_confidence",
        "cnn_margin",
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
            x_i = features[:, i]
            a = roc_auc_score(y, x_i)
            a = max(a, 1 - a)
            print(f"  {fname:<20} AUROC = {a:.4f}")

    print("\n" + "=" * 60)
    print("TARGET: min_eps (continuous) | Univariate AUROC (above median)")
    print("=" * 60)
    y_bin = (min_eps > np.median(min_eps)).astype(int)
    if y_bin.std() == 0:
        print("  Constant target, skipping.")
    else:
        for i, fname in enumerate(feature_names):
            x_i = features[:, i]
            a = roc_auc_score(y_bin, x_i)
            a = max(a, 1 - a)
            print(f"  {fname:<20} AUROC = {a:.4f}")


if __name__ == "__main__":
    main()
