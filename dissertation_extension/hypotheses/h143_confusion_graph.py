"""
Hypothesis H143: Confusion graph class centrality on Fashion-MNIST.

Train small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
Build a 10x10 class-confusion graph from FGSM attack successes:
  C[i, j] = P(land on class j | true class i, sample flipped).

Per test sample:
  - class_centrality: how attractive/central the true class is in the confusion graph (in-degree flow).
  - inverse_centrality: how peripheral/hard to reach the true class is (1 / class_centrality).

Features:
  - class_centrality
  - inverse_centrality
  - margin
  - mean_pix
  - std_pix

Targets:
  - flipped_FGSM
  - flipped_PGD
  - min_eps (min epsilon to flip via binary search)

Analysis:
  - Compute and print the 10x10 class-confusion matrix.
  - Univariate AUROC for each feature-target pair.
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


def train_model(train_set, seed=0):
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


def fgsm_attack_and_predictions(model, x, y, eps=EPS):
    """Run FGSM and return if flipped, and the adversarial predictions."""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    sign = x_adv.grad.sign().detach()
    adv = (x + eps * sign).clamp(0, 1)

    with torch.no_grad():
        logits = model(adv)
        pred_adv = logits.argmax(1)
        flipped = (pred_adv != y)
    return flipped, pred_adv


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
    print("Hypothesis H143: Confusion Graph Centrality Analysis")
    print("=" * 60)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Train model
    print("\nTraining CNN victim...")
    t0 = time.time()
    model = train_model(train_set)
    print(f"CNN trained in {time.time() - t0:.1f}s")

    # Stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    # Restrict to correctly classified samples
    model.eval()
    with torch.no_grad():
        pred = model(test_x).argmax(1)
        correct = (pred == test_y)

    x_c = test_x[correct]
    y_c = test_y[correct]
    N_correct = correct.sum().item()
    print(f"\nUsing {N_correct}/{N} correctly classified samples")

    # Run FGSM to build the confusion graph
    print("\nRunning FGSM to build the confusion matrix...")
    flipped_fgsm = []
    pred_adv_all = []
    for i in range(0, N_correct, 512):
        bx = x_c[i:i+512]
        by = y_c[i:i+512]
        flipped, pred_adv = fgsm_attack_and_predictions(model, bx, by)
        flipped_fgsm.append(flipped)
        pred_adv_all.append(pred_adv)

    flipped_fgsm = torch.cat(flipped_fgsm)
    pred_adv_all = torch.cat(pred_adv_all)

    # Build confusion matrix C
    # C[i, j] = P(land on j | true class i, sample flipped)
    confusion_matrix = np.zeros((N_CLASSES, N_CLASSES))
    y_c_np = y_c.cpu().numpy()
    pred_adv_np = pred_adv_all.cpu().numpy()
    flipped_np = flipped_fgsm.cpu().numpy()

    for i in range(N_CLASSES):
        # find correctly classified samples of class i that flipped
        flipped_mask = (y_c_np == i) & flipped_np
        n_flipped = flipped_mask.sum()
        if n_flipped > 0:
            counts = np.bincount(pred_adv_np[flipped_mask], minlength=N_CLASSES)
            confusion_matrix[i] = counts / float(n_flipped)
        else:
            confusion_matrix[i, i] = 1.0

    print("\n" + "=" * 60)
    print("FGSM FLIPPED CONFUSION MATRIX")
    print("=" * 60)
    header = "True \\ Adv " + " ".join([f"{k:<5}" for k in range(N_CLASSES)])
    print(header)
    for i in range(N_CLASSES):
        row = f"Class {i:<4}: " + " ".join([f"{confusion_matrix[i, j]:.3f}" for j in range(N_CLASSES)])
        print(row)

    # Compute class centrality metrics
    # class_centrality[c] = sum_{k != c} C[k, c]  (incoming flow)
    class_centrality = np.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        class_centrality[c] = sum(confusion_matrix[k, c] for k in range(N_CLASSES) if k != c)

    inverse_centrality = 1.0 / (class_centrality + 1e-5)

    print("\nClass Centrality Metrics:")
    for c in range(N_CLASSES):
        print(f"  Class {c:<2} - Centrality: {class_centrality[c]:.4f}, Inverse Centrality: {inverse_centrality[c]:.4f}")

    # Map centralities back to test samples
    sample_centrality = torch.tensor([class_centrality[y_c_np[i]] for i in range(N_correct)]).cpu()
    sample_inv_centrality = torch.tensor([inverse_centrality[y_c_np[i]] for i in range(N_correct)]).cpu()

    # Logit margin
    with torch.no_grad():
        logits = model(x_c)
        sorted_logits, _ = logits.sort(1, descending=True)
        margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).cpu()

    # Image stats
    flat_x = x_c.view(N_correct, -1)
    mean_pix = flat_x.mean(1).cpu()
    std_pix = flat_x.std(1).cpu()

    features = torch.stack([sample_centrality, sample_inv_centrality, margin, mean_pix, std_pix], dim=1).numpy()

    # Run remaining attacks
    print("\nRunning PGD-10 and min_eps attacks...")
    flipped_pgd = []
    min_eps = []

    for i in range(0, N_correct, 512):
        bx = x_c[i:i+512]
        by = y_c[i:i+512]
        flipped_pgd.append(pgd_attack(model, bx, by))
        min_eps.append(min_eps_to_flip(model, bx, by))

    flipped_fgsm = flipped_fgsm.cpu().numpy().astype(int)
    flipped_pgd = torch.cat(flipped_pgd).cpu().numpy().astype(int)
    min_eps = torch.cat(min_eps).cpu().numpy()

    print(f"  FGSM flip rate = {flipped_fgsm.mean():.4f}")
    print(f"  PGD-10 flip rate = {flipped_pgd.mean():.4f}")
    print(f"  Mean min_eps = {min_eps.mean():.4f}")

    # Analyze
    feature_names = [
        "class_centrality",
        "inverse_centrality",
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
