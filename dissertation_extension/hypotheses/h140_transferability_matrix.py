"""
Hypothesis H140: Adversarial transferability matrix on Fashion-MNIST.

Train K=4 CNNs (different seeds) on Fashion-MNIST for 10 epochs each.
Evaluate pairwise adversarial transferability.

For each sample:
  - transferability_score: fraction of model pairs (i, j) with i != j where an FGSM example
    generated on model i transfers to (fools) model j.

Features (computed on Model 0):
  - margin
  - mean_pix
  - std_pix

Targets:
  - transferability_score (continuous)
  - high_transferability (binary: score > median)

Analysis:
  - Compute and print the KxK transferability matrix.
  - Univariate AUROC for high-transferability prediction.
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
K_MODELS = 4


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


def train_model(train_set, seed):
    """Train a CNN with the given seed."""
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


def fgsm_perturb(model, x, y, eps=EPS):
    """Generate FGSM adversarial images."""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    sign = x_adv.grad.sign().detach()
    return (x + eps * sign).clamp(0, 1)


def main():
    print("=" * 60)
    print("Hypothesis H140: Adversarial Transferability Matrix")
    print("=" * 60)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Train K models
    models = []
    print(f"\nTraining K={K_MODELS} CNN models...")
    for k in range(K_MODELS):
        t0 = time.time()
        model = train_model(train_set, seed=k)
        model.eval()
        models.append(model)
        print(f"  Model {k} trained in {time.time() - t0:.1f}s")

    # Stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    # Restrict to samples correctly classified by ALL K models
    with torch.no_grad():
        correct_all = torch.ones(N, dtype=torch.bool, device=DEVICE)
        for model in models:
            pred = model(test_x).argmax(dim=1)
            correct_all &= (pred == test_y)

    x_c = test_x[correct_all]
    y_c = test_y[correct_all]
    N_correct = correct_all.sum().item()
    print(f"\nUsing {N_correct}/{N} samples correctly classified by all {K_MODELS} models")

    # Compute features on Model 0
    print("\nComputing features on Model 0...")
    with torch.no_grad():
        logits_0 = models[0](x_c)
        sorted_logits, _ = logits_0.sort(dim=1, descending=True)
        margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).cpu()

    flat_x = x_c.view(N_correct, -1)
    mean_pix = flat_x.mean(1).cpu()
    std_pix = flat_x.std(1).cpu()
    features = torch.stack([margin, mean_pix, std_pix], 1).numpy()

    # Generate FGSM examples on all models
    print("\nGenerating FGSM examples on all models...")
    adv_examples = []
    for k in range(K_MODELS):
        adv_k = []
        for i in range(0, N_correct, 512):
            bx = x_c[i:i+512]
            by = y_c[i:i+512]
            adv_k.append(fgsm_perturb(models[k], bx, by))
        adv_examples.append(torch.cat(adv_k, dim=0))

    # Evaluate pairwise transferability
    print("\nEvaluating pairwise transferability...")
    transfer_matrix = np.zeros((K_MODELS, K_MODELS))
    transfer_success_per_sample = torch.zeros(N_correct, device=DEVICE)
    n_pairs = 0

    for i in range(K_MODELS):
        for j in range(K_MODELS):
            # Check fooled rate
            fooled = []
            for b in range(0, N_correct, 512):
                bx_adv = adv_examples[i][b:b+512]
                by = y_c[b:b+512]
                with torch.no_grad():
                    pred_j = models[j](bx_adv).argmax(dim=1)
                    fooled.append(pred_j != by)
            fooled = torch.cat(fooled)
            transfer_matrix[i, j] = fooled.float().mean().item()

            if i != j:
                transfer_success_per_sample += fooled.float()
                n_pairs += 1

    # Transferability score
    transferability_score = (transfer_success_per_sample / float(n_pairs)).cpu().numpy()

    # Print matrix
    print("\n" + "=" * 60)
    print("TRANSFERABILITY MATRIX (Flip Rate)")
    print("=" * 60)
    header = "      " + " ".join([f"To M{k:<3}" for k in range(K_MODELS)])
    print(header)
    for i in range(K_MODELS):
        row = f"From M{i}: " + "  ".join([f"{transfer_matrix[i, j]:.3f}" for j in range(K_MODELS)])
        print(row)

    print(f"\nMean transfer rate (excluding self-attacks): {transferability_score.mean():.4f}")

    # Binarize transferability score for AUROC
    median_score = np.median(transferability_score)
    high_transfer = (transferability_score > median_score).astype(int)
    print(f"Binarization threshold (median score): {median_score:.4f}")
    print(f"High-transferability rate: {high_transfer.mean():.3f}")

    # Univariate AUROC
    feature_names = ["margin", "mean_pix", "std_pix"]
    print("\n" + "=" * 60)
    print("TARGET: High-Transferability | Univariate AUROC")
    print("=" * 60)
    if high_transfer.std() == 0:
        print("  Constant target, skipping.")
    else:
        for i, fname in enumerate(feature_names):
            x_i = features[:, i]
            a = roc_auc_score(high_transfer, x_i)
            a = max(a, 1 - a)
            print(f"  {fname:<15} AUROC = {a:.4f}")


if __name__ == "__main__":
    main()
