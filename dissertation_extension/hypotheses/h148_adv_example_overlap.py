"""
Hypothesis H148: Adversarial Example Overlap.

Train a small CNN on Fashion-MNIST for 10 epochs.
Generate FGSM, BIM, PGD, and MIM adversarials at eps=15/255.
Analyze the overlap between successful flips across the four attacks:
- Count how many attacks succeed per sample (attack_consensus: 0 to 4).
- Compute Jaccard similarities and Mutual Information between the flip sets.
- Univariate AUROC analysis for predicting "flipped by all 4 attacks".

Features:
- attack_consensus (sum of successes)
- margin (logits margin)
- mean_pix (mean pixel value)
- std_pix (std pixel value)

Target:
- flipped_by_all_4 (binary)
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, mutual_info_score

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


def compute_basic_features(test_x, model):
    """Compute margin, mean_pix, std_pix."""
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

    return margin, mean_pix, std_pix


# --- Attack Implementations ---

def attack_fgsm(model, x, y, eps=EPS):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    grad = x_adv.grad.sign().detach()
    adv = (x + eps * grad).clamp(0, 1)
    with torch.no_grad():
        return model(adv).argmax(1) != y, adv


def attack_bim(model, x, y, eps=EPS, steps=10):
    model.eval()
    alpha = eps / 5.0
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.sign().detach()
        x_adv = torch.clamp(x_adv + alpha * grad, x - eps, x + eps).clamp(0, 1).detach()
    with torch.no_grad():
        return model(x_adv).argmax(1) != y, x_adv


def attack_pgd(model, x, y, eps=EPS, steps=10):
    model.eval()
    alpha = eps / 5.0
    # Random restart start
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
        return model(x_adv).argmax(1) != y, x_adv


def attack_mim(model, x, y, eps=EPS, steps=10, decay=1.0):
    model.eval()
    alpha = eps / 5.0
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.detach()
        # Accumulate L1 normalized gradients into momentum
        grad_norm = torch.norm(grad.flatten(1), p=1, dim=1).view(-1, 1, 1, 1) + 1e-9
        momentum = decay * momentum + grad / grad_norm
        x_adv = torch.clamp(x_adv + alpha * momentum.sign(), x - eps, x + eps).clamp(0, 1).detach()
    with torch.no_grad():
        return model(x_adv).argmax(1) != y, x_adv


def main():
    print("=" * 70)
    print("Hypothesis H148: Adversarial Example Overlap (FGSM, BIM, PGD, MIM)")
    print("=" * 70)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("\nTraining model...")
    t0 = time.time()
    model, test_x, test_y = train_model(train_set, test_set)
    print(f"Training completed in {time.time() - t0:.1f}s")

    # Limit evaluation to a subset of correctly classified samples to run faster and save memory
    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        pred = logits.argmax(1)
        correct = (pred == test_y)

    x_c = test_x[correct][:1000]
    y_c = test_y[correct][:1000]
    N = x_c.size(0)
    print(f"\nUsing {N} correctly classified samples for evaluation.")

    # Compute baseline features
    margin, mean_pix, std_pix = compute_basic_features(x_c, model)

    # Generate attacks
    print("\nGenerating attacks...")
    attacks = {
        "FGSM": [],
        "BIM": [],
        "PGD": [],
        "MIM": []
    }

    # Batch attacks to prevent memory issues
    batch_size = 256
    for i in range(0, N, batch_size):
        xb = x_c[i:i+batch_size]
        yb = y_c[i:i+batch_size]

        f_fgsm, _ = attack_fgsm(model, xb, yb)
        f_bim, _ = attack_bim(model, xb, yb)
        f_pgd, _ = attack_pgd(model, xb, yb)
        f_mim, _ = attack_mim(model, xb, yb)

        attacks["FGSM"].append(f_fgsm)
        attacks["BIM"].append(f_bim)
        attacks["PGD"].append(f_pgd)
        attacks["MIM"].append(f_mim)

    for k in attacks:
        attacks[k] = torch.cat(attacks[k])

    # Convert to numpy for ease of comparison
    results_np = {k: v.cpu().numpy().astype(int) for k, v in attacks.items()}

    # Compute Jaccard and MI matrices
    names = list(attacks.keys())
    jaccard_mat = np.zeros((4, 4))
    mi_mat = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            v1, v2 = results_np[names[i]], results_np[names[j]]
            # Jaccard
            intersection = np.logical_and(v1, v2).sum()
            union = np.logical_or(v1, v2).sum()
            jaccard_mat[i, j] = intersection / union if union > 0 else 1.0
            # Mutual Information
            mi_mat[i, j] = mutual_info_score(v1, v2)

    print("\n" + "=" * 50)
    print("Jaccard Similarity Matrix of Attack Success")
    print("=" * 50)
    print(f"      {'  '.join([f'{n:<6}' for n in names])}")
    for i in range(4):
        row_str = "  ".join([f"{jaccard_mat[i, j]:.4f}" for j in range(4)])
        print(f"{names[i]:<5}: {row_str}")

    print("\n" + "=" * 50)
    print("Mutual Information Matrix of Attack Success")
    print("=" * 50)
    print(f"      {'  '.join([f'{n:<6}' for n in names])}")
    for i in range(4):
        row_str = "  ".join([f"{mi_mat[i, j]:.4f}" for j in range(4)])
        print(f"{names[i]:<5}: {row_str}")

    # Compute target and features
    success_sum = torch.stack(list(attacks.values())).float().sum(0)  # Shape (N,)
    flipped_by_all = (success_sum == 4).long()

    print(f"\nAttack Consensus Statistics (number of successful attacks per sample):")
    for s_count in range(5):
        pct = (success_sum == s_count).float().mean().item() * 100
        print(f"  Vulnerable to {s_count} attacks: {pct:.1f}%")

    # Univariate AUROC analysis for "flipped by all" target
    y = flipped_by_all.cpu().numpy()
    features = {
        "attack_consensus": success_sum.cpu().numpy(),
        "margin": margin.cpu().numpy(),
        "mean_pix": mean_pix.cpu().numpy(),
        "std_pix": std_pix.cpu().numpy()
    }

    print("\n" + "=" * 50)
    print("Univariate AUROC for Target: flipped_by_all_4")
    print("=" * 50)
    if y.std() == 0:
        print("Target has no variance (either all samples flipped by all, or none).")
    else:
        for fname, fval in features.items():
            a = roc_auc_score(y, fval)
            a_best = max(a, 1 - a)
            direction = "+" if a >= 0.5 else "-"
            print(f"  {fname:<18} AUROC = {a_best:.4f} (direction: {direction})")


if __name__ == "__main__":
    main()
