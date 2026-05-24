"""
Hypothesis H136: Class-normalized margin deviation predicts adversarial vulnerability.

Train small CNN on Fashion-MNIST for 10 epochs.
For each correctly classified test sample, compute its logit margin.
Per-class, compute the mean and standard deviation of logit margins across the test set.
For each test sample, compute the deviation from its class mean:
  deviation_from_class_mean = (margin - class_mean) / class_std

Features:
  - deviation_from_class_mean: margin normalized by class-specific mean and std
  - margin: absolute logit margin (top - 2nd logit)
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixel values

Targets:
  - flipped_fgsm: binary indicator of whether FGSM flips prediction
  - flipped_pgd: binary indicator of whether PGD flips prediction
  - min_eps: minimum epsilon to flip via FGSM binary search

Univariate AUROC for each feature-target pair. Compare absolute margin vs class-normalized margin deviation.
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


class CNN(nn.Module):
    """Small CNN matching diagnostic_test.py."""
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
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def fgsm_grad(model, x, y):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return grad.sign().detach()


def attack_fgsm(model, x, y, eps=EPS):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def attack_pgd(model, x, y, eps=EPS, alpha=2.0/255.0, steps=10):
    x_adv = x.clone().detach().requires_grad_(True)
    for _ in range(steps):
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = x_adv.clamp(0.0, 1.0).detach().requires_grad_(True)
    with torch.no_grad():
        return (model(x_adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def auroc_both_directions(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.5
    a = roc_auc_score(y_true, y_score)
    return max(a, 1.0 - a)


def main():
    print("Loading Fashion-MNIST dataset...")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    print("Training CNN model...")
    model = train_model(train_set)

    # Filter to correctly predicted test samples
    print("Evaluating model correctness...")
    with torch.no_grad():
        preds = []
        for i in range(0, N, 512):
            preds.append(model(test_x[i:i+512]).argmax(dim=1))
        preds = torch.cat(preds)
    correct = (preds == test_y)
    x_c = test_x[correct]
    y_c = test_y[correct]
    N_c = x_c.size(0)
    print(f"Correctly classified samples: {N_c}/{N}")

    # Compute absolute margins for correctly classified test samples
    print("Computing margins...")
    margins = torch.zeros(N_c, device=DEVICE)
    with torch.no_grad():
        for i in range(0, N_c, 512):
            logits = model(x_c[i:i+512])
            sorted_logits, _ = logits.sort(dim=1, descending=True)
            margins[i:i+512] = sorted_logits[:, 0] - sorted_logits[:, 1]
    
    margins_np = margins.cpu().numpy()
    y_c_np = y_c.cpu().numpy()

    # Compute per-class distribution of margins
    print("Computing per-class margin distributions...")
    class_means = {}
    class_stds = {}
    for c in range(10):
        class_mask = (y_c_np == c)
        if class_mask.any():
            class_margins = margins_np[class_mask]
            class_means[c] = np.mean(class_margins)
            class_stds[c] = np.std(class_margins)
            # quantiles for printout
            q25 = np.percentile(class_margins, 25)
            q50 = np.percentile(class_margins, 50)
            q75 = np.percentile(class_margins, 75)
            print(f"Class {c}: Mean={class_means[c]:.4f}, Std={class_stds[c]:.4f}, Q25={q25:.4f}, Q50={q50:.4f}, Q75={q75:.4f}")
        else:
            class_means[c] = 0.0
            class_stds[c] = 1.0

    # Compute class-normalized margin deviation
    deviation = np.zeros(N_c)
    for i in range(N_c):
        c = y_c_np[i]
        mu = class_means[c]
        sd = class_stds[c]
        deviation[i] = (margins_np[i] - mu) / (sd + 1e-12)

    # Compute targets in batches
    print("Computing attack targets...")
    flipped_fgsm_list = []
    flipped_pgd_list = []
    min_eps_list = []
    for i in range(0, N_c, 512):
        xb = x_c[i:i+512]
        yb = y_c[i:i+512]
        flipped_fgsm_list.append(attack_fgsm(model, xb, yb))
        flipped_pgd_list.append(attack_pgd(model, xb, yb))
        min_eps_list.append(min_eps_to_flip(model, xb, yb))

    y_fgsm = torch.cat(flipped_fgsm_list).cpu().numpy().astype(int)
    y_pgd = torch.cat(flipped_pgd_list).cpu().numpy().astype(int)
    y_mineps = torch.cat(min_eps_list).cpu().numpy()

    # Compute standard image features
    print("Computing features...")
    mean_pix_list = []
    std_pix_list = []

    for i in range(0, N_c, 512):
        xb = x_c[i:i+512]
        B_curr = xb.size(0)
        with torch.no_grad():
            flat_x = xb.view(B_curr, -1)
            mean_pix_list.append(flat_x.mean(dim=1))
            std_pix_list.append(flat_x.std(dim=1))

    feats = {
        "deviation_from_class_mean": deviation,
        "margin": margins_np,
        "mean_pix": torch.cat(mean_pix_list).cpu().numpy(),
        "std_pix": torch.cat(std_pix_list).cpu().numpy(),
    }

    # Evaluate Univariate AUROC
    print("\n" + "="*50)
    print("UNIVARIATE AUROC RESULTS")
    print("="*50)
    
    for feat_name, feat_val in feats.items():
        print(f"\nFeature: {feat_name}")
        auc_fgsm = auroc_both_directions(y_fgsm, feat_val)
        auc_pgd = auroc_both_directions(y_pgd, feat_val)
        # For continuous target min_eps, we partition at the median to compute AUROC
        median_eps = np.median(y_mineps)
        y_mineps_bin = (y_mineps <= median_eps).astype(int)
        auc_mineps = auroc_both_directions(y_mineps_bin, feat_val)
        
        print(f"  Target: flipped_fgsm  AUROC = {auc_fgsm:.4f}")
        print(f"  Target: flipped_pgd   AUROC = {auc_pgd:.4f}")
        print(f"  Target: min_eps       AUROC = {auc_mineps:.4f}")


if __name__ == "__main__":
    main()
