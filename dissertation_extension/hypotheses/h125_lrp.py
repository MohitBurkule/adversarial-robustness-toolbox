"""
Hypothesis H125: LRP Attribution vs Adversarial Vulnerability on Fashion-MNIST.

Trains a small CNN matching diagnostic_test.py for 10 epochs.
Implements basic LRP-epsilon (Bach et al., 2015) layer-wise relevance propagation.
For each sample, decomposes the true class logit backward to get an input relevance heatmap.

Computes LRP statistics for each test sample:
1. lrp_total: Sum of absolute relevance values.
2. lrp_entropy: Shannon entropy of normalized relevance distribution.
3. lrp_max: Maximum absolute relevance value in the input map.

Features:
- margin, mean_pix, std_pix
- lrp_total, lrp_entropy, lrp_max

Targets:
- flipped_FGSM, flipped_PGD, min_eps to flip

Univariate AUROC of each feature against the targets.
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


def fgsm_attack(model, x, y, eps=EPS):
    """FGSM attack."""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    with torch.no_grad():
        x_adv = (x_adv + eps * x_adv.grad.sign()).clamp(0, 1)
    return x_adv


def pgd_attack(model, x, y, eps=EPS, alpha=2.0/255.0, steps=10):
    """PGD-10 attack."""
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = (x_adv + alpha * x_adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return x_adv


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=8):
    """Binary search for minimum epsilon to flip prediction using PGD."""
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = x.clone().detach()
        alpha = mid / 4.0
        for _ in range(5):
            adv.requires_grad_()
            loss = F.cross_entropy(model(adv), y)
            loss.backward()
            with torch.no_grad():
                adv = (adv + alpha.view(-1, 1, 1, 1) * adv.grad.sign()).clamp(x - mid.view(-1, 1, 1, 1), x + mid.view(-1, 1, 1, 1)).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def lrp_forward_backward(model, x, target_class, eps_lrp=1e-5):
    """Perform LRP-epsilon attribution from target class logit to input space."""
    # Forward pass, saving activations
    a0 = x.clone().detach()
    
    a1 = F.relu(model.c1(a0))
    a2 = F.relu(model.c2(a1))
    
    a3_pool = F.max_pool2d(a2, kernel_size=2, stride=2)
    a3 = a3_pool.flatten(1)
    
    a4 = F.relu(model.fc1(a3))
    a5 = model.fc2(a4)
    
    # Target relevance
    R5 = torch.zeros_like(a5)
    R5[torch.arange(x.size(0)), target_class] = a5[torch.arange(x.size(0)), target_class]
    
    # 1. R5 -> R4 (fc2)
    z = F.linear(a4, model.fc2.weight)
    z = z + eps_lrp * torch.sign(z + 1e-12)
    s = R5 / z
    c = F.linear(s, model.fc2.weight.t())
    R4 = a4 * c
    
    # 2. R4 -> R3 (fc1)
    z = F.linear(a3, model.fc1.weight)
    z = z + eps_lrp * torch.sign(z + 1e-12)
    s = R4 / z
    c = F.linear(s, model.fc1.weight.t())
    R3 = a3 * c
    
    # 3. R3 -> R2 (MaxPool)
    R3_conv = R3.view_as(a3_pool)
    R2_pool = F.interpolate(R3_conv, scale_factor=2, mode='nearest')
    
    # 4. R2 -> R1 (c2)
    z = F.conv2d(a1, model.c2.weight, stride=model.c2.stride, padding=model.c2.padding)
    z = z + eps_lrp * torch.sign(z + 1e-12)
    s = R2_pool / z
    c = F.conv_transpose2d(s, model.c2.weight, stride=model.c2.stride, padding=model.c2.padding)
    if c.shape != a1.shape:
        c = F.interpolate(c, size=(a1.shape[2], a1.shape[3]), mode='bilinear', align_corners=False)
    R1 = a1 * c
    
    # 5. R1 -> R0 (c1)
    z = F.conv2d(a0, model.c1.weight, stride=model.c1.stride, padding=model.c1.padding)
    z = z + eps_lrp * torch.sign(z + 1e-12)
    s = R1 / z
    c = F.conv_transpose2d(s, model.c1.weight, stride=model.c1.stride, padding=model.c1.padding)
    if c.shape != a0.shape:
        c = F.interpolate(c, size=(a0.shape[2], a0.shape[3]), mode='bilinear', align_corners=False)
    R0 = a0 * c
    
    return R0


def train_model(train_set, seed=0):
    """Train CNN on Fashion-MNIST for 10 epochs."""
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
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch+1}/{EPOCHS} completed")
    return model


def compute_features_and_targets(model, x_c, y_c):
    """Compute per-sample features and vulnerability targets."""
    model.eval()
    N = x_c.size(0)
    
    # Baselines
    with torch.no_grad():
        logits = model(x_c)
        sorted_logits, _ = logits.sort(1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        flat_x = x_c.view(N, -1)
        mean_pix = flat_x.mean(1)
        std_pix = flat_x.std(1)
        
    # LRP features
    print("   Computing LRP attributions...")
    lrp_maps = lrp_forward_backward(model, x_c, y_c)
    lrp_flat = lrp_maps.view(N, -1).abs()
    
    lrp_total = lrp_flat.sum(1)
    
    # Normalized LRP for entropy
    lrp_normalized = lrp_flat / (lrp_total.unsqueeze(1) + 1e-12)
    lrp_entropy = - (lrp_normalized * torch.log(lrp_normalized + 1e-12)).sum(1)
    
    lrp_max = lrp_flat.max(1)[0]
    
    features = torch.stack([margin, mean_pix, std_pix, lrp_total, lrp_entropy, lrp_max], dim=1)
    
    # Targets
    print("   Evaluating attacks...")
    fgsm_adv = fgsm_attack(model, x_c, y_c)
    pgd_adv = pgd_attack(model, x_c, y_c)
    
    with torch.no_grad():
        flipped_fgsm = (model(fgsm_adv).argmax(dim=1) != y_c).long()
        flipped_pgd = (model(pgd_adv).argmax(dim=1) != y_c).long()
        
    min_eps = min_eps_to_flip(model, x_c, y_c)
    
    return features, flipped_fgsm, flipped_pgd, min_eps


def evaluate_auroc(features, target, feature_names):
    """Compute univariate AUROC scores."""
    scores = {}
    target_np = target.cpu().numpy()
    if target_np.std() == 0:
        return {name: 0.5 for name in feature_names}
    
    for i, name in enumerate(feature_names):
        feat_np = features[:, i].detach().cpu().numpy()
        a = roc_auc_score(target_np, feat_np)
        scores[name] = max(a, 1.0 - a)
    return scores


def main():
    print("=" * 70)
    print("Hypothesis H125: LRP Attribution vs Adversarial Vulnerability")
    print("=" * 70)
    
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    
    # Train model
    print("\nTraining CNN model...")
    model = train_model(train_set, seed=0)
    
    # We evaluate on a subset of 1000 test samples for efficiency
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    x_test, y_test = next(iter(test_loader))
    x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
    
    # Evaluate correctly classified samples
    model.eval()
    with torch.no_grad():
        correct = (model(x_test).argmax(dim=1) == y_test)
    x_c, y_c = x_test[correct], y_test[correct]
    print(f"Using {x_c.size(0)} correctly classified test samples")
    
    feats, fgsm_flip, pgd_flip, min_eps = compute_features_and_targets(model, x_c, y_c)
    
    feature_names = ["margin", "mean_pix", "std_pix", "lrp_total", "lrp_entropy", "lrp_max"]
    
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE & ROBUSTNESS")
    print("=" * 50)
    print(f"Clean Accuracy: {correct.float().mean().item():.4f}")
    print(f"FGSM Success: {fgsm_flip.float().mean().item():.4f}")
    print(f"PGD Success: {pgd_flip.float().mean().item():.4f}")
    print(f"Mean min_eps: {min_eps.mean().item():.4f}")
    
    print("\n" + "=" * 50)
    print("UNIVARIATE AUROC SCORES")
    print("=" * 50)
    print(f"{'Feature':<15} | {'vs FGSM Flip':<12} | {'vs PGD Flip':<12}")
    print("-" * 50)
    auroc_fgsm = evaluate_auroc(feats, fgsm_flip, feature_names)
    auroc_pgd = evaluate_auroc(feats, pgd_flip, feature_names)
    
    for name in feature_names:
        print(f"{name:<15} | {auroc_fgsm[name]:.4f}       | {auroc_pgd[name]:.4f}")


if __name__ == "__main__":
    main()
