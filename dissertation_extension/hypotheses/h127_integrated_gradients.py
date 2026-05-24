"""
Hypothesis H127: Integrated Gradients Attribution vs Adversarial Vulnerability on Fashion-MNIST.

Trains a small CNN matching diagnostic_test.py for 10 epochs.
Implements Integrated Gradients (Sundararajan et al., 2017) with a black (all-zeros) baseline over K=50 steps.

For each test sample:
- Computes Integrated Gradients (IG) attribution map for the true class logit.
- Extracts metrics:
  1. IG_total_attribution: Sum of absolute IG values.
  2. IG_entropy: Shannon entropy of normalized absolute IG values.
  3. IG_l2_norm: L2 norm of the IG map.

Features:
- margin, mean_pix, std_pix
- IG_total_attribution, IG_entropy, IG_l2_norm

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


def compute_integrated_gradients(model, x, y, K=50):
    """Compute Integrated Gradients attribution map with all-zeros baseline."""
    N = x.size(0)
    baseline = torch.zeros_like(x)
    grad_sum = torch.zeros_like(x)
    
    # Path integration via Riemann summation
    for k in range(1, K + 1):
        alpha = k / K
        x_interp = (baseline + alpha * (x - baseline)).clone().detach().requires_grad_(True)
        logits = model(x_interp)
        
        # Take gradient of target class logit
        logits_target = logits[torch.arange(N), y]
        logits_target.sum().backward()
        
        with torch.no_grad():
            if x_interp.grad is not None:
                grad_sum += x_interp.grad.detach()
                
    avg_grad = grad_sum / K
    ig = (x - baseline) * avg_grad
    return ig


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
        
    # Integrated Gradients features
    print("   Computing Integrated Gradients attributions...")
    ig_maps = compute_integrated_gradients(model, x_c, y_c, K=50)
    ig_flat = ig_maps.view(N, -1).abs()
    
    ig_l2_norm = ig_flat.norm(2, dim=1)
    ig_total_attribution = ig_flat.sum(1)
    
    # Shannon entropy of absolute IG
    ig_sum = ig_flat.sum(1) + 1e-12
    ig_normalized = ig_flat / ig_sum.unsqueeze(1)
    ig_entropy = - (ig_normalized * torch.log(ig_normalized + 1e-12)).sum(1)
    
    features = torch.stack([margin, mean_pix, std_pix, ig_total_attribution, ig_entropy, ig_l2_norm], dim=1)
    
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
        feat_np = features[:, i].cpu().numpy()
        a = roc_auc_score(target_np, feat_np)
        scores[name] = max(a, 1.0 - a)
    return scores


def main():
    print("=" * 70)
    print("Hypothesis H127: Integrated Gradients Attribution vs Vulnerability")
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
    
    feature_names = ["margin", "mean_pix", "std_pix", "IG_total_attribution", "IG_entropy", "IG_l2_norm"]
    
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
    print(f"{'Feature':<22} | {'vs FGSM Flip':<12} | {'vs PGD Flip':<12}")
    print("-" * 50)
    auroc_fgsm = evaluate_auroc(feats, fgsm_flip, feature_names)
    auroc_pgd = evaluate_auroc(feats, pgd_flip, feature_names)
    
    for name in feature_names:
        print(f"{name:<22} | {auroc_fgsm[name]:.4f}       | {auroc_pgd[name]:.4f}")


if __name__ == "__main__":
    main()
