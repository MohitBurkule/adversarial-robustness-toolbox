"""
Hypothesis H120: AWP (Adversarial Weight Perturbation) vs Vanilla CNN on Fashion-MNIST.

Trains two CNN models matching diagnostic_test.py:
1. Vanilla model
2. AWP model (Wu et al., 2020) with adversarial weight perturbation and FGSM-AT.

For both models:
- Computes per-sample features: margin, mean_pix, std_pix
- Computes vulnerability targets: FGSM flip (at eps=15/255), PGD flip (at eps=15/255, 10 steps), and min_eps to flip
- Evaluates univariate AUROC for each feature and target
- Outputs a comparison of feature-vulnerability relationships.
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


def train_model(train_set, mode='vanilla', seed=0, gamma=0.01):
    """Train CNN model, optionally with AWP and FGSM-AT."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    
    model = CNN(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            if mode == 'awp':
                # 1. Generate adversarial examples (FGSM-AT)
                model.eval()
                x_adv = fgsm_attack(model, x, y)
                model.train()
                
                # 2. Compute gradient w.r.t weights to find perturbation direction
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x_adv), y)
                loss.backward()
                
                # 3. Apply Adversarial Weight Perturbation (AWP)
                original_weights = {}
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            original_weights[name] = param.clone()
                            # Perturb weights adversarially: add gamma * sign(grad_w)
                            param.add_(gamma * param.grad.sign())
                            
                # 4. Compute loss and gradients w.r.t weights at perturbed weights
                optimizer.zero_grad()
                loss_awp = F.cross_entropy(model(x_adv), y)
                loss_awp.backward()
                
                # 5. Restore original weights (gradients remain unchanged)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in original_weights:
                            param.copy_(original_weights[name])
                            
                optimizer.step()
            else:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
                
        print(f"  [{mode}] Epoch {epoch+1}/{EPOCHS} completed")
    return model


def compute_features_and_targets(model, x_c, y_c):
    """Compute per-sample features and vulnerability targets."""
    model.eval()
    N = x_c.size(0)
    
    # Features
    with torch.no_grad():
        logits = model(x_c)
        sorted_logits, _ = logits.sort(1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        flat_x = x_c.view(N, -1)
        mean_pix = flat_x.mean(1)
        std_pix = flat_x.std(1)
        
    features = torch.stack([margin, mean_pix, std_pix], dim=1)
    
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
    print("Hypothesis H120: AWP Regularization vs Vanilla CNN")
    print("=" * 70)
    
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    
    # Train both models
    print("\nTraining Vanilla model...")
    model_vanilla = train_model(train_set, mode='vanilla', seed=0)
    
    print("\nTraining AWP model...")
    model_awp = train_model(train_set, mode='awp', seed=0, gamma=0.01)
    
    # We evaluate on a subset of 1000 test samples for efficiency
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    x_test, y_test = next(iter(test_loader))
    x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
    
    # Evaluate Vanilla
    print("\nEvaluating Vanilla model...")
    model_vanilla.eval()
    with torch.no_grad():
        correct_v = (model_vanilla(x_test).argmax(dim=1) == y_test)
    x_v, y_v = x_test[correct_v], y_test[correct_v]
    feats_v, fgsm_v, pgd_v, eps_v = compute_features_and_targets(model_vanilla, x_v, y_v)
    
    # Evaluate AWP
    print("\nEvaluating AWP model...")
    model_awp.eval()
    with torch.no_grad():
        correct_c = (model_awp(x_test).argmax(dim=1) == y_test)
    x_c, y_c = x_test[correct_c], y_test[correct_c]
    feats_c, fgsm_c, pgd_c, eps_c = compute_features_and_targets(model_awp, x_c, y_c)
    
    feature_names = ["margin", "mean_pix", "std_pix"]
    
    print("\n" + "=" * 50)
    print("VANILLA MODEL RESULTS")
    print("=" * 50)
    print(f"Accuracy: {correct_v.float().mean().item():.4f}")
    print(f"FGSM Success: {fgsm_v.float().mean().item():.4f}")
    print(f"PGD Success: {pgd_v.float().mean().item():.4f}")
    print(f"Mean min_eps: {eps_v.mean().item():.4f}")
    
    print("\nUnivariate AUROC vs PGD Flip (Vanilla):")
    auroc_pgd_v = evaluate_auroc(feats_v, pgd_v, feature_names)
    for name, score in auroc_pgd_v.items():
        print(f"  {name:<15} : {score:.4f}")
        
    print("\n" + "=" * 50)
    print("AWP MODEL RESULTS")
    print("=" * 50)
    print(f"Accuracy: {correct_c.float().mean().item():.4f}")
    print(f"FGSM Success: {fgsm_c.float().mean().item():.4f}")
    print(f"PGD Success: {pgd_c.float().mean().item():.4f}")
    print(f"Mean min_eps: {eps_c.mean().item():.4f}")
    
    print("\nUnivariate AUROC vs PGD Flip (AWP):")
    auroc_pgd_c = evaluate_auroc(feats_c, pgd_c, feature_names)
    for name, score in auroc_pgd_c.items():
        print(f"  {name:<15} : {score:.4f}")
        
    print("\n" + "=" * 50)
    print("SUMMARY OF AUROC CHANGES (AWP vs Vanilla PGD)")
    print("=" * 50)
    for name in feature_names:
        diff = auroc_pgd_c[name] - auroc_pgd_v[name]
        print(f"  {name:<15} : AWP={auroc_pgd_c[name]:.4f}  Vanilla={auroc_pgd_v[name]:.4f}  Change={diff:+.4f}")


if __name__ == "__main__":
    main()
