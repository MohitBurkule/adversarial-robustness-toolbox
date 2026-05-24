"""
Hypothesis H115: NES (Natural Evolution Strategies) Black-box Attack.

Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
Implement NES (Ilyas et al., 2018): for each test sample, estimate the gradient
using K=20 Gaussian perturbations (10 antithetic pairs) with score-function gradient estimation,
and apply a PGD step for 50 iterations at eps=15/255.

Features:
  - margin: logit margin
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixels
  - sobel_mean: spatial frequency proxy

Targets:
  - nes_flipped: binary success of NES
  - nes_queries_to_flip: queries required to flip (100 if never flips)
  - flipped_FGSM: binary success of standard single-step FGSM

Compare NES success with standard FGSM success.
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
EPS = 15.0 / 255.0
ALPHA = 2.0 / 255.0
K_NES = 20
SIGMA_NES = 0.001
MAX_ITER = 50
N_CLASSES = 10
N_SUB = 200  # Subsample size for NES black-box attack


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


def get_sobel_mean(x):
    """Compute mean Sobel filter magnitude for each image in a batch."""
    hx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
    hy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, hx, padding=1)
    gy = F.conv2d(x, hy, padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    return mag.mean(dim=(1, 2, 3))


def compute_features(test_x, model):
    """Compute features for test samples: margin, mean_pix, std_pix, sobel_mean."""
    N = test_x.size(0)
    model.eval()

    with torch.no_grad():
        logits = []
        sobel_vals = []
        for i in range(0, N, 512):
            batch_x = test_x[i:i+512]
            logits.append(model(batch_x))
            sobel_vals.append(get_sobel_mean(batch_x))
        logits = torch.cat(logits, 0)
        sobel_mean = torch.cat(sobel_vals, 0)

    sorted_logits, _ = logits.sort(1, descending=True)
    margin = (sorted_logits[:, 0] - sorted_logits[:, 1])

    flat_x = test_x.view(N, -1)
    mean_pix = flat_x.mean(1)
    std_pix = flat_x.std(1)

    features = torch.stack([margin, mean_pix, std_pix, sobel_mean], 1)
    return features


def nes_attack(model, x, y, eps=EPS, alpha=ALPHA, K=K_NES, sigma=SIGMA_NES, max_iter=MAX_ITER):
    """
    NES black-box gradient-estimation PGD attack.
    Estimates the gradient via score-function on K noise samples.
    """
    model.eval()
    x_orig = x.clone()
    x_adv = x.clone()
    
    with torch.no_grad():
        if model(x_adv.unsqueeze(0)).argmax(1).item() != y:
            return True, 0
            
    queries = 1
    
    for step in range(max_iter):
        grad = torch.zeros_like(x_adv)
        half_K = K // 2
        
        # Draw half_K Gaussian noises for antithetic sampling
        for _ in range(half_K):
            noise = torch.randn_like(x_adv)
            x_plus = torch.clamp(x_adv + sigma * noise, 0.0, 1.0)
            x_minus = torch.clamp(x_adv - sigma * noise, 0.0, 1.0)
            
            with torch.no_grad():
                logits_plus = model(x_plus.unsqueeze(0))
                logits_minus = model(x_minus.unsqueeze(0))
                queries += 2
                
                loss_plus = F.cross_entropy(logits_plus, torch.tensor([y], device=x.device)).item()
                loss_minus = F.cross_entropy(logits_minus, torch.tensor([y], device=x.device)).item()
                
            grad += (loss_plus - loss_minus) * noise
            
        grad = grad / (K * sigma)
        
        # PGD step using the estimated gradient sign
        x_adv = torch.clamp(x_adv + alpha * grad.sign(), x_orig - eps, x_orig + eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        with torch.no_grad():
            pred = model(x_adv.unsqueeze(0)).argmax(1).item()
            queries += 1
            if pred != y:
                return True, queries
                
    return False, queries


def fgsm_attack(model, x, y, eps=EPS):
    """Standard untargeted single-step FGSM attack."""
    model.eval()
    x_orig = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_orig), y)
    loss.backward()
    grad_sign = x_orig.grad.sign().detach()
    x_adv = torch.clamp(x + eps * grad_sign, 0.0, 1.0)
    with torch.no_grad():
        return (model(x_adv).argmax(dim=1) != y)


def main():
    print("=" * 60)
    print("Hypothesis H115: NES Zeroth-Order PGD Attack Analysis")
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

    print(f"Evaluating NES and FGSM on {x_sub.size(0)} correctly classified test samples...")

    # Compute features for subsample
    all_features = compute_features(x_sub, model)
    feature_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    t0 = time.time()
    
    nes_flipped = []
    nes_queries = []
    
    for i in range(x_sub.size(0)):
        flipped, q = nes_attack(model, x_sub[i], y_sub[i], eps=EPS, alpha=ALPHA, K=K_NES, sigma=SIGMA_NES, max_iter=MAX_ITER)
        nes_flipped.append(flipped)
        nes_queries.append(q)
        
    flipped_tensor = torch.tensor(nes_flipped, dtype=torch.long, device=DEVICE)
    queries_tensor = torch.tensor(nes_queries, dtype=torch.float, device=DEVICE)
    
    t1 = time.time()
    flipped_FGSM = fgsm_attack(model, x_sub, y_sub, EPS).long()
    
    print(f"Evaluations completed: NES ({t1-t0:.1f}s), FGSM ({time.time()-t1:.1f}s)")

    targets = {
        "nes_flipped": flipped_tensor,
        "flipped_FGSM": flipped_FGSM
    }

    # Print vulnerability statistics
    print(f"\nNES Attack Success Rate:  {flipped_tensor.float().mean().item():.4f}")
    print(f"FGSM Attack Success Rate: {flipped_FGSM.float().mean().item():.4f}")
    print(f"Mean NES Queries used:    {queries_tensor.mean().item():.2f}")

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


if __name__ == "__main__":
    main()
