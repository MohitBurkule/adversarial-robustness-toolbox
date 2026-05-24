"""
Hypothesis H116: Sign-OPT Decision-based Attack Analysis.

Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
Implement Sign-OPT (Cheng et al., 2020) decision-based search:
query the model with random directions, find the minimum epsilon to cross the decision boundary
along each direction via binary line search, and pick the best (minimum L2 perturbation) direction.

Budget: Subsample 200 test points, 50 search iterations each.

Features:
  - margin: logit margin
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixels

Targets:
  - sign_opt_perturbation_L2: continuous minimum L2 distance to cross the boundary
  - sign_opt_queries: total queries (model evaluations) used
  - sign_opt_vulnerable: binary indicator of whether L2 perturbation is below median

Compute univariate AUROC for the binarized target.
Analyze correlation of features with continuous L2 boundary distance.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
MAX_ITER_SIGN_OPT = 50
N_CLASSES = 10
N_SUB = 200  # Subsample size for Sign-OPT black-box attack


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


def compute_features(test_x, model):
    """Compute features for test samples: margin, mean_pix, std_pix."""
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

    features = torch.stack([margin, mean_pix, std_pix], 1)
    return features


def sign_opt_line_search(model, x, y, max_iter=MAX_ITER_SIGN_OPT):
    """
    Sign-OPT decision-based attack.
    Samples random directions, uses line search to find the nearest boundary point,
    and returns the minimum L2 perturbation distance and total queries used.
    """
    model.eval()
    best_l2 = float('inf')
    queries = 0
    
    with torch.no_grad():
        if model(x.unsqueeze(0)).argmax(1).item() != y:
            return 0.0, 1
            
    queries += 1
    
    for step in range(max_iter):
        # Sample random Gaussian direction
        theta = torch.randn_like(x)
        theta = theta / (torch.norm(theta) + 1e-8)
        
        # 1. Exponential line search to find a boundary crossing upper bound
        alpha_high = 0.1
        found = False
        for _ in range(7):
            adv = torch.clamp(x + alpha_high * theta, 0.0, 1.0)
            with torch.no_grad():
                pred = model(adv.unsqueeze(0)).argmax(1).item()
                queries += 1
            if pred != y:
                found = True
                break
            alpha_high *= 2.0
            
        if not found:
            continue
            
        # 2. Binary search to find the exact boundary crossing distance
        alpha_low = 0.0
        for _ in range(10):
            alpha_mid = (alpha_low + alpha_high) / 2.0
            adv = torch.clamp(x + alpha_mid * theta, 0.0, 1.0)
            with torch.no_grad():
                pred = model(adv.unsqueeze(0)).argmax(1).item()
                queries += 1
            if pred != y:
                alpha_high = alpha_mid
            else:
                alpha_low = alpha_mid
                
        # Compute L2 distance of the boundary perturbation
        pert_l2 = torch.norm(torch.clamp(x + alpha_high * theta, 0.0, 1.0) - x).item()
        if pert_l2 < best_l2:
            best_l2 = pert_l2
            
    if best_l2 == float('inf'):
        best_l2 = 10.0  # Sentinel value for no boundary found
        
    return best_l2, queries


def main():
    print("=" * 60)
    print("Hypothesis H116: Sign-OPT Decision Attack Analysis")
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

    print(f"Evaluating Sign-OPT on {x_sub.size(0)} correctly classified test samples...")

    # Compute features for subsample
    all_features = compute_features(x_sub, model)
    feature_names = ["margin", "mean_pix", "std_pix"]

    t0 = time.time()
    
    opt_l2 = []
    opt_queries = []
    
    for i in range(x_sub.size(0)):
        l2, q = sign_opt_line_search(model, x_sub[i], y_sub[i].item(), MAX_ITER_SIGN_OPT)
        opt_l2.append(l2)
        opt_queries.append(q)
        
    l2_tensor = torch.tensor(opt_l2, dtype=torch.float, device=DEVICE)
    queries_tensor = torch.tensor(opt_queries, dtype=torch.float, device=DEVICE)
    
    print(f"Sign-OPT evaluations completed in {time.time() - t0:.1f}s")

    # Binarize L2 perturbation to get vulnerability indicator (vulnerable if L2 distance is below median)
    l2_median = l2_tensor.median().item()
    sign_opt_vulnerable = (l2_tensor < l2_median).long()

    targets = {
        "sign_opt_vulnerable": sign_opt_vulnerable,
    }

    # Print vulnerability statistics
    print(f"\nSign-OPT Median L2 Perturbation: {l2_median:.4f}")
    print(f"Mean L2 Perturbation: {l2_tensor.mean().item():.4f}")
    print(f"Mean Queries used:     {queries_tensor.mean().item():.2f}")

    # Univariate AUROC analysis
    print("\n" + "=" * 60)
    print("UNIVARIATE AUROC ANALYSIS (Target: sign_opt_vulnerable)")
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

    # Spearman rank correlation of continuous L2 distance with features
    print("\n" + "=" * 60)
    print("SPEARMAN CORRELATION WITH CONTINUOUS L2 BOUNDARY DISTANCE")
    print("=" * 60)
    l2_np = l2_tensor.cpu().numpy()
    f_np = all_features.cpu().numpy()
    for i, fname in enumerate(feature_names):
        rho, pval = spearmanr(f_np[:, i], l2_np)
        print(f"  {fname:<15} -> Correlation: {rho:+.4f} (p-value: {pval:.2e})")


if __name__ == "__main__":
    main()
