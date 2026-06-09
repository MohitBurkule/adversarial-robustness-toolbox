"""
Hypothesis H114: ZOO (Zeroth Order Optimization) Black-box Attack.

Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
Implement ZOO (Chen et al., 2017) zeroth-order gradient estimation:
estimate pixel-wise gradients via finite differences on the adversarial loss (CE),
and perform coordinate-wise gradient ascent to flip predictions.

Budget: Subsample 200 test points, 100 iterations each.

Features:
  - margin: logit margin
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixels

Targets:
  - zoo_flipped: binary indicator of ZOO success
  - zoo_queries: total queries (model evaluations) used

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
ZOO_LR = 0.2
ZOO_H = 1e-4
MAX_ITER = 100
N_CLASSES = 10
N_SUB = 200  # Subsample size for ZOO black-box attack


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


def zoo_attack(model, x, y, lr=ZOO_LR, h=ZOO_H, max_iter=MAX_ITER):
    """
    ZOO zeroth-order coordinate-wise gradient ascent.
    Evaluates CE loss difference along a coordinate to estimate gradient.
    """
    model.eval()
    x_adv = x.clone()
    
    with torch.no_grad():
        if model(x_adv.unsqueeze(0)).argmax(1).item() != y:
            return True, 0
            
    queries = 1  # For initial prediction
    
    for step in range(max_iter):
        # Sample 10 random pixel coordinates to update in this step
        coords = torch.randperm(784)[:10]
        
        for coord in coords:
            r = coord // 28
            c = coord % 28
            
            # positive step
            x_plus = x_adv.clone()
            x_plus[0, r, c] = torch.clamp(x_plus[0, r, c] + h, 0.0, 1.0)
            
            # negative step
            x_minus = x_adv.clone()
            x_minus[0, r, c] = torch.clamp(x_minus[0, r, c] - h, 0.0, 1.0)
            
            with torch.no_grad():
                logits_plus = model(x_plus.unsqueeze(0))
                logits_minus = model(x_minus.unsqueeze(0))
                queries += 2
                
                loss_plus = F.cross_entropy(logits_plus, torch.tensor([y], device=x.device)).item()
                loss_minus = F.cross_entropy(logits_minus, torch.tensor([y], device=x.device)).item()
                
            # Finite differences gradient estimation
            grad = (loss_plus - loss_minus) / (2 * h)
            
            # Step in sign(gradient) to maximize cross-entropy loss (misclassify)
            x_adv[0, r, c] = torch.clamp(x_adv[0, r, c] + lr * np.sign(grad), 0.0, 1.0)
            
            # Query prediction to check if flipped
            with torch.no_grad():
                pred = model(x_adv.unsqueeze(0)).argmax(1).item()
                queries += 1
                if pred != y:
                    return True, queries
                    
    return False, queries


def main():
    print("=" * 60)
    print("Hypothesis H114: ZOO Zeroth-Order Attack Analysis")
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

    print(f"Evaluating ZOO on {x_sub.size(0)} correctly classified test samples...")

    # Compute features for subsample
    all_features = compute_features(x_sub, model)
    feature_names = ["margin", "mean_pix", "std_pix"]

    t0 = time.time()
    
    zoo_flipped = []
    zoo_queries = []
    
    for i in range(x_sub.size(0)):
        flipped, q = zoo_attack(model, x_sub[i], y_sub[i].item(), ZOO_LR, ZOO_H, MAX_ITER)
        zoo_flipped.append(flipped)
        zoo_queries.append(q)
        
    flipped_tensor = torch.tensor(zoo_flipped, dtype=torch.long, device=DEVICE)
    queries_tensor = torch.tensor(zoo_queries, dtype=torch.float, device=DEVICE)
    
    print(f"ZOO evaluations completed in {time.time() - t0:.1f}s")

    targets = {
        "zoo_flipped": flipped_tensor,
    }

    # Print vulnerability statistics
    print(f"\nZOO Attack Success Rate: {flipped_tensor.float().mean().item():.4f}")
    print(f"Mean Queries used: {queries_tensor.mean().item():.2f}")
    
    flipped_mask = flipped_tensor.cpu().numpy().astype(bool)
    if flipped_mask.any():
        mean_flipped_queries = queries_tensor[flipped_tensor.bool()].mean().item()
        print(f"Mean queries to flip (for successful attacks): {mean_flipped_queries:.2f}")

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
