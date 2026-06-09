"""
Hypothesis H113: SimBA (Simple Black-box Attack) Pixel-level Analysis.

Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
Implement SimBA-Pixel (Guo et al., 2019): for each test sample, iteratively query random pixel directions
and apply +eps or -eps depending on which decreases the correct class probability.

Budget: 200 queries per sample.

Features:
  - margin: logit margin
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixels
  - sobel_mean: spatial frequency proxy

Targets:
  - simba_flipped: binary indicator of whether SimBA flips the prediction within 200 queries
  - simba_queries_to_flip: the number of queries required to flip the prediction (200 if never flips)

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
SIMBA_EPS = 0.2
MAX_QUERIES = 200
N_CLASSES = 10
N_SUB = 200  # Subsample size for black-box iterative attacks to maintain fast execution


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


def simba_pixel(model, x, y, eps=SIMBA_EPS, max_queries=MAX_QUERIES):
    """
    SimBA-Pixel black-box decision attack.
    Queries random orthonormal pixel directions and applies +eps/-eps.
    """
    model.eval()
    x_adv = x.clone()
    
    with torch.no_grad():
        # Get baseline prediction and correct-class probability
        logits = model(x_adv.unsqueeze(0))
        p_orig = F.softmax(logits, dim=1)[0, y].item()
        
    queries = 1
    
    # 28x28 = 784 coordinates
    indices = torch.randperm(784)
    
    for idx in indices:
        if queries >= max_queries:
            break
            
        r = idx // 28
        c = idx % 28
        
        # 1. Try positive step
        x_temp = x_adv.clone()
        x_temp[0, r, c] = torch.clamp(x_temp[0, r, c] + eps, 0.0, 1.0)
        
        with torch.no_grad():
            logits = model(x_temp.unsqueeze(0))
            queries += 1
            if logits.argmax(1).item() != y:
                return True, queries
            p_plus = F.softmax(logits, dim=1)[0, y].item()
            
        if p_plus < p_orig:
            x_adv = x_temp
            p_orig = p_plus
            continue
            
        if queries >= max_queries:
            break
            
        # 2. Try negative step
        x_temp = x_adv.clone()
        x_temp[0, r, c] = torch.clamp(x_temp[0, r, c] - eps, 0.0, 1.0)
        
        with torch.no_grad():
            logits = model(x_temp.unsqueeze(0))
            queries += 1
            if logits.argmax(1).item() != y:
                return True, queries
            p_minus = F.softmax(logits, dim=1)[0, y].item()
            
        if p_minus < p_orig:
            x_adv = x_temp
            p_orig = p_minus
            
    with torch.no_grad():
        final_pred = model(x_adv.unsqueeze(0)).argmax(1).item()
        
    return (final_pred != y), queries


def main():
    print("=" * 60)
    print("Hypothesis H113: SimBA-Pixel Attack Analysis")
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

    print(f"Evaluating SimBA on {x_sub.size(0)} correctly classified test samples...")

    # Compute features for subsample
    all_features = compute_features(x_sub, model)
    feature_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    t0 = time.time()
    
    simba_flipped = []
    simba_queries = []
    
    for i in range(x_sub.size(0)):
        flipped, q = simba_pixel(model, x_sub[i], y_sub[i].item(), SIMBA_EPS, MAX_QUERIES)
        simba_flipped.append(flipped)
        simba_queries.append(q)
        
    flipped_tensor = torch.tensor(simba_flipped, dtype=torch.long, device=DEVICE)
    queries_tensor = torch.tensor(simba_queries, dtype=torch.float, device=DEVICE)
    
    print(f"SimBA evaluations completed in {time.time() - t0:.1f}s")

    targets = {
        "simba_flipped": flipped_tensor,
    }

    # Print vulnerability statistics
    print(f"\nSimBA Attack Success Rate: {flipped_tensor.float().mean().item():.4f}")
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

    # Spearman correlation of queries to flip with features (restricted to flipped samples)
    if flipped_mask.any():
        from scipy.stats import spearmanr
        print("\n" + "=" * 60)
        print("SPEARMAN CORRELATION WITH QUERIES TO FLIP (Successful Flips only)")
        print("=" * 60)
        q_np = queries_tensor.cpu().numpy()[flipped_mask]
        f_np = all_features.cpu().numpy()[flipped_mask]
        
        for i, fname in enumerate(feature_names):
            rho, pval = spearmanr(f_np[:, i], q_np)
            print(f"  {fname:<15} -> Correlation: {rho:+.4f} (p-value: {pval:.2e})")


if __name__ == "__main__":
    main()
