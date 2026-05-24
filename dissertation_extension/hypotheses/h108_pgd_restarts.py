"""
Hypothesis H108: PGD Random Restarts and Adversarial Vulnerability.

Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
Run PGD-10 at eps=15/255 with K in {1, 5, 10, 20} random restarts per sample.

Record:
  - flipped_at_K_restarts (binary for K in {1, 5, 10, 20})
  - min_restarts_to_flip: the restart number (1 to 20) where the sample first flips.
    If it never flips, set it to 21.

Features:
  - margin: logit margin (top - 2nd logit)
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixels
  - sobel_mean: mean Sobel filter gradient magnitude (spatial frequency proxy)

Compute univariate AUROC for each feature-target pair.
Analyze if samples needing more restarts have different feature distributions.
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
N_CLASSES = 10
N_SUB = 1000  # Subsample size to keep execution times reasonable


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


def pgd_attack_restarts(model, x, y, max_restarts=20, eps=EPS, alpha=ALPHA):
    """
    Run PGD-10 with random restarts.
    Returns:
      - first_flip_restart: 1-indexed restart where it first flipped, or 21 if it never flips.
      - flipped_at_k: dict mapping k to bool list
    """
    model.eval()
    B = x.size(0)
    first_flip = torch.full((B,), max_restarts + 1, dtype=torch.long, device=DEVICE)
    
    # We will record if flipped at each restart
    flipped_any = torch.zeros(B, dtype=torch.bool, device=DEVICE)

    for r in range(1, max_restarts + 1):
        # Initialize random perturbation inside L_inf ball
        noise = torch.FloatTensor(*x.shape).uniform_(-eps, eps).to(DEVICE)
        x_adv = torch.clamp(x + noise, 0.0, 1.0).detach().requires_grad_(True)

        for step in range(10):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            outputs = model(x_adv)
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            grad = x_adv.grad.sign().detach()
            
            # Projection step
            x_adv = x_adv + alpha * grad
            eta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + eta, min=0.0, max=1.0).detach()

        # Check which samples are successfully flipped in this restart
        with torch.no_grad():
            preds = model(x_adv).argmax(1)
            flipped = (preds != y)
            
        # Update first flip restart
        new_flips = flipped & ~flipped_any
        first_flip[new_flips] = r
        flipped_any = flipped_any | flipped

    return first_flip


def main():
    print("=" * 60)
    print("Hypothesis H108: PGD Random Restarts Analysis")
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

    print(f"Running restarts evaluation on {x_sub.size(0)} correctly classified test samples...")
    
    # Compute features for subsample
    all_features = compute_features(x_sub, model)
    feature_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    t0 = time.time()
    min_restarts_to_flip = pgd_attack_restarts(model, x_sub, y_sub, max_restarts=20, eps=EPS, alpha=ALPHA)
    print(f"PGD evaluation completed in {time.time() - t0:.1f}s")

    # Compute binary targets for K in {1, 5, 10, 20}
    flipped_at_1 = (min_restarts_to_flip <= 1).long()
    flipped_at_5 = (min_restarts_to_flip <= 5).long()
    flipped_at_10 = (min_restarts_to_flip <= 10).long()
    flipped_at_20 = (min_restarts_to_flip <= 20).long()

    targets = {
        "flipped_at_1": flipped_at_1,
        "flipped_at_5": flipped_at_5,
        "flipped_at_10": flipped_at_10,
        "flipped_at_20": flipped_at_20
    }

    # Print baseline stats
    print("\nAdversarial Success by Restart Limit K:")
    for K, t in [("1", flipped_at_1), ("5", flipped_at_5), ("10", flipped_at_10), ("20", flipped_at_20)]:
        print(f"  K = {K}: Success rate = {t.float().mean().item():.4f}")

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

    # Feature differences by restarts needed
    print("\n" + "=" * 60)
    print("ANALYSIS OF SAMPLES NEEDING MORE RESTARTS")
    print("=" * 60)
    
    # We partition samples by how many restarts were required:
    # Group A: Easy (flipped on 1st restart)
    # Group B: Medium (flipped in 2 to 10 restarts)
    # Group C: Hard (flipped in 11 to 20 restarts)
    # Group D: Robust (never flipped)
    restarts_np = min_restarts_to_flip.cpu().numpy()
    groups = {
        "Easy (K=1)": restarts_np == 1,
        "Medium (1<K<=10)": (restarts_np > 1) & (restarts_np <= 10),
        "Hard (10<K<=20)": (restarts_np > 10) & (restarts_np <= 20),
        "Robust (K>20)": restarts_np > 20
    }
    
    for name, mask in groups.items():
        count = mask.sum()
        if count == 0:
            print(f"  {name}: 0 samples")
            continue
        mean_feats = all_features[torch.tensor(mask, dtype=torch.bool)].mean(dim=0).cpu().numpy()
        print(f"  {name} (n={count}):")
        for i, fname in enumerate(feature_names):
            print(f"    {fname:<15}: {mean_feats[i]:.4f}")


if __name__ == "__main__":
    main()
