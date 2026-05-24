"""
Hypothesis H109: Momentum Iterative Method (MIM) Vulnerability Analysis.

Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
Implement MIM (Dong et al., 2018) at eps=15/255 with momentum decay=1.0 and 10 steps.

Features:
  - margin: logit margin (top - 2nd logit)
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixels
  - sobel_mean: spatial frequency proxy

Targets:
  - flipped_MIM: binary indicator of success at eps=15/255
  - MIM_min_eps: minimum L_inf eps that flips prediction using MIM (binary searched)

Comparison:
  - Compute FGSM min eps (single-step) and compare the rankings via rank correlation.

Univariate AUROC analysis for each feature-target pair.
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
EPS = 15.0 / 255.0
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


def mim_attack(model, x, y, eps, steps=10, decay=1.0):
    """
    Momentum Iterative Method (Dong et al., 2018).
    Returns success mask.
    """
    model.eval()
    B = x.size(0)
    alpha = 1.5 * eps / steps
    g = torch.zeros_like(x)
    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grad = x_adv.grad.detach()
        # L1 norm normalization of gradients
        grad_norm = torch.norm(grad, p=1, dim=(1, 2, 3), keepdim=True)
        grad = grad / (grad_norm + 1e-8)

        g = decay * g + grad
        x_adv = x_adv + alpha * g.sign()
        eta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + eta, min=0.0, max=1.0).detach()

    with torch.no_grad():
        preds = model(x_adv).argmax(1)
        return preds != y


def mim_min_eps_search(model, x, y, eps_max=0.3, steps=12):
    """Binary search for minimum eps to flip using MIM-10."""
    B = x.size(0)
    lo = torch.zeros(B, device=DEVICE)
    hi = torch.full((B,), eps_max, device=DEVICE)

    for _ in range(steps):
        mid = (lo + hi) / 2
        # Run batch-wise or element-wise. Since x is small we can do this in batch.
        flipped = mim_attack(model, x, y, mid.view(-1, 1, 1, 1), steps=10, decay=1.0)
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)

    return hi


def fgsm_min_eps_search(model, x, y, eps_max=0.3, steps=12):
    """Binary search for minimum eps to flip using standard single-step FGSM."""
    B = x.size(0)
    lo = torch.zeros(B, device=DEVICE)
    hi = torch.full((B,), eps_max, device=DEVICE)

    # Compute raw gradient sign once
    x_orig = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_orig), y)
    loss.backward()
    grad_sign = x_orig.grad.sign().detach()

    for _ in range(steps):
        mid = (lo + hi) / 2
        adv = torch.clamp(x + mid.view(-1, 1, 1, 1) * grad_sign, 0.0, 1.0)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)

    return hi


def main():
    print("=" * 60)
    print("Hypothesis H109: Momentum Iterative Method (MIM) Analysis")
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

    print(f"Evaluating MIM on {x_sub.size(0)} correctly classified test samples...")

    # Compute features for subsample
    all_features = compute_features(x_sub, model)
    feature_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    t0 = time.time()
    # 1. Evaluate MIM at eps=15/255
    flipped_MIM = mim_attack(model, x_sub, y_sub, EPS, steps=10, decay=1.0).long()
    print(f"MIM attack success at eps=15/255: {flipped_MIM.float().mean().item():.4f}")

    # 2. Binary search for min eps to flip via MIM
    print("Computing minimum eps to flip using MIM...")
    mim_min_eps = mim_min_eps_search(model, x_sub, y_sub)
    print(f"  Mean MIM min eps: {mim_min_eps.mean().item():.4f}")

    # 3. Binary search for min eps to flip via FGSM for comparison
    print("Computing minimum eps to flip using FGSM...")
    fgsm_min_eps = fgsm_min_eps_search(model, x_sub, y_sub)
    print(f"  Mean FGSM min eps: {fgsm_min_eps.mean().item():.4f}")
    print(f"MIM/FGSM evaluation completed in {time.time() - t0:.1f}s")

    # Univariate AUROC analysis against flipped_MIM
    print("\n" + "=" * 60)
    print("UNIVARIATE AUROC ANALYSIS (Target: flipped_MIM)")
    print("=" * 60)
    y_mim = flipped_MIM.cpu().numpy()
    if y_mim.std() == 0:
        print("Target flipped_MIM has no variance, skipping AUROC.")
    else:
        print(f"Target: flipped_MIM (Positive Rate = {y_mim.mean():.4f})")
        print(f"  {'Feature':<15} {'AUROC':>8} {'Direction':<10}")
        for i, fname in enumerate(feature_names):
            x_i = all_features[:, i].cpu().numpy()
            a = roc_auc_score(y_mim, x_i)
            direction = "+" if a >= 0.5 else "-"
            a = max(a, 1 - a)
            print(f"  {fname:<15} {a:>8.4f}  {direction}")

    # Spearman Rank Correlation between MIM and FGSM rankings
    print("\n" + "=" * 60)
    print("MIM vs FGSM VULNERABILITY RANKING COMPARISON")
    print("=" * 60)
    mim_ranks = mim_min_eps.cpu().numpy()
    fgsm_ranks = fgsm_min_eps.cpu().numpy()
    
    rho, pval = spearmanr(mim_ranks, fgsm_ranks)
    print(f"Spearman rank correlation coefficient: {rho:+.6f} (p-value: {pval:.2e})")
    
    # Analyze if ranking matches on a per-sample basis
    print("\nMean MIM/FGSM Min Eps by margin quartile:")
    margins = all_features[:, 0].cpu().numpy()
    quartiles = np.quantile(margins, [0.0, 0.25, 0.5, 0.75, 1.0])
    for q in range(4):
        mask = (margins >= quartiles[q]) & (margins <= quartiles[q+1])
        sub_mim = mim_ranks[mask].mean()
        sub_fgsm = fgsm_ranks[mask].mean()
        print(f"  Quartile {q+1} [{quartiles[q]:.2f}, {quartiles[q+1]:.2f}] (n={mask.sum()}):")
        print(f"    Mean MIM Min Eps:  {sub_mim:.4f}")
        print(f"    Mean FGSM Min Eps: {sub_fgsm:.4f}")


if __name__ == "__main__":
    main()
