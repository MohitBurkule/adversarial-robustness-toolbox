"""
Hypothesis H111: Expectation Over Transformations (EOT) Robust Attack.

Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
Implement EOT (Athalye et al., 2018) via PGD, averaging gradients over K=8 random rotations and translations.

Features:
  - margin: logit margin
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixels
  - sobel_mean: spatial frequency proxy

Targets:
  - flipped_EOT: binary indicator of EOT success at eps=15/255
  - flipped_PGD: binary indicator of plain PGD-10 success at eps=15/255

Compare EOT success ranking with plain PGD success ranking.
Compute univariate AUROC for both.
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
N_SUB = 500  # Reduced to 500 due to K=8 EOT compute requirements


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


def random_affine_differentiable(x, max_angle=10.0, max_translate=2.0):
    """Apply random differentiable rotation and translation using bilinear grid mapping."""
    B, C, H, W = x.size()
    # Convert angle to radians
    angle = (torch.rand(B, device=x.device) * 2.0 - 1.0) * (max_angle * np.pi / 180.0)
    tx = (torch.rand(B, device=x.device) * 2.0 - 1.0) * (max_translate / (W / 2.0))
    ty = (torch.rand(B, device=x.device) * 2.0 - 1.0) * (max_translate / (H / 2.0))
    
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty
    
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x_trans = F.grid_sample(x, grid, align_corners=False)
    return x_trans


def eot_attack(model, x, y, eps=EPS, alpha=ALPHA, steps=10, K=8):
    """EOT PGD attack with differentiable transformation averaging."""
    model.eval()
    B = x.size(0)
    
    # Initialize random perturbation inside L_inf ball
    noise = torch.FloatTensor(*x.shape).uniform_(-eps, eps).to(DEVICE)
    x_adv = torch.clamp(x + noise, 0.0, 1.0).detach().requires_grad_(True)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        
        # Average gradients over K transformation samples
        grad = torch.zeros_like(x_adv)
        for _ in range(K):
            # Differentiable transformation
            x_trans = random_affine_differentiable(x_adv, max_angle=10.0, max_translate=2.0)
            logits = model(x_trans)
            loss = F.cross_entropy(logits, y)
            
            # Compute gradient contribution
            temp_grad = torch.autograd.grad(loss, x_adv)[0]
            grad += temp_grad / K
            
        x_adv = x_adv + alpha * grad.sign()
        eta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + eta, min=0.0, max=1.0).detach()

    with torch.no_grad():
        # Evaluate on the original clean sample to check if the EOT pert is effective
        preds = model(x_adv).argmax(1)
        return preds != y


def plain_pgd_attack(model, x, y, eps=EPS, alpha=ALPHA, steps=10):
    """Standard PGD-10 attack (baseline without transformation)."""
    model.eval()
    noise = torch.FloatTensor(*x.shape).uniform_(-eps, eps).to(DEVICE)
    x_adv = torch.clamp(x + noise, 0.0, 1.0).detach().requires_grad_(True)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        grad = x_adv.grad.sign().detach()
        x_adv = x_adv + alpha * grad
        eta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + eta, min=0.0, max=1.0).detach()

    with torch.no_grad():
        preds = model(x_adv).argmax(1)
        return preds != y


def main():
    print("=" * 60)
    print("Hypothesis H111: Expectation Over Transformations (EOT) Attack")
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

    print(f"Evaluating attacks on {x_sub.size(0)} correctly classified test samples...")

    # Compute features for subsample
    all_features = compute_features(x_sub, model)
    feature_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    # 1. Plain PGD-10
    print("Running plain PGD-10 attack (baseline)...")
    t0 = time.time()
    flipped_PGD = plain_pgd_attack(model, x_sub, y_sub).long()
    print(f"  Plain PGD Success Rate: {flipped_PGD.float().mean().item():.4f}")

    # 2. EOT PGD-10
    print("Running EOT PGD-10 attack (K=8 rotation+translation)...")
    t1 = time.time()
    flipped_EOT = eot_attack(model, x_sub, y_sub, K=8).long()
    print(f"  EOT Success Rate: {flipped_EOT.float().mean().item():.4f}")
    print(f"Evaluations completed: PGD ({t1-t0:.1f}s), EOT ({time.time()-t1:.1f}s)")

    targets = {
        "flipped_PGD": flipped_PGD,
        "flipped_EOT": flipped_EOT
    }

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

    # Disagreement analysis between EOT and PGD
    print("\n" + "=" * 60)
    print("ATTACK CONSISTENCY COMPARISON")
    print("=" * 60)
    both_success = (flipped_PGD & flipped_EOT).float().mean().item()
    pgd_only = (flipped_PGD & (1 - flipped_EOT)).float().mean().item()
    eot_only = ((1 - flipped_PGD) & flipped_EOT).float().mean().item()
    neither = ((1 - flipped_PGD) & (1 - flipped_EOT)).float().mean().item()
    
    print(f"  Both attacks succeeded:              {both_success:.4f}")
    print(f"  Only plain PGD succeeded:            {pgd_only:.4f}")
    print(f"  Only EOT succeeded:                  {eot_only:.4f}")
    print(f"  Neither attack succeeded:            {neither:.4f}")


if __name__ == "__main__":
    main()
