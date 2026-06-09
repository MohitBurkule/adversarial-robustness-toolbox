"""
Hypothesis H110: Diverse Input Method (DIM) Transferability.

Train two small CNNs (Model A and Model B) matching diagnostic_test.py on Fashion-MNIST for 10 epochs each.
Implement DIM (Xie et al., 2019) at eps=15/255, 10 steps, which applies random resize and padding before gradients.

Features (computed on Model A):
  - margin: logit margin
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixels
  - sobel_mean: spatial frequency proxy

Targets:
  - flipped_DIM: binary indicator of success of DIM generated on Model A and evaluated on Model A
  - DIM_transfer_to_other_model: binary indicator of success of DIM generated on Model B and transferred to Model A

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


def train_model(train_set, test_set, seed):
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
        print(f"  Seed {seed}: Epoch {epoch+1}/{EPOCHS} completed")

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


def dim_attack(model, x, y, eps=EPS, alpha=ALPHA, steps=10, p=0.9, low=24, high=28):
    """
    Diverse Input Method (DIM) (Xie et al., 2019).
    Applies random resize + padding with probability p at each step.
    """
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        
        # input diversity transformation
        if torch.rand(1).item() < p:
            # Random resize size r
            r = torch.randint(low, high + 1, (1,)).item()
            x_resized = F.interpolate(x_adv, size=(r, r), mode='bilinear', align_corners=False)
            
            # Random padding
            pad_left = torch.randint(0, 28 - r + 1, (1,)).item()
            pad_right = 28 - r - pad_left
            pad_top = torch.randint(0, 28 - r + 1, (1,)).item()
            pad_bottom = 28 - r - pad_top
            x_trans = F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        else:
            x_trans = x_adv

        loss = F.cross_entropy(model(x_trans), y)
        loss.backward()
        
        grad = x_adv.grad.sign().detach()
        x_adv = x_adv + alpha * grad
        eta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + eta, min=0.0, max=1.0).detach()
        
    return x_adv


def main():
    print("=" * 60)
    print("Hypothesis H110: Diverse Input Method (DIM) Analysis")
    print("=" * 60)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Train Model A (seed 0) and Model B (seed 1)
    print("\nTraining Model A (victim, seed 0)...")
    t0 = time.time()
    model_A, test_x, test_y = train_model(train_set, test_set, seed=0)
    print(f"Model A training completed in {time.time() - t0:.1f}s")

    print("\nTraining Model B (surrogate for transfer, seed 1)...")
    t0 = time.time()
    model_B, _, _ = train_model(train_set, test_set, seed=1)
    print(f"Model B training completed in {time.time() - t0:.1f}s")

    # Filter to correctly classified samples on BOTH models
    model_A.eval()
    model_B.eval()
    with torch.no_grad():
        preds_A = []
        preds_B = []
        for i in range(0, test_x.size(0), 512):
            batch_x = test_x[i:i+512]
            preds_A.append(model_A(batch_x).argmax(1))
            preds_B.append(model_B(batch_x).argmax(1))
        preds_A = torch.cat(preds_A, 0)
        preds_B = torch.cat(preds_B, 0)
        correct = (preds_A == test_y) & (preds_B == test_y)

    x_c = test_x[correct]
    y_c = test_y[correct]

    # Subsample correctly classified instances to keep execution fast
    torch.manual_seed(0)
    indices = torch.randperm(x_c.size(0))[:N_SUB]
    x_sub = x_c[indices]
    y_sub = y_c[indices]

    print(f"Evaluating DIM on {x_sub.size(0)} double-correct test samples...")

    # Compute features for subsample based on Model A
    all_features = compute_features(x_sub, model_A)
    feature_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    # 1. Self DIM attack: generated on A, evaluated on A
    print("Running DIM attack on Model A (self)...")
    t0 = time.time()
    adv_self = dim_attack(model_A, x_sub, y_sub)
    with torch.no_grad():
        flipped_DIM = (model_A(adv_self).argmax(1) != y_sub).long()
    print(f"DIM self-attack success rate: {flipped_DIM.float().mean().item():.4f}")

    # 2. Transfer DIM attack: generated on B, evaluated on A
    print("Running DIM attack on Model B (transfer surrogate)...")
    adv_trans = dim_attack(model_B, x_sub, y_sub)
    with torch.no_grad():
        flipped_trans = (model_A(adv_trans).argmax(1) != y_sub).long()
    print(f"DIM transfer success rate (B -> A): {flipped_trans.float().mean().item():.4f}")
    print(f"DIM evaluations completed in {time.time() - t0:.1f}s")

    targets = {
        "flipped_DIM": flipped_DIM,
        "DIM_transfer_to_other_model": flipped_trans
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


if __name__ == "__main__":
    main()
