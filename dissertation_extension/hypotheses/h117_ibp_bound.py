"""
Hypothesis H117: Certified Interval Bound Propagation (IBP) Robustness.

Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
Implement IBP (Gowal et al., 2018) certified bounds: forward propagate the input L_inf box [x-eps, x+eps]
through the CNN using interval arithmetic to get upper and lower bounds on logits.

A sample is certified robust at epsilon if lower-bound of the true class logit > upper-bound of all other logits.
Use binary search to find the maximum epsilon at which each sample remains certified robust (the IBP certified radius).

Features:
  - margin: logit margin
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixels
  - IBP_certified_radius: the computed certified radius via IBP binary search

Targets:
  - flipped_FGSM: binary success of FGSM at eps=15/255
  - flipped_PGD: binary success of PGD-10 at eps=15/255

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


def ibp_forward(model, x, eps):
    """
    Perform Interval Bound Propagation forward pass.
    x is a single image or batch, eps is the L_inf bounds radius.
    Returns:
      - logits_L: lower bounds of logits
      - logits_U: upper bounds of logits
    """
    x_L = torch.clamp(x - eps, 0.0, 1.0)
    x_U = torch.clamp(x + eps, 0.0, 1.0)
    
    def conv2d_ibp(conv, h_L, h_U):
        W = conv.weight
        b = conv.bias
        W_plus = torch.clamp(W, min=0.0)
        W_minus = torch.clamp(W, max=0.0)
        
        # Propagate lower bounds and upper bounds through conv layer using weight signs
        z_L = F.conv2d(h_L, W_plus, bias=None, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups) + \
              F.conv2d(h_U, W_minus, bias=None, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        z_U = F.conv2d(h_U, W_plus, bias=None, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups) + \
              F.conv2d(h_L, W_minus, bias=None, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        
        if b is not None:
            z_L = z_L + b.view(1, -1, 1, 1)
            z_U = z_U + b.view(1, -1, 1, 1)
        return z_L, z_U
        
    def linear_ibp(linear, h_L, h_U):
        W = linear.weight
        b = linear.bias
        W_plus = torch.clamp(W, min=0.0)
        W_minus = torch.clamp(W, max=0.0)
        
        # Propagate bounds through linear layer
        z_L = F.linear(h_L, W_plus, bias=None) + F.linear(h_U, W_minus, bias=None)
        z_U = F.linear(h_U, W_plus, bias=None) + F.linear(h_L, W_minus, bias=None)
        
        if b is not None:
            z_L = z_L + b
            z_U = z_U + b
        return z_L, z_U

    # Propagate through c1 + relu
    z1_L, z1_U = conv2d_ibp(model.c1, x_L, x_U)
    x1_L, x1_U = F.relu(z1_L), F.relu(z1_U)
    
    # Propagate through c2 + relu -> maxpool
    z2_L, z2_U = conv2d_ibp(model.c2, x1_L, x1_U)
    x2_L, x2_U = F.relu(z2_L), F.relu(z2_U)
    x2_L = F.max_pool2d(x2_L, 2)
    x2_U = F.max_pool2d(x2_U, 2)
    
    # Flatten
    x2_L_flat = x2_L.flatten(1)
    x2_U_flat = x2_U.flatten(1)
    
    # Propagate through fc1 + relu
    z3_L, z3_U = linear_ibp(model.fc1, x2_L_flat, x2_U_flat)
    x3_L, x3_U = F.relu(z3_L), F.relu(z3_U)
    
    # Propagate through fc2 (logits)
    logits_L, logits_U = linear_ibp(model.fc2, x3_L, x3_U)
    return logits_L, logits_U


def certified_radius_ibp(model, x, y, max_eps=0.5, steps=10):
    """Binary search for the certified robustness radius using IBP."""
    lo = 0.0
    hi = max_eps
    
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        logits_L, logits_U = ibp_forward(model, x.unsqueeze(0), mid)
        
        # Check if lower bound of true class is strictly greater than upper bounds of other classes
        true_L = logits_L[0, y].item()
        
        mask = torch.ones(10, dtype=torch.bool, device=x.device)
        mask[y] = False
        max_other_U = logits_U[0, mask].max().item()
        
        if true_L > max_other_U:
            lo = mid
        else:
            hi = mid
            
    return lo


def fgsm_attack(model, x, y, eps=EPS):
    """Standard single-step FGSM attack."""
    model.eval()
    x_orig = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_orig), y)
    loss.backward()
    grad_sign = x_orig.grad.sign().detach()
    x_adv = torch.clamp(x + eps * grad_sign, 0.0, 1.0)
    with torch.no_grad():
        return (model(x_adv).argmax(dim=1) != y)


def pgd_attack(model, x, y, eps=EPS, alpha=ALPHA, steps=10):
    """Standard PGD-10 attack."""
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
        return (model(x_adv).argmax(dim=1) != y)


def main():
    print("=" * 60)
    print("Hypothesis H117: certified IBP Robustness Analysis")
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

    print(f"Evaluating certified bounds and attacks on {x_sub.size(0)} correctly classified test samples...")

    # Compute features for subsample
    all_features = compute_features(x_sub, model)
    
    # Compute certified radii
    print("Computing IBP certified robustness radii...")
    t0 = time.time()
    ibp_radii = []
    for i in range(x_sub.size(0)):
        ibp_radii.append(certified_radius_ibp(model, x_sub[i], y_sub[i].item(), max_eps=0.5, steps=10))
    ibp_radii_tensor = torch.tensor(ibp_radii, dtype=torch.float, device=DEVICE)
    print(f"IBP certification completed in {time.time() - t0:.1f}s")
    
    # Run empirical attacks
    print("Running empirical attacks...")
    flipped_FGSM = fgsm_attack(model, x_sub, y_sub, EPS).long()
    flipped_PGD = pgd_attack(model, x_sub, y_sub, EPS, ALPHA, steps=10).long()

    # Append IBP certified radius as a feature
    full_features = torch.cat([all_features, ibp_radii_tensor.unsqueeze(1)], dim=1)
    feature_names = ["margin", "mean_pix", "std_pix", "IBP_certified_radius"]

    targets = {
        "flipped_FGSM": flipped_FGSM,
        "flipped_PGD": flipped_PGD
    }

    # Print baseline stats
    print(f"\nEmpirical Attack Success Rates:")
    print(f"  FGSM Success Rate: {flipped_FGSM.float().mean().item():.4f}")
    print(f"  PGD-10 Success Rate: {flipped_PGD.float().mean().item():.4f}")
    print(f"Certified Robustness Stats:")
    print(f"  Mean IBP Certified Radius: {ibp_radii_tensor.mean().item():.4f}")
    print(f"  Percentage certified robust at eps={EPS:.4f}: {(ibp_radii_tensor >= EPS).float().mean().item()*100:.2f}%")

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
        print(f"  {'Feature':<22} {'AUROC':>8} {'Direction':<10}")
        for i, fname in enumerate(feature_names):
            x_i = full_features[:, i].cpu().numpy()
            a = roc_auc_score(y, x_i)
            direction = "+" if a >= 0.5 else "-"
            a = max(a, 1 - a)
            print(f"  {fname:<22} {a:>8.4f}  {direction}")


if __name__ == "__main__":
    main()
