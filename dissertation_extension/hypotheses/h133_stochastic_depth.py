"""
Hypothesis H133: Stochastic Depth training (Huang 2016) impacts adversarial vulnerability and predictor AUROCs.

Train two victim models on Fashion-MNIST for 10 epochs each:
  1. Vanilla CNN (standard architecture matching diagnostic_test.py).
  2. Stochastic Depth CNN (randomly dropping conv layers with probability 0.2 during training,
     using skip shortcuts to match channel and spatial dimensions).

For each model, compute features:
  - margin: logit margin (top - 2nd logit)
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixel values

Targets (computed separately per model):
  - flipped_fgsm: binary indicator of whether FGSM flips prediction
  - flipped_pgd: binary indicator of whether PGD flips prediction
  - min_eps: minimum epsilon to flip via FGSM binary search

Compare the univariate AUROCs of the features against the targets for both models.
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


# ---- 1. Standard CNN matching diagnostic_test.py ----
class VanillaCNN(nn.Module):
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


# ---- 2. Stochastic Depth CNN ----
class StochasticDepthCNN(nn.Module):
    def __init__(self, n=10, drop_prob=0.2):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)
        self.drop_prob = drop_prob
        
        # Shortcuts for shape matching (W_out = W_in - 2)
        self.skip1 = nn.Sequential(
            nn.Conv2d(1, 32, 1),
            nn.AvgPool2d(3, stride=1, padding=0)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(32, 64, 1),
            nn.AvgPool2d(3, stride=1, padding=0)
        )

    def forward(self, x):
        # Layer 1
        if self.training:
            if np.random.rand() > self.drop_prob:
                x1 = F.relu(self.c1(x))
            else:
                x1 = F.relu(self.skip1(x))
        else:
            x1 = (1.0 - self.drop_prob) * F.relu(self.c1(x)) + self.drop_prob * F.relu(self.skip1(x))

        # Layer 2
        if self.training:
            if np.random.rand() > self.drop_prob:
                x2 = F.relu(self.c2(x1))
            else:
                x2 = F.relu(self.skip2(x1))
        else:
            x2 = (1.0 - self.drop_prob) * F.relu(self.c2(x1)) + self.drop_prob * F.relu(self.skip2(x1))

        x = F.max_pool2d(x2, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


def train_model(model, train_set, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def fgsm_grad(model, x, y):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return grad.sign().detach()


def attack_fgsm(model, x, y, eps=EPS):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def attack_pgd(model, x, y, eps=EPS, alpha=2.0/255.0, steps=10):
    x_adv = x.clone().detach().requires_grad_(True)
    for _ in range(steps):
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = x_adv.clamp(0.0, 1.0).detach().requires_grad_(True)
    with torch.no_grad():
        return (model(x_adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def auroc_both_directions(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.5
    a = roc_auc_score(y_true, y_score)
    return max(a, 1.0 - a)


def evaluate_model_pipeline(model, name, train_set, test_x, test_y):
    print(f"\n--- Training {name} ---")
    model = train_model(model, train_set)
    
    N = test_x.size(0)
    with torch.no_grad():
        preds = []
        for i in range(0, N, 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct = (preds == test_y)
    x_c = test_x[correct]
    y_c = test_y[correct]
    N_c = x_c.size(0)
    print(f"{name} Correctly classified: {N_c}/{N}")
    
    # Targets
    print(f"Computing attack targets for {name}...")
    flipped_fgsm_list = []
    flipped_pgd_list = []
    min_eps_list = []
    for i in range(0, N_c, 512):
        xb = x_c[i:i+512]
        yb = y_c[i:i+512]
        flipped_fgsm_list.append(attack_fgsm(model, xb, yb))
        flipped_pgd_list.append(attack_pgd(model, xb, yb))
        min_eps_list.append(min_eps_to_flip(model, xb, yb))

    y_fgsm = torch.cat(flipped_fgsm_list).cpu().numpy().astype(int)
    y_pgd = torch.cat(flipped_pgd_list).cpu().numpy().astype(int)
    y_mineps = torch.cat(min_eps_list).cpu().numpy()

    # Features
    print(f"Computing features for {name}...")
    margin_list = []
    mean_pix_list = []
    std_pix_list = []
    for i in range(0, N_c, 512):
        xb = x_c[i:i+512]
        B_curr = xb.size(0)
        with torch.no_grad():
            logits = model(xb)
            sorted_logits, _ = logits.sort(dim=1, descending=True)
            margin_list.append(sorted_logits[:, 0] - sorted_logits[:, 1])
            
            flat_x = xb.view(B_curr, -1)
            mean_pix_list.append(flat_x.mean(dim=1))
            std_pix_list.append(flat_x.std(dim=1))

    feats = {
        "margin": torch.cat(margin_list).cpu().numpy(),
        "mean_pix": torch.cat(mean_pix_list).cpu().numpy(),
        "std_pix": torch.cat(std_pix_list).cpu().numpy(),
    }

    print(f"\nUNIVARIATE AUROC RESULTS for {name}:")
    for feat_name, feat_val in feats.items():
        auc_fgsm = auroc_both_directions(y_fgsm, feat_val)
        auc_pgd = auroc_both_directions(y_pgd, feat_val)
        
        median_eps = np.median(y_mineps)
        y_mineps_bin = (y_mineps <= median_eps).astype(int)
        auc_mineps = auroc_both_directions(y_mineps_bin, feat_val)
        
        print(f"  Feature: {feat_name:<10} | FGSM: {auc_fgsm:.4f} | PGD: {auc_pgd:.4f} | MinEps: {auc_mineps:.4f}")


def main():
    print("Loading Fashion-MNIST dataset...")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # Evaluate Vanilla Model
    vanilla_model = VanillaCNN(10).to(DEVICE)
    evaluate_model_pipeline(vanilla_model, "Vanilla CNN", train_set, test_x, test_y)

    # Evaluate Stochastic Depth Model
    stochastic_model = StochasticDepthCNN(10, drop_prob=0.2).to(DEVICE)
    evaluate_model_pipeline(stochastic_model, "Stochastic Depth CNN", train_set, test_x, test_y)


if __name__ == "__main__":
    main()
