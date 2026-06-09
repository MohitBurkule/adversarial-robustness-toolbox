"""
Hypothesis H135: Decomposed predictive uncertainty (aleatoric vs epistemic) predicts adversarial vulnerability.

Train K=5 CNN models with different seeds (0..4) on Fashion-MNIST for 10 epochs each.
For each test sample, compute three uncertainty measures:
  1. predictive_entropy = H(mean(softmax_k))
  2. aleatoric = mean(H(softmax_k))
  3. epistemic = predictive_entropy - aleatoric
Use Model 0 as the victim model.

Features:
  - predictive_entropy: total predictive entropy
  - aleatoric: aleatoric uncertainty component
  - epistemic: epistemic uncertainty component
  - margin: logit margin (top - 2nd logit) of the victim model (Model 0)
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixel values

Targets (w.r.t. Model 0):
  - flipped_fgsm: binary indicator of whether FGSM flips Model 0's prediction
  - flipped_pgd: binary indicator of whether PGD flips Model 0's prediction
  - min_eps: minimum epsilon to flip via FGSM binary search

Univariate AUROC for each feature-target pair.
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
K_MODELS = 5


class CNN(nn.Module):
    """Small CNN matching diagnostic_test.py."""
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


def train_model(train_set, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
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


def compute_entropy(probs, eps=1e-12):
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def main():
    print("Loading Fashion-MNIST dataset...")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    models = []
    for k in range(K_MODELS):
        print(f"Training model {k+1}/{K_MODELS}...")
        models.append(train_model(train_set, seed=k))

    victim = models[0]

    # Filter to correctly predicted test samples for Model 0
    print("Evaluating victim model correctness...")
    with torch.no_grad():
        preds = []
        for i in range(0, N, 512):
            preds.append(victim(test_x[i:i+512]).argmax(dim=1))
        preds = torch.cat(preds)
    correct_victim = (preds == test_y)
    x_c = test_x[correct_victim]
    y_c = test_y[correct_victim]
    N_c = x_c.size(0)
    print(f"Victim correctly classified samples: {N_c}/{N}")

    # Compute softmax outputs for all models in batches
    print("Evaluating models for uncertainty decomposition...")
    probs_stack_list = []
    
    with torch.no_grad():
        for i in range(0, N_c, 512):
            xb = x_c[i:i+512]
            B_curr = xb.size(0)
            
            # (B, K, 10)
            batch_probs = torch.zeros(B_curr, K_MODELS, 10, device=DEVICE)
            for k, m in enumerate(models):
                logits = m(xb)
                batch_probs[:, k] = F.softmax(logits, dim=1)
            probs_stack_list.append(batch_probs)

    probs_stack = torch.cat(probs_stack_list, dim=0) # (N_c, K, 10)
    probs_mean = probs_stack.mean(dim=1) # (N_c, 10)
    
    # 1. Total predictive entropy
    pred_entropy = compute_entropy(probs_mean).cpu().numpy()
    
    # 2. Aleatoric uncertainty
    ent_k = compute_entropy(probs_stack) # (N_c, K)
    aleatoric = ent_k.mean(dim=1).cpu().numpy()
    
    # 3. Epistemic uncertainty
    epistemic = pred_entropy - aleatoric

    # Compute targets in batches w.r.t. Model 0
    print("Computing attack targets...")
    flipped_fgsm_list = []
    flipped_pgd_list = []
    min_eps_list = []
    for i in range(0, N_c, 512):
        xb = x_c[i:i+512]
        yb = y_c[i:i+512]
        flipped_fgsm_list.append(attack_fgsm(victim, xb, yb))
        flipped_pgd_list.append(attack_pgd(victim, xb, yb))
        min_eps_list.append(min_eps_to_flip(victim, xb, yb))

    y_fgsm = torch.cat(flipped_fgsm_list).cpu().numpy().astype(int)
    y_pgd = torch.cat(flipped_pgd_list).cpu().numpy().astype(int)
    y_mineps = torch.cat(min_eps_list).cpu().numpy()

    # Compute features
    print("Computing features...")
    margin_list = []
    mean_pix_list = []
    std_pix_list = []

    for i in range(0, N_c, 512):
        xb = x_c[i:i+512]
        B_curr = xb.size(0)
        with torch.no_grad():
            logits = victim(xb)
            sorted_logits, _ = logits.sort(dim=1, descending=True)
            margin = sorted_logits[:, 0] - sorted_logits[:, 1]
            margin_list.append(margin)
            
            flat_x = xb.view(B_curr, -1)
            mean_pix_list.append(flat_x.mean(dim=1))
            std_pix_list.append(flat_x.std(dim=1))

    feats = {
        "predictive_entropy": pred_entropy,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "margin": torch.cat(margin_list).cpu().numpy(),
        "mean_pix": torch.cat(mean_pix_list).cpu().numpy(),
        "std_pix": torch.cat(std_pix_list).cpu().numpy(),
    }

    # Evaluate Univariate AUROC
    print("\n" + "="*50)
    print("UNIVARIATE AUROC RESULTS")
    print("="*50)
    
    for feat_name, feat_val in feats.items():
        print(f"\nFeature: {feat_name}")
        auc_fgsm = auroc_both_directions(y_fgsm, feat_val)
        auc_pgd = auroc_both_directions(y_pgd, feat_val)
        # For continuous target min_eps, we partition at the median to compute AUROC
        median_eps = np.median(y_mineps)
        y_mineps_bin = (y_mineps <= median_eps).astype(int)
        auc_mineps = auroc_both_directions(y_mineps_bin, feat_val)
        
        print(f"  Target: flipped_fgsm  AUROC = {auc_fgsm:.4f}")
        print(f"  Target: flipped_pgd   AUROC = {auc_pgd:.4f}")
        print(f"  Target: min_eps       AUROC = {auc_mineps:.4f}")


if __name__ == "__main__":
    main()
