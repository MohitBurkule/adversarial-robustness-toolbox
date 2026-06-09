"""
Hypothesis H129: Forgetting events across training epochs predict test sample adversarial vulnerability.

Train small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
Record per-sample correctness on test set at the end of each epoch.
Compute forgetting_event_count (Toneva ICLR 2019): number of correct -> wrong transitions.
Features:
  - forgetting_event_count: number of forgetting events on the test sample
  - margin: logit margin (top - 2nd logit)
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixel values

Targets:
  - flipped_fgsm: binary indicator of whether FGSM flips the prediction
  - flipped_pgd: binary indicator of whether PGD flips the prediction
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


def main():
    print("Loading Fashion-MNIST dataset...")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    print("Training CNN model with forgetting tracking...")
    torch.manual_seed(0)
    np.random.seed(0)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # correctness_history shape (N, EPOCHS)
    correctness_history = torch.zeros(N, EPOCHS, dtype=torch.bool, device=DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        
        # Evaluate on test set at the end of epoch
        model.eval()
        with torch.no_grad():
            for i in range(0, N, 512):
                logits = model(test_x[i:i+512])
                correctness_history[i:i+512, epoch] = (logits.argmax(dim=1) == test_y[i:i+512])
        
        print(f"Epoch {epoch+1}/{EPOCHS} done")

    # Compute forgetting events per test sample
    # Transitions from 1 (correct) to 0 (wrong)
    history_float = correctness_history.float()
    diff = history_float[:, :-1] - history_float[:, 1:]
    forgetting_events = (diff == 1).sum(dim=1).cpu().numpy()

    # Filter to correctly predicted test samples at final epoch
    final_correct = correctness_history[:, -1]
    x_c = test_x[final_correct]
    y_c = test_y[final_correct]
    forgetting_events_c = forgetting_events[final_correct.cpu().numpy()]
    N_c = x_c.size(0)
    print(f"Correctly classified samples at epoch 10: {N_c}/{N}")

    # Compute targets in batches
    print("Computing attack targets...")
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

    # Compute features
    print("Computing features...")
    margin_list = []
    mean_pix_list = []
    std_pix_list = []

    for i in range(0, N_c, 512):
        xb = x_c[i:i+512]
        B_curr = xb.size(0)
        with torch.no_grad():
            logits = model(xb)
            sorted_logits, _ = logits.sort(dim=1, descending=True)
            margin = sorted_logits[:, 0] - sorted_logits[:, 1]
            margin_list.append(margin)
            
            flat_x = xb.view(B_curr, -1)
            mean_pix_list.append(flat_x.mean(dim=1))
            std_pix_list.append(flat_x.std(dim=1))

    feats = {
        "forgetting_event_count": forgetting_events_c,
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
