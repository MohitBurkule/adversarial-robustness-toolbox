"""
Hypothesis H132: Snapshot ensemble disagreement predicts adversarial vulnerability.

Train small CNN on Fashion-MNIST for 30 epochs with a cosine annealing learning rate scheduler
having 3 cycles of 10 epochs. Save snapshot checkpoints at the end of each cycle (epochs 10, 20, 30).
Per test sample compute disagreement across the 3 snapshot models:
  - snap_disagreement: 1.0 - vote_agreement (where vote_agreement is the fraction of models agreeing on the majority prediction)
  - softmax_variance: average variance of the true-class softmax probability across the 3 snapshots
Use the last snapshot (Model 3) as the victim model.

Features:
  - snap_disagreement: snapshot ensemble vote disagreement
  - softmax_variance: softmax true-class probability variance across snapshots
  - margin: logit margin (top - 2nd logit) of the victim model (last snapshot)
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixel values

Targets (w.r.t. the victim model):
  - flipped_fgsm: binary indicator of whether FGSM flips prediction
  - flipped_pgd: binary indicator of whether PGD flips prediction
  - min_eps: minimum epsilon to flip via FGSM binary search

Univariate AUROC for each feature-target pair.
"""
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
CYCLE_LEN = 10
BATCH = 128
EPS = 15.0 / 255.0
LR_MAX = 1e-3


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

    print("Training CNN model with Cosine Annealing scheduler (3 cycles of 10 epochs)...")
    torch.manual_seed(0)
    np.random.seed(0)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_MAX)

    snapshots = []

    for epoch in range(EPOCHS):
        model.train()
        
        # Calculate Cosine Annealing Learning Rate manually
        # cos_theta = cos(pi * (epoch_in_cycle) / CYCLE_LEN)
        epoch_in_cycle = epoch % CYCLE_LEN
        lr = LR_MAX * 0.5 * (1.0 + np.cos(np.pi * epoch_in_cycle / CYCLE_LEN))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        
        # Save snapshot checkpoint at the end of each cycle (epochs 9, 19, 29)
        if (epoch + 1) % CYCLE_LEN == 0:
            print(f"Cycle completed at epoch {epoch+1}. Saving snapshot {len(snapshots)+1}.")
            # deepcopy the state dict to ensure we have a static copy
            snap = CNN(10).to(DEVICE)
            snap.load_state_dict(copy.deepcopy(model.state_dict()))
            snap.eval()
            snapshots.append(snap)

    victim = snapshots[-1] # The final model is our victim

    # Evaluate disagreement across 3 snapshots
    print("Evaluating snapshot ensemble on test set...")
    predictions = torch.zeros(N, len(snapshots), dtype=torch.long, device=DEVICE)
    confidences = torch.zeros(N, len(snapshots), device=DEVICE)
    correct_victim = torch.zeros(N, dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        # Victim correctness
        for i in range(0, N, 512):
            logits = victim(test_x[i:i+512])
            correct_victim[i:i+512] = (logits.argmax(dim=1) == test_y[i:i+512])

        # Get predictions and confidences for each snapshot
        for k, snap in enumerate(snapshots):
            for i in range(0, N, 512):
                logits = snap(test_x[i:i+512])
                predictions[i:i+512, k] = logits.argmax(dim=1)
                probs = F.softmax(logits, dim=1)
                confidences[i:i+512, k] = probs.gather(1, test_y[i:i+512].unsqueeze(1)).squeeze(1)

    # Compute vote disagreement:
    # vote_agreement is the fraction of models agreeing with the majority prediction.
    # Because K=3, the majority count can be 3 (agreement=1) or 2 (agreement=2/3=0.667).
    # vote_disagreement = 1.0 - vote_agreement
    vote_agreement = torch.zeros(N, device=DEVICE)
    for i in range(N):
        preds = predictions[i]
        vals, counts = torch.unique(preds, return_counts=True)
        majority_count = counts.max().item()
        vote_agreement[i] = majority_count / float(len(snapshots))

    snap_disagreement = 1.0 - vote_agreement
    softmax_variance = confidences.var(dim=1)

    # Filter to correctly predicted test samples for the victim
    x_c = test_x[correct_victim]
    y_c = test_y[correct_victim]
    snap_disagreement_c = snap_disagreement[correct_victim].cpu().numpy()
    softmax_variance_c = softmax_variance[correct_victim].cpu().numpy()
    N_c = x_c.size(0)
    print(f"Victim correctly classified samples: {N_c}/{N}")

    # Compute targets in batches
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
        "snap_disagreement": snap_disagreement_c,
        "softmax_variance": softmax_variance_c,
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
