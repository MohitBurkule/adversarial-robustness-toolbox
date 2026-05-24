"""
Hypothesis H150: Confusion-Class Probability Ratio.

Train a small CNN on Fashion-MNIST for 10 epochs.
For each test sample, compute the softmax probabilities:
- confusion_ratio = top2_softmax / top1_softmax
- top3_to_top1_ratio = top3_softmax / top1_softmax

Compare these ratio-based features with raw logit margin to see if ratios beat absolute margin:
- Features: confusion_ratio, top3_to_top1_ratio, margin, mean_pix, std_pix.
- Targets: flipped_FGSM, flipped_PGD, min_eps (binarized at median).
- Univariate AUROC.
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
N_CLASSES = 10


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


def compute_confusion_features(x, model):
    """Compute softmax-based probability ratios and raw logit margin."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    # Sort logits and probabilities descending
    sorted_logits, _ = logits.sort(dim=1, descending=True)
    sorted_probs, _ = probs.sort(dim=1, descending=True)

    # Margin: difference between top 1 and top 2 logits
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]

    # Ratios
    confusion_ratio = sorted_probs[:, 1] / (sorted_probs[:, 0] + 1e-9)
    top3_to_top1_ratio = sorted_probs[:, 2] / (sorted_probs[:, 0] + 1e-9)

    # Image statistics
    N = x.size(0)
    flat_x = x.view(N, -1)
    mean_pix = flat_x.mean(1)
    std_pix = flat_x.std(1)

    return (
        confusion_ratio.cpu().numpy(),
        top3_to_top1_ratio.cpu().numpy(),
        margin.cpu().numpy(),
        mean_pix.cpu().numpy(),
        std_pix.cpu().numpy()
    )


# --- Attack Implementations ---

def attack_fgsm(model, x, y, eps=EPS):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    grad = x_adv.grad.sign().detach()
    adv = (x + eps * grad).clamp(0, 1)
    with torch.no_grad():
        return model(adv).argmax(1) != y


def attack_pgd(model, x, y, eps=EPS, steps=10):
    model.eval()
    alpha = eps / 5.0
    x_adv = x.clone().detach() + (torch.rand_like(x) * 2 - 1) * eps
    x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.sign().detach()
        x_adv = torch.clamp(x_adv + alpha * grad, x - eps, x + eps).clamp(0, 1).detach()
    with torch.no_grad():
        return model(x_adv).argmax(1) != y


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Binary search for minimum FGSM perturbation to flip classification."""
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)

    x_grad = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_grad), y).backward()
    sign = x_grad.grad.sign().detach()

    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def main():
    print("=" * 70)
    print("Hypothesis H150: Confusion-Class Probability Ratio")
    print("=" * 70)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("\nTraining model...")
    t0 = time.time()
    model, test_x, test_y = train_model(train_set, test_set)
    print(f"Training completed in {time.time() - t0:.1f}s")

    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        pred = logits.argmax(1)
        correct = (pred == test_y)

    x_c = test_x[correct][:1000]
    y_c = test_y[correct][:1000]
    N = x_c.size(0)
    print(f"\nUsing {N} correctly classified samples for evaluation.")

    # Compute features
    print("\nComputing features...")
    confusion_ratio, top3_to_top1, margin, mean_pix, std_pix = compute_confusion_features(x_c, model)

    # Run attacks
    print("\nEvaluating attack outcomes...")
    batch_size = 256
    fgsm_success = []
    pgd_success = []
    min_eps_list = []

    for i in range(0, N, batch_size):
        xb = x_c[i:i+batch_size]
        yb = y_c[i:i+batch_size]

        fgsm_success.append(attack_fgsm(model, xb, yb))
        pgd_success.append(attack_pgd(model, xb, yb))
        min_eps_list.append(min_eps_to_flip(model, xb, yb))

    fgsm_success = torch.cat(fgsm_success)
    pgd_success = torch.cat(pgd_success)
    min_eps = torch.cat(min_eps_list)

    # Convert targets to numpy
    flipped_FGSM = fgsm_success.cpu().numpy().astype(int)
    flipped_PGD = pgd_success.cpu().numpy().astype(int)

    min_eps_np = min_eps.cpu().numpy()
    median_eps = np.median(min_eps_np)
    vulnerable_min_eps = (min_eps_np < median_eps).astype(int)

    targets = {
        "flipped_FGSM": flipped_FGSM,
        "flipped_PGD": flipped_PGD,
        "min_eps_below_median": vulnerable_min_eps
    }

    features = {
        "confusion_ratio": confusion_ratio,
        "top3_to_top1_ratio": top3_to_top1,
        "margin": margin,
        "mean_pix": mean_pix,
        "std_pix": std_pix
    }

    # Print stats
    print("\nTarget rates:")
    for tname, tval in targets.items():
        print(f"  {tname:<25}: positive rate = {tval.mean():.3f}")

    # Univariate AUROC analysis
    print("\n" + "=" * 60)
    print("Univariate AUROC Evaluation")
    print("=" * 60)

    results = {}
    for tname, y in targets.items():
        if y.std() == 0:
            print(f"\nTarget {tname} has no variance. Skipping AUROC.")
            continue
        print(f"\n--- Target: {tname} ---")
        results[tname] = {}
        for fname, x_i in features.items():
            a = roc_auc_score(y, x_i)
            a_best = max(a, 1 - a)
            direction = "+" if a >= 0.5 else "-"
            print(f"  {fname:<25} AUROC = {a_best:.4f} (direction: {direction})")
            results[tname][fname] = a_best

    # Compare confusion_ratio vs margin
    print("\n" + "=" * 60)
    print("Do Softmax-Ratio Features Beat Absolute Margin?")
    print("=" * 60)
    for tname in results:
        m_auc = results[tname]["margin"]
        cr_auc = results[tname]["confusion_ratio"]
        diff = cr_auc - m_auc
        win_str = "YES" if diff > 0 else "NO"
        print(f"  Target {tname:<25}: Confusion Ratio AUROC={cr_auc:.4f} vs Margin AUROC={m_auc:.4f} (Beat Margin? {win_str}, delta={diff:+.4f})")


if __name__ == "__main__":
    main()
