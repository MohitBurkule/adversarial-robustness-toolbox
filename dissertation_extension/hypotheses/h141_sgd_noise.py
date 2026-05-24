"""
Hypothesis H141: SGD noise sensitivity on Fashion-MNIST.

Train K=3 CNNs starting from the exact same weight initialization but with slightly
perturbed SGD trajectories (different data ordering and random noise added to gradients) for 10 epochs.

Per test sample:
  - vote_agreement: fraction of the K=3 models that agree on the correct class.
  - softmax_variance: variance of the true-class softmax probability across the K=3 models.

Features (victim = Model 0):
  - vote_agreement
  - softmax_variance
  - margin (from Model 0)
  - mean_pix
  - std_pix

Targets (on Model 0):
  - flipped_FGSM
  - flipped_PGD
  - min_eps (min epsilon to flip via binary search)

Analysis:
  - Univariate AUROC for all features against all targets.
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
K_MODELS = 3
INIT_SEED = 42


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


def train_perturbed_model(train_set, init_seed, shuffle_seed, grad_noise_scale=1e-4):
    """Train a CNN starting from identical init weights but with perturbed SGD."""
    # First, initialize the model with the shared init_seed
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    model = CNN(N_CLASSES).to(DEVICE)

    # Use a different seed for the DataLoader's shuffle to get different data ordering
    generator = torch.Generator()
    generator.manual_seed(shuffle_seed)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, generator=generator, num_workers=2)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()

            # Perturb gradients with small Gaussian noise
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * grad_noise_scale
                        p.grad.add_(noise)

            optimizer.step()

    return model


def fgsm_attack(model, x, y, eps=EPS):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    sign = x_adv.grad.sign().detach()
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS, alpha=2.0/255.0, steps=10):
    adv = x.clone().detach().requires_grad_(True)
    adv = adv + torch.FloatTensor(*adv.shape).uniform_(-eps, eps).to(DEVICE)
    adv = adv.clamp(0, 1)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        adv = (adv + alpha * adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    N = x.size(0)
    lo = torch.zeros(N, device=DEVICE)
    hi = torch.full((N,), eps_max, device=DEVICE)

    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    sign = x_adv.grad.sign().detach()

    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = (model(adv).argmax(1) != y)
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def main():
    print("=" * 60)
    print("Hypothesis H141: SGD Noise Sensitivity Analysis")
    print("=" * 60)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Train models
    models = []
    print(f"\nTraining K={K_MODELS} perturbed models from seed={INIT_SEED}...")
    for k in range(K_MODELS):
        t0 = time.time()
        model = train_perturbed_model(train_set, init_seed=INIT_SEED, shuffle_seed=2026+k)
        model.eval()
        models.append(model)
        print(f"  Model {k} trained in {time.time() - t0:.1f}s")

    # Stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    # Restrict to samples correctly classified by Model 0 (victim)
    with torch.no_grad():
        victim_pred = models[0](test_x).argmax(1)
        correct = (victim_pred == test_y)

    x_c = test_x[correct]
    y_c = test_y[correct]
    N_correct = correct.sum().item()
    print(f"\nUsing {N_correct}/{N} correctly classified samples from victim (Model 0)")

    # Compute disagreement metrics across the K=3 models
    print("\nComputing disagreement metrics...")
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for k in range(K_MODELS):
            logits_k = models[k](x_c)
            probs_k = F.softmax(logits_k, dim=1)
            all_preds.append(logits_k.argmax(dim=1))
            all_probs.append(probs_k[torch.arange(N_correct), y_c])

    # Convert to torch tensors
    all_preds = torch.stack(all_preds, dim=0)  # (K, N_correct)
    all_probs = torch.stack(all_probs, dim=0)  # (K, N_correct)

    # vote_agreement: fraction of the K=3 models predicting correctly
    vote_agreement = (all_preds == y_c).float().mean(dim=0).cpu()

    # softmax_variance: variance of the true-class softmax probability
    softmax_variance = all_probs.var(dim=0).cpu()

    # Margin on victim
    with torch.no_grad():
        logits_0 = models[0](x_c)
        sorted_logits, _ = logits_0.sort(1, descending=True)
        margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).cpu()

    # Basic image stats
    flat_x = x_c.view(N_correct, -1)
    mean_pix = flat_x.mean(1).cpu()
    std_pix = flat_x.std(1).cpu()

    # Stack features
    features = torch.stack([vote_agreement, softmax_variance, margin, mean_pix, std_pix], dim=1).numpy()

    # Run attacks
    print("\nRunning attacks on victim...")
    flipped_fgsm = []
    flipped_pgd = []
    min_eps = []

    for i in range(0, N_correct, 512):
        bx = x_c[i:i+512]
        by = y_c[i:i+512]
        flipped_fgsm.append(fgsm_attack(models[0], bx, by))
        flipped_pgd.append(pgd_attack(models[0], bx, by))
        min_eps.append(min_eps_to_flip(models[0], bx, by))

    flipped_fgsm = torch.cat(flipped_fgsm).cpu().numpy().astype(int)
    flipped_pgd = torch.cat(flipped_pgd).cpu().numpy().astype(int)
    min_eps = torch.cat(min_eps).cpu().numpy()

    print(f"  FGSM flip rate = {flipped_fgsm.mean():.4f}")
    print(f"  PGD-10 flip rate = {flipped_pgd.mean():.4f}")
    print(f"  Mean min_eps = {min_eps.mean():.4f}")

    # Analyze
    feature_names = [
        "vote_agreement",
        "softmax_variance",
        "margin",
        "mean_pix",
        "std_pix"
    ]

    for target_name, y in [("flipped_FGSM", flipped_fgsm), ("flipped_PGD", flipped_pgd)]:
        print("\n" + "=" * 60)
        print(f"TARGET: {target_name} | Univariate AUROC")
        print("=" * 60)
        if y.std() == 0:
            print("  Constant target, skipping.")
            continue
        for i, fname in enumerate(feature_names):
            x_i = features[:, i]
            a = roc_auc_score(y, x_i)
            a = max(a, 1 - a)
            print(f"  {fname:<20} AUROC = {a:.4f}")

    print("\n" + "=" * 60)
    print("TARGET: min_eps (continuous) | Univariate AUROC (above median)")
    print("=" * 60)
    y_bin = (min_eps > np.median(min_eps)).astype(int)
    if y_bin.std() == 0:
        print("  Constant target, skipping.")
    else:
        for i, fname in enumerate(feature_names):
            x_i = features[:, i]
            a = roc_auc_score(y_bin, x_i)
            a = max(a, 1 - a)
            print(f"  {fname:<20} AUROC = {a:.4f}")


if __name__ == "__main__":
    main()
