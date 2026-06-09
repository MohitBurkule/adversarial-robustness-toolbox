"""
H194 - Spectral frequency distribution of adversarial perturbations.

Hypothesis: adversarial perturbations from a naturally trained model are
concentrated in high-frequency components (>70% energy in top quartile of DCT
frequencies), while perturbations from an adversarially trained model spread
more uniformly across frequencies.

Methodology:
  - Train two models on Fashion-MNIST (n_train=6000, n_eval=500):
      Model A: standard cross-entropy training
      Model B: adversarial training (PGD-7, eps=0.3)
  - Generate PGD-10 perturbations (delta = x_adv - x_clean) for eval samples
  - Apply 2D DCT to each 28x28 perturbation
  - Divide DCT coefficients into 4 frequency quartiles (by L1 distance from DC)
  - Compute fraction of total energy in each quartile
  - Key test: is Q4 > 0.70 for standard model?

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from scipy.fft import dctn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH = 128
EPS = 0.1
PGD_STEPS = 10
ADV_EPS = 0.3
ADV_STEPS = 7
N_TRAIN = 6000
N_EVAL = 500
N_CLASSES = 10
SEED = 42


class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, n)

    def forward(self, x):
        x = F.relu(self.bn1(self.c1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.c2(x)))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_fashion_mnist():
    tf = transforms.ToTensor()
    tr = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
    te = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))])
    Ytr = torch.tensor([tr[i][1] for i in range(len(tr))])
    Xte = torch.stack([te[i][0] for i in range(len(te))])
    Yte = torch.tensor([te[i][1] for i in range(len(te))])
    return Xtr, Ytr, Xte, Yte


def subsample(Xtr, Ytr, Xte, Yte, n_train, n_eval, seed):
    g = torch.Generator().manual_seed(seed)
    idx_tr = torch.randperm(Xtr.size(0), generator=g)[:n_train]
    idx_te = torch.randperm(Xte.size(0), generator=g)[:n_eval]
    return Xtr[idx_tr], Ytr[idx_tr], Xte[idx_te], Yte[idx_te]


def pgd_attack(model, x, y, eps=EPS, steps=PGD_STEPS):
    alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def train_standard(model, Xtr, Ytr):
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model


def train_adversarial(model, Xtr, Ytr):
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    adv_alpha = 2.5 * ADV_EPS / ADV_STEPS
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            # generate adversarial training examples
            xa = pgd_attack(model, xb, yb, eps=ADV_EPS, steps=ADV_STEPS)
            loss = F.cross_entropy(model(xa), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model


def generate_perturbations(model, X, Y, batch=256):
    """Return perturbation array delta = x_adv - x_clean, shape (N, 28, 28)."""
    deltas = []
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i+batch], Y[i:i+batch]
        xa = pgd_attack(model, xb, yb)
        delta = (xa - xb).squeeze(1).cpu().numpy()  # (B, 28, 28)
        deltas.append(delta)
    return np.concatenate(deltas, axis=0)


def build_frequency_map(sz=28):
    """Assign each (i,j) DCT coefficient a frequency index = i + j (L1 distance from DC).
    Return quartile assignment: 0-3 for each coefficient."""
    freq = np.zeros((sz, sz), dtype=int)
    for i in range(sz):
        for j in range(sz):
            freq[i, j] = i + j
    max_freq = freq.max()
    # quartile boundaries: split unique freq values into 4 equal groups
    thresholds = np.percentile(freq.ravel(), [25, 50, 75])
    quartile = np.zeros((sz, sz), dtype=int)
    quartile[freq > thresholds[0]] = 1
    quartile[freq > thresholds[1]] = 2
    quartile[freq > thresholds[2]] = 3
    return quartile


def spectral_energy_fractions(deltas, quartile_map):
    """Compute mean energy fraction per quartile across all perturbations."""
    n = deltas.shape[0]
    q_energy = np.zeros((n, 4))
    for i in range(n):
        dct_coeff = dctn(deltas[i], norm='ortho')
        energy = dct_coeff ** 2
        total = energy.sum()
        if total < 1e-12:
            continue
        for q in range(4):
            q_energy[i, q] = energy[quartile_map == q].sum() / total
    return q_energy.mean(axis=0), q_energy


def l2_per_quartile(deltas, quartile_map):
    """Compute mean L2 norm of perturbation restricted to each frequency quartile."""
    n = deltas.shape[0]
    q_l2 = np.zeros((n, 4))
    for i in range(n):
        dct_coeff = dctn(deltas[i], norm='ortho')
        for q in range(4):
            mask = quartile_map == q
            q_l2[i, q] = np.sqrt((dct_coeff[mask] ** 2).sum())
    return q_l2.mean(axis=0)


def main():
    t0 = time.time()
    print("=" * 74)
    print("H194 - Spectral frequency distribution of adversarial perturbations")
    print("=" * 74)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Xtr_full, Ytr_full, Xte_full, Yte_full = load_fashion_mnist()
    Xtr, Ytr, Xte, Yte = subsample(Xtr_full, Ytr_full, Xte_full, Yte_full, N_TRAIN, N_EVAL, SEED)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)

    # Train standard model
    print("\n--- Training standard model (Model A) ---")
    torch.manual_seed(SEED)
    model_a = CNN().to(DEVICE)
    train_standard(model_a, Xtr, Ytr)
    with torch.no_grad():
        acc_a = float((model_a(Xte).argmax(1) == Yte).float().mean())
    print(f"  Clean accuracy: {acc_a:.4f}")

    # Train adversarially-trained model
    print("\n--- Training adversarial model (Model B, PGD-{}, eps={}) ---".format(ADV_STEPS, ADV_EPS))
    torch.manual_seed(SEED)
    model_b = CNN().to(DEVICE)
    train_adversarial(model_b, Xtr, Ytr)
    with torch.no_grad():
        acc_b = float((model_b(Xte).argmax(1) == Yte).float().mean())
    print(f"  Clean accuracy: {acc_b:.4f}")

    # Generate perturbations
    print("\n--- Generating PGD-10 perturbations ---")
    deltas_a = generate_perturbations(model_a, Xte, Yte)
    deltas_b = generate_perturbations(model_b, Xte, Yte)
    print(f"  Model A mean L2 perturbation: {np.sqrt((deltas_a**2).sum(axis=(1,2))).mean():.4f}")
    print(f"  Model B mean L2 perturbation: {np.sqrt((deltas_b**2).sum(axis=(1,2))).mean():.4f}")

    # Spectral analysis
    quartile_map = build_frequency_map(28)
    print("\n--- Spectral energy fractions (mean across samples) ---")
    fracs_a, _ = spectral_energy_fractions(deltas_a, quartile_map)
    fracs_b, _ = spectral_energy_fractions(deltas_b, quartile_map)

    print(f"\n  {'Quartile':>10}  {'Model A (std)':>14}  {'Model B (adv)':>14}")
    labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    for q in range(4):
        print(f"  {labels[q]:>10}  {fracs_a[q]:>14.4f}  {fracs_b[q]:>14.4f}")

    print(f"\n  Model A high-freq concentration (Q4): {fracs_a[3]:.4f}")
    print(f"  Model B high-freq concentration (Q4): {fracs_b[3]:.4f}")
    print(f"  Hypothesis (Q4 > 0.70 for Model A): {'SUPPORTED' if fracs_a[3] > 0.70 else 'NOT SUPPORTED'}")

    # L2 norms per quartile
    l2_a = l2_per_quartile(deltas_a, quartile_map)
    l2_b = l2_per_quartile(deltas_b, quartile_map)
    print("\n--- L2 norm of perturbation per frequency quartile ---")
    print(f"  {'Quartile':>10}  {'Model A (std)':>14}  {'Model B (adv)':>14}")
    for q in range(4):
        print(f"  {labels[q]:>10}  {l2_a[q]:>14.4f}  {l2_b[q]:>14.4f}")

    # Uniformity comparison
    from scipy.stats import entropy
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    kl_a = entropy(fracs_a, uniform)
    kl_b = entropy(fracs_b, uniform)
    print(f"\n  KL divergence from uniform: Model A = {kl_a:.4f}, Model B = {kl_b:.4f}")
    print(f"  Model B more uniform? {'YES' if kl_b < kl_a else 'NO'}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
