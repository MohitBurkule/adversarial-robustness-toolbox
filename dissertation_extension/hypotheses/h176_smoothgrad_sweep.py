"""
H176 - SmoothGrad sigma/K sensitivity sweep + direct gradient-norm-in-ball measurement.

Motivation (advisor critique, Paper 3):
  The headline "SmoothGrad collapses under adversarial training" (H169) was reported at a
  single hand-picked (K=50, sigma=0.1) and the proposed mechanism -- PGD-AT flattens the loss
  landscape *inside* the SmoothGrad noise neighbourhood -- was never directly measured. The two
  blocking experiments are:
    (1) sigma sweep   : AUROC vs sigma in {0.025, 0.05, 0.1, 0.2}   (K fixed = 50)
    (2) K sweep       : AUROC vs K in {5, 10, 25, 50, 100}          (sigma fixed = 0.1)
    (3) mechanism     : directly measure the distribution of ||grad_x CE|| at K random points
                        inside the sigma-ball, for vanilla vs PGD-AT. If AT collapses SmoothGrad,
                        the in-ball gradient norm should drop (flatter landscape) for AT.

This script trains a vanilla CNN and a PGD-AT CNN on Fashion-MNIST (patchable to other
datasets via run_with_patch.py), then runs all three analyses on EVAL_N correctly-classified
test samples. SmoothGrad is used as a per-sample vulnerability predictor: the predictor value
is the L2 norm of the noise-averaged input gradient; vulnerability label is PGD-10 flip.

A plain input-gradient-norm predictor (K=1, sigma=0, 1x compute) is reported as the baseline
that SmoothGrad must beat.
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
EVAL_N = 500

SIGMAS = [0.025, 0.05, 0.1, 0.2]
KS = [5, 10, 25, 50, 100]
SIGMA_FIXED = 0.1
K_FIXED = 50


class CNN(nn.Module):
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


def pgd_attack(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=10):
    model.eval()
    adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    adv = adv.clamp(0, 1)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        with torch.no_grad():
            adv = (adv + alpha * adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return adv.detach()


def pgd_flip(model, x, y, steps=10):
    x_adv = pgd_attack(model, x, y, steps=steps)
    with torch.no_grad():
        return (model(x_adv).argmax(1) != y)


def train_vanilla(train_set, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return model


def train_pgd_at(train_set, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = pgd_attack(model, x, y, alpha=2.0 / 255.0, steps=7)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(x_adv), y).backward()
            opt.step()
    return model


def input_grad_norm(model, x, y):
    """L2 norm of the plain input gradient of CE loss (K=1, sigma=0 baseline)."""
    xr = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xr), y)
    g, = torch.autograd.grad(loss, xr)
    return g.flatten(1).norm(dim=1)


def smoothgrad_norm(model, x, y, sigma, K):
    """L2 norm of the noise-averaged input gradient (SmoothGrad).

    For each of K noise draws, compute grad_x CE(model(x+noise), y); average the K
    gradients, then take per-sample L2 norm.
    """
    N = x.size(0)
    acc = torch.zeros_like(x)
    for _ in range(K):
        noise = torch.randn_like(x) * sigma
        xr = (x + noise).clamp(0, 1).detach().requires_grad_(True)
        loss = F.cross_entropy(model(xr), y)
        g, = torch.autograd.grad(loss, xr)
        acc += g.detach()
    acc /= K
    return acc.flatten(1).norm(dim=1)


def grad_norms_in_ball(model, x, y, sigma, K):
    """Return mean per-sample ||grad_x CE|| measured at K random points in the sigma-ball.

    This is the direct mechanism measurement: it does NOT average gradients (that is
    SmoothGrad); it averages the *norms*. A flatter landscape inside the ball (the claimed
    effect of AT) yields a smaller mean in-ball gradient norm.
    """
    N = x.size(0)
    norm_acc = torch.zeros(N, device=x.device)
    for _ in range(K):
        noise = torch.randn_like(x) * sigma
        xr = (x + noise).clamp(0, 1).detach().requires_grad_(True)
        loss = F.cross_entropy(model(xr), y)
        g, = torch.autograd.grad(loss, xr)
        norm_acc += g.detach().flatten(1).norm(dim=1)
    return norm_acc / K


def safe_auroc(label, score):
    label = np.asarray(label).astype(int)
    if label.std() == 0:
        return float("nan")
    a = roc_auc_score(label, np.asarray(score))
    return max(a, 1 - a)


def get_eval_set(model, test_x, test_y, n=EVAL_N):
    model.eval()
    with torch.no_grad():
        correct = (model(test_x).argmax(1) == test_y)
    idx = correct.nonzero(as_tuple=True)[0][:n]
    return test_x[idx], test_y[idx]


def compute_predictor(fn, model, x, y, batch=128):
    out = []
    for i in range(0, x.size(0), batch):
        out.append(fn(model, x[i:i + batch], y[i:i + batch]).detach().cpu())
    return torch.cat(out).numpy()


def main():
    print("=" * 74)
    print("H176 - SmoothGrad sigma/K sweep + gradient-norm-in-ball mechanism test")
    print("=" * 74)
    print(f"Device={DEVICE}  EVAL_N={EVAL_N}  EPS={EPS:.4f}")

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    models = {}
    print("\n--- Training Vanilla ---"); t0 = time.time()
    models["Vanilla"] = train_vanilla(train_set); print(f"  {time.time()-t0:.1f}s")
    print("--- Training PGD-AT ---"); t0 = time.time()
    models["PGD-AT"] = train_pgd_at(train_set); print(f"  {time.time()-t0:.1f}s")

    for mname, model in models.items():
        print("\n" + "=" * 74)
        print(f"MODEL: {mname}")
        print("=" * 74)
        ex, ey = get_eval_set(model, test_x, test_y)
        # vulnerability label
        flips = []
        for i in range(0, ex.size(0), 256):
            flips.append(pgd_flip(model, ex[i:i+256], ey[i:i+256]).cpu())
        flip = torch.cat(flips).numpy().astype(int)
        print(f"  eval samples={ex.size(0)}  PGD ASR={flip.mean():.3f}")

        # baseline: plain input grad norm
        base = compute_predictor(input_grad_norm, model, ex, ey)
        base_auroc = safe_auroc(flip, base)
        print(f"  [baseline] plain input-grad-norm (K=1, sigma=0) AUROC = {base_auroc:.4f}")

        # sigma sweep (K fixed)
        print(f"\n  sigma sweep (K={K_FIXED}):")
        print(f"    {'sigma':>7} {'SG-AUROC':>10} {'meanInBallGradNorm':>20}")
        for s in SIGMAS:
            sg = compute_predictor(lambda m, x, y: smoothgrad_norm(m, x, y, s, K_FIXED), model, ex, ey)
            ball = compute_predictor(lambda m, x, y: grad_norms_in_ball(m, x, y, s, K_FIXED), model, ex, ey)
            print(f"    {s:>7.3f} {safe_auroc(flip, sg):>10.4f} {ball.mean():>20.4f}")

        # K sweep (sigma fixed)
        print(f"\n  K sweep (sigma={SIGMA_FIXED}):")
        print(f"    {'K':>7} {'SG-AUROC':>10}")
        for k in KS:
            sg = compute_predictor(lambda m, x, y: smoothgrad_norm(m, x, y, SIGMA_FIXED, k), model, ex, ey)
            print(f"    {k:>7d} {safe_auroc(flip, sg):>10.4f}")

    print("\n" + "=" * 74)
    print("Interpretation: if SmoothGrad collapses under AT, the PGD-AT SG-AUROC should fall")
    print("toward (or below) its plain-grad baseline, AND its mean in-ball gradient norm")
    print("should be markedly lower than vanilla's at matched sigma (flatter loss landscape).")
    print("=" * 74)


if __name__ == "__main__":
    main()
