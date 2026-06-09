"""
H193 - Label noise increases adversarial vulnerability monotonically.

Hypothesis: injecting symmetric label noise at rates {0%, 5%, 10%, 20%} causes
PGD-10 attack success rate to increase monotonically, with the relationship
being superlinear (doubling noise > doubles ASR increase). Grounded in
arXiv:2207.03933, which proves label noise creates an irreducible lower bound
on adversarial risk.

Methodology:
  - For each noise_rate in [0.0, 0.05, 0.10, 0.20]:
      - Load Fashion-MNIST (n_train=4000, n_eval=500)
      - Apply symmetric label noise: randomly flip each label to a uniform
        random other class with probability noise_rate
      - Train a small CNN for 15 epochs
      - Evaluate: clean accuracy, PGD-10 ASR (eps=0.1), FGSM ASR (eps=0.1)
  - Report table: noise_rate, clean_acc, pgd_asr, fgsm_asr
  - Test superlinearity: compare ASR increments across noise doublings
  - Fit linear regression (noise_rate vs pgd_asr) and report R², slope

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.linear_model import LinearRegression

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH = 128
EPS = 0.1
PGD_STEPS = 10
N_TRAIN = 4000
N_EVAL = 500
N_CLASSES = 10
SEED = 42
NOISE_RATES = [0.0, 0.05, 0.10, 0.20]


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


def apply_label_noise(Y, noise_rate, seed):
    """Symmetric label noise: flip each label with prob noise_rate to a uniform random other class."""
    rng = np.random.RandomState(seed)
    Y_noisy = Y.clone()
    n = Y.size(0)
    mask = rng.random(n) < noise_rate
    for i in np.where(mask)[0]:
        candidates = [c for c in range(N_CLASSES) if c != Y[i].item()]
        Y_noisy[i] = rng.choice(candidates)
    return Y_noisy


def train(model, Xtr, Ytr):
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            out = model(Xtr[idx])
            loss = F.cross_entropy(out, Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model


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


def fgsm_attack(model, x, y, eps=EPS):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    g, = torch.autograd.grad(loss, x)
    return (x + eps * g.sign()).clamp(0, 1).detach()


def eval_asr(model, X, Y, attack_fn, batch=256):
    """ASR over originally-correct samples."""
    flips, correct = [], []
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i+batch], Y[i:i+batch]
        with torch.no_grad():
            corr = model(xb).argmax(1) == yb
        xa = attack_fn(model, xb, yb)
        with torch.no_grad():
            flip = model(xa).argmax(1) != yb
        flips.append(flip.cpu()); correct.append(corr.cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(correct).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() > 0 else float("nan")


def main():
    t0 = time.time()
    print("=" * 74)
    print("H193 - Label noise increases adversarial vulnerability monotonically")
    print("=" * 74)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Xtr_full, Ytr_full, Xte_full, Yte_full = load_fashion_mnist()
    Xtr, Ytr, Xte, Yte = subsample(Xtr_full, Ytr_full, Xte_full, Yte_full, N_TRAIN, N_EVAL, SEED)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)

    results = []
    for nr in NOISE_RATES:
        print(f"\n--- noise_rate = {nr:.2f} ---")
        Ytr_noisy = apply_label_noise(Ytr.cpu(), nr, seed=SEED).to(DEVICE)
        n_flipped = int((Ytr_noisy != Ytr).sum().item())
        print(f"  labels flipped: {n_flipped}/{N_TRAIN} ({100*n_flipped/N_TRAIN:.1f}%)")

        torch.manual_seed(SEED)
        model = CNN().to(DEVICE)
        train(model, Xtr, Ytr_noisy)

        with torch.no_grad():
            clean_acc = float((model(Xte).argmax(1) == Yte).float().mean())
        pgd_asr = eval_asr(model, Xte, Yte, pgd_attack)
        fgsm_asr = eval_asr(model, Xte, Yte, fgsm_attack)

        print(f"  clean_acc={clean_acc:.4f}  pgd_asr={pgd_asr:.4f}  fgsm_asr={fgsm_asr:.4f}")
        results.append((nr, clean_acc, pgd_asr, fgsm_asr))

    print("\n" + "=" * 74)
    print("RESULTS TABLE")
    print("=" * 74)
    print(f"  {'noise_rate':>10}  {'clean_acc':>10}  {'pgd_asr':>10}  {'fgsm_asr':>10}")
    for nr, ca, pa, fa in results:
        print(f"  {nr:>10.2f}  {ca:>10.4f}  {pa:>10.4f}  {fa:>10.4f}")

    # Superlinearity test
    print("\n--- Superlinearity test ---")
    asrs = [r[2] for r in results]
    deltas = []
    for i in range(1, len(asrs)):
        d = asrs[i] - asrs[i-1]
        nr_delta = NOISE_RATES[i] - NOISE_RATES[i-1]
        slope_seg = d / nr_delta if nr_delta > 0 else float("nan")
        deltas.append((NOISE_RATES[i-1], NOISE_RATES[i], d, slope_seg))
        print(f"  {NOISE_RATES[i-1]:.2f} -> {NOISE_RATES[i]:.2f}: "
              f"delta_ASR={d:+.4f}, marginal_slope={slope_seg:.4f}")

    slopes = [d[3] for d in deltas]
    superlinear = all(slopes[i] <= slopes[i+1] for i in range(len(slopes)-1))
    print(f"  Superlinear (marginal slope increasing)? {superlinear}")

    # Linear regression
    X_reg = np.array(NOISE_RATES).reshape(-1, 1)
    y_reg = np.array(asrs)
    lr = LinearRegression().fit(X_reg, y_reg)
    r2 = lr.score(X_reg, y_reg)
    print(f"\n--- Linear fit: noise_rate -> pgd_asr ---")
    print(f"  slope={lr.coef_[0]:.4f}  intercept={lr.intercept_:.4f}  R²={r2:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
