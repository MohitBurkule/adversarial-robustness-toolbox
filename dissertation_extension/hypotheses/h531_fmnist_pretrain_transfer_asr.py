"""
H531 – Fashion-MNIST Pretrain Transfer: Does Pretraining History Shape ASR?
=============================================================================

Motivation (from h530):
  Boundary inertia is real (rho(similarity, displacement) = -1.0).
  Its ASR consequence is non-monotonic: identical-task pretrain can protect
  (inherited well-trained boundary) but partial-alignment pretrain can harm
  (compromise boundary = exploitable structure).  The 2D toy was too clean.
  This experiment tests the effect on REAL Fashion-MNIST.

Pretraining scenarios (7 conditions):
  0. rand-init      – no pretrain, baseline
  1. MNIST-STD      – pretrain on MNIST digit recognition (different domain,
                      similar low-level features: 28x28 grayscale)
  2. MNIST-AT       – MNIST pretrain + PGD adversarial training (inherit
                      robust features from a different domain)
  3. FMNIST-FULL    – pretrain on full FMNIST 10-class (identical task,
                      maximum inertia — boundary barely moves)
  4. FMNIST-AT      – pretrain on FMNIST + PGD-AT (inherit robust features
                      from the same domain)
  5. FMNIST-DARK%   – auxiliary pretrain: predict dark pixel percentage
                      (5-class: 0-20%, 20-40%, ..., 80-100%)
                      Same visual features, completely different label space
  6. FMNIST-QUADRANT– auxiliary pretrain: predict which quadrant is brightest
                      (4-class spatial task, partial feature overlap)

All conditions fine-tuned on 2-class FMNIST subset (T-shirt=0 vs Trouser=1)
with N=100 training examples (50/class) — small enough for pretrain to matter.

Evaluation:
  • Clean accuracy on held-out 2-class test set (1000/class)
  • FGSM ASR (eps=0.1, 0.2, 0.3)
  • PGD-20 ASR (eps=0.1)

PASS: At least one pretrained condition achieves FGSM ASR (eps=0.2) that
      differs from rand-init by >5pp (either direction — finding IS the result).

Figure (2 rows):
  Row 0: ASR bar chart per condition (FGSM eps=0.2 + PGD) + clean acc
  Row 1: FGSM eps sweep for all 7 conditions
"""

import os, sys, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
OUT_DIR     = os.path.join(PROJECT_DIR, "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)

FIG_PATH = os.path.join(OUT_DIR, "h531_fmnist_pretrain_transfer_asr.png")
TXT_PATH = os.path.join(OUT_DIR, "h531_fmnist_pretrain_transfer_asr_output.txt")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42
BATCH       = 256
PRETRAIN_E  = 20          # pretrain epochs (full dataset)
FINETUNE_E  = 40          # fine-tune epochs (small subset)
LR          = 1e-3
PGD_STEPS   = 20
EPS_LIST    = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
EVAL_EPS    = 0.20
PGD_EPS     = 0.10
FINETUNE_N  = 50          # per class for fine-tune (100 total)
FINETUNE_CLASSES = [0, 1] # T-shirt vs Trouser


# ──────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path):
        self._f = open(path, "w"); self._s = sys.stdout
    def write(self, d): self._s.write(d); self._f.write(d)
    def flush(self): self._s.flush(); self._f.flush()
    def close(self): self._f.close()
    def __getattr__(self, n): return getattr(self._s, n)


# ──────────────────────────────────────────────────────────────────────────────
# Model: small CNN (same for all conditions)
# ──────────────────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                   # 14×14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                   # 7×7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x): return self.head(self.features(x))

    def replace_head(self, n_classes):
        """Swap classifier head for fine-tuning."""
        in_feats = self.head[-1].in_features
        self.head[-1] = nn.Linear(in_feats, n_classes)
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Attacks
# ──────────────────────────────────────────────────────────────────────────────
def fgsm(model, x, y, eps):
    xv = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xv), y).backward()
    return (x + eps * xv.grad.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps, step, steps):
    xa = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa = xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g = torch.autograd.grad(loss, xa)[0]
        xa = (xa.detach() + step * g.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return xa.detach()


def eval_asr_loader(model, loader, eps, attack="fgsm"):
    model.eval(); correct = flipped = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            ok = model(x).argmax(1) == y
        if ok.sum() == 0: continue
        xc, yc = x[ok], y[ok]
        xa = fgsm(model, xc, yc, eps) if attack == "fgsm" else \
             pgd(model, xc, yc, eps, eps/4, PGD_STEPS)
        with torch.no_grad():
            flipped += (model(xa).argmax(1) != yc).sum().item()
        correct  += ok.sum().item()
    return flipped / max(correct, 1)


def eval_acc_loader(model, loader):
    model.eval(); c = t = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            c += (model(x).argmax(1) == y).sum().item()
        t += len(y)
    return c / t


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────
def train_standard(model, loader, epochs, lr=LR):
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"      epoch {ep+1}/{epochs}")
    return model


def train_pgdat(model, loader, epochs, eps=0.1, lr=LR):
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            xa = pgd(model, x, y, eps=eps, step=eps/4, steps=7)
            opt.zero_grad()
            F.cross_entropy(model(xa), y).backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"      epoch {ep+1}/{epochs}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_fmnist_full():
    tf = transforms.ToTensor()
    tr = torchvision.datasets.FashionMNIST(DATA_DIR, train=True,  download=True, transform=tf)
    te = torchvision.datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)
    return (DataLoader(tr, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True),
            DataLoader(te, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True))


def load_mnist_full():
    tf = transforms.ToTensor()
    tr = torchvision.datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    te = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return (DataLoader(tr, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True),
            DataLoader(te, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True))


def make_finetune_loaders(fmnist_train, fmnist_test, n_per_class=FINETUNE_N, seed=SEED):
    """Extract balanced 2-class subset (T-shirt=0, Trouser=1)."""
    rng = np.random.default_rng(seed)

    def subset_2class(ds, n_per_class, remap):
        indices = {c: [] for c in FINETUNE_CLASSES}
        for i, (_, lbl) in enumerate(ds):
            if lbl in FINETUNE_CLASSES:
                indices[lbl].append(i)
        sel = []
        for c in FINETUNE_CLASSES:
            chosen = rng.choice(indices[c], n_per_class, replace=False)
            sel.extend(chosen.tolist())
        rng.shuffle(sel)

        xs, ys = [], []
        for idx in sel:
            x, y = ds[idx]
            xs.append(x)
            ys.append(torch.tensor(remap[y]))
        return TensorDataset(torch.stack(xs), torch.stack(ys))

    remap = {0: 0, 1: 1}   # T-shirt→0, Trouser→1
    ft_ds   = subset_2class(fmnist_train, n_per_class, remap)
    # Test: all T-shirt/Trouser from test set (no limit)
    test_ds = subset_2class(fmnist_test, 900, remap)

    ft_loader   = DataLoader(ft_ds,   batch_size=min(BATCH, len(ft_ds)),   shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
    return ft_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary task: dark pixel percentage (5-class)
# ──────────────────────────────────────────────────────────────────────────────
def make_dark_pct_loader(fmnist_ds):
    """Label each image by its dark-pixel percentage bucket (0-4)."""
    xs, ys = [], []
    for x, _ in DataLoader(fmnist_ds, batch_size=512, num_workers=2):
        dark_frac = (x < 0.3).float().mean(dim=[1, 2, 3])   # fraction below 0.3
        buckets   = (dark_frac * 5).long().clamp(0, 4)
        xs.append(x); ys.append(buckets)
    ds = TensorDataset(torch.cat(xs), torch.cat(ys))
    return DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)


# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary task: brightest quadrant (4-class)
# ──────────────────────────────────────────────────────────────────────────────
def make_quadrant_loader(fmnist_ds):
    """Label each image by which of the 4 quadrants (14×14) has highest mean brightness."""
    xs, ys = [], []
    for x, _ in DataLoader(fmnist_ds, batch_size=512, num_workers=2):
        # x: (B, 1, 28, 28)
        q = torch.stack([
            x[:, :, :14, :14].mean(dim=[1,2,3]),   # top-left
            x[:, :, :14, 14:].mean(dim=[1,2,3]),   # top-right
            x[:, :, 14:, :14].mean(dim=[1,2,3]),   # bottom-left
            x[:, :, 14:, 14:].mean(dim=[1,2,3]),   # bottom-right
        ], dim=1)                                    # (B, 4)
        labels = q.argmax(dim=1)
        xs.append(x); ys.append(labels)
    ds = TensorDataset(torch.cat(xs), torch.cat(ys))
    return DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    tee = Tee(TXT_PATH)
    sys.stdout = tee
    try:
        _run()
    finally:
        sys.stdout = tee._s
        tee.close()


def _run():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"Device: {DEVICE}")
    print("=" * 70)
    print("H531 – FMNIST Pretrain Transfer: Pretraining History Shapes ASR")
    print("=" * 70)
    print(f"Fine-tune: 2-class FMNIST (T-shirt vs Trouser), N={FINETUNE_N}/class")
    print(f"Pretrain epochs: {PRETRAIN_E}  |  Finetune epochs: {FINETUNE_E}")

    # ── Load datasets ─────────────────────────────────────────────────────────
    fmnist_train_loader, fmnist_test_loader = load_fmnist_full()
    mnist_train_loader,  _                  = load_mnist_full()
    ft_loader, test_loader = make_finetune_loaders(
        torchvision.datasets.FashionMNIST(DATA_DIR, train=True,
            download=True, transform=transforms.ToTensor()),
        torchvision.datasets.FashionMNIST(DATA_DIR, train=False,
            download=True, transform=transforms.ToTensor()),
    )

    fmnist_train_raw = torchvision.datasets.FashionMNIST(
        DATA_DIR, train=True, download=True, transform=transforms.ToTensor())

    print(f"\nFine-tune set: {len(ft_loader.dataset)} samples")
    print(f"Test set:      {len(test_loader.dataset)} samples")

    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # 0. Rand-init baseline
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("[0] Rand-init (no pretrain)")
    torch.manual_seed(SEED)
    m = CNN(n_classes=2).to(DEVICE)
    train_standard(m, ft_loader, FINETUNE_E)
    results["rand-init"] = _evaluate(m, test_loader)
    _print_result("rand-init", results["rand-init"])

    # ─────────────────────────────────────────────────────────────────────────
    # 1. MNIST-STD pretrain → fine-tune on FMNIST 2-class
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("[1] MNIST-STD pretrain → FMNIST fine-tune")
    torch.manual_seed(SEED)
    m_mnist_std = CNN(n_classes=10).to(DEVICE)
    train_standard(m_mnist_std, mnist_train_loader, PRETRAIN_E)
    m = copy.deepcopy(m_mnist_std).replace_head(2)
    train_standard(m, ft_loader, FINETUNE_E)
    results["MNIST-STD"] = _evaluate(m, test_loader)
    _print_result("MNIST-STD", results["MNIST-STD"])

    # ─────────────────────────────────────────────────────────────────────────
    # 2. MNIST-AT pretrain → fine-tune
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("[2] MNIST-AT pretrain → FMNIST fine-tune")
    torch.manual_seed(SEED)
    m_mnist_at = CNN(n_classes=10).to(DEVICE)
    train_pgdat(m_mnist_at, mnist_train_loader, PRETRAIN_E, eps=0.1)
    m = copy.deepcopy(m_mnist_at).replace_head(2)
    train_standard(m, ft_loader, FINETUNE_E)
    results["MNIST-AT"] = _evaluate(m, test_loader)
    _print_result("MNIST-AT", results["MNIST-AT"])

    # ─────────────────────────────────────────────────────────────────────────
    # 3. FMNIST-FULL pretrain (10-class, same task) → fine-tune
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("[3] FMNIST-FULL pretrain (10-class) → 2-class fine-tune")
    torch.manual_seed(SEED)
    m_fmnist_std = CNN(n_classes=10).to(DEVICE)
    train_standard(m_fmnist_std, fmnist_train_loader, PRETRAIN_E)
    m = copy.deepcopy(m_fmnist_std).replace_head(2)
    train_standard(m, ft_loader, FINETUNE_E)
    results["FMNIST-STD"] = _evaluate(m, test_loader)
    _print_result("FMNIST-STD", results["FMNIST-STD"])

    # ─────────────────────────────────────────────────────────────────────────
    # 4. FMNIST-AT pretrain → fine-tune
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("[4] FMNIST-AT pretrain → 2-class fine-tune")
    torch.manual_seed(SEED)
    m_fmnist_at = CNN(n_classes=10).to(DEVICE)
    train_pgdat(m_fmnist_at, fmnist_train_loader, PRETRAIN_E, eps=0.1)
    m = copy.deepcopy(m_fmnist_at).replace_head(2)
    train_standard(m, ft_loader, FINETUNE_E)
    results["FMNIST-AT"] = _evaluate(m, test_loader)
    _print_result("FMNIST-AT", results["FMNIST-AT"])

    # ─────────────────────────────────────────────────────────────────────────
    # 5. FMNIST-DARK% auxiliary pretrain → fine-tune
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("[5] FMNIST-DARK% auxiliary pretrain (predict dark pixel %) → fine-tune")
    dark_loader = make_dark_pct_loader(fmnist_train_raw)
    torch.manual_seed(SEED)
    m_dark = CNN(n_classes=5).to(DEVICE)
    train_standard(m_dark, dark_loader, PRETRAIN_E)
    m = copy.deepcopy(m_dark).replace_head(2)
    train_standard(m, ft_loader, FINETUNE_E)
    results["FMNIST-DARK%"] = _evaluate(m, test_loader)
    _print_result("FMNIST-DARK%", results["FMNIST-DARK%"])

    # ─────────────────────────────────────────────────────────────────────────
    # 6. FMNIST-QUADRANT auxiliary pretrain → fine-tune
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("[6] FMNIST-QUADRANT auxiliary pretrain (brightest quadrant) → fine-tune")
    quad_loader = make_quadrant_loader(fmnist_train_raw)
    torch.manual_seed(SEED)
    m_quad = CNN(n_classes=4).to(DEVICE)
    train_standard(m_quad, quad_loader, PRETRAIN_E)
    m = copy.deepcopy(m_quad).replace_head(2)
    train_standard(m, ft_loader, FINETUNE_E)
    results["FMNIST-QUAD"] = _evaluate(m, test_loader)
    _print_result("FMNIST-QUAD", results["FMNIST-QUAD"])

    # ─────────────────────────────────────────────────────────────────────────
    # Summary table
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Condition':<18} {'CleanAcc':>9} {'FGSM@0.1':>9} {'FGSM@0.2':>9} "
          f"{'FGSM@0.3':>9} {'PGD@0.1':>8}")
    print("-" * 72)
    for name, r in results.items():
        idx1 = EPS_LIST.index(0.10)
        idx2 = EPS_LIST.index(0.20)
        idx3 = EPS_LIST.index(0.30)
        print(f"{name:<18} {r['acc']:>9.4f} {r['fgsm'][idx1]:>9.4f} "
              f"{r['fgsm'][idx2]:>9.4f} {r['fgsm'][idx3]:>9.4f} {r['pgd']:>8.4f}")
    print("=" * 72)

    rand_asr = results["rand-init"]["fgsm"][EPS_LIST.index(EVAL_EPS)]
    max_diff = max(abs(r["fgsm"][EPS_LIST.index(EVAL_EPS)] - rand_asr)
                   for r in results.values())
    passed = max_diff > 0.05
    print(f"\nMax |ASR - rand-init| at eps={EVAL_EPS}: {max_diff:.4f}")
    print(f"=> {'PASS' if passed else 'FAIL'} (threshold > 0.05)")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure
    # ─────────────────────────────────────────────────────────────────────────
    print("\nGenerating figure ...")
    cond_names = list(results.keys())
    colors = ["#888888", "#3498db", "#1a5276", "#e74c3c", "#922b21",
              "#27ae60", "#1e8449"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "H531 – FMNIST Pretrain Transfer: How Pretraining History Shapes ASR\n"
        f"Fine-tune: T-shirt vs Trouser, N={FINETUNE_N}/class",
        fontsize=11, fontweight="bold"
    )

    # (0) FGSM eps sweep
    ax = axes[0]
    for name, col in zip(cond_names, colors):
        ax.plot(EPS_LIST, results[name]["fgsm"], "o-",
                color=col, label=name, lw=1.8, ms=6)
    ax.set_xlabel("FGSM ε", fontsize=10)
    ax.set_ylabel("ASR", fontsize=10)
    ax.set_title("FGSM ASR vs ε — all conditions", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=8)
    ax.axvline(EVAL_EPS, color="gray", ls=":", lw=1)

    # (1) Bar chart at eval_eps: FGSM + PGD side by side
    ax = axes[1]
    idx_eval = EPS_LIST.index(EVAL_EPS)
    fgsm_vals = [results[n]["fgsm"][idx_eval] for n in cond_names]
    pgd_vals  = [results[n]["pgd"]            for n in cond_names]
    acc_vals  = [results[n]["acc"]            for n in cond_names]
    x = np.arange(len(cond_names))
    w = 0.35
    b1 = ax.bar(x - w/2, fgsm_vals, w, label=f"FGSM ε={EVAL_EPS}",
                color=colors, alpha=0.85, edgecolor="k", lw=0.7)
    b2 = ax.bar(x + w/2, pgd_vals,  w, label=f"PGD-20 ε={PGD_EPS}",
                color=colors, alpha=0.45, edgecolor="k", lw=0.7, hatch="//")
    for bar, v in list(zip(b1, fgsm_vals)) + list(zip(b2, pgd_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    # Annotate clean acc below x-axis
    for i, (name, av) in enumerate(zip(cond_names, acc_vals)):
        ax.text(i, -0.12, f"{av:.3f}", ha="center", fontsize=7, color="navy",
                transform=ax.get_xaxis_transform())
    ax.text(-0.5, -0.12, "clean acc:", ha="right", fontsize=7, color="navy",
            transform=ax.get_xaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("ASR", fontsize=10)
    ax.set_title(f"FGSM & PGD ASR at ε={EVAL_EPS} / {PGD_EPS}\n"
                 "Blue = clean acc below bars", fontsize=9)
    ax.legend(fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.axhline(rand_asr, color="gray", ls="--", lw=1.2,
               label=f"rand-init ({rand_asr:.2f})")
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {FIG_PATH}")
    print(f"Text   → {TXT_PATH}")
    print("\nDone.")


def _evaluate(model, test_loader):
    model.eval()
    a = eval_acc_loader(model, test_loader)
    fgsm_asrs = [eval_asr_loader(model, test_loader, eps, "fgsm") for eps in EPS_LIST]
    pgd_asr   = eval_asr_loader(model, test_loader, PGD_EPS, "pgd")
    return {"acc": a, "fgsm": fgsm_asrs, "pgd": pgd_asr}


def _print_result(name, r):
    idx = EPS_LIST.index(EVAL_EPS)
    print(f"  => acc={r['acc']:.4f}  FGSM(ε={EVAL_EPS})={r['fgsm'][idx]:.4f}  "
          f"PGD(ε={PGD_EPS})={r['pgd']:.4f}")


if __name__ == "__main__":
    main()
