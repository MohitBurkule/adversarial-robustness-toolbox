"""
H533 – AT Reshapes Internal Representation Geometry
=====================================================

Motivation (from Ren et al. arXiv:2509.01235, NeurIPS MI Workshop 2025):
  Adversarial training does not just change the decision boundary — it
  restructures the internal feature space.  Specifically, AT is predicted to:
    1. INCREASE intra-class compactness (features cluster tighter within class)
    2. INCREASE inter-class separation (cluster centroids move further apart)
    3. DECREASE feature superposition / effective rank
       (fewer "dimensions" used → less interference between features)

  Feature superposition (Gorton & Lewis, NeurIPS MI Workshop 2025) is the
  mechanism: when a network encodes more features than it has dimensions,
  interference between features creates exploitable gradient directions.
  AT, by training on worst-case perturbations, is forced to find
  representations where these interference directions are smaller.

Experiment:
  Train 3 models on Fashion-MNIST (full 10-class):
    1. STD  — standard CE training
    2. AT   — PGD adversarial training (eps=0.1)
    3. HALF — train on half the epochs to observe geometry evolution

  Extract penultimate-layer features (128-dim) for all 10k test images.

  Metrics per model:
    a. Intra-class compactness: mean of per-class feature variance (trace of
       within-class covariance).  Lower = more compact.
    b. Inter-class separation: mean pairwise centroid distance.  Higher = better.
    c. Fisher criterion: inter / intra.  Higher = more linearly separable.
    d. Effective rank of feature matrix (ratio of sum of singular values to
       max singular value).  Lower = less superposition.
    e. Per-class ASR (FGSM, eps=0.1) vs per-class intra-class variance:
       does higher variance predict higher ASR?  (Spearman rho)

  Visualisation:
    Row 0: t-SNE of penultimate features (STD vs AT), coloured by class
    Row 1: per-class intra-class variance bar chart (STD vs AT)
    Row 2: singular value spectrum (STD vs AT) — superposition proxy
    Row 3: per-class ASR vs intra-class variance scatter

PASS:
  AT Fisher criterion > STD Fisher criterion (AT features more separable)
  AND effective rank(AT) < effective rank(STD)
  AND Spearman rho(intra_var, per_class_ASR) > 0.4 for at least one model.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
OUT_DIR     = os.path.join(PROJECT_DIR, "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_PATH = os.path.join(OUT_DIR, "h533_representation_geometry_at.png")
TXT_PATH = os.path.join(OUT_DIR, "h533_representation_geometry_at_output.txt")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42
EPOCHS     = 20
LR         = 1e-3
BATCH      = 256
FGSM_EPS   = 0.10
PGD_EPS    = 0.10
PGD_STEPS  = 7
TSNE_N     = 2000   # subsample for t-SNE speed

CLASS_NAMES = ["T-shirt","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Boot"]


class Tee:
    def __init__(self, path):
        self._f = open(path, "w", buffering=1); self._s = sys.stdout
    def write(self, d): self._s.write(d); self._f.write(d); self._f.flush()
    def flush(self): self._s.flush(); self._f.flush()
    def close(self): self._f.close()
    def __getattr__(self, n): return getattr(self._s, n)


# ──────────────────────────────────────────────────────────────────────────────
# Model — CNN with accessible penultimate layer
# ──────────────────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self, n_cls=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.penultimate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(128, n_cls)

    def forward(self, x):
        return self.classifier(self.penultimate(self.features(x)))

    def embed(self, x):
        """Return penultimate-layer features."""
        self.eval()
        with torch.no_grad():
            return self.penultimate(self.features(x))


# ──────────────────────────────────────────────────────────────────────────────
# Attacks
# ──────────────────────────────────────────────────────────────────────────────
def fgsm(model, x, y, eps):
    xv = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xv), y).backward()
    return (x + eps * xv.grad.sign()).clamp(0,1).detach()

def pgd(model, x, y, eps, steps=PGD_STEPS):
    xa = x.clone().detach() + torch.empty_like(x).uniform_(-eps,eps)
    xa = xa.clamp(0,1)
    step = eps / 4
    for _ in range(steps):
        xa = xa.requires_grad_(True)
        g = torch.autograd.grad(F.cross_entropy(model(xa),y), xa)[0]
        xa = (xa.detach() + step*g.sign()).clamp(x-eps, x+eps).clamp(0,1)
    return xa.detach()


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train(model, loader, epochs, at=False, eps=PGD_EPS):
    opt   = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if at:
                xa = pgd(model, x, y, eps, steps=7)
                model.train(); opt.zero_grad()
                F.cross_entropy(model(xa), y).backward()
            else:
                F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()
        if (ep+1) % 5 == 0:
            print(f"    ep {ep+1}/{epochs}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Geometry metrics
# ──────────────────────────────────────────────────────────────────────────────
def extract_features(model, loader):
    """Returns (feats: N×128, labels: N) numpy arrays."""
    feats, labs = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        f = model.embed(x).cpu().numpy()
        feats.append(f); labs.append(y.numpy())
    return np.vstack(feats), np.concatenate(labs)


def intra_class_variance(feats, labels, n_cls=10):
    """Per-class mean variance (trace of within-class covariance / dim)."""
    variances = []
    for c in range(n_cls):
        fc = feats[labels == c]
        if len(fc) < 2: variances.append(0.0); continue
        variances.append(np.var(fc, axis=0).mean())
    return np.array(variances)


def inter_class_separation(feats, labels, n_cls=10):
    """Mean pairwise distance between class centroids."""
    centroids = np.stack([feats[labels==c].mean(0) for c in range(n_cls)])
    dists = []
    for i in range(n_cls):
        for j in range(i+1, n_cls):
            dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    return float(np.mean(dists))


def fisher_criterion(feats, labels, n_cls=10):
    """Simplified Fisher ratio: inter_separation / mean(intra_var)."""
    inter = inter_class_separation(feats, labels, n_cls)
    intra = intra_class_variance(feats, labels, n_cls).mean()
    return inter / max(intra, 1e-8)


def effective_rank(feats):
    """
    Effective rank = exp(entropy of normalised singular values).
    Low effective rank → few dominant dimensions → less superposition.
    """
    _, sv, _ = np.linalg.svd(feats - feats.mean(0), full_matrices=False)
    sv = sv / sv.sum()
    sv = sv[sv > 1e-10]
    entropy = -np.sum(sv * np.log(sv))
    return float(np.exp(entropy))


def per_class_asr(model, loader, eps, n_cls=10):
    """FGSM ASR broken down per class."""
    correct  = np.zeros(n_cls)
    flipped  = np.zeros(n_cls)
    model.eval()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            ok = model(x).argmax(1) == y
        if ok.sum() == 0: continue
        xc, yc = x[ok], y[ok]
        xa = fgsm(model, xc, yc, eps)
        with torch.no_grad():
            preds_adv = model(xa).argmax(1)
        for c in range(n_cls):
            mask = yc == c
            if mask.sum() == 0: continue
            correct[c] += mask.sum().item()
            flipped[c] += (preds_adv[mask] != yc[mask]).sum().item()
    asr = np.where(correct > 0, flipped / correct, 0.0)
    return asr


# ──────────────────────────────────────────────────────────────────────────────
def main():
    tee = Tee(TXT_PATH)
    sys.stdout = tee
    try: _run()
    finally: sys.stdout = tee._s; tee.close()


def _run():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"Device: {DEVICE}")
    print("="*70)
    print("H533 – AT Reshapes Internal Representation Geometry (FMNIST)")
    print("="*70)

    tf = transforms.ToTensor()
    tr_ds = torchvision.datasets.FashionMNIST(DATA_DIR, train=True,  download=True, transform=tf)
    te_ds = torchvision.datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    # ── Train models ──────────────────────────────────────────────────────────
    print(f"\n[1/3] Training STD ({EPOCHS} epochs) ...")
    torch.manual_seed(SEED)
    std_model = CNN().to(DEVICE)
    train(std_model, tr_loader, EPOCHS, at=False)

    print(f"\n[2/3] Training AT-PGD (eps={PGD_EPS}, {EPOCHS} epochs) ...")
    torch.manual_seed(SEED)
    at_model = CNN().to(DEVICE)
    train(at_model, tr_loader, EPOCHS, at=True, eps=PGD_EPS)

    print(f"\n[3/3] Training HALF-STD ({EPOCHS//2} epochs) ...")
    torch.manual_seed(SEED)
    half_model = CNN().to(DEVICE)
    train(half_model, tr_loader, EPOCHS//2, at=False)

    # ── Extract features ──────────────────────────────────────────────────────
    print("\nExtracting penultimate features ...")
    feats_std,  labs = extract_features(std_model,  te_loader)
    feats_at,   _    = extract_features(at_model,   te_loader)
    feats_half, _    = extract_features(half_model, te_loader)

    # ── Geometry metrics ──────────────────────────────────────────────────────
    models_info = [
        ("STD",      std_model,  feats_std),
        ("AT",       at_model,   feats_at),
        ("HALF-STD", half_model, feats_half),
    ]

    print("\n" + "="*70)
    print(f"{'Model':<12} {'IntraVar':>10} {'InterSep':>10} {'Fisher':>10} {'EffRank':>10}")
    print("-"*70)
    geo = {}
    for name, m, feats in models_info:
        iv   = intra_class_variance(feats, labs).mean()
        sep  = inter_class_separation(feats, labs)
        fish = fisher_criterion(feats, labs)
        er   = effective_rank(feats)
        geo[name] = dict(intra=iv, inter=sep, fisher=fish, eff_rank=er,
                         intra_per_class=intra_class_variance(feats, labs))
        print(f"{name:<12} {iv:>10.4f} {sep:>10.4f} {fish:>10.4f} {er:>10.2f}")
    print("="*70)

    # ── Per-class ASR ─────────────────────────────────────────────────────────
    print("\nPer-class FGSM ASR ...")
    asr_std  = per_class_asr(std_model,  te_loader, FGSM_EPS)
    asr_at   = per_class_asr(at_model,   te_loader, FGSM_EPS)

    print(f"\n{'Class':<12} {'STD_intraVar':>13} {'AT_intraVar':>12} {'STD_ASR':>9} {'AT_ASR':>8}")
    print("-"*60)
    for c in range(10):
        print(f"{CLASS_NAMES[c]:<12} {geo['STD']['intra_per_class'][c]:>13.4f} "
              f"{geo['AT']['intra_per_class'][c]:>12.4f} "
              f"{asr_std[c]:>9.3f} {asr_at[c]:>8.3f}")

    # Spearman correlation: intra-class variance vs per-class ASR
    rho_std, p_std = spearmanr(geo['STD']['intra_per_class'], asr_std)
    rho_at,  p_at  = spearmanr(geo['AT']['intra_per_class'],  asr_at)
    print(f"\nSpearman rho(intra_var_STD, ASR_STD) = {rho_std:+.3f}  p={p_std:.4f}")
    print(f"Spearman rho(intra_var_AT,  ASR_AT)  = {rho_at:+.3f}  p={p_at:.4f}")

    # ── PASS ──────────────────────────────────────────────────────────────────
    crit1 = geo['AT']['fisher']   > geo['STD']['fisher']
    crit2 = geo['AT']['eff_rank'] < geo['STD']['eff_rank']
    crit3 = max(abs(rho_std), abs(rho_at)) > 0.4
    passed = crit1 and crit2 and crit3
    print("\n" + "="*70)
    print(f"PASS crit1 AT Fisher > STD Fisher:   {'OK' if crit1 else 'FAIL'}"
          f"  ({geo['AT']['fisher']:.3f} vs {geo['STD']['fisher']:.3f})")
    print(f"PASS crit2 AT EffRank < STD EffRank: {'OK' if crit2 else 'FAIL'}"
          f"  ({geo['AT']['eff_rank']:.2f} vs {geo['STD']['eff_rank']:.2f})")
    print(f"PASS crit3 |rho(intra,ASR)| > 0.4:  {'OK' if crit3 else 'FAIL'}"
          f"  (max={max(abs(rho_std),abs(rho_at)):.3f})")
    print(f"=> OVERALL: {'PASS' if passed else 'FAIL'}")

    # ── Figure ────────────────────────────────────────────────────────────────
    print("\nGenerating figure (t-SNE may take ~30s) ...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("H533 – AT Reshapes Internal Representation Geometry (FMNIST)\n"
                 "Penultimate-layer (128-dim) features extracted from STD vs AT models",
                 fontsize=11, fontweight="bold")

    COLORS10 = plt.cm.tab10(np.linspace(0, 1, 10))

    # t-SNE: STD and AT
    for col, (name, feats) in enumerate([("STD", feats_std), ("AT", feats_at)]):
        idx = np.random.default_rng(SEED).choice(len(feats), TSNE_N, replace=False)
        emb = TSNE(n_components=2, perplexity=40, random_state=SEED,
                   n_iter=500).fit_transform(feats[idx])
        ax = axes[0, col]
        for c in range(10):
            mask = labs[idx] == c
            ax.scatter(emb[mask, 0], emb[mask, 1], s=4, alpha=0.5,
                       color=COLORS10[c], label=CLASS_NAMES[c])
        ax.set_title(f"t-SNE: {name} features\n"
                     f"Fisher={geo[name]['fisher']:.2f}  EffRank={geo[name]['eff_rank']:.1f}",
                     fontsize=9)
        ax.tick_params(labelsize=7)
        if col == 1:
            ax.legend(fontsize=5.5, markerscale=2, ncol=2,
                      loc="upper right", framealpha=0.7)

    # Per-class intra-class variance
    ax = axes[0, 2]
    x = np.arange(10); w = 0.35
    ax.bar(x-w/2, geo['STD']['intra_per_class'], w,
           color=COLORS10, alpha=0.85, edgecolor="k", lw=0.6, label="STD")
    ax.bar(x+w/2, geo['AT']['intra_per_class'],  w,
           color=COLORS10, alpha=0.45, edgecolor="k", lw=0.6, hatch="//", label="AT")
    ax.set_xticks(x); ax.set_xticklabels([c[:5] for c in CLASS_NAMES],
                                          rotation=45, fontsize=7, ha="right")
    ax.set_ylabel("Mean intra-class variance", fontsize=8)
    ax.set_title("Per-class intra-class variance\n(lower = more compact)", fontsize=8.5)
    ax.legend(fontsize=8); ax.tick_params(labelsize=7)

    # Per-class ASR
    ax = axes[0, 3]
    ax.bar(x-w/2, asr_std, w, color=COLORS10, alpha=0.85, edgecolor="k", lw=0.6, label="STD")
    ax.bar(x+w/2, asr_at,  w, color=COLORS10, alpha=0.45, edgecolor="k", lw=0.6, hatch="//", label="AT")
    ax.set_xticks(x); ax.set_xticklabels([c[:5] for c in CLASS_NAMES],
                                          rotation=45, fontsize=7, ha="right")
    ax.set_ylabel(f"FGSM ASR (eps={FGSM_EPS})", fontsize=8)
    ax.set_title("Per-class FGSM ASR\n(some classes systematically harder to defend)", fontsize=8.5)
    ax.legend(fontsize=8); ax.tick_params(labelsize=7)

    # Singular value spectrum
    ax = axes[1, 0]
    for (name, _, feats), ls in zip(models_info, ["-","--",":"]):
        _, sv, _ = np.linalg.svd(feats[:2000] - feats[:2000].mean(0),
                                  full_matrices=False)
        sv_norm = sv / sv[0]
        ax.plot(sv_norm[:50], ls, lw=1.8,
                label=f"{name} (eff_rank={geo[name]['eff_rank']:.1f})")
    ax.set_xlabel("Singular value index", fontsize=8)
    ax.set_ylabel("Normalised singular value", fontsize=8)
    ax.set_title("Singular value spectrum\n(faster decay = less superposition)", fontsize=8.5)
    ax.legend(fontsize=8); ax.tick_params(labelsize=7)
    ax.set_yscale("log")

    # Scatter: intra-class var vs per-class ASR (STD)
    ax = axes[1, 1]
    for c in range(10):
        ax.scatter(geo['STD']['intra_per_class'][c], asr_std[c],
                   color=COLORS10[c], s=80, zorder=5, edgecolors="k", lw=0.5)
        ax.annotate(CLASS_NAMES[c][:5], (geo['STD']['intra_per_class'][c], asr_std[c]),
                    fontsize=6.5, xytext=(3,3), textcoords="offset points")
    ax.set_xlabel("Intra-class variance (STD)", fontsize=8)
    ax.set_ylabel("FGSM ASR (STD)", fontsize=8)
    ax.set_title(f"Intra-var vs ASR (STD)\nρ={rho_std:+.3f}  p={p_std:.3f}", fontsize=8.5)
    ax.tick_params(labelsize=7)

    # Scatter: intra-class var vs per-class ASR (AT)
    ax = axes[1, 2]
    for c in range(10):
        ax.scatter(geo['AT']['intra_per_class'][c], asr_at[c],
                   color=COLORS10[c], s=80, zorder=5, edgecolors="k", lw=0.5)
        ax.annotate(CLASS_NAMES[c][:5], (geo['AT']['intra_per_class'][c], asr_at[c]),
                    fontsize=6.5, xytext=(3,3), textcoords="offset points")
    ax.set_xlabel("Intra-class variance (AT)", fontsize=8)
    ax.set_ylabel("FGSM ASR (AT)", fontsize=8)
    ax.set_title(f"Intra-var vs ASR (AT)\nρ={rho_at:+.3f}  p={p_at:.3f}", fontsize=8.5)
    ax.tick_params(labelsize=7)

    # Summary bar: Fisher + EffRank for all 3 models
    ax = axes[1, 3]
    model_names = [n for n, _, _ in models_info]
    fisher_vals  = [geo[n]['fisher']   for n in model_names]
    erank_vals   = [geo[n]['eff_rank'] for n in model_names]
    xp = np.arange(3); ww = 0.35
    b1 = ax.bar(xp-ww/2, fisher_vals, ww, label="Fisher criterion",
                color=["steelblue","darkorange","gray"], alpha=0.85, edgecolor="k", lw=0.7)
    ax2 = ax.twinx()
    b2 = ax2.bar(xp+ww/2, erank_vals, ww, label="Effective rank",
                 color=["steelblue","darkorange","gray"], alpha=0.4, edgecolor="k", lw=0.7, hatch="//")
    for bar, v in zip(b1, fisher_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.1, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    for bar, v in zip(b2, erank_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.2, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xp); ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel("Fisher criterion (↑ better)", fontsize=8)
    ax2.set_ylabel("Effective rank (↓ less superposition)", fontsize=8)
    ax.set_title("Fisher criterion & Effective rank\nAT: higher Fisher, lower eff_rank?", fontsize=8.5)
    lines1, lbls1 = ax.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, lbls1+lbls2, fontsize=7.5)
    ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {FIG_PATH}")
    print(f"Text   → {TXT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
