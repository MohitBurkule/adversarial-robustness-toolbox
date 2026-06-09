"""
H201 - Per-class NC1 score predicts per-class adversarial vulnerability.

Hypothesis: per-class NC1 score (within-class feature variance / between-class
distance in penultimate layer) predicts per-class PGD-20 ASR with Pearson r > 0.6
across 10 Fashion-MNIST classes. Classes with high within-class spread (Shirt,
Coat) should have highest ASR; geometrically tight classes (Trouser, Bag) lowest.

Grounded in: arXiv:2311.07444 (neural collapse and robustness),
             arXiv:2501.19104 (NC1 varies by class separability).

Protocol:
  - Train a CNN on Fashion-MNIST (n_train=6000, 15 epochs).
  - Extract penultimate layer activations for all test samples (n_eval=1000).
  - For each class c compute NC1_c = S_W_c / D_B_c where:
      S_W_c = mean ||h - mu_c||^2 for samples in class c
      D_B_c = ||mu_c - mu_G||^2 (distance from class mean to global mean)
  - Generate PGD-20 adversarial examples (eps=0.3, step=0.03) for test set.
  - Compute per-class ASR (fraction of class c test samples fooled).
  - Report table, Pearson r, and scatter data.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import campaign.common as C

DEVICE = C.DEVICE
SEED = 42
N_TRAIN = 6000
N_EVAL = 1000
EPOCHS = 15
EPS = 0.3
PGD_STEPS = 20
PGD_ALPHA = 0.03

CLASS_NAMES = {
    0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot",
}


class CNNWithFeatures(nn.Module):
    """SmallCNN variant that exposes penultimate features."""
    def __init__(self, in_ch=1, size=28, n_classes=10, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.BatchNorm2d(width * 4),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        feat = size // 8
        self.flatten = nn.Flatten()
        self.penultimate = nn.Sequential(
            nn.Linear(width * 4 * feat * feat, 256), nn.ReLU()
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        h = self.flatten(self.features(x))
        h = self.penultimate(h)
        return self.head(h)

    def get_features(self, x):
        """Return penultimate layer activations (before final linear)."""
        h = self.flatten(self.features(x))
        return self.penultimate(h)


@torch.no_grad()
def extract_features(model, X, batch=256):
    model.eval()
    feats = []
    for i in range(0, X.size(0), batch):
        feats.append(model.get_features(X[i:i + batch]).cpu())
    return torch.cat(feats).numpy()


def compute_nc1_per_class(features, labels, n_classes=10):
    """Compute per-class NC1 = S_W_c / D_B_c."""
    mu_global = features.mean(axis=0)
    nc1 = np.zeros(n_classes)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() < 2:
            nc1[c] = float("nan")
            continue
        feats_c = features[mask]
        mu_c = feats_c.mean(axis=0)
        # within-class variance
        s_w = np.mean(np.sum((feats_c - mu_c) ** 2, axis=1))
        # between-class distance (class mean to global mean)
        d_b = np.sum((mu_c - mu_global) ** 2)
        nc1[c] = s_w / (d_b + 1e-10)
    return nc1


def compute_per_class_asr(model, X, Y, n_classes=10, eps=EPS,
                          steps=PGD_STEPS, alpha=PGD_ALPHA, batch=256):
    """Compute ASR per class (fraction of originally-correct samples fooled)."""
    model.eval()
    correct_all = []
    flipped_all = []
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        with torch.no_grad():
            correct = model(xb).argmax(1) == yb
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps, alpha=alpha)
        with torch.no_grad():
            flipped = model(xa).argmax(1) != yb
        correct_all.append(correct.cpu())
        flipped_all.append(flipped.cpu())
    correct_all = torch.cat(correct_all).numpy().astype(bool)
    flipped_all = torch.cat(flipped_all).numpy().astype(bool)
    Y_np = Y.cpu().numpy()

    asr = np.zeros(n_classes)
    clean_acc = np.zeros(n_classes)
    for c in range(n_classes):
        mask = Y_np == c
        clean_acc[c] = correct_all[mask].mean() if mask.sum() > 0 else float("nan")
        corr_c = correct_all[mask]
        flip_c = flipped_all[mask]
        if corr_c.sum() > 0:
            asr[c] = flip_c[corr_c].mean()
        else:
            asr[c] = float("nan")
    return asr, clean_acc


def main():
    print("=" * 74)
    print("H201 - Per-class NC1 score predicts per-class adversarial vulnerability")
    print("=" * 74)
    print(f"Device={DEVICE}  EPOCHS={EPOCHS}  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}")
    print(f"PGD: eps={EPS}  steps={PGD_STEPS}  alpha={PGD_ALPHA}")

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN,
                                          n_eval=N_EVAL, seed=SEED)

    # ---- Train CNN ----
    print("\n--- Training CNN ---")
    t0 = time.time()
    model = CNNWithFeatures(
        in_ch=meta["channels"], size=meta["size"],
        n_classes=meta["n_classes"]
    ).to(DEVICE)
    opt = C.make_optimizer(model, "adam", lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    print(f"  trained in {time.time() - t0:.1f}s")

    # ---- Clean accuracy ----
    with torch.no_grad():
        logits = []
        for i in range(0, Xte.size(0), 256):
            logits.append(model(Xte[i:i + 256]).cpu())
        logits = torch.cat(logits)
    acc = (logits.argmax(1) == Yte.cpu()).float().mean().item()
    print(f"  clean test accuracy: {acc:.4f}")

    # ---- Extract penultimate features ----
    print("\n--- Extracting penultimate features ---")
    features = extract_features(model, Xte)
    labels = Yte.cpu().numpy()
    print(f"  feature shape: {features.shape}")

    # ---- Compute per-class NC1 ----
    print("\n--- Computing per-class NC1 ---")
    nc1 = compute_nc1_per_class(features, labels, n_classes=meta["n_classes"])

    # ---- Compute per-class ASR ----
    print("\n--- Computing per-class PGD-20 ASR ---")
    t0 = time.time()
    asr, clean_acc = compute_per_class_asr(model, Xte, Yte,
                                            n_classes=meta["n_classes"])
    print(f"  PGD attack completed in {time.time() - t0:.1f}s")

    # ---- Results table ----
    print("\n--- Per-class results ---")
    print(f"  {'class':<12} {'NC1':>8} {'ASR':>8} {'clean_acc':>10}")
    print(f"  {'-' * 40}")
    for c in range(meta["n_classes"]):
        name = CLASS_NAMES.get(c, str(c))
        print(f"  {name:<12} {nc1[c]:>8.4f} {asr[c]:>8.4f} {clean_acc[c]:>10.4f}")

    # ---- Correlation ----
    valid = ~(np.isnan(nc1) | np.isnan(asr))
    if valid.sum() >= 3:
        r_p, p_p = pearsonr(nc1[valid], asr[valid])
        r_s, p_s = spearmanr(nc1[valid], asr[valid])
        print(f"\n--- Correlation (NC1 vs ASR) ---")
        print(f"  Pearson  r = {r_p:.4f}  (p = {p_p:.4f})")
        print(f"  Spearman r = {r_s:.4f}  (p = {p_s:.4f})")
        print(f"\n  Hypothesis (r > 0.6): {'SUPPORTED' if r_p > 0.6 else 'NOT SUPPORTED'}")
    else:
        print("\n  Not enough valid classes for correlation.")

    # ---- Scatter data ----
    print("\n--- Scatter data (NC1, ASR) ---")
    for c in range(meta["n_classes"]):
        name = CLASS_NAMES.get(c, str(c))
        print(f"  {name}: ({nc1[c]:.4f}, {asr[c]:.4f})")

    # ---- Rank analysis ----
    nc1_rank = np.argsort(np.argsort(-nc1))  # 0 = highest NC1
    asr_rank = np.argsort(np.argsort(-asr))
    print("\n--- Rank comparison (0 = highest) ---")
    print(f"  {'class':<12} {'NC1_rank':>9} {'ASR_rank':>9}")
    for c in range(meta["n_classes"]):
        name = CLASS_NAMES.get(c, str(c))
        print(f"  {name:<12} {nc1_rank[c]:>9} {asr_rank[c]:>9}")

    # ---- Expectation check ----
    print("\n--- Expectation check ---")
    # Shirt (6), Coat (4) expected high NC1/ASR; Trouser (1), Bag (8) expected low
    high_nc1 = [6, 4]
    low_nc1 = [1, 8]
    mean_high = np.mean([nc1[c] for c in high_nc1])
    mean_low = np.mean([nc1[c] for c in low_nc1])
    print(f"  Mean NC1 (Shirt, Coat):     {mean_high:.4f}")
    print(f"  Mean NC1 (Trouser, Bag):    {mean_low:.4f}")
    print(f"  Separation (high/low):      {mean_high / (mean_low + 1e-10):.2f}x")

    mean_high_asr = np.mean([asr[c] for c in high_nc1])
    mean_low_asr = np.mean([asr[c] for c in low_nc1])
    print(f"  Mean ASR (Shirt, Coat):     {mean_high_asr:.4f}")
    print(f"  Mean ASR (Trouser, Bag):    {mean_low_asr:.4f}")

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
