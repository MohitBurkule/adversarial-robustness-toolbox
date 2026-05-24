"""
H53: Adversarial robustness is class-asymmetric.

Tests whether some Fashion-MNIST classes are systematically more attackable than
others, whether within-class predictability of vulnerability (univariate AUROC
of victim margin / image stats) varies across classes, and which classes
adversarials transition to under FGSM (cross-class confusion).

Pipeline:
  1. Train a small CNN (matching diagnostic_test.py) on Fashion-MNIST, 10 epochs.
  2. Per-class:
        - clean accuracy
        - FGSM accuracy / flipped fraction (eps = 15/255)
        - PGD accuracy / flipped fraction (eps = 15/255, 10 steps, alpha=2/255)
        - mean victim margin (top1 - top2 logit, on correctly-classified samples)
        - mean image stats (mean intensity, std, edge_energy, foreground_pixels)
  3. Print a 10-class table.
  4. Within-class: univariate AUROC of margin / each image stat predicting
     binary FGSM-flip on correctly-classified samples of that class.
  5. Cross-class FGSM confusion matrix (true class x adversarial predicted class).
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = 2.0 / 255.0
SEED = 0

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


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


def train(model, train_set):
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        print(f" epoch {ep+1:02d}/{EPOCHS}  loss={loss_sum/total:.4f}  "
              f"train_acc={correct/total:.4f}  ({time.time()-t0:.1f}s)")
    model.eval()


def fgsm(model, x, y, eps=EPS):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps=EPS, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    # random start within eps-ball
    adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
    return adv.detach()


@torch.no_grad()
def batched_logits(model, x, bs=512):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(model(x[i:i+bs]))
    return torch.cat(out, 0)


def batched_attack(model, x, y, attack_fn, bs=256):
    out_pred = []
    for i in range(0, x.size(0), bs):
        adv = attack_fn(model, x[i:i+bs], y[i:i+bs])
        with torch.no_grad():
            out_pred.append(model(adv).argmax(1))
    return torch.cat(out_pred, 0)


def image_stats(x):
    """Per-sample image statistics.
    Returns dict of (N,) tensors: mean, std, edge_energy, foreground.
    """
    n = x.size(0)
    flat = x.view(n, -1)
    mean = flat.mean(1)
    std = flat.std(1)
    # edge energy: sum of squared finite differences
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    edge = (dx.pow(2).flatten(1).sum(1) + dy.pow(2).flatten(1).sum(1)).sqrt() / (28 * 28)
    foreground = (x > 0.1).float().flatten(1).mean(1)
    return {
        "mean_intensity": mean,
        "std_intensity": std,
        "edge_energy": edge,
        "foreground_frac": foreground,
    }


def safe_auroc(y, score):
    y = np.asarray(y).astype(int)
    if y.std() == 0 or len(np.unique(y)) < 2:
        return float("nan")
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    model = CNN(10).to(DEVICE)
    print("Training CNN victim on Fashion-MNIST ...")
    train(model, train_set)

    # Materialise test set on device
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)
    print(f"Test set: {N} samples")

    # Clean predictions, margins
    logits = batched_logits(model, test_x)
    clean_pred = logits.argmax(1)
    sorted_logits, _ = logits.sort(1, descending=True)
    margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).detach()

    # Attacks (full test set, untargeted)
    print("Running FGSM ...")
    fgsm_pred = batched_attack(model, test_x, test_y, fgsm, bs=256)
    print("Running PGD ...")
    pgd_pred = batched_attack(model, test_x, test_y, pgd, bs=256)

    stats = image_stats(test_x)
    stat_names = list(stats.keys())

    # ---------- Per-class table ----------
    print("\n===== Per-class robustness table =====")
    header = (f"{'cls':>3}  {'name':<13} {'n':>5} "
              f"{'clean_acc':>9} {'fgsm_acc':>9} {'fgsm_flip':>10} "
              f"{'pgd_acc':>8} {'pgd_flip':>9} {'margin':>8} "
              + " ".join(f"{s:>14}" for s in stat_names))
    print(header)
    per_class_rows = []
    for c in range(10):
        mask = (test_y == c)
        n_c = int(mask.sum().item())
        if n_c == 0:
            continue
        clean_correct = (clean_pred[mask] == c)
        clean_acc = clean_correct.float().mean().item()
        # fgsm/pgd accuracy = fraction still correct; flipped = of those originally correct, how many flipped
        fgsm_acc = (fgsm_pred[mask] == c).float().mean().item()
        pgd_acc = (pgd_pred[mask] == c).float().mean().item()
        if clean_correct.any():
            fgsm_flip = (fgsm_pred[mask][clean_correct] != c).float().mean().item()
            pgd_flip = (pgd_pred[mask][clean_correct] != c).float().mean().item()
            mean_margin = margin[mask][clean_correct].mean().item()
        else:
            fgsm_flip = pgd_flip = mean_margin = float("nan")
        stat_means = [stats[s][mask].mean().item() for s in stat_names]
        per_class_rows.append((c, n_c, clean_acc, fgsm_acc, fgsm_flip,
                               pgd_acc, pgd_flip, mean_margin, stat_means))
        print(f"{c:>3}  {FASHION_CLASSES[c]:<13} {n_c:>5} "
              f"{clean_acc:>9.4f} {fgsm_acc:>9.4f} {fgsm_flip:>10.4f} "
              f"{pgd_acc:>8.4f} {pgd_flip:>9.4f} {mean_margin:>8.3f} "
              + " ".join(f"{v:>14.4f}" for v in stat_means))

    # Summary ranking
    print("\n--- Classes ranked by FGSM flip rate (most attackable first) ---")
    ranked = sorted(per_class_rows, key=lambda r: -r[4])
    for r in ranked:
        print(f"  {FASHION_CLASSES[r[0]]:<13} fgsm_flip={r[4]:.4f}  "
              f"pgd_flip={r[6]:.4f}  mean_margin={r[7]:.3f}")

    # ---------- Within-class univariate AUROC ----------
    print("\n===== Within-class univariate AUROC predicting FGSM flip =====")
    print("(restricted to samples that were cleanly classified correctly)")
    feat_names = ["victim_margin"] + stat_names
    header = (f"{'cls':>3}  {'name':<13} {'n_eligible':>10} {'pos_rate':>9} "
              + " ".join(f"{f:>16}" for f in feat_names))
    print(header)
    within_rows = []
    for c in range(10):
        mask = (test_y == c) & (clean_pred == c)
        n_eligible = int(mask.sum().item())
        if n_eligible < 10:
            print(f"{c:>3}  {FASHION_CLASSES[c]:<13} {n_eligible:>10} (skipped, too few)")
            continue
        flipped = (fgsm_pred[mask] != c).cpu().numpy().astype(int)
        pos_rate = flipped.mean()
        feats = [margin[mask].cpu().numpy()] + [stats[s][mask].cpu().numpy()
                                                for s in stat_names]
        aurocs = [safe_auroc(flipped, f) for f in feats]
        within_rows.append((c, n_eligible, pos_rate, aurocs))
        print(f"{c:>3}  {FASHION_CLASSES[c]:<13} {n_eligible:>10} {pos_rate:>9.4f} "
              + " ".join(f"{a:>16.4f}" if not np.isnan(a) else f"{'nan':>16}"
                        for a in aurocs))

    # Variability across classes
    print("\n--- Spread of within-class univariate AUROC across classes ---")
    for i, fname in enumerate(feat_names):
        vals = np.array([r[3][i] for r in within_rows if not np.isnan(r[3][i])])
        if len(vals) == 0:
            continue
        print(f"  {fname:<18} min={vals.min():.4f} max={vals.max():.4f} "
              f"mean={vals.mean():.4f} std={vals.std():.4f}")

    # ---------- Cross-class FGSM confusion ----------
    print("\n===== FGSM cross-class confusion (rows = true class, cols = adv pred) =====")
    print("(restricted to samples originally classified correctly that flipped)")
    conf = np.zeros((10, 10), dtype=np.int64)
    flipped_global = (clean_pred == test_y) & (fgsm_pred != test_y)
    ty = test_y[flipped_global].cpu().numpy()
    fp = fgsm_pred[flipped_global].cpu().numpy()
    for t, p in zip(ty, fp):
        conf[t, p] += 1

    header = "true \\ adv  " + " ".join(f"{FASHION_CLASSES[c][:6]:>7}" for c in range(10)) + "   total"
    print(header)
    for r in range(10):
        total = conf[r].sum()
        print(f"{FASHION_CLASSES[r]:<11} " +
              " ".join(f"{conf[r, c]:>7d}" for c in range(10)) +
              f"   {total:>5d}")

    print("\n--- Top adversarial target per true class ---")
    for r in range(10):
        row = conf[r].copy()
        row[r] = 0  # ignore self (won't happen anyway since we filter flips)
        if row.sum() == 0:
            print(f"  {FASHION_CLASSES[r]:<13} (no flips)")
            continue
        top = int(np.argmax(row))
        share = row[top] / row.sum()
        print(f"  {FASHION_CLASSES[r]:<13} -> {FASHION_CLASSES[top]:<13} "
              f"({row[top]} of {row.sum()}, {share:.2%})")

    # ---------- Cross-class symmetry of attack flows ----------
    print("\n--- Asymmetry of flow A->B vs B->A (only top off-diagonal pairs) ---")
    pairs = []
    for a in range(10):
        for b in range(a + 1, 10):
            pairs.append((a, b, conf[a, b], conf[b, a]))
    pairs.sort(key=lambda t: -(t[2] + t[3]))
    for a, b, ab, ba in pairs[:10]:
        tot = ab + ba
        asym = abs(ab - ba) / tot if tot > 0 else 0.0
        print(f"  {FASHION_CLASSES[a]:<13} <-> {FASHION_CLASSES[b]:<13} "
              f"a->b={ab:>4d}  b->a={ba:>4d}  asym={asym:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
