"""
H103: Distance from a test sample to the nearest TRAINING sample of a *different*
class (in penultimate-feature space) is the cleanest "boundary distance" proxy
and predicts adversarial vulnerability.

Pipeline
--------
1. Train a small CNN (architecture matching diagnostic_test.py) on Fashion-MNIST
   for 10 epochs.
2. Subsample 5000 training points (stratified-ish via random sample).
3. Extract penultimate-layer features (output of fc1 after ReLU+dropout=identity
   at eval time) for the 5000 train sub-sample and for all 10k test points.
4. For each test sample, compute the minimum L2 distance in feature space to a
   training sample of a *different* class (using sklearn NearestNeighbors).
5. Per-sample features:
       - boundary_dist   (the H103 quantity)
       - final_margin    (logit-margin baseline)
       - mean_pix        (mean pixel intensity)
       - std_pix         (std of pixel intensities)
6. Vulnerability targets:
       - FGSM flip @ eps = 15/255
       - PGD  flip @ eps = 15/255 (10 steps, step = eps/4)
       - min_eps_to_flip (binary search on FGSM direction)
7. Univariate AUROC of each feature against each binary target, and Pearson
   correlation against min_eps. Restricted to samples the model classifies
   correctly.

This script is SELF-CONTAINED. It is intended to be saved and run later; this
file only WRITES code, does NOT execute it.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_STEP_SIZE = EPS_TEST / 4.0
N_TRAIN_SUB = 5000
SEED = 0


class CNN(nn.Module):
    """Matches diagnostic_test.py CNN. Penultimate = post-fc1 ReLU features (128-dim)."""
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def features(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return x  # 128-dim penultimate

    def forward(self, x):
        f = self.features(x)
        f = self.do2(f)
        return self.fc2(f)


def train_model(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    return model


@torch.no_grad()
def extract_features(model, x, batch=512):
    model.eval()
    out = []
    for i in range(0, x.size(0), batch):
        out.append(model.features(x[i:i+batch]).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def predict_logits(model, x, batch=512):
    model.eval()
    out = []
    for i in range(0, x.size(0), batch):
        out.append(model(x[i:i+batch]).cpu())
    return torch.cat(out, 0)


def fgsm_grad(model, x, y):
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS_TEST, steps=PGD_STEPS, step_size=PGD_STEP_SIZE):
    model.eval()
    x0 = x.clone().detach()
    # random start within eps ball
    adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        F.cross_entropy(model(adv), y).backward()
        with torch.no_grad():
            adv = adv + step_size * adv.grad.sign()
            adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
        adv = adv.detach()
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Binary search smallest eps along FGSM sign direction that flips prediction."""
    model.eval()
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


def batched(fn, x, y, batch=256):
    out = []
    for i in range(0, x.size(0), batch):
        out.append(fn(x[i:i+batch], y[i:i+batch]))
    return torch.cat(out, 0)


def boundary_distance(train_feats, train_labels, test_feats, test_labels):
    """
    For each test sample, the min L2 distance in feature space to any TRAINING
    sample whose class differs from the test sample's true class.

    Implementation: per-class NearestNeighbors index over training points; for
    each test sample, query each "other" class index for its nearest neighbour,
    take the min across other-class indices.
    """
    n_classes = int(max(train_labels.max(), test_labels.max())) + 1
    # build per-class indices
    indices = {}
    for c in range(n_classes):
        mask = train_labels == c
        if mask.sum() == 0:
            continue
        nn_idx = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        nn_idx.fit(train_feats[mask])
        indices[c] = nn_idx

    N = test_feats.shape[0]
    dists = np.full(N, np.inf, dtype=np.float64)
    for c, nn_idx in indices.items():
        same = test_labels == c
        other_mask = ~same  # test samples whose true class != c -> c is an "other" class for them
        if other_mask.sum() == 0:
            continue
        d, _ = nn_idx.kneighbors(test_feats[other_mask], n_neighbors=1, return_distance=True)
        d = d[:, 0]
        cur = dists[other_mask]
        new = np.minimum(cur, d)
        dists[other_mask] = new
    return dists


def univariate_auroc(score, y_binary):
    if y_binary.std() == 0:
        return float("nan")
    a = roc_auc_score(y_binary, score)
    return max(a, 1 - a)


def main():
    tf = transforms.ToTensor()
    print("loading Fashion-MNIST ...")
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(f"training small CNN for {EPOCHS} epochs on full train set ...")
    model = train_model(train_set)

    # subsample 5000 training points
    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(len(train_set), size=N_TRAIN_SUB, replace=False)
    sub_idx.sort()
    train_x_sub = torch.stack([train_set[i][0] for i in sub_idx]).to(DEVICE)
    train_y_sub = np.array([train_set[int(i)][1] for i in sub_idx], dtype=np.int64)

    # full test set in memory
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))],
                          dtype=torch.long, device=DEVICE)

    print("extracting penultimate features (train subset + test) ...")
    train_feats = extract_features(model, train_x_sub)
    test_feats = extract_features(model, test_x)

    print("computing boundary distance (min L2 to other-class train sample) ...")
    t0 = time.time()
    bdist = boundary_distance(train_feats, train_y_sub,
                              test_feats, test_y.cpu().numpy())
    print(f"  done ({time.time()-t0:.1f}s)  mean={bdist.mean():.4f}  median={np.median(bdist):.4f}")

    # pixel statistics
    test_x_np = test_x.detach().cpu().numpy().reshape(test_x.size(0), -1)
    mean_pix = test_x_np.mean(1)
    std_pix = test_x_np.std(1)

    # logit margin
    logits = predict_logits(model, test_x)
    sorted_logits, _ = logits.sort(1, descending=True)
    final_margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).numpy()
    final_pred = logits.argmax(1).to(DEVICE)

    # restrict to correctly-classified
    correct = (final_pred == test_y)
    correct_np = correct.cpu().numpy()
    print(f"correctly classified: {int(correct_np.sum())}/{correct_np.size}")

    x_c = test_x[correct]
    y_c = test_y[correct]
    bdist_c = bdist[correct_np]
    margin_c = final_margin[correct_np]
    mean_c = mean_pix[correct_np]
    std_c = std_pix[correct_np]

    # adversarial targets
    print("FGSM attack ...")
    fgsm_flip = batched(lambda a, b: fgsm_attack(model, a, b, EPS_TEST), x_c, y_c).cpu().numpy().astype(int)
    print(f"  FGSM flip rate = {fgsm_flip.mean():.3f}")

    print("PGD attack ...")
    pgd_flip = batched(lambda a, b: pgd_attack(model, a, b, EPS_TEST, PGD_STEPS, PGD_STEP_SIZE),
                       x_c, y_c).cpu().numpy().astype(int)
    print(f"  PGD flip rate = {pgd_flip.mean():.3f}")

    print("min_eps_to_flip ...")
    me_chunks = []
    for i in range(0, x_c.size(0), 256):
        me_chunks.append(min_eps_to_flip(model, x_c[i:i+256], y_c[i:i+256]))
    min_eps = torch.cat(me_chunks).cpu().numpy()
    print(f"  mean min_eps = {min_eps.mean():.4f}")

    # ---------- evaluation ----------
    feat_names = ["boundary_dist", "final_margin", "mean_pix", "std_pix"]
    feat_arrays = [bdist_c, margin_c, mean_c, std_c]

    print("\n========== Univariate AUROC (Fashion-MNIST) ==========")
    print(f"{'feature':<16} {'FGSM':>8} {'PGD':>8} {'min_eps_corr':>14}")
    for name, arr in zip(feat_names, feat_arrays):
        a_f = univariate_auroc(arr, fgsm_flip)
        a_p = univariate_auroc(arr, pgd_flip)
        cor = np.corrcoef(arr, min_eps)[0, 1]
        print(f"{name:<16} {a_f:>8.4f} {a_p:>8.4f} {cor:>+14.4f}")

    # also report positive rates for context
    print(f"\nFGSM positive rate: {fgsm_flip.mean():.3f}")
    print(f"PGD  positive rate: {pgd_flip.mean():.3f}")
    print(f"min_eps mean / median: {min_eps.mean():.4f} / {np.median(min_eps):.4f}")


if __name__ == "__main__":
    main()
