"""
Hypothesis H09: Per-sample distance (L2 in raw pixel space) to nearest same-class
training image predicts adversarial vulnerability.

Rationale (Maini 2022 etc.): out-of-distribution / "rare" test samples — those
that live far from any training neighbour — tend to be more adversarially
vulnerable. A cheap, model-agnostic proxy: for each test image, compute the
mean L2 distance to its K=5 nearest *same-class* training neighbours, and to
its K=5 nearest training neighbours regardless of class, and the gap between
the two. Higher distance => more atypical => predicted to be more attackable.

Pipeline (self-contained):
  1. Train a small CNN victim on Fashion-MNIST (10 epochs, Adam) — same
     architecture as dissertation_extension/diagnostic_test.py.
  2. For each TEST sample compute three pixel-space distance features:
        d_same  : mean L2 to 5 nearest same-class training neighbours
        d_any   : mean L2 to 5 nearest training neighbours of any class
        d_gap   : d_same - d_any  (>0 means same-class neighbours are
                  farther than the nearest neighbours overall)
     Using sklearn brute-force NearestNeighbors on a 5000-image random
     subsample of the training set (stratified per class for d_same).
  3. Baseline scalars per sample:
        victim_margin  : top1 - top2 logit gap of the victim on x
        mean_pix       : mean pixel value
        std_pix        : pixel-value std
  4. Vulnerability targets:
        flipped_FGSM         : FGSM eps=15/255 flips victim's top-1
        flipped_PGD          : PGD 10 steps, eps=15/255, alpha=eps/4
        min_eps_FGSM         : binary-searched smallest eps to flip with FGSM
                               (continuous)
  5. Univariate AUROC of each feature against each binary target,
     Pearson correlation with min_eps, and multivariate logistic regression
     ablations vs the margin baseline.

Run:  python h09_knn_distance.py
Outputs: stdout. Uses CUDA. Downloads data into /tmp/data.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
K = 5
SUBSAMPLE = 5000           # train images sampled for NN search
PGD_STEPS = 10
SEED = 0


# ----------------------------------------------------------------------------
# Model (matches diagnostic_test.py)
# ----------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2); x = self.do1(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.do2(x); return self.fc2(x)


def train_victim(train_set):
    torch.manual_seed(SEED); np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train(); t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        print(f"  epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Attacks
# ----------------------------------------------------------------------------
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, steps=PGD_STEPS):
    alpha = eps / 4.0
    adv = x.clone().detach()
    # random start within eps-ball
    adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
    adv = adv.clamp(0, 1)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        F.cross_entropy(model(adv), y).backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            adv = torch.max(torch.min(adv, x + eps), x - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf FGSM eps that flips."""
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


# ----------------------------------------------------------------------------
# Distance features
# ----------------------------------------------------------------------------
def compute_knn_distances(train_x_flat, train_y, test_x_flat, test_y, k=K):
    """
    Returns (d_same, d_any) numpy arrays of shape (N_test,)
    d_same: mean L2 to k nearest same-class training neighbours
    d_any : mean L2 to k nearest training neighbours of any class
    """
    N = test_x_flat.shape[0]
    d_same = np.zeros(N, dtype=np.float32)
    d_any = np.zeros(N, dtype=np.float32)

    # any-class
    nn_any = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    nn_any.fit(train_x_flat)
    dists, _ = nn_any.kneighbors(test_x_flat)
    d_any = dists.mean(axis=1)

    # per-class same-class
    for c in range(10):
        cls_mask = train_y == c
        if cls_mask.sum() < k:
            continue
        nn_c = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
        nn_c.fit(train_x_flat[cls_mask])
        test_mask = test_y == c
        if test_mask.sum() == 0:
            continue
        dists_c, _ = nn_c.kneighbors(test_x_flat[test_mask])
        d_same[test_mask] = dists_c.mean(axis=1)
    return d_same, d_any


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(feat_mat, feat_names, binary_targets, binary_names,
             min_eps, baseline_idx):
    Xs = StandardScaler().fit_transform(feat_mat)

    print("\n--- Univariate AUROC (binary targets) ---")
    print(f"{'feature':<18}" + "".join(f"{tn:>22}" for tn in binary_names))
    for i, fn in enumerate(feat_names):
        row = [f"{fn:<18}"]
        for j, tn in enumerate(binary_names):
            y = binary_targets[j]
            if y.std() == 0:
                row.append(f"{'(degenerate)':>22}")
                continue
            a = roc_auc_score(y, feat_mat[:, i])
            a = max(a, 1 - a)
            row.append(f"{a:>22.4f}")
        print("".join(row))

    print("\n--- Pearson correlation with min_eps_FGSM (continuous) ---")
    for i, fn in enumerate(feat_names):
        r = np.corrcoef(feat_mat[:, i], min_eps)[0, 1]
        print(f"  corr(min_eps, {fn:<18}) = {r:+.4f}")

    print("\n--- Multivariate logistic regression (binary targets) ---")
    # We compare three nested models per target:
    #   M0: margin only
    #   M1: margin + d_same + d_any + d_gap   (knn-distance features)
    #   M2: margin + ALL features (incl. pixel stats)
    knn_idx = [feat_names.index("d_same"),
               feat_names.index("d_any"),
               feat_names.index("d_gap")]
    for j, tn in enumerate(binary_names):
        y = binary_targets[j]
        if y.std() == 0:
            print(f"  target {tn}: degenerate, skipping")
            continue
        print(f"\n  target = {tn}  (pos rate = {y.mean():.3f})")

        def fit_auc(cols):
            X = Xs[:, cols]
            lr = LogisticRegression(max_iter=2000).fit(X, y)
            return roc_auc_score(y, lr.predict_proba(X)[:, 1]), lr

        auc0, _ = fit_auc([baseline_idx])
        auc1, lr1 = fit_auc([baseline_idx] + knn_idx)
        auc2, lr2 = fit_auc(list(range(Xs.shape[1])))
        print(f"    AUROC  margin-only            : {auc0:.4f}")
        print(f"    AUROC  margin + knn distances : {auc1:.4f}  (delta {auc1-auc0:+.4f})")
        print(f"    AUROC  margin + all features  : {auc2:.4f}  (delta {auc2-auc0:+.4f})")
        print(f"    coefficients (margin + knn):")
        names_used = [feat_names[baseline_idx]] + [feat_names[i] for i in knn_idx]
        for n, c in zip(names_used, lr1.coef_.flatten()):
            print(f"      {n:<18} {c:+.4f}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print(f"device = {DEVICE}")
    print("training victim CNN on Fashion-MNIST...")
    t0 = time.time()
    model = train_victim(train_set)
    print(f"  done ({time.time()-t0:.1f}s)")

    # Materialise tensors
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))])
    train_x = torch.stack([train_set[i][0] for i in range(len(train_set))])
    train_y = torch.tensor([train_set[i][1] for i in range(len(train_set))])

    # Random stratified-ish subsample of training set for NN search (speed).
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train_set), size=SUBSAMPLE, replace=False)
    tr_x_sub = train_x[idx].numpy().reshape(SUBSAMPLE, -1).astype(np.float32)
    tr_y_sub = train_y[idx].numpy()
    te_x_flat = test_x.numpy().reshape(len(test_set), -1).astype(np.float32)
    te_y_np = test_y.numpy()

    print(f"computing k-NN pixel distances (K={K}, train_subsample={SUBSAMPLE})...")
    t0 = time.time()
    d_same, d_any = compute_knn_distances(tr_x_sub, tr_y_sub, te_x_flat, te_y_np, k=K)
    d_gap = d_same - d_any
    print(f"  done ({time.time()-t0:.1f}s)")
    print(f"  d_same: mean={d_same.mean():.3f} std={d_same.std():.3f}")
    print(f"  d_any : mean={d_any.mean():.3f} std={d_any.std():.3f}")
    print(f"  d_gap : mean={d_gap.mean():.3f} std={d_gap.std():.3f}")

    # Baseline features
    mean_pix = te_x_flat.mean(axis=1)
    std_pix = te_x_flat.std(axis=1)

    # Victim margin & predictions
    test_x_dev = test_x.to(DEVICE); test_y_dev = test_y.to(DEVICE)
    N = test_x_dev.size(0)
    with torch.no_grad():
        logits_all = []
        for i in range(0, N, 512):
            logits_all.append(model(test_x_dev[i:i+512]))
        logits_all = torch.cat(logits_all, 0)
    sorted_logits, _ = logits_all.sort(1, descending=True)
    victim_margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).cpu().numpy()
    pred = logits_all.argmax(1)
    correct = (pred == test_y_dev)
    print(f"  victim clean accuracy = {correct.float().mean().item():.4f}")

    # Restrict to correctly-classified test samples (otherwise "flipping" is ill-defined).
    keep = correct.cpu().numpy()
    x_c = test_x_dev[correct]
    y_c = test_y_dev[correct]
    print(f"  using {keep.sum()} correctly-classified test samples")

    # Attacks (binary)
    print("computing FGSM eps=15/255 flips ...")
    t0 = time.time()
    f_fgsm = []
    for i in range(0, x_c.size(0), 512):
        f_fgsm.append(fgsm_flip(model, x_c[i:i+512], y_c[i:i+512]))
    f_fgsm = torch.cat(f_fgsm).cpu().numpy().astype(int)
    print(f"  done ({time.time()-t0:.1f}s)  flip rate={f_fgsm.mean():.3f}")

    print("computing PGD eps=15/255 flips ...")
    t0 = time.time()
    f_pgd = []
    for i in range(0, x_c.size(0), 512):
        f_pgd.append(pgd_flip(model, x_c[i:i+512], y_c[i:i+512]))
    f_pgd = torch.cat(f_pgd).cpu().numpy().astype(int)
    print(f"  done ({time.time()-t0:.1f}s)  flip rate={f_pgd.mean():.3f}")

    print("computing min_eps_FGSM (binary search) ...")
    t0 = time.time()
    me = []
    for i in range(0, x_c.size(0), 512):
        me.append(min_eps_to_flip(model, x_c[i:i+512], y_c[i:i+512]))
    min_eps = torch.cat(me).cpu().numpy()
    print(f"  done ({time.time()-t0:.1f}s)  mean min_eps={min_eps.mean():.4f}")

    # Restrict feature matrix to correctly-classified subset
    feat_names = ["victim_margin", "mean_pix", "std_pix", "d_same", "d_any", "d_gap"]
    feat_mat = np.stack([
        victim_margin[keep],
        mean_pix[keep],
        std_pix[keep],
        d_same[keep],
        d_any[keep],
        d_gap[keep],
    ], axis=1).astype(np.float64)

    binary_targets = [f_fgsm, f_pgd]
    binary_names = ["flipped_FGSM", "flipped_PGD"]

    evaluate(feat_mat, feat_names, binary_targets, binary_names,
             min_eps, baseline_idx=feat_names.index("victim_margin"))

    print("\n========== summary ==========")
    print("H09 asks: do raw-pixel k-NN distances predict adversarial vulnerability?")
    print("Look at: univariate AUROC of d_same/d_any/d_gap vs flipped_{FGSM,PGD},")
    print("        correlation with min_eps_FGSM (expected negative if H09 holds),")
    print("        and delta-AUROC when adding knn features to margin-only model.")


if __name__ == "__main__":
    main()
