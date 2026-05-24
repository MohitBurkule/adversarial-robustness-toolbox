"""
Hypothesis H31: Label diversity of K-nearest training neighbours in pixel space
predicts adversarial vulnerability.

Specifically, for each test sample we look at its K=20 nearest training
neighbours in raw-pixel L2 space and compute three diversity statistics over
their *labels*:

    majority_label_fraction    : fraction with the modal neighbour label
                                 (higher => purer neighbourhood)
    label_entropy              : Shannon entropy (nats) of the K-neighbour-label
                                 distribution (higher => more mixed)
    true_class_neighbour_fraction
                               : fraction of K neighbours whose label equals the
                                 victim's predicted class on the test sample
                                 (higher => neighbourhood agrees with victim)

Rationale: if a test sample sits among training samples of different classes
in pixel space, the local data manifold is mixed and the sample lies near a
true class boundary — predicted to be more adversarially vulnerable.

Pipeline (self-contained):
  1. Train small CNN victim on Fashion-MNIST (10 epochs Adam, same as
     diagnostic_test.py / h09).
  2. Random subsample 5000 training images. Use sklearn NearestNeighbors
     (brute, L2) over flattened pixels to fetch K=20 neighbours per test sample.
  3. Compute the three label-diversity features above
     (true_class_neighbour_fraction uses the victim's *predicted* class).
  4. Baseline scalars: victim_margin, mean_pix, std_pix.
  5. Targets:
        flipped_FGSM  : FGSM eps=15/255 flips victim's top-1
        flipped_PGD   : PGD 10 steps, eps=15/255, alpha=eps/4
        min_eps_FGSM  : binary-searched smallest FGSM eps that flips
  6. Univariate AUROC + Pearson with min_eps + multivariate logistic
     ablation: does KNN-label-diversity add over victim_margin?

Run:  python h31_knn_label_diversity.py
Outputs: stdout. Uses CUDA when available. Downloads data into /tmp/data.
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
K = 20
SUBSAMPLE = 5000
PGD_STEPS = 10
SEED = 0
N_CLASSES = 10


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
    model = CNN(N_CLASSES).to(DEVICE)
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
# Label-diversity features over K nearest pixel-space training neighbours
# ----------------------------------------------------------------------------
def compute_label_diversity(train_x_flat, train_y, test_x_flat, victim_pred,
                            k=K, n_classes=N_CLASSES):
    """
    Returns dict of arrays of shape (N_test,):
        majority_label_fraction
        label_entropy
        true_class_neighbour_fraction  (uses victim_pred as 'true class')
    """
    nn_obj = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    nn_obj.fit(train_x_flat)
    _, idx = nn_obj.kneighbors(test_x_flat)        # (N_test, k)
    neigh_labels = train_y[idx]                    # (N_test, k)

    N = test_x_flat.shape[0]
    maj_frac = np.zeros(N, dtype=np.float32)
    ent = np.zeros(N, dtype=np.float32)
    true_frac = np.zeros(N, dtype=np.float32)

    # Vectorised counts: one-hot histogram per row.
    # counts[i, c] = number of neighbours of test i with label c
    counts = np.zeros((N, n_classes), dtype=np.int32)
    rows = np.repeat(np.arange(N), k)
    np.add.at(counts, (rows, neigh_labels.ravel()), 1)
    probs = counts.astype(np.float32) / float(k)

    maj_frac = probs.max(axis=1)
    # Shannon entropy in nats; treat 0*log0 = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(probs > 0, np.log(probs), 0.0)
    ent = -(probs * logp).sum(axis=1)
    true_frac = probs[np.arange(N), victim_pred]

    return {
        "majority_label_fraction": maj_frac,
        "label_entropy": ent,
        "true_class_neighbour_fraction": true_frac,
    }


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(feat_mat, feat_names, binary_targets, binary_names,
             min_eps, baseline_idx, diversity_names):
    Xs = StandardScaler().fit_transform(feat_mat)

    print("\n--- Univariate AUROC (binary targets) ---")
    print(f"{'feature':<32}" + "".join(f"{tn:>22}" for tn in binary_names))
    for i, fn in enumerate(feat_names):
        row = [f"{fn:<32}"]
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
        print(f"  corr(min_eps, {fn:<32}) = {r:+.4f}")

    print("\n--- Multivariate logistic regression (binary targets) ---")
    diversity_idx = [feat_names.index(n) for n in diversity_names]
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
        auc1, lr1 = fit_auc([baseline_idx] + diversity_idx)
        auc2, lr2 = fit_auc(list(range(Xs.shape[1])))
        print(f"    AUROC  margin-only                 : {auc0:.4f}")
        print(f"    AUROC  margin + label-diversity    : {auc1:.4f}  (delta {auc1-auc0:+.4f})")
        print(f"    AUROC  margin + all features       : {auc2:.4f}  (delta {auc2-auc0:+.4f})")
        print(f"    coefficients (margin + label-diversity):")
        names_used = [feat_names[baseline_idx]] + [feat_names[i] for i in diversity_idx]
        for n, c in zip(names_used, lr1.coef_.flatten()):
            print(f"      {n:<32} {c:+.4f}")


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

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))])
    train_x = torch.stack([train_set[i][0] for i in range(len(train_set))])
    train_y = torch.tensor([train_set[i][1] for i in range(len(train_set))])

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train_set), size=SUBSAMPLE, replace=False)
    tr_x_sub = train_x[idx].numpy().reshape(SUBSAMPLE, -1).astype(np.float32)
    tr_y_sub = train_y[idx].numpy()
    te_x_flat = test_x.numpy().reshape(len(test_set), -1).astype(np.float32)

    # Victim margins + predictions
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
    pred_np = pred.cpu().numpy()
    correct = (pred == test_y_dev)
    print(f"  victim clean accuracy = {correct.float().mean().item():.4f}")

    print(f"computing K-NN label-diversity (K={K}, train_subsample={SUBSAMPLE})...")
    t0 = time.time()
    div = compute_label_diversity(tr_x_sub, tr_y_sub, te_x_flat, pred_np,
                                  k=K, n_classes=N_CLASSES)
    print(f"  done ({time.time()-t0:.1f}s)")
    for k_, v in div.items():
        print(f"  {k_:<32}: mean={v.mean():.4f}  std={v.std():.4f}")

    mean_pix = te_x_flat.mean(axis=1)
    std_pix = te_x_flat.std(axis=1)

    keep = correct.cpu().numpy()
    x_c = test_x_dev[correct]
    y_c = test_y_dev[correct]
    print(f"  using {keep.sum()} correctly-classified test samples")

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

    diversity_names = ["majority_label_fraction",
                       "label_entropy",
                       "true_class_neighbour_fraction"]
    feat_names = ["victim_margin", "mean_pix", "std_pix"] + diversity_names
    feat_mat = np.stack([
        victim_margin[keep],
        mean_pix[keep],
        std_pix[keep],
        div["majority_label_fraction"][keep],
        div["label_entropy"][keep],
        div["true_class_neighbour_fraction"][keep],
    ], axis=1).astype(np.float64)

    binary_targets = [f_fgsm, f_pgd]
    binary_names = ["flipped_FGSM", "flipped_PGD"]

    evaluate(feat_mat, feat_names, binary_targets, binary_names,
             min_eps, baseline_idx=feat_names.index("victim_margin"),
             diversity_names=diversity_names)

    print("\n========== summary ==========")
    print("H31 asks: does K-NN label diversity (in raw pixel space) predict")
    print("         adversarial vulnerability?")
    print("Expected if H31 holds:")
    print("  - label_entropy positively correlates with flip rate")
    print("  - majority_label_fraction & true_class_neighbour_fraction NEGATIVELY")
    print("    correlate with flip rate (and positively with min_eps_FGSM)")
    print("  - margin + label-diversity beats margin-only AUROC")


if __name__ == "__main__":
    main()
