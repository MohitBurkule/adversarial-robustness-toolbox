"""
H95: Cosine similarity (pixel space) to nearest training sample predicts
adversarial vulnerability.

Per test sample we compute:
  - cos_same   : cosine similarity to nearest training sample of the SAME (true) class
  - cos_other  : cosine similarity to nearest training sample of any OTHER class
  - cos_margin : cos_same - cos_other  (positive => closer to own class than to others)

Additional control features:
  - mean_pix, std_pix    : per-sample pixel statistics

Targets:
  - flipped_by_FGSM    (eps = 15/255)
  - flipped_by_PGD     (eps = 15/255, alpha=eps/4, 20 steps)
  - min_eps_to_flip    (FGSM binary search; continuous target)

Architecture matches dissertation_extension/diagnostic_test.py (small CNN,
10 epochs on Fashion-MNIST, Adam lr=1e-3).  We subsample 5000 training
images to make the nearest-neighbour search cheap.

We report univariate AUROC of each feature against the binary targets
and Pearson correlation for the continuous target.

Self-contained: run as
    python h95_cosine_to_train.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
N_TRAIN_SUB = 5000  # training subsample for nearest-neighbour cosine
SEED = 0


# ----- CNN matching diagnostic_test.py -----
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


# ----- attacks -----
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=None, steps=20):
    if alpha is None:
        alpha = eps / 4
    x_orig = x.clone().detach()
    # random start within eps-ball
    delta = (torch.rand_like(x_orig) * 2 - 1) * eps
    adv = (x_orig + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
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


# ----- training -----
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
        print(f"  epoch {ep+1}/{EPOCHS} done ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ----- cosine features -----
def cosine_features(test_x, test_y, train_x_sub, train_y_sub):
    """
    Returns cos_same, cos_other, cos_margin for each test sample.

    cos_same  = max over training samples of the SAME class of cosine(x, x_train)
    cos_other = max over training samples of a DIFFERENT class of cosine(x, x_train)
    cos_margin = cos_same - cos_other

    Operates in pixel space (flattened, L2-normalised).
    """
    N = test_x.size(0)
    M = train_x_sub.size(0)

    flat_test = test_x.view(N, -1)
    flat_train = train_x_sub.view(M, -1)

    # L2-normalise; add eps to avoid div-by-zero on all-zero patches
    nt = flat_test / (flat_test.norm(dim=1, keepdim=True) + 1e-12)
    nr = flat_train / (flat_train.norm(dim=1, keepdim=True) + 1e-12)

    cos_same = torch.full((N,), -1.0, device=test_x.device)
    cos_other = torch.full((N,), -1.0, device=test_x.device)

    CHUNK = 256
    for i in range(0, N, CHUNK):
        sl = slice(i, min(i + CHUNK, N))
        sims = nt[sl] @ nr.t()           # (chunk, M)
        ty = test_y[sl].unsqueeze(1)     # (chunk, 1)
        same_mask = (train_y_sub.unsqueeze(0) == ty)   # (chunk, M)
        other_mask = ~same_mask

        # If a class is missing from the training subsample (very unlikely with 5000),
        # masked_fill of -inf would give -inf max. We guard by clamping.
        sims_same = sims.masked_fill(~same_mask, -2.0)
        sims_other = sims.masked_fill(~other_mask, -2.0)

        cos_same[sl] = sims_same.max(dim=1).values
        cos_other[sl] = sims_other.max(dim=1).values

    cos_margin = cos_same - cos_other
    return cos_same, cos_other, cos_margin


# ----- main -----
def main():
    print(f"device={DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(f"training CNN for {EPOCHS} epochs on Fashion-MNIST...")
    t0 = time.time()
    model = train_model(train_set)
    print(f"  training done ({time.time()-t0:.1f}s)")

    # full test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # subsample training (class-stratified would be nicer but a random 5000
    # is comfortably balanced for Fashion-MNIST)
    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(len(train_set), N_TRAIN_SUB, replace=False)
    sub_idx_sorted = np.sort(sub_idx)
    sub = Subset(train_set, sub_idx_sorted.tolist())
    train_x_sub = torch.stack([sub[i][0] for i in range(len(sub))]).to(DEVICE)
    train_y_sub = torch.tensor([sub[i][1] for i in range(len(sub))]).to(DEVICE)
    # class coverage check
    classes_present = torch.unique(train_y_sub)
    if classes_present.numel() < 10:
        print(f"  WARNING: only {classes_present.numel()} classes in subsample")

    # restrict analysis to test samples Model classifies correctly
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct = preds == test_y
    print(f" {correct.sum().item()} / {test_x.size(0)} samples correctly classified")

    test_x_c = test_x[correct]
    test_y_c = test_y[correct]

    print(" computing cosine-to-train features...")
    t0 = time.time()
    cos_same, cos_other, cos_margin = cosine_features(
        test_x_c, test_y_c, train_x_sub, train_y_sub
    )
    print(f"  done ({time.time()-t0:.1f}s)")

    # control features
    flat = test_x_c.view(test_x_c.size(0), -1)
    mean_pix = flat.mean(dim=1)
    std_pix = flat.std(dim=1)

    feat_names = ["cos_same", "cos_other", "cos_margin", "mean_pix", "std_pix"]
    feats = torch.stack([cos_same, cos_other, cos_margin, mean_pix, std_pix], dim=1)

    # ----- targets -----
    print(" computing FGSM flip target (eps=15/255)...")
    t0 = time.time()
    fgsm_chunks = []
    for i in range(0, test_x_c.size(0), 512):
        fgsm_chunks.append(fgsm_flip(model, test_x_c[i:i+512], test_y_c[i:i+512]))
    fgsm_target = torch.cat(fgsm_chunks)
    print(f"  done ({time.time()-t0:.1f}s)  pos_rate={fgsm_target.float().mean():.3f}")

    print(" computing PGD flip target (eps=15/255, 20 steps)...")
    t0 = time.time()
    pgd_chunks = []
    for i in range(0, test_x_c.size(0), 512):
        pgd_chunks.append(pgd_flip(model, test_x_c[i:i+512], test_y_c[i:i+512]))
    pgd_target = torch.cat(pgd_chunks)
    print(f"  done ({time.time()-t0:.1f}s)  pos_rate={pgd_target.float().mean():.3f}")

    print(" computing min_eps_to_flip (FGSM binary search)...")
    t0 = time.time()
    me_chunks = []
    for i in range(0, test_x_c.size(0), 512):
        me_chunks.append(min_eps_to_flip(model, test_x_c[i:i+512], test_y_c[i:i+512]))
    min_eps = torch.cat(me_chunks)
    print(f"  done ({time.time()-t0:.1f}s)  mean min_eps={min_eps.mean():.4f}")

    # ----- univariate AUROC -----
    feats_np = feats.detach().cpu().numpy()
    binary_targets = {
        "FGSM_flip": fgsm_target.detach().cpu().numpy().astype(int),
        "PGD_flip": pgd_target.detach().cpu().numpy().astype(int),
    }
    min_eps_np = min_eps.detach().cpu().numpy()

    print("\n===== Univariate AUROC (binary targets) =====")
    print(f"{'feature':<14}" + "".join(f"{t:>14}" for t in binary_targets))
    for i, fn in enumerate(feat_names):
        row = f"{fn:<14}"
        x_i = feats_np[:, i]
        for t_name, y in binary_targets.items():
            if y.std() == 0:
                row += f"{'n/a':>14}"
                continue
            a = roc_auc_score(y, x_i)
            a = max(a, 1 - a)  # report direction-agnostic AUROC
            row += f"{a:>14.4f}"
        print(row)

    print("\n===== Pearson correlation with min_eps_to_flip (continuous) =====")
    for i, fn in enumerate(feat_names):
        cor = np.corrcoef(feats_np[:, i], min_eps_np)[0, 1]
        print(f"  corr(min_eps, {fn:<14}) = {cor:+.4f}")

    # summary positive rates
    print("\n===== Target summary =====")
    for t_name, y in binary_targets.items():
        print(f"  {t_name}: positive rate = {y.mean():.3f}  (n={len(y)})")
    print(f"  min_eps: mean={min_eps_np.mean():.4f} std={min_eps_np.std():.4f}")


if __name__ == "__main__":
    main()
