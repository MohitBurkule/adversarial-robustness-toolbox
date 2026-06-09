"""
Hypothesis H51: Cosine similarity between a sample's input-gradient and the
average input-gradient of all training samples in its (true) class predicts
adversarial vulnerability. Low cosine = atypical gradient direction =
more attackable.

Novel angle: we look at the *direction* of the input gradient, not just its
norm. A sample whose input-gradient aligns with the class-average direction
is "typical"; misaligned samples are atypical and hypothesised to be easier
to attack.

Pipeline:
  1. Train a small CNN victim on Fashion-MNIST (10 epochs).
  2. Compute input-gradients (CE loss wrt input) for 5000 training samples;
     unit-normalise each and average within each class -> per-class mean
     gradient direction (then re-normalised).
  3. For each test sample, compute its input-gradient direction (unit-norm)
     and the cosine with the mean direction of its true class.
  4. Features: grad_cosine_to_class_avg, victim_margin, input_grad_l2_norm,
     mean_pix, std_pix.
  5. Targets: flipped_FGSM (eps=15/255), flipped_PGD (eps=15/255, 10 steps),
     FGSM_min_eps (continuous + binarised by median).
  6. Univariate AUROC. Multivariate: does grad_cosine add over
     input_grad_norm + margin?

Self-contained — DO NOT run from this comment; the user just wants the code.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


DEVICE = torch.device("cuda")
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4
N_TRAIN_GRAD = 5000   # subsample of training set used for class means
N_CLASSES = 10
DATA_ROOT = "/tmp/data"
SEED = 0


# ------------------------- model -------------------------
class CNN(nn.Module):
    """Matches diagnostic_test.py architecture."""
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


def train_victim(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
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
        print(f"  epoch {ep+1}/{EPOCHS}  loss={loss_sum/total:.4f}  "
              f"acc={correct/total:.4f}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# --------------------- gradient utilities ----------------
def input_grad(model, x, y):
    """Return raw input gradient of CE loss wrt x.  Model must be in eval()."""
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y, reduction="sum")
    grad = torch.autograd.grad(loss, x)[0]
    return grad.detach()


def batched_input_grads(model, X, Y, batch=256):
    grads = torch.empty_like(X)
    for i in range(0, X.size(0), batch):
        g = input_grad(model, X[i:i+batch], Y[i:i+batch])
        grads[i:i+batch] = g
    return grads


def unit_normalise(g, eps=1e-12):
    """Flatten per-sample, divide by L2 norm."""
    flat = g.flatten(1)
    n = flat.norm(dim=1, keepdim=True).clamp_min(eps)
    return (flat / n).view_as(g), flat.norm(dim=1)


def compute_class_mean_directions(model, train_set):
    """Subsample N_TRAIN_GRAD training samples; per class compute mean of
    unit-normed input gradients; renormalise to unit length."""
    N = len(train_set)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(N, size=min(N_TRAIN_GRAD, N), replace=False)
    xs = torch.stack([train_set[int(i)][0] for i in idx]).to(DEVICE)
    ys = torch.tensor([train_set[int(i)][1] for i in idx]).to(DEVICE)
    grads = batched_input_grads(model, xs, ys)
    unit_g, _ = unit_normalise(grads)
    unit_flat = unit_g.flatten(1)  # (N, D)

    mean_dirs = torch.zeros(N_CLASSES, unit_flat.size(1), device=DEVICE)
    counts = torch.zeros(N_CLASSES, device=DEVICE)
    for c in range(N_CLASSES):
        m = (ys == c)
        if m.any():
            mean_dirs[c] = unit_flat[m].mean(0)
            counts[c] = m.sum()
    # renormalise to unit length
    mean_dirs = mean_dirs / mean_dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    print("  class-mean direction counts per class:", counts.cpu().int().tolist())
    return mean_dirs


# --------------------- attacks ---------------------------
def fgsm_attack(model, x, y, eps=EPS_TEST):
    g = input_grad(model, x, y)
    adv = (x + eps * g.sign()).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x_adv = x.clone().detach()
    # random start within eps ball
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="sum")
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(x_adv).argmax(1) != y)


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=15):
    """Binary search smallest eps (per-sample) that flips FGSM."""
    sign = input_grad(model, x, y).sign()
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def run_batched(fn, X, Y, batch=256, **kw):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(fn(X[i:i+batch], Y[i:i+batch], **kw))
    return torch.cat(outs)


# --------------------- feature pipeline ------------------
def compute_features(model, mean_dirs, X, Y):
    """Compute all per-sample features.  X and Y are on DEVICE."""
    N = X.size(0)
    # input gradients (single forward+backward per batch)
    grads = batched_input_grads(model, X, Y)
    unit_g, g_norm = unit_normalise(grads)
    unit_flat = unit_g.flatten(1)  # (N, D)

    # cosine with class mean direction (mean_dirs already unit-norm)
    cm = mean_dirs[Y]                       # (N, D)
    cos = (unit_flat * cm).sum(1)           # (N,)

    # victim margin (logit gap between top-1 and 2nd-best)
    with torch.no_grad():
        logits_chunks = []
        for i in range(0, N, 512):
            logits_chunks.append(model(X[i:i+512]))
        logits = torch.cat(logits_chunks, 0)
    sl, _ = logits.sort(1, descending=True)
    margin = sl[:, 0] - sl[:, 1]
    pred = logits.argmax(1)

    # pixel stats
    flat_x = X.flatten(1)
    mean_pix = flat_x.mean(1)
    std_pix = flat_x.std(1)

    feats = torch.stack([cos, margin, g_norm, mean_pix, std_pix], 1)
    names = ["grad_cosine_to_class_avg", "victim_margin",
             "input_grad_l2_norm", "mean_pix", "std_pix"]
    return feats, names, pred


# --------------------- evaluation ------------------------
def univariate_auroc(feats_np, names, y):
    print("  univariate AUROC:")
    for i, n in enumerate(names):
        a = roc_auc_score(y, feats_np[:, i])
        a = max(a, 1 - a)
        print(f"    {n:<28} {a:.4f}")


def multivariate(feats_np, names, y, focal="grad_cosine_to_class_avg",
                 baseline=("input_grad_l2_norm", "victim_margin")):
    Xs = StandardScaler().fit_transform(feats_np)
    # full
    full = LogisticRegression(max_iter=2000).fit(Xs, y)
    full_auc = roc_auc_score(y, full.predict_proba(Xs)[:, 1])

    # baseline (no focal)
    base_idx = [names.index(n) for n in baseline]
    base_lr = LogisticRegression(max_iter=2000).fit(Xs[:, base_idx], y)
    base_auc = roc_auc_score(y, base_lr.predict_proba(Xs[:, base_idx])[:, 1])

    # baseline + focal
    focal_idx = names.index(focal)
    cols = base_idx + [focal_idx]
    bf_lr = LogisticRegression(max_iter=2000).fit(Xs[:, cols], y)
    bf_auc = roc_auc_score(y, bf_lr.predict_proba(Xs[:, cols])[:, 1])

    print(f"  multivariate AUROC:")
    print(f"    full (all 5):                          {full_auc:.4f}")
    print(f"    baseline ({'+'.join(baseline)}):       {base_auc:.4f}")
    print(f"    baseline + {focal}:                    {bf_auc:.4f}")
    print(f"    delta from adding {focal}:             {bf_auc - base_auc:+.4f}")
    print("    standardised coefficients (full model):")
    for n, c in zip(names, full.coef_.flatten()):
        print(f"      {n:<28} {c:+.4f}")


def evaluate(feats, names, targets, target_names):
    feats_np = feats.detach().cpu().numpy()
    for y_t, t_name in zip(targets, target_names):
        y = y_t.detach().cpu().numpy().astype(int)
        if y.std() == 0:
            print(f"\n--- target {t_name}: degenerate (pos rate {y.mean():.3f}), skipping")
            continue
        print(f"\n--- target: {t_name}  (pos rate = {y.mean():.3f}, n={len(y)}) ---")
        univariate_auroc(feats_np, names, y)
        multivariate(feats_np, names, y)


# --------------------- main ------------------------------
def main():
    print("loading Fashion-MNIST ...")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print("training victim CNN ...")
    t0 = time.time()
    model = train_victim(train_set)
    print(f"  victim trained in {time.time()-t0:.1f}s")

    # materialise test tensors
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("computing per-class mean gradient directions from training subset ...")
    t0 = time.time()
    mean_dirs = compute_class_mean_directions(model, train_set)
    print(f"  done in {time.time()-t0:.1f}s")

    print("computing test-set features ...")
    t0 = time.time()
    feats, names, pred = compute_features(model, mean_dirs, test_x, test_y)
    print(f"  features {names} ready ({time.time()-t0:.1f}s)")

    # restrict to samples the victim classifies correctly
    correct = (pred == test_y)
    x_c, y_c, feats_c = test_x[correct], test_y[correct], feats[correct]
    print(f"  using {int(correct.sum())} correctly-classified test samples")

    # ------------- targets -------------
    print("computing FGSM (eps=15/255) flips ...")
    t0 = time.time()
    fgsm_flip = run_batched(lambda a, b: fgsm_attack(model, a, b), x_c, y_c)
    print(f"  done ({time.time()-t0:.1f}s) flip rate = {fgsm_flip.float().mean():.3f}")

    print("computing PGD (eps=15/255, 10 steps) flips ...")
    t0 = time.time()
    pgd_flip = run_batched(lambda a, b: pgd_attack(model, a, b), x_c, y_c)
    print(f"  done ({time.time()-t0:.1f}s) flip rate = {pgd_flip.float().mean():.3f}")

    print("computing FGSM min-eps (binary search) ...")
    t0 = time.time()
    min_eps = run_batched(lambda a, b: fgsm_min_eps(model, a, b), x_c, y_c)
    median = min_eps.median()
    fgsm_easy = (min_eps <= median)  # 1 = below-median eps -> easier to attack
    print(f"  done ({time.time()-t0:.1f}s) mean min_eps = {min_eps.mean():.4f}, "
          f"median = {median:.4f}")

    targets = [fgsm_flip, pgd_flip, fgsm_easy]
    target_names = ["flipped_FGSM", "flipped_PGD", "FGSM_min_eps_below_median"]

    print("\n========== H51 RESULTS (Fashion-MNIST) ==========")
    evaluate(feats_c, names, targets, target_names)

    # also report continuous correlation of grad_cosine with min_eps
    print("\n--- continuous: correlation of features with min_eps ---")
    feats_np = feats_c.detach().cpu().numpy()
    me_np = min_eps.detach().cpu().numpy()
    for i, n in enumerate(names):
        cor = np.corrcoef(feats_np[:, i], me_np)[0, 1]
        print(f"  corr(min_eps, {n:<28}) = {cor:+.4f}")


if __name__ == "__main__":
    main()
