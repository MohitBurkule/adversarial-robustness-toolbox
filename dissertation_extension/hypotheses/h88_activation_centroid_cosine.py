"""
H88: Activation-centroid cosine similarity as a predictor of adversarial vulnerability.

Hypothesis
----------
For each test sample, the cosine similarity between its penultimate-layer
activation and the *mean* penultimate activation of its predicted class
(computed over the training set) carries predictive signal about adversarial
vulnerability over and above the final-model logit margin.

Pipeline
--------
1. Train a small CNN (architecture matching diagnostic_test.py) on
   Fashion-MNIST for 10 epochs.
2. Per training class compute the mean penultimate activation (centroid).
3. For each correctly-classified test sample compute:
     - cos_own    : cosine to centroid of the predicted class
     - cos_other  : cosine to nearest *other* class centroid
     - cos_gap    : cos_own - cos_other
     - margin     : final logit margin (top - 2nd)
     - mean_pix   : per-image pixel mean
     - std_pix    : per-image pixel std
4. Compute three vulnerability targets:
     - flipped_by_FGSM      at eps = 15/255
     - flipped_by_PGD       at eps = 15/255, 10 steps, alpha = eps/4
     - min_eps_to_flip      binary-searched FGSM minimum eps
5. Univariate AUROC of each feature vs the binary targets, and Spearman-style
   correlation vs the continuous min_eps target.
6. Multivariate logistic regression with ablation against `margin` (the
   strong baseline): full feature set vs feature set with margin removed,
   and additionally cosine-only vs margin-only.

Code only - DO NOT run.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0
SEED = 0


# --------------------------------------------------------------------------- #
# Model (matches diagnostic_test.py)
# --------------------------------------------------------------------------- #
class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def features(self, x):
        """Return penultimate (post-ReLU, pre-dropout, pre-fc2) activations."""
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        h = self.features(x)
        return self.fc2(self.do2(h))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(train_set, n_classes=10, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Penultimate centroids over the training set
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_train_centroids(model, train_set, n_classes=10):
    """Mean penultimate activation per training-label class."""
    loader = DataLoader(train_set, 512, shuffle=False, num_workers=2)
    dim = model.fc1.out_features
    sums = torch.zeros(n_classes, dim, device=DEVICE)
    counts = torch.zeros(n_classes, device=DEVICE)
    model.eval()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        feats = model.features(x)
        for c in range(n_classes):
            mask = (y == c)
            if mask.any():
                sums[c] += feats[mask].sum(0)
                counts[c] += mask.sum()
    centroids = sums / counts.clamp(min=1).unsqueeze(1)
    return centroids  # (n_classes, dim)


# --------------------------------------------------------------------------- #
# Feature extraction for the test set
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_test_features(model, centroids, test_x, test_y, n_classes=10):
    """
    Returns
    -------
    feats     : (N, 6) tensor   cos_own, cos_other, cos_gap, margin, mean_pix, std_pix
    pred      : (N,)            predicted class (from final model)
    """
    N = test_x.size(0)
    feats_pen = []
    logits_all = []
    for i in range(0, N, 512):
        h = model.features(test_x[i:i+512])
        logits_all.append(model.fc2(h))
        feats_pen.append(h)
    feats_pen = torch.cat(feats_pen, 0)             # (N, D)
    logits = torch.cat(logits_all, 0)               # (N, C)
    pred = logits.argmax(1)
    sorted_logits, _ = logits.sort(1, descending=True)
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]

    # normalise
    fn = F.normalize(feats_pen, dim=1)              # (N, D)
    cn = F.normalize(centroids, dim=1)              # (C, D)
    cos_all = fn @ cn.t()                           # (N, C)

    cos_own = cos_all[torch.arange(N, device=DEVICE), pred]
    # nearest other-class centroid
    mask = F.one_hot(pred, n_classes).bool()
    cos_other_full = cos_all.masked_fill(mask, -1e9)
    cos_other = cos_other_full.max(1).values
    cos_gap = cos_own - cos_other

    mean_pix = test_x.view(N, -1).mean(1)
    std_pix = test_x.view(N, -1).std(1)

    feats = torch.stack([cos_own, cos_other, cos_gap,
                         margin, mean_pix, std_pix], 1)
    return feats, pred


# --------------------------------------------------------------------------- #
# Adversarial attacks
# --------------------------------------------------------------------------- #
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x_orig = x.clone().detach()
    # random start within the eps-ball
    delta = torch.empty_like(x).uniform_(-eps, eps)
    adv = (x_orig + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.min(torch.max(adv, x_orig - eps), x_orig + eps)
        adv = adv.clamp(0, 1)
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


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def batched(fn, x, y, batch=256, **kw):
    outs = []
    for i in range(0, x.size(0), batch):
        outs.append(fn(x[i:i+batch], y[i:i+batch], **kw))
    return torch.cat(outs)


def evaluate(feats, feat_names, targets_bin, target_names, min_eps):
    print("\n========== H88: activation-centroid cosine ==========")
    feats_np = feats.detach().cpu().numpy()
    Xs = StandardScaler().fit_transform(feats_np)

    # ---------- binary targets ----------
    for tname, ybin in zip(target_names, targets_bin):
        y = ybin.detach().cpu().numpy().astype(int)
        if y.std() == 0:
            print(f"\n  target {tname} is degenerate (pos rate {y.mean():.3f})")
            continue
        print(f"\n--- target: {tname}   (pos rate = {y.mean():.3f}) ---")

        # univariate AUROC
        for i, n in enumerate(feat_names):
            a = roc_auc_score(y, feats_np[:, i])
            a = max(a, 1 - a)
            print(f"   univariate AUROC  {n:<12} {a:.4f}")

        # multivariate ablation vs margin
        lr_full = LogisticRegression(max_iter=2000).fit(Xs, y)
        auc_full = roc_auc_score(y, lr_full.predict_proba(Xs)[:, 1])

        margin_idx = feat_names.index("margin")
        Xs_no_margin = np.delete(Xs, margin_idx, axis=1)
        lr_no_margin = LogisticRegression(max_iter=2000).fit(Xs_no_margin, y)
        auc_no_margin = roc_auc_score(y, lr_no_margin.predict_proba(Xs_no_margin)[:, 1])

        # cosine-only block (cos_own, cos_other, cos_gap)
        cos_idx = [feat_names.index(n) for n in ("cos_own", "cos_other", "cos_gap")]
        Xs_cos = Xs[:, cos_idx]
        lr_cos = LogisticRegression(max_iter=2000).fit(Xs_cos, y)
        auc_cos = roc_auc_score(y, lr_cos.predict_proba(Xs_cos)[:, 1])

        # margin-only baseline
        Xs_margin = Xs[:, [margin_idx]]
        lr_margin = LogisticRegression(max_iter=2000).fit(Xs_margin, y)
        auc_margin = roc_auc_score(y, lr_margin.predict_proba(Xs_margin)[:, 1])

        print(f"   multivariate AUROC (all 6 features):      {auc_full:.4f}")
        print(f"   multivariate AUROC (margin removed):      {auc_no_margin:.4f}")
        print(f"   AUROC cosine-only block (3 feats):        {auc_cos:.4f}")
        print(f"   AUROC margin-only baseline:               {auc_margin:.4f}")
        print(f"   Delta AUROC attributable to margin:       "
              f"{auc_full - auc_no_margin:+.4f}")
        print(f"   Delta AUROC cosine block over margin:     "
              f"{auc_cos - auc_margin:+.4f}")

        print("   standardised coefficients:")
        for n, c in zip(feat_names, lr_full.coef_.flatten()):
            print(f"     {n:<12} {c:+.4f}")

    # ---------- continuous min_eps ----------
    print("\n--- target: min_eps_to_flip (continuous) ---")
    y_cont = min_eps.detach().cpu().numpy()
    for i, n in enumerate(feat_names):
        cor = np.corrcoef(feats_np[:, i], y_cont)[0, 1]
        print(f"   corr(min_eps, {n:<12}) = {cor:+.4f}")
    ols_full = LinearRegression().fit(Xs, y_cont)
    r2_full = ols_full.score(Xs, y_cont)

    margin_idx = feat_names.index("margin")
    Xs_no_margin = np.delete(Xs, margin_idx, axis=1)
    ols_no_margin = LinearRegression().fit(Xs_no_margin, y_cont)
    r2_no_margin = ols_no_margin.score(Xs_no_margin, y_cont)

    cos_idx = [feat_names.index(n) for n in ("cos_own", "cos_other", "cos_gap")]
    Xs_cos = Xs[:, cos_idx]
    ols_cos = LinearRegression().fit(Xs_cos, y_cont)
    r2_cos = ols_cos.score(Xs_cos, y_cont)

    Xs_margin = Xs[:, [margin_idx]]
    ols_margin = LinearRegression().fit(Xs_margin, y_cont)
    r2_margin = ols_margin.score(Xs_margin, y_cont)

    print(f"  OLS R^2 all features:        {r2_full:.4f}")
    print(f"  OLS R^2 margin removed:      {r2_no_margin:.4f}")
    print(f"  OLS R^2 cosine-only block:   {r2_cos:.4f}")
    print(f"  OLS R^2 margin-only:         {r2_margin:.4f}")
    print(f"  Delta R^2 from margin:       {r2_full - r2_no_margin:+.4f}")
    print(f"  Delta R^2 cosine vs margin:  {r2_cos - r2_margin:+.4f}")
    print("  standardised coefficients:")
    for n, c in zip(feat_names, ols_full.coef_):
        print(f"   {n:<12} {c:+.6f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True,
                                      download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False,
                                     download=True, transform=tf)

    print("Training CNN on Fashion-MNIST for "
          f"{EPOCHS} epochs (seed={SEED}) ...")
    model = train_model(train_set, n_classes=10, seed=SEED)

    print("Computing per-class penultimate centroids on training set ...")
    centroids = compute_train_centroids(model, train_set, n_classes=10)

    # stack test set into memory
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("Computing per-sample features ...")
    feats, pred = compute_test_features(model, centroids, test_x, test_y,
                                        n_classes=10)
    feat_names = ["cos_own", "cos_other", "cos_gap",
                  "margin", "mean_pix", "std_pix"]

    # restrict to samples the model classifies correctly
    correct = pred == test_y
    print(f"  using {int(correct.sum())} / {test_y.numel()} correctly-classified samples")
    x_c = test_x[correct]
    y_c = test_y[correct]
    feats_c = feats[correct]

    print("Running FGSM attack (eps=15/255) ...")
    fgsm_flip = batched(lambda x, y: fgsm_attack(model, x, y, EPS_TEST),
                        x_c, y_c)
    print(f"  FGSM flip rate = {fgsm_flip.float().mean():.4f}")

    print(f"Running PGD attack (eps=15/255, steps={PGD_STEPS}) ...")
    pgd_flip = batched(lambda x, y: pgd_attack(model, x, y,
                                               EPS_TEST, PGD_ALPHA, PGD_STEPS),
                       x_c, y_c)
    print(f"  PGD flip rate = {pgd_flip.float().mean():.4f}")

    print("Computing min_eps_to_flip (binary-search FGSM) ...")
    min_eps = batched(lambda x, y: min_eps_to_flip(model, x, y),
                      x_c, y_c)
    print(f"  mean min_eps = {min_eps.mean():.4f}")

    evaluate(feats_c, feat_names,
             targets_bin=[fgsm_flip, pgd_flip],
             target_names=["FGSM_flip", "PGD_flip"],
             min_eps=min_eps)


if __name__ == "__main__":
    main()
