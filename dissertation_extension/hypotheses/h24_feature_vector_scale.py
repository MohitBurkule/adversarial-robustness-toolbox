"""
Hypothesis H24: Per-sample L2 norm of the penultimate-layer feature vector
predicts adversarial vulnerability.

Context: dissertation extension.  Lin et al. ("The Role of Feature Vector
Scale in the Adversarial Vulnerability of CNNs", MDPI Mathematics 2025,
doi 10.3390/math13183026) showed on CIFAR-10 that small-feature-norm samples
are disproportionately easy to attack: because logits = W @ feature, a small
feature norm shrinks the logit-gap (margin) and thereby the perturbation
budget needed to flip the prediction.

We test this on Fashion-MNIST with a small CNN whose penultimate layer is the
128-d output of fc1 + ReLU (matching diagnostic_test.py).

Features computed per test sample from the 128-d penultimate vector phi(x):
    - feature_l2_norm        ||phi(x)||_2
    - feature_linf_norm      ||phi(x)||_inf
    - feature_entropy        H[softmax(phi(x))]      (distributional spread)
    - inv_feature_l2_norm    1 / ||phi(x)||_2        (paper's "vulnerable" direction)

Baselines:
    - victim_margin          (eval-mode top1 - top2 logit gap)
    - mean_pix, std_pix

Targets (defined ONLY on samples the victim classifies correctly):
    - flipped_FGSM           binary,    L_inf eps = 15/255
    - flipped_PGD            binary,    20 steps, alpha = 2/255, eps = 15/255
    - FGSM_min_eps           continuous, per-sample L_inf binary search

Evaluation:
    1. Univariate AUROC of each feature against each binary target.
    2. Spearman of each feature against the continuous target.
    3. Multivariate logistic regression:
         margin-only  vs.  margin + feature-norm block
         all          vs.  all minus feature-norm block
       Does feature norm add over margin?  Does it BEAT margin alone?
       (margin is derived from logits = W @ feature, so the two are
       mechanistically linked; the comparison is informative.)
    4. Scatter summary: bucket samples by feature_l2_norm decile and
       report the FGSM flip-rate and mean FGSM_min_eps within each bucket
       — are small-norm samples really more attackable?

Self-contained: trains the victim from scratch (Adam, 10 epochs).  Downloads
Fashion-MNIST to /tmp/data.  Writes nothing else to disk.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0
SEED = 0


# ---------------------------------------------------------------------------
class CNN(nn.Module):
    """Matches diagnostic_test.py: two conv, two FC, dropout 0.25 / 0.5.
    Penultimate vector = 128-d output of fc1 + ReLU (post-dropout in train
    mode; here we always read it in eval mode so dropout is identity).
    """
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
        x = F.relu(self.fc1(x))      # <-- penultimate (128-d)
        return x

    def forward(self, x):
        f = self.features(x)
        f = self.do2(f)
        return self.fc2(f)


# ---------------------------------------------------------------------------
def train_victim(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done")
    model.eval()
    return model


# ---------------------------------------------------------------------------
@torch.no_grad()
def batched_logits(model, x, bs=512):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(model(x[i:i+bs]))
    return torch.cat(out, 0)


@torch.no_grad()
def batched_features(model, x, bs=512):
    model.eval()
    out = []
    for i in range(0, x.size(0), bs):
        out.append(model.features(x[i:i+bs]))
    return torch.cat(out, 0)


def fgsm_attack(model, x, y, eps=EPS_TEST):
    was_training = model.training
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_adv), y).backward()
    sign = x_adv.grad.sign().detach()
    if was_training:
        model.train()
    return (x + eps * sign).clamp(0, 1).detach()


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    was_training = model.training
    model.eval()
    x0 = x.clone().detach()
    delta = torch.empty_like(x0).uniform_(-eps, eps)
    x_adv = (x0 + delta).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)
    if was_training:
        model.train()
    return x_adv.detach()


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for the smallest L_inf eps that flips FGSM."""
    was_training = model.training
    model.eval()
    x_grad = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_grad), y).backward()
    sign = x_grad.grad.sign().detach()
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    if was_training:
        model.train()
    return hi


def batched_attack_flag(model, x, y, attack_fn, bs=256):
    flags = []
    model.eval()
    for i in range(0, x.size(0), bs):
        adv = attack_fn(model, x[i:i+bs], y[i:i+bs])
        with torch.no_grad():
            flags.append(model(adv).argmax(1) != y[i:i+bs])
    return torch.cat(flags, 0)


# ---------------------------------------------------------------------------
def feature_scale_features(model, x):
    """Compute per-sample summaries of the 128-d penultimate vector phi(x)."""
    phi = batched_features(model, x)            # (N, 128)
    eps_log = 1e-12
    l2 = phi.norm(p=2, dim=1)                   # (N,)
    linf = phi.norm(p=float("inf"), dim=1)      # (N,)
    sm = F.softmax(phi, dim=1)
    ent = -(sm * (sm + eps_log).log()).sum(dim=1)
    inv_l2 = 1.0 / (l2 + eps_log)
    return {
        "feature_l2_norm": l2,
        "feature_linf_norm": linf,
        "feature_entropy": ent,
        "inv_feature_l2_norm": inv_l2,
    }


def victim_margin(model, x):
    model.eval()
    with torch.no_grad():
        logits = batched_logits(model, x)
    s, _ = logits.sort(1, descending=True)
    return (s[:, 0] - s[:, 1])


# ---------------------------------------------------------------------------
def auroc_both_dirs(y, score):
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    tf = transforms.ToTensor()
    print("Loading Fashion-MNIST ...")
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print(f"Training victim CNN ({EPOCHS} epochs) on {DEVICE} ...")
    model = train_victim(train_set)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # restrict to correctly classified (vulnerability is only defined here)
    model.eval()
    with torch.no_grad():
        pred = batched_logits(model, test_x).argmax(1)
    correct = pred == test_y
    print(f"Test acc (eval mode): {correct.float().mean().item():.4f}")
    x = test_x[correct]
    y = test_y[correct]
    N = x.size(0)
    print(f"Using {N} correctly-classified samples.")

    # --- features -------------------------------------------------------
    print("Computing victim_margin (eval mode) ...")
    margin = victim_margin(model, x)

    print("Computing penultimate-feature scale statistics ...")
    fs = feature_scale_features(model, x)

    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))

    feat_tensors = [
        margin,
        fs["feature_l2_norm"],
        fs["feature_linf_norm"],
        fs["feature_entropy"],
        fs["inv_feature_l2_norm"],
        mean_pix,
        std_pix,
    ]
    names = [
        "victim_margin",
        "feature_l2_norm",
        "feature_linf_norm",
        "feature_entropy",
        "inv_feature_l2_norm",
        "mean_pix",
        "std_pix",
    ]
    fs_block_names = [
        "feature_l2_norm",
        "feature_linf_norm",
        "feature_entropy",
        "inv_feature_l2_norm",
    ]
    feats = torch.stack(feat_tensors, 1).detach().cpu().numpy()

    # --- targets --------------------------------------------------------
    print("Running FGSM attack ...")
    flip_fgsm = batched_attack_flag(model, x, y, fgsm_attack).cpu().numpy().astype(int)

    print("Running PGD attack ...")
    flip_pgd = batched_attack_flag(model, x, y, pgd_attack).cpu().numpy().astype(int)

    print("Running min-eps FGSM binary search ...")
    me = []
    for i in range(0, N, 256):
        me.append(min_eps_fgsm(model, x[i:i+256], y[i:i+256]))
    min_eps = torch.cat(me, 0).cpu().numpy()

    # --- evaluation -----------------------------------------------------
    print("\n========== H24: penultimate feature-vector scale vs vulnerability ==========")
    print(f"N = {N}  FGSM-flip rate = {flip_fgsm.mean():.3f}  "
          f"PGD-flip rate = {flip_pgd.mean():.3f}")
    print(f"min_eps FGSM:  mean={min_eps.mean():.4f}  median={np.median(min_eps):.4f}")
    print(f"feature_l2_norm: mean={feats[:, names.index('feature_l2_norm')].mean():.3f}  "
          f"min={feats[:, names.index('feature_l2_norm')].min():.3f}  "
          f"max={feats[:, names.index('feature_l2_norm')].max():.3f}")

    bin_targets = [("flipped_FGSM", flip_fgsm), ("flipped_PGD", flip_pgd)]

    Xs = StandardScaler().fit_transform(feats)
    margin_idx = names.index("victim_margin")
    fs_idxs = [names.index(n) for n in fs_block_names]
    non_fs_idxs = [i for i in range(len(names)) if i not in fs_idxs]

    for t_name, yv in bin_targets:
        if yv.std() == 0:
            print(f"\n--- target {t_name}: no variance, skipping ---")
            continue
        print(f"\n--- target: {t_name}  (positive rate = {yv.mean():.3f}) ---")

        per_feat_auc = {}
        for i, n in enumerate(names):
            a = auroc_both_dirs(yv, feats[:, i])
            per_feat_auc[n] = a
            print(f"   univariate AUROC  {n:<22} {a:.4f}")

        best_fs = max(fs_block_names, key=lambda n: per_feat_auc[n])
        print(f"   strongest single feature-scale stat: {best_fs}  "
              f"(AUROC = {per_feat_auc[best_fs]:.4f})")
        print(f"   margin-alone AUROC: {per_feat_auc['victim_margin']:.4f}  "
              f"=> feature-norm beats margin? "
              f"{per_feat_auc[best_fs] > per_feat_auc['victim_margin']}")

        # multivariate logistic regressions
        lr_full = LogisticRegression(max_iter=4000).fit(Xs, yv)
        auc_full = roc_auc_score(yv, lr_full.predict_proba(Xs)[:, 1])

        Xs_m = Xs[:, [margin_idx]]
        lr_m = LogisticRegression(max_iter=4000).fit(Xs_m, yv)
        auc_m = roc_auc_score(yv, lr_m.predict_proba(Xs_m)[:, 1])

        Xs_m_fs = Xs[:, [margin_idx] + fs_idxs]
        lr_m_fs = LogisticRegression(max_iter=4000).fit(Xs_m_fs, yv)
        auc_m_fs = roc_auc_score(yv, lr_m_fs.predict_proba(Xs_m_fs)[:, 1])

        Xs_fs_only = Xs[:, fs_idxs]
        lr_fs_only = LogisticRegression(max_iter=4000).fit(Xs_fs_only, yv)
        auc_fs_only = roc_auc_score(yv, lr_fs_only.predict_proba(Xs_fs_only)[:, 1])

        Xs_no_fs = Xs[:, non_fs_idxs]
        lr_no_fs = LogisticRegression(max_iter=4000).fit(Xs_no_fs, yv)
        auc_no_fs = roc_auc_score(yv, lr_no_fs.predict_proba(Xs_no_fs)[:, 1])

        print(f"   multivariate AUROC (margin only):              {auc_m:.4f}")
        print(f"   multivariate AUROC (feature-norm block only):  {auc_fs_only:.4f}")
        print(f"   multivariate AUROC (margin + feature-norm):    {auc_m_fs:.4f}")
        print(f"     Delta over margin-alone:                       {auc_m_fs - auc_m:+.4f}")
        print(f"   multivariate AUROC (all features):             {auc_full:.4f}")
        print(f"   multivariate AUROC (all minus feature-norm):   {auc_no_fs:.4f}")
        print(f"     Delta from removing feature-norm block:        {auc_full - auc_no_fs:+.4f}")
        print("   standardised coefficients (full model):")
        for n, c in zip(names, lr_full.coef_.flatten()):
            print(f"     {n:<22} {c:+.4f}")

    # --- continuous target ----------------------------------------------
    print(f"\n--- continuous target: FGSM_min_eps (lower = more vulnerable) ---")
    from scipy.stats import spearmanr
    for i, n in enumerate(names):
        r, p = spearmanr(feats[:, i], min_eps)
        print(f"   spearman  {n:<22} rho={r:+.4f}  p={p:.2e}")

    ols_full = LinearRegression().fit(Xs, min_eps)
    r2_full = ols_full.score(Xs, min_eps)
    ols_m = LinearRegression().fit(Xs[:, [margin_idx]], min_eps)
    r2_m = ols_m.score(Xs[:, [margin_idx]], min_eps)
    ols_m_fs = LinearRegression().fit(Xs[:, [margin_idx] + fs_idxs], min_eps)
    r2_m_fs = ols_m_fs.score(Xs[:, [margin_idx] + fs_idxs], min_eps)
    ols_fs_only = LinearRegression().fit(Xs[:, fs_idxs], min_eps)
    r2_fs_only = ols_fs_only.score(Xs[:, fs_idxs], min_eps)
    ols_no_fs = LinearRegression().fit(Xs[:, non_fs_idxs], min_eps)
    r2_no_fs = ols_no_fs.score(Xs[:, non_fs_idxs], min_eps)
    print(f"   OLS R^2 (margin only):                 {r2_m:.4f}")
    print(f"   OLS R^2 (feature-norm block only):     {r2_fs_only:.4f}")
    print(f"   OLS R^2 (margin + feature-norm):       {r2_m_fs:.4f}")
    print(f"     Delta over margin:                     {r2_m_fs - r2_m:+.4f}")
    print(f"   OLS R^2 (all features):                {r2_full:.4f}")
    print(f"   OLS R^2 (all minus feature-norm):      {r2_no_fs:.4f}")
    print(f"     Delta:                                 {r2_full - r2_no_fs:+.4f}")
    print("   OLS coefficients (full model):")
    for n, c in zip(names, ols_full.coef_.flatten()):
        print(f"     {n:<22} {c:+.4f}")

    # --- decile scatter: are small-feature-norm samples really more attackable? ---
    print("\n--- decile analysis: feature_l2_norm vs vulnerability ---")
    l2 = feats[:, names.index("feature_l2_norm")]
    deciles = np.quantile(l2, np.linspace(0, 1, 11))
    print(f"   {'decile':>6}  {'l2_range':>22}  {'n':>5}  "
          f"{'fgsm_flip':>10}  {'pgd_flip':>9}  {'mean_min_eps':>13}  {'mean_margin':>12}")
    margin_np = feats[:, margin_idx]
    for k in range(10):
        lo_q, hi_q = deciles[k], deciles[k+1]
        if k < 9:
            mask = (l2 >= lo_q) & (l2 < hi_q)
        else:
            mask = (l2 >= lo_q) & (l2 <= hi_q)
        n_k = mask.sum()
        if n_k == 0:
            continue
        print(f"   {k+1:>6}  [{lo_q:>8.3f},{hi_q:>8.3f}]  {n_k:>5}  "
              f"{flip_fgsm[mask].mean():>10.3f}  {flip_pgd[mask].mean():>9.3f}  "
              f"{min_eps[mask].mean():>13.4f}  {margin_np[mask].mean():>12.4f}")
    print("   (If H24 holds: flip-rate should DECREASE and mean_min_eps should "
          "INCREASE with feature_l2_norm decile.)")

    print("\nDone.")


if __name__ == "__main__":
    main()
