"""
Hypothesis H07: Per-sample local Lipschitz constant estimate predicts adversarial vulnerability.

L(x) approximates how much the model's logits change under small input perturbations:
    L_max(x) = max_{delta in B_inf(0, r)} || f(x + delta) - f(x) ||_2
    L_mean(x) = mean of the same quantity over K random samples
We also compute the input-gradient L2 norm of the cross-entropy loss as a cheap proxy.

High Lipschitz <=> sensitive logits <=> conjectured adversarial vulnerability.

Baselines: victim_margin, mean_pix, std_pix, sobel_mean.
Targets:   flipped_FGSM (eps=15/255), flipped_PGD, FGSM_min_eps_binary_search (continuous).

Self-contained: trains a small CNN on Fashion-MNIST (10 epochs), then evaluates
univariate AUROC for each feature and a multivariate ablation testing whether
Lipschitz adds predictive value over victim_margin.

NOTE: cwd-independent, writes nothing to disk besides downloading FashionMNIST
into /tmp/data.
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
LIP_RADIUS = 0.05
LIP_K = 50
SEED = 0


# ---------------------------------------------------------------------------
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


def fgsm_attack(model, x, y, eps=EPS_TEST):
    x_adv = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_adv), y).backward()
    sign = x_adv.grad.sign().detach()
    return (x + eps * sign).clamp(0, 1).detach()


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    # random start
    delta = torch.empty_like(x0).uniform_(-eps, eps)
    x_adv = (x0 + delta).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)
    return x_adv.detach()


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
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
    return hi  # smallest eps that flipped within budget


def batched_attack_flag(model, x, y, attack_fn, bs=256):
    flags = []
    for i in range(0, x.size(0), bs):
        adv = attack_fn(model, x[i:i+bs], y[i:i+bs])
        with torch.no_grad():
            flags.append(model(adv).argmax(1) != y[i:i+bs])
    return torch.cat(flags, 0)


# ---------------------------------------------------------------------------
def local_lipschitz(model, x, K=LIP_K, r=LIP_RADIUS, bs=64):
    """For each sample, sample K uniform perturbations in B_inf(0, r);
    return (L_max, L_mean) of || logits(x + delta) - logits(x) ||_2."""
    N = x.size(0)
    L_max = torch.zeros(N, device=DEVICE)
    L_mean = torch.zeros(N, device=DEVICE)
    model.eval()
    with torch.no_grad():
        for i in range(0, N, bs):
            xb = x[i:i+bs]
            B = xb.size(0)
            base = model(xb)                              # (B, C)
            # expand to (K*B, ...) — process in chunks of K to limit memory
            max_acc = torch.full((B,), -1.0, device=DEVICE)
            sum_acc = torch.zeros(B, device=DEVICE)
            for _ in range(K):
                delta = torch.empty_like(xb).uniform_(-r, r)
                xp = (xb + delta).clamp(0, 1)
                lp = model(xp)
                d = (lp - base).norm(dim=1)  # (B,)
                max_acc = torch.maximum(max_acc, d)
                sum_acc = sum_acc + d
            L_max[i:i+bs] = max_acc
            L_mean[i:i+bs] = sum_acc / K
    return L_max, L_mean


def input_grad_norm(model, x, y, bs=256):
    """|| grad_x CE-loss(model(x), y) ||_2 per sample."""
    N = x.size(0)
    out = torch.zeros(N, device=DEVICE)
    for i in range(0, N, bs):
        xb = x[i:i+bs].clone().detach().requires_grad_(True)
        yb = y[i:i+bs]
        # per-sample loss
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        g = torch.autograd.grad(loss, xb)[0]
        out[i:i+bs] = g.flatten(1).norm(dim=1)
    return out.detach()


def sobel_mean(x):
    """Mean absolute Sobel response per sample (edge density)."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return (gx.abs() + gy.abs()).mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
def victim_margin(model, x):
    with torch.no_grad():
        logits = batched_logits(model, x)
    s, _ = logits.sort(1, descending=True)
    return (s[:, 0] - s[:, 1])


# ---------------------------------------------------------------------------
def auroc_both_dirs(y, score):
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    tf = transforms.ToTensor()
    print("Loading Fashion-MNIST ...")
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print(f"Training victim CNN ({EPOCHS} epochs) on {DEVICE} ...")
    model = train_victim(train_set)

    # stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # restrict to correctly-classified samples (vulnerability only defined here)
    with torch.no_grad():
        pred = batched_logits(model, test_x).argmax(1)
    correct = pred == test_y
    print(f"Test acc: {correct.float().mean().item():.4f}")
    x = test_x[correct]
    y = test_y[correct]
    N = x.size(0)
    print(f"Using {N} correctly-classified samples.")

    # -------- features --------
    print("Computing victim_margin ...")
    margin = victim_margin(model, x)

    print(f"Computing local Lipschitz (K={LIP_K}, r={LIP_RADIUS}) ...")
    L_max, L_mean = local_lipschitz(model, x)

    print("Computing input-gradient L2 norm ...")
    grad_norm = input_grad_norm(model, x, y)

    print("Computing baseline pixel stats + sobel ...")
    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))
    sobel = sobel_mean(x)

    feats = torch.stack([margin, L_max, L_mean, grad_norm,
                         mean_pix, std_pix, sobel], 1).cpu().numpy()
    names = ["victim_margin", "lipschitz_max", "lipschitz_mean",
             "input_grad_norm", "mean_pix", "std_pix", "sobel_mean"]

    # -------- targets --------
    print("Running FGSM attack ...")
    flip_fgsm = batched_attack_flag(model, x, y, fgsm_attack).cpu().numpy().astype(int)

    print("Running PGD attack ...")
    flip_pgd = batched_attack_flag(model, x, y, pgd_attack).cpu().numpy().astype(int)

    print("Running min-eps FGSM binary search ...")
    min_eps = []
    bs = 256
    for i in range(0, N, bs):
        min_eps.append(min_eps_fgsm(model, x[i:i+bs], y[i:i+bs]))
    min_eps = torch.cat(min_eps, 0).cpu().numpy()

    # -------- evaluation --------
    print("\n========== H07: local Lipschitz vs adversarial vulnerability ==========")
    print(f"N = {N}  FGSM-flip rate = {flip_fgsm.mean():.3f}  PGD-flip rate = {flip_pgd.mean():.3f}")
    print(f"min_eps FGSM:  mean={min_eps.mean():.4f}  median={np.median(min_eps):.4f}")

    bin_targets = [("flipped_FGSM", flip_fgsm), ("flipped_PGD", flip_pgd)]

    for t_name, yv in bin_targets:
        if yv.std() == 0:
            print(f"\n--- target {t_name}: no variance, skipping ---")
            continue
        print(f"\n--- target: {t_name}  (positive rate = {yv.mean():.3f}) ---")
        for i, n in enumerate(names):
            a = auroc_both_dirs(yv, feats[:, i])
            print(f"   univariate AUROC  {n:<18} {a:.4f}")

        # multivariate: margin alone vs margin+Lipschitz features
        Xs = StandardScaler().fit_transform(feats)
        lr_full = LogisticRegression(max_iter=4000).fit(Xs, yv)
        auc_full = roc_auc_score(yv, lr_full.predict_proba(Xs)[:, 1])

        margin_idx = names.index("victim_margin")
        Xs_margin = Xs[:, [margin_idx]]
        lr_m = LogisticRegression(max_iter=4000).fit(Xs_margin, yv)
        auc_m = roc_auc_score(yv, lr_m.predict_proba(Xs_margin)[:, 1])

        lip_idxs = [names.index(n) for n in
                    ["lipschitz_max", "lipschitz_mean", "input_grad_norm"]]
        Xs_ml = Xs[:, [margin_idx] + lip_idxs]
        lr_ml = LogisticRegression(max_iter=4000).fit(Xs_ml, yv)
        auc_ml = roc_auc_score(yv, lr_ml.predict_proba(Xs_ml)[:, 1])

        # ablation: full minus Lipschitz family
        keep = [i for i in range(len(names)) if i not in lip_idxs]
        lr_noL = LogisticRegression(max_iter=4000).fit(Xs[:, keep], yv)
        auc_noL = roc_auc_score(yv, lr_noL.predict_proba(Xs[:, keep])[:, 1])

        print(f"   multivariate AUROC (margin only):                 {auc_m:.4f}")
        print(f"   multivariate AUROC (margin + Lipschitz triple):   {auc_ml:.4f}")
        print(f"     Delta over margin-alone:                          {auc_ml - auc_m:+.4f}")
        print(f"   multivariate AUROC (all 7 features):              {auc_full:.4f}")
        print(f"   multivariate AUROC (all minus Lipschitz triple):  {auc_noL:.4f}")
        print(f"     Delta from removing Lipschitz triple:             {auc_full - auc_noL:+.4f}")
        print(f"   standardised coefficients (full model):")
        for n, c in zip(names, lr_full.coef_.flatten()):
            print(f"     {n:<18} {c:+.4f}")

    # continuous target: min_eps_FGSM (lower = more vulnerable)
    print(f"\n--- continuous target: FGSM_min_eps (Spearman/OLS) ---")
    # univariate Spearman via rank correlation
    from scipy.stats import spearmanr
    for i, n in enumerate(names):
        r, p = spearmanr(feats[:, i], min_eps)
        print(f"   spearman  {n:<18} rho={r:+.4f}  p={p:.2e}")

    Xs = StandardScaler().fit_transform(feats)
    ols_full = LinearRegression().fit(Xs, min_eps)
    r2_full = ols_full.score(Xs, min_eps)
    margin_idx = names.index("victim_margin")
    ols_m = LinearRegression().fit(Xs[:, [margin_idx]], min_eps)
    r2_m = ols_m.score(Xs[:, [margin_idx]], min_eps)
    lip_idxs = [names.index(n) for n in
                ["lipschitz_max", "lipschitz_mean", "input_grad_norm"]]
    ols_ml = LinearRegression().fit(Xs[:, [margin_idx] + lip_idxs], min_eps)
    r2_ml = ols_ml.score(Xs[:, [margin_idx] + lip_idxs], min_eps)
    print(f"   OLS R^2 (margin only):              {r2_m:.4f}")
    print(f"   OLS R^2 (margin + Lipschitz triple):{r2_ml:.4f}")
    print(f"     Delta:                              {r2_ml - r2_m:+.4f}")
    print(f"   OLS R^2 (all 7 features):           {r2_full:.4f}")
    print(f"   OLS coefficients (full):")
    for n, c in zip(names, ols_full.coef_.flatten()):
        print(f"     {n:<18} {c:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
