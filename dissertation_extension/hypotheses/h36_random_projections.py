"""
H36: Random L_inf-projection of pixels combined with sign of the victim's input
gradient (i.e. how does loss change in K random directions of the input ball)
predicts vulnerability.

Idea: instead of computing the full input-Hessian or the Jacobian spectral norm,
sample K random unit-L_inf directions v (Rademacher signs in {-1, +1}^d), and
measure the per-sample loss change

    delta_k = L(x + eps * v_k, y) - L(x, y)

at a small eps. We then build cheap summary statistics over k = 1..K:

    rp_mean   = mean_k(delta_k)          # signed: positive = loss increases on avg
    rp_max    = max_k(delta_k)           # worst-case random-direction loss bump
    rp_std    = std_k(delta_k)           # spread / curvature-ish proxy
    rp_pos    = mean_k(1[delta_k > 0])   # fraction of directions that increase loss

These are zeroth-order proxies for input-space Lipschitz / Hessian behaviour.

Pipeline:
  1. Train a small CNN on Fashion-MNIST (10 epochs, Adam), matching diagnostic_test.py.
  2. Per (correctly classified) test sample, compute the four RP features with K=20,
     eps=0.05, Rademacher directions.
  3. Baselines: victim_margin, mean_pix, std_pix, input_grad_L2_norm
                (||grad_x L(x, y)||_2).
  4. Targets:
        - flipped_FGSM at eps = 15/255
        - flipped_PGD  at eps = 15/255, 20 steps, alpha = 2/255
        - FGSM_min_eps_binary_search (continuous; we report Spearman + rank-AUROC
          via -min_eps)
  5. Univariate AUROC per feature (Spearman for the continuous target).
  6. Multivariate logistic regression: does the random-projection feature set
     ADD over (input_grad_L2_norm + margin)? Compare AUROC of:
        (a) margin only
        (b) margin + input_grad_L2_norm
        (c) margin + input_grad_L2_norm + RP features

Refs (background, no internet calls performed):
  - Weng et al., "Evaluating the Robustness of Neural Networks: An Extreme Value
    Theory Approach" (CLEVER), 2018 -- random-direction Lipschitz proxy.
  - Wong & Kolter, randomized robustness certificates.
  - Cohen et al., randomized smoothing.

Run: python h36_random_projections.py
Data cached under /tmp/data
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0

# Random-projection probe settings
K_DIRS = 20
RP_EPS = 0.05  # L_inf radius for probing
SEED = 0


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


def train_model(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done in {time.time()-t0:.1f}s")
    model.eval()
    return model


def batched(fn, x, bs=256, **kw):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(fn(x[i:i+bs], **kw))
    return torch.cat(out, 0)


@torch.no_grad()
def softmax_logits(model, x):
    return F.softmax(model(x), dim=1)


@torch.no_grad()
def clean_predictions(model, x):
    p = batched(lambda b: softmax_logits(model, b), x)
    s = p.sort(1, descending=True)[0]
    margin = s[:, 0] - s[:, 1]
    return p, p.argmax(1), margin


def fgsm_grad_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def input_grad(model, x, y):
    """Return raw d L / d x per-sample (same shape as x)."""
    x = x.clone().detach().requires_grad_(True)
    # per-sample loss: sum is fine because grads are independent across batch
    logits = model(x)
    loss = F.cross_entropy(logits, y, reduction="sum")
    g = torch.autograd.grad(loss, x)[0]
    return g.detach()


def fgsm_attack(model, x, y, eps=EPS_TEST):
    s = fgsm_grad_sign(model, x, y)
    return (x + eps * s).clamp(0, 1)


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        g = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * g.sign()
        adv = torch.min(torch.max(adv, x0 - eps), x0 + eps).clamp(0, 1)
    return adv.detach()


def min_eps_fgsm_bsearch(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad_sign(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


@torch.no_grad()
def per_sample_ce(model, x, y):
    """Per-sample cross-entropy loss, shape [N]."""
    return F.cross_entropy(model(x), y, reduction="none")


def random_projection_features(model, x, y, K=K_DIRS, eps=RP_EPS, bs=64):
    """
    For each sample, draw K Rademacher directions v in {-1, +1}^d (same shape as x),
    compute delta_k = L(clip(x + eps * v_k), y) - L(x, y).
    Return rp_mean, rp_max, rp_std, rp_pos_frac each of shape [N].

    Inputs are clamped to [0, 1] after perturbation (matches the threat model used
    elsewhere). This means at pixels near 0 or 1 the effective step is reduced; this
    is intentional and consistent with how FGSM/PGD are evaluated above.
    """
    N = x.size(0)
    rp_mean = torch.zeros(N, device=DEVICE)
    rp_max = torch.full((N,), -float("inf"), device=DEVICE)
    rp_sumsq = torch.zeros(N, device=DEVICE)
    rp_pos = torch.zeros(N, device=DEVICE)

    # iterate over directions outside (so we keep memory low) and batches inside
    for k in range(K):
        # one direction per sample for this k (independent across N as well as k)
        for i in range(0, N, bs):
            xb = x[i:i+bs]
            yb = y[i:i+bs]
            # Rademacher direction, fresh per (k, sample)
            v = torch.empty_like(xb).bernoulli_(0.5).mul_(2).sub_(1)
            xp = (xb + eps * v).clamp(0, 1)
            with torch.no_grad():
                loss_pert = F.cross_entropy(model(xp), yb, reduction="none")
                loss_base = F.cross_entropy(model(xb), yb, reduction="none")
            delta = loss_pert - loss_base                                    # [b]
            rp_mean[i:i+xb.size(0)] += delta
            rp_max[i:i+xb.size(0)] = torch.maximum(rp_max[i:i+xb.size(0)], delta)
            rp_sumsq[i:i+xb.size(0)] += delta * delta
            rp_pos[i:i+xb.size(0)] += (delta > 0).float()

    rp_mean = rp_mean / K
    rp_var = rp_sumsq / K - rp_mean * rp_mean
    rp_std = rp_var.clamp_min(0).sqrt()
    rp_pos = rp_pos / K
    return rp_mean, rp_max, rp_std, rp_pos


def auroc_safe(y, score):
    if np.std(y) == 0:
        return float("nan")
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def main():
    print("device:", DEVICE)
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("Training victim CNN ...")
    model = train_model(train_set)

    print("Clean predictions ...")
    soft, pred, margin = clean_predictions(model, test_x)
    correct = (pred == test_y)
    print(f"  clean accuracy = {correct.float().mean().item():.4f}")
    x = test_x[correct]
    y = test_y[correct]
    margin_c = margin[correct]

    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))

    # ---- input gradient L2 norm baseline ----
    print("Input-gradient L2 norm ...")
    grad_l2 = []
    for i in range(0, x.size(0), 128):
        g = input_grad(model, x[i:i+128], y[i:i+128])
        grad_l2.append(g.flatten(1).norm(dim=1))
    grad_l2 = torch.cat(grad_l2)

    # ---- random projection features ----
    print(f"Random-projection probe  K={K_DIRS}  eps={RP_EPS}  (Rademacher) ...")
    torch.manual_seed(SEED + 1234)
    t0 = time.time()
    rp_mean, rp_max, rp_std, rp_pos = random_projection_features(model, x, y)
    print(f"  done in {time.time()-t0:.1f}s   "
          f"rp_mean mean={rp_mean.mean().item():+.4f}   "
          f"rp_max mean={rp_max.mean().item():+.4f}   "
          f"rp_std mean={rp_std.mean().item():.4f}   "
          f"rp_pos mean={rp_pos.mean().item():.3f}")

    # ---- targets ----
    print("FGSM @ eps=15/255 ...")
    flip_fgsm = []
    for i in range(0, x.size(0), 256):
        xb, yb = x[i:i+256], y[i:i+256]
        adv = fgsm_attack(model, xb, yb)
        with torch.no_grad():
            flip_fgsm.append(model(adv).argmax(1) != yb)
    flip_fgsm = torch.cat(flip_fgsm)

    print("PGD ...")
    flip_pgd = []
    for i in range(0, x.size(0), 256):
        xb, yb = x[i:i+256], y[i:i+256]
        adv = pgd_attack(model, xb, yb)
        with torch.no_grad():
            flip_pgd.append(model(adv).argmax(1) != yb)
    flip_pgd = torch.cat(flip_pgd)

    print("FGSM min-eps binary search ...")
    me = []
    for i in range(0, x.size(0), 256):
        me.append(min_eps_fgsm_bsearch(model, x[i:i+256], y[i:i+256]))
    min_eps = torch.cat(me)

    # numpy
    margin_np = margin_c.cpu().numpy()
    mean_pix_np = mean_pix.cpu().numpy()
    std_pix_np = std_pix.cpu().numpy()
    grad_l2_np = grad_l2.cpu().numpy()
    rp_mean_np = rp_mean.cpu().numpy()
    rp_max_np = rp_max.cpu().numpy()
    rp_std_np = rp_std.cpu().numpy()
    rp_pos_np = rp_pos.cpu().numpy()
    flip_fgsm_np = flip_fgsm.cpu().numpy().astype(int)
    flip_pgd_np = flip_pgd.cpu().numpy().astype(int)
    min_eps_np = min_eps.cpu().numpy()

    print(f"\nN={len(margin_np)}  "
          f"FGSM flip rate={flip_fgsm_np.mean():.3f}  "
          f"PGD flip rate={flip_pgd_np.mean():.3f}  "
          f"mean min_eps={min_eps_np.mean():.4f}")

    targets = {
        "flipped_FGSM": flip_fgsm_np,
        "flipped_PGD": flip_pgd_np,
    }
    feats = {
        # baselines
        "victim_margin":       margin_np,
        "mean_pix":            mean_pix_np,
        "std_pix":             std_pix_np,
        "input_grad_L2_norm":  grad_l2_np,
        # random-projection features
        "rp_mean":             rp_mean_np,
        "rp_max":              rp_max_np,
        "rp_std":              rp_std_np,
        "rp_pos_frac":         rp_pos_np,
    }

    # ---- univariate AUROC ----
    print("\n========== UNIVARIATE AUROC ==========")
    for tname, t in targets.items():
        print(f"\n target: {tname}  (pos rate = {t.mean():.3f})")
        for fname, fv in feats.items():
            a = auroc_safe(t, fv)
            print(f"   {fname:<22} AUROC = {a:.4f}")

    print("\n target: FGSM_min_eps  (Spearman with feature; -min_eps for AUROC-style sign)")
    for fname, fv in feats.items():
        rho, _ = spearmanr(fv, min_eps_np)
        print(f"   {fname:<22} Spearman = {rho:+.4f}")

    # ---- multivariate: does RP add over (margin + input_grad_L2_norm)? ----
    print("\n========== MULTIVARIATE: does RP add over (margin + input_grad_L2_norm)? ==========")
    X_m = margin_np.reshape(-1, 1)
    X_mg = np.stack([margin_np, grad_l2_np], axis=1)
    X_all = np.stack(
        [margin_np, grad_l2_np, rp_mean_np, rp_max_np, rp_std_np, rp_pos_np], axis=1
    )
    Xs_m = StandardScaler().fit_transform(X_m)
    Xs_mg = StandardScaler().fit_transform(X_mg)
    Xs_all = StandardScaler().fit_transform(X_all)

    summary = []
    for tname, t in targets.items():
        if t.std() == 0:
            continue
        lr_m  = LogisticRegression(max_iter=2000).fit(Xs_m,  t)
        lr_mg = LogisticRegression(max_iter=2000).fit(Xs_mg, t)
        lr_a  = LogisticRegression(max_iter=2000).fit(Xs_all, t)
        auc_m  = roc_auc_score(t, lr_m.predict_proba(Xs_m)[:, 1])
        auc_mg = roc_auc_score(t, lr_mg.predict_proba(Xs_mg)[:, 1])
        auc_a  = roc_auc_score(t, lr_a.predict_proba(Xs_all)[:, 1])
        print(f"\n {tname}:")
        print(f"   margin                  AUROC = {auc_m:.4f}")
        print(f"   margin + grad_L2        AUROC = {auc_mg:.4f}  "
              f"(delta vs margin      = {auc_mg-auc_m:+.4f})")
        print(f"   margin + grad_L2 + RP   AUROC = {auc_a:.4f}  "
              f"(delta vs margin+grad = {auc_a-auc_mg:+.4f})")
        print(f"   coefs (margin, grad_L2, rp_mean, rp_max, rp_std, rp_pos): "
              f"{[round(c, 3) for c in lr_a.coef_.flatten().tolist()]}")
        summary.append((tname, auc_m, auc_mg, auc_a, auc_a - auc_mg))

    print("\n===== SUMMARY =====")
    print(f"{'target':<16} {'margin':>8} {'+grad':>8} {'+grad+RP':>10} "
          f"{'delta RP':>10}")
    for tname, am, amg, aa, d in summary:
        print(f"{tname:<16} {am:>8.4f} {amg:>8.4f} {aa:>10.4f} {d:>+10.4f}")

    print("\nCaveats:")
    print(" - RP features use clamped perturbations; on near-saturated pixels the "
          "effective step is reduced (consistent with FGSM/PGD threat model here).")
    print(" - Rademacher directions are L_inf-unit (||v||_inf = 1); they are NOT "
          "L2-unit, so rp_* are NOT directly comparable to grad_L2 in magnitude.")
    print(" - K=20 is small; rp_mean/rp_std have noticeable Monte-Carlo variance, "
          "especially rp_pos_frac whose binomial SE is ~ sqrt(p(1-p)/K) ~ 0.11.")
    print(" - eps=0.05 sits between the FGSM eval eps (~0.059) and typical min_eps "
          "values, so RP probes a regime close to the eval threat model; results "
          "for other eps may differ.")
    print(" - Univariate AUROC uses max(a, 1-a); the multivariate logistic uses "
          "the signed feature, so direction (sign of coef) is informative.")
    print(" - Restricted to correctly-classified samples; targets are binary "
          "label flips (untargeted).")


if __name__ == "__main__":
    main()
