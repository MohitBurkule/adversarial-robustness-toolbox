"""
H17: Per-sample consistency under additive Gaussian noise is a strong vulnerability
predictor — equivalent to randomized smoothing's certified-radius proxy at a single
noise level.

Pipeline:
  1. Train small CNN on Fashion-MNIST (10 epochs, Adam) — see diagnostic_test.py.
  2. For each test sample, draw K=32 i.i.d. Gaussian-noise copies at sigma in
     {0.1, 0.25}. Compute:
        - vote_agreement: fraction whose argmax matches the clean argmax
        - softmax_l2_variance: mean L2 distance from clean softmax to noisy softmax
  3. Baselines: victim_margin, mean_pix, std_pix.
  4. Targets:
        - flipped_FGSM (eps = 15/255), flipped untargeted -> binary flip
        - flipped_PGD  (eps = 15/255, 20 steps, alpha = 2/255)
        - FGSM_min_eps_binary_search (continuous; cont. target -> rank-AUROC via -min_eps)
  5. Univariate AUROC per feature (also Spearman for the continuous min_eps target).
  6. Multivariate logistic regression: does noise consistency add over margin?
     Compare AUROC of (margin) vs (margin + noise consistency features) per sigma.
  7. Compare sigma=0.1 vs sigma=0.25 predictive signal.

Refs (background, no internet calls performed here):
  - Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing",
    ICML 2019.
  - Bahat & Shakhnarovich, "Classification confidence estimation with test-time
    data-augmentation", 2018 (related notion of consistency under perturbations).

Run: python h17_noise_consistency.py
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
K_NOISE = 32
SIGMAS = [0.1, 0.25]
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
    margin = (p.sort(1, descending=True)[0][:, 0] - p.sort(1, descending=True)[0][:, 1])
    return p, p.argmax(1), margin


def fgsm_grad_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


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
def noise_consistency(model, x, clean_softmax, clean_argmax, sigma, K=K_NOISE, bs=64):
    """Return (vote_agreement, softmax_l2_variance) tensors of shape [N]."""
    N = x.size(0)
    agree = torch.zeros(N, device=DEVICE)
    l2 = torch.zeros(N, device=DEVICE)
    for i in range(0, N, bs):
        xb = x[i:i+bs]
        cs = clean_softmax[i:i+bs]
        ca = clean_argmax[i:i+bs]
        # expand to [K, b, 1, 28, 28]
        b = xb.size(0)
        xx = xb.unsqueeze(0).expand(K, b, *xb.shape[1:])
        noise = torch.randn_like(xx) * sigma
        xn = (xx + noise).clamp(0, 1).reshape(K * b, *xb.shape[1:])
        p = F.softmax(model(xn), dim=1).reshape(K, b, -1)
        preds = p.argmax(2)                          # [K, b]
        agree[i:i+b] = (preds == ca.unsqueeze(0)).float().mean(0)
        # L2 distance per noisy copy from clean softmax, then averaged across K
        diff = p - cs.unsqueeze(0)
        l2[i:i+b] = diff.norm(dim=2).mean(0)
    return agree, l2


def auroc_safe(y, score):
    """AUROC, taking max(direction, 1-direction) so we report magnitude of signal."""
    if y.std() == 0:
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

    # restrict to correctly classified samples (those are the ones at risk of flipping)
    correct = (pred == test_y)
    print(f"  clean accuracy = {correct.float().mean().item():.4f}")
    x = test_x[correct]
    y = test_y[correct]
    soft_c = soft[correct]
    pred_c = pred[correct]
    margin_c = margin[correct]

    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))

    # --- noise consistency features per sigma ---
    feats_by_sigma = {}
    for sigma in SIGMAS:
        print(f"Noise consistency at sigma={sigma} (K={K_NOISE}) ...")
        t0 = time.time()
        torch.manual_seed(SEED + int(sigma * 1000))
        agree, l2 = noise_consistency(model, x, soft_c, pred_c, sigma)
        print(f"  done in {time.time()-t0:.1f}s  "
              f"agree mean={agree.mean().item():.3f}  "
              f"l2 mean={l2.mean().item():.3f}")
        feats_by_sigma[sigma] = {"vote_agreement": agree, "softmax_l2_variance": l2}

    # --- targets ---
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

    # numpy versions
    margin_np = margin_c.cpu().numpy()
    mean_pix_np = mean_pix.cpu().numpy()
    std_pix_np = std_pix.cpu().numpy()
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

    # ---- univariate AUROC ----
    print("\n========== UNIVARIATE AUROC ==========")
    baseline_feats = {
        "victim_margin": margin_np,
        "mean_pix": mean_pix_np,
        "std_pix": std_pix_np,
    }
    all_feats_per_sigma = {}
    for sigma in SIGMAS:
        feats = dict(baseline_feats)
        feats[f"vote_agreement@s{sigma}"] = feats_by_sigma[sigma]["vote_agreement"].cpu().numpy()
        feats[f"softmax_l2_var@s{sigma}"] = feats_by_sigma[sigma]["softmax_l2_variance"].cpu().numpy()
        all_feats_per_sigma[sigma] = feats

    for sigma in SIGMAS:
        print(f"\n--- sigma = {sigma} ---")
        feats = all_feats_per_sigma[sigma]
        for tname, t in targets.items():
            print(f" target: {tname}  (pos rate = {t.mean():.3f})")
            for fname, fv in feats.items():
                a = auroc_safe(t, fv)
                print(f"   {fname:<28} AUROC = {a:.4f}")
        # continuous target: rank AUROC <-> Spearman with min_eps
        print(f" target: FGSM_min_eps  (Spearman)")
        for fname, fv in feats.items():
            rho, _ = spearmanr(fv, min_eps_np)
            print(f"   {fname:<28} Spearman = {rho:+.4f}")

    # ---- multivariate: does noise consistency add over margin? ----
    print("\n========== MULTIVARIATE: noise consistency over margin ==========")
    summary = []
    for sigma in SIGMAS:
        print(f"\n--- sigma = {sigma} ---")
        agree_np = feats_by_sigma[sigma]["vote_agreement"].cpu().numpy()
        l2_np = feats_by_sigma[sigma]["softmax_l2_variance"].cpu().numpy()
        X_margin = margin_np.reshape(-1, 1)
        X_full = np.stack([margin_np, agree_np, l2_np], axis=1)
        Xs_m = StandardScaler().fit_transform(X_margin)
        Xs_f = StandardScaler().fit_transform(X_full)
        for tname, t in targets.items():
            if t.std() == 0:
                continue
            lr_m = LogisticRegression(max_iter=2000).fit(Xs_m, t)
            lr_f = LogisticRegression(max_iter=2000).fit(Xs_f, t)
            auc_m = roc_auc_score(t, lr_m.predict_proba(Xs_m)[:, 1])
            auc_f = roc_auc_score(t, lr_f.predict_proba(Xs_f)[:, 1])
            print(f" {tname}: margin-only AUROC = {auc_m:.4f}  "
                  f"margin+noise AUROC = {auc_f:.4f}  delta = {auc_f-auc_m:+.4f}")
            print(f"   coefs (margin, vote_agree, softmax_l2): "
                  f"{lr_f.coef_.flatten().tolist()}")
            summary.append((sigma, tname, auc_m, auc_f, auc_f - auc_m))

    # ---- compare sigmas head-to-head ----
    print("\n========== SIGMA COMPARISON (univariate vote_agreement AUROC) ==========")
    for tname, t in targets.items():
        row = []
        for sigma in SIGMAS:
            a = auroc_safe(t, feats_by_sigma[sigma]["vote_agreement"].cpu().numpy())
            row.append((sigma, a))
        best = max(row, key=lambda r: r[1])
        print(f" {tname}:  " +
              "  ".join(f"sigma={s} AUROC={a:.4f}" for s, a in row) +
              f"   ==>  best sigma = {best[0]}")

    print("\n===== SUMMARY (multivariate) =====")
    print(f"{'sigma':<7} {'target':<16} {'margin':>8} {'+noise':>8} {'delta':>8}")
    for s, t, am, af, d in summary:
        print(f"{s:<7} {t:<16} {am:>8.4f} {af:>8.4f} {d:>+8.4f}")


if __name__ == "__main__":
    main()
