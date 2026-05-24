"""
H77: Cohen et al. (2019) randomized smoothing certified L2 radius predicts
adversarial vulnerability on Fashion-MNIST.

Hypothesis
----------
For a sample x correctly classified by a standardly trained CNN, the
Cohen-certified L2 radius of the *smoothed* version of that classifier
(estimated with K=100 Monte-Carlo noisy forward passes at sigma=0.25 and a
Clopper-Pearson lower confidence bound on the majority-class probability) is
monotonically related to adversarial vulnerability: large certified radius
=> harder to attack with FGSM/PGD, larger min-eps to flip.

Pipeline
--------
  1. Train a small CNN matching dissertation_extension/diagnostic_test.py
     (same architecture) on Fashion-MNIST for 10 epochs (standard training,
     NOT noise augmentation - Cohen's certificate still applies but the
     radii will be small; that is fine for ranking purposes).
  2. For each test sample compute:
        - margin            (final-model logit margin, baseline)
        - mean_pix          (input mean pixel intensity)
        - std_pix           (input std pixel intensity)
        - certified_radius  (Cohen 2019 with K=100, sigma=0.25, alpha=0.001)
  3. Compute three vulnerability targets per (correctly classified) sample:
        - FGSM flip at eps = 15/255
        - PGD-20 flip at eps = 15/255
        - min_eps_FGSM via per-sample binary search
  4. Report univariate AUROC of each feature against each binary target,
     and Pearson correlation against the continuous min_eps target.

Cohen certified radius
----------------------
For smoothed classifier g(x) = argmax_c P_{eta~N(0, sigma^2 I)}[f(x+eta)=c],
the certificate is

    R(x) = sigma * Phi^{-1}( p_lower_A )

where p_lower_A is a Clopper-Pearson (1-alpha) lower confidence bound on
P[f(x+eta) = top-class]. If p_lower_A <= 0.5, no positive radius is
certified and we report 0. We use the standard Cohen "predict" -> "certify"
shortcut with a single noise batch of size K to keep this tractable: pick
the top class by Monte-Carlo majority vote, then Clopper-Pearson lower bound
on its count.

This file writes code only; it is NOT executed here.

References
----------
  - Cohen, Rosenfeld, Kolter. "Certified Adversarial Robustness via
    Randomized Smoothing", ICML 2019. https://arxiv.org/abs/1902.02918
  - Reference PyTorch implementation: https://github.com/locuslab/smoothing
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta, norm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = EPS_TEST / 8.0
SIGMA = 0.25
K_SMOOTH = 100        # Monte Carlo samples for Cohen certification
ALPHA_CP = 1e-3       # Clopper-Pearson confidence level
SEED = 0


# ---------------------------------------------------------------------------
# model (matches diagnostic_test.py)
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
# training
# ---------------------------------------------------------------------------
def train_model(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=2)
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
        print(f"  epoch {ep+1}/{EPOCHS} done in {time.time()-t0:.1f}s")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# attacks
# ---------------------------------------------------------------------------
def fgsm_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign = fgsm_sign(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """L_inf PGD with uniform-random start."""
    x0 = x.clone().detach()
    delta = (torch.rand_like(x0) * 2 - 1) * eps
    adv = (x0 + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps)
        adv = adv.clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    """Binary search smallest L_inf eps that flips FGSM (sign is eps-independent)."""
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_sign(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# ---------------------------------------------------------------------------
# Cohen randomized smoothing certified radius
# ---------------------------------------------------------------------------
def clopper_pearson_lower(k, n, alpha):
    """One-sided Clopper-Pearson lower bound on a binomial proportion.

    Returns the largest p such that P[Bin(n,p) >= k] <= alpha, equivalently
    the (alpha) quantile of Beta(k, n-k+1). When k == 0 the lower bound is 0.
    """
    if k == 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))


def cohen_certified_radius(model, x, sigma=SIGMA, K=K_SMOOTH, alpha=ALPHA_CP,
                           n_classes=10, mc_batch=100):
    """Compute Cohen-style L2 certified radius per sample.

    For each x_i, draw K i.i.d. samples eta ~ N(0, sigma^2 I), classify
    f(x_i + eta), take the most-frequent class c_hat, then certify radius
        R = sigma * Phi^{-1}( p_lower )
    where p_lower = ClopperPearson_lower(count_of_c_hat, K, alpha).
    If p_lower <= 0.5 we return 0.0 (no positive certificate).
    """
    n = x.size(0)
    radii = torch.zeros(n, device=DEVICE)
    top_class = torch.zeros(n, dtype=torch.long, device=DEVICE)
    model.eval()
    with torch.no_grad():
        for i in range(n):
            xi = x[i:i+1]
            counts = torch.zeros(n_classes, device=DEVICE)
            remaining = K
            while remaining > 0:
                bs = min(mc_batch, remaining)
                noise = torch.randn(bs, *xi.shape[1:], device=DEVICE) * sigma
                noisy = (xi.expand(bs, -1, -1, -1) + noise)
                # NB: Cohen does NOT clamp noisy inputs (the certificate is
                # over real-valued perturbations of x, not pixel-valid x).
                preds = model(noisy).argmax(1)
                counts += torch.bincount(preds, minlength=n_classes).float()
                remaining -= bs
            c_hat = int(counts.argmax().item())
            k = int(counts[c_hat].item())
            p_lower = clopper_pearson_lower(k, K, alpha)
            top_class[i] = c_hat
            if p_lower > 0.5:
                radii[i] = sigma * float(norm.ppf(p_lower))
            else:
                radii[i] = 0.0
    return radii, top_class


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def compute_basic_features(model, x, y, n_classes=10):
    """Returns final_margin (length n), mean_pix, std_pix, final_pred."""
    n = x.size(0)
    logits_all = []
    with torch.no_grad():
        for i in range(0, n, 512):
            logits_all.append(model(x[i:i+512]))
    logits = torch.cat(logits_all, 0)
    sorted_logits, _ = logits.sort(1, descending=True)
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]
    final_pred = logits.argmax(1)
    mean_pix = x.view(n, -1).mean(1)
    std_pix = x.view(n, -1).std(1)
    return margin, mean_pix, std_pix, final_pred


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def univariate_auroc(feats_np, names, target, target_name):
    print(f"\n--- target: {target_name}   (positive rate = {target.mean():.3f}) ---")
    for i, n in enumerate(names):
        try:
            a = roc_auc_score(target, feats_np[:, i])
        except ValueError:
            print(f"   {n:<20} AUROC undefined")
            continue
        a = max(a, 1 - a)
        print(f"   univariate AUROC  {n:<20} {a:.4f}")


def pearson_corr_to_min_eps(feats_np, names, min_eps_np):
    print(f"\n--- continuous target: min_eps_FGSM ---")
    for i, n in enumerate(names):
        cor = np.corrcoef(feats_np[:, i], min_eps_np)[0, 1]
        print(f"   corr(min_eps, {n:<20}) = {cor:+.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("Training small CNN on Fashion-MNIST ...")
    model = train_model(train_set)

    # Stack the test set on device
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # Basic features + restrict to samples model classifies correctly
    print("Computing basic features ...")
    margin, mean_pix, std_pix, final_pred = compute_basic_features(model, test_x, test_y)
    correct = final_pred == test_y
    print(f"  clean test accuracy = {correct.float().mean().item():.4f}")
    x_c = test_x[correct]
    y_c = test_y[correct]
    margin_c = margin[correct]
    mean_pix_c = mean_pix[correct]
    std_pix_c = std_pix[correct]
    n_c = x_c.size(0)
    print(f"  using {n_c} correctly-classified test samples")

    # Cohen certified radius per sample
    print(f"Computing Cohen certified radius "
          f"(sigma={SIGMA}, K={K_SMOOTH}, alpha={ALPHA_CP}) ...")
    t0 = time.time()
    radii_c, _ = cohen_certified_radius(model, x_c, sigma=SIGMA, K=K_SMOOTH,
                                        alpha=ALPHA_CP)
    print(f"  done in {time.time()-t0:.1f}s; "
          f"mean radius = {radii_c.mean().item():.4f}, "
          f"frac>0 = {(radii_c > 0).float().mean().item():.3f}")

    # Vulnerability targets (in batches for memory)
    print("Computing FGSM @ eps=15/255 ...")
    fgsm_res = []
    for i in range(0, n_c, 512):
        fgsm_res.append(fgsm_flip(model, x_c[i:i+512], y_c[i:i+512]))
    fgsm_res = torch.cat(fgsm_res)

    print("Computing PGD-20 @ eps=15/255 ...")
    pgd_res = []
    for i in range(0, n_c, 512):
        pgd_res.append(pgd_flip(model, x_c[i:i+512], y_c[i:i+512]))
    pgd_res = torch.cat(pgd_res)

    print("Computing min_eps_FGSM ...")
    me = []
    for i in range(0, n_c, 512):
        me.append(min_eps_fgsm(model, x_c[i:i+512], y_c[i:i+512]))
    min_eps = torch.cat(me)
    print(f"  mean min_eps = {min_eps.mean().item():.4f}")

    # Assemble feature matrix and evaluate
    feats = torch.stack([margin_c, mean_pix_c, std_pix_c, radii_c], dim=1)
    feat_names = ["margin", "mean_pix", "std_pix", "certified_radius"]
    feats_np = feats.detach().cpu().numpy()

    print("\n========== Fashion-MNIST: univariate AUROC ==========")
    univariate_auroc(feats_np, feat_names,
                     fgsm_res.detach().cpu().numpy().astype(int),
                     "FGSM_flip @ 15/255")
    univariate_auroc(feats_np, feat_names,
                     pgd_res.detach().cpu().numpy().astype(int),
                     "PGD20_flip @ 15/255")

    # Also dichotomise min_eps at median for a third binary target
    me_np = min_eps.detach().cpu().numpy()
    me_bin = (me_np < np.median(me_np)).astype(int)
    univariate_auroc(feats_np, feat_names, me_bin, "min_eps_FGSM < median")

    # Continuous correlation with min_eps
    pearson_corr_to_min_eps(feats_np, feat_names, me_np)

    print("\nDone.")


if __name__ == "__main__":
    main()
