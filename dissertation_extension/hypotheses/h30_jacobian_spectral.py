"""
Hypothesis H30: Per-sample spectral norm of the input-Jacobian
(J = d logits / d x, shape (C, D)) predicts adversarial vulnerability.

The spectral norm sigma_1(J(x)) = max_{||v||=1} ||J v||_2 upper-bounds the
worst-case L2 amplification of an input perturbation to the logits, and is
a tighter local-Lipschitz bound than the gradient L2 norm of any single
class / loss (which only sees one row / linear combination of rows of J).

We estimate sigma_1(J) via power iteration:
    v ~ N(0,I), v <- v / ||v||
    repeat:
        u = J v                  # forward Jacobian-vector product
        v = J^T u                # backward vector-Jacobian product
        v <- v / ||v||
    sigma_1 ~ ||J v_final||

We use torch.autograd.functional.jvp for the forward step and
torch.autograd.grad for the backward step (true reverse-mode VJP).

We also compute the Frobenius norm  ||J||_F = sqrt(sum_c ||grad logit_c||_2^2)
which is an upper bound on sigma_1 and a looser (but exact) Lipschitz proxy.

Baselines: victim_margin, mean_pix, std_pix, input_grad_L2_norm
           (gradient of CE loss w.r.t. input -- single direction).
Targets:   flipped_FGSM (eps = 15/255), flipped_PGD, FGSM_min_eps.

Self-contained: trains a small CNN on Fashion-MNIST (10 epochs, Adam),
matching dissertation_extension/diagnostic_test.py.  Writes nothing besides
the FashionMNIST download into /tmp/data.
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
POWER_ITERS = 5
SEED = 0


# ---------------------------------------------------------------------------
class CNN(nn.Module):
    """Same small CNN used in diagnostic_test.py / h07."""
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
    return hi


def batched_attack_flag(model, x, y, attack_fn, bs=256):
    flags = []
    for i in range(0, x.size(0), bs):
        adv = attack_fn(model, x[i:i+bs], y[i:i+bs])
        with torch.no_grad():
            flags.append(model(adv).argmax(1) != y[i:i+bs])
    return torch.cat(flags, 0)


# ---------------------------------------------------------------------------
def jacobian_spectral_and_frobenius(model, x, iters=POWER_ITERS, bs=64):
    """Per-sample spectral norm sigma_1(J) via power iteration on the
    input-Jacobian J = d logits / d x.

    Implementation detail: for each batch we treat the per-sample Jacobian
    as block-diagonal in the batched forward pass (different samples don't
    interact through the model), so a single batched JVP / VJP gives us all
    per-sample J v_i / J^T u_i at once.

    Returns (spectral, frobenius) tensors of shape (N,).
    """
    model.eval()
    N = x.size(0)
    spec = torch.zeros(N, device=DEVICE)
    frob = torch.zeros(N, device=DEVICE)
    C = None  # n_classes
    for i in range(0, N, bs):
        xb = x[i:i+bs].detach()
        B = xb.size(0)

        # ---- power iteration for sigma_1(J_b) per sample ----
        # v: input-shape vector per sample
        v = torch.randn_like(xb)
        v_flat = v.flatten(1)
        v_flat = v_flat / (v_flat.norm(dim=1, keepdim=True) + 1e-12)
        v = v_flat.view_as(xb)

        sigma = torch.zeros(B, device=DEVICE)
        for _ in range(iters):
            # forward JVP: u = J v   -> shape (B, C)
            xb_req = xb.clone().requires_grad_(True)
            # vjp via double-backward trick (functional jvp is fine too, but
            # this is reliable across torch versions).
            logits = model(xb_req)
            if C is None:
                C = logits.size(1)
            # surrogate scalar whose grad w.r.t. dummy_g equals J v
            dummy_g = torch.zeros_like(logits, requires_grad=True)
            # grad_x ( <dummy_g, logits> ) = J^T dummy_g (vector of shape xb)
            grad_x = torch.autograd.grad(
                (dummy_g * logits).sum(), xb_req, create_graph=True
            )[0]
            # grad of <grad_x, v> w.r.t. dummy_g = J v   (Pearlmutter trick)
            u = torch.autograd.grad((grad_x * v).sum(), dummy_g)[0]  # (B, C)
            u_norm = u.flatten(1).norm(dim=1)  # (B,)

            # backward VJP: v_new = J^T u  -> shape xb
            xb_req2 = xb.clone().requires_grad_(True)
            logits2 = model(xb_req2)
            v_new = torch.autograd.grad((u.detach() * logits2).sum(), xb_req2)[0]
            v_flat = v_new.flatten(1)
            v_norms = v_flat.norm(dim=1, keepdim=True) + 1e-12
            v_flat = v_flat / v_norms
            v = v_flat.view_as(xb)

            # Rayleigh-quotient style estimate sigma ~ ||u|| (since v unit norm)
            sigma = u_norm.detach()

        spec[i:i+B] = sigma

        # ---- Frobenius norm: sum_c ||grad_x logit_c||^2 ----
        # Do this with C backward passes (cheap for C=10) using a one-hot
        # weighting across the batch — same as h07's input-grad routine but
        # per class.
        f_sq = torch.zeros(B, device=DEVICE)
        for c in range(C):
            xb_req = xb.clone().requires_grad_(True)
            logits = model(xb_req)
            g = torch.autograd.grad(logits[:, c].sum(), xb_req)[0]
            f_sq = f_sq + g.flatten(1).pow(2).sum(dim=1)
        frob[i:i+B] = f_sq.sqrt()

    return spec.detach(), frob.detach()


def input_grad_norm(model, x, y, bs=256):
    """|| grad_x CE-loss(model(x), y) ||_2 per sample (single direction)."""
    N = x.size(0)
    out = torch.zeros(N, device=DEVICE)
    for i in range(0, N, bs):
        xb = x[i:i+bs].clone().detach().requires_grad_(True)
        yb = y[i:i+bs]
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        g = torch.autograd.grad(loss, xb)[0]
        out[i:i+bs] = g.flatten(1).norm(dim=1)
    return out.detach()


def victim_margin(model, x):
    with torch.no_grad():
        logits = batched_logits(model, x)
    s, _ = logits.sort(1, descending=True)
    return (s[:, 0] - s[:, 1])


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

    print(f"Computing Jacobian spectral norm (power iter x{POWER_ITERS}) "
          f"and Frobenius norm ...")
    jac_spec, jac_frob = jacobian_spectral_and_frobenius(model, x)

    print("Computing input-gradient L2 norm (CE-loss, single direction) ...")
    grad_norm = input_grad_norm(model, x, y)

    print("Computing baseline pixel stats ...")
    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))

    feats = torch.stack([margin, jac_spec, jac_frob, grad_norm,
                         mean_pix, std_pix], 1).cpu().numpy()
    names = ["victim_margin", "jacobian_spectral", "jacobian_frobenius",
             "input_grad_norm", "mean_pix", "std_pix"]

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
    print("\n========== H30: Jacobian spectral norm vs adversarial vulnerability ==========")
    print(f"N = {N}  FGSM-flip rate = {flip_fgsm.mean():.3f}  "
          f"PGD-flip rate = {flip_pgd.mean():.3f}")
    print(f"min_eps FGSM:  mean={min_eps.mean():.4f}  median={np.median(min_eps):.4f}")
    print(f"jac_spectral   mean={jac_spec.mean().item():.3f}  "
          f"median={jac_spec.median().item():.3f}")
    print(f"jac_frobenius  mean={jac_frob.mean().item():.3f}  "
          f"median={jac_frob.median().item():.3f}")
    # spec <= frob always:
    viol = (jac_spec > jac_frob + 1e-3).float().mean().item()
    print(f"sanity: fraction of samples with spec > frob (should be ~0): {viol:.4f}")

    bin_targets = [("flipped_FGSM", flip_fgsm), ("flipped_PGD", flip_pgd)]

    for t_name, yv in bin_targets:
        if yv.std() == 0:
            print(f"\n--- target {t_name}: no variance, skipping ---")
            continue
        print(f"\n--- target: {t_name}  (positive rate = {yv.mean():.3f}) ---")
        for i, n in enumerate(names):
            a = auroc_both_dirs(yv, feats[:, i])
            print(f"   univariate AUROC  {n:<20} {a:.4f}")

        Xs = StandardScaler().fit_transform(feats)
        margin_idx = names.index("victim_margin")
        grad_idx = names.index("input_grad_norm")
        spec_idx = names.index("jacobian_spectral")
        frob_idx = names.index("jacobian_frobenius")

        def fit_auc(cols):
            lr = LogisticRegression(max_iter=4000).fit(Xs[:, cols], yv)
            return roc_auc_score(yv, lr.predict_proba(Xs[:, cols])[:, 1])

        auc_m = fit_auc([margin_idx])
        auc_mg = fit_auc([margin_idx, grad_idx])
        auc_mgs = fit_auc([margin_idx, grad_idx, spec_idx])
        auc_mgsf = fit_auc([margin_idx, grad_idx, spec_idx, frob_idx])
        auc_full = fit_auc(list(range(len(names))))
        auc_no_jac = fit_auc([i for i in range(len(names))
                              if i not in (spec_idx, frob_idx)])

        print(f"   multivariate AUROC (margin only):                {auc_m:.4f}")
        print(f"   multivariate AUROC (margin + grad_norm):         {auc_mg:.4f}")
        print(f"   multivariate AUROC (margin + grad + jac_spec):   {auc_mgs:.4f}")
        print(f"     Delta from adding jac_spectral:                  {auc_mgs - auc_mg:+.4f}")
        print(f"   multivariate AUROC (margin + grad + spec + frob):{auc_mgsf:.4f}")
        print(f"     Delta from adding jac_frobenius:                 {auc_mgsf - auc_mgs:+.4f}")
        print(f"   multivariate AUROC (all features):               {auc_full:.4f}")
        print(f"   multivariate AUROC (all minus jac_{{spec,frob}}): {auc_no_jac:.4f}")
        print(f"     Delta from removing both Jacobian features:      {auc_full - auc_no_jac:+.4f}")

        lr_full = LogisticRegression(max_iter=4000).fit(Xs, yv)
        print(f"   standardised coefficients (full model):")
        for n, c in zip(names, lr_full.coef_.flatten()):
            print(f"     {n:<20} {c:+.4f}")

    # continuous target
    print(f"\n--- continuous target: FGSM_min_eps ---")
    from scipy.stats import spearmanr
    for i, n in enumerate(names):
        r, p = spearmanr(feats[:, i], min_eps)
        print(f"   spearman  {n:<20} rho={r:+.4f}  p={p:.2e}")

    Xs = StandardScaler().fit_transform(feats)
    margin_idx = names.index("victim_margin")
    grad_idx = names.index("input_grad_norm")
    spec_idx = names.index("jacobian_spectral")
    frob_idx = names.index("jacobian_frobenius")

    def fit_r2(cols):
        ols = LinearRegression().fit(Xs[:, cols], min_eps)
        return ols.score(Xs[:, cols], min_eps)

    r2_m = fit_r2([margin_idx])
    r2_mg = fit_r2([margin_idx, grad_idx])
    r2_mgs = fit_r2([margin_idx, grad_idx, spec_idx])
    r2_mgsf = fit_r2([margin_idx, grad_idx, spec_idx, frob_idx])
    r2_full = fit_r2(list(range(len(names))))
    print(f"   OLS R^2 (margin only):                {r2_m:.4f}")
    print(f"   OLS R^2 (margin + grad):              {r2_mg:.4f}")
    print(f"   OLS R^2 (margin + grad + jac_spec):   {r2_mgs:.4f}")
    print(f"     Delta from jac_spectral:              {r2_mgs - r2_mg:+.4f}")
    print(f"   OLS R^2 (margin + grad + spec + frob):{r2_mgsf:.4f}")
    print(f"   OLS R^2 (all features):               {r2_full:.4f}")

    ols_full = LinearRegression().fit(Xs, min_eps)
    print(f"   OLS coefficients (full):")
    for n, c in zip(names, ols_full.coef_.flatten()):
        print(f"     {n:<20} {c:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
