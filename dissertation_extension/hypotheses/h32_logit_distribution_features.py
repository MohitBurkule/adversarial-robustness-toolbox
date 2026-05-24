"""
Hypothesis H32: Richer logit-distribution features capture per-sample
                adversarial-vulnerability signal beyond the top1-top2 margin.

Context: dissertation extension.  The standard "victim margin" baseline uses
only 2 of the n_classes logits.  We test whether other simple summaries of the
full 10-d logit / softmax vector add information over this baseline:

    - softmax_entropy            H[softmax(z)]
    - softmax_max                max_c softmax(z)_c
    - top3_minus_top1            z_(3) - z_(1)   (a wider-margin variant; <=0)
    - logit_std                  std over all 10 logits
    - energy                     logsumexp(z) - max(z)        (>=0, "energy" gap)
    - kl_uniform                 KL( softmax(z) || Uniform )  = log C - H[p]

Baselines:
    - victim_margin              z_(1) - z_(2)        (eval-mode, deterministic)
    - mean_pix, std_pix

Targets (defined ONLY on samples the victim classifies correctly in eval mode):
    - flipped_FGSM    binary, L_inf eps = 15/255
    - flipped_PGD     binary, 20 steps alpha=2/255 eps=15/255
    - FGSM_min_eps    continuous, per-sample L_inf binary search

Evaluation:
    1. Univariate AUROC per feature per binary target;
       Spearman per feature for the continuous target.
    2. Multivariate logistic regression:
         margin-only                 vs.  margin + rich-logit block
         all features                vs.  all minus rich-logit block
       Reports the delta-AUROC: do the richer logit features add over margin?
    3. Identifies the strongest single rich-logit feature per target.

Self-contained: trains victim from scratch (Adam, 10 epochs), downloads
Fashion-MNIST to /tmp/data, writes nothing else to disk.

Caveats:
    - softmax_entropy, softmax_max, kl_uniform are deterministic monotone
      functions of the FULL softmax — they are strongly correlated with each
      other and with the margin (which dominates softmax when one logit is
      large).  Adding all of them in a logistic regression invites
      multicollinearity; we still standardise + use sklearn's L2-regularised
      LR so coefficients are stable, and report delta-AUROC rather than reading
      individual coefficients as causal.
    - top3_minus_top1 is <= 0 by construction; AUROC is taken in the best
      direction via auroc_both_dirs.
    - "Vulnerability" is conditioned on correctly-classified test samples
      (the only samples for which "flipped" is meaningful).
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
N_CLASSES = 10
SEED = 0


# ---------------------------------------------------------------------------
class CNN(nn.Module):
    """Matches diagnostic_test.py: two conv layers, two FC, dropout 0.25 / 0.5."""
    def __init__(self, n=N_CLASSES):
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
def logit_distribution_features(logits):
    """All deterministic, eval-mode features derived from the 10-d logit vector.

    Args:
        logits: (N, C) tensor
    Returns:
        dict of (N,) tensors
    """
    eps_log = 1e-12
    sm = F.softmax(logits, dim=1)                           # (N, C)
    log_sm = F.log_softmax(logits, dim=1)                   # (N, C)
    sorted_logits, _ = logits.sort(dim=1, descending=True)  # (N, C)

    margin = sorted_logits[:, 0] - sorted_logits[:, 1]                       # top1 - top2
    top3m1 = sorted_logits[:, 2] - sorted_logits[:, 0]                       # <= 0
    sm_ent = -(sm * (sm + eps_log).log()).sum(dim=1)                          # H[p]
    sm_max = sm.max(dim=1).values
    logit_std = logits.std(dim=1, unbiased=False)
    energy = torch.logsumexp(logits, dim=1) - logits.max(dim=1).values        # >= 0
    # KL(p || U) = sum_c p log(p / (1/C)) = log C - H[p]
    kl_unif = np.log(N_CLASSES) - sm_ent

    return {
        "victim_margin": margin,
        "softmax_entropy": sm_ent,
        "softmax_max": sm_max,
        "top3_minus_top1": top3m1,
        "logit_std": logit_std,
        "energy": energy,
        "kl_uniform": kl_unif,
    }


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

    model.eval()
    with torch.no_grad():
        all_logits = batched_logits(model, test_x)
    pred = all_logits.argmax(1)
    correct = pred == test_y
    print(f"Test acc (eval mode): {correct.float().mean().item():.4f}")
    x = test_x[correct]
    y = test_y[correct]
    logits = all_logits[correct]
    N = x.size(0)
    print(f"Using {N} correctly-classified samples.")

    # --- features -------------------------------------------------------
    print("Computing rich logit-distribution features (eval mode) ...")
    lf = logit_distribution_features(logits)

    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))

    names = [
        "victim_margin",            # baseline
        "softmax_entropy",
        "softmax_max",
        "top3_minus_top1",
        "logit_std",
        "energy",
        "kl_uniform",
        "mean_pix",                 # other baseline
        "std_pix",
    ]
    rich_block_names = [
        "softmax_entropy",
        "softmax_max",
        "top3_minus_top1",
        "logit_std",
        "energy",
        "kl_uniform",
    ]
    feat_tensors = [
        lf["victim_margin"],
        lf["softmax_entropy"],
        lf["softmax_max"],
        lf["top3_minus_top1"],
        lf["logit_std"],
        lf["energy"],
        lf["kl_uniform"],
        mean_pix,
        std_pix,
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
    print("\n========== H32: rich logit-distribution features vs adversarial vulnerability ==========")
    print(f"N = {N}  FGSM-flip rate = {flip_fgsm.mean():.3f}  "
          f"PGD-flip rate = {flip_pgd.mean():.3f}")
    print(f"min_eps FGSM:  mean={min_eps.mean():.4f}  median={np.median(min_eps):.4f}")

    bin_targets = [("flipped_FGSM", flip_fgsm), ("flipped_PGD", flip_pgd)]

    Xs = StandardScaler().fit_transform(feats)
    margin_idx = names.index("victim_margin")
    rich_idxs = [names.index(n) for n in rich_block_names]
    non_rich_idxs = [i for i in range(len(names)) if i not in rich_idxs]

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

        best_rich = max(rich_block_names, key=lambda n: per_feat_auc[n])
        print(f"   strongest single rich-logit feature: {best_rich}  "
              f"(AUROC = {per_feat_auc[best_rich]:.4f})")
        print(f"   baseline (victim_margin)            AUROC = {per_feat_auc['victim_margin']:.4f}")

        lr_full = LogisticRegression(max_iter=4000).fit(Xs, yv)
        auc_full = roc_auc_score(yv, lr_full.predict_proba(Xs)[:, 1])

        Xs_m = Xs[:, [margin_idx]]
        lr_m = LogisticRegression(max_iter=4000).fit(Xs_m, yv)
        auc_m = roc_auc_score(yv, lr_m.predict_proba(Xs_m)[:, 1])

        Xs_m_rich = Xs[:, [margin_idx] + rich_idxs]
        lr_m_rich = LogisticRegression(max_iter=4000).fit(Xs_m_rich, yv)
        auc_m_rich = roc_auc_score(yv, lr_m_rich.predict_proba(Xs_m_rich)[:, 1])

        Xs_no_rich = Xs[:, non_rich_idxs]
        lr_no_rich = LogisticRegression(max_iter=4000).fit(Xs_no_rich, yv)
        auc_no_rich = roc_auc_score(yv, lr_no_rich.predict_proba(Xs_no_rich)[:, 1])

        print(f"   multivariate AUROC (margin only):                {auc_m:.4f}")
        print(f"   multivariate AUROC (margin + rich-logit block):  {auc_m_rich:.4f}")
        print(f"     Delta over margin-alone:                         {auc_m_rich - auc_m:+.4f}")
        print(f"   multivariate AUROC (all features):               {auc_full:.4f}")
        print(f"   multivariate AUROC (all minus rich-logit block): {auc_no_rich:.4f}")
        print(f"     Delta from removing rich-logit block:            {auc_full - auc_no_rich:+.4f}")
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
    ols_m_rich = LinearRegression().fit(Xs[:, [margin_idx] + rich_idxs], min_eps)
    r2_m_rich = ols_m_rich.score(Xs[:, [margin_idx] + rich_idxs], min_eps)
    ols_no_rich = LinearRegression().fit(Xs[:, non_rich_idxs], min_eps)
    r2_no_rich = ols_no_rich.score(Xs[:, non_rich_idxs], min_eps)
    print(f"   OLS R^2 (margin only):                   {r2_m:.4f}")
    print(f"   OLS R^2 (margin + rich-logit block):     {r2_m_rich:.4f}")
    print(f"     Delta:                                   {r2_m_rich - r2_m:+.4f}")
    print(f"   OLS R^2 (all features):                  {r2_full:.4f}")
    print(f"   OLS R^2 (all minus rich-logit block):    {r2_no_rich:.4f}")
    print(f"     Delta:                                   {r2_full - r2_no_rich:+.4f}")
    print("   OLS coefficients (full model):")
    for n, c in zip(names, ols_full.coef_.flatten()):
        print(f"     {n:<22} {c:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
