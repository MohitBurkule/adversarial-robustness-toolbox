"""
Hypothesis H20: Monte-Carlo dropout uncertainty predicts adversarial vulnerability,
                and is essentially free since the architecture already contains dropout.

Context: dissertation extension.  Smith & Gal (2018, arXiv:1803.08533, UAI)
showed MC-dropout uncertainty (in particular the mutual information / BALD score)
can detect adversarial inputs.  We test the *inverse* direction: does the
MC-dropout uncertainty of a CLEAN test sample predict whether that sample will be
adversarially flipped by FGSM / PGD on the SAME victim?

Why "free": our victim CNN already has dropout layers (p=0.25 after conv pool
and p=0.5 after fc1, matching diagnostic_test.py).  By calling model.train() at
inference we re-enable the dropout masks and obtain K stochastic forward passes
per sample at no architecture / training cost.

MC-dropout features computed per test sample (K=20 stochastic passes):
    - predictive_entropy = H[mean_k softmax_k]                         (total uncertainty)
    - mutual_information = H[mean] - mean_k H[softmax_k]               (epistemic / BALD)
    - softmax_l2_variance = mean_c Var_k softmax_k[c]                  (variance proxy)
    - vote_agreement     = fraction of K argmax votes agreeing with majority
    - mean_softmax_max   = max prob of the averaged softmax (MC-confidence)

Baselines:
    - victim_margin (eval-mode, deterministic logit gap top1 - top2)
    - mean_pix, std_pix

Targets (defined ONLY on samples the victim classifies correctly in eval mode):
    - flipped_FGSM        binary,  L_inf eps = 15/255
    - flipped_PGD         binary,  20 steps, alpha = 2/255, eps = 15/255
    - FGSM_min_eps        continuous, per-sample L_inf binary search

Evaluation:
    1. Univariate AUROC of each feature against each binary target.
    2. Spearman of each feature against the continuous target.
    3. Multivariate logistic regression:
         margin-only        vs.  margin + MC-dropout block
         all-features       vs.  all-features minus MC-dropout block
       Reports delta AUROC: does MC-dropout add over the margin baseline?
    4. Identifies the strongest single MC-dropout feature per target.

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
MC_K = 20                # number of stochastic forward passes
SEED = 0


# ---------------------------------------------------------------------------
class CNN(nn.Module):
    """Matches diagnostic_test.py: two conv layers, two FC, dropout 0.25 / 0.5."""
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
    # ensure deterministic (eval mode) gradient
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
def mc_dropout_features(model, x, K=MC_K, bs=256):
    """K stochastic forward passes with dropout active (model.train()).
    BatchNorm is irrelevant — this CNN has none — so train() only toggles dropout.

    Returns dict of (N,) tensors:
        predictive_entropy, mutual_information, softmax_l2_variance,
        vote_agreement, mean_softmax_max
    """
    model.train()                    # enable dropout at inference
    N = x.size(0)
    pred_ent = torch.zeros(N, device=DEVICE)
    mut_inf = torch.zeros(N, device=DEVICE)
    sm_var = torch.zeros(N, device=DEVICE)
    vote_ag = torch.zeros(N, device=DEVICE)
    msm_max = torch.zeros(N, device=DEVICE)
    eps_log = 1e-12

    with torch.no_grad():
        for i in range(0, N, bs):
            xb = x[i:i+bs]
            B = xb.size(0)
            # stack K softmaxes: (K, B, C)
            sm_stack = []
            ent_stack = []
            arg_stack = []
            for _ in range(K):
                logits = model(xb)
                sm = F.softmax(logits, dim=1)
                sm_stack.append(sm)
                ent_stack.append(-(sm * (sm + eps_log).log()).sum(dim=1))   # (B,)
                arg_stack.append(sm.argmax(dim=1))
            sm_stack = torch.stack(sm_stack, dim=0)              # (K, B, C)
            ent_stack = torch.stack(ent_stack, dim=0)            # (K, B)
            arg_stack = torch.stack(arg_stack, dim=0)            # (K, B)

            mean_sm = sm_stack.mean(dim=0)                       # (B, C)
            H_mean = -(mean_sm * (mean_sm + eps_log).log()).sum(dim=1)   # (B,)
            mean_H = ent_stack.mean(dim=0)                       # (B,)
            bald = H_mean - mean_H                               # (B,)
            var_per_class = sm_stack.var(dim=0, unbiased=False)  # (B, C)
            l2var = var_per_class.mean(dim=1)                    # (B,)

            # majority vote agreement
            # mode per column of (K, B)
            modes, _ = torch.mode(arg_stack, dim=0)              # (B,)
            agree = (arg_stack == modes.unsqueeze(0)).float().mean(dim=0)  # (B,)

            pred_ent[i:i+B] = H_mean
            mut_inf[i:i+B] = bald
            sm_var[i:i+B] = l2var
            vote_ag[i:i+B] = agree
            msm_max[i:i+B] = mean_sm.max(dim=1).values

    model.eval()
    return {
        "predictive_entropy": pred_ent,
        "mutual_information": mut_inf,
        "softmax_l2_variance": sm_var,
        "vote_agreement": vote_ag,
        "mean_softmax_max": msm_max,
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

    print(f"Computing MC-dropout features  (K={MC_K} stochastic passes) ...")
    mc = mc_dropout_features(model, x, K=MC_K)

    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))

    feat_tensors = [
        margin,
        mc["predictive_entropy"],
        mc["mutual_information"],
        mc["softmax_l2_variance"],
        mc["vote_agreement"],
        mc["mean_softmax_max"],
        mean_pix,
        std_pix,
    ]
    names = [
        "victim_margin",
        "mc_pred_entropy",
        "mc_mutual_info",
        "mc_softmax_l2var",
        "mc_vote_agreement",
        "mc_mean_sm_max",
        "mean_pix",
        "std_pix",
    ]
    mc_block_names = [
        "mc_pred_entropy",
        "mc_mutual_info",
        "mc_softmax_l2var",
        "mc_vote_agreement",
        "mc_mean_sm_max",
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
    print("\n========== H20: MC-dropout uncertainty vs adversarial vulnerability ==========")
    print(f"N = {N}  FGSM-flip rate = {flip_fgsm.mean():.3f}  "
          f"PGD-flip rate = {flip_pgd.mean():.3f}")
    print(f"min_eps FGSM:  mean={min_eps.mean():.4f}  median={np.median(min_eps):.4f}")

    bin_targets = [("flipped_FGSM", flip_fgsm), ("flipped_PGD", flip_pgd)]

    Xs = StandardScaler().fit_transform(feats)
    margin_idx = names.index("victim_margin")
    mc_idxs = [names.index(n) for n in mc_block_names]
    non_mc_idxs = [i for i in range(len(names)) if i not in mc_idxs]

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

        # strongest single MC feature
        best_mc = max(mc_block_names, key=lambda n: per_feat_auc[n])
        print(f"   strongest single MC feature: {best_mc}  "
              f"(AUROC = {per_feat_auc[best_mc]:.4f})")

        # multivariate logistic regressions
        lr_full = LogisticRegression(max_iter=4000).fit(Xs, yv)
        auc_full = roc_auc_score(yv, lr_full.predict_proba(Xs)[:, 1])

        Xs_m = Xs[:, [margin_idx]]
        lr_m = LogisticRegression(max_iter=4000).fit(Xs_m, yv)
        auc_m = roc_auc_score(yv, lr_m.predict_proba(Xs_m)[:, 1])

        Xs_m_mc = Xs[:, [margin_idx] + mc_idxs]
        lr_m_mc = LogisticRegression(max_iter=4000).fit(Xs_m_mc, yv)
        auc_m_mc = roc_auc_score(yv, lr_m_mc.predict_proba(Xs_m_mc)[:, 1])

        Xs_no_mc = Xs[:, non_mc_idxs]
        lr_no_mc = LogisticRegression(max_iter=4000).fit(Xs_no_mc, yv)
        auc_no_mc = roc_auc_score(yv, lr_no_mc.predict_proba(Xs_no_mc)[:, 1])

        print(f"   multivariate AUROC (margin only):              {auc_m:.4f}")
        print(f"   multivariate AUROC (margin + MC block):        {auc_m_mc:.4f}")
        print(f"     Delta over margin-alone:                       {auc_m_mc - auc_m:+.4f}")
        print(f"   multivariate AUROC (all features):             {auc_full:.4f}")
        print(f"   multivariate AUROC (all minus MC block):       {auc_no_mc:.4f}")
        print(f"     Delta from removing MC block:                  {auc_full - auc_no_mc:+.4f}")
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
    ols_m_mc = LinearRegression().fit(Xs[:, [margin_idx] + mc_idxs], min_eps)
    r2_m_mc = ols_m_mc.score(Xs[:, [margin_idx] + mc_idxs], min_eps)
    ols_no_mc = LinearRegression().fit(Xs[:, non_mc_idxs], min_eps)
    r2_no_mc = ols_no_mc.score(Xs[:, non_mc_idxs], min_eps)
    print(f"   OLS R^2 (margin only):                 {r2_m:.4f}")
    print(f"   OLS R^2 (margin + MC block):           {r2_m_mc:.4f}")
    print(f"     Delta:                                 {r2_m_mc - r2_m:+.4f}")
    print(f"   OLS R^2 (all features):                {r2_full:.4f}")
    print(f"   OLS R^2 (all minus MC block):          {r2_no_mc:.4f}")
    print(f"     Delta:                                 {r2_full - r2_no_mc:+.4f}")
    print("   OLS coefficients (full model):")
    for n, c in zip(names, ols_full.coef_.flatten()):
        print(f"     {n:<22} {c:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
