"""
Hypothesis H94: Lakshminarayanan et al. (NeurIPS 2017, "Simple and Scalable
Predictive Uncertainty Estimation using Deep Ensembles") deep ensembles --- K
independently-trained models with different random seeds --- yield per-sample
epistemic uncertainty (predictive entropy of the mean softmax MINUS the mean of
per-member entropies) that predicts adversarial vulnerability of victim model 0.

Pipeline:
    1. Train K=5 small CNNs on Fashion-MNIST, 10 epochs each, seeds 0..K-1.
       Architecture matches diagnostic_test.py (two convs, two FCs, dropout
       0.25 / 0.5).  Member 0 is the victim model whose vulnerability we predict.
    2. For each test sample compute, over the K members:
         - mean_softmax            mean_k p_k(c|x)
         - predictive_entropy      H[ mean_softmax ]            (total uncertainty)
         - aleatoric_entropy       mean_k H[ p_k(.|x) ]         (data uncertainty)
         - epistemic_uncertainty   pred_ent - aleatoric_ent     (Bayesian / BALD)
         - mean_softmax_max        max_c mean_softmax(c)
         - vote_agreement          fraction of members agreeing with majority arg-max
       Plus baseline victim-model-0 margin (logit gap top1 - top2 in eval mode).
    3. Vulnerability targets evaluated on Model 0 over correctly-classified
       samples only:
         - flipped_FGSM    L_inf eps = 15/255
         - flipped_PGD     20 steps, alpha = 2/255, eps = 15/255
         - min_eps_FGSM    per-sample binary search for smallest flipping eps
    4. Univariate AUROC of each feature vs each binary target (both directions
       tried, max reported).  Spearman correlation vs the continuous min-eps
       target.

Self-contained: trains all ensemble members from scratch on the fly, downloads
Fashion-MNIST to /tmp/data, writes nothing to disk.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
K_ENSEMBLE = 5
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0


# ---------------------------------------------------------------------------
class CNN(nn.Module):
    """Matches diagnostic_test.py: two conv layers, two FCs, dropout 0.25 / 0.5."""
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
def train_member(seed, train_set):
    torch.manual_seed(seed)
    np.random.seed(seed)
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
        print(f"    member seed={seed}  epoch {ep+1}/{EPOCHS} done")
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
def batched_softmax(model, x, bs=512):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(F.softmax(model(x[i:i+bs]), dim=1))
    return torch.cat(out, 0)


def fgsm_attack(model, x, y, eps=EPS_TEST):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_adv), y).backward()
    sign = x_adv.grad.sign().detach()
    return (x + eps * sign).clamp(0, 1).detach()


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
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
    return x_adv.detach()


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
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
def ensemble_features(members, x, bs=256):
    """Per-sample deep-ensemble uncertainty features over K members.

    Returns dict of (N,) tensors:
        predictive_entropy   = H[ mean_k softmax_k ]            total uncertainty
        aleatoric_entropy    = mean_k H[ softmax_k ]            data uncertainty
        epistemic            = pred_ent - aleatoric             Bayesian / BALD
        mean_softmax_max     = max_c mean_softmax(c)
        vote_agreement       = fraction of members voting majority class
    """
    K = len(members)
    N = x.size(0)
    eps_log = 1e-12
    pred_ent = torch.zeros(N, device=DEVICE)
    aleat = torch.zeros(N, device=DEVICE)
    msm_max = torch.zeros(N, device=DEVICE)
    vote_ag = torch.zeros(N, device=DEVICE)
    for m in members:
        m.eval()

    with torch.no_grad():
        for i in range(0, N, bs):
            xb = x[i:i+bs]
            B = xb.size(0)
            sm_stack = []
            ent_stack = []
            arg_stack = []
            for m in members:
                sm = F.softmax(m(xb), dim=1)
                sm_stack.append(sm)
                ent_stack.append(-(sm * (sm + eps_log).log()).sum(dim=1))
                arg_stack.append(sm.argmax(dim=1))
            sm_stack = torch.stack(sm_stack, dim=0)         # (K, B, C)
            ent_stack = torch.stack(ent_stack, dim=0)       # (K, B)
            arg_stack = torch.stack(arg_stack, dim=0)       # (K, B)

            mean_sm = sm_stack.mean(dim=0)                  # (B, C)
            H_mean = -(mean_sm * (mean_sm + eps_log).log()).sum(dim=1)
            mean_H = ent_stack.mean(dim=0)
            modes, _ = torch.mode(arg_stack, dim=0)
            agree = (arg_stack == modes.unsqueeze(0)).float().mean(dim=0)

            pred_ent[i:i+B] = H_mean
            aleat[i:i+B] = mean_H
            msm_max[i:i+B] = mean_sm.max(dim=1).values
            vote_ag[i:i+B] = agree

    epistemic = pred_ent - aleat
    return {
        "predictive_entropy": pred_ent,
        "aleatoric_entropy": aleat,
        "epistemic": epistemic,
        "mean_softmax_max": msm_max,
        "vote_agreement": vote_ag,
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

    print(f"Training deep ensemble of K={K_ENSEMBLE} CNNs on {DEVICE} "
          f"({EPOCHS} epochs each) ...")
    members = []
    for k in range(K_ENSEMBLE):
        print(f"  -> training member {k} (seed={k}) ...")
        members.append(train_member(k, train_set))
    victim = members[0]

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # report per-member test accuracy
    for k, m in enumerate(members):
        with torch.no_grad():
            acc = (batched_logits(m, test_x).argmax(1) == test_y).float().mean().item()
        print(f"  member {k} test acc = {acc:.4f}")

    # restrict to samples victim (model 0) classifies correctly
    victim.eval()
    with torch.no_grad():
        pred0 = batched_logits(victim, test_x).argmax(1)
    correct = pred0 == test_y
    print(f"Victim (model 0) acc = {correct.float().mean().item():.4f}")
    x = test_x[correct]
    y = test_y[correct]
    N = x.size(0)
    print(f"Using {N} samples that victim classifies correctly.")

    # --- features -------------------------------------------------------
    print("Computing victim_margin (model 0, eval mode) ...")
    margin = victim_margin(victim, x)

    print(f"Computing deep-ensemble features over K={K_ENSEMBLE} members ...")
    ens = ensemble_features(members, x)

    feat_tensors = [
        margin,
        ens["predictive_entropy"],
        ens["aleatoric_entropy"],
        ens["epistemic"],
        ens["mean_softmax_max"],
        ens["vote_agreement"],
    ]
    names = [
        "victim_margin",
        "ens_pred_entropy",
        "ens_aleatoric",
        "ens_epistemic",
        "ens_mean_sm_max",
        "ens_vote_agreement",
    ]
    feats = torch.stack(feat_tensors, 1).detach().cpu().numpy()

    # --- targets --------------------------------------------------------
    print("Running FGSM attack on victim ...")
    flip_fgsm = batched_attack_flag(victim, x, y, fgsm_attack).cpu().numpy().astype(int)

    print("Running PGD attack on victim ...")
    flip_pgd = batched_attack_flag(victim, x, y, pgd_attack).cpu().numpy().astype(int)

    print("Running per-sample min-eps FGSM binary search on victim ...")
    me = []
    for i in range(0, N, 256):
        me.append(min_eps_fgsm(victim, x[i:i+256], y[i:i+256]))
    min_eps = torch.cat(me, 0).cpu().numpy()

    # --- evaluation -----------------------------------------------------
    print("\n========== H94: Deep-ensemble Bayesian uncertainty vs adversarial vulnerability ==========")
    print(f"N = {N}  FGSM-flip rate = {flip_fgsm.mean():.3f}  "
          f"PGD-flip rate = {flip_pgd.mean():.3f}")
    print(f"min_eps FGSM:  mean={min_eps.mean():.4f}  median={np.median(min_eps):.4f}")

    bin_targets = [("flipped_FGSM", flip_fgsm), ("flipped_PGD", flip_pgd)]

    for t_name, yv in bin_targets:
        if yv.std() == 0:
            print(f"\n--- target {t_name}: no variance, skipping ---")
            continue
        print(f"\n--- target: {t_name}  (positive rate = {yv.mean():.3f}) ---")
        for i, n in enumerate(names):
            a = auroc_both_dirs(yv, feats[:, i])
            print(f"   univariate AUROC  {n:<22} {a:.4f}")

    print(f"\n--- continuous target: min_eps_FGSM (lower = more vulnerable) ---")
    for i, n in enumerate(names):
        r, p = spearmanr(feats[:, i], min_eps)
        print(f"   spearman  {n:<22} rho={r:+.4f}  p={p:.2e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
