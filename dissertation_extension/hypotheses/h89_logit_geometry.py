"""
H89: Logit-geometry features predict adversarial vulnerability.

Hypothesis:
  Per-sample geometric features of the final logit vector carry a signal for
  adversarial vulnerability beyond simple margin-based features:
    - logit_l2_norm                   : ||z||_2  (overall logit magnitude)
    - angle_top2_to_rest              : angle (radians) between (z_true - z_2nd)
                                        and (z_true - mean(other_logits))
    - top3_gap                        : z_(1) - z_(3) (top-3 cluster spread)
    - fraction_of_logits_above_zero   : fraction of logit entries > 0

We additionally include classic baselines:
    - margin   : z_true - z_2nd
    - mean_pix : mean pixel intensity
    - std_pix  : pixel std

Targets (per-sample, binary unless noted):
    - FGSM @ eps=15/255 self
    - PGD  @ eps=15/255 self (10-step, alpha=eps/4)
    - min_eps to flip with FGSM (continuous; binary-search)

Model: small CNN matching diagnostic_test.py, trained 10 epochs on Fashion-MNIST.
Analysis: univariate AUROC of each feature against each binary target, and
Pearson correlation against min_eps.

This file is self-contained: it does not import from the rest of the project.
Run with:  python h89_logit_geometry.py
"""
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0
N_CLASSES = 10
SEED = 0


# --------------------- model: same CNN as diagnostic_test.py ----------------
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


# --------------------------------- training ---------------------------------
def train(seed, train_set):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ------------------------------ logit geometry ------------------------------
def compute_logit_features(model, x, y):
    """Return per-sample tensor of [logit_l2_norm, angle_top2_to_rest,
    top3_gap, fraction_logits_above_zero, margin]."""
    N = x.size(0)
    with torch.no_grad():
        logits_chunks = []
        for i in range(0, N, 512):
            logits_chunks.append(model(x[i:i+512]))
        logits = torch.cat(logits_chunks, 0)  # [N, C]
    C = logits.size(1)

    # logit l2 norm
    logit_l2 = logits.norm(dim=1)

    # top-3 gap
    sorted_l, _ = logits.sort(dim=1, descending=True)
    top3_gap = sorted_l[:, 0] - sorted_l[:, 2]

    # fraction above zero
    frac_pos = (logits > 0).float().mean(dim=1)

    # true logit and 2nd-best logit (excluding true class)
    one_hot_true = F.one_hot(y, C).bool()
    z_true = logits[torch.arange(N, device=logits.device), y]
    masked = logits.masked_fill(one_hot_true, float("-inf"))
    z_2nd_val, z_2nd_idx = masked.max(dim=1)

    # margin = z_true - z_2nd
    margin = z_true - z_2nd_val

    # mean of "other" logits = mean of all logits except true class
    sum_logits = logits.sum(dim=1)
    mean_other = (sum_logits - z_true) / (C - 1)

    # build vectors v1 = z_true - z_2nd (scalars) is degenerate as a vector.
    # Interpret geometrically: define direction-in-class-space vectors.
    # v1: one-hot(true) - one-hot(2nd), scaled by (z_true - z_2nd)
    # v2: one-hot(true) - uniform_over_others, scaled by (z_true - mean_other)
    # Compute angle between these two C-dim vectors.
    v1 = torch.zeros_like(logits)
    rows = torch.arange(N, device=logits.device)
    v1[rows, y] = z_true - z_2nd_val
    v1[rows, z_2nd_idx] = -(z_true - z_2nd_val)

    uniform = torch.full_like(logits, 1.0 / (C - 1))
    uniform[rows, y] = 0.0
    # v2 = (e_true - uniform_over_others) * (z_true - mean_other)
    v2 = -uniform.clone()
    v2[rows, y] = 1.0
    v2 = v2 * (z_true - mean_other).unsqueeze(1)

    n1 = v1.norm(dim=1).clamp_min(1e-12)
    n2 = v2.norm(dim=1).clamp_min(1e-12)
    cos = (v1 * v2).sum(dim=1) / (n1 * n2)
    cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos)

    return {
        "logit_l2_norm": logit_l2,
        "angle_top2_to_rest": angle,
        "top3_gap": top3_gap,
        "fraction_of_logits_above_zero": frac_pos,
        "margin": margin,
    }, logits


def compute_pixel_features(x):
    flat = x.flatten(1)
    return {"mean_pix": flat.mean(dim=1), "std_pix": flat.std(dim=1)}


# --------------------------------- attacks ----------------------------------
def fgsm_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign = fgsm_sign(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    delta = torch.zeros_like(x0).uniform_(-eps, eps)
    adv = (x0 + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    """Binary search per sample for smallest L_inf eps that flips FGSM."""
    sign = fgsm_sign(model, x, y)
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


def batched_attack(model, x, y, fn, **kw):
    out = []
    for i in range(0, x.size(0), 512):
        out.append(fn(model, x[i:i+512], y[i:i+512], **kw))
    return torch.cat(out)


# ------------------------------ AUROC reporting -----------------------------
def univariate_auroc(feats_np, names, y_bin):
    print(f"  positive rate = {y_bin.mean():.4f}")
    for i, n in enumerate(names):
        v = feats_np[:, i]
        try:
            a = roc_auc_score(y_bin, v)
        except ValueError:
            print(f"    {n:<32} AUROC = NaN (degenerate)")
            continue
        a_dir = max(a, 1 - a)
        print(f"    {n:<32} AUROC = {a_dir:.4f}  (raw={a:.4f})")


def correlations(feats_np, names, y_cont):
    print(f"  mean min_eps = {y_cont.mean():.4f}  std = {y_cont.std():.4f}")
    for i, n in enumerate(names):
        v = feats_np[:, i]
        if v.std() == 0:
            print(f"    {n:<32} corr = NaN (degenerate)")
            continue
        c = np.corrcoef(v, y_cont)[0, 1]
        print(f"    {n:<32} corr(min_eps) = {c:+.4f}")


# ----------------------------------- main -----------------------------------
def main():
    print(f"device = {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(f"training CNN for {EPOCHS} epochs on FashionMNIST (seed={SEED})...")
    t0 = time.time()
    model = train(SEED, train_set)
    print(f"training done in {time.time()-t0:.1f}s")

    # stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # logit-geometry features
    geom, _ = compute_logit_features(model, test_x, test_y)
    pix = compute_pixel_features(test_x)

    # restrict to samples the model classifies correctly
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct = preds == test_y
    print(f"clean accuracy = {correct.float().mean().item():.4f}")
    x_c, y_c = test_x[correct], test_y[correct]
    print(f"using {int(correct.sum())} correctly-classified samples")

    feat_names = [
        "logit_l2_norm",
        "angle_top2_to_rest",
        "top3_gap",
        "fraction_of_logits_above_zero",
        "margin",
        "mean_pix",
        "std_pix",
    ]
    feat_cols = [
        geom["logit_l2_norm"],
        geom["angle_top2_to_rest"],
        geom["top3_gap"],
        geom["fraction_of_logits_above_zero"],
        geom["margin"],
        pix["mean_pix"],
        pix["std_pix"],
    ]
    feats = torch.stack(feat_cols, dim=1)
    feats_c = feats[correct]
    feats_np = feats_c.detach().cpu().numpy()

    # ----- attacks -----
    print("\ncomputing FGSM self-flip @ eps=15/255 ...")
    t0 = time.time()
    fgsm = batched_attack(model, x_c, y_c, fgsm_flip, eps=EPS_TEST)
    print(f"  done ({time.time()-t0:.1f}s)  flip rate = {fgsm.float().mean():.4f}")

    print("computing PGD self-flip @ eps=15/255, 10-step ...")
    t0 = time.time()
    pgd = batched_attack(model, x_c, y_c, pgd_flip, eps=EPS_TEST,
                         alpha=PGD_ALPHA, steps=PGD_STEPS)
    print(f"  done ({time.time()-t0:.1f}s)  flip rate = {pgd.float().mean():.4f}")

    print("computing min_eps_FGSM (binary search) ...")
    t0 = time.time()
    me = batched_attack(model, x_c, y_c, min_eps_fgsm)
    print(f"  done ({time.time()-t0:.1f}s)  mean min_eps = {me.mean():.4f}")

    fgsm_np = fgsm.cpu().numpy().astype(int)
    pgd_np = pgd.cpu().numpy().astype(int)
    me_np = me.cpu().numpy()

    # ----- analysis -----
    print("\n========== univariate AUROC: FGSM_self_flip ==========")
    univariate_auroc(feats_np, feat_names, fgsm_np)

    print("\n========== univariate AUROC: PGD_self_flip ==========")
    univariate_auroc(feats_np, feat_names, pgd_np)

    print("\n========== Pearson correlation: min_eps_FGSM ==========")
    correlations(feats_np, feat_names, me_np)

    # Also AUROC for min_eps treated as binary at its median (vulnerable = below median)
    median = np.median(me_np)
    y_me_bin = (me_np <= median).astype(int)
    print(f"\n========== univariate AUROC: min_eps <= median ({median:.4f}) ==========")
    univariate_auroc(feats_np, feat_names, y_me_bin)


if __name__ == "__main__":
    main()
