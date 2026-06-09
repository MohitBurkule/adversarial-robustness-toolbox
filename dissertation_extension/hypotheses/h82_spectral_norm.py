"""
H82: Spectral-norm-regularised training (Miyato et al. 2018) changes the
per-sample vulnerability predictor ranking.

We train two victims on Fashion-MNIST for 10 epochs using the small CNN from
diagnostic_test.py:
  - vanilla
  - spectral-norm-constrained (each Linear/Conv2d wrapped with
    torch.nn.utils.spectral_norm to constrain layer Lipschitz constants)

Per-sample features computed on the test set from each victim:
  - margin     (top1 - top2 logit on the clean input)
  - mean_pix   (mean pixel intensity of the input)
  - std_pix    (std  pixel intensity of the input)
  - sobel_mean (mean absolute Sobel-filtered intensity, an edge-energy proxy)

Per-sample vulnerability targets per victim:
  - FGSM    : flipped by FGSM at eps=15/255
  - PGD     : flipped by 20-step PGD at eps=15/255, alpha=2/255
  - min_eps : per-sample binary-search smallest L_inf eps that flips FGSM

For each (victim, target) we compute the AUROC of every feature, and report
how the per-feature ranking differs between vanilla and spectral-norm victims.

DO NOT RUN automatically — written for offline execution.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
PGD_ALPHA = 2.0 / 255.0
PGD_STEPS = 20
EPOCHS = 10
BATCH = 128


# --------------------------------------------------------------------------- #
# Architecture (matches diagnostic_test.py; optionally spectral-normed)
# --------------------------------------------------------------------------- #
class CNN(nn.Module):
    def __init__(self, n=10, sn=False):
        super().__init__()
        wrap = spectral_norm if sn else (lambda m: m)
        self.c1 = wrap(nn.Conv2d(1, 32, 3))
        self.c2 = wrap(nn.Conv2d(32, 64, 3))
        self.fc1 = wrap(nn.Linear(64 * 12 * 12, 128))
        self.fc2 = wrap(nn.Linear(128, n))
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


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_victim(seed, train_set, use_sn, n_classes=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(n_classes, sn=use_sn).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"   epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Per-sample features
# --------------------------------------------------------------------------- #
SOBEL_X = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).view(1, 1, 3, 3)


def compute_features(model, x):
    """Returns dict of 1-D tensors length N: margin, mean_pix, std_pix, sobel_mean."""
    N = x.size(0)
    margins = []
    with torch.no_grad():
        for i in range(0, N, 512):
            logits = model(x[i:i + 512])
            sorted_logits, _ = logits.sort(1, descending=True)
            margins.append(sorted_logits[:, 0] - sorted_logits[:, 1])
    margin = torch.cat(margins)

    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))

    sx = SOBEL_X.to(x.device)
    sy = SOBEL_Y.to(x.device)
    sobel_means = []
    with torch.no_grad():
        for i in range(0, N, 512):
            chunk = x[i:i + 512]
            gx = F.conv2d(chunk, sx, padding=1)
            gy = F.conv2d(chunk, sy, padding=1)
            sobel_means.append((gx.abs() + gy.abs()).mean(dim=(1, 2, 3)))
    sobel_mean = torch.cat(sobel_means)

    return {"margin": margin,
            "mean_pix": mean_pix,
            "std_pix": std_pix,
            "sobel_mean": sobel_mean}


# --------------------------------------------------------------------------- #
# Attacks
# --------------------------------------------------------------------------- #
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x_orig = x.clone().detach()
    # random start in eps ball
    adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps to flip via FGSM."""
    lo = torch.zeros(x.size(0), device=x.device)
    hi = torch.full((x.size(0),), eps_max, device=x.device)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# --------------------------------------------------------------------------- #
# Driver per victim
# --------------------------------------------------------------------------- #
def evaluate_victim(model, x_full, y_full, label):
    # restrict to samples this victim classifies correctly (otherwise
    # vulnerability is trivially 1 / target is degenerate)
    with torch.no_grad():
        preds = []
        for i in range(0, x_full.size(0), 512):
            preds.append(model(x_full[i:i + 512]).argmax(1))
        preds = torch.cat(preds)
    correct = preds == y_full
    x, y = x_full[correct], y_full[correct]
    print(f"   [{label}] using {correct.sum().item()}/{x_full.size(0)} correctly-classified test samples")

    feats = compute_features(model, x)

    # FGSM
    fgsm_flip = []
    for i in range(0, x.size(0), 512):
        fgsm_flip.append(fgsm_attack(model, x[i:i + 512], y[i:i + 512]))
    fgsm_flip = torch.cat(fgsm_flip)

    # PGD
    pgd_flip = []
    for i in range(0, x.size(0), 512):
        pgd_flip.append(pgd_attack(model, x[i:i + 512], y[i:i + 512]))
    pgd_flip = torch.cat(pgd_flip)

    # min_eps
    me = []
    for i in range(0, x.size(0), 512):
        me.append(min_eps_to_flip(model, x[i:i + 512], y[i:i + 512]))
    min_eps = torch.cat(me)

    targets = {"FGSM": fgsm_flip.long().cpu().numpy(),
               "PGD": pgd_flip.long().cpu().numpy(),
               "min_eps": min_eps.cpu().numpy()}  # continuous; lower => more vulnerable

    feat_np = {k: v.cpu().numpy() for k, v in feats.items()}

    print(f"\n   [{label}] FGSM flip rate = {targets['FGSM'].mean():.3f}   "
          f"PGD flip rate = {targets['PGD'].mean():.3f}   "
          f"mean min_eps = {targets['min_eps'].mean():.4f}")

    results = {}  # results[target][feature] = auroc
    for t_name, t_vals in targets.items():
        if t_name == "min_eps":
            # convert to binary: vulnerable = below median min_eps
            med = np.median(t_vals)
            y_bin = (t_vals <= med).astype(int)
        else:
            y_bin = t_vals
        if y_bin.std() == 0:
            print(f"   [{label}] target {t_name} degenerate, skipping")
            continue
        results[t_name] = {}
        for f_name, f_vals in feat_np.items():
            a = roc_auc_score(y_bin, f_vals)
            a = max(a, 1 - a)
            results[t_name][f_name] = a
    return results


def print_comparison(res_vanilla, res_sn):
    feat_order = ["margin", "mean_pix", "std_pix", "sobel_mean"]
    targets = [t for t in ["FGSM", "PGD", "min_eps"] if t in res_vanilla and t in res_sn]
    print("\n=================  PER-FEATURE AUROC COMPARISON  =================")
    print(f"{'target':<10} {'feature':<14} {'vanilla':>9} {'spec_norm':>11} {'delta':>8}")
    for t in targets:
        for f in feat_order:
            v = res_vanilla[t][f]
            s = res_sn[t][f]
            print(f"{t:<10} {f:<14} {v:>9.4f} {s:>11.4f} {s - v:>+8.4f}")
        # ranking comparison
        rank_v = sorted(feat_order, key=lambda k: -res_vanilla[t][k])
        rank_s = sorted(feat_order, key=lambda k: -res_sn[t][k])
        print(f"   ranking [vanilla   ]: {rank_v}")
        print(f"   ranking [spec_norm ]: {rank_s}")
        print(f"   ranking-changed? {rank_v != rank_s}")
        print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    x_test = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("# Training victim 1 / 2: VANILLA")
    t0 = time.time()
    m_vanilla = train_victim(seed=0, train_set=train_set, use_sn=False)
    print(f"  vanilla trained in {time.time()-t0:.1f}s")

    print("\n# Training victim 2 / 2: SPECTRAL-NORM")
    t0 = time.time()
    m_sn = train_victim(seed=0, train_set=train_set, use_sn=True)
    print(f"  spectral-norm trained in {time.time()-t0:.1f}s")

    print("\n# Evaluating VANILLA")
    res_vanilla = evaluate_victim(m_vanilla, x_test, y_test, label="vanilla")
    print("\n# Evaluating SPECTRAL-NORM")
    res_sn = evaluate_victim(m_sn, x_test, y_test, label="spec_norm")

    print_comparison(res_vanilla, res_sn)


if __name__ == "__main__":
    main()
