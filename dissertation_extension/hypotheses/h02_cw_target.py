"""
H02: Carlini-Wagner (CW) attack as the vulnerability label gives a different
ranking of samples than FGSM, and image statistics may predict CW-flip
differently than FGSM-flip.

Prior result: image-statistics + final-logit margin predict FGSM-flip on
Fashion-MNIST at AUROC ~0.92. CW is a stronger attack that finds smaller
perturbations; the per-sample vulnerability ranking may differ, which would
mean a "vulnerability predictor" tuned on FGSM does not generalise to CW.

Pipeline
--------
1. Train a small CNN victim (same arch as diagnostic_test.py) on
   Fashion-MNIST for 10 epochs with Adam.
2. For every correctly-classified test sample compute:
     - CW (L2) perturbation magnitude  (via torchattacks if available,
       else a self-contained CW-L2 implementation)
     - CW success at a fixed L2 budget
     - FGSM L_inf success at eps = 15/255
     - FGSM minimum-eps-to-flip (via binary search on grad-sign direction)
3. Compute per-sample features:
     victim_margin, mean_pix, std_pix, sobel_mean, jpeg_q75
4. Univariate AUROC table: each feature  ->  P(CW-flip) and P(FGSM-flip).
5. Spearman rank-correlation between CW-perturbation-magnitude and
   FGSM-min-eps  (does the attack pick the same "easy" samples?).

Run:  python h02_cw_target.py
"""

from __future__ import annotations

import io
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr
from PIL import Image

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/tmp/data"
BATCH = 128
EPOCHS = 10
FGSM_EPS = 15.0 / 255.0
CW_L2_BUDGET = 1.5         # success threshold on L2 perturbation norm
CW_STEPS = 200
CW_LR = 5e-3
CW_C = 1.0                 # weight on adversarial loss in CW objective
CW_KAPPA = 0.0             # confidence margin in CW
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# model (matches diagnostic_test.py)
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, n: int = 10):
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


def train_victim(train_set):
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        print(f"  epoch {ep+1:2d}/{EPOCHS}  loss={running/len(train_set):.4f}  "
              f"({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# attacks
# ---------------------------------------------------------------------------
def fgsm_attack(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    adv = (x + eps * x.grad.sign()).clamp(0, 1).detach()
    return adv


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    x_ = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_), y).backward()
    sign = x_.grad.sign().detach()
    best = torch.full((x.size(0),), eps_max + 1.0, device=DEVICE)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            pred = model(adv).argmax(1)
        flipped = pred != y
        best = torch.where(flipped & (mid < best), mid, best)
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return best  # eps_max+1 if never flipped


def _try_torchattacks_cw(model, x, y):
    """Return (adv, used_torchattacks_bool)."""
    try:
        import torchattacks  # type: ignore
    except Exception:
        return None, False
    atk = torchattacks.CW(model, c=CW_C, kappa=CW_KAPPA,
                          steps=CW_STEPS, lr=CW_LR)
    adv = atk(x, y)
    return adv.detach(), True


def cw_l2_attack(model, x, y,
                 steps=CW_STEPS, lr=CW_LR, c=CW_C, kappa=CW_KAPPA):
    """
    Self-contained Carlini-Wagner L2 attack (untargeted).
    Reference: Carlini & Wagner, "Towards Evaluating the Robustness of NN", 2017.

    Optimises  ||delta||_2^2 + c * f(x+delta)
    with the change-of-variables  x_adv = 0.5*(tanh(w)+1)  so x_adv in [0,1].
    f(x') = max( Z(x')_y - max_{i!=y} Z(x')_i , -kappa )  for untargeted.
    """
    x = x.clone().detach()
    y = y.clone().detach()
    N = x.size(0)

    # invert tanh map; clamp to avoid atanh(+-1)
    x_clamped = x.clamp(1e-6, 1 - 1e-6)
    w = torch.atanh(2 * x_clamped - 1).detach().requires_grad_(True)

    opt = torch.optim.Adam([w], lr=lr)
    one_hot = F.one_hot(y, num_classes=10).float()

    best_l2 = torch.full((N,), float("inf"), device=DEVICE)
    best_adv = x.clone()

    for step in range(steps):
        adv = 0.5 * (torch.tanh(w) + 1)
        delta = adv - x
        l2 = (delta.view(N, -1) ** 2).sum(1)

        logits = model(adv)
        real = (one_hot * logits).sum(1)
        other = ((1 - one_hot) * logits - one_hot * 1e4).max(1).values
        f = torch.clamp(real - other, min=-kappa)
        loss = (l2 + c * f).sum()

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = logits.argmax(1)
            succ = pred != y
            improve = succ & (l2 < best_l2)
            best_l2 = torch.where(improve, l2, best_l2)
            if improve.any():
                best_adv[improve] = adv[improve]

    return best_adv.detach(), best_l2.sqrt().detach()  # return L2 norm


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
SOBEL_X = torch.tensor([[[[-1., 0., 1.],
                          [-2., 0., 2.],
                          [-1., 0., 1.]]]])
SOBEL_Y = torch.tensor([[[[-1., -2., -1.],
                          [ 0.,  0.,  0.],
                          [ 1.,  2.,  1.]]]])


def victim_margin(model, x):
    """top-1 logit minus top-2 logit."""
    with torch.no_grad():
        logits = []
        for i in range(0, x.size(0), 512):
            logits.append(model(x[i:i+512]))
        logits = torch.cat(logits, 0)
    s, _ = logits.sort(1, descending=True)
    return (s[:, 0] - s[:, 1]).cpu().numpy()


def image_stats(x):
    """mean_pix, std_pix, sobel_mean — all on CPU/GPU tensors."""
    x_cpu = x.detach()
    mean_pix = x_cpu.mean(dim=(1, 2, 3)).cpu().numpy()
    std_pix = x_cpu.std(dim=(1, 2, 3)).cpu().numpy()

    sx = SOBEL_X.to(x_cpu.device)
    sy = SOBEL_Y.to(x_cpu.device)
    gx = F.conv2d(x_cpu, sx, padding=1)
    gy = F.conv2d(x_cpu, sy, padding=1)
    sobel_mean = torch.sqrt(gx ** 2 + gy ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    return mean_pix, std_pix, sobel_mean


def jpeg_q75_size(x):
    """JPEG-compressed byte size at quality 75 — proxy for image complexity."""
    x_np = (x.detach().cpu().numpy() * 255.0).astype(np.uint8)  # (N,1,28,28)
    sizes = np.zeros(x_np.shape[0], dtype=np.float32)
    for i, arr in enumerate(x_np):
        img = Image.fromarray(arr[0], mode="L")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        sizes[i] = buf.tell()
    return sizes


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print(f"device: {DEVICE}")
    os.makedirs(DATA_DIR, exist_ok=True)

    tfm = T.ToTensor()
    train_set = torchvision.datasets.FashionMNIST(DATA_DIR, train=True,
                                                  download=True, transform=tfm)
    test_set = torchvision.datasets.FashionMNIST(DATA_DIR, train=False,
                                                 download=True, transform=tfm)

    print("training victim CNN ...")
    model = train_victim(train_set)

    # stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # filter to correctly classified samples (vulnerability only defined there)
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds, 0)
    correct = preds == test_y
    print(f"clean accuracy: {correct.float().mean().item():.4f}  "
          f"({correct.sum().item()} / {test_x.size(0)})")

    x = test_x[correct]
    y = test_y[correct]
    N = x.size(0)
    print(f"working set: {N} samples")

    # ----- FGSM ------------------------------------------------------------
    print("running FGSM (fixed eps) ...")
    fgsm_flip = np.zeros(N, dtype=np.float32)
    fgsm_min = np.zeros(N, dtype=np.float32)
    BS = 256
    for i in range(0, N, BS):
        xb, yb = x[i:i+BS], y[i:i+BS]
        adv = fgsm_attack(model, xb, yb, FGSM_EPS)
        with torch.no_grad():
            p = model(adv).argmax(1)
        fgsm_flip[i:i+BS] = (p != yb).cpu().numpy()
        fgsm_min[i:i+BS] = fgsm_min_eps(model, xb, yb).cpu().numpy()

    print(f"  FGSM eps={FGSM_EPS:.4f}  flip-rate = {fgsm_flip.mean():.4f}")

    # ----- CW --------------------------------------------------------------
    # Probe torchattacks once; if available use it for the whole run.
    print("running CW-L2 attack ...")
    probe_adv, use_ta = _try_torchattacks_cw(model, x[:4], y[:4])
    if use_ta:
        print("  using torchattacks.CW")
        import torchattacks
        atk = torchattacks.CW(model, c=CW_C, kappa=CW_KAPPA,
                              steps=CW_STEPS, lr=CW_LR)
    else:
        print("  using local CW-L2 implementation")

    cw_l2 = np.zeros(N, dtype=np.float32)
    cw_flip = np.zeros(N, dtype=np.float32)
    BS_CW = 128
    for i in range(0, N, BS_CW):
        xb, yb = x[i:i+BS_CW], y[i:i+BS_CW]
        if use_ta:
            adv = atk(xb, yb).detach()
            delta = adv - xb
            l2 = delta.view(xb.size(0), -1).norm(dim=1)
            with torch.no_grad():
                p = model(adv).argmax(1)
            succ = (p != yb)
            # samples not flipped -> set L2 to inf (will be treated as worst case)
            l2 = torch.where(succ, l2, torch.full_like(l2, float("inf")))
        else:
            adv, l2 = cw_l2_attack(model, xb, yb)
            with torch.no_grad():
                p = model(adv).argmax(1)
            succ = (p != yb)
        cw_l2[i:i+BS_CW] = l2.cpu().numpy()
        cw_flip[i:i+BS_CW] = (succ & (l2 <= CW_L2_BUDGET)).cpu().numpy()
        if (i // BS_CW) % 5 == 0:
            print(f"    [{i+xb.size(0)}/{N}]  "
                  f"cum CW-flip @ L2<={CW_L2_BUDGET}: "
                  f"{cw_flip[:i+xb.size(0)].mean():.4f}")

    finite = np.isfinite(cw_l2)
    print(f"  CW success rate (any L2): {finite.mean():.4f}")
    print(f"  CW-flip @ L2<={CW_L2_BUDGET}: {cw_flip.mean():.4f}")
    if finite.any():
        print(f"  median CW-L2 (successful): {np.median(cw_l2[finite]):.4f}")

    # ----- features --------------------------------------------------------
    print("computing image-stat features ...")
    margin = victim_margin(model, x)
    mean_pix, std_pix, sobel_mean = image_stats(x)
    jpeg = jpeg_q75_size(x)

    feats = {
        "victim_margin": margin,
        "mean_pix": mean_pix,
        "std_pix": std_pix,
        "sobel_mean": sobel_mean,
        "jpeg_q75": jpeg,
    }

    # ----- AUROC table -----------------------------------------------------
    # AUROC measures predictive value of feature for being flipped.
    # margin should be NEGATIVELY correlated with flip-prob, so we also report
    # AUROC of -margin (orientation just inverts auroc around 0.5).
    print()
    print("=" * 78)
    print(f"{'feature':<16s}  {'AUROC FGSM-flip':>16s}  {'AUROC CW-flip':>16s}")
    print("-" * 78)
    rows = []
    for name, vals in feats.items():
        # If a feature is constant within a class roc_auc_score errors; guard.
        try:
            a_fgsm = roc_auc_score(fgsm_flip.astype(int), vals)
        except ValueError:
            a_fgsm = float("nan")
        try:
            a_cw = roc_auc_score(cw_flip.astype(int), vals)
        except ValueError:
            a_cw = float("nan")
        # report the "better-oriented" auroc (max(a, 1-a)) plus sign
        def fmt(a):
            if math.isnan(a):
                return "   n/a "
            return f"{max(a, 1-a):.4f}{'+' if a >= 0.5 else '-'}"
        print(f"{name:<16s}  {fmt(a_fgsm):>16s}  {fmt(a_cw):>16s}")
        rows.append((name, a_fgsm, a_cw))
    print("=" * 78)
    print("  (suffix '+' means higher value -> more likely flipped;")
    print("   '-' means lower value -> more likely flipped)")

    # ----- ranking comparison ---------------------------------------------
    print()
    print("ranking comparison: CW-L2 vs FGSM-min-eps")
    # restrict to samples where BOTH attacks found a perturbation
    mask = np.isfinite(cw_l2) & (fgsm_min <= 0.3)
    print(f"  samples where both attacks succeed: {mask.sum()} / {N}")
    if mask.sum() > 10:
        rho, p = spearmanr(cw_l2[mask], fgsm_min[mask])
        r, _ = pearsonr(cw_l2[mask], fgsm_min[mask])
        print(f"  Spearman rho  (CW-L2 vs FGSM-min-eps): {rho:+.4f}  (p={p:.2e})")
        print(f"  Pearson  r    (CW-L2 vs FGSM-min-eps): {r:+.4f}")
    else:
        print("  too few overlapping samples — skipped")

    # AUROC: does CW-L2 predict FGSM-flip and vice versa?
    try:
        a = roc_auc_score(fgsm_flip.astype(int), -cw_l2_for_auroc(cw_l2))
        print(f"  AUROC( CW-L2 -> FGSM-flip ):  {a:.4f}")
    except ValueError:
        pass
    try:
        a = roc_auc_score(cw_flip.astype(int), -fgsm_min)
        print(f"  AUROC( FGSM-min-eps -> CW-flip ): {a:.4f}")
    except ValueError:
        pass

    # ----- save raw arrays for downstream analysis -------------------------
    out = {
        "fgsm_flip": fgsm_flip,
        "fgsm_min_eps": fgsm_min,
        "cw_l2": cw_l2,
        "cw_flip": cw_flip,
        **feats,
    }
    out_path = os.path.join(os.path.dirname(__file__), "h02_cw_target_results.npz")
    np.savez(out_path, **out)
    print(f"\nsaved per-sample arrays -> {out_path}")


def cw_l2_for_auroc(cw_l2):
    """Replace inf with a large finite value so sklearn accepts the array."""
    out = cw_l2.copy()
    finite_max = np.nanmax(out[np.isfinite(out)]) if np.isfinite(out).any() else 1.0
    out[~np.isfinite(out)] = finite_max * 10.0
    return out


if __name__ == "__main__":
    main()
