"""
Hypothesis H48: Feature Squeezing (Xu et al., NDSS 2018) as a per-sample
adversarial detector. Feature squeezing reduces input bit-depth and applies
local smoothing, then measures whether the model's softmax distribution
agrees with itself on the squeezed input. Large disagreement (in L1) signals
that the original input was likely adversarial.

We test the per-sample detection AUROC of Feature Squeezing on FGSM
(eps=15/255) against Fashion-MNIST, and compare against simple model-free
image statistics (mean_pix, std_pix, sobel_mean) plus the victim's margin.
Question: does feature squeezing, which is a model-aware detector, do
meaningfully better than these (mostly model-free) baselines that should
NOT distinguish clean from adversarial inputs?

Pipeline:
  1. Train small CNN victim on Fashion-MNIST (10 epochs Adam, same
     architecture as diagnostic_test.py).
  2. Build a balanced detection pool: take the first N/2 test samples
     unchanged (clean, label=0) and the next N/2 perturbed with FGSM at
     eps=15/255 (adversarial, label=1).
  3. Two squeezers:
       - bit-depth reduction to 3 bits per pixel
       - 3x3 median filter
     For each input x, compute s_b = bitdepth3(x), s_m = median3x3(x), and
     measure L1 distance between softmax(model(x)) and softmax(model(s_b))
     and softmax(model(s_m)). Per-sample FS score:
       fs_score = max(L1_bitdepth, L1_median)
  4. Baseline per-sample scores:
       - victim_margin = top1 - top2 logit on x
       - mean_pix, std_pix
       - sobel_mean: mean magnitude of Sobel gradients on x
  5. Targets: is_adversarial in {0,1}, 50/50 mix.
  6. Univariate AUROC for each score against is_adversarial.

Notes:
  - On Fashion-MNIST images are single-channel 28x28 in [0,1].
  - Bit-depth-3 quantises pixel intensities to 8 levels.
  - Median filter is applied per-image with a 3x3 kernel and reflect
    padding.
  - All inputs are clamped to [0,1] after FGSM.

Tools: PyTorch + CUDA. Data root /tmp/data.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
SEED = 0


# ----------------------------- model ---------------------------------------

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


# ----------------------------- training ------------------------------------

def train_victim():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tfm = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)

    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS}  {time.time()-t0:.1f}s")
    model.eval()
    return model, test_set


# ----------------------------- attack --------------------------------------

def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    adv = (x + eps * x.grad.sign()).clamp(0, 1).detach()
    return adv


# ----------------------------- squeezers -----------------------------------

def bit_depth_reduce(x, bits=3):
    """Reduce bit depth of x in [0,1] to `bits` bits/channel."""
    levels = (1 << bits) - 1  # 7 for 3 bits
    return torch.round(x * levels) / levels


def median_filter_3x3(x):
    """3x3 median filter with reflect padding, applied per (N,C,H,W)."""
    pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
    patches = pad.unfold(2, 3, 1).unfold(3, 3, 1)  # (N,C,H,W,3,3)
    patches = patches.contiguous().view(*patches.shape[:4], 9)
    return patches.median(dim=-1).values


def softmax_logits(model, x, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(F.softmax(model(x[i:i+bs]), dim=1))
    return torch.cat(out, 0)


def all_logits(model, x, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i+bs]))
    return torch.cat(out, 0)


# ----------------------------- sobel ---------------------------------------

def sobel_mean(x):
    """Per-image mean of Sobel gradient magnitude. x: (N,1,H,W) in [0,1]."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag.mean(dim=(1, 2, 3))


# ----------------------------- analysis ------------------------------------

def safe_auroc(y, score):
    y = np.asarray(y); score = np.asarray(score)
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, score)


# ----------------------------- main ----------------------------------------

def main():
    print(f"device: {DEVICE}")
    print("training victim ...")
    model, test_set = train_victim()

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)
    print(f"test set: {N}")

    # build a 50/50 mix: first half clean, second half FGSM'd
    half = N // 2
    clean_part = test_x[:half]
    clean_y = test_y[:half]
    adv_src = test_x[half:half + half]
    adv_src_y = test_y[half:half + half]

    print(f"generating FGSM (eps={EPS_TEST:.4f}) on {adv_src.size(0)} samples ...")
    adv_part = torch.zeros_like(adv_src)
    for i in range(0, adv_src.size(0), 512):
        xb = adv_src[i:i+512]; yb = adv_src_y[i:i+512]
        adv_part[i:i+512] = fgsm(model, xb, yb, EPS_TEST)

    pool_x = torch.cat([clean_part, adv_part], 0)
    is_adv = np.concatenate([np.zeros(clean_part.size(0), dtype=np.int64),
                             np.ones(adv_part.size(0), dtype=np.int64)])
    print(f"pool size: {pool_x.size(0)}  positives (adv): {is_adv.sum()}")

    # sanity: FGSM success rate on the adv half
    with torch.no_grad():
        adv_pred = all_logits(model, adv_part).argmax(1)
    flip_rate = (adv_pred != adv_src_y).float().mean().item()
    print(f"FGSM flip rate on adv half: {flip_rate:.4f}")

    # softmax on original
    print("computing softmax on original / bitdepth / median ...")
    p_orig = softmax_logits(model, pool_x)

    # squeezed versions
    x_bd = bit_depth_reduce(pool_x, bits=3)
    x_md = median_filter_3x3(pool_x)

    p_bd = softmax_logits(model, x_bd)
    p_md = softmax_logits(model, x_md)

    l1_bd = (p_orig - p_bd).abs().sum(dim=1).detach().cpu().numpy()
    l1_md = (p_orig - p_md).abs().sum(dim=1).detach().cpu().numpy()
    fs_score = np.maximum(l1_bd, l1_md)

    # victim margin on original
    logits_orig = all_logits(model, pool_x)
    sorted_l, _ = logits_orig.sort(1, descending=True)
    margin = (sorted_l[:, 0] - sorted_l[:, 1]).detach().cpu().numpy()

    # model-free image stats on pool inputs (post-attack, since attack
    # may shift pixel statistics)
    mean_pix = pool_x.mean(dim=(1, 2, 3)).detach().cpu().numpy()
    std_pix = pool_x.std(dim=(1, 2, 3)).detach().cpu().numpy()
    sobel = sobel_mean(pool_x).detach().cpu().numpy()

    feats = {
        "FS_max(bitdepth,median)": fs_score,
        "FS_L1_bitdepth3":          l1_bd,
        "FS_L1_median3x3":          l1_md,
        "victim_margin":            margin,
        "mean_pix":                 mean_pix,
        "std_pix":                  std_pix,
        "sobel_mean":               sobel,
    }

    print("\n=== Univariate detection AUROC (target: is_adversarial) ===")
    print(f"positive_rate={is_adv.mean():.3f}")
    for fname, fv in feats.items():
        a_pos = safe_auroc(is_adv, fv)
        a_neg = safe_auroc(is_adv, -fv)
        a = max(a_pos, a_neg)
        sign = "+" if a_pos >= a_neg else "-"
        print(f"  {fname:>26s}  AUROC={a:.4f}  (sign={sign})")

    print("\nDone.")


if __name__ == "__main__":
    main()
