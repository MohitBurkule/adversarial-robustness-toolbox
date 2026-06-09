"""
Hypothesis H41: L0 (sparse) adversarial attacks — perturb only K pixels —
give a different per-sample vulnerability ranking than L_inf FGSM.

Saliency-concentrated samples (high-entropy gradient -> low spatial entropy
i.e. saliency mass in few pixels) should be easier under L0; smooth samples
should be harder.

References:
  - Papernot et al. 2016, "The Limitations of Deep Learning in Adversarial Settings"
    (JSMA — Jacobian Saliency Map Attack).
  - Modas et al. 2019, "SparseFool".
  - Croce & Hein 2019, "Sparse and Imperceivable Adversarial Attacks".

Pipeline:
  1. Train a small CNN on Fashion-MNIST (10 epochs).
  2. L0 attack: for each sample, compute the gradient of the loss w.r.t. the
     input, identify top-K pixels by |grad|, and set each of those pixels to
     0 or 1 (whichever sign(grad) suggests increases the loss). Sweep
     K in {1, 5, 10, 20, 50, 100}, recording the minimum K that flips each
     sample (= 1e9 if never flips).
  3. Per-sample features: victim_margin, mean_pix, std_pix, sobel_mean,
     saliency_entropy (Shannon entropy of |grad| normalised to a prob. dist.).
  4. Targets: min_K (continuous), flipped_at_K10 (binary), flipped_FGSM_eps15.
  5. Univariate AUROC and Spearman correlation of each feature with min_K.
  6. Confusion: count samples flipped by L0-K=10 but not FGSM-eps=15/255 and
     vice versa.

Self-contained: run with `python h41_sparse_l0.py`.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
K_SWEEP = [1, 5, 10, 20, 50, 100]
K_BINARY = 10
EPS_FGSM = 15.0 / 255.0
EVAL_BATCH = 256
N_EVAL = 2000   # subset of test set for the expensive L0 sweep


# ---------------------------------------------------------------------- model
class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2); x = self.do1(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.do2(x); return self.fc2(x)


def train(model, train_loader):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done")
    model.eval()


# ----------------------------------------------------------------- attacks
def input_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y, reduction="sum")
    grad, = torch.autograd.grad(loss, x)
    return grad.detach()


def fgsm_flip(model, x, y, eps=EPS_FGSM):
    g = input_grad(model, x, y)
    adv = (x + eps * g.sign()).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def l0_attack(model, x, y, K):
    """Sparse L0 attack:
       1) compute grad of loss w.r.t. input,
       2) pick top-K pixels by |grad|,
       3) set each of those pixels to 1 if grad>0 else 0 (loss-increasing
          extreme). Returns the adversarial image and a flipped-mask.
    """
    g = input_grad(model, x, y)               # (B,1,H,W)
    B, C, H, W = x.shape
    flat = g.view(B, -1)
    abs_flat = flat.abs()
    # top-K indices
    _, idx = abs_flat.topk(K, dim=1)
    # target values: 1 if grad>0 else 0
    sign_top = torch.gather(flat, 1, idx).sign()
    target_vals = (sign_top > 0).float()      # 1 or 0
    adv = x.clone().view(B, -1)
    adv.scatter_(1, idx, target_vals)
    adv = adv.view(B, C, H, W).clamp(0, 1)
    with torch.no_grad():
        flipped = (model(adv).argmax(1) != y)
    return adv, flipped


def min_K_to_flip(model, x, y, K_list):
    """For each sample, return the smallest K in K_list that flips it
       (or a sentinel K_max+1 if none does)."""
    sentinel = max(K_list) + 1
    out = torch.full((x.size(0),), sentinel, dtype=torch.long, device=DEVICE)
    flipped_at_each = {}
    for K in sorted(K_list):
        _, flipped = l0_attack(model, x, y, K)
        flipped_at_each[K] = flipped
        # record first K at which a sample flipped
        new_flip = flipped & (out == sentinel)
        out[new_flip] = K
    return out, flipped_at_each


# ----------------------------------------------------------------- features
def sobel_mean(x):
    """Mean |sobel| edge response per image."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = (gx ** 2 + gy ** 2).sqrt()
    return mag.mean(dim=(1, 2, 3))


def saliency_entropy(model, x, y):
    """Shannon entropy of normalised |grad_x|.
       Low entropy -> mass concentrated on few pixels (saliency-concentrated).
       High entropy -> spread out (smooth)."""
    g = input_grad(model, x, y).abs()
    B = g.size(0)
    flat = g.view(B, -1)
    s = flat.sum(1, keepdim=True).clamp(min=1e-12)
    p = flat / s
    eps = 1e-12
    H = -(p * (p + eps).log()).sum(1)
    return H


def victim_margin(model, x, y):
    with torch.no_grad():
        logits = model(x)
    true_logit = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), -1e9)
    runner = masked.max(1).values
    return true_logit - runner


def compute_features(model, x, y):
    """Process in batches for memory."""
    N = x.size(0)
    margins, sals, sobels = [], [], []
    means, stds = [], []
    for i in range(0, N, EVAL_BATCH):
        xb = x[i:i+EVAL_BATCH]
        yb = y[i:i+EVAL_BATCH]
        margins.append(victim_margin(model, xb, yb))
        sals.append(saliency_entropy(model, xb, yb))
        sobels.append(sobel_mean(xb))
        means.append(xb.mean(dim=(1, 2, 3)))
        stds.append(xb.std(dim=(1, 2, 3)))
    return {
        "victim_margin": torch.cat(margins).cpu().numpy(),
        "saliency_entropy": torch.cat(sals).cpu().numpy(),
        "sobel_mean": torch.cat(sobels).cpu().numpy(),
        "mean_pix": torch.cat(means).cpu().numpy(),
        "std_pix": torch.cat(stds).cpu().numpy(),
    }


# ----------------------------------------------------------------- main
def main():
    print(f"device={DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)

    print("Training CNN ...")
    torch.manual_seed(0); np.random.seed(0)
    model = CNN().to(DEVICE)
    t0 = time.time()
    train(model, train_loader)
    print(f"  done in {time.time()-t0:.1f}s")

    # ----- gather a subset of correctly-classified test samples
    print("Selecting test subset ...")
    xs, ys = [], []
    with torch.no_grad():
        for i in range(len(test_set)):
            xs.append(test_set[i][0])
            ys.append(test_set[i][1])
            if len(xs) >= 4000:
                break
    x_all = torch.stack(xs).to(DEVICE)
    y_all = torch.tensor(ys, device=DEVICE)
    with torch.no_grad():
        preds = []
        for i in range(0, x_all.size(0), EVAL_BATCH):
            preds.append(model(x_all[i:i+EVAL_BATCH]).argmax(1))
        preds = torch.cat(preds)
    correct_mask = (preds == y_all)
    x_c = x_all[correct_mask][:N_EVAL]
    y_c = y_all[correct_mask][:N_EVAL]
    print(f"  using {x_c.size(0)} correctly-classified samples")

    # ----- features
    print("Computing per-sample features ...")
    feats = compute_features(model, x_c, y_c)

    # ----- L0 attack sweep
    print(f"Running L0 attack sweep K in {K_SWEEP} ...")
    t0 = time.time()
    min_K_chunks, flip_at_K_chunks = [], {K: [] for K in K_SWEEP}
    for i in range(0, x_c.size(0), EVAL_BATCH):
        xb, yb = x_c[i:i+EVAL_BATCH], y_c[i:i+EVAL_BATCH]
        m, per = min_K_to_flip(model, xb, yb, K_SWEEP)
        min_K_chunks.append(m)
        for K in K_SWEEP:
            flip_at_K_chunks[K].append(per[K])
    min_K = torch.cat(min_K_chunks).cpu().numpy()
    flip_at_K = {K: torch.cat(flip_at_K_chunks[K]).cpu().numpy().astype(int)
                 for K in K_SWEEP}
    print(f"  L0 sweep done in {time.time()-t0:.1f}s")
    sentinel = max(K_SWEEP) + 1
    success_rate_by_K = {K: flip_at_K[K].mean() for K in K_SWEEP}
    print("  cumulative flip-rate by K:")
    for K in K_SWEEP:
        print(f"    K={K:>4}  flipped={success_rate_by_K[K]:.3f}")
    print(f"  fraction never flipped (K>{max(K_SWEEP)}): "
          f"{(min_K == sentinel).mean():.3f}")

    # ----- FGSM target
    print(f"Running FGSM at eps={EPS_FGSM:.4f} ...")
    fgsm_chunks = []
    for i in range(0, x_c.size(0), EVAL_BATCH):
        fgsm_chunks.append(fgsm_flip(model, x_c[i:i+EVAL_BATCH], y_c[i:i+EVAL_BATCH]))
    fgsm_flipped = torch.cat(fgsm_chunks).cpu().numpy().astype(int)
    print(f"  FGSM flip rate: {fgsm_flipped.mean():.3f}")

    # ----- evaluate
    feat_names = ["victim_margin", "mean_pix", "std_pix",
                  "sobel_mean", "saliency_entropy"]
    print("\n===== Univariate AUROC =====")
    targets = {
        f"flipped_at_K{K_BINARY}": flip_at_K[K_BINARY],
        "flipped_FGSM_eps15": fgsm_flipped,
    }
    for tname, y_bin in targets.items():
        if y_bin.std() == 0:
            print(f"target {tname}: degenerate (rate={y_bin.mean():.3f}); skipping")
            continue
        print(f"\n-- target: {tname}  (pos rate={y_bin.mean():.3f}) --")
        for fn in feat_names:
            v = feats[fn]
            a = roc_auc_score(y_bin, v); a = max(a, 1 - a)
            print(f"  AUROC  {fn:<20} {a:.4f}")

    print("\n===== Spearman correlation with min_K (lower = more vulnerable) =====")
    # restrict to samples that did get flipped at some K (else min_K is sentinel)
    flipped_any = min_K < sentinel
    print(f"  using {flipped_any.sum()} samples flipped within K<={max(K_SWEEP)}")
    for fn in feat_names:
        rho, p = spearmanr(feats[fn][flipped_any], min_K[flipped_any])
        print(f"  Spearman(min_K, {fn:<20}) = {rho:+.4f}  (p={p:.2e})")

    # ----- confusion matrix L0 vs FGSM
    print("\n===== L0 (K={}) vs FGSM (eps={:.4f}) confusion =====".format(
        K_BINARY, EPS_FGSM))
    l0 = flip_at_K[K_BINARY].astype(bool)
    fg = fgsm_flipped.astype(bool)
    both = (l0 & fg).sum()
    only_l0 = (l0 & ~fg).sum()
    only_fg = (~l0 & fg).sum()
    neither = (~l0 & ~fg).sum()
    n = l0.size
    print(f"  both flipped:        {both}  ({both/n:.3f})")
    print(f"  L0 only (sparse):    {only_l0}  ({only_l0/n:.3f})")
    print(f"  FGSM only (L_inf):   {only_fg}  ({only_fg/n:.3f})")
    print(f"  neither flipped:     {neither}  ({neither/n:.3f})")
    # Cohen-style overlap: how aligned are the two rankings?
    from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
    kappa = cohen_kappa_score(l0.astype(int), fg.astype(int))
    mcc = matthews_corrcoef(l0.astype(int), fg.astype(int))
    print(f"  Cohen's kappa(L0 flipped, FGSM flipped): {kappa:+.4f}")
    print(f"  Matthews corr(L0 flipped, FGSM flipped): {mcc:+.4f}")

    # ----- key question: does saliency_entropy predict L0 differently than L_inf?
    print("\n===== Saliency entropy: differential AUROC L0 vs FGSM =====")
    se = feats["saliency_entropy"]
    a_l0 = roc_auc_score(flip_at_K[K_BINARY], se); a_l0 = max(a_l0, 1 - a_l0)
    a_fg = roc_auc_score(fgsm_flipped, se); a_fg = max(a_fg, 1 - a_fg)
    print(f"  AUROC(saliency_entropy -> L0_K{K_BINARY}):      {a_l0:.4f}")
    print(f"  AUROC(saliency_entropy -> FGSM_eps15):    {a_fg:.4f}")
    print(f"  Delta (L0 - FGSM):                         {a_l0 - a_fg:+.4f}")
    # Spearman with min_K (continuous) vs FGSM is binary, so we compare a
    # direction-consistent score: rank-correlation of saliency_entropy with
    # -min_K  (more vulnerable = lower min_K, higher -min_K).
    rho_se_l0, _ = spearmanr(se[flipped_any], -min_K[flipped_any])
    print(f"  Spearman(saliency_entropy, -min_K) on flipped-only = {rho_se_l0:+.4f}")

    # ----- save numeric outputs
    out = {
        "feat_names": feat_names,
        "feats": feats,
        "min_K": min_K,
        "flip_at_K": flip_at_K,
        "fgsm_flipped": fgsm_flipped,
        "K_sweep": K_SWEEP,
        "K_binary": K_BINARY,
        "eps_fgsm": EPS_FGSM,
    }
    np.savez("/tmp/data/h41_sparse_l0_results.npz",
             **{k: np.asarray(v) if not isinstance(v, dict) else v
                for k, v in out.items() if k != "flip_at_K" and k != "feats"},
             **{f"flip_at_K_{K}": flip_at_K[K] for K in K_SWEEP},
             **{f"feat_{k}": v for k, v in feats.items()})
    print("\nSaved arrays to /tmp/data/h41_sparse_l0_results.npz")


if __name__ == "__main__":
    main()
