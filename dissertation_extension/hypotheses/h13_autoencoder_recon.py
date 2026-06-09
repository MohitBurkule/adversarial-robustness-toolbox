"""
Hypothesis H13: Per-sample reconstruction error from a small convolutional
autoencoder (trained on the dataset, no labels used) predicts adversarial
vulnerability.

Atypical / out-of-distribution-ish samples should reconstruct worse and be
more adversarially vulnerable. The autoencoder feature is "model-free of
labels" - it never sees the class labels, so any predictive signal it carries
about adversarial vulnerability is a property of the input distribution
itself, not of the victim's decision boundary.

Pipeline:
  1. Train a small CNN victim on Fashion-MNIST (10 epochs Adam, mirrors the
     architecture in diagnostic_test.py).
  2. Separately train a small convolutional autoencoder on the same training
     set (4 epochs, MSE loss). Encoder: conv32 -> conv64 -> conv128 with
     maxpool; bottleneck dense=64; decoder mirrors with ConvTranspose. No
     labels are used anywhere.
  3. Per test sample, compute reconstruction MSE.
  4. Baseline features:
        - victim_margin   (top1 - top2 logit gap from the victim model)
        - mean_pix        (per-image mean pixel value)
        - std_pix         (per-image pixel std)
        - recon_err       (per-image MSE from the autoencoder)
  5. Vulnerability targets:
        - flipped_FGSM    (victim FGSM at eps=15/255)
        - flipped_PGD     (victim PGD-10 at eps=15/255, alpha=eps/4)
        - FGSM_min_eps    (per-sample binary search smallest L_inf eps that
                          flips FGSM; continuous, vulnerability = -min_eps)
  6. Univariate AUROC of each feature for each binary target, and Spearman /
     Pearson correlation with min_eps. Multivariate: does recon_err add over
     margin alone, and over the image-stat baseline (mean_pix, std_pix)?

Self-contained; only needs torch, torchvision, sklearn, scipy, numpy. Data is
downloaded to /tmp/data. Trains on CUDA.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

DEVICE = torch.device("cuda")
DATA_ROOT = "/tmp/data"
EPS_TEST = 15.0 / 255.0
VICTIM_EPOCHS = 10
AE_EPOCHS = 4
BATCH = 128
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    """Victim CNN, mirrors diagnostic_test.py."""

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


class ConvAutoencoder(nn.Module):
    """Small conv autoencoder for 28x28 single-channel images.

    Encoder: 28 -> conv32 -> pool -> 14 -> conv64 -> pool -> 7 -> conv128 -> 7
    Bottleneck dense (128*7*7 -> 64 -> 128*7*7).
    Decoder mirrors: 7 -> conv64 -> upsample -> 14 -> conv32 -> upsample -> 28
    -> conv1. Sigmoid output (pixels in [0,1]).
    """

    def __init__(self, bottleneck=64):
        super().__init__()
        # encoder
        self.e1 = nn.Conv2d(1, 32, 3, padding=1)       # 28x28
        self.e2 = nn.Conv2d(32, 64, 3, padding=1)      # 14x14
        self.e3 = nn.Conv2d(64, 128, 3, padding=1)     # 7x7
        self.enc_fc = nn.Linear(128 * 7 * 7, bottleneck)
        # decoder
        self.dec_fc = nn.Linear(bottleneck, 128 * 7 * 7)
        self.d1 = nn.Conv2d(128, 64, 3, padding=1)     # 7x7
        self.d2 = nn.Conv2d(64, 32, 3, padding=1)      # 14x14
        self.d3 = nn.Conv2d(32, 16, 3, padding=1)      # 28x28
        self.d_out = nn.Conv2d(16, 1, 3, padding=1)    # 28x28

    def encode(self, x):
        x = F.relu(self.e1(x))
        x = F.max_pool2d(x, 2)                         # -> 14
        x = F.relu(self.e2(x))
        x = F.max_pool2d(x, 2)                         # -> 7
        x = F.relu(self.e3(x))                         # 7
        x = x.flatten(1)
        z = self.enc_fc(x)
        return z

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.d1(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")    # 14
        x = F.relu(self.d2(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")    # 28
        x = F.relu(self.d3(x))
        x = torch.sigmoid(self.d_out(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_victim(train_set, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(VICTIM_EPOCHS):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  victim epoch {ep+1}/{VICTIM_EPOCHS}  loss={tot/n:.4f}  "
              f"({time.time()-t0:.1f}s)")
    model.eval()
    return model


def train_autoencoder(train_set, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # NB: train_set returns (image, label) but we DO NOT use the label
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    ae = ConvAutoencoder().to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    for ep in range(AE_EPOCHS):
        ae.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, _ in loader:                      # label discarded
            x = x.to(DEVICE)
            opt.zero_grad()
            recon = ae(x)
            loss = F.mse_loss(recon, x)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  AE   epoch {ep+1}/{AE_EPOCHS}    mse={tot/n:.6f}  "
              f"({time.time()-t0:.1f}s)")
    ae.eval()
    return ae


# ---------------------------------------------------------------------------
# Per-sample features
# ---------------------------------------------------------------------------

def victim_margin(model, x, batch=512):
    margins, preds = [], []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            logits = model(x[i:i+batch])
            srt, _ = logits.sort(1, descending=True)
            margins.append(srt[:, 0] - srt[:, 1])
            preds.append(logits.argmax(1))
    return torch.cat(margins), torch.cat(preds)


def recon_error(ae, x, batch=512):
    errs = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            xb = x[i:i+batch]
            r = ae(xb)
            e = ((r - xb) ** 2).flatten(1).mean(1)
            errs.append(e)
    return torch.cat(errs)


def pixel_stats(x):
    flat = x.flatten(1)
    return flat.mean(1), flat.std(1)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST, batch=512):
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        sign = fgsm_grad(model, xb, yb)
        adv = (xb + eps * sign).clamp(0, 1)
        with torch.no_grad():
            out.append(model(adv).argmax(1) != yb)
    return torch.cat(out)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS,
             batch=256):
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        delta = torch.empty_like(xb).uniform_(-eps, eps)
        adv = (xb + delta).clamp(0, 1).detach()
        for _ in range(steps):
            adv.requires_grad_(True)
            F.cross_entropy(model(adv), yb).backward()
            with torch.no_grad():
                adv = adv + alpha * adv.grad.sign()
                adv = torch.max(torch.min(adv, xb + eps), xb - eps).clamp(0, 1)
            adv = adv.detach()
        with torch.no_grad():
            out.append(model(adv).argmax(1) != yb)
    return torch.cat(out)


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15, batch=512):
    """Binary search per-sample smallest eps that flips FGSM. Direction is
    fixed at the eps=0 grad sign (standard FGSM)."""
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        sign = fgsm_grad(model, xb, yb)
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xb + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
            with torch.no_grad():
                flipped = model(adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        out.append(hi)
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def auroc_both_dirs(y, score):
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def evaluate(feats, feat_names, targets_bin, target_names, min_eps):
    feats_np = feats.cpu().numpy()
    Xs = StandardScaler().fit_transform(feats_np)
    print(f"\n--- univariate AUROC ---")
    print(f"{'feature':<15} " + "  ".join(f"{t:>22}" for t in target_names))
    for i, fname in enumerate(feat_names):
        row = f"{fname:<15} "
        for tgt in targets_bin:
            y = tgt.cpu().numpy().astype(int)
            if y.std() == 0:
                row += f"{'(degenerate)':>22}  "
            else:
                row += f"{auroc_both_dirs(y, feats_np[:, i]):>22.4f}  "
        print(row)

    print(f"\n--- correlation with FGSM_min_eps (vulnerability = -min_eps) ---")
    me = min_eps.cpu().numpy()
    for i, fname in enumerate(feat_names):
        sp, _ = spearmanr(feats_np[:, i], me)
        pe, _ = pearsonr(feats_np[:, i], me)
        print(f"  {fname:<15}  spearman={sp:+.4f}   pearson={pe:+.4f}")

    # Multivariate: does recon_err add over (a) margin only, (b) image stats only?
    margin_idx = feat_names.index("victim_margin")
    recon_idx = feat_names.index("recon_err")
    mean_idx = feat_names.index("mean_pix")
    std_idx = feat_names.index("std_pix")

    print(f"\n--- multivariate: does recon_err add information? ---")
    print(f"{'target':<22} {'margin':>8} {'+recon':>8} {'d_recon':>8}  "
          f"{'stats':>8} {'+recon':>8} {'d_recon':>8}  "
          f"{'all':>8}")
    for tgt, tname in zip(targets_bin, target_names):
        y = tgt.cpu().numpy().astype(int)
        if y.std() == 0:
            continue
        # margin only
        a_m = roc_auc_score(y, LogisticRegression(max_iter=2000)
                            .fit(Xs[:, [margin_idx]], y)
                            .predict_proba(Xs[:, [margin_idx]])[:, 1])
        # margin + recon
        cols = [margin_idx, recon_idx]
        a_mr = roc_auc_score(y, LogisticRegression(max_iter=2000)
                             .fit(Xs[:, cols], y)
                             .predict_proba(Xs[:, cols])[:, 1])
        # image stats only
        cols_s = [mean_idx, std_idx]
        a_s = roc_auc_score(y, LogisticRegression(max_iter=2000)
                            .fit(Xs[:, cols_s], y)
                            .predict_proba(Xs[:, cols_s])[:, 1])
        # image stats + recon
        cols_sr = [mean_idx, std_idx, recon_idx]
        a_sr = roc_auc_score(y, LogisticRegression(max_iter=2000)
                             .fit(Xs[:, cols_sr], y)
                             .predict_proba(Xs[:, cols_sr])[:, 1])
        # all
        a_all = roc_auc_score(y, LogisticRegression(max_iter=2000)
                              .fit(Xs, y)
                              .predict_proba(Xs)[:, 1])
        print(f"{tname:<22} {a_m:>8.4f} {a_mr:>8.4f} {a_mr-a_m:>+8.4f}  "
              f"{a_s:>8.4f} {a_sr:>8.4f} {a_sr-a_s:>+8.4f}  {a_all:>8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== H13: autoencoder reconstruction error as vulnerability proxy ===")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True,
                                      transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True,
                                     transform=tf)

    print("\n[1/4] training victim CNN")
    t0 = time.time()
    victim = train_victim(train_set, seed=0)
    print(f"  victim trained in {time.time()-t0:.1f}s")

    print("\n[2/4] training convolutional autoencoder (NO LABELS)")
    t0 = time.time()
    ae = train_autoencoder(train_set, seed=1)
    print(f"  AE trained in {time.time()-t0:.1f}s")

    # materialise test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n[3/4] computing per-sample features")
    margin, preds = victim_margin(victim, test_x)
    correct = preds == test_y
    print(f"  victim test accuracy = {correct.float().mean().item():.4f}")
    # restrict to correctly-classified samples (vulnerability is meaningful there)
    x_c = test_x[correct]
    y_c = test_y[correct]
    margin_c = margin[correct]
    print(f"  using {x_c.size(0)} correctly-classified test samples")

    recon = recon_error(ae, x_c)
    mean_pix, std_pix = pixel_stats(x_c)
    feats = torch.stack([margin_c, mean_pix, std_pix, recon], 1)
    feat_names = ["victim_margin", "mean_pix", "std_pix", "recon_err"]

    # sanity: recon-error summary stats
    print(f"  recon_err     mean={recon.mean():.5f}  std={recon.std():.5f}  "
          f"min={recon.min():.5f}  max={recon.max():.5f}")
    print(f"  victim_margin mean={margin_c.mean():.4f} std={margin_c.std():.4f}")

    print("\n[4/4] running attacks (FGSM, PGD, min-eps FGSM binary search)")
    t0 = time.time()
    flipped_fgsm = fgsm_flip(victim, x_c, y_c, eps=EPS_TEST)
    print(f"  FGSM flip rate eps={EPS_TEST:.4f}: "
          f"{flipped_fgsm.float().mean():.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    flipped_pgd = pgd_flip(victim, x_c, y_c, eps=EPS_TEST,
                           alpha=PGD_ALPHA, steps=PGD_STEPS)
    print(f"  PGD-{PGD_STEPS} flip rate eps={EPS_TEST:.4f}: "
          f"{flipped_pgd.float().mean():.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    min_eps = min_eps_fgsm(victim, x_c, y_c)
    print(f"  min_eps FGSM   mean={min_eps.mean():.4f}  "
          f"median={min_eps.median():.4f}  ({time.time()-t0:.1f}s)")

    targets_bin = [flipped_fgsm, flipped_pgd]
    target_names = ["flipped_FGSM", "flipped_PGD"]

    print("\n========== RESULTS ==========")
    evaluate(feats, feat_names, targets_bin, target_names, min_eps)
    print("\n=== H13 done ===")


if __name__ == "__main__":
    main()
