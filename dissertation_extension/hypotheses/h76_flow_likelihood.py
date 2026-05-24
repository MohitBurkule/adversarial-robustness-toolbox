"""
Hypothesis H76: Per-sample log-likelihood from a small RealNVP normalizing
flow (trained on Fashion-MNIST images, no labels) predicts adversarial
vulnerability.

Intuition: samples that lie in low-density regions of the data distribution
should be both (a) atypical and (b) more adversarially vulnerable. A
normalising flow provides an exact, tractable log p(x), which is a sharper
density estimator than a reconstruction-based proxy (cf. H13).

Pipeline:
  1. Train a small CNN victim on Fashion-MNIST (10 epochs Adam, mirrors the
     architecture in diagnostic_test.py).
  2. Separately train a small RealNVP flow on the same training set
     (5 epochs). The flow has 8 affine coupling layers operating on the
     flattened 28*28 = 784-dim input, alternating checkerboard-style binary
     masks. Each coupling layer's s/t network is a small MLP. No labels are
     used anywhere.
  3. Per test sample, compute log p(x) under the flow.
  4. Features:
        - victim_margin       (top1 - top2 logit gap)
        - mean_pix            (per-image mean pixel value)
        - std_pix             (per-image pixel std)
        - flow_log_likelihood (exact log p(x) under the flow)
  5. Vulnerability targets:
        - flipped_FGSM   (FGSM at eps=15/255)
        - flipped_PGD    (PGD-10 at eps=15/255, alpha=eps/4)
        - FGSM_min_eps   (per-sample binary search smallest L_inf eps that
                          flips FGSM; vulnerability = -min_eps)
  6. Univariate AUROC of each feature for each binary target, plus Spearman
     and Pearson correlation with min_eps. Multivariate ablation: does the
     flow log-likelihood add information over (a) margin alone and (b) the
     image-statistic baseline (mean_pix, std_pix)?

Notes on the flow:
  - We dequantise inputs (uniform noise in [0, 1/256)) and apply a logit
    transform with alpha=0.05, accumulating the Jacobian. This is the
    standard RealNVP preprocessing for [0,1] image pixels and avoids the
    flow having to model a sharp boundary at 0/1.
  - Affine couplings use the stable parameterisation s = tanh(log_s_raw)
    so the log-determinant per layer is sum(s * mask_complement).
  - Web-search-informed (RealNVP PyTorch reference implementations such as
    ANLGBOY/RealNVP-with-PyTorch and the codegenes.net guide); kept small
    (~8 coupling layers, 256-hidden MLP s/t nets) so 5 epochs is feasible
    on a single GPU in a few minutes.

Self-contained; only needs torch, torchvision, sklearn, scipy, numpy. Data
is downloaded to /tmp/data. Trains on CUDA.
"""
import math
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
FLOW_EPOCHS = 5
BATCH = 128
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0

IMG_DIM = 28 * 28
N_COUPLING = 8
HIDDEN = 256
LOGIT_ALPHA = 0.05   # standard RealNVP logit-transform alpha


# ---------------------------------------------------------------------------
# Victim model
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


# ---------------------------------------------------------------------------
# RealNVP flow
# ---------------------------------------------------------------------------

class STNet(nn.Module):
    """Small MLP producing scale (log_s) and translation (t) for a coupling
    layer. Both outputs have the full input dimensionality; the caller
    multiplies by the mask complement so that only the 'passive' half is
    actually transformed.
    """

    def __init__(self, dim=IMG_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * dim),
        )
        # zero-init last layer -> coupling layer is identity at start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        h = self.net(x)
        log_s, t = h.chunk(2, dim=1)
        # stabilise scale
        log_s = torch.tanh(log_s)
        return log_s, t


class AffineCoupling(nn.Module):
    """Affine coupling layer with a fixed binary mask.

    Forward (training, x -> z):
        passive = x * mask                          # untouched
        log_s, t = STNet(passive)
        active  = (1 - mask)
        z = passive + active * (x * exp(log_s) + t)
        log_det = sum(active * log_s)
    """

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.st = STNet(dim=mask.numel())

    def forward(self, x):
        passive = x * self.mask
        log_s, t = self.st(passive)
        active = 1.0 - self.mask
        log_s = log_s * active
        t = t * active
        z = passive + active * (x * torch.exp(log_s) + t)
        log_det = log_s.sum(dim=1)
        return z, log_det


def make_checkerboard_mask(dim, parity):
    """1-D 'checkerboard' mask over the flattened 28*28 vector. Parity 0
    masks even indices, parity 1 masks odd indices.
    """
    m = torch.arange(dim) % 2
    if parity == 0:
        m = 1 - m
    return m.float()


class RealNVP(nn.Module):
    """RealNVP-style flow: standard normal base, N affine coupling layers
    with alternating checkerboard masks, plus logit-pre-transform.

    log p(x) = log p(z) + sum_layers log|det df/dx|
              + log|det d(logit_transform)/dx|
    """

    def __init__(self, dim=IMG_DIM, n_coupling=N_COUPLING,
                 logit_alpha=LOGIT_ALPHA):
        super().__init__()
        self.dim = dim
        self.logit_alpha = logit_alpha
        layers = []
        for i in range(n_coupling):
            mask = make_checkerboard_mask(dim, parity=i % 2)
            layers.append(AffineCoupling(mask))
        self.layers = nn.ModuleList(layers)

    def logit_transform(self, x):
        """Map x in [0,1] -> R via y = logit(alpha + (1-2 alpha) x).
        Returns transformed y and log|det dy/dx|.
        """
        a = self.logit_alpha
        u = a + (1 - 2 * a) * x
        y = torch.log(u) - torch.log1p(-u)
        # dy/dx = (1 - 2 alpha) / (u (1-u))
        log_det = (math.log(1 - 2 * a)
                   - torch.log(u) - torch.log1p(-u)).sum(dim=1)
        return y, log_det

    def log_prob(self, x_img):
        """x_img: (B, 1, 28, 28) in [0,1]. Returns per-sample log p(x).

        Includes dequantisation in the caller (we expect x_img already to
        be a continuous, dequantised value in (0,1)).
        """
        B = x_img.size(0)
        x = x_img.view(B, -1)
        y, ld_logit = self.logit_transform(x)
        log_det = ld_logit
        z = y
        for layer in self.layers:
            z, ld = layer(z)
            log_det = log_det + ld
        # standard-normal base
        log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.dim * math.log(2 * math.pi)
        return log_pz + log_det


def dequantise(x):
    """Add uniform noise in [0, 1/256) to inputs already in [0,1] from
    ToTensor (which is x/255). After this, values are in (0, 1) almost
    surely, suitable for the logit transform.
    """
    return (x * 255.0 + torch.rand_like(x)) / 256.0


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


def train_flow(train_set, seed=2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    flow = RealNVP().to(DEVICE)
    opt = torch.optim.Adam(flow.parameters(), lr=5e-4)
    for ep in range(FLOW_EPOCHS):
        flow.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, _ in loader:                          # label discarded
            x = x.to(DEVICE)
            x = dequantise(x)
            opt.zero_grad()
            log_p = flow.log_prob(x)
            loss = -log_p.mean()
            loss.backward()
            # mild grad clip - flows can spike early in training
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 50.0)
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        # nats per dim for interpretability
        npd = (tot / n) / IMG_DIM
        print(f"  flow epoch {ep+1}/{FLOW_EPOCHS}  -log p={tot/n:.2f} "
              f"({npd:.4f} nats/dim)  ({time.time()-t0:.1f}s)")
    flow.eval()
    return flow


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


def flow_log_likelihood(flow, x, batch=256, n_dequant=4):
    """Average log p(x) over a few dequantisation noise draws to reduce
    variance. Returns per-sample log-likelihood in nats.
    """
    accum = torch.zeros(x.size(0), device=DEVICE)
    with torch.no_grad():
        for _ in range(n_dequant):
            for i in range(0, x.size(0), batch):
                xb = dequantise(x[i:i+batch])
                accum[i:i+batch] += flow.log_prob(xb)
    return accum / n_dequant


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
    """Binary-search per-sample smallest eps that flips FGSM (direction =
    eps=0 gradient sign).
    """
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
    print(f"{'feature':<22} " + "  ".join(f"{t:>22}" for t in target_names))
    for i, fname in enumerate(feat_names):
        row = f"{fname:<22} "
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
        print(f"  {fname:<22}  spearman={sp:+.4f}   pearson={pe:+.4f}")

    margin_idx = feat_names.index("victim_margin")
    flow_idx = feat_names.index("flow_log_likelihood")
    mean_idx = feat_names.index("mean_pix")
    std_idx = feat_names.index("std_pix")

    print(f"\n--- multivariate: does flow_log_likelihood add information? ---")
    print(f"{'target':<22} {'margin':>8} {'+flow':>8} {'d_flow':>8}  "
          f"{'stats':>8} {'+flow':>8} {'d_flow':>8}  "
          f"{'all':>8}")
    for tgt, tname in zip(targets_bin, target_names):
        y = tgt.cpu().numpy().astype(int)
        if y.std() == 0:
            continue
        a_m = roc_auc_score(y, LogisticRegression(max_iter=2000)
                            .fit(Xs[:, [margin_idx]], y)
                            .predict_proba(Xs[:, [margin_idx]])[:, 1])
        cols = [margin_idx, flow_idx]
        a_mf = roc_auc_score(y, LogisticRegression(max_iter=2000)
                             .fit(Xs[:, cols], y)
                             .predict_proba(Xs[:, cols])[:, 1])
        cols_s = [mean_idx, std_idx]
        a_s = roc_auc_score(y, LogisticRegression(max_iter=2000)
                            .fit(Xs[:, cols_s], y)
                            .predict_proba(Xs[:, cols_s])[:, 1])
        cols_sf = [mean_idx, std_idx, flow_idx]
        a_sf = roc_auc_score(y, LogisticRegression(max_iter=2000)
                             .fit(Xs[:, cols_sf], y)
                             .predict_proba(Xs[:, cols_sf])[:, 1])
        a_all = roc_auc_score(y, LogisticRegression(max_iter=2000)
                              .fit(Xs, y)
                              .predict_proba(Xs)[:, 1])
        print(f"{tname:<22} {a_m:>8.4f} {a_mf:>8.4f} {a_mf-a_m:>+8.4f}  "
              f"{a_s:>8.4f} {a_sf:>8.4f} {a_sf-a_s:>+8.4f}  {a_all:>8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== H76: normalising-flow log-likelihood as vulnerability proxy ===")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True,
                                      transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True,
                                     transform=tf)

    print("\n[1/4] training victim CNN")
    t0 = time.time()
    victim = train_victim(train_set, seed=0)
    print(f"  victim trained in {time.time()-t0:.1f}s")

    print("\n[2/4] training RealNVP flow (NO LABELS)")
    t0 = time.time()
    flow = train_flow(train_set, seed=2)
    print(f"  flow trained in {time.time()-t0:.1f}s")

    # materialise test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n[3/4] computing per-sample features")
    margin, preds = victim_margin(victim, test_x)
    correct = preds == test_y
    print(f"  victim test accuracy = {correct.float().mean().item():.4f}")
    x_c = test_x[correct]
    y_c = test_y[correct]
    margin_c = margin[correct]
    print(f"  using {x_c.size(0)} correctly-classified test samples")

    log_p = flow_log_likelihood(flow, x_c)
    mean_pix, std_pix = pixel_stats(x_c)
    feats = torch.stack([margin_c, mean_pix, std_pix, log_p], 1)
    feat_names = ["victim_margin", "mean_pix", "std_pix", "flow_log_likelihood"]

    print(f"  flow_log_likelihood mean={log_p.mean():.2f}  std={log_p.std():.2f}  "
          f"min={log_p.min():.2f}  max={log_p.max():.2f}  "
          f"({log_p.mean().item()/IMG_DIM:.4f} nats/dim avg)")
    print(f"  victim_margin       mean={margin_c.mean():.4f} std={margin_c.std():.4f}")

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
    print("\n=== H76 done ===")


if __name__ == "__main__":
    main()
