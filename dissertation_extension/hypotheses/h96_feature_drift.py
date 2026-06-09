"""
H96: Per-sample layer-by-layer feature drift between clean and a small noisy
perturbation (sigma=0.05) predicts FGSM/PGD vulnerability.

Hypothesis:
    For a clean input x and a noisy version x' = x + N(0, sigma^2 I) with
    sigma=0.05, the relative change in each layer's ReLU activations,
        d_l(x) = || relu_l(x) - relu_l(x + noise) || / || relu_l(x) ||,
    correlates with adversarial vulnerability of x. Samples whose internal
    features wander a lot under tiny input noise are intuitively closer to
    decision boundaries, hence easier to flip with FGSM/PGD.

Pipeline:
  1. Train the small CNN matching diagnostic_test.py on Fashion-MNIST for
     10 epochs with Adam (lr=1e-3, batch=128). The CNN exposes 4 ReLU
     activation layers:
        relu1 : F.relu(c1(x))                shape [B, 32, 26, 26]
        relu2 : F.relu(c2(relu1))            shape [B, 64, 24, 24]
        relu3 : F.relu(fc1(post-pool/flat))  shape [B, 128]
        relu4 : pre-softmax logits           shape [B, 10]   (no relu, but final feature layer)
        -- to stay faithful to "4 numbers, one per layer" we use:
           relu1, relu2, relu3 ReLU outputs, and the logits layer as 4th.
  2. For each test sample x (restrict to those correctly classified by the
     final model), draw a single Gaussian noise tensor with sigma=0.05,
     compute x' = clamp(x + noise, 0, 1), forward-pass both x and x',
     extract activations at each of the 4 layers, compute per-sample
     relative L2 drift:
         d_l = || a_l(x) - a_l(x') || / (|| a_l(x) || + eps).
  3. Build the feature vector:
        [drift_l1, drift_l2, drift_l3, drift_l4,
         victim_margin, mean_pix, std_pix]
  4. Compute three vulnerability targets per sample:
        - flipped_FGSM at eps = 15/255
        - flipped_PGD  at eps = 15/255, alpha = 2/255, 20 steps
        - FGSM_min_eps_binary_search (continuous)
  5. Univariate AUROC for each feature against each binary target;
     Spearman correlation with min_eps for the continuous one.

Run:
    python h96_feature_drift.py
Data cached under /tmp/data.
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
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0
SIGMA = 0.05
SEED = 0
EPS_NUM = 1e-8


class CNN(nn.Module):
    """Same architecture as diagnostic_test.py CNN; with intermediate-feature hooks."""

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

    def features(self, x):
        """Return the four per-layer feature tensors used to compute drift.
        relu1, relu2, relu3 (post-fc1), logits."""
        r1 = F.relu(self.c1(x))
        r2 = F.relu(self.c2(r1))
        p = F.max_pool2d(r2, 2).flatten(1)
        r3 = F.relu(self.fc1(p))
        r4 = self.fc2(r3)
        return r1, r2, r3, r4


def train_model(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done in {time.time()-t0:.1f}s")
    model.eval()
    return model


@torch.no_grad()
def clean_predictions(model, x, bs=256):
    softs, preds, margins = [], [], []
    for i in range(0, x.size(0), bs):
        logits = model(x[i:i+bs])
        s = F.softmax(logits, 1)
        sorted_s, _ = s.sort(1, descending=True)
        softs.append(s)
        preds.append(s.argmax(1))
        margins.append(sorted_s[:, 0] - sorted_s[:, 1])
    return torch.cat(softs), torch.cat(preds), torch.cat(margins)


@torch.no_grad()
def per_layer_drift(model, x, sigma=SIGMA, bs=128):
    """Return [N, 4] tensor of per-sample relative L2 drift per layer."""
    N = x.size(0)
    out = torch.zeros(N, 4, device=DEVICE)
    for i in range(0, N, bs):
        xb = x[i:i+bs]
        noise = torch.randn_like(xb) * sigma
        xb_n = (xb + noise).clamp(0, 1)
        fc = model.features(xb)
        fn = model.features(xb_n)
        for l in range(4):
            a = fc[l].flatten(1)
            b = fn[l].flatten(1)
            num = (a - b).norm(dim=1)
            den = a.norm(dim=1) + EPS_NUM
            out[i:i+xb.size(0), l] = num / den
    return out


def fgsm_grad_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS_TEST):
    s = fgsm_grad_sign(model, x, y)
    return (x + eps * s).clamp(0, 1)


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        g = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * g.sign()
        adv = torch.min(torch.max(adv, x0 - eps), x0 + eps).clamp(0, 1)
    return adv.detach()


def min_eps_fgsm_bsearch(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad_sign(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def auroc_safe(y, score):
    if y.std() == 0:
        return float("nan")
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def main():
    print("device:", DEVICE)
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("Training victim CNN ...")
    model = train_model(train_set)

    print("Clean predictions ...")
    soft, pred, margin = clean_predictions(model, test_x)
    correct = (pred == test_y)
    print(f"  clean accuracy = {correct.float().mean().item():.4f}")
    x = test_x[correct]
    y = test_y[correct]
    margin_c = margin[correct]

    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))

    print(f"Computing per-layer feature drift at sigma={SIGMA} ...")
    torch.manual_seed(SEED + 7)
    t0 = time.time()
    drift = per_layer_drift(model, x, sigma=SIGMA)
    print(f"  done in {time.time()-t0:.1f}s  "
          f"mean drift per layer = {[f'{drift[:,l].mean().item():.4f}' for l in range(4)]}")

    # ---- targets ----
    print("FGSM @ eps=15/255 ...")
    flip_fgsm = []
    for i in range(0, x.size(0), 256):
        xb, yb = x[i:i+256], y[i:i+256]
        adv = fgsm_attack(model, xb, yb)
        with torch.no_grad():
            flip_fgsm.append(model(adv).argmax(1) != yb)
    flip_fgsm = torch.cat(flip_fgsm)

    print("PGD ...")
    flip_pgd = []
    for i in range(0, x.size(0), 256):
        xb, yb = x[i:i+256], y[i:i+256]
        adv = pgd_attack(model, xb, yb)
        with torch.no_grad():
            flip_pgd.append(model(adv).argmax(1) != yb)
    flip_pgd = torch.cat(flip_pgd)

    print("FGSM min-eps binary search ...")
    me = []
    for i in range(0, x.size(0), 256):
        me.append(min_eps_fgsm_bsearch(model, x[i:i+256], y[i:i+256]))
    min_eps = torch.cat(me)

    # ---- assemble feature matrix ----
    feat_names = [
        "drift_relu1", "drift_relu2", "drift_relu3", "drift_logits",
        "victim_margin", "mean_pix", "std_pix",
    ]
    feats_np = np.stack([
        drift[:, 0].cpu().numpy(),
        drift[:, 1].cpu().numpy(),
        drift[:, 2].cpu().numpy(),
        drift[:, 3].cpu().numpy(),
        margin_c.cpu().numpy(),
        mean_pix.cpu().numpy(),
        std_pix.cpu().numpy(),
    ], axis=1)

    flip_fgsm_np = flip_fgsm.cpu().numpy().astype(int)
    flip_pgd_np = flip_pgd.cpu().numpy().astype(int)
    min_eps_np = min_eps.cpu().numpy()

    print(f"\nN={feats_np.shape[0]}  "
          f"FGSM flip rate={flip_fgsm_np.mean():.3f}  "
          f"PGD flip rate={flip_pgd_np.mean():.3f}  "
          f"mean min_eps={min_eps_np.mean():.4f}")

    targets_binary = {
        "flipped_FGSM": flip_fgsm_np,
        "flipped_PGD": flip_pgd_np,
    }

    # ---- univariate AUROC ----
    print("\n========== UNIVARIATE AUROC ==========")
    for tname, t in targets_binary.items():
        print(f"\n target: {tname}  (pos rate = {t.mean():.3f})")
        for i, fname in enumerate(feat_names):
            a = auroc_safe(t, feats_np[:, i])
            print(f"   {fname:<18} AUROC = {a:.4f}")

    # continuous target via Spearman
    print(f"\n target: FGSM_min_eps  (Spearman)")
    for i, fname in enumerate(feat_names):
        rho, _ = spearmanr(feats_np[:, i], min_eps_np)
        print(f"   {fname:<18} Spearman = {rho:+.4f}")


if __name__ == "__main__":
    main()
