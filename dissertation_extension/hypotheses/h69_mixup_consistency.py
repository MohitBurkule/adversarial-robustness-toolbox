"""
H69: Per-sample confidence drop under mixup with random partners predicts
adversarial vulnerability.

Hypothesis: Samples near a class boundary will lose true-class confidence
rapidly when mixed with random partners (alpha=0.1). The mean and max drop
in true-class softmax across K=16 random partners should correlate with
adversarial vulnerability (FGSM flip, PGD flip, min-eps).

Pipeline:
  1. Train small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
  2. For each test sample x with true label y:
       - select K=16 random partner indices from the test set
       - form x_mix = 0.9 * x + 0.1 * partner
       - record true-class softmax on x_mix
       - features: mean_drop = p_clean - mean_k(p_mix)
                   max_drop  = p_clean - min_k(p_mix)
  3. Augment with margin, mean_pix, std_pix (per sample).
  4. Targets:
       - FGSM_flip at eps=15/255
       - PGD_flip  at eps=15/255 (10 steps, step=2/255)
       - min_eps   binary search (continuous)
  5. Univariate AUROC for each feature vs each binary target;
     Spearman/Pearson corr against min_eps.

Run on samples Model A classifies correctly.

This file is self-contained: it trains its own model and computes
its own attacks; it does not depend on cached artefacts.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
K_PARTNERS = 16
MIXUP_ALPHA = 0.1   # weight on partner; x_mix = (1 - alpha) * x + alpha * partner
SEED = 0


class CNN(nn.Module):
    """Matches diagnostic_test.py CNN exactly."""
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


def train_model(train_set, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep + 1}/{EPOCHS} done")
    model.eval()
    return model


def model_logits(model, x, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            out.append(model(x[i:i + batch]))
    return torch.cat(out, 0)


def true_class_probs(model, x, y, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            p = F.softmax(model(x[i:i + batch]), 1)
            out.append(p[torch.arange(p.size(0), device=DEVICE), y[i:i + batch]])
    return torch.cat(out, 0)


def fgsm_grad_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack_success(model, x, y, eps=EPS_TEST, batch=512):
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        sign = fgsm_grad_sign(model, xb, yb)
        adv = (xb + eps * sign).clamp(0, 1)
        with torch.no_grad():
            out.append(model(adv).argmax(1) != yb)
    return torch.cat(out, 0)


def pgd_attack_success(model, x, y, eps=EPS_TEST, alpha=2.0 / 255.0, steps=10, batch=512):
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        x0 = xb.clone().detach()
        adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
        adv = adv.clamp(0, 1).detach()
        for _ in range(steps):
            adv.requires_grad_(True)
            loss = F.cross_entropy(model(adv), yb)
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + alpha * grad.sign()
            adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
        with torch.no_grad():
            out.append(model(adv).argmax(1) != yb)
    return torch.cat(out, 0)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15, batch=512):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        sign = fgsm_grad_sign(model, xb, yb)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xb + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
            with torch.no_grad():
                flipped = model(adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        out.append(hi)
    return torch.cat(out, 0)


def compute_mixup_features(model, test_x, test_y, K=K_PARTNERS, alpha=MIXUP_ALPHA,
                           batch=256, seed=SEED):
    """For each test sample, mix with K random partners drawn from test_x.

    Returns:
        mean_drop: p_clean - mean over K of p_mix on the true class
        max_drop:  p_clean - min  over K of p_mix on the true class
    """
    N = test_x.size(0)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # baseline clean true-class probability
    p_clean = true_class_probs(model, test_x, test_y, batch=512)

    # accumulators
    sum_pmix = torch.zeros(N, device=DEVICE)
    min_pmix = torch.full((N,), float("inf"), device=DEVICE)

    for k in range(K):
        # sample partner indices uniformly (with replacement) — re-roll any
        # collisions with self so the mix actually injects a different image
        partner_idx = torch.randint(0, N, (N,), generator=gen)
        self_idx = torch.arange(N)
        collide = partner_idx == self_idx
        while collide.any():
            partner_idx[collide] = torch.randint(0, N, (int(collide.sum()),),
                                                 generator=gen)
            collide = partner_idx == self_idx
        partner_idx = partner_idx.to(DEVICE)

        # compute mixed batch and true-class probs in chunks
        for i in range(0, N, batch):
            j = min(i + batch, N)
            x_self = test_x[i:j]
            x_part = test_x[partner_idx[i:j]]
            x_mix = ((1.0 - alpha) * x_self + alpha * x_part).clamp(0, 1)
            with torch.no_grad():
                p = F.softmax(model(x_mix), 1)
                p_true = p[torch.arange(p.size(0), device=DEVICE), test_y[i:j]]
            sum_pmix[i:j] += p_true
            min_pmix[i:j] = torch.minimum(min_pmix[i:j], p_true)

    mean_pmix = sum_pmix / K
    mean_drop = p_clean - mean_pmix
    max_drop = p_clean - min_pmix
    return mean_drop, max_drop, p_clean


def compute_margin(model, test_x, batch=512):
    logits = model_logits(model, test_x, batch=batch)
    s, _ = logits.sort(1, descending=True)
    return s[:, 0] - s[:, 1]


def univariate_auroc(feat, y_bin):
    """AUROC, oriented to the better direction."""
    f = feat.detach().cpu().numpy()
    yb = y_bin.detach().cpu().numpy().astype(int)
    if yb.std() == 0:
        return float("nan")
    a = roc_auc_score(yb, f)
    return max(a, 1 - a)


def main():
    print("==== H69: mixup-consistency vulnerability predictor ====")
    print(f"device={DEVICE}  EPOCHS={EPOCHS}  K={K_PARTNERS}  alpha={MIXUP_ALPHA}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(" training CNN ...")
    t0 = time.time()
    model = train_model(train_set, seed=SEED)
    print(f"  training done ({time.time() - t0:.1f}s)")

    # build full test tensors on device
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)
    print(f" test set: N={N}")

    # restrict to correctly-classified samples
    with torch.no_grad():
        preds = model_logits(model, test_x).argmax(1)
    correct = preds == test_y
    print(f" correctly classified: {int(correct.sum())} / {N}")
    x_c = test_x[correct]
    y_c = test_y[correct]

    # --- feature extraction ---
    print(" computing mixup features ...")
    t0 = time.time()
    mean_drop, max_drop, p_clean = compute_mixup_features(
        model, x_c, y_c, K=K_PARTNERS, alpha=MIXUP_ALPHA, seed=SEED)
    print(f"  done ({time.time() - t0:.1f}s)")
    print(f"  mean_drop:  mean={mean_drop.mean():.4f}  std={mean_drop.std():.4f}")
    print(f"  max_drop:   mean={max_drop.mean():.4f}   std={max_drop.std():.4f}")

    print(" computing margin / mean_pix / std_pix ...")
    margin = compute_margin(model, x_c)
    mean_pix = x_c.flatten(1).mean(1)
    std_pix = x_c.flatten(1).std(1)

    feats = torch.stack([mean_drop, max_drop, margin, p_clean, mean_pix, std_pix], 1)
    feat_names = ["mixup_mean_drop", "mixup_max_drop", "margin",
                  "p_clean", "mean_pix", "std_pix"]

    # --- targets ---
    print(" computing FGSM attack success ...")
    t0 = time.time()
    fgsm_flip = fgsm_attack_success(model, x_c, y_c, eps=EPS_TEST)
    print(f"  done ({time.time() - t0:.1f}s)  pos rate={fgsm_flip.float().mean():.3f}")

    print(" computing PGD attack success ...")
    t0 = time.time()
    pgd_flip = pgd_attack_success(model, x_c, y_c, eps=EPS_TEST)
    print(f"  done ({time.time() - t0:.1f}s)  pos rate={pgd_flip.float().mean():.3f}")

    print(" computing min_eps_to_flip ...")
    t0 = time.time()
    min_eps = min_eps_to_flip(model, x_c, y_c)
    print(f"  done ({time.time() - t0:.1f}s)  mean={min_eps.mean():.4f}")

    # --- univariate AUROCs (binary targets) ---
    binary_targets = [("FGSM_flip", fgsm_flip), ("PGD_flip", pgd_flip)]
    print("\n==== Univariate AUROC ====")
    print(f"{'feature':<20}" + "".join(f"{tn:>14}" for tn, _ in binary_targets))
    for i, fn in enumerate(feat_names):
        row = f"{fn:<20}"
        for _, tv in binary_targets:
            a = univariate_auroc(feats[:, i], tv)
            row += f"{a:>14.4f}"
        print(row)

    # --- correlation against continuous min_eps ---
    print("\n==== Correlation with min_eps_to_flip ====")
    me_np = min_eps.detach().cpu().numpy()
    print(f"{'feature':<20}{'pearson':>12}{'spearman':>12}")
    for i, fn in enumerate(feat_names):
        f_np = feats[:, i].detach().cpu().numpy()
        if np.std(f_np) == 0:
            pr, sr = float("nan"), float("nan")
        else:
            pr = float(np.corrcoef(f_np, me_np)[0, 1])
            sr = float(spearmanr(f_np, me_np).correlation)
        print(f"{fn:<20}{pr:>+12.4f}{sr:>+12.4f}")

    # also: AUROC of features against "low min_eps" (bottom-quartile = most vulnerable)
    print("\n==== Univariate AUROC vs 'low min_eps' (bottom quartile) ====")
    q = float(np.quantile(me_np, 0.25))
    low_eps = torch.as_tensor(me_np <= q, device=DEVICE)
    print(f"  threshold (25th pct) = {q:.4f}   pos rate = {low_eps.float().mean():.3f}")
    print(f"{'feature':<20}{'AUROC':>12}")
    for i, fn in enumerate(feat_names):
        a = univariate_auroc(feats[:, i], low_eps)
        print(f"{fn:<20}{a:>12.4f}")

    print("\n==== H69 done ====")


if __name__ == "__main__":
    main()
