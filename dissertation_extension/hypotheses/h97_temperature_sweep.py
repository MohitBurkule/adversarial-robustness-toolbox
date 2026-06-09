"""
H97: Temperature sweep of softmax_max as adversarial vulnerability indicator.

Hypothesis: How a sample's softmax responds to a temperature sweep (low T = peaked,
high T = flat) carries information beyond the standard margin. We probe the softmax
distribution at several temperatures and treat the resulting curve as a 'plasticity'
fingerprint for the sample.

Pipeline:
  1. Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
  2. For each test sample (that the model classifies correctly), compute:
        - softmax_max at T in {0.5, 1.0, 2.0, 4.0}
        - slope of softmax_max vs T (regression slope; a 'plasticity' measure)
        - margin (logit gap between top-1 and top-2)
        - mean_pix, std_pix (raw input statistics)
  3. Targets per sample:
        - FGSM flip at eps=15/255
        - PGD flip at eps=15/255
        - min_eps to flip via FGSM binary search (continuous)
  4. Univariate AUROC of each feature against the binary targets and
     Spearman/Pearson correlation against the continuous target.

Run last: write only. Do not execute.
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
TEMPS = [0.5, 1.0, 2.0, 4.0]
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0


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


def train_model(seed, train_set):
    torch.manual_seed(seed); np.random.seed(seed)
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
        print(f"  epoch {ep+1}/{EPOCHS} done")
    model.eval()
    return model


def batched_logits(model, x, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i+bs]))
    return torch.cat(out, 0)


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
    # random start within eps-ball
    adv = (x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        adv = adv.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def compute_temperature_features(logits, temps):
    """For each sample, compute softmax_max at each temperature and slope vs T."""
    feats = []
    sm_max_by_T = []
    for T in temps:
        sm = F.softmax(logits / T, dim=1)
        sm_max = sm.max(dim=1).values  # (N,)
        sm_max_by_T.append(sm_max)
        feats.append(sm_max)
    sm_max_mat = torch.stack(sm_max_by_T, dim=1)  # (N, len(temps))
    # slope via least-squares: slope = cov(T, sm_max) / var(T)
    T_vec = torch.tensor(temps, device=logits.device, dtype=sm_max_mat.dtype)
    T_mean = T_vec.mean()
    T_centered = T_vec - T_mean
    T_var = (T_centered ** 2).sum()
    sm_mean = sm_max_mat.mean(dim=1, keepdim=True)
    slope = ((sm_max_mat - sm_mean) * T_centered.unsqueeze(0)).sum(dim=1) / T_var
    feats.append(slope)
    return torch.stack(feats, dim=1)  # (N, len(temps)+1)


def main():
    print("# H97: temperature-sweep softmax features for adversarial vulnerability")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(" training CNN on Fashion-MNIST...")
    t0 = time.time()
    model = train_model(0, train_set)
    print(f"  done ({time.time()-t0:.1f}s)")

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print(" computing logits + final predictions...")
    logits = batched_logits(model, test_x)
    pred = logits.argmax(1)
    correct = pred == test_y
    print(f"  test accuracy = {correct.float().mean().item():.4f}")
    x_c = test_x[correct]
    y_c = test_y[correct]
    logits_c = logits[correct]
    N = x_c.size(0)
    print(f"  using {N} correctly classified samples")

    # margin (top1 - top2 logits)
    sorted_logits, _ = logits_c.sort(1, descending=True)
    margin = (sorted_logits[:, 0] - sorted_logits[:, 1])

    # pixel stats
    flat = x_c.view(N, -1)
    mean_pix = flat.mean(dim=1)
    std_pix = flat.std(dim=1)

    # temperature features
    print(" computing temperature-sweep features...")
    temp_feats = compute_temperature_features(logits_c, TEMPS)  # (N, len(TEMPS)+1)
    temp_names = [f"softmax_max_T{T}" for T in TEMPS] + ["softmax_max_slope_vs_T"]

    feat_names = temp_names + ["margin", "mean_pix", "std_pix"]
    feats = torch.cat([temp_feats,
                       margin.unsqueeze(1),
                       mean_pix.unsqueeze(1),
                       std_pix.unsqueeze(1)], dim=1)
    feats_np = feats.detach().cpu().numpy()

    # targets
    print(" computing FGSM flip target ...")
    t0 = time.time()
    fgsm_flips = []
    for i in range(0, N, 512):
        fgsm_flips.append(fgsm_attack(model, x_c[i:i+512], y_c[i:i+512]))
    fgsm_flip = torch.cat(fgsm_flips).cpu().numpy().astype(int)
    print(f"  done ({time.time()-t0:.1f}s)  flip rate = {fgsm_flip.mean():.3f}")

    print(" computing PGD flip target ...")
    t0 = time.time()
    pgd_flips = []
    for i in range(0, N, 256):
        pgd_flips.append(pgd_attack(model, x_c[i:i+256], y_c[i:i+256]))
    pgd_flip = torch.cat(pgd_flips).cpu().numpy().astype(int)
    print(f"  done ({time.time()-t0:.1f}s)  flip rate = {pgd_flip.mean():.3f}")

    print(" computing min_eps_to_flip target ...")
    t0 = time.time()
    me = []
    for i in range(0, N, 512):
        me.append(min_eps_to_flip(model, x_c[i:i+512], y_c[i:i+512]))
    min_eps = torch.cat(me).cpu().numpy()
    print(f"  done ({time.time()-t0:.1f}s)  mean min_eps = {min_eps.mean():.4f}")

    # univariate AUROC against binary targets
    print("\n=========== Univariate AUROC ===========")
    bin_targets = [("FGSM_flip", fgsm_flip), ("PGD_flip", pgd_flip)]
    for t_name, y in bin_targets:
        if y.std() == 0:
            print(f"  target {t_name} degenerate, skipping")
            continue
        print(f"\n--- target: {t_name}  (positive rate = {y.mean():.3f}) ---")
        for i, n in enumerate(feat_names):
            x_i = feats_np[:, i]
            a = roc_auc_score(y, x_i)
            a = max(a, 1 - a)
            print(f"   AUROC  {n:<28} {a:.4f}")

    # for continuous min_eps: report |Spearman| and AUROC-by-median-split
    print("\n=========== min_eps continuous ===========")
    median_eps = np.median(min_eps)
    y_bin = (min_eps <= median_eps).astype(int)  # "vulnerable" = small min_eps
    print(f"  median min_eps = {median_eps:.4f}; binarised as <= median => positive")
    print(f"\n--- target: min_eps (Spearman corr & median-split AUROC) ---")
    for i, n in enumerate(feat_names):
        x_i = feats_np[:, i]
        rho, _ = spearmanr(x_i, min_eps)
        a = roc_auc_score(y_bin, x_i)
        a = max(a, 1 - a)
        print(f"   {n:<28}  spearman={rho:+.4f}   AUROC_median_split={a:.4f}")


if __name__ == "__main__":
    main()
