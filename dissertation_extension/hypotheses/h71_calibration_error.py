"""
H71: Per-sample calibration error predicts adversarial vulnerability.

Hypothesis: Local expected calibration error (ECE) -- specifically, the gap
between a sample's max-softmax confidence and the empirical accuracy of
samples in the same confidence bin -- is correlated with adversarial
vulnerability. Samples whose confidence over-estimates the true accuracy of
their neighborhood should be easier to flip.

Pipeline:
  1. Train a small CNN (architecture matching diagnostic_test.py) on
     Fashion-MNIST for 10 epochs.
  2. For each test sample compute calibration-related features:
        - margin            (top1 - top2 logit)
        - mean_pix          (mean pixel value)
        - std_pix           (std of pixel values)
        - brier             (Brier score against a uniform target vector)
        - reliability_gap   (|max_prob - empirical_accuracy_in_conf_bin|)
     Auxiliary per-sample calibration quantities computed as a side-product
     (distance from confidence to true-class-probability) are also reported.
  3. Adversarial targets:
        - FGSM flip at eps = 15/255
        - PGD  flip at eps = 15/255 (10 steps, step=eps/4)
        - min_eps_FGSM (binary search smallest L_inf eps that flips FGSM)
  4. Univariate AUROC for each feature vs each binary target, plus Pearson
     correlation vs continuous min_eps.

Self-contained; this script writes code only and is not executed here.
"""
import time
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
N_BINS = 15  # for ECE / reliability binning


# --- CNN matching diagnostic_test.py exactly ---
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


def train_model(seed, train_set, n_classes=10):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done ({time.time()-t0:.1f}s)")
    model.eval()
    return model


def batched_logits(model, x, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i+bs]))
    return torch.cat(out, 0)


# --- attacks ---
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack_success(model, x, y, eps=EPS_TEST):
    flips = []
    for i in range(0, x.size(0), 512):
        xb, yb = x[i:i+512], y[i:i+512]
        sign = fgsm_grad(model, xb, yb)
        adv = (xb + eps * sign).clamp(0, 1)
        with torch.no_grad():
            flips.append(model(adv).argmax(1) != yb)
    return torch.cat(flips)


def pgd_attack_success(model, x, y, eps=EPS_TEST, steps=10, alpha=None):
    if alpha is None:
        alpha = eps / 4
    flips = []
    for i in range(0, x.size(0), 512):
        xb, yb = x[i:i+512], y[i:i+512]
        adv = xb.clone().detach() + torch.empty_like(xb).uniform_(-eps, eps)
        adv = adv.clamp(0, 1)
        for _ in range(steps):
            adv = adv.detach().requires_grad_(True)
            loss = F.cross_entropy(model(adv), yb)
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + alpha * grad.sign()
            adv = torch.max(torch.min(adv, xb + eps), xb - eps).clamp(0, 1)
        with torch.no_grad():
            flips.append(model(adv).argmax(1) != yb)
    return torch.cat(flips)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
    out = []
    for i in range(0, x.size(0), 512):
        xb, yb = x[i:i+512], y[i:i+512]
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        sign = fgsm_grad(model, xb, yb)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xb + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
            with torch.no_grad():
                flipped = model(adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        out.append(hi)
    return torch.cat(out)


# --- calibration features ---
def per_sample_reliability_gap(max_probs, correct, n_bins=N_BINS):
    """For each sample, return |max_prob - acc_in_its_conf_bin|.

    Bins are equal-width over [0, 1]; empirical accuracy is computed over all
    test samples falling in the same confidence bin.
    """
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=max_probs.device)
    # bin index in [0, n_bins-1]
    idx = torch.bucketize(max_probs, edges, right=False) - 1
    idx = idx.clamp(0, n_bins - 1)
    bin_acc = torch.zeros(n_bins, device=max_probs.device)
    bin_conf = torch.zeros(n_bins, device=max_probs.device)
    bin_count = torch.zeros(n_bins, device=max_probs.device)
    for b in range(n_bins):
        m = (idx == b)
        if m.any():
            bin_acc[b] = correct[m].float().mean()
            bin_conf[b] = max_probs[m].mean()
            bin_count[b] = m.sum().float()
    sample_bin_acc = bin_acc[idx]
    gap = (max_probs - sample_bin_acc).abs()
    return gap, bin_acc, bin_conf, bin_count, idx


def main():
    print("##### H71: per-sample calibration error vs adversarial vulnerability #####")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    n_classes = 10

    print(" training CNN on Fashion-MNIST...")
    model = train_model(0, train_set, n_classes)

    # stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)
    print(f" test set: {N} samples")

    # ---- logits / softmax features ----
    logits = batched_logits(model, test_x)
    probs = F.softmax(logits, 1)
    max_probs, pred = probs.max(1)
    correct = pred == test_y
    print(f" clean accuracy = {correct.float().mean().item():.4f}")

    sorted_logits, _ = logits.sort(1, descending=True)
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]

    # pixel stats
    mean_pix = test_x.view(N, -1).mean(1)
    std_pix = test_x.view(N, -1).std(1)

    # Brier vs uniform target (one-vs-rest with target = 1/K everywhere)
    uniform = torch.full_like(probs, 1.0 / n_classes)
    brier_uniform = ((probs - uniform) ** 2).sum(1)

    # distance from confidence (max prob) to true-class probability
    true_prob = probs[torch.arange(N, device=DEVICE), test_y]
    conf_to_true_gap = (max_probs - true_prob).abs()

    # reliability gap (per-sample local ECE proxy)
    reliability_gap, bin_acc, bin_conf, bin_count, bin_idx = per_sample_reliability_gap(
        max_probs, correct, n_bins=N_BINS
    )

    # overall ECE (informative print)
    total = bin_count.sum().clamp(min=1.0)
    ece = ((bin_count / total) * (bin_acc - bin_conf).abs()).sum().item()
    print(f" overall ECE (15 bins) = {ece:.4f}")

    # ---- adversarial targets (evaluate on samples model classifies correctly) ----
    print(" computing FGSM attack success...")
    fgsm_flip = fgsm_attack_success(model, test_x, test_y, eps=EPS_TEST)
    print(f"  FGSM flip rate (all samples) = {fgsm_flip.float().mean().item():.4f}")

    print(" computing PGD attack success...")
    pgd_flip = pgd_attack_success(model, test_x, test_y, eps=EPS_TEST, steps=10)
    print(f"  PGD flip rate (all samples) = {pgd_flip.float().mean().item():.4f}")

    print(" computing min_eps_to_flip (FGSM binary search)...")
    min_eps = min_eps_to_flip(model, test_x, test_y)
    print(f"  mean min_eps = {min_eps.mean().item():.4f}")

    # restrict to correctly-classified samples for evaluation
    mask = correct
    print(f" evaluating on {int(mask.sum().item())} correctly-classified samples")

    feat_names = ["margin", "mean_pix", "std_pix", "brier_uniform", "reliability_gap"]
    feats = torch.stack([margin, mean_pix, std_pix, brier_uniform, reliability_gap], 1)
    feats_c = feats[mask].detach().cpu().numpy()

    targets_bin = {
        "FGSM_flip":      fgsm_flip[mask].detach().cpu().numpy().astype(int),
        "PGD_flip":       pgd_flip[mask].detach().cpu().numpy().astype(int),
    }
    min_eps_c = min_eps[mask].detach().cpu().numpy()

    # ---- univariate AUROC ----
    print("\n========== Univariate AUROC ==========")
    for t_name, y_arr in targets_bin.items():
        if y_arr.std() == 0:
            print(f" target {t_name}: degenerate (pos rate = {y_arr.mean():.3f}) -- skipping")
            continue
        print(f"\n--- target: {t_name}  (pos rate = {y_arr.mean():.3f}) ---")
        for i, n in enumerate(feat_names):
            x_i = feats_c[:, i]
            a = roc_auc_score(y_arr, x_i)
            a = max(a, 1 - a)
            print(f"   univariate AUROC  {n:<18} {a:.4f}")

    # ---- min_eps (continuous) ----
    print("\n--- target: min_eps_to_flip (Pearson correlation) ---")
    for i, n in enumerate(feat_names):
        x_i = feats_c[:, i]
        if x_i.std() == 0:
            print(f"   {n:<18} degenerate")
            continue
        cor = np.corrcoef(x_i, min_eps_c)[0, 1]
        print(f"   corr(min_eps, {n:<18}) = {cor:+.4f}")

    # ---- diagnostic: confidence-vs-true-prob gap (reported but not in feature set
    #      because it is degenerate for correctly-classified samples where pred==true).
    print("\n--- diagnostic: confidence-to-true-class gap ---")
    ctg = conf_to_true_gap[mask].detach().cpu().numpy()
    print(f"   mean = {ctg.mean():.6f}  std = {ctg.std():.6f}")
    print("   (note: for correctly classified samples this equals 0; reported on full set below)")
    ctg_all = conf_to_true_gap.detach().cpu().numpy()
    print(f"   full set mean = {ctg_all.mean():.4f}  std = {ctg_all.std():.4f}")
    for t_name, t_tensor in [("FGSM_flip", fgsm_flip), ("PGD_flip", pgd_flip)]:
        y_full = t_tensor.detach().cpu().numpy().astype(int)
        if y_full.std() == 0:
            continue
        a = roc_auc_score(y_full, ctg_all)
        a = max(a, 1 - a)
        print(f"   univariate AUROC (full set) conf_to_true_gap vs {t_name}: {a:.4f}")


if __name__ == "__main__":
    main()
