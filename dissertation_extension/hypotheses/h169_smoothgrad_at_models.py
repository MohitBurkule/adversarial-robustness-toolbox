"""
Hypothesis H169: SmoothGrad Informativeness on Adversarially Trained Models.

H126 showed SmoothGrad L2 norm achieves ~0.97 AUROC on vanilla CNNs, matching
the decision-boundary margin. This script asks whether SmoothGrad retains that
informativeness on adversarially trained models, where the decision boundary
geometry is fundamentally different: expanded (larger margin) but potentially
less smooth due to adversarial training sharpening the loss landscape.

Three CNN variants are trained and evaluated:
  1. Vanilla CNN       — standard cross-entropy, seed=0.
  2. FGSM-AT CNN       — each batch trained on FGSM adversarial examples.
  3. PGD-AT CNN        — each batch trained on PGD-7 adversarial examples
                         (alpha=2/255, random init).

For each model, 1000 correctly-classified test samples are evaluated.

Per-sample features:
  - margin              : top1 - top2 logit
  - smoothgrad_l2_norm  : L2 norm of K=20 SmoothGrad map (sigma=0.1)
  - smoothgrad_max      : max absolute value of SmoothGrad map
  - input_grad_l2_norm  : L2 norm of plain single-pass input gradient
  - top1_prob           : softmax probability of predicted class

Vulnerability targets:
  - flipped_FGSM  : FGSM (eps=15/255) prediction flip (binary)
  - flipped_PGD   : PGD-10 (alpha=2/255, eps=15/255) prediction flip (binary)
  - min_eps       : binary-search minimum eps to flip (continuous)

Key question: Does SmoothGrad remain >= margin AUROC on AT models?
Printed comparison for each model:
  smoothgrad_l2_norm AUROC vs margin AUROC vs input_grad_l2_norm AUROC

Direction of change: does SmoothGrad's advantage over margin increase or
decrease under adversarial training?

Dataset: Fashion-MNIST (monkey-patched at runtime via patch_dataset.py).
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
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    """Small CNN matching diagnostic_test.py / H126 architecture."""

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
# Attack helpers
# ---------------------------------------------------------------------------

def fgsm_attack(model, x, y, eps=EPS):
    """Single-step FGSM attack; returns adversarial examples."""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    with torch.no_grad():
        x_adv = (x_adv + eps * x_adv.grad.sign()).clamp(0, 1)
    return x_adv.detach()


def pgd_attack(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=10):
    """PGD attack with random initialisation."""
    x_adv = x.clone().detach()
    x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = (x_adv + alpha * x_adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return x_adv.detach()


def pgd_train(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=7):
    """PGD-7 for adversarial training (random init, no grad tracking outside)."""
    x_adv = x.clone().detach()
    x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = (x_adv + alpha * x_adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return x_adv.detach()


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=8):
    """Binary-search minimum per-sample epsilon to flip prediction (PGD-5)."""
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        adv = x.clone().detach()
        alpha = mid / 4.0
        for _ in range(5):
            adv = adv.detach().requires_grad_(True)
            loss = F.cross_entropy(model(adv), y)
            loss.backward()
            with torch.no_grad():
                adv = (adv + alpha.view(-1, 1, 1, 1) * adv.grad.sign()).clamp(
                    x - mid.view(-1, 1, 1, 1), x + mid.view(-1, 1, 1, 1)
                ).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi.detach()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_loader(train_set, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(train_set, BATCH, shuffle=True, num_workers=2,
                      generator=g, pin_memory=True)


def train_vanilla(train_set, seed=0):
    """Train CNN with standard cross-entropy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = _make_loader(train_set, seed)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  [Vanilla] Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(loader):.4f}")
    return model


def train_fgsm_at(train_set, seed=0):
    """Adversarial training: each batch replaced with FGSM examples."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = _make_loader(train_set, seed)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Generate FGSM adversarial examples in train mode
            x_adv = fgsm_attack(model, x, y, eps=EPS)
            opt.zero_grad()
            loss = F.cross_entropy(model(x_adv), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  [FGSM-AT] Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(loader):.4f}")
    return model


def train_pgd_at(train_set, seed=0):
    """Adversarial training: each batch replaced with PGD-7 examples (alpha=2/255)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = _make_loader(train_set, seed)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = pgd_train(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=7)
            opt.zero_grad()
            loss = F.cross_entropy(model(x_adv), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  [PGD-AT]  Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(loader):.4f}")
    return model


# ---------------------------------------------------------------------------
# SmoothGrad & feature extraction
# ---------------------------------------------------------------------------

def smoothgrad_features(model, x, y, K=20, sigma=0.1):
    """
    Compute SmoothGrad attribution map processed in mini-batches of 100 to
    avoid OOM on GPU.

    Returns:
        l2_norm : (N,) tensor  -- L2 norm of averaged gradient map
        max_val : (N,) tensor  -- max absolute value of averaged gradient map
    """
    model.eval()
    N = x.size(0)
    grad_sum = torch.zeros_like(x)
    mini = 100

    for k in range(K):
        noise = torch.randn_like(x) * sigma
        x_noisy = (x + noise).clamp(0, 1).detach()
        # Process in mini-batches
        for start in range(0, N, mini):
            end = min(start + mini, N)
            xb = x_noisy[start:end].requires_grad_(True)
            yb = y[start:end]
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            with torch.no_grad():
                if xb.grad is not None:
                    grad_sum[start:end] += xb.grad.detach()

    sg = grad_sum / K  # (N, C, H, W)
    sg_flat = sg.view(N, -1).abs()
    l2_norm = sg_flat.norm(2, dim=1)
    max_val = sg_flat.max(dim=1)[0]
    return l2_norm, max_val


def input_grad_l2(model, x, y):
    """Plain single-pass input gradient L2 norm."""
    model.eval()
    N = x.size(0)
    norms = torch.zeros(N, device=DEVICE)
    mini = 100
    for start in range(0, N, mini):
        end = min(start + mini, N)
        xb = x[start:end].detach().requires_grad_(True)
        yb = y[start:end]
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        with torch.no_grad():
            if xb.grad is not None:
                norms[start:end] = xb.grad.view(end - start, -1).norm(2, dim=1)
    return norms


def compute_all_features(model, x_c, y_c):
    """
    Compute feature matrix and vulnerability targets for correctly-classified
    samples x_c, y_c.

    Returns:
        feats         : (N, 5) tensor  [margin, sg_l2, sg_max, ig_l2, top1_prob]
        flipped_fgsm  : (N,) long tensor
        flipped_pgd   : (N,) long tensor
        min_eps       : (N,) float tensor
    """
    model.eval()
    N = x_c.size(0)

    with torch.no_grad():
        logits = model(x_c)
        probs = torch.softmax(logits, dim=1)
        sorted_logits, _ = logits.sort(1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        top1_prob = probs.max(dim=1)[0]

    print("   Computing SmoothGrad (K=20, sigma=0.1)...")
    sg_l2, sg_max = smoothgrad_features(model, x_c, y_c, K=20, sigma=0.1)

    print("   Computing input gradient (1-pass)...")
    ig_l2 = input_grad_l2(model, x_c, y_c)

    feats = torch.stack([margin, sg_l2, sg_max, ig_l2, top1_prob], dim=1)

    print("   Running FGSM attack...")
    fgsm_adv = fgsm_attack(model, x_c, y_c, eps=EPS)
    print("   Running PGD-10 attack...")
    pgd_adv = pgd_attack(model, x_c, y_c, eps=EPS, alpha=2.0 / 255.0, steps=10)

    with torch.no_grad():
        flipped_fgsm = (model(fgsm_adv).argmax(1) != y_c).long()
        flipped_pgd = (model(pgd_adv).argmax(1) != y_c).long()

    print("   Binary-searching min_eps...")
    min_eps = min_eps_to_flip(model, x_c, y_c)

    return feats, flipped_fgsm, flipped_pgd, min_eps


# ---------------------------------------------------------------------------
# AUROC helper
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["margin", "smoothgrad_l2_norm", "smoothgrad_max",
                 "input_grad_l2_norm", "top1_prob"]


def auroc_scores(feats, target):
    """
    Univariate AUROC for each feature vs a binary or continuous target.
    Continuous targets are binarized at the median (below-median = 1 = easier to flip).
    Returns dict {name: auroc}.  Scores are always >= 0.5 (direction-agnostic).
    """
    target_np = target.cpu().numpy().astype(float)
    if target_np.std() == 0:
        return {n: 0.5 for n in FEATURE_NAMES}
    # Binarize continuous targets at median
    unique_vals = np.unique(target_np)
    if len(unique_vals) > 2:
        median = np.median(target_np)
        target_np = (target_np <= median).astype(float)
    out = {}
    for i, name in enumerate(FEATURE_NAMES):
        f = feats[:, i].cpu().numpy()
        a = roc_auc_score(target_np, f)
        out[name] = max(a, 1.0 - a)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_model(label, model, x_c, y_c, clean_acc):
    """Run full evaluation for one model; print and return AUROC tables."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")
    print(f"  Clean accuracy on full test set: {clean_acc:.4f}")
    print(f"  Correctly classified samples used: {x_c.size(0)}")

    feats, ff, fp, me = compute_all_features(model, x_c, y_c)

    fgsm_rate = ff.float().mean().item()
    pgd_rate = fp.float().mean().item()
    mean_me = me.mean().item()
    print(f"  FGSM flip rate (eps=15/255):  {fgsm_rate:.4f}")
    print(f"  PGD-10 flip rate (eps=15/255): {pgd_rate:.4f}")
    print(f"  Mean min_eps: {mean_me:.4f}")

    auc_fgsm = auroc_scores(feats, ff)
    auc_pgd = auroc_scores(feats, fp)
    auc_me = auroc_scores(feats, me)

    print(f"\n  {'Feature':<22} {'vs FGSM':>8} {'vs PGD':>8} {'vs min_eps':>10}")
    print(f"  {'-'*52}")
    for name in FEATURE_NAMES:
        print(f"  {name:<22} {auc_fgsm[name]:>8.4f} {auc_pgd[name]:>8.4f} {auc_me[name]:>10.4f}")

    return auc_fgsm, auc_pgd, auc_me


def main():
    t0 = time.time()
    print("=" * 70)
    print("Hypothesis H169: SmoothGrad Informativeness on Adversarially Trained Models")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Load 1000 test samples once; reuse across models
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    x_test, y_test = next(iter(test_loader))
    x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)

    models_cfg = [
        ("Vanilla CNN",  train_vanilla),
        ("FGSM-AT CNN",  train_fgsm_at),
        ("PGD-AT CNN",   train_pgd_at),
    ]

    results = {}  # label -> (auc_fgsm, auc_pgd, auc_me)

    for label, train_fn in models_cfg:
        print(f"\n{'='*70}")
        print(f"Training: {label}")
        print(f"{'='*70}")
        model = train_fn(train_set, seed=0)
        model.eval()

        with torch.no_grad():
            correct_mask = model(x_test).argmax(1) == y_test
        clean_acc = correct_mask.float().mean().item()
        x_c = x_test[correct_mask]
        y_c = y_test[correct_mask]

        auc_fgsm, auc_pgd, auc_me = evaluate_model(label, model, x_c, y_c, clean_acc)
        results[label] = (auc_fgsm, auc_pgd, auc_me)

    # ------------------------------------------------------------------
    # Key comparison: SmoothGrad vs margin vs input_grad, per model
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KEY COMPARISON: smoothgrad_l2_norm vs margin vs input_grad_l2_norm")
    print("(AUROC vs PGD-10 flip — primary robustness target)")
    print("=" * 70)
    print(f"{'Model':<18} {'sg_l2':>8} {'margin':>8} {'ig_l2':>8}  {'sg>margin?':>12}  {'sg_adv-sg_vanilla':>20}")

    sg_vanilla = results["Vanilla CNN"][1]["smoothgrad_l2_norm"]
    for label in ["Vanilla CNN", "FGSM-AT CNN", "PGD-AT CNN"]:
        af, ap, am = results[label]
        sg  = ap["smoothgrad_l2_norm"]
        mg  = ap["margin"]
        ig  = ap["input_grad_l2_norm"]
        adv = "YES" if sg >= mg else "NO"
        delta = sg - sg_vanilla
        print(f"  {label:<16} {sg:>8.4f} {mg:>8.4f} {ig:>8.4f}  {adv:>12}  {delta:>+20.4f}")

    print()
    sg_vanilla_val = results["Vanilla CNN"][1]["smoothgrad_l2_norm"]
    sg_fgsm_val = results["FGSM-AT CNN"][1]["smoothgrad_l2_norm"]
    sg_pgd_val = results["PGD-AT CNN"][1]["smoothgrad_l2_norm"]
    mg_vanilla = results["Vanilla CNN"][1]["margin"]
    mg_fgsm = results["FGSM-AT CNN"][1]["margin"]
    mg_pgd = results["PGD-AT CNN"][1]["margin"]

    adv_vanilla = sg_vanilla_val - mg_vanilla
    adv_fgsm = sg_fgsm_val - mg_fgsm
    adv_pgd = sg_pgd_val - mg_pgd

    print("SmoothGrad advantage over margin (AUROC gap, vs PGD flip):")
    print(f"  Vanilla CNN  : {adv_vanilla:+.4f}")
    print(f"  FGSM-AT CNN  : {adv_fgsm:+.4f}")
    print(f"  PGD-AT CNN   : {adv_pgd:+.4f}")

    if adv_pgd > adv_vanilla and adv_fgsm > adv_vanilla:
        direction = "INCREASES under both FGSM-AT and PGD-AT"
    elif adv_pgd < adv_vanilla and adv_fgsm < adv_vanilla:
        direction = "DECREASES under both FGSM-AT and PGD-AT"
    else:
        direction = "MIXED — changes direction between FGSM-AT and PGD-AT"

    print(f"\nDirection of SmoothGrad advantage under AT: {direction}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed/60:.1f} min")
    print("\nDone.")


if __name__ == "__main__":
    main()
