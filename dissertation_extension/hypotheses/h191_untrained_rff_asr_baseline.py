#!/usr/bin/env python3
"""
H191: Untrained RFF network ASR baseline.

Hypothesis: an untrained random Fourier feature first-layer + trained linear
classifier has a different (higher) ASR than a fully trained network with the
same capacity, because gradient-based optimization creates exploitable structure
that random features lack.

Three conditions (784->256->10):
1. Untrained RFF: z = cos(W_r @ x + b_r), frozen; trained linear head
2. Untrained+linear: W_r @ x (no cosine), frozen; trained linear head
3. Fully trained: standard MLP 784->256->10, end-to-end
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from campaign.common import (
    set_seed, load_dataset, dataset_meta,
    fgsm, pgd, logits_and_acc, DEVICE,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT = os.path.join(RESULTS_DIR, "h191_untrained_rff_asr_baseline_output.txt")

SEED = 42
N_TRAIN = 6000
N_EVAL = 500
EPS = 0.3
PGD_STEPS = 10
PGD_ALPHA = 0.03
HIDDEN = 256
N_CLASSES = 10
INPUT_DIM = 784  # 1*28*28
SIGMA = 1.0
EPOCHS = 30
BATCH = 128
LR = 1e-3


class RFFNet(nn.Module):
    """Random Fourier Feature network: frozen cos(W@x + b), trained linear head."""
    def __init__(self, in_dim, hidden, n_classes, sigma=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W_r = nn.Parameter(torch.randn(hidden, in_dim, generator=g) / sigma, requires_grad=False)
        self.b_r = nn.Parameter(torch.rand(hidden, generator=g) * 2 * np.pi, requires_grad=False)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x_flat = x.flatten(1)
        z = torch.cos(x_flat @ self.W_r.T + self.b_r)
        return self.head(z)


class RandomLinearNet(nn.Module):
    """Frozen random linear first layer (no cosine), trained linear head."""
    def __init__(self, in_dim, hidden, n_classes, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W_r = nn.Parameter(torch.randn(hidden, in_dim, generator=g) * 0.01, requires_grad=False)
        self.b_r = nn.Parameter(torch.zeros(hidden), requires_grad=False)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x_flat = x.flatten(1)
        z = F.relu(x_flat @ self.W_r.T + self.b_r)
        return self.head(z)


class FullMLP(nn.Module):
    """Standard fully trained MLP 784->256->10."""
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_head(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, lr=LR, verbose=False):
    """Train only requires_grad parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            out = model(Xtr[idx])
            loss = F.cross_entropy(out, Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if verbose and (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{epochs} loss={loss.item():.3f}")
    model.eval()
    return model


def attack_asr(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA, batch=256):
    """Return ASR and per-sample flip vector (over correctly classified samples)."""
    model.eval()
    flips_all, correct_all = [], []
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i+batch], Y[i:i+batch]
        with torch.no_grad():
            correct = model(xb).argmax(1) == yb
        if attack == "fgsm":
            xa = fgsm(model, xb, yb, eps)
        else:
            xa = pgd(model, xb, yb, eps, steps=steps, alpha=alpha)
        with torch.no_grad():
            flipped = model(xa).argmax(1) != yb
        flips_all.append(flipped.cpu())
        correct_all.append(correct.cpu())
    flips = torch.cat(flips_all).numpy()
    corr = torch.cat(correct_all).numpy().astype(bool)
    asr = float(flips[corr].mean()) if corr.sum() > 0 else float("nan")
    return asr, flips, corr


def transfer_asr(source_model, target_model, X, Y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA, batch=256):
    """Craft PGD AEs on source, evaluate on target. ASR over target-correct samples."""
    target_model.eval()
    source_model.eval()
    flips_all, correct_all = [], []
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i+batch], Y[i:i+batch]
        with torch.no_grad():
            correct = target_model(xb).argmax(1) == yb
        xa = pgd(source_model, xb, yb, eps, steps=steps, alpha=alpha)
        with torch.no_grad():
            flipped = target_model(xa).argmax(1) != yb
        flips_all.append(flipped.cpu())
        correct_all.append(correct.cpu())
    flips = torch.cat(flips_all).numpy()
    corr = torch.cat(correct_all).numpy().astype(bool)
    asr = float(flips[corr].mean()) if corr.sum() > 0 else float("nan")
    return asr, flips, corr


def main():
    t0 = time.time()
    set_seed(SEED)
    lines = ["H191: Untrained RFF network ASR baseline", ""]

    Xtr, Ytr, Xte, Yte = load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # --- Build and train three conditions ---
    conditions = {}

    # 1. RFF
    print("Training RFF network...")
    rff = RFFNet(INPUT_DIM, HIDDEN, N_CLASSES, sigma=SIGMA, seed=SEED).to(DEVICE)
    train_head(rff, Xtr, Ytr, verbose=True)
    conditions["RFF (frozen cos)"] = rff

    # 2. Random linear
    print("Training Random Linear network...")
    rlin = RandomLinearNet(INPUT_DIM, HIDDEN, N_CLASSES, seed=SEED).to(DEVICE)
    train_head(rlin, Xtr, Ytr, verbose=True)
    conditions["Random Linear (frozen)"] = rlin

    # 3. Fully trained MLP
    print("Training Fully Trained MLP...")
    mlp = FullMLP(INPUT_DIM, HIDDEN, N_CLASSES).to(DEVICE)
    train_head(mlp, Xtr, Ytr, verbose=True)
    conditions["Fully Trained MLP"] = mlp

    # --- Evaluate ---
    results = {}
    for name, model in conditions.items():
        print(f"\nEvaluating {name}...")
        _, clean_acc = logits_and_acc(model, Xte, Yte)
        pgd_asr, pgd_flips, pgd_corr = attack_asr(model, Xte, Yte, attack="pgd")
        fgsm_asr, _, _ = attack_asr(model, Xte, Yte, attack="fgsm")
        results[name] = {"clean_acc": clean_acc, "pgd_asr": pgd_asr,
                         "fgsm_asr": fgsm_asr, "pgd_flips": pgd_flips, "pgd_corr": pgd_corr}

    # Transfer: craft on fully trained, test on RFF and random linear
    print("\nComputing transfer ASR...")
    for name in ["RFF (frozen cos)", "Random Linear (frozen)"]:
        t_asr, _, _ = transfer_asr(conditions["Fully Trained MLP"], conditions[name], Xte, Yte)
        results[name]["transfer_asr"] = t_asr
    results["Fully Trained MLP"]["transfer_asr"] = results["Fully Trained MLP"]["pgd_asr"]  # self-transfer = white-box

    # --- Report table ---
    lines.append(f"{'Condition':<25s} {'CleanAcc':>9s} {'PGD_ASR':>9s} {'FGSM_ASR':>9s} {'TransASR':>9s}")
    lines.append("-" * 65)
    for name in ["RFF (frozen cos)", "Random Linear (frozen)", "Fully Trained MLP"]:
        r = results[name]
        lines.append(f"{name:<25s} {r['clean_acc']:9.4f} {r['pgd_asr']:9.4f} "
                     f"{r['fgsm_asr']:9.4f} {r['transfer_asr']:9.4f}")

    # --- Statistical test: RFF ASR vs Fully Trained ASR ---
    lines.append("\n--- Statistical comparison: RFF vs Fully Trained (PGD ASR) ---")
    rff_r = results["RFF (frozen cos)"]
    mlp_r = results["Fully Trained MLP"]

    # Per-sample comparison on jointly correct samples
    joint_corr = rff_r["pgd_corr"] & mlp_r["pgd_corr"]
    rff_flips = rff_r["pgd_flips"][joint_corr]
    mlp_flips = mlp_r["pgd_flips"][joint_corr]
    n_joint = joint_corr.sum()

    rff_flip_rate = rff_flips.mean()
    mlp_flip_rate = mlp_flips.mean()
    lines.append(f"Jointly correct samples: {n_joint}")
    lines.append(f"RFF flip rate: {rff_flip_rate:.4f}")
    lines.append(f"MLP flip rate: {mlp_flip_rate:.4f}")

    # McNemar's test (paired comparison)
    a = ((rff_flips == 1) & (mlp_flips == 0)).sum()  # RFF flipped, MLP not
    b = ((rff_flips == 0) & (mlp_flips == 1)).sum()  # MLP flipped, RFF not
    if a + b > 0:
        mcnemar_stat = (abs(a - b) - 1) ** 2 / (a + b)
        mcnemar_p = stats.chi2.sf(mcnemar_stat, 1)
    else:
        mcnemar_stat = 0.0
        mcnemar_p = 1.0
    lines.append(f"McNemar's test: a={a}, b={b}, chi2={mcnemar_stat:.4f}, p={mcnemar_p:.6f}")
    lines.append(f"RFF ASR {'>' if rff_flip_rate > mlp_flip_rate else '<='} Fully Trained ASR (p={mcnemar_p:.6f})")

    # Also: binomial test on RFF flip rate vs MLP flip rate
    rff_n_flips = int(rff_flips.sum())
    binom_p = stats.binom_test(rff_n_flips, n_joint, mlp_flip_rate, alternative='two-sided')
    lines.append(f"Binomial test (two-sided, H0: RFF rate = MLP rate): p={binom_p:.6f}")

    hypothesis_supported = rff_flip_rate < mlp_flip_rate and mcnemar_p < 0.05
    lines.append(f"\nHypothesis (RFF ASR < Fully Trained ASR): {hypothesis_supported}")
    lines.append(f"  RFF ASR = {rff_flip_rate:.4f}, MLP ASR = {mlp_flip_rate:.4f}, "
                 f"diff = {mlp_flip_rate - rff_flip_rate:.4f}")

    elapsed = time.time() - t0
    lines.append(f"\nElapsed: {elapsed:.1f}s")

    report = "\n".join(lines)
    print("\n" + report)
    with open(OUT, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
