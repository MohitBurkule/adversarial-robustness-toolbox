"""
H228 — Universal Adversarial Purifier
Train a small denoising CNN autoencoder to remove adversarial perturbations.

Architecture: Conv(1,32,3,pad=1)->ReLU->Conv(32,32,3,pad=1)->ReLU->Conv(32,1,3,pad=1)->Sigmoid
Training: for each batch generate PGD-3 adversarials, train P with MSE(P(x_adv), x_clean)
Use first 5000 training samples, 20 epochs.

Evaluate:
1. Clean accuracy: clf(P(x_clean)) — does purifier hurt clean accuracy?
2. FGSM recovery: clf(P(x_fgsm))
3. PGD recovery:  clf(P(x_pgd))
4. Cross-attack: purifier trained on PGD, tested on FGSM
5. Adaptive attack: grad through clf(P(x))
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign import common as C

os.makedirs("results/fashion_mnist", exist_ok=True)

SEED           = 0
N_EVAL         = 300
EPS            = 0.1
PURIFIER_N     = 5000
PURIFIER_EPOCHS = 20
C.set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Purifier architecture
# ---------------------------------------------------------------------------

class Purifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_purifier(purifier: Purifier, clf: nn.Module,
                   Xtr: torch.Tensor, Ytr: torch.Tensor,
                   attack: str = "pgd", epochs: int = PURIFIER_EPOCHS,
                   batch: int = 128) -> Purifier:
    opt = torch.optim.Adam(purifier.parameters(), lr=1e-3)
    n = Xtr.size(0)
    clf.eval()
    purifier.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0; steps = 0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb, yb = Xtr[idx], Ytr[idx]
            # generate adversarials with fixed clf
            if attack == "pgd":
                xa = C.pgd(clf, xb, yb, eps=EPS, steps=3)
            else:
                xa = C.fgsm(clf, xb, yb, eps=EPS)
            # train purifier: MSE(P(x_adv), x_clean)
            opt.zero_grad()
            recon = purifier(xa.detach())
            loss = F.mse_loss(recon, xb)
            loss.backward()
            opt.step()
            total_loss += loss.item(); steps += 1
        if (ep + 1) % 5 == 0:
            print(f"    epoch {ep+1}/{epochs}  mse={total_loss/steps:.5f}")
    purifier.eval()
    return purifier


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy(model, X, Y, batch=256):
    corr = 0
    for i in range(0, X.size(0), batch):
        corr += (model(X[i:i+batch]).argmax(1) == Y[i:i+batch]).sum().item()
    return corr / X.size(0)


@torch.no_grad()
def accuracy_with_purifier(clf, purifier, X, Y, batch=256):
    corr = 0
    for i in range(0, X.size(0), batch):
        xb = X[i:i+batch]; yb = Y[i:i+batch]
        xp = purifier(xb)
        corr += (clf(xp).argmax(1) == yb).sum().item()
    return corr / X.size(0)


def asr_standard(model, X, Y, attack, eps=EPS, steps=10, batch=64):
    """ASR without purifier."""
    model.eval()
    flips, corr = [], []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i+batch], Y[i:i+batch]
        with torch.no_grad():
            c = model(x).argmax(1) == y
        if attack == "fgsm":
            xa = C.fgsm(model, x, y, eps)
        else:
            xa = C.pgd(model, x, y, eps, steps)
        with torch.no_grad():
            f = model(xa).argmax(1) != y
        flips.append(f.cpu()); corr.append(c.cpu())
    flips = torch.cat(flips).numpy()
    corr  = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() > 0 else float("nan")


def recovery_rate(clf, purifier, X, Y, attack, eps=EPS, steps=10, batch=64):
    """
    Generate adversarials against clf (no purifier in attack graph).
    Measure fraction where clf(P(x_adv)).argmax == y.
    """
    clf.eval(); purifier.eval()
    correct_after, total = 0, 0
    for i in range(0, X.size(0), batch):
        x, y = X[i:i+batch], Y[i:i+batch]
        if attack == "fgsm":
            xa = C.fgsm(clf, x, y, eps)
        else:
            xa = C.pgd(clf, x, y, eps, steps)
        with torch.no_grad():
            xp = purifier(xa)
            preds = clf(xp).argmax(1)
        correct_after += (preds == y).sum().item()
        total += y.size(0)
    return correct_after / total if total > 0 else float("nan")


def adaptive_asr(clf, purifier, X, Y, eps=EPS, steps=10, batch=64):
    """
    Adaptive attack: compute gradient through the full pipeline clf(P(x)).
    Uses PGD.
    """
    clf.eval(); purifier.eval()
    # Wrap clf(purifier(x)) as a combined model
    class JointModel(nn.Module):
        def __init__(self, p, c):
            super().__init__()
            self.p = p; self.c = c
        def forward(self, x):
            return self.c(self.p(x))

    joint = JointModel(purifier, clf)
    flips, corr = [], []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i+batch], Y[i:i+batch]
        with torch.no_grad():
            c = clf(purifier(x)).argmax(1) == y
        xa = C.pgd(joint, x, y, eps=eps, steps=steps)
        with torch.no_grad():
            f = clf(purifier(xa)).argmax(1) != y
        flips.append(f.cpu()); corr.append(c.cpu())
    flips = torch.cat(flips).numpy()
    corr  = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() > 0 else float("nan")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
print("Loading Fashion-MNIST …")
Xtr_full, Ytr_full, Xte, Yte = C.load_dataset("fashion_mnist", n_eval=N_EVAL, seed=SEED)
Xte = Xte[:N_EVAL].to(device)
Yte = Yte[:N_EVAL].to(device)

meta = {"channels": 1, "size": 28, "n_classes": 10}

# Train classifier
print("\nTraining classifier on full training set …")
C.set_seed(SEED)
clf = C.build_model("cnn", meta, width=32, seed=SEED)
C.train_model(clf, Xtr_full, Ytr_full, epochs=10)
clf.eval()

clean_acc_no_purifier = accuracy(clf, Xte, Yte)
fgsm_asr_no_purifier  = asr_standard(clf, Xte, Yte, "fgsm")
pgd_asr_no_purifier   = asr_standard(clf, Xte, Yte, "pgd")
print(f"  Classifier only: clean={clean_acc_no_purifier:.3f}  fgsm_asr={fgsm_asr_no_purifier:.3f}  pgd_asr={pgd_asr_no_purifier:.3f}")

# Take first 5000 samples for purifier training
Xpur = Xtr_full[:PURIFIER_N]
Ypur = Ytr_full[:PURIFIER_N]

# Train PGD-purifier
print(f"\nTraining PGD purifier on {PURIFIER_N} samples ({PURIFIER_EPOCHS} epochs) …")
C.set_seed(SEED)
purifier_pgd = Purifier().to(device)
train_purifier(purifier_pgd, clf, Xpur, Ypur, attack="pgd")

# Evaluate PGD purifier
print("\nEvaluating PGD purifier …")
clean_with_p    = accuracy_with_purifier(clf, purifier_pgd, Xte, Yte)
recov_fgsm      = recovery_rate(clf, purifier_pgd, Xte, Yte, "fgsm")
recov_pgd       = recovery_rate(clf, purifier_pgd, Xte, Yte, "pgd")
adapt_asr       = adaptive_asr(clf, purifier_pgd, Xte, Yte)
print(f"  clean_acc (with purifier): {clean_with_p:.3f}")
print(f"  FGSM recovery rate:        {recov_fgsm:.3f}")
print(f"  PGD recovery rate:         {recov_pgd:.3f}")
print(f"  Adaptive PGD ASR:          {adapt_asr:.3f}")

# Cross-attack: purifier trained on PGD, tested on FGSM adversarials
# (recovery_rate already does this for "fgsm" above)
print(f"\nCross-attack generalisation (PGD-trained purifier vs FGSM adversarials):")
print(f"  FGSM recovery rate: {recov_fgsm:.3f}  (cross-attack)")

# Also train a FGSM-purifier for cross-attack the other direction
print(f"\nTraining FGSM purifier on {PURIFIER_N} samples …")
C.set_seed(SEED)
purifier_fgsm = Purifier().to(device)
train_purifier(purifier_fgsm, clf, Xpur, Ypur, attack="fgsm")

recov_fgsm_on_pgd = recovery_rate(clf, purifier_fgsm, Xte, Yte, "pgd")
recov_fgsm_on_fgsm = recovery_rate(clf, purifier_fgsm, Xte, Yte, "fgsm")
print(f"  FGSM-trained purifier: FGSM recovery={recov_fgsm_on_fgsm:.3f}  PGD recovery={recov_fgsm_on_pgd:.3f}")

# --- Summary table ---
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Metric':<45} {'Value':>10}")
print("-"*70)
print(f"{'No purifier — clean acc':<45} {clean_acc_no_purifier:>10.3f}")
print(f"{'No purifier — FGSM ASR':<45} {fgsm_asr_no_purifier:>10.3f}")
print(f"{'No purifier — PGD ASR':<45} {pgd_asr_no_purifier:>10.3f}")
print(f"{'PGD purifier — clean acc (clf(P(x_clean)))':<45} {clean_with_p:>10.3f}")
print(f"{'PGD purifier — FGSM recovery rate':<45} {recov_fgsm:>10.3f}")
print(f"{'PGD purifier — PGD recovery rate':<45} {recov_pgd:>10.3f}")
print(f"{'PGD purifier — adaptive PGD ASR':<45} {adapt_asr:>10.3f}")
print(f"{'FGSM purifier — FGSM recovery rate':<45} {recov_fgsm_on_fgsm:>10.3f}")
print(f"{'FGSM purifier — PGD recovery (cross-attack)':<45} {recov_fgsm_on_pgd:>10.3f}")
print("="*70)

print(f"\nConclusions:")
print(f"  Purifier clean accuracy drop: {clean_acc_no_purifier - clean_with_p:.3f}")
print(f"  PGD recovery (non-adaptive attack): {'GOOD' if recov_pgd > 0.5 else 'POOR'} ({recov_pgd:.3f})")
print(f"  Adaptive attack breaks purifier: {'YES' if adapt_asr > pgd_asr_no_purifier * 0.8 else 'PARTIALLY'} (adaptive_asr={adapt_asr:.3f})")
cross_ok = recov_fgsm > 0.5 or recov_fgsm_on_pgd > 0.5
print(f"  Cross-attack generalisation: {'YES' if cross_ok else 'NO'}")
print("\nDone.")
