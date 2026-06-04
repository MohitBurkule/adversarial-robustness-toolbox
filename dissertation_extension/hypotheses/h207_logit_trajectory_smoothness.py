"""
H207 - Logit margin trajectory smoothness during PGD as vulnerability predictor.

A sample with a monotonically decreasing margin under PGD is on a "gentle slope"
toward the boundary -- consistently exploitable. A sample with oscillating margin
is on a curved surface -- PGD may not converge. Smoothness of the trajectory
should predict final attack success beyond the initial margin alone.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0
N_EVAL = 200
EPS = 0.1
STEP_SIZE = 0.01
PGD_STEPS = 20

torch.manual_seed(SEED)
np.random.seed(SEED)

print("H207: Logit Trajectory Smoothness during PGD")
print("=" * 60)

Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]
META = {"channels": 1, "size": 28, "n_classes": 10}

print("Training CNN...")
model = C.build_model("cnn", META, width=32, seed=SEED)
C.train_model(model, Xtr, Ytr, epochs=10)
model.eval()


def compute_margin_batch(model, X, Y):
    """Compute top1-logit minus max-other-logit for each sample. Returns numpy (N,)."""
    with torch.no_grad():
        logits = model(X)
    # C.margin_of expects (logits_tensor, Y_tensor)
    return C.margin_of(logits.cpu(), Y.cpu())


# Compute margin trajectory during PGD
print(f"Running PGD-{PGD_STEPS} with trajectory recording (N={N_EVAL})...")

def pgd_with_margin_trajectory(model, X, Y, eps, steps, step_size):
    """Returns (X_adv, margin_traj) where margin_traj is (N, steps+1)."""
    X_adv = X.clone().detach()
    X_orig = X.clone().detach()
    margin_traj = []

    # step 0: clean margins
    m0 = compute_margin_batch(model, X_orig, Y)
    margin_traj.append(m0)

    for step in range(steps):
        X_adv = X_adv.requires_grad_(True)
        logits = model(X_adv)
        loss = F.cross_entropy(logits, Y)
        loss.backward()
        grad = X_adv.grad.detach()
        X_adv = X_adv.detach() + step_size * grad.sign()
        # Project back into eps-ball and [0,1]
        X_adv = torch.max(torch.min(X_adv, X_orig + eps), X_orig - eps).clamp(0, 1)

        m = compute_margin_batch(model, X_adv, Y)
        margin_traj.append(m)

    # margin_traj: list of (N,) arrays -> stack -> (steps+1, N) -> transpose -> (N, steps+1)
    margin_traj = np.stack(margin_traj, axis=0).T  # (N, steps+1)
    return X_adv.detach(), margin_traj

X_pgd, margin_traj = pgd_with_margin_trajectory(model, Xte, Yte, EPS, PGD_STEPS, STEP_SIZE)

# Attack success labels
with torch.no_grad():
    orig_preds = model(Xte).argmax(1)
    pgd_preds = model(X_pgd).argmax(1)
pgd_success = (pgd_preds != orig_preds).cpu().float().numpy()

# FGSM for comparison
X_fgsm = C.fgsm(model, Xte, Yte, eps=EPS)
with torch.no_grad():
    fgsm_preds = model(X_fgsm).argmax(1)
fgsm_success = (fgsm_preds != orig_preds).cpu().float().numpy()

print(f"FGSM ASR: {fgsm_success.mean():.4f}")
print(f"PGD-{PGD_STEPS} ASR: {pgd_success.mean():.4f}")

# Trajectory metrics per sample
initial_margin = margin_traj[:, 0]  # (N,)
final_margin = margin_traj[:, -1]   # (N,)

# Monotonicity: fraction of steps where margin decreases
diffs = np.diff(margin_traj, axis=1)  # (N, steps)
monotonicity = (diffs < 0).mean(axis=1)  # fraction of decreasing steps

# Total variation
total_variation = np.abs(diffs).sum(axis=1)  # sum of absolute changes

print("\n--- Trajectory Statistics ---")
print(f"Initial margin: mean={initial_margin.mean():.4f}, std={initial_margin.std():.4f}")
print(f"Final margin:   mean={final_margin.mean():.4f}, std={final_margin.std():.4f}")
print(f"Monotonicity:   mean={monotonicity.mean():.4f} (1=perfectly decreasing)")
print(f"Total variation: mean={total_variation.mean():.4f}")

# AUROCs
def safe_auroc(labels, scores):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float('nan')
    return roc_auc_score(labels, scores)

# For AUROC: higher score = more likely to succeed
# Initial margin: lower = more vulnerable → use -initial_margin
auroc_init_fgsm = safe_auroc(fgsm_success, -initial_margin)
auroc_init_pgd  = safe_auroc(pgd_success,  -initial_margin)
auroc_mono_pgd  = safe_auroc(pgd_success,   monotonicity)   # higher monotonicity = more vulnerable
auroc_tv_pgd    = safe_auroc(pgd_success,   total_variation)  # higher TV = bigger oscillation = boundary nearby = more vulnerable
auroc_final_pgd = safe_auroc(pgd_success,  -final_margin)    # final margin (near-tautological)

print("\n--- AUROC Results ---")
print(f"AUROC (−initial_margin → FGSM success):      {auroc_init_fgsm:.4f}  [baseline]")
print(f"AUROC (−initial_margin → PGD success):       {auroc_init_pgd:.4f}  [baseline]")
print(f"AUROC (monotonicity    → PGD success):       {auroc_mono_pgd:.4f}")
print(f"AUROC (total_variation  → PGD success):       {auroc_tv_pgd:.4f}")
print(f"AUROC (−final_margin   → PGD success):       {auroc_final_pgd:.4f}  [upper bound]")

# Spearman between metrics
rho_mono_init, _ = spearmanr(monotonicity, initial_margin)
rho_tv_init, _ = spearmanr(total_variation, initial_margin)
print(f"\nSpearman rho (monotonicity vs initial_margin): {rho_mono_init:.4f}")
print(f"Spearman rho (total_variation vs initial_margin): {rho_tv_init:.4f}")

# Mean trajectory for success vs fail groups
succ_idx = pgd_success == 1
fail_idx = pgd_success == 0
if succ_idx.sum() > 5 and fail_idx.sum() > 5:
    traj_succ = margin_traj[succ_idx].mean(axis=0)
    traj_fail = margin_traj[fail_idx].mean(axis=0)
    print(f"\n--- Mean Margin Trajectory (success N={succ_idx.sum()}, fail N={fail_idx.sum()}) ---")
    print(f"Step:  {'0':>6} {'5':>6} {'10':>6} {'15':>6} {'20':>6}")
    print(f"Succ:  {traj_succ[0]:>6.3f} {traj_succ[5]:>6.3f} {traj_succ[10]:>6.3f} {traj_succ[15]:>6.3f} {traj_succ[20]:>6.3f}")
    print(f"Fail:  {traj_fail[0]:>6.3f} {traj_fail[5]:>6.3f} {traj_fail[10]:>6.3f} {traj_fail[15]:>6.3f} {traj_fail[20]:>6.3f}")

print("\n--- Interpretation ---")
gain_mono = auroc_mono_pgd - auroc_init_pgd if not np.isnan(auroc_mono_pgd) else 0
gain_tv = auroc_tv_pgd - auroc_init_pgd if not np.isnan(auroc_tv_pgd) else 0
if gain_mono > 0.01 or gain_tv > 0.01:
    print("SUPPORTED: Trajectory smoothness adds predictive value beyond initial margin.")
else:
    print("NOT SUPPORTED: Trajectory metrics do not improve beyond initial margin AUROC.")
    print("Initial margin captures the essential vulnerability geometry.")
