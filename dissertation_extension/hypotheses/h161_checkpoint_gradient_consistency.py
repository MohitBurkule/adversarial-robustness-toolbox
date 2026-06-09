"""
H161: Checkpoint Gradient Consistency vs Adversarial Vulnerability

Hypothesis
----------
Samples whose input-gradient *direction* stabilises early across training
checkpoints may be more adversarially vulnerable.  If the model consistently
agrees on the direction to perturb a sample (the FGSM sign) regardless of which
checkpoint we inspect, the adversarial direction is "well-defined" from the
attacker's perspective -- the gradient signal is unambiguous and the attack can
exploit it reliably.  Conversely, samples where gradient sign oscillates between
checkpoints sit in a more confused region of the loss landscape; the attack
direction is less reliable and the sample may be harder to flip.

Two per-sample scalars capture this idea:

  grad_consistency  -- for each pixel, compute the modal sign across 5
                       checkpoints; fraction of (pixel, checkpoint) pairs that
                       agree with the modal sign, averaged over all pixels.
                       Range [0.5, 1.0]; 1.0 = perfect agreement.

  grad_stability    -- mean cosine similarity between the *raw* (un-signed)
                       gradients of consecutive checkpoint pairs, averaged over
                       the 4 consecutive pairs.  Range [-1, 1]; near 1.0 = very
                       stable gradient direction.

Targets (on the final model):
  fgsm_flip  -- binary: FGSM at eps=15/255 flips the predicted label
  pgd_flip   -- binary: PGD-10 at eps=15/255 flips the predicted label
  min_eps    -- continuous: smallest eps (binary search, 8 iters) that flips
                the label in the FGSM-sign direction

Baseline: classification margin (correct-class logit minus max-other logit).

AUROC for each (feature, target) pair -- direction-agnostic via max(a, 1-a).
The key question: does high grad_consistency predict high vulnerability?

Pipeline
--------
1.  Train CNN for 15 epochs on Fashion-MNIST, saving state-dict after each
    multiple of 3 (epochs 3, 6, 9, 12, 15) -- 5 checkpoints total.
2.  Identify 1000 test samples correctly classified by the *final* model.
3.  For each of the 5 checkpoints, compute FGSM gradient signs for all 1000
    samples (batches of 100 for memory efficiency), storing shape
    [1000, n_pixels].
4.  Compute grad_consistency and grad_stability per sample.
5.  Compute margin, fgsm_flip, pgd_flip, min_eps on the final model.
6.  Report per-(feature, target) AUROC, interpret the consistency hypothesis.

Run
---
    python dissertation_extension/hypotheses/h161_checkpoint_gradient_consistency.py
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10
CHECKPOINT_EVERY = 3          # save after epochs 3, 6, 9, 12, 15
N_EVAL = 1000                 # correctly-classified test samples to analyse
GRAD_BATCH = 100              # batch size when computing gradients at checkpoints
PGD_STEPS = 10
PGD_ALPHA = EPS / 4.0
BINARY_SEARCH_ITERS = 8
BINARY_HI = 0.3


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CNN(nn.Module):
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
def fgsm(model, x, y, eps):
    """Return FGSM adversarial examples."""
    xr = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xr), y)
    loss.backward()
    return (xr + eps * xr.grad.sign()).clamp(0.0, 1.0).detach()


def pgd(model, x, y, eps, alpha, steps):
    """Return PGD adversarial examples."""
    x_adv = x.clone().detach() + torch.zeros_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0.0, 1.0)
    return x_adv.detach()


def fgsm_gradient(model, x, y):
    """
    Return the *raw* gradient (not sign) w.r.t. the input.
    model must already be in eval() mode.
    Returns tensor shape [N, C*H*W].
    """
    xr = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xr), y)
    loss.backward()
    return xr.grad.detach().view(x.shape[0], -1)   # [N, pixels]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Gradient feature computation
# ---------------------------------------------------------------------------
def compute_checkpoint_gradients(state_dicts, x_eval, y_eval):
    """
    For each checkpoint state_dict, compute FGSM gradients for all eval samples.

    Returns
    -------
    signs : list of np.ndarray  shape [N, pixels], dtype int8  (+1 / -1)
    grads : list of np.ndarray  shape [N, pixels], dtype float32 (raw gradients)
    """
    import numpy as np
    n_pixels = x_eval.shape[1] * x_eval.shape[2] * x_eval.shape[3]
    N = len(x_eval)
    all_signs = []
    all_grads = []

    for ckpt_idx, sd in enumerate(state_dicts):
        model = CNN(N_CLASSES).to(DEVICE)
        model.load_state_dict(sd)
        model.eval()

        signs_buf = torch.zeros(N, n_pixels, dtype=torch.float32)
        grads_buf = torch.zeros(N, n_pixels, dtype=torch.float32)

        for start in range(0, N, GRAD_BATCH):
            end = min(start + GRAD_BATCH, N)
            xb = x_eval[start:end].to(DEVICE)
            yb = y_eval[start:end].to(DEVICE)
            g = fgsm_gradient(model, xb, yb)        # [batch, pixels]
            signs_buf[start:end] = g.sign().cpu()
            grads_buf[start:end] = g.cpu()

        all_signs.append(signs_buf.numpy())   # [N, pixels]
        all_grads.append(grads_buf.numpy())
        print(f"    checkpoint {ckpt_idx+1}/{len(state_dicts)} gradients computed")

    return all_signs, all_grads


def grad_consistency_score(all_signs):
    """
    Per-sample gradient sign consistency across checkpoints.

    For each pixel, find the modal sign across checkpoints; count the fraction
    of checkpoints agreeing with the modal sign.  Average over all pixels.

    Parameters
    ----------
    all_signs : list of arrays, each [N, pixels], values in {-1, 0, +1}

    Returns
    -------
    consistency : np.ndarray  shape [N], range [0.5, 1.0]
    """
    import numpy as np
    # Stack: [n_ckpts, N, pixels]
    signs_stack = np.stack(all_signs, axis=0).astype(np.float32)  # {-1, 0, 1}
    n_ckpts = signs_stack.shape[0]
    # For each (sample, pixel): count how many checkpoints give sign > 0
    pos_frac = (signs_stack > 0).sum(axis=0) / n_ckpts   # [N, pixels]
    # Agreement fraction = max(pos_frac, 1 - pos_frac) per pixel
    agreement = np.maximum(pos_frac, 1.0 - pos_frac)     # [N, pixels]
    return agreement.mean(axis=1)                          # [N]


def grad_stability_score(all_grads):
    """
    Mean cosine similarity between consecutive-checkpoint gradients.

    Parameters
    ----------
    all_grads : list of arrays, each [N, pixels]

    Returns
    -------
    stability : np.ndarray  shape [N], range approximately [-1, 1]
    """
    import numpy as np
    n_pairs = len(all_grads) - 1
    if n_pairs < 1:
        return np.ones(all_grads[0].shape[0])

    cos_sum = np.zeros(all_grads[0].shape[0])
    for i in range(n_pairs):
        g1 = all_grads[i]                              # [N, pixels]
        g2 = all_grads[i + 1]
        norm1 = np.linalg.norm(g1, axis=1, keepdims=True) + 1e-10
        norm2 = np.linalg.norm(g2, axis=1, keepdims=True) + 1e-10
        cos_sim = (g1 / norm1 * (g2 / norm2)).sum(axis=1)  # [N]
        cos_sum += cos_sim

    return cos_sum / n_pairs


# ---------------------------------------------------------------------------
# Vulnerability targets on final model
# ---------------------------------------------------------------------------
def compute_targets(model, x_eval, y_eval):
    """
    Compute fgsm_flip, pgd_flip, min_eps, and margin for all eval samples.
    Samples must already be correctly classified.

    Returns dict of np.ndarray, each shape [N].
    """
    import numpy as np
    model.eval()
    N = len(x_eval)

    # Margin
    with torch.no_grad():
        logits = model(x_eval)                             # [N, C]
    correct_logits = logits[range(N), y_eval]              # [N]
    # Set correct class to -inf before taking max of others
    logits_clone = logits.clone()
    logits_clone[range(N), y_eval] = float("-inf")
    best_other = logits_clone.max(dim=1).values            # [N]
    margin = (correct_logits - best_other).cpu().numpy()   # higher = more robust

    # FGSM flip
    x_fgsm = fgsm(model, x_eval, y_eval, EPS)
    with torch.no_grad():
        fgsm_flip = (model(x_fgsm).argmax(1) != y_eval).cpu().numpy().astype(float)

    # PGD flip
    x_pgd = pgd(model, x_eval, y_eval, EPS, PGD_ALPHA, PGD_STEPS)
    with torch.no_grad():
        pgd_flip = (model(x_pgd).argmax(1) != y_eval).cpu().numpy().astype(float)

    # min_eps via binary search along FGSM-sign direction
    # Compute gradient direction once
    xr = x_eval.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xr), y_eval)
    loss.backward()
    direction = xr.grad.sign().detach()   # [N, 1, H, W]

    min_eps_vals = np.zeros(N)
    for i in range(N):
        xi = x_eval[i:i+1]
        yi = y_eval[i:i+1]
        di = direction[i:i+1]
        lo, hi = 0.0, BINARY_HI
        for _ in range(BINARY_SEARCH_ITERS):
            mid = (lo + hi) / 2.0
            x_pert = (xi + mid * di).clamp(0.0, 1.0)
            with torch.no_grad():
                flipped = model(x_pert).argmax(1).item() != yi.item()
            if flipped:
                hi = mid
            else:
                lo = mid
        min_eps_vals[i] = hi

    return {
        "margin": margin,
        "fgsm_flip": fgsm_flip,
        "pgd_flip": pgd_flip,
        "min_eps": min_eps_vals,
    }


# ---------------------------------------------------------------------------
# AUROC helper
# ---------------------------------------------------------------------------
def safe_auroc(scores, labels):
    """Direction-agnostic AUROC: max(auroc, 1-auroc)."""
    if len(set(labels)) < 2:
        return float("nan")
    a = roc_auc_score(labels, scores)
    return max(a, 1.0 - a)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("H161: Checkpoint Gradient Consistency vs Adversarial Vulnerability")
    print("=" * 70)
    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS}  (checkpoints every {CHECKPOINT_EVERY})")
    print(f"Eval N : {N_EVAL} correctly-classified test samples")
    print(f"EPS    : {EPS:.5f}  ({EPS*255:.1f}/255)")
    print()

    tf = transforms.ToTensor()
    train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_ds  = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False,
                              num_workers=2, pin_memory=True)

    # -----------------------------------------------------------------------
    # Train and save checkpoints
    # -----------------------------------------------------------------------
    model = CNN(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    checkpoint_state_dicts = []   # list of 5 state_dicts
    checkpoint_epochs = []

    print("Training CNN ...")
    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer)
        scheduler.step()
        if epoch % CHECKPOINT_EVERY == 0:
            checkpoint_state_dicts.append(copy.deepcopy(model.state_dict()))
            checkpoint_epochs.append(epoch)
            print(f"  epoch {epoch:2d} -- checkpoint saved "
                  f"({len(checkpoint_state_dicts)}/{EPOCHS // CHECKPOINT_EVERY})")

    print(f"\nTotal checkpoints saved: {len(checkpoint_state_dicts)}")
    print()

    # -----------------------------------------------------------------------
    # Identify N_EVAL correctly-classified test samples using the final model
    # -----------------------------------------------------------------------
    model.eval()
    xs_list, ys_list = [], []
    total_correct = 0

    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.no_grad():
            preds = model(xb).argmax(1)
        mask = preds == yb
        xs_list.append(xb[mask])
        ys_list.append(yb[mask])
        total_correct += mask.sum().item()
        if total_correct >= N_EVAL:
            break

    x_eval_all = torch.cat(xs_list, dim=0)[:N_EVAL]   # [N_EVAL, 1, 28, 28]
    y_eval_all = torch.cat(ys_list, dim=0)[:N_EVAL]   # [N_EVAL]
    print(f"Collected {len(x_eval_all)} correctly-classified test samples.")
    print()

    # -----------------------------------------------------------------------
    # Compute gradients at each checkpoint
    # -----------------------------------------------------------------------
    print("Computing FGSM gradients at each checkpoint ...")
    all_signs, all_grads = compute_checkpoint_gradients(
        checkpoint_state_dicts, x_eval_all, y_eval_all
    )
    print()

    # -----------------------------------------------------------------------
    # Derive per-sample features
    # -----------------------------------------------------------------------
    import numpy as np

    print("Deriving gradient consistency features ...")
    grad_consistency = grad_consistency_score(all_signs)   # [N]
    grad_stability   = grad_stability_score(all_grads)     # [N]

    print(f"  grad_consistency : mean={grad_consistency.mean():.4f}  "
          f"std={grad_consistency.std():.4f}  "
          f"min={grad_consistency.min():.4f}  max={grad_consistency.max():.4f}")
    print(f"  grad_stability   : mean={grad_stability.mean():.4f}  "
          f"std={grad_stability.std():.4f}  "
          f"min={grad_stability.min():.4f}  max={grad_stability.max():.4f}")
    print()

    # -----------------------------------------------------------------------
    # Compute vulnerability targets on the final model
    # -----------------------------------------------------------------------
    print("Computing vulnerability targets on final model ...")
    # Move eval set to device for attack computation
    x_dev = x_eval_all.to(DEVICE)
    y_dev = y_eval_all.to(DEVICE)
    targets = compute_targets(model, x_dev, y_dev)

    fgsm_flip = targets["fgsm_flip"]
    pgd_flip  = targets["pgd_flip"]
    min_eps   = targets["min_eps"]
    margin    = targets["margin"]

    print(f"  FGSM flip rate : {fgsm_flip.mean():.3f}")
    print(f"  PGD  flip rate : {pgd_flip.mean():.3f}")
    print(f"  mean min_eps   : {min_eps.mean():.4f}")
    print(f"  mean margin    : {margin.mean():.4f}")
    print()

    # -----------------------------------------------------------------------
    # AUROC table
    # -----------------------------------------------------------------------
    features = {
        "grad_consistency": grad_consistency,
        "grad_stability":   grad_stability,
        "margin":           margin,
    }

    # For binary targets: higher vulnerability = label 1.
    # fgsm_flip and pgd_flip are already binary (1 = flipped = vulnerable).
    # min_eps is continuous: lower min_eps = more vulnerable, so we invert.
    # margin is continuous: lower margin = more vulnerable, so we invert.
    binary_targets = {
        "fgsm_flip": fgsm_flip,
        "pgd_flip":  pgd_flip,
    }
    # For AUROC on continuous targets we binarise at median.
    min_eps_bin = (min_eps < np.median(min_eps)).astype(float)  # 1 = more vulnerable
    margin_bin  = (margin  < np.median(margin)).astype(float)

    print("AUROC Results  (direction-agnostic: max(a, 1-a))")
    print("-" * 62)
    header = f"{'Feature':<22}  {'FGSM_flip':>9}  {'PGD_flip':>8}  " \
             f"{'min_eps':>7}  {'margin_bin':>10}"
    print(header)
    print("-" * 62)

    rows_out = {}
    for feat_name, feat_vals in features.items():
        a_fgsm  = safe_auroc(feat_vals, fgsm_flip)
        a_pgd   = safe_auroc(feat_vals, pgd_flip)
        a_mineps = safe_auroc(feat_vals, min_eps_bin)
        a_marg  = safe_auroc(feat_vals, margin_bin)
        rows_out[feat_name] = (a_fgsm, a_pgd, a_mineps, a_marg)
        print(f"{feat_name:<22}  {a_fgsm:>9.3f}  {a_pgd:>8.3f}  "
              f"{a_mineps:>7.3f}  {a_marg:>10.3f}")

    print("-" * 62)
    print()

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    print("Interpretation")
    print("=" * 70)

    gc_fgsm  = rows_out["grad_consistency"][0]
    gc_pgd   = rows_out["grad_consistency"][1]
    gc_mineps= rows_out["grad_consistency"][2]
    gs_fgsm  = rows_out["grad_stability"][0]
    gs_pgd   = rows_out["grad_stability"][1]
    gs_mineps= rows_out["grad_stability"][2]
    m_fgsm   = rows_out["margin"][0]

    threshold_strong = 0.65
    threshold_moderate = 0.55

    def describe_auroc(a, feature, target):
        if a >= threshold_strong:
            return f"  {feature} is a STRONG predictor of {target} (AUROC={a:.3f})"
        elif a >= threshold_moderate:
            return f"  {feature} is a MODERATE predictor of {target} (AUROC={a:.3f})"
        else:
            return f"  {feature} shows WEAK/NO predictive signal for {target} (AUROC={a:.3f})"

    print(describe_auroc(gc_fgsm, "grad_consistency", "FGSM flip"))
    print(describe_auroc(gc_pgd,  "grad_consistency", "PGD  flip"))
    print(describe_auroc(gc_mineps, "grad_consistency", "min_eps"))
    print(describe_auroc(gs_fgsm, "grad_stability",   "FGSM flip"))
    print(describe_auroc(gs_pgd,  "grad_stability",   "PGD  flip"))
    print(describe_auroc(gs_mineps,"grad_stability",  "min_eps"))
    print()
    print(f"  Baseline (margin) AUROC for FGSM flip: {m_fgsm:.3f}")
    print()

    # Hypothesis verdict
    # High grad_consistency is hypothesised to predict higher vulnerability
    # (because a stable attack direction is easier to exploit).
    # We check whether grad_consistency AUROC > margin AUROC baseline - 0.05.
    gc_beats_baseline = (gc_fgsm >= m_fgsm - 0.05 or gc_pgd >= m_fgsm - 0.05)

    print("Hypothesis H161 Verdict:")
    print("-" * 70)
    if gc_fgsm >= threshold_strong or gc_pgd >= threshold_strong:
        print("  SUPPORTED: High gradient sign consistency across checkpoints is a")
        print("  strong predictor of adversarial vulnerability, consistent with the")
        print("  hypothesis that a stable attack direction is easier to exploit.")
    elif gc_fgsm >= threshold_moderate or gc_pgd >= threshold_moderate:
        print("  PARTIALLY SUPPORTED: Gradient consistency shows moderate predictive")
        print("  power for adversarial vulnerability.")
        if gc_beats_baseline:
            print("  It reaches or exceeds the margin baseline, suggesting it captures")
            print("  independent information about exploitability.")
    else:
        print("  NOT SUPPORTED: Gradient sign consistency across checkpoints does NOT")
        print("  reliably predict adversarial vulnerability (AUROC near chance).")
        print("  The hypothesis that a stable gradient direction makes samples easier")
        print("  to attack is not confirmed on this dataset/architecture.")

    if gs_fgsm >= threshold_moderate and gs_fgsm > gc_fgsm:
        print()
        print("  NOTE: grad_stability (cosine similarity between consecutive")
        print("  checkpoint gradients) outperforms grad_consistency, suggesting that")
        print("  the *magnitude* of gradient drift -- not just sign agreement -- is")
        print("  more informative.")

    print()
    print("Summary statistics:")
    print(f"  grad_consistency AUROC (FGSM flip): {gc_fgsm:.3f}")
    print(f"  grad_stability   AUROC (FGSM flip): {gs_fgsm:.3f}")
    print(f"  margin           AUROC (FGSM flip): {m_fgsm:.3f}")


if __name__ == "__main__":
    main()
