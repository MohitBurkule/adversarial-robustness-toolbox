"""
H260 — Targeted vs Untargeted Attack Difficulty
Hypothesis: Targeted attack is harder than untargeted, and the clean margin predicts
this gap. Samples with high margin show greater reduction in targeted success relative
to untargeted success — margin protects more against targeted perturbation.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0; EPS = 0.1; N_EVAL = 300; PGD_STEPS = 20; PGD_ALPHA = 0.01
META = {"channels": 1, "size": 28, "n_classes": 10}
torch.manual_seed(SEED); np.random.seed(SEED)

Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = C.build_model("cnn", META, width=32, seed=SEED)
C.train_model(model, Xtr, Ytr, epochs=10)
model.eval()


def pgd_targeted(mdl: nn.Module, X: torch.Tensor, Y_true: torch.Tensor,
                 Y_target: torch.Tensor, eps: float, steps: int, alpha: float,
                 batch: int = 64) -> torch.Tensor:
    """
    Targeted PGD: minimise loss for Y_target (maximise correct-to-target confusion).
    Processes in batches to stay memory-efficient.
    """
    results = []
    for start in range(0, len(X), batch):
        xb  = X[start:start + batch].clone()
        ytb = Y_target[start:start + batch]
        xb_orig = xb.clone()

        xb = xb + torch.empty_like(xb).uniform_(-eps, eps)
        xb = xb.clamp(0.0, 1.0)
        xb.requires_grad_(True)

        for _ in range(steps):
            logits = mdl(xb)
            # targeted: minimise loss for target class
            loss = -F.cross_entropy(logits, ytb)
            loss.backward()
            with torch.no_grad():
                xb_data = xb.data + alpha * xb.grad.data.sign()
                xb_data = torch.max(torch.min(xb_data, xb_orig + eps), xb_orig - eps)
                xb_data = xb_data.clamp(0.0, 1.0)
            xb = xb_data.clone().requires_grad_(True)

        results.append(xb.detach())
    return torch.cat(results, dim=0)


def main():
    t0 = time.time()
    print("=" * 60)
    print("H260 — Targeted vs Untargeted Attack Difficulty")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Section 1: clean accuracy and margin
    # ------------------------------------------------------------------ #
    print("\n[1] Clean evaluation ...")
    with torch.no_grad():
        logits_c, acc_c = C.logits_and_acc(model, Xte, Yte)
        preds_c = logits_c.argmax(1)
        margins = C.margin(model, Xte, Yte, batch=512)
    correctly_classified = (preds_c.cpu() == Yte.cpu()).cpu().numpy()  # (N,) bool
    print(f"  Clean accuracy: {acc_c:.4f}")
    print(f"  Clean margin  : mean={margins.mean():.4f}  std={margins.std():.4f}")
    print(f"  Correctly classified: {correctly_classified.sum()} / {N_EVAL}")

    # ------------------------------------------------------------------ #
    # Section 2: untargeted PGD
    # ------------------------------------------------------------------ #
    print("\n[2] Untargeted PGD attack ...")
    for p in model.parameters():
        p.requires_grad_(True)
    Xadv_untarg = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    model.eval()

    with torch.no_grad():
        logits_u, _ = C.logits_and_acc(model, Xadv_untarg, Yte)
        preds_u = logits_u.argmax(1)
        untarg_success = (correctly_classified & (preds_u.cpu().numpy() != Yte.cpu().numpy())).astype(int)
    print(f"  Untargeted ASR: {untarg_success.mean():.4f}  ({untarg_success.sum()}/{N_EVAL})")

    # ------------------------------------------------------------------ #
    # Section 3: targeted PGD (target = (true + 1) % 10)
    # ------------------------------------------------------------------ #
    print("\n[3] Targeted PGD attack (target = (true + 1) % 10) ...")
    Y_target = (Yte + 1) % META["n_classes"]

    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    Xadv_targ = pgd_targeted(model, Xte, Yte, Y_target,
                              eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    model.eval()

    with torch.no_grad():
        logits_t, _ = C.logits_and_acc(model, Xadv_targ, Yte)
        preds_t = logits_t.argmax(1)
        # targeted success: classified as target class
        targ_success = (correctly_classified & (preds_t.cpu().numpy() == Y_target.cpu().numpy())).astype(int)
    print(f"  Targeted ASR  : {targ_success.mean():.4f}  ({targ_success.sum()}/{N_EVAL})")

    # ------------------------------------------------------------------ #
    # Section 4: difficulty gap per sample
    # ------------------------------------------------------------------ #
    print("\n[4] Computing per-sample difficulty gap ...")
    gap = untarg_success.astype(float) - targ_success.astype(float)
    # gap > 0: untargeted succeeds but targeted fails (targeted is harder)
    # gap < 0: targeted succeeds but untargeted fails (unusual)
    # gap = 0: both succeed or both fail
    gap_positive = (gap > 0).sum()
    gap_negative = (gap < 0).sum()
    gap_zero     = (gap == 0).sum()
    print(f"  gap > 0 (untarg only): {gap_positive}")
    print(f"  gap = 0 (both or neither): {gap_zero}")
    print(f"  gap < 0 (targ only):   {gap_negative}")
    print(f"  Mean gap: {gap.mean():+.4f}  std={gap.std():.4f}")

    # ------------------------------------------------------------------ #
    # Section 5: correlation of gap with margin
    # ------------------------------------------------------------------ #
    print("\n[5] Correlation: gap vs margin")
    print("-" * 55)
    rho_gap_marg, p_gap_marg = spearmanr(gap, margins)
    print(f"  Spearman rho (gap vs margin):             {rho_gap_marg:+.4f}  p={p_gap_marg:.4f}")

    rho_u_marg, p_u_marg = spearmanr(untarg_success.astype(float), -margins)
    rho_t_marg, p_t_marg = spearmanr(targ_success.astype(float),   -margins)
    print(f"  Spearman rho (untarg_success vs -margin): {rho_u_marg:+.4f}  p={p_u_marg:.4f}")
    print(f"  Spearman rho (targ_success   vs -margin): {rho_t_marg:+.4f}  p={p_t_marg:.4f}")

    # ------------------------------------------------------------------ #
    # Section 6: AUROC
    # ------------------------------------------------------------------ #
    print("\n[6] AUROC — does margin predict attack success?")
    print("-" * 55)
    for label, arr in [("untarg_success", untarg_success), ("targ_success", targ_success)]:
        if arr.sum() > 0 and arr.sum() < N_EVAL:
            auc = roc_auc_score(arr, -margins)
            print(f"  AUROC (-margin -> {label}): {auc:.4f}")
        else:
            print(f"  AUROC (-margin -> {label}): skipped (trivial)")

    # ------------------------------------------------------------------ #
    # Section 7: margin quantile analysis
    # ------------------------------------------------------------------ #
    print("\n[7] Attack success rate by margin quartile")
    print("-" * 55)
    quartiles = np.percentile(margins, [25, 50, 75])
    bounds = [-np.inf] + list(quartiles) + [np.inf]
    for q in range(4):
        mask = (margins >= bounds[q]) & (margins < bounds[q + 1])
        if mask.sum() == 0:
            continue
        u_asr = untarg_success[mask].mean()
        t_asr = targ_success[mask].mean()
        diff  = u_asr - t_asr
        print(f"  Q{q+1} (n={mask.sum():3d}, margin [{bounds[q]:+.3f}, {bounds[q+1]:+.3f})):  "
              f"untarg={u_asr:.4f}  targ={t_asr:.4f}  gap={diff:+.4f}")

    # ------------------------------------------------------------------ #
    # Section 8: summary
    # ------------------------------------------------------------------ #
    print("\n[8] Summary")
    print("-" * 55)
    overall_gap = untarg_success.mean() - targ_success.mean()
    print(f"  Overall ASR gap (untarg - targ): {overall_gap:+.4f}")
    print(f"  Targeted is harder: {'YES' if overall_gap > 0 else 'NO'}")
    print(f"  Margin correlates with gap (rho={rho_gap_marg:+.4f}): "
          f"{'YES (p<0.05)' if p_gap_marg < 0.05 else 'NOT SIGNIFICANT'}")

    print(f"\nDone in {time.time() - t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
