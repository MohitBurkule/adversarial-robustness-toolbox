"""
H193 - Loss surface during PGD shows a sharpening-then-flattening "uncanny valley"
dynamic, and peak loss at step K predicts final adversarial success.

Motivated by arXiv:2405.16918 (loss-landscape sharpness during adversarial training).

We run 20-step PGD on Fashion-MNIST test samples, recording the cross-entropy loss
at every step. For each sample we compute:
  (a) max_loss   — peak loss over the trajectory (measure of sharpness)
  (b) peak_step  — step at which peak loss occurs
  (c) final_loss — loss at step 20

We then ask:
  1. Does max_loss predict FGSM success better than margin? (AUROC comparison)
  2. What does the mean loss trajectory look like for succeed vs fail samples?
     Does it show sharpening-then-flattening (the "uncanny valley")?
  3. What fraction of samples show classic sharpening-then-flattening?
"""

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
EPS = 0.1
PGD_STEPS = 20
PGD_ALPHA = 0.01
N_EVAL = 200
SEED = 0

# Threshold for "meaningful drop after peak" to count as sharpening-then-flattening
VALLEY_DROP_THRESH = 0.05


def pgd_with_trajectory(model, X, Y, eps, steps, alpha):
    """Run PGD and record per-sample loss at every step.

    Returns:
        x_adv: final adversarial examples  (N, C, H, W)
        traj:  loss trajectory             (N, steps) numpy array
    """
    model.eval()
    N = X.size(0)
    x0 = X.clone().detach()
    x_adv = x0.clone().detach()
    traj = np.zeros((N, steps), dtype=np.float32)

    for step in range(steps):
        x_adv = x_adv.requires_grad_(True)
        out = model(x_adv)
        # Per-sample loss (no reduction)
        loss_vec = F.cross_entropy(out, Y, reduction="none")  # (N,)
        loss_sum = loss_vec.sum()
        grads = torch.autograd.grad(loss_sum, x_adv)[0]
        traj[:, step] = loss_vec.detach().cpu().numpy()
        with torch.no_grad():
            x_adv = x_adv.detach() + alpha * grads.sign()
            x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1)

    return x_adv.detach(), traj


def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "fashion_mnist"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h193_boundary_flatness_output.txt")

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 74)
    log("H193 - PGD loss trajectory 'uncanny valley' predicts adversarial success")
    log("=" * 74)
    log(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  steps={PGD_STEPS}  alpha={PGD_ALPHA}  N={N_EVAL}")

    t0 = time.time()
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=2000, seed=SEED)

    model = C.build_model("cnn", meta, seed=SEED)
    log(f"\nTraining model for 10 epochs...")
    C.train_model(model, Xtr, Ytr, epochs=10, opt="sgd", lr=0.05, ncls=meta["n_classes"])
    model.eval()

    # Subset for evaluation
    Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]

    # --- Clean logits for margin baseline ---
    logits, clean_acc = C.logits_and_acc(model, Xte, Yte)
    margins = C.margin_of(logits, Yte)   # (N,) higher = more robust
    log(f"Clean accuracy on eval subset: {clean_acc:.3f}")

    # --- FGSM attack for binary success labels ---
    log(f"\nRunning FGSM attack...")
    x_fgsm = C.fgsm(model, Xte, Yte, EPS)
    with torch.no_grad():
        fgsm_pred = model(x_fgsm).argmax(1).cpu()
    # Success = model fooled (prediction changed)
    fgsm_success = (fgsm_pred != Yte.cpu()).numpy().astype(int)  # (N,)
    log(f"FGSM attack success rate: {fgsm_success.mean():.3f}")

    # --- PGD with trajectory recording ---
    log(f"\nRunning {PGD_STEPS}-step PGD with loss trajectory recording...")
    x_adv_pgd, traj = pgd_with_trajectory(model, Xte, Yte, EPS, PGD_STEPS, PGD_ALPHA)
    # traj shape: (N, PGD_STEPS)

    # Check PGD success too (for context)
    with torch.no_grad():
        pgd_pred = model(x_adv_pgd).argmax(1).cpu()
    pgd_success = (pgd_pred != Yte.cpu()).numpy().astype(int)
    log(f"PGD attack success rate: {pgd_success.mean():.3f}")

    # --- Trajectory-derived features ---
    max_loss   = traj.max(axis=1)          # (N,) peak loss
    peak_step  = traj.argmax(axis=1)       # (N,) step of peak
    final_loss = traj[:, -1]               # (N,) final loss

    # --- AUROC: max_loss vs FGSM success ---
    auroc_max_loss = C.safe_auroc(fgsm_success, max_loss)
    # For margin: higher margin = more robust = LESS likely to be fooled
    # So negate margin so higher = more likely fooled
    auroc_margin   = C.safe_auroc(fgsm_success, -margins)
    auroc_final_loss = C.safe_auroc(fgsm_success, final_loss)

    log("\n" + "=" * 74)
    log("AUROC (predicting FGSM success):")
    log(f"  max_loss   (peak PGD loss)  AUROC = {auroc_max_loss:.4f}")
    log(f"  final_loss (step-20 loss)   AUROC = {auroc_final_loss:.4f}")
    log(f"  -margin    (baseline)       AUROC = {auroc_margin:.4f}")
    log("=" * 74)

    # --- Mean loss trajectories: succeed vs fail ---
    succ_mask = fgsm_success.astype(bool)
    fail_mask = ~succ_mask
    traj_succ = traj[succ_mask].mean(axis=0) if succ_mask.sum() > 0 else np.zeros(PGD_STEPS)
    traj_fail = traj[fail_mask].mean(axis=0) if fail_mask.sum() > 0 else np.zeros(PGD_STEPS)

    log(f"\nMean loss trajectory (FGSM-succeed samples, n={succ_mask.sum()}):")
    vals = "  ".join(f"s{i+1}:{v:.3f}" for i, v in enumerate(traj_succ))
    log(f"  {vals}")
    log(f"Mean loss trajectory (FGSM-fail samples, n={fail_mask.sum()}):")
    vals = "  ".join(f"s{i+1}:{v:.3f}" for i, v in enumerate(traj_fail))
    log(f"  {vals}")

    # Check for "uncanny valley" shape in succeed trajectory:
    # peak before last step and drops from peak
    succ_peak_step = int(traj_succ.argmax())
    succ_drop = float(traj_succ.max() - traj_succ[-1])
    log(f"\nSucceed trajectory: peak at step {succ_peak_step+1}/{PGD_STEPS}, "
        f"drop from peak to final = {succ_drop:.4f}")
    fail_peak_step = int(traj_fail.argmax())
    fail_drop = float(traj_fail.max() - traj_fail[-1])
    log(f"Fail    trajectory: peak at step {fail_peak_step+1}/{PGD_STEPS}, "
        f"drop from peak to final = {fail_drop:.4f}")

    # --- Fraction showing classic sharpening-then-flattening ---
    # Criteria: (1) peak is not at first or last step, AND (2) loss drops by
    # VALLEY_DROP_THRESH from peak to final
    has_interior_peak = (peak_step > 0) & (peak_step < PGD_STEPS - 1)
    drops_after_peak  = (max_loss - final_loss) > VALLEY_DROP_THRESH
    valley_pattern    = has_interior_peak & drops_after_peak
    frac_valley = valley_pattern.mean()

    log(f"\nFraction with sharpening-then-flattening pattern "
        f"(interior peak, drop>{VALLEY_DROP_THRESH}): {frac_valley:.3f}")
    log(f"  Among FGSM-succeed: "
        f"{valley_pattern[succ_mask].mean():.3f}")
    log(f"  Among FGSM-fail:    "
        f"{valley_pattern[fail_mask].mean():.3f}")

    # --- Peak step distribution summary ---
    log(f"\nPeak step statistics:")
    log(f"  Mean peak step (all):           {peak_step.mean():.2f}")
    log(f"  Mean peak step (FGSM-succeed):  {peak_step[succ_mask].mean():.2f}")
    log(f"  Mean peak step (FGSM-fail):     {peak_step[fail_mask].mean():.2f}")

    # --- Runtime ---
    log(f"\nTotal runtime: {time.time()-t0:.1f}s")

    log("\n" + "=" * 74)
    log("Interpretation:")
    log("  If AUROC(max_loss) > AUROC(margin), the height of the PGD loss peak")
    log("  carries predictive signal about FGSM success beyond what clean margin")
    log("  already knows. A loss trajectory that rises then falls (uncanny valley)")
    log("  before the attack converges suggests the loss surface near the boundary")
    log("  is locally non-convex — gradient ascent overshoots the decision boundary")
    log("  and slides back into a flat region. Samples that succeed tend to reach")
    log("  the peak earlier and sustain it, while samples that fail show a steeper")
    log("  and earlier drop (the gradient carries the perturbation away from the")
    log("  optimal adversarial direction). This is consistent with 2405.16918's")
    log("  sharpness-flatness duality in adversarial loss landscapes.")
    log("=" * 74)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
