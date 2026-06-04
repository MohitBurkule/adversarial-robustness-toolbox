"""
H190 - Step-to-Flip (STF) as a per-sample vulnerability predictor vs margin.

Instead of binary PGD success/fail at fixed steps, measure at each step whether
the prediction has flipped. STF = first step where flip occurs (21 if no flip in
20 steps). Low STF = high vulnerability.

Metrics:
  1. AUROC: -STF → FGSM success (binary)
  2. AUROC: -margin → FGSM success (baseline)
  3. AUROC: -STF → PGD-20 success
  4. Spearman rho between STF and margin
  5. Distribution of STF: fraction flipping at step 1, 5, 10, 20, never
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
EPS = 0.1
PGD_MAX_STEPS = 20
STEP_SIZE = EPS / 4
N_EVAL = 300


def compute_stf(model, X, Y, eps=EPS, max_steps=PGD_MAX_STEPS, step_size=STEP_SIZE):
    """
    Compute Step-to-Flip for each sample in X.
    Returns array of shape (N,) with values in {1, ..., max_steps, max_steps+1}.
    max_steps+1 means no flip occurred.
    """
    model.eval()
    N = X.shape[0]
    stf = torch.full((N,), max_steps + 1, dtype=torch.float32, device=X.device)

    # Track which samples still haven't flipped
    not_flipped = torch.ones(N, dtype=torch.bool, device=X.device)

    # Get original predictions
    with torch.no_grad():
        orig_preds = model(X).argmax(dim=1)

    # PGD initialisation: start from X (no random start for determinism)
    delta = torch.zeros_like(X)
    delta.requires_grad_(True)

    # We need per-step iteration; rebuild graph each step
    X_adv = X.clone().detach()
    delta = torch.zeros_like(X_adv, requires_grad=False)

    for step in range(1, max_steps + 1):
        delta_var = delta.clone().detach().requires_grad_(True)
        logits = model(torch.clamp(X_adv + delta_var, 0.0, 1.0))
        # untargeted: maximise CE w.r.t. true labels
        loss = F.cross_entropy(logits, Y, reduction='sum')
        loss.backward()

        with torch.no_grad():
            delta = delta + step_size * delta_var.grad.sign()
            delta = delta.clamp(-eps, eps)

            # Check which samples have flipped this step
            adv_preds = model(torch.clamp(X_adv + delta, 0.0, 1.0)).argmax(dim=1)
            newly_flipped = not_flipped & (adv_preds != orig_preds)
            stf[newly_flipped] = float(step)
            not_flipped = not_flipped & ~newly_flipped

            if not not_flipped.any():
                break  # all flipped

    return stf.cpu().numpy()


def main():
    t0 = time.time()
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 60)
    log("H190: Step-to-Flip (STF) as per-sample vulnerability predictor")
    log("=" * 60)

    # Load data
    log("\n[1] Loading Fashion-MNIST …")
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    X = Xte[:N_EVAL]
    Y = Yte[:N_EVAL]
    log(f"    Using {N_EVAL} test samples  (X shape: {tuple(X.shape)})")

    # Build and train model
    log("\n[2] Building and training CNN …")
    model = C.build_model("cnn", meta, width=32, seed=0)
    C.train_model(model, Xtr, Ytr, epochs=10)

    # Clean accuracy
    with torch.no_grad():
        logits_te = model(X)
        clean_preds = logits_te.argmax(dim=1)
        clean_acc = (clean_preds == Y).float().mean().item()
    log(f"    Clean accuracy on eval subset: {clean_acc:.3f}")

    # Compute STF
    log("\n[3] Computing Step-to-Flip (max 20 PGD steps) …")
    stf = compute_stf(model, X, Y, eps=EPS, max_steps=PGD_MAX_STEPS)
    log(f"    STF computed. min={stf.min():.0f}, max={stf.max():.0f}, "
        f"mean={stf.mean():.2f}, median={np.median(stf):.1f}")

    # Compute margin
    log("\n[4] Computing margin …")
    logits_m, _ = C.logits_and_acc(model, X, Y)
    margin = C.margin_of(logits_m, Y)
    log(f"    Margin computed. min={margin.min():.4f}, max={margin.max():.4f}, "
        f"mean={margin.mean():.4f}")

    # FGSM binary labels
    log("\n[5] Computing FGSM success labels (eps=0.1) …")
    X_fgsm = C.fgsm(model, X, Y, eps=EPS)
    with torch.no_grad():
        fgsm_preds = model(X_fgsm).argmax(dim=1)
    fgsm_success = (fgsm_preds != Y).cpu().numpy().astype(int)
    log(f"    FGSM attack success rate: {fgsm_success.mean():.3f}")

    # PGD-20 binary labels
    log("\n[6] Computing PGD-20 success labels …")
    # If STF <= 20, PGD-20 succeeded
    pgd20_success = (stf <= PGD_MAX_STEPS).astype(int)
    log(f"    PGD-20 attack success rate: {pgd20_success.mean():.3f}")

    # AUROC calculations
    log("\n[7] AUROC calculations …")

    # Only compute AUROC when both classes present
    def safe_auroc(scores, labels, label=""):
        if labels.sum() == 0 or labels.sum() == len(labels):
            log(f"    AUROC {label}: N/A (only one class in labels)")
            return float("nan")
        auc = roc_auc_score(labels, scores)
        log(f"    AUROC {label}: {auc:.4f}")
        return auc

    auc_stf_fgsm  = safe_auroc(-stf,    fgsm_success,  "-STF → FGSM success")
    auc_margin_fgsm = safe_auroc(-margin, fgsm_success, "-margin → FGSM success")
    auc_stf_pgd   = safe_auroc(-stf,    pgd20_success, "-STF → PGD-20 success")

    # Spearman rho between STF and margin
    log("\n[8] Spearman correlation: STF vs margin …")
    rho, pval = spearmanr(stf, margin)
    log(f"    Spearman rho = {rho:.4f}  (p = {pval:.4e})")

    # STF distribution
    log("\n[9] STF distribution …")
    total = len(stf)
    for cutoff, label in [(1, "flip at step 1"),
                          (5, "flip at step ≤5"),
                          (10, "flip at step ≤10"),
                          (20, "flip at step ≤20"),
                          (PGD_MAX_STEPS + 1, "never flipped (STF=21)")]:
        if label.startswith("never"):
            count = (stf == PGD_MAX_STEPS + 1).sum()
        else:
            count = (stf <= cutoff).sum()
        log(f"    {label}: {count}/{total} ({100*count/total:.1f}%)")

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  AUROC -STF   → FGSM success : {auc_stf_fgsm:.4f}")
    log(f"  AUROC -margin→ FGSM success : {auc_margin_fgsm:.4f}")
    log(f"  AUROC -STF   → PGD-20 success: {auc_stf_pgd:.4f}")
    log(f"  Spearman rho (STF vs margin) : {rho:.4f}  p={pval:.2e}")

    if not np.isnan(auc_stf_fgsm) and not np.isnan(auc_margin_fgsm):
        if auc_stf_fgsm > auc_margin_fgsm:
            log("\n  VERDICT: STF OUTPERFORMS margin as FGSM vulnerability predictor.")
        elif auc_stf_fgsm > auc_margin_fgsm - 0.01:
            log("\n  VERDICT: STF is COMPARABLE to margin as FGSM vulnerability predictor.")
        else:
            log("\n  VERDICT: margin OUTPERFORMS STF as FGSM vulnerability predictor.")

    log(f"\nTotal runtime: {time.time() - t0:.1f}s")

    # Write output
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h190_step_to_flip_output.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nOutput saved to {out_path}")


if __name__ == "__main__":
    main()
