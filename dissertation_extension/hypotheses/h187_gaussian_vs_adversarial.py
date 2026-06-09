"""
H187 - Per-sample adversarial vulnerability correlates with sensitivity to Gaussian noise.

Paper 2601.14519 shows adversarial and random noise robustness are not equivalent at
aggregate level, but the per-sample correlation is unexplored. Here we test whether
a sample's susceptibility to Gaussian noise (σ=0.1, 20 trials) predicts susceptibility
to FGSM and PGD attacks at the per-sample level.

Metrics:
  - AUROC: Gaussian noise success -> FGSM success
  - AUROC: Gaussian noise success -> PGD success
  - AUROC: FGSM success -> PGD success (baseline comparison)
  - Fraction of samples: both fail / both succeed / only adversarial / only Gaussian

A high AUROC would suggest aggregate-level findings extend to per-sample correlations.
A low AUROC would confirm that the two phenomena are orthogonal even at the sample level.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DATASET = "fashion_mnist"
N = 500
EPS = 0.1
PGD_STEPS = 10
GAUSSIAN_SIGMA = 0.1
GAUSSIAN_REPEATS = 20
SEED = 0

META = {"channels": 1, "size": 28, "n_classes": 10}


def gaussian_success(model, X, Y, sigma=0.1, repeats=20, seed=0):
    """For each sample, check if ANY of `repeats` Gaussian perturbations flips the prediction."""
    model.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        clean_preds = model(X).argmax(1)
        # Only track samples that are correctly classified
        correct_mask = (clean_preds == Y).cpu().numpy()

        flipped = torch.zeros(len(X), dtype=torch.bool, device=X.device)
        for _ in range(repeats):
            noise = torch.randn_like(X) * sigma
            Xn = (X + noise).clamp(0.0, 1.0)
            preds = model(Xn).argmax(1)
            flipped |= (preds != clean_preds)

    return flipped.cpu().numpy(), correct_mask


def adversarial_success(model, X, Y, method="fgsm"):
    """Binary success (prediction flips on correctly-classified samples)."""
    model.eval()
    with torch.no_grad():
        clean_preds = model(X).argmax(1)
        correct_mask = (clean_preds == Y).cpu().numpy()

    if method == "fgsm":
        Xadv = C.fgsm(model, X, Y, eps=EPS)
    elif method == "pgd":
        Xadv = C.pgd(model, X, Y, eps=EPS, steps=PGD_STEPS)
    else:
        raise ValueError(f"Unknown method: {method}")

    with torch.no_grad():
        adv_preds = model(Xadv).argmax(1)
        success = (adv_preds != clean_preds).cpu().numpy()

    return success, correct_mask


def safe_auroc(y_true, y_score):
    """Return AUROC or NaN if only one class present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def main():
    t0 = time.time()
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("H187: Per-sample Gaussian noise sensitivity vs adversarial vulnerability")
    log("=" * 70)
    log(f"Dataset: {DATASET} | N={N} | eps={EPS} | sigma={GAUSSIAN_SIGMA} | repeats={GAUSSIAN_REPEATS}")
    log()

    # Load data
    log("Loading dataset...")
    Xtr, Ytr, Xte, Yte = C.load_dataset(DATASET)
    X = Xte[:N]
    Y = Yte[:N]
    log(f"Test subset: {X.shape}")
    log()

    # Build and train model
    log("Building and training model (CNN width=32, seed=0, 10 epochs)...")
    model = C.build_model("cnn", META, width=32, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=10)
    log()

    # Evaluate clean accuracy
    model.eval()
    with torch.no_grad():
        clean_acc = (model(X).argmax(1) == Y).float().mean().item()
    log(f"Clean accuracy on test subset: {clean_acc:.4f}")
    log()

    # Compute attack successes
    log("Computing FGSM successes...")
    fgsm_succ, fgsm_correct = adversarial_success(model, X, Y, method="fgsm")

    log("Computing PGD successes...")
    pgd_succ, pgd_correct = adversarial_success(model, X, Y, method="pgd")

    log("Computing Gaussian noise successes (20 trials)...")
    gauss_succ, gauss_correct = gaussian_success(model, X, Y, sigma=GAUSSIAN_SIGMA, repeats=GAUSSIAN_REPEATS)
    log()

    # Restrict to correctly classified samples (attacks only meaningful there)
    both_correct = fgsm_correct & pgd_correct & gauss_correct
    log(f"Samples correctly classified (all methods): {both_correct.sum()} / {N}")
    log()

    f = fgsm_succ[both_correct].astype(int)
    p = pgd_succ[both_correct].astype(int)
    g = gauss_succ[both_correct].astype(int)

    # AUROC calculations
    auroc_gauss_fgsm = safe_auroc(f, g)
    auroc_gauss_pgd  = safe_auroc(p, g)
    auroc_fgsm_pgd   = safe_auroc(p, f)

    log("=" * 70)
    log("AUROC Results (on correctly-classified samples only)")
    log("=" * 70)
    log(f"  Gaussian -> FGSM  : {auroc_gauss_fgsm:.4f}  (predicts FGSM success from Gaussian)")
    log(f"  Gaussian -> PGD   : {auroc_gauss_pgd:.4f}  (predicts PGD success from Gaussian)")
    log(f"  FGSM -> PGD       : {auroc_fgsm_pgd:.4f}  (baseline: FGSM predicts PGD)")
    log()

    # Attack success rates
    log("Attack success rates (on correctly-classified samples):")
    log(f"  FGSM ASR  : {f.mean():.4f}")
    log(f"  PGD ASR   : {p.mean():.4f}")
    log(f"  Gaussian  : {g.mean():.4f}")
    log()

    # Contingency breakdown
    n_cc = len(f)

    def frac(mask): return mask.sum() / n_cc

    both_fail_fg   = frac((f == 0) & (g == 0))
    both_succ_fg   = frac((f == 1) & (g == 1))
    only_fgsm      = frac((f == 1) & (g == 0))
    only_gauss     = frac((f == 0) & (g == 1))

    both_fail_pg   = frac((p == 0) & (g == 0))
    both_succ_pg   = frac((p == 1) & (g == 1))
    only_pgd       = frac((p == 1) & (g == 0))
    only_gauss2    = frac((p == 0) & (g == 1))

    log("Contingency fractions - FGSM vs Gaussian (on correctly-classified):")
    log(f"  Both fail (robust to both)   : {both_fail_fg:.4f}")
    log(f"  Both succeed (vulnerable)    : {both_succ_fg:.4f}")
    log(f"  Only FGSM succeeds           : {only_fgsm:.4f}")
    log(f"  Only Gaussian succeeds       : {only_gauss:.4f}")
    log()

    log("Contingency fractions - PGD vs Gaussian (on correctly-classified):")
    log(f"  Both fail (robust to both)   : {both_fail_pg:.4f}")
    log(f"  Both succeed (vulnerable)    : {both_succ_pg:.4f}")
    log(f"  Only PGD succeeds            : {only_pgd:.4f}")
    log(f"  Only Gaussian succeeds       : {only_gauss2:.4f}")
    log()

    # Interpretation
    log("=" * 70)
    log("Interpretation")
    log("=" * 70)
    threshold_high = 0.65
    threshold_near_chance = 0.55

    def interpret_auroc(name, val):
        if np.isnan(val):
            return f"  {name}: NaN (degenerate — check class balance)"
        elif val >= threshold_high:
            return f"  {name}: {val:.4f} — moderate-to-strong correlation (per-sample linkage exists)"
        elif val >= threshold_near_chance:
            return f"  {name}: {val:.4f} — weak correlation (slightly above chance)"
        else:
            return f"  {name}: {val:.4f} — near chance (orthogonal vulnerabilities)"

    log(interpret_auroc("Gaussian -> FGSM", auroc_gauss_fgsm))
    log(interpret_auroc("Gaussian -> PGD ", auroc_gauss_pgd))
    log(interpret_auroc("FGSM     -> PGD ", auroc_fgsm_pgd))
    log()

    elapsed = time.time() - t0
    log(f"Elapsed: {elapsed:.1f}s")
    log("DONE")

    # Save output
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h187_gaussian_vs_adversarial_output.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nOutput saved to: {out_path}")


if __name__ == "__main__":
    main()
