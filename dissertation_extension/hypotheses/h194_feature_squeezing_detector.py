"""
H194 - Feature squeezing prediction-change magnitude as per-sample vulnerability predictor.

Xu et al. (2018) use feature squeezing (bit-depth reduction) as adversarial detection.
We test whether the clean-image prediction-change magnitude after squeezing predicts
adversarial vulnerability *before* any attack is run.

Squeezing score: quantize pixels to 2-bit (3 levels) or 4-bit (7 levels), then compute
L1 distance between original and squeezed softmax distributions.

High squeezing score ⇒ sample is near a decision boundary ⇒ should be easier to attack.

Key outputs:
  1. AUROC: 2-bit squeezing score → FGSM success
  2. AUROC: 4-bit squeezing score → FGSM success
  3. AUROC: 2-bit squeezing score → PGD success
  4. AUROC: margin → FGSM/PGD success (baseline)
  5. Fraction of samples where squeezing changes predicted class
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N = 500
EPS = 0.1
PGD_STEPS = 10
SEED = 0


def squeeze(X: torch.Tensor, levels: int) -> torch.Tensor:
    """Quantize pixel values to `levels` uniformly-spaced steps in [0,1]."""
    # levels=3 → 2-bit approx; levels=7 → ~3-bit
    return (X * (levels - 1)).round() / (levels - 1)


def squeezing_score(model: torch.nn.Module, X: torch.Tensor, levels: int) -> torch.Tensor:
    """L1 distance between original and squeezed softmax distributions (per sample)."""
    model.eval()
    with torch.no_grad():
        p_orig = F.softmax(model(X), dim=1)
        X_sq = squeeze(X, levels).clamp(0.0, 1.0)
        p_sq = F.softmax(model(X_sq), dim=1)
        score = (p_orig - p_sq).abs().sum(dim=1)
    return score


def run():
    t0 = time.time()
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 60)
    log("H194  Feature-squeezing as vulnerability predictor")
    log("=" * 60)

    # ── data & model ──────────────────────────────────────────────
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    Xte, Yte = Xte[:N], Yte[:N]
    log(f"Dataset : {DS}  |  test subset : {N}")

    model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                          width=32, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=10)
    model.eval()

    # ── clean accuracy ────────────────────────────────────────────
    with torch.no_grad():
        logits_clean = model(Xte)
    pred_clean = logits_clean.argmax(dim=1)
    clean_acc = (pred_clean == Yte).float().mean().item()
    log(f"Clean accuracy : {clean_acc:.4f}")

    # ── squeezing scores ──────────────────────────────────────────
    sq2 = squeezing_score(model, Xte, levels=3).cpu().numpy()   # 2-bit (3 levels)
    sq4 = squeezing_score(model, Xte, levels=7).cpu().numpy()   # ~3-bit (7 levels)

    # fraction where top-1 class changes after squeezing
    with torch.no_grad():
        pred_sq2 = F.softmax(model(squeeze(Xte, 3).clamp(0, 1)), dim=1).argmax(dim=1)
        pred_sq4 = F.softmax(model(squeeze(Xte, 7).clamp(0, 1)), dim=1).argmax(dim=1)

    frac_change2 = (pred_sq2 != pred_clean).float().mean().item()
    frac_change4 = (pred_sq4 != pred_clean).float().mean().item()
    log(f"Fraction class-change after 2-bit squeeze : {frac_change2:.4f}")
    log(f"Fraction class-change after 4-bit squeeze : {frac_change4:.4f}")

    # ── margin baseline ───────────────────────────────────────────
    logits_m, _ = C.logits_and_acc(model, Xte, Yte)
    margin = C.margin_of(logits_m, Yte)

    # ── attacks ───────────────────────────────────────────────────
    log("Running FGSM …")
    X_fgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    with torch.no_grad():
        pred_fgsm = model(X_fgsm).argmax(dim=1)
    fgsm_success = (pred_fgsm != Yte).cpu().numpy().astype(int)

    log("Running PGD …")
    X_pgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS)
    with torch.no_grad():
        pred_pgd = model(X_pgd).argmax(dim=1)
    pgd_success = (pred_pgd != Yte).cpu().numpy().astype(int)

    fgsm_rate = fgsm_success.mean()
    pgd_rate  = pgd_success.mean()
    log(f"FGSM attack success rate : {fgsm_rate:.4f}")
    log(f"PGD  attack success rate : {pgd_rate:.4f}")

    # ── AUROCs ───────────────────────────────────────────────────
    log("")
    log("── AUROCs (higher = better predictor of vulnerability) ──")

    def safe_auroc(scores, labels, name):
        if labels.sum() == 0 or labels.sum() == len(labels):
            log(f"  {name}: n/a (degenerate labels)")
            return float("nan")
        auc = roc_auc_score(labels, scores)
        log(f"  {name}: {auc:.4f}")
        return auc

    log("FGSM success prediction:")
    auc_sq2_fgsm  = safe_auroc(sq2,    fgsm_success, "2-bit squeeze score → FGSM")
    auc_sq4_fgsm  = safe_auroc(sq4,    fgsm_success, "4-bit squeeze score → FGSM")
    auc_margin_fgsm = safe_auroc(-margin, fgsm_success, "Neg-margin          → FGSM")

    log("PGD success prediction:")
    auc_sq2_pgd   = safe_auroc(sq2,    pgd_success,  "2-bit squeeze score → PGD ")
    auc_sq4_pgd   = safe_auroc(sq4,    pgd_success,  "4-bit squeeze score → PGD ")
    auc_margin_pgd  = safe_auroc(-margin, pgd_success,  "Neg-margin          → PGD ")

    # ── per-decile breakdown ──────────────────────────────────────
    log("")
    log("── Per-decile FGSM success rate (by 2-bit squeeze score) ──")
    deciles = np.percentile(sq2, np.arange(0, 100, 10))
    bins = np.digitize(sq2, deciles)
    for b in range(1, 11):
        mask = bins == b
        if mask.sum() == 0:
            continue
        rate = fgsm_success[mask].mean()
        log(f"  Decile {b:2d}  (sq2 ≥ {deciles[b-1]:.4f}): "
            f"n={mask.sum():3d}  FGSM-success={rate:.3f}")

    # ── score statistics ──────────────────────────────────────────
    log("")
    log("── Squeeze score statistics ──")
    log(f"  2-bit: mean={sq2.mean():.4f}  std={sq2.std():.4f}  "
        f"min={sq2.min():.4f}  max={sq2.max():.4f}")
    log(f"  4-bit: mean={sq4.mean():.4f}  std={sq4.std():.4f}  "
        f"min={sq4.min():.4f}  max={sq4.max():.4f}")

    # ── summary ───────────────────────────────────────────────────
    log("")
    log("── Summary ──")
    log(f"  AUROC sq2→FGSM : {auc_sq2_fgsm:.4f}  |  sq4→FGSM : {auc_sq4_fgsm:.4f}")
    log(f"  AUROC sq2→PGD  : {auc_sq2_pgd:.4f}  |  sq4→PGD  : {auc_sq4_pgd:.4f}")
    log(f"  AUROC margin→FGSM : {auc_margin_fgsm:.4f}  |  margin→PGD : {auc_margin_pgd:.4f}")

    # verdict
    best_sq_fgsm = max(auc_sq2_fgsm, auc_sq4_fgsm)
    log("")
    if best_sq_fgsm > auc_margin_fgsm + 0.02:
        verdict = "SUPPORTED — squeezing score beats margin baseline for FGSM prediction."
    elif best_sq_fgsm > 0.6:
        verdict = "PARTIAL — squeezing score is above-chance but does not beat margin."
    else:
        verdict = "NOT SUPPORTED — squeezing score is near chance (AUROC ≤ 0.6)."
    log(f"Verdict: {verdict}")

    elapsed = time.time() - t0
    log(f"\nElapsed: {elapsed:.1f}s")

    # write output
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h194_feature_squeezing_detector_output.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    run()
