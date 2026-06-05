"""
H237 - Broken Symmetry: symmetry gap predicts adversarial vulnerability.

Compute symmetry_gap = |margin(x) - margin(hflip(x))| for each test sample.
Also try: |margin(x) - margin(x_brightness_shifted)| and
          |margin(x) - margin(x_contrast_shifted)|.
Measure AUROC of symmetry_gap → FGSM/PGD success.
"""
import os, sys, time
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

SEED = 0
EPS = 0.1
N_EVAL = 300

def compute_auroc(scores, labels):
    if not HAS_SKLEARN:
        return float('nan')
    try:
        if len(np.unique(labels)) < 2:
            return float('nan')
        return roc_auc_score(labels, scores)
    except Exception:
        return float('nan')

def pgd_success(model, X, Y):
    Xadv = C.pgd(model, X, Y, eps=EPS, steps=10, alpha=0.01)
    model.eval()
    with torch.no_grad():
        preds = model(Xadv).argmax(1).cpu()
    return (preds != Y.cpu()).numpy().astype(int)

def fgsm_success(model, X, Y):
    Xadv = C.fgsm(model, X, Y, eps=EPS)
    model.eval()
    with torch.no_grad():
        preds = model(Xadv).argmax(1).cpu()
    return (preds != Y.cpu()).numpy().astype(int)

def main():
    print("=== H237: Broken Symmetry ===")
    t0 = time.time()

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]

    # [1] Train model
    print("\n[1] Training model...")
    model = C.build_model("cnn", meta, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=10)

    # [2] Compute margins on original and transformed inputs
    print("[2] Computing margins...")
    margins_orig = np.array(C.margin(model, Xte_e))

    # Horizontal flip
    Xte_hflip = torch.stack([TF.hflip(x) for x in Xte_e])
    margins_hflip = np.array(C.margin(model, Xte_hflip))

    # Brightness shift (+0.2 clipped to [0,1])
    Xte_bright = (Xte_e + 0.2).clamp(0, 1)
    margins_bright = np.array(C.margin(model, Xte_bright))

    # Contrast shift: multiply by 1.5, re-center, clip
    mean_val = Xte_e.mean(dim=(2, 3), keepdim=True)
    Xte_contrast = ((Xte_e - mean_val) * 1.5 + mean_val).clamp(0, 1)
    margins_contrast = np.array(C.margin(model, Xte_contrast))

    # [3] Symmetry gaps
    gap_hflip = np.abs(margins_orig - margins_hflip)
    gap_bright = np.abs(margins_orig - margins_bright)
    gap_contrast = np.abs(margins_orig - margins_contrast)

    print(f"    Symmetry gap (hflip):    mean={gap_hflip.mean():.4f}")
    print(f"    Symmetry gap (bright):   mean={gap_bright.mean():.4f}")
    print(f"    Symmetry gap (contrast): mean={gap_contrast.mean():.4f}")

    # [4] FGSM and PGD success
    print("[3] Running FGSM and PGD attacks...")
    fgsm_succ = fgsm_success(model, Xte_e, Yte_e)
    pgd_succ = pgd_success(model, Xte_e, Yte_e)
    print(f"    FGSM ASR: {fgsm_succ.mean():.3f}, PGD ASR: {pgd_succ.mean():.3f}")

    # [4] AUROC: symmetry_gap → attack success
    print("[4] Computing AUROCs...")
    results = {}
    for gap_name, gap in [("hflip", gap_hflip),
                           ("brightness", gap_bright),
                           ("contrast", gap_contrast)]:
        auc_fgsm = compute_auroc(gap, fgsm_succ)
        auc_pgd = compute_auroc(gap, pgd_succ)
        results[gap_name] = (auc_fgsm, auc_pgd)
        print(f"    {gap_name:12s}: AUROC(→FGSM)={auc_fgsm:.3f}, "
              f"AUROC(→PGD)={auc_pgd:.3f}")

    # Also AUROC of margin → success (baseline predictor)
    auc_margin_fgsm = compute_auroc(-margins_orig, fgsm_succ)
    auc_margin_pgd = compute_auroc(-margins_orig, pgd_succ)
    print(f"    {'margin':12s}: AUROC(→FGSM)={auc_margin_fgsm:.3f}, "
          f"AUROC(→PGD)={auc_margin_pgd:.3f}  [baseline]")

    print(f"\n--- Summary ---")
    print(f"FGSM ASR={fgsm_succ.mean():.3f}, PGD ASR={pgd_succ.mean():.3f}")
    for gap_name, (af, ap) in results.items():
        print(f"  {gap_name:12s} gap: AUROC→FGSM={af:.3f}, AUROC→PGD={ap:.3f}")
    print(f"  Margin (baseline):  AUROC→FGSM={auc_margin_fgsm:.3f}, "
          f"AUROC→PGD={auc_margin_pgd:.3f}")
    print("Interpretation: AUROC > 0.6 for symmetry gap suggests asymmetric "
          "sensitivity predicts vulnerability.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
