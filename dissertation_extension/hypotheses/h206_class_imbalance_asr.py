"""
H206 - Minority-class adversarial vulnerability scales with imbalance ratio.

Hypothesis: minority-class FGSM ASR increases log-linearly with imbalance
ratio, and at ratio 50:1 is >=20pp higher than majority-class ASR -- even
though clean accuracy gap is <10pp.

Grounded in: arXiv:2503.06461 (ICLR 2025), arXiv:2503.01924 (CVPR 2025).

Methodology:
  - Majority classes: 1 (Trouser), 7 (Sneaker), 8 (Bag) -- visually distinct
  - Minority classes: 0 (T-shirt), 2 (Pullover), 6 (Shirt) -- visually similar
  - For imbalance ratio r in [1, 5, 10, 20, 50]:
      majority: 600 samples/class; minority: 600/r samples/class
  - Train CNN, evaluate on balanced test set (100/class)
  - Measure per-group clean acc and FGSM ASR

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign import common as C

SEED = 42
EPS = 0.1
EPOCHS = 15
MAJORITY_CLASSES = [1, 7, 8]
MINORITY_CLASSES = [0, 2, 6]
ALL_CLASSES = sorted(MAJORITY_CLASSES + MINORITY_CLASSES)
SAMPLES_PER_MAJ = 600
TEST_PER_CLASS = 100
RATIOS = [1, 5, 10, 20, 50]

CLASS_NAMES = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
               5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Boot"}


def build_imbalanced_set(Xtr, Ytr, ratio, seed=42):
    """Subsample: majority classes get SAMPLES_PER_MAJ, minority get SAMPLES_PER_MAJ/ratio."""
    rng = np.random.RandomState(seed)
    indices = []
    for c in ALL_CLASSES:
        cls_idx = (Ytr.cpu() == c).nonzero(as_tuple=True)[0].numpy()
        if c in MAJORITY_CLASSES:
            n = min(SAMPLES_PER_MAJ, len(cls_idx))
        else:
            n = min(max(SAMPLES_PER_MAJ // ratio, 1), len(cls_idx))
        chosen = rng.choice(cls_idx, size=n, replace=False)
        indices.extend(chosen.tolist())
    indices = torch.tensor(indices, device=Xtr.device)
    return Xtr[indices], Ytr[indices]


def build_balanced_test(Xte, Yte, per_class=TEST_PER_CLASS, seed=42):
    """Balanced test set with per_class samples per class."""
    rng = np.random.RandomState(seed)
    indices = []
    for c in ALL_CLASSES:
        cls_idx = (Yte.cpu() == c).nonzero(as_tuple=True)[0].numpy()
        n = min(per_class, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n, replace=False)
        indices.extend(chosen.tolist())
    indices = torch.tensor(indices, device=Xte.device)
    return Xte[indices], Yte[indices]


def group_metrics(model, X, Y, classes, eps=EPS):
    """Average clean acc and FGSM ASR across a group of classes."""
    accs, asrs = [], []
    for c in classes:
        mask = Y == c
        if mask.sum() == 0:
            continue
        xc, yc = X[mask], Y[mask]
        _, acc = C.logits_and_acc(model, xc, yc)
        res = C.attack_success(model, xc, yc, attack="fgsm", eps=eps)
        accs.append(acc)
        asrs.append(res["asr"])
    return np.mean(accs), np.mean(asrs)


def main():
    C.set_seed(SEED)
    # Load full dataset (we need enough samples for high ratios)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=None, n_eval=None, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")

    # Build balanced test set
    Xte_bal, Yte_bal = build_balanced_test(Xte, Yte)

    print("=" * 74)
    print("H206 - Minority-class adversarial vulnerability vs imbalance ratio")
    print("=" * 74)
    print(f"Device={C.DEVICE}  seed={SEED}  eps={EPS}")
    print(f"Majority classes: {[CLASS_NAMES[c] for c in MAJORITY_CLASSES]}")
    print(f"Minority classes: {[CLASS_NAMES[c] for c in MINORITY_CLASSES]}")
    print(f"Test set: {Xte_bal.size(0)} samples ({TEST_PER_CLASS}/class)")

    results = []
    for r in RATIOS:
        print(f"\n--- Ratio {r}:1 ---")
        Xtr_imb, Ytr_imb = build_imbalanced_set(Xtr, Ytr, r)
        # Count per class
        for c in ALL_CLASSES:
            n = (Ytr_imb == c).sum().item()
            print(f"  class {c} ({CLASS_NAMES[c]}): {n} train samples")

        t0 = time.time()
        model = C.build_model("cnn", meta)
        C.train_model(model, Xtr_imb, Ytr_imb, epochs=EPOCHS)
        elapsed = time.time() - t0

        maj_acc, maj_asr = group_metrics(model, Xte_bal, Yte_bal, MAJORITY_CLASSES)
        min_acc, min_asr = group_metrics(model, Xte_bal, Yte_bal, MINORITY_CLASSES)

        row = {
            "ratio": r,
            "maj_clean": maj_acc,
            "min_clean": min_acc,
            "maj_asr": maj_asr,
            "min_asr": min_asr,
            "clean_gap": maj_acc - min_acc,
            "asr_gap": min_asr - maj_asr,
        }
        results.append(row)
        print(f"  maj_clean={maj_acc:.4f}  min_clean={min_acc:.4f}  "
              f"maj_asr={maj_asr:.4f}  min_asr={min_asr:.4f}  "
              f"asr_gap={row['asr_gap']*100:+.1f}pp  ({elapsed:.1f}s)")

    # --- Summary table ---
    print("\n" + "=" * 74)
    print("SUMMARY TABLE")
    print("=" * 74)
    print(f"{'Ratio':>6} {'Maj clean':>10} {'Min clean':>10} {'Clean gap':>10} "
          f"{'Maj ASR':>10} {'Min ASR':>10} {'ASR gap':>10}")
    print("-" * 68)
    for row in results:
        print(f"{row['ratio']:>6} {row['maj_clean']:>10.4f} {row['min_clean']:>10.4f} "
              f"{row['clean_gap']*100:>+10.1f}pp {row['maj_asr']:>10.4f} "
              f"{row['min_asr']:>10.4f} {row['asr_gap']*100:>+10.1f}pp")

    # --- Log-linear fit ---
    print("\n--- Log-linear fit: asr_gap ~ a * log(ratio) + b ---")
    log_r = np.log(np.array([row["ratio"] for row in results]))
    gaps = np.array([row["asr_gap"] for row in results])
    if len(log_r) >= 2:
        coeffs = np.polyfit(log_r, gaps, 1)
        print(f"  slope={coeffs[0]:.4f}  intercept={coeffs[1]:.4f}")
        residuals = gaps - np.polyval(coeffs, log_r)
        ss_res = (residuals ** 2).sum()
        ss_tot = ((gaps - gaps.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  R^2 = {r2:.4f}")

    # --- Hypothesis test ---
    print("\n" + "=" * 74)
    print("HYPOTHESIS TEST")
    print("=" * 74)
    r50_row = [row for row in results if row["ratio"] == 50][0]
    asr_gap_50 = r50_row["asr_gap"]
    clean_gap_50 = r50_row["clean_gap"]
    print(f"  At ratio 50:1:")
    print(f"    ASR gap (min - maj) = {asr_gap_50*100:.1f}pp")
    print(f"    Clean gap (maj - min) = {clean_gap_50*100:.1f}pp")
    h1 = asr_gap_50 >= 0.20
    h2 = clean_gap_50 < 0.10
    print(f"  ASR gap >= 20pp: {'SUPPORTED' if h1 else 'NOT SUPPORTED'} ({asr_gap_50*100:.1f}pp)")
    print(f"  Clean gap < 10pp: {'SUPPORTED' if h2 else 'NOT SUPPORTED'} ({clean_gap_50*100:.1f}pp)")
    print(f"  H206 overall: {'SUPPORTED' if (h1 and h2) else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "h206_class_imbalance_asr_output.txt")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    text = buf.getvalue()
    print(text)
    with open(outpath, "w") as f:
        f.write(text)
    print(f"\nSaved to {outpath}")
