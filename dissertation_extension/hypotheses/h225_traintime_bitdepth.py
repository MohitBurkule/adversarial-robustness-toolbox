"""
H225 - Training on quantised images: does reduced bit-depth confer adversarial robustness?

For bit_depth in [3, 4, 5, 6, 8]:
  Mode A - train on quantised, test on original 8-bit
  Mode B - train on original, test on quantised (baseline)
  Mode C - train and test on same quantised bit depth

Key question: does the model trained on 3-4 bit images learn representations
robust to adversarial perturbations in the quantisation noise floor?
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_EVAL = 300
SEED = 0
EPS = 0.1
BIT_DEPTHS = [3, 4, 5, 6, 8]
EPOCHS = 10

os.makedirs("results/fashion_mnist", exist_ok=True)


def quantise(x, bit_depth):
    """Quantise tensor x in [0,1] to bit_depth bits."""
    levels = 2 ** bit_depth - 1
    return torch.round(x * levels) / levels


def eval_model(model, Xte_clean, Xte_eval, Yte):
    """Evaluate model: clean acc on Xte_eval, FGSM/PGD ASR on Xte_clean."""
    model.eval()
    # Clean accuracy on eval set
    with torch.no_grad():
        preds = model(Xte_eval).argmax(1)
    clean_acc = float((preds == Yte).float().mean())

    # FGSM ASR (attack on clean test images)
    Xadv_fgsm = C.fgsm(model, Xte_clean, Yte, eps=EPS)
    with torch.no_grad():
        fgsm_preds = model(Xadv_fgsm).argmax(1)
    fgsm_asr = float((fgsm_preds != Yte).float().mean())

    # PGD ASR
    Xadv_pgd = C.pgd(model, Xte_clean, Yte, eps=EPS, steps=10, alpha=0.01)
    with torch.no_grad():
        pgd_preds = model(Xadv_pgd).argmax(1)
    pgd_asr = float((pgd_preds != Yte).float().mean())

    # Mean margin
    mean_margin = float(C.margin(model, Xte_clean, Yte).mean())

    return clean_acc, fgsm_asr, pgd_asr, mean_margin


def main():
    print("=" * 75)
    print("H225 - Training bit-depth quantisation and adversarial robustness")
    print("=" * 75)

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr_orig, Ytr, Xte_orig, Yte = C.load_dataset(DS, n_eval=N_EVAL, seed=SEED)
    print(f"Device={C.DEVICE}  N_eval={N_EVAL}  eps={EPS}  epochs={EPOCHS}")
    print(f"Training set size: {Xtr_orig.size(0)}")

    rows = []

    for bit in BIT_DEPTHS:
        print(f"\n--- bit_depth={bit} ---")
        Xtr_q = quantise(Xtr_orig, bit)
        Xte_q = quantise(Xte_orig, bit)

        # Mode A: train on quantised, test on ORIGINAL (no quant at test)
        t0 = time.time()
        C.set_seed(SEED)
        mA = C.build_model("cnn", meta, width=32, seed=SEED)
        C.train_model(mA, Xtr_q, Ytr, epochs=EPOCHS)
        ca, fa, pa, ma = eval_model(mA, Xte_orig, Xte_orig, Yte)
        rows.append({"bit": bit, "mode": "A_train_q_test_orig",
                     "clean_acc": ca, "fgsm_asr": fa, "pgd_asr": pa, "mean_margin": ma})
        print(f"  Mode A (train-q, test-orig): clean={ca:.3f}  fgsm_asr={fa:.3f}  pgd_asr={pa:.3f}  margin={ma:.3f}  ({time.time()-t0:.1f}s)")

        # Mode B: train on original, test on quantised (baseline: does quant at test help?)
        t0 = time.time()
        C.set_seed(SEED)
        mB = C.build_model("cnn", meta, width=32, seed=SEED)
        C.train_model(mB, Xtr_orig, Ytr, epochs=EPOCHS)
        # Test: evaluate clean on quantised, adversarial on original
        cb, fb, pb, mb = eval_model(mB, Xte_orig, Xte_q, Yte)
        rows.append({"bit": bit, "mode": "B_train_orig_test_q",
                     "clean_acc": cb, "fgsm_asr": fb, "pgd_asr": pb, "mean_margin": mb})
        print(f"  Mode B (train-orig, test-q):  clean={cb:.3f}  fgsm_asr={fb:.3f}  pgd_asr={pb:.3f}  margin={mb:.3f}  ({time.time()-t0:.1f}s)")

        # Mode C: train and test on same quantised bit depth
        t0 = time.time()
        C.set_seed(SEED)
        mC = C.build_model("cnn", meta, width=32, seed=SEED)
        C.train_model(mC, Xtr_q, Ytr, epochs=EPOCHS)
        cc, fc, pc, mc = eval_model(mC, Xte_q, Xte_q, Yte)
        rows.append({"bit": bit, "mode": "C_train_q_test_q",
                     "clean_acc": cc, "fgsm_asr": fc, "pgd_asr": pc, "mean_margin": mc})
        print(f"  Mode C (train-q, test-q):     clean={cc:.3f}  fgsm_asr={fc:.3f}  pgd_asr={pc:.3f}  margin={mc:.3f}  ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 75)
    print("FULL TABLE")
    print(f"{'bit':>4}  {'mode':<25}  {'clean_acc':>10}  {'fgsm_asr':>9}  {'pgd_asr':>8}  {'mean_margin':>12}")
    print("-" * 75)
    for r in rows:
        print(f"{r['bit']:>4}  {r['mode']:<25}  {r['clean_acc']:>10.4f}  {r['fgsm_asr']:>9.4f}  {r['pgd_asr']:>8.4f}  {r['mean_margin']:>12.4f}")

    print("\n" + "=" * 75)
    print("ANALYSIS: Mode A (train-quantised, test-original) robustness vs bit depth:")
    mode_a = [r for r in rows if r["mode"] == "A_train_q_test_orig"]
    for r in mode_a:
        print(f"  bit={r['bit']}  pgd_asr={r['pgd_asr']:.4f}  clean_acc={r['clean_acc']:.4f}")

    # Is 3 or 4 bit training more robust than 8-bit baseline?
    asr_3bit = next((r["pgd_asr"] for r in mode_a if r["bit"] == 3), float("nan"))
    asr_8bit = next((r["pgd_asr"] for r in mode_a if r["bit"] == 8), float("nan"))
    print(f"\n  PGD ASR: 3-bit training={asr_3bit:.4f}  vs  8-bit training={asr_8bit:.4f}")
    if asr_3bit < asr_8bit - 0.02:
        print("  HYPOTHESIS SUPPORTED: Low-bit training reduces adversarial success rate.")
        print("  Model trained on 3-4 bit images learns to ignore sub-4-bit variation.")
    else:
        print("  HYPOTHESIS NOT SUPPORTED: Bit-depth reduction alone does not confer robustness.")
    print("=" * 75)


if __name__ == "__main__":
    main()
