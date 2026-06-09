"""
H205 - Catastrophic robustness forgetting in sequential learning.

Hypothesis: After sequentially training on Task 2 classes, ASR on Task 1
classes increases by >=15pp more than clean accuracy drops on Task 1 --
robustness is forgotten faster than clean accuracy.

Grounded in: arXiv:2402.11196, arXiv:2510.09181.

Methodology:
  - Fashion-MNIST class splits:
      Task 1: classes 0-4 (T-shirt, Trouser, Pullover, Dress, Coat)
      Task 2: classes 5-9 (Sandal, Shirt, Sneaker, Bag, Boot)
  - Jointly trained baseline: train CNN on all 10 classes (6k samples, 15 epochs)
  - Sequential model:
      Phase 1: train CNN on Task 1 only (classes 0-4, 3k samples, 15 epochs)
      Phase 2: continue training on Task 2 only (classes 5-9, 3k samples, 10 epochs)
  - Evaluate on Task 1 test samples only:
      clean_forgetting = joint_clean_acc_t1 - sequential_clean_acc_t1
      robustness_forgetting = sequential_asr_t1 - joint_asr_t1
  - Test: robustness_forgetting_pgd >= clean_forgetting + 15pp

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign import common as C

SEED = 42
N_TRAIN = 6000
N_EVAL = 1000
EPOCHS_JOINT = 15
EPOCHS_SEQ_P1 = 15
EPOCHS_SEQ_P2 = 10
EPS = 0.1
TASK1_CLASSES = [0, 1, 2, 3, 4]
TASK2_CLASSES = [5, 6, 7, 8, 9]
CLASS_NAMES = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
               5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Boot"}


def filter_classes(X, Y, classes):
    """Return subset of X, Y where Y is in the given class list."""
    mask = torch.zeros(Y.size(0), dtype=torch.bool, device=Y.device)
    for c in classes:
        mask |= (Y == c)
    return X[mask], Y[mask]


def per_class_metrics(model, X, Y, classes, eps=EPS):
    """Per-class clean acc and FGSM/PGD ASR on given classes."""
    results = {}
    for c in classes:
        mask = Y == c
        if mask.sum() == 0:
            results[c] = {"clean_acc": float("nan"), "fgsm_asr": float("nan"), "pgd_asr": float("nan")}
            continue
        xc, yc = X[mask], Y[mask]
        _, acc = C.logits_and_acc(model, xc, yc)
        fgsm_res = C.attack_success(model, xc, yc, attack="fgsm", eps=eps)
        pgd_res = C.attack_success(model, xc, yc, attack="pgd", eps=eps, steps=10)
        results[c] = {"clean_acc": acc, "fgsm_asr": fgsm_res["asr"], "pgd_asr": pgd_res["asr"]}
    return results


def main():
    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # Test set: Task 1 only
    Xte_t1, Yte_t1 = filter_classes(Xte, Yte, TASK1_CLASSES)

    # Training subsets
    Xtr_t1, Ytr_t1 = filter_classes(Xtr, Ytr, TASK1_CLASSES)
    Xtr_t2, Ytr_t2 = filter_classes(Xtr, Ytr, TASK2_CLASSES)

    print("=" * 74)
    print("H205 - Catastrophic robustness forgetting in sequential learning")
    print("=" * 74)
    print(f"Device={C.DEVICE}  seed={SEED}  eps={EPS}")
    print(f"Task 1 classes: {TASK1_CLASSES}  Task 2 classes: {TASK2_CLASSES}")
    print(f"Train: joint={Xtr.size(0)}, T1={Xtr_t1.size(0)}, T2={Xtr_t2.size(0)}")
    print(f"Test (Task 1 only): {Xte_t1.size(0)} samples")

    # --- Joint baseline ---
    print("\n--- Joint baseline (all 10 classes, 15 epochs) ---")
    t0 = time.time()
    joint_model = C.build_model("cnn", meta)
    C.train_model(joint_model, Xtr, Ytr, epochs=EPOCHS_JOINT, verbose=True)
    print(f"  Training time: {time.time()-t0:.1f}s")

    _, joint_clean_t1 = C.logits_and_acc(joint_model, Xte_t1, Yte_t1)
    joint_fgsm = C.attack_success(joint_model, Xte_t1, Yte_t1, attack="fgsm", eps=EPS)
    joint_pgd = C.attack_success(joint_model, Xte_t1, Yte_t1, attack="pgd", eps=EPS, steps=10)
    print(f"  Joint Task1 clean_acc={joint_clean_t1:.4f}  fgsm_asr={joint_fgsm['asr']:.4f}  pgd_asr={joint_pgd['asr']:.4f}")

    # --- Sequential model ---
    print("\n--- Sequential model ---")
    print("  Phase 1: Task 1 only, 15 epochs")
    t0 = time.time()
    seq_model = C.build_model("cnn", meta)
    C.train_model(seq_model, Xtr_t1, Ytr_t1, epochs=EPOCHS_SEQ_P1, verbose=True)
    print(f"  Phase 1 time: {time.time()-t0:.1f}s")

    # Check Task 1 performance after Phase 1 (before forgetting)
    _, p1_clean_t1 = C.logits_and_acc(seq_model, Xte_t1, Yte_t1)
    print(f"  After Phase 1: Task1 clean_acc={p1_clean_t1:.4f}")

    print("  Phase 2: Task 2 only, 10 epochs (continue training, no reset)")
    t0 = time.time()
    C.train_model(seq_model, Xtr_t2, Ytr_t2, epochs=EPOCHS_SEQ_P2, verbose=True)
    print(f"  Phase 2 time: {time.time()-t0:.1f}s")

    _, seq_clean_t1 = C.logits_and_acc(seq_model, Xte_t1, Yte_t1)
    seq_fgsm = C.attack_success(seq_model, Xte_t1, Yte_t1, attack="fgsm", eps=EPS)
    seq_pgd = C.attack_success(seq_model, Xte_t1, Yte_t1, attack="pgd", eps=EPS, steps=10)
    print(f"  Sequential Task1 clean_acc={seq_clean_t1:.4f}  fgsm_asr={seq_fgsm['asr']:.4f}  pgd_asr={seq_pgd['asr']:.4f}")

    # --- Forgetting metrics ---
    clean_forgetting = joint_clean_t1 - seq_clean_t1
    rob_forgetting_fgsm = seq_fgsm["asr"] - joint_fgsm["asr"]
    rob_forgetting_pgd = seq_pgd["asr"] - joint_pgd["asr"]

    print("\n" + "=" * 74)
    print("FORGETTING METRICS (Task 1)")
    print("=" * 74)
    print(f"  clean_forgetting       = {clean_forgetting:+.4f} ({clean_forgetting*100:+.1f}pp)")
    print(f"  robustness_forgetting_fgsm = {rob_forgetting_fgsm:+.4f} ({rob_forgetting_fgsm*100:+.1f}pp)")
    print(f"  robustness_forgetting_pgd  = {rob_forgetting_pgd:+.4f} ({rob_forgetting_pgd*100:+.1f}pp)")
    print(f"  gap (pgd rob forgetting - clean forgetting) = {(rob_forgetting_pgd - clean_forgetting)*100:+.1f}pp")

    hypothesis_holds = (rob_forgetting_pgd - clean_forgetting) >= 0.15
    print(f"\n  H205 hypothesis (gap >= 15pp): {'SUPPORTED' if hypothesis_holds else 'NOT SUPPORTED'}")

    # --- Per-class breakdown ---
    print("\n" + "=" * 74)
    print("PER-CLASS BREAKDOWN (Task 1 classes)")
    print("=" * 74)
    joint_pc = per_class_metrics(joint_model, Xte_t1, Yte_t1, TASK1_CLASSES)
    seq_pc = per_class_metrics(seq_model, Xte_t1, Yte_t1, TASK1_CLASSES)

    print(f"{'Class':<12} {'Joint clean':>11} {'Seq clean':>11} {'Joint PGD':>11} {'Seq PGD':>11} {'Clean forg':>11} {'Rob forg':>11}")
    print("-" * 80)
    for c in TASK1_CLASSES:
        jc = joint_pc[c]["clean_acc"]
        sc = seq_pc[c]["clean_acc"]
        jp = joint_pc[c]["pgd_asr"]
        sp = seq_pc[c]["pgd_asr"]
        cf = jc - sc
        rf = sp - jp
        print(f"{CLASS_NAMES[c]:<12} {jc:>11.4f} {sc:>11.4f} {jp:>11.4f} {sp:>11.4f} {cf:>+11.4f} {rf:>+11.4f}")

    print("\n" + "=" * 74)
    print("CONCLUSION")
    print("=" * 74)
    print(f"If robustness forgetting >> clean forgetting, sequential learning")
    print(f"catastrophically erodes adversarial robustness even when clean accuracy")
    print(f"partially survives -- consistent with arXiv:2402.11196.")


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "h205_catastrophic_robustness_forgetting_output.txt")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    text = buf.getvalue()
    print(text)
    with open(outpath, "w") as f:
        f.write(text)
    print(f"\nSaved to {outpath}")
