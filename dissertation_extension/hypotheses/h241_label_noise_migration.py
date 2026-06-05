"""
H241 - Label Noise Migration: vulnerability shifts across noise levels.

Train 5 models with increasing label noise: 0%, 5%, 10%, 20%, 50%.
For each model, identify the set of vulnerable test samples (PGD success).
Measure Jaccard overlap between vulnerable sets across noise levels.
Hypothesis: vulnerability "migrates" to different samples under noise rather
than disappearing. Also measure: does margin AUROC degrade with noise?
"""
import os, sys, time
import numpy as np
import torch

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
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.50]

def add_label_noise(Y, noise_frac, n_classes, rng):
    """Randomly flip a fraction of labels to a different class."""
    Y_noisy = Y.clone()
    n_flip = int(len(Y) * noise_frac)
    flip_idx = rng.choice(len(Y), n_flip, replace=False)
    for i in flip_idx:
        orig = Y_noisy[i].item()
        new_label = rng.integers(0, n_classes)
        while new_label == orig:
            new_label = rng.integers(0, n_classes)
        Y_noisy[i] = new_label
    return Y_noisy

def pgd_success(model, X, Y):
    Xadv = C.pgd(model, X, Y, eps=EPS, steps=10, alpha=0.01)
    model.eval()
    with torch.no_grad():
        preds = model(Xadv).argmax(1).cpu()
    return (preds != Y.cpu()).numpy().astype(int)

def jaccard(a, b):
    a_set = set(np.where(a)[0])
    b_set = set(np.where(b)[0])
    if len(a_set | b_set) == 0:
        return float('nan')
    return len(a_set & b_set) / len(a_set | b_set)

def main():
    print("=== H241: Label Noise Migration ===")
    t0 = time.time()

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]
    n_classes = meta['n_classes']
    rng = np.random.default_rng(SEED)

    vuln_sets = []
    clean_accs = []
    margins_list = []
    aurocs = []

    for noise in NOISE_LEVELS:
        print(f"\n[noise={noise:.0%}] Adding label noise and training...")

        if noise > 0:
            Ytr_noisy = add_label_noise(Ytr, noise, n_classes, rng)
        else:
            Ytr_noisy = Ytr.clone()

        actual_noise = (Ytr_noisy != Ytr).float().mean().item()
        print(f"  Actual noise: {actual_noise:.3f}")

        C.set_seed(SEED)
        model = C.build_model("cnn", meta, seed=SEED)
        C.train_model(model, Xtr, Ytr_noisy, epochs=10)

        model.eval()
        with torch.no_grad():
            preds_clean = model(Xte_e).argmax(1).cpu()
        clean_acc = (preds_clean == Yte_e.cpu()).float().mean().item()
        clean_accs.append(clean_acc)

        margins = np.array(C.margin(model, Xte_e))
        margins_list.append(margins)

        succ = pgd_success(model, Xte_e, Yte_e)
        vuln_sets.append(succ)

        auroc = float('nan')
        if HAS_SKLEARN:
            try:
                if len(np.unique(succ)) == 2:
                    auroc = roc_auc_score(succ, -margins)
            except Exception:
                pass
        aurocs.append(auroc)

        print(f"  Clean acc: {clean_acc:.3f}, PGD ASR: {succ.mean():.3f}, "
              f"Margin AUROC: {auroc:.3f}")

    # [Jaccard matrix]
    print("\n[Jaccard overlap matrix between vulnerable sets]")
    n = len(NOISE_LEVELS)
    jaccard_matrix = np.zeros((n, n))
    header = "       " + "".join(f"{nl:.0%}  " for nl in NOISE_LEVELS)
    print(header)
    for i in range(n):
        row = f"  {NOISE_LEVELS[i]:.0%}:  "
        for j in range(n):
            j_val = jaccard(vuln_sets[i], vuln_sets[j])
            jaccard_matrix[i, j] = j_val
            row += f"{j_val:.3f}  "
        print(row)

    print(f"\n--- Summary ---")
    print(f"{'Noise':>8} | {'Clean Acc':>10} | {'PGD ASR':>8} | {'Margin AUROC':>12}")
    print("-" * 50)
    for i, noise in enumerate(NOISE_LEVELS):
        print(f"  {noise:.0%}     | {clean_accs[i]:>10.3f} | "
              f"{vuln_sets[i].mean():>8.3f} | {aurocs[i]:>12.3f}")
    print("\nJaccard(0% vs 50%):", f"{jaccard_matrix[0, -1]:.3f}")
    print("Interpretation: low Jaccard(0%,50%) + similar ASR => "
          "vulnerability migrates to different samples under noise.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
