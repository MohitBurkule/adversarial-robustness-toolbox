"""
H239 - Dead Neuron Topology: ReLU dead neurons predict adversarial vulnerability.

For each test sample, count the number of ReLU neurons that output exactly 0
(dead for this sample) across all layers. Register forward hooks on all ReLU
modules. dead_count = sum of zero activations.
Measure: Spearman ρ(dead_count, margin), AUROC(dead_count → FGSM success),
compare fgsm_success vs pgd_success per dead_count quartile.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

try:
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

SEED = 0
EPS = 0.1
N_EVAL = 300

def count_dead_neurons(model, X, batch=64):
    """Count dead ReLU neurons per sample across all ReLU layers."""
    relu_modules = [m for m in model.modules() if isinstance(m, nn.ReLU)]
    if not relu_modules:
        # Fallback: try any activation-like modules
        relu_modules = [m for m in model.modules()
                        if hasattr(m, 'inplace') and isinstance(m, nn.ReLU)]

    activations_list = []
    hooks = []

    def make_hook(storage):
        def hook(module, inp, out):
            storage.append(out.detach().cpu())
        return hook

    dead_counts = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i+batch]
            batch_activations = []

            # register hooks
            for m in relu_modules:
                storage = []
                batch_activations.append(storage)
                h = m.register_forward_hook(make_hook(storage))
                hooks.append(h)

            _ = model(xb)

            # remove hooks
            for h in hooks:
                h.remove()
            hooks.clear()

            # count zeros per sample
            n_samples = xb.size(0)
            for s in range(n_samples):
                dead = 0
                for layer_acts in batch_activations:
                    if layer_acts:
                        act = layer_acts[0]  # shape: (batch, ...)
                        dead += (act[s] == 0).sum().item()
                dead_counts.append(dead)

    return np.array(dead_counts)

def main():
    print("=== H239: Dead Neuron Topology ===")
    t0 = time.time()

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]

    # [1] Train model
    print("\n[1] Training model...")
    model = C.build_model("cnn", meta, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=10)

    # [2] Count dead neurons per test sample
    print("[2] Counting dead ReLU neurons per test sample...")
    dead_counts = count_dead_neurons(model, Xte_e)
    print(f"    Dead neuron count: mean={dead_counts.mean():.1f}, "
          f"std={dead_counts.std():.1f}, "
          f"min={dead_counts.min()}, max={dead_counts.max()}")

    # [3] Compute margins and attack success
    print("[3] Computing margins...")
    margins = np.array(C.margin(model, Xte_e))

    print("[4] Running FGSM and PGD attacks...")
    Xfgsm = C.fgsm(model, Xte_e, Yte_e, eps=EPS)
    model.eval()
    with torch.no_grad():
        fgsm_succ = (model(Xfgsm).argmax(1).cpu() != Yte_e.cpu()).numpy().astype(int)

    Xpgd = C.pgd(model, Xte_e, Yte_e, eps=EPS, steps=10, alpha=0.01)
    with torch.no_grad():
        pgd_succ = (model(Xpgd).argmax(1).cpu() != Yte_e.cpu()).numpy().astype(int)

    print(f"    FGSM ASR: {fgsm_succ.mean():.3f}, PGD ASR: {pgd_succ.mean():.3f}")

    # [5] Metrics
    print("[5] Computing metrics...")
    rho_dead_margin = float('nan')
    auroc_dead_fgsm = float('nan')
    auroc_dead_pgd = float('nan')

    if HAS_SKLEARN:
        try:
            rho_dead_margin = spearmanr(dead_counts, margins).correlation
        except Exception:
            pass
        try:
            if len(np.unique(fgsm_succ)) == 2:
                auroc_dead_fgsm = roc_auc_score(fgsm_succ, dead_counts)
        except Exception:
            pass
        try:
            if len(np.unique(pgd_succ)) == 2:
                auroc_dead_pgd = roc_auc_score(pgd_succ, dead_counts)
        except Exception:
            pass

    # [6] Quartile comparison of FGSM vs PGD discrepancy
    print("[6] Dead count quartile analysis (FGSM vs PGD discrepancy)...")
    quartile_edges = np.percentile(dead_counts, [0, 25, 50, 75, 100])
    for q in range(4):
        lo, hi = quartile_edges[q], quartile_edges[q+1]
        mask = (dead_counts >= lo) & (dead_counts <= hi)
        if mask.sum() == 0:
            continue
        fa = fgsm_succ[mask].mean()
        pa = pgd_succ[mask].mean()
        disc = fa - pa
        print(f"    Q{q+1} (dead∈[{lo:.0f},{hi:.0f}]): n={mask.sum()}, "
              f"FGSM_ASR={fa:.3f}, PGD_ASR={pa:.3f}, "
              f"discrepancy(FGSM-PGD)={disc:+.3f}")

    print(f"\n--- Summary ---")
    print(f"Mean dead neurons: {dead_counts.mean():.1f}")
    print(f"Spearman ρ(dead_count, margin):     {rho_dead_margin:.3f}")
    print(f"AUROC(dead_count → FGSM success):   {auroc_dead_fgsm:.3f}")
    print(f"AUROC(dead_count → PGD success):    {auroc_dead_pgd:.3f}")
    print("Interpretation: high dead_count with FGSM≫PGD discrepancy in Q4 "
          "signals gradient masking (dead neurons zero out gradients).")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
