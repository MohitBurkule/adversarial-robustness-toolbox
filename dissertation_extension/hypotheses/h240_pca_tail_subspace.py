"""
H240 - PCA Tail Subspace: adversarial perturbations live in low-variance directions.

Fit PCA on flattened training images (top K=50,100,200,400 components).
For each FGSM perturbation δ, compute fraction of energy in PCA tail subspace:
  tail_energy = ||δ - PCA_projection(δ)||² / ||δ||²
Does tail_energy predict attack success? Hypothesis: adversarial perturbations
live in low-variance directions → tail_energy high for successful attacks.
Measure: AUROC(tail_energy → FGSM success) at each K.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

SEED = 0
EPS = 0.1
N_EVAL = 300
K_VALUES = [50, 100, 200, 400]

def main():
    print("=== H240: PCA Tail Subspace ===")
    t0 = time.time()

    if not HAS_SKLEARN:
        print("sklearn not available — cannot run PCA analysis.")
        return

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]

    # [1] Train model
    print("\n[1] Training model...")
    model = C.build_model("cnn", meta, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=10)

    # [2] Fit PCA on training images
    print("[2] Fitting PCA on training images...")
    Xtr_np = Xtr.cpu().numpy().reshape(len(Xtr), -1)  # (N, 784)
    Xte_np = Xte_e.cpu().numpy().reshape(N_EVAL, -1)

    max_k = max(K_VALUES)
    pca = PCA(n_components=max_k, random_state=SEED)
    pca.fit(Xtr_np)
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    for k in K_VALUES:
        print(f"    PCA K={k:4d}: explained variance = {explained_var[k-1]:.4f}")

    # PCA components: (max_k, 784)
    components = pca.components_  # (max_k, 784)

    # [3] FGSM attack and perturbation
    print("[3] Running FGSM attack...")
    Xfgsm = C.fgsm(model, Xte_e, Yte_e, eps=EPS)
    model.eval()
    with torch.no_grad():
        fgsm_succ = (model(Xfgsm).argmax(1).cpu() != Yte_e.cpu()).numpy().astype(int)
    print(f"    FGSM ASR: {fgsm_succ.mean():.3f}")

    # Perturbation delta
    delta_np = (Xfgsm - Xte_e).cpu().numpy().reshape(N_EVAL, -1)  # (N, 784)

    # [4] Compute tail energy at each K
    print("[4] Computing tail energy at each K...")
    auroc_results = {}

    for k in K_VALUES:
        comps_k = components[:k]  # (k, 784)
        # Project delta onto PCA subspace
        coords = delta_np @ comps_k.T        # (N, k)
        proj = coords @ comps_k              # (N, 784) — projection in image space

        # Tail energy
        delta_norm_sq = (delta_np ** 2).sum(axis=1)  # (N,)
        tail_sq = ((delta_np - proj) ** 2).sum(axis=1)  # (N,)

        # Avoid division by zero
        safe_norm = np.where(delta_norm_sq > 1e-12, delta_norm_sq, 1.0)
        tail_energy = tail_sq / safe_norm  # fraction in tail
        tail_energy[delta_norm_sq < 1e-12] = 0.0

        try:
            if len(np.unique(fgsm_succ)) == 2:
                auroc = roc_auc_score(fgsm_succ, tail_energy)
            else:
                auroc = float('nan')
        except Exception:
            auroc = float('nan')

        try:
            rho = spearmanr(tail_energy, fgsm_succ).correlation
        except Exception:
            rho = float('nan')

        auroc_results[k] = (auroc, rho, tail_energy.mean(),
                            tail_energy[fgsm_succ == 1].mean() if fgsm_succ.sum() > 0 else float('nan'),
                            tail_energy[fgsm_succ == 0].mean() if (fgsm_succ == 0).sum() > 0 else float('nan'))
        print(f"    K={k:4d}: AUROC={auroc:.3f}, ρ={rho:.3f}, "
              f"mean_tail={tail_energy.mean():.4f}, "
              f"tail_succ={auroc_results[k][3]:.4f}, "
              f"tail_fail={auroc_results[k][4]:.4f}")

    # Also use margin as baseline
    margins = np.array(C.margin(model, Xte_e))
    try:
        auroc_margin = roc_auc_score(fgsm_succ, -margins)
    except Exception:
        auroc_margin = float('nan')
    print(f"    Margin baseline AUROC: {auroc_margin:.3f}")

    print(f"\n--- Summary ---")
    print(f"FGSM ASR: {fgsm_succ.mean():.3f}")
    for k in K_VALUES:
        au, rh, mt, ts, tf = auroc_results[k]
        print(f"  K={k:4d}: AUROC={au:.3f}, Spearman_ρ={rh:.3f}, "
              f"tail_energy(success)={ts:.4f} vs (fail)={tf:.4f}")
    print(f"  Margin baseline AUROC: {auroc_margin:.3f}")
    print("Interpretation: AUROC>0.5 for tail_energy => adversarial perturbations "
          "prefer low-variance PCA directions.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
