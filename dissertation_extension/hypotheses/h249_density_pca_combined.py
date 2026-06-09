"""
H249 - Density + PCA Combined Predictor.

Combine neighbourhood density (H206 proxy) with PCA tail energy (H240).
Multi-predictor: does density + tail_energy + margin together improve AUROC
over margin alone?
Use logistic regression meta-predictor on [density_r10, tail_energy_k100, margin]
→ pgd_success.
Measure: individual AUROCs and combined AUROC, feature importances.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from scipy.spatial import cKDTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

SEED = 0
EPS = 0.1
N_EVAL = 300
EPOCHS = 10
K_PCA = 100          # PCA components for tail energy
DENSITY_K = 10       # k-NN for density

def compute_density(X_query, X_ref, k=10):
    """
    Neighbourhood density: inverse of mean distance to k nearest neighbours
    in flattened pixel space.
    Returns density array (N_query,).
    """
    X_q = X_query.cpu().numpy().reshape(len(X_query), -1)
    X_r = X_ref.cpu().numpy().reshape(len(X_ref), -1)
    tree = cKDTree(X_r)
    dists, _ = tree.query(X_q, k=k+1)  # +1 because query includes itself if present
    dists = dists[:, 1:]  # exclude self if it appears
    mean_dists = dists.mean(axis=1)
    density = 1.0 / (mean_dists + 1e-8)
    return density

def compute_tail_energy(delta_np, components):
    """Fraction of perturbation energy in PCA tail subspace."""
    coords = delta_np @ components.T
    proj = coords @ components
    delta_norm_sq = (delta_np ** 2).sum(axis=1)
    tail_sq = ((delta_np - proj) ** 2).sum(axis=1)
    safe_norm = np.where(delta_norm_sq > 1e-12, delta_norm_sq, 1.0)
    tail_energy = tail_sq / safe_norm
    tail_energy[delta_norm_sq < 1e-12] = 0.0
    return tail_energy

def main():
    print("=== H249: Density + PCA Combined Predictor ===")
    t0 = time.time()

    if not HAS_SKLEARN:
        print("sklearn not available — cannot run.")
        return

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]

    # [1] Train model
    print("\n[1] Training model...")
    model = C.build_model("cnn", meta, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=EPOCHS)

    # [2] PGD attack to get labels
    print("[2] Running PGD attack...")
    Xadv = C.pgd(model, Xte_e, Yte_e, eps=EPS, steps=10, alpha=0.01)
    model.eval()
    with torch.no_grad():
        pgd_succ = (model(Xadv).argmax(1).cpu() != Yte_e.cpu()).numpy().astype(int)
    print(f"    PGD ASR: {pgd_succ.mean():.3f}")

    # [3] Compute margin
    print("[3] Computing margins...")
    margins = np.array(C.margin(model, Xte_e))

    # [4] Compute neighbourhood density
    print(f"[4] Computing neighbourhood density (k={DENSITY_K})...")
    density = compute_density(Xte_e, Xtr[:5000], k=DENSITY_K)
    print(f"    Density: mean={density.mean():.4f}, std={density.std():.4f}")

    # [5] Compute PCA tail energy
    print(f"[5] Computing PCA tail energy (K={K_PCA})...")
    Xtr_np = Xtr.cpu().numpy().reshape(len(Xtr), -1)
    pca = PCA(n_components=K_PCA, random_state=SEED)
    pca.fit(Xtr_np)
    components = pca.components_  # (K_PCA, 784)

    # FGSM perturbation
    Xfgsm = C.fgsm(model, Xte_e, Yte_e, eps=EPS)
    delta_np = (Xfgsm - Xte_e).cpu().numpy().reshape(N_EVAL, -1)
    tail_energy = compute_tail_energy(delta_np, components)
    print(f"    Tail energy: mean={tail_energy.mean():.4f}, "
          f"std={tail_energy.std():.4f}")

    # [6] Individual AUROCs
    print("[6] Individual AUROCs...")
    try:
        auroc_margin = roc_auc_score(pgd_succ, -margins) if len(np.unique(pgd_succ)) == 2 else float('nan')
    except Exception:
        auroc_margin = float('nan')
    try:
        auroc_density = roc_auc_score(pgd_succ, -density) if len(np.unique(pgd_succ)) == 2 else float('nan')
    except Exception:
        auroc_density = float('nan')
    try:
        auroc_tail = roc_auc_score(pgd_succ, tail_energy) if len(np.unique(pgd_succ)) == 2 else float('nan')
    except Exception:
        auroc_tail = float('nan')

    print(f"    AUROC(margin):      {auroc_margin:.3f}")
    print(f"    AUROC(density):     {auroc_density:.3f}")
    print(f"    AUROC(tail_energy): {auroc_tail:.3f}")

    # [7] Combined logistic regression meta-predictor
    print("[7] Logistic regression meta-predictor...")
    features = np.column_stack([-margins, -density, tail_energy])
    feature_names = ['neg_margin', 'neg_density', 'tail_energy']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    try:
        lr_model = LogisticRegression(random_state=SEED, max_iter=1000)

        # 5-fold cross-validated AUROC
        cv_scores = cross_val_score(lr_model, features_scaled, pgd_succ,
                                     cv=5, scoring='roc_auc')
        combined_auroc = cv_scores.mean()
        combined_std = cv_scores.std()

        # Fit on all data for coefficients
        lr_model.fit(features_scaled, pgd_succ)
        coefs = lr_model.coef_[0]
        print(f"    Combined AUROC (5-fold CV): {combined_auroc:.3f} ± {combined_std:.3f}")
        print(f"    Feature importances (|coef|):")
        for name, coef in zip(feature_names, coefs):
            print(f"      {name:>15}: {coef:+.4f}")
    except Exception as e:
        combined_auroc = float('nan')
        print(f"    LogisticRegression failed: {e}")

    # Pairwise combinations
    print("[8] Pairwise combination AUROCs...")
    pairs = [
        ('margin+density', np.column_stack([-margins, -density])),
        ('margin+tail',    np.column_stack([-margins, tail_energy])),
        ('density+tail',   np.column_stack([-density, tail_energy])),
    ]
    for pair_name, feats in pairs:
        try:
            feats_s = StandardScaler().fit_transform(feats)
            lr = LogisticRegression(random_state=SEED, max_iter=1000)
            cv = cross_val_score(lr, feats_s, pgd_succ, cv=5, scoring='roc_auc')
            print(f"    {pair_name:>20}: {cv.mean():.3f} ± {cv.std():.3f}")
        except Exception as e:
            print(f"    {pair_name:>20}: failed ({e})")

    print(f"\n--- Summary ---")
    print(f"PGD ASR: {pgd_succ.mean():.3f}")
    print(f"Individual AUROCs: margin={auroc_margin:.3f}, "
          f"density={auroc_density:.3f}, tail_energy={auroc_tail:.3f}")
    print(f"Combined (all 3) AUROC: {combined_auroc:.3f}")
    print("Interpretation: combined AUROC > best individual => "
          "density/tail carry complementary information beyond margin alone.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
