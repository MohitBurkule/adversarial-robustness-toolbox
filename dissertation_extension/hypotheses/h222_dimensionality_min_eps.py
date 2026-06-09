"""
H222 - Input dimensionality vs minimum epsilon to attack — Simon-Gabriel √n scaling.

Use PCA to project images to n_components ∈ [10, 25, 50, 100, 200, 400, 784].
For each n: fit PCA on Xtr, project Xte[:300] to n dims, reconstruct.
Train a fresh CNN on PCA-reconstructed training images.
For each model: sweep eps ∈ [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3] and compute FGSM ASR.
Find eps_50 = smallest eps where FGSM ASR ≥ 0.5.
Fit log(eps_50) = a + b*log(n) — check if b ≈ -0.5 (Simon-Gabriel prediction).
Also measure: mean L1 norm of input gradient vs n_components.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

os.makedirs("results/fashion_mnist", exist_ok=True)

try:
    from sklearn.decomposition import PCA
except ImportError:
    raise ImportError("scikit-learn required: pip install scikit-learn")

DS = "fashion_mnist"
N_EVAL = 300
SEED = 0
N_COMPONENTS_LIST = [10, 25, 50, 100, 200, 400, 784]
EPS_SWEEP = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
EPOCHS = 7
BATCH = 128


def pca_reconstruct(pca, X_flat_np):
    """Project to PCA space and reconstruct. Returns numpy float32."""
    projected = pca.transform(X_flat_np)
    reconstructed = pca.inverse_transform(projected)
    return reconstructed.astype(np.float32)


def np_to_tensor(X_np, shape, device):
    """Reshape flat numpy array to (N, C, H, W) tensor on device."""
    return torch.from_numpy(X_np).view(-1, *shape).clamp(0, 1).to(device)


def compute_grad_l1(model, Xte, Yte, n_samples=N_EVAL):
    """Mean L1 norm of input gradient (per image)."""
    model.eval()
    X = Xte[:n_samples]
    Y = Yte[:n_samples]
    X.requires_grad_(False)
    Xc = X.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(Xc), Y)
    loss.backward()
    grad = Xc.grad.detach()
    # L1 norm per image: sum of abs values
    l1 = grad.view(X.size(0), -1).abs().sum(dim=1)
    return float(l1.mean().cpu())


def main():
    print("=" * 74)
    print("H222 - Input dimensionality vs minimum epsilon (Simon-Gabriel √n test)")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  N_EVAL={N_EVAL}")
    print(f"n_components: {N_COMPONENTS_LIST}")
    print(f"eps_sweep: {EPS_SWEEP}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    im_shape = (meta["channels"], meta["size"], meta["size"])
    flat_dim = meta["channels"] * meta["size"] * meta["size"]

    # Load full training set and eval set
    Xtr_full, Ytr_full, Xte_full, Yte_full = C.load_dataset(DS, n_eval=N_EVAL, seed=SEED)

    # Move to CPU for PCA
    Xtr_np = Xtr_full.cpu().numpy().reshape(-1, flat_dim)
    Xte_np = Xte_full.cpu().numpy().reshape(-1, flat_dim)

    results = []

    for n_comp in N_COMPONENTS_LIST:
        print(f"\n[n_components={n_comp}]")
        t0 = time.time()
        C.set_seed(SEED)

        # Fit PCA on training data
        actual_n = min(n_comp, flat_dim, Xtr_np.shape[0])
        pca = PCA(n_components=actual_n, random_state=SEED)
        pca.fit(Xtr_np)

        # Reconstruct training images
        Xtr_rec_np = pca_reconstruct(pca, Xtr_np)
        Xte_rec_np = pca_reconstruct(pca, Xte_np)

        Xtr_rec = np_to_tensor(Xtr_rec_np, im_shape, C.DEVICE)
        Xte_rec = np_to_tensor(Xte_rec_np, im_shape, C.DEVICE)
        Ytr = Ytr_full
        Yte = Yte_full

        # Train fresh CNN
        model = C.build_model("cnn", meta, width=32, seed=SEED)
        C.train_model(model, Xtr_rec, Ytr, epochs=EPOCHS, opt="sgd", lr=0.05, ncls=10)

        # Clean accuracy
        with torch.no_grad():
            logits, clean_acc = C.logits_and_acc(model, Xte_rec, Yte)

        # Gradient L1 norm
        grad_l1 = compute_grad_l1(model, Xte_rec, Yte)

        # Sweep eps -> find eps_50
        asr_by_eps = {}
        eps_50 = float("nan")
        for eps in EPS_SWEEP:
            xa = C.fgsm(model, Xte_rec, Yte, eps=eps)
            with torch.no_grad():
                adv_pred = model(xa).argmax(1).cpu()
                clean_pred = model(Xte_rec).argmax(1).cpu()
            correct = (clean_pred == Yte.cpu())
            flipped = (adv_pred != Yte.cpu())
            asr = float(flipped[correct].float().mean()) if correct.sum() > 0 else float("nan")
            asr_by_eps[eps] = asr
            if np.isnan(eps_50) and asr >= 0.5:
                eps_50 = eps

        elapsed = time.time() - t0
        print(f"  clean_acc={clean_acc:.3f}  grad_L1={grad_l1:.4f}  "
              f"eps_50={eps_50}  ({elapsed:.1f}s)")
        print(f"  ASR by eps: " +
              "  ".join(f"eps={e}:{asr_by_eps[e]:.2f}" for e in EPS_SWEEP))

        results.append({
            "n_comp": actual_n,
            "clean_acc": clean_acc,
            "grad_l1": grad_l1,
            "eps_50": eps_50,
            "asr_by_eps": asr_by_eps,
        })

    # Summary table
    print("\n" + "=" * 74)
    print(f"{'n_comp':>8} {'clean_acc':>10} {'grad_L1':>10} {'eps_50':>8}")
    print("-" * 40)
    for r in results:
        print(f"{r['n_comp']:>8} {r['clean_acc']:>10.3f} {r['grad_l1']:>10.4f} "
              f"{r['eps_50']:>8}")

    # Log-log regression: fit log(eps_50) = a + b*log(n)
    valid = [(r["n_comp"], r["eps_50"]) for r in results
             if not np.isnan(r["eps_50"]) and r["eps_50"] > 0]

    print("\n" + "=" * 74)
    print("Simon-Gabriel √n scaling test: log(eps_50) = a + b*log(n)")
    if len(valid) >= 2:
        ns = np.array([v[0] for v in valid], dtype=float)
        eps50s = np.array([v[1] for v in valid], dtype=float)
        log_n = np.log(ns)
        log_eps = np.log(eps50s)
        b, a = np.polyfit(log_n, log_eps, 1)
        print(f"  Fitted: b={b:.4f}  a={a:.4f}")
        print(f"  Simon-Gabriel prediction: b ≈ -0.5")
        print(f"  Deviation from -0.5: {b - (-0.5):+.4f}")
        if abs(b - (-0.5)) < 0.15:
            print("  => b is close to -0.5: CONSISTENT with Simon-Gabriel √n scaling")
        else:
            print("  => b deviates from -0.5: NOT fully consistent with Simon-Gabriel √n")
    else:
        print("  Insufficient valid eps_50 points for regression.")

    # Log-log regression for grad L1 vs n
    print("\nGrad L1 vs n_components log-log fit:")
    ns_all = np.array([r["n_comp"] for r in results], dtype=float)
    gl1_all = np.array([r["grad_l1"] for r in results], dtype=float)
    valid_g = ns_all > 0
    if valid_g.sum() >= 2:
        b_g, a_g = np.polyfit(np.log(ns_all[valid_g]), np.log(gl1_all[valid_g]), 1)
        print(f"  log(grad_L1) = {a_g:.4f} + {b_g:.4f}*log(n)")
        print(f"  Expected: b ≈ +0.5 (√n scaling of gradient L1 norm)")

    print("=" * 74)


if __name__ == "__main__":
    main()
