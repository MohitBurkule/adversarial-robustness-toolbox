"""
H223 - How many noise samples K needed for smoothed robustness signal to converge?

Train base CNN. For K in [1,5,10,20,50,100,200]:
  Compute smoothed prediction for each test sample: run model K times with
  N(0,0.1^2) noise, take majority vote. Compute soft margin:
    smoothed_margin = fraction_correct_class - fraction_second_best_class.
For each K:
  - AUROC(smoothed_margin -> PGD_success)
  - Spearman rho(smoothed_margin_K, logit_margin)
  - Mean abs difference |smoothed_margin_K_norm - logit_margin_norm|  (both normalised to [0,1])
Plot convergence: at what K does AUROC plateau (within 0.01 of K=200)?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_EVAL = 300
SEED = 0
EPS = 0.1
SIGMA = 0.1
K_LIST = [1, 5, 10, 20, 50, 100, 200]

os.makedirs("results/fashion_mnist", exist_ok=True)


def smoothed_margin(model, X, Y, K, sigma=0.1, batch_inner=64):
    """For each sample in X, run model K times with N(0,sigma^2) noise.
    Return soft margin: fraction_correct_class - fraction_second_best_class.
    """
    model.eval()
    N = X.size(0)
    n_classes = 10
    vote_counts = torch.zeros(N, n_classes)

    for k in range(K):
        noise = torch.randn_like(X) * sigma
        xn = (X + noise).clamp(0, 1)
        with torch.no_grad():
            for i in range(0, N, batch_inner):
                logits = model(xn[i:i+batch_inner]).cpu()
                preds = logits.argmax(1)
                for j, p in enumerate(preds):
                    vote_counts[i+j, p] += 1

    vote_frac = vote_counts / K  # (N, n_classes)
    Y_cpu = Y.cpu()
    # fraction for correct class
    correct_frac = vote_frac[torch.arange(N), Y_cpu]
    # fraction for second-best class
    tmp = vote_frac.clone()
    tmp[torch.arange(N), Y_cpu] = -1.0
    second_frac = tmp.max(1).values
    sm = (correct_frac - second_frac).numpy()
    return sm


def spearman_rho(a, b):
    from scipy.stats import spearmanr
    try:
        r, _ = spearmanr(a, b)
        return float(r)
    except Exception:
        return float("nan")


def normalise_01(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return arr - mn
    return (arr - mn) / (mx - mn)


def main():
    print("=" * 70)
    print("H223 - Smoothing K convergence for robustness signal")
    print("=" * 70)

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_eval=N_EVAL, seed=SEED)
    print(f"Device={C.DEVICE}  N_eval={N_EVAL}  sigma={SIGMA}  eps={EPS}")

    # Train base model
    t0 = time.time()
    model = C.build_model("cnn", meta, width=32, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=10)
    print(f"Training done in {time.time()-t0:.1f}s")

    # Compute logit margin (reference)
    logit_margin = C.margin(model, Xte, Yte)  # numpy (N,)

    # Compute PGD labels on original model
    print("Computing PGD adversarial examples...")
    t0 = time.time()
    Xadv = C.pgd(model, Xte, Yte, eps=EPS, steps=10, alpha=0.01)
    with torch.no_grad():
        pgd_preds = model(Xadv).argmax(1).cpu()
    pgd_success = (pgd_preds != Yte.cpu()).numpy().astype(int)
    print(f"PGD done in {time.time()-t0:.1f}s. ASR={pgd_success.mean():.3f}")

    # For each K, compute smoothed margin
    results = []
    logit_margin_norm = normalise_01(logit_margin)

    print(f"\n{'K':>5}  {'AUROC':>8}  {'Spearman_rho':>13}  {'Mean_abs_diff':>14}")
    print("-" * 50)
    auroc_at_k = {}
    for K in K_LIST:
        t0 = time.time()
        sm = smoothed_margin(model, Xte, Yte, K=K, sigma=SIGMA)
        auroc = C.safe_auroc(pgd_success, -sm)  # lower margin -> higher attack success
        sm_norm = normalise_01(sm)
        rho = spearman_rho(sm, logit_margin)
        mad = float(np.mean(np.abs(sm_norm - logit_margin_norm)))
        elapsed = time.time() - t0
        auroc_at_k[K] = auroc
        row = {"K": K, "AUROC": auroc, "Spearman_rho": rho, "Mean_abs_diff": mad, "time_s": elapsed}
        results.append(row)
        print(f"{K:>5}  {auroc:>8.4f}  {rho:>13.4f}  {mad:>14.4f}  ({elapsed:.1f}s)")

    # Find convergence point: min K where |AUROC_K - AUROC_200| <= 0.01
    auroc_200 = auroc_at_k[200]
    conv_k = None
    for K in K_LIST:
        if abs(auroc_at_k[K] - auroc_200) <= 0.01:
            conv_k = K
            break

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  AUROC at K=200 (reference):  {auroc_200:.4f}")
    print(f"  Convergence K (AUROC within 0.01 of K=200): {conv_k}")
    print(f"  Interpretation: K>={conv_k} provides a stable smoothed robustness signal")
    print(f"  Spearman rho shows rank correlation with logit margin across K values")
    print("=" * 70)


if __name__ == "__main__":
    main()
