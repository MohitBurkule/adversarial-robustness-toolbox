"""
H235 - Overfitting Echo: spurious correlations → adversarial boundary.

Find samples where training loss at the final epoch is highest (proxy for
memorised/hard samples). Remove top-K most-memorised training samples. Retrain.
Do the test samples that were previously adversarially vulnerable become more
robust? Measure: Jaccard overlap of vulnerable set before vs after removal.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0
EPS = 0.1
N_EVAL = 300
EPOCHS = 10
REMOVE_FRACS = [0.05, 0.10, 0.20]

def get_per_sample_loss(model, X, Y, batch=256):
    """Return per-sample cross-entropy loss (no grad)."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i+batch]
            yb = Y[i:i+batch]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction='none')
            losses.append(loss.cpu())
    return torch.cat(losses).numpy()

def pgd_success(model, X, Y):
    """Return boolean array: True if PGD attack succeeded."""
    Xadv = C.pgd(model, X, Y, eps=EPS, steps=10, alpha=0.01)
    model.eval()
    with torch.no_grad():
        preds = model(Xadv).argmax(1).cpu()
    return (preds != Y.cpu()).numpy()

def jaccard(a, b):
    a, b = set(np.where(a)[0]), set(np.where(b)[0])
    if len(a | b) == 0:
        return float('nan')
    return len(a & b) / len(a | b)

def main():
    print("=== H235: Overfitting Echo ===")
    t0 = time.time()

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]

    # [1] Train baseline model
    print("\n[1] Training baseline model...")
    model = C.build_model("cnn", meta, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=EPOCHS)

    # [2] Compute per-sample training loss (memorisation proxy)
    print("[2] Computing per-sample training loss...")
    train_losses = get_per_sample_loss(model, Xtr, Ytr)
    print(f"    Train loss: mean={train_losses.mean():.4f}, "
          f"max={train_losses.max():.4f}, min={train_losses.min():.4f}")

    # [3] Identify vulnerable test samples (baseline)
    print("[3] Identifying adversarially vulnerable test samples (baseline)...")
    baseline_vuln = pgd_success(model, Xte_e, Yte_e)
    n_vuln_base = baseline_vuln.sum()
    print(f"    Baseline vulnerable: {n_vuln_base}/{N_EVAL} "
          f"({100*n_vuln_base/N_EVAL:.1f}%)")

    # [4] Remove top-K memorised samples, retrain, measure Jaccard
    print("\n[4] Removing top-K memorised samples and retraining...")
    results = []
    sorted_idx = np.argsort(train_losses)[::-1]  # descending loss

    for frac in REMOVE_FRACS:
        k = int(len(Xtr) * frac)
        remove_idx = set(sorted_idx[:k].tolist())
        keep_mask = np.array([i not in remove_idx for i in range(len(Xtr))])
        keep_mask_t = torch.from_numpy(keep_mask)

        Xtr_pruned = Xtr[keep_mask_t]
        Ytr_pruned = Ytr[keep_mask_t]

        C.set_seed(SEED)
        model_pruned = C.build_model("cnn", meta, seed=SEED)
        C.train_model(model_pruned, Xtr_pruned, Ytr_pruned, epochs=EPOCHS)

        pruned_vuln = pgd_success(model_pruned, Xte_e, Yte_e)
        n_vuln_pruned = pruned_vuln.sum()
        j = jaccard(baseline_vuln, pruned_vuln)

        # clean accuracy
        model_pruned.eval()
        with torch.no_grad():
            preds = model_pruned(Xte_e).argmax(1).cpu()
        clean_acc = (preds == Yte_e.cpu()).float().mean().item()

        results.append({
            'frac': frac, 'k': k,
            'n_vuln': n_vuln_pruned,
            'jaccard': j,
            'clean_acc': clean_acc,
        })
        print(f"    frac={frac:.0%} (k={k}): vuln={n_vuln_pruned} "
              f"({100*n_vuln_pruned/N_EVAL:.1f}%), "
              f"Jaccard={j:.3f}, clean_acc={clean_acc:.3f}")

    print(f"\n--- Summary ---")
    print(f"Baseline vulnerable: {n_vuln_base}/{N_EVAL}")
    for r in results:
        print(f"  Remove top-{r['frac']:.0%} (k={r['k']}): "
              f"vuln={r['n_vuln']} ({100*r['n_vuln']/N_EVAL:.1f}%), "
              f"Jaccard={r['jaccard']:.3f}, clean_acc={r['clean_acc']:.3f}")
    print(f"Interpretation: Jaccard<1 = vulnerability set shifts; "
          f"lower Jaccard = more migration of vulnerabilities after removal.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
