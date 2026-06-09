"""
H236 - Witness Sample: gradient-similarity influence proxy.

For each test sample, find the training sample whose removal most increases
its margin. Use gradient similarity as influence proxy:
  influence(train_i, test_j) = dot(grad_train_i, grad_test_j)
Remove top-1 influencer per test sample (batched). Retrain without those
samples. Measure: does margin of corresponding test sample increase?
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
N_TRAIN_GRAD = 500   # subsample training grads to keep memory reasonable

def get_gradients(model, X, Y, batch=64):
    """Return flattened parameter gradients per sample (averaged over params)."""
    model.eval()
    grads = []
    for i in range(0, len(X), batch):
        xb = X[i:i+batch]
        yb = Y[i:i+batch]
        g_batch = []
        for xi, yi in zip(xb, yb):
            model.zero_grad()
            logits = model(xi.unsqueeze(0))
            loss = F.cross_entropy(logits, yi.unsqueeze(0))
            loss.backward()
            g = torch.cat([p.grad.flatten() for p in model.parameters()
                           if p.grad is not None])
            g_batch.append(g.detach().cpu())
        grads.append(torch.stack(g_batch))
    return torch.cat(grads)  # (N, D)

def main():
    print("=== H236: Witness Sample ===")
    t0 = time.time()

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]

    # [1] Train baseline model
    print("\n[1] Training baseline model...")
    model = C.build_model("cnn", meta, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=EPOCHS)

    # [2] Baseline margins
    print("[2] Computing baseline margins...")
    margins_before = np.array(C.margin(model, Xte_e))
    print(f"    Mean margin before: {margins_before.mean():.4f}")

    # [3] Compute gradients on subset of training and all test samples
    print(f"[3] Computing gradients (train subset={N_TRAIN_GRAD}, test={N_EVAL})...")
    # Use a random subset of training for tractability
    rng = np.random.default_rng(SEED)
    tr_idx = rng.choice(len(Xtr), N_TRAIN_GRAD, replace=False)
    Xtr_sub = Xtr[tr_idx]
    Ytr_sub = Ytr[tr_idx]

    print("    Computing training gradients...")
    train_grads = get_gradients(model, Xtr_sub, Ytr_sub)  # (N_TRAIN_GRAD, D)
    print("    Computing test gradients...")
    test_grads = get_gradients(model, Xte_e, Yte_e)        # (N_EVAL, D)

    # Normalise for cosine similarity
    train_grads_n = F.normalize(train_grads, dim=1)
    test_grads_n = F.normalize(test_grads, dim=1)

    # [4] For each test sample, find top-1 influencing training sample
    print("[4] Computing influence (gradient cosine similarity)...")
    # influence matrix: (N_EVAL, N_TRAIN_GRAD)
    influence = test_grads_n @ train_grads_n.T  # (N_EVAL, N_TRAIN_GRAD)
    top_influencer_local = influence.argmax(dim=1).numpy()  # index into tr_idx
    top_influencer_global = tr_idx[top_influencer_local]

    # unique training samples to remove
    samples_to_remove = set(top_influencer_global.tolist())
    print(f"    Unique training samples to remove: {len(samples_to_remove)}")

    # [5] Retrain without those samples
    print("[5] Retraining without top-1 influencers...")
    keep_mask = np.ones(len(Xtr), dtype=bool)
    for idx in samples_to_remove:
        keep_mask[idx] = False
    keep_t = torch.from_numpy(keep_mask)
    Xtr_pruned = Xtr[keep_t]
    Ytr_pruned = Ytr[keep_t]

    C.set_seed(SEED)
    model2 = C.build_model("cnn", meta, seed=SEED)
    C.train_model(model2, Xtr_pruned, Ytr_pruned, epochs=EPOCHS)

    # [6] Margins after removal
    print("[6] Computing margins after removal...")
    margins_after = np.array(C.margin(model2, Xte_e))

    margin_delta = margins_after - margins_before
    increased = (margin_delta > 0).sum()
    mean_delta = margin_delta.mean()

    print(f"\n--- Summary ---")
    print(f"Training samples removed: {len(samples_to_remove)}/{len(Xtr)}")
    print(f"Mean margin before: {margins_before.mean():.4f}")
    print(f"Mean margin after:  {margins_after.mean():.4f}")
    print(f"Mean margin delta:  {mean_delta:+.4f}")
    print(f"Samples with increased margin: {increased}/{N_EVAL} "
          f"({100*increased/N_EVAL:.1f}%)")
    print(f"Hypothesis: removing top-1 influencer should increase margin "
          f"of corresponding test sample.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
