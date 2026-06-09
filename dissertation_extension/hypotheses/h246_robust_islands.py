"""
H246 - Robust Islands: persistence of high-margin samples during PGD-AT.

Train with PGD-AT for 50 epochs, saving margin per sample every 5 epochs.
After peak robustness (early epochs), does margin variance increase as mean
decreases? Also: identify "robust island" samples (top-10% margin at final
epoch) — were they also top-10% at epoch 5? Measure persistence.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0
EPS = 0.1
N_EVAL = 300
AT_EPOCHS = 50
CHECKPOINT_EVERY = 5
BATCH_SIZE = 128
AT_STEPS = 5

def train_pgd_at_epoch(model, Xtr, Ytr, optimizer, eps, pgd_steps, alpha):
    """Train one epoch of PGD adversarial training."""
    model.train()
    n = len(Xtr)
    perm = torch.randperm(n)
    total_loss = 0.0
    n_batches = 0
    for i in range(0, n, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        xb = Xtr[idx]
        yb = Ytr[idx]
        xb_adv = C.pgd(model, xb, yb, eps=eps, steps=pgd_steps, alpha=alpha)
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(xb_adv), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches

def main():
    print("=== H246: Robust Islands ===")
    t0 = time.time()

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]

    # [1] PGD-AT training with checkpoints
    print(f"\n[1] PGD-AT training for {AT_EPOCHS} epochs, "
          f"checkpoints every {CHECKPOINT_EVERY} epochs...")
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, seed=SEED)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=1e-4)

    checkpoints = {}  # epoch -> margins array
    margin_means = []
    margin_stds = []
    epochs_recorded = []

    for epoch in range(1, AT_EPOCHS + 1):
        loss = train_pgd_at_epoch(model, Xtr, Ytr, optimizer,
                                  eps=EPS, pgd_steps=AT_STEPS, alpha=EPS/4)

        if epoch % CHECKPOINT_EVERY == 0:
            margins = np.array(C.margin(model, Xte_e))
            checkpoints[epoch] = margins
            mean_m = margins.mean()
            std_m = margins.std()
            margin_means.append(mean_m)
            margin_stds.append(std_m)
            epochs_recorded.append(epoch)

            model.eval()
            Xadv = C.pgd(model, Xte_e, Yte_e, eps=EPS, steps=10, alpha=0.01)
            with torch.no_grad():
                asr = (model(Xadv).argmax(1).cpu() != Yte_e.cpu()).float().mean().item()
            print(f"    Epoch {epoch:3d}: loss={loss:.4f}, "
                  f"margin_mean={mean_m:.4f}, margin_std={std_m:.4f}, "
                  f"PGD_ASR={asr:.3f}")

    # [2] Find peak robustness epoch (highest mean margin)
    peak_idx = int(np.argmax(margin_means))
    peak_epoch = epochs_recorded[peak_idx]
    print(f"\n[2] Peak robustness at epoch {peak_epoch} "
          f"(margin_mean={margin_means[peak_idx]:.4f})")

    # [3] Analyse variance after peak
    print("[3] Margin mean and variance trajectory after peak...")
    for i, ep in enumerate(epochs_recorded):
        indicator = " <- PEAK" if ep == peak_epoch else ""
        print(f"    Epoch {ep:3d}: mean={margin_means[i]:.4f}, "
              f"std={margin_stds[i]:.4f}{indicator}")

    # [4] Robust island persistence
    print("\n[4] Robust island persistence analysis...")
    ISLAND_FRAC = 0.10

    # Robust island at epoch 5 (first checkpoint)
    ep_early = epochs_recorded[0]
    ep_final = epochs_recorded[-1]
    margins_early = checkpoints[ep_early]
    margins_final = checkpoints[ep_final]

    thresh_early = np.percentile(margins_early, 100 * (1 - ISLAND_FRAC))
    thresh_final = np.percentile(margins_final, 100 * (1 - ISLAND_FRAC))

    island_early = set(np.where(margins_early >= thresh_early)[0])
    island_final = set(np.where(margins_final >= thresh_final)[0])

    overlap = island_early & island_final
    persistence = len(overlap) / len(island_final) if island_final else float('nan')

    print(f"    Top-{ISLAND_FRAC:.0%} island size: "
          f"epoch {ep_early}={len(island_early)}, "
          f"epoch {ep_final}={len(island_final)}")
    print(f"    Overlap: {len(overlap)}")
    print(f"    Persistence (overlap/final_island): {persistence:.3f}")

    # Check if variance increases after peak
    post_peak_stds = margin_stds[peak_idx:]
    post_peak_means = margin_means[peak_idx:]
    var_increasing = all(post_peak_stds[i] <= post_peak_stds[i+1]
                         for i in range(len(post_peak_stds)-1))

    print(f"\n--- Summary ---")
    print(f"Peak robustness at epoch {peak_epoch}")
    print(f"Margin mean at peak: {margin_means[peak_idx]:.4f}, "
          f"at final: {margin_means[-1]:.4f}")
    print(f"Margin std at peak: {margin_stds[peak_idx]:.4f}, "
          f"at final: {margin_stds[-1]:.4f}")
    print(f"Variance monotonically increases after peak: {var_increasing}")
    print(f"Robust island persistence (epoch {ep_early} → {ep_final}): "
          f"{persistence:.3f}")
    print("Interpretation: persistence < 0.5 = island membership unstable; "
          "variance increasing after peak = robust overfitting spreading.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
