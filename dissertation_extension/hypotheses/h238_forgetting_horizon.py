"""
H238 - Forgetting Horizon: catastrophic forgetting events predict adversarial vulnerability.

Train model for 30 epochs, recording per-sample correct/incorrect at each epoch.
Count "forgetting events" = number of times a sample flips from correct to
incorrect during training. Does forgetting_count predict adversarial vulnerability?
Measure: AUROC(forgetting_count → PGD success), Spearman ρ(forgetting_count, margin).
"""
import os, sys, time
import numpy as np
import torch

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
TOTAL_EPOCHS = 30
RECORD_EVERY = 1   # record every epoch

def eval_correctness(model, X, Y, batch=256):
    """Return boolean numpy array of per-sample correctness."""
    model.eval()
    correct = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i+batch]
            yb = Y[i:i+batch]
            preds = model(xb).argmax(1).cpu()
            correct.append((preds == yb.cpu()).numpy())
    return np.concatenate(correct)

def main():
    print("=== H238: Forgetting Horizon ===")
    t0 = time.time()

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")

    # Use training subset for forgetting tracking (full set can be slow)
    N_TRACK = min(3000, len(Xtr))
    Xtr_track = Xtr[:N_TRACK]
    Ytr_track = Ytr[:N_TRACK]
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]

    # [1] Train epoch by epoch, recording correctness
    print(f"\n[1] Training for {TOTAL_EPOCHS} epochs, tracking forgetting on "
          f"{N_TRACK} training samples...")
    model = C.build_model("cnn", meta, seed=SEED)

    # We need manual epoch-by-epoch training
    # C.train_model trains all at once — we call it one epoch at a time
    correctness_history = []  # list of (N_TRACK,) bool arrays

    for epoch in range(1, TOTAL_EPOCHS + 1):
        C.train_model(model, Xtr_track, Ytr_track, epochs=1)
        if epoch % RECORD_EVERY == 0:
            corr = eval_correctness(model, Xtr_track, Ytr_track)
            correctness_history.append(corr)
            if epoch % 5 == 0:
                print(f"    Epoch {epoch:3d}: train_acc={corr.mean():.3f}")

    # [2] Count forgetting events: transitions from correct → incorrect
    print("[2] Counting forgetting events...")
    history_arr = np.stack(correctness_history, axis=0)  # (T, N_TRACK)
    forgetting_count = np.zeros(N_TRACK, dtype=int)
    for t in range(1, len(correctness_history)):
        flip = history_arr[t-1] & ~history_arr[t]  # was correct, now wrong
        forgetting_count += flip.astype(int)

    print(f"    Forgetting count: mean={forgetting_count.mean():.2f}, "
          f"max={forgetting_count.max()}, "
          f"zero-forgetting={( forgetting_count==0).sum()}/{N_TRACK}")

    # [3] PGD attack on training-tracked samples
    print("[3] Running PGD attack on tracked training samples...")
    Xadv = C.pgd(model, Xtr_track, Ytr_track, eps=EPS, steps=10, alpha=0.01)
    model.eval()
    with torch.no_grad():
        adv_preds = model(Xadv).argmax(1).cpu()
    pgd_succ = (adv_preds != Ytr_track.cpu()).numpy().astype(int)

    margins = np.array(C.margin(model, Xtr_track))

    # [4] AUROC and Spearman
    print("[4] Computing metrics...")
    auroc = float('nan')
    spearman_fc_margin = float('nan')
    spearman_fc_pgd = float('nan')

    if HAS_SKLEARN:
        try:
            if len(np.unique(pgd_succ)) == 2:
                auroc = roc_auc_score(pgd_succ, forgetting_count)
        except Exception:
            pass
        try:
            spearman_fc_margin = spearmanr(forgetting_count, margins).correlation
        except Exception:
            pass
        try:
            spearman_fc_pgd = spearmanr(forgetting_count, pgd_succ).correlation
        except Exception:
            pass

    # Quartile analysis
    quartiles = np.percentile(forgetting_count, [25, 50, 75])
    print(f"    Forgetting count quartiles: {quartiles}")
    for q_lo, q_hi, label in [
        (0, quartiles[0], "Q1 (low)"),
        (quartiles[0], quartiles[1], "Q2"),
        (quartiles[1], quartiles[2], "Q3"),
        (quartiles[2], forgetting_count.max()+1, "Q4 (high)"),
    ]:
        mask = (forgetting_count >= q_lo) & (forgetting_count < q_hi)
        if mask.sum() > 0:
            print(f"    {label}: n={mask.sum()}, "
                  f"pgd_asr={pgd_succ[mask].mean():.3f}, "
                  f"margin={margins[mask].mean():.4f}")

    print(f"\n--- Summary ---")
    print(f"AUROC(forgetting_count → PGD success): {auroc:.3f}")
    print(f"Spearman ρ(forgetting_count, margin):  {spearman_fc_margin:.3f}")
    print(f"Spearman ρ(forgetting_count, pgd_succ):{spearman_fc_pgd:.3f}")
    print(f"Overall PGD ASR: {pgd_succ.mean():.3f}")
    print("Interpretation: AUROC>0.6 and negative Spearman(fc,margin) "
          "=> more forgetting → more vulnerable.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
