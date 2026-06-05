"""
H218 - Robust overfitting: which samples overfit first?

Train PGD-AT model for 40 epochs, record per-sample margins at checkpoints
[5,10,15,20,25,30,35,40]. Classify samples as 'overfit victims' if their
peak margin substantially exceeds their epoch-40 margin (delta > 0.1).

Questions:
1. What fraction of samples are overfit victims?
2. Were overfit victims low-margin samples at epoch 5? (Spearman ρ)
3. Does the margin AUROC (predicting PGD success) change across checkpoints?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS          = "fashion_mnist"
N_EVAL      = 200
SEED        = 0
EPS         = 0.1
TOTAL_EPOCHS = 40
CHECKPOINTS  = [5, 10, 15, 20, 25, 30, 35, 40]
DELTA_THRESH = 0.1   # margin drop to qualify as overfit victim


# ---------------------------------------------------------------------------
# Train N more epochs from current model state
# ---------------------------------------------------------------------------
def train_n_epochs(model, Xtr, Ytr, n_epochs, opt, sched,
                   batch=128, adv_eps=EPS, adv_steps=7):
    model.train()
    n = Xtr.size(0)
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_adv = C.pgd(model, xb, yb, eps=adv_eps, steps=adv_steps,
                           alpha=2.5 * adv_eps / adv_steps)
            opt.zero_grad()
            F.cross_entropy(model(xb_adv), yb).backward()
            opt.step()
        sched.step()
    model.eval()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("=== H218: Robust Overfitting — per-sample margin dynamics ===")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  N_EVAL={N_EVAL}  eps={EPS}")
    print(f"Checkpoints: {CHECKPOINTS}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_EVAL, seed=SEED)

    model = C.build_model("cnn", meta, width=32, seed=SEED)
    opt   = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                             lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_EPOCHS)

    # --- 1. Train with checkpoints ---
    print("\n--- 1. Training (PGD-AT, 40 epochs with margin snapshots) ---")
    margins_at_ckpt = {}   # checkpoint epoch -> numpy array shape (N_EVAL,)
    auroc_at_ckpt   = {}
    epoch_done = 0
    t0 = time.time()

    for ckpt in CHECKPOINTS:
        n_more = ckpt - epoch_done
        train_n_epochs(model, Xtr, Ytr, n_more, opt, sched)
        epoch_done = ckpt

        mgn = C.margin(model, Xte, Yte)
        margins_at_ckpt[ckpt] = mgn

        # AUROC margin -> PGD success (snapshot)
        pgd_res = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=10)
        corr = pgd_res["correct"].astype(bool)
        auroc = C.safe_auroc(pgd_res["flips"][corr], -mgn[corr])
        auroc_at_ckpt[ckpt] = auroc

        _, clean_acc = C.logits_and_acc(model, Xte, Yte)
        print(f"  epoch {ckpt:>2}: clean={clean_acc:.3f}  "
              f"margin mean={mgn.mean():.3f} std={mgn.std():.3f}  "
              f"AUROC={auroc:.3f}  PGD_ASR={pgd_res['asr']:.3f}")

    print(f"  Total training time: {time.time()-t0:.1f}s")

    # --- 2. Per-sample analysis ---
    print("\n--- 2. Per-sample overfit analysis ---")
    margin_matrix = np.stack([margins_at_ckpt[c] for c in CHECKPOINTS], axis=1)
    # N_EVAL × len(CHECKPOINTS)

    peak_margin  = margin_matrix.max(axis=1)
    final_margin = margin_matrix[:, -1]
    delta        = peak_margin - final_margin   # how much margin dropped from peak

    overfit_mask = delta > DELTA_THRESH
    frac_overfit = overfit_mask.mean()
    print(f"  Fraction of overfit victims (delta > {DELTA_THRESH}): {frac_overfit:.3f}")

    early_margin = margin_matrix[:, 0]   # epoch 5
    rho, pval = spearmanr(early_margin, delta)
    print(f"  Spearman ρ(early_margin@ep5, delta): {rho:.4f}  p={pval:.4e}")
    print("  (negative ρ: low-margin samples at ep5 tend to overfit more)")

    # --- 3. Fraction of overfit victims per quartile of initial margin ---
    print("\n--- 3. Overfit fraction by initial-margin quartile ---")
    quartile_edges = np.percentile(early_margin, [0, 25, 50, 75, 100])
    for q in range(4):
        lo, hi = quartile_edges[q], quartile_edges[q+1]
        mask = (early_margin >= lo) & (early_margin <= hi)
        frac = overfit_mask[mask].mean() if mask.sum() > 0 else float("nan")
        print(f"  Q{q+1} (margin {lo:.3f}..{hi:.3f}): "
              f"n={mask.sum()}  overfit_frac={frac:.3f}")

    # --- Summary ---
    print("\n" + "=" * 74)
    print("--- Summary ---")
    print(f"{'Epoch':>6} {'Mgn_mean':>9} {'Mgn_std':>8} {'AUROC':>7}")
    for ckpt in CHECKPOINTS:
        mgn = margins_at_ckpt[ckpt]
        print(f"  {ckpt:>4}  {mgn.mean():>9.3f}  {mgn.std():>8.3f}  {auroc_at_ckpt[ckpt]:>7.3f}")
    print(f"\n  Overfit victim fraction: {frac_overfit:.3f}")
    print(f"  Spearman ρ(early_margin, delta): {rho:.4f}")
    print("=" * 74)
    print("Interpretation: if low-margin samples at epoch 5 show larger delta,")
    print("robust overfitting preferentially harms initially-hard samples — those")
    print("that the model barely learned to classify robustly before overfitting.")
    print("=" * 74)


if __name__ == "__main__":
    main()
