"""
H204 - Randomized smoothing discards hard samples: low-margin samples lose
       certifiability under Cohen et al. (2019) smoothing.

Motivated by 2410.06895 (SOTA certified training improves ACR by ignoring hard
samples). We verify on Fashion-MNIST that the per-sample certified radius
r_i = sigma * Phi^{-1}(p_A) correlates strongly with the vanilla decision
margin, and that bottom-quartile margin samples have a dramatically higher
abstain rate than top-quartile samples.

Key outputs
-----------
1. Smoothed-model clean accuracy (majority-vote over M=100 noise draws)
2. Average Certified Radius (ACR) and global abstain fraction
3. Spearman rho: vanilla margin vs certified radius
4. Q1 abstain rate vs Q4 abstain rate
5. AUROC: vanilla margin -> certifiability (r>0 binary label)
"""
import os, sys, time
import numpy as np
import torch
from scipy.stats import spearmanr, norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ── config ────────────────────────────────────────────────────────────────────
SIGMA   = 0.25   # smoothing noise std
M       = 100    # Monte-Carlo draws per sample
N_TEST  = 300    # samples from Fashion-MNIST test set
SEED    = 0
META    = {"channels": 1, "size": 28, "n_classes": 10}


# ── helpers ───────────────────────────────────────────────────────────────────

def majority_vote(model, x_single, sigma, M, seed=0):
    """
    Run M noisy forward passes for a single sample (1,C,H,W) and return
    (predicted_class, p_A) where p_A is the fraction of votes for top class.
    """
    torch.manual_seed(seed)
    noise = torch.randn(M, *x_single.shape[1:], device=x_single.device) * sigma
    x_rep = x_single.expand(M, -1, -1, -1) + noise
    x_rep = x_rep.clamp(0.0, 1.0)
    with torch.no_grad():
        logits = model(x_rep)            # (M, n_classes)
    votes = logits.argmax(dim=1)         # (M,)
    counts = torch.bincount(votes, minlength=META["n_classes"])
    top_cls = int(counts.argmax())
    p_A = float(counts[top_cls]) / M
    return top_cls, p_A


def certified_radius(p_A, sigma):
    """Cohen et al. 2019: r = sigma * Phi^{-1}(p_A) if p_A > 0.5, else 0."""
    if p_A > 0.5:
        return sigma * norm.ppf(p_A)
    return 0.0


def batch_smooth(model, X, sigma, M):
    """
    Vectorised smoothing: add noise to the *whole batch* at once to avoid a
    Python loop over samples.  Returns (pred_classes, p_A_array).
    """
    N = X.size(0)
    # Expand: (N, C, H, W) -> (N*M, C, H, W) then add noise
    X_rep = X.unsqueeze(1).expand(-1, M, -1, -1, -1).reshape(N * M, *X.shape[1:])
    torch.manual_seed(42)
    noise = torch.randn_like(X_rep) * sigma
    X_noisy = (X_rep + noise).clamp(0.0, 1.0)

    # Forward in chunks to avoid OOM
    chunk = 512
    all_votes = []
    with torch.no_grad():
        for i in range(0, N * M, chunk):
            all_votes.append(model(X_noisy[i:i + chunk]).argmax(dim=1))
    all_votes = torch.cat(all_votes).reshape(N, M)   # (N, M)

    # majority vote and p_A per sample
    preds, p_As = [], []
    for i in range(N):
        counts = torch.bincount(all_votes[i], minlength=META["n_classes"])
        top_cls = int(counts.argmax())
        preds.append(top_cls)
        p_As.append(float(counts[top_cls]) / M)
    return np.array(preds, dtype=int), np.array(p_As)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 74)
    print("H204 - Randomized smoothing discards hard (low-margin) samples")
    print("=" * 74)
    print(f"sigma={SIGMA}  M={M}  N_test={N_TEST}  seed={SEED}  device={C.DEVICE}")

    # 1. Data ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    Xtr, Ytr, Xte_full, Yte_full = C.load_dataset("fashion_mnist")
    Xte = Xte_full[:N_TEST]
    Yte = Yte_full[:N_TEST]
    print(f"Data loaded: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    # 2. Train model ───────────────────────────────────────────────────────────
    model = C.build_model("cnn", META, width=32, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=10)
    model.eval()

    # 3. Vanilla clean accuracy and per-sample margin ──────────────────────────
    with torch.no_grad():
        logits_te = model(Xte).cpu()
    vanilla_preds = logits_te.argmax(dim=1)
    vanilla_acc = float((vanilla_preds == Yte.cpu()).float().mean())
    margins = C.margin_of(logits_te, Yte.cpu())   # numpy array (N,)
    print(f"Vanilla clean accuracy: {vanilla_acc:.4f}")
    print(f"Vanilla margin  mean={margins.mean():.3f}  min={margins.min():.3f}  max={margins.max():.3f}")

    # 4. Randomised smoothing ──────────────────────────────────────────────────
    print(f"\nRunning randomised smoothing (M={M} draws per sample)...")
    t1 = time.time()
    smooth_preds, p_As = batch_smooth(model, Xte, SIGMA, M)
    print(f"  done in {time.time()-t1:.1f}s")

    # Smoothed model clean accuracy (majority-vote vs true labels)
    smooth_acc = float((smooth_preds == Yte.cpu().numpy()).mean())

    # Certified radii
    radii = np.array([certified_radius(p, SIGMA) for p in p_As])
    abstain = (radii == 0.0)
    ACR = float(radii.mean())
    abstain_rate = float(abstain.mean())

    print(f"\nSmoothed model clean accuracy (majority vote): {smooth_acc:.4f}")
    print(f"ACR (avg certified radius):                   {ACR:.4f}")
    print(f"Global abstain rate (r=0):                    {abstain_rate:.4f}")

    # 5. Spearman rho: vanilla margin vs certified radius ──────────────────────
    rho, pval = spearmanr(margins, radii)
    print(f"\nSpearman rho (margin vs radius): {rho:.4f}  p={pval:.2e}")

    # 6. Q1 vs Q4 abstain rate ─────────────────────────────────────────────────
    q25 = np.percentile(margins, 25)
    q75 = np.percentile(margins, 75)
    q1_mask = margins <= q25
    q4_mask = margins >= q75
    q1_abstain = float(abstain[q1_mask].mean())
    q4_abstain = float(abstain[q4_mask].mean())
    print(f"\nMargin quartile analysis:")
    print(f"  Q1 (low margin  <= {q25:.2f}): n={q1_mask.sum()}  abstain rate={q1_abstain:.4f}")
    print(f"  Q4 (high margin >= {q75:.2f}): n={q4_mask.sum()}  abstain rate={q4_abstain:.4f}")
    print(f"  Ratio Q1/Q4 abstain: {q1_abstain/(q4_abstain+1e-9):.1f}x")

    # 7. AUROC: vanilla margin -> certifiability ───────────────────────────────
    certifiable = (~abstain).astype(int)
    auroc = C.safe_auroc(certifiable, margins)
    print(f"\nAUROC (margin -> certifiable): {auroc:.4f}")

    # 8. Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  Vanilla clean acc:          {vanilla_acc:.4f}")
    print(f"  Smoothed clean acc:         {smooth_acc:.4f}")
    print(f"  ACR:                        {ACR:.4f}")
    print(f"  Global abstain rate:        {abstain_rate:.4f}")
    print(f"  Spearman rho (margin/r):    {rho:.4f}  (p={pval:.2e})")
    print(f"  Q1 abstain rate:            {q1_abstain:.4f}")
    print(f"  Q4 abstain rate:            {q4_abstain:.4f}")
    print(f"  AUROC margin->certifiable:  {auroc:.4f}")
    print(f"  Total runtime:              {total_time:.1f}s")
    print("=" * 74)
    print("INTERPRETATION")
    if rho > 0.3 and pval < 0.05:
        print("  [CONFIRMED] Strong positive Spearman rho: high-margin samples")
        print("  receive larger certified radii.  Low-margin (hard) samples are")
        print("  disproportionately pushed to the abstain region (r=0), consistent")
        print("  with 2410.06895 — certified training gains ACR by abandoning hard")
        print("  samples rather than genuinely hardening their decision boundaries.")
    elif rho > 0.1 and pval < 0.05:
        print("  [PARTIAL] Moderate positive Spearman rho; trend holds but weaker")
        print("  than expected.  Hard samples still abstain more than easy ones.")
    else:
        print("  [INCONCLUSIVE] Weak or non-significant Spearman rho.  The margin-")
        print("  radius link is not clearly established on this model/dataset.")
    print("=" * 74)


if __name__ == "__main__":
    main()
