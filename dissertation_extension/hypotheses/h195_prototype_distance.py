"""
H195 - Distance to nearest wrong-class prototype predicts adversarial vulnerability.

Hypothesis: in the penultimate feature space, a sample's distance to the nearest
WRONG-class centroid (prototype) is a stronger predictor of adversarial vulnerability
than the raw softmax margin.

Intuition: a sample geometrically close to an adversarial class centroid already
lies near the wrong-class region; a small perturbation suffices to cross the
boundary.  By contrast, margin is a one-dimensional signal that can be inflated
by confidence calibration without reflecting true geometric security.

Procedure:
  1. Train SmallCNN on Fashion-MNIST (full train set).
  2. Extract penultimate (256-d) features for all train samples via forward hook.
  3. Compute per-class centroids in feature space (mean of correct-class features).
  4. For each test sample (Xte[:500]):
       a. min_wrong_dist  = min L2 distance to any WRONG-class centroid
       b. own_dist        = L2 distance to own class centroid
       c. proto_margin    = min_wrong_dist - own_dist   (signed gap)
       d. softmax margin  = C.margin_of(logits, Y)
  5. Run FGSM and PGD attacks; record per-sample success.
  6. AUROC(-min_wrong_dist → attack success) vs AUROC(margin → attack success).
  7. Spearman ρ between min_wrong_dist and softmax margin.

Expected finding: AUROC for prototype distance ≥ AUROC for margin on both attacks.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS        = "fashion_mnist"
N_TE      = 500
EPS       = 0.1
PGD_STEPS = 10
SEEDS     = [0, 1, 2]


# ---------------------------------------------------------------------------
# Feature extraction via forward hook on the penultimate activation
# ---------------------------------------------------------------------------
def extract_penultimate(model, X, batch=256):
    """Return (N, 256) penultimate features by hooking head[2] (ReLU after first Linear)."""
    captured = {}

    def hook_fn(module, inp, out):
        captured["feat"] = out.detach().cpu()

    # SmallCNN.head = Sequential(Flatten, Linear->256, Act, Linear->n_classes)
    # head[2] is the activation after the first linear — the penultimate representation
    handle = model.head[2].register_forward_hook(hook_fn)

    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            _ = model(X[i:i + batch])
            feats.append(captured["feat"])
    handle.remove()
    return torch.cat(feats, dim=0)   # (N, 256)


# ---------------------------------------------------------------------------
# Prototype distance computation
# ---------------------------------------------------------------------------
def compute_centroids(features, labels, n_classes):
    """Mean feature per class. Returns (n_classes, D) tensor."""
    centroids = []
    for c in range(n_classes):
        mask = labels == c
        centroids.append(features[mask].mean(0))
    return torch.stack(centroids)   # (K, D)


def prototype_distances(features, labels, centroids, n_classes):
    """
    For each sample return:
      min_wrong_dist : L2 to nearest WRONG-class centroid (lower = more vulnerable)
      own_dist       : L2 to own class centroid
    """
    # (N, K) pairwise L2
    diffs = features.unsqueeze(1) - centroids.unsqueeze(0)   # (N, K, D)
    dists = diffs.norm(dim=2)                                  # (N, K)

    N = features.size(0)
    min_wrong = torch.full((N,), float("inf"))
    own_d     = torch.zeros(N)

    for i in range(N):
        c = labels[i].item()
        own_d[i] = dists[i, c]
        wrong_dists = torch.cat([dists[i, :c], dists[i, c + 1:]])
        min_wrong[i] = wrong_dists.min()

    return min_wrong.numpy(), own_d.numpy()


# ---------------------------------------------------------------------------
# Per-sample attack success
# ---------------------------------------------------------------------------
def attack_success_per_sample(model, X, Y):
    """Returns boolean arrays (fgsm_succ, pgd_succ) — success = clean correct & adv flipped."""
    with torch.no_grad():
        logits_clean, _ = C.logits_and_acc(model, X, Y)
        clean_correct = (logits_clean.argmax(1).cpu() == Y.cpu()).numpy().astype(bool)

    Xfgsm = C.fgsm(model, X, Y, eps=EPS)
    Xpgd  = C.pgd(model, X, Y, eps=EPS, steps=PGD_STEPS)

    with torch.no_grad():
        logits_fgsm, _ = C.logits_and_acc(model, Xfgsm, Y)
        logits_pgd,  _ = C.logits_and_acc(model, Xpgd,  Y)
        fgsm_flip = (logits_fgsm.argmax(1).cpu() != Y.cpu()).numpy().astype(bool)
        pgd_flip  = (logits_pgd.argmax(1).cpu()  != Y.cpu()).numpy().astype(bool)

    fgsm_succ = clean_correct & fgsm_flip
    pgd_succ  = clean_correct & pgd_flip
    return fgsm_succ, pgd_succ, clean_correct


# ---------------------------------------------------------------------------
# Single seed run
# ---------------------------------------------------------------------------
def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    n_classes = meta["n_classes"]

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=None, n_eval=N_TE, seed=seed)
    Xte, Yte = Xte[:N_TE], Yte[:N_TE]

    model = C.build_model("cnn", meta, width=32, seed=seed)
    C.train_model(model, Xtr, Ytr, epochs=10)

    # Clean accuracy check
    _, acc_clean = C.logits_and_acc(model, Xte, Yte)
    print(f"  [seed={seed}] clean acc={acc_clean:.3f}")

    # --- feature extraction ---
    t0 = time.time()
    feat_tr = extract_penultimate(model, Xtr)   # (N_tr, 256)
    feat_te = extract_penultimate(model, Xte)   # (500, 256)
    print(f"  features extracted in {time.time()-t0:.1f}s  shape={feat_tr.shape}")

    # --- centroids from train ---
    centroids = compute_centroids(feat_tr, Ytr.cpu(), n_classes)

    # --- distances for test samples ---
    min_wrong_dist, own_dist = prototype_distances(feat_te, Yte.cpu(), centroids, n_classes)
    proto_margin = min_wrong_dist - own_dist   # positive = own class closer

    # --- softmax margin ---
    logits_te, _ = C.logits_and_acc(model, Xte, Yte)
    soft_margin  = C.margin_of(logits_te, Yte.cpu())

    # --- attack success ---
    fgsm_succ, pgd_succ, clean_correct = attack_success_per_sample(model, Xte, Yte)
    n_correct = clean_correct.sum()

    # Restrict metrics to correctly-classified samples
    idx = clean_correct
    mwd  = min_wrong_dist[idx]
    pm   = proto_margin[idx]
    sm   = soft_margin[idx]
    fs   = fgsm_succ[idx].astype(int)
    ps   = pgd_succ[idx].astype(int)

    def safe_auroc(scores, labels):
        if labels.sum() == 0 or labels.sum() == len(labels):
            return float("nan")
        return roc_auc_score(labels, scores)

    # Predictor: -min_wrong_dist (higher = closer to wrong centroid = more vulnerable)
    auroc_fgsm_proto = safe_auroc(-mwd, fs)
    auroc_pgd_proto  = safe_auroc(-mwd, ps)

    # Baseline predictor: -margin (lower margin = more vulnerable)
    auroc_fgsm_marg  = safe_auroc(-sm, fs)
    auroc_pgd_marg   = safe_auroc(-sm, ps)

    # Also try proto_margin directly (positive = secure)
    auroc_fgsm_pm = safe_auroc(-pm, fs)
    auroc_pgd_pm  = safe_auroc(-pm, ps)

    # Spearman correlation between min_wrong_dist and softmax margin
    rho_wrong_marg, pval_wrong = spearmanr(min_wrong_dist[idx], sm)
    rho_proto_marg, pval_proto = spearmanr(pm, sm)

    # ASR summary
    asr_fgsm = fs.mean()
    asr_pgd  = ps.mean()

    return dict(
        seed=seed,
        n_correct=int(n_correct),
        asr_fgsm=float(asr_fgsm),
        asr_pgd=float(asr_pgd),
        auroc_fgsm_proto=float(auroc_fgsm_proto),
        auroc_pgd_proto=float(auroc_pgd_proto),
        auroc_fgsm_pm=float(auroc_fgsm_pm),
        auroc_pgd_pm=float(auroc_pgd_pm),
        auroc_fgsm_marg=float(auroc_fgsm_marg),
        auroc_pgd_marg=float(auroc_pgd_marg),
        rho_wrong_vs_margin=float(rho_wrong_marg),
        pval_wrong_vs_margin=float(pval_wrong),
        rho_proto_margin_vs_softmax=float(rho_proto_marg),
        pval_proto_margin_vs_softmax=float(pval_proto),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("H195 — Prototype Distance vs Margin as Adversarial Vulnerability Predictor")
    print(f"Dataset: {DS}  |  N_te={N_TE}  |  eps={EPS}  |  seeds={SEEDS}")
    print("=" * 70)

    all_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        r = run_seed(seed)
        all_results.append(r)

        print(f"  ASR FGSM={r['asr_fgsm']:.3f}  PGD={r['asr_pgd']:.3f}")
        print(f"  AUROC (-min_wrong_dist → FGSM): {r['auroc_fgsm_proto']:.3f}")
        print(f"  AUROC (-min_wrong_dist → PGD):  {r['auroc_pgd_proto']:.3f}")
        print(f"  AUROC (-proto_margin   → FGSM): {r['auroc_fgsm_pm']:.3f}")
        print(f"  AUROC (-proto_margin   → PGD):  {r['auroc_pgd_pm']:.3f}")
        print(f"  AUROC (-margin         → FGSM): {r['auroc_fgsm_marg']:.3f}")
        print(f"  AUROC (-margin         → PGD):  {r['auroc_pgd_marg']:.3f}")
        print(f"  Spearman ρ(min_wrong_dist, margin)  = {r['rho_wrong_vs_margin']:.3f}  p={r['pval_wrong_vs_margin']:.2e}")
        print(f"  Spearman ρ(proto_margin, soft_marg) = {r['rho_proto_margin_vs_softmax']:.3f}  p={r['pval_proto_margin_vs_softmax']:.2e}")

    # Aggregate across seeds
    print("\n" + "=" * 70)
    print("AGGREGATE (mean ± std across seeds)")
    print("=" * 70)

    keys = [
        "auroc_fgsm_proto", "auroc_pgd_proto",
        "auroc_fgsm_pm", "auroc_pgd_pm",
        "auroc_fgsm_marg", "auroc_pgd_marg",
        "rho_wrong_vs_margin", "rho_proto_margin_vs_softmax",
        "asr_fgsm", "asr_pgd",
    ]
    for k in keys:
        vals = np.array([r[k] for r in all_results])
        print(f"  {k:<42s}: {vals.mean():.3f} ± {vals.std():.3f}")

    # Verdict
    proto_fgsm = np.mean([r["auroc_fgsm_proto"] for r in all_results])
    marg_fgsm  = np.mean([r["auroc_fgsm_marg"]  for r in all_results])
    proto_pgd  = np.mean([r["auroc_pgd_proto"]  for r in all_results])
    marg_pgd   = np.mean([r["auroc_pgd_marg"]   for r in all_results])

    print("\nVERDICT")
    if proto_fgsm > marg_fgsm and proto_pgd > marg_pgd:
        print("  SUPPORTED — prototype distance outperforms margin on both FGSM and PGD.")
    elif proto_fgsm > marg_fgsm or proto_pgd > marg_pgd:
        print("  PARTIAL — prototype distance outperforms margin on one attack only.")
    else:
        print("  NOT SUPPORTED — softmax margin is at least as good as prototype distance.")

    delta_fgsm = proto_fgsm - marg_fgsm
    delta_pgd  = proto_pgd  - marg_pgd
    print(f"  Δ AUROC FGSM (proto − margin) = {delta_fgsm:+.3f}")
    print(f"  Δ AUROC PGD  (proto − margin) = {delta_pgd:+.3f}")


if __name__ == "__main__":
    t_start = time.time()
    main()
    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")
