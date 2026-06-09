"""
H493 - C-score and adversarial brittleness: do atypical training samples
       drive the lowest robust margins after PGD-AT?

Seed papers
-----------
  * Jiang, Mirzasoleiman, Bartlett, Tibshirani, Recht et al. (2021)
    "Characterizing Structural Regularities of Labeled Data in Overparameterized
    Models." ICML.  Defines the consistency score (C-score): for a sample (x,y),
    the fraction of independently-trained held-out classifiers that predict y
    correctly when (x,y) is *not* in their training set.  Low C-score = atypical
    / hard-to-learn / on the long tail.

  * Carlini, Erlingsson, Papernot (2019)
    "Distribution Density, Tails, and Outliers in Machine Learning: Metrics and
    Applications."  Argues tail/outlier samples disproportionately drive
    privacy, memorization, and -- crucially here -- robustness pathologies.

  * Feldman (2020)
    "Does Learning Require Memorization? A Short Tale about a Long Tail."  STOC.
    Theoretical basis for why a heavy-tailed label distribution forces the model
    to memorize rare/atypical points, plausibly producing small-margin decision
    regions around them.

Hypothesis
----------
Atypical (low-C-score) Fashion-MNIST training points become the lowest-margin
points of a PGD-adversarially trained SmallCNN.  Concretely:

    rho( -C_score , -margin_AT )  >  0.4         (Spearman, per training point)

i.e. low C <=> low robust margin.  Removing the bottom-5% C-score tail before
PGD-AT should *improve* robust accuracy (because we no longer waste capacity
memorizing the brittle tail).

Critique-driven approximation
-----------------------------
A faithful C-score needs hundreds of held-out classifiers.  At sub-scale we
approximate it with k=5 cross-fitted SmallCNNs trained with standard (non-AT)
ERM: each fold trains on 4/5 of N and we record, for every held-out training
point, P(yhat == y | trained without this fold).  We average the held-out
softmax probability on the true class across folds; high = typical, low =
atypical.  This is the standard small-scale C-score surrogate used in the
follow-up literature.

Controls / pipeline
-------------------
  (1) 5-fold cross-fit STD SmallCNN on N=6000 Fashion-MNIST training points
      -> per-sample approximate C-score in [0,1].
  (2) Train a PGD-AT SmallCNN on the full N=6000 (eps=0.1, 7-step PGD).
  (3) Per-sample final clean margin and PGD-robust margin under the AT model.
  (4) Spearman / Pearson correlation between C-score and margins.
  (5) Ablation: re-train PGD-AT excluding the bottom-5% C-score samples;
      compare clean acc + PGD-robust acc on the held-out test set.

NOTE: This file is intentionally NOT executed here -- it is queued for the
background runner.  All compute should happen on the campaign worker.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:  # pragma: no cover - scipy is in the campaign env
    spearmanr = None
    pearsonr = None


DS = "fashion_mnist"
SEED = 0
N_TRAIN = 6000
N_EVAL = 2000
K_FOLDS = 5
STD_EPOCHS = 6           # per-fold ERM training for the C-score surrogate
AT_EPOCHS = 8            # PGD-AT training epochs
EPS = 0.1                # L-inf budget (matches the fashion campaign default)
PGD_STEPS = 7            # for both AT and evaluation
BOTTOM_FRAC = 0.05       # ablation: drop bottom 5% C-score points before re-AT


# ---------------------------------------------------------------------------
# (1) k-fold cross-fit C-score surrogate
# ---------------------------------------------------------------------------
def kfold_cscore(Xtr, Ytr, meta, k=K_FOLDS, epochs=STD_EPOCHS, seed=SEED):
    """For each training point i, average held-out P(y=Y_i | x_i) across the
    folds in which i is in the validation slice.  Higher -> more typical.
    """
    n = Xtr.size(0)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    folds = [perm[f::k] for f in range(k)]            # disjoint index lists

    p_true = torch.zeros(n)                            # held-out P(true class)
    counts = torch.zeros(n)

    for f in range(k):
        val_idx = torch.tensor(folds[f], dtype=torch.long)
        tr_mask = torch.ones(n, dtype=torch.bool)
        tr_mask[val_idx] = False
        Xf, Yf = Xtr[tr_mask], Ytr[tr_mask]
        Xv, Yv = Xtr[val_idx], Ytr[val_idx]

        m = C.build_model("cnn", meta, seed=seed * 100 + f)
        C.train_model(m, Xf, Yf, epochs=epochs, opt="sgd", lr=0.05,
                      ncls=meta["n_classes"])
        m.eval()
        with torch.no_grad():
            # batched softmax on validation slice
            outs = []
            for i in range(0, Xv.size(0), 512):
                outs.append(torch.softmax(m(Xv[i:i + 512]), dim=1).cpu())
            probs = torch.cat(outs, dim=0)
        p_true[val_idx] += probs.gather(1, Yv.cpu()[:, None]).squeeze(1)
        counts[val_idx] += 1.0
        print(f"  [C-score fold {f+1}/{k}] trained on {Xf.size(0)} pts, "
              f"scored {Xv.size(0)} held-out pts")
        del m

    cscore = (p_true / counts.clamp_min(1.0)).numpy()
    return cscore


# ---------------------------------------------------------------------------
# (3) per-sample margins under a trained model
# ---------------------------------------------------------------------------
def per_sample_margins(model, X, Y, eps=EPS, steps=PGD_STEPS, batch=256):
    """Return (clean_margin, robust_margin) numpy arrays of shape (N,)."""
    model.eval()
    clean_parts, rob_parts = [], []
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        with torch.no_grad():
            clean_logits = model(xb).cpu()
        clean_parts.append(C.margin_of(clean_logits, yb))
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            rob_logits = model(xa).cpu()
        rob_parts.append(C.margin_of(rob_logits, yb))
    return np.concatenate(clean_parts), np.concatenate(rob_parts)


# ---------------------------------------------------------------------------
# (5) full ablation: PGD-AT with vs without bottom-C-score tail
# ---------------------------------------------------------------------------
def train_at_and_eval(Xtr, Ytr, Xte, Yte, meta, tag, seed=SEED):
    m = C.build_model("cnn", meta, seed=seed)
    C.train_model(m, Xtr, Ytr, epochs=AT_EPOCHS, opt="sgd", lr=0.05,
                  ncls=meta["n_classes"],
                  adv_train=True, adv_eps=EPS, adv_steps=PGD_STEPS)
    _, clean_acc = C.logits_and_acc(m, Xte, Yte)
    res = C.attack_success(m, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    # robust acc = fraction predicted correctly under PGD attack (on full set)
    # attack_success only reports flips among originally-correct, so reconstruct:
    rob_correct = int((~res["flips"] & res["correct"]).sum())
    robust_acc = rob_correct / Xte.size(0)
    print(f"  [{tag}] clean_acc={clean_acc:.3f}  pgd_asr={res['asr']:.3f}  "
          f"robust_acc={robust_acc:.3f}  (N_train={Xtr.size(0)})")
    return {"tag": tag, "clean_acc": clean_acc, "asr": res["asr"],
            "robust_acc": robust_acc, "n_train": int(Xtr.size(0))}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("H493 - C-score (atypicality) vs PGD-AT margin on Fashion-MNIST")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  N_train={N_TRAIN}  N_eval={N_EVAL}")
    print(f"k_folds={K_FOLDS}  eps={EPS}  pgd_steps={PGD_STEPS}  "
          f"std_epochs={STD_EPOCHS}  at_epochs={AT_EPOCHS}")
    print(f"bottom_frac dropped in ablation = {BOTTOM_FRAC}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # ---- (1) approximate C-score via k-fold cross-fit STD SmallCNN ----------
    print("\n[1] Cross-fitting standard SmallCNN to estimate C-score "
          f"(k={K_FOLDS}) ...")
    t0 = time.time()
    cscore = kfold_cscore(Xtr, Ytr, meta, k=K_FOLDS, epochs=STD_EPOCHS, seed=SEED)
    print(f"    C-score done in {time.time()-t0:.1f}s   "
          f"mean={cscore.mean():.3f}  med={np.median(cscore):.3f}  "
          f"min={cscore.min():.3f}  max={cscore.max():.3f}")

    # ---- (2) train PGD-AT SmallCNN on the full N ---------------------------
    print(f"\n[2] Training PGD-AT SmallCNN on full N={N_TRAIN} ...")
    t0 = time.time()
    m_at = C.build_model("cnn", meta, seed=SEED)
    C.train_model(m_at, Xtr, Ytr, epochs=AT_EPOCHS, opt="sgd", lr=0.05,
                  ncls=meta["n_classes"],
                  adv_train=True, adv_eps=EPS, adv_steps=PGD_STEPS)
    _, at_clean = C.logits_and_acc(m_at, Xte, Yte)
    at_eval = C.attack_success(m_at, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    at_robust = float((~at_eval["flips"] & at_eval["correct"]).sum()) / Xte.size(0)
    print(f"    PGD-AT trained in {time.time()-t0:.1f}s  "
          f"clean_acc={at_clean:.3f}  pgd_asr={at_eval['asr']:.3f}  "
          f"robust_acc={at_robust:.3f}")

    # ---- (3) per-sample margins on the *training* set under AT model -------
    print("\n[3] Per-sample clean + robust margins on training set ...")
    t0 = time.time()
    clean_marg, robust_marg = per_sample_margins(m_at, Xtr, Ytr,
                                                 eps=EPS, steps=PGD_STEPS)
    print(f"    margins computed in {time.time()-t0:.1f}s   "
          f"clean: mean={clean_marg.mean():.2f}  "
          f"robust: mean={robust_marg.mean():.2f}")

    # ---- (4) correlation: C-score vs margin --------------------------------
    print("\n[4] Correlations (per-sample, on train) ...")
    def _corr(a, b):
        if spearmanr is None:
            return float("nan"), float("nan")
        sp = spearmanr(a, b).correlation
        pe = pearsonr(a, b)[0]
        return float(sp), float(pe)

    sp_clean, pe_clean = _corr(cscore, clean_marg)
    sp_rob, pe_rob = _corr(cscore, robust_marg)
    # also: corr of (-C) with (-margin) is the same as corr of C with margin,
    # but we also report on the AT robust margin's per-class-deciled tail.
    print(f"    Spearman( C , clean_margin  ) = {sp_clean:+.3f}   "
          f"Pearson = {pe_clean:+.3f}")
    print(f"    Spearman( C , robust_margin ) = {sp_rob:+.3f}   "
          f"Pearson = {pe_rob:+.3f}")
    hypothesis_supported = sp_rob is not None and sp_rob > 0.4
    print(f"    Hypothesis threshold rho>0.4 on robust margin "
          f"=> {'SUPPORTED' if hypothesis_supported else 'NOT supported'}")

    # bottom-vs-top decile robust margin gap (effect-size sanity check)
    q_lo = np.quantile(cscore, 0.10)
    q_hi = np.quantile(cscore, 0.90)
    bot_marg = robust_marg[cscore <= q_lo].mean()
    top_marg = robust_marg[cscore >= q_hi].mean()
    print(f"    robust margin  bottom-10% C: {bot_marg:+.2f}   "
          f"top-10% C: {top_marg:+.2f}   gap={top_marg-bot_marg:+.2f}")

    # ---- (5) ablation: drop bottom-5% C-score, re-train PGD-AT -------------
    print(f"\n[5] Ablation: re-train PGD-AT excluding bottom {BOTTOM_FRAC:.0%} "
          "C-score samples ...")
    cutoff = np.quantile(cscore, BOTTOM_FRAC)
    keep = cscore > cutoff
    Xtr2 = Xtr[torch.from_numpy(keep).to(Xtr.device)]
    Ytr2 = Ytr[torch.from_numpy(keep).to(Ytr.device)]
    ab_full = train_at_and_eval(Xtr, Ytr, Xte, Yte, meta, "AT_full",   seed=SEED)
    ab_drop = train_at_and_eval(Xtr2, Ytr2, Xte, Yte, meta, "AT_drop_bot5", seed=SEED)
    delta_clean  = ab_drop["clean_acc"]  - ab_full["clean_acc"]
    delta_robust = ab_drop["robust_acc"] - ab_full["robust_acc"]

    # ---- HEADLINE ----------------------------------------------------------
    print("\n" + "=" * 74)
    print("HEADLINE verdict")
    print("=" * 74)
    print(f"  Spearman( C-score , robust margin ) on train = {sp_rob:+.3f}")
    print(f"  bottom-10% vs top-10% C robust-margin gap    = {top_marg-bot_marg:+.2f}")
    print(f"  AT_full     : clean={ab_full['clean_acc']:.3f}  "
          f"robust={ab_full['robust_acc']:.3f}")
    print(f"  AT_drop_bot5: clean={ab_drop['clean_acc']:.3f}  "
          f"robust={ab_drop['robust_acc']:.3f}")
    print(f"  delta(robust_acc) from dropping atypical tail = {delta_robust:+.3f}")
    if hypothesis_supported and delta_robust > 0.0:
        verdict = ("SUPPORTED: atypical (low-C) samples are the lowest-margin "
                   "points under PGD-AT and dropping them improves robust acc.")
    elif hypothesis_supported and delta_robust <= 0.0:
        verdict = ("PARTIAL: C-score correlates with robust margin as predicted, "
                   "but removing the atypical tail does NOT raise robust acc "
                   "(tail samples may still be needed for generalization).")
    elif (not hypothesis_supported) and delta_robust > 0.02:
        verdict = ("PARTIAL: weak C/margin correlation but dropping atypicals "
                   "still helps -- another atypicality proxy (or memorization "
                   "channel) may be the real culprit.")
    else:
        verdict = ("REFUTED: C-score does not track robust margin and removing "
                   "the atypical tail does not improve robust accuracy. "
                   "Atypicality (Feldman/Carlini long-tail) is not the dominant "
                   "driver of PGD-AT brittleness on Fashion-MNIST.")
    print(f"  VERDICT: {verdict}")
    print("=" * 74)
    print("Refs: Jiang et al. 2021 (C-score); Carlini, Erlingsson, Papernot 2019")
    print("      (Distribution Density, Tails, and Outliers); Feldman 2020")
    print("      (Does Learning Require Memorization? A Short Tale about a Long Tail).")
    print("=" * 74)


if __name__ == "__main__":
    main()
