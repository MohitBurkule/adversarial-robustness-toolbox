"""
H189 - Ensemble disagreement predicts per-sample adversarial vulnerability
        better than single-model margin.

Motivated by SED paper (arXiv:2409.16797): samples where ensemble members
disagree lie near decision boundaries and should be more vulnerable to attacks.

We train 3 independent CNNs (seeds 0,1,2) on Fashion-MNIST and compare four
scores as predictors of whether FGSM / PGD will successfully fool the model:

  (A) Single-model margin: correct-class logit minus max-other logit (model0).
      Higher margin = more confident = harder to fool → negate for AUROC.
  (B) Ensemble disagreement (entropy): entropy of mean softmax across 3 models.
      Higher entropy = more disagreement = more vulnerable.
  (C) Ensemble disagreement (variance): mean variance of per-class softmax probs
      across 3 models. Higher = more disagreement.
  (D) Ensemble mean margin: mean of margin scores from all 3 models (negated).

Attack success is assessed on model0.  Evaluation subset: Xte[:300].
"""

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS         = "fashion_mnist"
N_EVAL     = 300
EPS        = 0.1
PGD_STEPS  = 10
SEEDS      = [0, 1, 2]


@torch.no_grad()
def softmax_probs(model, X, batch=256):
    """Return (N, n_classes) numpy array of softmax probabilities."""
    parts = []
    for i in range(0, X.size(0), batch):
        parts.append(F.softmax(model(X[i:i+batch]), dim=1).cpu())
    return torch.cat(parts).numpy()


def attack_labels(model, X, Y, attack="fgsm"):
    """Return binary label array: 1 = attack succeeded on a correct sample, 0 = not."""
    model.eval()
    # clean correctness
    with torch.no_grad():
        clean_pred = model(X).argmax(1).cpu()
    correct_mask = (clean_pred == Y.cpu()).numpy().astype(bool)

    if attack == "fgsm":
        Xa = C.fgsm(model, X, Y, EPS)
    else:
        Xa = C.pgd(model, X, Y, EPS, PGD_STEPS)

    with torch.no_grad():
        adv_pred = model(Xa).argmax(1).cpu()
    flipped = (adv_pred != Y.cpu()).numpy().astype(bool)

    # attack success = was correct AND got flipped
    return (correct_mask & flipped).astype(int), correct_mask


def main():
    print("=" * 74)
    print("H189 - Ensemble disagreement vs single-model margin for predicting")
    print("       per-sample adversarial vulnerability (FGSM & PGD)")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  N_eval={N_EVAL}  eps={EPS}")

    t0 = time.time()
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]
    print(f"Dataset loaded. Xte subset: {Xte.shape}")

    # -----------------------------------------------------------------------
    # Train 3 independent CNNs
    # -----------------------------------------------------------------------
    models = []
    for s in SEEDS:
        C.set_seed(s)
        m = C.build_model("cnn", meta, seed=s)
        C.train_model(m, Xtr, Ytr, epochs=10)
        m.eval()
        models.append(m)
        print(f"  Model seed={s} trained.")

    model0 = models[0]

    # -----------------------------------------------------------------------
    # Per-sample softmax probabilities from each model  →  ensemble scores
    # -----------------------------------------------------------------------
    probs = [softmax_probs(m, Xte) for m in models]   # list of (N, C) arrays
    probs_stack = np.stack(probs, axis=0)              # (3, N, C)

    mean_probs = probs_stack.mean(axis=0)              # (N, C)

    # (B) Entropy of mean softmax (higher = more disagreement)
    eps_clip = 1e-9
    ent_mean = -np.sum(mean_probs * np.log(mean_probs + eps_clip), axis=1)  # (N,)

    # (C) Mean variance across models per sample (average over classes)
    var_probs = probs_stack.var(axis=0).mean(axis=1)   # (N,)

    # (A) Single-model margin from model0 (negate: higher margin = less vulnerable)
    logits0, acc0 = C.logits_and_acc(model0, Xte, Yte)
    margin0 = C.margin_of(logits0, Yte)                # (N,) higher = more confident
    neg_margin0 = -margin0                             # higher = more vulnerable

    # (D) Ensemble mean margin (negated)
    all_margins = []
    for m in models:
        lg, _ = C.logits_and_acc(m, Xte, Yte)
        all_margins.append(C.margin_of(lg, Yte))
    neg_mean_margin = -np.stack(all_margins, axis=0).mean(axis=0)  # (N,)

    print(f"\nClean accuracy (model0): {acc0:.3f}")

    # -----------------------------------------------------------------------
    # Attack labels (model0)
    # -----------------------------------------------------------------------
    print("\nRunning FGSM attacks on model0 ...")
    fgsm_labels, correct_mask = attack_labels(model0, Xte, Yte, attack="fgsm")
    print(f"  FGSM success rate (of all eval): {fgsm_labels.mean():.3f}  "
          f"({fgsm_labels.sum()}/{N_EVAL})")

    print("Running PGD attacks on model0 ...")
    pgd_labels, _ = attack_labels(model0, Xte, Yte, attack="pgd")
    print(f"  PGD  success rate (of all eval): {pgd_labels.mean():.3f}  "
          f"({pgd_labels.sum()}/{N_EVAL})")

    # -----------------------------------------------------------------------
    # AUROC computation (use full N_EVAL labels — 0s for initially wrong samples
    # are fine since an already-wrong sample always has label=0)
    # -----------------------------------------------------------------------
    def safe_auroc(scores, labels):
        if labels.sum() == 0 or labels.sum() == len(labels):
            return float("nan")
        return roc_auc_score(labels, scores)

    results = {}
    for attack_name, labels in [("FGSM", fgsm_labels), ("PGD", pgd_labels)]:
        results[attack_name] = {
            "auroc_margin":       safe_auroc(neg_margin0,    labels),
            "auroc_ent_ensemble": safe_auroc(ent_mean,       labels),
            "auroc_var_ensemble": safe_auroc(var_probs,      labels),
            "auroc_mean_margin":  safe_auroc(neg_mean_margin, labels),
        }

    # Spearman correlation between ensemble entropy and single-model margin (negated)
    rho_fgsm, p_fgsm = spearmanr(ent_mean, neg_margin0)
    rho_pgd,  p_pgd  = spearmanr(ent_mean, neg_margin0)  # same x/y

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("AUROC RESULTS (higher = better predictor of attack success)")
    print("=" * 74)
    header = f"{'Score':<35} {'FGSM AUROC':>12} {'PGD AUROC':>12}"
    print(header)
    print("-" * 60)
    rows_data = [
        ("(A) Single-model margin (neg)",   "auroc_margin"),
        ("(B) Ensemble entropy (mean probs)","auroc_ent_ensemble"),
        ("(C) Ensemble prob variance",       "auroc_var_ensemble"),
        ("(D) Ensemble mean margin (neg)",   "auroc_mean_margin"),
    ]
    for label, key in rows_data:
        fval = results["FGSM"][key]
        pval = results["PGD"][key]
        fs = f"{fval:.4f}" if fval == fval else "  nan "
        ps = f"{pval:.4f}" if pval == pval else "  nan "
        print(f"{label:<35} {fs:>12} {ps:>12}")

    print("\n" + "=" * 74)
    print("SPEARMAN CORRELATION: ensemble entropy vs neg-single-margin")
    print(f"  rho = {rho_fgsm:.4f}   p = {p_fgsm:.4e}")
    print("  (positive rho => high disagreement samples also have low margin)")
    print("=" * 74)

    print("\nINTERPRETATION:")
    fgsm_ens_better = results["FGSM"]["auroc_ent_ensemble"] > results["FGSM"]["auroc_margin"]
    pgd_ens_better  = results["PGD"]["auroc_ent_ensemble"]  > results["PGD"]["auroc_margin"]
    if fgsm_ens_better and pgd_ens_better:
        print("  SUPPORT: Ensemble disagreement (entropy) outperforms single-model")
        print("  margin on both FGSM and PGD, consistent with SED (2409.16797).")
    elif fgsm_ens_better or pgd_ens_better:
        print("  PARTIAL SUPPORT: Ensemble entropy beats single-model margin on one")
        print("  attack but not both.")
    else:
        print("  NOT SUPPORTED: Single-model margin predicts attack success at least")
        print("  as well as ensemble disagreement — boundary proximity captured by")
        print("  the margin of a single model already.")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
