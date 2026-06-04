"""
H199 - Cosine similarity between clean input gradient and adversarial
       perturbation predicts attack success and transferability.

High cosine alignment between the clean gradient and the actual PGD
perturbation direction implies the sample lies near a locally linear
decision boundary, making it easier to fool and more transferable.

We train two independent CNNs (seeds 0, 1) on Fashion-MNIST and measure:

  (1) AUROC: cosine_sim(grad_A_flat, pgd_delta_flat) → PGD success on model_A
  (2) AUROC: cosine_sim(grad_A_flat, fgsm_delta_A_flat) → transfer success on model_B
  (3) AUROC: neg-margin (model_A) → transfer success on model_B  [baseline]
  (4) Mean cosine similarity for successful vs failed transfers (FGSM)

Notes:
  - FGSM delta is sign(grad_A), so cosine_sim(grad_A, sign(grad_A)) = 1 for
    non-zero gradients.  We therefore use the *raw* grad (not sign) vs FGSM
    delta to get meaningful variance across samples.
  - For PGD the accumulated perturbation can differ from the clean gradient,
    so cosine similarity is a genuine predictor there.

Evaluation subset: Xte[:300].
"""

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS         = "fashion_mnist"
N_EVAL     = 300
EPS        = 0.1
PGD_STEPS  = 10
SEEDS      = [0, 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def input_grad(model, X, Y):
    """Return per-sample input gradients, shape (N, 1, 28, 28)."""
    model.eval()
    X_req = X.clone().requires_grad_(True)
    logits = model(X_req)
    loss = F.cross_entropy(logits, Y)
    loss.backward()
    return X_req.grad.detach().clone()


def pgd_attack(model, X, Y, eps, steps, alpha=None):
    """Return adversarial examples via PGD (L-inf)."""
    if alpha is None:
        alpha = eps / steps * 2
    model.eval()
    Xadv = X.clone().detach()
    Xadv = Xadv + torch.empty_like(Xadv).uniform_(-eps, eps)
    Xadv = torch.clamp(Xadv, 0., 1.)
    for _ in range(steps):
        Xadv.requires_grad_(True)
        loss = F.cross_entropy(model(Xadv), Y)
        loss.backward()
        with torch.no_grad():
            Xadv = Xadv + alpha * Xadv.grad.sign()
            Xadv = torch.max(torch.min(Xadv, X + eps), X - eps)
            Xadv = torch.clamp(Xadv, 0., 1.)
    return Xadv.detach()


def fgsm_attack(model, X, Y, eps):
    """Return FGSM adversarial examples."""
    model.eval()
    X_req = X.clone().requires_grad_(True)
    loss = F.cross_entropy(model(X_req), Y)
    loss.backward()
    return (X + eps * X_req.grad.detach().sign()).clamp(0., 1.)


def batch_cosine_sim(a, b):
    """
    Per-sample cosine similarity.
    a, b: (N, D) numpy arrays.
    Returns (N,) array.
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return np.sum((a / a_norm) * (b / b_norm), axis=1)


def safe_auroc(scores, labels):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return roc_auc_score(labels, scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 74)
    print("H199 - Gradient cosine similarity predicts attack success &")
    print("       adversarial transferability on Fashion-MNIST")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  N_eval={N_EVAL}  eps={EPS}  "
          f"pgd_steps={PGD_STEPS}")

    t0 = time.time()
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]
    print(f"Dataset loaded. Xte subset: {Xte.shape}")

    # -----------------------------------------------------------------------
    # Train two independent CNNs
    # -----------------------------------------------------------------------
    models = []
    for s in SEEDS:
        C.set_seed(s)
        m = C.build_model("cnn", meta, seed=s)
        C.train_model(m, Xtr, Ytr, epochs=10)
        m.eval()
        models.append(m)
        print(f"  Model seed={s} trained.")

    model_A, model_B = models[0], models[1]

    # -----------------------------------------------------------------------
    # Clean accuracy
    # -----------------------------------------------------------------------
    logits_A, acc_A = C.logits_and_acc(model_A, Xte, Yte)
    logits_B, acc_B = C.logits_and_acc(model_B, Xte, Yte)
    print(f"\nClean accuracy — model_A: {acc_A:.3f}   model_B: {acc_B:.3f}")

    # -----------------------------------------------------------------------
    # Clean input gradient on model_A
    # -----------------------------------------------------------------------
    print("\nComputing clean input gradients (model_A) ...")
    grad_A = input_grad(model_A, Xte, Yte)          # (N,1,28,28)
    grad_A_flat = grad_A.cpu().numpy().reshape(N_EVAL, -1)   # (N, 784)

    # -----------------------------------------------------------------------
    # FGSM on model_A
    # -----------------------------------------------------------------------
    print("Running FGSM on model_A ...")
    Xfgsm_A = fgsm_attack(model_A, Xte, Yte, EPS)
    delta_fgsm_flat = (Xfgsm_A - Xte).cpu().numpy().reshape(N_EVAL, -1)

    with torch.no_grad():
        pred_clean_A = model_A(Xte).argmax(1).cpu().numpy()
        pred_fgsm_A  = model_A(Xfgsm_A).argmax(1).cpu().numpy()
        pred_fgsm_B  = model_B(Xfgsm_A).argmax(1).cpu().numpy()

    Yte_np = Yte.cpu().numpy()
    correct_A = (pred_clean_A == Yte_np)

    # Attack success on model_A (FGSM)
    fgsm_success_A = (correct_A & (pred_fgsm_A != Yte_np)).astype(int)
    print(f"  FGSM success rate on model_A: {fgsm_success_A.mean():.3f}  "
          f"({fgsm_success_A.sum()}/{N_EVAL})")

    # Transfer success: model_A FGSM fools model_B (from initially correct samples)
    with torch.no_grad():
        pred_clean_B = model_B(Xte).argmax(1).cpu().numpy()
    correct_B  = (pred_clean_B == Yte_np)
    # Transfer success: both models correct on clean, B flipped by A's FGSM
    transfer_success = (correct_A & correct_B & (pred_fgsm_B != Yte_np)).astype(int)
    print(f"  Transfer success (A→B, FGSM): {transfer_success.mean():.3f}  "
          f"({transfer_success.sum()}/{N_EVAL})")

    # -----------------------------------------------------------------------
    # PGD on model_A
    # -----------------------------------------------------------------------
    print("Running PGD on model_A ...")
    Xpgd_A = pgd_attack(model_A, Xte, Yte, EPS, PGD_STEPS)
    delta_pgd_flat = (Xpgd_A - Xte).cpu().numpy().reshape(N_EVAL, -1)

    with torch.no_grad():
        pred_pgd_A = model_A(Xpgd_A).argmax(1).cpu().numpy()

    pgd_success_A = (correct_A & (pred_pgd_A != Yte_np)).astype(int)
    print(f"  PGD success rate on model_A:  {pgd_success_A.mean():.3f}  "
          f"({pgd_success_A.sum()}/{N_EVAL})")

    # -----------------------------------------------------------------------
    # Cosine similarities
    # -----------------------------------------------------------------------
    print("\nComputing cosine similarities ...")
    cos_grad_pgd  = batch_cosine_sim(grad_A_flat, delta_pgd_flat)   # grad vs PGD delta
    cos_grad_fgsm = batch_cosine_sim(grad_A_flat, delta_fgsm_flat)  # grad vs FGSM delta

    # Margin of model_A (negate for AUROC: higher neg-margin = more vulnerable)
    margin_A = C.margin_of(logits_A, Yte)
    neg_margin_A = -margin_A

    # -----------------------------------------------------------------------
    # AUROC results
    # -----------------------------------------------------------------------
    auroc_cos_pgd_vs_pgd_success  = safe_auroc(cos_grad_pgd,  pgd_success_A)
    auroc_cos_fgsm_vs_transfer    = safe_auroc(cos_grad_fgsm, transfer_success)
    auroc_margin_vs_transfer      = safe_auroc(neg_margin_A,  transfer_success)
    auroc_cos_pgd_vs_transfer     = safe_auroc(cos_grad_pgd,  transfer_success)

    # -----------------------------------------------------------------------
    # Mean cosine sim for successful vs failed transfers
    # -----------------------------------------------------------------------
    cos_transfer1 = cos_grad_fgsm[transfer_success == 1]
    cos_transfer0 = cos_grad_fgsm[transfer_success == 0]
    mean_cos_transfer1 = cos_transfer1.mean() if len(cos_transfer1) > 0 else float("nan")
    mean_cos_transfer0 = cos_transfer0.mean() if len(cos_transfer0) > 0 else float("nan")

    mw_stat, mw_p = (float("nan"), float("nan"))
    if len(cos_transfer1) > 0 and len(cos_transfer0) > 0:
        mw_stat, mw_p = mannwhitneyu(cos_transfer1, cos_transfer0, alternative="greater")

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("AUROC RESULTS (higher = better predictor)")
    print("=" * 74)
    print(f"{'Predictor':<48} {'AUROC':>8}")
    print("-" * 57)
    print(f"{'(1) cos(grad_A, PGD_delta) → PGD success (model_A)':<48} "
          f"{auroc_cos_pgd_vs_pgd_success:>8.4f}")
    print(f"{'(2) cos(grad_A, FGSM_delta) → transfer success (A→B)':<48} "
          f"{auroc_cos_fgsm_vs_transfer:>8.4f}")
    print(f"{'(3) neg-margin(model_A) → transfer success [baseline]':<48} "
          f"{auroc_margin_vs_transfer:>8.4f}")
    print(f"{'(4) cos(grad_A, PGD_delta) → transfer success (A→B)':<48} "
          f"{auroc_cos_pgd_vs_transfer:>8.4f}")

    print("\n" + "=" * 74)
    print("MEAN COSINE SIM: grad_A vs FGSM_delta (by transfer outcome)")
    print("=" * 74)
    print(f"  Successful transfer (n={len(cos_transfer1):3d}): "
          f"mean cos = {mean_cos_transfer1:.4f}")
    print(f"  Failed transfer    (n={len(cos_transfer0):3d}): "
          f"mean cos = {mean_cos_transfer0:.4f}")
    print(f"  Mann-Whitney U (greater): stat={mw_stat:.1f}  p={mw_p:.4e}")

    print("\n" + "=" * 74)
    print("INTERPRETATION")
    print("=" * 74)
    # Assess support
    h_support_pgd  = auroc_cos_pgd_vs_pgd_success > 0.55
    h_support_xfer = auroc_cos_fgsm_vs_transfer   > 0.55
    cos_beats_margin = auroc_cos_fgsm_vs_transfer  > auroc_margin_vs_transfer

    if h_support_pgd and h_support_xfer:
        verdict = "SUPPORTED"
        explanation = (
            "Cosine alignment between the clean gradient and the adversarial "
            "perturbation direction is a meaningful predictor of both PGD "
            "success and adversarial transferability, consistent with the "
            "locally-linear boundary hypothesis."
        )
    elif h_support_pgd or h_support_xfer:
        verdict = "PARTIALLY SUPPORTED"
        explanation = (
            "Cosine alignment predicts one of the two outcomes (PGD success "
            "or transfer) but not both."
        )
    else:
        verdict = "NOT SUPPORTED"
        explanation = (
            "Cosine alignment between the clean gradient and adversarial "
            "perturbation direction does not reliably predict attack success "
            "or transfer on this dataset/architecture."
        )

    print(f"  Verdict: {verdict}")
    print(f"  {explanation}")
    if cos_beats_margin:
        print("  Cosine similarity outperforms neg-margin as a transfer predictor.")
    else:
        print("  neg-margin (baseline) is at least as good as cosine similarity")
        print("  for predicting transfer success.")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
