"""
H208 - Cross-architecture vulnerability consistency.

Hypothesis: the vulnerability ranking of test samples is highly consistent
across different architectures (CNN-small, CNN-wide, MLP), confirming that
vulnerability is data-geometric rather than architecture-specific.

We test:
  1. Spearman rho of per-sample logit margins across all architecture pairs.
  2. Kendall's tau of per-sample margin RANKS across all pairs.
  3. Cross-architecture AUROC: does model-A's margin predict model-B's FGSM
     attack success?  (diagonal = own AUROC, off-diagonal = cross AUROC)

Expected finding (data-geometric view): rho/tau >> 0 and cross-AUROC close
to own-AUROC, meaning architectures mostly agree about which samples are
hard/easy to attack.
"""
import os, sys, time
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS         = "fashion_mnist"
N_EVAL     = 300        # Xte[:300]
EPS        = 0.1        # L-inf FGSM budget (matches campaign default)
EPOCHS     = 10


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def compute_margins(model, X, Y):
    """Signed logit margin: correct-class logit minus max-other-class logit."""
    model.eval()
    margins = []
    batch = 256
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            xb = X[i:i+batch]
            yb = Y[i:i+batch]
            logits = model(xb)                         # (B, C)
            correct_logit = logits[range(len(yb)), yb] # (B,)
            # zero out correct class to find max-other
            logits_copy = logits.clone()
            logits_copy[range(len(yb)), yb] = float('-inf')
            max_other = logits_copy.max(dim=1).values  # (B,)
            margins.append((correct_logit - max_other).cpu().numpy())
    return np.concatenate(margins)                      # (N,)


def compute_fgsm_success(model, X, Y, eps=EPS):
    """
    Returns a binary array: 1 if FGSM fooled the model (orig correct, adv wrong).
    Samples already misclassified clean are excluded (NaN → 0 for simplicity;
    we mark originally wrong samples as not-attacked=0).
    """
    model.eval()
    batch = 128
    success = []
    for i in range(0, X.size(0), batch):
        xb = X[i:i+batch]
        yb = Y[i:i+batch]
        with torch.no_grad():
            clean_pred = model(xb).argmax(1)
        # FGSM
        xb_adv = C.fgsm(model, xb, yb, eps)
        with torch.no_grad():
            adv_pred = model(xb_adv).argmax(1)
        # success = originally correct AND adversarial wrong
        orig_correct = (clean_pred.cpu() == yb.cpu())
        adv_wrong    = (adv_pred.cpu()   != yb.cpu())
        s = (orig_correct & adv_wrong).float().numpy()
        success.append(s)
    return np.concatenate(success)                      # (N,) binary


def safe_auroc(labels, scores):
    """AUROC; returns NaN if only one class present."""
    if len(np.unique(labels)) < 2:
        return float('nan')
    return roc_auc_score(labels, -scores)   # lower margin → higher attack prob


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 74)
    print("H208 - Cross-architecture vulnerability consistency")
    print("=" * 74)

    t0 = time.time()
    C.set_seed(0)
    meta = C.dataset_meta(DS)

    print(f"\nLoading {DS} ...")
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    Xte = Xte[:N_EVAL]
    Yte = Yte[:N_EVAL]
    print(f"  train={Xtr.shape[0]}  eval={Xte.shape[0]}")

    # -----------------------------------------------------------------------
    # Build & train three architectures
    # -----------------------------------------------------------------------
    arch_defs = [
        ("CNN-small", "cnn",  {"width": 16, "seed": 0}),
        ("CNN-wide",  "cnn",  {"width": 64, "seed": 0}),
        ("MLP",       "mlp",  {"width": 256}),
    ]

    results = {}   # arch_name -> {"model", "margins", "fgsm_success", "clean_acc", "asr"}

    for name, arch, kw in arch_defs:
        print(f"\n--- {name} ({arch}, {kw}) ---")
        C.set_seed(kw.get("seed", 0))
        model = C.build_model(arch, meta, **kw)
        C.train_model(model, Xtr, Ytr, epochs=EPOCHS)

        # clean accuracy
        _, clean_acc = C.logits_and_acc(model, Xte, Yte)
        print(f"  clean acc = {clean_acc:.4f}")

        # per-sample margins
        margins = compute_margins(model, Xte, Yte)

        # FGSM success
        fgsm_succ = compute_fgsm_success(model, Xte, Yte)
        asr = fgsm_succ.mean()
        print(f"  FGSM ASR  = {asr:.4f}")

        results[name] = {
            "model": model,
            "margins": margins,
            "fgsm_success": fgsm_succ,
            "clean_acc": clean_acc,
            "asr": asr,
        }

    arch_names = list(results.keys())
    n = len(arch_names)

    # -----------------------------------------------------------------------
    # Pairwise Spearman rho of margins
    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("Pairwise Spearman rho of per-sample margins")
    print("=" * 74)
    rho_mat  = np.full((n, n), np.nan)
    tau_mat  = np.full((n, n), np.nan)

    col_w = max(len(a) for a in arch_names) + 2
    header = " " * col_w + "".join(f"{a:>{col_w}}" for a in arch_names)
    print(header)
    for i, ai in enumerate(arch_names):
        row_rho = f"{ai:<{col_w}}"
        row_tau = f"{ai:<{col_w}}"
        for j, aj in enumerate(arch_names):
            mi = results[ai]["margins"]
            mj = results[aj]["margins"]
            rho, _ = spearmanr(mi, mj)
            tau, _ = kendalltau(mi, mj)
            rho_mat[i, j] = rho
            tau_mat[i, j] = tau
            row_rho += f"{rho:>{col_w}.4f}"
            row_tau += f"{tau:>{col_w}.4f}"
        print(row_rho)

    print("\nKendall tau of margin ranks:")
    print(header)
    for i, ai in enumerate(arch_names):
        row = f"{ai:<{col_w}}"
        for j in range(n):
            row += f"{tau_mat[i,j]:>{col_w}.4f}"
        print(row)

    # -----------------------------------------------------------------------
    # Cross-architecture AUROC matrix
    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("Cross-architecture AUROC: rows=score source, cols=label source")
    print("  cell(i,j) = AUROC of model-i margin predicting model-j FGSM success")
    print("  diagonal  = own AUROC (self-prediction baseline)")
    print("=" * 74)

    auroc_mat = np.full((n, n), np.nan)
    print(header)
    for i, ai in enumerate(arch_names):
        row = f"{ai:<{col_w}}"
        for j, aj in enumerate(arch_names):
            labels = results[aj]["fgsm_success"]
            scores = results[ai]["margins"]
            auc = safe_auroc(labels, scores)
            auroc_mat[i, j] = auc
            row += f"{auc:>{col_w}.4f}"
        print(row)

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    # off-diagonal rho/tau
    off_rho  = [rho_mat[i,j] for i in range(n) for j in range(n) if i != j]
    off_tau  = [tau_mat[i,j] for i in range(n) for j in range(n) if i != j]
    diag_auc = [auroc_mat[i,i] for i in range(n) if not np.isnan(auroc_mat[i,i])]
    off_auc  = [auroc_mat[i,j] for i in range(n) for j in range(n)
                if i != j and not np.isnan(auroc_mat[i,j])]

    mean_off_rho = np.mean(off_rho)
    mean_off_tau = np.mean(off_tau)
    mean_diag_auc = np.mean(diag_auc)
    mean_off_auc  = np.mean(off_auc) if off_auc else float('nan')

    print("\n" + "=" * 74)
    print("Summary")
    print("=" * 74)
    print(f"  Mean off-diagonal Spearman rho  : {mean_off_rho:.4f}")
    print(f"  Mean off-diagonal Kendall tau   : {mean_off_tau:.4f}")
    print(f"  Mean own (diagonal) AUROC       : {mean_diag_auc:.4f}")
    print(f"  Mean cross (off-diag) AUROC     : {mean_off_auc:.4f}")
    print(f"  Cross-AUROC vs own-AUROC gap    : {mean_off_auc - mean_diag_auc:+.4f}")
    print(f"\n  Total runtime: {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("Interpretation")
    print("=" * 74)
    if mean_off_rho > 0.5:
        print("  HIGH cross-architecture margin correlation (rho > 0.5): vulnerability")
        print("  ordering is largely shared across architectures, consistent with a")
        print("  DATA-GEOMETRIC explanation — certain input-space regions are close to")
        print("  decision boundaries regardless of the specific classifier.")
    elif mean_off_rho > 0.2:
        print("  MODERATE cross-architecture correlation (0.2 < rho <= 0.5): partial")
        print("  data-geometric signal, but architecture-specific geometry also matters.")
    else:
        print("  LOW cross-architecture correlation (rho <= 0.2): vulnerability is")
        print("  largely architecture-specific, not data-geometric.")

    if abs(mean_off_auc - mean_diag_auc) < 0.05:
        print("  Cross-AUROC is close to own-AUROC (gap < 0.05): a model's margin is")
        print("  nearly as predictive of ANOTHER model's attack success as of its own,")
        print("  further supporting the data-geometric view.")
    else:
        print(f"  Cross-AUROC gap = {mean_off_auc - mean_diag_auc:+.4f}: meaningful drop in")
        print("  predictability across architectures — some architecture-specificity.")
    print("=" * 74)


if __name__ == "__main__":
    main()
