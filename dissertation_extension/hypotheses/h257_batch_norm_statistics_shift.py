"""
H257 — Batch Norm Statistics Shift vs Adversarial Vulnerability
Hypothesis: The shift in batch normalisation running statistics (mean activations)
between clean and adversarial inputs correlates with per-sample vulnerability.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0; EPS = 0.1; N_EVAL = 300
META = {"channels": 1, "size": 28, "n_classes": 10}
torch.manual_seed(SEED); np.random.seed(SEED)

Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = C.build_model("cnn", META, width=32, seed=SEED)
C.train_model(model, Xtr, Ytr, epochs=10)
model.eval()


def collect_bn_activations(mdl: nn.Module, X: torch.Tensor, batch: int = 64):
    """
    Run X through mdl and collect post-BN activations for every BatchNorm2d layer.
    Returns a dict: {layer_name: (N, C) mean activation per sample per channel}.
    """
    hooks = []
    store = {}

    for name, module in mdl.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            def make_hook(n):
                def hook_fn(mod, inp, out):
                    # out: (B, C, H, W) — take spatial mean per sample per channel
                    store.setdefault(n, []).append(out.detach().mean(dim=(2, 3)).cpu())
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for start in range(0, len(X), batch):
            xb = X[start:start + batch]
            _ = mdl(xb)

    for h in hooks:
        h.remove()

    # Concatenate batches: {name: (N, C) tensor}
    return {k: torch.cat(v, dim=0) for k, v in store.items()}


def per_sample_bn_shift(clean_acts: dict, adv_acts: dict) -> np.ndarray:
    """
    Mean absolute deviation in BN activations between clean and adversarial,
    averaged across all BN layers and channels. Returns (N,) array.
    """
    shifts = []
    for name in clean_acts:
        if name not in adv_acts:
            continue
        diff = (clean_acts[name] - adv_acts[name]).abs()  # (N, C)
        shifts.append(diff.mean(dim=1))  # (N,)
    if not shifts:
        return np.zeros(len(next(iter(clean_acts.values()))))
    return torch.stack(shifts, dim=0).mean(dim=0).numpy()  # (N,)


def main():
    t0 = time.time()
    print("=" * 60)
    print("H257 — Batch Norm Statistics Shift vs Vulnerability")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Section 1: check BN layers exist
    # ------------------------------------------------------------------ #
    bn_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    print(f"\n[1] Found {len(bn_layers)} BatchNorm2d layers:")
    for name, _ in bn_layers:
        print(f"     {name}")
    if len(bn_layers) == 0:
        print("  WARNING: No BN layers found — cannot run this hypothesis.")
        return

    # ------------------------------------------------------------------ #
    # Section 2: generate adversarial examples
    # ------------------------------------------------------------------ #
    print("\n[2] Generating adversarial examples (PGD) ...")
    for p in model.parameters():
        p.requires_grad_(True)
    Xadv = C.pgd(model, Xte, Yte, eps=EPS, steps=10, alpha=0.01)
    model.eval()

    # ------------------------------------------------------------------ #
    # Section 3: collect BN activations
    # ------------------------------------------------------------------ #
    print("\n[3] Collecting BN activations for clean and adversarial inputs ...")
    clean_acts = collect_bn_activations(model, Xte)
    adv_acts   = collect_bn_activations(model, Xadv)

    for name in clean_acts:
        c_shape = tuple(clean_acts[name].shape)
        a_shape = tuple(adv_acts[name].shape)
        print(f"     Layer {name}: clean={c_shape}, adv={a_shape}")

    # ------------------------------------------------------------------ #
    # Section 4: compute per-sample BN shift
    # ------------------------------------------------------------------ #
    print("\n[4] Computing per-sample BN shift ...")
    bn_shift = per_sample_bn_shift(clean_acts, adv_acts)
    print(f"  BN shift: mean={bn_shift.mean():.6f}  std={bn_shift.std():.6f}  "
          f"min={bn_shift.min():.6f}  max={bn_shift.max():.6f}")

    # ------------------------------------------------------------------ #
    # Section 5: compute vulnerability labels
    # ------------------------------------------------------------------ #
    print("\n[5] Computing vulnerability (PGD success) and clean margin ...")
    with torch.no_grad():
        logits_clean, _ = C.logits_and_acc(model, Xte, Yte)
        preds_clean = logits_clean.argmax(1)
        logits_adv, _ = C.logits_and_acc(model, Xadv, Yte)
        preds_adv = logits_adv.argmax(1)
        pgd_success = ((preds_clean.cpu() == Yte.cpu()) & (preds_adv.cpu() != Yte.cpu())).cpu().numpy().astype(int)
        margins = C.margin(model, Xte, Yte, batch=512)

    print(f"  PGD ASR  : {pgd_success.mean():.4f}  ({pgd_success.sum()}/{N_EVAL})")
    print(f"  Margin   : mean={margins.mean():.4f}  std={margins.std():.4f}")

    # ------------------------------------------------------------------ #
    # Section 6: per-BN-layer shift analysis
    # ------------------------------------------------------------------ #
    print("\n[6] Per-layer BN shift statistics")
    print("-" * 55)
    for name in clean_acts:
        if name not in adv_acts:
            continue
        layer_shift = (clean_acts[name] - adv_acts[name]).abs().mean(dim=1).numpy()
        rho_succ, p_succ = spearmanr(layer_shift, pgd_success)
        rho_marg, p_marg = spearmanr(layer_shift, -margins)
        print(f"  {name:<30s}  shift_mean={layer_shift.mean():.5f}"
              f"  rho_vs_pgd={rho_succ:+.4f}(p={p_succ:.3f})"
              f"  rho_vs_margin={rho_marg:+.4f}(p={p_marg:.3f})")

    # ------------------------------------------------------------------ #
    # Section 7: aggregate correlation and AUROC
    # ------------------------------------------------------------------ #
    print("\n[7] Aggregate BN shift correlation and AUROC")
    print("-" * 55)
    rho_succ, p_succ = spearmanr(bn_shift, pgd_success)
    rho_marg, p_marg = spearmanr(bn_shift, -margins)
    print(f"  Spearman rho (BN shift vs PGD success): {rho_succ:+.4f}  p={p_succ:.4f}")
    print(f"  Spearman rho (BN shift vs -margin):     {rho_marg:+.4f}  p={p_marg:.4f}")

    if pgd_success.sum() > 0 and pgd_success.sum() < len(pgd_success):
        auc = roc_auc_score(pgd_success, bn_shift)
        print(f"  AUROC (BN shift -> PGD success):        {auc:.4f}")

    print(f"\nDone in {time.time() - t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
