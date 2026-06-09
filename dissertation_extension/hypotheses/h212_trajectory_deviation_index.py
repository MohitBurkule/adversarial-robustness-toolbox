"""
H212 - Trajectory Deviation Index (TDI): a geometric blind spot of adversarial training.

Motivation (arXiv:2604.21395, April 2026):
  PGD-AT lowers the Frobenius norm of the input-gradient (smoother landscape) but simultaneously
  *increases* the Trajectory Deviation Index -- the fraction of each sample's gradient that is
  aligned with the class-level label-correlated direction. Higher TDI means adversarial
  perturbations are more predictable and exploitable. This is a theorem-level geometric
  blind spot: AT cannot fully fix robustness by smoothing alone.

Protocol:
  - Train two SmallCNN models on Fashion-MNIST (n_train=6000, 15 epochs):
      A) standard cross-entropy
      B) PGD-AT (eps=0.3, steps=7)
  - For each model, on 200 test samples:
      * Compute input gradient g = nabla_x L(f(x), y)
      * Gradient norm ||g||_2
      * Class-level label-correlated direction v_y = mean gradient direction for class y
        (estimated from 50 samples per class)
      * TDI(x) = |cos(g, v_y)|
  - Report mean gradient norm, mean TDI, per-class TDI breakdown.
  - Confirm AT lowers gradient norm but raises TDI.
  - Also report PGD-10 ASR for both models.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign.common import (
    set_seed, load_dataset, dataset_meta, build_model, train_model,
    attack_success, logits_and_acc, DEVICE,
)

SEED = 42
N_TRAIN = 6000
N_EVAL = 200
N_CLASS_SAMPLES = 50  # samples per class for estimating v_y
EPOCHS = 15
EPS_AT = 0.3
AT_STEPS = 7
EPS_EVAL = 0.3
DATASET = "fashion_mnist"


def compute_input_gradient(model, x, y):
    """Compute gradient of CE loss w.r.t. input x. Returns gradient tensor same shape as x."""
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    g, = torch.autograd.grad(loss, x)
    return g.detach()


def compute_class_directions(model, X, Y, n_classes=10, n_per_class=50):
    """Compute mean gradient direction v_y for each class y."""
    directions = {}
    for c in range(n_classes):
        idx = (Y == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            directions[c] = torch.zeros(X.shape[1:], device=DEVICE)
            continue
        sel = idx[:n_per_class]
        grads = []
        for i in sel:
            g = compute_input_gradient(model, X[i:i+1], Y[i:i+1])
            grads.append(g.squeeze(0))
        grads = torch.stack(grads)
        mean_g = grads.mean(dim=0)
        # normalise to unit direction
        norm = mean_g.flatten().norm()
        if norm > 1e-12:
            mean_g = mean_g / norm
        directions[c] = mean_g
    return directions


def compute_tdi(model, X, Y, class_dirs):
    """Compute per-sample TDI = |cos(g, v_y)| and gradient norms."""
    tdis = []
    norms = []
    for i in range(X.size(0)):
        g = compute_input_gradient(model, X[i:i+1], Y[i:i+1]).squeeze(0)
        g_flat = g.flatten()
        g_norm = g_flat.norm().item()
        norms.append(g_norm)

        v_y = class_dirs[Y[i].item()].flatten()
        v_norm = v_y.norm().item()
        if g_norm < 1e-12 or v_norm < 1e-12:
            tdis.append(0.0)
        else:
            cos_sim = (g_flat @ v_y).item() / (g_norm * v_norm)
            tdis.append(abs(cos_sim))
    return np.array(tdis), np.array(norms)


def main():
    print("=" * 74)
    print("H212 - Trajectory Deviation Index (TDI): geometric blind spot of AT")
    print("=" * 74)
    print(f"Device={DEVICE}  DATASET={DATASET}  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}")
    print(f"EPOCHS={EPOCHS}  EPS_AT={EPS_AT}  EPS_EVAL={EPS_EVAL}")
    t0 = time.time()

    set_seed(SEED)
    meta = dataset_meta(DATASET)
    Xtr, Ytr, Xte, Yte = load_dataset(DATASET, n_train=N_TRAIN, n_eval=1000, seed=SEED)
    # use first N_EVAL for TDI analysis, full 1000 for class direction estimation
    Xeval, Yeval = Xte[:N_EVAL], Yte[:N_EVAL]

    # --- Model A: standard training ---
    print("\n--- Training Model A (standard) ---")
    model_std = build_model("cnn", meta, width=32)
    train_model(model_std, Xtr, Ytr, epochs=EPOCHS, batch=128, opt="adam", lr=1e-3)
    _, acc_std = logits_and_acc(model_std, Xeval, Yeval)
    print(f"  clean acc = {acc_std:.4f}")

    # --- Model B: PGD-AT ---
    print("\n--- Training Model B (PGD-AT, eps={EPS_AT}) ---")
    set_seed(SEED)
    model_at = build_model("cnn", meta, width=32)
    train_model(model_at, Xtr, Ytr, epochs=EPOCHS, batch=128, opt="adam", lr=1e-3,
                adv_train=True, adv_eps=EPS_AT, adv_steps=AT_STEPS)
    _, acc_at = logits_and_acc(model_at, Xeval, Yeval)
    print(f"  clean acc = {acc_at:.4f}")

    # --- Compute class-level gradient directions ---
    print("\n--- Computing class-level gradient directions ---")
    dirs_std = compute_class_directions(model_std, Xte, Yte, n_classes=meta["n_classes"],
                                        n_per_class=N_CLASS_SAMPLES)
    dirs_at = compute_class_directions(model_at, Xte, Yte, n_classes=meta["n_classes"],
                                       n_per_class=N_CLASS_SAMPLES)

    # --- Compute TDI and gradient norms ---
    print("--- Computing TDI and gradient norms on eval set ---")
    tdi_std, norms_std = compute_tdi(model_std, Xeval, Yeval, dirs_std)
    tdi_at, norms_at = compute_tdi(model_at, Xeval, Yeval, dirs_at)

    # --- PGD ASR ---
    print("--- Computing PGD-10 ASR ---")
    asr_std = attack_success(model_std, Xeval, Yeval, attack="pgd", eps=EPS_EVAL, steps=10)["asr"]
    asr_at = attack_success(model_at, Xeval, Yeval, attack="pgd", eps=EPS_EVAL, steps=10)["asr"]

    # --- Results ---
    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)

    print(f"\n{'metric':<30} {'standard':>12} {'PGD-AT':>12}")
    print("-" * 54)
    print(f"{'clean accuracy':<30} {acc_std:>12.4f} {acc_at:>12.4f}")
    print(f"{'mean gradient norm':<30} {norms_std.mean():>12.4f} {norms_at.mean():>12.4f}")
    print(f"{'std gradient norm':<30} {norms_std.std():>12.4f} {norms_at.std():>12.4f}")
    print(f"{'mean TDI':<30} {tdi_std.mean():>12.4f} {tdi_at.mean():>12.4f}")
    print(f"{'std TDI':<30} {tdi_std.std():>12.4f} {tdi_at.std():>12.4f}")
    print(f"{'PGD-10 ASR (eps={EPS_EVAL})':<30} {asr_std:>12.4f} {asr_at:>12.4f}")

    # Per-class TDI
    print(f"\n{'class':<8} {'TDI(std)':>10} {'TDI(AT)':>10} {'delta':>10}")
    print("-" * 38)
    for c in range(meta["n_classes"]):
        mask = (Yeval.cpu() == c).numpy()
        if mask.sum() == 0:
            continue
        t_s = tdi_std[mask].mean()
        t_a = tdi_at[mask].mean()
        print(f"{c:<8} {t_s:>10.4f} {t_a:>10.4f} {t_a - t_s:>+10.4f}")

    # Hypothesis test
    norm_lower = norms_std.mean() > norms_at.mean()
    tdi_higher = tdi_at.mean() > tdi_std.mean()
    print(f"\n--- Hypothesis test ---")
    print(f"  gradient_norm(std) > gradient_norm(AT): {norm_lower}  "
          f"({norms_std.mean():.4f} vs {norms_at.mean():.4f})")
    print(f"  TDI(AT) > TDI(std):                    {tdi_higher}  "
          f"({tdi_at.mean():.4f} vs {tdi_std.mean():.4f})")
    if norm_lower and tdi_higher:
        print("  => CONFIRMED: AT lowers gradient norm but raises TDI (geometric blind spot)")
    elif norm_lower and not tdi_higher:
        print("  => PARTIAL: AT lowers gradient norm but TDI does not increase")
    else:
        print("  => NOT CONFIRMED: gradient norm pattern does not match prediction")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
