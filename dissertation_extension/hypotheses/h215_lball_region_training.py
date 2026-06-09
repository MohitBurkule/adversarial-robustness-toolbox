"""
H215 - L-Ball Region Training: does training over the epsilon-ball produce
better-calibrated margins?

Compare 3 training regimes on Fashion-MNIST:
  1. Standard training (baseline)
  2. Smooth-AT: add Gaussian noise σ=0.1 to inputs during training
  3. FGSM-AT: standard FGSM adversarial training with eps=0.1

Key metric: Spearman ρ between per-sample margin and a certified-radius proxy
computed via Monte-Carlo smoothing (50 forward passes with σ=0.1 noise,
fraction correct → certified_radius = σ * Φ⁻¹(p_A)).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm as scipy_norm, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS        = "fashion_mnist"
N_EVAL    = 300
SEED      = 0
EPS       = 0.1
SIGMA     = 0.1
N_SMOOTH  = 50    # Monte-Carlo samples for certified radius
EPOCHS    = 10
BATCH     = 128


# ---------------------------------------------------------------------------
# certified-radius proxy
# ---------------------------------------------------------------------------
@torch.no_grad()
def certified_radius_proxy(model, X, Y, sigma=SIGMA, n_samples=N_SMOOTH, batch=128):
    """For each sample compute fraction of noisy forwards that are correct, then
    certified_radius = σ * Φ⁻¹(p_A) clipped to [0, ∞).
    """
    n = X.size(0)
    correct_counts = torch.zeros(n, device=X.device)
    for _ in range(n_samples):
        noise = torch.randn_like(X) * sigma
        Xn = (X + noise).clamp(0, 1)
        for i in range(0, n, batch):
            out = model(Xn[i:i+batch])
            correct_counts[i:i+batch] += (out.argmax(1) == Y[i:i+batch]).float()
    p_A = (correct_counts / n_samples).cpu().numpy()
    # clamp p_A so ppf stays finite and positive
    p_A_safe = np.clip(p_A, 0.501, 0.999)
    cr = sigma * scipy_norm.ppf(p_A_safe)
    # samples where model was never correct → radius = 0
    cr[p_A <= 0.5] = 0.0
    return np.clip(cr, 0.0, np.inf)


# ---------------------------------------------------------------------------
# custom training loop with Smooth-AT noise injection
# ---------------------------------------------------------------------------
def train_smooth_at(model, Xtr, Ytr, sigma=SIGMA, epochs=EPOCHS, batch=BATCH):
    """Standard training with Gaussian noise added to each batch."""
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb, yb = Xtr[idx], Ytr[idx]
            noise = torch.randn_like(xb) * sigma
            xb_noisy = (xb + noise).clamp(0, 1)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_noisy), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_fgsm_at(model, Xtr, Ytr, eps=EPS, epochs=EPOCHS, batch=BATCH):
    """FGSM adversarial training."""
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb, yb = Xtr[idx], Ytr[idx]
            # FGSM step (keep model in train mode)
            xb_adv = C.fgsm(model, xb, yb, eps)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("=== H215: L-Ball Region Training — margin calibration ===")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  sigma={SIGMA}  N_EVAL={N_EVAL}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_EVAL, seed=SEED)

    regimes = [
        ("standard",  None),
        ("smooth_at", "smooth"),
        ("fgsm_at",   "fgsm"),
    ]

    results = {}
    for name, mode in regimes:
        print(f"\n--- 1. Training: {name} ---")
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32, seed=SEED)

        if mode is None:
            C.train_model(model, Xtr, Ytr, epochs=EPOCHS, opt="sgd", lr=0.05, ncls=meta["n_classes"])
        elif mode == "smooth":
            train_smooth_at(model, Xtr, Ytr)
        elif mode == "fgsm":
            train_fgsm_at(model, Xtr, Ytr)

        # --- clean accuracy
        _, clean_acc = C.logits_and_acc(model, Xte, Yte)
        print(f"  clean_acc = {clean_acc:.3f}")

        # --- attack ASRs
        fgsm_res = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
        pgd_res  = C.attack_success(model, Xte, Yte, attack="pgd",  eps=EPS, steps=10)
        print(f"  FGSM ASR = {fgsm_res['asr']:.3f}  PGD ASR = {pgd_res['asr']:.3f}")

        # --- margins
        mgn = C.margin(model, Xte, Yte)
        print(f"  margin mean={mgn.mean():.3f} std={mgn.std():.3f}")

        # --- certified radius proxy
        print(f"  Computing certified-radius proxy ({N_SMOOTH} MC samples)...")
        cr = certified_radius_proxy(model, Xte, Yte)
        print(f"  cert_radius mean={cr.mean():.4f} std={cr.std():.4f}")

        # Spearman ρ between margin and certified radius
        rho, pval = spearmanr(mgn, cr)
        print(f"  Spearman rho(margin, cert_radius) = {rho:.4f}  p={pval:.4e}")

        results[name] = {
            "clean_acc": clean_acc,
            "fgsm_asr": fgsm_res["asr"],
            "pgd_asr":  pgd_res["asr"],
            "margin_mean": float(mgn.mean()),
            "margin_std":  float(mgn.std()),
            "cr_mean":  float(cr.mean()),
            "cr_std":   float(cr.std()),
            "spearman_rho": float(rho),
            "spearman_p":   float(pval),
        }
        print(f"  ({time.time()-t0:.1f}s)")

    # --- Summary table ---
    print("\n" + "=" * 74)
    print("--- Summary ---")
    print(f"{'Regime':<14} {'CleanAcc':>9} {'FGSM_ASR':>9} {'PGD_ASR':>8} "
          f"{'Mgn_mean':>9} {'CR_mean':>8} {'Spearman_ρ':>11}")
    for name, r in results.items():
        print(f"{name:<14} {r['clean_acc']:>9.3f} {r['fgsm_asr']:>9.3f} "
              f"{r['pgd_asr']:>8.3f} {r['margin_mean']:>9.3f} "
              f"{r['cr_mean']:>8.4f} {r['spearman_rho']:>11.4f}")
    print("=" * 74)
    print("Interpretation: higher Spearman ρ(margin, cert_radius) indicates better")
    print("margin calibration — margin predicts certified robustness. Smooth-AT is")
    print("expected to show tighter correlation than baseline; FGSM-AT may widen")
    print("margins overall but may not improve calibration versus smoothed model.")
    print("=" * 74)


if __name__ == "__main__":
    main()
