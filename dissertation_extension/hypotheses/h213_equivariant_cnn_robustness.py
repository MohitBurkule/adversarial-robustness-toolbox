"""
H213 - Equivariant CNN architecture provides free robustness without adversarial training.

Motivation (arXiv:2510.16171, NeurIPS 2025):
  Rotation/reflection-equivariant CNNs achieve lower adversarial attack success rates than
  standard CNNs of the same capacity, even WITHOUT adversarial training. This suggests
  architecture-as-defense: the symmetry constraint implicitly smooths the loss landscape.

  Since e2cnn may not be installed, we approximate D4-equivariance at TEST TIME via an
  ensemble of 8 dihedral transforms (4 rotations x 2 reflections), averaging softmax outputs.

Three inference conditions (all using the SAME standard-trained CNN):
  1. Standard: plain forward pass
  2. D4 ensemble: average softmax over 8 dihedral transforms
  3. D4 + noise ensemble: same + Gaussian noise sigma=0.05 per transform

Attacks:
  - FGSM oblivious (eps=0.1, crafted on standard model)
  - PGD-10 oblivious (crafted on standard model)
  - FGSM adaptive (PGD-20 through the D4-ensemble mean logits)

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign.common import (
    set_seed, load_dataset, dataset_meta, build_model, train_model,
    logits_and_acc, fgsm, pgd, DEVICE,
)

SEED = 42
N_TRAIN = 6000
N_EVAL = 500
EPOCHS = 15
EPS = 0.1
NOISE_SIGMA = 0.05
DATASET = "fashion_mnist"


# ---------------------------------------------------------------------------
# D4 dihedral transforms (8 elements: 4 rotations x {identity, flip})
# ---------------------------------------------------------------------------
def d4_transforms(x):
    """Return list of 8 transformed versions of x (B,C,H,W)."""
    out = []
    for k in range(4):
        rotated = torch.rot90(x, k, dims=(-2, -1))
        out.append(rotated)
        out.append(torch.flip(rotated, dims=[-1]))
    return out


def d4_inverse(x, idx):
    """Inverse of the idx-th D4 transform (for inverse-transforming gradients)."""
    k = idx // 2
    is_flipped = idx % 2 == 1
    if is_flipped:
        x = torch.flip(x, dims=[-1])
    # inverse rotation: rotate by (4 - k)
    if k > 0:
        x = torch.rot90(x, 4 - k, dims=(-2, -1))
    return x


# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------
def ensemble_predict(model, x, noise_sigma=0.0):
    """D4 ensemble: average softmax over 8 dihedral transforms. Returns mean logits."""
    model.eval()
    probs_sum = None
    transforms = d4_transforms(x)
    for t in transforms:
        if noise_sigma > 0:
            t = t + torch.randn_like(t) * noise_sigma
            t = t.clamp(0, 1)
        with torch.no_grad():
            logits = model(t)
            p = F.softmax(logits, dim=-1)
        if probs_sum is None:
            probs_sum = p
        else:
            probs_sum = probs_sum + p
    # return log-probs as "logits" for consistency
    return torch.log(probs_sum / 8.0 + 1e-12)


def ensemble_predict_differentiable(model, x, noise_sigma=0.0):
    """Differentiable D4 ensemble for adaptive attack (gradients flow through transforms)."""
    probs_sum = None
    transforms = d4_transforms(x)
    for t in transforms:
        if noise_sigma > 0:
            t = t + torch.randn_like(t) * noise_sigma
            t = t.clamp(0, 1)
        logits = model(t)
        p = F.softmax(logits, dim=-1)
        if probs_sum is None:
            probs_sum = p
        else:
            probs_sum = probs_sum + p
    return torch.log(probs_sum / 8.0 + 1e-12)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def oblivious_attack(model, X, Y, attack_fn):
    """Craft adversarials on the standard model, then evaluate on each inference mode."""
    model.eval()
    return attack_fn(model, X, Y)


def adaptive_pgd_d4(model, x, y, eps=EPS, steps=20, alpha=None, noise_sigma=0.0):
    """PGD attack through the D4 ensemble (adaptive)."""
    if alpha is None:
        alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    xa = x0.clone() + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    model.eval()
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        logits = ensemble_predict_differentiable(model, xa, noise_sigma=noise_sigma)
        loss = F.cross_entropy(logits, y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_asr(model, X_adv, Y, predict_fn):
    """ASR on originally-correct samples given pre-crafted adversarials."""
    # check which are correct on clean (use predict_fn identity -- caller handles)
    # Here we just check: does predict_fn(X_adv) flip from Y?
    correct = []
    flipped = []
    bs = 64
    for i in range(0, X_adv.size(0), bs):
        xb, yb = X_adv[i:i+bs], Y[i:i+bs]
        with torch.no_grad():
            pred = predict_fn(xb).argmax(1)
        flipped.append((pred != yb).cpu())
    return torch.cat(flipped).float().numpy()


def compute_metrics(model, Xclean, Yclean, Xfgsm, Xpgd, label, noise_sigma=0.0):
    """Compute clean acc, oblivious FGSM/PGD ASR for a given inference mode."""
    bs = 64

    if label == "standard":
        def predict(x):
            return model(x)
    elif label == "d4_ensemble":
        def predict(x):
            return ensemble_predict(model, x, noise_sigma=0.0)
    elif label == "d4_noise_ensemble":
        def predict(x):
            return ensemble_predict(model, x, noise_sigma=noise_sigma)
    else:
        raise ValueError(label)

    # Clean accuracy
    correct_clean = []
    for i in range(0, Xclean.size(0), bs):
        with torch.no_grad():
            pred = predict(Xclean[i:i+bs]).argmax(1)
        correct_clean.append((pred == Yclean[i:i+bs]).cpu())
    clean_acc = torch.cat(correct_clean).float().mean().item()

    # Which samples are correctly classified clean?
    corr_mask = torch.cat(correct_clean).numpy().astype(bool)

    # FGSM oblivious ASR (on originally-correct samples)
    flips_fgsm = []
    for i in range(0, Xfgsm.size(0), bs):
        with torch.no_grad():
            pred = predict(Xfgsm[i:i+bs]).argmax(1)
        flips_fgsm.append((pred != Yclean[i:i+bs]).cpu())
    flips_fgsm = torch.cat(flips_fgsm).numpy()
    fgsm_asr = float(flips_fgsm[corr_mask].mean()) if corr_mask.sum() > 0 else float("nan")

    # PGD oblivious ASR
    flips_pgd = []
    for i in range(0, Xpgd.size(0), bs):
        with torch.no_grad():
            pred = predict(Xpgd[i:i+bs]).argmax(1)
        flips_pgd.append((pred != Yclean[i:i+bs]).cpu())
    flips_pgd = torch.cat(flips_pgd).numpy()
    pgd_asr = float(flips_pgd[corr_mask].mean()) if corr_mask.sum() > 0 else float("nan")

    return clean_acc, fgsm_asr, pgd_asr, corr_mask


def main():
    print("=" * 74)
    print("H213 - D4-equivariant ensemble: architecture-as-defense")
    print("=" * 74)
    print(f"Device={DEVICE}  DATASET={DATASET}  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}")
    print(f"EPOCHS={EPOCHS}  EPS={EPS}  NOISE_SIGMA={NOISE_SIGMA}")
    t0 = time.time()

    set_seed(SEED)
    meta = dataset_meta(DATASET)
    Xtr, Ytr, Xte, Yte = load_dataset(DATASET, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # --- Train a single standard CNN ---
    print("\n--- Training standard CNN ---")
    model = build_model("cnn", meta, width=32)
    train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=128, opt="adam", lr=1e-3)
    model.eval()

    # --- Craft oblivious adversarials (on standard model) ---
    print("--- Crafting oblivious adversarials ---")
    Xfgsm_parts, Xpgd_parts = [], []
    bs = 128
    for i in range(0, N_EVAL, bs):
        xb, yb = Xte[i:i+bs], Yte[i:i+bs]
        Xfgsm_parts.append(fgsm(model, xb, yb, eps=EPS))
        Xpgd_parts.append(pgd(model, xb, yb, eps=EPS, steps=10))
    Xfgsm_obliv = torch.cat(Xfgsm_parts)
    Xpgd_obliv = torch.cat(Xpgd_parts)

    # --- Evaluate three inference modes ---
    results = {}
    for mode in ["standard", "d4_ensemble", "d4_noise_ensemble"]:
        sigma = NOISE_SIGMA if mode == "d4_noise_ensemble" else 0.0
        clean_acc, fgsm_asr, pgd_asr, corr_mask = compute_metrics(
            model, Xte, Yte, Xfgsm_obliv, Xpgd_obliv, mode, noise_sigma=sigma)
        results[mode] = {"clean_acc": clean_acc, "fgsm_obliv_asr": fgsm_asr,
                         "pgd_obliv_asr": pgd_asr}
        print(f"  {mode}: clean={clean_acc:.4f} fgsm_obliv={fgsm_asr:.4f} pgd_obliv={pgd_asr:.4f}")

    # --- Adaptive attack on D4 ensemble ---
    print("--- Crafting adaptive adversarials (PGD-20 through D4 ensemble) ---")
    Xadapt_parts = []
    for i in range(0, N_EVAL, bs):
        xb, yb = Xte[i:i+bs], Yte[i:i+bs]
        Xadapt_parts.append(adaptive_pgd_d4(model, xb, yb, eps=EPS, steps=20))
    Xadapt = torch.cat(Xadapt_parts)

    # Evaluate adaptive on D4 ensemble
    adapt_flips = []
    adapt_corr = []
    for i in range(0, N_EVAL, 64):
        xb, yb = Xte[i:i+64], Yte[i:i+64]
        with torch.no_grad():
            clean_pred = ensemble_predict(model, xb).argmax(1)
            adv_pred = ensemble_predict(model, Xadapt[i:i+64]).argmax(1)
        adapt_corr.append((clean_pred == yb).cpu())
        adapt_flips.append((adv_pred != yb).cpu())
    adapt_corr = torch.cat(adapt_corr).numpy().astype(bool)
    adapt_flips = torch.cat(adapt_flips).numpy()
    fgsm_adapt_asr = float(adapt_flips[adapt_corr].mean()) if adapt_corr.sum() > 0 else float("nan")
    results["d4_ensemble"]["fgsm_adaptive_asr"] = fgsm_adapt_asr

    # Also adaptive on D4+noise
    Xadapt_noise_parts = []
    for i in range(0, N_EVAL, bs):
        xb, yb = Xte[i:i+bs], Yte[i:i+bs]
        Xadapt_noise_parts.append(adaptive_pgd_d4(model, xb, yb, eps=EPS, steps=20,
                                                    noise_sigma=NOISE_SIGMA))
    Xadapt_noise = torch.cat(Xadapt_noise_parts)

    adapt_n_flips = []
    adapt_n_corr = []
    for i in range(0, N_EVAL, 64):
        xb, yb = Xte[i:i+64], Yte[i:i+64]
        with torch.no_grad():
            clean_pred = ensemble_predict(model, xb, noise_sigma=NOISE_SIGMA).argmax(1)
            adv_pred = ensemble_predict(model, Xadapt_noise[i:i+64], noise_sigma=NOISE_SIGMA).argmax(1)
        adapt_n_corr.append((clean_pred == yb).cpu())
        adapt_n_flips.append((adv_pred != yb).cpu())
    adapt_n_corr = torch.cat(adapt_n_corr).numpy().astype(bool)
    adapt_n_flips = torch.cat(adapt_n_flips).numpy()
    adapt_noise_asr = float(adapt_n_flips[adapt_n_corr].mean()) if adapt_n_corr.sum() > 0 else float("nan")
    results["d4_noise_ensemble"]["fgsm_adaptive_asr"] = adapt_noise_asr

    # --- Final table ---
    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    print(f"\n{'inference_mode':<22} {'clean_acc':>10} {'fgsm_obliv':>12} {'pgd_obliv':>12} {'adaptive':>12}")
    print("-" * 68)
    for mode in ["standard", "d4_ensemble", "d4_noise_ensemble"]:
        r = results[mode]
        adapt = r.get("fgsm_adaptive_asr", float("nan"))
        print(f"{mode:<22} {r['clean_acc']:>10.4f} {r['fgsm_obliv_asr']:>12.4f} "
              f"{r['pgd_obliv_asr']:>12.4f} {adapt:>12.4f}")

    # --- Hypothesis test ---
    std_fgsm = results["standard"]["fgsm_obliv_asr"]
    d4_fgsm = results["d4_ensemble"]["fgsm_obliv_asr"]
    d4_adapt = results["d4_ensemble"]["fgsm_adaptive_asr"]
    delta_obliv = std_fgsm - d4_fgsm
    print(f"\n--- Hypothesis test ---")
    print(f"  D4 ensemble lowers oblivious FGSM ASR by {delta_obliv*100:.1f}pp "
          f"(threshold: >=5pp)")
    if delta_obliv >= 0.05:
        print(f"  => CONFIRMED: D4 ensemble provides >=5pp oblivious FGSM defense")
    else:
        print(f"  => NOT CONFIRMED: D4 ensemble benefit < 5pp")

    print(f"\n  Adaptive attack ASR on D4 ensemble: {d4_adapt:.4f}")
    print(f"  Oblivious ASR on D4 ensemble:       {d4_fgsm:.4f}")
    if d4_adapt > d4_fgsm + 0.05:
        print(f"  => Adaptive attack COLLAPSES D4 benefit (delta={d4_adapt-d4_fgsm:.4f})")
    else:
        print(f"  => D4 benefit partly SURVIVES adaptive attack")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
