"""
H276 - Spectral norm regularisation as implicit adversarial training.

Spectral normalisation (SN) constrains the Lipschitz constant of each layer
by normalising its weight matrix by its largest singular value (spectral norm).
A globally bounded Lipschitz constant bounds how much the output can change
for a given input perturbation, which theoretically should limit adversarial
vulnerability.

Four models:
  (a) Baseline: no regularisation
  (b) SN-only: torch.nn.utils.spectral_norm on all Conv2d and Linear layers
  (c) WD-only: weight decay lambda=0.01 (L2 regularisation, different mechanism)
  (d) SN+WD: both spectral norm and weight decay

For each model, also estimate the per-layer spectral norm (largest singular value)
after training to verify whether SN actually reduces it.

Key question: does Lipschitz bounding via spectral norm give any adversarial
robustness without explicit adversarial training?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS        = "fashion_mnist"
SEED      = 0
N_TRAIN   = 10000
EPOCHS    = 10
LR        = 0.05
MOMENTUM  = 0.9
BATCH     = 128
EPS       = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
WD_STRONG = 0.01   # strong weight decay for condition (c) and (d)
WD_BASE   = 5e-4   # normal weight decay for (a) and (b)
OUT_FILE  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h276_spectral_norm_implicit_at_output.txt"
)


# ---------------------------------------------------------------------------
# Spectral norm application
# ---------------------------------------------------------------------------

def apply_spectral_norm(model):
    """Apply spectral_norm to all Conv2d and Linear layers in-place."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.utils.spectral_norm(module)
    return model


def estimate_spectral_norm(module):
    """Estimate the spectral norm (largest singular value) of a layer's weight."""
    if isinstance(module, nn.Conv2d):
        # Reshape weight to 2D: (out_channels, in_channels * kH * kW)
        W = module.weight.data.reshape(module.weight.size(0), -1)
    elif isinstance(module, nn.Linear):
        W = module.weight.data
    else:
        return None
    # Use SVD; for large matrices use power iteration for speed
    W_cpu = W.float().cpu()
    if W_cpu.numel() < 200 * 200:
        try:
            sv = torch.linalg.svdvals(W_cpu)
            return float(sv[0])
        except Exception:
            pass
    # Power iteration fallback
    v = torch.randn(W_cpu.shape[1])
    v = v / (v.norm() + 1e-12)
    for _ in range(20):
        u = W_cpu @ v
        u = u / (u.norm() + 1e-12)
        v = W_cpu.T @ u
        v = v / (v.norm() + 1e-12)
    return float((u @ (W_cpu @ v)).item())


def collect_spectral_norms(model):
    """Return dict {layer_name: spectral_norm_value} for conv and linear layers."""
    sns = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            sn_val = estimate_spectral_norm(module)
            if sn_val is not None:
                sns[name] = sn_val
    return sns


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, X, Y, wd):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=wd)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = X[idx].to(C.DEVICE), Y[idx].to(C.DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"    epoch {ep+1}/{EPOCHS}  loss={total_loss:.3f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, Xte, Yte, label):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, fgsm_acc = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, pgd_acc = C.logits_and_acc(model, Xpgd, Yte)
    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))
    fgsm_asr, pgd_asr = 1.0 - fgsm_acc, 1.0 - pgd_acc
    print(f"  [{label}] clean={clean_acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  "
          f"PGD_ASR={pgd_asr:.4f}  margin={mean_margin:.4f}")
    return dict(clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                pgd_asr=pgd_asr, mean_margin=mean_margin)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    C.set_seed(SEED)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    print("Loading data...")
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    idx = torch.randperm(len(Xtr))[:N_TRAIN]
    Xtr, Ytr = Xtr[idx], Ytr[idx]
    Xte, Yte = Xte[:500], Yte[:500]
    print(f"  train={len(Xtr)}  test={len(Xte)}")

    configs = [
        ("baseline",  False, WD_BASE),
        ("sn_only",   True,  WD_BASE),
        ("wd_only",   False, WD_STRONG),
        ("sn_and_wd", True,  WD_STRONG),
    ]
    results     = {}
    sn_profiles = {}

    for name, use_sn, wd in configs:
        print(f"\n--- Training {name} (sn={use_sn}, wd={wd}) ---")
        C.set_seed(SEED)
        model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                              width=32, seed=SEED)
        model.to(C.DEVICE)
        if use_sn:
            apply_spectral_norm(model)

        train_model(model, Xtr, Ytr, wd=wd)

        r = evaluate(model, Xte, Yte, label=name)
        sn_vals = collect_spectral_norms(model)
        mean_sn = float(np.mean(list(sn_vals.values()))) if sn_vals else float("nan")
        print(f"  [{name}] mean_spectral_norm={mean_sn:.4f}")
        r["mean_spectral_norm"] = mean_sn
        results[name]     = r
        sn_profiles[name] = sn_vals

    elapsed = time.time() - t0

    lines = [
        "H276 - Spectral Norm as Implicit Adversarial Training\n",
        "=" * 70 + "\n\n",
        f"N_train={N_TRAIN}  epochs={EPOCHS}  lr={LR}  eps={EPS}  pgd_steps={PGD_STEPS}\n",
        f"WD_strong={WD_STRONG}  WD_base={WD_BASE}\n\n",
        f"{'Model':<14} {'CleanAcc':>10} {'FGSM_ASR':>10} {'PGD_ASR':>10} "
        f"{'Margin':>10} {'MeanSN':>10}\n",
        "-" * 64 + "\n",
    ]
    for name, r in results.items():
        lines.append(
            f"{name:<14} {r['clean_acc']:>10.4f} {r['fgsm_asr']:>10.4f} "
            f"{r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f} "
            f"{r['mean_spectral_norm']:>10.4f}\n"
        )

    lines.append("\nPer-layer spectral norms:\n")
    for name, sv in sn_profiles.items():
        lines.append(f"  {name}:\n")
        for layer, val in sv.items():
            lines.append(f"    {layer}: {val:.4f}\n")

    lines.append(f"\nElapsed: {elapsed:.1f}s\n\n")
    lines.append("ANALYSIS\n--------\n")
    base = results["baseline"]
    sn   = results["sn_only"]
    wd   = results["wd_only"]
    snwd = results["sn_and_wd"]
    lines.append(f"SN reduces mean spectral norm: {sn['mean_spectral_norm'] < base['mean_spectral_norm']}"
                 f"  (base={base['mean_spectral_norm']:.4f}, sn={sn['mean_spectral_norm']:.4f})\n")
    lines.append(f"SN-only PGD_ASR delta vs baseline:   {sn['pgd_asr']-base['pgd_asr']:+.4f}\n")
    lines.append(f"WD-only PGD_ASR delta vs baseline:   {wd['pgd_asr']-base['pgd_asr']:+.4f}\n")
    lines.append(f"SN+WD   PGD_ASR delta vs baseline:   {snwd['pgd_asr']-base['pgd_asr']:+.4f}\n")
    verdict = ("Spectral norm gives meaningful robustness for free."
               if sn["pgd_asr"] < base["pgd_asr"] - 0.02
               else "Spectral norm alone does NOT provide meaningful adversarial robustness without AT.")
    lines.append(f"Verdict: {verdict}\n")

    with open(OUT_FILE, "w") as f:
        f.writelines(lines)
    print(f"\nResults written to {OUT_FILE}")
