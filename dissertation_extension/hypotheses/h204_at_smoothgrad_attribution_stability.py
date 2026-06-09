"""
H204 - Adversarial training stabilises SmoothGrad attribution maps.

Hypothesis: PGD adversarial examples have SmoothGrad attribution maps with
mean cosine similarity <0.3 to clean-input maps for standard models, but >0.6
for AT models -- AT stabilises attributions by 2x.

Cites: arXiv:2603.07302, arXiv:2411.05837.

Methodology:
  - Standard model vs PGD-AT model on Fashion-MNIST (n_train=6000, 15 epochs).
  - SmoothGrad: (1/N) sum of input gradients over N=20 noisy copies (sigma=0.1).
  - Metrics: cosine similarity, relative L2 distance, sign stability between
    clean and adversarial SmoothGrad maps.
  - For 5 example samples, print top-10 pixel overlap between clean and adv maps.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign.common import (
    DEVICE, set_seed, load_dataset, dataset_meta,
    build_model, train_model, pgd, logits_and_acc,
)

SEED = 42
N_TRAIN = 6000
N_EVAL = 200
EPOCHS = 15
LR = 0.05
OPT = "sgd"
BATCH = 128
SG_N = 20        # SmoothGrad samples
SG_SIGMA = 0.1   # SmoothGrad noise std
ATK_EPS = 0.1
ATK_STEPS = 10


# ---- SmoothGrad -------------------------------------------------------------

def smoothgrad(model, x, y, n_samples=SG_N, sigma=SG_SIGMA):
    """Compute SmoothGrad attribution for a single input x (1,C,H,W).
    Returns attribution of same shape as x, on CPU."""
    model.eval()
    x = x.unsqueeze(0) if x.dim() == 3 else x  # ensure (1,C,H,W)
    accum = torch.zeros_like(x)
    for _ in range(n_samples):
        xn = (x + torch.randn_like(x) * sigma).clamp(0, 1).detach().requires_grad_(True)
        loss = F.cross_entropy(model(xn), y.unsqueeze(0) if y.dim() == 0 else y)
        g, = torch.autograd.grad(loss, xn)
        accum += g.detach()
    return (accum / n_samples).squeeze(0).cpu()  # (C,H,W)


def smoothgrad_batch(model, X, Y, n_samples=SG_N, sigma=SG_SIGMA):
    """Compute SmoothGrad for each sample independently. Returns list of tensors."""
    attrs = []
    for i in range(X.size(0)):
        sg = smoothgrad(model, X[i], Y[i], n_samples, sigma)
        attrs.append(sg)
    return attrs


# ---- metrics ----------------------------------------------------------------

def cosine_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = (a_flat.norm() * b_flat.norm())
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / denom)


def relative_l2(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    norm_a = a_flat.norm()
    if norm_a < 1e-12:
        return float('nan')
    return float((a_flat - b_flat).norm() / norm_a)


def sign_stability(a, b):
    sa = (a.flatten() >= 0).float()
    sb = (b.flatten() >= 0).float()
    return float((sa == sb).float().mean())


# ---- main -------------------------------------------------------------------

def main():
    t0 = time.time()
    set_seed(SEED)
    meta = dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # Train standard model
    print("--- Training STANDARD model ---")
    set_seed(SEED)
    std_model = build_model("cnn", meta, width=32, bn=True).to(DEVICE)
    train_model(std_model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                opt=OPT, lr=LR, ncls=meta["n_classes"], verbose=True)
    _, std_acc = logits_and_acc(std_model, Xte, Yte)
    print(f"  Standard clean acc: {std_acc:.4f}")

    # Train AT model
    print("\n--- Training AT model ---")
    set_seed(SEED)
    at_model = build_model("cnn", meta, width=32, bn=True).to(DEVICE)
    train_model(at_model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                opt=OPT, lr=LR, ncls=meta["n_classes"],
                adv_train=True, adv_eps=0.3, adv_steps=7, verbose=True)
    _, at_acc = logits_and_acc(at_model, Xte, Yte)
    print(f"  AT clean acc: {at_acc:.4f}")

    # Generate adversarial examples
    print("\n--- Generating PGD adversarial examples ---")
    X_adv_std = pgd(std_model, Xte, Yte, eps=ATK_EPS, steps=ATK_STEPS)
    X_adv_at = pgd(at_model, Xte, Yte, eps=ATK_EPS, steps=ATK_STEPS)

    # Compute SmoothGrad attributions
    print("--- Computing SmoothGrad (standard model, clean) ---")
    sg_std_clean = smoothgrad_batch(std_model, Xte, Yte)
    print("--- Computing SmoothGrad (standard model, adv) ---")
    sg_std_adv = smoothgrad_batch(std_model, X_adv_std, Yte)
    print("--- Computing SmoothGrad (AT model, clean) ---")
    sg_at_clean = smoothgrad_batch(at_model, Xte, Yte)
    print("--- Computing SmoothGrad (AT model, adv) ---")
    sg_at_adv = smoothgrad_batch(at_model, X_adv_at, Yte)

    # Compute metrics
    def compute_metrics(sg_clean_list, sg_adv_list):
        cos_sims, l2_dists, sign_stabs = [], [], []
        for sc, sa in zip(sg_clean_list, sg_adv_list):
            cos_sims.append(cosine_sim(sc, sa))
            l2_dists.append(relative_l2(sc, sa))
            sign_stabs.append(sign_stability(sc, sa))
        return {
            "cosine_sim_mean": np.mean(cos_sims),
            "cosine_sim_std": np.std(cos_sims),
            "l2_dist_mean": np.nanmean(l2_dists),
            "l2_dist_std": np.nanstd(l2_dists),
            "sign_stab_mean": np.mean(sign_stabs),
            "sign_stab_std": np.std(sign_stabs),
        }

    std_metrics = compute_metrics(sg_std_clean, sg_std_adv)
    at_metrics = compute_metrics(sg_at_clean, sg_at_adv)

    # Top-10 pixel overlap for 5 examples
    def top_k_overlap(sg_clean_list, sg_adv_list, k=10, n_examples=5):
        overlaps = []
        for i in range(min(n_examples, len(sg_clean_list))):
            sc = sg_clean_list[i].flatten().abs()
            sa = sg_adv_list[i].flatten().abs()
            top_clean = set(sc.topk(k).indices.tolist())
            top_adv = set(sa.topk(k).indices.tolist())
            overlap = len(top_clean & top_adv)
            overlaps.append((i, sorted(top_clean), sorted(top_adv), overlap))
        return overlaps

    std_overlaps = top_k_overlap(sg_std_clean, sg_std_adv)
    at_overlaps = top_k_overlap(sg_at_clean, sg_at_adv)

    # Hypothesis test
    ratio = at_metrics["cosine_sim_mean"] / max(std_metrics["cosine_sim_mean"], 1e-9)
    hypothesis_met = (std_metrics["cosine_sim_mean"] < 0.3 and
                      at_metrics["cosine_sim_mean"] > 0.6)

    # ---- report -------------------------------------------------------------
    elapsed = time.time() - t0
    lines = []
    lines.append("=" * 72)
    lines.append("H204 - AT Stabilises SmoothGrad Attribution Maps")
    lines.append("=" * 72)
    lines.append(f"Dataset: Fashion-MNIST  n_train={N_TRAIN}  n_eval={N_EVAL}  seed={SEED}")
    lines.append(f"SmoothGrad: N={SG_N}  sigma={SG_SIGMA}  |  Attack: PGD-{ATK_STEPS} eps={ATK_EPS}")
    lines.append(f"Standard model clean acc: {std_acc:.4f}")
    lines.append(f"AT model clean acc:       {at_acc:.4f}")
    lines.append("")

    header = f"{'Model':<12} {'CosSim':>10} {'±std':>8} {'RelL2':>10} {'±std':>8} {'SignStab':>10} {'±std':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in [("Standard", std_metrics), ("AT", at_metrics)]:
        lines.append(f"{name:<12} {m['cosine_sim_mean']:>10.4f} {m['cosine_sim_std']:>8.4f} "
                      f"{m['l2_dist_mean']:>10.4f} {m['l2_dist_std']:>8.4f} "
                      f"{m['sign_stab_mean']:>10.4f} {m['sign_stab_std']:>8.4f}")

    lines.append("")
    lines.append(f"Cosine similarity ratio (AT / Standard): {ratio:.2f}x")
    lines.append(f"Hypothesis (std<0.3, AT>0.6, AT>=2x std): {'SUPPORTED' if hypothesis_met else 'NOT SUPPORTED'}")
    lines.append("")

    # Top-10 pixel overlap examples
    for label, overlaps in [("Standard", std_overlaps), ("AT", at_overlaps)]:
        lines.append(f"Top-10 pixel overlap ({label} model, 5 examples):")
        for idx, tc, ta, ov in overlaps:
            lines.append(f"  Sample {idx}: clean={tc}  adv={ta}  overlap={ov}/10")
        lines.append("")

    lines.append(f"Elapsed: {elapsed:.1f}s")
    report = "\n".join(lines)
    print(report)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "h204_at_smoothgrad_attribution_stability_output.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
