"""
H269 - Does the position of Gaussian noise injection (early vs late hidden layer)
affect adversarial robustness differently?

Five model variants are trained:
  - baseline  : no noise injection
  - noise_b1  : noise added after block1 activations (early)
  - noise_b2  : noise added after block2 activations (mid)
  - noise_b3  : noise added after block3 activations (late)

Noise is injected via register_forward_hook; the hook checks model.training so
noise is disabled at eval time. After training we measure:
  - clean_acc
  - FGSM attack success rate (ASR)
  - PGD attack success rate  (ASR)
  - mean margin
  - mean input-space gradient norm  ‖∇_x L‖ averaged over 500 test samples

Key question: does early-layer noise push the decision boundary further from the
input manifold than late-layer noise?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign import common as C


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------

def make_noise_hook(sigma: float, model_ref):
    """Return a forward hook that adds Gaussian noise during training only."""
    def hook(module, input, output):
        if model_ref[0].training:
            return output + torch.randn_like(output) * sigma
        return output
    return hook


def attach_noise_hook(model, position: str, sigma: float):
    """
    position in {"block1", "block2", "block3"}.
    Returns handle list so hooks can be removed before eval.
    """
    model_ref = [model]
    # features is a flat Sequential; each block is 4 layers (Conv, BN, ReLU, Pool)
    # Hook onto the last layer of each block (MaxPool2d)
    n_layers = len(model.features)
    lpb = n_layers // 3  # layers per block
    target_layer = {
        "block1": model.features[lpb - 1],      # end of block 1
        "block2": model.features[2 * lpb - 1],   # end of block 2
        "block3": model.features[3 * lpb - 1],   # end of block 3
    }[position]
    h = target_layer.register_forward_hook(make_noise_hook(sigma, model_ref))
    return [h]


# ---------------------------------------------------------------------------
# Gradient norm at input
# ---------------------------------------------------------------------------

def mean_input_grad_norm(model, X, Y, n_samples=500):
    """Average ‖∇_x CE‖_2 over n_samples test examples."""
    model.eval()
    idx = torch.randperm(len(X))[:n_samples]
    Xs = X[idx].to(C.DEVICE).requires_grad_(True)
    Ys = Y[idx].to(C.DEVICE)
    logits = model(Xs)
    loss = nn.CrossEntropyLoss()(logits, Ys)
    loss.backward()
    grad_norms = Xs.grad.detach().view(n_samples, -1).norm(dim=1)
    return grad_norms.mean().item()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_asr(model, X, Y, Xadv):
    """Fraction of initially-correct samples flipped by the attack."""
    model.eval()
    with torch.no_grad():
        clean_pred = model(X.to(C.DEVICE)).argmax(1).cpu().numpy()
        adv_pred   = model(Xadv.to(C.DEVICE)).argmax(1).cpu().numpy()
    y_np = Y.cpu().numpy()
    correct_clean = clean_pred == y_np
    fooled = correct_clean & (adv_pred != y_np)
    if correct_clean.sum() == 0:
        return 0.0
    return float(fooled.sum() / correct_clean.sum())


def evaluate_model(model, Xte, Yte, tag=""):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    _, clean_acc = C.logits_and_acc(model, Xte, Yte)

    Xfgsm = C.fgsm(model, Xte, Yte, eps=0.1)
    fgsm_asr = compute_asr(model, Xte, Yte, Xfgsm)

    Xpgd = C.pgd(model, Xte, Yte, eps=0.1, steps=10, alpha=0.01)
    pgd_asr = compute_asr(model, Xte, Yte, Xpgd)

    margins = C.margin(model, Xte, Yte)
    mean_mg = float(np.mean(margins))

    grad_norm = mean_input_grad_norm(model, Xte, Yte)

    print(f"  [{tag}]  clean_acc={clean_acc:.4f}  fgsm_asr={fgsm_asr:.4f}  "
          f"pgd_asr={pgd_asr:.4f}  mean_margin={mean_mg:.4f}  "
          f"grad_norm={grad_norm:.4f}")

    return dict(tag=tag, clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                pgd_asr=pgd_asr, mean_margin=mean_mg, grad_norm=grad_norm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SIGMA       = 0.1
EPOCHS      = 10
RESULTS_DIR = os.path.join(os.path.dirname(__file__),
                           "..", "results", "fashion_mnist")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "h269_noise_layer_position_output.txt")

    C.set_seed(0)
    print("Loading Fashion-MNIST …")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")

    variants = [
        ("baseline", None),
        ("noise_b1", "block1"),
        ("noise_b2", "block2"),
        ("noise_b3", "block3"),
    ]

    results = []
    for tag, position in variants:
        print(f"\n=== Training: {tag} ===")
        C.set_seed(0)
        model = C.build_model("cnn",
                              {"channels": 1, "size": 28, "n_classes": 10},
                              width=32, seed=0)
        handles = []
        if position is not None:
            handles = attach_noise_hook(model, position, SIGMA)

        t0 = time.time()
        C.train_model(model, Xtr, Ytr, epochs=EPOCHS)
        elapsed = time.time() - t0

        for h in handles:
            h.remove()

        print(f"  Training time: {elapsed:.1f}s")
        res = evaluate_model(model, Xte, Yte, tag=tag)
        res["train_time_s"] = elapsed
        results.append(res)

    # Summary table
    col = 12
    header = (f"{'Variant':<{col}} {'CleanAcc':>9} {'FGSM_ASR':>9} "
              f"{'PGD_ASR':>8} {'Margin':>8} {'GradNorm':>9}")
    sep = "-" * len(header)
    rows = []
    for r in results:
        rows.append(
            f"{r['tag']:<{col}} {r['clean_acc']:>9.4f} {r['fgsm_asr']:>9.4f} "
            f"{r['pgd_asr']:>8.4f} {r['mean_margin']:>8.4f} {r['grad_norm']:>9.4f}"
        )

    table = "\n".join([header, sep] + rows)
    print("\n\n" + table)

    base = results[0]
    finding_lines = [table, "\n\nKey Findings (delta vs baseline):"]
    for r in results[1:]:
        finding_lines.append(
            f"  {r['tag']}: ΔFGSM_ASR={r['fgsm_asr']-base['fgsm_asr']:+.4f}  "
            f"ΔPGD_ASR={r['pgd_asr']-base['pgd_asr']:+.4f}  "
            f"ΔGradNorm={r['grad_norm']-base['grad_norm']:+.4f}  "
            f"ΔMargin={r['mean_margin']-base['mean_margin']:+.4f}"
        )

    output = "\n".join(finding_lines)
    print(output)

    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
