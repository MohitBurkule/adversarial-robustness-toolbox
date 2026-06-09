"""
H270 - Does Manifold Mixup (hidden layer interpolation) improve adversarial
robustness more than input-space Mixup?

Four model variants:
  - baseline       : standard cross-entropy training
  - input_mixup    : Mixup in pixel space (λ ~ Beta(0.4, 0.4))
  - manifold_mixup : at each batch pick a random layer from
                     {input, block1_out, block2_out, block3_out}, interpolate
                     activations with λ ~ Beta(0.4, 0.4), continue forward
                     from that point, use soft/interpolated labels
  - adv_mixup      : Mixup between clean and FGSM examples in input space

Metrics: clean_acc, FGSM_ASR, PGD_ASR, mean_margin

Key question: does manifold mixup beat input mixup for FGSM robustness?
Does any variant beat the baseline for PGD?
"""
import os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign import common as C


ALPHA       = 0.4
EPOCHS      = 10
BATCH_SIZE  = 128
RESULTS_DIR = os.path.join(os.path.dirname(__file__),
                           "..", "results", "fashion_mnist")


# ---------------------------------------------------------------------------
# Mixup helpers
# ---------------------------------------------------------------------------

def mixup_data(x, y_onehot, alpha=0.4):
    """Standard input-space mixup. Returns mixed_x, mixed_y (soft labels)."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return mixed_x, mixed_y


def soft_cross_entropy(logits, soft_labels):
    """Cross-entropy with soft targets."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def train_baseline(model, Xtr, Ytr, epochs=EPOCHS):
    model.to(C.DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds  = TensorDataset(Xtr, Ytr)
    dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()


def train_input_mixup(model, Xtr, Ytr, epochs=EPOCHS, alpha=ALPHA):
    model.to(C.DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_cls = 10
    ds = TensorDataset(Xtr, Ytr)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
            yb_oh = F.one_hot(yb, n_cls).float()
            mixed_x, mixed_y = mixup_data(xb, yb_oh, alpha)
            opt.zero_grad()
            soft_cross_entropy(model(mixed_x), mixed_y).backward()
            opt.step()


def forward_to_block(model, x, block_idx):
    """
    Run forward pass through model.features up to (and including) block_idx.
    block_idx: -1 = before any block (raw input), 0,1,2 = after block 0,1,2.
    Returns activations tensor.
    """
    if block_idx == -1:
        return x
    out = x
    for i in range(block_idx + 1):
        out = model.features[i](out)
    return out


def forward_from_block(model, act, block_idx):
    """
    Continue forward from activations produced after block_idx.
    Runs remaining feature blocks then the head.
    """
    out = act
    for i in range(block_idx + 1, len(model.features)):
        out = model.features[i](out)
    out = model.head(out)
    return out


def train_manifold_mixup(model, Xtr, Ytr, epochs=EPOCHS, alpha=ALPHA):
    """
    At each batch pick a random mixing layer from {input(-1), block0, block1, block2}.
    """
    model.to(C.DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_cls = 10
    ds = TensorDataset(Xtr, Ytr)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    layer_choices = [-1, 0, 1, 2]   # -1 = input space
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
            yb_oh = F.one_hot(yb, n_cls).float()
            mix_layer = random.choice(layer_choices)

            # Forward to chosen layer
            act = forward_to_block(model, xb, mix_layer)

            # Mixup in that space
            lam = float(np.random.beta(alpha, alpha))
            idx = torch.randperm(act.size(0), device=act.device)
            mixed_act = lam * act + (1 - lam) * act[idx]
            mixed_y   = lam * yb_oh + (1 - lam) * yb_oh[idx]

            # Continue forward from mixed activations
            logits = forward_from_block(model, mixed_act, mix_layer)

            opt.zero_grad()
            soft_cross_entropy(logits, mixed_y).backward()
            opt.step()


def train_adv_mixup(model, Xtr, Ytr, epochs=EPOCHS, alpha=ALPHA,
                    fgsm_eps=0.1):
    """
    Mixup between clean and FGSM adversarial examples in input space.
    """
    model.to(C.DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_cls = 10
    ds = TensorDataset(Xtr, Ytr)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
            yb_oh = F.one_hot(yb, n_cls).float()

            # Generate FGSM perturbation inline
            model.eval()
            xb_adv = xb.detach().requires_grad_(True)
            loss_adv = F.cross_entropy(model(xb_adv), yb)
            loss_adv.backward()
            xb_fgsm = (xb + fgsm_eps * xb_adv.grad.sign()).clamp(0, 1).detach()
            model.train()

            # Mixup clean vs FGSM
            lam = float(np.random.beta(alpha, alpha))
            mixed_x = lam * xb + (1 - lam) * xb_fgsm
            mixed_y = lam * yb_oh + (1 - lam) * yb_oh   # same labels

            opt.zero_grad()
            soft_cross_entropy(model(mixed_x), mixed_y).backward()
            opt.step()


# ---------------------------------------------------------------------------
# Evaluation helpers (shared with h269 pattern)
# ---------------------------------------------------------------------------

def compute_asr(model, X, Y, Xadv):
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

    print(f"  [{tag}]  clean_acc={clean_acc:.4f}  fgsm_asr={fgsm_asr:.4f}  "
          f"pgd_asr={pgd_asr:.4f}  mean_margin={mean_mg:.4f}")
    return dict(tag=tag, clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                pgd_asr=pgd_asr, mean_margin=mean_mg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            "h270_manifold_mixup_vs_input_mixup_output.txt")

    C.set_seed(0)
    print("Loading Fashion-MNIST …")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")

    train_fns = [
        ("baseline",       train_baseline),
        ("input_mixup",    train_input_mixup),
        ("manifold_mixup", train_manifold_mixup),
        ("adv_mixup",      train_adv_mixup),
    ]

    results = []
    for tag, train_fn in train_fns:
        print(f"\n=== Training: {tag} ===")
        C.set_seed(0)
        model = C.build_model("cnn",
                              {"channels": 1, "size": 28, "n_classes": 10},
                              width=32, seed=0)
        t0 = time.time()
        train_fn(model, Xtr, Ytr)
        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.1f}s")
        res = evaluate_model(model, Xte, Yte, tag=tag)
        res["train_time_s"] = elapsed
        results.append(res)

    # Summary table
    col = 16
    header = (f"{'Variant':<{col}} {'CleanAcc':>9} {'FGSM_ASR':>9} "
              f"{'PGD_ASR':>8} {'Margin':>8}")
    sep = "-" * len(header)
    rows = [f"{r['tag']:<{col}} {r['clean_acc']:>9.4f} {r['fgsm_asr']:>9.4f} "
            f"{r['pgd_asr']:>8.4f} {r['mean_margin']:>8.4f}"
            for r in results]
    table = "\n".join([header, sep] + rows)
    print("\n\n" + table)

    base = results[0]
    finding_lines = [table, "\n\nKey Findings (delta vs baseline):"]
    for r in results[1:]:
        finding_lines.append(
            f"  {r['tag']}: ΔFGSM_ASR={r['fgsm_asr']-base['fgsm_asr']:+.4f}  "
            f"ΔPGD_ASR={r['pgd_asr']-base['pgd_asr']:+.4f}  "
            f"ΔMargin={r['mean_margin']-base['mean_margin']:+.4f}"
        )

    # Specific questions
    mm  = next(r for r in results if r["tag"] == "manifold_mixup")
    im  = next(r for r in results if r["tag"] == "input_mixup")
    finding_lines.append(
        f"\nManifold Mixup vs Input Mixup — ΔFGSM_ASR={mm['fgsm_asr']-im['fgsm_asr']:+.4f}"
    )
    any_beats_pgd = any(r["pgd_asr"] < base["pgd_asr"] for r in results[1:])
    finding_lines.append(
        f"Any variant beats baseline PGD_ASR: {any_beats_pgd}"
    )

    output = "\n".join(finding_lines)
    print(output)

    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
