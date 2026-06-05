"""
H282 - Gradient sign consistency and FGSM effectiveness.

FGSM uses sign(∇_x L). If gradient signs are consistent across samples of the
same class (all pointing the same direction), the signed gradient is highly
structured and FGSM is effective. If signs are random/inconsistent, FGSM is
less effective.

Measure per-pixel sign consistency:
    sign_consistency(p) = |mean(sign(∇_x L)[:, p])| ∈ [0, 1]

Then:
  - Mean sign consistency per class and over the whole test set
  - Correlation between per-sample sign consistency and FGSM success
  - Comparison between standard model and PGD-adversarially-trained model

Hypothesis: adversarially trained models have LOWER sign consistency (gradient
signs are less predictable, making sign-based FGSM less effective).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS        = "fashion_mnist"
N_TRAIN   = 6000
N_EVAL    = 300
EPOCHS    = 10
EPS       = 0.1
PGD_STEPS = 10
SEED      = 0

OUT_PATH  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h282_gradient_sign_entropy_output.txt"
)


def input_gradients(model, X, Y, batch=64):
    """Return sign(∇_x L) as CPU tensor (N, D) and raw ∇_x L (N, D)."""
    signs, raw = [], []
    N = X.shape[0]
    model.eval()
    for i in range(0, N, batch):
        xb = X[i: i + batch].clone().detach().to(C.DEVICE).requires_grad_(True)
        yb = Y[i: i + batch].to(C.DEVICE)
        logits = model(xb)
        loss   = F.cross_entropy(logits, yb, reduction="sum")
        g      = torch.autograd.grad(loss, xb)[0].detach().cpu()
        raw.append(g.reshape(g.shape[0], -1))
        signs.append(g.sign().reshape(g.shape[0], -1))
    return torch.cat(signs, dim=0), torch.cat(raw, dim=0)


def per_sample_sign_consistency(signs):
    """
    signs: (N, D) tensor of {-1, 0, +1}
    Per-sample sign consistency = mean over pixels of |sign_i|
    (fraction of pixels that have a definite non-zero sign).
    Just returns |signs|.mean(dim=1) as a proxy.
    Actually we want consistency ACROSS samples, so we compute mean sign per
    pixel and average the absolute value.
    But for per-sample, we measure how much that sample's sign pattern
    aligns with the class mean sign pattern.
    """
    # class-mean sign for each pixel: (D,)
    mean_sign = signs.float().mean(dim=0)  # (D,)
    # alignment of each sample with the class mean
    align = (signs.float() * mean_sign).mean(dim=1)  # (N,)
    return align.numpy()


def train_pgd_at(model, Xtr, Ytr, epochs=EPOCHS, eps=EPS, steps=PGD_STEPS,
                 batch=128):
    """Simple PGD adversarial training."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N   = Xtr.shape[0]
    model.train()
    for ep in range(epochs):
        perm   = torch.randperm(N)
        ep_loss = 0.0
        for i in range(0, N, batch):
            idx = perm[i: i + batch]
            xb  = Xtr[idx].to(C.DEVICE)
            yb  = Ytr[idx].to(C.DEVICE)
            # generate adversarial examples
            xadv = C.pgd(model, xb, yb, eps=eps, steps=steps)
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(xadv), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    AT ep {ep+1}/{epochs}  loss={ep_loss / max(1, N // batch):.4f}")
    model.eval()


def analyse_model(model, Xte, Yte, label, lines):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    signs, grads = input_gradients(model, Xte, Yte)  # (N, D)
    N_CLASSES = 10
    D = signs.shape[1]

    # ── global sign consistency ───────────────────────────────────────────────
    global_mean_sign = signs.float().mean(dim=0)          # (D,)
    global_sc        = global_mean_sign.abs().mean().item()

    lines.append(f"\n  [{label}] global sign consistency: {global_sc:.4f}")

    # ── per-class sign consistency ────────────────────────────────────────────
    lines.append(f"  [{label}] per-class sign consistency:")
    for c in range(N_CLASSES):
        mask = (Yte.cpu() == c)
        if mask.sum() < 2:
            continue
        sc_c = signs[mask].float().mean(dim=0).abs().mean().item()
        lines.append(f"    class {c}: {sc_c:.4f}  (n={mask.sum().item()})")

    # ── FGSM success vs sign consistency ─────────────────────────────────────
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    with torch.no_grad():
        fgsm_pred = model(Xfgsm).argmax(1).cpu()
    fgsm_ok = (fgsm_pred != Yte.cpu()).float().numpy()  # 1=success

    # per-sample alignment with class-mean sign
    align = per_sample_sign_consistency(signs)

    # spearman correlation between alignment and fgsm success
    from scipy import stats as sp_stats
    rho, pval = sp_stats.spearmanr(align, fgsm_ok)
    lines.append(f"  [{label}] Spearman rho(sign_alignment, FGSM_success)="
                 f"{rho:.4f}  p={pval:.4f}")
    lines.append(f"  [{label}] FGSM ASR overall: {fgsm_ok.mean():.4f}")

    # mean alignment for FGSM-vulnerable vs robust
    vuln_align = align[fgsm_ok == 1].mean() if (fgsm_ok == 1).any() else float("nan")
    rob_align  = align[fgsm_ok == 0].mean() if (fgsm_ok == 0).any() else float("nan")
    lines.append(f"  [{label}] mean sign_alignment: vulnerable={vuln_align:.4f}  "
                 f"robust={rob_align:.4f}")

    return global_sc


def main():
    C.set_seed(SEED)
    lines = []
    lines.append("=" * 74)
    lines.append("H282 - Gradient sign consistency and FGSM effectiveness")
    lines.append("=" * 74)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_train={N_TRAIN}  "
                 f"N_eval={N_EVAL}  eps={EPS}")

    t0 = time.time()
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    Xtr, Ytr = Xtr[:N_TRAIN], Ytr[:N_TRAIN]
    Xte, Yte = Xte[:N_EVAL].to(C.DEVICE), Yte[:N_EVAL].to(C.DEVICE)

    # ── standard model ────────────────────────────────────────────────────────
    print("\nTraining standard model ...")
    C.set_seed(SEED)
    std_model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, seed=SEED)
    C.train_model(std_model, Xtr, Ytr, epochs=EPOCHS)
    std_model.eval()

    # ── PGD-AT model ─────────────────────────────────────────────────────────
    print("\nTraining PGD-AT model ...")
    C.set_seed(SEED)
    at_model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, seed=SEED)
    at_model.to(C.DEVICE)
    train_pgd_at(at_model, Xtr, Ytr, epochs=EPOCHS)
    at_model.eval()

    lines.append("\nSign consistency analysis:")
    sc_std = analyse_model(std_model, Xte, Yte, "standard", lines)
    sc_at  = analyse_model(at_model,  Xte, Yte, "PGD-AT",   lines)

    lines.append("\n" + "=" * 74)
    lines.append(f"Global sign consistency: standard={sc_std:.4f}  AT={sc_at:.4f}")
    lines.append(f"Total runtime: {time.time() - t0:.1f}s")
    lines.append("=" * 74)
    lines.append("Interpretation: if AT model has lower sign consistency, the model")
    lines.append("has learned gradients that are less sign-aligned across samples,")
    lines.append("making FGSM less effective (it moves in a less predictable direction).")
    lines.append("Positive Spearman rho = sign alignment predicts FGSM success.")
    lines.append("=" * 74)

    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
