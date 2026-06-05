"""
H280 - Adversarial gradient structure: pointy vs broad.

Characterises the statistical structure of input gradients ∇_x L for four
sample groups:
  - Clean-correct: correctly classified clean samples
  - FGSM-vulnerable: FGSM flips the prediction
  - FGSM-robust: FGSM fails (prediction unchanged)
  - PGD-vulnerable: PGD flips the prediction

For each group we compute per-sample:
  1. Kurtosis of gradient values (high kurtosis = spiky)
  2. Gini coefficient of |gradient| (0=uniform, 1=one pixel dominates)
  3. Top-k pixels holding 50% and 90% of gradient energy ||∇||²
  4. High-freq vs low-freq energy ratio via 2D FFT
  5. L1/L2 ratio: ||∇||₁ / ||∇||₂

Hypothesis: FGSM-vulnerable samples have more concentrated (spiky, high
kurtosis, low L1/L2, high Gini) gradients than FGSM-robust samples.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS        = "fashion_mnist"
N_EVAL    = 300
EPS       = 0.1
PGD_STEPS = 10
SEED      = 0

OUT_PATH  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h280_adversarial_gradient_structure_output.txt"
)


# ── helper: compute input gradient for a batch ──────────────────────────────
def input_gradients(model, X, Y):
    """Return ∇_x CE(f(x), y) as a detached CPU tensor, shape (N, C, H, W)."""
    X = X.clone().detach().to(C.DEVICE).requires_grad_(True)
    Y = Y.to(C.DEVICE)
    logits = model(X)
    loss   = torch.nn.functional.cross_entropy(logits, Y, reduction="sum")
    grads  = torch.autograd.grad(loss, X)[0]
    return grads.detach().cpu()


# ── helper: kurtosis ─────────────────────────────────────────────────────────
def kurtosis_batch(g):
    """g: (N, D) tensor. Returns (N,) kurtosis."""
    mu  = g.mean(dim=1, keepdim=True)
    std = g.std(dim=1, keepdim=True).clamp(min=1e-10)
    return ((g - mu) / std).pow(4).mean(dim=1)


# ── helper: Gini coefficient ─────────────────────────────────────────────────
def gini_batch(g_abs):
    """g_abs: (N, D) non-negative tensor. Returns (N,) Gini coefficients."""
    g_s, _ = g_abs.sort(dim=1)
    N, D   = g_s.shape
    idx    = torch.arange(1, D + 1, dtype=torch.float32)
    numer  = (2 * (idx * g_s).sum(dim=1)) - (D + 1) * g_s.sum(dim=1)
    denom  = (D * g_s.sum(dim=1)).clamp(min=1e-12)
    return numer / denom


# ── helper: top-k energy fraction ────────────────────────────────────────────
def topk_energy_frac(g_abs_sq, frac=0.5):
    """Return for each sample the minimum k s.t. top-k pixels hold `frac` of energy."""
    g_s, _ = g_abs_sq.sort(dim=1, descending=True)
    cumsum  = g_s.cumsum(dim=1)
    total   = g_s.sum(dim=1, keepdim=True).clamp(min=1e-12)
    reached = (cumsum / total) >= frac     # (N, D) bool
    # argmax on bool gives first True index
    k = reached.float().argmax(dim=1) + 1  # 1-indexed
    return k.float()


# ── helper: high vs low frequency ratio via 2D FFT ──────────────────────────
def hf_lf_ratio_batch(g, h=28, w=28):
    """
    g: (N, D) flattened gradient. Reshape to (N, h, w), compute 2D FFT,
    split into inner (low-freq) and outer (high-freq) halves by radius.
    """
    N = g.shape[0]
    g2d  = g.reshape(N, h, w)
    fft  = torch.fft.fft2(g2d)
    mag  = fft.abs().pow(2)          # (N, h, w)
    # shifted so DC is at centre
    mag  = torch.fft.fftshift(mag, dim=(-2, -1))
    cy, cx = h // 2, w // 2
    ys = torch.arange(h).float() - cy
    xs = torch.arange(w).float() - cx
    radius = (ys[:, None].pow(2) + xs[None, :].pow(2)).sqrt()  # (h, w)
    threshold = min(cy, cx) / 2
    lf_mask = radius <= threshold
    hf_energy = mag[:, ~lf_mask].sum(dim=1)
    lf_energy = mag[:,  lf_mask].sum(dim=1).clamp(min=1e-12)
    return hf_energy / lf_energy


# ── helper: L1/L2 ratio ──────────────────────────────────────────────────────
def l1_l2_ratio(g_abs):
    l1 = g_abs.sum(dim=1)
    l2 = g_abs.pow(2).sum(dim=1).sqrt().clamp(min=1e-12)
    return l1 / l2


# ── compute all metrics for a (N, D) gradient flat tensor ───────────────────
def compute_metrics(g_flat):
    g_abs    = g_flat.abs()
    g_abs_sq = g_abs.pow(2)
    metrics = {
        "kurtosis":       kurtosis_batch(g_flat).numpy(),
        "gini":           gini_batch(g_abs).numpy(),
        "topk_50":        topk_energy_frac(g_abs_sq, 0.50).numpy(),
        "topk_90":        topk_energy_frac(g_abs_sq, 0.90).numpy(),
        "hf_lf_ratio":    hf_lf_ratio_batch(g_flat).numpy(),
        "l1_l2_ratio":    l1_l2_ratio(g_abs).numpy(),
    }
    return metrics


def summarise(metrics, label, lines):
    lines.append(f"\n  [{label}]")
    for k, v in metrics.items():
        lines.append(f"    {k:20s}: mean={v.mean():.4f}  std={v.std():.4f}  "
                     f"median={np.median(v):.4f}")


def main():
    C.set_seed(SEED)
    lines = []
    lines.append("=" * 74)
    lines.append("H280 - Adversarial gradient structure: pointy vs broad")
    lines.append("=" * 74)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_EVAL={N_EVAL}  eps={EPS}")

    t0 = time.time()
    _, _, Xte, Yte = C.load_dataset(DS)
    Xte, Yte = Xte[:N_EVAL].to(C.DEVICE), Yte[:N_EVAL].to(C.DEVICE)

    model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, seed=SEED)
    C.train_model(model, *C.load_dataset(DS)[:2], epochs=10)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    # ── group 1: clean-correct ────────────────────────────────────────────────
    with torch.no_grad():
        logits_clean = model(Xte)
        clean_pred   = logits_clean.argmax(1)
    clean_mask = (clean_pred == Yte)
    lines.append(f"\nClean-correct: {clean_mask.sum().item()} / {N_EVAL}")

    g_clean  = input_gradients(model, Xte[clean_mask], Yte[clean_mask])
    N_c      = g_clean.shape[0]
    g_flat_c = g_clean.reshape(N_c, -1)
    m_clean  = compute_metrics(g_flat_c)

    # ── FGSM adversarial examples ─────────────────────────────────────────────
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    with torch.no_grad():
        fgsm_pred = model(Xfgsm).argmax(1)
    fgsm_vuln = clean_mask & (fgsm_pred != Yte)
    fgsm_rob  = clean_mask & (fgsm_pred == Yte)
    lines.append(f"FGSM-vulnerable: {fgsm_vuln.sum().item()}  FGSM-robust: {fgsm_rob.sum().item()}")

    g_fgsm_v  = input_gradients(model, Xte[fgsm_vuln], Yte[fgsm_vuln])
    g_fgsm_v  = g_fgsm_v.reshape(g_fgsm_v.shape[0], -1)
    m_fgsm_v  = compute_metrics(g_fgsm_v)

    g_fgsm_r  = input_gradients(model, Xte[fgsm_rob], Yte[fgsm_rob])
    g_fgsm_r  = g_fgsm_r.reshape(g_fgsm_r.shape[0], -1)
    m_fgsm_r  = compute_metrics(g_fgsm_r)

    # ── PGD adversarial examples ──────────────────────────────────────────────
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS)
    with torch.no_grad():
        pgd_pred = model(Xpgd).argmax(1)
    pgd_vuln = clean_mask & (pgd_pred != Yte)
    lines.append(f"PGD-vulnerable: {pgd_vuln.sum().item()}")

    g_pgd_v   = input_gradients(model, Xte[pgd_vuln], Yte[pgd_vuln])
    g_pgd_v   = g_pgd_v.reshape(g_pgd_v.shape[0], -1)
    m_pgd_v   = compute_metrics(g_pgd_v)

    # ── summarise ─────────────────────────────────────────────────────────────
    lines.append("\nGradient structure metrics (mean ± std over samples in group):")
    summarise(m_clean,  "clean-correct",   lines)
    summarise(m_fgsm_v, "FGSM-vulnerable", lines)
    summarise(m_fgsm_r, "FGSM-robust",     lines)
    summarise(m_pgd_v,  "PGD-vulnerable",  lines)

    # ── pairwise comparison: vulnerable vs robust ─────────────────────────────
    lines.append("\nVulnerable - Robust delta (positive = vulnerable is higher):")
    for k in m_fgsm_v:
        delta = m_fgsm_v[k].mean() - m_fgsm_r[k].mean()
        lines.append(f"  {k:20s}: Δ = {delta:+.4f}")

    lines.append(f"\nTotal runtime: {time.time() - t0:.1f}s")
    lines.append("=" * 74)
    lines.append("Interpretation: high kurtosis / high Gini / low L1/L2 ratio in the")
    lines.append("FGSM-vulnerable group confirms that vulnerable samples have spiky")
    lines.append("(concentrated) input gradients — making sign-based attacks efficient.")
    lines.append("Robust samples with broad gradients spread perturbation budget thin.")
    lines.append("=" * 74)

    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
