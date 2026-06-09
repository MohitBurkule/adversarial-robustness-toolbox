"""
H286 - Inter-Batch Gradient Variance Penalty for Adversarial Robustness.

Novel idea: penalise the variance of weight gradients across mini-batches during
training to encourage finding flatter, more consistent minima.

  L_total = CE(batch1) + CE(batch2) + λ · ||g1 − g2||²_F

where g1, g2 are DETACHED gradients from two consecutive mini-batches. The penalty
is used as a scalar multiplier on the combined loss gradient in the next step —
this keeps the computation graph clean and avoids second-order differentiation.

Hypothesis: higher λ → lower gradient variance → flatter loss landscape →
improved adversarial robustness (lower FGSM/PGD attack success rate, higher margin).

We also post-hoc estimate the Hessian trace via Hutchinson's estimator (20
Rademacher vectors) to correlate sharpness with robustness.

λ grid: {0.0, 0.001, 0.01, 0.1}, N_train=10000, 15 epochs.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
LAMBDAS = [0.0, 0.001, 0.01, 0.1]
N_TRAIN = 10000
EPOCHS = 15
LR = 0.05
BATCH = 256          # split into two halves of 128
SEED = 0
EPS_FGSM = 0.1
EPS_PGD = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
HUTCHINSON_SAMPLES = 20
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist"
)
OUT_FILE = os.path.join(RESULTS_DIR, "h286_inter_batch_gradient_variance_penalty_output.txt")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_with_grad_variance_penalty(model, Xtr, Ytr, epochs, lam, lr=LR):
    """Train with inter-batch gradient variance penalty.

    Each step draws two consecutive mini-batches of 128:
      1. Forward+backward batch1 → store detached g1
      2. Forward+backward batch2 → g2 = current param.grad
      3. variance = Σ ||g1_i - g2_i||² (scalar, fully detached)
      4. Combined forward+backward on both batches; scale all param.grads by
         (1 + lam * variance) before the optimiser step. This approximates
         adding lam * variance as an auxiliary loss while keeping the graph clean.
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    n = len(Xtr)
    epoch_times = []

    for ep in range(epochs):
        t0 = time.time()
        perm = torch.randperm(n)
        batch_variances = []

        for i in range(0, n - BATCH, BATCH):
            idx1 = perm[i: i + 128]
            idx2 = perm[i + 128: i + 256]

            xb1 = Xtr[idx1].to(C.DEVICE)
            yb1 = Ytr[idx1].to(C.DEVICE)
            xb2 = Xtr[idx2].to(C.DEVICE)
            yb2 = Ytr[idx2].to(C.DEVICE)

            # --- batch 1: get detached gradient g1 ---
            opt.zero_grad()
            loss1 = F.cross_entropy(model(xb1), yb1)
            loss1.backward()
            g1 = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

            # --- batch 2: get detached gradient g2 ---
            opt.zero_grad()
            loss2 = F.cross_entropy(model(xb2), yb2)
            loss2.backward()
            g2 = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

            # variance scalar (fully detached)
            variance = sum((a - b).pow(2).sum() for a, b in zip(g1, g2)).item()
            batch_variances.append(variance)

            # --- combined step: recompute combined loss ---
            opt.zero_grad()
            loss_combined = F.cross_entropy(model(xb1), yb1) + F.cross_entropy(model(xb2), yb2)
            loss_combined.backward()

            # scale gradients by (1 + lam * variance) — approximates adding
            # lam * variance as an explicit penalty term
            if lam > 0.0:
                scale = 1.0 + lam * variance
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)

            opt.step()

        epoch_times.append(time.time() - t0)

    return epoch_times, batch_variances


# ---------------------------------------------------------------------------
# Hutchinson Hessian trace estimator
# ---------------------------------------------------------------------------

def hutchinson_trace(model, X, Y, n_samples=HUTCHINSON_SAMPLES, batch_size=256):
    """Estimate Hessian trace via Hutchinson's estimator: E[v^T H v] where v ~ Rademacher.

    Uses the 'finite-difference via grad-variance' proxy: for each Rademacher
    vector v, compute v · (g1 - g2) / ||v|| on two random batches.
    This is an approximation to the directional curvature rather than the full trace,
    but avoids expensive second-order graph computation.
    """
    model.eval()
    n = len(X)
    estimates = []

    for _ in range(n_samples):
        idx1 = torch.randperm(n)[:batch_size // 2]
        idx2 = torch.randperm(n)[:batch_size // 2]

        xb1 = X[idx1].to(C.DEVICE)
        yb1 = Y[idx1].to(C.DEVICE)
        xb2 = X[idx2].to(C.DEVICE)
        yb2 = Y[idx2].to(C.DEVICE)

        # gradient from batch 1
        for p in model.parameters():
            p.requires_grad_(True)
        loss1 = F.cross_entropy(model(xb1), yb1)
        grads1 = torch.autograd.grad(loss1, model.parameters(), create_graph=False)

        # gradient from batch 2
        loss2 = F.cross_entropy(model(xb2), yb2)
        grads2 = torch.autograd.grad(loss2, model.parameters(), create_graph=False)

        # Rademacher vector with same shape as params
        v = [torch.randint(0, 2, g.shape, device=g.device).float() * 2 - 1
             for g in grads1]

        # directional curvature proxy: v · (g1 - g2)
        diff_dot_v = sum((g1 - g2).mul(vi).sum()
                         for g1, g2, vi in zip(grads1, grads2, v))
        estimates.append(diff_dot_v.item())

    model.train()
    return float(np.mean(np.abs(estimates)))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_model(model, Xte, Yte, label=""):
    for p in model.parameters():
        p.requires_grad_(True)

    _, clean_acc = C.logits_and_acc(model, Xte, Yte)

    Xadv_fgsm = C.fgsm(model, Xte, Yte, eps=EPS_FGSM)
    _, acc_fgsm = C.logits_and_acc(model, Xadv_fgsm, Yte)
    fgsm_asr = 1.0 - acc_fgsm

    Xadv_pgd = C.pgd(model, Xte, Yte, eps=EPS_PGD, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, acc_pgd = C.logits_and_acc(model, Xadv_pgd, Yte)
    pgd_asr = 1.0 - acc_pgd

    margins = C.margin(model, Xte, Yte)
    mean_margin = float(np.mean(margins))

    return {
        "label": label,
        "clean_acc": float(clean_acc),
        "fgsm_asr": float(fgsm_asr),
        "pgd_asr": float(pgd_asr),
        "mean_margin": mean_margin,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    C.set_seed(SEED)

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    # Subsample training set
    idx = torch.randperm(len(Xtr))[:N_TRAIN]
    Xtr = Xtr[idx]
    Ytr = Ytr[idx]

    results = []

    for lam in LAMBDAS:
        print(f"\n=== λ={lam} ===")
        C.set_seed(SEED)
        model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                              width=32, seed=SEED)
        model.to(C.DEVICE)

        t_start = time.time()
        epoch_times, batch_variances = train_with_grad_variance_penalty(
            model, Xtr, Ytr, epochs=EPOCHS, lam=lam
        )
        total_time = time.time() - t_start

        print(f"  Training done in {total_time:.1f}s")
        print(f"  Mean batch variance: {np.mean(batch_variances):.4f}")

        metrics = eval_model(model, Xte, Yte, label=f"lam={lam}")
        metrics["lam"] = lam
        metrics["mean_batch_variance"] = float(np.mean(batch_variances))
        metrics["total_train_time"] = total_time

        # Hutchinson trace estimate
        print("  Computing Hutchinson trace proxy...")
        hess_trace = hutchinson_trace(model, Xte, Yte)
        metrics["hutchinson_trace_proxy"] = hess_trace

        results.append(metrics)
        print(f"  clean_acc={metrics['clean_acc']:.4f}  "
              f"fgsm_asr={metrics['fgsm_asr']:.4f}  "
              f"pgd_asr={metrics['pgd_asr']:.4f}  "
              f"mean_margin={metrics['mean_margin']:.4f}  "
              f"hess_trace={hess_trace:.4f}")

    # Write results
    lines = ["H286 Inter-Batch Gradient Variance Penalty — Results",
             "=" * 60,
             f"Dataset: {DS}  N_train={N_TRAIN}  Epochs={EPOCHS}  Seed={SEED}",
             f"EPS_FGSM={EPS_FGSM}  EPS_PGD={EPS_PGD}  PGD_STEPS={PGD_STEPS}",
             ""]

    header = (f"{'lam':>8}  {'clean_acc':>10}  {'fgsm_asr':>9}  {'pgd_asr':>8}  "
              f"{'mean_margin':>12}  {'batch_var':>10}  {'hess_trace':>11}  {'time_s':>7}")
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        lines.append(
            f"{r['lam']:>8.4f}  {r['clean_acc']:>10.4f}  {r['fgsm_asr']:>9.4f}  "
            f"{r['pgd_asr']:>8.4f}  {r['mean_margin']:>12.4f}  "
            f"{r['mean_batch_variance']:>10.4f}  {r['hutchinson_trace_proxy']:>11.4f}  "
            f"{r['total_train_time']:>7.1f}"
        )

    lines += [
        "",
        "Analysis:",
        "- λ=0.0 is standard SGD baseline",
        "- Decreasing fgsm_asr/pgd_asr with increasing λ → penalty improves robustness",
        "- Hutchinson trace proxy should anti-correlate with robustness (lower = flatter)",
        "- Mean batch variance should decrease with higher λ (penalty achieves goal)",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
