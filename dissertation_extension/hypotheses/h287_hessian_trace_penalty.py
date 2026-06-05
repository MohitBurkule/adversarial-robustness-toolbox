"""
H287 - Hutchinson Hessian Trace Penalty for Adversarial Robustness.

Well-established regulariser (Niu et al. 2022) but not previously evaluated
for adversarial robustness on Fashion-MNIST.

    L_total = L_CE(x, y) + λ · v^T H v

where v is a random Rademacher vector and v^T H v is estimated via the
Hutchinson trick using a single Hessian-vector product (create_graph=True).

Implementation:
  1. Compute loss; get first-order gradients with create_graph=True
  2. Sample Rademacher vector v (same shape as params)
  3. Compute g·v (scalar), then differentiate w.r.t. params → Hv
  4. Hutchinson estimate: sum(v_i * Hv_i) ≈ trace(H) / n_params
  5. total_loss = loss + lam * hess_trace_est → .backward() → opt.step()

λ grid: {0.0, 0.0001, 0.001, 0.01}, N_train=6000, 10 epochs (expensive).
Metrics: clean_acc, FGSM_ASR, PGD_ASR, mean_margin, training time per epoch.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
LAMBDAS = [0.0, 0.0001, 0.001, 0.01]
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH_SIZE = 128
SEED = 0
EPS_FGSM = 0.1
EPS_PGD = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist"
)
OUT_FILE = os.path.join(RESULTS_DIR, "h287_hessian_trace_penalty_output.txt")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_with_hessian_trace_penalty(model, Xtr, Ytr, epochs, lam, lr=LR):
    """SGD training with Hutchinson Hessian trace penalty.

    Each step:
      1. Compute CE loss → get grads with create_graph=True
      2. Sample Rademacher v; compute g·v; differentiate → Hv
      3. hess_trace_est = Σ v_i * Hv_i  (in-graph scalar)
      4. total_loss = loss + lam * hess_trace_est → backward → step
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    n = len(Xtr)
    params = list(model.parameters())
    epoch_times = []
    hess_estimates_all = []

    for ep in range(epochs):
        t0 = time.time()
        perm = torch.randperm(n)
        ep_hess = []

        for i in range(0, n - BATCH_SIZE, BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            xb = Xtr[idx].to(C.DEVICE)
            yb = Ytr[idx].to(C.DEVICE)

            opt.zero_grad()

            if lam == 0.0:
                # Standard CE training
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                ep_hess.append(0.0)
            else:
                # Single forward with create_graph=True for Hutchinson HVP
                loss = F.cross_entropy(model(xb), yb)
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
                v_list = [
                    torch.randint(0, 2, g.shape, device=g.device).float() * 2 - 1
                    for g in grads
                ]
                gv = sum((g * v).sum() for g, v in zip(grads, v_list))
                # retain_graph=True so loss graph survives for total.backward()
                Hvp = torch.autograd.grad(gv, params, retain_graph=True)
                hess_trace_est = sum(
                    (v * hv).sum() for v, hv in zip(v_list, Hvp)
                )
                ep_hess.append(hess_trace_est.item())
                total = loss + lam * hess_trace_est
                total.backward()

            opt.step()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        hess_estimates_all.extend(ep_hess)

        mean_hess = float(np.mean(ep_hess)) if ep_hess else 0.0
        print(
            f"  ep {ep+1:02d}/{epochs}  "
            f"mean_hess_trace={mean_hess:.4f}  "
            f"epoch_time={elapsed:.1f}s  lam={lam}"
        )

    return epoch_times, hess_estimates_all


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_model(model, Xte, Yte):
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
    idx = torch.randperm(len(Xtr))[:N_TRAIN]
    Xtr, Ytr = Xtr[idx], Ytr[idx]

    results = []

    for lam in LAMBDAS:
        print(f"\n=== λ={lam} ===")
        C.set_seed(SEED)
        model = C.build_model(
            "cnn", {"channels": 1, "size": 28, "n_classes": 10}, width=32, seed=SEED
        )
        model.to(C.DEVICE)

        t_start = time.time()
        epoch_times, hess_ests = train_with_hessian_trace_penalty(
            model, Xtr, Ytr, epochs=EPOCHS, lam=lam
        )
        total_time = time.time() - t_start

        metrics = eval_model(model, Xte, Yte)
        metrics["lam"] = lam
        metrics["total_train_time"] = total_time
        metrics["mean_epoch_time"] = float(np.mean(epoch_times))
        metrics["mean_hess_trace_est"] = float(np.mean(np.abs(hess_ests))) if hess_ests else 0.0

        results.append(metrics)
        print(
            f"  clean_acc={metrics['clean_acc']:.4f}  "
            f"fgsm_asr={metrics['fgsm_asr']:.4f}  "
            f"pgd_asr={metrics['pgd_asr']:.4f}  "
            f"mean_margin={metrics['mean_margin']:.4f}  "
            f"mean_epoch_time={metrics['mean_epoch_time']:.1f}s"
        )

    # Write results
    lines = [
        "H287 Hutchinson Hessian Trace Penalty — Results",
        "=" * 60,
        f"Dataset: {DS}  N_train={N_TRAIN}  Epochs={EPOCHS}  Seed={SEED}",
        f"EPS_FGSM={EPS_FGSM}  EPS_PGD={EPS_PGD}  PGD_STEPS={PGD_STEPS}",
        "",
    ]

    header = (
        f"{'lam':>8}  {'clean_acc':>10}  {'fgsm_asr':>9}  {'pgd_asr':>8}  "
        f"{'mean_margin':>12}  {'hess_trace_est':>15}  {'mean_ep_s':>9}  {'total_s':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        lines.append(
            f"{r['lam']:>8.5f}  {r['clean_acc']:>10.4f}  {r['fgsm_asr']:>9.4f}  "
            f"{r['pgd_asr']:>8.4f}  {r['mean_margin']:>12.4f}  "
            f"{r['mean_hess_trace_est']:>15.4f}  {r['mean_epoch_time']:>9.1f}  "
            f"{r['total_train_time']:>7.1f}"
        )

    lines += [
        "",
        "Analysis:",
        "- λ=0.0 is standard SGD baseline",
        "- Lower fgsm_asr/pgd_asr with higher λ → Hessian penalty improves robustness",
        "- Mean hess_trace_est should decrease with higher λ (flatter minimum achieved)",
        "- Note: create_graph=True is expensive; expect 3–5× slower training than baseline",
        "- Compare epoch times across λ to quantify cost",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
