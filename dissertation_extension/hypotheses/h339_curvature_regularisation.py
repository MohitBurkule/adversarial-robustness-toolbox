"""H339: Curvature regularisation via input Hessian trace (Hutchinson).

Penalise curvature of loss surface w.r.t. input x (not weights).
κ = Tr(H_x L) estimated by Hutchinson: sample v ~ Rademacher, compute v^T H_x L v.
H_x is the Hessian of L w.r.t. input x.
λ grid: [0, 0.001, 0.01].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

OUT = "results/fashion_mnist/h339_curvature_regularisation_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0


def hutchinson_hessian_trace(loss, x, n_samples=1):
    """Estimate Tr(H_x L) using Hutchinson's estimator.

    For each Rademacher vector v: v^T H v = v^T ∇_x (∇_x L · v)
    """
    g, = torch.autograd.grad(loss, x, create_graph=True)
    traces = []
    for _ in range(n_samples):
        v = torch.randint_like(x, 0, 2).float() * 2 - 1  # Rademacher
        # ∇_x (g · v) = H_x v
        gv = (g * v).sum()
        hv, = torch.autograd.grad(gv, x, retain_graph=True, create_graph=False)
        traces.append((hv * v).sum())
    return torch.stack(traces).mean()


def train_with_curvature_penalty(model, Xtr, Ytr, lam):
    C.set_seed(SEED)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = Xtr[idx].clone().detach().requires_grad_(True)
            yb = Ytr[idx]
            opt.zero_grad()
            logits = model(xb)
            ce_loss = F.cross_entropy(logits, yb)
            if lam > 0:
                # Hutchinson trace estimate
                trace = hutchinson_hessian_trace(ce_loss, xb, n_samples=1)
                # Penalise positive curvature (trace can be negative)
                penalty = lam * trace
                total_loss = ce_loss + penalty
            else:
                total_loss = ce_loss
            total_loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1-float(acc_fgsm), pgd_asr=1-float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))


def estimate_input_curvature(model, X, Y, n=50):
    """Estimate average input Hessian trace on test set."""
    model.eval()
    traces = []
    for k in range(min(n, X.size(0))):
        xi = X[k:k+1].clone().requires_grad_(True)
        yi = Y[k:k+1]
        loss = F.cross_entropy(model(xi), yi)
        try:
            t = hutchinson_hessian_trace(loss, xi, n_samples=3)
            traces.append(t.item())
        except Exception:
            pass
    return np.mean(traces) if traces else float('nan'), np.std(traces) if traces else float('nan')


def main():
    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    lines = ["H339: Curvature Regularisation (Input Hessian Trace)", "="*60]
    lines.append(f"Settings: N_TRAIN={N_TRAIN}, EPOCHS={EPOCHS}, LR={LR}, BATCH={BATCH}")
    lines.append("Curvature = Tr(H_x L) via Hutchinson estimator (v~Rademacher)\n")

    lam_grid = [0, 0.001, 0.01]
    results = {}
    for lam in lam_grid:
        print(f"Training λ={lam}...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_with_curvature_penalty(model, Xtr, Ytr, lam=lam)
        r = eval_model(model, Xte, Yte)
        print(f"  Estimating curvature...")
        curv_mean, curv_std = estimate_input_curvature(model, Xte, Yte, n=30)
        key = f"lam={lam}"
        results[key] = {**r, "curvature": curv_mean, "curvature_std": curv_std}
        lines.append(f"{key}: clean={r['clean_acc']:.4f} fgsm_asr={r['fgsm_asr']:.4f} pgd_asr={r['pgd_asr']:.4f} margin={r['mean_margin']:.4f} curvature={curv_mean:.4f}±{curv_std:.4f}")

    lines.append("\n" + "="*60)
    lines.append("SUMMARY")
    lines.append(f"{'Condition':<12} {'Clean':>8} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Margin':>10} {'Curvature':>12}")
    for key, r in results.items():
        lines.append(f"{key:<12} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f} {r['curvature']:>12.4f}")

    out_text = "\n".join(lines)
    print(out_text)
    with open(OUT, "w") as f:
        f.write(out_text)
    print(f"\nResults written to {OUT}")

if __name__ == "__main__":
    main()
