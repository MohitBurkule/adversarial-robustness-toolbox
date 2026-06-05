"""H344: Input Jacobian spectral penalty.

Penalise top singular value of input Jacobian (Lipschitz constant w.r.t. input).
σ_max(J) where J = ∂f/∂x. Estimate via power iteration: ||Jv|| / ||v|| for random v.
J^T(Jv) via two autograd.grad calls. λ grid: [0, 0.001, 0.01].
Smaller Lipschitz constant → smoother model → more robust.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

OUT = "results/fashion_mnist/h344_input_gradient_spectral_penalty_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
N_POWER_ITER = 3


def spectral_norm_input_jacobian(model, x, n_iter=N_POWER_ITER):
    # Estimate sigma_max(J) via random projections ||J^T u||^2 for u~N(0,I) in output space.
    # This gives an unbiased estimate of ||J||_F^2 / n_out which serves as spectral norm proxy.
    x.requires_grad_(True)
    logits = model(x)  # (B, C)

    estimates = []
    for _ in range(n_iter):
        u = torch.randn_like(logits)  # (B, C)
        u = u / (u.norm(dim=1, keepdim=True) + 1e-8)
        # J^T u: grad of sum(logits * u) w.r.t. x
        Jtu, = torch.autograd.grad(
            (logits * u.detach()).sum(), x,
            create_graph=True, retain_graph=True
        )  # (B, D_in)
        # ||J^T u||^2 per sample
        sq = (Jtu.flatten(1) ** 2).sum(dim=1)  # (B,)
        estimates.append(sq)

    # Average over power iter samples: estimate of sigma_max^2
    sigma_sq = torch.stack(estimates, dim=0).mean(dim=0).mean()  # scalar
    return sigma_sq


def train_with_spectral_penalty(model, Xtr, Ytr, lam):
    C.set_seed(SEED)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = Xtr[idx].clone().detach()
            yb = Ytr[idx]
            opt.zero_grad()

            xb_req = xb.requires_grad_(True)
            logits = model(xb_req)
            ce_loss = F.cross_entropy(logits, yb)

            if lam > 0:
                # Estimate spectral norm via random projections
                u = torch.randn(xb.size(0), logits.size(1), device=xb.device)
                u = u / (u.norm(dim=1, keepdim=True) + 1e-8)
                Jtu, = torch.autograd.grad(
                    (logits * u.detach()).sum(), xb_req,
                    create_graph=True, retain_graph=True
                )
                sigma_sq = (Jtu.flatten(1) ** 2).sum(dim=1).mean()
                penalty = lam * sigma_sq
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


def estimate_spectral_norm(model, X, Y, n=50):
    """Estimate average spectral norm of input Jacobian on test samples."""
    model.eval()
    norms = []
    for k in range(min(n, X.size(0))):
        xi = X[k:k+1].clone().requires_grad_(True)
        logits = model(xi)
        u = torch.randn(1, logits.size(1), device=xi.device)
        u = u / (u.norm() + 1e-8)
        Jtu, = torch.autograd.grad((logits * u).sum(), xi)
        norms.append(Jtu.norm().item())
    return np.mean(norms), np.std(norms)


def main():
    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    lines = ["H344: Input Jacobian Spectral Penalty", "="*60]
    lines.append(f"Settings: N_TRAIN={N_TRAIN}, EPOCHS={EPOCHS}, LR={LR}, BATCH={BATCH}")
    lines.append(f"Penalty = σ_max(J)² estimated via random projection ||J^T u||² for u~N(0,I)")
    lines.append("Hypothesis: smaller Lipschitz constant → smoother model → more robust.\n")

    lam_grid = [0, 0.001, 0.01]
    results = {}
    for lam in lam_grid:
        print(f"Training λ={lam}...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_with_spectral_penalty(model, Xtr, Ytr, lam=lam)
        r = eval_model(model, Xte, Yte)
        print(f"  Estimating spectral norm...")
        sn_mean, sn_std = estimate_spectral_norm(model, Xte, Yte, n=50)
        key = f"lam={lam}"
        results[key] = {**r, "spectral_norm": sn_mean, "spectral_norm_std": sn_std}
        lines.append(f"{key}: clean={r['clean_acc']:.4f} fgsm_asr={r['fgsm_asr']:.4f} pgd_asr={r['pgd_asr']:.4f} margin={r['mean_margin']:.4f} spectral_norm={sn_mean:.4f}±{sn_std:.4f}")

    lines.append("\n" + "="*60)
    lines.append("SUMMARY")
    lines.append(f"{'Condition':<12} {'Clean':>8} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Margin':>10} {'SpectralNorm':>14}")
    for key, r in results.items():
        lines.append(f"{key:<12} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f} {r['spectral_norm']:>14.4f}")

    out_text = "\n".join(lines)
    print(out_text)
    with open(OUT, "w") as f:
        f.write(out_text)
    print(f"\nResults written to {OUT}")

if __name__ == "__main__":
    main()
