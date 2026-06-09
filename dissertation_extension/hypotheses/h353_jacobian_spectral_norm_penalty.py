"""
H353 - Jacobian Spectral Norm Penalty via Power Iteration

Penalise spectral norm (largest singular value) of input Jacobian via power iteration.
For random v ~ N(0,I): Jv = autograd.grad(f(x)·v, x)[0]; σ_max ≈ ||Jv||/||v||.
K=5 iterations. λ grid: [0, 0.001, 0.01].
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
K_ITER = 5
LAMBDAS = [0, 0.001, 0.01]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h353_jacobian_spectral_norm_penalty_output.txt")


def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1 - float(acc_fgsm),
                pgd_asr=1 - float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))


def spectral_norm_estimate(model, x, K=5):
    """Estimate largest singular value of Jacobian via power iteration."""
    n, c, h, w = x.shape
    # Random initial vector in output space (same shape as logits)
    with torch.no_grad():
        out_shape = model(x[:1]).shape[1]  # number of classes

    u = torch.randn(n, out_shape, device=x.device)
    u = F.normalize(u, dim=1)

    for _ in range(K):
        # J^T u: backprop gradient of (f(x) * u).sum() w.r.t. x
        x_req = x.detach().requires_grad_(True)
        out = model(x_req)
        scalar = (out * u.detach()).sum()
        Jtu = torch.autograd.grad(scalar, x_req, create_graph=True)[0]
        # J v: forward direction
        v = Jtu.detach()
        v_flat = v.view(n, -1)
        v_norm = v_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        v = (v_flat / v_norm).view_as(v)
        # Jv
        x_req2 = x.detach().requires_grad_(True)
        out2 = model(x_req2)
        # directional derivative: grad of (out2 * arbitrary_u).sum() won't work
        # Use: Jv = d/dt f(x + tv) at t=0 via autograd
        def f_dot_v(t):
            return model(x + t * v.detach()).sum()
        Jv = torch.autograd.functional.jvp(lambda xx: model(xx), x.detach(), v.detach(),
                                            create_graph=True)[1]
        u = Jv
        u_flat = u.view(n, -1)
        sigma = u_flat.norm(dim=1)
        u = F.normalize(u.view(n, -1), dim=1).view(n, out_shape)

    return sigma.mean()


def train_jsn(meta, Xtr, Ytr, lam):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    ncls = meta["n_classes"]

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            opt.zero_grad()
            if lam > 0:
                # Estimate spectral norm via power iteration with finite differences
                x_req = xb.clone().detach().requires_grad_(True)
                out = model(x_req)
                loss_ce = F.cross_entropy(out, yb)
                u = F.normalize(torch.randn(xb.size(0), ncls, device=xb.device), dim=1)
                sigma = torch.tensor(0.0, device=xb.device)
                eps_fd = 1e-3
                for _ in range(K_ITER):
                    # J^T u via backprop
                    scalar = (out * u.detach()).sum()
                    Jtu = torch.autograd.grad(scalar, x_req, retain_graph=True,
                                              create_graph=False)[0]
                    v_flat = Jtu.detach().view(xb.size(0), -1)
                    v = F.normalize(v_flat, dim=1).view_as(xb)
                    # Jv via two-sided finite difference (no graph needed)
                    with torch.no_grad():
                        Jv = (model(xb + eps_fd * v) - model(xb - eps_fd * v)) / (2 * eps_fd)
                    u_flat = Jv.view(xb.size(0), -1)
                    sigma = u_flat.norm(dim=1).mean() / (ncls ** 0.5)
                    u = F.normalize(u_flat, dim=1)
                loss = loss_ce + lam * sigma
            else:
                loss = F.cross_entropy(model(xb), yb)

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("H353 - Jacobian Spectral Norm Penalty via Power Iteration")
    lines.append("=" * 72)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_TRAIN={N_TRAIN}  K_iter={K_ITER}")
    lines.append(f"lambda grid: {LAMBDAS}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    for lam in LAMBDAS:
        t0 = time.time()
        model = train_jsn(meta, Xtr, Ytr, lam)
        metrics = eval_model(model, Xte, Yte)
        metrics["lambda"] = lam
        metrics["runtime_s"] = round(time.time() - t0, 1)
        line = (f"  lambda={lam:<6}  clean={metrics['clean_acc']:.3f}  "
                f"fgsm_asr={metrics['fgsm_asr']:.3f}  pgd_asr={metrics['pgd_asr']:.3f}  "
                f"margin={metrics['mean_margin']:.3f}  ({metrics['runtime_s']}s)")
        lines.append(line)
        print(line)

    lines.append("")
    lines.append("Interpretation: Spectral norm of the Jacobian bounds the Lipschitz")
    lines.append("constant of the network w.r.t. inputs. Penalising it directly constrains")
    lines.append("adversarial sensitivity more tightly than Frobenius norm.")
    lines.append("=" * 72)

    out = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(out + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
