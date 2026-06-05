"""
H345 - ELLE: Efficient Local Linearity Enforcement (ICLR 2024)

Penalise the difference between the actual loss at a perturbed point and its
first-order Taylor approximation. No double backprop needed — two forward passes.

penalty = |L(x+δ) - [L(x) + ∇_x L · δ]|  where δ ~ Uniform with ||δ||_inf = eps_taylor.
λ grid: [0, 0.01, 0.1, 1.0], eps_taylor=0.05.
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
EPS_TAYLOR = 0.05
LAMBDAS = [0, 0.01, 0.1, 1.0]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h345_elle_local_linearity_output.txt")


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


def train_elle(meta, Xtr, Ytr, lam, eps_taylor):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            # Compute gradient of loss w.r.t. input
            xb_req = xb.clone().detach().requires_grad_(True)
            loss_clean = F.cross_entropy(model(xb_req), yb)
            grad_x = torch.autograd.grad(loss_clean, xb_req)[0].detach()

            # Random delta with ||delta||_inf = eps_taylor
            delta = torch.empty_like(xb).uniform_(-eps_taylor, eps_taylor)
            x_pert = (xb + delta).clamp(0, 1).detach()

            opt.zero_grad()
            # Loss at clean point
            out_clean = model(xb)
            l_clean = F.cross_entropy(out_clean, yb)

            # Loss at perturbed point
            l_pert = F.cross_entropy(model(x_pert), yb)

            # First-order Taylor approximation: L(x) + grad_x · delta
            taylor = l_clean.detach() + (grad_x * delta).sum(dim=(1, 2, 3)).mean()

            # ELLE penalty
            if lam > 0:
                elle_penalty = (l_pert - taylor).abs()
                loss = l_clean + lam * elle_penalty
            else:
                loss = l_clean

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("H345 - ELLE: Efficient Local Linearity Enforcement (ICLR 2024)")
    lines.append("=" * 72)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_TRAIN={N_TRAIN}  eps_taylor={EPS_TAYLOR}")
    lines.append(f"lambda grid: {LAMBDAS}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    results = []
    for lam in LAMBDAS:
        t0 = time.time()
        model = train_elle(meta, Xtr, Ytr, lam, EPS_TAYLOR)
        metrics = eval_model(model, Xte, Yte)
        metrics["lambda"] = lam
        metrics["runtime_s"] = round(time.time() - t0, 1)
        results.append(metrics)
        line = (f"  lambda={lam:<5}  clean={metrics['clean_acc']:.3f}  "
                f"fgsm_asr={metrics['fgsm_asr']:.3f}  pgd_asr={metrics['pgd_asr']:.3f}  "
                f"margin={metrics['mean_margin']:.3f}  ({metrics['runtime_s']}s)")
        lines.append(line)
        print(line)

    lines.append("")
    lines.append("Interpretation: ELLE enforces that the loss surface is locally linear")
    lines.append("around training points, cheaply (no double backprop). Higher lambda")
    lines.append("should reduce PGD ASR by discouraging curvature that PGD exploits.")
    lines.append("=" * 72)

    out = "\n".join(lines)
    print(out.split("\n")[0])
    with open(OUT_FILE, "w") as f:
        f.write(out + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
