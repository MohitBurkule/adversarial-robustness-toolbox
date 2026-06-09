"""H335: Input gradient norm normalisation during training.

Instead of penalising ||∇_x L||², normalise the gradient direction.
Three conditions:
  - standard baseline
  - normalised-grad: add auxiliary loss term encouraging unit-norm input gradient
  - normalised + penalty: normalised + ||∇_x L||² penalty
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

OUT = "results/fashion_mnist/h335_input_gradient_norm_normalisation_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0


def train_standard(model, Xtr, Ytr):
    C.set_seed(SEED)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_normalised_grad(model, Xtr, Ytr, add_penalty=False, lam=0.01):
    """
    Normalised gradient training:
    - Compute ∇_x CE (with create_graph=True)
    - Normalise: g_hat = g / (||g|| + eps)
    - Add regularisation that penalises deviation from unit norm: (||g|| - 1)²
    - Optionally also add ||g||² penalty
    """
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
            # Compute input gradient with graph
            g, = torch.autograd.grad(ce_loss, xb, create_graph=True)
            # Per-sample gradient norms
            g_flat = g.flatten(1)
            g_norm = g_flat.norm(dim=1)  # (B,)
            # Normalisation penalty: encourage g_norm ~ 1
            norm_penalty = lam * ((g_norm - 1.0) ** 2).mean()
            if add_penalty:
                mag_penalty = lam * (g_norm ** 2).mean()
                total_loss = ce_loss + norm_penalty + mag_penalty
            else:
                total_loss = ce_loss + norm_penalty
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


def compute_grad_norm_stats(model, X, Y, n=200):
    """Compute mean/std of input gradient norms."""
    model.eval()
    norms = []
    X_sub = X[:n]
    Y_sub = Y[:n]
    for k in range(X_sub.size(0)):
        xi = X_sub[k:k+1].clone().requires_grad_(True)
        yi = Y_sub[k:k+1]
        loss = F.cross_entropy(model(xi), yi)
        g, = torch.autograd.grad(loss, xi)
        norms.append(g.norm().item())
    return np.mean(norms), np.std(norms)


def main():
    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    lines = ["H335: Input Gradient Norm Normalisation", "="*60]
    lines.append(f"Settings: N_TRAIN={N_TRAIN}, EPOCHS={EPOCHS}, LR={LR}, BATCH={BATCH}")

    conditions = [
        ("standard", False, False),
        ("normalised_grad", True, False),
        ("normalised+penalty", True, True),
    ]

    results = {}
    for name, normalised, add_penalty in conditions:
        print(f"Training {name}...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        if not normalised:
            train_standard(model, Xtr, Ytr)
        else:
            train_normalised_grad(model, Xtr, Ytr, add_penalty=add_penalty, lam=0.01)

        r = eval_model(model, Xte, Yte)
        gn_mean, gn_std = compute_grad_norm_stats(model, Xte, Yte, n=100)
        results[name] = {**r, "grad_norm_mean": gn_mean, "grad_norm_std": gn_std}

        lines.append(f"\nCondition: {name}")
        lines.append(f"  clean={r['clean_acc']:.4f} fgsm_asr={r['fgsm_asr']:.4f} pgd_asr={r['pgd_asr']:.4f} margin={r['mean_margin']:.4f}")
        lines.append(f"  grad_norm_mean={gn_mean:.4f} ± {gn_std:.4f}")

    lines.append("\n" + "="*60)
    lines.append("SUMMARY")
    lines.append(f"{'Condition':<25} {'Clean':>8} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'GradNorm':>10}")
    for name, r in results.items():
        lines.append(f"{name:<25} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['grad_norm_mean']:>10.4f}")

    out_text = "\n".join(lines)
    print(out_text)
    with open(OUT, "w") as f:
        f.write(out_text)
    print(f"\nResults written to {OUT}")

if __name__ == "__main__":
    main()
