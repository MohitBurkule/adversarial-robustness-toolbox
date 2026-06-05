"""H337: Gradient magnitude entropy penalty.

Penalise low entropy of gradient magnitude distribution.
gradient_magnitude = |∇_x L| (elementwise), normalise to distribution, compute entropy.
High entropy = gradient spread across many dimensions = less exploitable.
Penalty = -entropy(|∇_x L|). λ grid: [0, 0.01, 0.1, 1.0].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

OUT = "results/fashion_mnist/h337_gradient_magnitude_entropy_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0


def gradient_entropy(g, eps=1e-8):
    """Compute entropy of elementwise gradient magnitude distribution.
    g: (B, C, H, W) gradients.
    Returns scalar: mean entropy across batch.
    """
    g_flat = g.flatten(1).abs()  # (B, D)
    # Normalise to probability distribution per sample
    g_sum = g_flat.sum(dim=1, keepdim=True) + eps
    p = g_flat / g_sum  # (B, D)
    # Entropy: -sum(p * log(p))
    ent = -(p * torch.log(p + eps)).sum(dim=1)  # (B,)
    return ent.mean()


def train_with_entropy_penalty(model, Xtr, Ytr, lam):
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
                g, = torch.autograd.grad(ce_loss, xb, create_graph=True)
                # Penalty = -entropy (penalise low entropy = encourage high entropy)
                ent = gradient_entropy(g)
                # Negative entropy penalty: we want to MAXIMISE entropy, so penalty = -entropy
                penalty = lam * (-ent)
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


def compute_entropy_stats(model, X, Y, n=200):
    """Compute mean entropy of gradient magnitude distribution."""
    model.eval()
    entropies = []
    for k in range(min(n, X.size(0))):
        xi = X[k:k+1].clone().requires_grad_(True)
        yi = Y[k:k+1]
        loss = F.cross_entropy(model(xi), yi)
        g, = torch.autograd.grad(loss, xi)
        g_flat = g.flatten().abs()
        g_sum = g_flat.sum() + 1e-8
        p = g_flat / g_sum
        ent = -(p * torch.log(p + 1e-8)).sum().item()
        entropies.append(ent)
    return np.mean(entropies), np.std(entropies)


def main():
    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    lines = ["H337: Gradient Magnitude Entropy Penalty", "="*60]
    lines.append(f"Settings: N_TRAIN={N_TRAIN}, EPOCHS={EPOCHS}, LR={LR}, BATCH={BATCH}")
    lines.append("Penalty = -entropy(|∇_x L|) — encourages high entropy (diffuse gradient)\n")

    lam_grid = [0, 0.01, 0.1, 1.0]
    results = {}
    for lam in lam_grid:
        print(f"Training λ={lam}...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_with_entropy_penalty(model, Xtr, Ytr, lam=lam)
        r = eval_model(model, Xte, Yte)
        ent_mean, ent_std = compute_entropy_stats(model, Xte, Yte, n=100)
        key = f"lam={lam}"
        results[key] = {**r, "grad_entropy": ent_mean, "grad_entropy_std": ent_std}
        lines.append(f"{key}: clean={r['clean_acc']:.4f} fgsm_asr={r['fgsm_asr']:.4f} pgd_asr={r['pgd_asr']:.4f} margin={r['mean_margin']:.4f} grad_ent={ent_mean:.4f}±{ent_std:.4f}")

    lines.append("\n" + "="*60)
    lines.append("SUMMARY")
    lines.append(f"{'Condition':<12} {'Clean':>8} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Margin':>10} {'GradEnt':>10}")
    for key, r in results.items():
        lines.append(f"{key:<12} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f} {r['grad_entropy']:>10.4f}")

    out_text = "\n".join(lines)
    print(out_text)
    with open(OUT, "w") as f:
        f.write(out_text)
    print(f"\nResults written to {OUT}")

if __name__ == "__main__":
    main()
