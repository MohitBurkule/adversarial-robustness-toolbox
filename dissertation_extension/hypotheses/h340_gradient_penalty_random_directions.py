"""H340: Gradient penalty at random perturbation directions.

Instead of penalising ||∇_x L||² at clean x, penalise gradient at random perturbation:
x_rand = x + ε*u where u ~ Uniform(ball). Penalty = ||∇_{x_rand} L(x_rand)||².
ε=0.05, K=3 random directions averaged.
λ grid: [0, 0.01, 0.1].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

OUT = "results/fashion_mnist/h340_gradient_penalty_random_directions_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.05
K = 3


def train_with_random_dir_penalty(model, Xtr, Ytr, lam, eps=EPS, k=K):
    C.set_seed(SEED)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = Xtr[idx]
            yb = Ytr[idx]
            opt.zero_grad()
            # Clean CE loss (no grad tracking for x here)
            ce_loss = F.cross_entropy(model(xb), yb)

            if lam > 0:
                penalties = []
                for _ in range(k):
                    # Random perturbation u ~ Uniform(-eps, eps) in L∞ ball
                    u = torch.empty_like(xb).uniform_(-eps, eps)
                    x_rand = (xb + u).clamp(0, 1).detach().requires_grad_(True)
                    logits_rand = model(x_rand)
                    loss_rand = F.cross_entropy(logits_rand, yb)
                    g, = torch.autograd.grad(loss_rand, x_rand, create_graph=True)
                    g_sq = (g.flatten(1) ** 2).sum(dim=1).mean()
                    penalties.append(g_sq)
                penalty = lam * torch.stack(penalties).mean()
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


def main():
    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    lines = ["H340: Gradient Penalty at Random Directions", "="*60]
    lines.append(f"Settings: N_TRAIN={N_TRAIN}, EPOCHS={EPOCHS}, LR={LR}, BATCH={BATCH}")
    lines.append(f"Random perturbation ε={EPS}, K={K} directions averaged\n")

    lam_grid = [0, 0.01, 0.1]
    results = {}
    for lam in lam_grid:
        print(f"Training λ={lam}...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_with_random_dir_penalty(model, Xtr, Ytr, lam=lam)
        r = eval_model(model, Xte, Yte)
        key = f"lam={lam}"
        results[key] = r
        lines.append(f"{key}: clean={r['clean_acc']:.4f} fgsm_asr={r['fgsm_asr']:.4f} pgd_asr={r['pgd_asr']:.4f} margin={r['mean_margin']:.4f}")

    lines.append("\n" + "="*60)
    lines.append("SUMMARY")
    lines.append(f"{'Condition':<12} {'Clean':>8} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Margin':>10}")
    for key, r in results.items():
        lines.append(f"{key:<12} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f}")

    out_text = "\n".join(lines)
    print(out_text)
    with open(OUT, "w") as f:
        f.write(out_text)
    print(f"\nResults written to {OUT}")

if __name__ == "__main__":
    main()
