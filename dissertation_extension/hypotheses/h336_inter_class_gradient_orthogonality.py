"""H336: Inter-class gradient orthogonality penalty.

Penalise alignment between gradients of different classes.
Per-class mean gradient g_c, penalty = sum_{c≠c'} |cosine(g_c, g_c')|.
λ grid: [0, 0.001, 0.01].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

OUT = "results/fashion_mnist/h336_inter_class_gradient_orthogonality_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
N_CLASSES = 10


def train_with_orthogonality_penalty(model, Xtr, Ytr, lam):
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
                # Compute per-class mean gradients
                class_grads = []
                classes_present = yb.unique()
                valid_classes = []
                for c in classes_present:
                    mask = (yb == c)
                    if mask.sum() < 2:
                        continue
                    xc = xb[mask]
                    lc_loss = F.cross_entropy(model(xc), yb[mask])
                    gc, = torch.autograd.grad(lc_loss, xb, retain_graph=True, create_graph=True,
                                              allow_unused=True)
                    if gc is None:
                        # Gradient w.r.t. xb, but only xc contributed; recompute properly
                        gc2, = torch.autograd.grad(lc_loss, xb, retain_graph=True,
                                                   create_graph=True, allow_unused=True)
                        if gc2 is None:
                            continue
                        gc = gc2[mask].flatten(1).mean(0)
                    else:
                        gc = gc[mask].flatten(1).mean(0)
                    class_grads.append(gc)
                    valid_classes.append(c.item())

                if len(class_grads) >= 2:
                    # Stack: (C, D)
                    G = torch.stack(class_grads, dim=0)
                    G_norm = G / (G.norm(dim=1, keepdim=True) + 1e-8)
                    # Gram matrix
                    gram = G_norm @ G_norm.T  # (C, C)
                    # Off-diagonal absolute cosine similarity
                    mask_off = ~torch.eye(len(class_grads), dtype=torch.bool, device=gram.device)
                    penalty = lam * gram[mask_off].abs().mean()
                    total_loss = ce_loss + penalty
                else:
                    total_loss = ce_loss
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

    lines = ["H336: Inter-Class Gradient Orthogonality Penalty", "="*60]
    lines.append(f"Settings: N_TRAIN={N_TRAIN}, EPOCHS={EPOCHS}, LR={LR}, BATCH={BATCH}")
    lines.append("Penalty = sum_{c≠c'} |cosine(g_c, g_c')| where g_c = per-class mean gradient\n")

    lam_grid = [0, 0.001, 0.01]
    results = {}
    for lam in lam_grid:
        print(f"Training λ={lam}...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_with_orthogonality_penalty(model, Xtr, Ytr, lam=lam)
        r = eval_model(model, Xte, Yte)
        key = f"lam={lam}"
        results[key] = r
        lines.append(f"{key}: clean={r['clean_acc']:.4f} fgsm_asr={r['fgsm_asr']:.4f} pgd_asr={r['pgd_asr']:.4f} margin={r['mean_margin']:.4f}")

    lines.append("\n" + "="*60)
    lines.append("SUMMARY")
    lines.append(f"{'Condition':<15} {'Clean':>8} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Margin':>10}")
    for key, r in results.items():
        lines.append(f"{key:<15} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f}")

    out_text = "\n".join(lines)
    print(out_text)
    with open(OUT, "w") as f:
        f.write(out_text)
    print(f"\nResults written to {OUT}")

if __name__ == "__main__":
    main()
