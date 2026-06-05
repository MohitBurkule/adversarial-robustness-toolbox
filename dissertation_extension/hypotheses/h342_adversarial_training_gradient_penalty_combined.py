"""H342: Full comparison — AT and gradient penalty combined.

6 conditions:
(a) baseline
(b) FGSM-AT
(c) grad penalty λ=0.01 (clean x)
(d) FGSM-AT + grad penalty on clean x
(e) FGSM-AT + grad penalty on adversarial x_adv
(f) half-half: 50% clean+penalty, 50% AT samples
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

OUT = "results/fashion_mnist/h342_adversarial_training_gradient_penalty_combined_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
LAM = 0.01
FGSM_EPS = 0.1


def grad_penalty(model, xb, yb, lam):
    """Compute gradient penalty ||∇_x L||² w.r.t. xb."""
    xb_req = xb.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xb_req), yb)
    g, = torch.autograd.grad(loss, xb_req, create_graph=True)
    return lam * (g.flatten(1) ** 2).sum(dim=1).mean()


def train_condition(model, Xtr, Ytr, condition):
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

            if condition == 'baseline':
                loss = F.cross_entropy(model(xb), yb)

            elif condition == 'fgsm_at':
                xadv = C.fgsm(model, xb, yb, eps=FGSM_EPS)
                loss = F.cross_entropy(model(xadv), yb)

            elif condition == 'grad_penalty':
                ce_loss = F.cross_entropy(model(xb), yb)
                pen = grad_penalty(model, xb, yb, LAM)
                loss = ce_loss + pen

            elif condition == 'fgsm_at_grad_penalty_clean':
                xadv = C.fgsm(model, xb, yb, eps=FGSM_EPS)
                ce_loss = F.cross_entropy(model(xadv), yb)
                pen = grad_penalty(model, xb, yb, LAM)
                loss = ce_loss + pen

            elif condition == 'fgsm_at_grad_penalty_adv':
                xadv = C.fgsm(model, xb, yb, eps=FGSM_EPS)
                ce_loss = F.cross_entropy(model(xadv), yb)
                pen = grad_penalty(model, xadv, yb, LAM)
                loss = ce_loss + pen

            elif condition == 'half_half':
                # 50% clean + penalty, 50% AT
                half = xb.size(0) // 2
                xb_clean = xb[:half]
                yb_clean = yb[:half]
                xb_at = xb[half:]
                yb_at = yb[half:]
                xadv = C.fgsm(model, xb_at, yb_at, eps=FGSM_EPS)
                ce_clean = F.cross_entropy(model(xb_clean), yb_clean)
                pen = grad_penalty(model, xb_clean, yb_clean, LAM)
                ce_at = F.cross_entropy(model(xadv), yb_at)
                loss = 0.5 * (ce_clean + pen) + 0.5 * ce_at

            else:
                raise ValueError(condition)

            loss.backward()
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

    lines = ["H342: AT + Gradient Penalty Combined", "="*60]
    lines.append(f"Settings: N_TRAIN={N_TRAIN}, EPOCHS={EPOCHS}, LR={LR}, BATCH={BATCH}, λ={LAM}, FGSM_eps={FGSM_EPS}")
    lines.append("6 conditions: baseline, fgsm_at, grad_penalty, fgsm_at+clean_pen, fgsm_at+adv_pen, half-half\n")

    conditions = [
        ('baseline', '(a) Baseline'),
        ('fgsm_at', '(b) FGSM-AT'),
        ('grad_penalty', '(c) Grad penalty (clean x)'),
        ('fgsm_at_grad_penalty_clean', '(d) FGSM-AT + grad penalty on clean x'),
        ('fgsm_at_grad_penalty_adv', '(e) FGSM-AT + grad penalty on adv x'),
        ('half_half', '(f) Half-half: 50% clean+penalty, 50% AT'),
    ]

    results = {}
    for cond, desc in conditions:
        print(f"Training {cond}...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_condition(model, Xtr, Ytr, condition=cond)
        r = eval_model(model, Xte, Yte)
        results[cond] = r
        lines.append(f"{desc}")
        lines.append(f"  clean={r['clean_acc']:.4f} fgsm_asr={r['fgsm_asr']:.4f} pgd_asr={r['pgd_asr']:.4f} margin={r['mean_margin']:.4f}")

    lines.append("\n" + "="*60)
    lines.append("SUMMARY")
    lines.append(f"{'Condition':<35} {'Clean':>8} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Margin':>10}")
    for (cond, desc), r in zip(conditions, results.values()):
        lines.append(f"{cond:<35} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f}")

    out_text = "\n".join(lines)
    print(out_text)
    with open(OUT, "w") as f:
        f.write(out_text)
    print(f"\nResults written to {OUT}")

if __name__ == "__main__":
    main()
