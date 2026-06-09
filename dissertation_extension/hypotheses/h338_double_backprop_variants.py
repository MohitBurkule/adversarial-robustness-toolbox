"""H338: Double backprop variants.

Compare:
(a) ||∇_x L||² penalty — standard
(b) ||∇_x L||² only on correctly classified samples
(c) ||∇_x L||² only on incorrectly classified samples
(d) ||∇_x L||² weighted by per-sample loss magnitude
All at λ=0.01.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

OUT = "results/fashion_mnist/h338_double_backprop_variants_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
LAM = 0.01


def train_variant(model, Xtr, Ytr, variant):
    """
    variant: 'baseline', 'standard', 'correct_only', 'incorrect_only', 'loss_weighted'
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
            per_sample_loss = F.cross_entropy(logits, yb, reduction='none')
            ce_loss = per_sample_loss.mean()

            if variant == 'baseline':
                total_loss = ce_loss
            else:
                g, = torch.autograd.grad(ce_loss, xb, create_graph=True)
                g_flat = g.flatten(1)  # (B, D)
                g_sq = (g_flat ** 2).sum(dim=1)  # (B,)

                if variant == 'standard':
                    penalty = LAM * g_sq.mean()
                elif variant == 'correct_only':
                    with torch.no_grad():
                        correct_mask = (logits.argmax(1) == yb).float()
                    if correct_mask.sum() > 0:
                        penalty = LAM * (g_sq * correct_mask).sum() / (correct_mask.sum() + 1e-8)
                    else:
                        penalty = torch.tensor(0.0, device=xb.device)
                elif variant == 'incorrect_only':
                    with torch.no_grad():
                        incorrect_mask = (logits.argmax(1) != yb).float()
                    if incorrect_mask.sum() > 0:
                        penalty = LAM * (g_sq * incorrect_mask).sum() / (incorrect_mask.sum() + 1e-8)
                    else:
                        penalty = torch.tensor(0.0, device=xb.device)
                elif variant == 'loss_weighted':
                    # Weight each sample's gradient penalty by its loss magnitude
                    with torch.no_grad():
                        weights = per_sample_loss.detach()
                        weights = weights / (weights.sum() + 1e-8) * weights.size(0)
                    penalty = LAM * (g_sq * weights).mean()
                else:
                    penalty = torch.tensor(0.0, device=xb.device)

                total_loss = ce_loss + penalty
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

    lines = ["H338: Double Backprop Variants", "="*60]
    lines.append(f"Settings: N_TRAIN={N_TRAIN}, EPOCHS={EPOCHS}, LR={LR}, BATCH={BATCH}, λ={LAM}")
    lines.append("Variants: baseline, standard, correct_only, incorrect_only, loss_weighted\n")

    variants = ['baseline', 'standard', 'correct_only', 'incorrect_only', 'loss_weighted']
    descriptions = {
        'baseline': '(a) No penalty',
        'standard': '(b) ||∇_x L||² on all samples',
        'correct_only': '(c) ||∇_x L||² on correct samples only',
        'incorrect_only': '(d) ||∇_x L||² on incorrect samples only',
        'loss_weighted': '(e) ||∇_x L||² weighted by loss magnitude',
    }

    results = {}
    for variant in variants:
        print(f"Training {variant}...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_variant(model, Xtr, Ytr, variant=variant)
        r = eval_model(model, Xte, Yte)
        results[variant] = r
        lines.append(f"{descriptions[variant]}")
        lines.append(f"  clean={r['clean_acc']:.4f} fgsm_asr={r['fgsm_asr']:.4f} pgd_asr={r['pgd_asr']:.4f} margin={r['mean_margin']:.4f}")

    lines.append("\n" + "="*60)
    lines.append("SUMMARY")
    lines.append(f"{'Variant':<20} {'Clean':>8} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Margin':>10}")
    for v, r in results.items():
        lines.append(f"{v:<20} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f}")

    out_text = "\n".join(lines)
    print(out_text)
    with open(OUT, "w") as f:
        f.write(out_text)
    print(f"\nResults written to {OUT}")

if __name__ == "__main__":
    main()
