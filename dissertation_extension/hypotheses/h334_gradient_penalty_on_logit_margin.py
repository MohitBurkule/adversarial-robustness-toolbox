"""H334: Gradient penalty on logit margin loss instead of CE.

Instead of penalising ||∇_x CE||², use margin loss L_margin = max(0, 1 - margin).
Penalty = ||∇_x L_margin||². Compare to CE-penalty and baseline.
λ grid: [0, 0.001, 0.01, 0.1].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

OUT = "results/fashion_mnist/h334_gradient_penalty_on_logit_margin_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0


def margin_loss(logits, y):
    """max(0, 1 - margin) where margin = correct_logit - max_other_logit."""
    correct = logits.gather(1, y.view(-1, 1)).squeeze(1)
    tmp = logits.clone()
    tmp.scatter_(1, y.view(-1, 1), -1e9)
    other = tmp.max(1).values
    return F.relu(1.0 - (correct - other)).mean()


def train_with_penalty(model, Xtr, Ytr, lam, penalty_type="margin"):
    """Train with gradient penalty. penalty_type: 'margin' or 'ce'."""
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
                if penalty_type == "margin":
                    pen_loss = margin_loss(logits, yb)
                else:
                    pen_loss = ce_loss
                g, = torch.autograd.grad(pen_loss, xb, create_graph=True)
                penalty = lam * (g ** 2).sum(dim=(1, 2, 3)).mean()
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

    lines = ["H334: Gradient Penalty on Logit Margin Loss", "="*60]
    lines.append(f"Settings: N_TRAIN={N_TRAIN}, EPOCHS={EPOCHS}, LR={LR}, BATCH={BATCH}")

    lam_grid = [0, 0.001, 0.01, 0.1]
    results = {}

    # Margin penalty
    for lam in lam_grid:
        print(f"Training margin-penalty λ={lam}...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_with_penalty(model, Xtr, Ytr, lam=lam, penalty_type="margin")
        r = eval_model(model, Xte, Yte)
        key = f"margin_lam={lam}"
        results[key] = r
        lines.append(f"\n{key}: clean={r['clean_acc']:.4f} fgsm_asr={r['fgsm_asr']:.4f} pgd_asr={r['pgd_asr']:.4f} margin={r['mean_margin']:.4f}")

    # CE penalty for comparison
    for lam in [0.001, 0.01, 0.1]:
        print(f"Training CE-penalty λ={lam}...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_with_penalty(model, Xtr, Ytr, lam=lam, penalty_type="ce")
        r = eval_model(model, Xte, Yte)
        key = f"ce_lam={lam}"
        results[key] = r
        lines.append(f"\n{key}: clean={r['clean_acc']:.4f} fgsm_asr={r['fgsm_asr']:.4f} pgd_asr={r['pgd_asr']:.4f} margin={r['mean_margin']:.4f}")

    lines.append("\n" + "="*60)
    lines.append("SUMMARY")
    lines.append(f"{'Condition':<25} {'Clean':>8} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Margin':>10}")
    for key, r in results.items():
        lines.append(f"{key:<25} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f}")

    out_text = "\n".join(lines)
    print(out_text)
    with open(OUT, "w") as f:
        f.write(out_text)
    print(f"\nResults written to {OUT}")

if __name__ == "__main__":
    main()
