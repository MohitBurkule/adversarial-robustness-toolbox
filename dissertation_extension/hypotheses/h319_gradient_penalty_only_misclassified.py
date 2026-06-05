"""H319: Apply input gradient penalty only on samples model currently misclassifies."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign import common as C

N_TRAIN = 6000; EPOCHS = 10; LR = 0.05; BATCH = 128; SEED = 0
LAM = 0.01
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)

def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters(): p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1-float(acc_fgsm),
                pgd_asr=1-float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))

def train_selective_gp(mode, Xtr, Ytr, meta):
    """
    mode: 'none' (no penalty), 'uniform' (all samples), 'misclassified' (high-loss only)
    """
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = Xtr[idx].clone().detach().requires_grad_(True)
            yb = Ytr[idx]
            opt.zero_grad()
            out = model(xb)
            per_sample_loss = F.cross_entropy(out, yb, reduction='none')
            ce_loss = per_sample_loss.mean()

            if mode == 'none':
                loss = ce_loss
            elif mode == 'uniform':
                g = torch.autograd.grad(ce_loss, xb, create_graph=True)[0]
                gp = LAM * g.flatten(1).norm(dim=1).pow(2).mean()
                loss = ce_loss + gp
            elif mode == 'misclassified':
                mean_l = per_sample_loss.mean().detach()
                std_l = per_sample_loss.std().detach()
                hard_mask = (per_sample_loss.detach() > mean_l + std_l)
                if hard_mask.sum() > 0:
                    # Compute gradient, apply penalty only on hard samples
                    g = torch.autograd.grad(ce_loss, xb, create_graph=True)[0]
                    g_norm_sq = g.flatten(1).norm(dim=1).pow(2)
                    gp = LAM * (g_norm_sq * hard_mask.float()).mean()
                    loss = ce_loss + gp
                else:
                    loss = ce_loss
            else:
                loss = ce_loss

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    modes = ['none', 'uniform', 'misclassified']
    lines = ["H319: Gradient Penalty - Selective vs Uniform vs None", "="*60]
    for mode in modes:
        model = train_selective_gp(mode, Xtr, Ytr, meta)
        res = eval_model(model, Xte, Yte)
        line = (f"mode={mode:15s}: clean_acc={res['clean_acc']:.4f}, "
                f"fgsm_asr={res['fgsm_asr']:.4f}, pgd_asr={res['pgd_asr']:.4f}, "
                f"mean_margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line)
    out_path = os.path.join(RESULTS_DIR, "h319_gradient_penalty_only_misclassified_output.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
