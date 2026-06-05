"""H328: Input gradient norm clipping — only penalise samples above threshold tau."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

N_TRAIN, EPOCHS, LR, BATCH, SEED = 6000, 10, 0.05, 128, 0
LAM_GP = 0.01
OUT = "results/fashion_mnist/h328_input_gradient_clipping_training_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters(): p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1-float(acc_fgsm), pgd_asr=1-float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))

def train_condition(model, Xtr, Ytr, tau=0.0, lam=0.01):
    """tau=0.0 means penalise all (no clipping)."""
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
            xb_g = xb.clone().detach().requires_grad_(True)
            ce = F.cross_entropy(model(xb_g), yb)
            if lam > 0:
                g, = torch.autograd.grad(ce, xb_g, create_graph=True)
                g_norms = g.flatten(1).norm(dim=1)  # per-sample norms
                if tau > 0:
                    mask = (g_norms > tau).float()
                    penalty = (g_norms ** 2 * mask).sum() / (mask.sum() + 1e-8)
                else:
                    penalty = (g_norms ** 2).mean()
                loss = ce + lam * penalty
            else:
                loss = ce
            loss.backward()
            opt.step()
        sched.step()
    model.eval()

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    lines = ["H328: Input Gradient Clipping Training\n"]

    conditions = [
        ("baseline",       None, 0.0),
        ("tau=0(all)",     0.0,  LAM_GP),
        ("tau=0.5",        0.5,  LAM_GP),
        ("tau=1.0",        1.0,  LAM_GP),
        ("tau=2.0",        2.0,  LAM_GP),
    ]
    for name, tau, lam in conditions:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        if tau is None:
            train_condition(model, Xtr, Ytr, tau=0.0, lam=0.0)
        else:
            train_condition(model, Xtr, Ytr, tau=tau, lam=lam)
        res = eval_model(model, Xte, Yte)
        line = (f"{name}: clean={res['clean_acc']:.4f} fgsm_asr={res['fgsm_asr']:.4f} "
                f"pgd_asr={res['pgd_asr']:.4f} margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line + "\n")

    with open(OUT, "w") as f: f.writelines(lines)
    print(f"Saved to {OUT}")

if __name__ == "__main__":
    main()
