"""H329: Weight decay sweep and gradient interaction analysis."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

N_TRAIN, EPOCHS, LR, BATCH, SEED = 6000, 10, 0.05, 128, 0
OUT = "results/fashion_mnist/h329_weight_decay_gradient_interaction_output.txt"
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

def compute_input_grad_norm(model, Xte, Yte, batch=256):
    norms = []
    model.train()
    for i in range(0, min(Xte.size(0), 1000), batch):
        xb = Xte[i:i+batch].clone().detach().requires_grad_(True)
        yb = Yte[i:i+batch]
        loss = F.cross_entropy(model(xb), yb)
        g, = torch.autograd.grad(loss, xb)
        norms.append(g.flatten(1).norm(dim=1).detach().cpu())
    model.eval()
    return float(torch.cat(norms).mean())

def compute_weight_grad_norm(model, Xte, Yte, batch=256):
    norms = []
    model.train()
    for i in range(0, min(Xte.size(0), 500), batch):
        xb = Xte[i:i+batch]
        yb = Yte[i:i+batch]
        model.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        total = 0.0
        count = 0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
                count += 1
        norms.append(total ** 0.5)
    model.eval()
    return float(np.mean(norms))

def train_condition(model, Xtr, Ytr, wd=5e-4):
    C.set_seed(SEED)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=wd)
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

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    lines = ["H329: Weight Decay vs Gradient Interaction\n",
             "wd | clean_acc | fgsm_asr | pgd_asr | mean_margin | inp_grad_norm | wt_grad_norm\n"]

    wd_grid = [0, 1e-5, 1e-4, 5e-4, 1e-3, 1e-2]
    for wd in wd_grid:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_condition(model, Xtr, Ytr, wd=wd)
        res = eval_model(model, Xte, Yte)
        inp_gn = compute_input_grad_norm(model, Xte, Yte)
        wt_gn = compute_weight_grad_norm(model, Xte, Yte)
        line = (f"wd={wd:.0e}: clean={res['clean_acc']:.4f} fgsm_asr={res['fgsm_asr']:.4f} "
                f"pgd_asr={res['pgd_asr']:.4f} margin={res['mean_margin']:.4f} "
                f"inp_grad={inp_gn:.4f} wt_grad={wt_gn:.4f}")
        print(line)
        lines.append(line + "\n")

    with open(OUT, "w") as f: f.writelines(lines)
    print(f"Saved to {OUT}")

if __name__ == "__main__":
    main()
