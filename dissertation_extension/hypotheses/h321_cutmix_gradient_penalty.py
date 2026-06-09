"""H321: CutMix augmentation + input gradient penalty comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign import common as C

N_TRAIN, EPOCHS, LR, BATCH, SEED = 6000, 10, 0.05, 128, 0
OUT = "results/fashion_mnist/h321_cutmix_gradient_penalty_output.txt"
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

def rand_bbox(size, lam):
    W = size[2]; H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def cutmix_batch(xb, yb, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(xb.size(0), device=xb.device)
    x1, y1, x2, y2 = rand_bbox(xb.shape, lam)
    xm = xb.clone()
    xm[:, :, x1:x2, y1:y2] = xb[perm, :, x1:x2, y1:y2]
    lam_actual = 1 - (x2 - x1) * (y2 - y1) / (xb.shape[2] * xb.shape[3])
    return xm, yb, yb[perm], lam_actual

def input_grad_penalty(model, xb, yb):
    xb_g = xb.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xb_g), yb)
    g, = torch.autograd.grad(loss, xb_g, create_graph=True)
    return (g ** 2).sum(dim=(1,2,3)).mean()

def train_condition(model, Xtr, Ytr, use_cutmix=False, lam_gp=0.0):
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
            if use_cutmix:
                xm, ya, yb2, lam = cutmix_batch(xb, yb)
                out = model(xm)
                loss = lam * F.cross_entropy(out, ya) + (1 - lam) * F.cross_entropy(out, yb2)
            else:
                out = model(xb)
                loss = F.cross_entropy(out, yb)
            if lam_gp > 0:
                gp = input_grad_penalty(model, xb, yb)
                loss = loss + lam_gp * gp
            loss.backward()
            opt.step()
        sched.step()
    model.eval()

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    lines = ["H321: CutMix + Gradient Penalty\n"]

    conditions = [
        ("baseline",           False, 0.0),
        ("cutmix_only",        True,  0.0),
        ("cutmix+gp(0.01)",    True,  0.01),
        ("gp_only(0.01)",      False, 0.01),
    ]
    for name, use_cm, lam in conditions:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_condition(model, Xtr, Ytr, use_cutmix=use_cm, lam_gp=lam)
        res = eval_model(model, Xte, Yte)
        line = (f"{name}: clean={res['clean_acc']:.4f} fgsm_asr={res['fgsm_asr']:.4f} "
                f"pgd_asr={res['pgd_asr']:.4f} margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line + "\n")

    with open(OUT, "w") as f: f.writelines(lines)
    print(f"Saved to {OUT}")

if __name__ == "__main__":
    main()
