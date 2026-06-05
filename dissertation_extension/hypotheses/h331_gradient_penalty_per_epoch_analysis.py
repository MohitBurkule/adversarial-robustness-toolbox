"""H331: Track gradient norm and robustness per epoch for baseline vs gradient penalty."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

N_TRAIN, EPOCHS, LR, BATCH, SEED = 6000, 10, 0.05, 128, 0
LAM_GP = 0.01
OUT = "results/fashion_mnist/h331_gradient_penalty_per_epoch_analysis_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

def fgsm_asr(model, Xte, Yte, eps=0.1):
    for p in model.parameters(): p.requires_grad_(True)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc = C.logits_and_acc(model, Xfgsm, Yte)
    return 1 - float(acc)

def pgd_asr(model, Xte, Yte, eps=0.1, steps=10, alpha=0.01):
    for p in model.parameters(): p.requires_grad_(True)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=steps, alpha=alpha)
    _, acc = C.logits_and_acc(model, Xpgd, Yte)
    return 1 - float(acc)

def mean_input_grad_norm(model, Xte, Yte, batch=256):
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

def input_grad_penalty(model, xb, yb):
    xb_g = xb.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xb_g), yb)
    g, = torch.autograd.grad(loss, xb_g, create_graph=True)
    return (g ** 2).sum(dim=(1,2,3)).mean()

def train_and_track(model, Xtr, Ytr, Xte, Yte, use_gp=False, lam=0.01):
    C.set_seed(SEED)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    records = []
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            if use_gp:
                xb_g = xb.clone().detach().requires_grad_(True)
                ce = F.cross_entropy(model(xb_g), yb)
                gp = input_grad_penalty(model, xb, yb)
                loss = ce + lam * gp
            else:
                loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        _, clean = C.logits_and_acc(model, Xte, Yte)
        fa = fgsm_asr(model, Xte, Yte)
        pa = pgd_asr(model, Xte, Yte)
        gn = mean_input_grad_norm(model, Xte, Yte)
        records.append((ep+1, float(clean), fa, pa, gn))
    return records

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    lines = ["H331: Per-Epoch Gradient Penalty Analysis\n",
             "condition | epoch | clean_acc | fgsm_asr | pgd_asr | inp_grad_norm\n"]

    for cname, use_gp in [("baseline", False), ("gp(lam=0.01)", True)]:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        records = train_and_track(model, Xtr, Ytr, Xte, Yte, use_gp=use_gp, lam=LAM_GP)
        for (ep, clean, fa, pa, gn) in records:
            line = (f"{cname} ep={ep}: clean={clean:.4f} fgsm_asr={fa:.4f} "
                    f"pgd_asr={pa:.4f} inp_grad={gn:.4f}")
            print(line)
            lines.append(line + "\n")

    with open(OUT, "w") as f: f.writelines(lines)
    print(f"Saved to {OUT}")

if __name__ == "__main__":
    main()
