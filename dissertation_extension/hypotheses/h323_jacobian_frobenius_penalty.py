"""H323: Jacobian Frobenius norm penalty via random projections."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

N_TRAIN, EPOCHS, LR, BATCH, SEED = 6000, 10, 0.05, 128, 0
K_PROJ = 5
OUT = "results/fashion_mnist/h323_jacobian_frobenius_penalty_output.txt"
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

def jacobian_frob_penalty(model, xb, K=5):
    """Estimate ||J||_F^2 via K random projections."""
    penalties = []
    for _ in range(K):
        v = torch.randn(xb.size(0), 10, device=xb.device)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        xb_g = xb.clone().detach().requires_grad_(True)
        out = model(xb_g)
        proj = (out * v).sum()
        g, = torch.autograd.grad(proj, xb_g, create_graph=True)
        penalties.append((g ** 2).sum(dim=(1,2,3)))
    return torch.stack(penalties, dim=0).mean(0).mean()

def input_grad_penalty(model, xb, yb):
    xb_g = xb.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xb_g), yb)
    g, = torch.autograd.grad(loss, xb_g, create_graph=True)
    return (g ** 2).sum(dim=(1,2,3)).mean()

def train_condition(model, Xtr, Ytr, mode="baseline", lam=0.0):
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
            if mode == "jacobian" and lam > 0:
                xb_g = xb.clone().detach().requires_grad_(True)
                ce = F.cross_entropy(model(xb_g), yb)
                jp = jacobian_frob_penalty(model, xb, K=K_PROJ)
                loss = ce + lam * jp
            elif mode == "inputgrad" and lam > 0:
                gp = input_grad_penalty(model, xb, yb)
                xb_g = xb.clone().detach().requires_grad_(True)
                ce = F.cross_entropy(model(xb_g), yb)
                loss = ce + lam * gp
            else:
                loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    lines = ["H323: Jacobian Frobenius Penalty\n"]

    conditions = [
        ("baseline",           "baseline",  0.0),
        ("jacobian(lam=0.001)","jacobian",  0.001),
        ("jacobian(lam=0.01)", "jacobian",  0.01),
        ("inputgrad(lam=0.001)","inputgrad",0.001),
        ("inputgrad(lam=0.01)", "inputgrad",0.01),
    ]
    for name, mode, lam in conditions:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_condition(model, Xtr, Ytr, mode=mode, lam=lam)
        res = eval_model(model, Xte, Yte)
        line = (f"{name}: clean={res['clean_acc']:.4f} fgsm_asr={res['fgsm_asr']:.4f} "
                f"pgd_asr={res['pgd_asr']:.4f} margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line + "\n")

    with open(OUT, "w") as f: f.writelines(lines)
    print(f"Saved to {OUT}")

if __name__ == "__main__":
    main()
