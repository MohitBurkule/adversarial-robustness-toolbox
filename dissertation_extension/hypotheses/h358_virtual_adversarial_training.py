"""
H358: Virtual Adversarial Training (VAT) — Miyato et al. 2018.
Penalise KL divergence between f(x) and f(x+r_adv) where r_adv is found via power iteration.
Lambda grid: [0, 0.1, 1.0, 10.0], eps=0.1.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import campaign.common as C

N_TRAIN = 6000
EPOCHS  = 10
LR      = 0.05
BATCH   = 128
SEED    = 0
EPS     = 0.1
LAMBDAS = [0, 0.1, 1.0, 10.0]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h358_virtual_adversarial_training_output.txt")


def vat_loss(model, x, eps=0.1, xi=1e-6, K=1):
    """Virtual adversarial loss via K-step power iteration."""
    with torch.no_grad():
        logits_clean = model(x)
        p_clean = F.softmax(logits_clean, dim=1)

    # random unit vector
    d = torch.randn_like(x)
    d = d / (d.view(d.size(0), -1).norm(dim=1).view(-1,1,1,1) + 1e-8)

    for _ in range(K):
        x_adv = (x + xi * d).detach().requires_grad_(True)
        logits_adv = model(x_adv)
        p_adv = F.softmax(logits_adv, dim=1)
        kl = F.kl_div(F.log_softmax(logits_adv, dim=1), p_clean, reduction='batchmean')
        g, = torch.autograd.grad(kl, x_adv)
        d = g.detach()
        d = d / (d.view(d.size(0), -1).norm(dim=1).view(-1,1,1,1) + 1e-8)

    x_vat = (x + eps * d).clamp(0, 1).detach()
    logits_vat = model(x_vat)
    loss_vat = F.kl_div(F.log_softmax(logits_vat, dim=1), p_clean, reduction='batchmean')
    return loss_vat


def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1-float(acc_fgsm),
                pgd_asr=1-float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))


def train_vat(model, Xtr, Ytr, lam):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xb.requires_grad_(False)
            opt.zero_grad()
            out = model(xb)
            ce = F.cross_entropy(out, yb)
            if lam > 0:
                vl = vat_loss(model, xb, eps=EPS)
                loss = ce + lam * vl
            else:
                loss = ce
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = ["H358: Virtual Adversarial Training (VAT)\n" + "="*50]
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    for lam in LAMBDAS:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        train_vat(model, Xtr, Ytr, lam)
        res = eval_model(model, Xte, Yte)
        line = (f"lambda={lam:5.1f} | clean={res['clean_acc']:.3f} "
                f"fgsm_asr={res['fgsm_asr']:.3f} pgd_asr={res['pgd_asr']:.3f} "
                f"margin={res['mean_margin']:.3f}")
        print(line)
        lines.append(line)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
