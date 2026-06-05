"""H330: Proximal gradient (L1 soft-thresholding) training."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

N_TRAIN, EPOCHS, LR, BATCH, SEED = 6000, 10, 0.05, 128, 0
OUT = "results/fashion_mnist/h330_proximal_gradient_training_output.txt"
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

def prox_l1(model, lam):
    """Apply L1 soft-thresholding (proximal operator) to all parameters."""
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.sign() * (p.data.abs() - lam).clamp(min=0)

def train_condition(model, Xtr, Ytr, opt_name="sgd", prox_lam=0.0):
    C.set_seed(SEED)
    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
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
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            if prox_lam > 0:
                prox_l1(model, prox_lam)
        sched.step()
    model.eval()

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    lines = ["H330: Proximal Gradient Training\n"]

    conditions = [
        ("SGD_baseline",           "sgd",  0.0),
        ("SGD+prox(1e-4)",         "sgd",  1e-4),
        ("SGD+prox(1e-3)",         "sgd",  1e-3),
        ("Adam_baseline",          "adam", 0.0),
        ("Adam+prox(1e-4)",        "adam", 1e-4),
        ("Adam+prox(1e-3)",        "adam", 1e-3),
    ]
    for name, opt_name, plam in conditions:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_condition(model, Xtr, Ytr, opt_name=opt_name, prox_lam=plam)
        res = eval_model(model, Xte, Yte)
        line = (f"{name}: clean={res['clean_acc']:.4f} fgsm_asr={res['fgsm_asr']:.4f} "
                f"pgd_asr={res['pgd_asr']:.4f} margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line + "\n")

    with open(OUT, "w") as f: f.writelines(lines)
    print(f"Saved to {OUT}")

if __name__ == "__main__":
    main()
