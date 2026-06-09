"""H311: Symmetric Cross Entropy - SCE = alpha*CE(p,q) + beta*RCE(q,p)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign import common as C

N_TRAIN = 6000; EPOCHS = 10; LR = 0.05; BATCH = 128; SEED = 0
N_CLASSES = 10
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)

def symmetric_ce_loss(logits, targets, alpha=1.0, beta=0.5, eps=1e-7):
    """SCE = alpha*CE(p,q) + beta*RCE(q,p)."""
    n_classes = logits.size(1)
    # Forward CE
    ce = F.cross_entropy(logits, targets)
    # Reverse CE: CE(q, p) = -p * log(q+eps); q=softmax(logits), p=one-hot
    q = F.softmax(logits, dim=1)
    p_onehot = F.one_hot(targets, num_classes=n_classes).float()
    rce = (-p_onehot * torch.log(q + eps)).sum(dim=1).mean()
    return alpha * ce + beta * rce

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

def train_sce(beta, Xtr, Ytr, meta):
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
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = symmetric_ce_loss(out, yb, alpha=1.0, beta=beta)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    betas = [0, 0.1, 0.5, 1.0]
    lines = ["H311: Symmetric CE (beta grid) vs Robustness", "="*60]
    for beta in betas:
        model = train_sce(beta, Xtr, Ytr, meta)
        res = eval_model(model, Xte, Yte)
        label = "CE" if beta == 0 else f"SCE(b={beta})"
        line = (f"beta={beta:.1f} ({label}): clean_acc={res['clean_acc']:.4f}, "
                f"fgsm_asr={res['fgsm_asr']:.4f}, pgd_asr={res['pgd_asr']:.4f}, "
                f"mean_margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line)
    out_path = os.path.join(RESULTS_DIR, "h311_symmetric_cross_entropy_output.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
