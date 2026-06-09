"""
H366: Smooth Activation Swap — ReLU vs SiLU (Swish).
Conditions: ReLU baseline, SiLU, SiLU + FGSM-AT, ReLU + FGSM-AT.
No gradient penalty — just activation change.
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

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h366_smooth_activation_swap_output.txt")


def replace_relu_with_silu(model):
    """Recursively replace nn.ReLU with nn.SiLU."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.SiLU())
        else:
            replace_relu_with_silu(module)
    return model


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


def train_model_simple(model, Xtr, Ytr, adv_train=False):
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
            if adv_train:
                xb = C.fgsm(model, xb, yb, eps=EPS)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = ["H366: Smooth Activation Swap (ReLU vs SiLU)\n" + "="*50]
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    conditions = [
        ("relu_baseline", False),
        ("silu_baseline", False),
        ("relu_fgsm_at", True),
        ("silu_fgsm_at", True),
    ]

    for cond, adv_train in conditions:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        if "silu" in cond:
            replace_relu_with_silu(model)
        train_model_simple(model, Xtr, Ytr, adv_train=adv_train)
        res = eval_model(model, Xte, Yte)
        line = (f"condition={cond} adv_train={adv_train} | clean={res['clean_acc']:.3f} "
                f"fgsm_asr={res['fgsm_asr']:.3f} pgd_asr={res['pgd_asr']:.3f} "
                f"margin={res['mean_margin']:.3f}")
        print(line)
        lines.append(line)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
