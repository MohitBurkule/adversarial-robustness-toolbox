"""H332: Transfer attack robustness — surrogate FGSM/PGD attacks on target models."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

N_TRAIN, EPOCHS, LR, BATCH, SEED = 6000, 10, 0.05, 128, 0
SURROGATE_SEED = 42
AT_EPS = 0.1
OUT = "results/fashion_mnist/h332_gradient_penalty_transfer_attack_output.txt"
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

def eval_transfer(target, Xfgsm_surr, Xpgd_surr, Yte):
    target.eval()
    _, acc_fgsm = C.logits_and_acc(target, Xfgsm_surr, Yte)
    _, acc_pgd = C.logits_and_acc(target, Xpgd_surr, Yte)
    return 1 - float(acc_fgsm), 1 - float(acc_pgd)

def input_grad_penalty(model, xb, yb):
    xb_g = xb.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xb_g), yb)
    g, = torch.autograd.grad(loss, xb_g, create_graph=True)
    return (g ** 2).sum(dim=(1,2,3)).mean()

def train_model(model, Xtr, Ytr, mode="baseline", seed=SEED):
    C.set_seed(seed)
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
            if mode == "gp":
                xb_g = xb.clone().detach().requires_grad_(True)
                ce = F.cross_entropy(model(xb_g), yb)
                gp = input_grad_penalty(model, xb, yb)
                loss = ce + 0.01 * gp
            elif mode == "fgsm_at":
                model.eval()
                xb_adv = C.fgsm(model, xb, yb, eps=AT_EPS)
                model.train()
                loss = F.cross_entropy(model(xb_adv), yb)
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
    lines = ["H332: Transfer Attack Robustness\n"]

    # Train surrogate (different seed)
    C.set_seed(SURROGATE_SEED)
    surrogate = C.build_model("cnn", meta)
    train_model(surrogate, Xtr, Ytr, mode="baseline", seed=SURROGATE_SEED)
    surrogate.eval()
    for p in surrogate.parameters(): p.requires_grad_(True)

    # Generate transfer attacks from surrogate
    print("Generating transfer attacks from surrogate...")
    Xfgsm_surr = C.fgsm(surrogate, Xte, Yte, eps=AT_EPS)
    Xpgd_surr = C.pgd(surrogate, Xte, Yte, eps=AT_EPS, steps=10, alpha=0.01)

    # Evaluate surrogate direct attacks (white-box)
    _, surr_acc = C.logits_and_acc(surrogate, Xte, Yte)
    lines.append(f"Surrogate model clean acc: {float(surr_acc):.4f}\n")

    # Train and evaluate target models
    target_conditions = [
        ("baseline",  "baseline"),
        ("gp(0.01)",  "gp"),
        ("FGSM-AT",   "fgsm_at"),
    ]
    for tname, tmode in target_conditions:
        C.set_seed(SEED)
        target = C.build_model("cnn", meta)
        train_model(target, Xtr, Ytr, mode=tmode, seed=SEED)
        # White-box eval
        res_wb = eval_model(target, Xte, Yte)
        # Transfer eval
        tr_fgsm, tr_pgd = eval_transfer(target, Xfgsm_surr, Xpgd_surr, Yte)
        line = (f"target={tname}: "
                f"clean={res_wb['clean_acc']:.4f} "
                f"wb_fgsm_asr={res_wb['fgsm_asr']:.4f} wb_pgd_asr={res_wb['pgd_asr']:.4f} "
                f"transfer_fgsm_asr={tr_fgsm:.4f} transfer_pgd_asr={tr_pgd:.4f} "
                f"margin={res_wb['mean_margin']:.4f}")
        print(line)
        lines.append(line + "\n")

    with open(OUT, "w") as f: f.writelines(lines)
    print(f"Saved to {OUT}")

if __name__ == "__main__":
    main()
