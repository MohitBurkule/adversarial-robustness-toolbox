"""H317: Full cosine annealing of gradient penalty + warmup/cooldown variants."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign import common as C

N_TRAIN = 6000; EPOCHS = 10; LR = 0.05; BATCH = 128; SEED = 0
LAM_MAX = 0.01
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_lam(schedule, epoch, total_epochs=10, lam_max=LAM_MAX):
    t = epoch; T = total_epochs
    if schedule == "fixed":
        return lam_max
    elif schedule == "none":
        return 0.0
    elif schedule == "cosine_anneal":
        # λ_max * 0.5 * (1 + cos(π*t/T))
        return lam_max * 0.5 * (1 + math.cos(math.pi * t / T))
    elif schedule == "warmup":
        # 0 -> lam_max over first half
        half = T // 2
        return lam_max * min(1.0, t / max(1, half))
    elif schedule == "cooldown":
        # lam_max -> 0 over second half
        half = T // 2
        if t < half:
            return lam_max
        return lam_max * max(0.0, 1.0 - (t - half) / max(1, T - half))
    elif schedule == "warmup_cooldown":
        # warm-up first half, cool-down second half
        half = T // 2
        if t <= half:
            return lam_max * (t / max(1, half))
        else:
            return lam_max * (1.0 - (t - half) / max(1, T - half))
    return lam_max

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

def train_schedule(schedule, Xtr, Ytr, meta):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        lam = get_lam(schedule, ep, EPOCHS)
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = Xtr[idx].clone().detach().requires_grad_(True)
            yb = Ytr[idx]
            opt.zero_grad()
            out = model(xb)
            ce_loss = F.cross_entropy(out, yb)
            if lam > 0:
                g = torch.autograd.grad(ce_loss, xb, create_graph=True)[0]
                gp = lam * g.flatten(1).norm(dim=1).pow(2).mean()
                loss = ce_loss + gp
            else:
                loss = ce_loss
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    schedules = ["none", "fixed", "cosine_anneal", "warmup", "cooldown", "warmup_cooldown"]
    lines = ["H317: Gradient Penalty Warmup/Cooldown Schedules vs Robustness", "="*60]
    for sched_name in schedules:
        model = train_schedule(sched_name, Xtr, Ytr, meta)
        res = eval_model(model, Xte, Yte)
        line = (f"schedule={sched_name:20s}: clean_acc={res['clean_acc']:.4f}, "
                f"fgsm_asr={res['fgsm_asr']:.4f}, pgd_asr={res['pgd_asr']:.4f}, "
                f"mean_margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line)
    out_path = os.path.join(RESULTS_DIR, "h317_gradient_penalty_warmup_cooldown_output.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
