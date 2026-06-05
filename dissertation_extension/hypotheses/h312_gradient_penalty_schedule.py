"""H312: Warm-up schedule for input gradient penalty."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
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
    """Return lambda for gradient penalty given schedule type."""
    t = epoch  # 0-indexed
    T = total_epochs
    if schedule == "fixed":
        return lam_max
    elif schedule == "linear_warmup":
        return lam_max * min(1.0, t / (T // 2))
    elif schedule == "cosine":
        import math
        return lam_max * 0.5 * (1 + math.cos(math.pi * t / T))
    elif schedule == "step":
        # Step: 0 for first 3 epochs, full for remaining
        return 0.0 if t < 3 else lam_max
    elif schedule == "none":
        return 0.0
    return lam_max

def train_with_schedule(schedule, Xtr, Ytr, meta):
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

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    schedules = ["none", "fixed", "linear_warmup", "cosine", "step"]
    lines = ["H312: Gradient Penalty Schedule vs Robustness", "="*60]
    for sched_name in schedules:
        model = train_with_schedule(sched_name, Xtr, Ytr, meta)
        res = eval_model(model, Xte, Yte)
        line = (f"schedule={sched_name:15s}: clean_acc={res['clean_acc']:.4f}, "
                f"fgsm_asr={res['fgsm_asr']:.4f}, pgd_asr={res['pgd_asr']:.4f}, "
                f"mean_margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line)
    out_path = os.path.join(RESULTS_DIR, "h312_gradient_penalty_schedule_output.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
