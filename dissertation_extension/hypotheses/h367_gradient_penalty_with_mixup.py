"""
H367: Gradient Penalty with Mixup (AugMax-style).
Conditions: (1) standard mixup, (2) adversarial mixup, (3) clean + grad penalty, (4) adv mixup + grad penalty.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import campaign.common as C

N_TRAIN = 6000
EPOCHS  = 10
LR      = 0.05
BATCH   = 128
SEED    = 0
EPS     = 0.1
LAM_GP  = 0.01  # gradient penalty lambda

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h367_gradient_penalty_with_mixup_output.txt")


def grad_penalty(model, xb, yb):
    xb_req = xb.detach().requires_grad_(True)
    out = model(xb_req)
    ce = F.cross_entropy(out, yb)
    g, = torch.autograd.grad(ce, xb_req, create_graph=True)
    return (g.view(g.size(0), -1) ** 2).sum(1).mean()


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


def train_with_mode(model, Xtr, Ytr, mode):
    """mode: std_mixup | adv_mixup | std_gradpen | adv_mixup_gradpen"""
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
            opt.zero_grad()

            use_mixup = "mixup" in mode
            use_adv   = "adv" in mode
            use_gp    = "gradpen" in mode

            if use_adv:
                x_adv = C.fgsm(model, xb, yb, eps=EPS)
                # adversarial mixup: x_mix = 0.5*clean + 0.5*adv
                x_in = 0.5 * xb + 0.5 * x_adv
            elif use_mixup:
                lam = np.random.beta(0.2, 0.2)
                perm2 = torch.randperm(xb.size(0), device=xb.device)
                x_in = lam * xb + (1 - lam) * xb[perm2]
            else:
                x_in = xb

            out = model(x_in)
            if use_mixup and not use_adv:
                ce = lam * F.cross_entropy(out, yb) + (1 - lam) * F.cross_entropy(out, yb[perm2])
            else:
                ce = F.cross_entropy(out, yb)

            if use_gp:
                gp = grad_penalty(model, xb, yb)
                loss = ce + LAM_GP * gp
            else:
                loss = ce

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = ["H367: Gradient Penalty with Mixup (AugMax-style)\n" + "="*50]
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    modes = ["std_mixup", "adv_mixup", "std_gradpen", "adv_mixup_gradpen"]

    for mode in modes:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        train_with_mode(model, Xtr, Ytr, mode)
        res = eval_model(model, Xte, Yte)
        line = (f"mode={mode} | clean={res['clean_acc']:.3f} "
                f"fgsm_asr={res['fgsm_asr']:.3f} pgd_asr={res['pgd_asr']:.3f} "
                f"margin={res['mean_margin']:.3f}")
        print(line)
        lines.append(line)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
