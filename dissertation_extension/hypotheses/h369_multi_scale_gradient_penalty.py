"""
H369: Multi-Scale Gradient Penalty.
Penalise input gradient at multiple resolutions via average pooling then upsample.
Scales: [1.0, 0.75, 0.5]. Lambda grid: [0, 0.001, 0.01].
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
SCALES  = [1.0, 0.75, 0.5]
LAMBDAS = [0, 0.001, 0.01]
IMG_SIZE = 28

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h369_multi_scale_gradient_penalty_output.txt")


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


def multi_scale_grad_penalty(model, xb, yb, scales, img_size=28):
    total = 0.0
    for s in scales:
        if s < 1.0:
            sz = max(1, int(img_size * s))
            # downsample then upsample back
            x_small = F.interpolate(xb, size=sz, mode='bilinear', align_corners=False)
            x_up = F.interpolate(x_small, size=img_size, mode='bilinear', align_corners=False)
        else:
            x_up = xb
        x_req = x_up.detach().requires_grad_(True)
        out = model(x_req)
        ce = F.cross_entropy(out, yb)
        g, = torch.autograd.grad(ce, x_req, create_graph=True)
        total = total + (g.view(g.size(0), -1) ** 2).sum(1).mean()
    return total / len(scales)


def train_msgp(model, Xtr, Ytr, lam):
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
            out = model(xb)
            ce = F.cross_entropy(out, yb)
            if lam > 0:
                pen = multi_scale_grad_penalty(model, xb, yb, SCALES, IMG_SIZE)
                loss = ce + lam * pen
            else:
                loss = ce
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = ["H369: Multi-Scale Gradient Penalty\n" + "="*50]
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    for lam in LAMBDAS:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        train_msgp(model, Xtr, Ytr, lam)
        res = eval_model(model, Xte, Yte)
        line = (f"lambda={lam:.4f} | clean={res['clean_acc']:.3f} "
                f"fgsm_asr={res['fgsm_asr']:.3f} pgd_asr={res['pgd_asr']:.3f} "
                f"margin={res['mean_margin']:.3f}")
        print(line)
        lines.append(line)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
