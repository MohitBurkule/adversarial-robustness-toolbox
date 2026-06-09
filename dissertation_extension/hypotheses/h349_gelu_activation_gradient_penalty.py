"""
H349 - GELU activation + Input Gradient Norm Penalty (arXiv 2409.20139)

Smooth activations (GELU) make gradient norm regularisation more effective.
Conditions: ReLU+no penalty, ReLU+penalty(λ=0.01), GELU+no penalty, GELU+penalty(λ=0.01).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
LAM = 0.01

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h349_gelu_activation_gradient_penalty_output.txt")


def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1 - float(acc_fgsm),
                pgd_asr=1 - float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))


def train_condition(meta, Xtr, Ytr, act, use_penalty):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, act=act)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            opt.zero_grad()
            if use_penalty:
                xb_req = xb.clone().detach().requires_grad_(True)
                out = model(xb_req)
                loss_ce = F.cross_entropy(out, yb)
                grad_x = torch.autograd.grad(loss_ce, xb_req, create_graph=True)[0]
                grad_norm = grad_x.flatten(1).norm(dim=1).mean()
                loss = loss_ce + LAM * grad_norm
            else:
                out = model(xb)
                loss = F.cross_entropy(out, yb)

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("H349 - GELU Activation + Input Gradient Norm Penalty (arXiv 2409.20139)")
    lines.append("=" * 72)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_TRAIN={N_TRAIN}  lambda={LAM}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    conditions = [
        ("relu", False, "ReLU+no_penalty"),
        ("relu", True,  "ReLU+penalty"),
        ("gelu", False, "GELU+no_penalty"),
        ("gelu", True,  "GELU+penalty"),
    ]
    for act, use_pen, name in conditions:
        t0 = time.time()
        model = train_condition(meta, Xtr, Ytr, act, use_pen)
        metrics = eval_model(model, Xte, Yte)
        metrics["runtime_s"] = round(time.time() - t0, 1)
        line = (f"  {name:<22}  clean={metrics['clean_acc']:.3f}  "
                f"fgsm_asr={metrics['fgsm_asr']:.3f}  pgd_asr={metrics['pgd_asr']:.3f}  "
                f"margin={metrics['mean_margin']:.3f}  ({metrics['runtime_s']}s)")
        lines.append(line)
        print(line)

    lines.append("")
    lines.append("Interpretation: Smooth activations (GELU) enable meaningful gradient norm")
    lines.append("penalties (ReLU has near-zero gradient almost everywhere after training).")
    lines.append("GELU+penalty should show stronger robustness improvement than ReLU+penalty.")
    lines.append("=" * 72)

    out = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(out + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
