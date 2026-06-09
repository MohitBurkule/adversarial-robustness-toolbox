"""
H360: Class-Conditional Gradient Alignment.
Per-class mean input gradient penalty: within-class variance + inter-class similarity.
λ1=0.01 (within-class var), λ2=0.001 (inter-class sim).
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
LAMBDA1 = 0.01   # within-class variance
LAMBDA2 = 0.001  # inter-class similarity

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h360_class_conditional_gradient_alignment_output.txt")


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


def class_gradient_penalty(model, xb, yb, n_classes=10, lam1=0.01, lam2=0.001):
    xb_req = xb.detach().requires_grad_(True)
    out = model(xb_req)
    loss_ce = F.cross_entropy(out, yb)
    grads, = torch.autograd.grad(loss_ce, xb_req, create_graph=True)
    # grads: (B, C, H, W) -> flatten per sample
    g_flat = grads.view(grads.size(0), -1)  # (B, D)

    # per-class mean gradients
    class_means = []
    class_grads = []
    for c in range(n_classes):
        mask = (yb == c)
        if mask.sum() == 0:
            continue
        gc = g_flat[mask]  # (nc, D)
        class_grads.append(gc)
        class_means.append(gc.mean(0))

    # within-class variance penalty
    within_var = 0.0
    for gc, gm in zip(class_grads, class_means):
        within_var = within_var + ((gc - gm.unsqueeze(0)) ** 2).mean()

    # inter-class cosine similarity penalty
    inter_sim = 0.0
    count = 0
    n_cls_present = len(class_means)
    for i in range(n_cls_present):
        for j in range(i+1, n_cls_present):
            gi = class_means[i]
            gj = class_means[j]
            cos = F.cosine_similarity(gi.unsqueeze(0), gj.unsqueeze(0))
            inter_sim = inter_sim + cos.abs()
            count += 1
    if count > 0:
        inter_sim = inter_sim / count

    penalty = lam1 * within_var + lam2 * inter_sim
    return penalty


def train_ccga(model, Xtr, Ytr, lam1, lam2):
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
            if lam1 > 0 or lam2 > 0:
                pen = class_gradient_penalty(model, xb, yb, lam1=lam1, lam2=lam2)
                loss = ce + pen
            else:
                loss = ce
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = ["H360: Class-Conditional Gradient Alignment\n" + "="*50]
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    for lam1, lam2, label in [(0, 0, "baseline"), (LAMBDA1, LAMBDA2, "ccga")]:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        train_ccga(model, Xtr, Ytr, lam1, lam2)
        res = eval_model(model, Xte, Yte)
        line = (f"condition={label} lam1={lam1} lam2={lam2} | "
                f"clean={res['clean_acc']:.3f} fgsm_asr={res['fgsm_asr']:.3f} "
                f"pgd_asr={res['pgd_asr']:.3f} margin={res['mean_margin']:.3f}")
        print(line)
        lines.append(line)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
