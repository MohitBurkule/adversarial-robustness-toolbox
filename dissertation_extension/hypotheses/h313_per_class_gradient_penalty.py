"""H313: Per-class gradient penalty magnitudes based on per-class accuracy."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign import common as C

N_TRAIN = 6000; EPOCHS = 10; LR = 0.05; BATCH = 128; SEED = 0
LAM = 0.01
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def per_class_acc(model, Xval, Yval, n_classes=10):
    model.eval()
    with torch.no_grad():
        logits, _ = C.logits_and_acc(model, Xval, Yval)
        preds = logits.argmax(1)
    accs = []
    Yval_cpu = Yval.cpu()
    for c in range(n_classes):
        mask = Yval_cpu == c
        if mask.sum() == 0:
            accs.append(1.0)
        else:
            accs.append(float((preds[mask] == Yval_cpu[mask]).float().mean()))
    return accs

def train_per_class_gp(use_per_class, Xtr, Ytr, meta, n_classes=10):
    C.set_seed(SEED)
    # Split train into train/val for computing per-class acc
    n = Xtr.size(0)
    val_size = 600
    Xval, Yval = Xtr[:val_size], Ytr[:val_size]
    Xtr2, Ytr2 = Xtr[val_size:], Ytr[val_size:]

    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # Initially uniform class weights
    class_lams = torch.ones(n_classes, device=C.DEVICE) * LAM

    n2 = Xtr2.size(0)
    for ep in range(EPOCHS):
        model.train()
        # Update per-class lambda every 2 epochs based on val accuracy
        if use_per_class and ep % 2 == 0 and ep > 0:
            accs = per_class_acc(model, Xval, Yval, n_classes)
            accs_t = torch.tensor(accs, device=C.DEVICE)
            # lambda_c = LAM * (1 / acc_c), capped to avoid explosion
            class_lams = LAM * (1.0 / (accs_t + 1e-3))
            class_lams = class_lams.clamp(0, LAM * 10)
            model.train()

        perm = torch.randperm(n2, device=Xtr2.device)
        for i in range(0, n2, BATCH):
            idx = perm[i:i+BATCH]
            xb = Xtr2[idx].clone().detach().requires_grad_(True)
            yb = Ytr2[idx]
            opt.zero_grad()
            out = model(xb)
            ce_loss = F.cross_entropy(out, yb)
            # Per-sample lambda based on class
            lam_per_sample = class_lams[yb]  # (batch,)
            g = torch.autograd.grad(ce_loss, xb, create_graph=True)[0]
            g_norm_sq = g.flatten(1).norm(dim=1).pow(2)  # (batch,)
            gp = (lam_per_sample * g_norm_sq).mean()
            loss = ce_loss + gp
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    lines = ["H313: Per-Class Gradient Penalty vs Uniform Penalty", "="*60]
    for name, use_per_class in [("uniform_lambda", False), ("per_class_lambda", True)]:
        model = train_per_class_gp(use_per_class, Xtr, Ytr, meta)
        res = eval_model(model, Xte, Yte)
        line = (f"{name}: clean_acc={res['clean_acc']:.4f}, "
                f"fgsm_asr={res['fgsm_asr']:.4f}, pgd_asr={res['pgd_asr']:.4f}, "
                f"mean_margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line)
    out_path = os.path.join(RESULTS_DIR, "h313_per_class_gradient_penalty_output.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
