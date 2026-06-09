"""H325: Confidence regularisation vs entropy regularisation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

N_TRAIN, EPOCHS, LR, BATCH, SEED = 6000, 10, 0.05, 128, 0
OUT = "results/fashion_mnist/h325_confidence_regularisation_output.txt"
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

def train_condition(model, Xtr, Ytr, mode="baseline", lam=0.0):
    C.set_seed(SEED)
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
            out = model(xb)
            ce = F.cross_entropy(out, yb)
            if mode == "conf" and lam > 0:
                p = F.softmax(out, dim=1)
                conf_penalty = p.max(dim=1).values.mean()
                loss = ce + lam * conf_penalty
            elif mode == "entropy" and lam > 0:
                p = F.softmax(out, dim=1)
                entropy = -(p * (p + 1e-8).log()).sum(dim=1).mean()
                loss = ce + lam * (-entropy)  # penalise low entropy = penalise overconfidence
            else:
                loss = ce
            loss.backward()
            opt.step()
        sched.step()
    model.eval()

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    lines = ["H325: Confidence Regularisation\n"]

    conditions = [
        ("baseline",          "baseline", 0.0),
        ("conf(lam=0.01)",    "conf",     0.01),
        ("conf(lam=0.1)",     "conf",     0.1),
        ("conf(lam=1.0)",     "conf",     1.0),
        ("entropy(lam=0.01)", "entropy",  0.01),
        ("entropy(lam=0.1)",  "entropy",  0.1),
        ("entropy(lam=1.0)",  "entropy",  1.0),
    ]
    for name, mode, lam in conditions:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_condition(model, Xtr, Ytr, mode=mode, lam=lam)
        res = eval_model(model, Xte, Yte)
        line = (f"{name}: clean={res['clean_acc']:.4f} fgsm_asr={res['fgsm_asr']:.4f} "
                f"pgd_asr={res['pgd_asr']:.4f} margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line + "\n")

    with open(OUT, "w") as f: f.writelines(lines)
    print(f"Saved to {OUT}")

if __name__ == "__main__":
    main()
