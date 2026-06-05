"""
H359: Adversarial Logit Pairing (ALP) — Kannan et al. 2018.
loss = CE(x,y) + λ * ||f(x) - f(x_adv)||²
x_adv via FGSM(eps=0.1). Forces clean and adversarial logits to match.
Lambda grid: [0, 0.1, 1.0, 10.0].
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
LAMBDAS = [0, 0.1, 1.0, 10.0]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h359_adversarial_logit_pairing_output.txt")


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


def train_alp(model, Xtr, Ytr, lam):
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
            # clean logits
            logits_clean = model(xb)
            ce = F.cross_entropy(logits_clean, yb)
            if lam > 0:
                # FGSM adversarial
                x_adv = C.fgsm(model, xb, yb, eps=EPS)
                logits_adv = model(x_adv)
                alp = ((logits_clean.detach() - logits_adv) ** 2).mean()
                loss = ce + lam * alp
            else:
                loss = ce
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = ["H359: Adversarial Logit Pairing (ALP)\n" + "="*50]
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    for lam in LAMBDAS:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        train_alp(model, Xtr, Ytr, lam)
        res = eval_model(model, Xte, Yte)
        line = (f"lambda={lam:5.1f} | clean={res['clean_acc']:.3f} "
                f"fgsm_asr={res['fgsm_asr']:.3f} pgd_asr={res['pgd_asr']:.3f} "
                f"margin={res['mean_margin']:.3f}")
        print(line)
        lines.append(line)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
