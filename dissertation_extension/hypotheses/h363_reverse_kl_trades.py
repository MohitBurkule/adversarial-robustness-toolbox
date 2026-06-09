"""
H363: Reverse-KL TRADES.
Standard TRADES: CE(x,y) + β*KL(f(x)||f(x_adv)).
Reverse: CE(x,y) + β*KL(f(x_adv)||f(x)).
x_adv via PGD maximising KL(f(x_adv)||f(x)).
Beta grid: [1, 3, 6].
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
PGD_STEPS = 7
BETAS   = [1, 3, 6]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h363_reverse_kl_trades_output.txt")


def pgd_rev_kl(model, x, eps, steps=7):
    """PGD maximising KL(f(x_adv)||f(x)) = reverse KL."""
    alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    with torch.no_grad():
        p_clean = F.softmax(model(x0), dim=1)
    for _ in range(steps):
        xa.requires_grad_(True)
        logits_adv = model(xa)
        # KL(adv||clean) = sum p_adv * log(p_adv/p_clean)
        log_p_adv = F.log_softmax(logits_adv, dim=1)
        p_adv = log_p_adv.exp()
        rev_kl = (p_adv * (log_p_adv - p_clean.log())).sum(1).mean()
        g, = torch.autograd.grad(rev_kl, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


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


def train_rev_kl_trades(model, Xtr, Ytr, beta):
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
            # CE on clean
            out_clean = model(xb)
            ce = F.cross_entropy(out_clean, yb)
            # Reverse-KL regularisation
            x_adv = pgd_rev_kl(model, xb, EPS, steps=PGD_STEPS)
            out_adv = model(x_adv)
            p_clean = F.softmax(out_clean.detach(), dim=1)
            log_p_adv = F.log_softmax(out_adv, dim=1)
            p_adv = log_p_adv.exp()
            rev_kl = (p_adv * (log_p_adv - p_clean.log())).sum(1).mean()
            loss = ce + beta * rev_kl
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = ["H363: Reverse-KL TRADES\n" + "="*50]
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    for beta in BETAS:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        train_rev_kl_trades(model, Xtr, Ytr, beta)
        res = eval_model(model, Xte, Yte)
        line = (f"beta={beta} | clean={res['clean_acc']:.3f} "
                f"fgsm_asr={res['fgsm_asr']:.3f} pgd_asr={res['pgd_asr']:.3f} "
                f"margin={res['mean_margin']:.3f}")
        print(line)
        lines.append(line)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
