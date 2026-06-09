"""H304: TRADES objective (Zhang et al. 2019) on Fashion-MNIST."""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/run/media/mohit/NewVolume1/adversarial-robustness-toolbox/dissertation_extension")
from campaign import common as C

N_TRAIN, EPOCHS, LR, BATCH, SEED = 6000, 10, 0.05, 128, 0
EPS, PGD_STEPS, PGD_ALPHA = 0.1, 10, 0.01

def eval_model(model, Xte, Yte):
    for p in model.parameters(): p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))
    return dict(clean_acc=float(clean_acc), fgsm_asr=1-float(acc_fgsm),
                pgd_asr=1-float(acc_pgd), mean_margin=mean_margin)

def pgd_on_kl(model, x, steps=10, eps=0.1, alpha=0.01):
    """PGD maximising KL divergence from clean logits (TRADES inner step)."""
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)
    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(0, 1)
    x0 = x.detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        kl = F.kl_div(F.log_softmax(model(x_adv), dim=1), p_clean, reduction='batchmean')
        g, = torch.autograd.grad(kl, x_adv)
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1)
    return x_adv.detach()

def train_trades(meta, Xtr, Ytr, beta):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            if beta > 0:
                x_adv = pgd_on_kl(model, xb, steps=10, eps=EPS, alpha=0.01)
                model.train()
                out_clean = model(xb)
                out_adv = model(x_adv)
                loss_ce = F.cross_entropy(out_clean, yb)
                loss_kl = F.kl_div(
                    F.log_softmax(out_adv, dim=1),
                    F.softmax(out_clean, dim=1),
                    reduction='batchmean')
                loss = loss_ce + beta * loss_kl
            else:
                out = model(xb)
                loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    results = []
    for beta in [0, 1, 3, 6]:
        t0 = time.time()
        model = train_trades(meta, Xtr, Ytr, beta)
        t1 = time.time()
        r = eval_model(model, Xte, Yte)
        r.update({"beta": beta, "time_s": round(t1-t0, 1)})
        results.append(r)
        print(f"beta={beta}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
              f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s")

    import os; os.makedirs("results/fashion_mnist", exist_ok=True)
    out = "results/fashion_mnist/h304_trades_objective_output.txt"
    with open(out, "w") as f:
        f.write("H304: TRADES Objective\n" + "="*60 + "\n")
        for r in results:
            f.write(f"beta={r['beta']}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
                    f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s\n")
    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
