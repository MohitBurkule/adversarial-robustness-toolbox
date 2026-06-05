"""H303: Adversarial Logit Pairing (Kannan et al. 2018)."""
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

def train_alp(meta, Xtr, Ytr, lam, fgsm_at=False):
    """ALP: CE(x,y) + lambda * ||f(x) - f(x_adv)||^2
    fgsm_at: also compare to standard FGSM-AT (lam=0 with adv training)
    """
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

            # Compute FGSM adversarial
            xb_tmp = xb.detach().requires_grad_(True)
            loss_tmp = F.cross_entropy(model(xb_tmp), yb)
            g, = torch.autograd.grad(loss_tmp, xb_tmp)
            x_adv = (xb + EPS * g.sign()).clamp(0, 1).detach()

            if fgsm_at:
                # Standard FGSM adversarial training
                out_adv = model(x_adv)
                loss = F.cross_entropy(out_adv, yb)
            else:
                # ALP: clean CE + lambda * ||f(x) - f(x_adv)||^2
                out_clean = model(xb)
                out_adv = model(x_adv)
                loss_ce = F.cross_entropy(out_clean, yb)
                pairing_loss = lam * ((out_clean - out_adv) ** 2).sum(1).mean()
                loss = loss_ce + pairing_loss

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    results = []

    # Standard FGSM-AT comparison
    t0 = time.time()
    model = train_alp(meta, Xtr, Ytr, lam=0, fgsm_at=True)
    t1 = time.time()
    r = eval_model(model, Xte, Yte)
    r.update({"lam": "fgsm_at", "time_s": round(t1-t0, 1)})
    results.append(r)
    print(f"fgsm_at: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
          f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s")

    # ALP with different lambdas
    for lam in [0, 0.1, 1.0, 10.0]:
        t0 = time.time()
        model = train_alp(meta, Xtr, Ytr, lam=lam, fgsm_at=False)
        t1 = time.time()
        r = eval_model(model, Xte, Yte)
        r.update({"lam": lam, "time_s": round(t1-t0, 1)})
        results.append(r)
        print(f"ALP lam={lam}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
              f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s")

    import os; os.makedirs("results/fashion_mnist", exist_ok=True)
    out = "results/fashion_mnist/h303_adversarial_logit_pairing_output.txt"
    with open(out, "w") as f:
        f.write("H303: Adversarial Logit Pairing\n" + "="*60 + "\n")
        for r in results:
            f.write(f"lam={r['lam']}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
                    f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s\n")
    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
