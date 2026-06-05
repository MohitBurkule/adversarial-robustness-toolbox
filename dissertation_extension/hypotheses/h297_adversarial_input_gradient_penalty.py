"""H297: Adversarial input gradient penalty — penalise ||grad_{x_adv} L(x_adv)||^2."""
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

def train_with_adv_grad_penalty(meta, Xtr, Ytr, lam, on_adv=True):
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
            if lam > 0:
                # Standard clean loss
                out_clean = model(xb)
                loss_clean = F.cross_entropy(out_clean, yb)
                if on_adv:
                    # Compute FGSM adversarial (no graph through this)
                    xb_tmp = xb.detach().requires_grad_(True)
                    g_fgsm, = torch.autograd.grad(
                        F.cross_entropy(model(xb_tmp), yb), xb_tmp)
                    x_adv = (xb + EPS * g_fgsm.sign()).clamp(0, 1).detach()
                    x_adv.requires_grad_(True)
                    loss_adv = F.cross_entropy(model(x_adv), yb)
                    g_adv, = torch.autograd.grad(loss_adv, x_adv, create_graph=True)
                    penalty = lam * (g_adv ** 2).sum(dim=(1,2,3)).mean()
                    (loss_clean + penalty).backward()
                else:
                    x_clean = xb.requires_grad_(True)
                    loss2 = F.cross_entropy(model(x_clean), yb)
                    g_c, = torch.autograd.grad(loss2, x_clean, create_graph=True)
                    penalty = lam * (g_c ** 2).sum(dim=(1,2,3)).mean()
                    (loss_clean + penalty).backward()
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
    # Adversarial gradient penalty
    for lam in [0, 0.001, 0.01, 0.1]:
        t0 = time.time()
        model = train_with_adv_grad_penalty(meta, Xtr, Ytr, lam, on_adv=True)
        t1 = time.time()
        r = eval_model(model, Xte, Yte)
        r.update({"lam": lam, "mode": "adv_grad_penalty", "time_s": round(t1-t0, 1)})
        results.append(r)
        print(f"adv_grad lam={lam}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
              f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f}")

    # Clean gradient penalty for comparison
    for lam in [0.001, 0.01, 0.1]:
        t0 = time.time()
        model = train_with_adv_grad_penalty(meta, Xtr, Ytr, lam, on_adv=False)
        t1 = time.time()
        r = eval_model(model, Xte, Yte)
        r.update({"lam": lam, "mode": "clean_grad_penalty", "time_s": round(t1-t0, 1)})
        results.append(r)
        print(f"clean_grad lam={lam}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
              f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f}")

    import os; os.makedirs("results/fashion_mnist", exist_ok=True)
    out = "results/fashion_mnist/h297_adversarial_input_gradient_penalty_output.txt"
    with open(out, "w") as f:
        f.write("H297: Adversarial Input Gradient Penalty\n" + "="*60 + "\n")
        for r in results:
            f.write(f"{r['mode']} lam={r['lam']}: clean={r['clean_acc']:.3f} "
                    f"fgsm_asr={r['fgsm_asr']:.3f} pgd_asr={r['pgd_asr']:.3f} "
                    f"margin={r['mean_margin']:.3f} time={r['time_s']}s\n")
    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
