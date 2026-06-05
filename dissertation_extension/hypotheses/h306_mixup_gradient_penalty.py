"""H306: Mixup + input gradient penalty combination."""
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

def train_combined(meta, Xtr, Ytr, use_mixup, grad_lam, alpha=0.2):
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

            if use_mixup:
                lam_mix = float(np.random.beta(alpha, alpha))
                perm2 = torch.randperm(xb.size(0), device=xb.device)
                xb_mix = lam_mix * xb + (1 - lam_mix) * xb[perm2]
                yb2 = yb[perm2]
                if grad_lam > 0:
                    xb_mix_req = xb_mix.detach().requires_grad_(True)
                    out = model(xb_mix_req)
                    loss = (lam_mix * F.cross_entropy(out, yb, reduction='none') +
                            (1 - lam_mix) * F.cross_entropy(out, yb2, reduction='none')).mean()
                    g, = torch.autograd.grad(loss, xb_mix_req, create_graph=True)
                    penalty = grad_lam * (g ** 2).sum(dim=(1,2,3)).mean()
                    (loss + penalty).backward()
                else:
                    out = model(xb_mix)
                    loss = (lam_mix * F.cross_entropy(out, yb, reduction='none') +
                            (1 - lam_mix) * F.cross_entropy(out, yb2, reduction='none')).mean()
                    loss.backward()
            else:
                if grad_lam > 0:
                    xb_req = xb.detach().requires_grad_(True)
                    out = model(xb_req)
                    loss = F.cross_entropy(out, yb)
                    g, = torch.autograd.grad(loss, xb_req, create_graph=True)
                    penalty = grad_lam * (g ** 2).sum(dim=(1,2,3)).mean()
                    (loss + penalty).backward()
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
    conditions = [
        ("baseline",            False, 0.0),
        ("mixup_only",          True,  0.0),
        ("grad_penalty_only",   False, 0.01),
        ("mixup_grad_penalty",  True,  0.01),
    ]
    results = []
    for name, use_mixup, grad_lam in conditions:
        t0 = time.time()
        model = train_combined(meta, Xtr, Ytr, use_mixup, grad_lam)
        t1 = time.time()
        r = eval_model(model, Xte, Yte)
        r.update({"condition": name, "time_s": round(t1-t0, 1)})
        results.append(r)
        print(f"{name}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
              f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s")

    import os; os.makedirs("results/fashion_mnist", exist_ok=True)
    out = "results/fashion_mnist/h306_mixup_gradient_penalty_output.txt"
    with open(out, "w") as f:
        f.write("H306: Mixup + Gradient Penalty\n" + "="*60 + "\n")
        for r in results:
            f.write(f"{r['condition']}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
                    f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s\n")
    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
