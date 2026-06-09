"""H302: Gradient Centralization (Yong et al. 2020) for adversarial robustness."""
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

def apply_gc(model):
    """Apply gradient centralization as a gradient hook."""
    hooks = []
    for p in model.parameters():
        if p.dim() > 1:  # Only for weight tensors (not biases)
            def make_hook(param):
                def hook(grad):
                    # Center gradient: subtract mean over all dims except output dim
                    mean = grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)
                    return grad - mean
                return hook
            hooks.append(p.register_hook(make_hook(p)))
    return hooks

def train_with_gc(meta, Xtr, Ytr, mode):
    """mode: 'baseline', 'gc', 'gc_momentum', 'gc_adam'"""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    if mode in ("gc", "gc_momentum"):
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    elif mode == "gc_adam":
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
    else:  # baseline
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    use_gc = mode != "baseline"
    if use_gc:
        hooks = apply_gc(model)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        sched.step()

    if use_gc:
        for h in hooks:
            h.remove()
    model.eval()
    return model

def main():
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    modes = ["baseline", "gc", "gc_momentum", "gc_adam"]
    results = []
    for mode in modes:
        t0 = time.time()
        model = train_with_gc(meta, Xtr, Ytr, mode)
        t1 = time.time()
        r = eval_model(model, Xte, Yte)
        r.update({"mode": mode, "time_s": round(t1-t0, 1)})
        results.append(r)
        print(f"mode={mode}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
              f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s")

    import os; os.makedirs("results/fashion_mnist", exist_ok=True)
    out = "results/fashion_mnist/h302_gradient_centralization_output.txt"
    with open(out, "w") as f:
        f.write("H302: Gradient Centralization\n" + "="*60 + "\n")
        for r in results:
            f.write(f"mode={r['mode']}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
                    f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s\n")
    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
