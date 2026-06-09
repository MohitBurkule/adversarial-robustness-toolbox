"""H300: Compare 4 gradient clipping strategies during training."""
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

def train_with_clipping(meta, Xtr, Ytr, clip_mode, clip_val):
    """clip_mode: 'none', 'global_norm', 'per_layer_norm', 'value'"""
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
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            if clip_mode == 'global_norm':
                nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            elif clip_mode == 'per_layer_norm':
                for p in model.parameters():
                    if p.grad is not None:
                        nn.utils.clip_grad_norm_([p], clip_val)
            elif clip_mode == 'value':
                nn.utils.clip_grad_value_(model.parameters(), clip_val)
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    conditions = [("none", None, None)]
    for T in [0.5, 1.0, 5.0]:
        conditions.append(("global_norm", T, T))
        conditions.append(("per_layer_norm", T, T))
        conditions.append(("value", T, T))

    results = []
    for mode, clip_val, T in conditions:
        t0 = time.time()
        model = train_with_clipping(meta, Xtr, Ytr, mode, clip_val)
        t1 = time.time()
        r = eval_model(model, Xte, Yte)
        r.update({"mode": mode, "T": T, "time_s": round(t1-t0, 1)})
        results.append(r)
        print(f"mode={mode} T={T}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
              f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s")

    import os; os.makedirs("results/fashion_mnist", exist_ok=True)
    out = "results/fashion_mnist/h300_gradient_clipping_robustness_output.txt"
    with open(out, "w") as f:
        f.write("H300: Gradient Clipping Robustness\n" + "="*60 + "\n")
        for r in results:
            f.write(f"mode={r['mode']} T={r['T']}: clean={r['clean_acc']:.3f} "
                    f"fgsm_asr={r['fgsm_asr']:.3f} pgd_asr={r['pgd_asr']:.3f} "
                    f"margin={r['mean_margin']:.3f} time={r['time_s']}s\n")
    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
