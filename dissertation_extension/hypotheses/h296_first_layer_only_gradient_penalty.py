"""H296: First-layer-only gradient penalty (penalise gradient w.r.t. block0 output)."""
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

def train_with_first_layer_penalty(meta, Xtr, Ytr, lam):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)

    # SmallCNN.features[0:4] = block0 (Conv, BN, ReLU, MaxPool)
    # features[4:] = rest
    block0_end_idx = 3  # MaxPool2d is at index 3

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()

            if lam > 0:
                # Pass through block0, capture output
                h = xb.detach()
                for k in range(block0_end_idx + 1):
                    h = model.features[k](h)
                h.requires_grad_(True)
                # Pass rest of model
                h2 = h
                for k in range(block0_end_idx + 1, len(model.features)):
                    h2 = model.features[k](h2)
                out = model.head(h2)
                loss = F.cross_entropy(out, yb)
                g_h, = torch.autograd.grad(loss, h, create_graph=True)
                penalty = lam * (g_h ** 2).mean()
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
    lambdas = [0, 0.001, 0.01, 0.1]
    results = []
    for lam in lambdas:
        t0 = time.time()
        model = train_with_first_layer_penalty(meta, Xtr, Ytr, lam)
        t1 = time.time()
        r = eval_model(model, Xte, Yte)
        r["time_s"] = round(t1-t0, 1)
        r["lam"] = lam
        results.append(r)
        print(f"lam={lam}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
              f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s")

    import os; os.makedirs("results/fashion_mnist", exist_ok=True)
    out = "results/fashion_mnist/h296_first_layer_only_gradient_penalty_output.txt"
    with open(out, "w") as f:
        f.write("H296: First-Layer-Only Gradient Penalty\n")
        f.write("="*60 + "\n")
        for r in results:
            f.write(f"lam={r['lam']}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
                    f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s\n")
    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
