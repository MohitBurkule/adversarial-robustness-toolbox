"""H295: Combined input + activation gradient penalty during training."""
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

def train_with_penalty(meta, Xtr, Ytr, lam1, lam2):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)

    # Identify the 3 CNN blocks in SmallCNN features
    # features = block0(0:4) + block1(4:8) + block2(8:12)
    block_ends = [3, 7, 11]  # indices of MaxPool2d layers (outputs of each block)

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            opt.zero_grad()

            # Capture activation outputs
            activations = []
            hooks = []

            if lam2 > 0:
                for k in block_ends:
                    layer = model.features[k]
                    def make_hook(lst):
                        def hook(m, inp, out):
                            lst.append(out)
                        return hook
                    hooks.append(layer.register_forward_hook(make_hook(activations)))

            xb_req = xb.detach().requires_grad_(lam1 > 0)
            out = model(xb_req)
            loss = F.cross_entropy(out, yb)

            total_penalty = torch.tensor(0.0, device=C.DEVICE)

            if lam1 > 0:
                g_x, = torch.autograd.grad(loss, xb_req, create_graph=True)
                total_penalty = total_penalty + lam1 * (g_x ** 2).sum(dim=(1,2,3)).mean()

            if lam2 > 0:
                for h in hooks:
                    h.remove()
                for act in activations:
                    g_h, = torch.autograd.grad(loss, act, retain_graph=True, create_graph=True)
                    total_penalty = total_penalty + lam2 * (g_h ** 2).mean()

            total_loss = loss + total_penalty
            total_loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    conditions = [
        ("baseline",        0.0,  0.0),
        ("lam1=0.01",       0.01, 0.0),
        ("lam2=0.01",       0.0,  0.01),
        ("lam1=lam2=0.01",  0.01, 0.01),
        ("lam1=0.01,lam2=0.001", 0.01, 0.001),
    ]
    results = []
    for name, lam1, lam2 in conditions:
        t0 = time.time()
        model = train_with_penalty(meta, Xtr, Ytr, lam1, lam2)
        t1 = time.time()
        r = eval_model(model, Xte, Yte)
        r["time_s"] = round(t1-t0, 1)
        r["condition"] = name
        results.append(r)
        print(f"{name}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
              f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s")

    out = "results/fashion_mnist/h295_combined_input_activation_gradient_penalty_output.txt"
    import os; os.makedirs("results/fashion_mnist", exist_ok=True)
    with open(out, "w") as f:
        f.write("H295: Combined Input + Activation Gradient Penalty\n")
        f.write("="*60 + "\n")
        for r in results:
            f.write(f"{r['condition']}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
                    f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s\n")
    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
