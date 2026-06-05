"""H298: Gradient cosine alignment penalty — force gradients across batch to align."""
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

def train_with_cosine_penalty(meta, Xtr, Ytr, lam):
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
                # Compute per-sample gradients via vmap-style approach using individual grad
                # Efficient: compute jacobian of loss w.r.t. flat x
                xb_req = xb.detach().requires_grad_(True)
                # Compute per-sample losses, then sum for jacobian
                out = model(xb_req)
                losses = F.cross_entropy(out, yb, reduction='none')
                # Gradient of sum = sum of gradients, but we need per-sample
                # Use autograd with create_graph for the penalty
                g, = torch.autograd.grad(losses.sum(), xb_req, create_graph=True)
                # g shape: (B, C, H, W)
                g_flat = g.flatten(1)  # (B, D)
                # Mean gradient
                g_mean = g_flat.mean(0, keepdim=True)  # (1, D)
                # Cosine similarity of each sample gradient with mean
                cos_sim = F.cosine_similarity(g_flat, g_mean.expand_as(g_flat), dim=1)
                # Penalty = 1 - mean cosine similarity (want high similarity)
                penalty = lam * (1 - cos_sim.mean())
                clean_loss = losses.mean()
                (clean_loss + penalty).backward()
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
    for lam in [0, 0.001, 0.01, 0.1]:
        t0 = time.time()
        model = train_with_cosine_penalty(meta, Xtr, Ytr, lam)
        t1 = time.time()
        r = eval_model(model, Xte, Yte)
        r.update({"lam": lam, "time_s": round(t1-t0, 1)})
        results.append(r)
        print(f"lam={lam}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
              f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s")

    import os; os.makedirs("results/fashion_mnist", exist_ok=True)
    out = "results/fashion_mnist/h298_gradient_cosine_alignment_penalty_output.txt"
    with open(out, "w") as f:
        f.write("H298: Gradient Cosine Alignment Penalty\n" + "="*60 + "\n")
        for r in results:
            f.write(f"lam={r['lam']}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
                    f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s\n")
    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
