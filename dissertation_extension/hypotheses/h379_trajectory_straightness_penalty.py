"""
H379 – Trajectory straightness penalty: penalise curvature of weight trajectory.

AT has more coherent gradient direction. Penalise curvature:
curvature_t = 1 - cosine(Δw_t, Δw_{t-1}).
If curvature > τ, scale gradient down (not a differentiable loss term).
τ = 0.8 (allow up to 0.2 cosine deviation). λ grid: [0, 0.01, 0.1].
λ here controls how aggressively we suppress high-curvature updates.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000; EPOCHS = 10; LR = 0.05; BATCH = 128; SEED = 0
EPS = 0.1; PGD_STEPS = 10; PGD_ALPHA = 0.01

def eval_model(model, Xte, Yte):
    for p in model.parameters(): p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1-float(acc_fgsm),
                pgd_asr=1-float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))

def _flat_params(model):
    return torch.cat([p.data.flatten() for p in model.parameters()])

def train_straight(model, Xtr, Ytr, lam, tau=0.8):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    prev_update = None
    prev_params = _flat_params(model)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            loss.backward()

            if lam > 0 and prev_update is not None:
                # Compute what current update would be
                cur_grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                cos = F.cosine_similarity(cur_grad.unsqueeze(0), prev_update.unsqueeze(0)).item()
                curvature = 1.0 - cos
                if curvature > (1.0 - tau):
                    # Scale gradient down proportionally
                    scale = max(0.0, 1.0 - lam * (curvature - (1.0 - tau)))
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)

            before = _flat_params(model)
            opt.step()
            after = _flat_params(model)
            prev_update = after - before
        sched.step()
    model.eval()
    return model

def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    lambdas = [0, 0.01, 0.1]
    results = {}
    for lam in lambdas:
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        model = train_straight(model, Xtr, Ytr, lam)
        res = eval_model(model, Xte, Yte)
        res["lambda"] = lam
        res["time"] = time.time() - t0
        results[f"lam={lam}"] = res
        print(f"  lam={lam}: {res}")

    print("\n=== H379 RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: clean={v['clean_acc']:.4f} fgsm_asr={v['fgsm_asr']:.4f} "
              f"pgd_asr={v['pgd_asr']:.4f} margin={v['mean_margin']:.4f}")

    base = results["lam=0"]["pgd_asr"]
    best_l = min([l for l in lambdas if l > 0],
                 key=lambda l: results[f"lam={l}"]["pgd_asr"])
    best = results[f"lam={best_l}"]
    delta = base - best["pgd_asr"]
    print(f"\nBest lambda={best_l}: PGD ASR reduced by {delta:.4f}")
    print(f"SUPPORTED = {delta > 0.02}")

if __name__ == "__main__":
    main()
