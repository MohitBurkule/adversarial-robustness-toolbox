"""
H380 – Weight norm trajectory control: penalise ||W_t||_F growth rate.

AT weight norm grows more slowly. Penalise growth beyond allowed_growth.
penalty = max(0, ||W_t||_F - ||W_0||_F - allowed_growth).
allowed_growth = 1.0 (AT-like). λ grid: [0, 0.1, 1.0].
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

def _weight_norm(model):
    return torch.sqrt(sum((p.data**2).sum() for p in model.parameters())).item()

def train_norm_control(model, Xtr, Ytr, lam, allowed_growth=1.0):
    init_norm = _weight_norm(model)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])

            if lam > 0:
                # Differentiable norm penalty
                cur_norm = torch.sqrt(sum((p**2).sum() for p in model.parameters()))
                excess = F.relu(cur_norm - init_norm - allowed_growth)
                loss = loss + lam * excess

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    final_norm = _weight_norm(model)
    return model, final_norm - init_norm

def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    lambdas = [0, 0.1, 1.0]
    results = {}
    for lam in lambdas:
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        model, norm_growth = train_norm_control(model, Xtr, Ytr, lam)
        res = eval_model(model, Xte, Yte)
        res["lambda"] = lam
        res["norm_growth"] = norm_growth
        res["time"] = time.time() - t0
        results[f"lam={lam}"] = res
        print(f"  lam={lam}: growth={norm_growth:.3f} {res}")

    print("\n=== H380 RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: clean={v['clean_acc']:.4f} fgsm_asr={v['fgsm_asr']:.4f} "
              f"pgd_asr={v['pgd_asr']:.4f} margin={v['mean_margin']:.4f} growth={v['norm_growth']:.3f}")

    base = results["lam=0"]["pgd_asr"]
    best_l = min([l for l in lambdas if l > 0],
                 key=lambda l: results[f"lam={l}"]["pgd_asr"])
    best = results[f"lam={best_l}"]
    delta = base - best["pgd_asr"]
    print(f"\nBest lambda={best_l}: PGD ASR reduced by {delta:.4f}")
    print(f"SUPPORTED = {delta > 0.02}")

if __name__ == "__main__":
    main()
