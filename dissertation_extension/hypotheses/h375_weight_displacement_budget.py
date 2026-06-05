"""
H375 – Weight displacement budget: constrain ||W_t - W_0|| to AT-like levels.

AT ends closer to init (8.2 vs 9.5 L2 displacement). After each optimizer step,
project weights back if displacement exceeds budget.
Budget grid: [7.0, 8.0, 9.0, 10.0 (unconstrained)].
"""
import os, sys, time, copy
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

def _displacement(model, init_params):
    diff = []
    for p, p0 in zip(model.parameters(), init_params):
        diff.append((p.data - p0).flatten())
    return torch.cat(diff).norm().item()

def _project(model, init_params, budget):
    """Project weights back to ball of radius budget around init."""
    diff = []
    for p, p0 in zip(model.parameters(), init_params):
        diff.append((p.data - p0).flatten())
    d = torch.cat(diff)
    norm = d.norm().item()
    if norm > budget:
        scale = budget / norm
        for p, p0 in zip(model.parameters(), init_params):
            p.data.copy_(p0 + (p.data - p0) * scale)

def train_budget(model, Xtr, Ytr, budget):
    init_params = [p.data.clone() for p in model.parameters()]
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
            loss.backward()
            opt.step()
            if budget < 10.0:
                _project(model, init_params, budget)
        sched.step()
    model.eval()
    final_disp = _displacement(model, init_params)
    return model, final_disp

def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    budgets = [7.0, 8.0, 9.0, 10.0]
    results = {}
    for b in budgets:
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        model, final_disp = train_budget(model, Xtr, Ytr, b)
        res = eval_model(model, Xte, Yte)
        res["budget"] = b
        res["final_displacement"] = final_disp
        res["time"] = time.time() - t0
        label = f"budget={b}" if b < 10 else "unconstrained"
        results[label] = res
        print(f"  {label}: disp={final_disp:.2f} {res}")

    print("\n=== H375 RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: clean={v['clean_acc']:.4f} fgsm_asr={v['fgsm_asr']:.4f} "
              f"pgd_asr={v['pgd_asr']:.4f} margin={v['mean_margin']:.4f} disp={v['final_displacement']:.2f}")

    base = results["unconstrained"]["pgd_asr"]
    best_b = min([b for b in budgets if b < 10],
                 key=lambda b: results[f"budget={b}"]["pgd_asr"])
    best = results[f"budget={best_b}"]
    delta = base - best["pgd_asr"]
    print(f"\nBest budget={best_b}: PGD ASR reduced by {delta:.4f}")
    print(f"SUPPORTED = {delta > 0.02}")

if __name__ == "__main__":
    main()
