"""
H377 – Gradient step length budget: constrain total path length in weight space.

AT travels shorter total path (13.1 vs 16.3). Constrain cumulative step length.
Effective LR at step t = LR * max(0, 1 - cumulative_path / path_budget).
Path budget grid: [10.0, 13.0, 16.0 (unconstrained)].
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

def train_path_budget(model, Xtr, Ytr, path_budget):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    cumulative_path = 0.0
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            loss.backward()

            if path_budget < 16.0:
                # Scale gradient to respect budget
                scale = max(0.0, 1.0 - cumulative_path / path_budget)
                if scale < 1.0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)

            # Snapshot before step to measure step length
            old = [p.data.clone() for p in model.parameters()]
            opt.step()
            # Measure actual step length
            step_len = torch.sqrt(sum(((p.data - o)**2).sum()
                                      for p, o in zip(model.parameters(), old))).item()
            cumulative_path += step_len
        sched.step()
    model.eval()
    return model, cumulative_path

def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    budgets = [10.0, 13.0, 16.0]
    results = {}
    for b in budgets:
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        model, total_path = train_path_budget(model, Xtr, Ytr, b)
        res = eval_model(model, Xte, Yte)
        res["path_budget"] = b
        res["actual_path"] = total_path
        res["time"] = time.time() - t0
        label = f"budget={b}" if b < 16 else "unconstrained"
        results[label] = res
        print(f"  {label}: path={total_path:.2f} {res}")

    print("\n=== H377 RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: clean={v['clean_acc']:.4f} fgsm_asr={v['fgsm_asr']:.4f} "
              f"pgd_asr={v['pgd_asr']:.4f} margin={v['mean_margin']:.4f} path={v['actual_path']:.2f}")

    base = results["unconstrained"]["pgd_asr"]
    best_b = min([b for b in budgets if b < 16],
                 key=lambda b: results[f"budget={b}"]["pgd_asr"])
    best = results[f"budget={best_b}"]
    delta = base - best["pgd_asr"]
    print(f"\nBest budget={best_b}: PGD ASR reduced by {delta:.4f}")
    print(f"SUPPORTED = {delta > 0.02}")

if __name__ == "__main__":
    main()
