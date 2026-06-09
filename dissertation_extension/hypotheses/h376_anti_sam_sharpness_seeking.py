"""
H376 – Anti-SAM: deliberately seek sharp minima.

AT ends in sharper weight-space minima (Liu et al. 2020, Chen & Gu 2021).
Anti-SAM: perturb weights to MAXIMISE loss, then compute gradient at that
perturbed point, then update original weights with that gradient.
This is the OPPOSITE of SAM (which perturbs to maximise, then minimises).
ρ grid: [0, 0.01, 0.05, 0.1].
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

def train_anti_sam(model, Xtr, Ytr, rho):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            if rho > 0:
                # Step 1: compute gradient at current weights
                opt.zero_grad()
                loss1 = F.cross_entropy(model(xb), yb)
                loss1.backward()

                # Compute perturbation direction (ascent)
                grad_norm = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None))
                if grad_norm > 1e-12:
                    # Save original weights, perturb to sharp point
                    old_params = []
                    for p in model.parameters():
                        old_params.append(p.data.clone())
                        if p.grad is not None:
                            p.data.add_(rho * p.grad / grad_norm)  # ascend

                    # Step 2: compute gradient at perturbed point
                    opt.zero_grad()
                    loss2 = F.cross_entropy(model(xb), yb)
                    loss2.backward()

                    # Restore original weights
                    for p, old in zip(model.parameters(), old_params):
                        p.data.copy_(old)

                    # Step 3: update with gradient from perturbed point
                    opt.step()
                else:
                    opt.step()
            else:
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()
        sched.step()
    model.eval()
    return model

def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    rhos = [0, 0.01, 0.05, 0.1]
    results = {}
    for rho in rhos:
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        model = train_anti_sam(model, Xtr, Ytr, rho)
        res = eval_model(model, Xte, Yte)
        res["rho"] = rho
        res["time"] = time.time() - t0
        results[f"rho={rho}"] = res
        print(f"  rho={rho}: {res}")

    print("\n=== H376 RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: clean={v['clean_acc']:.4f} fgsm_asr={v['fgsm_asr']:.4f} "
              f"pgd_asr={v['pgd_asr']:.4f} margin={v['mean_margin']:.4f}")

    base = results["rho=0"]["pgd_asr"]
    best_rho = min([r for r in rhos if r > 0],
                   key=lambda r: results[f"rho={r}"]["pgd_asr"])
    best = results[f"rho={best_rho}"]
    delta = base - best["pgd_asr"]
    print(f"\nBest rho={best_rho}: PGD ASR reduced by {delta:.4f}")
    print(f"SUPPORTED = {delta > 0.02}")

if __name__ == "__main__":
    main()
