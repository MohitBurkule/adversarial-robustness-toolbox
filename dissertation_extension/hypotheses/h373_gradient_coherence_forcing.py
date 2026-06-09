"""
H373 – Gradient coherence forcing during standard training.

AT has higher consecutive-batch gradient cosine similarity (0.223 vs 0.177).
We force this during standard training: after computing batch gradient, check
cosine similarity with previous batch gradient. If cos_sim < threshold τ,
scale gradient down: g_t *= max(0, cos_sim / τ).
τ grid: [0.0 (no filtering), 0.1, 0.2, 0.3].
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

def _flat_grad(model):
    gs = []
    for p in model.parameters():
        if p.grad is not None:
            gs.append(p.grad.detach().flatten())
    return torch.cat(gs) if gs else None

def train_coherence(model, Xtr, Ytr, tau, epochs=EPOCHS, batch=BATCH, lr=LR):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    prev_grad = None
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            if tau > 0 and prev_grad is not None:
                cur = _flat_grad(model)
                cos = F.cosine_similarity(cur.unsqueeze(0), prev_grad.unsqueeze(0)).item()
                if cos < tau:
                    scale = max(0.0, cos / tau)
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)
            prev_grad = _flat_grad(model)
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    taus = [0.0, 0.1, 0.2, 0.3]
    results = {}
    for tau in taus:
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        model = train_coherence(model, Xtr, Ytr, tau)
        res = eval_model(model, Xte, Yte)
        res["tau"] = tau
        res["time"] = time.time() - t0
        results[f"tau={tau}"] = res
        print(f"  tau={tau}: {res}")

    print("\n=== H373 RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: clean={v['clean_acc']:.4f} fgsm_asr={v['fgsm_asr']:.4f} "
              f"pgd_asr={v['pgd_asr']:.4f} margin={v['mean_margin']:.4f}")

    # Does coherence forcing improve robustness?
    base = results["tau=0.0"]
    best_tau = min([t for t in taus if t > 0],
                   key=lambda t: results[f"tau={t}"]["pgd_asr"])
    best = results[f"tau={best_tau}"]
    delta = base["pgd_asr"] - best["pgd_asr"]
    print(f"\nBest tau={best_tau}: PGD ASR reduced by {delta:.4f}")
    print(f"SUPPORTED = {delta > 0.02}")

if __name__ == "__main__":
    main()
