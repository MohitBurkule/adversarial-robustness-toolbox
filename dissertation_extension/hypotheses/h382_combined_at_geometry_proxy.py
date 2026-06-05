"""
H382 – Combined AT geometry proxy: can we approximate AT robustness without
adversarial examples by combining the best geometry constraints?

Conditions:
  (a) baseline (standard training)
  (b) gradient coherence forcing τ=0.2
  (c) layer LRs [1.5, 1.0, 0.8, 0.5]
  (d) all three combined (coherence + layer LRs + displacement budget=8.0)
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

def _project(model, init_params, budget):
    diff = []
    for p, p0 in zip(model.parameters(), init_params):
        diff.append((p.data - p0).flatten())
    d = torch.cat(diff)
    norm = d.norm().item()
    if norm > budget:
        scale = budget / norm
        for p, p0 in zip(model.parameters(), init_params):
            p.data.copy_(p0 + (p.data - p0) * scale)

def _get_param_groups(model, multipliers):
    groups = []
    groups.append({"params": list(model.features[0:4].parameters()), "lr": LR * multipliers[0]})
    groups.append({"params": list(model.features[4:8].parameters()), "lr": LR * multipliers[1]})
    groups.append({"params": list(model.features[8:12].parameters()), "lr": LR * multipliers[2]})
    groups.append({"params": list(model.head.parameters()), "lr": LR * multipliers[3]})
    return groups

def train_combined(model, Xtr, Ytr, use_coherence=False, tau=0.2,
                   layer_mults=None, use_budget=False, budget=8.0):
    init_params = [p.data.clone() for p in model.parameters()] if use_budget else None

    if layer_mults is not None:
        groups = _get_param_groups(model, layer_mults)
        opt = torch.optim.SGD(groups, lr=LR, momentum=0.9, weight_decay=5e-4)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    prev_grad = None
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            loss.backward()

            if use_coherence and prev_grad is not None:
                cur = _flat_grad(model)
                cos = F.cosine_similarity(cur.unsqueeze(0), prev_grad.unsqueeze(0)).item()
                if cos < tau:
                    scale = max(0.0, cos / tau)
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)
            if use_coherence:
                prev_grad = _flat_grad(model)

            opt.step()
            if use_budget:
                _project(model, init_params, budget)
        sched.step()
    model.eval()
    return model

def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    conditions = {
        "baseline": dict(use_coherence=False, layer_mults=None, use_budget=False),
        "coherence_only": dict(use_coherence=True, tau=0.2, layer_mults=None, use_budget=False),
        "layer_lr_only": dict(use_coherence=False, layer_mults=[1.5, 1.0, 0.8, 0.5], use_budget=False),
        "all_combined": dict(use_coherence=True, tau=0.2, layer_mults=[1.5, 1.0, 0.8, 0.5],
                            use_budget=True, budget=8.0),
    }

    # Also train actual AT for reference
    print("Training AT reference...")
    C.set_seed(SEED)
    at_model = C.build_model("cnn", meta)
    at_model = C.train_model(at_model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                             lr=LR, adv_train=True, adv_eps=EPS, ncls=10)
    at_res = eval_model(at_model, Xte, Yte)
    print(f"  AT: {at_res}")

    results = {"actual_AT": at_res}
    for name, kwargs in conditions.items():
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        model = train_combined(model, Xtr, Ytr, **kwargs)
        res = eval_model(model, Xte, Yte)
        res["time"] = time.time() - t0
        results[name] = res
        print(f"  {name}: {res}")

    print("\n=== H382 RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: clean={v['clean_acc']:.4f} fgsm_asr={v['fgsm_asr']:.4f} "
              f"pgd_asr={v['pgd_asr']:.4f} margin={v['mean_margin']:.4f}")

    base = results["baseline"]["pgd_asr"]
    combined = results["all_combined"]["pgd_asr"]
    at = results["actual_AT"]["pgd_asr"]
    delta = base - combined
    at_delta = base - at
    print(f"\nCombined vs baseline PGD ASR delta: {delta:.4f}")
    print(f"AT vs baseline PGD ASR delta: {at_delta:.4f}")
    if at_delta > 0:
        print(f"Combined recovers {delta/at_delta*100:.1f}% of AT robustness gain")
    print(f"SUPPORTED = {delta > 0.02}")

if __name__ == "__main__":
    main()
