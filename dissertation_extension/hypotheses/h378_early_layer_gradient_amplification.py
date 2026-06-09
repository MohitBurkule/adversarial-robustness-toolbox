"""
H378 – Early-layer gradient amplification matching AT's movement pattern.

Amplify gradients for early layers (matching AT's 1.5x early-layer displacement).
Use gradient hooks: block0 * α_0, block1 * α_1, block2 * α_2, head * α_h.
Conditions: AT-mimic [2.0,1.0,0.5,0.3], uniform [1,1,1,1], inverse [0.3,0.5,1.0,2.0].
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

def train_grad_amp(model, Xtr, Ytr, alphas):
    """alphas = [block0, block1, block2, head] gradient multipliers."""
    # Register gradient hooks on parameters
    blocks = [
        list(model.features[0:4].parameters()),
        list(model.features[4:8].parameters()),
        list(model.features[8:12].parameters()),
        list(model.head.parameters()),
    ]
    hooks = []
    for block_params, alpha in zip(blocks, alphas):
        for p in block_params:
            if alpha != 1.0:
                h = p.register_hook(lambda g, a=alpha: g * a)
                hooks.append(h)

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
        sched.step()
    model.eval()
    for h in hooks:
        h.remove()
    return model

def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    conditions = {
        "uniform":    [1.0, 1.0, 1.0, 1.0],
        "at_mimic":   [2.0, 1.0, 0.5, 0.3],
        "inverse":    [0.3, 0.5, 1.0, 2.0],
    }
    results = {}
    for name, alphas in conditions.items():
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        model = train_grad_amp(model, Xtr, Ytr, alphas)
        res = eval_model(model, Xte, Yte)
        res["alphas"] = alphas
        res["time"] = time.time() - t0
        results[name] = res
        print(f"  {name}: {res}")

    print("\n=== H378 RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: clean={v['clean_acc']:.4f} fgsm_asr={v['fgsm_asr']:.4f} "
              f"pgd_asr={v['pgd_asr']:.4f} margin={v['mean_margin']:.4f}")

    base = results["uniform"]["pgd_asr"]
    mimic = results["at_mimic"]["pgd_asr"]
    delta = base - mimic
    print(f"\nAT-mimic vs uniform PGD ASR delta: {delta:.4f}")
    print(f"SUPPORTED = {delta > 0.02}")

if __name__ == "__main__":
    main()
