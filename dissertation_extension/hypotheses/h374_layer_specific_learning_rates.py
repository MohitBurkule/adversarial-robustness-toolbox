"""
H374 – Layer-specific learning rates mimicking AT weight displacement pattern.

AT reshapes early layers more (1.5x displacement), head less (0.8x).
Mimic with per-layer LR multipliers.
Conditions: uniform, AT-mimicking [1.5, 1.0, 0.8, 0.5], inverse [0.5, 0.8, 1.0, 1.5].
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

def _get_param_groups(model, multipliers):
    """multipliers = [block0_mult, block1_mult, block2_mult, head_mult]"""
    groups = []
    groups.append({"params": list(model.features[0:4].parameters()), "lr": LR * multipliers[0]})
    groups.append({"params": list(model.features[4:8].parameters()), "lr": LR * multipliers[1]})
    groups.append({"params": list(model.features[8:12].parameters()), "lr": LR * multipliers[2]})
    groups.append({"params": list(model.head.parameters()), "lr": LR * multipliers[3]})
    return groups

def train_layerlr(model, Xtr, Ytr, multipliers):
    groups = _get_param_groups(model, multipliers)
    opt = torch.optim.SGD(groups, lr=LR, momentum=0.9, weight_decay=5e-4)
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
    return model

def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    conditions = {
        "uniform":    [1.0, 1.0, 1.0, 1.0],
        "at_mimic":   [1.5, 1.0, 0.8, 0.5],
        "inverse":    [0.5, 0.8, 1.0, 1.5],
    }
    results = {}
    for name, mults in conditions.items():
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        model = train_layerlr(model, Xtr, Ytr, mults)
        res = eval_model(model, Xte, Yte)
        res["multipliers"] = mults
        res["time"] = time.time() - t0
        results[name] = res
        print(f"  {name}: {res}")

    print("\n=== H374 RESULTS ===")
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
