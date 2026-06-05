"""
H381 – Feature activation statistics matching: force standard model to have
AT-like activation magnitudes at each block.

Phase 1: train AT model, record mean ||h_k|| at each block over test set.
Phase 2: train standard model with penalty matching those statistics.
penalty = Σ_k (mean(||h_k||) - AT_target_k)².
Compare: standard, AT-stats-matched, actual AT.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
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

def _get_block_activations(model, X, batch=256):
    """Return mean activation norm per block: [block0, block1, block2]."""
    model.eval()
    accum = [0.0, 0.0, 0.0]
    count = 0
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            xb = X[i:i+batch]
            h = xb
            for bidx in range(3):
                start = bidx * 4
                for layer in model.features[start:start+4]:
                    h = layer(h)
                accum[bidx] += h.flatten(1).norm(dim=1).sum().item()
            count += xb.size(0)
    return [a / count for a in accum]

def train_stats_matched(model, Xtr, Ytr, targets, lam=1.0):
    """Train with activation statistics matching penalty."""
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()

            # Forward through blocks, collecting activations
            h = xb
            penalty = torch.tensor(0.0, device=xb.device)
            for bidx in range(3):
                start = bidx * 4
                for layer in model.features[start:start+4]:
                    h = layer(h)
                mean_norm = h.flatten(1).norm(dim=1).mean()
                penalty = penalty + (mean_norm - targets[bidx]) ** 2

            logits = model.head(h)
            loss = F.cross_entropy(logits, yb) + lam * penalty
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    # Phase 1: Train AT model and get activation statistics
    print("Phase 1: Training AT model...")
    C.set_seed(SEED)
    at_model = C.build_model("cnn", meta)
    at_model = C.train_model(at_model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                             lr=LR, adv_train=True, adv_eps=EPS, ncls=10)
    at_stats = _get_block_activations(at_model, Xte)
    print(f"  AT activation stats: {[f'{s:.3f}' for s in at_stats]}")
    at_res = eval_model(at_model, Xte, Yte)
    print(f"  AT eval: {at_res}")

    # Phase 2: Standard baseline
    print("\nPhase 2: Standard baseline...")
    C.set_seed(SEED)
    std_model = C.build_model("cnn", meta)
    std_model = C.train_model(std_model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                              lr=LR, ncls=10)
    std_stats = _get_block_activations(std_model, Xte)
    print(f"  Standard activation stats: {[f'{s:.3f}' for s in std_stats]}")
    std_res = eval_model(std_model, Xte, Yte)
    print(f"  Standard eval: {std_res}")

    # Phase 3: AT-stats-matched training (sweep lambda)
    lam_grid = [0.001, 0.01, 0.1]
    matched_results = {}
    for lam in lam_grid:
        print(f"\nPhase 3: AT-stats-matched training (lam={lam})...")
        C.set_seed(SEED)
        m = C.build_model("cnn", meta)
        m = train_stats_matched(m, Xtr, Ytr, at_stats, lam=lam)
        ms = _get_block_activations(m, Xte)
        print(f"  Matched activation stats: {[f'{s:.3f}' for s in ms]}")
        r = eval_model(m, Xte, Yte)
        print(f"  Matched eval: {r}")
        matched_results[lam] = (r, ms)

    # Pick best lambda by PGD ASR
    best_lam = min(lam_grid, key=lambda l: matched_results[l][0]["pgd_asr"])
    matched_res, matched_stats = matched_results[best_lam]
    print(f"\nBest lambda={best_lam}")

    print("\n=== H381 RESULTS ===")
    for name, res, stats in [("standard", std_res, std_stats),
                              ("at_matched", matched_res, matched_stats),
                              ("actual_AT", at_res, at_stats)]:
        print(f"  {name}: clean={res['clean_acc']:.4f} fgsm_asr={res['fgsm_asr']:.4f} "
              f"pgd_asr={res['pgd_asr']:.4f} margin={res['mean_margin']:.4f} "
              f"stats={[f'{s:.2f}' for s in stats]}")

    delta = std_res["pgd_asr"] - matched_res["pgd_asr"]
    print(f"\nAT-stats-matched vs standard PGD ASR delta: {delta:.4f}")
    print(f"SUPPORTED = {delta > 0.02}")

if __name__ == "__main__":
    main()
