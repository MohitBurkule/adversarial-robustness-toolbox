"""
H247 - Jointly Robust Samples: samples robust to ALL attacks simultaneously.

Find samples robust to FGSM, PGD-10, and C&W (approximated as PGD L2).
jointly_robust = set where all three attacks fail.
Measure: (1) size of jointly_robust set,
         (2) do they cluster in feature space (mean pairwise distance),
         (3) are they the same across 3 seeds? Cross-seed Jaccard overlap.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0
EPS = 0.1
N_EVAL = 300
EPOCHS = 10
SEEDS = [0, 1, 2]
CW_STEPS = 20  # more steps for C&W proxy

def pgd_l2(model, X, Y, eps=1.0, steps=20, alpha=0.1):
    """PGD attack under L2 norm (C&W proxy)."""
    model.eval()
    X_orig = X.clone().detach()
    delta = torch.zeros_like(X)
    delta.requires_grad_(True)
    for _ in range(steps):
        logits = model(torch.clamp(X_orig + delta, 0, 1))
        loss = F.cross_entropy(logits, Y)
        loss.backward()
        with torch.no_grad():
            grad = delta.grad.clone()
            # L2 normalised step
            grad_flat = grad.flatten(1)
            grad_norm = grad_flat.norm(dim=1).clamp(min=1e-12)
            grad_normalised = (grad / grad_norm.view(-1, 1, 1, 1))
            delta = delta + alpha * grad_normalised
            # L2 projection
            delta_flat = delta.flatten(1)
            delta_norm = delta_flat.norm(dim=1)
            scale = (eps / delta_norm.clamp(min=1e-12)).clamp(max=1.0)
            delta = delta * scale.view(-1, 1, 1, 1)
        delta = delta.detach().requires_grad_(True)
    return torch.clamp(X_orig + delta.detach(), 0, 1)

def attack_success(model, X, Y, attack='fgsm'):
    model.eval()
    if attack == 'fgsm':
        Xadv = C.fgsm(model, X, Y, eps=EPS)
    elif attack == 'pgd':
        Xadv = C.pgd(model, X, Y, eps=EPS, steps=10, alpha=0.01)
    elif attack == 'cw_l2':
        Xadv = pgd_l2(model, X, Y, eps=1.0, steps=CW_STEPS, alpha=0.05)
    with torch.no_grad():
        preds = model(Xadv).argmax(1).cpu()
    return (preds != Y.cpu()).numpy().astype(bool)

def get_features(model, X, batch=256):
    """Extract penultimate layer features."""
    model.eval()
    layers = list(model.children())
    last_linear_idx = None
    for i, layer in enumerate(layers):
        if isinstance(layer, torch.nn.Linear):
            last_linear_idx = i

    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i+batch]
            if last_linear_idx is not None:
                x = xb
                for layer in layers[:last_linear_idx]:
                    x = layer(x)
                if x.dim() > 2:
                    x = x.flatten(1)
            else:
                x = model(xb)
            feats.append(x.cpu())
    return torch.cat(feats)

def mean_pairwise_l2(features):
    """Mean pairwise L2 distance for a set of feature vectors."""
    if len(features) < 2:
        return float('nan')
    # Use torch.cdist on CPU
    feats = features.float()
    n = len(feats)
    if n > 200:
        feats = feats[:200]
    dists = torch.cdist(feats, feats)
    # Upper triangle only
    mask = torch.triu(torch.ones(len(feats), len(feats)), diagonal=1).bool()
    return dists[mask].mean().item()

def main():
    print("=== H247: Jointly Robust Samples ===")
    t0 = time.time()

    meta = C.dataset_meta("fashion_mnist")
    Xtr_all, Ytr_all, Xte_all, Yte_all = C.load_dataset("fashion_mnist")
    Xte_e = Xte_all[:N_EVAL]
    Yte_e = Yte_all[:N_EVAL]

    jointly_robust_per_seed = {}
    models = {}

    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        C.set_seed(seed)
        model = C.build_model("cnn", meta, seed=seed)
        C.train_model(model, Xtr_all, Ytr_all, epochs=EPOCHS)
        models[seed] = model

        fgsm_succ = attack_success(model, Xte_e, Yte_e, 'fgsm')
        pgd_succ = attack_success(model, Xte_e, Yte_e, 'pgd')
        cw_succ = attack_success(model, Xte_e, Yte_e, 'cw_l2')

        jointly_robust = ~fgsm_succ & ~pgd_succ & ~cw_succ
        jointly_robust_per_seed[seed] = jointly_robust

        print(f"  FGSM ASR: {fgsm_succ.mean():.3f}, "
              f"PGD ASR: {pgd_succ.mean():.3f}, "
              f"CW(L2) ASR: {cw_succ.mean():.3f}")
        print(f"  Jointly robust: {jointly_robust.sum()}/{N_EVAL} "
              f"({100*jointly_robust.mean():.1f}%)")

    # [Feature clustering analysis for seed 0]
    print("\n[Feature clustering of jointly robust set (seed 0)]...")
    seed0_jr = jointly_robust_per_seed[0]
    features_all = get_features(models[0], Xte_e)

    if seed0_jr.sum() >= 2:
        jr_features = features_all[torch.from_numpy(seed0_jr)]
        mean_dist_jr = mean_pairwise_l2(jr_features)
    else:
        mean_dist_jr = float('nan')

    # Random subset of same size
    rng = np.random.default_rng(SEED)
    n_jr = seed0_jr.sum()
    if n_jr >= 2:
        rand_idx = rng.choice(N_EVAL, n_jr, replace=False)
        rand_features = features_all[rand_idx]
        mean_dist_rand = mean_pairwise_l2(rand_features)
    else:
        mean_dist_rand = float('nan')

    print(f"  Mean pairwise L2 (jointly robust): {mean_dist_jr:.4f}")
    print(f"  Mean pairwise L2 (random subset):  {mean_dist_rand:.4f}")

    # [Cross-seed Jaccard overlap]
    print("\n[Cross-seed Jaccard overlaps]...")
    for s1, s2 in [(0,1), (0,2), (1,2)]:
        a = set(np.where(jointly_robust_per_seed[s1])[0])
        b = set(np.where(jointly_robust_per_seed[s2])[0])
        j = len(a & b) / len(a | b) if len(a | b) > 0 else float('nan')
        print(f"  Seed {s1} vs Seed {s2}: Jaccard={j:.3f} "
              f"(|A|={len(a)}, |B|={len(b)}, |A∩B|={len(a&b)})")

    print(f"\n--- Summary ---")
    for seed in SEEDS:
        jr = jointly_robust_per_seed[seed]
        print(f"  Seed {seed}: jointly robust = {jr.sum()}/{N_EVAL} "
              f"({100*jr.mean():.1f}%)")
    print(f"Feature clustering: jr_dist={mean_dist_jr:.4f} vs "
          f"random_dist={mean_dist_rand:.4f}")
    print("Interpretation: lower pairwise dist for jr set => clustering in "
          "feature space; high cross-seed Jaccard => robust set is consistent.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
