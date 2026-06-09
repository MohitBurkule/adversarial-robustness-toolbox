"""
H199 - CutMix augmentation provides robustness without adversarial training.

Hypothesis:
  CutMix-augmented standard training achieves higher FGSM/PGD robustness than
  Cutout, Mixup, or no augmentation, without adversarial training.
  Predicted FGSM accuracy at eps=4/255: CutMix ~33-40%, Mixup ~28-33%,
  Cutout ~24-28%, baseline ~20-25%.

Protocol:
  Train 4 small CNNs on Fashion-MNIST (n_train=6000, n_eval=500) for 15 epochs:
    - Baseline: no augmentation
    - Cutout: mask a random 14x14 square region per sample
    - Mixup: convex combination alpha=0.4 (Beta(0.4,0.4))
    - CutMix: paste random patch from another sample, mix labels by area ratio
  Evaluate: clean_acc, FGSM_acc (eps=4/255), PGD-7_acc (eps=4/255, step=0.004)
  Log robust acc at epochs [5, 10, 15] for robust overfitting check.
  Test rank order: CutMix > Mixup > Cutout > baseline for pgd7_acc.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign.common import (
    DEVICE, set_seed, load_dataset, dataset_meta,
    build_model, train_model, make_optimizer, logits_and_acc, fgsm, pgd,
)

# ── hyperparameters ──────────────────────────────────────────────────────────
DATASET    = "fashion_mnist"
N_TRAIN    = 6000
N_EVAL     = 500
SEED       = 42
EPOCHS     = 15
BATCH      = 128
EPS        = 4.0 / 255.0   # ~0.0157
PGD_STEPS  = 7
PGD_ALPHA  = 0.004
CUTOUT_SZ  = 14
MIXUP_ALPHA = 0.4
CHECKPOINT_EPOCHS = [5, 10, 15]


def apply_cutout(x, size=CUTOUT_SZ):
    """Zero-mask a random square region per sample."""
    b, c, h, w = x.shape
    x = x.clone()
    for i in range(b):
        cy = torch.randint(0, h, (1,)).item()
        cx = torch.randint(0, w, (1,)).item()
        y0 = max(0, cy - size // 2)
        y1 = min(h, cy + size // 2)
        x0 = max(0, cx - size // 2)
        x1 = min(w, cx + size // 2)
        x[i, :, y0:y1, x0:x1] = 0.0
    return x


def apply_mixup(x, y, ncls, alpha=MIXUP_ALPHA):
    """Mixup: convex combination of pairs."""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    return x_mix, y, y[perm], lam


def apply_cutmix(x, y, ncls):
    """CutMix: paste a random patch from another sample, mix labels by area ratio."""
    b, c, h, w = x.shape
    lam = np.random.beta(1.0, 1.0)  # area ratio
    perm = torch.randperm(b, device=x.device)

    # sample bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    cy = torch.randint(0, h, (1,)).item()
    cx = torch.randint(0, w, (1,)).item()
    y0 = max(0, cy - cut_h // 2)
    y1 = min(h, cy + cut_h // 2)
    x0 = max(0, cx - cut_w // 2)
    x1 = min(w, cx + cut_w // 2)

    x_mix = x.clone()
    x_mix[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
    # adjust lambda to actual area ratio
    lam_actual = 1.0 - (y1 - y0) * (x1 - x0) / (h * w)
    return x_mix, y, y[perm], lam_actual


def train_augmented(model, Xtr, Ytr, epochs, aug_type, ncls, checkpoints, Xte, Yte):
    """Train with augmentation, return model and checkpoint robust accuracies."""
    opt = make_optimizer(model, "sgd", 0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    checkpoint_results = {}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            if aug_type == "cutout":
                xb = apply_cutout(xb)
                out = model(xb)
                loss = F.cross_entropy(out, yb)
            elif aug_type == "mixup":
                xm, ya, yb2, lam = apply_mixup(xb, yb, ncls)
                out = model(xm)
                loss = lam * F.cross_entropy(out, ya) + (1 - lam) * F.cross_entropy(out, yb2)
            elif aug_type == "cutmix":
                xm, ya, yb2, lam = apply_cutmix(xb, yb, ncls)
                out = model(xm)
                loss = lam * F.cross_entropy(out, ya) + (1 - lam) * F.cross_entropy(out, yb2)
            else:  # baseline
                out = model(xb)
                loss = F.cross_entropy(out, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        if ep in checkpoints:
            model.eval()
            _, clean_acc = logits_and_acc(model, Xte, Yte)
            # FGSM robust acc
            xf = fgsm(model, Xte, Yte, EPS)
            with torch.no_grad():
                fgsm_acc = (model(xf).argmax(1) == Yte).float().mean().item()
            # PGD robust acc
            xp = pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            with torch.no_grad():
                pgd_acc = (model(xp).argmax(1) == Yte).float().mean().item()
            checkpoint_results[ep] = {
                "clean_acc": clean_acc, "fgsm_acc": fgsm_acc, "pgd7_acc": pgd_acc
            }
            print(f"    ep={ep}: clean={clean_acc:.4f} fgsm={fgsm_acc:.4f} pgd7={pgd_acc:.4f}")

    model.eval()
    return model, checkpoint_results


def main():
    t0 = time.time()
    set_seed(SEED)
    meta = dataset_meta(DATASET)
    ncls = meta["n_classes"]
    Xtr, Ytr, Xte, Yte = load_dataset(DATASET, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    aug_types = ["baseline", "cutout", "mixup", "cutmix"]
    all_results = {}

    for aug in aug_types:
        print(f"\n{'='*60}")
        print(f"Training: {aug}")
        set_seed(SEED)
        model = build_model("cnn", meta, width=32)
        model, ckpts = train_augmented(
            model, Xtr, Ytr, EPOCHS, aug, ncls, CHECKPOINT_EPOCHS, Xte, Yte
        )
        all_results[aug] = ckpts

    # ── output ───────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("H199 - Augmentation Robustness (CutMix vs Mixup vs Cutout)")
    lines.append("=" * 70)
    lines.append(f"Dataset: {DATASET}  n_train={N_TRAIN}  n_eval={N_EVAL}  seed={SEED}")
    lines.append(f"Epochs: {EPOCHS}  eps={EPS:.4f} (4/255)  PGD steps={PGD_STEPS}")
    lines.append("")

    # final results table
    lines.append(f"{'Augmentation':<12} {'clean_acc':>10} {'fgsm_acc':>10} {'pgd7_acc':>10}")
    lines.append("-" * 46)
    final_pgd = {}
    for aug in aug_types:
        r = all_results[aug][EPOCHS]
        final_pgd[aug] = r["pgd7_acc"]
        lines.append(f"{aug:<12} {r['clean_acc']:>10.4f} {r['fgsm_acc']:>10.4f} {r['pgd7_acc']:>10.4f}")

    # epoch breakdown
    lines.append("\nRobust accuracy over training (robust overfitting check):")
    for aug in aug_types:
        lines.append(f"\n  {aug}:")
        lines.append(f"    {'Epoch':>6} {'clean_acc':>10} {'fgsm_acc':>10} {'pgd7_acc':>10}")
        for ep in CHECKPOINT_EPOCHS:
            r = all_results[aug][ep]
            lines.append(f"    {ep:>6} {r['clean_acc']:>10.4f} {r['fgsm_acc']:>10.4f} {r['pgd7_acc']:>10.4f}")

    # rank order test
    rank = sorted(aug_types, key=lambda a: final_pgd[a], reverse=True)
    expected = ["cutmix", "mixup", "cutout", "baseline"]
    rank_match = rank == expected

    lines.append("")
    lines.append("=" * 70)
    lines.append("KEY TESTS")
    lines.append("=" * 70)
    lines.append(f"PGD-7 robust acc rank: {' > '.join(f'{a}({final_pgd[a]:.4f})' for a in rank)}")
    lines.append(f"Expected rank: {' > '.join(expected)}")
    lines.append(f"Rank order matches hypothesis: {rank_match}")
    lines.append(f"CutMix > baseline: {final_pgd['cutmix'] > final_pgd['baseline']}")
    lines.append(f"\nTotal time: {time.time()-t0:.1f}s")

    output = "\n".join(lines)
    print("\n" + output)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", DATASET), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", DATASET,
                            "h199_augmentation_robustness_output.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
