"""
H216 - Which augmentation gives best robustness against random-direction attack?

Train 5 models with different augmentation strategies on Fashion-MNIST, then
attack with FGSM, Random-FGSM (L∞ unit random direction × eps), and PGD-10.

Key metric: ASR(random_FGSM) / ASR(FGSM) ratio — does augmentation diversity
specifically help against non-gradient attacks more than gradient attacks?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS     = "fashion_mnist"
N_EVAL = 300
SEED   = 0
EPS    = 0.1
EPOCHS = 10
BATCH  = 128


# ---------------------------------------------------------------------------
# augmentation helpers (operate on a batch tensor in [0,1])
# ---------------------------------------------------------------------------
def aug_hflip_crop(xb):
    """RandomHFlip + RandomCrop(28, padding=2)."""
    # horizontal flip with p=0.5
    flip_mask = torch.rand(xb.size(0)) < 0.5
    xb = xb.clone()
    xb[flip_mask] = xb[flip_mask].flip(-1)
    # random crop: pad by 2 then crop back to 28
    xb_pad = F.pad(xb, [2, 2, 2, 2], mode="reflect")   # (N,1,32,32)
    N, C, H, W = xb_pad.shape
    out = []
    for i in range(N):
        r = torch.randint(0, H - 28 + 1, (1,)).item()
        c = torch.randint(0, W - 28 + 1, (1,)).item()
        out.append(xb_pad[i:i+1, :, r:r+28, c:c+28])
    return torch.cat(out, dim=0)


def aug_cutout(xb, patch=8):
    """Zero out a random 8×8 patch per sample."""
    xb = xb.clone()
    N, C, H, W = xb.shape
    for i in range(N):
        r = torch.randint(0, H - patch + 1, (1,)).item()
        c = torch.randint(0, W - patch + 1, (1,)).item()
        xb[i, :, r:r+patch, c:c+patch] = 0.0
    return xb


def aug_brightness(xb, strength=0.2):
    """Random brightness jitter for grayscale: multiply by factor in [1-s, 1+s]."""
    factor = 1.0 + (torch.rand(xb.size(0), 1, 1, 1, device=xb.device) * 2 - 1) * strength
    return (xb * factor).clamp(0, 1)


def aug_gaussian_noise(xb, sigma=0.05):
    return (xb + torch.randn_like(xb) * sigma).clamp(0, 1)


def no_aug(xb):
    return xb


AUGMENTATIONS = {
    "baseline":      no_aug,
    "hflip_crop":    aug_hflip_crop,
    "cutout":        aug_cutout,
    "brightness":    aug_brightness,
    "gaussian_noise": aug_gaussian_noise,
}


# ---------------------------------------------------------------------------
# training with augmentation
# ---------------------------------------------------------------------------
def train_with_aug(model, Xtr, Ytr, aug_fn, epochs=EPOCHS, batch=BATCH):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_aug = aug_fn(xb)
            opt.zero_grad()
            F.cross_entropy(model(xb_aug), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Random-FGSM attack: x + eps * sign(rand_unit_direction)
# "unit L-inf" = sign of a standard-normal vector
# ---------------------------------------------------------------------------
def random_fgsm(model, X, Y, eps=EPS):
    """Random L∞ direction attack: x + eps * sign(randn)."""
    delta = torch.randn_like(X).sign() * eps
    Xa = (X + delta).clamp(0, 1)
    return Xa


def attack_asr(model, X, Y, attack_fn):
    """Fraction of originally-correct samples flipped after attack_fn."""
    with torch.no_grad():
        correct = model(X).argmax(1) == Y
    Xa = attack_fn(model, X, Y)
    with torch.no_grad():
        flipped = model(Xa).argmax(1) != Y
    corr = correct.cpu().numpy().astype(bool)
    flip = flipped.cpu().numpy()
    return float(flip[corr].mean()) if corr.sum() > 0 else float("nan")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("=== H216: Augmentation vs Random-Direction Attack ===")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  N_EVAL={N_EVAL}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_EVAL, seed=SEED)

    results = {}

    for aug_name, aug_fn in AUGMENTATIONS.items():
        print(f"\n--- Training: {aug_name} ---")
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32, seed=SEED)
        train_with_aug(model, Xtr, Ytr, aug_fn)

        _, clean_acc = C.logits_and_acc(model, Xte, Yte)

        asr_fgsm   = attack_asr(model, Xte, Yte, lambda m, x, y: C.fgsm(m, x, y, EPS))
        asr_rfgsm  = attack_asr(model, Xte, Yte, lambda m, x, y: random_fgsm(m, x, y, EPS))
        asr_pgd    = attack_asr(model, Xte, Yte,
                                lambda m, x, y: C.pgd(m, x, y, EPS, steps=10))

        ratio = asr_rfgsm / asr_fgsm if asr_fgsm > 0 else float("nan")

        results[aug_name] = {
            "clean_acc": clean_acc,
            "asr_fgsm":  asr_fgsm,
            "asr_rfgsm": asr_rfgsm,
            "asr_pgd":   asr_pgd,
            "ratio_rfgsm_fgsm": ratio,
        }

        print(f"  clean_acc={clean_acc:.3f}  FGSM={asr_fgsm:.3f}  "
              f"RandFGSM={asr_rfgsm:.3f}  PGD={asr_pgd:.3f}  "
              f"ratio={ratio:.3f}  ({time.time()-t0:.1f}s)")

    # --- Summary table ---
    print("\n" + "=" * 74)
    print("--- Summary ---")
    print(f"{'Augmentation':<18} {'CleanAcc':>9} {'FGSM_ASR':>9} {'Rand_ASR':>9} "
          f"{'PGD_ASR':>8} {'Rand/FGSM':>10}")
    for aug_name, r in results.items():
        print(f"{aug_name:<18} {r['clean_acc']:>9.3f} {r['asr_fgsm']:>9.3f} "
              f"{r['asr_rfgsm']:>9.3f} {r['asr_pgd']:>8.3f} "
              f"{r['ratio_rfgsm_fgsm']:>10.3f}")
    print("=" * 74)
    print("Interpretation: ratio < 1 means augmented model is proportionally MORE")
    print("robust to random attacks than to gradient attacks — augmentation diversity")
    print("may confuse random directions more than targeted gradient steps.")
    print("A ratio closer to 1.0 for augmented models vs baseline indicates that")
    print("augmentation does NOT specifically confer extra defence vs random attacks.")
    print("=" * 74)


if __name__ == "__main__":
    main()
