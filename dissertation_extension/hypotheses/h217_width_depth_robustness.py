"""
H217 - Model width vs depth: which dimension improves adversarial robustness more?

Build 6 architectures with ~100K parameters each. Train each with PGD-AT.
Compare: clean accuracy, FGSM ASR, PGD ASR, AUROC(-margin → PGD success).

Key question: at the same parameter budget, does width beat depth for robustness?
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
# Architecture builders (all return nn.Sequential-based nn.Module on DEVICE)
# ---------------------------------------------------------------------------
def _conv_block(in_ch, out_ch, bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return layers


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# All models expect input (N,1,28,28)

class WideShallow(nn.Module):
    """2 conv layers, 64 filters, 1 maxpool each."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(1, 64), nn.MaxPool2d(2),
            *_conv_block(64, 64), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 10))

    def forward(self, x):
        return self.head(self.features(x))


class WideMedium(nn.Module):
    """3 conv layers, 48 filters."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(1, 48), nn.MaxPool2d(2),
            *_conv_block(48, 48), nn.MaxPool2d(2),
            *_conv_block(48, 48),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48 * 7 * 7, 10))

    def forward(self, x):
        return self.head(self.features(x))


class NarrowDeep(nn.Module):
    """5 conv layers, 16 filters."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(1, 16), nn.MaxPool2d(2),
            *_conv_block(16, 16),
            *_conv_block(16, 16), nn.MaxPool2d(2),
            *_conv_block(16, 16),
            *_conv_block(16, 16),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 10))

    def forward(self, x):
        return self.head(self.features(x))


class Standard(nn.Module):
    """3 conv layers, 32 filters (baseline ~SmallCNN)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(1, 32), nn.MaxPool2d(2),
            *_conv_block(32, 32), nn.MaxPool2d(2),
            *_conv_block(32, 32),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

    def forward(self, x):
        return self.head(self.features(x))


class VeryWide(nn.Module):
    """2 conv layers, 96 filters, 1 FC hidden."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(1, 96), nn.MaxPool2d(2),
            *_conv_block(96, 96), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.head(self.features(x))


class VeryDeep(nn.Module):
    """6 conv layers, 16 filters."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(1, 16), nn.MaxPool2d(2),
            *_conv_block(16, 16),
            *_conv_block(16, 16), nn.MaxPool2d(2),
            *_conv_block(16, 16),
            *_conv_block(16, 16),
            *_conv_block(16, 16),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 10))

    def forward(self, x):
        return self.head(self.features(x))


ARCHITECTURES = {
    "wide_shallow": WideShallow,
    "wide_medium":  WideMedium,
    "narrow_deep":  NarrowDeep,
    "standard":     Standard,
    "very_wide":    VeryWide,
    "very_deep":    VeryDeep,
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("=== H217: Model Width vs Depth — adversarial robustness ===")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  N_EVAL={N_EVAL}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_EVAL, seed=SEED)

    print("\n--- 1. Architecture parameter counts ---")
    for arch_name, arch_cls in ARCHITECTURES.items():
        m = arch_cls()
        print(f"  {arch_name:<14} params={count_params(m):,}")

    results = {}

    for arch_name, arch_cls in ARCHITECTURES.items():
        print(f"\n--- 2. Training (PGD-AT): {arch_name} ---")
        t0 = time.time()
        C.set_seed(SEED)
        model = arch_cls().to(C.DEVICE)

        C.train_model(model, Xtr, Ytr, epochs=EPOCHS, opt="sgd", lr=0.05,
                      ncls=meta["n_classes"], adv_train=True, adv_eps=EPS, adv_steps=7)

        _, clean_acc = C.logits_and_acc(model, Xte, Yte)

        fgsm_res = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
        pgd_res  = C.attack_success(model, Xte, Yte, attack="pgd",  eps=EPS, steps=10)

        mgn = C.margin(model, Xte, Yte)

        # AUROC(-margin → PGD success), over originally-correct samples
        corr_mask = pgd_res["correct"].astype(bool)
        auroc = C.safe_auroc(pgd_res["flips"][corr_mask], -mgn[corr_mask])

        results[arch_name] = {
            "params":    count_params(model),
            "clean_acc": clean_acc,
            "fgsm_asr":  fgsm_res["asr"],
            "pgd_asr":   pgd_res["asr"],
            "margin_mean": float(mgn.mean()),
            "auroc":     auroc,
        }

        print(f"  params={count_params(model):,}  clean={clean_acc:.3f}  "
              f"FGSM={fgsm_res['asr']:.3f}  PGD={pgd_res['asr']:.3f}  "
              f"mgn={mgn.mean():.3f}  AUROC={auroc:.3f}  ({time.time()-t0:.1f}s)")

    # --- Summary ---
    print("\n" + "=" * 74)
    print("--- Summary ---")
    print(f"{'Architecture':<14} {'Params':>8} {'CleanAcc':>9} {'FGSM_ASR':>9} "
          f"{'PGD_ASR':>8} {'MgnMean':>8} {'AUROC':>7}")
    for arch_name, r in results.items():
        print(f"{arch_name:<14} {r['params']:>8,} {r['clean_acc']:>9.3f} "
              f"{r['fgsm_asr']:>9.3f} {r['pgd_asr']:>8.3f} "
              f"{r['margin_mean']:>8.3f} {r['auroc']:>7.3f}")
    print("=" * 74)
    print("Interpretation: lower PGD ASR / higher margin under AT at similar param")
    print("count indicates better robustness. If wide architectures consistently")
    print("outperform deep ones, width is the more efficient axis for robustness.")
    print("=" * 74)


if __name__ == "__main__":
    main()
