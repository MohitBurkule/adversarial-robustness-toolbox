"""
H203 - BatchNorm vs LayerNorm adversarial vulnerability (ASR comparison).

Hypothesis: Replacing BatchNorm with LayerNorm in a CNN reduces PGD-20 ASR
by >=8 percentage points with <1pp clean accuracy loss. Mechanism: BN exploits
batch statistics mismatch between train/test; adversarial perturbations shift
running statistics. LayerNorm normalises per-sample, eliminating this vector.

Grounded in: arXiv:2405.11708 (ABNN), arXiv:2410.06921, arXiv:1905.02161.

Architecture: Conv(1,32,3,pad=1)->Norm->ReLU->MaxPool ->
              Conv(32,64,3,pad=1)->Norm->ReLU->MaxPool ->
              Flatten->FC(64*7*7,256)->ReLU->FC(256,10)

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign.common import (
    DEVICE, set_seed, load_dataset, dataset_meta,
    train_model, fgsm, pgd, attack_success, logits_and_acc,
)

SEED = 42
N_TRAIN = 6000
N_EVAL = 500
EPOCHS = 20
LR = 0.05
OPT = "sgd"
BATCH = 128


# ---- models -----------------------------------------------------------------

class NormCNN(nn.Module):
    """3-conv CNN parameterised by normalisation type: 'bn' or 'ln'."""

    def __init__(self, norm_type="bn", in_ch=1, size=28, n_classes=10):
        super().__init__()
        self.norm_type = norm_type

        def make_norm(channels, h, w):
            if norm_type == "bn":
                return nn.BatchNorm2d(channels)
            else:
                # GroupNorm(1, C) == LayerNorm over C,H,W
                return nn.GroupNorm(1, channels)

        # Block 1: in_ch -> 32, spatial /2
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.norm1 = make_norm(32, size, size)
        # Block 2: 32 -> 64, spatial /2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm2 = make_norm(64, size // 2, size // 2)
        # Head
        feat_dim = 64 * (size // 4) * (size // 4)  # after two MaxPool2d(2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.norm1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.norm2(self.conv2(x))), 2)
        return self.head(x)


# ---- BN running-stats shift under attack ------------------------------------

def measure_bn_shift(model, X_clean, Y, eps=0.1, steps=20, batch=256):
    """For each BN layer, compute ||mean_clean - mean_adv|| over batches."""
    model.eval()
    bn_layers = [(name, m) for name, m in model.named_modules()
                 if isinstance(m, nn.BatchNorm2d)]
    if not bn_layers:
        return {}

    # Collect batch means under clean and adversarial inputs
    clean_means = {name: [] for name, _ in bn_layers}
    adv_means = {name: [] for name, _ in bn_layers}

    hooks = []
    current_store = clean_means

    def make_hook(name):
        def hook_fn(module, inp, out):
            # inp[0] is the input to BN; compute batch mean over N,H,W
            current_store[name].append(inp[0].detach().mean(dim=(0, 2, 3)).cpu())
        return hook_fn

    for name, m in bn_layers:
        hooks.append(m.register_forward_hook(make_hook(name)))

    # Clean forward
    model.train()  # so BN computes batch stats (not running stats)
    with torch.no_grad():
        for i in range(0, min(X_clean.size(0), 1000), batch):
            model(X_clean[i:i + batch])

    # Adversarial forward
    current_store = adv_means
    model.eval()
    for i in range(0, min(X_clean.size(0), 1000), batch):
        xb = X_clean[i:i + batch]
        yb = Y[i:i + batch]
        xa = pgd(model, xb, yb, eps=eps, steps=steps)
        model.train()
        with torch.no_grad():
            model(xa)
        model.eval()

    for h in hooks:
        h.remove()

    shifts = {}
    for name, _ in bn_layers:
        cm = torch.stack(clean_means[name]).mean(0)
        am = torch.stack(adv_means[name]).mean(0)
        shifts[name] = float(torch.norm(cm - am).item())

    model.eval()
    return shifts


# ---- main -------------------------------------------------------------------

def main():
    t0 = time.time()
    set_seed(SEED)
    meta = dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    results = []

    for norm_type in ["bn", "ln"]:
        set_seed(SEED)
        model = NormCNN(norm_type=norm_type, in_ch=meta["channels"],
                        size=meta["size"], n_classes=meta["n_classes"]).to(DEVICE)

        print(f"\n--- Training {norm_type.upper()} model ---")
        train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                    opt=OPT, lr=LR, ncls=meta["n_classes"], verbose=True)

        # Clean accuracy
        _, clean_acc = logits_and_acc(model, Xte, Yte)

        # FGSM ASR (eps=0.1)
        fgsm_res = attack_success(model, Xte, Yte, attack="fgsm", eps=0.1)

        # PGD-20 ASR (eps=0.1)
        pgd01_res = attack_success(model, Xte, Yte, attack="pgd", eps=0.1, steps=20)

        # PGD-20 ASR (eps=0.3)
        pgd03_res = attack_success(model, Xte, Yte, attack="pgd", eps=0.3, steps=20)

        # BN stats shift (only for BN model)
        bn_shifts = {}
        if norm_type == "bn":
            bn_shifts = measure_bn_shift(model, Xte, Yte, eps=0.1, steps=20)

        row = {
            "norm_type": norm_type.upper(),
            "clean_acc": clean_acc,
            "fgsm_asr": fgsm_res["asr"],
            "pgd20_asr_01": pgd01_res["asr"],
            "pgd20_asr_03": pgd03_res["asr"],
            "bn_shifts": bn_shifts,
        }
        results.append(row)
        print(f"  {norm_type.upper()}: clean={clean_acc:.4f}  fgsm={fgsm_res['asr']:.4f}  "
              f"pgd20@0.1={pgd01_res['asr']:.4f}  pgd20@0.3={pgd03_res['asr']:.4f}")

    # ---- report -------------------------------------------------------------
    elapsed = time.time() - t0
    bn_row = results[0]
    ln_row = results[1]
    delta_pgd20 = bn_row["pgd20_asr_01"] - ln_row["pgd20_asr_01"]
    delta_clean = abs(bn_row["clean_acc"] - ln_row["clean_acc"])
    hypothesis_met = delta_pgd20 >= 0.08 and delta_clean < 0.01

    lines = []
    lines.append("=" * 72)
    lines.append("H203 - BatchNorm vs LayerNorm Adversarial Vulnerability")
    lines.append("=" * 72)
    lines.append(f"Dataset: Fashion-MNIST  n_train={N_TRAIN}  n_eval={N_EVAL}  seed={SEED}")
    lines.append(f"Architecture: 2-conv NormCNN  epochs={EPOCHS}  lr={LR}  opt={OPT}")
    lines.append("")
    lines.append(f"{'Norm':<6} {'Clean':>8} {'FGSM':>8} {'PGD20@0.1':>10} {'PGD20@0.3':>10}")
    lines.append("-" * 46)
    for r in results:
        lines.append(f"{r['norm_type']:<6} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>8.4f} "
                      f"{r['pgd20_asr_01']:>10.4f} {r['pgd20_asr_03']:>10.4f}")
    lines.append("")
    lines.append(f"Delta PGD20@0.1 ASR (BN - LN): {delta_pgd20:+.4f}  ({delta_pgd20*100:+.1f}pp)")
    lines.append(f"Delta clean accuracy:           {delta_clean:.4f}  ({delta_clean*100:.1f}pp)")
    lines.append(f"Hypothesis (>=8pp ASR drop, <1pp clean loss): {'SUPPORTED' if hypothesis_met else 'NOT SUPPORTED'}")
    lines.append("")

    # BN running-stats shift
    if bn_row["bn_shifts"]:
        lines.append("BN running-stats shift under PGD attack (||mu_clean - mu_adv||):")
        for name, shift in bn_row["bn_shifts"].items():
            lines.append(f"  {name}: {shift:.4f}")
        lines.append("")

    lines.append(f"Elapsed: {elapsed:.1f}s")
    report = "\n".join(lines)
    print(report)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "h203_batchnorm_vs_layernorm_asr_output.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
