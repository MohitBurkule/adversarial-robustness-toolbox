"""
H197 - Architecture match vs data overlap in adversarial transferability.

Hypothesis:
  Architecture similarity is a stronger driver of adversarial transfer ASR
  than training data overlap. Same-arch transfer (CNN→CNN) yields higher ASR
  than cross-arch (CNN→MLP) even when data overlaps completely, and reducing
  data overlap (80% subset) causes a smaller ASR drop than switching architecture.

Protocol:
  Build 4 models on Fashion-MNIST (n_train=6000):
    M1: SmallCNN, source, natural, full 6k data, seed=42
    M2: SmallCNN, target, same arch, same 6k data, seed=99
    M3: SmallCNN, target, same arch, 80% data subset (random 4800), seed=99
    M4: MLP 784→256→10, target, different arch, same 6k data, seed=42

  All trained 15 epochs to >85% clean accuracy.

  Generate PGD-L∞ adversarial examples on M1 (eps=0.1, steps=20, step=0.01)
  for 500 test samples.

  Evaluate transfer ASR on M2, M3, M4 (fraction of M1-adversarial examples
  that fool each target, restricted to samples M1 classifies correctly).

  Compute linear CKA between M1 and each target on penultimate-layer
  activations for the 500 clean test samples.

  Test: transfer_asr(M2) > transfer_asr(M3) > transfer_asr(M4)
        AND CKA rank order consistent with ASR rank order.

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
    build_model, train_model, logits_and_acc, pgd,
)

# ── hyperparameters ──────────────────────────────────────────────────────────
DATASET    = "fashion_mnist"
N_TRAIN    = 6000
N_EVAL     = 1000
SEED       = 42
EPOCHS     = 15
N_ATTACK   = 500    # samples for transfer attack
PGD_EPS    = 0.1    # L-inf epsilon
PGD_STEPS  = 20
PGD_ALPHA  = 0.01
DATA_OVERLAP_FRAC = 0.8  # M3 trains on 80% of M1's data


# ── custom MLP (2-layer, 784→256→10) ────────────────────────────────────────
class SmallMLP(nn.Module):
    def __init__(self, in_ch=1, size=28, n_classes=10):
        super().__init__()
        d = in_ch * size * size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── penultimate-layer feature extractors ─────────────────────────────────────
@torch.no_grad()
def get_penultimate_cnn(model, X, batch=256):
    """Extract penultimate features (before final linear) from SmallCNN."""
    feats = []
    for i in range(0, X.size(0), batch):
        xb = X[i:i+batch]
        h = model.features(xb)
        h = model.head[:-1](h)  # Flatten + Linear(256) + ReLU, skip last Linear
        feats.append(h.cpu())
    return torch.cat(feats)


@torch.no_grad()
def get_penultimate_mlp(model, X, batch=256):
    """Extract penultimate features from SmallMLP (after first ReLU)."""
    feats = []
    for i in range(0, X.size(0), batch):
        xb = X[i:i+batch]
        # net = [Flatten, Linear(784,256), ReLU, Linear(256,10)]
        h = xb
        for layer in list(model.net.children())[:-1]:
            h = layer(h)
        feats.append(h.cpu())
    return torch.cat(feats)


# ── linear CKA ──────────────────────────────────────────────────────────────
def linear_cka(X, Y):
    """Linear CKA between two feature matrices (N x d1) and (N x d2).
    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    cross = torch.norm(Y.T @ X, p='fro') ** 2
    xx = torch.norm(X.T @ X, p='fro')
    yy = torch.norm(Y.T @ Y, p='fro')
    return (cross / (xx * yy + 1e-12)).item()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    meta = dataset_meta(DATASET)

    # load full data
    set_seed(SEED)
    Xtr, Ytr, Xte, Yte = load_dataset(DATASET, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # 80% subset for M3
    set_seed(99)
    n_sub = int(N_TRAIN * DATA_OVERLAP_FRAC)
    sub_idx = torch.randperm(Xtr.size(0))[:n_sub]
    Xtr_sub, Ytr_sub = Xtr[sub_idx], Ytr[sub_idx]
    data_overlap_actual = n_sub / N_TRAIN

    # attack / CKA samples
    Xa, Ya = Xte[:N_ATTACK], Yte[:N_ATTACK]

    # ── train models ─────────────────────────────────────────────────────────
    models = {}

    # M1: SmallCNN, source, seed=42
    print("Training M1 (SmallCNN, source, full data, seed=42) ...")
    set_seed(42)
    m1 = build_model("cnn", meta, width=32)
    train_model(m1, Xtr, Ytr, epochs=EPOCHS, opt="sgd", lr=0.05)
    _, acc1 = logits_and_acc(m1, Xte, Yte)
    models["M1_source"] = {"model": m1, "arch": "CNN", "data": "full", "acc": acc1}
    print(f"  M1 clean acc: {acc1:.4f}")

    # M2: SmallCNN, same arch, same data, seed=99
    print("Training M2 (SmallCNN, same arch, full data, seed=99) ...")
    set_seed(99)
    m2 = build_model("cnn", meta, width=32)
    train_model(m2, Xtr, Ytr, epochs=EPOCHS, opt="sgd", lr=0.05)
    _, acc2 = logits_and_acc(m2, Xte, Yte)
    models["M2_same_arch_same_data"] = {"model": m2, "arch": "CNN", "data": "full", "acc": acc2}
    print(f"  M2 clean acc: {acc2:.4f}")

    # M3: SmallCNN, same arch, 80% data, seed=99
    print("Training M3 (SmallCNN, same arch, 80% data, seed=99) ...")
    set_seed(99)
    m3 = build_model("cnn", meta, width=32)
    train_model(m3, Xtr_sub, Ytr_sub, epochs=EPOCHS, opt="sgd", lr=0.05)
    _, acc3 = logits_and_acc(m3, Xte, Yte)
    models["M3_same_arch_less_data"] = {"model": m3, "arch": "CNN", "data": "80%", "acc": acc3}
    print(f"  M3 clean acc: {acc3:.4f}")

    # M4: SmallMLP, different arch, same data, seed=42
    print("Training M4 (SmallMLP, diff arch, full data, seed=42) ...")
    set_seed(42)
    m4 = SmallMLP(in_ch=meta["channels"], size=meta["size"],
                  n_classes=meta["n_classes"]).to(DEVICE)
    train_model(m4, Xtr, Ytr, epochs=EPOCHS, opt="adam", lr=0.001)
    _, acc4 = logits_and_acc(m4, Xte, Yte)
    models["M4_diff_arch_same_data"] = {"model": m4, "arch": "MLP", "data": "full", "acc": acc4}
    print(f"  M4 clean acc: {acc4:.4f}")

    # ── generate adversarial examples on M1 ──────────────────────────────────
    print(f"\nGenerating PGD-Linf adversarial examples on M1 (eps={PGD_EPS}, "
          f"steps={PGD_STEPS}) for {N_ATTACK} samples ...")
    m1.eval()

    # find correctly classified samples
    with torch.no_grad():
        clean_preds = []
        for i in range(0, Xa.size(0), 256):
            clean_preds.append(m1(Xa[i:i+256]).argmax(1))
        clean_preds = torch.cat(clean_preds)
    correct_mask = (clean_preds == Ya)
    print(f"  M1 correct on attack set: {correct_mask.sum().item()}/{N_ATTACK}")

    # generate adversarial examples for all (evaluate transfer on correct subset)
    X_adv = pgd(m1, Xa, Ya, eps=PGD_EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)

    # ── evaluate transfer ASR ────────────────────────────────────────────────
    print("\nTransfer ASR (fraction of M1-correct samples fooled):")
    transfer_results = {}

    for name, info in models.items():
        if name == "M1_source":
            continue
        model_t = info["model"]
        model_t.eval()
        with torch.no_grad():
            adv_preds = []
            for i in range(0, X_adv.size(0), 256):
                adv_preds.append(model_t(X_adv[i:i+256]).argmax(1))
            adv_preds = torch.cat(adv_preds)
        # ASR = fraction of M1-correct samples where target is fooled
        fooled = (adv_preds[correct_mask] != Ya[correct_mask])
        asr = fooled.float().mean().item()
        transfer_results[name] = asr
        print(f"  {name}: ASR = {asr:.4f}")

    # ── compute CKA ──────────────────────────────────────────────────────────
    print("\nComputing linear CKA (penultimate layer, clean samples) ...")
    feat_m1 = get_penultimate_cnn(m1, Xa)

    cka_results = {}
    for name, info in models.items():
        if name == "M1_source":
            continue
        if info["arch"] == "CNN":
            feat_t = get_penultimate_cnn(info["model"], Xa)
        else:
            feat_t = get_penultimate_mlp(info["model"], Xa)
        cka = linear_cka(feat_m1, feat_t)
        cka_results[name] = cka
        print(f"  CKA(M1, {name}): {cka:.4f}")

    # ── test hypothesis ──────────────────────────────────────────────────────
    asr_m2 = transfer_results["M2_same_arch_same_data"]
    asr_m3 = transfer_results["M3_same_arch_less_data"]
    asr_m4 = transfer_results["M4_diff_arch_same_data"]

    cka_m2 = cka_results["M2_same_arch_same_data"]
    cka_m3 = cka_results["M3_same_arch_less_data"]
    cka_m4 = cka_results["M4_diff_arch_same_data"]

    asr_order_ok = (asr_m2 > asr_m3 > asr_m4)
    # architecture effect = drop from M2 to M4 (same data, diff arch)
    arch_effect = asr_m2 - asr_m4
    # data effect = drop from M2 to M3 (same arch, less data)
    data_effect = asr_m2 - asr_m3
    arch_stronger = arch_effect > data_effect

    # CKA rank consistency
    cka_order = (cka_m2 >= cka_m3 >= cka_m4)

    elapsed = time.time() - t0

    # ── print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("H197 - Architecture Match vs Data Overlap in Adversarial Transferability")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  dataset={DATASET}, n_train={N_TRAIN}, n_eval={N_EVAL}, seed={SEED}")
    print(f"  PGD-Linf: eps={PGD_EPS}, steps={PGD_STEPS}, alpha={PGD_ALPHA}")
    print(f"  data_overlap_M3={DATA_OVERLAP_FRAC}, n_attack={N_ATTACK}")

    print(f"\n{'Target':<30} {'Arch':>5} {'Data':>6} {'CleanAcc':>9} {'TransASR':>9} {'CKA':>8}")
    print("-" * 70)
    for name in ["M2_same_arch_same_data", "M3_same_arch_less_data", "M4_diff_arch_same_data"]:
        info = models[name]
        print(f"{name:<30} {info['arch']:>5} {info['data']:>6} {info['acc']:>9.4f} "
              f"{transfer_results[name]:>9.4f} {cka_results[name]:>8.4f}")

    print(f"\nArchitecture effect (M2-M4 ASR drop): {arch_effect:.4f}")
    print(f"Data overlap effect (M2-M3 ASR drop):  {data_effect:.4f}")
    print(f"Architecture stronger than data: {'YES' if arch_stronger else 'NO'}")
    print(f"ASR order M2>M3>M4: {'YES' if asr_order_ok else 'NO'}")
    print(f"CKA rank consistent with ASR: {'YES' if cka_order else 'NO'}")

    supported = arch_stronger and asr_order_ok
    print(f"\nHypothesis: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")
    print(f"Elapsed: {elapsed:.1f}s")

    # ── save results ─────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h197_transfer_architecture_vs_data_output.txt")
    with open(out_path, "w") as f:
        f.write("H197 - Architecture Match vs Data Overlap in Adversarial Transferability\n")
        f.write("=" * 70 + "\n")
        f.write(f"dataset={DATASET}, n_train={N_TRAIN}, n_eval={N_EVAL}, seed={SEED}\n")
        f.write(f"PGD-Linf: eps={PGD_EPS}, steps={PGD_STEPS}, alpha={PGD_ALPHA}\n")
        f.write(f"data_overlap_M3={DATA_OVERLAP_FRAC}, n_attack={N_ATTACK}\n\n")
        f.write(f"{'Target':<30} {'Arch':>5} {'Data':>6} {'CleanAcc':>9} {'TransASR':>9} {'CKA':>8}\n")
        f.write("-" * 70 + "\n")
        for name in ["M2_same_arch_same_data", "M3_same_arch_less_data", "M4_diff_arch_same_data"]:
            info = models[name]
            f.write(f"{name:<30} {info['arch']:>5} {info['data']:>6} {info['acc']:>9.4f} "
                    f"{transfer_results[name]:>9.4f} {cka_results[name]:>8.4f}\n")
        f.write(f"\nArchitecture effect (M2-M4): {arch_effect:.4f}\n")
        f.write(f"Data overlap effect (M2-M3): {data_effect:.4f}\n")
        f.write(f"Arch stronger: {'YES' if arch_stronger else 'NO'}\n")
        f.write(f"ASR order M2>M3>M4: {'YES' if asr_order_ok else 'NO'}\n")
        f.write(f"CKA rank consistent: {'YES' if cka_order else 'NO'}\n")
        f.write(f"\nHypothesis: {'SUPPORTED' if supported else 'NOT SUPPORTED'}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
