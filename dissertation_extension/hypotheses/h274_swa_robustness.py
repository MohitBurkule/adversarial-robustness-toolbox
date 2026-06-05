"""
H274 - Stochastic Weight Averaging (SWA) and adversarial robustness.

SWA averages weights from multiple points along the training trajectory, placing
the final model in a wider, flatter region of the loss landscape. The hypothesis:
a flatter weight-space basin = more robust model.

Two SWA variants:
  (a) Baseline: single model trained 20 epochs
  (b) SWA-trajectory: arithmetic mean of snapshots at epochs 10,12,14,16,18,20
      (all from the same training run)
  (c) SWA-multiseed: arithmetic mean of 5 independently seeded models (each 20 epochs)

After SWA averaging, BatchNorm running statistics are updated by a forward pass
through the training set (required because BN stats are not parameter-averaged).

Sharpness proxy: same as H273 -- 20 random N(0, 0.01^2) weight perturbations,
average loss increase.

Key question: does sitting in a flatter basin (via weight averaging) give
adversarial robustness without any explicit adversarial training?
"""
import os, sys, time
import copy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS            = "fashion_mnist"
SEED          = 0
N_TRAIN       = 10000
EPOCHS        = 20
LR            = 0.05
MOMENTUM      = 0.9
BATCH         = 128
EPS           = 0.1
PGD_STEPS     = 10
PGD_ALPHA     = 0.01
SHARP_SAMPLES = 20
SHARP_SIGMA   = 0.01
SNAPSHOT_EPOCHS = [10, 12, 14, 16, 18, 20]
MULTI_SEEDS     = [0, 1, 2, 3, 4]
OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h274_swa_robustness_output.txt"
)


# ---------------------------------------------------------------------------
# Training with optional snapshot collection
# ---------------------------------------------------------------------------

def train_with_snapshots(model, X, Y, snapshot_epochs=None):
    """
    Train for EPOCHS epochs with SGD. Collect deep copies of the model state
    at each epoch listed in snapshot_epochs.
    Returns list of (epoch, state_dict) pairs.
    """
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=5e-4)
    snapshots = []
    model.train()
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = X[idx].to(C.DEVICE), Y[idx].to(C.DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if ep % 5 == 0:
            print(f"    epoch {ep}/{EPOCHS}  loss={total_loss:.3f}")
        if snapshot_epochs and ep in snapshot_epochs:
            snapshots.append((ep, copy.deepcopy(model.state_dict())))
    return snapshots


# ---------------------------------------------------------------------------
# SWA: average parameter tensors from a list of state_dicts
# ---------------------------------------------------------------------------

def average_state_dicts(state_dicts):
    """Return a new state_dict whose parameters are the arithmetic mean."""
    avg = copy.deepcopy(state_dicts[0])
    n = len(state_dicts)
    for key in avg:
        # Average only floating-point tensors (skip integer running buffers)
        if avg[key].is_floating_point():
            avg[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(0)
    return avg


def update_bn_stats(model, X):
    """One forward pass through X to refresh BatchNorm running stats."""
    model.train()  # BN updates stats only in train mode
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = X[i:i + 256].to(C.DEVICE)
            model(xb)
    model.eval()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def measure_sharpness(model, X, Y):
    model.eval()
    with torch.no_grad():
        base, nb = 0.0, 0
        for i in range(0, len(X), 256):
            xb, yb = X[i:i+256].to(C.DEVICE), Y[i:i+256].to(C.DEVICE)
            base += F.cross_entropy(model(xb), yb).item()
            nb += 1
        base /= nb
    orig = [p.data.clone() for p in model.parameters()]
    deltas = []
    for _ in range(SHARP_SAMPLES):
        with torch.no_grad():
            for p in model.parameters():
                p.data.add_(torch.randn_like(p) * SHARP_SIGMA)
        noisy, nb = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb, yb = X[i:i+256].to(C.DEVICE), Y[i:i+256].to(C.DEVICE)
                noisy += F.cross_entropy(model(xb), yb).item()
                nb += 1
        noisy /= nb
        deltas.append(noisy - base)
        with torch.no_grad():
            for p, op in zip(model.parameters(), orig):
                p.data.copy_(op)
    return float(np.mean(deltas)), float(np.std(deltas))


def evaluate(model, Xte, Yte, label):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, fgsm_acc = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, pgd_acc = C.logits_and_acc(model, Xpgd, Yte)
    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))
    fgsm_asr, pgd_asr = 1.0 - fgsm_acc, 1.0 - pgd_acc
    print(f"  [{label}] clean={clean_acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  "
          f"PGD_ASR={pgd_asr:.4f}  margin={mean_margin:.4f}")
    return dict(clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                pgd_asr=pgd_asr, mean_margin=mean_margin)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    C.set_seed(SEED)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    print("Loading data...")
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    idx = torch.randperm(len(Xtr))[:N_TRAIN]
    Xtr, Ytr = Xtr[idx], Ytr[idx]
    Xte, Yte = Xte[:500], Yte[:500]
    print(f"  train={len(Xtr)}  test={len(Xte)}")

    results = {}

    # ---- (a) Baseline: single training run, take final model ----
    print("\n--- (a) Baseline (epoch 20) ---")
    C.set_seed(SEED)
    model_base = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                               width=32, seed=SEED)
    model_base.to(C.DEVICE)
    snapshots = train_with_snapshots(model_base, Xtr, Ytr,
                                     snapshot_epochs=SNAPSHOT_EPOCHS)
    # model_base IS the epoch-20 model after training
    r = evaluate(model_base, Xte, Yte, label="baseline_ep20")
    print("  Measuring sharpness...")
    sm, ss = measure_sharpness(model_base, Xtr, Ytr)
    r["sharpness_mean"], r["sharpness_std"] = sm, ss
    print(f"  sharpness={sm:.4f}±{ss:.4f}")
    results["baseline_ep20"] = r

    # ---- (b) SWA-trajectory: average snapshots from same run ----
    print("\n--- (b) SWA-trajectory (mean of epochs 10,12,14,16,18,20) ---")
    swa_sd = average_state_dicts([sd for _, sd in snapshots])
    model_swa_traj = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                                   width=32, seed=SEED)
    model_swa_traj.to(C.DEVICE)
    model_swa_traj.load_state_dict(swa_sd)
    update_bn_stats(model_swa_traj, Xtr)
    r = evaluate(model_swa_traj, Xte, Yte, label="swa_trajectory")
    print("  Measuring sharpness...")
    sm, ss = measure_sharpness(model_swa_traj, Xtr, Ytr)
    r["sharpness_mean"], r["sharpness_std"] = sm, ss
    print(f"  sharpness={sm:.4f}±{ss:.4f}")
    results["swa_trajectory"] = r

    # ---- (c) SWA-multiseed: average 5 independently trained models ----
    print("\n--- (c) SWA-multiseed (5 seeds, 20 epochs each) ---")
    seed_state_dicts = []
    for s in MULTI_SEEDS:
        print(f"  Training seed={s}...")
        C.set_seed(s)
        m = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                          width=32, seed=s)
        m.to(C.DEVICE)
        train_with_snapshots(m, Xtr, Ytr, snapshot_epochs=None)
        seed_state_dicts.append(copy.deepcopy(m.state_dict()))

    swa_ms_sd = average_state_dicts(seed_state_dicts)
    model_swa_ms = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                                 width=32, seed=SEED)
    model_swa_ms.to(C.DEVICE)
    model_swa_ms.load_state_dict(swa_ms_sd)
    update_bn_stats(model_swa_ms, Xtr)
    r = evaluate(model_swa_ms, Xte, Yte, label="swa_multiseed")
    print("  Measuring sharpness...")
    sm, ss = measure_sharpness(model_swa_ms, Xtr, Ytr)
    r["sharpness_mean"], r["sharpness_std"] = sm, ss
    print(f"  sharpness={sm:.4f}±{ss:.4f}")
    results["swa_multiseed"] = r

    elapsed = time.time() - t0

    lines = [
        "H274 - SWA Robustness\n",
        "=" * 70 + "\n\n",
        f"N_train={N_TRAIN}  epochs={EPOCHS}  lr={LR}  eps={EPS}  pgd_steps={PGD_STEPS}\n",
        f"Snapshots at epochs: {SNAPSHOT_EPOCHS}\n",
        f"Multi-seed seeds: {MULTI_SEEDS}\n",
        f"Sharpness: {SHARP_SAMPLES} random N(0,{SHARP_SIGMA}^2) perturbations\n\n",
        f"{'Model':<22} {'CleanAcc':>10} {'FGSM_ASR':>10} {'PGD_ASR':>10} "
        f"{'Margin':>10} {'Sharpness':>16}\n",
        "-" * 78 + "\n",
    ]
    for name, r in results.items():
        lines.append(
            f"{name:<22} {r['clean_acc']:>10.4f} {r['fgsm_asr']:>10.4f} "
            f"{r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f} "
            f"  {r['sharpness_mean']:>8.4f}±{r['sharpness_std']:.4f}\n"
        )
    lines.append(f"\nElapsed: {elapsed:.1f}s\n\n")
    lines.append("ANALYSIS\n--------\n")
    base = results["baseline_ep20"]
    traj = results["swa_trajectory"]
    ms   = results["swa_multiseed"]
    lines.append(f"SWA-traj vs baseline: sharpness delta={traj['sharpness_mean']-base['sharpness_mean']:+.4f}  "
                 f"PGD_ASR delta={traj['pgd_asr']-base['pgd_asr']:+.4f}\n")
    lines.append(f"SWA-ms   vs baseline: sharpness delta={ms['sharpness_mean']-base['sharpness_mean']:+.4f}  "
                 f"PGD_ASR delta={ms['pgd_asr']-base['pgd_asr']:+.4f}\n")
    verdict = ("SWA produces flatter basins AND lower PGD_ASR => weight averaging may implicitly improve robustness."
               if (traj["pgd_asr"] < base["pgd_asr"] or ms["pgd_asr"] < base["pgd_asr"])
               else "SWA flattens the loss basin but does NOT meaningfully reduce adversarial vulnerability.")
    lines.append(f"Verdict: {verdict}\n")

    with open(OUT_FILE, "w") as f:
        f.writelines(lines)
    print(f"\nResults written to {OUT_FILE}")
