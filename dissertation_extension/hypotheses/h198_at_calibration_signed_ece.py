"""
H198 - Adversarial training causes underconfidence (signed-ECE flip).

Hypothesis:
  Standard training causes overconfidence (signed-ECE < 0) while PGD-AT causes
  underconfidence (signed-ECE > 0); the magnitude of signed-ECE anti-correlates
  with clean accuracy across an AT epsilon sweep.

Protocol:
  Train 5 models on Fashion-MNIST (n_train=6000, n_eval=500):
    - Standard (eps=0)
    - PGD-AT with eps in {0.05, 0.1, 0.2, 0.3}
  For each model compute:
    - Clean accuracy
    - ECE (10 equal-width confidence bins)
    - Signed-ECE: positive = underconfident (acc > conf), negative = overconfident
    - Reliability diagram bins (avg_conf, avg_acc per bin)
  Report table, test signed-ECE sign flip, and Pearson r(eps, signed_ece).

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign.common import (
    DEVICE, set_seed, load_dataset, dataset_meta,
    build_model, train_model, logits_and_acc,
)

# ── hyperparameters ──────────────────────────────────────────────────────────
DATASET    = "fashion_mnist"
N_TRAIN    = 6000
N_EVAL     = 500
SEED       = 42
EPOCHS     = 15
N_BINS     = 10
EPS_LIST   = [0.0, 0.05, 0.1, 0.2, 0.3]


def compute_calibration(logits, labels, n_bins=N_BINS):
    """Compute ECE, signed-ECE, and per-bin reliability data."""
    probs = torch.softmax(logits, dim=1)
    confidences, preds = probs.max(dim=1)
    accuracies = (preds == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    signed_ece = 0.0
    bins_data = []
    n = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        bin_size = mask.sum().item()
        if bin_size == 0:
            bins_data.append((lo, hi, 0, float("nan"), float("nan")))
            continue
        avg_conf = confidences[mask].mean().item()
        avg_acc = accuracies[mask].mean().item()
        weight = bin_size / n
        ece += weight * abs(avg_acc - avg_conf)
        signed_ece += weight * (avg_acc - avg_conf)  # positive = underconfident
        bins_data.append((lo, hi, bin_size, avg_conf, avg_acc))

    return ece, signed_ece, bins_data


def main():
    t0 = time.time()
    set_seed(SEED)
    meta = dataset_meta(DATASET)
    Xtr, Ytr, Xte, Yte = load_dataset(DATASET, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    results = []
    for eps in EPS_LIST:
        adv_train = eps > 0
        label = f"AT-eps={eps:.2f}" if adv_train else "Standard"
        print(f"\n{'='*60}")
        print(f"Training: {label}")
        model = build_model("cnn", meta, width=32)
        model = train_model(
            model, Xtr, Ytr, epochs=EPOCHS, batch=128, opt="sgd", lr=0.05,
            adv_train=adv_train, adv_eps=eps, adv_steps=7, verbose=True,
        )
        logits, acc = logits_and_acc(model, Xte, Yte)
        ece, signed_ece, bins_data = compute_calibration(logits, Yte.cpu())
        results.append({
            "eps": eps, "label": label, "clean_acc": acc,
            "ece": ece, "signed_ece": signed_ece, "bins": bins_data,
        })
        print(f"  clean_acc={acc:.4f}  ECE={ece:.4f}  signed_ECE={signed_ece:+.4f}")

    # ── analysis ─────────────────────────────────────────────────────────────
    eps_arr = np.array([r["eps"] for r in results])
    sece_arr = np.array([r["signed_ece"] for r in results])
    r_val, p_val = pearsonr(eps_arr, sece_arr)

    std_sece = results[0]["signed_ece"]
    high_at_sece = results[-1]["signed_ece"]
    sign_flip = (std_sece < 0) and (high_at_sece > 0)

    # ── output ───────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("H198 - AT Calibration: Signed-ECE Flip")
    lines.append("=" * 70)
    lines.append(f"Dataset: {DATASET}  n_train={N_TRAIN}  n_eval={N_EVAL}  seed={SEED}")
    lines.append(f"Epochs: {EPOCHS}  Bins: {N_BINS}")
    lines.append("")

    # main table
    lines.append(f"{'Model':<18} {'eps':>5} {'clean_acc':>10} {'ECE':>8} {'signed_ECE':>12}")
    lines.append("-" * 58)
    for r in results:
        lines.append(f"{r['label']:<18} {r['eps']:>5.2f} {r['clean_acc']:>10.4f} "
                      f"{r['ece']:>8.4f} {r['signed_ece']:>+12.4f}")

    lines.append("")
    lines.append("Reliability diagram (per-bin avg_conf, avg_acc):")
    for r in results:
        lines.append(f"\n  {r['label']}:")
        lines.append(f"    {'Bin':>12} {'Count':>6} {'AvgConf':>8} {'AvgAcc':>8} {'Gap':>8}")
        for lo, hi, cnt, conf, acc in r["bins"]:
            gap = f"{acc - conf:+.4f}" if cnt > 0 else "   n/a"
            conf_s = f"{conf:.4f}" if cnt > 0 else "   n/a"
            acc_s = f"{acc:.4f}" if cnt > 0 else "   n/a"
            lines.append(f"    [{lo:.1f},{hi:.1f}){cnt:>6} {conf_s:>8} {acc_s:>8} {gap:>8}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("KEY TESTS")
    lines.append("=" * 70)
    lines.append(f"Standard signed-ECE:  {std_sece:+.4f}  ({'overconfident' if std_sece < 0 else 'underconfident'})")
    lines.append(f"AT-eps=0.30 signed-ECE: {high_at_sece:+.4f}  ({'overconfident' if high_at_sece < 0 else 'underconfident'})")
    lines.append(f"Sign flip (std<0 → AT>0): {sign_flip}")
    lines.append(f"Pearson r(eps, signed_ECE) = {r_val:+.4f}  (p={p_val:.4f})")
    lines.append(f"Hypothesis supported: {sign_flip and r_val > 0.5}")
    lines.append(f"\nTotal time: {time.time()-t0:.1f}s")

    output = "\n".join(lines)
    print("\n" + output)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", DATASET), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", DATASET,
                            "h198_at_calibration_signed_ece_output.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
