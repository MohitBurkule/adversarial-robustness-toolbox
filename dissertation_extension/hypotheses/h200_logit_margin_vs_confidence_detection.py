"""
H200 - Logit margin vs max-confidence as adversarial detector.

Hypothesis:
  Logit margin (top_logit - second_logit) is a significantly better detector of
  PGD adversarial examples than max softmax confidence, especially for PGD-AT
  models -- where max-confidence AUROC drops to ~0.60 but logit-margin AUROC
  remains ~0.80. This extends the dissertation's core finding that logit margin
  predicts vulnerability.

Protocol:
  Train 2 models on Fashion-MNIST (n_train=6000):
    A: standard cross-entropy (15 epochs)
    B: PGD-AT (eps=0.3, steps=7, 15 epochs)
  For each model generate PGD-10 adversarial examples (eps=0.3) on 500 test samples.
  Compute per-sample: max_softmax_confidence, logit_margin, entropy.
  Binary detection (clean=0, adv=1) AUROC using each feature as score.
  Key test: auroc_margin > auroc_conf for the AT model.
  Also report threshold vs TPR/FPR at 5 operating points per model.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign.common import (
    DEVICE, set_seed, load_dataset, dataset_meta,
    build_model, train_model, logits_and_acc, pgd, safe_auroc,
)

# ── hyperparameters ──────────────────────────────────────────────────────────
DATASET    = "fashion_mnist"
N_TRAIN    = 6000
N_EVAL     = 500
SEED       = 42
EPOCHS     = 15
ADV_EPS    = 0.3
PGD_STEPS  = 10


def compute_detection_features(model, X, batch=256):
    """Compute max_confidence, logit_margin, entropy for each sample."""
    model.eval()
    confs, margins, entropies = [], [], []
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            xb = X[i:i + batch]
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            # max confidence
            max_conf, _ = probs.max(dim=1)
            confs.append(max_conf.cpu())
            # logit margin: top1 - top2
            top2 = logits.topk(2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            margins.append(margin.cpu())
            # entropy
            ent = -(probs * (probs + 1e-10).log()).sum(dim=1)
            entropies.append(ent.cpu())
    return (torch.cat(confs).numpy(), torch.cat(margins).numpy(),
            torch.cat(entropies).numpy())


def tpr_fpr_at_thresholds(labels, scores, n_points=5):
    """Compute TPR/FPR at n_points evenly-spaced score thresholds."""
    thresholds = np.linspace(scores.min(), scores.max(), n_points + 2)[1:-1]
    results = []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        results.append((t, tpr, fpr))
    return results


def main():
    t0 = time.time()
    set_seed(SEED)
    meta = dataset_meta(DATASET)
    Xtr, Ytr, Xte, Yte = load_dataset(DATASET, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    models = {}
    # Model A: standard
    print("Training Model A (standard)...")
    model_a = build_model("cnn", meta, width=32)
    model_a = train_model(model_a, Xtr, Ytr, epochs=EPOCHS, batch=128, opt="sgd", lr=0.05,
                          verbose=True)
    _, acc_a = logits_and_acc(model_a, Xte, Yte)
    models["Standard"] = model_a

    # Model B: PGD-AT
    print("\nTraining Model B (PGD-AT eps=0.3)...")
    set_seed(SEED)
    model_b = build_model("cnn", meta, width=32)
    model_b = train_model(model_b, Xtr, Ytr, epochs=EPOCHS, batch=128, opt="sgd", lr=0.05,
                          adv_train=True, adv_eps=ADV_EPS, adv_steps=7, verbose=True)
    _, acc_b = logits_and_acc(model_b, Xte, Yte)
    models["PGD-AT"] = model_b

    # ── detection evaluation ─────────────────────────────────────────────────
    all_results = {}
    for name, model in models.items():
        print(f"\nEvaluating detection for {name}...")
        # clean features
        conf_c, marg_c, ent_c = compute_detection_features(model, Xte)
        # generate adversarial examples
        Xadv = pgd(model, Xte, Yte, eps=ADV_EPS, steps=PGD_STEPS)
        conf_a, marg_a, ent_a = compute_detection_features(model, Xadv)

        # binary labels: clean=0, adv=1
        labels = np.concatenate([np.zeros(len(conf_c)), np.ones(len(conf_a))])

        # scores: lower conf/margin → more likely adversarial, so negate for AUROC
        score_conf = np.concatenate([-conf_c, -conf_a])
        score_marg = np.concatenate([-marg_c, -marg_a])
        # higher entropy → more likely adversarial
        score_ent = np.concatenate([ent_c, ent_a])

        auroc_conf = safe_auroc(labels, score_conf)
        auroc_marg = safe_auroc(labels, score_marg)
        auroc_ent = safe_auroc(labels, score_ent)

        # operating points for logit margin detector
        ops_margin = tpr_fpr_at_thresholds(labels, score_marg)
        ops_conf = tpr_fpr_at_thresholds(labels, score_conf)

        all_results[name] = {
            "auroc_conf": auroc_conf, "auroc_margin": auroc_marg,
            "auroc_entropy": auroc_ent,
            "ops_margin": ops_margin, "ops_conf": ops_conf,
        }
        print(f"  AUROC conf={auroc_conf:.4f}  margin={auroc_marg:.4f}  entropy={auroc_ent:.4f}")

    # ── output ───────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("H200 - Logit Margin vs Max-Confidence Adversarial Detection")
    lines.append("=" * 70)
    lines.append(f"Dataset: {DATASET}  n_train={N_TRAIN}  n_eval={N_EVAL}  seed={SEED}")
    lines.append(f"Epochs: {EPOCHS}  adv_eps={ADV_EPS}  PGD steps={PGD_STEPS}")
    lines.append(f"Model A (Standard) clean_acc: {acc_a:.4f}")
    lines.append(f"Model B (PGD-AT)   clean_acc: {acc_b:.4f}")
    lines.append("")

    # main AUROC table
    lines.append(f"{'Model':<12} {'AUROC_conf':>11} {'AUROC_margin':>13} {'AUROC_entropy':>14}")
    lines.append("-" * 54)
    for name in ["Standard", "PGD-AT"]:
        r = all_results[name]
        lines.append(f"{name:<12} {r['auroc_conf']:>11.4f} {r['auroc_margin']:>13.4f} "
                      f"{r['auroc_entropy']:>14.4f}")

    # operating points
    for name in ["Standard", "PGD-AT"]:
        r = all_results[name]
        lines.append(f"\n  {name} - Logit margin detector operating points:")
        lines.append(f"    {'Threshold':>10} {'TPR':>8} {'FPR':>8}")
        for t, tpr, fpr in r["ops_margin"]:
            lines.append(f"    {t:>10.4f} {tpr:>8.4f} {fpr:>8.4f}")
        lines.append(f"  {name} - Max-confidence detector operating points:")
        lines.append(f"    {'Threshold':>10} {'TPR':>8} {'FPR':>8}")
        for t, tpr, fpr in r["ops_conf"]:
            lines.append(f"    {t:>10.4f} {tpr:>8.4f} {fpr:>8.4f}")

    # key tests
    margin_wins_at = all_results["PGD-AT"]["auroc_margin"] > all_results["PGD-AT"]["auroc_conf"]
    margin_gap = all_results["PGD-AT"]["auroc_margin"] - all_results["PGD-AT"]["auroc_conf"]

    lines.append("")
    lines.append("=" * 70)
    lines.append("KEY TESTS")
    lines.append("=" * 70)
    lines.append(f"AT model: margin AUROC > conf AUROC? {margin_wins_at}  (gap={margin_gap:+.4f})")
    lines.append(f"Standard model: margin AUROC = {all_results['Standard']['auroc_margin']:.4f}, "
                  f"conf AUROC = {all_results['Standard']['auroc_conf']:.4f}")
    lines.append(f"Hypothesis supported (margin > conf for AT model): {margin_wins_at}")
    lines.append(f"\nTotal time: {time.time()-t0:.1f}s")

    output = "\n".join(lines)
    print("\n" + output)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", DATASET), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", DATASET,
                            "h200_logit_margin_vs_confidence_detection_output.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
