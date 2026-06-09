"""
H430 - Label smoothing as a defense: gradient masking artifact or true robustness?

Hypothesis
----------
Label smoothing (LS) with parameter α replaces hard one-hot targets with a
soft distribution (1−α on the true class, α/(K−1) on others).  This reduces
maximum logit confidence at convergence.  Müller et al. (NeurIPS 2019, "When
does label smoothing help?") show LS calibrates confidence but also COMPRESSES
the logit gap between classes.  Shafahi et al. (arXiv:1910.11585) note that
lower-confidence models appear more robust to gradient-based attacks in naive
evaluation, yet this may be a gradient-masking artifact: if the loss surface
is flatter near clean inputs, iterative gradient attacks stall without actually
finding a decision boundary that is farther away.

Prediction:
  - PGD attack-success DECREASES with α when evaluated with standard steps.
  - Transfer attack (adversarials crafted on the α=0 model, evaluated on the
    smoothed model) should be LESS affected — if true robustness, transfer
    should also drop; if masking, transfer stays high because the boundary
    geometry (not gradient) decides success.
  - Logit confidence (max-softmax) decreases with α, explaining the PGD drop.
  - Conclusion: LS gives apparent robustness mainly through gradient masking,
    not a genuine shift in decision boundary geometry.

Experiment
----------
For each α ∈ {0, 0.1, 0.2, 0.5} and SEEDS:
  1. Train a CNN with label-smoothing cross-entropy loss.
  2. Record clean accuracy and mean max-softmax confidence.
  3. Evaluate PGD attack-success (white-box, ε=0.1, 20 steps) on that model.
  4. Evaluate transfer attack-success: adversarials crafted on the α=0 model
     of the same seed, then evaluated on the smoothed model.
  5. Compare PGD vs transfer trends across α to diagnose masking.

References
----------
- Müller et al. (2019) "When does label smoothing help?" NeurIPS.
- Shafahi et al. (2019) "Label smoothing and logit squeezing: A replacement
  for mixup for smoothing distributions over classes." arXiv:1910.11585.
- Hypothesis: LS lowers logit confidence → gradient-based attacks stall
  (masking), but transfer attacks expose the true (unchanged) geometry.
"""
import os, sys, time, pathlib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DS       = "fashion_mnist"
SEEDS    = [0, 1, 2]
ALPHAS   = [0.0, 0.1, 0.2, 0.5]   # label-smoothing strengths
EPS      = 0.1                     # L-inf budget
PGD_STEPS = 20
OUT_FILE = pathlib.Path(__file__).resolve().parent.parent / \
           "results" / "fashion_mnist" / "h430_label_smoothing_defense_output.txt"


# ---------------------------------------------------------------------------
# Label-smoothing loss
# ---------------------------------------------------------------------------
def smooth_ce(logits, targets, alpha, n_classes):
    """Cross-entropy with label smoothing (Müller et al. 2019)."""
    log_prob = F.log_softmax(logits, dim=1)
    # one-hot
    one_hot = torch.zeros_like(log_prob).scatter_(1, targets.unsqueeze(1), 1.0)
    # soft target
    smooth = one_hot * (1.0 - alpha) + (alpha / n_classes)
    return -(smooth * log_prob).sum(dim=1).mean()


def train_smoothed(meta, Xtr, Ytr, alpha, seed):
    """Train a CNN with label-smoothed CE; return trained model."""
    C.set_seed(seed)
    model = C.build_model("cnn", meta, seed=seed)
    n_classes = meta["n_classes"]
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9,
                          weight_decay=1e-4)
    model.train()
    batch = 128
    epochs = 8
    idx = torch.randperm(Xtr.size(0), generator=torch.Generator().manual_seed(seed))
    Xtr_s, Ytr_s = Xtr[idx], Ytr[idx]
    for _ in range(epochs):
        for start in range(0, Xtr_s.size(0), batch):
            xb = Xtr_s[start:start + batch]
            yb = Ytr_s[start:start + batch]
            opt.zero_grad()
            loss = smooth_ce(model(xb), yb, alpha, n_classes)
            loss.backward()
            opt.step()
    model.eval()
    return model


def max_softmax_confidence(model, X, batch=512):
    """Mean of max softmax probability over X."""
    vals = []
    with torch.no_grad():
        for start in range(0, X.size(0), batch):
            xb = X[start:start + batch]
            probs = F.softmax(model(xb), dim=1)
            vals.append(probs.max(dim=1).values.cpu())
    return float(torch.cat(vals).mean())


def transfer_attack_success(src_model, tgt_model, X, Y, eps, steps):
    """Craft adversarials on src_model; measure success on tgt_model."""
    advs = C.pgd(src_model, X, Y, eps, steps)
    with torch.no_grad():
        preds = tgt_model(advs).argmax(1).cpu()
    return float((preds != Y.cpu()).float().mean())


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------
def run_seed(seed, meta, Xtr, Ytr, Xte, Yte):
    results = {}

    # Train baseline (α=0) first — needed for transfer attacks
    base_model = train_smoothed(meta, Xtr, Ytr, alpha=0.0, seed=seed)

    for alpha in ALPHAS:
        t0 = time.time()
        if alpha == 0.0:
            model = base_model
        else:
            model = train_smoothed(meta, Xtr, Ytr, alpha=alpha, seed=seed)

        # Clean accuracy
        _, clean_acc = C.logits_and_acc(model, Xte, Yte)

        # Confidence
        conf = max_softmax_confidence(model, Xte)

        # White-box PGD attack-success
        pgd_asr = C.attack_success(model, Xte, Yte, attack="pgd",
                                   eps=EPS, steps=PGD_STEPS)

        # Transfer attack-success (advs from α=0 model → this model)
        if alpha == 0.0:
            transfer_asr = pgd_asr   # same model — transfer == white-box
        else:
            transfer_asr = transfer_attack_success(
                base_model, model, Xte, Yte, EPS, PGD_STEPS)

        results[alpha] = {
            "clean_acc":    clean_acc,
            "confidence":   conf,
            "pgd_asr":      pgd_asr,
            "transfer_asr": transfer_asr,
            "runtime_s":    round(time.time() - t0, 1),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 74)
    out("H430 - Label smoothing as defense: masking or true robustness?")
    out("=" * 74)
    out(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  pgd_steps={PGD_STEPS}")
    out(f"Alphas={ALPHAS}  seeds={SEEDS}")
    out()

    meta = C.dataset_meta(DS)
    # Aggregate across seeds: {alpha: {metric: [values]}}
    agg = {a: {"clean_acc": [], "confidence": [],
               "pgd_asr": [], "transfer_asr": []} for a in ALPHAS}

    for seed in SEEDS:
        out(f"--- Seed {seed} ---")
        C.set_seed(seed)
        Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=2000,
                                             seed=seed)
        seed_res = run_seed(seed, meta, Xtr, Ytr, Xte, Yte)

        out(f"  {'alpha':>6}  {'clean_acc':>9}  {'confidence':>10}  "
            f"{'pgd_asr':>8}  {'transfer_asr':>13}  {'time_s':>7}")
        for a in ALPHAS:
            r = seed_res[a]
            out(f"  {a:>6.2f}  {r['clean_acc']:>9.3f}  {r['confidence']:>10.3f}  "
                f"{r['pgd_asr']:>8.3f}  {r['transfer_asr']:>13.3f}  "
                f"{r['runtime_s']:>7.1f}")
            for k in agg[a]:
                agg[a][k].append(r[k])
        out()

    # Summary table
    out("=" * 74)
    out("MEAN across seeds")
    out(f"  {'alpha':>6}  {'clean_acc':>9}  {'confidence':>10}  "
        f"{'pgd_asr':>8}  {'transfer_asr':>13}  {'pgd_drop':>9}  {'xfer_drop':>9}")
    base_pgd  = float(np.mean(agg[0.0]["pgd_asr"]))
    base_xfer = float(np.mean(agg[0.0]["transfer_asr"]))
    for a in ALPHAS:
        ca   = float(np.mean(agg[a]["clean_acc"]))
        conf = float(np.mean(agg[a]["confidence"]))
        pgd  = float(np.mean(agg[a]["pgd_asr"]))
        xfer = float(np.mean(agg[a]["transfer_asr"]))
        pgd_drop  = base_pgd  - pgd
        xfer_drop = base_xfer - xfer
        out(f"  {a:>6.2f}  {ca:>9.3f}  {conf:>10.3f}  "
            f"{pgd:>8.3f}  {xfer:>13.3f}  {pgd_drop:>+9.3f}  {xfer_drop:>+9.3f}")

    out()
    out("=" * 74)
    out("Interpretation")
    out("-" * 74)
    out("Masking signature: PGD-ASR drops substantially with α, while")
    out("transfer-ASR drops little or not at all.")
    out("True robustness signature: both PGD-ASR and transfer-ASR decrease")
    out("proportionally, and max-softmax confidence remains moderate.")
    out()
    out("If pgd_drop >> xfer_drop: label smoothing IS gradient masking —")
    out("  logit compression flattens gradients and stalls PGD, but the")
    out("  decision boundary is unchanged (transfer probes true geometry).")
    out("If pgd_drop ≈ xfer_drop: label smoothing provides genuine robustness")
    out("  (boundary geometry shifts, not just gradient signal).")
    out()
    out("References:")
    out("  Müller et al. (NeurIPS 2019) 'When does label smoothing help?'")
    out("  Shafahi et al. (arXiv:1910.11585) label smoothing + robustness.")
    out("=" * 74)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines) + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
