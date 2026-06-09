"""
H200 - What fraction of Fashion-MNIST adversarials are "label-incongruent"?

Paper 2410.12671 (DUCAT) finds up to 40% of CIFAR-10 adversarials are
label-incongruent (the perturbed image visually belongs to a different class
than the original).  For Fashion-MNIST (lower inter-class ambiguity) we
expect a smaller but nonzero fraction.

We approximate visual congruence via a second independently-trained model B
(seed=1): if model_B also predicts a class different from the true label on the
adversarial example, the adversarial is "label-incongruent" — the perturbation
has pushed the image past a genuine perceptual boundary, not just exploited
model_A's idiosyncratic decision surface.

Measurements:
  1. Overall fraction of fooling adversarials that are label-incongruent.
  2. Per-class breakdown of label-incongruence rate.
  3. Most common adversarial target classes for label-incongruent examples.
  4. Correlation: is label-incongruence rate higher for low-margin (clean) samples?
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ config ---
DATASET   = "fashion_mnist"
N_EVAL    = 500
EPS       = 0.1
PGD_STEPS = 10
EPOCHS    = 10
WIDTH     = 32

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "h200_label_incongruent_adversarials_output.txt")

# ------------------------------------------------------------------ helpers --

def get_softmax_margins(model, X, Y):
    """Return softmax probability of true class minus max-other-class probability."""
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(X), dim=1)
    true_prob = probs.gather(1, Y[:, None]).squeeze(1)
    tmp = probs.clone()
    tmp[torch.arange(tmp.size(0)), Y.cpu() if Y.device.type == "cpu" else Y] = -1e9
    other_prob = tmp.max(1).values
    return (true_prob - other_prob).cpu().numpy()


def correlation(x, y):
    """Pearson r between two numpy arrays."""
    if len(x) < 2:
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float((xm * ym).sum() / denom) if denom > 0 else float("nan")


# -------------------------------------------------------------------- main ---

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lines = []

    def pr(*args, **kw):
        msg = " ".join(str(a) for a in args)
        print(msg, **kw)
        lines.append(msg)

    pr("=" * 74)
    pr("H200 - Label-Incongruent Adversarial Examples on Fashion-MNIST")
    pr("=" * 74)
    pr(f"Device={C.DEVICE}  eps={EPS}  pgd_steps={PGD_STEPS}  "
       f"n_eval={N_EVAL}  epochs={EPOCHS}  width={WIDTH}")
    pr()

    # ---- load data -----------------------------------------------------------
    t0 = time.time()
    pr("Loading dataset …")
    Xtr, Ytr, Xte, Yte = C.load_dataset(DATASET, n_eval=N_EVAL)
    Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]
    pr(f"  train={Xtr.size(0)}  eval={Xte.size(0)}")

    # ---- train model A -------------------------------------------------------
    pr("\nTraining model_A (seed=0) …")
    meta = C.dataset_meta(DATASET)
    C.set_seed(0)
    model_A = C.build_model("cnn", meta, width=WIDTH, seed=0)
    C.train_model(model_A, Xtr, Ytr, epochs=EPOCHS, opt="sgd", lr=0.05, ncls=10)
    with torch.no_grad():
        acc_A = float((model_A(Xte).argmax(1) == Yte).float().mean())
    pr(f"  model_A test acc = {acc_A:.4f}")

    # ---- train model B -------------------------------------------------------
    pr("\nTraining model_B (seed=1) …")
    C.set_seed(1)
    model_B = C.build_model("cnn", meta, width=WIDTH, seed=1)
    C.train_model(model_B, Xtr, Ytr, epochs=EPOCHS, opt="sgd", lr=0.05, ncls=10)
    with torch.no_grad():
        acc_B = float((model_B(Xte).argmax(1) == Yte).float().mean())
    pr(f"  model_B test acc = {acc_B:.4f}")

    # ---- clean softmax margins (for correlation analysis) --------------------
    pr("\nComputing clean softmax margins from model_A …")
    clean_margins = get_softmax_margins(model_A, Xte, Yte)
    pr(f"  mean margin = {clean_margins.mean():.4f}  "
       f"std = {clean_margins.std():.4f}  "
       f"min = {clean_margins.min():.4f}")

    # ---- generate adversarials -----------------------------------------------
    pr(f"\nGenerating PGD adversarials (eps={EPS}, steps={PGD_STEPS}) on model_A …")
    X_adv = C.pgd(model_A, Xte, Yte, eps=EPS, steps=PGD_STEPS)

    model_A.eval()
    model_B.eval()
    with torch.no_grad():
        adv_class_A = model_A(X_adv).argmax(1)
        adv_class_B = model_B(X_adv).argmax(1)
        clean_class_A = model_A(Xte).argmax(1)

    # mask: adversarials that actually fool model_A
    Y_cpu   = Yte.cpu()
    adv_A   = adv_class_A.cpu()
    adv_B   = adv_class_B.cpu()
    fooled  = adv_A != Y_cpu                       # PGD fooled model_A
    n_total = fooled.sum().item()

    pr(f"  Total eval samples : {N_EVAL}")
    pr(f"  Fooled model_A     : {n_total}  "
       f"(ASR = {n_total/N_EVAL:.3f})")

    if n_total == 0:
        pr("\nNo adversarials generated — cannot proceed.")
        with open(OUTPUT_FILE, "w") as f:
            f.write("\n".join(lines))
        return

    # ---- label incongruence --------------------------------------------------
    # Among fooling adversarials, model_B also disagrees with true label
    fooled_np = fooled.numpy().astype(bool)
    adv_B_np  = adv_B.numpy()
    Y_np      = Y_cpu.numpy()
    adv_A_np  = adv_A.numpy()

    incongruent = fooled_np & (adv_B_np != Y_np)   # model_B also sees different class
    congruent   = fooled_np & (adv_B_np == Y_np)   # model_B still sees true class

    n_incongruent = incongruent.sum()
    n_congruent   = congruent.sum()
    frac_incon    = n_incongruent / n_total

    pr()
    pr("=" * 74)
    pr("1. OVERALL LABEL-INCONGRUENCE RESULTS")
    pr("=" * 74)
    pr(f"  Fooling adversarials    : {n_total}")
    pr(f"  Label-incongruent       : {n_incongruent}  "
       f"({frac_incon*100:.1f}%)")
    pr(f"  Label-congruent         : {n_congruent}  "
       f"({n_congruent/n_total*100:.1f}%)")
    pr(f"  (DUCAT CIFAR-10 baseline: ~40%)")

    # ---- per-class breakdown -------------------------------------------------
    pr()
    pr("=" * 74)
    pr("2. PER-CLASS LABEL-INCONGRUENCE RATE")
    pr("=" * 74)
    pr(f"  {'Class':20s}  {'Fooled':>7}  {'Incon':>7}  {'Rate':>7}")
    pr(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*7}")
    per_class_incon = {}
    for c in range(10):
        mask_c  = (Y_np == c) & fooled_np
        n_c     = mask_c.sum()
        n_c_inc = (mask_c & (adv_B_np != Y_np)).sum()
        rate    = n_c_inc / n_c if n_c > 0 else float("nan")
        per_class_incon[c] = {"n_fooled": int(n_c), "n_incon": int(n_c_inc), "rate": rate}
        pr(f"  {CLASS_NAMES[c]:20s}  {n_c:7d}  {n_c_inc:7d}  {rate:7.3f}")

    # ---- most common adversarial target classes for incongruent examples -----
    pr()
    pr("=" * 74)
    pr("3. MOST COMMON ADVERSARIAL TARGET CLASSES (label-incongruent only)")
    pr("=" * 74)
    inc_targets = adv_A_np[incongruent]
    if len(inc_targets) > 0:
        target_counts = np.bincount(inc_targets, minlength=10)
        order = np.argsort(-target_counts)
        pr(f"  {'Target class':20s}  {'Count':>7}  {'Frac':>7}")
        pr(f"  {'-'*20}  {'-'*7}  {'-'*7}")
        for c in order:
            if target_counts[c] > 0:
                pr(f"  {CLASS_NAMES[c]:20s}  {target_counts[c]:7d}  "
                   f"{target_counts[c]/len(inc_targets):7.3f}")
    else:
        pr("  (no label-incongruent adversarials found)")

    # model_B target distribution for incongruent adversarials
    pr()
    pr("  Model_B predicted classes for incongruent adversarials:")
    inc_B_targets = adv_B_np[incongruent]
    if len(inc_B_targets) > 0:
        tbc = np.bincount(inc_B_targets, minlength=10)
        order_b = np.argsort(-tbc)
        pr(f"  {'Target class (B)':20s}  {'Count':>7}  {'Frac':>7}")
        pr(f"  {'-'*20}  {'-'*7}  {'-'*7}")
        for c in order_b:
            if tbc[c] > 0:
                pr(f"  {CLASS_NAMES[c]:20s}  {tbc[c]:7d}  "
                   f"{tbc[c]/len(inc_B_targets):7.3f}")

    # ---- correlation: margin vs incongruence ---------------------------------
    pr()
    pr("=" * 74)
    pr("4. MARGIN vs LABEL-INCONGRUENCE CORRELATION")
    pr("=" * 74)
    fooled_margins = clean_margins[fooled_np]
    fooled_incon   = (adv_B_np[fooled_np] != Y_np[fooled_np]).astype(float)

    r = correlation(fooled_margins, fooled_incon)
    pr(f"  Pearson r (clean margin vs incongruent): {r:.4f}")
    pr(f"  Interpretation: negative r means lower-margin samples are MORE "
       f"likely to be label-incongruent.")

    # Tertile breakdown
    if n_total >= 10:
        tertiles = np.percentile(fooled_margins, [33, 67])
        for label, lo, hi in [
            ("Low margin  (< p33)", -np.inf, tertiles[0]),
            ("Mid margin  (p33-p67)", tertiles[0], tertiles[1]),
            ("High margin (> p67)", tertiles[1], np.inf),
        ]:
            mask_t = (fooled_margins > lo) & (fooled_margins <= hi)
            n_t    = mask_t.sum()
            n_inc  = fooled_incon[mask_t].sum()
            rate   = n_inc / n_t if n_t > 0 else float("nan")
            pr(f"  {label}: n={n_t:4d}  incon_rate={rate:.3f}")

    # ---- summary verdict -----------------------------------------------------
    pr()
    pr("=" * 74)
    pr("VERDICT")
    pr("=" * 74)
    pr(f"  Fashion-MNIST label-incongruence rate: {frac_incon*100:.1f}%")
    if frac_incon >= 0.30:
        verdict = "HIGH — comparable to CIFAR-10 DUCAT findings (~40%)."
    elif frac_incon >= 0.15:
        verdict = "MODERATE — smaller than CIFAR-10 as expected for lower-ambiguity data."
    elif frac_incon >= 0.05:
        verdict = "LOW — Fashion-MNIST adversarials are mostly model-specific artifacts."
    else:
        verdict = "VERY LOW — adversarials almost entirely exploit model_A's idiosyncratic surface."
    pr(f"  Classification: {verdict}")
    pr(f"  Total runtime: {time.time()-t0:.1f}s")
    pr("=" * 74)

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
