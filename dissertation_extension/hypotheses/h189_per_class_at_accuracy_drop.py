"""
H189 - Per-class clean accuracy drop from adversarial training correlates
       with class boundary proximity.

Hypothesis: classes whose natural samples sit close to decision boundaries
(low logit margin) suffer the largest clean accuracy drop from adversarial
training.

Methodology:
  - Train standard model and AT model (PGD-based, eps=0.3) on Fashion-MNIST (6k).
  - For each of 10 classes: clean acc (standard), clean acc (AT), acc drop,
    mean logit margin (standard model).
  - Pearson correlation between acc_drop and mean_margin across 10 classes.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import campaign.common as C

# ---- config ----------------------------------------------------------------
SEED = 42
N_TRAIN = 6000
N_EVAL = 1000
EPS = 0.3
AT_EPOCHS = 12
STD_EPOCHS = 12
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
OUT_FILE = os.path.join(RESULTS_DIR, "h189_per_class_at_accuracy_drop_output.txt")

FMNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main():
    t0 = time.time()
    C.set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = ["=" * 70, "H189 - Per-class AT Accuracy Drop vs Boundary Proximity", "=" * 70, ""]

    # load data
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")

    # train standard model
    print("Training standard model...")
    std_model = C.build_model("cnn", meta, width=32)
    C.train_model(std_model, Xtr, Ytr, epochs=STD_EPOCHS, opt="adam", lr=1e-3)
    std_model.eval()

    # train AT model
    print("Training AT model (PGD-based)...")
    at_model = C.build_model("cnn", meta, width=32)
    C.train_model(at_model, Xtr, Ytr, epochs=AT_EPOCHS, opt="adam", lr=1e-3,
                  adv_train=True, adv_eps=EPS, adv_steps=7)
    at_model.eval()

    # overall accuracies
    _, std_acc = C.logits_and_acc(std_model, Xte, Yte)
    _, at_acc = C.logits_and_acc(at_model, Xte, Yte)
    lines.append(f"Overall clean acc (standard): {std_acc:.4f}")
    lines.append(f"Overall clean acc (AT):       {at_acc:.4f}")
    lines.append(f"Overall acc drop:             {std_acc - at_acc:.4f}")
    lines.append("")

    # per-class analysis
    Yte_cpu = Yte.cpu()
    std_logits, _ = C.logits_and_acc(std_model, Xte, Yte)
    at_logits, _ = C.logits_and_acc(at_model, Xte, Yte)
    std_margins = C.margin_of(std_logits, Yte_cpu)

    class_accs_std = []
    class_accs_at = []
    class_drops = []
    class_margins = []

    header = f"{'Class':<15} {'Std_acc':>8} {'AT_acc':>8} {'Drop':>8} {'Mean_margin':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for c in range(10):
        mask = (Yte_cpu == c).numpy()
        n_c = mask.sum()
        if n_c == 0:
            continue

        std_pred = std_logits[mask].argmax(1).numpy()
        at_pred = at_logits[mask].argmax(1).numpy()
        acc_std = (std_pred == c).mean()
        acc_at = (at_pred == c).mean()
        drop = acc_std - acc_at
        mean_margin = std_margins[mask].mean()

        class_accs_std.append(acc_std)
        class_accs_at.append(acc_at)
        class_drops.append(drop)
        class_margins.append(mean_margin)

        lines.append(f"{FMNIST_CLASSES[c]:<15} {acc_std:>8.4f} {acc_at:>8.4f} "
                     f"{drop:>8.4f} {mean_margin:>12.3f}")

    lines.append("")

    # correlation
    drops_arr = np.array(class_drops)
    margins_arr = np.array(class_margins)

    r, pval = pearsonr(margins_arr, drops_arr)
    lines.append(f"Pearson r(mean_margin, acc_drop): r={r:.4f}, p={pval:.4f}")
    lines.append("")

    # interpretation
    if r < -0.3:
        lines.append("Direction: NEGATIVE correlation -- classes with lower margin (closer to")
        lines.append("  decision boundary) tend to have LARGER accuracy drops from AT.")
        lines.append("  This SUPPORTS the hypothesis.")
    elif r > 0.3:
        lines.append("Direction: POSITIVE correlation -- classes with higher margin have larger drops.")
        lines.append("  This CONTRADICTS the hypothesis.")
    else:
        lines.append("Weak or no correlation -- boundary proximity does not clearly predict")
        lines.append("  which classes lose the most accuracy from AT.")
    lines.append("")

    # also report adversarial robustness of AT model per class
    lines.append("--- Adversarial robustness (AT model, PGD-10, eps=0.3) per class ---")
    res = C.attack_success(at_model, Xte, Yte, attack="pgd", eps=EPS, steps=10)
    lines.append(f"Overall AT model ASR: {res['asr']:.4f}")
    lines.append("")

    # scatter data for manual plotting
    lines.append("--- Scatter data (for plotting) ---")
    lines.append(f"{'Class':<15} {'mean_margin':>12} {'acc_drop':>10}")
    for c in range(10):
        lines.append(f"{FMNIST_CLASSES[c]:<15} {class_margins[c]:>12.3f} {class_drops[c]:>10.4f}")
    lines.append("")

    elapsed = time.time() - t0
    lines.append(f"Elapsed: {elapsed:.1f}s")

    report = "\n".join(lines)
    print(report)
    with open(OUT_FILE, "w") as f:
        f.write(report)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
