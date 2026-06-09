"""
H516 - Sparse (L0) adversarial attacks: how many pixels must change to flip?

Paper: "σ-zero: Gradient-based optimization of ℓ0-norm adversarial examples"
(ICLR 2025). Sparse attacks constrain the NUMBER of perturbed pixels (L0 norm)
rather than the perturbation magnitude (L∞/L2). This is practically relevant:
a few corrupted pixels (e.g. dead sensor pixels, raindrops on signs) can fool
classifiers.

Hypothesis: (1) Fashion-MNIST CNNs can be flipped by modifying very few pixels
(≤5% of 784 = ~39 pixels) on most samples. (2) PGD-AT models require
significantly MORE pixel modifications than STD models (AT raises the L0
barrier). (3) Per-class L0 vulnerability correlates with per-class L∞
vulnerability (classes that are easy to attack under one norm are easy under
the other).

Method: greedy pixel attack (Jacobian Saliency Map Approach / JSMA-style):
  - Compute gradient of target-class logit w.r.t. each pixel
  - Iteratively flip the pixel with largest gradient magnitude to its
    extreme value (0 or 1)
  - Stop when prediction changes or pixel budget exhausted
  - Record number of pixels needed per sample

Controls: 3 seeds, STD vs PGD-AT, same architecture.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
N_SAMPLES = 300
MAX_PIXELS = 80  # max pixels to perturb (~10% of 784)
ADV_EPS_TRAIN = 0.1
ADV_STEPS_TRAIN = 7


def greedy_l0_attack(model, x, y_true, max_pixels=MAX_PIXELS):
    """Greedy L0 attack: iteratively flip the highest-gradient pixel.

    Returns: (n_pixels_to_flip, success_bool)
    """
    model.eval()
    x_adv = x.clone().unsqueeze(0)  # (1, C, H, W)
    modified = set()
    n_flat = x.numel()

    for step in range(max_pixels):
        x_adv_var = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv_var)
        # maximize loss of true class = minimize true-class logit
        loss = -logits[0, y_true]
        loss.backward()
        grad = x_adv_var.grad.detach().reshape(-1).clone()

        # mask already-modified pixels
        for idx in modified:
            grad[idx] = 0.0

        if grad.abs().max() == 0:
            break

        # pick pixel with largest |grad|
        best_idx = grad.abs().argmax().item()
        modified.add(best_idx)

        # flip to extreme value (0 or 1) in direction of gradient
        flat = x_adv.reshape(-1)
        if grad[best_idx] > 0:
            flat[best_idx] = 0.0  # decrease (since we're minimizing true logit)
        else:
            flat[best_idx] = 1.0
        x_adv = flat.reshape(x_adv.shape).clone()

        # check if flipped
        with torch.no_grad():
            pred = model(x_adv).argmax(1).item()
        if pred != y_true:
            return len(modified), True

    return len(modified), False


def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=2000, seed=seed)

    idx = torch.randperm(Xte.size(0), generator=torch.Generator().manual_seed(seed))[:N_SAMPLES]
    Xs, Ys = Xte[idx], Yte[idx]

    # STD model
    std_model = C.build_model("cnn", meta, seed=seed)
    C.train_model(std_model, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05, ncls=meta["n_classes"])

    # PGD-AT model
    at_model = C.build_model("cnn", meta, seed=seed)
    C.train_model(at_model, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05, ncls=meta["n_classes"],
                  adv_train=True, adv_eps=ADV_EPS_TRAIN, adv_steps=ADV_STEPS_TRAIN)

    results = {"seed": seed, "std": [], "at": []}
    Y_np = Ys.cpu().numpy()

    for tag, model in [("std", std_model), ("at", at_model)]:
        # check clean accuracy first
        with torch.no_grad():
            clean_pred = model(Xs).argmax(1).cpu().numpy()
        clean_correct = (clean_pred == Y_np)

        pixels_needed = []
        successes = 0
        total_attacked = 0

        for i in range(N_SAMPLES):
            if not clean_correct[i]:
                continue
            total_attacked += 1
            n_pix, success = greedy_l0_attack(model, Xs[i], int(Y_np[i]))
            if success:
                successes += 1
                pixels_needed.append(n_pix)

        results[tag] = {
            "clean_acc": float(clean_correct.mean()),
            "l0_asr": successes / max(total_attacked, 1),
            "mean_pixels": float(np.mean(pixels_needed)) if pixels_needed else float("nan"),
            "median_pixels": float(np.median(pixels_needed)) if pixels_needed else float("nan"),
            "total_attacked": total_attacked,
            "successes": successes,
        }

        # Per-class L0 vulnerability
        per_class = {}
        for c in range(meta["n_classes"]):
            mask_c = (Y_np == c) & clean_correct
            if mask_c.sum() == 0:
                per_class[c] = float("nan")
                continue
            pix_c = []
            for i in np.where(mask_c)[0]:
                n_pix, success = greedy_l0_attack(model, Xs[i], int(Y_np[i]))
                if success:
                    pix_c.append(n_pix)
            per_class[c] = float(np.mean(pix_c)) if pix_c else float("nan")
        results[tag]["per_class_mean_pixels"] = per_class

    return results


def main():
    t0 = time.time()
    all_results = [run_seed(s) for s in SEEDS]

    print("=" * 72)
    print("H516 — Sparse L0 Adversarial Attack Analysis")
    print("=" * 72)

    for r in all_results:
        print(f"\n--- Seed {r['seed']} ---")
        for tag in ["std", "at"]:
            d = r[tag]
            print(f"  [{tag.upper()}] Clean={d['clean_acc']:.3f}  L0-ASR={d['l0_asr']:.3f}  "
                  f"Mean-pixels={d['mean_pixels']:.1f}  Median-pixels={d['median_pixels']:.1f}  "
                  f"({d['successes']}/{d['total_attacked']} flipped)")

    print("\n" + "=" * 72)
    print("CROSS-SEED SUMMARY")
    print("=" * 72)
    for tag in ["std", "at"]:
        asrs = [r[tag]["l0_asr"] for r in all_results]
        mpix = [r[tag]["mean_pixels"] for r in all_results]
        print(f"  [{tag.upper()}] L0-ASR={np.mean(asrs):.3f}±{np.std(asrs):.3f}  "
              f"Mean-pixels={np.nanmean(mpix):.1f}±{np.nanstd(mpix):.1f}")

    std_mpix = np.nanmean([r["std"]["mean_pixels"] for r in all_results])
    at_mpix = np.nanmean([r["at"]["mean_pixels"] for r in all_results])
    print(f"\n  L0 barrier: STD={std_mpix:.1f} pixels vs AT={at_mpix:.1f} pixels")
    if at_mpix > std_mpix:
        print("  → PGD-AT raises the L0 pixel barrier (harder to attack sparsely)")
    else:
        print("  → PGD-AT does NOT raise the L0 pixel barrier")

    # Per-class comparison (last seed)
    print("\n  Per-class mean pixels to flip (last seed):")
    r = all_results[-1]
    for c in range(10):
        s = r["std"]["per_class_mean_pixels"].get(c, float("nan"))
        a = r["at"]["per_class_mean_pixels"].get(c, float("nan"))
        print(f"    Class {c}: STD={s:.1f}  AT={a:.1f}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
