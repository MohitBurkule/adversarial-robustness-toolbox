"""
H515 - Frequency-domain analysis of adversarial vulnerability.

Paper: "Intriguing Frequency Interpretation of Adversarial Robustness for CNNs
and ViTs" (arXiv:2506.12875, June 2025). Key finding: adversarial perturbations
concentrate attack power in mid-to-high frequency bands for CNNs, while low-to-
mid frequencies matter more for transformers.

Hypothesis: On Fashion-MNIST CNNs, adversarial perturbations (PGD) contain
disproportionately more energy in the HIGH-frequency DCT bands compared to
the original image energy distribution. Furthermore, PGD-AT models produce
perturbations whose frequency profile is more spread (flatter) than STD models,
because AT forces the attacker to use a wider frequency range.

Experiment:
  (1) Train STD and PGD-AT CNNs on Fashion-MNIST.
  (2) Generate PGD adversarial examples for 500 test samples.
  (3) Compute 2D DCT of: original images, adversarial images, and the
      perturbation delta = adv - orig.
  (4) Partition the DCT spectrum into LOW / MID / HIGH frequency bands
      (by distance from DC component) and compute fraction of total energy
      in each band.
  (5) Compare band energy ratios between STD vs AT perturbations.
  (6) Measure correlation between high-freq perturbation energy and attack
      success (label flip).

Controls: 3 seeds, same architecture, matched PGD attack parameters.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
N_SAMPLES = 500
ADV_EPS = 0.15
ADV_STEPS = 20
ADV_EPS_TRAIN = 0.1
ADV_STEPS_TRAIN = 7


def dct2(x_np):
    """2D DCT via scipy."""
    from scipy.fft import dctn
    return dctn(x_np, type=2, norm='ortho', axes=(-2, -1))


def freq_band_energy(dct_coeffs, h, w):
    """Split DCT coefficients into LOW/MID/HIGH bands by Manhattan distance
    from DC component (top-left corner). Returns dict of band energies."""
    total_energy = (dct_coeffs ** 2).sum()
    if total_energy < 1e-15:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "total": 0.0}

    max_dist = h + w - 2
    low_thresh = max_dist / 3
    high_thresh = 2 * max_dist / 3

    low_e = 0.0
    mid_e = 0.0
    high_e = 0.0

    for i in range(h):
        for j in range(w):
            d = i + j
            e = dct_coeffs[i, j] ** 2
            if d <= low_thresh:
                low_e += e
            elif d <= high_thresh:
                mid_e += e
            else:
                high_e += e

    return {
        "low": float(low_e / total_energy),
        "mid": float(mid_e / total_energy),
        "high": float(high_e / total_energy),
        "total": float(total_energy),
    }


def analyze_freq(model, X, Y, label):
    """Generate PGD adversarial examples, compute DCT band energies."""
    model.eval()
    X_adv = C.pgd(model, X, Y, eps=ADV_EPS, steps=ADV_STEPS)

    with torch.no_grad():
        pred_clean = model(X).argmax(1).cpu().numpy()
        pred_adv = model(X_adv).argmax(1).cpu().numpy()

    Y_np = Y.cpu().numpy()
    clean_correct = (pred_clean == Y_np)
    flipped = clean_correct & (pred_adv != Y_np)
    asr = flipped.sum() / max(clean_correct.sum(), 1)

    X_np = X.cpu().numpy()
    Xa_np = X_adv.cpu().numpy()
    delta_np = Xa_np - X_np

    n = X_np.shape[0]
    h, w = X_np.shape[-2], X_np.shape[-1]

    # Compute band energies for perturbations
    bands_delta = {"low": [], "mid": [], "high": []}
    bands_orig = {"low": [], "mid": [], "high": []}

    for i in range(n):
        img = X_np[i].squeeze()  # (H, W)
        pert = delta_np[i].squeeze()

        dct_orig = dct2(img)
        dct_pert = dct2(pert)

        be_orig = freq_band_energy(dct_orig, h, w)
        be_pert = freq_band_energy(dct_pert, h, w)

        for b in ["low", "mid", "high"]:
            bands_orig[b].append(be_orig[b])
            bands_delta[b].append(be_pert[b])

    # Per-sample high-freq energy vs flip success correlation
    high_e = np.array(bands_delta["high"])
    flipped_f = flipped.astype(float)
    if high_e.std() > 0 and flipped_f.std() > 0:
        corr = float(np.corrcoef(high_e, flipped_f)[0, 1])
    else:
        corr = float("nan")

    return {
        "label": label,
        "asr": float(asr),
        "orig_band_means": {b: float(np.mean(bands_orig[b])) for b in ["low", "mid", "high"]},
        "pert_band_means": {b: float(np.mean(bands_delta[b])) for b in ["low", "mid", "high"]},
        "high_freq_flip_corr": corr,
    }


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

    std_res = analyze_freq(std_model, Xs, Ys, "STD")
    at_res = analyze_freq(at_model, Xs, Ys, "PGD-AT")

    return {"seed": seed, "std": std_res, "at": at_res}


def main():
    t0 = time.time()
    results = [run_seed(s) for s in SEEDS]

    print("=" * 72)
    print("H515 — Frequency-Domain Adversarial Robustness Analysis")
    print("=" * 72)

    for r in results:
        print(f"\n--- Seed {r['seed']} ---")
        for tag in ["std", "at"]:
            d = r[tag]
            print(f"  [{d['label']}] ASR={d['asr']:.3f}")
            print(f"    Original image bands: L={d['orig_band_means']['low']:.3f} "
                  f"M={d['orig_band_means']['mid']:.3f} H={d['orig_band_means']['high']:.3f}")
            print(f"    Perturbation bands:   L={d['pert_band_means']['low']:.3f} "
                  f"M={d['pert_band_means']['mid']:.3f} H={d['pert_band_means']['high']:.3f}")
            print(f"    High-freq ↔ flip corr: {d['high_freq_flip_corr']:.3f}")

    # Cross-seed summary
    print("\n" + "=" * 72)
    print("CROSS-SEED SUMMARY")
    print("=" * 72)
    for tag in ["std", "at"]:
        asrs = [r[tag]["asr"] for r in results]
        highs = [r[tag]["pert_band_means"]["high"] for r in results]
        corrs = [r[tag]["high_freq_flip_corr"] for r in results]
        label = results[0][tag]["label"]
        print(f"  [{label}] ASR={np.mean(asrs):.3f}±{np.std(asrs):.3f}  "
              f"Pert-High={np.mean(highs):.3f}±{np.std(highs):.3f}  "
              f"HighFreq-Flip-Corr={np.nanmean(corrs):.3f}")

    # Key comparison
    std_high = np.mean([r["std"]["pert_band_means"]["high"] for r in results])
    at_high = np.mean([r["at"]["pert_band_means"]["high"] for r in results])
    print(f"\n  Perturbation high-freq energy: STD={std_high:.3f} vs AT={at_high:.3f}")
    if at_high > std_high:
        print("  → AT forces perturbations to higher frequencies (attacker must work harder)")
    else:
        print("  → AT does NOT push perturbations to higher frequencies")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
