"""
H500 - Tsipras et al. 2019: "Robustness May Be at Odds with Accuracy" - on F-MNIST.

SEED (paper 5 / Theme G6, Tsipras, Santurkar, Engstrom, Turner, Madry, ICLR 2019,
"Robustness May Be at Odds with Accuracy"):
  In a synthetic Gaussian-mixture setting they construct a distribution where a
  Bayes-optimal accurate classifier is provably non-robust; PGD adversarial
  training therefore *must* trade clean accuracy for robustness. Empirically on
  MNIST/CIFAR they show clean accuracy drops monotonically with the training
  perturbation budget eps_train. The headline quantity is the slope of clean-acc
  versus eps_train.

HYPOTHESIS: On F-MNIST with SmallCNN, training with PGD-AT at
  eps_train in {0.0, 0.05, 0.10, 0.15, 0.20, 0.30}
clean accuracy at eps_test = 0.0 is monotone decreasing in eps_train. The linear
regression slope d(clean-acc)/d(eps_train) is between roughly -0.3 and -0.5
("accuracy-drop per unit eps") at N_train = 6000. A perfectly linear
trade-off would also be consistent with a quadratic fit having a small
second-order coefficient.

CRITIQUE SEED: Tsipras' theoretical example is *constructed* (a designed
Gaussian-mixture where robust+accurate is provably impossible). It does not
imply that real image data has this property. Two important rebuttals are:

  * Yang, Rashtchian, Zhang, Salakhutdinov, Chaudhuri, NeurIPS 2020,
    "A Closer Look at Accuracy vs. Robustness": they show CIFAR/MNIST/etc are
    r-separated for the relevant epsilon, so a perfectly robust + perfectly
    accurate classifier *exists in principle*; observed trade-offs therefore
    reflect model class / optimisation, not an information-theoretic barrier.
  * Raghunathan, Xie, Yang, Duchi, Liang, ICML 2020, "Understanding and
    Mitigating the Tradeoff Between Robustness and Accuracy": adversarial
    training can hurt generalisation even when the underlying clean-vs-robust
    trade-off is mild; what we measure is partly an optimiser/regulariser
    effect.
  * Schmidt, Santurkar, Tsipras, Talwar, Madry, NeurIPS 2018, "Adversarially
    Robust Generalization Requires More Data": robust training is sample-hungry,
    so the apparent trade-off may shrink with more training data.

Therefore the F-MNIST trade-off here could be MILD, ABSENT, or sample-size
dependent; we document the observed slope, fit linear *and* quadratic, look at
per-class clean accuracy (do some classes GAIN clean acc as eps_train rises?),
and repeat at N=2000 and N=6000 to test the Schmidt prediction.

CONTROLS:
  (1) PGD-AT at eps_train in {0.0, 0.05, 0.10, 0.15, 0.20, 0.30}.
  (2) Measure clean accuracy and PGD attack-success-rate at fixed
      eps_test = 0.10 for each trained model.
  (3) Fit clean-acc(eps_train) with linear and quadratic regression; report
      slope, intercept, quadratic curvature, R^2.
  (4) Per-class clean accuracy across eps_train: does any class GAIN clean
      accuracy as eps_train grows?
  (5) Repeat the whole sweep at N_train = 2000 and N_train = 6000 to test
      whether the trade-off shrinks with more data (Schmidt 2018 prediction).
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
EPS_TRAINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
N_TRAINS = [2000, 6000]
EPS_TEST = 0.10
PGD_STEPS_TEST = 10
PGD_STEPS_TRAIN = 7
SEED = 0
EPOCHS = 6
N_EVAL = 2000


def _linfit(xs, ys):
    """Return (slope, intercept, r2) for a simple linear fit y = a*x + b."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < 2:
        return float("nan"), float("nan"), float("nan")
    a, b = np.polyfit(xs, ys, 1)
    yhat = a * xs + b
    ss_res = float(((ys - yhat) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), r2


def _quadfit(xs, ys):
    """Return (a2, a1, a0, r2) for y = a2 x^2 + a1 x + a0."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    a2, a1, a0 = np.polyfit(xs, ys, 2)
    yhat = a2 * xs ** 2 + a1 * xs + a0
    ss_res = float(((ys - yhat) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a2), float(a1), float(a0), r2


def per_class_acc(model, X, Y, ncls):
    """Return per-class clean accuracy array, length ncls."""
    lg, _ = C.logits_and_acc(model, X, Y)
    pred = lg.argmax(1).cpu().numpy()
    Y_np = Y.cpu().numpy()
    out = np.full(ncls, float("nan"))
    for c in range(ncls):
        m = Y_np == c
        if m.sum() > 0:
            out[c] = float((pred[m] == c).mean())
    return out


def train_and_eval(eps_train, n_train, seed):
    """Train one PGD-AT model at the given eps_train and measure clean acc,
    PGD-ASR @ eps_test, and per-class clean acc on a fixed eval set."""
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=n_train, n_eval=N_EVAL, seed=seed)

    model = C.build_model("cnn", meta, seed=seed)
    adv_train = eps_train > 0.0
    C.train_model(model, Xtr, Ytr,
                  epochs=EPOCHS, opt="sgd", lr=0.05, ncls=meta["n_classes"],
                  adv_train=adv_train, adv_eps=eps_train,
                  adv_steps=PGD_STEPS_TRAIN)

    # clean acc on the eval set
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)

    # PGD attack success rate at fixed eps_test (on originally-correct samples,
    # matches the rest of the campaign).
    asr_info = C.attack_success(model, Xte, Yte, attack="pgd",
                                eps=EPS_TEST, steps=PGD_STEPS_TEST)
    pgd_asr = asr_info["asr"]

    # per-class clean acc
    pc = per_class_acc(model, Xte, Yte, meta["n_classes"])

    return {
        "eps_train": eps_train,
        "n_train": n_train,
        "clean_acc": float(clean_acc),
        "pgd_asr": float(pgd_asr),
        "per_class_acc": pc.tolist(),
    }


def analyse_sweep(rows, n_train):
    """Compute slope / quadratic curvature / monotonicity for one N_train sweep."""
    rows = [r for r in rows if r["n_train"] == n_train]
    rows = sorted(rows, key=lambda r: r["eps_train"])
    xs = [r["eps_train"] for r in rows]
    ys = [r["clean_acc"] for r in rows]
    slope, intercept, r2 = _linfit(xs, ys)
    a2, a1, a0, r2q = _quadfit(xs, ys)
    # monotonicity: count non-increases between consecutive eps_train levels
    diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    n_drops = sum(1 for d in diffs if d <= 0)
    monotone_frac = n_drops / max(1, len(diffs))
    drop_total = ys[0] - ys[-1]
    # per-class GAIN: any class whose acc at max eps_train > acc at eps_train=0
    pc0 = np.asarray(rows[0]["per_class_acc"])
    pcL = np.asarray(rows[-1]["per_class_acc"])
    gainers = [c for c in range(len(pc0)) if pcL[c] > pc0[c]]
    return {
        "n_train": n_train,
        "xs": xs, "ys": ys,
        "slope": slope, "intercept": intercept, "r2_lin": r2,
        "quad_a2": a2, "quad_a1": a1, "quad_a0": a0, "r2_quad": r2q,
        "monotone_frac": monotone_frac,
        "drop_total": drop_total,
        "gain_classes": gainers,
        "per_class_0": pc0.tolist(),
        "per_class_max": pcL.tolist(),
    }


def main():
    print("=" * 78)
    print("H500 - Tsipras et al. 2019 robustness/accuracy trade-off on F-MNIST")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  eps_test={EPS_TEST}")
    print(f"eps_train sweep = {EPS_TRAINS}")
    print(f"N_train sweep   = {N_TRAINS}  (Schmidt-2018 sample-complexity test)")
    print("Seed: Tsipras et al. 2019 'Robustness May Be at Odds with Accuracy'.")
    print("Critique: Yang 2020 (r-separation), Raghunathan 2020 (AT hurts gen),")
    print("          Schmidt 2018 (robust generalisation needs more data).")
    print("-" * 78)

    rows = []
    for n_train in N_TRAINS:
        for eps_train in EPS_TRAINS:
            t0 = time.time()
            r = train_and_eval(eps_train, n_train, SEED)
            r["runtime_s"] = round(time.time() - t0, 1)
            rows.append(r)
            pc_str = ", ".join(f"{v:.2f}" for v in r["per_class_acc"])
            print(f"[N={n_train:5d}  eps_train={eps_train:.2f}]  "
                  f"clean={r['clean_acc']:.3f}  PGD-ASR@{EPS_TEST}={r['pgd_asr']:.3f}  "
                  f"({r['runtime_s']}s)")
            print(f"    per-class clean acc: [{pc_str}]")

    print("\n" + "=" * 78)
    print("TRADE-OFF FITS  (clean_acc vs eps_train)")
    print("=" * 78)
    summaries = []
    for n_train in N_TRAINS:
        s = analyse_sweep(rows, n_train)
        summaries.append(s)
        print(f"\nN_train = {n_train}")
        print(f"  eps_train       : {['%.2f' % x for x in s['xs']]}")
        print(f"  clean_acc       : {['%.3f' % y for y in s['ys']]}")
        print(f"  linear fit      : slope={s['slope']:+.3f}  intercept={s['intercept']:.3f}  R^2={s['r2_lin']:.3f}")
        print(f"  quadratic fit   : a2={s['quad_a2']:+.3f}  a1={s['quad_a1']:+.3f}  a0={s['quad_a0']:.3f}  R^2={s['r2_quad']:.3f}")
        print(f"  drop total      : {s['drop_total']:+.3f}  (acc@eps0 - acc@eps_max)")
        print(f"  monotone frac   : {s['monotone_frac']:.2f}  (frac of consecutive non-increases)")
        print(f"  classes GAINING : {s['gain_classes']}  (per-class clean acc higher at eps_max than at eps=0)")

    # Sample-complexity comparison.
    if len(summaries) == 2:
        a, b = summaries  # N=2000, N=6000
        d_slope = b["slope"] - a["slope"]
        d_drop = b["drop_total"] - a["drop_total"]
        print("\n" + "-" * 78)
        print("Schmidt 2018 prediction: trade-off should SHRINK as N grows.")
        print(f"  slope    : N={a['n_train']}: {a['slope']:+.3f}    "
              f"N={b['n_train']}: {b['slope']:+.3f}    delta = {d_slope:+.3f}")
        print(f"  drop     : N={a['n_train']}: {a['drop_total']:+.3f}    "
              f"N={b['n_train']}: {b['drop_total']:+.3f}    delta = {d_drop:+.3f}")

    # HEADLINE
    main_summary = next((s for s in summaries if s["n_train"] == 6000), summaries[-1])
    slope = main_summary["slope"]
    drop = main_summary["drop_total"]
    monotone = main_summary["monotone_frac"]
    gainers = main_summary["gain_classes"]

    in_predicted = -0.5 <= slope <= -0.3
    strong_tradeoff = slope < -0.3
    mild_tradeoff = -0.3 <= slope < -0.1
    absent_tradeoff = slope >= -0.1

    if strong_tradeoff and monotone >= 0.8:
        verdict = "STRONG Tsipras-style trade-off observed on F-MNIST."
    elif strong_tradeoff and monotone < 0.8:
        verdict = "Strong slope but non-monotone: trade-off real but noisy."
    elif mild_tradeoff:
        verdict = "MILD trade-off on F-MNIST (closer to Yang 2020 'r-separated' regime)."
    elif absent_tradeoff:
        verdict = "NO meaningful trade-off on F-MNIST (Yang 2020 supported)."
    else:
        verdict = "Indeterminate trade-off shape."

    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(f"  N=6000 PGD-AT clean-acc(eps_train) slope = {slope:+.3f}    "
          f"(Tsipras prediction: ~ -0.3 to -0.5; in range: {in_predicted})")
    print(f"  total clean-acc drop across eps_train sweep = {drop:+.3f}")
    print(f"  monotone fraction = {monotone:.2f}  ({'monotone' if monotone == 1.0 else 'NOT strictly monotone'})")
    print(f"  classes that GAIN clean acc at eps_max = {gainers}  "
          f"({'none -> uniform cost' if not gainers else 'some classes BENEFIT from AT'})")
    print(f"  VERDICT: {verdict}")
    print("=" * 78)
    print("Note (critique seed): a mild/absent slope here is CONSISTENT with Yang")
    print("2020's r-separation argument that F-MNIST may admit accurate+robust")
    print("classifiers, and with Raghunathan 2020's view that the trade-off we DO")
    print("see is partly an optimisation/generalisation artefact. The N=2000 vs")
    print("N=6000 comparison probes the Schmidt 2018 sample-complexity prediction.")
    print("=" * 78)


if __name__ == "__main__":
    main()
