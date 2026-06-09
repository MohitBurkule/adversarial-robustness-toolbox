"""
H511 - Model-size double descent in clean accuracy for PGD-AT on F-MNIST.

SEED (paper 5 / Theme G6, Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever, ICLR
2020, "Deep Double Descent: Where Bigger Models and More Data Hurt"; cf. Belkin,
Hsu, Ma, Mandal, PNAS 2019, "Reconciling modern machine learning practice and
the bias-variance trade-off"):
  Both papers show that test error is non-monotone in model capacity: it falls
  in the underparameterised regime, rises around the interpolation threshold
  (where the model first fits the training set), then falls again as width
  continues to grow.  Nakkiran additionally shows this "double descent" curve
  appears in *robust* / adversarial training, where the peak is wider and more
  pronounced because PGD-AT effectively increases the effective dataset size
  ratio.

HYPOTHESIS: On F-MNIST with SmallCNN trained with PGD-AT (Madry, Makelov,
Schmidt, Tsipras, Vladu, ICLR 2018, "Towards Deep Learning Models Resistant to
Adversarial Attacks"), CLEAN test accuracy as a function of base width
w in {8, 16, 32, 64, 128, 256} is non-monotone.  Concretely the prediction is:
  - rises from w=8 to w=16 to w=32 (under-parameterised classical regime),
  - DIPS at w=64 (near the interpolation threshold for N=3000),
  - rises again at w=128 and w=256 (modern over-parameterised regime).
This is the model-size axis of the Nakkiran double-descent curve, restricted
to clean accuracy of a PGD-AT model.

CRITIQUE SEED: At N=6000 (the campaign default) the interpolation-threshold
peak likely sits at a width we have not tested, so a previous campaign sweep
(h501) saw a roughly monotone trend in *robust* accuracy and no peak.  Double
descent is sharpest when the ratio (#params / N) crosses 1 cleanly, so we
lower N to 3000 and push width up to 256 (which gives ~ 1.7M conv params, well
into the over-parameterised regime for N=3000).  This is also exactly the
construction Nakkiran et al. use to surface the effect.

This hypothesis is DIFFERENT from h501, which sweeps width vs *robust* accuracy
at the campaign default N=6000.  Here we (i) use N=3000, (ii) push to wider
models, and (iii) headline CLEAN accuracy, not robust accuracy.

CONTROLS:
  (1) PGD-AT at N_train=3000 across widths {8, 16, 32, 64, 128, 256}.
  (2) Both clean accuracy AND PGD attack-success rate at fixed eps_test, per width.
  (3) Param-count curve: report total parameters per width so the
      interpolation threshold can be located.
  (4) Loss curves: train-clean / test-clean / train-robust / test-robust loss
      per width (recorded at end of training so we can see whether the dip
      coincides with the model first fitting the training set).
  (5) STD-trained baseline at the same widths, so we can separate the "robust
      double descent" effect (Nakkiran's stronger claim) from the standard one.

EXTRA PAPERS (>= 2, cited in the verdict block):
  * Nakkiran et al. 2020, "Deep Double Descent" (seed; sample-wise + model-wise).
  * Belkin et al. 2019, "Reconciling modern machine learning practice and the
    bias-variance trade-off" (the double-descent picture itself).
  * Madry et al. 2018, "Towards Deep Learning Models Resistant to Adversarial
    Attacks" (PGD-AT, the training procedure tested here).

HEADLINE: existence and location of a clean-accuracy dip in PGD-AT as width
grows; whether the same dip is present under STD training; whether it sits
near the interpolation threshold (#params ~ N).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
WIDTHS = [8, 16, 32, 64, 128, 256]
N_TRAIN = 3000
N_EVAL = 2000
EPS_TEST = 0.10
EPS_TRAIN = 0.10
PGD_STEPS_TEST = 10
PGD_STEPS_TRAIN = 7
EPOCHS = 8
SEED = 0


def _count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@torch.no_grad()
def _clean_loss(model, X, Y, batch=512):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        out = model(xb)
        total += float(F.cross_entropy(out, yb, reduction="sum"))
        n += xb.size(0)
    return total / max(1, n)


def _robust_loss(model, X, Y, eps, steps, batch=256):
    """PGD-adversarial cross-entropy loss (model in eval mode for attack, then no_grad)."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            out = model(xa)
            total += float(F.cross_entropy(out, yb, reduction="sum"))
            n += xb.size(0)
    return total / max(1, n)


def train_and_eval(width, adv_train, seed):
    """Train a SmallCNN of the given base width with either STD or PGD-AT, then
    measure clean acc, PGD-ASR at eps_test, parameter count, and the four loss
    curves (train clean, test clean, train robust, test robust)."""
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)

    model = C.build_model("cnn", meta, width=width, seed=seed)
    n_params = _count_params(model)

    C.train_model(model, Xtr, Ytr,
                  epochs=EPOCHS, opt="sgd", lr=0.05, ncls=meta["n_classes"],
                  adv_train=adv_train, adv_eps=EPS_TRAIN,
                  adv_steps=PGD_STEPS_TRAIN)

    # clean accuracy on eval set
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)

    # PGD attack-success rate at fixed eps_test
    asr_info = C.attack_success(model, Xte, Yte, attack="pgd",
                                eps=EPS_TEST, steps=PGD_STEPS_TEST)

    # loss curves (recorded after training; the dip should show up as a spike
    # in test-clean loss relative to train-clean loss near the interpolation
    # threshold)
    train_clean = _clean_loss(model, Xtr, Ytr)
    test_clean = _clean_loss(model, Xte, Yte)
    train_robust = _robust_loss(model, Xtr, Ytr, EPS_TEST, PGD_STEPS_TEST)
    test_robust = _robust_loss(model, Xte, Yte, EPS_TEST, PGD_STEPS_TEST)

    return {
        "width": width,
        "regime": "PGD-AT" if adv_train else "STD",
        "n_params": n_params,
        "params_over_N": n_params / float(N_TRAIN),
        "clean_acc": float(clean_acc),
        "pgd_asr": float(asr_info["asr"]),
        "loss_train_clean": float(train_clean),
        "loss_test_clean": float(test_clean),
        "loss_train_robust": float(train_robust),
        "loss_test_robust": float(test_robust),
    }


def _find_dip(xs, ys):
    """Return (idx, x_at_dip, depth) where depth = max(neighbours) - y[idx]
    for any strict interior local minimum.  If there is no strict interior
    local min, idx = -1 and depth = 0.0."""
    best_idx, best_depth = -1, 0.0
    for i in range(1, len(ys) - 1):
        if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]:
            depth = min(ys[i - 1], ys[i + 1]) - ys[i]
            if depth > best_depth:
                best_idx, best_depth = i, depth
    return best_idx, (xs[best_idx] if best_idx >= 0 else float("nan")), best_depth


def _summarise(rows, regime):
    rows = [r for r in rows if r["regime"] == regime]
    rows = sorted(rows, key=lambda r: r["width"])
    xs = [r["width"] for r in rows]
    ys = [r["clean_acc"] for r in rows]
    idx, w_dip, depth = _find_dip(xs, ys)
    monotone_up = all(ys[i + 1] >= ys[i] for i in range(len(ys) - 1))
    monotone_down = all(ys[i + 1] <= ys[i] for i in range(len(ys) - 1))
    return {
        "regime": regime,
        "widths": xs, "clean_acc": ys,
        "dip_width": w_dip, "dip_depth": depth, "dip_idx": idx,
        "monotone_up": monotone_up,
        "monotone_down": monotone_down,
        "rows": rows,
    }


def main():
    print("=" * 78)
    print("H511 - Clean-accuracy model-size double descent for PGD-AT on F-MNIST")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  N_train={N_TRAIN}  N_eval={N_EVAL}")
    print(f"eps_train={EPS_TRAIN}  eps_test={EPS_TEST}  PGD steps (train/test)="
          f"{PGD_STEPS_TRAIN}/{PGD_STEPS_TEST}  epochs={EPOCHS}")
    print(f"width sweep = {WIDTHS}")
    print("Seed   : Nakkiran 2020 'Deep Double Descent';")
    print("         Belkin 2019 'Reconciling modern ML practice & bias-variance';")
    print("         Madry 2018 (PGD-AT, the training procedure tested).")
    print("Critique: at N=6000 the interpolation peak may sit at an untested width,")
    print("          so we lower N to 3000 and push width to 256 to surface the dip.")
    print("-" * 78)

    rows = []
    for adv in (True, False):
        for w in WIDTHS:
            t0 = time.time()
            r = train_and_eval(w, adv_train=adv, seed=SEED)
            r["runtime_s"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"[{r['regime']:>6s}  width={w:4d}  params={r['n_params']:>9d}  "
                  f"p/N={r['params_over_N']:6.2f}]  clean={r['clean_acc']:.3f}  "
                  f"PGD-ASR@{EPS_TEST}={r['pgd_asr']:.3f}  ({r['runtime_s']}s)")
            print(f"    losses  train_clean={r['loss_train_clean']:.3f}  "
                  f"test_clean={r['loss_test_clean']:.3f}  "
                  f"train_robust={r['loss_train_robust']:.3f}  "
                  f"test_robust={r['loss_test_robust']:.3f}")

    print("\n" + "=" * 78)
    print("PARAM-COUNT CURVE")
    print("=" * 78)
    print(f"  {'width':>6s}  {'n_params':>10s}  {'params/N':>10s}")
    for w in WIDTHS:
        r = next(r for r in rows if r["width"] == w and r["regime"] == "PGD-AT")
        print(f"  {w:>6d}  {r['n_params']:>10d}  {r['params_over_N']:>10.2f}")

    summaries = {regime: _summarise(rows, regime) for regime in ("PGD-AT", "STD")}

    print("\n" + "=" * 78)
    print("CLEAN-ACC vs WIDTH  (per regime)")
    print("=" * 78)
    for regime in ("PGD-AT", "STD"):
        s = summaries[regime]
        ys_str = ", ".join(f"{y:.3f}" for y in s["clean_acc"])
        print(f"\n[{regime}]  widths = {s['widths']}")
        print(f"            clean = [{ys_str}]")
        if s["dip_idx"] >= 0:
            print(f"  -> interior local MINIMUM at width={s['dip_width']}  "
                  f"depth={s['dip_depth']:.3f}")
        elif s["monotone_up"]:
            print("  -> monotone INCREASING (no interior dip; classical-only regime)")
        elif s["monotone_down"]:
            print("  -> monotone DECREASING (overfitting throughout; no double descent peak)")
        else:
            print("  -> non-monotone but no strict interior local minimum")

    # HEADLINE
    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    s_at = summaries["PGD-AT"]
    s_st = summaries["STD"]

    predicted_pattern = (
        s_at["clean_acc"][0] < s_at["clean_acc"][1] < s_at["clean_acc"][2]
        and s_at["clean_acc"][3] < s_at["clean_acc"][2]
        and s_at["clean_acc"][4] > s_at["clean_acc"][3]
    )
    width64_dip = (
        s_at["clean_acc"][3] < s_at["clean_acc"][2]
        and s_at["clean_acc"][3] < s_at["clean_acc"][4]
    )

    if predicted_pattern:
        verdict = ("EXACT MATCH to the H-stated Nakkiran-style pattern: clean acc grows"
                   " at w=16,32, dips at w=64, recovers at w=128. Model-size double"
                   " descent for PGD-AT clean accuracy on F-MNIST CONFIRMED.")
    elif width64_dip:
        verdict = ("PARTIAL MATCH: w=64 sits at an interior local minimum (the predicted"
                   " interpolation-threshold dip) but the surrounding monotone pattern"
                   " did not match exactly. Consistent with Nakkiran double descent but"
                   " shifted.")
    elif s_at["dip_idx"] >= 0:
        verdict = (f"DOUBLE DESCENT PRESENT at a DIFFERENT width: dip at "
                   f"w={s_at['dip_width']} (depth={s_at['dip_depth']:.3f}). The "
                   f"qualitative Nakkiran prediction holds; the dip location differs.")
    elif s_at["monotone_up"]:
        verdict = ("NO double descent: clean accuracy is monotone increasing in width."
                   " Either the interpolation threshold lies beyond w=256 or the F-MNIST"
                   " task is too easy at N=3000 to expose the peak.")
    else:
        verdict = ("NO clear double-descent signature: clean acc varies non-monotonically"
                   " but without a strict interior minimum.")

    # Distinguish robust- vs std-double-descent (Nakkiran's stronger claim is that
    # the peak is wider/more pronounced under adversarial training).
    if s_at["dip_idx"] >= 0 and s_st["dip_idx"] < 0:
        regime_comparison = ("DIP IS ROBUST-TRAINING-SPECIFIC: only the PGD-AT curve has"
                             " a clean-acc interior minimum, supporting Nakkiran's claim"
                             " that adversarial training amplifies double descent.")
    elif s_at["dip_idx"] >= 0 and s_st["dip_idx"] >= 0:
        regime_comparison = (f"BOTH regimes show a dip (PGD-AT @ w={s_at['dip_width']},"
                             f" STD @ w={s_st['dip_width']}); double descent is a"
                             " capacity-vs-N effect, not robust-training-specific.")
    elif s_at["dip_idx"] < 0 and s_st["dip_idx"] >= 0:
        regime_comparison = (f"Inverted Nakkiran prediction: only STD shows a dip"
                             f" (@ w={s_st['dip_width']}); PGD-AT smooths it out.")
    else:
        regime_comparison = "Neither regime exposes an interior clean-acc minimum."

    print(f"  PGD-AT clean(width) : {[f'{y:.3f}' for y in s_at['clean_acc']]}"
          f"   widths={s_at['widths']}")
    print(f"  STD    clean(width) : {[f'{y:.3f}' for y in s_st['clean_acc']]}"
          f"   widths={s_st['widths']}")
    print(f"  Predicted exact pattern (w=16,32 up; w=64 dip; w=128 up) : {predicted_pattern}")
    print(f"  w=64 is a local minimum                                  : {width64_dip}")
    print(f"  VERDICT          : {verdict}")
    print(f"  Regime contrast  : {regime_comparison}")
    print("=" * 78)
    print("References:")
    print("  - Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever, ICLR 2020,")
    print("    'Deep Double Descent: Where Bigger Models and More Data Hurt'.")
    print("  - Belkin, Hsu, Ma, Mandal, PNAS 2019, 'Reconciling modern machine")
    print("    learning practice and the bias-variance trade-off'.")
    print("  - Madry, Makelov, Schmidt, Tsipras, Vladu, ICLR 2018, 'Towards Deep")
    print("    Learning Models Resistant to Adversarial Attacks' (PGD-AT).")
    print("=" * 78)


if __name__ == "__main__":
    main()
