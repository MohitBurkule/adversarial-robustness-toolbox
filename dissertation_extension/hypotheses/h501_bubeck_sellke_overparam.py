"""
H501 - Bubeck-Sellke overparameterization: does doubling SmallCNN width yield
       a measurable robust-acc gain on Fashion-MNIST at N=6000?

Seed papers
-----------
  * Bubeck & Sellke (NeurIPS 2021)  "A Universal Law of Robustness via
    Isoperimetry."  Proves that for a generic, sufficiently-smooth interpolating
    classifier on N samples of input dimension d to be O(1)-Lipschitz (i.e.
    adversarially robust at eps=O(1)), the model must contain
        p  >=  N * d / eps^2
    parameters (up to log factors).  In particular, for fixed N and d, robust
    interpolation REQUIRES overparameterization in p.  For Fashion-MNIST
    (d=784) with N=6000 and eps=0.1 this gives a back-of-envelope floor
        p_min  ~  6000 * 784 / 0.01  =  4.7e8 params,
    which a width-32 SmallCNN (~0.3M params) is 1500x short of.  The theorem is
    asymptotic & for shallow one-hidden-layer ReLU; we are therefore testing
    only the qualitative consequence: more parameters -> better robust acc, on
    a finite conv net well inside the underparameterized regime.

  * Madry, Makelov, Schmidt, Tsipras, Vladu (ICLR 2018)  "Towards Deep Learning
    Models Resistant to Adversarial Attacks."  Empirical companion piece: shows
    that capacity (width / depth) is required to *fit* PGD-AT loss at all, and
    that ResNet width 1->4 monotonically raises robust acc on CIFAR-10.  This
    is the practical reading of the Bubeck-Sellke law we actually test here.

  * Wu, Xia, Wang (NeurIPS 2020)  "Adversarial Weight Perturbation Helps Robust
    Generalization."  Sometimes summarized as "wider networks don't help
    [robust generalization beyond a point]" -- their finding is that the robust
    generalization gap *grows* with width even as robust train acc improves, so
    the width->robustness curve can be non-monotonic / saturating once the model
    overfits the AT loss.  Useful counter-control: if our widest model has the
    best PGD-AT *train* loss but a *worse* test robust acc than width 64, that
    would match Wu et al.'s narrative.

  * Pang, Xu, Dong, Su, Zhu (ICLR 2020)  "Rethinking Softmax Cross-Entropy Loss
    for Adversarial Robustness."  Shows that the standard CE used in PGD-AT
    leaves robustness on the table relative to margin-shaping losses; cited
    here as a reminder that *width* is one axis (Bubeck-Sellke) and *loss
    shape* is an orthogonal one -- we keep loss fixed (CE) to isolate the width
    effect.

Hypothesis
----------
Holding everything else fixed (PGD-AT, eps=0.1, 7-step PGD, CE loss, SGD, same
data, same epochs), doubling the SmallCNN base width through
{16, 32, 64, 128} yields a *monotonic* increase in PGD robust accuracy on
Fashion-MNIST test, with at least one doubling step delivering >=2pp absolute
robust-acc gain.  This is the directional consequence of Bubeck-Sellke /
Madry-2018 even though we are nowhere near the asymptotic regime.

Critique-driven caveats
-----------------------
  * Bubeck-Sellke is proven for one-hidden-layer ReLU networks under a
    Lipschitz-interpolation hypothesis; SmallCNN is a tiny 3-block conv net
    with batchnorm and is *not* in that regime.  We are therefore testing the
    folklore "more params -> more robust" claim that Madry-2018 and the
    Wide-ResNet literature popularised, with Bubeck-Sellke as the theoretical
    seed -- not a faithful empirical instantiation of the theorem.
  * Wu et al. predict the curve can saturate or reverse; we explicitly look
    for that with a loss-vs-width control.
  * At N=6000 we are also far from interpolation, so the theorem's "robust
    *interpolation*" precondition is not met.  The hypothesis is therefore
    one-sided (gain) but allows saturation as a partial outcome.

Controls / pipeline
-------------------
  (1) Train PGD-AT SmallCNN at widths W in {16, 32, 64, 128}.  Same seed, same
      data, same epochs, same optimizer, same eps/steps.
  (2) Per width: parameter count (M), clean acc, PGD-ASR, PGD robust acc,
      mean robust margin on the test set.
  (3) Compute "robust efficiency" = robust_acc per million parameters.  Tells
      us whether the gain (if any) is worth the capacity.
  (4) Per-class breakdown of robust acc at the widest model (W=128) -- which
      Fashion-MNIST classes (shirt vs T-shirt vs pullover) absorb the gain?
  (5) Loss-vs-width curve: final-epoch PGD-AT *training* loss + final clean +
      final adversarial test loss, to check for overfitting (Wu-style).

Verdict logic
-------------
  SUPPORTED if robust_acc(W) is monotonic non-decreasing in W AND
                max_W robust_acc - min_W robust_acc >= 0.02.
  PARTIAL   if there is a net gain (>=0.02) but the curve is non-monotonic
                (saturation or reversal -- Wu-style overfitting plausible).
  REFUTED   if the spread is <0.02 (width simply does not matter at this
                budget) OR if robust acc *decreases* monotonically with width.

NOTE: This file is intentionally NOT executed here -- it is queued for the
background runner.  All compute happens on the campaign worker.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C


DS = "fashion_mnist"
SEED = 0
N_TRAIN = 6000
N_EVAL = 2000
WIDTHS = [16, 32, 64, 128]
AT_EPOCHS = 8
EPS = 0.1
PGD_STEPS = 7


def _param_count(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@torch.no_grad()
def _per_class_acc(model, X, Y, n_classes, batch=256):
    """Return per-class accuracy array of length n_classes."""
    model.eval()
    preds = []
    for i in range(0, X.size(0), batch):
        preds.append(model(X[i:i + batch]).argmax(1).cpu())
    preds = torch.cat(preds)
    y = Y.cpu()
    out = np.zeros(n_classes)
    for c in range(n_classes):
        m = (y == c)
        out[c] = float((preds[m] == c).float().mean()) if m.sum() > 0 else float("nan")
    return out


def _per_class_robust_acc(model, X, Y, n_classes, eps=EPS, steps=PGD_STEPS, batch=256):
    """Per-class fraction-correct under PGD attack."""
    model.eval()
    preds_adv = []
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            preds_adv.append(model(xa).argmax(1).cpu())
    preds_adv = torch.cat(preds_adv)
    y = Y.cpu()
    out = np.zeros(n_classes)
    for c in range(n_classes):
        m = (y == c)
        out[c] = float((preds_adv[m] == c).float().mean()) if m.sum() > 0 else float("nan")
    return out


def _train_loss_and_test_losses(model, Xtr, Ytr, Xte, Yte, eps=EPS, steps=PGD_STEPS, batch=256):
    """Final-state PGD-AT-style train loss (on adv examples) + clean & adv test loss."""
    import torch.nn.functional as F
    model.eval()

    # train adversarial loss (mirrors PGD-AT objective)
    losses = []
    for i in range(0, Xtr.size(0), batch):
        xb, yb = Xtr[i:i + batch], Ytr[i:i + batch]
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            losses.append(F.cross_entropy(model(xa), yb, reduction="sum").item())
    adv_train_loss = sum(losses) / Xtr.size(0)

    # clean test loss
    losses = []
    with torch.no_grad():
        for i in range(0, Xte.size(0), batch):
            xb, yb = Xte[i:i + batch], Yte[i:i + batch]
            losses.append(F.cross_entropy(model(xb), yb, reduction="sum").item())
    clean_test_loss = sum(losses) / Xte.size(0)

    # adv test loss
    losses = []
    for i in range(0, Xte.size(0), batch):
        xb, yb = Xte[i:i + batch], Yte[i:i + batch]
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            losses.append(F.cross_entropy(model(xa), yb, reduction="sum").item())
    adv_test_loss = sum(losses) / Xte.size(0)

    return adv_train_loss, clean_test_loss, adv_test_loss


def run_width(width, Xtr, Ytr, Xte, Yte, meta, seed=SEED):
    """Train PGD-AT SmallCNN at given width and collect metrics."""
    C.set_seed(seed)
    m = C.build_model("cnn", meta, width=width, seed=seed)
    n_params = _param_count(m)

    t0 = time.time()
    C.train_model(m, Xtr, Ytr, epochs=AT_EPOCHS, opt="sgd", lr=0.05,
                  ncls=meta["n_classes"],
                  adv_train=True, adv_eps=EPS, adv_steps=PGD_STEPS)
    train_s = time.time() - t0

    # clean test acc
    _, clean_acc = C.logits_and_acc(m, Xte, Yte)

    # PGD attack on test
    res = C.attack_success(m, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    rob_correct = int((~res["flips"] & res["correct"]).sum())
    robust_acc = rob_correct / Xte.size(0)

    # mean robust margin on test
    rob_margins = []
    m.eval()
    for i in range(0, Xte.size(0), 256):
        xb, yb = Xte[i:i + 256], Yte[i:i + 256]
        xa = C.pgd(m, xb, yb, eps=EPS, steps=PGD_STEPS)
        with torch.no_grad():
            lg = m(xa).cpu()
        rob_margins.append(C.margin_of(lg, yb))
    mean_rob_margin = float(np.concatenate(rob_margins).mean())

    # robust efficiency = robust acc per million params
    rob_eff = robust_acc / (n_params / 1e6) if n_params > 0 else float("nan")

    row = {
        "width": width,
        "n_params": n_params,
        "params_M": n_params / 1e6,
        "clean_acc": clean_acc,
        "pgd_asr": res["asr"],
        "robust_acc": robust_acc,
        "mean_robust_margin": mean_rob_margin,
        "robust_eff_per_Mparam": rob_eff,
        "train_s": round(train_s, 1),
    }
    print(f"  [W={width:>3}] params={n_params/1e6:.3f}M  "
          f"clean={clean_acc:.3f}  asr={res['asr']:.3f}  "
          f"robust={robust_acc:.3f}  rob_margin={mean_rob_margin:+.2f}  "
          f"eff={rob_eff:.3f}/M  ({train_s:.1f}s)")
    return m, row


def main():
    print("=" * 78)
    print("H501 - Bubeck-Sellke overparam: width vs robust acc on F-MNIST SmallCNN")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  N_train={N_TRAIN}  N_eval={N_EVAL}")
    print(f"widths={WIDTHS}  at_epochs={AT_EPOCHS}  eps={EPS}  pgd_steps={PGD_STEPS}")

    # Bubeck-Sellke back-of-envelope param floor for the chosen (N, d, eps)
    d_in = 28 * 28
    p_min_BS = N_TRAIN * d_in / (EPS ** 2)
    print(f"Bubeck-Sellke floor p_min ~ N*d/eps^2 = "
          f"{N_TRAIN}*{d_in}/{EPS**2:.3f} = {p_min_BS:.2e} params")
    print("(SmallCNN at all tested widths is FAR below this floor; we test only")
    print(" the qualitative folklore 'wider -> more robust', not the theorem.)")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # ------------------------------------------------------------------------
    # (1+2+3) PGD-AT at each width with full metric battery
    # ------------------------------------------------------------------------
    print("\n[1-3] PGD-AT sweep over widths ...")
    rows = []
    widest_model = None
    for w in WIDTHS:
        m, r = run_width(w, Xtr, Ytr, Xte, Yte, meta, seed=SEED)
        rows.append(r)
        if w == max(WIDTHS):
            widest_model = m
        else:
            del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------------
    # (4) per-class breakdown at widest model
    # ------------------------------------------------------------------------
    FASHION_CLASSES = ["Tshirt", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankleboot"]
    print(f"\n[4] Per-class breakdown @ widest W={max(WIDTHS)} ...")
    pc_clean = _per_class_acc(widest_model, Xte, Yte, meta["n_classes"])
    pc_rob   = _per_class_robust_acc(widest_model, Xte, Yte, meta["n_classes"],
                                     eps=EPS, steps=PGD_STEPS)
    print(f"  {'class':<10} {'clean':>7} {'robust':>7}")
    for c in range(meta["n_classes"]):
        name = FASHION_CLASSES[c] if c < len(FASHION_CLASSES) else f"c{c}"
        print(f"  {name:<10} {pc_clean[c]:>7.3f} {pc_rob[c]:>7.3f}")

    # ------------------------------------------------------------------------
    # (5) loss-vs-width: train adv loss, test clean loss, test adv loss
    # ------------------------------------------------------------------------
    print("\n[5] Loss-vs-width overfitting check (re-train one model per width "
          "with same seed for losses; widest re-used from above) ...")
    loss_rows = []
    for w, base_row in zip(WIDTHS, rows):
        if w == max(WIDTHS):
            m = widest_model
        else:
            C.set_seed(SEED)
            m = C.build_model("cnn", meta, width=w, seed=SEED)
            C.train_model(m, Xtr, Ytr, epochs=AT_EPOCHS, opt="sgd", lr=0.05,
                          ncls=meta["n_classes"],
                          adv_train=True, adv_eps=EPS, adv_steps=PGD_STEPS)
        atl, cte, ate = _train_loss_and_test_losses(m, Xtr, Ytr, Xte, Yte,
                                                    eps=EPS, steps=PGD_STEPS)
        gap = ate - atl   # adv test loss - adv train loss (Wu robust-gen gap)
        loss_rows.append({"width": w, "adv_train_loss": atl,
                          "clean_test_loss": cte, "adv_test_loss": ate,
                          "robust_gen_gap": gap})
        print(f"  [W={w:>3}] adv_train_loss={atl:.3f}  clean_test_loss={cte:.3f}  "
              f"adv_test_loss={ate:.3f}  robust_gen_gap={gap:+.3f}")
        if w != max(WIDTHS):
            del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------------
    # HEADLINE
    # ------------------------------------------------------------------------
    rob_accs = [r["robust_acc"] for r in rows]
    clean_accs = [r["clean_acc"] for r in rows]
    spread = max(rob_accs) - min(rob_accs)
    monotonic = all(rob_accs[i] <= rob_accs[i + 1] + 1e-9 for i in range(len(rob_accs) - 1))
    best_double_gain = max(rob_accs[i + 1] - rob_accs[i] for i in range(len(rob_accs) - 1))

    print("\n" + "=" * 78)
    print("HEADLINE verdict")
    print("=" * 78)
    print(f"  widths               : {WIDTHS}")
    print(f"  params (M)           : {[round(r['params_M'], 3) for r in rows]}")
    print(f"  clean acc            : {[round(a, 3) for a in clean_accs]}")
    print(f"  robust acc           : {[round(a, 3) for a in rob_accs]}")
    print(f"  robust eff / Mparam  : {[round(r['robust_eff_per_Mparam'], 3) for r in rows]}")
    print(f"  robust gen gap       : {[round(lr['robust_gen_gap'], 3) for lr in loss_rows]}")
    print(f"  spread (max-min rob) : {spread:+.3f}")
    print(f"  best single doubling : {best_double_gain:+.3f}")
    print(f"  monotonic in W       : {monotonic}")

    if monotonic and spread >= 0.02:
        verdict = ("SUPPORTED: PGD robust acc is monotonic non-decreasing in width "
                   "and the total spread exceeds 2pp -- consistent with the "
                   "Bubeck-Sellke / Madry-2018 'more parameters -> more robust' "
                   "directional claim, even at SmallCNN scale.")
    elif spread >= 0.02:
        verdict = ("PARTIAL: a >=2pp robust-acc gain exists across widths but the "
                   "curve is non-monotonic -- consistent with Wu et al. (2020) "
                   "robust-generalization saturation / reversal at larger width.")
    elif all(rob_accs[i] >= rob_accs[i + 1] - 1e-9 for i in range(len(rob_accs) - 1)) \
            and spread >= 0.02:
        verdict = ("REFUTED (reversed): robust acc DECREASES with width -- strong "
                   "Wu-style robust overfitting; Bubeck-Sellke folklore fails at "
                   "this regime.")
    else:
        verdict = ("REFUTED: width simply does not move robust accuracy by 2pp at "
                   "this (SmallCNN, F-MNIST, N=6000, eps=0.1) budget. The "
                   "Bubeck-Sellke law's qualitative prediction is not detectable "
                   "here -- either we are too far below the parameter floor, or "
                   "data/loss-shape (cf. Pang 2020) dominates the width axis.")
    print(f"  VERDICT: {verdict}")
    print("=" * 78)
    print("Refs: Bubeck & Sellke (NeurIPS 2021, universal law of robustness);")
    print("      Madry et al. (ICLR 2018, capacity & PGD-AT);")
    print("      Wu, Xia, Wang (NeurIPS 2020, AWP / robust generalization gap);")
    print("      Pang et al. (ICLR 2020, rethinking softmax CE for AT).")
    print("=" * 78)


if __name__ == "__main__":
    main()
