"""
H506 - eps_train vs input-space margin: scaling law and saturation.

Seed (campaign gap-map G6 / dissertation paper-5 section 5):
  Folklore from Madry et al. (ICLR 2018, "Towards Deep Learning Models Resistant
  to Adversarial Attacks") and its follow-ups says that PGD-AT widens the
  empirical input-space margin roughly in proportion to the training-time
  epsilon - and that pushing eps_train too far collapses clean accuracy.
  Neither the slope nor the saturation knee is quantified for Fashion-MNIST in
  the literature; we measure both.

Hypothesis (precise form):
  For PGD-AT (Madry-style, L_inf, 7 steps, SmallCNN, Fashion-MNIST):
    (H1)  median min-eps-to-flip = c * eps_train + b, with c in [1.2, 1.5]
          and |b| <= 0.01 over eps_train in {0.05, 0.10, 0.15}.
    (H2)  clean accuracy collapses (drops by >=10 percentage points vs std)
          and the linear scaling breaks (residual > 0.02 vs linear fit)
          once eps_train >= 0.25 (saturation regime).
    (H3)  the per-class slope c_k varies by less than 2x across classes - i.e.
          scaling is roughly class-uniform on F-MNIST.

Critique seed (advisor):
  This is "Madry implicit" - the qualitative claim is folklore. The novelty is
  the F-MNIST quantification of c, of the saturation knee, and of per-class
  heterogeneity. We are NOT claiming a theorem; we report the empirical fit,
  its residuals, its p25/p50/p75 envelope, and explicit per-class slopes.

Extra papers cited:
  * Madry et al., ICLR 2018 - PGD-AT baseline and the eps_train -> robust radius
    intuition.
  * Athalye, Carlini & Wagner, ICML 2018 / Athalye 2020 thesis "On the State
    of Robust Adversarial Machine Learning" - cautions that "robust radius"
    must be measured with a strong adaptive attack and against gradient
    masking; we use PGD-40-restart-2 bisection as a strong upper bound (cf.
    h486 critique).
  * Wang et al., ICLR 2020 - "Improving Adversarial Robustness Requires
    Revisiting Misclassified Examples" (MART): shows the misclassified-sample
    subset behaves differently under AT; we report robust-acc@0.1 and
    per-class slope to see whether saturation hits the hard classes first.

Controls (per the brief):
  (1) PGD-AT trained at eps_train in {0.02, 0.05, 0.10, 0.15, 0.20, 0.30}
      plus an STD baseline (eps_train = 0).
  (2) At each defence, compute per-sample min-eps-to-flip on the SAME 1000 test
      samples; report median (p50), p25, p75.
  (3) Linear fit median(eps_train) = c * eps_train + b on the moderate-regime
      subset {0.05, 0.10, 0.15} (the hypothesised linear band), plus a fit on
      the FULL sweep for contrast; report c, b, R^2, max residual.
  (4) Clean accuracy and robust-acc-at-eps=0.10 (PGD-40-restart-2) per model.
  (5) Per-class median min-eps; per-class linear slope c_k over the same
      moderate subset; range max_k c_k / min_k c_k.

This script writes a human-readable report to:
    results/fashion_mnist/h506_eps_vs_margin_scaling_output.txt
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DS = "fashion_mnist"
SEED = 0
N_EVAL = 1000
N_TRAIN = 20000
EPOCHS = 8

# eps_train sweep:  0.0 = STD baseline (no AT)
EPS_TRAIN_SWEEP = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
# moderate band used for the headline linear fit
LINEAR_BAND = [0.05, 0.10, 0.15]
# saturation threshold: H2 claims linear breaks AND clean acc collapses here.
SATURATION_EPS = 0.25
SATURATION_REGIME = [e for e in EPS_TRAIN_SWEEP if e >= SATURATION_EPS]

# Margin / robust-acc probing config
ROBUST_ACC_EPS = 0.10
BISECT_LO = 0.0
BISECT_HI = 0.4   # extend above eps_train=0.3 so we don't censor too aggressively
BISECT_DEPTH = 12
PGD_STEPS_EVAL = 40
PGD_RESTARTS = 2

N_CLASSES = 10
CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
)
OUT_PATH = os.path.join(RESULTS_DIR, "h506_eps_vs_margin_scaling_output.txt")


# ---------------------------------------------------------------------------
# Attack helpers (L_inf PGD with multi-restart; matches h486 style)
# ---------------------------------------------------------------------------
def pgd_linf(model, x, y, eps, steps=40, alpha=None, restarts=1, random_start=True):
    if alpha is None:
        alpha = max(eps / 10.0, 1e-4)
    x0 = x.clone().detach()
    best_adv = x0.clone()
    best_flipped = torch.zeros(x0.size(0), dtype=torch.bool, device=x0.device)
    for _ in range(restarts):
        xa = x0.clone()
        if random_start and eps > 0:
            xa = xa + torch.empty_like(xa).uniform_(-eps, eps)
            xa = xa.clamp(0.0, 1.0)
        for _ in range(steps):
            xa.requires_grad_(True)
            loss = F.cross_entropy(model(xa), y)
            g, = torch.autograd.grad(loss, xa)
            xa = xa.detach() + alpha * g.sign()
            xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0.0, 1.0)
        with torch.no_grad():
            flipped = (model(xa).argmax(1) != y)
        upgrade = flipped & (~best_flipped)
        if upgrade.any():
            best_adv[upgrade] = xa[upgrade]
            best_flipped = best_flipped | flipped
        if best_flipped.all():
            break
    return best_adv.detach(), best_flipped


def min_eps_to_flip_bisect(model, x, y, lo=BISECT_LO, hi=BISECT_HI,
                            depth=BISECT_DEPTH, steps=PGD_STEPS_EVAL,
                            restarts=PGD_RESTARTS, batch=256):
    """Per-sample L_inf binary search. UPPER BOUND on true margin (cf. h486)."""
    N = x.size(0)
    out_min = torch.full((N,), float(hi), device=x.device)
    out_success = torch.zeros(N, dtype=torch.bool, device=x.device)
    for i in range(0, N, batch):
        bx = x[i:i + batch]
        by = y[i:i + batch]
        n = bx.size(0)
        _, flipped_hi = pgd_linf(model, bx, by, eps=hi, steps=steps,
                                  restarts=restarts, random_start=True)
        out_success[i:i + batch] = flipped_hi
        lo_t = torch.full((n,), float(lo), device=bx.device)
        hi_t = torch.full((n,), float(hi), device=bx.device)
        for _ in range(depth):
            mid = 0.5 * (lo_t + hi_t)
            xa = bx.clone()
            xa = xa + (torch.empty_like(xa).uniform_(-1.0, 1.0)
                       * mid.view(-1, 1, 1, 1))
            xa = xa.clamp(0.0, 1.0)
            alpha = (mid / 10.0).clamp(min=1e-4)
            for _step in range(steps):
                xa.requires_grad_(True)
                loss = F.cross_entropy(model(xa), by)
                g, = torch.autograd.grad(loss, xa)
                step_vec = alpha.view(-1, 1, 1, 1) * g.sign()
                xa = xa.detach() + step_vec
                lower = bx - mid.view(-1, 1, 1, 1)
                upper = bx + mid.view(-1, 1, 1, 1)
                xa = torch.min(torch.max(xa, lower), upper).clamp(0.0, 1.0)
            with torch.no_grad():
                flipped = (model(xa).argmax(1) != by)
            hi_t = torch.where(flipped & flipped_hi, mid, hi_t)
            lo_t = torch.where((~flipped) & flipped_hi, mid, lo_t)
        est = torch.where(flipped_hi, hi_t, torch.full_like(hi_t, float(hi)))
        out_min[i:i + batch] = est
    return out_min, out_success


# ---------------------------------------------------------------------------
# Training: STD (eps_train=0) and PGD-AT at arbitrary eps_train
# ---------------------------------------------------------------------------
def train_one(eps_train, Xtr, Ytr, meta, seed=SEED):
    C.set_seed(seed)
    m = C.build_model("cnn", meta)
    if eps_train <= 0:
        return C.train_model(m, Xtr, Ytr, epochs=EPOCHS, batch=128,
                              opt="adam", lr=1e-3)
    return C.train_model(m, Xtr, Ytr, epochs=EPOCHS, batch=128, opt="adam", lr=1e-3,
                          adv_train=True, adv_eps=eps_train, adv_steps=7)


# ---------------------------------------------------------------------------
# Robust accuracy at fixed eps
# ---------------------------------------------------------------------------
def robust_acc_at(model, X, Y, eps, steps=PGD_STEPS_EVAL, restarts=PGD_RESTARTS,
                  batch=256):
    n = X.size(0)
    n_correct_under_attack = 0
    for i in range(0, n, batch):
        bx = X[i:i + batch]; by = Y[i:i + batch]
        # only attack originally-correct samples (standard convention);
        # samples that were already wrong count as "not robust"
        with torch.no_grad():
            clean_correct = (model(bx).argmax(1) == by)
        if clean_correct.any():
            xc = bx[clean_correct]; yc = by[clean_correct]
            _, flipped = pgd_linf(model, xc, yc, eps=eps, steps=steps,
                                   restarts=restarts, random_start=True)
            n_correct_under_attack += int((~flipped).sum())
    return n_correct_under_attack / float(n)


# ---------------------------------------------------------------------------
# Linear fit utility (median ~ c * eps_train + b)
# ---------------------------------------------------------------------------
def linear_fit(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    c, b = np.polyfit(xs, ys, 1)
    yhat = c * xs + b
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    max_resid = float(np.max(np.abs(ys - yhat)))
    return float(c), float(b), r2, max_resid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fout = open(OUT_PATH, "w")

    def log(*args, **kw):
        msg = " ".join(str(a) for a in args)
        print(msg, **kw)
        fout.write(msg + "\n")
        fout.flush()

    log("=" * 78)
    log("H506 - eps_train vs input-space margin: scaling and saturation")
    log("=" * 78)
    log(f"Device: {C.DEVICE}   dataset: {DS}   seed: {SEED}")
    log(f"PGD-AT sweep eps_train = {EPS_TRAIN_SWEEP} (0.0 = STD baseline)")
    log(f"Train subset: {N_TRAIN}  epochs: {EPOCHS}  attack: PGD-7 (train), "
        f"PGD-{PGD_STEPS_EVAL}x{PGD_RESTARTS} (eval)")
    log(f"Margin estimator: per-sample bisection in [{BISECT_LO}, {BISECT_HI}], "
        f"depth={BISECT_DEPTH}, on {N_EVAL} fixed test samples.")
    log(f"Linear band (headline fit):  eps_train in {LINEAR_BAND}")
    log(f"Saturation threshold:        eps_train >= {SATURATION_EPS}")
    log("")
    log("CRITIQUE: PGD bisection gives an UPPER BOUND on the true L_inf margin")
    log("(Athalye/Carlini/Wagner ICML 2018; cf. h486). Since all defences are")
    log("attacked under identical PGD config, cross-defence trends are valid.")
    log("")
    log("Refs:")
    log("  * Madry et al., ICLR 2018 (PGD-AT, eps_train baseline).")
    log("  * Athalye, Carlini, Wagner, ICML 2018; Athalye 2020 (attack-quality")
    log("    cautions; necessity of strong adaptive PGD for robust-radius claims).")
    log("  * Wang et al., ICLR 2020 (MART; misclassified examples behave")
    log("    differently under AT - motivates per-class slope reporting).")
    log("")

    # ---- Data ------------------------------------------------------------
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    Yte_np = Yte.cpu().numpy()
    log(f"Loaded {DS}: Xtr {tuple(Xtr.shape)}, Xte {tuple(Xte.shape)}")
    log("")

    # ---- Train all defences ---------------------------------------------
    models = {}
    timings = {}
    for eps_train in EPS_TRAIN_SWEEP:
        name = "STD" if eps_train == 0.0 else f"AT@{eps_train:.2f}"
        t0 = time.time()
        log(f"--- Training {name} (eps_train={eps_train}) ---")
        m = train_one(eps_train, Xtr, Ytr, meta, seed=SEED)
        m.eval()
        timings[name] = time.time() - t0
        models[name] = m
        with torch.no_grad():
            clean_acc = float((m(Xte).argmax(1) == Yte).float().mean())
        log(f"    trained in {timings[name]:6.1f}s   clean_acc={clean_acc:.4f}")
    log("")

    # ---- Compute clean acc, robust acc @ ROBUST_ACC_EPS, min-eps ---------
    clean_acc = {}
    rob_acc = {}
    minE = {}
    success = {}
    for eps_train in EPS_TRAIN_SWEEP:
        name = "STD" if eps_train == 0.0 else f"AT@{eps_train:.2f}"
        m = models[name]
        with torch.no_grad():
            clean_acc[name] = float((m(Xte).argmax(1) == Yte).float().mean())
        t0 = time.time()
        rob_acc[name] = robust_acc_at(m, Xte, Yte, eps=ROBUST_ACC_EPS)
        log(f"[{name}] robust_acc@{ROBUST_ACC_EPS:.2f} = {rob_acc[name]:.4f}  "
            f"({time.time()-t0:.1f}s)")
        t0 = time.time()
        me, sc = min_eps_to_flip_bisect(m, Xte, Yte)
        minE[name] = me.cpu().numpy()
        success[name] = sc.cpu().numpy()
        log(f"    min-eps done in {time.time()-t0:6.1f}s  "
            f"censored@{BISECT_HI}={int((~sc).sum())}/{N_EVAL}")
    log("")

    # ---- Section A: clean acc / robust acc / margin summary -------------
    log("=" * 78)
    log("Section A: per-model summary")
    log("=" * 78)
    header = (f"{'model':<10}  {'eps_tr':>7}  {'clean_acc':>10}  "
              f"{'rob_acc@'+f'{ROBUST_ACC_EPS:.2f}':>13}  "
              f"{'min_p25':>9}  {'min_p50':>9}  {'min_p75':>9}  {'censor':>7}")
    log(header)
    log("-" * len(header))
    summary = {}
    for eps_train in EPS_TRAIN_SWEEP:
        name = "STD" if eps_train == 0.0 else f"AT@{eps_train:.2f}"
        me = minE[name]
        p25, p50, p75 = np.quantile(me, [0.25, 0.50, 0.75])
        censor = int((~success[name]).sum())
        summary[name] = {
            "eps_train": eps_train,
            "clean_acc": clean_acc[name],
            "rob_acc": rob_acc[name],
            "p25": float(p25), "p50": float(p50), "p75": float(p75),
            "censor": censor,
        }
        log(f"{name:<10}  {eps_train:>7.3f}  {clean_acc[name]:>10.4f}  "
            f"{rob_acc[name]:>13.4f}  "
            f"{p25:>9.4f}  {p50:>9.4f}  {p75:>9.4f}  {censor:>7d}")
    log("")

    # ---- Section B: linear fit on moderate band -------------------------
    log("=" * 78)
    log("Section B: linear fit  median(min-eps) = c * eps_train + b")
    log("=" * 78)
    # moderate band
    band_xs = LINEAR_BAND
    band_ys = [summary[f"AT@{e:.2f}"]["p50"] for e in band_xs]
    c_band, b_band, r2_band, mres_band = linear_fit(band_xs, band_ys)
    log(f"Moderate band eps_train in {band_xs}:")
    log(f"    median p50 values = {[round(v, 4) for v in band_ys]}")
    log(f"    fit: c = {c_band:+.4f}   b = {b_band:+.4f}   "
        f"R^2 = {r2_band:.4f}   max|resid| = {mres_band:.4f}")
    log("")

    # full sweep fit (excluding STD baseline since eps_train=0 anchors trivially)
    full_xs = [e for e in EPS_TRAIN_SWEEP if e > 0]
    full_ys = [summary[f"AT@{e:.2f}"]["p50"] for e in full_xs]
    c_full, b_full, r2_full, mres_full = linear_fit(full_xs, full_ys)
    log(f"Full PGD-AT sweep eps_train in {full_xs}:")
    log(f"    median p50 values = {[round(v, 4) for v in full_ys]}")
    log(f"    fit: c = {c_full:+.4f}   b = {b_full:+.4f}   "
        f"R^2 = {r2_full:.4f}   max|resid| = {mres_full:.4f}")
    log("")

    # Per-quantile slopes (p25, p75) on moderate band - sanity check
    for q_label, q in [("p25", 0.25), ("p75", 0.75)]:
        ys_q = [summary[f"AT@{e:.2f}"][q_label] for e in band_xs]
        cq, bq, r2q, mrq = linear_fit(band_xs, ys_q)
        log(f"  Slope on {q_label} (moderate band): c={cq:+.4f}  b={bq:+.4f}  "
            f"R^2={r2q:.4f}  max|resid|={mrq:.4f}")
    log("")

    # ---- Section C: saturation diagnostic -------------------------------
    log("=" * 78)
    log("Section C: saturation diagnostic")
    log("=" * 78)
    std_clean = summary["STD"]["clean_acc"]
    log(f"STD baseline clean accuracy = {std_clean:.4f}")
    log(f"{'eps_train':>10}  {'clean_acc':>10}  {'drop_vs_STD':>12}  "
        f"{'p50_pred':>10}  {'p50_obs':>10}  {'resid':>10}  {'sat_flag':>8}")
    log("-" * 80)
    any_saturation = False
    for eps_train in [e for e in EPS_TRAIN_SWEEP if e > 0]:
        name = f"AT@{eps_train:.2f}"
        s = summary[name]
        pred_p50 = c_band * eps_train + b_band
        resid = s["p50"] - pred_p50
        drop = std_clean - s["clean_acc"]
        sat = (eps_train >= SATURATION_EPS) and (drop >= 0.10) and (abs(resid) >= 0.02)
        any_saturation = any_saturation or sat
        log(f"{eps_train:>10.3f}  {s['clean_acc']:>10.4f}  {drop:>+12.4f}  "
            f"{pred_p50:>10.4f}  {s['p50']:>10.4f}  {resid:>+10.4f}  "
            f"{'YES' if sat else 'no':>8}")
    log("")
    log(f"H2 (saturation at eps_train >= {SATURATION_EPS}): "
        f"clean_acc drop >= 10pp AND |residual from moderate-band fit| >= 0.02")
    log(f"H2 met for any eps_train in saturation regime? "
        f"{'YES' if any_saturation else 'NO'}")
    log("")

    # ---- Section D: per-class slopes -----------------------------------
    log("=" * 78)
    log("Section D: per-class median min-eps and per-class linear slope")
    log("=" * 78)
    log(f"Slope c_k computed on the moderate band eps_train in {LINEAR_BAND}.")
    log("")
    # per-class median tables for the moderate band + STD reference
    band_models = ["STD"] + [f"AT@{e:.2f}" for e in band_xs]
    header_cols = ["class"] + band_models + ["c_k", "b_k", "R^2"]
    log("  " + "  ".join(f"{h:>10}" for h in header_cols))
    log("  " + "-" * (12 * len(header_cols)))
    class_slopes = {}
    for k in range(N_CLASSES):
        sel = (Yte_np == k)
        if sel.sum() < 5:
            log(f"  {CLASS_NAMES[k]:>10}  (n<5, skipped)")
            continue
        row_meds = []
        for name in band_models:
            row_meds.append(float(np.median(minE[name][sel])))
        # slope on AT@... only (exclude STD anchor)
        xs_k = band_xs
        ys_k = row_meds[1:]
        c_k, b_k, r2_k, _ = linear_fit(xs_k, ys_k)
        class_slopes[k] = (c_k, b_k, r2_k)
        cells = [CLASS_NAMES[k]] + [f"{v:.4f}" for v in row_meds] + \
                [f"{c_k:+.3f}", f"{b_k:+.3f}", f"{r2_k:.3f}"]
        log("  " + "  ".join(f"{c:>10}" for c in cells))
    valid_cs = [v[0] for v in class_slopes.values() if v[0] == v[0]]
    if valid_cs:
        cmin, cmax = min(valid_cs), max(valid_cs)
        ratio = cmax / cmin if cmin > 1e-6 else float("inf")
        log("")
        log(f"Per-class slope range: min(c_k) = {cmin:+.3f}   "
            f"max(c_k) = {cmax:+.3f}   max/min = {ratio:.2f}x")
        h3_met = ratio < 2.0
        log(f"H3 (per-class slope spread < 2x): {'MET' if h3_met else 'VIOLATED'}")
    else:
        h3_met = False
        log("H3: not enough valid per-class slopes")
    log("")

    # ---- HEADLINE verdict ------------------------------------------------
    log("=" * 78)
    log("HEADLINE VERDICT")
    log("=" * 78)
    h1_met = (1.2 <= c_band <= 1.5) and (abs(b_band) <= 0.01)
    log(f"H1 (moderate-band slope c in [1.2, 1.5], |b| <= 0.01):")
    log(f"    measured c = {c_band:+.4f}, b = {b_band:+.4f}, R^2 = {r2_band:.4f}")
    log(f"    => {'MET' if h1_met else 'NOT MET'}")
    log(f"H2 (saturation at eps_train >= {SATURATION_EPS}):  "
        f"{'MET' if any_saturation else 'NOT MET'}")
    log(f"H3 (per-class slope spread < 2x):  "
        f"{'MET' if h3_met else 'NOT MET'}")
    log("")
    n_met = int(h1_met) + int(any_saturation) + int(h3_met)
    if n_met == 3:
        verdict = (f"SUPPORTED: median min-eps scales linearly with eps_train "
                   f"(c={c_band:+.3f}) in the moderate band, saturates above "
                   f"eps_train={SATURATION_EPS}, and per-class scaling is "
                   f"approximately uniform on Fashion-MNIST.")
    elif n_met >= 1:
        verdict = (f"PARTIAL: {n_met}/3 sub-hypotheses met. "
                   f"H1={h1_met}  H2={any_saturation}  H3={h3_met}.  "
                   f"Moderate-band slope c={c_band:+.3f}, b={b_band:+.4f}.")
    else:
        verdict = (f"REJECTED: none of H1/H2/H3 hold. Moderate-band fit "
                   f"c={c_band:+.3f}, b={b_band:+.4f}, R^2={r2_band:.3f}.")
    log(verdict)
    log("")
    log(f"Output written to: {OUT_PATH}")
    fout.close()


if __name__ == "__main__":
    main()
