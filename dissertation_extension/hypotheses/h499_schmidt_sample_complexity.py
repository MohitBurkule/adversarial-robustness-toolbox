"""
H499 - Schmidt et al. sample-complexity gap for adversarial robustness on F-MNIST.

Seed (CAMPAIGN_GAP_MAP §5 / §3 G6 theory):
    Schmidt, Santurkar, Tsipras, Talwar, Madry (NeurIPS 2018),
    "Adversarially Robust Generalization Requires More Data". arXiv:1804.11285.
    On a Gaussian model they prove the sample complexity for a robust classifier
    is sqrt(d) times that for a standard classifier; on a Bernoulli model the
    standard task is solvable from O(1) samples while robust needs poly(d).

Hypothesis (empirical F-MNIST instantiation):
    The training-set size required to reach 90% of asymptotic accuracy is
        N_robust_knee >= 3 x N_clean_knee
    AND this gap_ratio grows monotonically with eps.

Critique seed:
    Schmidt's bounds are toy-distribution (Gaussian / Bernoulli). On real
    F-MNIST images with a SmallCNN we expect the qualitative gap to survive
    (since F-MNIST images live near a low-dim manifold like the Gaussian
    cluster model) but the ratio may be smaller than sqrt(d) ~ sqrt(784) = 28.
    We probe the qualitative claim and quantify the empirical ratio.

Extra papers (cited; not direct dependencies):
    - Yin, Ramchandran, Bartlett (ICML 2019), "Rademacher Complexity for
      Adversarially Robust Generalization". arXiv:1810.11914. Shows the
      adversarial Rademacher complexity for linear classifiers has an extra
      sqrt(d) factor vs standard - same direction as Schmidt.
    - Khim & Loh (2018), "Adversarial Risk Bounds via Function
      Transformation". arXiv:1810.09519. Function-transformation bounds also
      predict a larger robust generalization gap.
    - Carmon, Raghunathan, Schmidt, Liang, Duchi (NeurIPS 2019),
      "Unlabeled Data Improves Adversarial Robustness". arXiv:1905.13736.
      Shows the gap can be partially closed with unlabeled data: i.e. the
      sample-complexity gap is *real* and exploitable, supporting Schmidt.

Design:
    Sweep N_TRAIN in {500, 1000, 2000, 4000, 6000, 12000, 30000} (capped at the
    full F-MNIST training-set size, 60000) at fixed 10 epochs.

    Two training modes per N:
      (a) STD : standard cross-entropy training.
      (b) AT  : PGD adversarial training at eps_train = 0.1, 7 PGD steps.

    Three evaluation epsilons in {0.05, 0.1, 0.2}, eval via PGD-20.

    For each (mode, eps) we get a curve acc(N).  Two curve fits are tried:
        exp:    a + b * (1 - exp(-N/tau))
        power:  a - b * N^(-gamma)
    For each curve we report:
        - asymptote estimate  (max of fit value at N=Nmax and observed @ Nmax)
        - knee N_90 = smallest N s.t. fitted-acc >= 0.9 * asymptote
          (linear interpolation between sweep points if needed)

    Derived quantity per eps:
        gap_ratio(eps) = N_robust_knee(eps) / N_clean_knee
    where N_clean_knee uses the STD model on clean test.

    Bonus diagnostic: per-class robust-acc at smallest and largest N, to see
    whether small N hurts robustness uniformly or concentrates on a few classes
    (Schmidt's lower bound is for the worst-case distribution; if a couple of
    F-MNIST classes carry the sample-complexity cost, that's an empirical
    refinement of the theory).

Verdict rule:
    SUPPORT  if median over eps of gap_ratio(eps) >= 3.0 AND gap_ratio is
             monotonic non-decreasing in eps.
    PARTIAL  if median >= 1.5 but either monotonicity or the 3x threshold fails.
    REJECT   otherwise.

DO NOT EXECUTE. This script is write-only per the task spec.
"""
import os, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
N_SWEEP = [500, 1000, 2000, 4000, 6000, 12000, 30000]
N_TRAIN_FULL = 60000              # F-MNIST has 60k train; cap if N > 60000
EPOCHS = 10
BATCH = 128
LR = 0.05
N_EVAL = 2000
EPS_TRAIN_AT = 0.1                # PGD-AT training budget
ADV_TRAIN_STEPS = 7
EPS_EVAL_LIST = [0.05, 0.1, 0.2]
PGD_EVAL_STEPS = 20
KNEE_FRAC = 0.90                  # "knee" = 90% of asymptote


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _clean_acc(model, X, Y, batch=512):
    _, a = C.logits_and_acc(model, X, Y, batch=batch)
    return a


def _robust_acc(model, X, Y, eps, steps=PGD_EVAL_STEPS, batch=256):
    """Accuracy under PGD-`steps` at the given eps. Returns float in [0,1]."""
    model.eval()
    correct = 0
    total = 0
    for i in range(0, X.size(0), batch):
        xb = X[i:i + batch]
        yb = Y[i:i + batch]
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            pred = model(xa).argmax(1)
        correct += int((pred == yb).sum().item())
        total += int(yb.size(0))
    return correct / max(1, total)


def _per_class_robust_acc(model, X, Y, eps, steps=PGD_EVAL_STEPS, batch=256, ncls=10):
    model.eval()
    per_correct = np.zeros(ncls, dtype=np.int64)
    per_total = np.zeros(ncls, dtype=np.int64)
    for i in range(0, X.size(0), batch):
        xb = X[i:i + batch]; yb = Y[i:i + batch]
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            pred = model(xa).argmax(1).cpu().numpy()
        yb_np = yb.cpu().numpy()
        for c in range(ncls):
            m = (yb_np == c)
            per_total[c] += int(m.sum())
            per_correct[c] += int(((pred == yb_np) & m).sum())
    return (per_correct / np.maximum(1, per_total)).tolist()


# ---------------------------------------------------------------------------
# curve fitting + knee extraction
# ---------------------------------------------------------------------------
def _fit_exp(ns, accs):
    """Fit acc(N) = a + b * (1 - exp(-N/tau)). Grid + local refine, pure-numpy."""
    ns = np.asarray(ns, dtype=np.float64)
    accs = np.asarray(accs, dtype=np.float64)
    best = (np.inf, (accs[0], max(0.0, accs[-1] - accs[0]), max(1.0, ns[-1] / 3)))
    for tau in np.geomspace(max(50.0, ns[0] / 4), ns[-1] * 10, 40):
        feat = 1.0 - np.exp(-ns / tau)
        # solve [1, feat] * [a; b] = accs in least squares
        A = np.stack([np.ones_like(feat), feat], axis=1)
        coef, *_ = np.linalg.lstsq(A, accs, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        resid = float(np.mean((A @ coef - accs) ** 2))
        if resid < best[0]:
            best = (resid, (a, b, float(tau)))
    a, b, tau = best[1]
    def f(n):
        n = np.asarray(n, dtype=np.float64)
        return a + b * (1 - np.exp(-n / tau))
    return f, {"a": a, "b": b, "tau": tau, "rmse": math.sqrt(best[0])}


def _fit_power(ns, accs):
    """Fit acc(N) = a - b * N^(-gamma). Grid over gamma, linear over (a,b)."""
    ns = np.asarray(ns, dtype=np.float64)
    accs = np.asarray(accs, dtype=np.float64)
    best = (np.inf, (accs[-1], 0.1, 0.5))
    for gamma in np.linspace(0.05, 1.5, 40):
        feat = -(ns ** (-gamma))
        A = np.stack([np.ones_like(feat), feat], axis=1)
        coef, *_ = np.linalg.lstsq(A, accs, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        resid = float(np.mean((A @ coef - accs) ** 2))
        if resid < best[0]:
            best = (resid, (a, b, float(gamma)))
    a, b, gamma = best[1]
    def f(n):
        n = np.asarray(n, dtype=np.float64)
        return a - b * (n ** (-gamma))
    return f, {"a": a, "b": b, "gamma": gamma, "rmse": math.sqrt(best[0])}


def _knee(ns, accs, frac=KNEE_FRAC):
    """Return (N_knee, asymptote, fit_kind, fit_params).
    Pick the better-fitting of exp / power, take asymptote = max(fit(Nmax), max(obs)),
    and find the smallest N (linear-interp) where fit(N) >= frac * asymptote.
    Robust to non-monotonic noise.
    """
    ns = list(ns)
    fexp, pexp = _fit_exp(ns, accs)
    fpow, ppow = _fit_power(ns, accs)
    use_exp = pexp["rmse"] <= ppow["rmse"]
    f = fexp if use_exp else fpow
    params = pexp if use_exp else ppow
    kind = "exp" if use_exp else "power"
    Nmax = ns[-1]
    asymptote = float(max(f(Nmax), max(accs)))
    target = frac * asymptote
    grid = np.geomspace(ns[0] / 2, Nmax * 2, 400)
    vals = f(grid)
    above = np.where(vals >= target)[0]
    if len(above) == 0:
        knee = float(Nmax * 2)  # never reached within probed range
    else:
        i = int(above[0])
        if i == 0:
            knee = float(grid[0])
        else:
            # linear interp in log-N space
            x0, x1 = math.log(grid[i - 1]), math.log(grid[i])
            y0, y1 = float(vals[i - 1]), float(vals[i])
            t = (target - y0) / max(1e-9, (y1 - y0))
            knee = float(math.exp(x0 + t * (x1 - x0)))
    return knee, asymptote, kind, params


# ---------------------------------------------------------------------------
# per-seed sweep
# ---------------------------------------------------------------------------
def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    ncls = meta["n_classes"]
    # Load the FULL train pool once (n_train=None) so we can re-subset per N.
    Xtr_full, Ytr_full, Xte, Yte = C.load_dataset(DS, n_train=None, n_eval=N_EVAL, seed=seed)
    n_pool = Xtr_full.size(0)

    # Deterministic per-seed shuffle of the train pool, then take prefixes.
    g = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    perm = torch.randperm(n_pool, generator=g).to(Xtr_full.device)
    Xtr_full = Xtr_full[perm]; Ytr_full = Ytr_full[perm]

    rows = []           # one row per (N, mode)
    for N in N_SWEEP:
        n_use = min(N, n_pool)
        Xtr, Ytr = Xtr_full[:n_use], Ytr_full[:n_use]

        for mode in ("std", "at"):
            C.set_seed(seed * 1000 + N + (0 if mode == "std" else 1))
            model = C.build_model("cnn", meta, seed=seed)
            t0 = time.time()
            C.train_model(
                model, Xtr, Ytr,
                epochs=EPOCHS, batch=BATCH, opt="sgd", lr=LR,
                ncls=ncls,
                adv_train=(mode == "at"),
                adv_eps=EPS_TRAIN_AT,
                adv_steps=ADV_TRAIN_STEPS,
                verbose=False,
            )
            train_s = time.time() - t0

            clean = _clean_acc(model, Xte, Yte)
            row = {
                "seed": seed, "N": int(n_use), "N_requested": int(N), "mode": mode,
                "clean_acc": float(clean),
                "train_s": round(train_s, 1),
            }
            for eps in EPS_EVAL_LIST:
                row[f"rob_acc_eps{eps}"] = float(_robust_acc(model, Xte, Yte, eps))
            # per-class robust at eps_train, only for the endpoints (small + large N)
            if N == N_SWEEP[0] or N == N_SWEEP[-1]:
                row["per_class_rob_eps0.1"] = _per_class_robust_acc(
                    model, Xte, Yte, eps=0.1, ncls=ncls)
            rows.append(row)
            print(f"[seed {seed}] N={n_use:>5d} mode={mode:>3s}  "
                  f"clean={clean:.3f}  "
                  f"rob0.05={row['rob_acc_eps0.05']:.3f}  "
                  f"rob0.10={row['rob_acc_eps0.1']:.3f}  "
                  f"rob0.20={row['rob_acc_eps0.2']:.3f}  "
                  f"({train_s:.0f}s)")
    return rows


# ---------------------------------------------------------------------------
# aggregation and verdict
# ---------------------------------------------------------------------------
def _curve_from_rows(rows, mode, key):
    """Return (Ns_sorted, mean_metric_over_seeds) for the given mode+metric key."""
    by_N = {}
    for r in rows:
        if r["mode"] != mode:
            continue
        by_N.setdefault(r["N"], []).append(r[key])
    Ns = sorted(by_N.keys())
    means = [float(np.mean(by_N[n])) for n in Ns]
    return Ns, means


def main():
    print("=" * 78)
    print("H499 - Schmidt et al. sample-complexity gap (clean vs robust) on F-MNIST")
    print("=" * 78)
    print(f"device={C.DEVICE}  dataset={DS}  arch=SmallCNN  epochs={EPOCHS}")
    print(f"N sweep              : {N_SWEEP}")
    print(f"train modes          : STD vs PGD-AT(eps={EPS_TRAIN_AT}, {ADV_TRAIN_STEPS} steps)")
    print(f"eval eps             : {EPS_EVAL_LIST}   PGD-{PGD_EVAL_STEPS} eval")
    print(f"knee fraction        : {KNEE_FRAC} of asymptote")
    print(f"seeds                : {SEEDS}")
    print("-" * 78)

    all_rows = []
    for s in SEEDS:
        all_rows.extend(run_seed(s))

    # ---- knee analysis ------------------------------------------------------
    # Clean knee: from STD model's clean accuracy curve.
    Ns_c, clean_mean = _curve_from_rows(all_rows, mode="std", key="clean_acc")
    N_clean_knee, asy_clean, kind_clean, params_clean = _knee(Ns_c, clean_mean)

    print("\n" + "=" * 78)
    print("CURVE FITS  (mean across seeds)")
    print("=" * 78)
    print(f"\nSTD clean acc            asymptote={asy_clean:.3f}  "
          f"knee_N90={N_clean_knee:.0f}  fit={kind_clean}  params={params_clean}")
    for N, a in zip(Ns_c, clean_mean):
        print(f"   N={N:>5d}  clean={a:.3f}")

    knees_robust = {}     # eps -> N_robust_knee (using AT model + that eps)
    asymptotes_robust = {}
    for eps in EPS_EVAL_LIST:
        Ns_r, rob_mean = _curve_from_rows(all_rows, mode="at", key=f"rob_acc_eps{eps}")
        N_rob, asy_rob, kind_rob, params_rob = _knee(Ns_r, rob_mean)
        knees_robust[eps] = N_rob
        asymptotes_robust[eps] = asy_rob
        print(f"\nAT robust acc eps={eps}  asymptote={asy_rob:.3f}  "
              f"knee_N90={N_rob:.0f}  fit={kind_rob}  params={params_rob}")
        for N, a in zip(Ns_r, rob_mean):
            print(f"   N={N:>5d}  rob={a:.3f}")

    # ---- gap ratios ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("SAMPLE-COMPLEXITY GAP  (Schmidt et al. 2018)")
    print("=" * 78)
    gap_ratios = []
    for eps in EPS_EVAL_LIST:
        gr = knees_robust[eps] / max(1.0, N_clean_knee)
        gap_ratios.append(gr)
        print(f"  eps={eps}:  N_robust_knee={knees_robust[eps]:8.0f}   "
              f"N_clean_knee={N_clean_knee:8.0f}   gap_ratio={gr:6.2f}")

    median_gap = float(np.median(gap_ratios))
    monotonic_in_eps = all(gap_ratios[i] <= gap_ratios[i + 1] + 1e-6
                           for i in range(len(gap_ratios) - 1))
    print(f"\n  median gap_ratio over eps : {median_gap:.2f}")
    print(f"  monotonic in eps          : {monotonic_in_eps}")

    if median_gap >= 3.0 and monotonic_in_eps:
        verdict = "SUPPORT"
    elif median_gap >= 1.5:
        verdict = "PARTIAL"
    else:
        verdict = "REJECT"

    # ---- per-class robust-acc diagnostic at endpoints -----------------------
    print("\n" + "=" * 78)
    print("PER-CLASS ROBUST ACC  (AT, eps=0.1)  -- smallest vs largest N")
    print("=" * 78)
    def _avg_per_class(rows, mode, N_target):
        vals = [r["per_class_rob_eps0.1"] for r in rows
                if r["mode"] == mode and r["N"] == N_target and "per_class_rob_eps0.1" in r]
        if not vals:
            return None
        arr = np.asarray(vals)  # (n_seeds, ncls)
        return arr.mean(axis=0).tolist()
    N_lo = min(r["N"] for r in all_rows)
    N_hi = max(r["N"] for r in all_rows)
    pc_lo = _avg_per_class(all_rows, "at", N_lo)
    pc_hi = _avg_per_class(all_rows, "at", N_hi)
    fmnist_names = ["Tshirt", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle"]
    if pc_lo is not None and pc_hi is not None:
        print(f"   {'class':<10s}  N={N_lo:>5d}    N={N_hi:>5d}    delta")
        for c in range(len(pc_lo)):
            d = pc_hi[c] - pc_lo[c]
            print(f"   {fmnist_names[c]:<10s}  {pc_lo[c]:.3f}      {pc_hi[c]:.3f}      "
                  f"{d:+.3f}")
        worst_classes = sorted(range(len(pc_lo)),
                               key=lambda c: pc_hi[c] - pc_lo[c], reverse=True)[:3]
        print(f"   top-3 N-hungriest classes (largest robust-acc gain): "
              f"{[fmnist_names[c] for c in worst_classes]}")

    # ---- headline -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(f"  median gap_ratio          : {median_gap:.2f}  (threshold for SUPPORT: 3.00)")
    print(f"  monotone-in-eps           : {monotonic_in_eps}")
    print(f"  N_clean_knee              : {N_clean_knee:.0f}")
    print(f"  N_robust_knee per eps     : "
          f"{ {eps: round(knees_robust[eps]) for eps in EPS_EVAL_LIST} }")
    print(f"  VERDICT                   : {verdict}")
    print("=" * 78)
    print("Interpretation:")
    print("  Schmidt et al. (2018) prove that a Gaussian-cluster classifier needs ~sqrt(d)")
    print("  more samples to be robust than to be standardly accurate. Yin et al. (2019)")
    print("  and Khim & Loh (2018) give Rademacher / function-transform bounds with the")
    print("  same direction. Carmon et al. (2019) further show the gap can be partly")
    print("  closed with unlabeled data. Here we test the qualitative form of the claim")
    print("  on F-MNIST: does the sample-size needed to hit 90% of asymptotic robust acc")
    print("  exceed the clean-acc analogue, and does the multiplicative gap grow with the")
    print("  attack budget? A SUPPORT verdict gives an empirical, image-data instance of")
    print("  Schmidt's theory; PARTIAL means the gap exists but does not reach 3x; REJECT")
    print("  would indicate F-MNIST is too low-complexity to expose the gap at our N range.")
    print("=" * 78)


if __name__ == "__main__":
    main()
