"""
H505 - Tail-index of the per-sample margin distribution (Margin-CDF tail / G6).

Seed (Section 5 / Section 3 G6 of the campaign plan): the empirical CDF of
input-space margins ("min-eps-to-flip" under L_inf) has interesting *bulk*
behaviour (see H486), but the headline claim of "AT removes adversarial
brittleness" is really a claim about the LEFT TAIL of the margin distribution
- i.e. the brittlest few percent of samples. We therefore ask whether the
left tail of the margin distribution is *power-law / heavy-tailed* under
standard training, and whether adversarial training (PGD-AT) thins it into
an essentially exponential / light-tailed regime.

Hypothesis (G6 seed):
  * STD model:    left-tail of margins is Pareto-like with tail-index alpha < 2
                  (fat tail; a non-negligible probability mass of samples that
                  flip at near-zero eps).
  * PGD-AT model: alpha > 3 (thin tail; near-exponential), i.e. AT
                  specifically targets the brittlest quantile and converts the
                  power-law tail into a near-exponential one. This is a
                  stronger and falsifiable claim than "AT raises the mean
                  robust radius".

Critique applied (from peer review of the campaign):
  * Tail-index estimation (Hill estimator, Hill 1975) requires MANY samples,
    so we draw 1000 fixed F-MNIST test inputs (out of 10000) for the
    binary-search-margin evaluations rather than the smaller subsets used
    by earlier hypotheses. The hypothesis itself is about the FULL test
    distribution; the 1000-sample budget is the largest tractable size
    under PGD-bisection cost.
  * Hill applies to the original heavy-tailed variable. Margins are
    bounded below by 0 and we are interested in SMALL margins (brittle
    samples), so we transform to Z_i = 1 / margin_i: small margin ->
    large Z -> Hill applied to upper order statistics of Z is equivalent
    to a left-tail-index estimator for margin. We DO NOT take a log of the
    margin directly before Hill - Hill is for heavy-tailed variables, not
    for log-margin which is essentially the link function of an exponential
    fit. This is documented explicitly below.
  * Hill estimates depend on the choice of order-statistic cut "k". We
    therefore sweep k over bottom-5% / 10% / 20% of margins and report
    all three. We additionally run a Kolmogorov-Smirnov goodness-of-fit
    test of the margin distribution against an exponential null on the
    FULL empirical CDF (Massey 1951) as a complementary, parameter-free
    diagnostic for tail-heaviness.
  * Right-censored samples (those whose adversarial example was not found
    inside the PGD-bisection budget [0, 0.3]) are EXCLUDED from the Hill
    estimator (they are not in the brittle tail by definition). They are
    counted and reported.

Controls (per the seed):
  (1) STD vs PGD-AT trained side-by-side on F-MNIST SmallCNN, same seed.
  (2) Min-eps-to-flip on 1000 samples via PGD binary search (depth 12,
      PGD-40 with 2 restarts at each midpoint).
  (3) Hill tail-index alpha-hat using bottom 5% / 10% / 20% of margins.
  (4) Full empirical KS test of the margin distribution vs Exponential(lambda)
      with lambda fitted by MLE (lambda-hat = 1 / mean(margin)).
  (5) Per-class tail-index (k = bottom-20% of class-conditional margins) for
      a class-resolved view: does AT thin tails uniformly, or only on the
      worst classes?

Important methodological note on the Hill estimator:
  The Hill estimator H_k of a sample x_1, ..., x_n with order statistics
  x_(1) >= x_(2) >= ... >= x_(n) is
       H_k = (1/k) * sum_{i=1}^{k} log(x_(i)) - log(x_(k+1))
  and estimates 1/alpha for a Pareto-like upper tail. To probe a LEFT tail
  of the margin distribution we apply Hill to Z_i = 1 / margin_i (upper
  tail of Z is the left tail of margin). We then report alpha_hat = 1/H_k.
  Crucially, we DO NOT log-transform margin before Hill: Hill operates on
  the original heavy-tailed variable, and a log-margin would already be
  an exponential-family quantity that breaks the Pareto assumption.

External works cited (>= 2 required by seed):
  * Carlini et al. (NeurIPS 2019, "Distribution Density, Tails, and Outliers
    in Machine Learning: Metrics and Applications") - tail-aware metrics
    for adversarial examples; argues that brittleness is concentrated in
    the data distribution's tail, motivating tail-specific reporting.
  * Feldman (STOC 2020, "Does Learning Require Memorization? A Short
    Tale about a Long Tail") - shows that long-tailed data distributions
    force memorisation of tail points; relevant because the brittle
    samples in F-MNIST are most likely tail / atypical points.
  * Hill (Annals of Statistics 1975, "A simple general approach to
    inference about the tail of a distribution") - the Hill estimator
    itself.

Output written to results/fashion_mnist/h505_margin_cdf_tail_output.txt.
"""
import os
import sys
import time
import math
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
N_TEST_POOL = 10000     # full F-MNIST test set
N_EVAL = 1000           # binary-search-margin evaluations (per seed/§5)
N_TRAIN = 20000         # subset of FMNIST train for speed
EPOCHS = 8
AT_EPS = 0.1
BISECT_LO = 0.0
BISECT_HI = 0.3
BISECT_DEPTH = 12
PGD_STEPS_EVAL = 40
PGD_RESTARTS = 2
HILL_FRACS = (0.05, 0.10, 0.20)
CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
N_CLASSES = 10
PER_CLASS_TAIL_FRAC = 0.20

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
)
OUT_PATH = os.path.join(RESULTS_DIR, "h505_margin_cdf_tail_output.txt")


# ---------------------------------------------------------------------------
# Attack helpers (multi-restart L_inf PGD + per-sample bisection)
# ---------------------------------------------------------------------------
def pgd_linf(model, x, y, eps, steps=40, alpha=None, restarts=1, random_start=True):
    """Standard L_inf PGD with optional restarts. Returns (x_adv, flipped)."""
    if alpha is None:
        alpha = max(eps / 10.0, 1e-4)
    x0 = x.clone().detach()
    best_adv = x0.clone()
    best_flipped = torch.zeros(x0.size(0), dtype=torch.bool, device=x0.device)
    for r in range(restarts):
        xa = x0.clone()
        if random_start:
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
    """Per-sample binary search for the smallest eps in [lo, hi] such that PGD
    at that eps flips the prediction. Returns (min_eps, flipped_at_hi).

    NOTE: this is an UPPER BOUND on the true L_inf margin (PGD may miss
    adversarials). Differences between defences remain valid under identical
    attack configuration.
    """
    N = x.size(0)
    out_min = torch.full((N,), float(hi), device=x.device)
    out_success = torch.zeros(N, dtype=torch.bool, device=x.device)

    for i in range(0, N, batch):
        bx = x[i:i + batch]
        by = y[i:i + batch]
        n = bx.size(0)

        # 1) check the maximum budget first
        _, flipped_hi = pgd_linf(model, bx, by, eps=hi, steps=steps,
                                  restarts=restarts, random_start=True)
        out_success[i:i + batch] = flipped_hi

        # 2) per-sample bisection
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
# Training: STD and PGD-AT
# ---------------------------------------------------------------------------
def train_std(Xtr, Ytr, meta, seed=SEED):
    C.set_seed(seed)
    m = C.build_model("cnn", meta)
    return C.train_model(m, Xtr, Ytr, epochs=EPOCHS, batch=128, opt="adam", lr=1e-3)


def train_pgd_at(Xtr, Ytr, meta, seed=SEED):
    C.set_seed(seed)
    m = C.build_model("cnn", meta)
    return C.train_model(m, Xtr, Ytr, epochs=EPOCHS, batch=128, opt="adam", lr=1e-3,
                          adv_train=True, adv_eps=AT_EPS, adv_steps=7)


# ---------------------------------------------------------------------------
# Tail-statistics
# ---------------------------------------------------------------------------
def hill_estimator(z, k):
    """Hill (1975) tail-index estimator on the UPPER tail of `z`.

    Returns (alpha_hat, H_k) where H_k = (1/k) sum_{i=1}^k log z_(i) - log z_(k+1)
    and alpha_hat = 1 / H_k.

    Inputs:
      z : 1-D numpy array of strictly positive samples.
      k : number of upper order statistics to use (1 <= k <= len(z)-1).
    """
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z) & (z > 0)]
    n = z.size
    if n < 3 or k < 1 or k >= n:
        return float("nan"), float("nan")
    # descending sort: z_sorted[0] is the largest
    z_sorted = np.sort(z)[::-1]
    log_top = np.log(z_sorted[:k])
    log_anchor = math.log(z_sorted[k])
    H_k = float(log_top.mean() - log_anchor)
    if H_k <= 0:
        return float("nan"), H_k
    return float(1.0 / H_k), H_k


def left_tail_alpha(margins, frac):
    """Estimate the LEFT-tail index of `margins` via Hill on Z = 1/margin.

    `frac` selects how many upper order statistics of Z (== lower order
    statistics of margin) to use. Returns (alpha_hat, k_used, n_used).
    """
    m = np.asarray(margins, dtype=np.float64)
    m = m[np.isfinite(m) & (m > 0)]
    if m.size < 10:
        return float("nan"), 0, m.size
    z = 1.0 / m
    k = max(1, int(round(frac * m.size)))
    k = min(k, m.size - 2)
    alpha, _ = hill_estimator(z, k)
    return alpha, k, m.size


def ks_against_exponential(margins):
    """Kolmogorov-Smirnov two-sided test of `margins` against an exponential
    distribution with rate lambda_hat = 1 / mean(margins) (MLE).

    Returns (D, p_approx, lambda_hat). The p-value uses the asymptotic
    Kolmogorov distribution (Massey 1951) and is asymptotic; the
    distribution parameter is estimated from the data, so the reported
    p-value is OPTIMISTIC (Lilliefors correction is not applied). This is
    documented in the output.
    """
    m = np.asarray(margins, dtype=np.float64)
    m = m[np.isfinite(m) & (m > 0)]
    n = m.size
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    mean_m = float(m.mean())
    if mean_m <= 0:
        return float("nan"), float("nan"), float("nan")
    lam = 1.0 / mean_m
    m_sorted = np.sort(m)
    # empirical CDF jumps i/n at m_sorted[i-1]
    i = np.arange(1, n + 1)
    F_emp_upper = i / n
    F_emp_lower = (i - 1) / n
    F_theo = 1.0 - np.exp(-lam * m_sorted)
    D_plus = float(np.max(F_emp_upper - F_theo))
    D_minus = float(np.max(F_theo - F_emp_lower))
    D = max(D_plus, D_minus)
    # asymptotic p-value (Kolmogorov distribution); valid for large n.
    # Q(lam) = 2 * sum_{j=1..inf} (-1)^(j-1) exp(-2 j^2 lam^2)
    lam_stat = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * D
    s = 0.0
    for j in range(1, 101):
        term = ((-1) ** (j - 1)) * math.exp(-2.0 * j * j * lam_stat * lam_stat)
        s += term
        if abs(term) < 1e-12:
            break
    p = max(0.0, min(1.0, 2.0 * s))
    return D, p, lam


def quantiles(x, qs=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)):
    return {f"p{int(q*100):02d}": float(np.quantile(np.asarray(x), q)) for q in qs}


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
    log("H505 - Tail-index of per-sample margin distribution (STD vs PGD-AT)")
    log("=" * 78)
    log(f"Device: {C.DEVICE}")
    log(f"Dataset: {DS}  (full test pool N={N_TEST_POOL})")
    log(f"Margin-eval budget: N_EVAL={N_EVAL} (PGD-bisection on fixed subset)")
    log(f"Bisection: PGD-{PGD_STEPS_EVAL} in [{BISECT_LO},{BISECT_HI}], "
        f"depth={BISECT_DEPTH}, restarts={PGD_RESTARTS}")
    log(f"Hill fractions: {HILL_FRACS}    Per-class tail frac: {PER_CLASS_TAIL_FRAC}")
    log("")
    log("Hypothesis (G6): STD has alpha<2 (Pareto, fat left tail);")
    log("                 PGD-AT has alpha>3 (near-exponential, thin tail).")
    log("")
    log("Methodological note: Hill operates on the ORIGINAL heavy-tailed")
    log("variable. We apply Hill to Z = 1/margin so that the LEFT tail of the")
    log("margin distribution becomes the upper tail of Z. We DO NOT take the")
    log("log of the margin before Hill - log-margin is already an")
    log("exponential-family quantity and would invalidate the Pareto")
    log("assumption Hill is built on.")
    log("")
    log("References cited:")
    log("  Hill (Annals of Statistics 1975): the Hill tail-index estimator.")
    log("  Carlini et al. (NeurIPS 2019): tails / outliers in ML metrics.")
    log("  Feldman (STOC 2020): long-tailed data forces tail memorisation.")
    log("  Massey (1951): one-sample Kolmogorov-Smirnov test.")
    log("")

    # ---- Data -----------------------------------------------------------
    meta = C.dataset_meta(DS)
    # Load full F-MNIST test set (10000), then we pick 1000 fixed indices.
    Xtr, Ytr, Xte_full, Yte_full = C.load_dataset(
        DS, n_train=N_TRAIN, n_eval=N_TEST_POOL, seed=SEED,
    )
    log(f"Loaded {DS}: Xtr {tuple(Xtr.shape)}, Xte_full {tuple(Xte_full.shape)}")

    # fixed deterministic subset of N_EVAL for margin bisection
    g = torch.Generator(device="cpu").manual_seed(SEED)
    idx = torch.randperm(Xte_full.size(0), generator=g)[:N_EVAL]
    Xte = Xte_full[idx]
    Yte = Yte_full[idx]
    log(f"Margin-eval subset: N={N_EVAL} samples (deterministic, seed={SEED})")
    log("")

    # ---- Train STD and PGD-AT ------------------------------------------
    log("--- Training STD ---")
    t0 = time.time()
    m_std = train_std(Xtr, Ytr, meta)
    m_std.eval()
    t_std = time.time() - t0
    with torch.no_grad():
        acc_std = float((m_std(Xte_full).argmax(1) == Yte_full).float().mean())
    log(f"    STD trained in {t_std:6.1f}s    clean_acc={acc_std:.4f}")

    log("--- Training PGD-AT ---")
    t0 = time.time()
    m_at = train_pgd_at(Xtr, Ytr, meta)
    m_at.eval()
    t_at = time.time() - t0
    with torch.no_grad():
        acc_at = float((m_at(Xte_full).argmax(1) == Yte_full).float().mean())
    log(f"    PGD-AT trained in {t_at:6.1f}s    clean_acc={acc_at:.4f}")
    log("")

    # ---- Per-sample min-eps via PGD bisection ---------------------------
    models = {"STD": m_std, "PGD-AT": m_at}
    minE = {}
    success = {}
    for name, m in models.items():
        log(f"[{name}] PGD-bisection on N={N_EVAL}...")
        t0 = time.time()
        me, sc = min_eps_to_flip_bisect(m, Xte, Yte)
        minE[name] = me.cpu().numpy()
        success[name] = sc.cpu().numpy()
        n_cens = int((~sc).sum())
        log(f"    done in {time.time()-t0:6.1f}s   "
            f"censored@{BISECT_HI}={n_cens}/{N_EVAL}   "
            f"mean_min_eps={float(np.mean(minE[name])):.4f}")
    log("")

    # ---- Margin summary (percentile sanity) -----------------------------
    log("=" * 78)
    log("Section A: per-defence margin percentiles (sanity)")
    log("=" * 78)
    qs = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")
    header = f"{'model':<8}  " + "  ".join(f"{q:>7}" for q in qs)
    log(header); log("-" * len(header))
    for name in ("STD", "PGD-AT"):
        Q = quantiles(minE[name])
        log(f"{name:<8}  " + "  ".join(f"{Q[q]:>7.4f}" for q in qs))
    log("")

    # ---- Hill tail-index estimates --------------------------------------
    log("=" * 78)
    log("Section B: Hill tail-index alpha-hat on Z = 1 / margin")
    log("=" * 78)
    log("(only non-censored samples are used; right-censored ones are NOT")
    log(" in the brittle tail by construction.)")
    log("")
    log(f"{'model':<8}  {'frac':>5}  {'k':>5}  {'n':>5}  {'alpha_hat':>10}")
    log("-" * 50)
    alpha_summary = {}
    for name in ("STD", "PGD-AT"):
        keep = success[name].astype(bool)
        ms = minE[name][keep]
        alpha_summary[name] = {}
        for frac in HILL_FRACS:
            alpha, k_used, n_used = left_tail_alpha(ms, frac)
            alpha_summary[name][frac] = alpha
            log(f"{name:<8}  {frac:>5.2f}  {k_used:>5d}  {n_used:>5d}  {alpha:>10.4f}")
    log("")
    log("Interpretation:")
    log("  * alpha < 2 -> Pareto / fat left tail (heavy brittleness).")
    log("  * 2 <= alpha < 3 -> heavy but finite-variance tail.")
    log("  * alpha >= 3 -> thin / near-exponential tail.")
    log("")

    # ---- Kolmogorov-Smirnov vs Exponential ------------------------------
    log("=" * 78)
    log("Section C: KS goodness-of-fit vs Exponential(lambda_hat)")
    log("=" * 78)
    log("Null: margin ~ Exp(lambda) with lambda fitted by MLE (1 / mean).")
    log("Reject H0 -> margin distribution is NOT exponential (likely heavier).")
    log("p-values are asymptotic Kolmogorov (Massey 1951); since lambda was")
    log("estimated from the data the reported p is OPTIMISTIC (no Lilliefors")
    log("correction) - treat very small p as the meaningful signal.")
    log("")
    log(f"{'model':<8}  {'lambda_hat':>10}  {'D':>10}  {'p_asymp':>10}")
    log("-" * 46)
    ks_summary = {}
    for name in ("STD", "PGD-AT"):
        keep = success[name].astype(bool)
        ms = minE[name][keep]
        D, p, lam = ks_against_exponential(ms)
        ks_summary[name] = (D, p, lam)
        log(f"{name:<8}  {lam:>10.4f}  {D:>10.4f}  {p:>10.4e}")
    log("")

    # ---- Per-class tail-index -------------------------------------------
    log("=" * 78)
    log("Section D: per-class tail-index (frac={:.2f})".format(PER_CLASS_TAIL_FRAC))
    log("=" * 78)
    log("Does AT thin tails uniformly across classes, or only on the worst?")
    log("")
    Yte_np = Yte.cpu().numpy()
    hdr = (f"{'class':<12}  {'n_std':>5}  {'alpha_STD':>9}  "
           f"{'n_AT':>5}  {'alpha_AT':>9}  {'delta':>9}")
    log(hdr); log("-" * len(hdr))
    per_class_alpha = {}
    for c in range(N_CLASSES):
        sel = (Yte_np == c)
        if sel.sum() < 10:
            log(f"{CLASS_NAMES[c]:<12}  (n<10, skipped)")
            continue
        keep_std = sel & success["STD"].astype(bool)
        keep_at = sel & success["PGD-AT"].astype(bool)
        ms_std = minE["STD"][keep_std]
        ms_at = minE["PGD-AT"][keep_at]
        a_std, _, n_std = left_tail_alpha(ms_std, PER_CLASS_TAIL_FRAC)
        a_at, _, n_at = left_tail_alpha(ms_at, PER_CLASS_TAIL_FRAC)
        per_class_alpha[c] = (a_std, a_at)
        if (a_std == a_std) and (a_at == a_at):  # both finite
            delta_str = f"{a_at - a_std:+.4f}"
        else:
            delta_str = "n/a"
        log(f"{CLASS_NAMES[c]:<12}  {n_std:>5d}  {a_std:>9.4f}  "
            f"{n_at:>5d}  {a_at:>9.4f}  {delta_str:>9}")
    log("")

    # ---- Headline verdict ------------------------------------------------
    log("=" * 78)
    log("HEADLINE VERDICT")
    log("=" * 78)

    a_std_main = alpha_summary["STD"].get(0.10, float("nan"))
    a_at_main = alpha_summary["PGD-AT"].get(0.10, float("nan"))
    log(f"  STD     alpha-hat (frac=0.10) = {a_std_main:.4f}")
    log(f"  PGD-AT  alpha-hat (frac=0.10) = {a_at_main:.4f}")
    log(f"  KS_STD : D={ks_summary['STD'][0]:.4f}, p={ks_summary['STD'][1]:.2e}")
    log(f"  KS_AT  : D={ks_summary['PGD-AT'][0]:.4f}, p={ks_summary['PGD-AT'][1]:.2e}")

    # robust hypothesis check across all three fractions
    supports_std_fat = all(
        (alpha_summary["STD"][f] == alpha_summary["STD"][f])  # not nan
        and alpha_summary["STD"][f] < 2.0
        for f in HILL_FRACS
    )
    supports_at_thin = all(
        (alpha_summary["PGD-AT"][f] == alpha_summary["PGD-AT"][f])
        and alpha_summary["PGD-AT"][f] > 3.0
        for f in HILL_FRACS
    )

    if supports_std_fat and supports_at_thin:
        verdict = ("SUPPORTED: STD margin distribution is Pareto-like with "
                   "alpha<2 across all tail fractions, while PGD-AT pushes "
                   "alpha>3 (near-exponential). AT specifically thins the "
                   "left tail.")
    elif supports_at_thin and not supports_std_fat:
        verdict = ("PARTIAL: PGD-AT yields a thin tail (alpha>3) but the STD "
                   "tail is not as heavy as alpha<2. AT does thin the tail "
                   "but the STD baseline is already not extremely heavy.")
    elif supports_std_fat and not supports_at_thin:
        verdict = ("PARTIAL: STD is Pareto-like (alpha<2) but PGD-AT does "
                   "NOT achieve alpha>3 across all fractions. AT improves "
                   "the tail but not all the way to near-exponential.")
    else:
        # check at least directional thinning
        try:
            directional = a_at_main > a_std_main + 0.5
        except Exception:
            directional = False
        if directional:
            verdict = ("WEAK: neither alpha<2 (STD) nor alpha>3 (AT) bounds "
                       "are met cleanly, but PGD-AT increases alpha-hat by "
                       ">0.5 over STD at frac=0.10, consistent with "
                       "directional tail thinning by AT.")
        else:
            verdict = ("REJECTED: PGD-AT does not visibly thin the left "
                       "tail of the margin distribution beyond STD. The "
                       "G6 power-law-to-exponential hypothesis is not "
                       "supported on F-MNIST/SmallCNN.")
    log("")
    log(verdict)
    log("")
    log(f"Output written to: {OUT_PATH}")
    fout.close()


if __name__ == "__main__":
    main()
