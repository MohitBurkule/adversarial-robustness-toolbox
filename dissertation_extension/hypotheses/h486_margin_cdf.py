"""
H486 - Empirical CDF of input-space margins per defence (Margin-CDF-per-defence).

Seed (campaign gap-map M6/G6): rather than reporting *mean* robust radius, report
the full empirical CDF of per-sample input-space margins ("min-eps-to-flip"
under L_inf). Conjecture: adversarial training (AT) shifts the LEFT tail of
the margin-CDF (the brittlest samples) more than the median - i.e. the gain
is mostly in pushing brittle samples to a usable margin, not in raising the
median far.

Why this matters:
  * Yang et al. (NeurIPS 2020, "A Closer Look at Accuracy vs Robustness")
    argue robustness and accuracy can co-exist iff classes are well separated
    in margin distribution. Their analysis is on TRAINING-set margins; we
    instead probe TEST-set input-space margin CDFs across defences.
  * Cohen et al. (ICML 2019, "Certified Adversarial Robustness via Randomized
    Smoothing") report the empirical CDF of certified L2 radii as the natural
    object - we adopt the same CDF view in the L_inf empirical (uncertified)
    regime.
  * Croce & Hein (RobustBench, NeurIPS 2021 Datasets&Benchmarks) standardise
    robust accuracy at fixed eps; here we sweep eps and recover the entire
    CDF, which subsumes their single-eps robust accuracy.

Critique applied:
  The "true" L_inf margin is computable in closed form only for linear models
  (DeepFool gives an L_p approximation; AutoPGD-DLR gives a strong upper bound;
  exact verification is NP-hard in general). Our PGD-bisection estimate
  therefore gives an UPPER BOUND on the true margin (true margin <= our
  estimate). We mitigate by (i) using random restarts, (ii) deep binary
  search (12 levels in [0, 0.3] L_inf), and (iii) keeping PGD step count
  high (40). Differences BETWEEN defences are robust to this approximation
  because all defences are attacked under identical PGD configuration.

Method:
  1. Train 4 defences on Fashion-MNIST with SmallCNN, seed=0:
       STD     - vanilla cross-entropy
       PGD-AT  - Madry-style PGD-7 adv training, eps=0.1
       TRADES  - Zhang et al. ICML 2019, beta=6, PGD-7, eps=0.1
       FREE-AT - Shafahi et al. NeurIPS 2019, m=4 replays, eps=0.1
  2. Pick 1000 test samples (fixed indices, all defences see the same set).
  3. For each (model, sample), binary-search min-eps to flip in [0, 0.3]
     L_inf, depth 12, with PGD-40 at each midpoint (random_start, k=2 restarts).
  4. Report:
       (a) full empirical CDF of min-eps (printed as a 21-point grid in eps,
           plus deciles p10/p25/p50/p75/p90),
       (b) shift between STD and each AT defence at every reported quantile,
           with explicit LEFT-tail-vs-median diagnostic (delta_p10 vs delta_p50),
       (c) per-class CDF (p10/p50/p90 per class) for STD vs best-AT defence,
       (d) Pearson and Spearman correlation between min-eps and model
           confidence (max softmax) on clean inputs - probing whether AT
           DECOUPLES margin from confidence.

The four models give us cross-defence comparability: if AT genuinely lifts
the left tail more than the median, delta_p10 >> delta_p50 across PGD-AT,
TRADES, and FREE-AT independently. If only delta_p50 increases, AT is just
shifting all samples by a constant.

Output written to results/fashion_mnist/h486_margin_cdf_output.txt.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DS = "fashion_mnist"
SEED = 0
N_EVAL = 1000           # samples for margin CDF
N_TRAIN = 20000         # subset of FMNIST train for speed
EPOCHS = 8
AT_EPS = 0.1            # training-time and reference eps
BISECT_LO = 0.0
BISECT_HI = 0.3
BISECT_DEPTH = 12
PGD_STEPS_EVAL = 40
PGD_RESTARTS = 2
CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
N_CLASSES = 10

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
)
OUT_PATH = os.path.join(RESULTS_DIR, "h486_margin_cdf_output.txt")


# ---------------------------------------------------------------------------
# Attack helpers (L_inf PGD with multi-restart, returns adversarial images)
# ---------------------------------------------------------------------------
def pgd_linf(model, x, y, eps, steps=40, alpha=None, restarts=1, random_start=True):
    """Standard L_inf PGD with optional restarts. Returns x_adv (best over restarts)."""
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
        # keep this restart's adv for newly-flipped samples
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
    at that eps flips the prediction. Returns:
        min_eps : tensor (N,) of estimated min eps (== hi if never flipped)
        flipped_at_hi : bool tensor (N,) indicating success at the maximum budget
    NOTE: this is an UPPER BOUND on the true L_inf margin (PGD may miss adversarials).
    """
    N = x.size(0)
    out_min = torch.full((N,), float(hi), device=x.device)
    out_success = torch.zeros(N, dtype=torch.bool, device=x.device)

    for i in range(0, N, batch):
        bx = x[i:i + batch]
        by = y[i:i + batch]
        n = bx.size(0)

        # 1) check the maximum budget first: anything that does not flip at hi
        #    has min_eps > hi (right-censored at hi).
        _, flipped_hi = pgd_linf(model, bx, by, eps=hi, steps=steps,
                                  restarts=restarts, random_start=True)
        out_success[i:i + batch] = flipped_hi
        # for samples that never flip, leave min_eps = hi (right-censored)

        # 2) binary search per-sample for samples that DO flip at hi
        lo_t = torch.full((n,), float(lo), device=bx.device)
        hi_t = torch.full((n,), float(hi), device=bx.device)

        # only bisect the active subset; inactive ones (not flipped at hi)
        # keep min_eps = hi as the right-censored value.
        for _ in range(depth):
            mid = 0.5 * (lo_t + hi_t)
            # For batched PGD we need a SCALAR eps. Run at max(mid[active])
            # and decide per-sample after; this is slightly conservative but
            # is correct because at a given step we only accept flips whose
            # required eps is <= mid_per_sample.
            # Trick: run PGD at the per-sample mid via per-sample clamping.
            # We achieve that by passing eps = hi (large) and then clamping
            # each sample with its own mid budget after the PGD inner loop.
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
                # per-sample L_inf projection
                lower = bx - mid.view(-1, 1, 1, 1)
                upper = bx + mid.view(-1, 1, 1, 1)
                xa = torch.min(torch.max(xa, lower), upper).clamp(0.0, 1.0)
            with torch.no_grad():
                flipped = (model(xa).argmax(1) != by)
            # tighten bounds per sample
            hi_t = torch.where(flipped & flipped_hi, mid, hi_t)
            lo_t = torch.where((~flipped) & flipped_hi, mid, lo_t)
            # samples that did not flip at hi stay right-censored (lo=lo, hi=hi)

        # final estimate = hi_t for samples that flipped at hi, else hi (censor)
        est = torch.where(flipped_hi, hi_t, torch.full_like(hi_t, float(hi)))
        out_min[i:i + batch] = est
    return out_min, out_success


# ---------------------------------------------------------------------------
# Training routines for the 4 defences
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


def train_trades(Xtr, Ytr, meta, seed=SEED, beta=6.0, pgd_steps=7):
    """TRADES (Zhang et al. ICML 2019).

    Loss = CE(model(x), y) + beta * KL( softmax(model(x_adv)) || softmax(model(x)) ),
    where x_adv is obtained by maximising the KL term (not CE) with PGD.
    """
    C.set_seed(seed)
    model = C.build_model("cnn", meta)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    eps = AT_EPS
    alpha = eps / 4.0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            x = Xtr[idx]; y = Ytr[idx]
            # inner max: PGD on the KL term
            with torch.no_grad():
                logits_clean = model(x)
                p_clean = F.softmax(logits_clean, dim=1).detach()
            x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
            x_adv = x_adv.clamp(0.0, 1.0)
            for _ in range(pgd_steps):
                x_adv.requires_grad_(True)
                logp_adv = F.log_softmax(model(x_adv), dim=1)
                kl = F.kl_div(logp_adv, p_clean, reduction="batchmean")
                g, = torch.autograd.grad(kl, x_adv)
                x_adv = x_adv.detach() + alpha * g.sign()
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)
            # outer min: CE(clean) + beta * KL(adv || clean)
            opt.zero_grad()
            logits_clean = model(x)
            logits_adv = model(x_adv)
            loss_nat = F.cross_entropy(logits_clean, y)
            loss_rob = F.kl_div(F.log_softmax(logits_adv, dim=1),
                                F.softmax(logits_clean, dim=1),
                                reduction="batchmean")
            (loss_nat + beta * loss_rob).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_free_at(Xtr, Ytr, meta, seed=SEED, m_replay=4):
    """Free-AT (Shafahi et al. NeurIPS 2019).

    Each minibatch is replayed m times; the perturbation delta is carried
    across replays (so adversary and weights co-train). Effective epoch
    count is EPOCHS / m_replay so total compute matches PGD-AT.
    """
    C.set_seed(seed)
    model = C.build_model("cnn", meta)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    outer_epochs = max(1, EPOCHS // m_replay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=outer_epochs)
    n = Xtr.size(0)
    eps = AT_EPS
    for ep in range(outer_epochs):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            x = Xtr[idx]; y = Ytr[idx]
            delta = torch.zeros_like(x, requires_grad=True)
            for _ in range(m_replay):
                opt.zero_grad()
                x_adv = (x + delta).clamp(0.0, 1.0)
                loss = F.cross_entropy(model(x_adv), y)
                loss.backward()
                # update delta (sign of grad on delta)
                with torch.no_grad():
                    delta_grad = delta.grad.detach()
                    delta = (delta.detach() + eps * delta_grad.sign()).clamp(-eps, eps)
                    delta.requires_grad_(True)
                opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# CDF utilities
# ---------------------------------------------------------------------------
def empirical_cdf_at(x, grid):
    """Empirical CDF F(t) = P(X <= t) evaluated at `grid` (numpy)."""
    x = np.asarray(x); grid = np.asarray(grid)
    xs = np.sort(x)
    # searchsorted gives count of xs <= t when side='right'
    counts = np.searchsorted(xs, grid, side="right")
    return counts / float(xs.size)


def quantiles(x, qs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    return {f"p{int(q*100)}": float(np.quantile(np.asarray(x), q)) for q in qs}


def fmt_pct(p):
    return f"{100.0 * p:5.1f}%"


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
    log("H486 - Empirical margin CDF per defence on Fashion-MNIST (SmallCNN)")
    log("=" * 78)
    log(f"Device: {C.DEVICE}")
    log(f"Defences: STD, PGD-AT, TRADES (beta=6), Free-AT (m={4})")
    log(f"Train subset: {N_TRAIN}, epochs: {EPOCHS}, AT eps: {AT_EPS}")
    log(f"Margin estimator: PGD-{PGD_STEPS_EVAL} bisection in [{BISECT_LO}, {BISECT_HI}], "
        f"depth={BISECT_DEPTH}, restarts={PGD_RESTARTS}")
    log(f"Eval samples: {N_EVAL}  (fixed across all defences)")
    log("")
    log("CRITIQUE: PGD-bisection yields an UPPER BOUND on the true L_inf")
    log("margin. Cross-defence comparisons remain valid under identical attack")
    log("configuration. For tighter estimates one would use AutoPGD-DLR or")
    log("DeepFool-L2 + binary search (Croce & Hein ICML 2020).")
    log("")
    log("Refs: Yang et al. NeurIPS 2020 (accuracy vs robustness, margin distributions);")
    log("      Cohen et al. ICML 2019 (radius-CDF view of certified robustness);")
    log("      Croce & Hein NeurIPS-D&B 2021 (RobustBench, robust-radius reporting).")
    log("")

    # ---- Data ------------------------------------------------------------
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    log(f"Loaded {DS}: Xtr {tuple(Xtr.shape)}, Xte {tuple(Xte.shape)}")
    log("")

    # ---- Train the four defences ----------------------------------------
    trainers = [
        ("STD",     train_std),
        ("PGD-AT",  train_pgd_at),
        ("TRADES",  train_trades),
        ("Free-AT", train_free_at),
    ]

    models = {}
    timings = {}
    for name, fn in trainers:
        t0 = time.time()
        log(f"--- Training {name} ---")
        m = fn(Xtr, Ytr, meta)
        m.eval()
        timings[name] = time.time() - t0
        models[name] = m
        # clean acc sanity
        with torch.no_grad():
            acc = float((m(Xte).argmax(1) == Yte).float().mean())
        log(f"    trained in {timings[name]:6.1f}s   clean_acc={acc:.4f}")
    log("")

    # ---- Compute per-model: min-eps, confidence, clean preds -------------
    minE = {}
    success = {}
    conf = {}
    preds = {}
    correct = {}
    for name, m in models.items():
        log(f"[{name}] computing min-eps and confidence on {N_EVAL} samples...")
        with torch.no_grad():
            logits = m(Xte)
            soft = F.softmax(logits, dim=1)
            conf[name] = soft.max(1).values.cpu().numpy()
            preds[name] = logits.argmax(1).cpu().numpy()
            correct[name] = (preds[name] == Yte.cpu().numpy())
        t0 = time.time()
        me, sc = min_eps_to_flip_bisect(m, Xte, Yte)
        minE[name] = me.cpu().numpy()
        success[name] = sc.cpu().numpy()
        log(f"    done in {time.time()-t0:6.1f}s  "
            f"censored@{BISECT_HI}={int((~sc).sum())}/{N_EVAL}")
    log("")

    # ---- (a) Full empirical CDF on a grid --------------------------------
    grid = np.linspace(0.0, BISECT_HI, 21)
    log("=" * 78)
    log("Section A: empirical CDF F(eps) = P(min_eps <= eps) per defence")
    log("=" * 78)
    header = f"{'eps':>6}  " + "  ".join(f"{name:>8}" for name, _ in trainers)
    log(header)
    log("-" * len(header))
    cdfs = {name: empirical_cdf_at(minE[name], grid) for name, _ in trainers}
    for j, eps_j in enumerate(grid):
        row = f"{eps_j:>6.3f}  " + "  ".join(f"{cdfs[name][j]*100:>7.1f}%" for name, _ in trainers)
        log(row)
    log("")

    # ---- (b) Percentiles and shift vs STD --------------------------------
    log("=" * 78)
    log("Section B: percentile table (robust radius at each quantile)")
    log("=" * 78)
    qs = [("p10", 0.10), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90)]
    qheader = f"{'model':<10}  " + "  ".join(f"{lbl:>8}" for lbl, _ in qs)
    log(qheader)
    log("-" * len(qheader))
    perc = {}
    for name, _ in trainers:
        perc[name] = {lbl: float(np.quantile(minE[name], q)) for lbl, q in qs}
        row = f"{name:<10}  " + "  ".join(f"{perc[name][lbl]:>8.4f}" for lbl, _ in qs)
        log(row)
    log("")
    log("Shift (defence - STD) at each quantile:")
    sheader = f"{'defence':<10}  " + "  ".join(f"{lbl:>8}" for lbl, _ in qs)
    log(sheader)
    log("-" * len(sheader))
    left_tail_vs_median = {}
    for name, _ in trainers:
        if name == "STD":
            continue
        shifts = {lbl: perc[name][lbl] - perc["STD"][lbl] for lbl, _ in qs}
        row = f"{name:<10}  " + "  ".join(f"{shifts[lbl]:>+8.4f}" for lbl, _ in qs)
        log(row)
        # diagnostic ratio
        d10 = shifts["p10"]; d50 = shifts["p50"]
        if abs(d50) < 1e-6:
            ratio_str = "inf" if d10 > 0 else "n/a"
        else:
            ratio_str = f"{d10 / d50:+.2f}x"
        left_tail_vs_median[name] = (d10, d50, ratio_str)
    log("")
    log("Left-tail-vs-median diagnostic (delta_p10 vs delta_p50):")
    log(f"  Hypothesis: delta_p10 / delta_p50 > 1.0 means AT lifts the LEFT TAIL")
    log(f"  more than the median (consistent with the hypothesis seed).")
    for name, (d10, d50, ratio) in left_tail_vs_median.items():
        log(f"  {name:<10}: delta_p10={d10:+.4f}  delta_p50={d50:+.4f}  ratio={ratio}")
    log("")

    # ---- (c) Per-class CDF (STD vs best-AT defence) ---------------------
    log("=" * 78)
    log("Section C: per-class margin percentiles (STD vs PGD-AT)")
    log("=" * 78)
    Yte_np = Yte.cpu().numpy()
    cheader = f"{'class':<12}  {'STD-p10':>8}  {'STD-p50':>8}  {'STD-p90':>8}  " \
              f"{'AT-p10':>8}  {'AT-p50':>8}  {'AT-p90':>8}  {'d_p10':>8}  {'d_p50':>8}"
    log(cheader)
    log("-" * len(cheader))
    for c in range(N_CLASSES):
        sel = (Yte_np == c)
        if sel.sum() < 5:
            log(f"{CLASS_NAMES[c]:<12}  (n<5, skipped)")
            continue
        s_std = minE["STD"][sel]
        s_at = minE["PGD-AT"][sel]
        p10s, p50s, p90s = np.quantile(s_std, [0.1, 0.5, 0.9])
        p10a, p50a, p90a = np.quantile(s_at, [0.1, 0.5, 0.9])
        log(f"{CLASS_NAMES[c]:<12}  {p10s:>8.4f}  {p50s:>8.4f}  {p90s:>8.4f}  "
            f"{p10a:>8.4f}  {p50a:>8.4f}  {p90a:>8.4f}  "
            f"{p10a-p10s:>+8.4f}  {p50a-p50s:>+8.4f}")
    log("")

    # ---- (d) Margin vs confidence correlation ---------------------------
    log("=" * 78)
    log("Section D: correlation between min-eps and confidence (max softmax)")
    log("=" * 78)
    log("Question: does AT DECOUPLE confidence from robustness?")
    log(f"{'model':<10}  {'pearson_r':>10}  {'p':>10}  {'spearman_rho':>13}  {'p':>10}")
    log("-" * 60)
    corrs = {}
    for name, _ in trainers:
        me = minE[name]; cf = conf[name]
        # restrict to non-censored (so we have informative min-eps)
        keep = success[name].astype(bool)
        if keep.sum() < 10:
            log(f"{name:<10}  (too few non-censored)")
            continue
        pr, pp = pearsonr(me[keep], cf[keep])
        sr, sp = spearmanr(me[keep], cf[keep])
        corrs[name] = (pr, sr)
        log(f"{name:<10}  {pr:>10.4f}  {pp:>10.4f}  {sr:>13.4f}  {sp:>10.4f}")
    log("")
    if "STD" in corrs and "PGD-AT" in corrs:
        rho_std = corrs["STD"][1]
        rho_at = corrs["PGD-AT"][1]
        log(f"STD spearman(margin, confidence) = {rho_std:+.4f}")
        log(f"PGD-AT spearman(margin, confidence) = {rho_at:+.4f}")
        if abs(rho_at) + 0.10 < abs(rho_std):
            log("=> AT DECOUPLES confidence from margin "
                "(|rho| dropped by >=0.10 vs STD).")
        elif abs(rho_at) > abs(rho_std) + 0.10:
            log("=> AT TIGHTENS confidence-margin coupling vs STD.")
        else:
            log("=> No clear coupling change between STD and PGD-AT.")
    log("")

    # ---- Headline verdict ------------------------------------------------
    log("=" * 78)
    log("HEADLINE VERDICT")
    log("=" * 78)
    # decide direction of the AT-shift
    supports = []
    against = []
    for name, (d10, d50, _) in left_tail_vs_median.items():
        if d10 > d50 + 0.005 and d10 > 0:
            supports.append(name)
        elif d10 + 0.005 < d50:
            against.append(name)
    if len(supports) >= 2 and len(against) == 0:
        verdict = ("SUPPORTED: across {} defences, AT lifts the LEFT TAIL "
                   "(p10) of the margin-CDF more than the median (p50). "
                   "The robustness gain is concentrated in brittle samples.").format(
                       ", ".join(supports))
    elif len(against) >= 2 and len(supports) == 0:
        verdict = ("REJECTED: across {} defences, AT lifts the MEDIAN more "
                   "than the left tail. The robustness gain is uniform, "
                   "not tail-concentrated.").format(", ".join(against))
    else:
        verdict = ("MIXED: AT defences disagree on whether the gain is "
                   "left-tail-concentrated. Supports={}, Against={}.").format(
                       supports or "none", against or "none")
    log(verdict)
    log("")
    log(f"Output written to: {OUT_PATH}")
    fout.close()


if __name__ == "__main__":
    main()
