"""
H446 - SmoothAdv certified training (Salman et al., NeurIPS 2019).

Critique / why this matters (campaign-internal):
  Gap G2 in CAMPAIGN_GAP_MAP.md: certified defences are nearly absent. H117 IBP
  was a probe, H204/H268 used randomised smoothing only as an *eval* on
  standard models, and H327 ran a Gaussian-noise training run but never
  emitted a certified L2 radius. The campaign therefore still has zero
  *certified* accuracy numbers for any defended model. SmoothAdv (Salman 2019)
  combines Cohen 2019 noise-augmented (Gaussian) training with adversarial
  perturbations computed *on the smoothed classifier* (i.e., PGD whose loss is
  the soft-smooth cross-entropy averaged over K noise samples). The deployed
  classifier g(x) = argmax_c P(f(x+eta)=c), eta ~ N(0, sigma^2 I), comes with
  a closed-form L2 certificate R = sigma * Phi^{-1}(p_A) (Cohen Theorem 1).

  We therefore report:
    (a) clean acc of the smoothed classifier (M0=100 selection, M1=1000 est),
    (b) empirical PGD-Linf-ASR at the campaign's eps=0.1 (white-box on the
        underlying base classifier f, since g is deterministic at sampling
        budget M1; this matches H391's audit conventions),
    (c) certified L2 radii via Cohen 2019 certify():
            - n0 = 100 (selection)
            - n  = 1000 (estimation)
            - alpha = 0.001 (one-sided Clopper-Pearson lower bound)
        Reported at thresholds R in {0.0, 0.25, 0.5, 0.75, 1.0}.

  Controls (the actual scientific contrast):
    C0: vanilla SGD baseline (no noise, no adv) - reference for "no defence".
    C1: vanilla Cohen RS training (Gaussian noise augmentation, no adv)
        at sigma in {0.12, 0.25, 0.5}                       <-- Cohen 2019.
    C2: SmoothAdv with K=2 noise samples, m_train=2 PGD-on-smoothed steps,
        at sigma in {0.12, 0.25, 0.5}                       <-- Salman 2019.
    C3: SmoothAdv "deeper inner" - K=2, m_train=3 PGD-on-smoothed steps,
        at sigma=0.25 only (compute budget) - probes whether more inner
        steps help under the campaign's 6k / 10-epoch regime.

  Sigmas {0.12, 0.25, 0.5} are exactly the three sigma settings in Cohen 2019
  Table 2 / Salman 2019 Table 1.

Extra papers (>=2 required by brief):
  * Cohen, Rosenfeld, Kolter (2019) "Certified Adversarial Robustness via
    Randomized Smoothing", ICML 2019 - we reimplement Algorithm 1 (CERTIFY)
    verbatim: M0=100 / M1=1000 / alpha=0.001, with one-sided Clopper-Pearson
    via scipy.stats.beta if available, else Pearson-approx fallback.
  * Salman, Li, Razenshteyn, Zhang, Zhang, Bubeck, Yang (2019) "Provably
    Robust Deep Learning via Adversarially Trained Smoothed Classifiers",
    NeurIPS 2019 - the SmoothAdv objective itself. Inner attack maximises
    the soft-smoothed CE loss
            L_smooth(x, y) = -log E_eta softmax(f(x+eta))[y]
    via PGD with steps m_train on the L2 ball of radius eps_train = sigma
    (per Salman Sec. 3.2 "epsilon = sigma").
  * Zhai, Dan, Cao, Suggala, Goyal, Zhao, Wang, Ravikumar (2020) "MACER:
    Attack-free and Scalable Robust Training via Maximizing Certified
    Radius", ICLR 2020 - cited as the comparison anchor (h447 will run it).
  * Yang, Duchi, Murthy (2020) "Randomized Smoothing of All Shapes and Sizes",
    ICML 2020 - theoretical generalisation establishing that Cohen's L2 is
    tight (we rely on the Gaussian L2 case here).

Threat-model contrast - IMPORTANT for honest reporting:
  Cohen/Salman certificates are L2. The campaign eval is Linf eps=0.1. We
  report both: certified L2 radius (the defence's *native* threat model)
  AND empirical Linf PGD ASR (the campaign's standard threat). An Linf
  ball of radius 0.1 on 28x28 grayscale has L2 diameter <= 0.1 * sqrt(784)
  = 2.8 in the worst case, so a certified-L2 radius of e.g. 0.5 does NOT
  automatically translate to Linf=0.1 robustness; we discuss this in the
  verdict.

Config (campaign standard):
  N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9, wd=5e-4),
  SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.

No execution - this script is staged only. ASCII output.
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

# ---- config --------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
N_CERTIFY = 500            # subsample of test set for certification (M0+M1 cost)
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01

# Cohen 2019 CERTIFY hyperparameters (Algorithm 1)
N0_SELECT = 100            # M0 - selection batch
N1_ESTIM = 1000            # M1 - estimation batch
ALPHA_CERT = 0.001         # one-sided confidence
CERT_BATCH = 200           # noise-sample batch when estimating counts

# SmoothAdv training settings
K_TRAIN = 2                # noise samples per training adv step (Salman M=2)
M_TRAIN_DEFAULT = 2        # inner PGD-on-smoothed steps
EPS_TRAIN_FACTOR = 1.0     # eps_train = sigma * factor   (Salman Sec 3.2)

# certified radius thresholds to report acc at
RADIUS_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "fashion_mnist",
                   "h446_smoothadv_certified_output.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


# ---- Clopper-Pearson lower bound (Cohen 2019 Algorithm 1, line "LowerConfBound")
def lower_conf_bound(k, n, alpha):
    """One-sided Clopper-Pearson lower bound for Binomial(n, p), given k successes.

    Returns the lower bound on p such that
        P(Binomial(n, p_lower) >= k) <= alpha.
    Uses scipy.stats.beta.ppf if available (exact); otherwise a
    Wilson-score-style normal approximation as a fallback (less tight).
    """
    if k == 0:
        return 0.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(alpha, k, n - k + 1))
    except Exception:
        # Wilson lower bound (fallback - looser, still valid in spirit)
        from math import sqrt
        p_hat = k / n
        z = 3.0902  # ~one-sided 0.999
        denom = 1 + z * z / n
        centre = p_hat + z * z / (2 * n)
        spread = z * sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
        return max(0.0, (centre - spread) / denom)


# ---- inverse normal CDF (Phi^{-1}) without scipy if scipy missing
def phi_inv(p):
    """Inverse standard-normal CDF. Uses scipy.stats.norm.ppf if available,
    else the Beasley-Springer-Moro rational approximation."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:
        # Acklam's approximation
        a = [-3.969683028665376e+01,  2.209460984245205e+02,
             -2.759285104469687e+02,  1.383577518672690e+02,
             -3.066479806614716e+01,  2.506628277459239e+00]
        b = [-5.447609879822406e+01,  1.615858368580409e+02,
             -1.556989798598866e+02,  6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00,  2.938163982698783e+00]
        d = [7.784695709041462e-03,  3.224671290700398e-01,
             2.445134137142996e+00,  3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                   ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        if p <= phigh:
            q = p - 0.5
            r = q * q
            return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                   (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


# ---- Cohen 2019 Algorithm 1 (CERTIFY) ------------------------------------
@torch.no_grad()
def _count_class_predictions(model, x, n_samples, sigma, n_classes, batch):
    """Sample n_samples noisy versions of x, count predicted-class hits.

    Returns a length-n_classes numpy array of counts.
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    remaining = n_samples
    while remaining > 0:
        b = min(batch, remaining)
        x_rep = x.unsqueeze(0).expand(b, *x.shape).contiguous()
        noise = torch.randn_like(x_rep) * sigma
        logits = model((x_rep + noise).clamp(0, 1))
        preds = logits.argmax(1).cpu().numpy()
        for c in preds:
            counts[c] += 1
        remaining -= b
    return counts


@torch.no_grad()
def certify_one(model, x, sigma, n0, n1, alpha, n_classes, batch):
    """Cohen 2019 CERTIFY for a single sample x. Returns (pred, radius).
    pred = -1 -> ABSTAIN.
    """
    # Selection phase
    c0 = _count_class_predictions(model, x, n0, sigma, n_classes, batch)
    c_A = int(c0.argmax())
    # Estimation phase
    c1 = _count_class_predictions(model, x, n1, sigma, n_classes, batch)
    n_A = int(c1[c_A])
    p_A = lower_conf_bound(n_A, n1, alpha)
    if p_A <= 0.5:
        return -1, 0.0
    radius = sigma * phi_inv(p_A)
    return c_A, float(radius)


@torch.no_grad()
def certify_dataset(model, X, Y, sigma, n0=N0_SELECT, n1=N1_ESTIM,
                    alpha=ALPHA_CERT, n_classes=10, batch=CERT_BATCH):
    """Run CERTIFY across (X, Y). Returns dict with:
       preds, radii (np arrays of length N), and acc-at-R for each R in
       RADIUS_THRESHOLDS (certified accuracy = fraction with correct
       prediction AND radius >= R; abstains count as wrong)."""
    model.eval()
    N = X.size(0)
    preds = np.full(N, -1, dtype=np.int64)
    radii = np.zeros(N, dtype=np.float64)
    for i in range(N):
        p, r = certify_one(model, X[i], sigma, n0, n1, alpha, n_classes, batch)
        preds[i] = p
        radii[i] = r
    Y_np = Y.cpu().numpy()
    correct = (preds == Y_np)
    acc_at_R = {R: float(((radii >= R) & correct).mean()) for R in RADIUS_THRESHOLDS}
    abstain_rate = float((preds == -1).mean())
    return {
        "preds": preds, "radii": radii,
        "acc_at_R": acc_at_R,
        "abstain": abstain_rate,
        "mean_radius_correct": float(radii[correct].mean()) if correct.any() else 0.0,
        "median_radius_correct": float(np.median(radii[correct])) if correct.any() else 0.0,
    }


# ---- SmoothAdv inner attack ---------------------------------------------
def pgd_on_smoothed(model, x, y, sigma, K, eps, steps, alpha):
    """PGD that maximises soft-smoothed cross-entropy:
       L(x) = -log E_{eta~N(0,sigma^2 I)} softmax(f(x+eta))[y]
    L2-constrained perturbation with budget eps, steps `steps`, step `alpha`.
    Follows Salman 2019 Sec 3.2 (their "SmoothAdv_PGD").
    """
    x0 = x.clone().detach()
    delta = torch.zeros_like(x0).normal_(0, 1e-3).requires_grad_(True)
    for _ in range(steps):
        # average softmax over K noise samples
        b = x0.size(0)
        x_rep = (x0 + delta).unsqueeze(1).expand(b, K, *x0.shape[1:]).contiguous()
        x_rep = x_rep.view(b * K, *x0.shape[1:])
        noise = torch.randn_like(x_rep) * sigma
        logits = model((x_rep + noise).clamp(0, 1))
        probs = F.softmax(logits, dim=1).view(b, K, -1).mean(dim=1)
        # log of mean prob of true class
        true_p = probs.gather(1, y[:, None]).squeeze(1).clamp_min(1e-12)
        loss = -torch.log(true_p).mean()
        g, = torch.autograd.grad(loss, delta)
        # L2 step
        g_flat = g.view(b, -1)
        g_norm = g_flat.norm(dim=1).clamp_min(1e-12)
        step = alpha * g / g_norm.view(b, *([1] * (g.dim() - 1)))
        delta = (delta.detach() + step)
        # project onto L2 ball of radius eps
        d_flat = delta.view(b, -1)
        d_norm = d_flat.norm(dim=1).clamp_min(1e-12)
        factor = torch.minimum(torch.ones_like(d_norm), eps / d_norm)
        delta = (d_flat * factor[:, None]).view_as(x0)
        # keep in image domain
        delta = ((x0 + delta).clamp(0, 1) - x0).detach().requires_grad_(True)
    return (x0 + delta.detach()).clamp(0, 1)


# ---- training conditions -------------------------------------------------
def train_condition(model, Xtr, Ytr, sigma, mode, K=K_TRAIN, m=M_TRAIN_DEFAULT):
    """mode in {"baseline", "rs", "smoothadv"}.
       sigma applies for rs / smoothadv. For smoothadv, eps_train = sigma."""
    C.set_seed(SEED)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    eps_train = sigma * EPS_TRAIN_FACTOR
    alpha_train = 2.0 * eps_train / max(1, m)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            if mode == "smoothadv" and sigma > 0:
                # inner: PGD-on-smoothed (model in eval for stable BN during attack)
                model.eval()
                x_adv = pgd_on_smoothed(model, xb, yb, sigma=sigma, K=K,
                                        eps=eps_train, steps=m, alpha=alpha_train)
                model.train()
                # outer: smoothed CE on adv input, K noise samples
                b = x_adv.size(0)
                x_rep = x_adv.unsqueeze(1).expand(b, K, *x_adv.shape[1:]).contiguous()
                x_rep = x_rep.view(b * K, *x_adv.shape[1:])
                noise = torch.randn_like(x_rep) * sigma
                logits = model((x_rep + noise).clamp(0, 1))
                logits = logits.view(b, K, -1).mean(dim=1)
                loss = F.cross_entropy(logits, yb)
            elif mode == "rs" and sigma > 0:
                # Cohen RS training: K noise samples, averaged loss
                b = xb.size(0)
                x_rep = xb.unsqueeze(1).expand(b, K, *xb.shape[1:]).contiguous()
                x_rep = x_rep.view(b * K, *xb.shape[1:])
                noise = torch.randn_like(x_rep) * sigma
                logits = model((x_rep + noise).clamp(0, 1))
                logits = logits.view(b, K, -1).mean(dim=1)
                loss = F.cross_entropy(logits, yb)
            else:
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- empirical eval (Linf PGD, campaign threat model) --------------------
def empirical_eval(model, Xte, Yte):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    margin_mean = float(np.mean(C.margin(model, Xte, Yte)))
    return {
        "clean_acc": float(clean_acc),
        "fgsm_asr": 1.0 - float(acc_fgsm),
        "pgd_asr": 1.0 - float(acc_pgd),
        "mean_margin": margin_mean,
    }


# ---- driver --------------------------------------------------------------
def fmt_acc_at_R(d):
    return ", ".join(f"R>={R}:{d[R]:.3f}" for R in RADIUS_THRESHOLDS)


def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # subsample for certify (expensive: M0+M1 = 1100 fwd passes per point)
    g = torch.Generator().manual_seed(SEED)
    cidx = torch.randperm(Xte.size(0), generator=g)[:N_CERTIFY]
    Xc, Yc = Xte[cidx], Yte[cidx]

    lines = []
    def emit(msg):
        print(msg, flush=True)
        lines.append(msg + "\n")
        with open(OUT, "w") as f:
            f.writelines(lines)

    emit("H446 - SmoothAdv certified training (Salman 2019) on Fashion-MNIST")
    emit("=" * 76)
    emit(f"Config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} N_CERTIFY={N_CERTIFY}")
    emit(f"        EPOCHS={EPOCHS} LR={LR} BATCH={BATCH} SEED={SEED}")
    emit(f"        EPS_Linf={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    emit(f"Cert:   N0={N0_SELECT} N1={N1_ESTIM} alpha={ALPHA_CERT}")
    emit(f"Train:  K_train={K_TRAIN} m_train_default={M_TRAIN_DEFAULT}"
         f" eps_train_factor={EPS_TRAIN_FACTOR}")
    emit("Threat-model note: certificates are L2 (Cohen 2019). Empirical PGD is Linf eps=0.1.")
    emit("-" * 76)

    # Conditions: (name, mode, sigma, m_train)
    conditions = [
        ("C0_baseline_no_defence",            "baseline",  0.0,  0),
        ("C1_RS_Cohen_sigma=0.12",            "rs",        0.12, 0),
        ("C1_RS_Cohen_sigma=0.25",            "rs",        0.25, 0),
        ("C1_RS_Cohen_sigma=0.50",            "rs",        0.50, 0),
        ("C2_SmoothAdv_sigma=0.12_m=2",       "smoothadv", 0.12, 2),
        ("C2_SmoothAdv_sigma=0.25_m=2",       "smoothadv", 0.25, 2),
        ("C2_SmoothAdv_sigma=0.50_m=2",       "smoothadv", 0.50, 2),
        ("C3_SmoothAdv_sigma=0.25_m=3",       "smoothadv", 0.25, 3),
    ]

    summary_rows = []
    for (name, mode, sigma, m) in conditions:
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        train_condition(model, Xtr, Ytr, sigma=sigma, mode=mode,
                        K=K_TRAIN, m=(m if m > 0 else M_TRAIN_DEFAULT))
        train_s = time.time() - t0

        # Empirical (base classifier, Linf)
        emp = empirical_eval(model, Xte, Yte)

        # Certified (smoothed classifier) - only when sigma > 0
        if sigma > 0:
            t1 = time.time()
            cert = certify_dataset(model, Xc, Yc, sigma=sigma)
            cert_s = time.time() - t1
        else:
            cert = None
            cert_s = 0.0

        emit("")
        emit(f"[{name}]  train={train_s:.1f}s  cert={cert_s:.1f}s")
        emit(f"  empirical (Linf eps={EPS}):"
             f"  clean={emp['clean_acc']:.4f}"
             f"  fgsm_asr={emp['fgsm_asr']:.4f}"
             f"  pgd_asr={emp['pgd_asr']:.4f}"
             f"  margin={emp['mean_margin']:.4f}")
        if cert is not None:
            emit(f"  certified L2 (sigma={sigma}):"
                 f"  abstain={cert['abstain']:.3f}"
                 f"  mean_R(correct)={cert['mean_radius_correct']:.4f}"
                 f"  median_R(correct)={cert['median_radius_correct']:.4f}")
            emit(f"  certified acc-at-R: {fmt_acc_at_R(cert['acc_at_R'])}")
        else:
            emit("  certified L2: N/A (sigma=0)")

        row = {
            "name": name, "mode": mode, "sigma": sigma, "m_train": m,
            "clean": emp["clean_acc"], "fgsm_asr": emp["fgsm_asr"],
            "pgd_asr": emp["pgd_asr"], "margin": emp["mean_margin"],
            "cert_acc_at_R": (cert["acc_at_R"] if cert else None),
            "cert_mean_R": (cert["mean_radius_correct"] if cert else None),
            "abstain": (cert["abstain"] if cert else None),
        }
        summary_rows.append(row)

    # ----- compact summary table -----
    emit("")
    emit("=" * 76)
    emit("SUMMARY")
    emit("-" * 76)
    header = (f"{'condition':<32} {'clean':>6} {'pgd_asr':>8} "
              f"{'cR=0.0':>7} {'cR=0.25':>8} {'cR=0.5':>7} {'cR=0.75':>8} {'cR=1.0':>7}")
    emit(header)
    for r in summary_rows:
        if r["cert_acc_at_R"] is None:
            ca = ["  -  "] * len(RADIUS_THRESHOLDS)
        else:
            ca = [f"{r['cert_acc_at_R'][R]:.3f}" for R in RADIUS_THRESHOLDS]
        emit(f"{r['name']:<32} {r['clean']:>6.3f} {r['pgd_asr']:>8.3f} "
             f"{ca[0]:>7} {ca[1]:>8} {ca[2]:>7} {ca[3]:>8} {ca[4]:>7}")

    # ----- verdict -----
    emit("")
    emit("=" * 76)
    emit("VERDICT (rubric)")
    emit("-" * 76)
    emit("PRIMARY claim under test: SmoothAdv (Salman 2019) produces a model with")
    emit("a non-trivial CERTIFIED L2 accuracy at moderate radii (R >= 0.25), and")
    emit("dominates vanilla Cohen RS at matched sigma on the certified-acc-at-R")
    emit("curve.")
    emit("")
    emit("SUPPORTED iff: for at least one sigma in {0.12,0.25,0.5}, SmoothAdv has")
    emit("  (a) certified acc at R=0.25 > 0.30, AND")
    emit("  (b) certified acc at R=0.25 strictly > matched-sigma Cohen RS.")
    emit("PARTIAL  iff: (a) holds but (b) fails - Cohen RS already captures the gain.")
    emit("NOT SUPPORTED iff: certified acc at R=0.25 < 0.20 for all SmoothAdv runs.")
    emit("")
    emit("SECONDARY (Linf transfer): the campaign's eps=0.1 Linf PGD ASR is")
    emit("reported on the *base* (non-smoothed) network. SmoothAdv is NOT designed")
    emit("for Linf - a low certified-L2 radius (e.g., R<0.1) is consistent with")
    emit("high Linf PGD ASR. Read both columns; the certified-L2 column is the")
    emit("scientifically primary one and the campaign's first such number.")
    emit("")
    emit("CAVEATS:")
    emit("  * Single seed (SEED=0), N_CERTIFY=500 - certified acc has Clopper-")
    emit("    Pearson built-in but the test subset size still bounds resolution to")
    emit("    ~0.04 absolute.")
    emit("  * 10-epoch / 6k-train budget is below Cohen/Salman's CIFAR-10 setting;")
    emit("    radii are expected to be smaller than their Table 1 numbers.")
    emit("  * eps_train = sigma (Salman default). A finer sweep is left to h447 /")
    emit("    MACER follow-up.")
    emit("")
    emit("End of H446.")


if __name__ == "__main__":
    main()
