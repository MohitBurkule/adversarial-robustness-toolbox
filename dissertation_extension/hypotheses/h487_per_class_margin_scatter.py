"""
Hypothesis H487 - Per-class scatter of (mean-margin, robust-acc): does within-class
margin VARIANCE predict per-class adversarial robustness?

Seed (Section 5 / Section 3 / Goal G8 / Method M6).

Hypothesis
----------
For a PGD-AT model on Fashion-MNIST, classes with higher *within-class margin
variance* have lower robust accuracy at eps=0.1 -- the intuition being that a
high-variance margin distribution has a long left tail of brittle samples that
PGD picks off first, even when the *mean* margin of the class is healthy.

Why this is not trivial
-----------------------
Per-class clean accuracy and class fairness in adversarial training are well
documented (Xu et al., "To Be Robust or To Be Fair: Towards Fairness in
Adversarial Training", ICML 2021; Tian et al., "Analysis and Applications of
Class-wise Robustness of Adversarial Training", KDD 2021). Both papers identify
PERSISTENT class disparities under AT and link them mostly to inter-class
geometry / class hardness, with margin MEAN as the usual proxy. The H487 seed
goes further: it asks whether the *second moment* (and *third*, via skew) of
the within-class margin distribution carries additional predictive signal --
i.e. whether the LEFT TAIL of a class's margin histogram drives its robust-acc
beyond what the class's mean margin already explains.

Critique-driven controls
------------------------
(C1) Margin variance is naturally correlated with margin mean (heteroskedasticity:
     classes with larger separation tend also to have wider spread). We therefore
     compute Pearson AND Spearman correlations of (margin_mean, margin_std,
     margin_skew) against per-class robust-acc, AND a PARTIAL Pearson correlation
     of (margin_std vs robust-acc) controlling for both margin_mean and clean-acc.
     This is the load-bearing control.

(C2) Fashion-MNIST class 6 ("Shirt") is the canonical hard class -- it dominates
     confusion-matrix mass and could singlehandedly drive any correlation. We
     therefore report all stats both INCLUDING and EXCLUDING class 6 to bound
     its leverage. We also flag any class with Cook's-distance >= 4/n_classes
     under an OLS fit of robust-acc on margin_std.

(C3) Confounding with clean accuracy: a class that is hard CLEANLY will also be
     hard ADVERSARIALLY, so part of any margin_std -> robust-acc signal could be
     mediated by clean-acc. The partial correlation controls for this.

(C4) Model-dependence: a single STD or PGD-AT model could give us a spurious
     pattern. We train THREE models on the same SmallCNN backbone:
       * STD   -- standard cross-entropy
       * PGD-AT -- Madry-style L-inf PGD adversarial training (eps=0.1)
       * TRADES -- KL-regularised AT (Zhang et al. ICML 2019)
     and report the correlation table per model. The hypothesis is *about the
     PGD-AT model*; the other two are sanity checks: we want the (std, robust-acc)
     link to be STRONGER for the robust models than for STD (because for STD the
     robust-acc is near-zero for every class so there is no variance to predict).

(C5) Drill-down for the hardest-3 classes (by robust-acc, on the PGD-AT model):
     dump per-sample clean-margin histograms (text-mode quantile sketch) so we
     can SEE whether the failure mode is "low mean" or "fat left tail".

Citations (the four required by the seed)
-----------------------------------------
  [Xu2021]    H. Xu et al., "To Be Robust or To Be Fair: Towards Fairness in
              Adversarial Training", ICML 2021. arXiv:2010.06121.
              -- shows that PGD-AT systematically widens the class-wise
                 robust-accuracy gap; their fairness gap is between BEST and
                 WORST class robust-acc. We extend by asking what *feature* of
                 the per-class margin distribution predicts which class is worst.

  [Tian2021]  Q. Tian et al., "Analysis and Applications of Class-wise Robustness
              of Adversarial Training", KDD 2021.
              -- attributes class-wise robustness gaps to inter-class confusability
                 and visualises per-class boundary distances. Their per-class
                 metric is closely related to our `margin_mean`; we test whether
                 `margin_std` adds signal beyond `margin_mean`.

  [Madry2018] A. Madry et al., "Towards Deep Learning Models Resistant to
              Adversarial Attacks", ICLR 2018. -- PGD-AT recipe used here.

  [Zhang2019] H. Zhang et al., "Theoretically Principled Trade-off between
              Robustness and Accuracy", ICML 2019. -- TRADES recipe used here.

Outputs (printed to results/fashion_mnist/h487_per_class_margin_scatter_output.txt
when redirected on run)
----------------------
  * Per-class table: clean-acc, robust-acc(eps=0.1, PGD-20), margin_mean,
    margin_std, margin_skew, n_class -- one block per model (STD, PGD-AT, TRADES).
  * Correlation table: Pearson & Spearman of {mean, std, skew} vs robust-acc,
    per model, with-and-without class 6.
  * Partial correlation: partial Pearson r(margin_std, robust_acc | margin_mean,
    clean_acc) on the PGD-AT model -- the primary critique-driven control.
  * Hardest-3-classes drill-down: text-mode quantile sketch (P5/P25/P50/P75/P95)
    of per-sample clean margins.
  * HEADLINE verdict line at the very bottom.
"""
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DS = "fashion_mnist"
SEED = 0
N_TRAIN = 12000          # subset large enough to train all three models in <few-min
N_EVAL = 5000            # large eval pool so per-class counts (~500/class) are stable
EPS = 0.1                # L-inf budget (matches the rest of the F-MNIST campaign)
PGD_STEPS_TRAIN = 7
PGD_STEPS_EVAL = 20
EPOCHS = 8
TRADES_BETA = 6.0        # standard TRADES weighting

FMNIST_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------------------
# TRADES training (Zhang et al. ICML 2019)
# ---------------------------------------------------------------------------
def _pgd_for_trades(model, x, eps, steps, alpha):
    """KL-maximising inner attack for TRADES (no labels)."""
    model.eval()
    with torch.no_grad():
        logits_clean = model(x)
        p_clean = F.softmax(logits_clean, dim=1)
    x0 = x.clone().detach()
    xa = x0 + 0.001 * torch.randn_like(x0)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        loss = F.kl_div(F.log_softmax(model(xa), dim=1), p_clean, reduction="batchmean")
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    model.train()
    return xa.detach()


def train_trades(model, Xtr, Ytr, epochs=EPOCHS, batch=128, lr=0.05,
                 eps=EPS, steps=PGD_STEPS_TRAIN, beta=TRADES_BETA):
    """Madry-style optimiser, TRADES KL-regularised loss."""
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    alpha = 2.5 * eps / steps
    n = Xtr.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        model.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = _pgd_for_trades(model, xb, eps=eps, steps=steps, alpha=alpha)
            model.train()
            opt.zero_grad()
            logits_clean = model(xb)
            logits_adv = model(xa)
            loss_nat = F.cross_entropy(logits_clean, yb)
            loss_rob = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean, dim=1),
                reduction="batchmean",
            )
            loss = loss_nat + beta * loss_rob
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------
def per_class_stats(model, Xte, Yte, ncls, eps=EPS, steps=PGD_STEPS_EVAL):
    """For each class c, compute:
       clean_acc[c], robust_acc[c], margin_mean[c], margin_std[c], margin_skew[c],
       n_class[c], and the raw per-sample clean margins margins_by_class[c].
    """
    model.eval()
    # clean logits + per-sample margin
    logits_clean, _ = C.logits_and_acc(model, Xte, Yte)            # CPU tensor
    clean_pred = logits_clean.argmax(1)
    clean_correct = (clean_pred == Yte.cpu()).numpy().astype(bool)
    margins = C.margin_of(logits_clean, Yte)                       # numpy (N,)

    # robust accuracy: PGD-20 untargeted, eps in L-inf
    flips = []
    batch = 256
    for i in range(0, Xte.size(0), batch):
        xa = C.pgd(model, Xte[i:i + batch], Yte[i:i + batch], eps=eps, steps=steps)
        with torch.no_grad():
            pred_a = model(xa).argmax(1).cpu()
        flips.append((pred_a == Yte[i:i + batch].cpu()).numpy())  # 1=still correct
    robust_correct = np.concatenate(flips).astype(bool)

    Y_np = Yte.cpu().numpy()
    out = {
        "clean_acc": np.zeros(ncls),
        "robust_acc": np.zeros(ncls),
        "margin_mean": np.zeros(ncls),
        "margin_std": np.zeros(ncls),
        "margin_skew": np.zeros(ncls),
        "n_class": np.zeros(ncls, dtype=int),
        "margins_by_class": {},
    }
    for c in range(ncls):
        m = (Y_np == c)
        nc = int(m.sum())
        out["n_class"][c] = nc
        if nc == 0:
            continue
        out["clean_acc"][c] = float(clean_correct[m].mean())
        out["robust_acc"][c] = float(robust_correct[m].mean())
        cm = margins[m]
        out["margins_by_class"][c] = cm
        out["margin_mean"][c] = float(np.mean(cm))
        sd = float(np.std(cm))
        out["margin_std"][c] = sd
        if sd > 1e-12:
            out["margin_skew"][c] = float(np.mean(((cm - cm.mean()) / sd) ** 3))
        else:
            out["margin_skew"][c] = 0.0
    return out


# ---------------------------------------------------------------------------
# Correlation utilities (pure numpy -- no scipy dependency)
# ---------------------------------------------------------------------------
def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a):
    """Average-rank for ties; mirrors scipy.stats.rankdata."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=np.float64)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def _spearman(x, y):
    if len(x) < 3:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _partial_pearson(y, x, controls):
    """Partial Pearson r(y, x | controls). controls is a list/array (n_obs, k).
    Implements via residualisation against [1, controls].
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    Z = np.asarray(controls, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = y.size
    if n < Z.shape[1] + 3:
        return float("nan")
    Zc = np.concatenate([np.ones((n, 1)), Z], axis=1)
    # least-squares residuals of y and x on Zc
    bx, *_ = np.linalg.lstsq(Zc, x, rcond=None)
    by, *_ = np.linalg.lstsq(Zc, y, rcond=None)
    rx = x - Zc @ bx
    ry = y - Zc @ by
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _cooks_distance(y, x):
    """Cook's distance for simple OLS y ~ a + b*x. Returns array length n."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = y.size
    if n < 4 or x.std() < 1e-12:
        return np.full(n, np.nan)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    H = X @ XtX_inv @ X.T
    h = np.diag(H)
    beta = XtX_inv @ X.T @ y
    yhat = X @ beta
    resid = y - yhat
    sse = float((resid ** 2).sum())
    p = 2
    mse = sse / max(n - p, 1)
    if mse < 1e-18:
        return np.full(n, 0.0)
    D = (resid ** 2 / (p * mse)) * (h / (1 - h) ** 2)
    return D


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
def print_per_class_table(stats, model_name):
    print(f"\n--- Per-class table : {model_name} ---")
    print(f"{'cls':>3} {'name':<13} {'n':>4} {'clean':>7} {'robust':>7} "
          f"{'m_mean':>8} {'m_std':>7} {'m_skew':>7}")
    for c in range(len(stats["n_class"])):
        print(f"{c:>3} {FMNIST_CLASS_NAMES[c]:<13} "
              f"{stats['n_class'][c]:>4} "
              f"{stats['clean_acc'][c]:>7.3f} "
              f"{stats['robust_acc'][c]:>7.3f} "
              f"{stats['margin_mean'][c]:>8.3f} "
              f"{stats['margin_std'][c]:>7.3f} "
              f"{stats['margin_skew'][c]:>7.3f}")


def print_corr_block(stats, model_name, label, idx):
    """Pearson/Spearman of margin stats vs robust_acc over the index set `idx`."""
    rob = stats["robust_acc"][idx]
    mn = stats["margin_mean"][idx]
    sd = stats["margin_std"][idx]
    sk = stats["margin_skew"][idx]
    print(f"\n[corr] {model_name} -- {label} (n={len(idx)})")
    print(f"   margin_mean vs robust_acc : Pearson={_pearson(mn, rob):+.3f}  "
          f"Spearman={_spearman(mn, rob):+.3f}")
    print(f"   margin_std  vs robust_acc : Pearson={_pearson(sd, rob):+.3f}  "
          f"Spearman={_spearman(sd, rob):+.3f}")
    print(f"   margin_skew vs robust_acc : Pearson={_pearson(sk, rob):+.3f}  "
          f"Spearman={_spearman(sk, rob):+.3f}")


def quantile_sketch(arr, qs=(5, 25, 50, 75, 95)):
    return [float(np.percentile(arr, q)) for q in qs]


def drill_down_hardest(stats, model_name, k=3):
    """Per-sample margin sketch for the k hardest classes (by robust_acc)."""
    rob = stats["robust_acc"]
    order = np.argsort(rob)[:k]
    print(f"\n--- Hardest-{k}-classes drill-down ({model_name}) ---")
    print(f"   {'cls':>3} {'name':<13} {'rob':>6} {'n':>5} "
          f"  {'P5':>7} {'P25':>7} {'P50':>7} {'P75':>7} {'P95':>7}")
    for c in order:
        cm = stats["margins_by_class"].get(int(c), np.array([]))
        if cm.size == 0:
            continue
        qs = quantile_sketch(cm)
        print(f"   {c:>3} {FMNIST_CLASS_NAMES[c]:<13} "
              f"{rob[c]:>6.3f} {cm.size:>5d}  "
              f"{qs[0]:>7.3f} {qs[1]:>7.3f} {qs[2]:>7.3f} {qs[3]:>7.3f} {qs[4]:>7.3f}")
        # crude left-tail flag: fraction of margins < 0 (i.e. wrong even pre-attack)
        neg_frac = float((cm < 0).mean())
        print(f"      left-tail (margin<0): {neg_frac:.3f}   "
              f"(P5 << mean => fat left tail signals brittle minority)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("H487 - Per-class margin scatter: does within-class margin VARIANCE")
    print("       predict per-class robust-acc on a PGD-AT model? (F-MNIST, SmallCNN)")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  seed={SEED}  eps={EPS}  "
          f"n_train={N_TRAIN}  n_eval={N_EVAL}  epochs={EPOCHS}")
    print("Citations: Xu2021 (fairness AT), Tian2021 (classwise robustness), "
          "Madry2018, Zhang2019 (TRADES)")
    print("=" * 78)

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    ncls = meta["n_classes"]
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # ---- train three models on identical data ----
    models = {}

    print("\n[1/3] Training STD (vanilla CE) ...")
    t0 = time.time()
    m_std = C.build_model("cnn", meta, seed=SEED)
    C.train_model(m_std, Xtr, Ytr, epochs=EPOCHS, opt="sgd", lr=0.05, ncls=ncls)
    models["STD"] = m_std
    print(f"   done in {time.time() - t0:.1f}s")

    print("\n[2/3] Training PGD-AT (Madry, eps=0.1, steps=7) ...")
    t0 = time.time()
    m_at = C.build_model("cnn", meta, seed=SEED)
    C.train_model(m_at, Xtr, Ytr, epochs=EPOCHS, opt="sgd", lr=0.05, ncls=ncls,
                  adv_train=True, adv_eps=EPS, adv_steps=PGD_STEPS_TRAIN)
    models["PGD-AT"] = m_at
    print(f"   done in {time.time() - t0:.1f}s")

    print(f"\n[3/3] Training TRADES (beta={TRADES_BETA}) ...")
    t0 = time.time()
    m_tr = C.build_model("cnn", meta, seed=SEED)
    train_trades(m_tr, Xtr, Ytr, epochs=EPOCHS, beta=TRADES_BETA,
                 eps=EPS, steps=PGD_STEPS_TRAIN)
    models["TRADES"] = m_tr
    print(f"   done in {time.time() - t0:.1f}s")

    # ---- per-class stats for each model ----
    all_stats = {}
    for name, model in models.items():
        print(f"\nEvaluating {name} per-class (PGD-{PGD_STEPS_EVAL}, eps={EPS}) ...")
        t0 = time.time()
        s = per_class_stats(model, Xte, Yte, ncls, eps=EPS, steps=PGD_STEPS_EVAL)
        all_stats[name] = s
        print(f"   done in {time.time() - t0:.1f}s   "
              f"overall clean={s['clean_acc'].mean():.3f}  "
              f"overall robust={s['robust_acc'].mean():.3f}")
        print_per_class_table(s, name)

    # ---- correlation tables ----
    all_idx = np.arange(ncls)
    no6_idx = np.array([c for c in range(ncls) if c != 6])  # drop "Shirt"

    print("\n" + "=" * 78)
    print("CORRELATIONS: per-class margin stats vs robust-acc")
    print("(Pearson and Spearman; n=10 classes overall, n=9 dropping 'Shirt')")
    print("=" * 78)
    for name, s in all_stats.items():
        print_corr_block(s, name, "all 10 classes", all_idx)
        print_corr_block(s, name, "excl. class 6 (Shirt)", no6_idx)

    # ---- partial correlation (the headline control) on PGD-AT ----
    s_at = all_stats["PGD-AT"]
    controls = np.column_stack([s_at["margin_mean"], s_at["clean_acc"]])
    pr_std = _partial_pearson(s_at["robust_acc"], s_at["margin_std"], controls)
    pr_skew = _partial_pearson(s_at["robust_acc"], s_at["margin_skew"], controls)
    # also: marginal Pearson for direct comparison
    raw_std = _pearson(s_at["margin_std"], s_at["robust_acc"])
    raw_skew = _pearson(s_at["margin_skew"], s_at["robust_acc"])

    print("\n" + "=" * 78)
    print("PARTIAL CORRELATION  (PGD-AT model, n=10 classes)")
    print("=" * 78)
    print("  raw Pearson r(margin_std,  robust_acc)                         = "
          f"{raw_std:+.3f}")
    print("  partial    r(margin_std,  robust_acc | margin_mean, clean_acc) = "
          f"{pr_std:+.3f}")
    print("  raw Pearson r(margin_skew, robust_acc)                         = "
          f"{raw_skew:+.3f}")
    print("  partial    r(margin_skew, robust_acc | margin_mean, clean_acc) = "
          f"{pr_skew:+.3f}")
    print("  -> If |partial| stays sizeable (>= ~0.4) and same sign as raw,")
    print("     margin_std carries signal beyond what mean + clean-acc already give.")

    # ---- Cook's distance for the OLS robust_acc ~ margin_std on PGD-AT ----
    cd = _cooks_distance(s_at["robust_acc"], s_at["margin_std"])
    thresh = 4.0 / ncls
    print("\n  Cook's distance (OLS robust_acc ~ margin_std, PGD-AT):")
    for c in range(ncls):
        flag = "  <-- influential" if (np.isfinite(cd[c]) and cd[c] > thresh) else ""
        print(f"     class {c} ({FMNIST_CLASS_NAMES[c]:<13}) D={cd[c]:.3f}{flag}")
    print(f"  threshold = 4/n = {thresh:.3f}")

    # ---- hardest-3 drill-downs ----
    print("\n" + "=" * 78)
    print("HARDEST-3-CLASSES DRILL-DOWN  (sample-level margin quantile sketches)")
    print("=" * 78)
    for name, s in all_stats.items():
        drill_down_hardest(s, name, k=3)

    # ---- headline verdict ----
    # Verdict logic:
    #   * H487 is supported on PGD-AT if margin_std is NEGATIVELY correlated with
    #     robust_acc (Pearson AND Spearman both <= -0.4) AND the partial r given
    #     (margin_mean, clean_acc) is still <= -0.3.
    #   * Otherwise: rejected (or "explained by mean").
    sd_pear = _pearson(s_at["margin_std"], s_at["robust_acc"])
    sd_spear = _spearman(s_at["margin_std"], s_at["robust_acc"])
    sd_pear_no6 = _pearson(s_at["margin_std"][no6_idx], s_at["robust_acc"][no6_idx])

    if (np.isfinite(sd_pear) and np.isfinite(pr_std)
            and sd_pear <= -0.4 and sd_spear <= -0.4 and pr_std <= -0.3):
        verdict = "SUPPORTED"
        gloss = ("within-class margin VARIANCE has additional predictive power "
                 "for per-class robust-acc on PGD-AT, beyond margin mean and "
                 "clean-acc.")
    elif (np.isfinite(sd_pear) and np.isfinite(pr_std)
          and sd_pear <= -0.3 and pr_std > -0.3):
        verdict = "MEDIATED"
        gloss = ("raw margin_std correlates with robust-acc, but the relationship "
                 "is explained away once margin_mean and clean-acc are controlled "
                 "for -- consistent with Tian2021's mean-margin account.")
    else:
        verdict = "REJECTED"
        gloss = ("no robust negative association between margin variance and "
                 "per-class robust-acc on PGD-AT was found.")

    print("\n" + "=" * 78)
    print(f"HEADLINE: {verdict} -- {gloss}")
    print(f"  raw Pearson r(std, robust_acc)         = {sd_pear:+.3f}")
    print(f"  raw Spearman rho(std, robust_acc)      = {sd_spear:+.3f}")
    print(f"  partial r(std, robust_acc | mean, clean)= {pr_std:+.3f}")
    print(f"  excl. Shirt (cls 6) Pearson            = {sd_pear_no6:+.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
