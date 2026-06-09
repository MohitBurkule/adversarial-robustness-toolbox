"""
H503 - Certified radius (randomized smoothing) vs empirical PGD-L2 margin.

Seed papers (§5 / §3 G6):
  * Cohen, Rosenfeld & Kolter, "Certified Adversarial Robustness via Randomized
    Smoothing", ICML 2019.  Per-sample certified L2 radius
        R = sigma * Phi^{-1}(p_A)
    where p_A is a (Clopper-Pearson) lower bound on the smoothed classifier's
    probability of returning the top class under Gaussian noise N(0, sigma^2 I).
  * Salman et al., "Provably Robust Deep Learning via Adversarially Trained
    Smoothed Classifiers", NeurIPS 2019.  Smoothed-AT base classifier trained
    with Gaussian noise injection (here we use noise sigma=0.25 during training,
    matching the certification sigma -- the simple Cohen-style "Gaussian
    augmentation" variant that Salman's paper builds on and improves).
  * Yang et al., "Randomized Smoothing of All Shapes and Sizes", ICML 2020.
    Establishes the geometry of smoothing certificates beyond the L2 ball and
    motivates comparing certified radii to *L2* empirical margins specifically.

Hypothesis (advisor critique baked in):
  Certified L2 radius from randomized smoothing correlates STRONGLY (Spearman
  rho > 0.6) with the empirical PGD-L2 min-margin computed on the *smoothed*
  classifier (Monte-Carlo majority-vote), but only MODERATELY (rho ~ 0.3) with
  the PGD-L2 margin of the *base* classifier.  Comparing certificate to base-
  classifier empirical margin (a common informal practice) is the wrong target:
  the certificate is a statement about the smoothed classifier, so the
  empirical reference must also be smoothed.

Controls:
  (1) Two base classifiers per seed:
        STD          - standard cross-entropy training, no noise.
        SMOOTHED_AT  - Salman-style Gaussian-noise training at sigma=0.25
                       (each minibatch input gets fresh N(0, sigma^2 I) noise).
  (2) Randomized-smoothing certification (Cohen Alg. CERTIFY):
        n0 = 100 noise samples to pick top class,
        n  = 1000 noise samples to lower-bound p_A,
        alpha = 0.001 confidence, 200 test points.
      Certified radius for non-abstained samples is sigma * Phi^{-1}(p_A_lb);
      abstained samples get radius 0 and are tracked separately.
  (3) Empirical PGD-L2 margin on the SMOOTHED classifier:
      Binary search on the L2 budget; "model" call is a Monte-Carlo smoothed
      vote (n_mc = 64 noise samples) so the attack sees the *same* classifier
      the certificate describes.  Returns the smallest L2 perturbation that
      flips the smoothed prediction (PGD-L2 with 20 steps, 5 restarts).
  (4) Empirical PGD-L2 margin on the BASE classifier with the same protocol
      (no smoothing at attack time).
  (5) Spearman rho between certified radius and (a) smoothed empirical margin,
      (b) base empirical margin.  Done on the non-abstained, originally-correct
      subset for fairness.
  (6) Coverage: fraction abstained at alpha = 0.001 on each model.

DO NOT EXECUTE.  Output (when later run) lands in
results/fashion_mnist/h503_certified_radius_vs_margin_output.txt
via the campaign driver's redirect convention.
"""
import os, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
DS = "fashion_mnist"
SEEDS = [0, 1, 2]
SIGMA = 0.25          # smoothing / training noise std
N0 = 100              # noise samples to pick top class
N_CERT = 1000         # noise samples to bound p_A
ALPHA = 0.001         # certification confidence
N_TEST = 200          # certified / attacked test points
N_MC_ATTACK = 64      # Monte-Carlo samples per forward during smoothed attack
PGD_L2_STEPS = 20
PGD_L2_RESTARTS = 5
L2_BUDGETS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # bisection grid (L2)
TRAIN_EPOCHS = 6


# --------------------------------------------------------------------------
# Clopper-Pearson lower confidence bound on a Binomial proportion.
#   p_lb such that P[Bin(n, p_lb) >= k] = alpha.
# We avoid scipy by using the Beta-inverse identity via math.lgamma + bisection.
# --------------------------------------------------------------------------
def _binom_sf_ge(k, n, p):
    """P[Bin(n,p) >= k] = I_p(k, n-k+1), regularised incomplete beta."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    # use torch.special.betainc for stability
    return float(torch.special.betainc(torch.tensor(float(k)),
                                       torch.tensor(float(n - k + 1)),
                                       torch.tensor(float(p))))


def cp_lower_bound(k, n, alpha):
    """Clopper-Pearson lower bound: largest p with P[Bin(n,p) >= k] <= alpha."""
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _binom_sf_ge(k, n, mid) <= alpha:
            lo = mid
        else:
            hi = mid
    return lo


def phi_inv(p):
    """Inverse standard-normal CDF via erfinv."""
    p = float(min(max(p, 1e-12), 1 - 1e-12))
    return math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * p - 1.0)))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_noisy(model, Xtr, Ytr, sigma, epochs=TRAIN_EPOCHS, batch=128, lr=0.05):
    """Salman-style Gaussian augmentation: add fresh N(0, sigma^2 I) each step."""
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            xb = (xb + sigma * torch.randn_like(xb)).clamp(0, 1)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# --------------------------------------------------------------------------
# Smoothed classifier wrapper (Monte-Carlo majority vote)
# --------------------------------------------------------------------------
@torch.no_grad()
def _mc_counts(model, x, n_mc, sigma, ncls, batch_mc=64):
    """For a SINGLE input x of shape (1,C,H,W), return class counts under n_mc
    Gaussian noise samples."""
    counts = torch.zeros(ncls, device=x.device)
    done = 0
    while done < n_mc:
        b = min(batch_mc, n_mc - done)
        xs = x.expand(b, -1, -1, -1) + sigma * torch.randn(b, *x.shape[1:], device=x.device)
        xs = xs.clamp(0, 1)
        preds = model(xs).argmax(1)
        counts += torch.bincount(preds, minlength=ncls).float()
        done += b
    return counts


def smoothed_predict_batch(model, X, n_mc, sigma, ncls):
    """Top-1 vote per row.  Returns LongTensor (N,)."""
    out = torch.empty(X.size(0), dtype=torch.long, device=X.device)
    for i in range(X.size(0)):
        out[i] = _mc_counts(model, X[i:i + 1], n_mc, sigma, ncls).argmax()
    return out


# --------------------------------------------------------------------------
# Cohen CERTIFY (Algorithm 1, Cohen 2019)
# --------------------------------------------------------------------------
def certify(model, X, sigma, n0, n_cert, alpha, ncls):
    """Return (top_class, certified_radius, abstained)
       arrays of length X.size(0).  radius = 0 when abstained."""
    radii = np.zeros(X.size(0))
    top = -np.ones(X.size(0), dtype=np.int64)
    abstain = np.zeros(X.size(0), dtype=bool)
    for i in range(X.size(0)):
        x = X[i:i + 1]
        c0 = _mc_counts(model, x, n0, sigma, ncls)
        c_hat = int(c0.argmax())
        c1 = _mc_counts(model, x, n_cert, sigma, ncls)
        n_a = int(c1[c_hat].item())
        p_lb = cp_lower_bound(n_a, n_cert, alpha)
        if p_lb <= 0.5:
            abstain[i] = True
            top[i] = c_hat
            radii[i] = 0.0
        else:
            top[i] = c_hat
            radii[i] = sigma * phi_inv(p_lb)
    return top, radii, abstain


# --------------------------------------------------------------------------
# PGD-L2 min-margin (binary-search-like budget sweep)
# --------------------------------------------------------------------------
def _pgd_l2_step(loss, x_adv, x0, eps, alpha):
    g, = torch.autograd.grad(loss, x_adv)
    # normalise grad
    flat = g.view(g.size(0), -1)
    n = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    g_unit = (flat / n).view_as(g)
    x_new = x_adv.detach() + alpha * g_unit
    # project to L2 ball of radius eps around x0
    delta = x_new - x0
    flat_d = delta.view(delta.size(0), -1)
    dn = flat_d.norm(dim=1, keepdim=True).clamp_min(1e-12)
    factor = torch.clamp(eps / dn, max=1.0)
    delta = (flat_d * factor).view_as(delta)
    return (x0 + delta).clamp(0, 1).detach()


def pgd_l2(pred_fn, x, y, eps, steps=PGD_L2_STEPS, restarts=PGD_L2_RESTARTS,
           ncls=10):
    """L2 PGD using a generic prediction function `pred_fn(x) -> logits`.
    Returns the worst (most-flipping) adversarial across restarts."""
    best_adv = x.clone().detach()
    best_flipped = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    alpha = 2.5 * eps / steps
    for r in range(restarts):
        # random init inside L2 ball
        noise = torch.randn_like(x)
        flat = noise.view(noise.size(0), -1)
        flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        radius = torch.rand(x.size(0), 1, device=x.device) * eps
        delta = (flat * radius).view_as(x)
        x_adv = (x + delta).clamp(0, 1).detach()
        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits = pred_fn(x_adv)
            loss = F.cross_entropy(logits, y)
            x_adv = _pgd_l2_step(loss, x_adv, x, eps, alpha)
        with torch.no_grad():
            flipped = pred_fn(x_adv).argmax(1) != y
        upd = flipped & (~best_flipped)
        best_adv[upd] = x_adv[upd]
        best_flipped = best_flipped | flipped
    return best_adv, best_flipped


def empirical_l2_min_margin(pred_fn, X, Y, budgets=L2_BUDGETS):
    """For each row, smallest budget in `budgets` that flips the prediction;
    +inf if no budget flips.  Returns numpy array."""
    n = X.size(0)
    flipped_at = np.full(n, np.inf)
    remaining = torch.ones(n, dtype=torch.bool, device=X.device)
    for eps in budgets:
        if not remaining.any():
            break
        idx = torch.where(remaining)[0]
        x_sub, y_sub = X[idx], Y[idx]
        _, flipped = pgd_l2(pred_fn, x_sub, y_sub, eps=eps)
        sub_np = flipped.cpu().numpy()
        idx_np = idx.cpu().numpy()
        for j, f in zip(idx_np, sub_np):
            if f and not np.isfinite(flipped_at[j]):
                flipped_at[j] = eps
        remaining[idx[flipped]] = False
    return flipped_at


# --------------------------------------------------------------------------
# Smoothed-classifier prediction function for the attack.
# We use a *soft* MC estimate (average softmax over noise samples) so PGD has
# a well-defined gradient through the smoothed prediction.  This is the
# standard differentiable surrogate used in the Salman/SmoothAdv line of work.
# --------------------------------------------------------------------------
def make_smoothed_predfn(model, sigma, n_mc, batch_mc=32):
    def pred(x):
        # average softmax over n_mc noise draws; gradients flow through model.
        accum = 0.0
        done = 0
        while done < n_mc:
            b = min(batch_mc, n_mc - done)
            # tile each row b times along a new leading axis, then merge
            noise = sigma * torch.randn(b, *x.shape, device=x.device)
            xs = (x.unsqueeze(0) + noise).clamp(0, 1)             # (b,N,C,H,W)
            xs = xs.view(b * x.size(0), *x.shape[1:])
            logits = model(xs).view(b, x.size(0), -1)
            accum = accum + F.softmax(logits, dim=-1).sum(0)
            done += b
        probs = accum / n_mc
        # log to give CE-meaningful logits
        return torch.log(probs.clamp_min(1e-12))
    return pred


def make_base_predfn(model):
    def pred(x):
        return model(x)
    return pred


# --------------------------------------------------------------------------
# Spearman rho (no scipy dependence)
# --------------------------------------------------------------------------
def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    ra = _rankdata(a[mask])
    rb = _rankdata(b[mask])
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


def _rankdata(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=float)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


# --------------------------------------------------------------------------
# Per-seed run
# --------------------------------------------------------------------------
def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    ncls = meta["n_classes"]
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_TEST, seed=seed)

    # --- two base classifiers ---
    m_std = C.build_model("cnn", meta, seed=seed)
    C.train_model(m_std, Xtr, Ytr, epochs=TRAIN_EPOCHS, opt="sgd", lr=0.05, ncls=ncls)

    m_sat = C.build_model("cnn", meta, seed=seed)
    train_noisy(m_sat, Xtr, Ytr, sigma=SIGMA, epochs=TRAIN_EPOCHS, lr=0.05)

    out = {"seed": seed}

    for tag, model in [("STD", m_std), ("SMOOTHED_AT", m_sat)]:
        # (a) certification
        top, radii, abstain = certify(model, Xte, SIGMA, N0, N_CERT, ALPHA, ncls)
        coverage_abstain = float(abstain.mean())

        # (b) "smoothed correct" mask: vote matches label AND not abstained.
        y_np = Yte.cpu().numpy()
        smooth_correct = (~abstain) & (top == y_np)

        # (c) empirical L2 min-margin on SMOOTHED classifier
        sm_pred = make_smoothed_predfn(model, SIGMA, N_MC_ATTACK)
        m2_smooth = empirical_l2_min_margin(sm_pred, Xte, Yte)

        # (d) empirical L2 min-margin on BASE classifier
        base_pred = make_base_predfn(model)
        m2_base = empirical_l2_min_margin(base_pred, Xte, Yte)

        # (e) Spearman correlations on smoothed-correct, non-abstained subset.
        # Replace inf (never flipped) with the largest budget * 1.5 so rank is
        # well-defined for both arms.
        cap = max(L2_BUDGETS) * 1.5
        m2s = np.where(np.isfinite(m2_smooth), m2_smooth, cap)
        m2b = np.where(np.isfinite(m2_base), m2_base, cap)

        rho_smooth = spearman(radii[smooth_correct], m2s[smooth_correct])
        rho_base = spearman(radii[smooth_correct], m2b[smooth_correct])

        out[tag] = {
            "coverage_abstain": coverage_abstain,
            "mean_cert_radius": float(radii[smooth_correct].mean()) if smooth_correct.any() else float("nan"),
            "mean_smoothed_l2_margin": float(np.mean(m2s[smooth_correct])) if smooth_correct.any() else float("nan"),
            "mean_base_l2_margin": float(np.mean(m2b[smooth_correct])) if smooth_correct.any() else float("nan"),
            "rho_cert_vs_smoothed_margin": rho_smooth,
            "rho_cert_vs_base_margin": rho_base,
            "n_smooth_correct": int(smooth_correct.sum()),
        }

    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("H503 - Certified L2 radius (randomized smoothing) vs empirical PGD-L2 margin")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  sigma={SIGMA}  n0={N0}  n_cert={N_CERT}")
    print(f"alpha={ALPHA}  n_test={N_TEST}  n_mc_attack={N_MC_ATTACK}")
    print(f"PGD-L2: steps={PGD_L2_STEPS} restarts={PGD_L2_RESTARTS} budgets={L2_BUDGETS}")
    print("Seeds:  Cohen 2019; Salman 2019 (smoothed-AT); Yang 2020 (smoothing geometry).")
    print()

    all_rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        all_rows.append(r)
        for tag in ["STD", "SMOOTHED_AT"]:
            d = r[tag]
            print(f"[seed {s} | {tag}]  ({r['runtime_s']}s, cumulative)")
            print(f"  abstain frac (alpha={ALPHA})       : {d['coverage_abstain']:.3f}")
            print(f"  mean certified L2 radius            : {d['mean_cert_radius']:.3f}")
            print(f"  mean empirical L2 margin (smoothed) : {d['mean_smoothed_l2_margin']:.3f}")
            print(f"  mean empirical L2 margin (base)     : {d['mean_base_l2_margin']:.3f}")
            print(f"  Spearman rho  cert vs SMOOTHED      : {d['rho_cert_vs_smoothed_margin']:.3f}")
            print(f"  Spearman rho  cert vs BASE          : {d['rho_cert_vs_base_margin']:.3f}")
            print()

    # aggregate
    def agg(tag, key):
        v = [r[tag][key] for r in all_rows if r[tag][key] == r[tag][key]]
        return float(np.mean(v)) if v else float("nan")

    print("=" * 78)
    print("MEANS across seeds")
    for tag in ["STD", "SMOOTHED_AT"]:
        print(f"  [{tag}]")
        print(f"    abstain frac                       : {agg(tag,'coverage_abstain'):.3f}")
        print(f"    cert radius                        : {agg(tag,'mean_cert_radius'):.3f}")
        print(f"    PGD-L2 margin (smoothed)           : {agg(tag,'mean_smoothed_l2_margin'):.3f}")
        print(f"    PGD-L2 margin (base)               : {agg(tag,'mean_base_l2_margin'):.3f}")
        print(f"    rho cert vs SMOOTHED               : {agg(tag,'rho_cert_vs_smoothed_margin'):.3f}")
        print(f"    rho cert vs BASE                   : {agg(tag,'rho_cert_vs_base_margin'):.3f}")
    print("=" * 78)

    # ---------------- HEADLINE verdict ----------------
    rho_s = agg("SMOOTHED_AT", "rho_cert_vs_smoothed_margin")
    rho_b = agg("SMOOTHED_AT", "rho_cert_vs_base_margin")
    if rho_s > 0.6 and rho_b < 0.45 and (rho_s - rho_b) > 0.2:
        verdict = ("SUPPORTED: certified L2 radius tracks the SMOOTHED classifier's "
                   "empirical PGD-L2 margin (rho>0.6) but only moderately the base "
                   "classifier's margin (rho<~0.45). Confirms Cohen/Salman/Yang: the "
                   "certificate is a statement about the smoothed model.")
    elif rho_s > rho_b + 0.1:
        verdict = ("PARTIAL: rho(smoothed) > rho(base) in the predicted direction, but "
                   "the smoothed correlation falls short of the >0.6 threshold or the "
                   "gap is small. Mechanism present but weak at this MC budget.")
    else:
        verdict = ("NOT SUPPORTED: certified radius does not track the smoothed-"
                   "classifier margin more strongly than the base margin. Either "
                   "the MC estimates are too noisy, or sigma is misaligned with the "
                   "L2 attack scale on F-MNIST/SmallCNN.")
    print("HEADLINE:")
    print("  " + verdict)
    print("=" * 78)
    print("Interpretation: This addresses the advisor critique that comparing a")
    print("smoothed-classifier certificate to a base-classifier empirical margin is")
    print("apples-to-oranges. We show both. Where the literature's promise holds,")
    print("rho(cert, smoothed-margin) dominates rho(cert, base-margin), and the gap")
    print("widens for the smoothed-AT model trained per Salman 2019.")
    print("=" * 78)


if __name__ == "__main__":
    main()
