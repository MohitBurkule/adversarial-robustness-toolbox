"""
H496 - Minimal sufficient activation statistics for per-sample robustness.

Seed (Hypothesis-set Section 5 / Section 3 G8):
    Can a tiny set of per-layer mean/std activation statistics extracted from a
    PGD-AT model linearly predict each sample's min-eps-to-flip? The hypothesis
    is that K~6-10 scalar features (one or two summary stats per ReLU layer)
    achieve Spearman rho > 0.5 with the per-sample minimum L-inf radius needed
    to flip the prediction. If true, robustness is encoded in a very low-dim
    "activation signature" rather than diffuse over all neurons.

Critique seed:
    Predictive power might come purely from a confidence proxy (the final
    pre-softmax-logit margin). We CONTROL for this by also fitting Ridge on the
    feature set with the logit-layer statistics removed, and report whether
    Spearman correlation collapses.

Related work cited (>=2):
    * Bai et al. 2021, "Improving Adversarial Robustness via Channel-wise
      Activation Suppressing" (CAS) - shows channel-level activation
      statistics carry robustness-relevant signal.
    * Wang et al. 2019, "ME-Net" - per-sample reconstruction quality
      (an activation-level summary) correlates with robustness.
    * Croce & Hein 2022, "Sparse-RS" - per-sample query-budget / min-eps
      varies enormously across samples; the very existence of a heavy-tailed
      per-sample min-eps distribution motivates predicting it from features.
    * Yang et al. 2020, "A Closer Look at Accuracy vs. Robustness" - per-sample
      robustness varies more with local input geometry than with global accuracy.

Controls / Protocol:
    (1) PGD-AT model (SmallCNN, F-MNIST, eps=0.1, 7 PGD steps in training).
    (2) On 2000 clean correctly-classified test samples, extract per-ReLU-layer
        (mean, std) of post-ReLU activations: 3 conv-block ReLUs + 1 head ReLU
        + (logit-mean, logit-std on pre-softmax logits) -> ~10 features.
    (3) Per-sample min-eps-to-flip via PGD binary search (untargeted, 10 steps
        per probe) over [0, 0.30].
    (4) Fit Ridge regression on standardised features predicting log(min_eps).
        Report R^2 (held-out 5-fold) and Spearman rho.
    (5) Ablation: drop the two logit-layer features and refit. Report whether
        rho drops below 0.5 (confidence-proxy attribution) or holds (genuine
        activation-geometry signal).

Verdict: HEADLINE = rho_full vs rho_no_logits, and whether the H8 threshold
(rho > 0.5) is met with K<=10 features.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
EPS_TRAIN = 0.1
EPS_MAX = 0.30          # upper bound for binary search
BS_ITERS = 10           # binary-search steps -> ~3e-4 resolution
PGD_STEPS = 10
N_PROBE = 2000          # samples for the feature / regression analysis


# ---------------------------------------------------------------------------
# Per-sample min-eps via PGD binary search (untargeted)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _preds(model, X, batch=512):
    out = []
    for i in range(0, X.size(0), batch):
        out.append(model(X[i:i + batch]).argmax(1))
    return torch.cat(out)


def min_eps_pgd_binsearch(model, X, Y, eps_max=EPS_MAX, iters=BS_ITERS,
                          pgd_steps=PGD_STEPS, batch=256):
    """For each sample, binary-search smallest eps in [0, eps_max] for which
    untargeted PGD flips the label. Samples never flipped at eps_max are given
    eps_max (right-censored). Returns numpy array (N,)."""
    n = X.size(0)
    lo = torch.zeros(n, device=X.device)
    hi = torch.full((n,), float(eps_max), device=X.device)

    # First check that eps_max actually flips them; otherwise mark censored.
    flipped_at_max = torch.zeros(n, dtype=torch.bool, device=X.device)
    for i in range(0, n, batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        xa = C.pgd(model, xb, yb, eps_max, steps=pgd_steps)
        with torch.no_grad():
            flipped_at_max[i:i + batch] = (model(xa).argmax(1) != yb)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        flipped_mid = torch.zeros(n, dtype=torch.bool, device=X.device)
        for i in range(0, n, batch):
            xb, yb = X[i:i + batch], Y[i:i + batch]
            eps_b = mid[i:i + batch]
            # We can't pass per-sample eps directly to C.pgd; iterate per
            # unique batch with a scalar approximation by max(eps_b). Simpler:
            # run PGD per-sample-eps by scaling each sample's perturbation.
            x0 = xb.clone().detach()
            xa = x0 + torch.empty_like(x0).uniform_(-1, 1) * eps_b[:, None, None, None]
            xa = xa.clamp(0, 1)
            for _step in range(pgd_steps):
                xa.requires_grad_(True)
                loss = F.cross_entropy(model(xa), yb)
                g, = torch.autograd.grad(loss, xa)
                alpha = (2.5 * eps_b / pgd_steps)[:, None, None, None]
                xa = xa.detach() + alpha * g.sign()
                lo_b = (x0 - eps_b[:, None, None, None])
                hi_b = (x0 + eps_b[:, None, None, None])
                xa = torch.min(torch.max(xa, lo_b), hi_b).clamp(0, 1)
            with torch.no_grad():
                flipped_mid[i:i + batch] = (model(xa).argmax(1) != yb)
        # shrink hi if flipped; raise lo otherwise
        hi = torch.where(flipped_mid, mid, hi)
        lo = torch.where(flipped_mid, lo, mid)

    min_eps = hi.clone()
    # right-censor those that never flipped at eps_max
    min_eps[~flipped_at_max] = float(eps_max)
    return min_eps.cpu().numpy(), flipped_at_max.cpu().numpy()


# ---------------------------------------------------------------------------
# Per-layer activation features via forward hooks on SmallCNN
# ---------------------------------------------------------------------------
def extract_activation_features(model, X, batch=256):
    """Capture post-ReLU activation (mean, std) per ReLU layer + logit (mean,
    std) on pre-softmax logits. Returns (N, K) numpy array and feature names.

    SmallCNN topology:
        features = [Conv-BN-ReLU-MaxPool] x 3 (3 ReLUs)
        head     = [Flatten, Linear, ReLU, Linear]  (1 more ReLU, then logits)
    """
    import torch.nn as nn
    relu_outputs = []
    handles = []

    def make_hook(idx):
        def hook(mod, inp, out):
            relu_outputs[idx].append(out.detach())
        return hook

    relu_modules = []
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            relu_modules.append(m)
    relu_outputs.clear()
    relu_outputs.extend([[] for _ in relu_modules])
    for i, m in enumerate(relu_modules):
        handles.append(m.register_forward_hook(make_hook(i)))

    feats = []
    logits_chunks = []
    model.eval()
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            for ro in relu_outputs:
                ro.clear()
            xb = X[i:i + batch]
            lg = model(xb)
            logits_chunks.append(lg.detach().cpu())
            # collapse each ReLU output to (mean, std) per-sample
            per_layer = []
            for ro in relu_outputs:
                a = ro[0]  # (B, C, H, W) or (B, D)
                a_flat = a.flatten(1)
                per_layer.append(a_flat.mean(1, keepdim=True))
                per_layer.append(a_flat.std(1, keepdim=True))
            feats.append(torch.cat(per_layer, dim=1).cpu())
    for h in handles:
        h.remove()

    feats = torch.cat(feats, dim=0).numpy()        # (N, 2*num_relus)
    logits = torch.cat(logits_chunks, dim=0).numpy()
    # Append (logit_mean, logit_std) over the K-class logits as the "logit
    # layer" features that the critique flags as a possible confidence proxy.
    lg_mean = logits.mean(axis=1, keepdims=True)
    lg_std = logits.std(axis=1, keepdims=True)
    X_full = np.concatenate([feats, lg_mean, lg_std], axis=1)

    names = []
    for i in range(len(relu_modules)):
        names += [f"relu{i+1}_mean", f"relu{i+1}_std"]
    names += ["logit_mean", "logit_std"]
    return X_full, names


# ---------------------------------------------------------------------------
# Ridge + Spearman (sklearn)
# ---------------------------------------------------------------------------
def evaluate_ridge(Xf, y, n_splits=5, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    preds = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(Xf):
        sc = StandardScaler().fit(Xf[tr])
        Xt, Xv = sc.transform(Xf[tr]), sc.transform(Xf[te])
        m = Ridge(alpha=alpha).fit(Xt, y[tr])
        preds[te] = m.predict(Xv)
    # R^2 (held-out)
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rho, _ = spearmanr(preds, y)
    return float(r2), float(rho)


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------
def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=2500, seed=seed)

    model = C.build_model("cnn", meta, seed=seed)
    # PGD-adversarial training: control (1).
    C.train_model(model, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05,
                  ncls=meta["n_classes"], adv_train=True,
                  adv_eps=EPS_TRAIN, adv_steps=7)

    # Use clean correctly-classified test samples only (so min-eps is a flip,
    # not a correction).
    lg, _ = C.logits_and_acc(model, Xte, Yte)
    corr = (lg.argmax(1) == Yte.cpu()).numpy().astype(bool)
    idx_all = np.where(corr)[0]
    rng = np.random.RandomState(seed)
    rng.shuffle(idx_all)
    idx = idx_all[:N_PROBE]
    Xp, Yp = Xte[idx], Yte[idx]
    n = Xp.size(0)

    # Control (3): per-sample min-eps-to-flip.
    t0 = time.time()
    min_eps, flipped = min_eps_pgd_binsearch(model, Xp, Yp)
    t_eps = time.time() - t0

    # Control (2): per-layer activation features.
    feats, names = extract_activation_features(model, Xp)
    K = feats.shape[1]

    # Target = log(min_eps). Right-censored points sit at eps_max so we
    # clip away from log(0) by adding a small floor.
    y = np.log(np.maximum(min_eps, 1e-4))

    # Control (4): Ridge with ALL features.
    r2_full, rho_full = evaluate_ridge(feats, y)
    # Control (5): ablation - drop logit_mean & logit_std (the final two cols).
    logit_cols = [i for i, nm in enumerate(names) if nm.startswith("logit_")]
    keep = [i for i in range(feats.shape[1]) if i not in logit_cols]
    r2_no, rho_no = evaluate_ridge(feats[:, keep], y)

    # Sanity baseline: predict the constant mean -> rho = 0; we also report
    # a single-feature baseline using only logit_mean to gauge the confidence
    # proxy on its own.
    r2_lm, rho_lm = evaluate_ridge(feats[:, [names.index("logit_mean")]], y)

    return {
        "seed": seed,
        "n_samples": int(n),
        "K_features": int(K),
        "fraction_flipped_at_eps_max": float(flipped.mean()),
        "min_eps_median": float(np.median(min_eps)),
        "min_eps_p25": float(np.percentile(min_eps, 25)),
        "min_eps_p75": float(np.percentile(min_eps, 75)),
        "r2_full": r2_full,
        "rho_full": rho_full,
        "r2_no_logits": r2_no,
        "rho_no_logits": rho_no,
        "r2_logitmean_only": r2_lm,
        "rho_logitmean_only": rho_lm,
        "t_eps_s": round(t_eps, 1),
        "feature_names": names,
    }


def main():
    print("=" * 78)
    print("H496 - Minimal sufficient activation statistics for per-sample robustness")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  eps_train={EPS_TRAIN}  N_probe={N_PROBE}")
    print(f"Binary search: range [0,{EPS_MAX}] x {BS_ITERS} iters x PGD-{PGD_STEPS} per probe")
    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"\n[seed {s}] K={r['K_features']} features ({r['runtime_s']}s, eps-search {r['t_eps_s']}s)")
        print(f"  feature names: {r['feature_names']}")
        print(f"  min_eps: median={r['min_eps_median']:.4f}  p25={r['min_eps_p25']:.4f}  p75={r['min_eps_p75']:.4f}")
        print(f"  flipped at eps_max ({EPS_MAX}): {r['fraction_flipped_at_eps_max']:.3f}")
        print(f"  FULL  (K={r['K_features']:>2}) : R^2={r['r2_full']:+.3f}  Spearman rho={r['rho_full']:+.3f}")
        print(f"  NO LOGIT FEATS    : R^2={r['r2_no_logits']:+.3f}  Spearman rho={r['rho_no_logits']:+.3f}")
        print(f"  logit_mean only   : R^2={r['r2_logitmean_only']:+.3f}  Spearman rho={r['rho_logitmean_only']:+.3f}")

    def m(k):
        v = [r[k] for r in rows if isinstance(r[k], float) and r[k] == r[k]]
        return sum(v) / len(v) if v else float("nan")

    rho_full = m("rho_full")
    rho_no = m("rho_no_logits")
    rho_lm = m("rho_logitmean_only")

    print("\n" + "=" * 78)
    print("MEANS across seeds")
    print(f"  rho_full        : {rho_full:+.3f}   (target: > 0.5)")
    print(f"  rho_no_logits   : {rho_no:+.3f}   (collapse => confidence-proxy)")
    print(f"  rho_logitmean   : {rho_lm:+.3f}   (single-feature confidence baseline)")
    print(f"  R^2_full        : {m('r2_full'):+.3f}")
    print(f"  R^2_no_logits   : {m('r2_no_logits'):+.3f}")
    print("=" * 78)

    # ------------------------------------------------------------------ verdict
    H8 = 0.5
    if rho_full > H8 and rho_no > H8:
        verdict = ("SUPPORTED. A K<=10 activation-stat signature predicts per-sample "
                   "min-eps with Spearman > 0.5 even after removing logit features, so "
                   "the signal is genuine activation geometry, not a logit-confidence "
                   "proxy. Robustness has a low-dim sufficient statistic (cf. CAS, "
                   "Bai 2021).")
    elif rho_full > H8 and rho_no <= H8:
        verdict = ("PARTIAL. Full-feature rho clears 0.5 but collapses when logit "
                   "features are dropped: most of the predictive power was a confidence "
                   "proxy (consistent with the critique). Internal ReLU statistics "
                   "alone are NOT a minimal sufficient signature.")
    elif rho_full <= H8:
        verdict = ("REFUTED. Even with all K activation+logit features Spearman stays "
                   "below 0.5, so 6-10 layer-mean statistics are NOT sufficient to "
                   "predict per-sample min-eps. Robustness signature is higher-dim "
                   "than the K8 hypothesis claims.")
    else:
        verdict = "INCONCLUSIVE."

    print("\nHEADLINE VERDICT:")
    print(verdict)
    print("=" * 78)


if __name__ == "__main__":
    main()
