"""
H268 - Certified vs empirical robustness.

Certified radius via randomised smoothing (sigma=0.1, K=100 samples) vs empirical
robustness (min eps to flip via binary search).

For each test sample:
  - certified_radius = sigma * Phi^-1(smoothed_top_prob)
  - empirical min_eps via binary search over FGSM eps

Do certified and empirical robustness rank samples the same way? Spearman rho.
Are certified-robust samples a subset of empirically-robust samples?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm as scipy_norm, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ── config ────────────────────────────────────────────────────────────────────
DS        = "fashion_mnist"
N_EVAL    = 200      # binary search is slow
EPS_STD   = 0.1
SEED      = 0
META      = {"channels": 1, "size": 28, "n_classes": 10}
EPOCHS    = 10
WIDTH     = 32
RS_SIGMA  = 0.1      # randomised smoothing sigma
RS_K      = 100      # smoothing samples
EPS_LO    = 0.01
EPS_HI    = 0.5
BSEARCH_K = 10       # binary search iterations
CERT_THRESH = 0.0    # certified_radius > 0 = certified robust

OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def randomised_smoothing_cert(model, X, Y, sigma, K, n_classes=10):
    """For each sample, compute certified_radius = sigma * Phi^-1(top_class_prob).
    Uses Monte-Carlo estimate of smoothed classifier.
    Returns (certified_radius array, smoothed_top_prob array, smooth_pred array).
    """
    model.eval()
    N = X.size(0)
    class_counts = np.zeros((N, n_classes), dtype=int)

    for _ in range(K):
        noise = torch.randn_like(X) * sigma
        Xn = (X + noise).clamp(0, 1)
        with torch.no_grad():
            preds = model(Xn).argmax(1).cpu().numpy()
        for i, p in enumerate(preds):
            class_counts[i, p] += 1

    # Smoothed top-1 class and its probability
    smooth_pred = class_counts.argmax(axis=1)
    top_probs = class_counts.max(axis=1) / K

    # Certified radius: sigma * Phi^-1(top_prob), only valid if top_prob > 0.5
    cert_radius = np.where(
        top_probs > 0.5,
        sigma * scipy_norm.ppf(top_probs.clip(1e-6, 1 - 1e-6)),
        -np.inf
    )
    return cert_radius, top_probs, smooth_pred


def binary_search_min_eps(model, X, Y, eps_lo=0.01, eps_hi=0.5, k=10):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    N = X.size(0)
    min_eps = np.full(N, np.inf)
    for idx in range(N):
        x = X[idx:idx+1]
        y = Y[idx:idx+1]
        with torch.no_grad():
            if model(x).argmax(1).item() != y.item():
                min_eps[idx] = 0.0
                continue
        xa = C.fgsm(model, x, y, eps=eps_hi)
        with torch.no_grad():
            if model(xa).argmax(1).item() == y.item():
                continue
        lo, hi = eps_lo, eps_hi
        for _ in range(k):
            mid = (lo + hi) / 2
            xa = C.fgsm(model, x, y, eps=mid)
            with torch.no_grad():
                flipped = model(xa).argmax(1).item() != y.item()
            if flipped:
                hi = mid
            else:
                lo = mid
        min_eps[idx] = hi
    return min_eps


# ── MAIN ──────────────────────────────────────────────────────────────────────
log("=" * 72)
log("H268 - Certified vs Empirical Robustness")
log("=" * 72)
t_total = time.time()

log(f"\nLoading {DS} ...")
Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
Xev, Yev = Xte[:N_EVAL], Yte[:N_EVAL]
log(f"  Train: {Xtr.shape}  Eval: {Xev.shape}")

log("\n[1] Training model ...")
t0 = time.time()
model = C.build_model("cnn", META, width=WIDTH, seed=SEED)
C.train_model(model, Xtr, Ytr, epochs=EPOCHS)
with torch.no_grad():
    clean_acc = (model(Xev).argmax(1) == Yev).float().mean().item()
log(f"    Done in {time.time()-t0:.1f}s  clean_acc={clean_acc:.4f}")

log(f"\n[2] Randomised smoothing (sigma={RS_SIGMA}, K={RS_K}) ...")
t0 = time.time()
cert_radius, top_probs, smooth_pred = randomised_smoothing_cert(
    model, Xev, Yev, sigma=RS_SIGMA, K=RS_K)
n_certified = int((cert_radius > CERT_THRESH).sum())
log(f"    Done in {time.time()-t0:.1f}s")
log(f"    Certified robust (radius>0): {n_certified}/{N_EVAL}")
log(f"    Certified radius stats: mean={cert_radius[cert_radius>-np.inf].mean():.4f}  "
    f"median={np.median(cert_radius[cert_radius>-np.inf]):.4f}")
log(f"    Smooth accuracy (agrees with clean pred): "
    f"{float((smooth_pred == Yev.cpu().numpy()).mean()):.4f}")

log(f"\n[3] Binary-search empirical min_eps (k={BSEARCH_K}) ...")
t0 = time.time()
min_eps = binary_search_min_eps(model, Xev, Yev,
                                 eps_lo=EPS_LO, eps_hi=EPS_HI, k=BSEARCH_K)
log(f"    Done in {time.time()-t0:.1f}s")
finite = np.isfinite(min_eps)
log(f"    {int(finite.sum())}/{N_EVAL} samples empirically flippable")
log(f"    min_eps stats (finite): mean={min_eps[finite].mean():.4f}  "
    f"median={np.median(min_eps[finite]):.4f}")

log("\n[4] Correlation analysis ...")
# Only use samples that are: correctly classified, have finite min_eps, certified radius is valid
correct_clean = (smooth_pred == Yev.cpu().numpy())
valid = correct_clean & finite
log(f"  Valid samples for correlation: {int(valid.sum())}")

if valid.sum() > 10:
    cr_v = cert_radius[valid]
    me_v = min_eps[valid]
    rho, pval = spearmanr(cr_v, me_v)
    log(f"  Spearman rho(cert_radius, min_eps) = {rho:.4f}  p={pval:.4e}")
    log(f"  (Positive rho = certified robust samples also empirically robust)")
else:
    rho, pval = float("nan"), float("nan")
    log("  Too few valid samples.")

log("\n[5] Certified vs Empirical subset analysis ...")
# Are certified-robust samples ALSO empirically robust?
cert_mask = cert_radius > CERT_THRESH
emp_robust = min_eps > EPS_STD  # empirically robust against EPS_STD

cert_and_correct = cert_mask & correct_clean
emp_and_correct  = emp_robust & correct_clean

log(f"  Certified robust: {int(cert_mask.sum())}  "
    f"Empirically robust (min_eps>{EPS_STD}): {int(emp_robust.sum())}")

if cert_and_correct.sum() > 0:
    overlap = int((cert_and_correct & emp_and_correct).sum())
    frac = overlap / int(cert_and_correct.sum())
    log(f"  Of {int(cert_and_correct.sum())} certified-robust samples, "
        f"{overlap} ({frac:.1%}) are also empirically robust")
    log(f"  (Certified ⊆ Empirical: {'YES' if frac > 0.8 else 'NO, only ' + f'{frac:.0%}'})")

log("\n" + "=" * 72)
log("SUMMARY")
log("=" * 72)
log(f"  clean_acc             = {clean_acc:.4f}")
log(f"  n_certified_robust    = {n_certified}/{N_EVAL}")
log(f"  n_empirically_robust  = {int(emp_robust.sum())}/{N_EVAL}")
log(f"  Spearman rho(CR, ME)  = {rho:.4f}  p={pval:.4e}")

log("\nINTERPRETATION")
log("-" * 72)
if rho > 0.4 and pval < 0.05:
    verdict = "SUPPORTED: certified and empirical robustness rank samples similarly"
elif rho > 0.2:
    verdict = "PARTIAL: moderate agreement between certified and empirical robustness"
else:
    verdict = "NOT SUPPORTED: certified and empirical robustness are poorly correlated"
log(f"  Verdict: {verdict}")

log(f"\nTotal time: {time.time()-t_total:.1f}s")
log("=" * 72)

out_file = os.path.join(OUT_DIR, "h268_noise_certified_vs_empirical_robustness_output.txt")
with open(out_file, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults written to {out_file}", flush=True)

if __name__ == "__main__":
    pass
