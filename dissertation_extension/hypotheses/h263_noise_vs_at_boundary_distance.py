"""
H263 - Noise training vs adversarial training: boundary distance comparison.

Compare boundary distance (measured as min_eps_to_flip via binary search over eps)
between:
  1. Standard training
  2. Gaussian noise sigma=0.2
  3. PGD adversarial training

For each model, binary-search eps in [0.01..0.5] to find smallest eps where FGSM
flips each sample. Plot distribution of min_eps per model. Spearman rho between
min_eps and clean margin.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ── config ────────────────────────────────────────────────────────────────────
DS        = "fashion_mnist"
N_EVAL    = 200          # keep small — binary search is expensive
EPS_STD   = 0.1
SIGMA     = 0.2
AT_EPS    = 0.1
AT_STEPS  = 7
SEED      = 0
META      = {"channels": 1, "size": 28, "n_classes": 10}
EPOCHS    = 10
WIDTH     = 32
EPS_LO    = 0.01
EPS_HI    = 0.5
BSEARCH_K = 10           # binary search iterations

OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def train_with_noise(model, X, Y, sigma, epochs=10, lr=0.05, batch=128):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=X.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = X[idx], Y[idx]
            if sigma > 0:
                xb = (xb + torch.randn_like(xb) * sigma).clamp(0, 1)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def binary_search_min_eps(model, X, Y, eps_lo=0.01, eps_hi=0.5, k=10):
    """For each sample, find minimum FGSM eps that flips prediction.
    Returns numpy array of shape (N,); inf if never flipped."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    N = X.size(0)
    min_eps = np.full(N, np.inf)
    # For each sample run binary search
    for idx in range(N):
        x = X[idx:idx+1]
        y = Y[idx:idx+1]
        with torch.no_grad():
            if model(x).argmax(1).item() != y.item():
                min_eps[idx] = 0.0
                continue
        lo, hi = eps_lo, eps_hi
        # check if flippable at hi
        xa = C.fgsm(model, x, y, eps=hi)
        with torch.no_grad():
            if model(xa).argmax(1).item() == y.item():
                continue  # never flipped; leave inf
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
log("H263 - Noise vs Adversarial Training: Boundary Distance")
log("=" * 72)
t_total = time.time()

log(f"\nLoading {DS} ...")
Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
Xev, Yev = Xte[:N_EVAL], Yte[:N_EVAL]
log(f"  Train: {Xtr.shape}  Eval: {Xev.shape}")

models = {}

log("\n[1] Training STANDARD model ...")
t0 = time.time()
m = C.build_model("cnn", META, width=WIDTH, seed=SEED)
train_with_noise(m, Xtr, Ytr, sigma=0.0, epochs=EPOCHS)
models["standard"] = m
log(f"    Done in {time.time()-t0:.1f}s")

log(f"\n[2] Training NOISE model (sigma={SIGMA}) ...")
t0 = time.time()
m = C.build_model("cnn", META, width=WIDTH, seed=SEED)
train_with_noise(m, Xtr, Ytr, sigma=SIGMA, epochs=EPOCHS)
models["noise"] = m
log(f"    Done in {time.time()-t0:.1f}s")

log(f"\n[3] Training PGD-AT model (eps={AT_EPS}, steps={AT_STEPS}) ...")
t0 = time.time()
m = C.build_model("cnn", META, width=WIDTH, seed=SEED)
C.train_model(m, Xtr, Ytr, epochs=EPOCHS, adv_train=True, adv_eps=AT_EPS, adv_steps=AT_STEPS)
models["pgd_at"] = m
log(f"    Done in {time.time()-t0:.1f}s")

log(f"\n[4] Binary-search min_eps for each model (N={N_EVAL}, k={BSEARCH_K}) ...")
min_eps_results = {}
margins_results = {}

for name, model in models.items():
    t0 = time.time()
    log(f"  [{name}] computing min_eps ...")
    min_eps = binary_search_min_eps(model, Xev, Yev,
                                     eps_lo=EPS_LO, eps_hi=EPS_HI, k=BSEARCH_K)
    mar = C.margin(model, Xev, Yev)

    finite = np.isfinite(min_eps)
    n_fin = int(finite.sum())
    log(f"    Finite flips: {n_fin}/{N_EVAL}  "
        f"mean_min_eps={min_eps[finite].mean():.4f}  "
        f"median={np.median(min_eps[finite]):.4f}  "
        f"p25={np.percentile(min_eps[finite],25):.4f}  "
        f"p75={np.percentile(min_eps[finite],75):.4f}")

    rho, pval = spearmanr(min_eps[finite], mar[finite])
    log(f"    Spearman rho(min_eps, margin) = {rho:.4f}  p={pval:.4e}")

    min_eps_results[name] = min_eps
    margins_results[name] = mar
    log(f"    Done in {time.time()-t0:.1f}s")

# ── Summary table ─────────────────────────────────────────────────────────────
log("\n" + "=" * 72)
log("SUMMARY TABLE")
log("=" * 72)
log(f"{'Model':>10}  {'mean_min_eps':>13}  {'median_min_eps':>15}  {'mean_margin':>12}  {'spearman_rho':>13}")
log("-" * 72)
for name in ["standard", "noise", "pgd_at"]:
    me = min_eps_results[name]
    mar = margins_results[name]
    finite = np.isfinite(me)
    rho, _ = spearmanr(me[finite], mar[finite]) if finite.sum() > 5 else (float('nan'), 1.0)
    log(f"{name:>10}  {me[finite].mean():>13.4f}  {np.median(me[finite]):>15.4f}  "
        f"{mar.mean():>12.4f}  {rho:>13.4f}")

log("\nINTERPRETATION")
log("-" * 72)
for a, b in [("standard", "noise"), ("standard", "pgd_at"), ("noise", "pgd_at")]:
    me_a = min_eps_results[a]
    me_b = min_eps_results[b]
    fa = np.isfinite(me_a)
    fb = np.isfinite(me_b)
    both = fa & fb
    if both.sum() > 0:
        delta = me_b[both].mean() - me_a[both].mean()
        log(f"  {b} vs {a}: mean min_eps delta = {delta:+.4f} "
            f"({'more robust' if delta > 0 else 'less robust'})")

log(f"\nTotal time: {time.time()-t_total:.1f}s")
log("=" * 72)

out_file = os.path.join(OUT_DIR, "h263_noise_vs_at_boundary_distance_output.txt")
with open(out_file, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults written to {out_file}", flush=True)

if __name__ == "__main__":
    pass
