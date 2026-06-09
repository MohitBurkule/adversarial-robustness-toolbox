"""
H254 - Cross-epsilon vulnerability rank preservation.

Hypothesis: The same samples remain vulnerable across different perturbation
budgets. If vulnerability rankings are stable, a sample that is easy to attack
at eps=0.01 should also be easy at eps=0.2. We compute per-sample PGD success
at eps in {0.01, 0.05, 0.1, 0.2} and compute all pairwise Spearman correlations
of the resulting binary vulnerability vectors.

Pipeline
--------
1. Train CNN (seed=0, 10 epochs).
2. For each eps in [0.01, 0.05, 0.1, 0.2]:
   - Run PGD (steps=10) on Xte[:N_EVAL].
   - Record binary success vector (1 = attack succeeded).
3. Pairwise Spearman rho matrix over the 4 eps conditions.
4. Also compute % of samples that are consistently vulnerable (success at ALL eps)
   vs consistently robust (failure at ALL eps) vs inconsistent.
"""
import os, sys, time
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ── config ───────────────────────────────────────────────────────────────────
DS        = "fashion_mnist"
N_EVAL    = 300
EPS_LIST  = [0.01, 0.05, 0.1, 0.2]
PGD_STEPS = 10
SEED      = 0
META      = {"channels": 1, "size": 28, "n_classes": 10}
OUT_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "fashion_mnist")
OUT_FILE  = os.path.join(OUT_DIR, "h254_cross_eps_rank_preservation_output.txt")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

lines = []
def log(s=""):
    print(s)
    lines.append(str(s))

# ── data & model ──────────────────────────────────────────────────────────────
log("=" * 70)
log("H254  Cross-Epsilon Vulnerability Rank Preservation")
log("=" * 70)
t0 = time.time()

log(f"\n[1] Loading {DS} ...")
Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]
log(f"    Eval: {Xte.shape}")

log(f"\n[2] Training CNN (seed={SEED}, epochs=10) ...")
model = C.build_model("cnn", META, width=32, seed=SEED)
C.train_model(model, Xtr, Ytr, epochs=10)
model.eval()

# ── attack at each eps ────────────────────────────────────────────────────────
log(f"\n[3] PGD attacks at eps in {EPS_LIST} ...")
for p in model.parameters(): p.requires_grad_(True)

success_matrix = np.zeros((len(EPS_LIST), N_EVAL), dtype=int)

for i, eps in enumerate(EPS_LIST):
    log(f"    eps={eps:.3f} ...")
    X_pgd = C.pgd(model, Xte, Yte, eps=eps, steps=PGD_STEPS)
    with torch.no_grad():
        preds = model(X_pgd).argmax(1).cpu().numpy()
    success = (preds != Yte.cpu().numpy()).astype(int)
    success_matrix[i] = success
    log(f"      ASR = {success.mean():.4f}")

# ── pairwise Spearman rho matrix ──────────────────────────────────────────────
log(f"\n[4] Pairwise Spearman rho (per-sample success vectors)")
n_eps = len(EPS_LIST)

# header
header = f"{'':>8}"
for eps in EPS_LIST:
    header += f"  eps={eps:.2f}"
log(header)

rho_matrix = np.zeros((n_eps, n_eps))
p_matrix   = np.zeros((n_eps, n_eps))

for i in range(n_eps):
    row = f"eps={EPS_LIST[i]:.2f}"
    for j in range(n_eps):
        if i == j:
            rho_matrix[i, j] = 1.0
            row += f"  {'1.0000':>8}"
        else:
            rho, p = spearmanr(success_matrix[i], success_matrix[j])
            rho_matrix[i, j] = rho
            p_matrix[i, j]   = p
            row += f"  {rho:>+8.4f}"
    log(row)

log(f"\n[5] p-values for each pair:")
for i in range(n_eps):
    for j in range(i+1, n_eps):
        rho = rho_matrix[i, j]
        p   = p_matrix[i, j]
        sig = "significant" if p < 0.05 else "not significant"
        log(f"    eps={EPS_LIST[i]:.2f} vs eps={EPS_LIST[j]:.2f}: rho={rho:+.4f}, p={p:.6f} -> {sig}")

# ── consistency analysis ──────────────────────────────────────────────────────
log(f"\n[6] Sample Consistency Analysis")
n_successes = success_matrix.sum(axis=0)  # (N,): count of eps levels where success

always_vuln    = (n_successes == n_eps).sum()
always_robust  = (n_successes == 0).sum()
inconsistent   = N_EVAL - always_vuln - always_robust

log(f"    Always vulnerable (success at all {n_eps} eps): {always_vuln:>4} ({always_vuln/N_EVAL:.3f})")
log(f"    Always robust     (failure at all {n_eps} eps): {always_robust:>4} ({always_robust/N_EVAL:.3f})")
log(f"    Inconsistent      (mixed across eps):           {inconsistent:>4} ({inconsistent/N_EVAL:.3f})")

log(f"\n    Distribution of per-sample success counts:")
for k in range(n_eps + 1):
    count = (n_successes == k).sum()
    bar   = "#" * int(count / N_EVAL * 40)
    log(f"    successes={k}/{n_eps}: {count:>4} ({count/N_EVAL:.3f})  {bar}")

# ── monotonicity check ────────────────────────────────────────────────────────
log(f"\n[7] Monotonicity: does ASR increase monotonically with eps?")
asr_per_eps = success_matrix.mean(axis=1)
for i, eps in enumerate(EPS_LIST):
    log(f"    eps={eps:.2f} -> ASR={asr_per_eps[i]:.4f}")

is_monotone = all(asr_per_eps[i] <= asr_per_eps[i+1] + 1e-6 for i in range(len(asr_per_eps)-1))
log(f"    Monotone increasing: {is_monotone}")

# ── mean rank correlation ─────────────────────────────────────────────────────
upper_rhos = [rho_matrix[i, j] for i in range(n_eps) for j in range(i+1, n_eps)]
mean_rho   = np.mean(upper_rhos)
log(f"\n[8] Mean pairwise Spearman rho across all eps pairs: {mean_rho:+.4f}")

log(f"\n[9] Interpretation")
if mean_rho > 0.5:
    log("    Rankings are strongly preserved across eps: the same samples tend to")
    log("    remain vulnerable regardless of perturbation budget.")
elif mean_rho > 0.2:
    log("    Rankings are moderately preserved: significant overlap in vulnerable")
    log("    samples but notable reshuffling at different eps.")
else:
    log("    Rankings are weakly preserved: different eps budgets expose largely")
    log("    different subsets of vulnerable samples.")

log(f"\nTotal time: {time.time() - t0:.1f}s")
log("=" * 70)

with open(OUT_FILE, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    pass
