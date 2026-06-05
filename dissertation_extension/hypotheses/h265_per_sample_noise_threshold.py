"""
H265 - Per-sample noise destabilisation threshold.

For each test sample: find the minimum sigma such that adding N(0,sigma^2) noise
K=50 times causes the model to misclassify it MORE than 50% of the time.
Call this sigma_destabilise per sample.

Hypothesis: samples with LOW sigma_destabilise (easily destabilised by noise) are
the same samples easily attacked adversarially — both probe the same boundary distance.
Correlate sigma_destabilise with margin (Spearman rho) and AUROC predicting FGSM success.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ── config ────────────────────────────────────────────────────────────────────
DS          = "fashion_mnist"
N_EVAL      = 300       # keep small — K=50 noise passes per sample
EPS         = 0.1
SEED        = 0
META        = {"channels": 1, "size": 28, "n_classes": 10}
EPOCHS      = 10
WIDTH       = 32
K           = 50        # noise samples per test sample
SIGMA_GRID  = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2]

OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def compute_sigma_destabilise(model, X, Y, sigma_grid, K=50):
    """For each sample, find smallest sigma in sigma_grid where error rate > 50%."""
    model.eval()
    N = X.size(0)
    sigma_dest = np.full(N, np.inf)

    for si, sigma in enumerate(sigma_grid):
        if sigma == 0.0:
            continue
        # Monte-Carlo noise: K replicates
        # Process in batches to avoid OOM
        err_counts = np.zeros(N)
        for k_rep in range(K):
            noise = torch.randn_like(X) * sigma
            Xn = (X + noise).clamp(0, 1)
            with torch.no_grad():
                preds = model(Xn).argmax(1).cpu()
            wrong = (preds != Y.cpu()).numpy()
            err_counts += wrong.astype(float)

        err_rate = err_counts / K
        newly_destabilised = (err_rate > 0.5) & np.isinf(sigma_dest)
        sigma_dest[newly_destabilised] = sigma
        log(f"  sigma={sigma:.2f}: {int(newly_destabilised.sum())} newly destabilised "
            f"({int(np.isfinite(sigma_dest).sum())}/{N} total)")

    return sigma_dest


# ── MAIN ──────────────────────────────────────────────────────────────────────
log("=" * 72)
log("H265 - Per-Sample Noise Destabilisation Threshold")
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
log(f"    Done in {time.time()-t0:.1f}s")

# Baseline accuracy
with torch.no_grad():
    logits = model(Xev)
    clean_acc = (logits.argmax(1) == Yev).float().mean().item()
log(f"    Clean acc: {clean_acc:.4f}")

log(f"\n[2] Computing sigma_destabilise (K={K} noise reps per sample) ...")
t0 = time.time()
sigma_dest = compute_sigma_destabilise(model, Xev, Yev, SIGMA_GRID, K=K)
log(f"    Done in {time.time()-t0:.1f}s")
finite = np.isfinite(sigma_dest)
log(f"    {int(finite.sum())}/{N_EVAL} samples destabilised within sigma_grid")
log(f"    mean sigma_dest (finite) = {sigma_dest[finite].mean():.4f}"
    if finite.any() else "    No samples destabilised")

log("\n[3] Computing FGSM vulnerability ...")
for p in model.parameters():
    p.requires_grad_(True)
Xf = C.fgsm(model, Xev, Yev, eps=EPS)
with torch.no_grad():
    fgsm_preds = model(Xf).argmax(1).cpu()
    clean_preds = model(Xev).argmax(1).cpu()
correctly_classified = (clean_preds == Yev.cpu()).numpy()
fgsm_success = (fgsm_preds != Yev.cpu()).numpy()
fgsm_asr = float(fgsm_success[correctly_classified].mean()) if correctly_classified.any() else float("nan")
log(f"    FGSM ASR = {fgsm_asr:.4f}")

log("\n[4] Computing margins ...")
mar = C.margin(model, Xev, Yev)
log(f"    mean_margin = {mar.mean():.4f}")

log("\n[5] Correlation analysis ...")
# Use only finite sigma_dest values
mask = finite & correctly_classified
log(f"    Using {int(mask.sum())} samples (finite sigma_dest AND correctly classified)")

if mask.sum() > 10:
    sd_m = sigma_dest[mask]
    mar_m = mar[mask]
    fgsm_m = fgsm_success[mask].astype(float)

    rho_margin, p_margin = spearmanr(sd_m, mar_m)
    log(f"    Spearman rho(sigma_dest, margin) = {rho_margin:.4f}  p={p_margin:.4e}")
    log(f"    (Positive rho means larger sigma_dest = more robust — consistent with hypothesis)")

    if len(np.unique(fgsm_m)) > 1:
        # AUROC: does sigma_dest predict FGSM success?
        # Low sigma_dest → should predict high FGSM success → use -sigma_dest as score
        try:
            auroc = roc_auc_score(fgsm_m, -sd_m)
        except Exception:
            auroc = float("nan")
        log(f"    AUROC(-sigma_dest → FGSM success) = {auroc:.4f}")
        log(f"    (>0.5 means low sigma_dest predicts FGSM success — supports hypothesis)")
    else:
        log("    Skipping AUROC: only one class in FGSM success labels")

    # Distribution summary
    log("\n    sigma_dest distribution:")
    for q in [0, 25, 50, 75, 100]:
        log(f"      p{q:>3}: {np.percentile(sd_m, q):.4f}")
else:
    rho_margin, auroc = float("nan"), float("nan")
    log("    Too few samples for correlation.")

log("\nINTERPRETATION")
log("-" * 72)
if abs(rho_margin) > 0.3 and rho_margin > 0:
    verdict = "SUPPORTED: sigma_dest correlates with margin — noise and adversarial probe same geometry"
elif abs(rho_margin) > 0.15:
    verdict = "WEAK SUPPORT: moderate correlation between sigma_dest and margin"
else:
    verdict = "NOT SUPPORTED: sigma_dest does not correlate with adversarial vulnerability"
log(f"  Verdict: {verdict}")
log(f"  rho(sigma_dest, margin)={rho_margin:.4f}")

log(f"\nTotal time: {time.time()-t_total:.1f}s")
log("=" * 72)

out_file = os.path.join(OUT_DIR, "h265_per_sample_noise_threshold_output.txt")
with open(out_file, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults written to {out_file}", flush=True)

if __name__ == "__main__":
    pass
