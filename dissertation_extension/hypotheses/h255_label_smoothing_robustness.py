"""
H255 - Label smoothing as an adversarial robustness regulariser.

Hypothesis: Training with label smoothing encourages the model to produce less
peaked logit distributions, which may reduce adversarial vulnerability by
lowering the gradient signal available to the attacker. We test 4 smoothing
levels (0.0, 0.05, 0.1, 0.2) and measure clean accuracy, FGSM ASR, PGD ASR,
and mean margin.

Pipeline
--------
1. Train 4 CNNs with label_smooth in [0.0, 0.05, 0.1, 0.2], same seed.
2. Evaluate each on Xte[:N_EVAL]: clean acc, FGSM ASR, PGD ASR, mean margin.
3. Print results table.
4. Spearman rho between smoothing alpha and each robustness metric.
"""
import os, sys, time
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ── config ───────────────────────────────────────────────────────────────────
DS             = "fashion_mnist"
N_EVAL         = 300
EPS            = 0.1
PGD_STEPS      = 10
SEED           = 0
META           = {"channels": 1, "size": 28, "n_classes": 10}
SMOOTH_ALPHAS  = [0.0, 0.05, 0.1, 0.2]
OUT_DIR        = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "fashion_mnist")
OUT_FILE       = os.path.join(OUT_DIR, "h255_label_smoothing_robustness_output.txt")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

lines = []
def log(s=""):
    print(s)
    lines.append(str(s))

# ── data ──────────────────────────────────────────────────────────────────────
log("=" * 70)
log("H255  Label Smoothing vs Adversarial Robustness")
log("=" * 70)
t0 = time.time()

log(f"\n[1] Loading {DS} ...")
Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]
log(f"    Train: {Xtr.shape}  Eval: {Xte.shape}")

# ── per-smoothing loop ────────────────────────────────────────────────────────
log(f"\n[2] Training 4 CNNs with label_smooth in {SMOOTH_ALPHAS} ...")

results = []
for alpha in SMOOTH_ALPHAS:
    log(f"\n  --- label_smooth={alpha} ---")
    model = C.build_model("cnn", META, width=32, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=10, label_smooth=alpha)
    model.eval()

    # clean accuracy
    with torch.no_grad():
        logits = model(Xte)
        clean_acc = (logits.argmax(1).cpu() == Yte.cpu()).float().mean().item()
    log(f"    clean acc   = {clean_acc:.4f}")

    # FGSM
    for p in model.parameters(): p.requires_grad_(True)
    X_fgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    with torch.no_grad():
        fgsm_asr = (model(X_fgsm).argmax(1).cpu() != Yte.cpu()).float().mean().item()
    log(f"    FGSM ASR    = {fgsm_asr:.4f}")

    # PGD
    X_pgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS)
    with torch.no_grad():
        pgd_asr = (model(X_pgd).argmax(1).cpu() != Yte.cpu()).float().mean().item()
    log(f"    PGD  ASR    = {pgd_asr:.4f}")

    # mean margin
    mar = C.margin(model, Xte, Yte)
    mean_margin = float(mar.mean())
    log(f"    mean margin = {mean_margin:.4f}")

    # logit confidence (mean max softmax)
    with torch.no_grad():
        import torch.nn.functional as F
        probs = F.softmax(model(Xte), dim=1)
        mean_conf = probs.max(dim=1).values.mean().item()
    log(f"    mean conf   = {mean_conf:.4f}")

    results.append({"alpha": alpha, "clean_acc": clean_acc, "fgsm_asr": fgsm_asr,
                    "pgd_asr": pgd_asr, "mean_margin": mean_margin, "mean_conf": mean_conf})

# ── summary table ─────────────────────────────────────────────────────────────
log(f"\n[3] Summary Table")
log("-" * 78)
log(f"{'Smooth':>8} {'CleanAcc':>10} {'FGSM_ASR':>10} {'PGD_ASR':>9} {'MeanMargin':>12} {'MeanConf':>10}")
log("-" * 78)
for r in results:
    log(f"{r['alpha']:>8.2f} {r['clean_acc']:>10.4f} {r['fgsm_asr']:>10.4f} "
        f"{r['pgd_asr']:>9.4f} {r['mean_margin']:>12.4f} {r['mean_conf']:>10.4f}")
log("-" * 78)

# ── delta relative to baseline (alpha=0) ─────────────────────────────────────
log(f"\n[4] Delta relative to baseline (alpha=0.0)")
base = results[0]
log(f"    {'Smooth':>8} {'dCleanAcc':>11} {'dFGSM_ASR':>11} {'dPGD_ASR':>10} {'dMeanMargin':>13}")
for r in results[1:]:
    log(f"    {r['alpha']:>8.2f} "
        f"{r['clean_acc'] - base['clean_acc']:>+11.4f} "
        f"{r['fgsm_asr'] - base['fgsm_asr']:>+11.4f} "
        f"{r['pgd_asr'] - base['pgd_asr']:>+10.4f} "
        f"{r['mean_margin'] - base['mean_margin']:>+13.4f}")

# ── Spearman correlations ─────────────────────────────────────────────────────
log(f"\n[5] Spearman Correlations (smoothing alpha vs metrics)")
alphas   = np.array([r["alpha"]       for r in results])
fgsm_arr = np.array([r["fgsm_asr"]   for r in results])
pgd_arr  = np.array([r["pgd_asr"]    for r in results])
mar_arr  = np.array([r["mean_margin"] for r in results])
acc_arr  = np.array([r["clean_acc"]  for r in results])

rho_fgsm, p_fgsm = spearmanr(alphas, fgsm_arr)
rho_pgd,  p_pgd  = spearmanr(alphas, pgd_arr)
rho_mar,  p_mar  = spearmanr(alphas, mar_arr)
rho_acc,  p_acc  = spearmanr(alphas, acc_arr)

log(f"    rho(alpha, FGSM_ASR)   = {rho_fgsm:+.4f}  (p={p_fgsm:.4f})")
log(f"    rho(alpha, PGD_ASR)    = {rho_pgd:+.4f}  (p={p_pgd:.4f})")
log(f"    rho(alpha, mean_margin)= {rho_mar:+.4f}  (p={p_mar:.4f})")
log(f"    rho(alpha, clean_acc)  = {rho_acc:+.4f}  (p={p_acc:.4f})")

log(f"\n[6] Interpretation")
robustness_improves = rho_pgd < 0  # higher alpha -> lower PGD ASR = more robust
log(f"    Smoothing vs PGD ASR: rho={rho_pgd:+.3f} -> "
    f"{'robustness IMPROVES with smoothing' if robustness_improves else 'robustness DOES NOT improve with smoothing'}")

sig_pgd = "significant" if p_pgd < 0.05 else "not significant"
log(f"    Effect on PGD robustness: {sig_pgd} at alpha=0.05")

if rho_acc < 0 and abs(rho_acc) > 0.5:
    log("    Warning: label smoothing reduces clean accuracy noticeably.")
else:
    log("    Clean accuracy impact: minimal or positive.")

clean_cost    = acc_arr[-1] - acc_arr[0]
robust_gain   = pgd_arr[0] - pgd_arr[-1]   # positive = ASR decreased = more robust
log(f"\n    At alpha=0.2 vs baseline:")
log(f"      Clean acc change: {clean_cost:+.4f}")
log(f"      PGD ASR change:   {-robust_gain:+.4f}  (negative = more robust)")

log(f"\nTotal time: {time.time() - t0:.1f}s")
log("=" * 70)

with open(OUT_FILE, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    pass
