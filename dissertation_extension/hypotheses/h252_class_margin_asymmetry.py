"""
H252 - Per-class adversarial vulnerability asymmetry.

Hypothesis: Fashion-MNIST classes are not equally easy to attack. Some classes
(e.g. Shirt, Pullover) sit closer to decision boundaries and should have
systematically higher FGSM/PGD ASR. We test this with a chi-squared test for
uniformity of attack successes across classes.

Pipeline
--------
1. Train CNN (seed=0, 10 epochs).
2. On Xte[:N_EVAL]: compute per-class FGSM ASR and PGD ASR.
3. Chi-squared test: are attack successes uniform across classes?
4. Rank classes by PGD ASR. Identify most/least robust.
5. Print bar-chart-style ASCII visualisation.
"""
import os, sys, time
import numpy as np
import torch
from scipy.stats import chi2_contingency, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ── config ───────────────────────────────────────────────────────────────────
DS          = "fashion_mnist"
N_EVAL      = 300
EPS         = 0.1
PGD_STEPS   = 10
SEED        = 0
META        = {"channels": 1, "size": 28, "n_classes": 10}
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot"]
OUT_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
OUT_FILE    = os.path.join(OUT_DIR, "h252_class_margin_asymmetry_output.txt")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

lines = []
def log(s=""):
    print(s)
    lines.append(str(s))

def ascii_bar(val, width=30):
    filled = int(round(val * width))
    return "[" + "#" * filled + " " * (width - filled) + "]"

# ── data & model ──────────────────────────────────────────────────────────────
log("=" * 70)
log("H252  Per-Class Adversarial Vulnerability Asymmetry")
log("=" * 70)
t0 = time.time()

log(f"\n[1] Loading {DS} ...")
Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]
log(f"    Eval set: {Xte.shape}, classes distribution: {np.bincount(Yte.cpu().numpy())}")

log(f"\n[2] Training CNN (seed={SEED}, epochs=10) ...")
model = C.build_model("cnn", META, width=32, seed=SEED)
C.train_model(model, Xtr, Ytr, epochs=10)
model.eval()

# ── attacks ───────────────────────────────────────────────────────────────────
log(f"\n[3] Running FGSM (eps={EPS}) ...")
for p in model.parameters(): p.requires_grad_(True)
X_fgsm = C.fgsm(model, Xte, Yte, eps=EPS)
with torch.no_grad():
    fgsm_preds = model(X_fgsm).argmax(1).cpu().numpy()
    clean_preds = model(Xte).argmax(1).cpu().numpy()
fgsm_flip = (fgsm_preds != Yte.cpu().numpy()).astype(int)
log(f"    FGSM overall ASR: {fgsm_flip.mean():.4f}")

log(f"\n[4] Running PGD (eps={EPS}, steps={PGD_STEPS}) ...")
X_pgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS)
with torch.no_grad():
    pgd_preds = model(X_pgd).argmax(1).cpu().numpy()
pgd_flip = (pgd_preds != Yte.cpu().numpy()).astype(int)
log(f"    PGD  overall ASR: {pgd_flip.mean():.4f}")

# ── per-class stats ────────────────────────────────────────────────────────────
log(f"\n[5] Per-Class Statistics")
log("-" * 80)
log(f"{'Class':<14} {'N':>4} {'CleanAcc':>9} {'FGSM_ASR':>9} {'PGD_ASR':>9}  PGD Bar")
log("-" * 80)

Y_np = Yte.cpu().numpy()
per_class_fgsm = []
per_class_pgd  = []
per_class_n    = []

# chi-squared table: [class x {success, failure}] for FGSM and PGD
fgsm_contingency = []
pgd_contingency  = []

for c in range(10):
    idx = np.where(Y_np == c)[0]
    n   = len(idx)
    if n == 0:
        per_class_fgsm.append(np.nan)
        per_class_pgd.append(np.nan)
        per_class_n.append(0)
        log(f"{CLASS_NAMES[c]:<14} {n:>4}  (no samples)")
        continue

    clean_acc = (clean_preds[idx] == Y_np[idx]).mean()
    fgsm_asr  = fgsm_flip[idx].mean()
    pgd_asr   = pgd_flip[idx].mean()
    per_class_fgsm.append(fgsm_asr)
    per_class_pgd.append(pgd_asr)
    per_class_n.append(n)

    fgsm_contingency.append([int(fgsm_flip[idx].sum()), int((1 - fgsm_flip[idx]).sum())])
    pgd_contingency.append([int(pgd_flip[idx].sum()),   int((1 - pgd_flip[idx]).sum())])

    bar = ascii_bar(pgd_asr)
    log(f"{CLASS_NAMES[c]:<14} {n:>4}  {clean_acc:>8.3f}  {fgsm_asr:>8.3f}  {pgd_asr:>8.3f}  {bar}")

log("-" * 80)

# ── chi-squared test ───────────────────────────────────────────────────────────
log(f"\n[6] Chi-Squared Test for Uniformity of Attack Successes")

fgsm_table = np.array(fgsm_contingency)
pgd_table  = np.array(pgd_contingency)

try:
    chi2_fgsm, p_fgsm, dof_fgsm, _ = chi2_contingency(fgsm_table + 0.5)  # Yates correction via smoothing
    chi2_pgd,  p_pgd,  dof_pgd,  _ = chi2_contingency(pgd_table + 0.5)
except Exception as e:
    log(f"    Chi-squared failed: {e}")
    chi2_fgsm = p_fgsm = dof_fgsm = float('nan')
    chi2_pgd  = p_pgd  = dof_pgd  = float('nan')

log(f"    FGSM chi2={chi2_fgsm:.4f}, dof={dof_fgsm}, p={p_fgsm:.6f}")
log(f"    PGD  chi2={chi2_pgd:.4f},  dof={dof_pgd},  p={p_pgd:.6f}")

sig_fgsm = "SIGNIFICANT" if p_fgsm < 0.05 else "not significant"
sig_pgd  = "SIGNIFICANT" if p_pgd  < 0.05 else "not significant"
log(f"    FGSM: attack success is NOT uniform across classes -> {sig_fgsm}")
log(f"    PGD:  attack success is NOT uniform across classes -> {sig_pgd}")

# ── rank by PGD ASR ────────────────────────────────────────────────────────────
log(f"\n[7] Class Rankings by PGD ASR")
pgd_arr = np.array(per_class_pgd)
valid   = [(i, v) for i, v in enumerate(pgd_arr) if not np.isnan(v)]
ranked  = sorted(valid, key=lambda x: x[1], reverse=True)

log("    Most vulnerable (highest PGD ASR):")
for rank, (ci, asr) in enumerate(ranked[:3], 1):
    log(f"      #{rank}  {CLASS_NAMES[ci]:<14}  PGD_ASR={asr:.4f}  FGSM_ASR={per_class_fgsm[ci]:.4f}")

log("    Most robust (lowest PGD ASR):")
for rank, (ci, asr) in enumerate(ranked[-3:][::-1], 1):
    log(f"      #{rank}  {CLASS_NAMES[ci]:<14}  PGD_ASR={asr:.4f}  FGSM_ASR={per_class_fgsm[ci]:.4f}")

# ── Spearman between FGSM and PGD ASR ────────────────────────────────────────
fgsm_arr = np.array(per_class_fgsm)
valid_mask = ~(np.isnan(fgsm_arr) | np.isnan(pgd_arr))
rho, p = spearmanr(fgsm_arr[valid_mask], pgd_arr[valid_mask])
log(f"\n[8] Spearman rho(FGSM_ASR, PGD_ASR) across classes = {rho:+.4f}  (p={p:.4f})")

log(f"\n[9] Interpretation")
if p_pgd < 0.05:
    log("    Chi-squared test confirms non-uniform vulnerability: some classes are")
    log("    systematically easier to attack (reject H0 of uniformity).")
else:
    log("    Chi-squared test does NOT reject uniformity: class differences may be")
    log("    due to chance.")

log(f"\nTotal time: {time.time() - t0:.1f}s")
log("=" * 70)

with open(OUT_FILE, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    pass
