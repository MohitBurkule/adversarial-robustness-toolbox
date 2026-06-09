"""
H261 - Targeted noise hardening: adding more training noise to vulnerable samples.

Hypothesis: For samples that are easy to attack (low margin / high vulnerability),
adding more Gaussian noise during training of THOSE specific samples will push the
decision boundary further away, making them harder to attack. Samples that are
already hard to attack get normal training.

Experiment design:
1. Train a baseline CNN (standard training, no extra noise)
2. Compute per-sample FGSM vulnerability on the TRAINING set (proxy: logit margin)
3. Identify the bottom 25% most vulnerable training samples ("easy targets")
4. Retrain from scratch with adaptive noise: easy-target samples get sigma=0.15
   added during training, remaining samples get sigma=0.02 (minimal noise)
5. Also train a "uniform noise" model where ALL samples get sigma=0.1 noise (ablation)
6. Evaluate all 3 models on test set: clean acc, FGSM ASR, PGD ASR, mean margin
7. Key question: does the targeted model improve robustness on the originally-vulnerable
   samples specifically, or does it just trade clean accuracy?

Also measure: do the originally-vulnerable training samples have HIGHER margin
improvement than the non-targeted samples in the retrained model?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ── config ───────────────────────────────────────────────────────────────────
DS          = "fashion_mnist"
N_EVAL      = 500
EPS         = 0.1
PGD_STEPS   = 10
SEED        = 0
META        = {"channels": 1, "size": 28, "n_classes": 10}
EPOCHS      = 10
WIDTH       = 32
VULN_FRAC   = 0.25   # bottom 25% by margin = "easy targets"
SIGMA_HIGH  = 0.15   # noise for vulnerable samples
SIGMA_LOW   = 0.02   # noise for non-vulnerable samples
SIGMA_UNI   = 0.1    # noise for uniform-noise ablation

OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "fashion_mnist")
OUT_FILE = os.path.join(OUT_DIR, "h261_targeted_noise_hardening_output.txt")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

lines = []
def log(s=""):
    print(s)
    lines.append(str(s))


# ── custom training function with per-sample noise ────────────────────────────
def train_with_noise(model, X, Y, noise_sigma, epochs=10, lr=0.01, batch=128):
    """Train model with Gaussian noise sigma applied uniformly to all samples."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    ds = TensorDataset(X.cpu(), Y.cpu())
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    for ep in range(epochs):
        total_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(C.DEVICE)
            yb = yb.to(C.DEVICE)
            noise = torch.randn_like(xb) * noise_sigma
            xb_n = (xb + noise).clamp(0.0, 1.0)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_n), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    model.eval()


def train_with_adaptive_noise(model, X, Y, vuln_mask, sigma_high, sigma_low,
                               epochs=10, lr=0.01, batch=128):
    """
    Train model with per-sample adaptive noise:
    - samples where vuln_mask[i]=True  get N(0, sigma_high^2) noise
    - samples where vuln_mask[i]=False get N(0, sigma_low^2)  noise
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # store mask as tensor on CPU alongside data
    mask_t = vuln_mask.cpu()   # bool tensor, shape [N]
    ds = TensorDataset(X.cpu(), Y.cpu(), mask_t.float())
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    for ep in range(epochs):
        for xb, yb, mb in loader:
            xb = xb.to(C.DEVICE)
            yb = yb.to(C.DEVICE)
            mb = mb.to(C.DEVICE)  # 1.0 where vulnerable, 0.0 otherwise
            sigma = mb * sigma_high + (1.0 - mb) * sigma_low  # per-sample sigma
            # sigma shape: [B], expand to [B,1,1,1] for broadcasting
            sigma = sigma.view(-1, 1, 1, 1)
            noise = torch.randn_like(xb) * sigma
            xb_n = (xb + noise).clamp(0.0, 1.0)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_n), yb)
            loss.backward()
            opt.step()
    model.eval()


def eval_model(model, Xte, Yte):
    """Return dict of clean_acc, fgsm_asr, pgd_asr, mean_margin."""
    Xte_ev = Xte[:N_EVAL]
    Yte_ev = Yte[:N_EVAL]
    model.eval()
    with torch.no_grad():
        logits = model(Xte_ev)
        clean_acc = (logits.argmax(1).cpu() == Yte_ev.cpu()).float().mean().item()

    for p in model.parameters(): p.requires_grad_(True)
    X_fgsm = C.fgsm(model, Xte_ev, Yte_ev, eps=EPS)
    with torch.no_grad():
        fgsm_asr = (model(X_fgsm).argmax(1).cpu() != Yte_ev.cpu()).float().mean().item()

    X_pgd = C.pgd(model, Xte_ev, Yte_ev, eps=EPS, steps=PGD_STEPS)
    with torch.no_grad():
        pgd_asr = (model(X_pgd).argmax(1).cpu() != Yte_ev.cpu()).float().mean().item()

    mar = C.margin(model, Xte_ev, Yte_ev)
    mean_margin = float(mar.mean())

    return {"clean_acc": clean_acc, "fgsm_asr": fgsm_asr,
            "pgd_asr": pgd_asr, "mean_margin": mean_margin}


# ── MAIN ─────────────────────────────────────────────────────────────────────
log("=" * 72)
log("H261 - Targeted Noise Hardening for Vulnerable Samples")
log("=" * 72)
t_total = time.time()

log(f"\n[1] Loading {DS} ...")
Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
log(f"    Train: {Xtr.shape}  Test: {Xte.shape}")

# ── Step 1: Train baseline model ──────────────────────────────────────────────
log(f"\n[2] Training BASELINE model (standard, no extra noise) ...")
t0 = time.time()
model_base = C.build_model("cnn", META, width=WIDTH, seed=SEED)
C.train_model(model_base, Xtr, Ytr, epochs=EPOCHS)
log(f"    Done in {time.time()-t0:.1f}s")

# ── Step 2: Compute per-sample margin on training set ─────────────────────────
log(f"\n[3] Computing per-sample margin on training set ...")
t0 = time.time()
# Use a subset for speed if training set is large; use full set for correctness
model_base.eval()
# margin returns numpy array
mar_train = C.margin(model_base, Xtr, Ytr)
log(f"    Margin computed for {len(mar_train)} training samples in {time.time()-t0:.1f}s")
log(f"    Margin stats: mean={mar_train.mean():.4f}  min={mar_train.min():.4f}  "
    f"p25={np.percentile(mar_train,25):.4f}  max={mar_train.max():.4f}")

# ── Step 3: Identify bottom 25% vulnerable samples ────────────────────────────
log(f"\n[4] Identifying bottom {int(VULN_FRAC*100)}% most vulnerable training samples ...")
thresh = np.percentile(mar_train, VULN_FRAC * 100)
vuln_mask_np = mar_train <= thresh
vuln_mask = torch.tensor(vuln_mask_np, dtype=torch.bool, device=Xtr.device)
n_vuln = int(vuln_mask.sum().item())
log(f"    Vulnerability threshold (margin <= {thresh:.4f}): {n_vuln} samples "
    f"({n_vuln/len(mar_train)*100:.1f}%)")

# ── Step 4: Retrain with ADAPTIVE noise ──────────────────────────────────────
log(f"\n[5] Training ADAPTIVE-NOISE model "
    f"(vulnerable: sigma={SIGMA_HIGH}, rest: sigma={SIGMA_LOW}) ...")
t0 = time.time()
model_adapt = C.build_model("cnn", META, width=WIDTH, seed=SEED)
train_with_adaptive_noise(model_adapt, Xtr, Ytr, vuln_mask,
                           SIGMA_HIGH, SIGMA_LOW, epochs=EPOCHS)
log(f"    Done in {time.time()-t0:.1f}s")

# ── Step 5: Train UNIFORM noise model (ablation) ─────────────────────────────
log(f"\n[6] Training UNIFORM-NOISE model (all samples: sigma={SIGMA_UNI}) ...")
t0 = time.time()
model_uni = C.build_model("cnn", META, width=WIDTH, seed=SEED)
train_with_noise(model_uni, Xtr, Ytr, noise_sigma=SIGMA_UNI, epochs=EPOCHS)
log(f"    Done in {time.time()-t0:.1f}s")

# ── Step 6: Evaluate all 3 models ─────────────────────────────────────────────
log(f"\n[7] Evaluating all 3 models on test set (N={N_EVAL}) ...")
results = {}
for name, model in [("baseline", model_base), ("adaptive", model_adapt),
                     ("uniform", model_uni)]:
    t0 = time.time()
    r = eval_model(model, Xte, Yte)
    r["runtime_s"] = round(time.time()-t0, 1)
    results[name] = r
    log(f"  [{name}] clean_acc={r['clean_acc']:.4f}  FGSM_ASR={r['fgsm_asr']:.4f}  "
        f"PGD_ASR={r['pgd_asr']:.4f}  mean_margin={r['mean_margin']:.4f}  "
        f"({r['runtime_s']}s)")

# ── Step 7: Margin improvement on originally-vulnerable vs non-targeted samples
log(f"\n[8] Per-group margin analysis: vulnerable vs non-vulnerable training samples ...")
log(f"    (Using train-set margin from BASELINE vs ADAPTIVE model)")

# Compute margins from adaptive model on the TRAINING set
model_adapt.eval()
mar_adapt = C.margin(model_adapt, Xtr, Ytr)

vuln_mask_cpu = vuln_mask_np.astype(bool)
non_vuln_mask_cpu = ~vuln_mask_cpu

# Baseline margins by group
base_vuln_mean    = float(mar_train[vuln_mask_cpu].mean())
base_nonvuln_mean = float(mar_train[non_vuln_mask_cpu].mean())

# Adaptive margins by group
adapt_vuln_mean    = float(mar_adapt[vuln_mask_cpu].mean())
adapt_nonvuln_mean = float(mar_adapt[non_vuln_mask_cpu].mean())

delta_vuln    = adapt_vuln_mean    - base_vuln_mean
delta_nonvuln = adapt_nonvuln_mean - base_nonvuln_mean

log(f"    Group           | Baseline margin | Adaptive margin | Delta")
log(f"    ----------------+-----------------+-----------------+--------")
log(f"    Vulnerable      | {base_vuln_mean:>15.4f} | {adapt_vuln_mean:>15.4f} | {delta_vuln:>+7.4f}")
log(f"    Non-vulnerable  | {base_nonvuln_mean:>15.4f} | {adapt_nonvuln_mean:>15.4f} | {delta_nonvuln:>+7.4f}")

targeted_lift = delta_vuln - delta_nonvuln
log(f"\n    Targeted lift (delta_vuln - delta_nonvuln): {targeted_lift:+.4f}")
log(f"    (Positive = vulnerable samples benefited MORE from adaptive noise)")

# ── Summary table ─────────────────────────────────────────────────────────────
log(f"\n[9] Summary Table")
log("-" * 80)
log(f"{'Model':>12}  {'CleanAcc':>10}  {'FGSM_ASR':>10}  {'PGD_ASR':>9}  {'MeanMargin':>12}")
log("-" * 80)
for name in ["baseline", "adaptive", "uniform"]:
    r = results[name]
    log(f"{name:>12}  {r['clean_acc']:>10.4f}  {r['fgsm_asr']:>10.4f}  "
        f"{r['pgd_asr']:>9.4f}  {r['mean_margin']:>12.4f}")
log("-" * 80)

# Delta vs baseline
log(f"\n[10] Delta vs BASELINE")
log(f"{'Model':>12}  {'dCleanAcc':>11}  {'dFGSM_ASR':>11}  {'dPGD_ASR':>10}  {'dMeanMargin':>13}")
base = results["baseline"]
for name in ["adaptive", "uniform"]:
    r = results[name]
    log(f"{name:>12}  "
        f"{r['clean_acc']  - base['clean_acc']:>+11.4f}  "
        f"{r['fgsm_asr']   - base['fgsm_asr']:>+11.4f}  "
        f"{r['pgd_asr']    - base['pgd_asr']:>+10.4f}  "
        f"{r['mean_margin']- base['mean_margin']:>+13.4f}")

# ── Interpretation ────────────────────────────────────────────────────────────
log(f"\n[11] Interpretation")
adapt_pgd_delta = results["adaptive"]["pgd_asr"] - results["baseline"]["pgd_asr"]
adapt_acc_delta = results["adaptive"]["clean_acc"] - results["baseline"]["clean_acc"]
uni_pgd_delta   = results["uniform"]["pgd_asr"] - results["baseline"]["pgd_asr"]

log(f"    Adaptive vs Baseline: PGD_ASR {adapt_pgd_delta:+.4f}, CleanAcc {adapt_acc_delta:+.4f}")
log(f"    Uniform  vs Baseline: PGD_ASR {uni_pgd_delta:+.4f}")

if adapt_pgd_delta < -0.01 and adapt_acc_delta > -0.02:
    verdict = "SUPPORTED: targeted noise hardening improves robustness with minimal clean accuracy cost"
elif adapt_pgd_delta < -0.01 and adapt_acc_delta <= -0.02:
    verdict = "PARTIAL: robustness improves but at meaningful clean accuracy cost"
elif adapt_pgd_delta < uni_pgd_delta:
    verdict = "PARTIAL: adaptive noise is more efficient than uniform noise, but robustness gain is modest"
else:
    verdict = "NOT SUPPORTED: adaptive noise does not clearly outperform baseline or uniform noise"

log(f"\n    Verdict: {verdict}")
log(f"    Targeted lift on vulnerable group: {targeted_lift:+.4f} "
    f"({'positive — targeted samples improved more' if targeted_lift > 0 else 'negative — no targeted benefit'})")

adaptive_better_than_uniform = results["adaptive"]["pgd_asr"] < results["uniform"]["pgd_asr"]
log(f"    Adaptive better than uniform noise: {adaptive_better_than_uniform}")

log(f"\nTotal time: {time.time()-t_total:.1f}s")
log("=" * 72)

with open(OUT_FILE, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    pass
