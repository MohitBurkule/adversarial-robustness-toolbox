"""
H262 - Gaussian noise sigma sweep during training.

Sweep Gaussian noise sigma in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5].
For each sigma: train CNN with Gaussian noise augmentation.
Measure: clean_acc, FGSM_ASR, PGD_ASR, mean_margin.
Also compute certified_radius proxy = sigma * Phi^-1(clean_acc) using scipy.

Key question: at what sigma does PGD ASR start dropping? Is there a threshold?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm as scipy_norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ── config ────────────────────────────────────────────────────────────────────
DS       = "fashion_mnist"
N_EVAL   = 500
EPS      = 0.1
PGD_STEPS = 10
SEED     = 0
META     = {"channels": 1, "size": 28, "n_classes": 10}
EPOCHS   = 10
WIDTH    = 32
SIGMAS   = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def train_with_noise(model, X, Y, noise_sigma, epochs=10, lr=0.05, batch=128):
    """Train with Gaussian noise of given sigma added to inputs."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=X.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = X[idx], Y[idx]
            if noise_sigma > 0.0:
                noise = torch.randn_like(xb) * noise_sigma
                xb = (xb + noise).clamp(0.0, 1.0)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_model(model, Xte, Yte, n_eval):
    Xev, Yev = Xte[:n_eval], Yte[:n_eval]
    model.eval()
    with torch.no_grad():
        logits = model(Xev)
        clean_acc = (logits.argmax(1) == Yev).float().mean().item()

    for p in model.parameters():
        p.requires_grad_(True)

    Xf = C.fgsm(model, Xev, Yev, eps=EPS)
    with torch.no_grad():
        fgsm_asr = (model(Xf).argmax(1) != Yev).float().mean().item()

    Xp = C.pgd(model, Xev, Yev, eps=EPS, steps=PGD_STEPS)
    with torch.no_grad():
        pgd_asr = (model(Xp).argmax(1) != Yev).float().mean().item()

    mar = C.margin(model, Xev, Yte[:n_eval])
    mean_margin = float(mar.mean())

    return clean_acc, fgsm_asr, pgd_asr, mean_margin


# ── MAIN ──────────────────────────────────────────────────────────────────────
log("=" * 72)
log("H262 - Gaussian Noise Sigma Sweep During Training")
log("=" * 72)
t_total = time.time()

log(f"\nLoading {DS} ...")
Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
log(f"  Train: {Xtr.shape}  Test: {Xte.shape}")

results = []

for sigma in SIGMAS:
    log(f"\n--- sigma={sigma:.2f} ---")
    t0 = time.time()
    model = C.build_model("cnn", META, width=WIDTH, seed=SEED)
    train_with_noise(model, Xtr, Ytr, noise_sigma=sigma, epochs=EPOCHS)
    log(f"  Training done in {time.time()-t0:.1f}s")

    clean_acc, fgsm_asr, pgd_asr, mean_margin = eval_model(model, Xte, Yte, N_EVAL)

    # certified radius proxy: sigma * Phi^-1(clean_acc), clamped to valid domain
    p = float(np.clip(clean_acc, 1e-6, 1.0 - 1e-6))
    cert_radius = float(sigma * scipy_norm.ppf(p)) if sigma > 0.0 else 0.0

    results.append({
        "sigma": sigma,
        "clean_acc": clean_acc,
        "fgsm_asr": fgsm_asr,
        "pgd_asr": pgd_asr,
        "mean_margin": mean_margin,
        "cert_radius": cert_radius,
    })
    log(f"  clean_acc={clean_acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  "
        f"PGD_ASR={pgd_asr:.4f}  mean_margin={mean_margin:.4f}  cert_radius={cert_radius:.4f}")

log("\n" + "=" * 72)
log("SIGMA SWEEP TABLE")
log("=" * 72)
hdr = f"{'sigma':>6}  {'clean_acc':>10}  {'FGSM_ASR':>10}  {'PGD_ASR':>9}  {'mean_margin':>12}  {'cert_radius':>12}"
log(hdr)
log("-" * len(hdr))
for r in results:
    log(f"{r['sigma']:>6.2f}  {r['clean_acc']:>10.4f}  {r['fgsm_asr']:>10.4f}  "
        f"{r['pgd_asr']:>9.4f}  {r['mean_margin']:>12.4f}  {r['cert_radius']:>12.4f}")

# Find threshold where PGD ASR starts dropping
log("\nINTERPRETATION")
log("-" * 72)
baseline_pgd = results[0]["pgd_asr"]
threshold_sigma = None
for r in results[1:]:
    if r["pgd_asr"] < baseline_pgd - 0.01:
        threshold_sigma = r["sigma"]
        break

if threshold_sigma is not None:
    log(f"PGD ASR starts dropping (>1pp below baseline={baseline_pgd:.4f}) "
        f"at sigma={threshold_sigma:.2f}")
else:
    log(f"PGD ASR does not clearly drop below baseline={baseline_pgd:.4f} in the sweep.")

# Best robustness point
best = min(results, key=lambda r: r["pgd_asr"])
log(f"Best PGD robustness: sigma={best['sigma']:.2f}  "
    f"PGD_ASR={best['pgd_asr']:.4f}  clean_acc={best['clean_acc']:.4f}")

# Clean acc at which noise becomes damaging (>2pp drop)
acc_base = results[0]["clean_acc"]
for r in results:
    if r["clean_acc"] < acc_base - 0.02:
        log(f"Clean accuracy degrades >2pp below baseline at sigma={r['sigma']:.2f} "
            f"(acc={r['clean_acc']:.4f})")
        break

log(f"\nNote: certified_radius proxy = sigma * Phi^-1(clean_acc) "
    f"(randomised-smoothing lower bound if classifier certifiably correct with prob=clean_acc)")
log(f"\nTotal time: {time.time()-t_total:.1f}s")
log("=" * 72)

out_file = os.path.join(OUT_DIR, "h262_noise_sigma_sweep_output.txt")
with open(out_file, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults written to {out_file}", flush=True)

if __name__ == "__main__":
    pass
