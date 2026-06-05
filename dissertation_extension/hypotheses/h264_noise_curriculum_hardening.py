"""
H264 - Curriculum noise hardening.

Curriculum noise: start training with high noise sigma=0.3, anneal to sigma=0.0
over epochs (sigma = 0.3 * (1 - epoch/total_epochs)).

Compare to:
  - constant sigma=0.15
  - constant sigma=0.0 (baseline)
  - reverse curriculum (start low, end high)

Measure: clean_acc, FGSM_ASR, PGD_ASR, mean_margin.
Hypothesis: annealing noise helps the model first learn a robust coarse structure,
then refine.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ── config ────────────────────────────────────────────────────────────────────
DS        = "fashion_mnist"
N_EVAL    = 500
EPS       = 0.1
PGD_STEPS = 10
SEED      = 0
META      = {"channels": 1, "size": 28, "n_classes": 10}
EPOCHS    = 10
WIDTH     = 32
SIGMA_MAX = 0.3
SIGMA_MID = 0.15

OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def train_curriculum(model, X, Y, sigma_schedule, epochs=10, lr=0.05, batch=128):
    """sigma_schedule: list of len=epochs giving sigma for each epoch."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.size(0)
    for ep in range(epochs):
        sigma = sigma_schedule[ep]
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
    mar = C.margin(model, Xev, Yev)
    return clean_acc, fgsm_asr, pgd_asr, float(mar.mean())


# ── MAIN ──────────────────────────────────────────────────────────────────────
log("=" * 72)
log("H264 - Curriculum Noise Hardening")
log("=" * 72)
t_total = time.time()

log(f"\nLoading {DS} ...")
Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
log(f"  Train: {Xtr.shape}  Test: {Xte.shape}")

# Build schedules
schedules = {
    "baseline":          [0.0] * EPOCHS,
    "constant_0.15":     [SIGMA_MID] * EPOCHS,
    "anneal_hi_to_lo":   [SIGMA_MAX * (1 - ep / EPOCHS) for ep in range(EPOCHS)],
    "reverse_lo_to_hi":  [SIGMA_MAX * (ep / EPOCHS) for ep in range(EPOCHS)],
}

results = {}
for name, sched in schedules.items():
    log(f"\n--- {name} ---")
    log(f"  schedule: {[f'{s:.3f}' for s in sched]}")
    t0 = time.time()
    model = C.build_model("cnn", META, width=WIDTH, seed=SEED)
    train_curriculum(model, Xtr, Ytr, sched, epochs=EPOCHS)
    clean_acc, fgsm_asr, pgd_asr, mean_margin = eval_model(model, Xte, Yte, N_EVAL)
    log(f"  clean_acc={clean_acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  "
        f"PGD_ASR={pgd_asr:.4f}  mean_margin={mean_margin:.4f}  "
        f"({time.time()-t0:.1f}s)")
    results[name] = dict(clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                         pgd_asr=pgd_asr, mean_margin=mean_margin)

log("\n" + "=" * 72)
log("SUMMARY TABLE")
log("=" * 72)
hdr = f"{'Model':>20}  {'clean_acc':>10}  {'FGSM_ASR':>10}  {'PGD_ASR':>9}  {'mean_margin':>12}"
log(hdr)
log("-" * len(hdr))
for name in schedules:
    r = results[name]
    log(f"{name:>20}  {r['clean_acc']:>10.4f}  {r['fgsm_asr']:>10.4f}  "
        f"{r['pgd_asr']:>9.4f}  {r['mean_margin']:>12.4f}")

log("\nDELTA vs BASELINE")
base = results["baseline"]
log(f"{'Model':>20}  {'dCleanAcc':>11}  {'dFGSM_ASR':>11}  {'dPGD_ASR':>10}  {'dMargin':>10}")
for name in [k for k in schedules if k != "baseline"]:
    r = results[name]
    log(f"{name:>20}  "
        f"{r['clean_acc']-base['clean_acc']:>+11.4f}  "
        f"{r['fgsm_asr']-base['fgsm_asr']:>+11.4f}  "
        f"{r['pgd_asr']-base['pgd_asr']:>+10.4f}  "
        f"{r['mean_margin']-base['mean_margin']:>+10.4f}")

log("\nINTERPRETATION")
log("-" * 72)
anneal = results["anneal_hi_to_lo"]
reverse = results["reverse_lo_to_hi"]
const = results["constant_0.15"]

if anneal["pgd_asr"] < const["pgd_asr"] and anneal["pgd_asr"] < reverse["pgd_asr"]:
    verdict = "SUPPORTED: annealing (hi-to-lo) gives best PGD robustness — coarse-to-fine learning"
elif reverse["pgd_asr"] < anneal["pgd_asr"]:
    verdict = "REFUTED: reverse curriculum (lo-to-hi) is better — fine-to-coarse learning"
elif anneal["pgd_asr"] < base["pgd_asr"] - 0.01:
    verdict = "PARTIAL: annealing helps vs baseline but not clearly better than constant noise"
else:
    verdict = "NOT SUPPORTED: no clear benefit of curriculum noise scheduling"
log(f"  Verdict: {verdict}")

log(f"\nTotal time: {time.time()-t_total:.1f}s")
log("=" * 72)

out_file = os.path.join(OUT_DIR, "h264_noise_curriculum_hardening_output.txt")
with open(out_file, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults written to {out_file}", flush=True)

if __name__ == "__main__":
    pass
