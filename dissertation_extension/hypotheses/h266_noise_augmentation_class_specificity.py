"""
H266 - Noise augmentation class specificity.

Does noise augmentation help some classes more than others?
Train: baseline vs noise sigma=0.2.
For each of the 10 Fashion-MNIST classes: compute per-class FGSM ASR and PGD ASR.
Measure: which classes benefit most from noise training?
Is there a correlation between class difficulty (baseline ASR) and noise benefit (ASR reduction)?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ── config ────────────────────────────────────────────────────────────────────
DS        = "fashion_mnist"
N_EVAL    = 2000       # need per-class counts — use more samples
EPS       = 0.1
PGD_STEPS = 10
SEED      = 0
META      = {"channels": 1, "size": 28, "n_classes": 10}
EPOCHS    = 10
WIDTH     = 32
SIGMA     = 0.2
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

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


def per_class_asr(model, X, Y, attack, eps, steps=10, n_classes=10):
    """Return per-class ASR array."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    with torch.no_grad():
        clean_preds = model(X).argmax(1).cpu()
    correct = (clean_preds == Y.cpu()).numpy()

    if attack == "fgsm":
        Xa = C.fgsm(model, X, Y, eps=eps)
    else:
        Xa = C.pgd(model, X, Y, eps=eps, steps=steps)

    with torch.no_grad():
        adv_preds = model(Xa).argmax(1).cpu().numpy()

    Y_np = Y.cpu().numpy()
    class_asr = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = (Y_np == c) & correct
        if mask.sum() > 0:
            flipped = (adv_preds[mask] != Y_np[mask])
            class_asr[c] = float(flipped.mean())
    return class_asr


# ── MAIN ──────────────────────────────────────────────────────────────────────
log("=" * 72)
log("H266 - Noise Augmentation Class Specificity")
log("=" * 72)
t_total = time.time()

log(f"\nLoading {DS} ...")
Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_eval=N_EVAL)
Xev, Yev = Xte, Yte
log(f"  Train: {Xtr.shape}  Eval: {Xev.shape}")

log("\n[1] Training BASELINE model ...")
t0 = time.time()
model_base = C.build_model("cnn", META, width=WIDTH, seed=SEED)
train_with_noise(model_base, Xtr, Ytr, sigma=0.0, epochs=EPOCHS)
log(f"    Done in {time.time()-t0:.1f}s")

log(f"\n[2] Training NOISE model (sigma={SIGMA}) ...")
t0 = time.time()
model_noise = C.build_model("cnn", META, width=WIDTH, seed=SEED)
train_with_noise(model_noise, Xtr, Ytr, sigma=SIGMA, epochs=EPOCHS)
log(f"    Done in {time.time()-t0:.1f}s")

log("\n[3] Computing per-class metrics ...")
results = {}
for name, model in [("baseline", model_base), ("noise", model_noise)]:
    with torch.no_grad():
        logits = model(Xev)
        clean_acc = (logits.argmax(1) == Yev).float().mean().item()
    log(f"\n  [{name}] clean_acc={clean_acc:.4f}")

    fgsm_asr_cls = per_class_asr(model, Xev, Yev, "fgsm", EPS)
    pgd_asr_cls  = per_class_asr(model, Xev, Yev, "pgd", EPS, PGD_STEPS)

    log(f"  Per-class FGSM ASR: {[f'{v:.3f}' for v in fgsm_asr_cls]}")
    log(f"  Per-class PGD  ASR: {[f'{v:.3f}' for v in pgd_asr_cls]}")

    results[name] = dict(clean_acc=clean_acc, fgsm=fgsm_asr_cls, pgd=pgd_asr_cls)

log("\n" + "=" * 72)
log("PER-CLASS RESULTS TABLE")
log("=" * 72)
hdr = f"{'Class':>12}  {'Base_FGSM':>10}  {'Noise_FGSM':>11}  {'dFGSM':>7}  {'Base_PGD':>9}  {'Noise_PGD':>10}  {'dPGD':>7}"
log(hdr)
log("-" * len(hdr))
fgsm_base = results["baseline"]["fgsm"]
fgsm_noise = results["noise"]["fgsm"]
pgd_base  = results["baseline"]["pgd"]
pgd_noise  = results["noise"]["pgd"]

for c in range(10):
    df = fgsm_noise[c] - fgsm_base[c]
    dp = pgd_noise[c] - pgd_base[c]
    log(f"{CLASS_NAMES[c]:>12}  {fgsm_base[c]:>10.4f}  {fgsm_noise[c]:>11.4f}  "
        f"{df:>+7.4f}  {pgd_base[c]:>9.4f}  {pgd_noise[c]:>10.4f}  {dp:>+7.4f}")

log("\n[4] Correlation: class difficulty vs noise benefit ...")
df_vec = fgsm_noise - fgsm_base   # negative = noise helped
dp_vec = pgd_noise  - pgd_base

rho_f, pf = spearmanr(fgsm_base, -df_vec)  # harder class → more benefit?
rho_p, pp = spearmanr(pgd_base,  -dp_vec)
log(f"  Spearman rho(baseline_FGSM_ASR, FGSM benefit) = {rho_f:.4f}  p={pf:.4e}")
log(f"  Spearman rho(baseline_PGD_ASR,  PGD  benefit) = {rho_p:.4f}  p={pp:.4e}")
log(f"  (Positive rho = harder classes benefit more from noise augmentation)")

log("\nINTERPRETATION")
log("-" * 72)
n_helped_fgsm = int((df_vec < -0.01).sum())
n_helped_pgd  = int((dp_vec < -0.01).sum())
log(f"  Classes where FGSM ASR dropped >1pp: {n_helped_fgsm}/10")
log(f"  Classes where PGD  ASR dropped >1pp: {n_helped_pgd}/10")

if rho_p > 0.4:
    verdict = "SUPPORTED: harder classes benefit more from noise augmentation"
elif rho_p > 0.1:
    verdict = "WEAK SUPPORT: slight tendency for harder classes to benefit"
else:
    verdict = "NOT SUPPORTED: no correlation between class difficulty and noise benefit"
log(f"  Verdict: {verdict}")

log(f"\nTotal time: {time.time()-t_total:.1f}s")
log("=" * 72)

out_file = os.path.join(OUT_DIR, "h266_noise_augmentation_class_specificity_output.txt")
with open(out_file, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults written to {out_file}", flush=True)

if __name__ == "__main__":
    pass
