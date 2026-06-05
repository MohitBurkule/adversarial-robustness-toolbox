"""
H267 - Input noise vs weight noise vs dropout.

Compare three noise types:
  1. Input noise during training: add N(0,sigma^2) to inputs
  2. Weight noise during training: add N(0,sigma^2) to all weights each forward pass
  3. Dropout (p=0.3) — classic weight stochasticity

All at matched "noise level" (sigma chosen to produce similar clean accuracy degradation).
Measure: FGSM_ASR, PGD_ASR, mean_margin. Which noise type most efficiently moves
the decision boundary?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

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
INPUT_SIGMA  = 0.15
WEIGHT_SIGMA = 0.02    # weight sigma needs to be much smaller
DROPOUT_P    = 0.3

OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def train_input_noise(model, X, Y, sigma, epochs=10, lr=0.05, batch=128):
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


def train_weight_noise(model, X, Y, sigma, epochs=10, lr=0.05, batch=128):
    """Add Gaussian noise to all weights at each forward pass."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=X.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = X[idx], Y[idx]
            opt.zero_grad()
            # Save original params, add noise, forward, restore
            orig_params = {}
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        orig_params[name] = param.data.clone()
                        param.data.add_(torch.randn_like(param) * sigma)
            loss = F.cross_entropy(model(xb), yb)
            # Restore before backward
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in orig_params:
                        param.data.copy_(orig_params[name])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def build_dropout_model(meta, width, seed):
    """Build CNN with dropout after each activation."""
    ch, sz, ncls = meta["channels"], meta["size"], meta["n_classes"]

    class DropoutCNN(nn.Module):
        def __init__(self, p=0.3):
            super().__init__()
            def block(i, o):
                return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                        nn.ReLU(), nn.Dropout2d(p), nn.MaxPool2d(2)]
            self.features = nn.Sequential(
                *block(ch, width), *block(width, width*2), *block(width*2, width*4))
            feat = sz // 8
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(width*4*feat*feat, 256), nn.ReLU(), nn.Dropout(p),
                nn.Linear(256, ncls))

        def forward(self, x):
            return self.head(self.features(x))

    torch.manual_seed(seed)
    return DropoutCNN(p=DROPOUT_P).to(C.DEVICE)


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
log("H267 - Input Noise vs Weight Noise vs Dropout")
log("=" * 72)
t_total = time.time()

log(f"\nLoading {DS} ...")
Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
log(f"  Train: {Xtr.shape}  Test: {Xte.shape}")

results = {}

log("\n[1] Training BASELINE (no noise) ...")
t0 = time.time()
m = C.build_model("cnn", META, width=WIDTH, seed=SEED)
train_input_noise(m, Xtr, Ytr, sigma=0.0, epochs=EPOCHS)
results["baseline"] = eval_model(m, Xte, Yte, N_EVAL)
log(f"    Done in {time.time()-t0:.1f}s  "
    f"clean_acc={results['baseline'][0]:.4f}  PGD_ASR={results['baseline'][2]:.4f}")

log(f"\n[2] Training INPUT NOISE (sigma={INPUT_SIGMA}) ...")
t0 = time.time()
m = C.build_model("cnn", META, width=WIDTH, seed=SEED)
train_input_noise(m, Xtr, Ytr, sigma=INPUT_SIGMA, epochs=EPOCHS)
results["input_noise"] = eval_model(m, Xte, Yte, N_EVAL)
log(f"    Done in {time.time()-t0:.1f}s  "
    f"clean_acc={results['input_noise'][0]:.4f}  PGD_ASR={results['input_noise'][2]:.4f}")

log(f"\n[3] Training WEIGHT NOISE (sigma={WEIGHT_SIGMA}) ...")
t0 = time.time()
m = C.build_model("cnn", META, width=WIDTH, seed=SEED)
train_weight_noise(m, Xtr, Ytr, sigma=WEIGHT_SIGMA, epochs=EPOCHS)
results["weight_noise"] = eval_model(m, Xte, Yte, N_EVAL)
log(f"    Done in {time.time()-t0:.1f}s  "
    f"clean_acc={results['weight_noise'][0]:.4f}  PGD_ASR={results['weight_noise'][2]:.4f}")

log(f"\n[4] Training DROPOUT (p={DROPOUT_P}) ...")
t0 = time.time()
m = build_dropout_model(META, WIDTH, SEED)
train_input_noise(m, Xtr, Ytr, sigma=0.0, epochs=EPOCHS)  # no input noise; dropout handles stochasticity
results["dropout"] = eval_model(m, Xte, Yte, N_EVAL)
log(f"    Done in {time.time()-t0:.1f}s  "
    f"clean_acc={results['dropout'][0]:.4f}  PGD_ASR={results['dropout'][2]:.4f}")

log("\n" + "=" * 72)
log("SUMMARY TABLE")
log("=" * 72)
hdr = f"{'Model':>14}  {'clean_acc':>10}  {'FGSM_ASR':>10}  {'PGD_ASR':>9}  {'mean_margin':>12}"
log(hdr)
log("-" * len(hdr))
for name in ["baseline", "input_noise", "weight_noise", "dropout"]:
    clean_acc, fgsm_asr, pgd_asr, mean_margin = results[name]
    log(f"{name:>14}  {clean_acc:>10.4f}  {fgsm_asr:>10.4f}  "
        f"{pgd_asr:>9.4f}  {mean_margin:>12.4f}")

log("\nDELTA vs BASELINE")
base = results["baseline"]
log(f"{'Model':>14}  {'dCleanAcc':>11}  {'dFGSM_ASR':>11}  {'dPGD_ASR':>10}  {'dMargin':>10}")
for name in ["input_noise", "weight_noise", "dropout"]:
    r = results[name]
    log(f"{name:>14}  "
        f"{r[0]-base[0]:>+11.4f}  "
        f"{r[1]-base[1]:>+11.4f}  "
        f"{r[2]-base[2]:>+10.4f}  "
        f"{r[3]-base[3]:>+10.4f}")

log("\nINTERPRETATION")
log("-" * 72)
pgd_asrs = {n: results[n][2] for n in ["input_noise", "weight_noise", "dropout"]}
best_name = min(pgd_asrs, key=pgd_asrs.get)
log(f"  Most PGD-robust noise type: {best_name} (PGD_ASR={pgd_asrs[best_name]:.4f})")
log(f"  Parameter notes: input_sigma={INPUT_SIGMA}, weight_sigma={WEIGHT_SIGMA}, dropout_p={DROPOUT_P}")
log(f"  Note: sigma values not calibrated to identical accuracy — interpret delta clean_acc")

log(f"\nTotal time: {time.time()-t_total:.1f}s")
log("=" * 72)

out_file = os.path.join(OUT_DIR, "h267_input_noise_vs_weight_noise_output.txt")
with open(out_file, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults written to {out_file}", flush=True)

if __name__ == "__main__":
    pass
