"""
H395 — Inference-time randomized smoothing on a standard model.

Hypothesis: adding Gaussian noise + averaging logits at INFERENCE ONLY (no retraining)
buys robustness, and an EOT-aware attack that differentiates through the smoothing
erodes the naive gain.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import campaign.common as C
import torch
import torch.nn.functional as F
import numpy as np

SEED = 0
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
EPS = 0.1
PGD_STEPS = 10
K_SMOOTH = 20   # noise draws for smoothed classifier evaluation
K_EOT = 10      # noise draws per gradient step in EOT-PGD
SIGMA_SWEEP = [0.0, 0.1, 0.25, 0.5]

C.set_seed(SEED)
DEVICE = C.DEVICE

# ---- data & model -----------------------------------------------------------
meta = {"channels": 1, "size": 28, "n_classes": 10}
Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)

model = C.build_model("cnn", meta, width=32, seed=SEED)
opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                      lr=LR, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

print("Training standard model...")
model.train()
n = Xtr.size(0)
for ep in range(EPOCHS):
    perm = torch.randperm(n, device=DEVICE)
    for i in range(0, n, BATCH):
        idx = perm[i:i+BATCH]
        xb, yb = Xtr[idx], Ytr[idx]
        opt.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        opt.step()
    sched.step()
model.eval()

_, clean_acc = C.logits_and_acc(model, Xte, Yte)
print(f"Clean accuracy: {clean_acc:.4f}")


# ---- smoothed classifier ----------------------------------------------------
@torch.no_grad()
def smoothed_predict(x, sigma, K=K_SMOOTH):
    """Return predicted class via averaging logits over K noise draws."""
    if sigma == 0.0:
        return model(x).argmax(1)
    # x: (B, C, H, W)
    B = x.size(0)
    logit_sum = torch.zeros(B, 10, device=DEVICE)
    for _ in range(K):
        noise = torch.randn_like(x) * sigma
        logit_sum += model((x + noise).clamp(0, 1))
    return logit_sum.argmax(1)


@torch.no_grad()
def smoothed_acc(x, y, sigma):
    preds = smoothed_predict(x, sigma)
    return float((preds == y).float().mean())


# ---- NAIVE PGD: craft on underlying model, evaluate on smoothed -------------
def naive_pgd_asr(x, y, sigma):
    """Craft adversarial on base model; evaluate success on smoothed classifier."""
    xa = C.pgd(model, x, y, eps=EPS, steps=PGD_STEPS)
    with torch.no_grad():
        smooth_preds = smoothed_predict(xa, sigma)
    return float((smooth_preds != y).float().mean())


# ---- EOT-PGD: gradient averaged over K fresh noise samples ------------------
def eot_pgd(x, y, sigma, eps=EPS, steps=PGD_STEPS, K=K_EOT):
    """PGD where each step averages gradient over K noise samples through smoothing."""
    alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        # average loss over K noise draws
        total_loss = torch.zeros(1, device=DEVICE)
        for _ in range(K):
            noise = torch.randn_like(xa) * sigma
            logits = model((xa + noise).clamp(0, 1))
            total_loss = total_loss + F.cross_entropy(logits, y)
        (total_loss / K).backward()
        with torch.no_grad():
            xa = xa + alpha * xa.grad.sign()
            xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def eot_pgd_asr(x, y, sigma):
    if sigma == 0.0:
        # degenerate: just plain PGD
        xa = C.pgd(model, x, y, eps=EPS, steps=PGD_STEPS)
    else:
        xa = eot_pgd(x, y, sigma)
    with torch.no_grad():
        smooth_preds = smoothed_predict(xa, sigma)
    return float((smooth_preds != y).float().mean())


# ---- sweep ------------------------------------------------------------------
print("\n" + "="*65)
print(f"{'sigma':>8} | {'clean_smoothed':>14} | {'naive_pgd_asr':>13} | {'eot_pgd_asr':>11}")
print("-"*65)

results = []
# evaluate on a batch for speed
X_eval, Y_eval = Xte[:500], Yte[:500]

for sigma in SIGMA_SWEEP:
    print(f"  sigma={sigma} ...", flush=True)
    c_acc = smoothed_acc(X_eval, Y_eval, sigma)
    n_asr = naive_pgd_asr(X_eval, Y_eval, sigma)
    e_asr = eot_pgd_asr(X_eval, Y_eval, sigma)
    results.append((sigma, c_acc, n_asr, e_asr))
    print(f"{sigma:>8.2f} | {c_acc:>14.4f} | {n_asr:>13.4f} | {e_asr:>11.4f}")

print("="*65)

# ---- verdict ----------------------------------------------------------------
best = max(results, key=lambda r: r[1] - r[3])   # best clean - eot_asr tradeoff
naive_mean = np.mean([r[2] for r in results])
eot_mean   = np.mean([r[3] for r in results])
gap = naive_mean - eot_mean

print(f"\nMean naive_pgd_asr={naive_mean:.4f}, mean eot_pgd_asr={eot_mean:.4f}, gap={gap:.4f}")
print(f"Best sigma for clean-robust tradeoff: {best[0]} "
      f"(clean_acc={best[1]:.4f}, eot_asr={best[3]:.4f})")

if gap > 0.1:
    verdict = ("VERDICT: Inference-time smoothing provides only APPARENT robustness — "
               f"naive ASR is {naive_mean:.3f} vs EOT-PGD ASR {eot_mean:.3f} "
               f"(gap={gap:.3f}). EOT attack largely erodes the naive gain.")
elif eot_mean < 0.3:
    verdict = (f"VERDICT: Inference-time smoothing provides REAL partial robustness — "
               f"EOT-PGD ASR ({eot_mean:.3f}) remains well below naive ({naive_mean:.3f}). "
               f"Best sigma={best[0]}.")
else:
    verdict = (f"VERDICT: Mixed — naive ASR={naive_mean:.3f}, EOT ASR={eot_mean:.3f}; "
               f"smoothing reduces naive attack but EOT partially recovers. "
               f"Best sigma={best[0]}.")

print(verdict)
