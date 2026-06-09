"""
H209 - Random-Input Distillation: does RSLAD's robustness transfer require real data?

H146 showed that a student trained on adversarial examples from a PGD-AT teacher
inherits strong robustness (PGD ASR 11% vs 96% vanilla). But is that because:
  (a) the student learns the teacher's DECISION GEOMETRY (boundary shape), or
  (b) the student just learns to mimic the teacher's OUTPUT DISTRIBUTION
     regardless of what the inputs look like?

We test (b) by training the student on PURE RANDOM NOISE inputs with the
teacher's soft predictions as targets. If robustness transfers even with
random inputs, the effect is about output-distribution matching, not geometry.
If it doesn't transfer, H146's result requires real adversarial data to work.

Protocol:
  Teacher : PGD-AT CNN (same as H146)
  Student A: trained on adversarial examples from Teacher (H146 replication)
  Student B: trained on random uniform noise [0,1] with Teacher's soft labels
  Student C: trained on random Gaussian noise clipped to [0,1]

Compare: clean acc, FGSM ASR, PGD ASR, margin AUROC for all three.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0
EPS = 0.1
PGD_STEPS = 10
STEP_SIZE = EPS / PGD_STEPS
EPOCHS = 10
BATCH = 128
N_EVAL = 500
META = {"channels": 1, "size": 28, "n_classes": 10}

torch.manual_seed(SEED)
np.random.seed(SEED)

print("H209: Random-Input Distillation")
print("=" * 60)
print("Does RSLAD robustness transfer survive random inputs?")
print()

Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
Xte_eval, Yte_eval = Xte[:N_EVAL], Yte[:N_EVAL]

# ── Teacher: PGD-AT ──────────────────────────────────────────────────────────
print("[Teacher] Training PGD-AT CNN...")
t0 = time.time()
teacher = C.build_model("cnn", META, width=32, seed=SEED)

# Manual PGD-AT loop
opt = torch.optim.SGD(teacher.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
device = next(teacher.parameters()).device
loader = torch.utils.data.DataLoader(
    list(zip(Xtr, Ytr)), batch_size=BATCH, shuffle=True)

for epoch in range(EPOCHS):
    teacher.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        # PGD-3 for speed during training
        xb_adv = xb.clone().detach()
        for _ in range(3):
            xb_adv = xb_adv.requires_grad_(True)
            loss = F.cross_entropy(teacher(xb_adv), yb)
            loss.backward()
            xb_adv = (xb_adv.detach() + (EPS/3) * xb_adv.grad.sign())
            xb_adv = torch.max(torch.min(xb_adv, xb + EPS), xb - EPS).clamp(0, 1)
        opt.zero_grad()
        loss = F.cross_entropy(teacher(xb_adv.detach()), yb)
        loss.backward()
        opt.step()
teacher.eval()
print(f"  Teacher trained in {time.time()-t0:.1f}s")

# ── Student A: adversarial-input distillation (H146 replication) ─────────────
print("[Student A] Adversarial-input distillation...")
t0 = time.time()
studentA = C.build_model("cnn", META, width=32, seed=SEED+1)
optA = torch.optim.Adam(studentA.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    studentA.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        # Generate adversarials from teacher
        xb_adv = xb.clone().detach()
        for _ in range(3):
            xb_adv = xb_adv.requires_grad_(True)
            loss = F.cross_entropy(teacher(xb_adv), yb)
            loss.backward()
            xb_adv = (xb_adv.detach() + (EPS/3) * xb_adv.grad.sign())
            xb_adv = torch.max(torch.min(xb_adv, xb + EPS), xb - EPS).clamp(0, 1)
        # Soft targets from teacher
        with torch.no_grad():
            soft_labels = F.softmax(teacher(xb_adv.detach()) / 4.0, dim=1)
        optA.zero_grad()
        log_probs = F.log_softmax(studentA(xb_adv.detach()), dim=1)
        loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
        loss.backward()
        optA.step()
studentA.eval()
print(f"  Student A trained in {time.time()-t0:.1f}s")

# ── Student B: uniform random noise distillation ──────────────────────────────
print("[Student B] Uniform random noise distillation...")
t0 = time.time()
studentB = C.build_model("cnn", META, width=32, seed=SEED+2)
optB = torch.optim.Adam(studentB.parameters(), lr=1e-3)
n_batches = len(Xtr) // BATCH

for epoch in range(EPOCHS):
    studentB.train()
    for _ in range(n_batches):
        # Random uniform noise as input
        xb_rand = torch.rand(BATCH, 1, 28, 28, device=device)
        with torch.no_grad():
            soft_labels = F.softmax(teacher(xb_rand) / 4.0, dim=1)
        optB.zero_grad()
        log_probs = F.log_softmax(studentB(xb_rand), dim=1)
        loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
        loss.backward()
        optB.step()
studentB.eval()
print(f"  Student B trained in {time.time()-t0:.1f}s")

# ── Student C: Gaussian noise distillation ────────────────────────────────────
print("[Student C] Gaussian noise distillation...")
t0 = time.time()
studentC = C.build_model("cnn", META, width=32, seed=SEED+3)
optC = torch.optim.Adam(studentC.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    studentC.train()
    for _ in range(n_batches):
        # Gaussian noise as input
        xb_gauss = torch.randn(BATCH, 1, 28, 28, device=device).clamp(0, 1)
        with torch.no_grad():
            soft_labels = F.softmax(teacher(xb_gauss) / 4.0, dim=1)
        optC.zero_grad()
        log_probs = F.log_softmax(studentC(xb_gauss), dim=1)
        loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
        loss.backward()
        optC.step()
studentC.eval()
print(f"  Student C trained in {time.time()-t0:.1f}s")

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, Xte, Yte, name):
    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        preds = logits.argmax(1)
        clean_acc = (preds == Yte).float().mean().item()

    with torch.no_grad():
        logits_m = model(Xte)
    margins = C.margin_of(logits_m.cpu(), Yte.cpu())
    margins_np = margins
    margins_np = np.array(margins) if not isinstance(margins, np.ndarray) else margins

    X_fgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    with torch.no_grad():
        fgsm_success = (model(X_fgsm).argmax(1) != Yte).cpu().float().numpy()

    X_pgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=STEP_SIZE)
    with torch.no_grad():
        pgd_success = (model(X_pgd).argmax(1) != Yte).cpu().float().numpy()

    try:
        auroc_pgd = roc_auc_score(pgd_success, -margins_np)
    except Exception:
        auroc_pgd = float('nan')

    print(f"\n  {name}:")
    print(f"    Clean acc:           {clean_acc:.4f}")
    print(f"    FGSM ASR:            {fgsm_success.mean():.4f}")
    print(f"    PGD ASR:             {pgd_success.mean():.4f}")
    print(f"    Mean margin:         {margins_np.mean():.4f}")
    print(f"    AUROC (margin→PGD):  {auroc_pgd:.4f}")
    return {
        'clean_acc': clean_acc,
        'fgsm_asr': fgsm_success.mean(),
        'pgd_asr': pgd_success.mean(),
        'mean_margin': margins_np.mean(),
        'auroc_pgd': auroc_pgd
    }

print("\n--- Results ---")
rA = evaluate(studentA, Xte_eval, Yte_eval, "Student A (adversarial inputs — H146 replication)")
rB = evaluate(studentB, Xte_eval, Yte_eval, "Student B (uniform random noise inputs)")
rC = evaluate(studentC, Xte_eval, Yte_eval, "Student C (Gaussian noise inputs)")

print("\n--- Summary Table ---")
print(f"{'Metric':<25} {'Stud-A (adv)':>14} {'Stud-B (uniform)':>16} {'Stud-C (gauss)':>14}")
print("-" * 72)
for k in ['clean_acc', 'fgsm_asr', 'pgd_asr', 'mean_margin', 'auroc_pgd']:
    print(f"  {k:<23} {rA[k]:>14.4f} {rB[k]:>16.4f} {rC[k]:>14.4f}")

print("\n--- Interpretation ---")
if rB['pgd_asr'] < 0.3 or rC['pgd_asr'] < 0.3:
    print("SURPRISING: Random-input distillation achieves meaningful robustness.")
    print("Teacher output distribution alone is sufficient — geometry not required.")
elif rA['pgd_asr'] < 0.3 and rB['pgd_asr'] > 0.7:
    print("CONFIRMED: Adversarial inputs are necessary for robustness transfer.")
    print("Random inputs produce no robustness — effect requires data geometry.")
else:
    print("MIXED: Partial robustness transfer from random inputs.")
    print(f"  Adv distil PGD ASR: {rA['pgd_asr']:.3f}")
    print(f"  Rand distil PGD ASR: {rB['pgd_asr']:.3f}")
