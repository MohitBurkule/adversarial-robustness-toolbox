"""
H210 - Gradient Trajectory Replay: does learning live in the update direction?

Standard training: model A starts at W_0, SGD moves through W_1...W_T.
The net displacement is D = W_T - W_0 (a single vector in weight space).

KEY INSIGHT (from Opus review): applying deltas Δ_1...Δ_T sequentially to a
different init is EQUIVALENT to a single translation W_0' + D. Sequential
ordering is irrelevant; this is just a linear displacement test.

This script explicitly tests THREE distinct things:

  (A) NET DISPLACEMENT: does W_0' + D learn? (=sequential replay)
      This tests: "can the final model's net change, applied from any start,
      produce a classifier?" i.e. does the delta carry function information?

  (B) DIRECTION ONLY: does W_0' + ||D|| * (D/||D||) learn?
      Same displacement magnitude but unit-normalized direction.
      Tests if magnitude or direction matters more.

  (C) RANDOM DIRECTION, SAME MAGNITUDE: W_0'' + ||D|| * random_unit_vector
      Control: random displacement of the same L2 size.
      If (A) is meaningful, it should beat (C).

  (D) PERMUTATION-ALIGNED REPLAY (key fix from literature):
      Apply D to W_0' but first permute neurons to align W_0' to W_0's layout
      (approximate, via greedy column matching on first layer weights).
      arXiv:2305.14122 shows alignment is needed for trajectory transfer.

The main question: does D carry learning, or is it just initialisation?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0
EPS = 0.1
EPOCHS = 8
BATCH = 256
N_EVAL = 500
META = {"channels": 1, "size": 28, "n_classes": 10}

torch.manual_seed(SEED)
np.random.seed(SEED)

print("H210: Gradient Trajectory Replay (corrected design)")
print("=" * 60)
print("Testing: does the net weight displacement carry learning?")
print()

Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
Xte_eval, Yte_eval = Xte[:N_EVAL], Yte[:N_EVAL]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = torch.utils.data.DataLoader(
    list(zip(Xtr, Ytr)), batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED))

# ── Model A: normal training ──────────────────────────────────────────────────
print("[Model A] Normal training (seed=0)...")
t0 = time.time()
modelA = C.build_model("cnn", META, width=32, seed=SEED)
W0_A = [p.data.clone() for p in modelA.parameters()]  # save initial weights

opt = torch.optim.SGD(modelA.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
for epoch in range(EPOCHS):
    modelA.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        F.cross_entropy(modelA(xb), yb).backward()
        opt.step()
modelA.eval()

# Net displacement D = W_T - W_0
W_T_A = [p.data.clone() for p in modelA.parameters()]
D = [wt - w0 for wt, w0 in zip(W_T_A, W0_A)]
D_norm = sum((d**2).sum().item() for d in D) ** 0.5
print(f"  Done in {time.time()-t0:.1f}s  ||D|| = {D_norm:.4f}")

# ── Model B: different init + net displacement (=sequential replay) ───────────
print("[Model B] Random init (seed=42) + net displacement D...")
modelB = C.build_model("cnn", META, width=32, seed=42)
with torch.no_grad():
    for p, d in zip(modelB.parameters(), D):
        p.data.add_(d)
modelB.eval()

# ── Model C: different init + direction-only displacement ─────────────────────
print("[Model C] Random init (seed=42) + unit direction of D, same ||D||...")
modelC = C.build_model("cnn", META, width=32, seed=42)
with torch.no_grad():
    for p, d in zip(modelC.parameters(), D):
        p.data.add_(d * (D_norm / (D_norm + 1e-10)))  # unit direction already, since D is one vector
        # actually: scale unit_d by D_norm → same as D itself
        # instead normalise per-param then rescale:
        # unit_d = d / (||D|| + eps) * ||D|| = d — same thing
        # Correct: apply D/||D|| * ||D|| per-tensor, which is just D
        # So C = B in this formulation. Make it per-tensor normalised:
        pass
# Redo C with per-param unit vectors scaled by mean param norm
modelC = C.build_model("cnn", META, width=32, seed=42)
with torch.no_grad():
    per_param_norms = [d.norm().item() + 1e-10 for d in D]
    mean_pnorm = np.mean(per_param_norms)
    for p, d, pnorm in zip(modelC.parameters(), D, per_param_norms):
        unit_d = d / pnorm  # unit direction per parameter tensor
        p.data.add_(unit_d * mean_pnorm)  # apply scaled unit direction
modelC.eval()

# ── Model D: random init + random displacement of same magnitude ──────────────
print("[Model D] Random init (seed=42) + random displacement ||D||...")
torch.manual_seed(999)
modelD = C.build_model("cnn", META, width=32, seed=42)
with torch.no_grad():
    rand_deltas = [torch.randn_like(d) for d in D]
    rand_norm = sum((r**2).sum().item() for r in rand_deltas) ** 0.5
    scale = D_norm / (rand_norm + 1e-10)
    for p, r in zip(modelD.parameters(), rand_deltas):
        p.data.add_(r * scale)
modelD.eval()

# ── Model E: permutation-aligned displacement ─────────────────────────────────
print("[Model E] Aligned displacement (greedy first-layer permutation matching)...")
modelE = C.build_model("cnn", META, width=32, seed=42)
W0_B = [p.data.clone() for p in modelE.parameters()]  # init of B/C/D/E

# Greedy neuron permutation on first conv layer
# W_A[0]: (out_channels, in_channels, kH, kW) = (32, 1, 3, 3)
# W_B[0]: same shape
# Match each output neuron of B to the closest output neuron of A
pA_flat = W0_A[0].view(W0_A[0].size(0), -1)  # (32, 9)
pB_flat = W0_B[0].view(W0_B[0].size(0), -1)
# Greedy matching: for each A neuron, find closest unmatched B neuron
available = list(range(pB_flat.size(0)))
perm = []
for i in range(pA_flat.size(0)):
    dists = torch.cdist(pA_flat[i:i+1], pB_flat[available]).squeeze(0)
    best = available[dists.argmin().item()]
    perm.append(best)
    available.remove(best)
perm = torch.tensor(perm, dtype=torch.long)

# Apply permutation to model E's weights (first layer output channels only)
with torch.no_grad():
    for p, d in zip(modelE.parameters(), D):
        p.data.add_(d)
    # Re-order first layer by permutation
    modelE_params = list(modelE.parameters())
    modelE_params[0].data = modelE_params[0].data[perm]  # reorder output channels
    # Note: full alignment would require matching all layers; first-layer only is approximate
modelE.eval()

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, Xte, Yte, name):
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1)
        clean_acc = (preds == Yte).float().mean().item()
    margins_np = C.margin(model, Xte)
    X_pgd = C.pgd(model, Xte, Yte, eps=EPS, steps=10, alpha=0.01)
    with torch.no_grad():
        pgd_success = (model(X_pgd).argmax(1) != Yte).cpu().float().numpy()
    pgd_asr = pgd_success.mean()
    try:
        auroc = roc_auc_score(pgd_success, -margins_np)
    except Exception:
        auroc = float('nan')
    print(f"  {name}: clean={clean_acc:.4f}  PGD_ASR={pgd_asr:.4f}  AUROC={auroc:.4f}")
    return dict(clean_acc=clean_acc, pgd_asr=pgd_asr, auroc=auroc)

print("\n--- Results ---")
rA = evaluate(modelA, Xte_eval, Yte_eval, "A (normal training)")
rB = evaluate(modelB, Xte_eval, Yte_eval, "B (different init + net D — same as seq replay)")
rC = evaluate(modelC, Xte_eval, Yte_eval, "C (different init + per-param unit direction)")
rD = evaluate(modelD, Xte_eval, Yte_eval, "D (different init + random displacement ||D||)")
rE = evaluate(modelE, Xte_eval, Yte_eval, "E (approx permutation-aligned D)")

print("\n--- Summary ---")
print(f"{'Model':<45} {'Clean Acc':>10} {'PGD ASR':>10}")
print("-" * 67)
for name, r in [("A normal training", rA), ("B net displacement (random init)", rB),
                ("C unit direction (random init)", rC), ("D random displacement", rD),
                ("E permutation-aligned", rE)]:
    print(f"  {name:<43} {r['clean_acc']:>10.4f} {r['pgd_asr']:>10.4f}")

print("\n--- Interpretation ---")
print(f"Net displacement vs random displacement: {rB['clean_acc']:.3f} vs {rD['clean_acc']:.3f}")
if rB['clean_acc'] > rD['clean_acc'] + 0.05:
    print("CONFIRMED: Net displacement D carries learning signal beyond random displacement.")
    print("The trained gradient direction is meaningful independent of starting point.")
elif rB['clean_acc'] > 0.7:
    print("PARTIAL: B learns but may not significantly beat random displacement.")
    print(f"Both B ({rB['clean_acc']:.3f}) and D ({rD['clean_acc']:.3f}) may learn from sheer displacement size.")
else:
    print("NOT CONFIRMED: Net displacement from random init fails to produce a classifier.")
    print("Weight initialisation is entangled with the gradient trajectory — cannot be separated.")

if rE['clean_acc'] > rB['clean_acc'] + 0.03:
    print(f"\nALIGNMENT MATTERS: Permutation-aligned model E ({rE['clean_acc']:.3f}) > unaligned B ({rB['clean_acc']:.3f})")
    print("Consistent with arXiv:2305.14122: neuron alignment is required for trajectory transfer.")
