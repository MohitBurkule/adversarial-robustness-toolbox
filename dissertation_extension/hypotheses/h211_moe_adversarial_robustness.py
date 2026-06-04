"""
H211 - Mixture of Experts: how does adding/removing experts change adversarial robustness?

A Mixture of Experts (MoE) model has:
  - A router that, for each input, assigns weights to K expert sub-networks
  - K expert CNNs that each produce a prediction
  - Final output = weighted sum of expert outputs

Questions:
  1. Does more experts = more robust? (diversity hypothesis)
  2. Does removing the most-used expert hurt robustness more than removing a
     rarely-used expert? (importance hypothesis)
  3. Do different experts specialise on different vulnerability regions?
     (do samples that fool expert i also fool expert j?)
  4. Is an MoE's margin AUROC better or worse than a single model?

We train MoE with K=1,2,4,8 experts and compare robustness at each scale.
Then for K=4, we ablate each expert and measure robustness degradation.
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
EPOCHS = 10
BATCH = 128
N_EVAL = 500
META = {"channels": 1, "size": 28, "n_classes": 10}

torch.manual_seed(SEED)
np.random.seed(SEED)

print("H211: Mixture of Experts — Adversarial Robustness")
print("=" * 60)

Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
Xte_eval, Yte_eval = Xte[:N_EVAL], Yte[:N_EVAL]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── MoE architecture ──────────────────────────────────────────────────────────
class ExpertCNN(nn.Module):
    """Small CNN expert (lightweight for MoE scaling)."""
    def __init__(self, n_classes=10, width=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width*2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width*2 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def features_out(self, x):
        return self.features(x).flatten(1)


class MoEModel(nn.Module):
    """Soft MoE: router assigns continuous weights to K experts."""
    def __init__(self, n_experts=4, n_classes=10, width=16):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            ExpertCNN(n_classes, width) for _ in range(n_experts)
        ])
        # Router: simple linear on flattened input
        feat_dim = width * 2 * 7 * 7
        # Use first expert's feature extractor to get routing features
        self.router = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width*2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(feat_dim, n_experts)
        )
        self.disabled_experts = set()

    def forward(self, x):
        # Router weights
        gate_logits = self.router(x)  # (B, K)
        # Zero out disabled experts
        if self.disabled_experts:
            mask = torch.ones(self.n_experts, device=x.device)
            for i in self.disabled_experts:
                mask[i] = 0.0
            gate_logits = gate_logits * mask.unsqueeze(0)
        gates = F.softmax(gate_logits, dim=1)  # (B, K)

        # Expert outputs: (B, K, n_classes)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)

        # Weighted sum: (B, n_classes)
        out = (gates.unsqueeze(-1) * expert_outs).sum(dim=1)
        return out

    def router_weights(self, x):
        """Return routing weights for analysis. (B, K)"""
        with torch.no_grad():
            gate_logits = self.router(x)
            return F.softmax(gate_logits, dim=1)


def train_moe(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        list(zip(Xtr, Ytr)), batch_size=batch, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
    model.eval()
    return model


def evaluate(model, Xte, Yte, name="model"):
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1)
        clean_acc = (preds == Yte).float().mean().item()

    margins = C.margin(model, Xte)
    margins_np = np.array(margins) if not isinstance(margins, np.ndarray) else margins

    X_fgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    with torch.no_grad():
        fgsm_asr = (model(X_fgsm).argmax(1) != Yte).float().mean().item()

    X_pgd = C.pgd(model, Xte, Yte, eps=EPS, steps=10, alpha=0.01)
    with torch.no_grad():
        pgd_success = (model(X_pgd).argmax(1) != Yte).cpu().float().numpy()
    pgd_asr = pgd_success.mean()

    try:
        auroc = roc_auc_score(pgd_success, -margins_np)
    except Exception:
        auroc = float('nan')

    return dict(name=name, clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                pgd_asr=pgd_asr, auroc=auroc)


# ── 1. Scale study: K=1,2,4,8 experts ────────────────────────────────────────
print("\n[Part 1] Scaling: K = 1, 2, 4, 8 experts")
scale_results = []
for k in [1, 2, 4, 8]:
    print(f"  Training K={k}...", end=" ", flush=True)
    t0 = time.time()
    torch.manual_seed(SEED)
    model = MoEModel(n_experts=k, n_classes=10, width=16).to(device)
    train_moe(model, Xtr, Ytr)
    r = evaluate(model, Xte_eval, Yte_eval, f"MoE K={k}")
    r['n_experts'] = k
    r['train_time'] = time.time() - t0
    scale_results.append(r)
    print(f"  clean={r['clean_acc']:.3f}  PGD_ASR={r['pgd_asr']:.3f}  AUROC={r['auroc']:.3f}  ({r['train_time']:.0f}s)")

# ── 2. Expert ablation on K=4 ─────────────────────────────────────────────────
print("\n[Part 2] Expert ablation on K=4 model")
torch.manual_seed(SEED)
moe4 = MoEModel(n_experts=4, n_classes=10, width=16).to(device)
train_moe(moe4, Xtr, Ytr)

# Measure router usage per expert
with torch.no_grad():
    weights = moe4.router_weights(Xte_eval)  # (N_EVAL, 4)
mean_usage = weights.mean(0).cpu().numpy()
print(f"  Mean router weight per expert: {[f'{u:.3f}' for u in mean_usage]}")
most_used = int(mean_usage.argmax())
least_used = int(mean_usage.argmin())
print(f"  Most used: expert {most_used} ({mean_usage[most_used]:.3f})")
print(f"  Least used: expert {least_used} ({mean_usage[least_used]:.3f})")

ablation_results = []
# Baseline (no ablation)
r_base = evaluate(moe4, Xte_eval, Yte_eval, "K=4 baseline (all experts)")
ablation_results.append(r_base)
print(f"  Baseline: clean={r_base['clean_acc']:.3f}  PGD_ASR={r_base['pgd_asr']:.3f}")

# Ablate each expert
for i in range(4):
    moe4.disabled_experts = {i}
    label = f"K=4 remove expert {i} ({'most used' if i==most_used else 'least used' if i==least_used else 'mid'})"
    r = evaluate(moe4, Xte_eval, Yte_eval, label)
    r['removed_expert'] = i
    r['expert_usage'] = float(mean_usage[i])
    ablation_results.append(r)
    print(f"  Remove expert {i}: clean={r['clean_acc']:.3f}  PGD_ASR={r['pgd_asr']:.3f}  "
          f"ΔPGD={r['pgd_asr']-r_base['pgd_asr']:+.3f}")
moe4.disabled_experts = set()

# ── 3. Expert specialisation: do experts disagree on vulnerability? ───────────
print("\n[Part 3] Expert specialisation on K=4 model")
moe4.eval()
X_pgd_full = C.pgd(moe4, Xte_eval, Yte_eval, eps=EPS, steps=10, alpha=0.01)
with torch.no_grad():
    full_pgd_success = (moe4(X_pgd_full).argmax(1) != Yte_eval).cpu().float().numpy()

# Test each expert individually (without router)
expert_asrs = []
for i, expert in enumerate(moe4.experts):
    expert.eval()
    X_pgd_ei = C.pgd(expert, Xte_eval, Yte_eval, eps=EPS, steps=10, alpha=0.01)
    with torch.no_grad():
        asr_i = (expert(X_pgd_ei).argmax(1) != Yte_eval).float().mean().item()
    expert_asrs.append(asr_i)
    print(f"  Expert {i} standalone PGD ASR: {asr_i:.4f}")

# Cross-expert agreement: if sample fools expert i, does it fool expert j?
print(f"\n  Full MoE PGD ASR: {full_pgd_success.mean():.4f}")
print(f"  Expert standalone ASRs: {[f'{a:.3f}' for a in expert_asrs]}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n--- Scale Study Results ---")
print(f"{'K':>4} {'Clean Acc':>10} {'FGSM ASR':>10} {'PGD ASR':>10} {'AUROC':>8}")
for r in scale_results:
    print(f"  {r['n_experts']:>2}   {r['clean_acc']:>10.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['auroc']:>8.4f}")

print("\n--- Interpretation ---")
asrs = [r['pgd_asr'] for r in scale_results]
if asrs[-1] < asrs[0]:
    print("SUPPORTED: More experts → lower PGD ASR (diversity improves robustness)")
else:
    print("NOT SUPPORTED: Expert count does not improve robustness.")
    print("  Robustness is determined by training regime, not model ensemble size.")

# Check if most-used expert removal hurts more
abl_pgd = {r.get('removed_expert', -1): r['pgd_asr'] for r in ablation_results}
if most_used in abl_pgd and least_used in abl_pgd:
    delta_most = abl_pgd[most_used] - r_base['pgd_asr']
    delta_least = abl_pgd[least_used] - r_base['pgd_asr']
    print(f"\nRemoving most-used expert {most_used}: ΔPGD_ASR = {delta_most:+.4f}")
    print(f"Removing least-used expert {least_used}: ΔPGD_ASR = {delta_least:+.4f}")
    if delta_most > delta_least + 0.02:
        print("IMPORTANCE: Most-used expert is critical for robustness.")
    else:
        print("NO IMPORTANCE EFFECT: Expert usage ≠ adversarial importance.")
