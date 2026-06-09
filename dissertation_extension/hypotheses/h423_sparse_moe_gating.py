"""
H423 - Sparse Mixture-of-Experts gating for adversarial robustness.

Hypothesis: sparse top-k routing forces an attacker to flip BOTH the gating
decision AND the expert prediction simultaneously, raising effective attack
difficulty relative to a monolithic network or dense ensemble.

Reference: Shazeer et al. (2017) "Outrageously Large Neural Networks: The
Sparsely-Gated Mixture-of-Experts Layer." ICLR 2017.

Four conditions (all use the same CNN backbone width, same training config):
  1. baseline      -- single SmallCNN, no MoE.
  2. dense_ensemble -- 4 CNNs, output = mean of all logits (dense, k=4).
  3. sparse_moe_k1 -- 4 expert CNNs, gating selects top-1 per input.
  4. sparse_moe_k2 -- 4 expert CNNs, gating selects top-2 per input.

The gating network is a small linear layer over a flattened 7x7 pooled feature
map shared across experts (a lightweight shared trunk). During training the gate
uses a straight-through estimator: the top-k hard mask is applied in the forward
pass but gradients flow through the soft gate scores (following Shazeer 2017
Sec. 2). A load-balancing auxiliary loss (coefficient 0.01) encourages even
expert utilisation.

Config: N_TRAIN=6000, N_EVAL=2000, N_EXPERTS=4, EPOCHS=12, LR=0.05,
BATCH=128, SGD mom=0.9 wd=5e-4, SEED=0, EPS=0.1, PGD_STEPS=10.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config ------------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
N_EXPERTS = 4
EPOCHS = 12
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
LB_COEF = 0.01          # load-balancing loss coefficient
WIDTH = 32
META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h423_sparse_moe_gating_output.txt",
)

# ---- shared CNN trunk (feature extractor, no head) ---------------------------

class CNNTrunk(nn.Module):
    """Three conv-pool blocks; returns (B, width*4, feat, feat) feature maps."""
    def __init__(self, in_ch=1, width=32):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.net = nn.Sequential(
            *block(in_ch, width),
            *block(width, width * 2),
            *block(width * 2, width * 4),
        )

    def forward(self, x):
        return self.net(x)


class ExpertHead(nn.Module):
    """Linear classification head on flattened trunk output."""
    def __init__(self, feat_dim, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, h):
        return self.fc2(F.relu(self.fc1(h)))


# ---- Sparse MoE model --------------------------------------------------------

class SparseMoE(nn.Module):
    """
    Shared CNN trunk -> N_EXPERTS expert heads + 1 gating network.

    Gate: linear(flat_feat) -> softmax scores; top-k hard mask applied in
    forward, gradients flow through soft scores (straight-through).
    """
    def __init__(self, n_experts=4, k=1, in_ch=1, size=28, n_classes=10,
                 width=32):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.trunk = CNNTrunk(in_ch, width)
        feat = size // 8           # 28 -> 3 (after 3x MaxPool2)
        feat_dim = width * 4 * feat * feat
        self.experts = nn.ModuleList(
            [ExpertHead(feat_dim, n_classes) for _ in range(n_experts)]
        )
        self.gate = nn.Linear(feat_dim, n_experts)

    def forward(self, x, return_gate=False):
        h = self.trunk(x).flatten(1)          # (B, feat_dim)
        scores = self.gate(h)                  # (B, n_experts) raw logits
        soft = F.softmax(scores, dim=-1)       # (B, n_experts)

        # top-k hard mask (straight-through: detach mask, keep soft for grad)
        _, topk_idx = soft.topk(self.k, dim=-1)   # (B, k)
        hard = torch.zeros_like(soft)
        hard.scatter_(1, topk_idx, 1.0)
        # straight-through: hard in forward, soft gradient
        gate_weights = hard + soft - soft.detach()  # (B, n_experts)

        # weighted sum of expert logits
        expert_outs = torch.stack(
            [e(h) for e in self.experts], dim=2
        )  # (B, n_classes, n_experts)
        out = (expert_outs * gate_weights.unsqueeze(1)).sum(2)  # (B, n_classes)

        if return_gate:
            return out, soft
        return out

    def load_balance_loss(self, x):
        """Auxiliary loss: encourage uniform expert utilisation."""
        _, soft = self.forward(x, return_gate=True)
        # variance of mean gate weights across batch -> minimise
        mean_per_expert = soft.mean(0)  # (n_experts,)
        target = torch.full_like(mean_per_expert, 1.0 / self.n_experts)
        return F.mse_loss(mean_per_expert, target)


# ---- Dense ensemble model ----------------------------------------------------

class DenseEnsemble(nn.Module):
    """4 independent full SmallCNNs; logits averaged (k = n_experts)."""
    def __init__(self, n_experts=4, in_ch=1, size=28, n_classes=10, width=32):
        super().__init__()
        self.members = nn.ModuleList(
            [C.SmallCNN(in_ch, size, n_classes, width=width) for _ in range(n_experts)]
        )

    def forward(self, x):
        return torch.stack([m(x) for m in self.members], dim=0).mean(0)


# ---- training helpers --------------------------------------------------------

def _sgd(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def train_baseline(Xtr, Ytr):
    C.set_seed(SEED)
    model = C.SmallCNN(META["channels"], META["size"], META["n_classes"],
                       width=WIDTH).to(C.DEVICE)
    opt = _sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_dense_ensemble(Xtr, Ytr):
    C.set_seed(SEED)
    model = DenseEnsemble(N_EXPERTS, META["channels"], META["size"],
                          META["n_classes"], WIDTH).to(C.DEVICE)
    opt = _sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            # sum of individual CE losses (identical to mean-logit CE when
            # all members are identical capacity)
            loss = sum(F.cross_entropy(m(xb), yb) for m in model.members) / N_EXPERTS
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_sparse_moe(Xtr, Ytr, k):
    C.set_seed(SEED)
    model = SparseMoE(N_EXPERTS, k, META["channels"], META["size"],
                      META["n_classes"], WIDTH).to(C.DEVICE)
    opt = _sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            out, soft = model(xb, return_gate=True)
            ce = F.cross_entropy(out, yb)
            # load-balancing auxiliary loss
            mean_gate = soft.mean(0)
            target = torch.full_like(mean_gate, 1.0 / N_EXPERTS)
            lb = F.mse_loss(mean_gate, target)
            loss = ce + LB_COEF * lb
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def accuracy(model, X, Y):
    correct = 0
    for i in range(0, X.size(0), 256):
        xb, yb = X[i:i + 256], Y[i:i + 256]
        correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / Y.size(0)


def asr(model, X, Y, attack):
    res = C.attack_success(model, X, Y, attack=attack, eps=EPS,
                           steps=PGD_STEPS, batch=128)
    return res["asr"]


# ---- main --------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("H423 - Sparse MoE Gating (Shazeer 2017)")
    log(f"Dataset={DS}  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}  "
        f"N_EXPERTS={N_EXPERTS}  EPOCHS={EPOCHS}  EPS={EPS}")
    log(f"Hypothesis: sparse top-k routing raises attack difficulty by "
        f"forcing attacker to jointly fool gate + winning expert(s).")
    log()

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN,
                                          n_eval=N_EVAL, seed=SEED)

    conditions = [
        ("baseline",      lambda: train_baseline(Xtr, Ytr)),
        ("dense_k4",      lambda: train_dense_ensemble(Xtr, Ytr)),
        ("sparse_moe_k1", lambda: train_sparse_moe(Xtr, Ytr, k=1)),
        ("sparse_moe_k2", lambda: train_sparse_moe(Xtr, Ytr, k=2)),
    ]

    results = {}
    for name, train_fn in conditions:
        t0 = time.time()
        log(f"--- {name} ---")
        model = train_fn()
        clean_acc = accuracy(model, Xte, Yte)
        fgsm_asr = asr(model, Xte, Yte, "fgsm")
        pgd_asr  = asr(model, Xte, Yte, "pgd")
        elapsed = time.time() - t0
        results[name] = dict(clean_acc=clean_acc,
                             fgsm_asr=fgsm_asr, pgd_asr=pgd_asr)
        log(f"  clean_acc={clean_acc:.4f}  fgsm_asr={fgsm_asr:.4f}"
            f"  pgd_asr={pgd_asr:.4f}  ({elapsed:.1f}s)")
        log()

    log("=== SUMMARY ===")
    log(f"{'condition':<18} {'clean_acc':>10} {'fgsm_asr':>10} {'pgd_asr':>10}")
    for name, r in results.items():
        log(f"{name:<18} {r['clean_acc']:>10.4f} {r['fgsm_asr']:>10.4f}"
            f" {r['pgd_asr']:>10.4f}")

    log()
    base = results["baseline"]
    for name in ("dense_k4", "sparse_moe_k1", "sparse_moe_k2"):
        r = results[name]
        delta_fgsm = base["fgsm_asr"] - r["fgsm_asr"]
        delta_pgd  = base["pgd_asr"]  - r["pgd_asr"]
        log(f"{name}: DELTA fgsm_asr={delta_fgsm:+.4f}  pgd_asr={delta_pgd:+.4f}"
            f"  (positive = more robust than baseline)")

    log()
    log("Ref: Shazeer et al. (2017) ICLR - Sparsely-Gated MoE.")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
