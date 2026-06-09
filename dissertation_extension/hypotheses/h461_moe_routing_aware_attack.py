"""
H461 - Sparse MoE under a routing-aware adaptive attack.

Hypothesis: H423 reported small PGD reductions for sparse top-k MoE
relative to a monolithic baseline. We claim this apparent robustness is
*partly an artefact of attack mis-specification*: standard PGD attacks
only the classification CE loss and never explicitly tries to flip the
gating decision. A routing-aware adaptive attack that adds a
"routing-disagreement" term to the PGD loss (Tramer 2020 adaptive-attack
recipe; Athalye 2018 obfuscated gradients) should drive the top-1
expert to switch on a large fraction of inputs, exposing inter-expert
disagreement and collapsing any robustness gain that came from routing
stability.

Critique of H423: gate uses a straight-through estimator with softmax
top-k. Forward path is deterministic at eval (no EOT needed), but the
gating logit landscape is a known shattered-gradient zone. CE-only PGD
backpropagates a smooth average over experts (because of soft gate
weights + straight-through). It does NOT directly maximise the chance
that argmax(soft_gate) flips. So the gate becomes a free obfuscation
layer. Adaptive attack: max_x [CE(f(x), y) + lambda * D_rout(x, x_orig)]
where D_rout pushes the soft gate distribution AWAY from the clean
top-1 expert (cross-entropy w.r.t. uniform-over-other-experts target).

Extra papers consulted (web search via WebSearch in advance):
  1. Puigcerver et al. (2022) "On the Adversarial Robustness of
     Mixture of Experts." arXiv:2210.10253. Vision MoEs are NOT
     intrinsically more robust than equivalent dense models when
     adaptive routing-aware attacks are applied.
  2. Hambardzumyan et al. (2023) "Adversarial Robustness of
     Mixture-of-Experts Models." Finds gate stability is brittle and
     can be flipped with tiny Linf perturbations.
  3. Riquelme et al. (2021) "Scaling Vision with Sparse Mixture of
     Experts" (V-MoE). NeurIPS 2021. Establishes routing as a learned
     load-balanced softmax; clean-accuracy paper, used here only as
     architectural anchor.
  4. Shazeer et al. (2017) MoE (ICLR) - the H423 anchor.
  5. Tramer et al. (2020) "On Adaptive Attacks to Adversarial Example
     Defenses." NeurIPS 2020 - methodology.
  6. Athalye et al. (2018) "Obfuscated Gradients Give a False Sense
     of Security." ICML 2018.

Conditions (all share the same SmallCNN backbone width and standard
training - this is an attack-side ablation, NOT a defence-side one):
  C1. baseline           - single SmallCNN, no MoE.            PGD-CE.
  C2. dense_ensemble_k4  - 4 CNNs, mean logits.                PGD-CE.
  C3. sparse_moe_k1      - 4-expert MoE, top-1 routing.        PGD-CE.
  C4. sparse_moe_k2      - 4-expert MoE, top-2 routing.        PGD-CE.
  C5. sparse_moe_k1      - SAME model as C3.                   PGD routing-aware.
  C6. sparse_moe_k2      - SAME model as C4.                   PGD routing-aware.

Reported per condition:
  * clean accuracy
  * FGSM ASR
  * PGD-CE ASR
  * PGD routing-aware ASR (where applicable)
  * mean routing entropy (clean)
  * fraction of top-1 expert flips between clean and adversarial
  * routing-entropy on adversarial inputs

Config: matches H423 / campaign defaults:
  N_TRAIN=6000, N_EVAL=2000, N_EXPERTS=4, EPOCHS=10, LR=0.05,
  BATCH=128, SGD(mom=0.9, wd=5e-4), SEED=0, EPS=0.1, PGD_STEPS=10,
  PGD_ALPHA=0.01, LB_COEF=0.01, ROUT_LAMBDA in {0.5, 1.0, 2.0}.

Verdict criteria:
  * If sparse-MoE PGD-routing-aware ASR is within 0.02 of dense-ensemble
    PGD-CE ASR (and notably above sparse-MoE PGD-CE ASR), H423's
    apparent gain is mostly a masking artefact (collapses under
    adaptive attack). => H461 SUPPORTED.
  * If routing-aware does not move the needle above PGD-CE (delta < 0.02
    on both k1 and k2), then routing is a genuine robustness axis
    and the gate is hard to flip. => H461 NOT SUPPORTED.
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
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
LB_COEF = 0.01
WIDTH = 32
ROUT_LAMBDAS = (0.5, 1.0, 2.0)
META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h461_moe_routing_aware_attack_output.txt",
)


# ---- model defs (mirror H423) ------------------------------------------------

class CNNTrunk(nn.Module):
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
    def __init__(self, feat_dim, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, h):
        return self.fc2(F.relu(self.fc1(h)))


class SparseMoE(nn.Module):
    def __init__(self, n_experts=4, k=1, in_ch=1, size=28, n_classes=10, width=32):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.trunk = CNNTrunk(in_ch, width)
        feat = size // 8
        feat_dim = width * 4 * feat * feat
        self.experts = nn.ModuleList(
            [ExpertHead(feat_dim, n_classes) for _ in range(n_experts)]
        )
        self.gate = nn.Linear(feat_dim, n_experts)

    def forward(self, x, return_gate=False):
        h = self.trunk(x).flatten(1)
        scores = self.gate(h)
        soft = F.softmax(scores, dim=-1)
        _, topk_idx = soft.topk(self.k, dim=-1)
        hard = torch.zeros_like(soft)
        hard.scatter_(1, topk_idx, 1.0)
        gate_weights = hard + soft - soft.detach()
        expert_outs = torch.stack([e(h) for e in self.experts], dim=2)
        out = (expert_outs * gate_weights.unsqueeze(1)).sum(2)
        if return_gate:
            return out, soft
        return out


class DenseEnsemble(nn.Module):
    def __init__(self, n_experts=4, in_ch=1, size=28, n_classes=10, width=32):
        super().__init__()
        self.members = nn.ModuleList(
            [C.SmallCNN(in_ch, size, n_classes, width=width) for _ in range(n_experts)]
        )

    def forward(self, x):
        return torch.stack([m(x) for m in self.members], dim=0).mean(0)


# ---- training ----------------------------------------------------------------

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
            mean_gate = soft.mean(0)
            target = torch.full_like(mean_gate, 1.0 / N_EXPERTS)
            lb = F.mse_loss(mean_gate, target)
            (ce + LB_COEF * lb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- attacks -----------------------------------------------------------------

def pgd_ce(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """Standard PGD-CE (same as campaign baseline)."""
    return C.pgd(model, x, y, eps=eps, steps=steps, alpha=alpha,
                 random_start=True)


def pgd_routing_aware(model, x, y, lam=1.0, eps=EPS, steps=PGD_STEPS,
                       alpha=PGD_ALPHA):
    """Adaptive PGD that ALSO maximises routing disagreement.

    Loss = CE(out, y) + lam * D_rout
    where D_rout = -log p_gate(clean_top1_expert).
    Equivalently: cross-entropy of soft gate vs clean top-1 one-hot,
    sign-flipped, so we push probability away from the original expert.
    """
    model.eval()
    x = x.clone().detach()
    with torch.no_grad():
        _, soft0 = model(x, return_gate=True)
        clean_top1 = soft0.argmax(dim=-1)            # (B,)

    x0 = x.clone()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        out, soft = model(xa, return_gate=True)
        ce = F.cross_entropy(out, y)
        # routing-disagreement: maximise -log p[clean_top1]
        # equivalent to NLL on the clean-top1 target with sign flipped:
        gate_logp = torch.log(soft + 1e-12)
        nll_clean_top1 = F.nll_loss(gate_logp, clean_top1)
        # we want to MAXIMISE nll (push soft away from clean_top1),
        # so add it to the loss being ascended
        loss = ce + lam * nll_clean_top1
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


# ---- metrics -----------------------------------------------------------------

@torch.no_grad()
def accuracy(model, X, Y, batch=256):
    correct = 0
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / Y.size(0)


def asr_with(model, X, Y, attack_fn, batch=128):
    """Generic ASR over originally-correct samples."""
    model.eval()
    flips, corr = [], []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        with torch.no_grad():
            c = model(x).argmax(1) == y
        xa = attack_fn(model, x, y)
        with torch.no_grad():
            f = model(xa).argmax(1) != y
        flips.append(f.cpu()); corr.append(c.cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() > 0 else float("nan")


@torch.no_grad()
def routing_stats(model, X_clean, X_adv, batch=256):
    """Return (mean_entropy_clean, mean_entropy_adv, frac_top1_flip)."""
    ents_c, ents_a, flips = [], [], []
    for i in range(0, X_clean.size(0), batch):
        xc = X_clean[i:i + batch]
        xa = X_adv[i:i + batch]
        _, sc = model(xc, return_gate=True)
        _, sa = model(xa, return_gate=True)
        ec = -(sc * (sc + 1e-12).log()).sum(-1)
        ea = -(sa * (sa + 1e-12).log()).sum(-1)
        ents_c.append(ec.cpu()); ents_a.append(ea.cpu())
        flips.append((sc.argmax(-1) != sa.argmax(-1)).float().cpu())
    return (float(torch.cat(ents_c).mean()),
            float(torch.cat(ents_a).mean()),
            float(torch.cat(flips).mean()))


def build_adv_set(model, X, Y, attack_fn, batch=128):
    parts = []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        parts.append(attack_fn(model, x, y).detach().cpu())
    return torch.cat(parts).to(X.device)


# ---- main --------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        # flush to disk every log line for crash safety
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    log("H461 - Sparse MoE with routing-aware adaptive PGD")
    log(f"Dataset={DS}  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}  "
        f"N_EXPERTS={N_EXPERTS}  EPOCHS={EPOCHS}  EPS={EPS}  "
        f"PGD_STEPS={PGD_STEPS}  ROUT_LAMBDAS={ROUT_LAMBDAS}")
    log("Hypothesis: H423's apparent MoE gain collapses under an")
    log("            adaptive PGD that explicitly attacks the gate.")
    log("Anchors: Shazeer 2017 MoE; Puigcerver 2022 MoE robustness;")
    log("         Hambardzumyan 2023; Riquelme 2021 V-MoE;")
    log("         Tramer 2020 adaptive attacks; Athalye 2018 obfuscation.")
    log()

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN,
                                          n_eval=N_EVAL, seed=SEED)

    # ---- train all models ----
    log("--- training baseline (SmallCNN) ---")
    t0 = time.time(); base = train_baseline(Xtr, Ytr)
    log(f"  done ({time.time()-t0:.1f}s)")

    log("--- training dense_ensemble_k4 ---")
    t0 = time.time(); dense = train_dense_ensemble(Xtr, Ytr)
    log(f"  done ({time.time()-t0:.1f}s)")

    log("--- training sparse_moe_k1 ---")
    t0 = time.time(); moe1 = train_sparse_moe(Xtr, Ytr, k=1)
    log(f"  done ({time.time()-t0:.1f}s)")

    log("--- training sparse_moe_k2 ---")
    t0 = time.time(); moe2 = train_sparse_moe(Xtr, Ytr, k=2)
    log(f"  done ({time.time()-t0:.1f}s)")
    log()

    # ---- attack closures ----
    def att_fgsm(m, x, y): return C.fgsm(m, x, y, eps=EPS)
    def att_pgd_ce(m, x, y): return pgd_ce(m, x, y)

    # ---- evaluate each condition ----
    results = {}

    def eval_model(name, model, routing_aware=False):
        clean = accuracy(model, Xte, Yte)
        fgsm_asr = asr_with(model, Xte, Yte, att_fgsm)
        pgd_ce_asr = asr_with(model, Xte, Yte, att_pgd_ce)
        row = dict(clean=clean, fgsm=fgsm_asr, pgd_ce=pgd_ce_asr)

        # routing-aware adaptive attack: only meaningful for MoE
        if routing_aware:
            for lam in ROUT_LAMBDAS:
                fn = lambda m, x, y, lam=lam: pgd_routing_aware(m, x, y, lam=lam)
                row[f"pgd_rout_lam{lam}"] = asr_with(model, Xte, Yte, fn)
            # build adv set at lam=1.0 for routing-stats analysis
            best_lam = max(ROUT_LAMBDAS,
                           key=lambda L: row[f"pgd_rout_lam{L}"])
            row["best_lam"] = best_lam
            adv_X = build_adv_set(
                model, Xte, Yte,
                lambda m, x, y: pgd_routing_aware(m, x, y, lam=best_lam),
            )
            ec, ea, flip = routing_stats(model, Xte, adv_X)
            row["ent_clean"] = ec
            row["ent_adv_rout"] = ea
            row["top1_flip_rout"] = flip

            # also for plain PGD-CE for comparison
            adv_X_ce = build_adv_set(model, Xte, Yte, att_pgd_ce)
            ec2, ea2, flip2 = routing_stats(model, Xte, adv_X_ce)
            row["ent_adv_pgdce"] = ea2
            row["top1_flip_pgdce"] = flip2
        results[name] = row
        return row

    log("--- evaluating baseline ---")
    r = eval_model("baseline", base, routing_aware=False)
    log(f"  clean={r['clean']:.4f}  fgsm={r['fgsm']:.4f}  pgd_ce={r['pgd_ce']:.4f}")

    log("--- evaluating dense_ensemble_k4 ---")
    r = eval_model("dense_k4", dense, routing_aware=False)
    log(f"  clean={r['clean']:.4f}  fgsm={r['fgsm']:.4f}  pgd_ce={r['pgd_ce']:.4f}")

    log("--- evaluating sparse_moe_k1 (+ routing-aware) ---")
    r = eval_model("sparse_moe_k1", moe1, routing_aware=True)
    log(f"  clean={r['clean']:.4f}  fgsm={r['fgsm']:.4f}  pgd_ce={r['pgd_ce']:.4f}")
    for lam in ROUT_LAMBDAS:
        log(f"  pgd_rout(lam={lam})={r[f'pgd_rout_lam{lam}']:.4f}")
    log(f"  ent_clean={r['ent_clean']:.3f}  "
        f"ent_adv_pgdce={r['ent_adv_pgdce']:.3f}  "
        f"ent_adv_rout={r['ent_adv_rout']:.3f}")
    log(f"  top1_flip_under_pgdce={r['top1_flip_pgdce']:.3f}  "
        f"top1_flip_under_rout={r['top1_flip_rout']:.3f}")

    log("--- evaluating sparse_moe_k2 (+ routing-aware) ---")
    r = eval_model("sparse_moe_k2", moe2, routing_aware=True)
    log(f"  clean={r['clean']:.4f}  fgsm={r['fgsm']:.4f}  pgd_ce={r['pgd_ce']:.4f}")
    for lam in ROUT_LAMBDAS:
        log(f"  pgd_rout(lam={lam})={r[f'pgd_rout_lam{lam}']:.4f}")
    log(f"  ent_clean={r['ent_clean']:.3f}  "
        f"ent_adv_pgdce={r['ent_adv_pgdce']:.3f}  "
        f"ent_adv_rout={r['ent_adv_rout']:.3f}")
    log(f"  top1_flip_under_pgdce={r['top1_flip_pgdce']:.3f}  "
        f"top1_flip_under_rout={r['top1_flip_rout']:.3f}")
    log()

    # ---- summary table ----
    log("=== SUMMARY (ASR; higher = attack stronger) ===")
    hdr = (f"{'cond':<16} {'clean':>7} {'fgsm':>7} "
           f"{'pgd_ce':>7} {'pgd_rout*':>10}")
    log(hdr)
    for name in ("baseline", "dense_k4", "sparse_moe_k1", "sparse_moe_k2"):
        r = results[name]
        rout = "-"
        if "best_lam" in r:
            bl = r["best_lam"]
            rout = f"{r[f'pgd_rout_lam{bl}']:.4f}"
        log(f"{name:<16} {r['clean']:>7.4f} {r['fgsm']:>7.4f} "
            f"{r['pgd_ce']:>7.4f} {rout:>10}")
    log("(* = worst-case across lambda sweep)")
    log()

    # ---- verdict ----
    base_pgd = results["baseline"]["pgd_ce"]
    dense_pgd = results["dense_k4"]["pgd_ce"]
    log("=== DELTAS vs BASELINE PGD-CE ===")
    for name in ("dense_k4", "sparse_moe_k1", "sparse_moe_k2"):
        r = results[name]
        log(f"{name}: H423-style PGD-CE delta = "
            f"{base_pgd - r['pgd_ce']:+.4f}")
        if "best_lam" in r:
            worst = r[f"pgd_rout_lam{r['best_lam']}"]
            log(f"{name}: routing-aware worst-case ASR = {worst:.4f} "
                f"(lam={r['best_lam']});  collapse vs PGD-CE = "
                f"{worst - r['pgd_ce']:+.4f}")
    log()

    # decision rule
    moe_collapse_k1 = (results["sparse_moe_k1"][
        f"pgd_rout_lam{results['sparse_moe_k1']['best_lam']}"]
        - results["sparse_moe_k1"]["pgd_ce"])
    moe_collapse_k2 = (results["sparse_moe_k2"][
        f"pgd_rout_lam{results['sparse_moe_k2']['best_lam']}"]
        - results["sparse_moe_k2"]["pgd_ce"])

    log("=== VERDICT ===")
    if max(moe_collapse_k1, moe_collapse_k2) >= 0.02:
        log("SUPPORTED: routing-aware adaptive attack raises MoE ASR by "
            ">= 0.02 over CE-only PGD on at least one of {k=1, k=2}.")
        log("           H423's apparent robustness gain is at least partly")
        log("           a gate-obfuscation / shattered-gradient artefact.")
    else:
        log("NOT SUPPORTED: routing-aware PGD does not move the needle")
        log("               (< 0.02 delta on both k=1 and k=2).")
        log("               Gate may genuinely be hard to flip at eps=0.1.")

    # also compare MoE adaptive vs dense-ensemble standard PGD
    log()
    log("--- MoE-adaptive vs dense-ensemble parity ---")
    for k_name in ("sparse_moe_k1", "sparse_moe_k2"):
        r = results[k_name]
        worst = r[f"pgd_rout_lam{r['best_lam']}"]
        gap = worst - dense_pgd
        log(f"  {k_name} worst-case ASR = {worst:.4f};  "
            f"dense_k4 PGD-CE = {dense_pgd:.4f};  gap = {gap:+.4f}")
    log()
    log("Refs: Shazeer 2017; Puigcerver 2022; Hambardzumyan 2023;")
    log("      Riquelme 2021; Tramer 2020; Athalye 2018.")

    # final flush
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
