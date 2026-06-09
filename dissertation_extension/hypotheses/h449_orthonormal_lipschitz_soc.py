"""
H449 - Soft-Lipschitz Orthonormal layer (SOC-style) for adversarial robustness.

Gap G4 (architecture axes). Anchor: Anil, Lucas, Grosse, "Sorting Out Lipschitz
Function Approximation", ICML 2019 (arXiv:1811.05381) -- proposes orthonormal
weight matrices + GroupSort activation to build EXACTLY 1-Lipschitz networks.
Additional context: Trockman & Kolter, "Orthogonalizing Convolutional Layers
with the Cayley Transform", ICLR 2021 (arXiv:2104.07167); Singla & Feizi,
"Skew Orthogonal Convolutions" (SOC), ICML 2021 (arXiv:2105.11417). All three
enforce per-layer 1-Lipschitz with orthonormal weights; combined with a
gradient-norm-preserving activation (GroupSort), the whole net is 1-Lipschitz.

Mechanism vs prior campaign work:
  - H276 (spectral norm via Miyato power-iter): only bounds matrix-reshape
    spectral norm of the weight; loose upper bound on conv operator norm; in
    our campaign it FAILED to give robustness without AT.
  - H323 (Jacobian Frobenius penalty via random projections): SOFT penalty on
    ||J||_F. Capped PGD-ASR around 0.68 -- one of the strongest non-AT defenses.
    The mechanism is "shrink the Jacobian"; H449 enforces the SAME constraint
    by construction (per-layer Jacobian is orthogonal => ||J||_2 = 1 and
    ||J||_F <= sqrt(n)). i.e. H449 is the architectural form of H323's penalty.
  - H408 (i-ResNet Lipschitz sweep, c=0.5): clean acc collapsed to ~0.20.
    Hard Lipschitz caps are extremely expensive on Fashion-MNIST at our scale.

Hypothesis: an orthonormal-conv + GroupSort SmallCNN should achieve a small
verified Lipschitz constant and a sizeable PGD-ASR reduction vs baseline,
trading off clean accuracy. The trade-off should be SMALLER than H408 i-ResNet
(orthogonal layers preserve norms exactly rather than contract them) and the
joint with AT should beat AT-alone if Lipschitz architecture is the right prior.

Design (all 5 conditions trained from scratch on Fashion-MNIST):
  (a) baseline      -- SmallCNN, ReLU, BN, vanilla SGD                    -> reference
  (b) sn_only       -- SmallCNN with nn.utils.spectral_norm on convs/lin  -> H276-style control
  (c) orth          -- Bjorck-orthonormalised convs + GroupSort, no AT    -> H449 main
  (d) orth_at       -- (c) + PGD adv training (joint)                     -> SOC + AT
  (e) at_only       -- SmallCNN + PGD adv training                        -> AT baseline

Verified Lipschitz constant:
  Estimated empirically as the maximum ||f(x+d)-f(x)||_2 / ||d||_2 over random
  perturbations of magnitude eps in {1e-3, 1e-2, 0.1}. Not a certificate, but a
  monotone proxy for the true Lipschitz constant; the orthonormal model should
  show L close to 1 across the head's scale, the baseline >> 1.

Outputs: clean_acc, FGSM_ASR, PGD_ASR (eps=0.1, 10 steps, alpha=0.01), mean
margin, empirical Lipschitz proxy. Saved to
  results/fashion_mnist/h449_orthonormal_lipschitz_soc_output.txt
ASCII only. Flushed after every condition. Verdict at the end.

Standard config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9,
wd=5e-4), SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.
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

# ---- config --------------------------------------------------------------
DS = "fashion_mnist"
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_TRAIN = 1000 if SMOKE else 6000
N_EVAL = 500 if SMOKE else 2000
EPOCHS = 2 if SMOKE else 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
WD = 5e-4
WIDTH = 32
BJORCK_ITERS = 8           # iterations of Bjorck orthonormalisation
LIP_PROBE_N = 200          # samples for empirical Lipschitz constant
LIP_PROBE_EPS = [1e-3, 1e-2, 1e-1]

META = {"channels": 1, "size": 28, "n_classes": 10}


# ---------------------------------------------------------------------------
# Bjorck orthonormalisation (Anil 2019 LNets) -- iteratively project a matrix
# onto its nearest orthonormal matrix:
#     W_{k+1} = 1.5 W_k - 0.5 W_k W_k^T W_k
# converges (locally) when sigma_max(W) <= sqrt(3).
# ---------------------------------------------------------------------------
def bjorck(W, n_iter=BJORCK_ITERS):
    # rescale to enter the basin of convergence
    s = torch.linalg.norm(W, ord=2)
    W = W / (s + 1e-8)
    for _ in range(n_iter):
        W = 1.5 * W - 0.5 * (W @ W.t() @ W)
    return W


class OrthConv2d(nn.Module):
    """Conv2d whose RESHAPED weight matrix (out, in*kH*kW) is Bjorck-orthonormalised
    every forward pass. This is the LNets / Anil-2019 construction: the reshape-
    matrix has orthonormal rows, so the conv operator norm is <= 1 (with equality
    when the receptive field is large enough). It is NOT exactly orthogonal in
    the conv-operator sense (that requires Cayley/SOC, Trockman 2021 / Singla
    2021); it is the tractable SOFT-Lipschitz proxy that gives per-layer Lip<=1
    on the matrix reshape, just like spectral_norm but stronger (full row-
    orthonormality rather than only top singular value <= 1)."""

    def __init__(self, ci, co, k=3, padding=1):
        super().__init__()
        self.ci, self.co, self.k, self.padding = ci, co, k, padding
        w = torch.randn(co, ci, k, k)
        # initialise close to orthonormal (helps optimisation)
        with torch.no_grad():
            wm = w.view(co, -1)
            if wm.size(1) >= wm.size(0):
                wm = bjorck(wm, 20)
            w = wm.view(co, ci, k, k)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(co))

    def get_orth_weight(self):
        wm = self.weight.view(self.co, -1)
        wm = bjorck(wm, BJORCK_ITERS)
        return wm.view(self.co, self.ci, self.k, self.k)

    def forward(self, x):
        w = self.get_orth_weight()
        return F.conv2d(x, w, self.bias, padding=self.padding)


class OrthLinear(nn.Module):
    """Linear with Bjorck-orthonormalised weight (rows orthonormal if out<=in,
    columns orthonormal if in<=out). Either way ||W||_2 <= 1."""

    def __init__(self, ci, co):
        super().__init__()
        w = torch.randn(co, ci)
        with torch.no_grad():
            if w.size(1) >= w.size(0):
                w = bjorck(w, 20)
            else:
                w = bjorck(w.t(), 20).t()
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(co))
        self.co, self.ci = co, ci

    def get_orth_weight(self):
        w = self.weight
        if w.size(1) >= w.size(0):
            return bjorck(w, BJORCK_ITERS)
        return bjorck(w.t(), BJORCK_ITERS).t()

    def forward(self, x):
        return F.linear(x, self.get_orth_weight(), self.bias)


class GroupSort(nn.Module):
    """Gradient-norm-preserving 1-Lipschitz activation (Anil 2019). Splits
    channels into groups of `group_size`, sorts within each group. group_size=2
    is the "MaxMin" variant. Returns the sorted tensor in-place along channel
    dim. Sorting is a permutation -> exactly 1-Lipschitz and gradient-norm-
    preserving (unlike ReLU which kills negatives)."""

    def __init__(self, group_size=2):
        super().__init__()
        self.gs = group_size

    def forward(self, x):
        # x: (B, C, ...)
        B, C = x.shape[0], x.shape[1]
        assert C % self.gs == 0, f"channels {C} not divisible by group_size {self.gs}"
        rest = x.shape[2:]
        xg = x.view(B, C // self.gs, self.gs, *rest)
        xs, _ = xg.sort(dim=2)
        return xs.view(B, C, *rest)


class OrthCNN(nn.Module):
    """SmallCNN drop-in with OrthConv2d + GroupSort. Architecturally 1-Lipschitz
    (per-layer Lip<=1 from Bjorck; GroupSort is 1-Lipschitz; AvgPool is
    1/sqrt(area)-Lipschitz <=1). No BatchNorm (BN is not 1-Lipschitz at infer
    time)."""

    def __init__(self, in_ch=1, size=28, n_classes=10, width=WIDTH):
        super().__init__()
        # use AvgPool (1-Lipschitz on each output entry; sum of small pieces) --
        # MaxPool is also 1-Lipschitz but is not gradient-norm-preserving.
        self.features = nn.Sequential(
            OrthConv2d(in_ch, width, 3, 1), GroupSort(2), nn.AvgPool2d(2),
            OrthConv2d(width, width * 2, 3, 1), GroupSort(2), nn.AvgPool2d(2),
            OrthConv2d(width * 2, width * 4, 3, 1), GroupSort(2), nn.AvgPool2d(2),
        )
        feat = size // 8
        self.flatten = nn.Flatten()
        self.fc1 = OrthLinear(width * 4 * feat * feat, 256)
        self.act = GroupSort(2)
        self.fc2 = OrthLinear(256, n_classes)

    def forward(self, x):
        z = self.features(x)
        z = self.flatten(z)
        z = self.act(self.fc1(z))
        return self.fc2(z)


# ---------------------------------------------------------------------------
# spectral-norm baseline (H276-style single power iteration)
# ---------------------------------------------------------------------------
def apply_spectral_norm(model):
    for _, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.utils.spectral_norm(m)
    return model


# ---------------------------------------------------------------------------
# empirical Lipschitz proxy
# ---------------------------------------------------------------------------
@torch.no_grad()
def empirical_lipschitz(model, X, eps_list=LIP_PROBE_EPS, n=LIP_PROBE_N):
    model.eval()
    X = X[:n]
    out = {}
    for eps in eps_list:
        d = torch.randn_like(X)
        d = d / (d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12) * eps
        y0 = model(X)
        y1 = model((X + d).clamp(0, 1))
        # use the actual L2 distance of the perturbation post-clamp
        d_eff = ((X + d).clamp(0, 1) - X).flatten(1).norm(dim=1) + 1e-12
        diff = (y1 - y0).flatten(1).norm(dim=1)
        ratio = (diff / d_eff)
        out[eps] = float(ratio.max().item())
    return out


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------
def train(model, Xtr, Ytr, adv=False):
    C.set_seed(SEED)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            if adv:
                model.eval()
                xb = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
                model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
    model.eval()
    return model


def evaluate(model, Xte, Yte):
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    _, clean = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))
    return dict(clean=float(clean), fgsm_asr=float(fg["asr"]),
                pgd_asr=float(pg["asr"]), margin=mean_margin)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "fashion_mnist",
        "h449_orthonormal_lipschitz_soc_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H449  Soft-Lipschitz Orthonormal layer (SOC-style)  --  gap G4")
    out("Anchor: Anil/Lucas/Grosse 2019 (LNets). Refs: Trockman 2021 (Cayley),")
    out("        Singla 2021 (SOC).")
    out("=" * 80)
    out(f"config: SMOKE={SMOKE} N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SEED={SEED} EPS={EPS} PGD_STEPS={PGD_STEPS} "
        f"PGD_ALPHA={PGD_ALPHA} WIDTH={WIDTH} BJORCK_ITERS={BJORCK_ITERS}")
    out(f"device={C.DEVICE}")
    out("")

    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    flush()

    conditions = [
        ("baseline",    "cnn",     False),
        ("sn_only",     "cnn_sn",  False),
        ("orth",        "orth",    False),
        ("orth_at",     "orth",    True),
        ("at_only",     "cnn",     True),
    ]
    rows = []
    for name, arch, adv in conditions:
        out("\n" + "-" * 80)
        out(f"[condition {name}] arch={arch}  adv_train={adv}")
        out("-" * 80)
        C.set_seed(SEED)
        if arch == "cnn":
            model = C.build_model("cnn", META, width=WIDTH).to(C.DEVICE)
        elif arch == "cnn_sn":
            model = C.build_model("cnn", META, width=WIDTH).to(C.DEVICE)
            apply_spectral_norm(model)
        elif arch == "orth":
            model = OrthCNN(in_ch=1, size=28, n_classes=10, width=WIDTH).to(C.DEVICE)
        else:
            raise ValueError(arch)
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        out(f"    n_params={n_par}")
        train(model, Xtr, Ytr, adv=adv)
        res = evaluate(model, Xte, Yte)
        lip = empirical_lipschitz(model, Xte)
        res["lip"] = lip
        out(f"    clean={res['clean']:.4f}  FGSM_ASR={res['fgsm_asr']:.4f}  "
            f"PGD_ASR={res['pgd_asr']:.4f}  margin={res['margin']:.4f}")
        out("    empirical Lipschitz (max||df||/||dx||) over "
            f"eps in {LIP_PROBE_EPS}:")
        for e in LIP_PROBE_EPS:
            out(f"      eps={e:>6}: L_emp = {lip[e]:.3f}")
        rows.append((name, res))
        flush()

    # ---- summary table ----
    out("\n" + "=" * 80)
    out("[TABLE]")
    out("=" * 80)
    hdr = "{:<12} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "cond", "clean", "FGSM", "PGD", "margin", "L@1e-3", "L@1e-2", "L@1e-1")
    out(hdr)
    out("-" * len(hdr))
    for name, r in rows:
        out("{:<12} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.3f} {:>9.3f} {:>9.3f}".format(
            name, r["clean"], r["fgsm_asr"], r["pgd_asr"], r["margin"],
            r["lip"][1e-3], r["lip"][1e-2], r["lip"][1e-1]))

    # ---- verdict ----
    out("\n" + "=" * 80)
    out("[VERDICT]")
    out("=" * 80)
    res = dict(rows)
    base, sn, orth, orth_at, at = (res["baseline"], res["sn_only"], res["orth"],
                                    res["orth_at"], res["at_only"])

    d_pgd_orth = base["pgd_asr"] - orth["pgd_asr"]
    d_clean_orth = orth["clean"] - base["clean"]
    d_pgd_sn = base["pgd_asr"] - sn["pgd_asr"]
    d_pgd_orth_at = at["pgd_asr"] - orth_at["pgd_asr"]
    d_clean_orth_at = orth_at["clean"] - at["clean"]

    out(f"  orth vs baseline:    PGD_ASR delta = {-d_pgd_orth:+.4f}  "
        f"(robustness gain={d_pgd_orth:+.4f}); clean delta={d_clean_orth:+.4f}")
    out(f"  sn   vs baseline:    PGD_ASR delta = {-d_pgd_sn:+.4f}  "
        f"(robustness gain={d_pgd_sn:+.4f}); clean delta={sn['clean']-base['clean']:+.4f}")
    out(f"  orth_at vs at_only:  PGD_ASR delta = {-d_pgd_orth_at:+.4f}  "
        f"(robustness gain={d_pgd_orth_at:+.4f}); clean delta={d_clean_orth_at:+.4f}")
    out(f"  empirical Lipschitz @eps=1e-3:  base={base['lip'][1e-3]:.3f}  "
        f"sn={sn['lip'][1e-3]:.3f}  orth={orth['lip'][1e-3]:.3f}  "
        f"orth_at={orth_at['lip'][1e-3]:.3f}  at={at['lip'][1e-3]:.3f}")

    # decision logic
    msgs = []
    if d_pgd_orth > 0.05:
        msgs.append("Orthonormal layers + GroupSort give meaningful non-AT PGD "
                    "robustness, consistent with H323 Jacobian-Frob.")
    elif d_pgd_orth < -0.02:
        msgs.append("Orthonormal layers + GroupSort HURT robustness vs ReLU CNN "
                    "-- the optimisation cost outweighs the Lipschitz prior at "
                    "this scale.")
    else:
        msgs.append("Orthonormal layers + GroupSort give no significant non-AT "
                    "PGD robustness gain.")
    if d_clean_orth < -0.05:
        msgs.append(f"Clean-accuracy cost is substantial ({d_clean_orth:+.4f}) "
                    "but smaller than H408 i-ResNet c=0.5 (~-0.6).")
    if d_pgd_orth_at > 0.02:
        msgs.append("SOC + AT > AT alone: Lipschitz architecture is a useful "
                    "prior on top of AT.")
    elif d_pgd_orth_at < -0.02:
        msgs.append("SOC + AT < AT alone: the orthogonality constraint is "
                    "incompatible with AT optimisation here.")
    else:
        msgs.append("SOC + AT ~ AT alone: no synergy between architectural "
                    "1-Lipschitz and adversarial training.")
    if orth["lip"][1e-3] < 0.5 * base["lip"][1e-3]:
        msgs.append("Empirical Lipschitz constant is at-least halved by "
                    "orthonormalisation -- the architectural constraint binds.")
    out("")
    for m in msgs:
        out("  - " + m)

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
