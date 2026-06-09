"""
H455 - Manifold-Mixup-AT (Fashion-MNIST).

Seed: extends H124 / H270 + Verma 2019 (Manifold-Mixup, ICML). Gap G3 (loss /
training-objective variants of AT, mixup family). Anchor papers reviewed:
  * Verma et al., "Manifold Mixup: Better Representations by Interpolating
    Hidden States", ICML 2019. Mix pairs of samples at a *randomly-chosen*
    hidden layer; mix labels with the same lambda ~ Beta(alpha, alpha).
    Promotes flatter class-conditional manifolds at intermediate layers.
  * Pang et al., "Mixup Inference: Better Exploiting Mixup to Defend
    Adversarial Attacks" + "Bag of Tricks for AT" (2020/2021). Vanilla input
    Mixup combined with AT helps modestly under PGD-10 but degrades under
    stronger attack; warns that mixup-AT can MASK gradient signal.
  * Lee, Lee, Yoon, Hwang, "Adversarial Vertex Mixup" (CVPR 2020). Argues
    vanilla mixup blurs the AT decision boundary; proposes "adversarial
    vertex mixup" to *combat* the masking effect.

Critique (key methodological point on attack policy):
  Manifold-Mixup mixes at a HIDDEN layer during training. The PGD threat
  model is bound by ||delta||_inf <= eps in INPUT space. There is no direct
  conflict because at inference the network is deterministic (no mixing);
  the same input-space PGD bound applies. However, two pitfalls must be
  controlled for:
    (1) During the AT inner-max we MUST disable mixing -- otherwise the
        model is stochastic and PGD's max becomes meaningless. We attack
        the clean forward pass (no mix), then apply the mix on the adv input
        for the outer-min step. This matches Pang's "Mixup-AT" practice and
        Lee 2020's adversarial-vertex-mixup setup.
    (2) Hidden-space mixing can *gradient-mask*: gradients of the (mixed)
        loss w.r.t. the input become small because two samples'
        representations cancel. We MUST run a transfer-attack control --
        if PGD-AT baseline adversarials transfer with lower ASR than
        white-box, that is a masking signature (Athalye 2018; Tramer 2020).

Conditions (single seed; all from-scratch retrain to keep config consistent):
  1. PGD-AT baseline                       (no mixup)
  2. Manifold-Mixup standalone (no AT)     alpha=0.4, random layer in {1,2,3}
  3. Manifold-Mixup + AT, mix at layer 1   alpha=0.4
  4. Manifold-Mixup + AT, mix at layer 2   alpha=0.4
  5. Manifold-Mixup + AT, mix at layer 3   alpha=0.4
  6. Manifold-Mixup + AT, random layer     alpha=0.2
  7. Manifold-Mixup + AT, random layer     alpha=0.4
  8. Manifold-Mixup + AT, random layer     alpha=1.0
  (Standard-trained model used as the SURROGATE for transfer-attack control.)

Standard config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128,
SGD(mom=0.9, wd=5e-4), SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.

Outputs:
  results/fashion_mnist/h455_manifold_mixup_at_output.txt  (ASCII).
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
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h455_manifold_mixup_at_output.txt"
)


# ---------------------------------------------------------------------------
# Manifold-Mixup-capable SmallCNN clone.
# Mirrors C.SmallCNN; exposes a forward with an optional mix(layer, perm, lam).
#   layer = 0  : mix in INPUT space (== vanilla mixup; included for sanity)
#   layer = 1  : mix after block 0 (post pool 1)   -- shallow features
#   layer = 2  : mix after block 1 (post pool 2)   -- mid features
#   layer = 3  : mix after block 2 (post pool 3)   -- deep features
#   layer = None : no mixing (eval / attack-time / unmixed forward)
# ---------------------------------------------------------------------------
class MixCNN(nn.Module):
    def __init__(self, in_ch=1, size=28, n_classes=10, width=32):
        super().__init__()
        w = width
        self.block0 = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, padding=1), nn.BatchNorm2d(w),
            nn.ReLU(), nn.MaxPool2d(2))
        self.block1 = nn.Sequential(
            nn.Conv2d(w, w * 2, 3, padding=1), nn.BatchNorm2d(w * 2),
            nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(w * 2, w * 4, 3, padding=1), nn.BatchNorm2d(w * 4),
            nn.ReLU(), nn.MaxPool2d(2))
        feat = size // 8
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 4 * feat * feat, 256), nn.ReLU(),
            nn.Linear(256, n_classes))

    def forward(self, x, mix_layer=None, perm=None, lam=1.0):
        if mix_layer == 0 and perm is not None:
            x = lam * x + (1 - lam) * x[perm]
        h = self.block0(x)
        if mix_layer == 1 and perm is not None:
            h = lam * h + (1 - lam) * h[perm]
        h = self.block1(h)
        if mix_layer == 2 and perm is not None:
            h = lam * h + (1 - lam) * h[perm]
        h = self.block2(h)
        if mix_layer == 3 and perm is not None:
            h = lam * h + (1 - lam) * h[perm]
        return self.head(h)


def build_mix_cnn(seed):
    C.set_seed(seed)
    return MixCNN(in_ch=1, size=28, n_classes=10, width=32).to(C.DEVICE)


def build_std_cnn(seed):
    """A vanilla SmallCNN from common (for the standard surrogate in transfer)."""
    C.set_seed(seed)
    return C.build_model("cnn", META, width=32, seed=seed)


def make_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


# ---------------------------------------------------------------------------
# PGD against a MixCNN: ALWAYS attack the clean (unmixed) forward.
# We pass mix_layer=None inside the inner-max so the model is deterministic.
# ---------------------------------------------------------------------------
def pgd_unmixed(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA,
                random_start=True):
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        xa = xa + torch.empty_like(xa).uniform_(-eps, eps)
        xa = xa.clamp(0, 1)
    was_training = model.training
    model.eval()  # freeze BN running stats during attack
    for _ in range(steps):
        xa.requires_grad_(True)
        logits = model(xa, mix_layer=None)  # NO MIXING during attack
        loss = F.cross_entropy(logits, y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    if was_training:
        model.train()
    return xa.detach()


def fgsm_unmixed(model, x, y, eps=EPS):
    was_training = model.training
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x, mix_layer=None), y)
    g, = torch.autograd.grad(loss, x)
    xa = (x + eps * g.sign()).clamp(0, 1).detach()
    if was_training:
        model.train()
    return xa


# ---------------------------------------------------------------------------
# Training loop.
#   mode : "std"    standard training, no AT, no mixup
#          "pgdat"  PGD-AT, no mixup
#          "mm"     Manifold-Mixup, no AT
#          "mmat"   Manifold-Mixup + PGD-AT (the H455 condition)
#   mix_layer_policy : int in {0,1,2,3}, or "rand123" (random in {1,2,3})
#   alpha : Beta concentration parameter
# ---------------------------------------------------------------------------
def train(model, Xtr, Ytr, mode, mix_layer_policy=None, alpha=0.4, seed=SEED):
    rng = np.random.RandomState(seed)
    opt = make_optimizer(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n - BATCH, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            # ---- inner-max (AT step) on the UNMIXED forward -----------
            if mode in ("pgdat", "mmat"):
                xb = pgd_unmixed(model, xb, yb, eps=EPS, steps=PGD_STEPS,
                                 alpha=PGD_ALPHA)
                model.train()  # back to train mode after attack

            opt.zero_grad()

            if mode in ("std", "pgdat"):
                out = model(xb, mix_layer=None)
                loss = F.cross_entropy(out, yb)
            else:
                # Manifold-Mixup (optionally on adversarial input).
                if isinstance(mix_layer_policy, int):
                    L = mix_layer_policy
                else:  # "rand123"
                    L = int(rng.choice([1, 2, 3]))
                lam = float(rng.beta(alpha, alpha))
                bperm = torch.randperm(xb.size(0), device=xb.device)
                out = model(xb, mix_layer=L, perm=bperm, lam=lam)
                ya, yb2 = yb, yb[bperm]
                loss = (lam * F.cross_entropy(out, ya)
                        + (1.0 - lam) * F.cross_entropy(out, yb2))

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation on a MixCNN (always with mix_layer=None at eval).
# ---------------------------------------------------------------------------
@torch.no_grad()
def clean_acc_mix(model, X, Y, batch=512):
    model.eval()
    correct = 0
    for i in range(0, X.size(0), batch):
        logits = model(X[i:i + batch], mix_layer=None)
        correct += int((logits.argmax(1) == Y[i:i + batch]).sum().item())
    return correct / X.size(0)


def eval_full(model, Xte, Yte):
    """clean_acc, FGSM ASR, PGD ASR, mean margin (unmixed forward throughout)."""
    model.eval()
    acc = clean_acc_mix(model, Xte, Yte)

    # FGSM ASR
    fgsm_flip, fgsm_corr = [], []
    for i in range(0, Xte.size(0), 256):
        x, y = Xte[i:i + 256], Yte[i:i + 256]
        with torch.no_grad():
            corr = model(x, mix_layer=None).argmax(1) == y
        xa = fgsm_unmixed(model, x, y, eps=EPS)
        with torch.no_grad():
            flip = model(xa, mix_layer=None).argmax(1) != y
        fgsm_flip.append(flip.cpu()); fgsm_corr.append(corr.cpu())
    fgsm_flip = torch.cat(fgsm_flip).numpy()
    fgsm_corr = torch.cat(fgsm_corr).numpy().astype(bool)
    fgsm_asr = float(fgsm_flip[fgsm_corr].mean()) if fgsm_corr.sum() else float("nan")

    # PGD ASR
    pgd_flip, pgd_corr = [], []
    for i in range(0, Xte.size(0), 256):
        x, y = Xte[i:i + 256], Yte[i:i + 256]
        with torch.no_grad():
            corr = model(x, mix_layer=None).argmax(1) == y
        xa = pgd_unmixed(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
        with torch.no_grad():
            flip = model(xa, mix_layer=None).argmax(1) != y
        pgd_flip.append(flip.cpu()); pgd_corr.append(corr.cpu())
    pgd_flip = torch.cat(pgd_flip).numpy()
    pgd_corr = torch.cat(pgd_corr).numpy().astype(bool)
    pgd_asr = float(pgd_flip[pgd_corr].mean()) if pgd_corr.sum() else float("nan")

    # Mean margin (on Xte, true labels)
    parts = []
    with torch.no_grad():
        for i in range(0, Xte.size(0), 512):
            parts.append(model(Xte[i:i + 512], mix_layer=None).cpu())
    logits = torch.cat(parts)
    margin = float(np.mean(C.margin_of(logits, Yte)))
    return dict(clean_acc=acc, fgsm_asr=fgsm_asr, pgd_asr=pgd_asr,
                mean_margin=margin)


# ---------------------------------------------------------------------------
# Transfer-attack masking check.
# Generate PGD advs on a STANDARD surrogate; measure ASR on each defended
# model. If white-box ASR is *much* lower than transfer ASR, the defended
# model is genuinely robust. If white-box ASR is much HIGHER than transfer
# (i.e., transfer < white-box), still informative. The clear masking signal
# is white-box << transfer (Athalye 2018): adversaries built against a
# standard model are MORE effective than the white-box attack, indicating
# gradient masking on the defended model.
# ---------------------------------------------------------------------------
def build_transfer_advs(surrogate_std, Xte, Yte):
    """Generate PGD-10 adversarials against the standard surrogate."""
    surrogate_std.eval()
    advs = []
    for i in range(0, Xte.size(0), 256):
        x, y = Xte[i:i + 256], Yte[i:i + 256]
        xa = C.pgd(surrogate_std, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
        advs.append(xa.detach())
    return torch.cat(advs, dim=0)


def transfer_asr(target_model, X_adv, Xte, Yte):
    """ASR of `X_adv` evaluated on target_model (restricted to originally-correct)."""
    target_model.eval()
    flips, corr = [], []
    for i in range(0, Xte.size(0), 256):
        x_clean, y = Xte[i:i + 256], Yte[i:i + 256]
        xa = X_adv[i:i + 256]
        with torch.no_grad():
            c = target_model(x_clean, mix_layer=None).argmax(1) == y \
                if isinstance(target_model, MixCNN) \
                else target_model(x_clean).argmax(1) == y
            f = target_model(xa, mix_layer=None).argmax(1) != y \
                if isinstance(target_model, MixCNN) \
                else target_model(xa).argmax(1) != y
        flips.append(f.cpu()); corr.append(c.cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() else float("nan")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
CONDITIONS = [
    # (label, mode, mix_layer_policy, alpha)
    ("PGD-AT baseline",                    "pgdat", None,     None),
    ("ManifoldMixup (no AT, rand layer)",  "mm",    "rand123", 0.4),
    ("MM+AT, layer=1, alpha=0.4",          "mmat",  1,         0.4),
    ("MM+AT, layer=2, alpha=0.4",          "mmat",  2,         0.4),
    ("MM+AT, layer=3, alpha=0.4",          "mmat",  3,         0.4),
    ("MM+AT, rand layer, alpha=0.2",       "mmat",  "rand123", 0.2),
    ("MM+AT, rand layer, alpha=0.4",       "mmat",  "rand123", 0.4),
    ("MM+AT, rand layer, alpha=1.0",       "mmat",  "rand123", 1.0),
]


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H455  Manifold-Mixup-AT  (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        device = {C.DEVICE}")
    out("")
    out("Attack policy: PGD inner-max ALWAYS runs on the UNMIXED forward")
    out("(model.eval(), mix_layer=None). Mixup is applied only to the outer-min")
    out("training step. At test time the model is deterministic so the input-")
    out("space PGD bound is meaningful.")
    out("")
    out("Conditions:")
    for lbl, mode, mlp, a in CONDITIONS:
        out(f"  - {lbl:<40} mode={mode}  mix_layer={mlp}  alpha={a}")
    out("")
    flush_file()

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN,
                                        n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")
    flush_file()

    # ---- standard surrogate (for transfer-attack masking control) ----
    out("[A] training STANDARD surrogate (for transfer-attack control)...")
    surrogate = build_std_cnn(SEED)
    opt = make_optimizer(surrogate, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    surrogate.train()
    for ep in range(EPOCHS):
        p = torch.randperm(Xtr.size(0), device=Xtr.device)
        for i in range(0, Xtr.size(0) - BATCH, BATCH):
            idx = p[i:i + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(surrogate(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
    surrogate.eval()
    _, s_acc = C.logits_and_acc(surrogate, Xte, Yte)
    s_pgd = C.attack_success(surrogate, Xte, Yte, attack="pgd",
                             eps=EPS, steps=PGD_STEPS)["asr"]
    out(f"    surrogate: clean_acc={s_acc:.4f}  PGD_ASR(self)={s_pgd:.4f}")
    out("    building transfer adversarials (PGD-10 against surrogate)...")
    X_adv_transfer = build_transfer_advs(surrogate, Xte, Yte)
    out(f"    transfer adv set: {tuple(X_adv_transfer.shape)}  "
        f"({time.time()-t0:.0f}s)")
    out("")
    flush_file()

    # ---- run conditions ----
    rows = []
    for ci, (lbl, mode, mlp, a) in enumerate(CONDITIONS):
        out("-" * 80)
        out(f"[B.{ci+1}] {lbl}")
        out("-" * 80)
        ct0 = time.time()
        model = build_mix_cnn(SEED)
        train(model, Xtr, Ytr, mode=mode, mix_layer_policy=mlp, alpha=a or 0.4,
              seed=SEED)
        m = eval_full(model, Xte, Yte)
        t_asr = transfer_asr(model, X_adv_transfer, Xte, Yte)
        m["transfer_asr"] = t_asr
        m["label"] = lbl
        m["time_s"] = time.time() - ct0
        rows.append(m)
        out(f"    clean_acc={m['clean_acc']:.4f}  FGSM_ASR={m['fgsm_asr']:.4f}  "
            f"PGD_ASR={m['pgd_asr']:.4f}  margin={m['mean_margin']:.4f}")
        out(f"    transfer_PGD_ASR(from std surrogate) = {t_asr:.4f}")
        # masking diagnostic
        gap = t_asr - m["pgd_asr"]
        if gap > 0.05:
            tag = "MASKING-SUSPECT  (transfer > white-box by >0.05)"
        elif gap < -0.05:
            tag = "genuine          (white-box > transfer; expected)"
        else:
            tag = "ambiguous        (|transfer - white-box| <= 0.05)"
        out(f"    masking-check: transfer-WB = {gap:+.4f}  -> {tag}")
        out(f"    time={m['time_s']:.1f}s   (total {time.time()-t0:.0f}s)")
        out("")
        flush_file()

    # ---- summary table ----
    out("=" * 80)
    out("[C] MAIN TABLE")
    out("=" * 80)
    hdr = ("{:<40} {:>9} {:>9} {:>9} {:>9} {:>10} {:>8}".format(
        "condition", "clean", "FGSM_ASR", "PGD_ASR", "TR_ASR",
        "margin", "time_s"))
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<40} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>10.4f} {:>8.1f}".format(
            r["label"], r["clean_acc"], r["fgsm_asr"], r["pgd_asr"],
            r["transfer_asr"], r["mean_margin"], r["time_s"]))
    out("-" * len(hdr))
    out("Legend: TR_ASR = ASR of transfer adversarials built on the standard")
    out("        surrogate; if TR_ASR >> PGD_ASR -> gradient masking suspected.")
    out("")

    # ---- verdict ----
    out("=" * 80)
    out("[D] VERDICT")
    out("=" * 80)
    base = rows[0]                                # PGD-AT baseline
    mm_only = rows[1]                             # MM standalone
    at_mmat = [r for r in rows if r["label"].startswith("MM+AT")]

    best_mmat = min(at_mmat, key=lambda r: r["pgd_asr"])
    gain_vs_at = base["pgd_asr"] - best_mmat["pgd_asr"]
    acc_drop = best_mmat["clean_acc"] - base["clean_acc"]
    masking_best = best_mmat["transfer_asr"] - best_mmat["pgd_asr"]

    out(f"  PGD-AT baseline:           clean={base['clean_acc']:.4f}  "
        f"PGD_ASR={base['pgd_asr']:.4f}")
    out(f"  ManifoldMixup-only:        clean={mm_only['clean_acc']:.4f}  "
        f"PGD_ASR={mm_only['pgd_asr']:.4f}  "
        f"(masking-gap {mm_only['transfer_asr']-mm_only['pgd_asr']:+.4f})")
    out(f"  best MM+AT condition:      {best_mmat['label']}")
    out(f"      clean={best_mmat['clean_acc']:.4f}  "
        f"PGD_ASR={best_mmat['pgd_asr']:.4f}  "
        f"TR_ASR={best_mmat['transfer_asr']:.4f}  "
        f"masking-gap={masking_best:+.4f}")
    out(f"  PGD gain vs PGD-AT:        {gain_vs_at:+.4f}  "
        f"(positive => MM+AT MORE robust)")
    out(f"  clean-acc change vs PGD-AT:{acc_drop:+.4f}")

    # one-line verdict
    if masking_best > 0.05:
        one = ("MASKING: best MM+AT shows transfer ASR materially above its "
               "white-box PGD ASR; robustness gain may be obfuscated gradients, "
               "not a real defence.")
    elif gain_vs_at > 0.02 and acc_drop >= -0.02:
        one = ("YES: Manifold-Mixup + AT improves PGD robustness over PGD-AT "
               "without losing clean accuracy, and passes the transfer-attack "
               "masking check.")
    elif gain_vs_at > 0.02 and acc_drop < -0.02:
        one = ("PARTIAL: MM+AT improves PGD robustness but pays a clean-accuracy "
               "cost (>0.02). Tradeoff condition met.")
    elif abs(gain_vs_at) <= 0.02:
        one = ("NULL: MM+AT ties PGD-AT within +/-0.02 PGD ASR. No evidence the "
               "hidden-layer mix point adds robustness beyond AT itself.")
    else:
        one = ("NEGATIVE: MM+AT is WORSE than PGD-AT on PGD ASR. Mixing at hidden "
               "layers harms the AT objective at this scale.")
    out("  ONE-LINE VERDICT: " + one)
    out("")
    out(f"done in {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
