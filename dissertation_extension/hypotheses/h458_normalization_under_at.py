"""
H458 - GroupNorm vs BatchNorm under Adversarial Training (Fashion-MNIST).

Question (gap G4, architecture axes under AT):
  Xie & Yuille 2020 ("Intriguing Properties of Adversarial Training at Scale",
  ICLR'20) argued that BatchNorm is harmful under adversarial training: the
  clean and adversarial mini-batches form a two-domain mixture and BN's
  running statistics get pulled between them, producing a domain-mismatch
  that hurts robustness. Awais et al. 2021 ("Adversarially Robust Deep
  Learning with Optimal-Transport-Regularized Divergences" / a line of work
  on AT with norm-free or GN backbones) and Galloway et al. 2019 ("Batch
  Normalization is a Cause of Adversarial Vulnerability", ICML-W'19) both
  suggest BN is implicated in adversarial fragility, while sample-wise norms
  (GroupNorm, LayerNorm) or no-norm avoid the running-stats domain mixture.

  Does this transfer to a small CNN on Fashion-MNIST? Specifically: at matched
  compute / init / optimiser, does swapping BN -> GN, LN, or None recover
  robustness under PGD-AT relative to BN+AT?

Hypothesis:
  Under PGD adversarial training, models with sample-wise normalisation
  (GroupNorm / LayerNorm) or no normalisation will achieve LOWER robust ASR
  than the BatchNorm counterpart, while standard CE training will show the
  opposite or no difference (BN helps clean accuracy). The BN-AT model will
  additionally exhibit a measurable divergence between clean and adversarial
  running statistics (a diagnostic for the domain-mixture problem).

Design (4 norms x 2 training regimes = 8 runs, paired seeds):
  Norms      : "none", "bn", "gn", "ln"  (all inserted at the same locations
               in a width=32 SmallCNN-style net; matched param count modulo
               BN/GN/LN affine parameters).
  Regimes    : "ce" (standard CE) and "at" (PGD-AT with EPS=0.1, 7 inner
               steps as in common.train_model adv_train path).
  Metrics    : clean_acc, FGSM_ASR, PGD_ASR (eps=0.1, steps=10, alpha=0.01).
  Diagnostic : for BN-AT, after training, recompute BN running stats on
               (a) clean train batch and (b) PGD adv train batch with
               model.train(); report L2 distance between the two means and
               variances summed across all BN layers (BN train-stats
               divergence).
  Controls   : identical seed, identical optimiser (SGD mom=0.9 wd=5e-4),
               identical LR/epochs/batch, identical width. The only knob
               that varies is the normalisation layer (and the AT switch).

Config: N_TRAIN=6000 N_EVAL=2000 EPOCHS=10 LR=0.05 BATCH=128
        SGD(mom=0.9, wd=5e-4) SEED=0 EPS=0.1 PGD_STEPS=10 PGD_ALPHA=0.01.
        AT inner: PGD-7, alpha=2.5*eps/7, random start.

Output: results/fashion_mnist/h458_normalization_under_at_output.txt  (ASCII).
This script DOES NOT execute on import; run via .venv/bin/python.
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
AT_INNER_STEPS = 7
WIDTH = 32
GN_GROUPS = 8                    # GroupNorm groups (must divide channel counts)
NORMS = ["none", "bn", "gn", "ln"]
REGIMES = ["ce", "at"]

META = {"channels": 1, "size": 28, "n_classes": 10}


# ---- model with selectable normalisation ---------------------------------
class _LN2d(nn.Module):
    """LayerNorm over (C,H,W) per-sample (i.e. GroupNorm with 1 group).

    Functionally identical to nn.GroupNorm(1, C) and matches the standard
    convention used by ResNet-LN / ConvNeXt for 'LayerNorm' on conv feature
    maps.  Per-sample, fully spatial, no running stats.
    """
    def __init__(self, C):
        super().__init__()
        self.gn = nn.GroupNorm(1, C)

    def forward(self, x):
        return self.gn(x)


def _make_norm(kind, C_out):
    if kind == "none":
        return nn.Identity()
    if kind == "bn":
        return nn.BatchNorm2d(C_out)
    if kind == "gn":
        g = GN_GROUPS
        while C_out % g != 0 and g > 1:
            g //= 2
        return nn.GroupNorm(g, C_out)
    if kind == "ln":
        return _LN2d(C_out)
    raise ValueError(kind)


class NormCNN(nn.Module):
    """SmallCNN clone but the normalisation layer kind is configurable.

    Architecture matches campaign.common.SmallCNN at width=WIDTH so prior
    numbers in the campaign remain comparable. Only the norm layer inside
    each conv block changes.
    """
    def __init__(self, in_ch=1, size=28, n_classes=10, width=32, norm="bn"):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), _make_norm(norm, o),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(
            *block(in_ch, width),
            *block(width, width * 2),
            *block(width * 2, width * 4),
        )
        feat = size // 8
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * feat * feat, 256), nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def _make_optimizer_sgd(model, lr):
    """SGD mom=0.9 wd=5e-4 (config); common.make_optimizer uses wd=1e-4."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def train_one(norm, regime, Xtr, Ytr, seed):
    """Train a NormCNN from scratch.

    regime == "ce" : standard cross-entropy training.
    regime == "at" : PGD-AT inner loop (eps=EPS, steps=AT_INNER_STEPS).
    """
    C.set_seed(seed)
    model = NormCNN(META["channels"], META["size"], META["n_classes"],
                    width=WIDTH, norm=norm).to(C.DEVICE)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            if regime == "at":
                xb = C.pgd(model, xb, yb, eps=EPS, steps=AT_INNER_STEPS,
                           alpha=2.5 * EPS / AT_INNER_STEPS)
                # Note: this re-enters .train() implicitly because pgd uses
                # forward in train mode if model.train() is set. We keep the
                # BN running stats updated under adversarial inputs, which
                # IS exactly the Xie 2020 phenomenon we want to expose.
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_robustness(model, X, Y):
    """clean acc, FGSM ASR, PGD ASR on (X,Y)."""
    _, acc = C.logits_and_acc(model, X, Y)
    fg = C.attack_success(model, X, Y, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


def bn_stats_divergence(model, Xtr, Ytr, n_probe=512):
    """Diagnostic: how far apart are BN running stats when re-estimated on
    clean vs PGD-adversarial data?

    Procedure (Xie 2020-flavoured):
      1. Snapshot current BN running_mean / running_var per BN layer.
      2. Reset them; put model in .train() and forward a fresh clean batch
         with momentum=1.0 so running stats == that batch's stats.
      3. Restore snapshot. Now reset again and forward a PGD adv batch the
         same way.
      4. Return summed L2 distance between the two clean/adv stat sets.

    Returns float('nan') if the model has no BatchNorm2d layers.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if len(bns) == 0:
        return float("nan"), float("nan")
    # snapshot
    snap = [(b.running_mean.detach().clone(), b.running_var.detach().clone(),
             b.momentum) for b in bns]

    # pick a probe batch
    g = torch.Generator(device="cpu").manual_seed(0)
    idx = torch.randperm(Xtr.size(0), generator=g)[:n_probe].to(Xtr.device)
    xc, yc = Xtr[idx], Ytr[idx]
    xa = C.pgd(model, xc, yc, eps=EPS, steps=AT_INNER_STEPS,
               alpha=2.5 * EPS / AT_INNER_STEPS)

    def _refit_and_grab(xb):
        # reset + momentum=1 so one forward overwrites running stats with batch stats
        for b in bns:
            b.running_mean.zero_()
            b.running_var.fill_(1.0)
            b.momentum = 1.0
        model.train()
        with torch.no_grad():
            _ = model(xb)
        model.eval()
        return [(b.running_mean.detach().clone(),
                 b.running_var.detach().clone()) for b in bns]

    clean_stats = _refit_and_grab(xc)
    adv_stats = _refit_and_grab(xa)

    # restore snapshot
    for b, (rm, rv, mom) in zip(bns, snap):
        b.running_mean.copy_(rm)
        b.running_var.copy_(rv)
        b.momentum = mom

    d_mean = 0.0
    d_var = 0.0
    for (cm, cv), (am, av) in zip(clean_stats, adv_stats):
        d_mean += float(torch.linalg.vector_norm(cm - am).item())
        d_var += float(torch.linalg.vector_norm(cv - av).item())
    return d_mean, d_var


def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h458_normalization_under_at_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H458  GroupNorm vs BatchNorm under PGD-AT  (Fashion-MNIST, gap G4)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} "
        f"AT_INNER_STEPS={AT_INNER_STEPS} WIDTH={WIDTH} GN_GROUPS={GN_GROUPS}")
    out(f"        norms = {NORMS}     regimes = {REGIMES}")
    out(f"        device = {C.DEVICE}")
    out("")
    out("Prior art:")
    out("  Xie & Yuille 2020 (ICLR) 'Intriguing Properties of AT at Scale':")
    out("    BN running stats become a clean/adv mixture under AT and hurt robust acc.")
    out("  Galloway et al. 2019 (ICML-W) 'BatchNorm is a Cause of Adv Vulnerability':")
    out("    BN-trained nets show systematically higher gradient-attack success.")
    out("  Awais et al. 2021 / Benz et al. 2021: GN/LN or norm-free backbones avoid")
    out("    the two-domain running-stats failure mode.")
    out("Confound watch: BN-AT vs GN-AT only fair at matched init/opt/seed/compute.")
    out("")

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")
    flush_file()

    # ---- run the 4x2 grid ----
    rows = []
    bn_diag = {}
    for regime in REGIMES:
        out("-" * 80)
        out(f"[regime={regime}]")
        out("-" * 80)
        for norm in NORMS:
            tag = f"{norm}-{regime}"
            out(f"  training {tag} ...")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            t1 = time.time()
            model = train_one(norm, regime, Xtr, Ytr, SEED)
            train_s = time.time() - t1
            acc, fgsm, pgd = eval_robustness(model, Xte, Yte)
            out(f"    {tag}: clean_acc={acc:.4f}  FGSM_ASR={fgsm:.4f}  "
                f"PGD_ASR={pgd:.4f}  ({train_s:.1f}s)")
            row = {"norm": norm, "regime": regime, "acc": acc,
                   "fgsm": fgsm, "pgd": pgd, "train_s": train_s}
            if norm == "bn":
                d_mean, d_var = bn_stats_divergence(model, Xtr, Ytr)
                row["bn_d_mean"] = d_mean
                row["bn_d_var"] = d_var
                bn_diag[regime] = (d_mean, d_var)
                out(f"           BN-stats clean-vs-adv divergence: "
                    f"L2(d_mean)={d_mean:.4f}  L2(d_var)={d_var:.4f}")
            rows.append(row)
            flush_file()
        out("")

    # ---- MAIN TABLE ----
    out("=" * 80)
    out("[main table]  norm x regime")
    out("=" * 80)
    hdr = "{:<6} {:<7} {:>10} {:>10} {:>10}".format(
        "norm", "regime", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<6} {:<7} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["norm"], r["regime"], r["acc"], r["fgsm"], r["pgd"]))
    out("-" * len(hdr))
    out("")
    out("BN clean-vs-adv stats divergence (diagnostic):")
    for regime, (dm, dv) in bn_diag.items():
        out(f"  regime={regime}: L2(d_mean)={dm:.4f}  L2(d_var)={dv:.4f}")

    # ---- VERDICT ----
    out("")
    out("=" * 80)
    out("[verdict]")
    out("=" * 80)
    def find(norm, regime):
        for r in rows:
            if r["norm"] == norm and r["regime"] == regime:
                return r
        return None

    # robust PGD-ASR under AT for each norm
    at_rows = {r["norm"]: r for r in rows if r["regime"] == "at"}
    ce_rows = {r["norm"]: r for r in rows if r["regime"] == "ce"}
    bn_at_pgd = at_rows["bn"]["pgd"]
    best_norm = min(at_rows.items(), key=lambda kv: kv[1]["pgd"])[0]
    best_pgd = at_rows[best_norm]["pgd"]
    gain_vs_bn = bn_at_pgd - best_pgd                    # positive => better than BN-AT
    gn_vs_bn = bn_at_pgd - at_rows["gn"]["pgd"]
    ln_vs_bn = bn_at_pgd - at_rows["ln"]["pgd"]
    none_vs_bn = bn_at_pgd - at_rows["none"]["pgd"]

    out("Under PGD-AT (lower PGD_ASR = more robust):")
    out(f"  none-AT PGD_ASR = {at_rows['none']['pgd']:.4f}  (vs BN-AT: {none_vs_bn:+.4f})")
    out(f"  bn-AT   PGD_ASR = {at_rows['bn']['pgd']:.4f}  (reference)")
    out(f"  gn-AT   PGD_ASR = {at_rows['gn']['pgd']:.4f}  (vs BN-AT: {gn_vs_bn:+.4f})")
    out(f"  ln-AT   PGD_ASR = {at_rows['ln']['pgd']:.4f}  (vs BN-AT: {ln_vs_bn:+.4f})")
    out(f"  best norm under AT = {best_norm}  (PGD_ASR={best_pgd:.4f}, "
        f"gain vs BN-AT = {gain_vs_bn:+.4f})")

    out("Under standard CE training (sanity, BN usually helps clean acc):")
    out(f"  none-CE clean={ce_rows['none']['acc']:.4f}  PGD_ASR={ce_rows['none']['pgd']:.4f}")
    out(f"  bn-CE   clean={ce_rows['bn']['acc']:.4f}  PGD_ASR={ce_rows['bn']['pgd']:.4f}")
    out(f"  gn-CE   clean={ce_rows['gn']['acc']:.4f}  PGD_ASR={ce_rows['gn']['pgd']:.4f}")
    out(f"  ln-CE   clean={ce_rows['ln']['acc']:.4f}  PGD_ASR={ce_rows['ln']['pgd']:.4f}")

    if "at" in bn_diag:
        dm_at, dv_at = bn_diag["at"]
        dm_ce, dv_ce = bn_diag.get("ce", (float("nan"), float("nan")))
        out("")
        out(f"BN-stats divergence (clean vs PGD-adv mini-batch, post-training):")
        out(f"  CE-trained BN: L2(d_mean)={dm_ce:.4f}  L2(d_var)={dv_ce:.4f}")
        out(f"  AT-trained BN: L2(d_mean)={dm_at:.4f}  L2(d_var)={dv_at:.4f}")

    # one-line verdict
    sig = 0.02
    if gain_vs_bn > sig and best_norm != "bn":
        verdict = (f"YES: replacing BN with {best_norm} under PGD-AT cuts PGD_ASR "
                   f"by {gain_vs_bn:+.4f} on Fashion-MNIST -- consistent with "
                   f"Xie 2020.")
    elif gain_vs_bn > sig and best_norm == "bn":
        verdict = ("UNEXPECTED: BN remained the most robust under AT (gain test "
                   "is vacuous).")
    elif abs(gain_vs_bn) <= sig:
        verdict = ("NO: at matched compute/init on Fashion-MNIST, normalisation "
                   "choice under PGD-AT does not meaningfully shift PGD_ASR "
                   f"(best gain vs BN-AT = {gain_vs_bn:+.4f}, within {sig}).")
    else:
        verdict = ("MIXED: BN-AT outperformed alternatives -- counter to the "
                   "Xie 2020 prediction at this scale.")
    out("")
    out("ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
