"""
H429 - Focal loss as an implicit adversarial defense.

Lin et al. 2017 (RetinaNet) proposed focal loss to down-weight easy examples:
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
where p_t is the model's probability for the true class and gamma >= 0.
gamma=0 recovers standard cross-entropy.

Hypothesis: boundary / hard samples have LOW p_t, so (1-p_t)^gamma -> 1 and
they keep their full gradient signal.  Easy interior samples have HIGH p_t, so
(1-p_t)^gamma -> 0 and they are discounted.  This implicit up-weighting of
boundary samples may act as soft margin-maximization, reducing adversarial
attack success (per the H404 finding that boundary samples are the adversarially
vulnerable ones).

Design:
  - Sweep gamma in {0, 1, 2, 5}.  gamma=0 is the CE baseline.
  - Each condition: fresh CNN (width=32), trained from scratch.
  - Eval: clean acc, FGSM ASR, PGD ASR (eps=0.1).
  - Also report margin statistics (mean, std) as a proxy for margin-maximization.
  - Print partial results after each gamma; flush to file after each.

Config: N_TRAIN=6000, N_EVAL=2000, EPOCHS=10, LR=0.05, BATCH=128,
        SGD mom=0.9 wd=5e-4, SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config ------------------------------------------------------------------
DS        = "fashion_mnist"
N_TRAIN   = 6000
N_EVAL    = 2000
EPOCHS    = 10
LR        = 0.05
BATCH     = 128
SEED      = 0
EPS       = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
GAMMAS    = [0, 1, 2, 5]   # focal loss focusing parameter

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h429_focal_loss_defense_output.txt",
)


# ---- focal loss --------------------------------------------------------------
def focal_loss(logits, targets, gamma: float):
    """Focal loss: FL = -(1-p_t)^gamma * log(p_t).
    gamma=0 -> standard cross-entropy."""
    if gamma == 0:
        return F.cross_entropy(logits, targets)
    log_p    = F.log_softmax(logits, dim=1)                    # (B, C)
    p_t      = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
    weight   = (1.0 - p_t).pow(gamma)
    loss     = -(weight * log_p.gather(1, targets.unsqueeze(1)).squeeze(1)).mean()
    return loss


# ---- training ----------------------------------------------------------------
def train_focal(Xtr, Ytr, gamma: float, seed: int):
    """Train a fresh CNN with focal loss (gamma). Returns eval-mode model."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt   = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=5e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n     = Xtr.size(0)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            focal_loss(model(xb), yb, gamma).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- eval --------------------------------------------------------------------
def eval_condition(model, Xte, Yte):
    """Returns (clean_acc, fgsm_asr, pgd_asr, mean_margin, std_margin)."""
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fgsm   = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)["asr"]
    pgd    = C.attack_success(model, Xte, Yte, attack="pgd",
                              eps=EPS, steps=PGD_STEPS)["asr"]
    mg     = C.margin(model, Xte, Yte)   # numpy (N,)
    return acc, fgsm, pgd, float(mg.mean()), float(mg.std())


# ---- main --------------------------------------------------------------------
def main():
    t0    = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H429  Focal loss as implicit adversarial defense (gamma sweep)")
    out("=" * 80)
    out(f"config: DS={DS} N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} "
        f"LR={LR} BATCH={BATCH}")
    out(f"        SGD(mom=0.9,wd=5e-4) SEED={SEED} EPS={EPS} "
        f"PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        GAMMAS={GAMMAS}  device={C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    rows = []
    for ci, gamma in enumerate(GAMMAS):
        label = f"gamma={gamma}" + ("  [CE baseline]" if gamma == 0 else "")
        out(f"[{ci+1}/{len(GAMMAS)}] training with {label} ...")
        model = train_focal(Xtr, Ytr, gamma, SEED)
        acc, fgsm, pgd, mg_mean, mg_std = eval_condition(model, Xte, Yte)
        rows.append({
            "gamma": gamma, "acc": acc, "fgsm": fgsm, "pgd": pgd,
            "mg_mean": mg_mean, "mg_std": mg_std,
        })
        out(f"    clean_acc={acc:.4f}  FGSM_ASR={fgsm:.4f}  PGD_ASR={pgd:.4f}  "
            f"margin_mean={mg_mean:.4f}  margin_std={mg_std:.4f}  "
            f"({time.time()-t0:.0f}s)")
        flush_file()

    # ---- summary table -------------------------------------------------------
    out("")
    out("=" * 80)
    out("SUMMARY TABLE")
    out("=" * 80)
    hdr = "{:<10} {:>10} {:>10} {:>10} {:>12} {:>12}".format(
        "gamma", "clean_acc", "FGSM_ASR", "PGD_ASR", "margin_mean", "margin_std")
    out(hdr)
    out("-" * len(hdr))
    base = rows[0]
    for r in rows:
        out("{:<10} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.4f} {:>12.4f}".format(
            r["gamma"], r["acc"], r["fgsm"], r["pgd"], r["mg_mean"], r["mg_std"]))
    out("-" * len(hdr))
    out("")
    out("Delta vs CE baseline (gamma=0):")
    for r in rows[1:]:
        out("  gamma={}: d_PGD_ASR={:+.4f}  d_clean_acc={:+.4f}  "
            "d_margin_mean={:+.4f}".format(
                r["gamma"],
                r["pgd"]     - base["pgd"],
                r["acc"]     - base["acc"],
                r["mg_mean"] - base["mg_mean"],
            ))

    # ---- verdict -------------------------------------------------------------
    out("")
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)
    best = min(rows[1:], key=lambda r: r["pgd"])   # best focal (non-CE)
    d_pgd  = best["pgd"]     - base["pgd"]         # negative => more robust
    d_acc  = best["acc"]     - base["acc"]
    d_mg   = best["mg_mean"] - base["mg_mean"]
    acc_ok = best["acc"] >= base["acc"] - 0.02

    out(f"  best focal gamma={best['gamma']}: PGD_ASR {base['pgd']:.4f} -> "
        f"{best['pgd']:.4f} ({d_pgd:+.4f})")
    out(f"  clean acc change: {d_acc:+.4f}  ({'preserved' if acc_ok else 'dropped >0.02'})")
    out(f"  margin_mean change: {d_mg:+.4f}  "
        f"({'increased -> implicit margin-max' if d_mg > 0 else 'no margin-max signal'})")

    if d_pgd < -0.02 and acc_ok:
        verdict = ("YES: focal loss meaningfully reduces adversarial ASR while "
                   "preserving clean accuracy, consistent with implicit margin-maximization.")
    elif d_pgd < -0.02 and not acc_ok:
        verdict = ("PARTIAL: focal loss reduces ASR but at a clean-accuracy cost; "
                   "trade-off not favourable for a pure defense.")
    elif -0.02 <= d_pgd <= 0.02 and acc_ok:
        verdict = ("NEUTRAL: focal loss does not meaningfully change adversarial "
                   "robustness; focusing on boundary samples via gamma does not "
                   "translate to lower ASR at this scale.")
    else:
        verdict = ("NO: focal loss hurts both clean accuracy and robustness at "
                   "this scale; the implicit margin-maximization hypothesis is not supported.")

    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
