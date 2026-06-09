"""
H415 - Fast Adversarial Training (Wong et al. 2020, arXiv:2001.03994)

Hypothesis: Single-step FGSM-AT with random initialisation (Fast-AT) and a
cyclic learning-rate schedule matches multi-step PGD-AT robustness at a
fraction of the compute cost — but is susceptible to "catastrophic overfitting"
(CO), where PGD robustness suddenly collapses to near-zero mid-training.

Gradient alignment (GA) — cosine similarity between the single-step FGSM
gradient and the true loss gradient — is monitored per epoch as an early
warning signal for CO.  Standard FGSM-AT (no random init) is included as a
CO-prone control.

Design:
  1. STD      — standard training (clean, no AT) — upper bound on clean acc.
  2. FGSM_AT  — single-step FGSM-AT, no random init, cosine-annealing LR.
  3. FAST_AT  — single-step FGSM-AT + uniform random init in [-eps,eps]
                + cyclic (triangular) LR  (Wong et al. 2020 recipe).
  4. PGD_AT10 — multi-step PGD-AT with 10 steps, cosine-annealing LR
                (the gold-standard baseline; expensive but reliable).

  All four trained on the same N_TRAIN=6000 Fashion-MNIST samples for
  EPOCHS=10 epochs, BATCH=128, SGD mom=0.9 wd=5e-4.

Metrics per epoch (for CO detection):
  - Training-set FGSM adversarial accuracy (proxy for AT success)
  - Gradient-alignment score (GA): mean cosine-sim between g_fgsm and g_clean

Final evaluation (test set):
  - Clean accuracy
  - FGSM ASR  (eps=0.1)
  - PGD-10 ASR (eps=0.1) — key robustness metric

Reference: Wong, Rice & Kolter (2020). "Fast is better than free: Revisiting
adversarial training." ICLR 2020. arXiv:2001.03994.

Config: N_TRAIN=6000 EPOCHS=10 BATCH=128 SEED=0 eps=0.1
Output: results/fashion_mnist/h415_fast_adversarial_training_output.txt
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

# ---- config -----------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10

# Cyclic LR params (triangular, one full cycle = EPOCHS epochs)
LR_MIN = 0.0
LR_MAX = 0.2        # peak LR for Fast-AT (Wong et al. use ~0.2 for CIFAR-10)
LR_STD = 0.05       # fixed peak for STD / FGSM_AT / PGD_AT10 (cosine annealing)

META = {"channels": 1, "size": 28, "n_classes": 10}

# ---- output -----------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(_REPO, "results", "fashion_mnist",
                        "h415_fast_adversarial_training_output.txt")


# ---- helpers ----------------------------------------------------------------

def _sgd(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def _cosine_lr(opt, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)


def _cyclic_lr(opt, epochs, steps_per_epoch):
    """Triangular cyclic LR that completes one full cycle over all epochs."""
    total_steps = epochs * steps_per_epoch
    return torch.optim.lr_scheduler.CyclicLR(
        opt,
        base_lr=LR_MIN,
        max_lr=LR_MAX,
        step_size_up=total_steps // 2,
        step_size_down=total_steps - total_steps // 2,
        cycle_momentum=False,
    )


def fgsm_rand(model, x, y, eps, random_init=True):
    """FGSM with optional uniform random initialisation."""
    x0 = x.clone().detach()
    if random_init:
        delta = torch.empty_like(x0).uniform_(-eps, eps)
        xa = (x0 + delta).clamp(0, 1).requires_grad_(True)
    else:
        xa = x0.clone().requires_grad_(True)
    loss = F.cross_entropy(model(xa), y)
    g, = torch.autograd.grad(loss, xa)
    xa_adv = (xa.detach() + eps * g.sign()).clamp(0, 1)
    # project back into eps-ball around x0
    xa_adv = torch.min(torch.max(xa_adv, x0 - eps), x0 + eps).clamp(0, 1)
    return xa_adv.detach(), g.detach()


def gradient_alignment(model, x, y, eps):
    """Cosine similarity between FGSM perturbation gradient and clean gradient.

    High GA (near 1) means the single-step approximation is accurate.
    GA collapsing toward 0 or going negative is an early CO warning sign.
    """
    model.eval()
    # clean gradient
    xc = x.clone().detach().requires_grad_(True)
    lc = F.cross_entropy(model(xc), y)
    gc, = torch.autograd.grad(lc, xc)

    # FGSM-rand gradient
    xr = x.clone().detach()
    delta = torch.empty_like(xr).uniform_(-eps, eps)
    xr = (xr + delta).clamp(0, 1).requires_grad_(True)
    lr_ = F.cross_entropy(model(xr), y)
    gr, = torch.autograd.grad(lr_, xr)

    gc_flat = gc.view(gc.size(0), -1)
    gr_flat = gr.view(gr.size(0), -1)
    cos = F.cosine_similarity(gc_flat, gr_flat, dim=1).mean().item()
    model.train()
    return cos


def train_epoch_adv_ga(model, Xtr, Ytr, opt, sched, mode, cyclic):
    """One training epoch; returns (mean_loss, adv_train_acc, grad_align).

    mode: "std" | "fgsm_at" | "fast_at" | "pgd_at10"
    cyclic: True if the scheduler is CyclicLR (step per batch), else step per epoch.
    grad_align measured only for fgsm-based modes on a small batch sample.
    """
    model.train()
    n = Xtr.size(0)
    perm = torch.randperm(n, device=Xtr.device)
    total_loss, total_corr, total_samples = 0.0, 0, 0
    ga_vals = []

    for i in range(0, n, BATCH):
        idx = perm[i: i + BATCH]
        xb, yb = Xtr[idx], Ytr[idx]

        if mode == "std":
            xadv = xb
        elif mode == "fgsm_at":
            xadv, g_fgsm = fgsm_rand(model, xb, yb, EPS, random_init=False)
            # GA: compare with fresh clean gradient
            if i == 0:  # measure once per epoch (cheap)
                ga_vals.append(gradient_alignment(model, xb, yb, EPS))
        elif mode == "fast_at":
            xadv, g_fgsm = fgsm_rand(model, xb, yb, EPS, random_init=True)
            if i == 0:
                ga_vals.append(gradient_alignment(model, xb, yb, EPS))
        elif mode == "pgd_at10":
            xadv = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS,
                         alpha=2.5 * EPS / PGD_STEPS, random_start=True)

        opt.zero_grad()
        out = model(xadv)
        loss = F.cross_entropy(out, yb)
        loss.backward()
        opt.step()
        if cyclic:
            sched.step()   # CyclicLR steps per batch

        total_loss += loss.item() * yb.size(0)
        total_corr += (out.detach().argmax(1) == yb).sum().item()
        total_samples += yb.size(0)

    mean_ga = float(np.mean(ga_vals)) if ga_vals else float("nan")
    return (total_loss / total_samples,
            total_corr / total_samples,
            mean_ga)


def train_condition(name, mode, Xtr, Ytr, seed):
    """Train a fresh SmallCNN for EPOCHS epochs under `mode`.

    Returns (model, epoch_log) where epoch_log is list of dicts per epoch.
    """
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32)
    n_steps = (N_TRAIN + BATCH - 1) // BATCH

    if mode == "fast_at":
        opt = _sgd(model, LR_MAX)   # CyclicLR controls actual rate
        sched = _cyclic_lr(opt, EPOCHS, n_steps)
        cyclic = True
    else:
        opt = _sgd(model, LR_STD)
        sched = _cosine_lr(opt, EPOCHS)
        cyclic = False

    epoch_log = []
    for ep in range(EPOCHS):
        loss, tr_acc, ga = train_epoch_adv_ga(
            model, Xtr, Ytr, opt, sched, mode, cyclic)
        if not cyclic:
            sched.step()
        current_lr = opt.param_groups[0]["lr"]
        epoch_log.append({
            "ep": ep + 1,
            "loss": loss,
            "tr_acc": tr_acc,
            "ga": ga,
            "lr": current_lr,
        })

    model.eval()
    return model, epoch_log


def eval_model(model, Xte, Yte):
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    fgsm_res = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pgd_res = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS,
                               steps=PGD_STEPS)
    return clean_acc, fgsm_res["asr"], pgd_res["asr"]


# ---- catastrophic overfitting detection -------------------------------------

def detect_co(epoch_log):
    """Simple CO detector: flag if PGD-era proxy (train-adv-acc) drops >0.15
    in a single epoch after epoch 3, OR if GA drops below 0.05.

    Returns list of (epoch, reason) tuples.
    """
    flags = []
    for i in range(1, len(epoch_log)):
        prev, curr = epoch_log[i - 1], epoch_log[i]
        if curr["ep"] > 3:
            drop = prev["tr_acc"] - curr["tr_acc"]
            if drop > 0.15:
                flags.append((curr["ep"],
                               f"adv-train-acc drop {drop:.3f} (>{0.15})"))
        if not np.isnan(curr["ga"]) and curr["ga"] < 0.05:
            flags.append((curr["ep"],
                           f"gradient-alignment {curr['ga']:.4f} < 0.05"))
    return flags


# ---- main -------------------------------------------------------------------

def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H415  Fast Adversarial Training — Wong et al. 2020 (arXiv:2001.03994)")
    out("=" * 80)
    out(f"config: DS={DS}  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}  EPOCHS={EPOCHS}")
    out(f"        BATCH={BATCH}  SEED={SEED}  EPS={EPS}  PGD_STEPS={PGD_STEPS}")
    out(f"        LR_STD={LR_STD} (cosine)  LR_MAX={LR_MAX} (cyclic, Fast-AT)")
    out(f"        device={C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                         seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    conditions = [
        ("STD",       "std",      "Standard training (no AT)"),
        ("FGSM_AT",   "fgsm_at",  "FGSM-AT, no random init, cosine LR"),
        ("FAST_AT",   "fast_at",  "Fast-AT: FGSM + random init + cyclic LR"),
        ("PGD_AT10",  "pgd_at10", "PGD-AT-10: multi-step, cosine LR (baseline)"),
    ]

    results = {}

    for cname, mode, desc in conditions:
        out("─" * 80)
        out(f"[{cname}]  {desc}")
        out("─" * 80)
        tc = time.time()
        model, epoch_log = train_condition(mode, mode, Xtr, Ytr, SEED)

        # epoch table
        hdr = f"  {'ep':>3}  {'loss':>7}  {'tr_adv_acc':>10}  {'GA':>7}  {'lr':>8}"
        out(hdr)
        out("  " + "-" * (len(hdr) - 2))
        for row in epoch_log:
            ga_str = f"{row['ga']:7.4f}" if not np.isnan(row["ga"]) else "    n/a"
            out(f"  {row['ep']:>3}  {row['loss']:7.4f}  {row['tr_acc']:10.4f}"
                f"  {ga_str}  {row['lr']:8.6f}")

        # CO detection
        co_flags = detect_co(epoch_log)
        if co_flags:
            out(f"  *** CATASTROPHIC OVERFITTING SIGNALS ({len(co_flags)}): ***")
            for ep, reason in co_flags:
                out(f"      epoch {ep}: {reason}")
        else:
            out("  (no catastrophic-overfitting signals detected)")

        # eval
        clean_acc, fgsm_asr, pgd_asr = eval_model(model, Xte, Yte)
        results[cname] = {
            "clean_acc": clean_acc,
            "fgsm_asr": fgsm_asr,
            "pgd_asr": pgd_asr,
            "co_flags": co_flags,
            "epoch_log": epoch_log,
        }
        elapsed = time.time() - tc
        out(f"  => clean_acc={clean_acc:.4f}  FGSM_ASR={fgsm_asr:.4f}"
            f"  PGD_ASR={pgd_asr:.4f}  ({elapsed:.0f}s)")
        out("")
        flush_file()

    # ---- summary table -------------------------------------------------------
    out("=" * 80)
    out("SUMMARY TABLE")
    out("=" * 80)
    hdr = ("{:<12} {:>10} {:>10} {:>10} {:>6}".format(
        "Condition", "clean_acc", "FGSM_ASR", "PGD_ASR", "CO?"))
    out(hdr)
    out("-" * len(hdr))
    for cname, _, _ in conditions:
        r = results[cname]
        co = "YES" if r["co_flags"] else "no"
        out("{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>6}".format(
            cname, r["clean_acc"], r["fgsm_asr"], r["pgd_asr"], co))
    out("-" * len(hdr))

    # ---- gradient alignment summary -----------------------------------------
    out("")
    out("GRADIENT ALIGNMENT (mean cosine-sim per epoch, FGSM-based methods):")
    for cname in ("FGSM_AT", "FAST_AT"):
        r = results[cname]
        ga_vals = [row["ga"] for row in r["epoch_log"] if not np.isnan(row["ga"])]
        if ga_vals:
            out(f"  {cname}: min={min(ga_vals):.4f}  mean={np.mean(ga_vals):.4f}"
                f"  max={max(ga_vals):.4f}  final={ga_vals[-1]:.4f}")

    # ---- analysis & verdict --------------------------------------------------
    out("")
    out("=" * 80)
    out("ANALYSIS")
    out("=" * 80)

    std   = results["STD"]
    fgsm  = results["FGSM_AT"]
    fast  = results["FAST_AT"]
    pgd10 = results["PGD_AT10"]

    clean_cost_fast = std["clean_acc"] - fast["clean_acc"]
    clean_cost_pgd  = std["clean_acc"] - pgd10["clean_acc"]
    rob_gap_fgsm    = pgd10["pgd_asr"] - fgsm["pgd_asr"]   # + => fgsm worse
    rob_gap_fast    = pgd10["pgd_asr"] - fast["pgd_asr"]   # + => fast worse
    fast_vs_fgsm    = fgsm["pgd_asr"] - fast["pgd_asr"]    # + => fast better

    out(f"  STD    clean={std['clean_acc']:.4f}  PGD_ASR={std['pgd_asr']:.4f}")
    out(f"  FGSM_AT clean={fgsm['clean_acc']:.4f}  PGD_ASR={fgsm['pgd_asr']:.4f}"
        f"  (CO: {'YES' if fgsm['co_flags'] else 'no'})")
    out(f"  FAST_AT clean={fast['clean_acc']:.4f}  PGD_ASR={fast['pgd_asr']:.4f}"
        f"  (CO: {'YES' if fast['co_flags'] else 'no'})")
    out(f"  PGD10   clean={pgd10['clean_acc']:.4f}  PGD_ASR={pgd10['pgd_asr']:.4f}")
    out("")
    out(f"  Fast-AT vs FGSM-AT PGD_ASR improvement : {fast_vs_fgsm:+.4f}"
        f"  (positive => Fast-AT more robust)")
    out(f"  Fast-AT vs PGD-10  PGD_ASR gap         : {rob_gap_fast:+.4f}"
        f"  (positive => Fast-AT worse)")
    out(f"  Clean-acc cost Fast-AT vs STD           : {clean_cost_fast:+.4f}")
    out(f"  Clean-acc cost PGD-10  vs STD           : {clean_cost_pgd:+.4f}")

    out("")
    # Catastrophic overfitting verdict
    co_fgsm = bool(fgsm["co_flags"])
    co_fast = bool(fast["co_flags"])
    if co_fgsm and not co_fast:
        co_verdict = ("FGSM-AT suffers catastrophic overfitting; Fast-AT's random "
                      "init prevents it — consistent with Wong et al.")
    elif co_fgsm and co_fast:
        co_verdict = ("Both FGSM-AT and Fast-AT show CO signals — cyclic LR alone "
                      "insufficient; eps or LR tuning may be needed.")
    elif not co_fgsm and not co_fast:
        co_verdict = ("Neither method shows CO signals at this epoch/eps regime "
                      "(short training may not trigger CO).")
    else:
        co_verdict = ("Unexpected: Fast-AT shows CO while FGSM-AT does not "
                      "— check gradient-alignment trace.")

    out(f"  CO verdict: {co_verdict}")

    out("")
    # Efficiency verdict
    if rob_gap_fast < 0.05:
        eff_verdict = ("Fast-AT achieves PGD_ASR within 0.05 of PGD-10 while using "
                       "~10x fewer gradient steps — efficiency hypothesis SUPPORTED.")
    elif rob_gap_fast < 0.15:
        eff_verdict = ("Fast-AT is moderately worse than PGD-10 (gap < 0.15) — "
                       "partial efficiency gain, consistent with short training regime.")
    else:
        eff_verdict = ("Fast-AT is substantially worse than PGD-10 — single-step AT "
                       "does not generalise well at this scale/epoch count.")

    out(f"  Efficiency verdict: {eff_verdict}")

    out("")
    out(f"Total elapsed: {time.time() - t0:.1f}s")
    out("=" * 80)

    flush_file()
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
