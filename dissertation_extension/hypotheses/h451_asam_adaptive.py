"""
H451 - ASAM (Adaptive Sharpness-Aware Minimisation) vs SAM vs Anti-SAM.

Anchor: Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization for
Scale-Invariant Learning of Deep Neural Networks", ICML 2021.

Context / critique
------------------
SAM (Foret 2021) perturbs weights toward the worst-case direction at a
fixed radius rho, then minimises the resulting "sharp" loss. Its weakness
is that the fixed-rho sphere ignores per-parameter scale: a uniform rho
applied to differently-scaled parameters does not yield a scale-invariant
notion of sharpness. ASAM (Kwon 2021) fixes this by SCALING the
perturbation by |w| (elementwise), making sharpness scale-invariant and
typically allowing much larger nominal rho (e.g. rho=0.5..2.0 vs base-SAM
rho~0.05). GSAM (Zhuang 2022) and F-SAM are later variants that decompose
the gradient further; we focus on ASAM here.

In our campaign, H273 found base-SAM was WORSE than SGD for adversarial
robustness, while H376 found anti-SAM (perturb TOWARD sharper minima) was
POSITIVE (PGD ASR 0.79 vs ~0.92). This raises two questions:

  Q1.  Does ASAM differ from SAM enough to flip the H273 verdict?
       ASAM's elementwise scaling could either (a) help, by finding
       more "calibrated" flat minima that happen to be robust, or
       (b) make the situation worse, by even more aggressively
       smoothing the loss landscape (Tsipras-style accuracy/robustness
       trade pushes ASAM closer to the standard-accuracy regime).
  Q2.  Is the H376 anti-SAM finding robust to the ASAM variant?
       We also test "anti-ASAM" - perturb adaptively TOWARD the sharp
       region with elementwise-|w| scaling - to see whether the
       positive anti-SAM effect is preserved under scale-invariance.

Controls
--------
  - sgd_baseline          : standard SGD (replicates H273 baseline)
  - sam_rho005            : base-SAM rho=0.05 (replicates H273 SAM)
  - anti_sam_rho005       : anti-SAM rho=0.05 (replicates H376 best)
  - asam_rho{0.05,0.1,0.5,1.0}     : ASAM rho sweep
  - anti_asam_rho{0.05,0.1,0.5,1.0}: anti-ASAM (mirror) rho sweep
  - clean + FGSM + PGD-10 eval
  - per-class clean accuracy and per-class PGD ASR breakdown

Expected outcome
----------------
If ASAM merely deepens the SAM effect, ASAM PGD ASR will be >= SAM PGD ASR
(worse robustness). If the H376 finding is geometry-driven (sharper minima
= more robust), anti-ASAM should mirror anti-SAM's win. Falsifiable: a
single rho where ASAM beats SGD by >0.02 in PGD ASR would refute the
"SAM-family always hurts robustness" reading of H273; conversely an
anti-ASAM that fails to match anti-SAM would suggest the H376 effect is
specific to non-adaptive perturbation.

Related work cited:
  Foret et al. 2021 (SAM, ICLR)
  Kwon et al. 2021 (ASAM, ICML)            <- anchor
  Zhuang et al. 2022 (GSAM, ICLR)
  Liu et al. 2020 (sharpness <-> AT)
  Foret 2021 / Andriushchenko 2022 (SAM <-> robustness)
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
DS        = "fashion_mnist"
SEED      = 0
N_TRAIN   = 6000
EPOCHS    = 10
LR        = 0.05
MOMENTUM  = 0.9
WD        = 5e-4
BATCH     = 128
EPS       = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01

# Rho schedule for ASAM / anti-ASAM (Kwon-2021 recommends rho ~10x SAM).
ASAM_RHOS = [0.05, 0.1, 0.5, 1.0]

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h451_asam_adaptive_output.txt"
)


# ---------------------------------------------------------------------------
# SAM / ASAM family training functions (manual, no external optimiser).
#
# For each variant we do two forward/backward passes:
#   step 1: grad at current w
#   step 2: temporarily perturb w -> w + e(w, g, rho), grad there
#   step 3: restore w, update with the step-2 gradient
#
# direction:   "ascent"  -> SAM / ASAM           (perturb to max loss)
#              "descent" -> anti-SAM / anti-ASAM (perturb to min loss)
# adaptive:    False -> rho * g / ||g||_2        (base SAM)
#              True  -> rho * (|w| * g) / ||(|w| * g)||_2  (ASAM)
# ---------------------------------------------------------------------------

def _compute_perturbation(model, rho, adaptive):
    """Return list of per-parameter perturbation tensors e_w."""
    eps_w = []
    if adaptive:
        # ||(|w| * g)||_2 over all params (Kwon Eq. 6, p=2).
        sq = 0.0
        for p in model.parameters():
            if p.grad is None:
                eps_w.append(None)
                continue
            sq = sq + (p.data.abs() * p.grad).pow(2).sum()
        norm = torch.sqrt(sq) + 1e-12
        out = []
        i = 0
        for p in model.parameters():
            if p.grad is None:
                out.append(None)
                continue
            # e_w = rho * |w|^2 * g / norm  (scale-invariant; Kwon p. 5).
            out.append(rho * (p.data.abs() ** 2) * p.grad / norm)
            i += 1
        return out
    else:
        # plain SAM: rho * g / ||g||_2.
        sq = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            sq = sq + p.grad.pow(2).sum()
        norm = torch.sqrt(sq) + 1e-12
        out = []
        for p in model.parameters():
            if p.grad is None:
                out.append(None)
                continue
            out.append(rho * p.grad / norm)
        return out


def train_sam_family(model, Xtr, Ytr, rho, direction, adaptive, label=""):
    """Generic trainer covering SGD / SAM / ASAM / anti-SAM / anti-ASAM."""
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        ep_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            if rho <= 0:
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                continue

            # --- step 1: grad at w ---
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()

            # --- step 2: perturb w -> w + sign * e_w ---
            e_w = _compute_perturbation(model, rho, adaptive)
            sign = +1.0 if direction == "ascent" else -1.0
            old_p = []
            with torch.no_grad():
                for p, e in zip(model.parameters(), e_w):
                    old_p.append(p.data.clone())
                    if e is not None:
                        p.data.add_(sign * e)

            # --- step 3: grad at perturbed point ---
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()

            # restore weights, then take an SGD step using the
            # gradient computed at the perturbed point.
            with torch.no_grad():
                for p, op in zip(model.parameters(), old_p):
                    p.data.copy_(op)
            opt.step()

            ep_loss += float(loss.item())
        sched.step()
        if (ep + 1) % 5 == 0:
            print(f"    {label} epoch {ep+1}/{EPOCHS}  loss={ep_loss:.3f}",
                  flush=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def eval_model(model, Xte, Yte, label):
    """Clean / FGSM / PGD ASR + per-class breakdown + mean margin."""
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    logits_pgd, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))

    # per-class clean acc and per-class PGD ASR (out of originally-correct).
    Yte_cpu = Yte.cpu().numpy()
    pred_pgd = logits_pgd.argmax(1).numpy()
    # need clean preds too:
    with torch.no_grad():
        clean_pred = []
        for i in range(0, Xte.size(0), 512):
            clean_pred.append(model(Xte[i:i+512]).argmax(1).cpu())
        clean_pred = torch.cat(clean_pred).numpy()
    per_class_clean = {}
    per_class_pgd_asr = {}
    for c in range(10):
        mask = Yte_cpu == c
        if mask.sum() == 0:
            per_class_clean[c] = float("nan")
            per_class_pgd_asr[c] = float("nan")
            continue
        per_class_clean[c] = float((clean_pred[mask] == c).mean())
        corr_mask = (clean_pred == Yte_cpu) & mask
        if corr_mask.sum() == 0:
            per_class_pgd_asr[c] = float("nan")
        else:
            per_class_pgd_asr[c] = float(
                (pred_pgd[corr_mask] != Yte_cpu[corr_mask]).mean()
            )

    out = dict(
        clean_acc=float(clean_acc),
        fgsm_asr=1.0 - float(acc_fgsm),
        pgd_asr=1.0 - float(acc_pgd),
        mean_margin=mean_margin,
        per_class_clean=per_class_clean,
        per_class_pgd_asr=per_class_pgd_asr,
    )
    print(f"  [{label}] clean={out['clean_acc']:.4f}  "
          f"FGSM_ASR={out['fgsm_asr']:.4f}  "
          f"PGD_ASR={out['pgd_asr']:.4f}  margin={out['mean_margin']:.4f}",
          flush=True)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)
    # use the same eval slice convention as the campaign (n_eval=2000 default)
    print(f"train={Xtr.size(0)}  test={Xte.size(0)}", flush=True)

    # (name, rho, direction, adaptive)
    configs = [
        ("sgd_baseline",       0.00, "ascent",  False),
        ("sam_rho005",         0.05, "ascent",  False),
        ("anti_sam_rho005",    0.05, "descent", False),
    ]
    for rho in ASAM_RHOS:
        configs.append((f"asam_rho{rho}",      rho, "ascent",  True))
    for rho in ASAM_RHOS:
        configs.append((f"anti_asam_rho{rho}", rho, "descent", True))

    results = {}
    for name, rho, direction, adaptive in configs:
        print(f"\n--- Training {name} (rho={rho}, dir={direction}, "
              f"adaptive={adaptive}) ---", flush=True)
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        train_sam_family(model, Xtr, Ytr, rho=rho, direction=direction,
                         adaptive=adaptive, label=name)
        r = eval_model(model, Xte, Yte, label=name)
        r["rho"] = rho
        r["direction"] = direction
        r["adaptive"] = adaptive
        results[name] = r

    # -----------------------------------------------------------------------
    # write report
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0
    lines = []
    lines.append("H451 - ASAM (Adaptive SAM) vs SAM vs Anti-SAM\n")
    lines.append("=" * 78 + "\n\n")
    lines.append("Anchor: Kwon et al. 2021 (ASAM, ICML 2021)\n")
    lines.append("Refs : Foret 2021 (SAM), Zhuang 2022 (GSAM)\n")
    lines.append(f"Config: N_train={N_TRAIN} epochs={EPOCHS} lr={LR} "
                 f"batch={BATCH} eps={EPS} pgd_steps={PGD_STEPS}\n")
    lines.append(f"Rho sweep: {ASAM_RHOS}\n\n")

    lines.append(
        f"{'Variant':<22} {'rho':>6} {'dir':>8} {'adap':>5} "
        f"{'CleanAcc':>9} {'FGSM_ASR':>9} {'PGD_ASR':>9} {'Margin':>9}\n"
    )
    lines.append("-" * 88 + "\n")
    for name, r in results.items():
        lines.append(
            f"{name:<22} {r['rho']:>6.2f} {r['direction']:>8} "
            f"{str(r['adaptive']):>5} "
            f"{r['clean_acc']:>9.4f} {r['fgsm_asr']:>9.4f} "
            f"{r['pgd_asr']:>9.4f} {r['mean_margin']:>9.4f}\n"
        )

    # Per-class breakdown for the headline rows.
    headline = ["sgd_baseline", "sam_rho005", "anti_sam_rho005"] + \
               [f"asam_rho{r}" for r in ASAM_RHOS] + \
               [f"anti_asam_rho{r}" for r in ASAM_RHOS]
    lines.append("\nPer-class breakdown (clean acc / PGD ASR)\n")
    lines.append("-" * 78 + "\n")
    cls_hdr = "Variant".ljust(22) + " " + \
              " ".join(f"c{c}".rjust(7) for c in range(10)) + "\n"
    lines.append(cls_hdr)
    for name in headline:
        r = results[name]
        lines.append(name.ljust(22) + " CLN " +
                     " ".join(f"{r['per_class_clean'][c]:>5.2f}"
                              for c in range(10)) + "\n")
        lines.append(" " * 22 + " ASR " +
                     " ".join(f"{r['per_class_pgd_asr'][c]:>5.2f}"
                              for c in range(10)) + "\n")

    # -----------------------------------------------------------------------
    # analysis / verdict
    # -----------------------------------------------------------------------
    sgd  = results["sgd_baseline"]
    sam  = results["sam_rho005"]
    asam = results["anti_sam_rho005"]

    best_asam_name = min((f"asam_rho{r}" for r in ASAM_RHOS),
                         key=lambda n: results[n]["pgd_asr"])
    best_asam = results[best_asam_name]
    best_anti_asam_name = min((f"anti_asam_rho{r}" for r in ASAM_RHOS),
                              key=lambda n: results[n]["pgd_asr"])
    best_anti_asam = results[best_anti_asam_name]

    lines.append("\nANALYSIS\n--------\n")
    lines.append(f"Baseline SGD              PGD_ASR = {sgd['pgd_asr']:.4f}\n")
    lines.append(f"SAM   rho=0.05            PGD_ASR = {sam['pgd_asr']:.4f}  "
                 f"(delta vs SGD = {sam['pgd_asr']-sgd['pgd_asr']:+.4f})\n")
    lines.append(f"Anti-SAM rho=0.05         PGD_ASR = {asam['pgd_asr']:.4f}  "
                 f"(delta vs SGD = {asam['pgd_asr']-sgd['pgd_asr']:+.4f})\n")
    lines.append(f"Best ASAM ({best_asam_name})    "
                 f"PGD_ASR = {best_asam['pgd_asr']:.4f}  "
                 f"(delta vs SGD = {best_asam['pgd_asr']-sgd['pgd_asr']:+.4f})\n")
    lines.append(f"Best anti-ASAM ({best_anti_asam_name})  "
                 f"PGD_ASR = {best_anti_asam['pgd_asr']:.4f}  "
                 f"(delta vs SGD = {best_anti_asam['pgd_asr']-sgd['pgd_asr']:+.4f})\n")

    # Three falsifiable claims.
    asam_helps   = best_asam["pgd_asr"]      < sgd["pgd_asr"]  - 0.02
    asam_matches = abs(best_asam["pgd_asr"]  - sam["pgd_asr"]) <= 0.02
    anti_asam_wins = best_anti_asam["pgd_asr"] < sgd["pgd_asr"] - 0.02

    lines.append("\nClaims\n------\n")
    lines.append(f"C1. ASAM beats SGD (PGD ASR drop > 0.02): {asam_helps}\n")
    lines.append(f"C2. ASAM ties base SAM (|delta| <= 0.02): {asam_matches}\n")
    lines.append(f"C3. Anti-ASAM mirrors anti-SAM win (>0.02 drop vs SGD): "
                 f"{anti_asam_wins}\n")

    if anti_asam_wins and not asam_helps:
        verdict = ("Anti-SAM finding REPLICATES under scale-invariance "
                   "(anti-ASAM also robust); standard ASAM does NOT help. "
                   "H376 sign of effect is the geometry that matters.")
    elif asam_helps and not anti_asam_wins:
        verdict = ("ASAM (adaptive flatness-seeking) helps robustness here, "
                   "contradicting the H273 SAM result. The fixed-rho of base "
                   "SAM may have been the obstacle, not flatness itself.")
    elif asam_helps and anti_asam_wins:
        verdict = ("Both ASAM and anti-ASAM beat SGD: scale-invariant "
                   "perturbation of either sign helps. Effect likely from "
                   "ASAM's adaptive |w|-scaling, not the direction.")
    else:
        verdict = ("Neither ASAM nor anti-ASAM beats SGD by >0.02. "
                   "ASAM does NOT differ from SAM enough to matter for "
                   "robustness at this scale; H376 may be specific to "
                   "non-adaptive perturbation.")
    lines.append(f"\nVerdict: {verdict}\n")
    lines.append(f"\nElapsed: {elapsed:.1f}s\n")

    with open(OUT_FILE, "w") as f:
        f.writelines(lines)
        f.flush()
    print(f"\nResults written to {OUT_FILE}", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
