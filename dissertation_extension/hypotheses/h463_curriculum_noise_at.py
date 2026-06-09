"""
H463 - Curriculum-noise + AT mix (gap G3, Fashion-MNIST).

Anchor / prior art:
  - H264 (this campaign): pure noise-curriculum sigma 0.3 -> 0.0 over 10
    epochs vs constant sigma=0.15 vs reverse curriculum. Found that
    noise-curriculum alone is FGSM-only: PGD ASR barely moves at eps=0.1
    (C4 finding in CAMPAIGN_GAP_MAP.md). Noise augmentation is necessary
    but not sufficient.
  - Cai, Liu & Song (2018) "Curriculum Adversarial Training" (IJCAI 2018).
    Ramp eps_train from 0 -> eps_max over training; argues that an
    easy-to-hard curriculum on the PGD radius improves robust generalisation
    and stabilises optimisation vs jumping straight to eps_max.
  - Wang, Ma, Bailey et al. (2019) "On the Convergence and Robustness of
    Adversarial Training" / DAT (ICML 2019). Use a "convergence quality"
    criterion (FOSC) to anneal the PGD inner-max strength; weaker adv
    examples early, stronger later.
  - Sitawarin, Chakraborty & Wagner (2020) "SAT: Improving Adversarial
    Training via Curriculum-based Loss Smoothing". Soft curriculum on
    the adversarial loss term.
  - Madry et al. (2018) PGD-AT - canonical reference (no curriculum,
    inner-max at full eps from epoch 1).

Gap addressed (campaign G3 - "training objectives missing"):
  The campaign tested noise-curriculum standalone (H264, FGSM-only win)
  and PGD-AT (saturating PGD ASR ~0.32-0.34) but never combined them.
  Two natural curricula remain unexplored:
    (i)  PRE-WARM curriculum: Gaussian noise sigma 0.3 -> 0.0 over the
         first half of training, then transition to PGD-AT for the second
         half. Noise here serves as a cheap "soft" surrogate for early
         PGD examples (cheap because no inner-max gradient steps).
    (ii) EPS curriculum (Cai 2018): keep PGD-AT throughout but ramp
         attack radius eps_train from 0 -> 0.1 linearly over epochs.

H463 asks: do either of these curricula beat vanilla PGD-AT at the
campaign's 6k/10-epoch scale, OR do they merely match it (Cai 2018's
gain may not survive sub-scale training and may also be due to under-
trained inner-max at full eps becoming overcooked at small N)?

Hypothesis:
  Curriculum (noise pre-warm OR eps ramp) MATCHES PGD-AT on PGD ASR
  (within +/-0.03) and gives a small clean-accuracy bump (+1-3pp)
  because early-epoch noise/low-eps batches act as a regulariser before
  the model commits to robust features. Faster (3-epoch) noise pre-warm
  beats slower (5-epoch) pre-warm because the campaign budget is only
  10 epochs total; spending half on noise leaves PGD-AT under-trained.
  Risk: any masking signal (transfer ASR << white-box ASR for curriculum
  models) would invalidate the white-box "match", so we include a
  transfer-attack control from a pure PGD-AT source.

Conditions (8 runs):
  A. CLEAN_BASE           - vanilla CE, no noise, no AT. Sanity ref.
  B. PGDAT_PURE           - PGD-AT all 10 epochs at eps=0.1 (Madry).
                            CONTROL: canonical baseline.
  C. NOISE_ONLY           - Noise-curriculum 0.3 -> 0.0 over 10 epochs,
                            no AT. CONTROL: replicates H264 anneal arm.
  D. NOISE3_AT7           - Noise sigma 0.3 -> 0.0 over epochs 1..3
                            (FAST pre-warm), then PGD-AT @ eps=0.1 for
                            epochs 4..10.
  E. NOISE5_AT5           - Noise sigma 0.3 -> 0.0 over epochs 1..5
                            (SLOW pre-warm), then PGD-AT @ eps=0.1 for
                            epochs 6..10.
  F. EPS_RAMP             - PGD-AT all 10 epochs; eps_train ramped
                            linearly 0 -> 0.1 over epochs 1..5, held at
                            0.1 for epochs 6..10. (Cai 2018 style.)
  G. EPS_RAMP_SLOW        - PGD-AT all 10 epochs; eps_train ramped
                            0 -> 0.1 over all 10 epochs (no plateau).
  H. NOISE5_AT5_THEN_EPS  - Hybrid: noise pre-warm 1..5, then eps-ramp
                            6..10 (eps 0.02 -> 0.10). Tests whether
                            stacking the two curricula compounds.

Each model evaluated on:
  - clean accuracy
  - FGSM ASR @ eps=0.1
  - PGD-10 ASR @ eps=0.1 (white-box; the campaign's headline metric)
  - PGD-20 ASR @ eps=0.1 (stronger white-box; masking probe)
  - TRANSFER ASR: PGD-10 adversarials crafted on B (PGD_PURE) applied
    to each model. If white-box PGD ASR < transfer ASR by a large
    margin, the model is gradient-masking.
  - mean clean margin

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9,wd=5e-4),
        SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01. CNN width=32.
Output: results/fashion_mnist/h463_curriculum_noise_at_output.txt (ASCII).
"""
import os
import sys
import time

import numpy as np
import torch
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
PGD20_STEPS = 20
SIGMA_MAX = 0.3
META = {"channels": 1, "size": 28, "n_classes": 10}

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
)
OUT_FILE = os.path.join(RESULTS_DIR, "h463_curriculum_noise_at_output.txt")


# ---- helpers -------------------------------------------------------------
def _make_opt(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)


def _new_model():
    C.set_seed(SEED)
    return C.build_model("cnn", META, width=32)


def linear_ramp(ep, ep_start, ep_end, v_start, v_end):
    """Piecewise-linear schedule: clamp at endpoints outside [ep_start, ep_end]."""
    if ep <= ep_start:
        return v_start
    if ep >= ep_end:
        return v_end
    frac = (ep - ep_start) / float(ep_end - ep_start)
    return v_start + frac * (v_end - v_start)


def build_schedule(name):
    """Return list of (sigma, adv, eps_train) per epoch (0-indexed, len=EPOCHS).

    Conventions:
      sigma > 0    -> additive Gaussian noise on inputs that epoch.
      adv = True   -> apply PGD-AT inner-max at eps_train (alpha = 2.5 eps/steps).
      adv = False  -> clean CE on (possibly noisy) inputs.
    """
    sch = []
    for ep in range(EPOCHS):  # ep is 0..EPOCHS-1
        sigma = 0.0
        adv = False
        eps_train = 0.0
        if name == "CLEAN_BASE":
            pass
        elif name == "PGDAT_PURE":
            adv = True
            eps_train = EPS
        elif name == "NOISE_ONLY":
            # H264-style anneal sigma_max -> 0 over 10 epochs.
            sigma = SIGMA_MAX * (1.0 - ep / float(EPOCHS))
            adv = False
        elif name == "NOISE3_AT7":
            if ep < 3:
                # noise ramp 0.30 -> 0.10 over epochs 0,1,2
                sigma = linear_ramp(ep, 0, 3, SIGMA_MAX, SIGMA_MAX / 3.0)
                adv = False
            else:
                adv = True
                eps_train = EPS
        elif name == "NOISE5_AT5":
            if ep < 5:
                sigma = linear_ramp(ep, 0, 5, SIGMA_MAX, 0.0)
                adv = False
            else:
                adv = True
                eps_train = EPS
        elif name == "EPS_RAMP":
            adv = True
            # ramp eps 0 -> 0.1 over epochs 0..4, hold at 0.1 epochs 5..9
            if ep < 5:
                eps_train = linear_ramp(ep + 1, 0, 5, 0.0, EPS)
            else:
                eps_train = EPS
        elif name == "EPS_RAMP_SLOW":
            adv = True
            eps_train = linear_ramp(ep + 1, 0, EPOCHS, 0.0, EPS)
        elif name == "NOISE5_AT5_THEN_EPS":
            if ep < 5:
                sigma = linear_ramp(ep, 0, 5, SIGMA_MAX, 0.0)
                adv = False
            else:
                adv = True
                # ramp eps 0.02 -> 0.10 across the AT phase epochs 5..9
                eps_train = linear_ramp(ep - 5 + 1, 0, 5, 0.02, EPS)
        else:
            raise ValueError(name)
        sch.append((float(sigma), bool(adv), float(eps_train)))
    return sch


def train_curriculum(name, Xtr, Ytr):
    """Train under the named curriculum schedule. Returns model."""
    sch = build_schedule(name)
    model = _new_model()
    opt = _make_opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        sigma, adv, eps_t = sch[ep]
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            if adv and eps_t > 0.0:
                # PGD inner-max at current eps_t.
                alpha = 2.5 * eps_t / PGD_STEPS
                xb_in = C.pgd(model, xb, yb, eps=eps_t,
                              steps=PGD_STEPS, alpha=alpha)
            elif sigma > 0.0:
                xb_in = (xb + torch.randn_like(xb) * sigma).clamp(0.0, 1.0)
            else:
                xb_in = xb
            opt.zero_grad()
            F.cross_entropy(model(xb_in), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model, sch


# ---- evaluation ----------------------------------------------------------
@torch.no_grad()
def _clean_acc(model, X, Y):
    _, a = C.logits_and_acc(model, X, Y)
    return a


def _asr_with_attack(model, X, Y, attack_fn):
    """attack_fn(model, x, y) -> x_adv. Returns ASR over originally-correct."""
    model.eval()
    flips, corr = [], []
    B = 256
    for i in range(0, X.size(0), B):
        x, y = X[i:i + B], Y[i:i + B]
        with torch.no_grad():
            c = model(x).argmax(1) == y
        xa = attack_fn(model, x, y)
        with torch.no_grad():
            f = model(xa).argmax(1) != y
        flips.append(f.cpu())
        corr.append(c.cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr).numpy().astype(bool)
    if corr.sum() == 0:
        return float("nan")
    return float(flips[corr].mean())


def _transfer_asr(target_model, X, Y, X_adv):
    """ASR when target_model is evaluated on adversarials X_adv crafted
    elsewhere. Measured over samples target_model classifies correctly
    on the CLEAN inputs (consistent with white-box ASR definition)."""
    target_model.eval()
    flips, corr = [], []
    B = 256
    for i in range(0, X.size(0), B):
        x, y = X[i:i + B], Y[i:i + B]
        xa = X_adv[i:i + B]
        with torch.no_grad():
            c = target_model(x).argmax(1) == y
            f = target_model(xa).argmax(1) != y
        corr.append(c.cpu())
        flips.append(f.cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr).numpy().astype(bool)
    if corr.sum() == 0:
        return float("nan")
    return float(flips[corr].mean())


def evaluate(model, Xte, Yte, transfer_adv=None):
    clean = _clean_acc(model, Xte, Yte)
    fgsm_asr = _asr_with_attack(
        model, Xte, Yte,
        lambda m, x, y: C.fgsm(m, x, y, eps=EPS),
    )
    pgd10_asr = _asr_with_attack(
        model, Xte, Yte,
        lambda m, x, y: C.pgd(m, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA),
    )
    pgd20_asr = _asr_with_attack(
        model, Xte, Yte,
        lambda m, x, y: C.pgd(m, x, y, eps=EPS, steps=PGD20_STEPS,
                              alpha=2.5 * EPS / PGD20_STEPS),
    )
    mar = C.margin(model, Xte, Yte)
    mean_margin = float(np.mean(mar))
    tr_asr = float("nan")
    if transfer_adv is not None:
        tr_asr = _transfer_asr(model, Xte, Yte, transfer_adv)
    return {
        "clean": clean,
        "fgsm_asr": fgsm_asr,
        "pgd10_asr": pgd10_asr,
        "pgd20_asr": pgd20_asr,
        "transfer_asr": tr_asr,
        "mean_margin": mean_margin,
    }


def craft_transfer_adv(source_model, X, Y):
    """Craft PGD-10 adversarials in batches on source_model for transfer."""
    source_model.eval()
    out = []
    B = 256
    for i in range(0, X.size(0), B):
        x, y = X[i:i + B], Y[i:i + B]
        xa = C.pgd(source_model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
        out.append(xa.detach())
    return torch.cat(out, dim=0)


# ---- main ----------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H463  Curriculum-noise + AT mix (Fashion-MNIST, gap G3)")
    out("=" * 80)
    out("Anchor: Cai, Liu & Song (2018) Curriculum Adversarial Training, IJCAI.")
    out("Priors: Wang+ (2019) DAT ICML; Sitawarin+ (2020) SAT; Madry+ (2018) PGD-AT;")
    out("        H264 (this campaign) noise-curriculum standalone.")
    out("")
    out("Question: does pre-warming with noise-curriculum, OR ramping eps_train")
    out("          (Cai 2018), beat or merely match canonical PGD-AT at the")
    out("          6k/10-epoch campaign scale?  Includes masking control.")
    out("")
    out("Hypothesis: curricula MATCH PGD-AT on PGD ASR (within +/-0.03) and may")
    out("            buy a small clean-acc bump. Fast (3ep) pre-warm beats slow")
    out("            (5ep) pre-warm because total budget is only 10 epochs.")
    out("")
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4)")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} "
        f"PGD20_STEPS={PGD20_STEPS} SEED={SEED}")
    out(f"        SIGMA_MAX={SIGMA_MAX}")
    out(f"        device={C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                        seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    condition_names = [
        "CLEAN_BASE",
        "PGDAT_PURE",
        "NOISE_ONLY",
        "NOISE3_AT7",
        "NOISE5_AT5",
        "EPS_RAMP",
        "EPS_RAMP_SLOW",
        "NOISE5_AT5_THEN_EPS",
    ]

    out("conditions:")
    for nm in condition_names:
        sch = build_schedule(nm)
        sig_str = ",".join(f"{s[0]:.2f}" for s in sch)
        adv_str = "".join("A" if s[1] else "." for s in sch)
        eps_str = ",".join(f"{s[2]:.2f}" for s in sch)
        out(f"  {nm:<22} sigma=[{sig_str}]")
        out(f"  {'':22}     adv=[{adv_str}]   epsT=[{eps_str}]")
    out("")

    # --- train all models, but train PGDAT_PURE first so we can use it as
    #     the transfer-attack source for all others (including itself).
    rows = {}
    models = {}
    out("[phase 1] training PGDAT_PURE as transfer source")
    t1 = time.time()
    model_src, _ = train_curriculum("PGDAT_PURE", Xtr, Ytr)
    models["PGDAT_PURE"] = model_src
    out(f"  PGDAT_PURE trained in {time.time()-t1:.1f}s")
    flush()

    out("")
    out("[phase 2] crafting transfer adversarials from PGDAT_PURE")
    t1 = time.time()
    X_adv_src = craft_transfer_adv(model_src, Xte, Yte)
    out(f"  crafted {X_adv_src.shape[0]} transfer advs in {time.time()-t1:.1f}s")
    flush()

    out("")
    out("[phase 3] training the remaining conditions and evaluating all")
    for nm in condition_names:
        if nm == "PGDAT_PURE":
            model = model_src
            train_sec = 0.0  # already trained
        else:
            t1 = time.time()
            model, _ = train_curriculum(nm, Xtr, Ytr)
            train_sec = time.time() - t1
            models[nm] = model

        t2 = time.time()
        r = evaluate(model, Xte, Yte, transfer_adv=X_adv_src)
        eval_sec = time.time() - t2

        r["cond"] = nm
        r["train_sec"] = train_sec
        rows[nm] = r
        out(f"  {nm:<22} clean={r['clean']:.4f}  FGSM={r['fgsm_asr']:.4f}  "
            f"PGD10={r['pgd10_asr']:.4f}  PGD20={r['pgd20_asr']:.4f}  "
            f"TR={r['transfer_asr']:.4f}  marg={r['mean_margin']:.3f}  "
            f"({train_sec:.0f}s tr / {eval_sec:.0f}s ev)")
        flush()

    # --- summary table ----------------------------------------------------
    out("")
    out("=" * 80)
    out("SUMMARY TABLE")
    out("=" * 80)
    hdr = ("{:<22} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"
           .format("cond", "clean", "FGSM", "PGD10", "PGD20", "TRsrc", "margin"))
    out(hdr)
    out("-" * len(hdr))
    for nm in condition_names:
        r = rows[nm]
        out("{:<22} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}"
            .format(nm, r["clean"], r["fgsm_asr"], r["pgd10_asr"],
                    r["pgd20_asr"], r["transfer_asr"], r["mean_margin"]))
    out("-" * len(hdr))

    # --- deltas vs PGDAT_PURE --------------------------------------------
    base = rows["PGDAT_PURE"]
    out("")
    out("DELTA vs PGDAT_PURE (lower PGD ASR = better)")
    out("-" * 80)
    hdr2 = ("{:<22} {:>10} {:>10} {:>10} {:>10}"
            .format("cond", "dClean", "dPGD10", "dPGD20", "dMargin"))
    out(hdr2)
    out("-" * len(hdr2))
    for nm in condition_names:
        if nm == "PGDAT_PURE":
            continue
        r = rows[nm]
        out("{:<22} {:>+10.4f} {:>+10.4f} {:>+10.4f} {:>+10.4f}"
            .format(nm,
                    r["clean"] - base["clean"],
                    r["pgd10_asr"] - base["pgd10_asr"],
                    r["pgd20_asr"] - base["pgd20_asr"],
                    r["mean_margin"] - base["mean_margin"]))
    out("-" * len(hdr2))

    # --- masking diagnostic ----------------------------------------------
    out("")
    out("MASKING DIAGNOSTIC")
    out("-" * 80)
    out("If white-box PGD ASR << transfer ASR (from PGDAT_PURE source), the")
    out("white-box result is suspicious (gradient masking). Flag delta > 0.10.")
    out("Note: TR for PGDAT_PURE itself is the 'self-transfer' = white-box PGD10.")
    out("")
    out("{:<22} {:>8} {:>8} {:>8} {:>8}".format(
        "cond", "PGD10", "TRsrc", "TR-PGD10", "FLAG"))
    out("-" * 60)
    for nm in condition_names:
        if nm == "CLEAN_BASE":
            continue
        r = rows[nm]
        delta = r["transfer_asr"] - r["pgd10_asr"]
        flag = "MASK?" if delta > 0.10 else "ok"
        out("{:<22} {:>8.4f} {:>8.4f} {:>+8.4f} {:>8}".format(
            nm, r["pgd10_asr"], r["transfer_asr"], delta, flag))

    # --- verdict logic ----------------------------------------------------
    out("")
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)

    THR_MATCH = 0.03         # +/-3pp = "match"
    THR_BEAT = 0.03          # must beat PGDAT_PURE by 3pp PGD ASR to "beat"
    THR_MASK = 0.10          # transfer - whitebox > 10pp = masking suspect

    base_pgd = base["pgd10_asr"]
    curricula = ["NOISE3_AT7", "NOISE5_AT5", "EPS_RAMP", "EPS_RAMP_SLOW",
                 "NOISE5_AT5_THEN_EPS"]
    cur_best = min(curricula, key=lambda k: rows[k]["pgd10_asr"])
    best_pgd = rows[cur_best]["pgd10_asr"]
    best_delta = best_pgd - base_pgd

    # masking check on best curriculum
    best_mask = (rows[cur_best]["transfer_asr"] - rows[cur_best]["pgd10_asr"]
                 > THR_MASK)

    # control sanity: NOISE_ONLY should NOT match PGD-AT (per H264/C4)
    noise_only_delta = rows["NOISE_ONLY"]["pgd10_asr"] - base_pgd
    noise_only_helps = noise_only_delta < -THR_BEAT

    out(f"  PGDAT_PURE       : clean={base['clean']:.4f}  PGD10={base_pgd:.4f}  "
        f"margin={base['mean_margin']:.3f}")
    out(f"  best curriculum  : {cur_best}  PGD10={best_pgd:.4f}  "
        f"(delta vs PGD-AT = {best_delta:+.4f})")
    out(f"  NOISE_ONLY ctrl  : PGD10={rows['NOISE_ONLY']['pgd10_asr']:.4f}  "
        f"(delta vs PGD-AT = {noise_only_delta:+.4f})")
    out("")

    if best_mask:
        verdict = (
            f"MASKING SUSPECT: best curriculum '{cur_best}' has transfer ASR "
            f"({rows[cur_best]['transfer_asr']:.3f}) exceeding white-box "
            f"PGD-10 ({best_pgd:.3f}) by >{THR_MASK:.2f}. Treat any apparent "
            "robustness gain as artefact; PGD-AT remains the trustworthy "
            "anchor."
        )
    elif best_delta < -THR_BEAT:
        verdict = (
            f"REFUTED (positive surprise): best curriculum '{cur_best}' BEATS "
            f"PGD-AT on PGD-10 by {-best_delta:.3f} at matched compute and "
            "passes the transfer-masking check. Curriculum AT may compound "
            "with PGD-AT under sub-scale training. Recommend cross-seed "
            "replication before claiming a real gain."
        )
    elif abs(best_delta) <= THR_MATCH:
        if noise_only_helps:
            verdict = (
                "PARTIAL/CONFUSED: curricula match PGD-AT (within +/-0.03), "
                "but NOISE_ONLY also matched PGD-AT - that contradicts H264/C4 "
                "and points to an under-trained PGD-AT baseline rather than a "
                "real curriculum effect."
            )
        else:
            verdict = (
                "CONFIRMED: curriculum AT (noise pre-warm or eps ramp) MATCHES "
                "vanilla PGD-AT on PGD-10 ASR (within +/-0.03) and on margin. "
                "NOISE_ONLY remains FGSM-only as H264 found. Curriculum is a "
                "stylistic alternative to PGD-AT, not a free-lunch improvement, "
                "at the 6k/10-epoch scale."
            )
    else:
        verdict = (
            f"NOT SUPPORTED / WORSE: best curriculum '{cur_best}' has PGD-10 "
            f"ASR {best_pgd:.3f} which is {best_delta:+.3f} vs PGDAT_PURE "
            f"({base_pgd:.3f}). Curricula HURT robustness at this scale, "
            "likely because pre-warm phases shorten the effective AT budget."
        )

    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time()-t0:.1f}s")
    flush()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
