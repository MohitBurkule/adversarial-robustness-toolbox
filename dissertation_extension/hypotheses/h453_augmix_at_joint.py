"""
H453 - AugMix x PGD-AT joint training (gap G3, Fashion-MNIST).

Anchor: Hendrycks et al. (2020) "AugMix: A Simple Data Processing Method to
Improve Robustness and Uncertainty" (ICLR 2020).

Companion priors:
  - Modas et al. (2022) "PRIME: A Few Primitives Can Boost Robustness to
    Common Corruptions" (ECCV 2022). Found that simple augmentation
    primitives stack with adversarial training but with diminishing returns
    on Linf robustness (corruption-AT is largely a separate threat model).
  - Kireev, Andriushchenko, Flammarion (2022) "On the effectiveness of
    adversarial training against common corruptions" (UAI 2022). Showed
    standard PGD-AT already buys partial common-corruption robustness, but
    Linf-AT and corruption-AT optimize different objectives; combining them
    can give cross-threat gains OR neutral interference depending on eps and
    augmentation magnitude.

Gap addressed (campaign G3 - "cross-threat-model joint defences"):
  H433 ran AugMix STANDALONE and asked whether AugMix's JSD consistency
  loss transfers to Linf adversarial robustness. The current question is
  the JOINT case: when you train with BOTH AugMix (corruption-style
  hardening) AND PGD-AT (Linf hardening) at varying mix-weights vs
  adv-weights, do the two defences stack (additive on PGD AND corruptions),
  interfere (one cancels the other), or trade off (better on one, worse on
  the other)?

Hypothesis (H453):
  AugMix + PGD-AT joint training yields PGD robustness comparable to
  PGD-AT alone (no stacking on Linf) AND corruption robustness comparable
  to AugMix-only (no stacking on corruptions), i.e. each defence
  saturates its own threat model and joint training is essentially the
  MAX of the two, not the SUM. Strong AugMix weight may slightly degrade
  PGD-AT (interference); strong adv weight may slightly degrade
  corruption robustness.

Design:
  - 6 conditions in main grid + 2 control conditions = 8 runs.
  - Main grid: 3 mix-weights (LAMBDA_JSD in {0, 6, 12}) x 2 adv-weights
    (ADV_FRAC in {0.5, 1.0}). LAMBDA_JSD=0 collapses to PGD-AT baseline
    (when ADV_FRAC=1) or PGD-AT-mixbatch (when ADV_FRAC=0.5); LAMBDA_JSD=12
    is Hendrycks 2020 default.
  - Controls (explicit, even though some overlap the grid):
      * PGD-AT pure (ADV_FRAC=1, LAMBDA_JSD=0)        -- corresponds to
        canonical PGD-AT (~PGD ASR 0.33 from RESULTS_SUMMARY).
      * AugMix-only (ADV_FRAC=0, LAMBDA_JSD=12)       -- replicates H433.
      * Clean baseline (ADV_FRAC=0, LAMBDA_JSD=0)     -- vanilla CE.
  - For each run, eval on:
      * clean accuracy
      * PGD-10 ASR at eps=0.1                          (Linf threat)
      * Gaussian-noise corruption accuracy at sigma=0.2 (corruption proxy)
      * Gaussian-blur corruption accuracy (3x3 kernel, sigma=1.0)
      * Brightness-shift corruption accuracy (+0.15 then clamp)

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9, wd=5e-4),
        SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.
Output: results/fashion_mnist/h453_augmix_at_joint_output.txt (ASCII).
"""
import os
import sys
import time
import math

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
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01

# AugMix hyper-params (mirror H433 defaults)
K_CHAINS = 3
ALPHA = 1.0
MAX_TRANSLATE = 0.15
MAX_ROTATE = 30.0
MAX_SHEAR = 15.0
MAX_CONTRAST = 0.5
OPS_PER_CHAIN = 3

# Grid for joint study
LAMBDA_JSD_GRID = [0.0, 6.0, 12.0]   # AugMix JSD weight (0 = no AugMix)
ADV_FRAC_GRID = [0.5, 1.0]            # fraction of each batch that is PGD-attacked

# Corruption-proxy parameters
GAUSS_SIGMA = 0.2
BLUR_SIGMA = 1.0
BRIGHT_SHIFT = 0.15

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h453_augmix_at_joint_output.txt")


# ---- AugMix ops (pure PyTorch, ported from H433) -----------------------------

def _affine(x, angle=0.0, translate=(0.0, 0.0), shear=0.0):
    N, _, _, _ = x.shape
    angle_r = math.radians(angle)
    shear_r = math.radians(shear)
    cos_a, sin_a = math.cos(angle_r), math.sin(angle_r)
    m00 = cos_a
    m01 = -sin_a + math.tan(shear_r) * cos_a
    m10 = sin_a
    m11 = cos_a + math.tan(shear_r) * sin_a
    tx = translate[0] * 2.0
    ty = translate[1] * 2.0
    theta = torch.tensor([[m00, m01, tx], [m10, m11, ty]],
                         dtype=x.dtype, device=x.device).unsqueeze(0).expand(N, -1, -1)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=False)


def aug_translate_x(x, m): return _affine(x, translate=(m * MAX_TRANSLATE, 0.0))
def aug_translate_y(x, m): return _affine(x, translate=(0.0, m * MAX_TRANSLATE))
def aug_rotate(x, m):       return _affine(x, angle=m * MAX_ROTATE)
def aug_shear_x(x, m):      return _affine(x, shear=m * MAX_SHEAR)
def aug_contrast(x, m):
    factor = 1.0 + m * MAX_CONTRAST
    return ((x - 0.5) * factor + 0.5).clamp(0, 1)


_ALL_OPS = [aug_translate_x, aug_translate_y, aug_rotate, aug_shear_x, aug_contrast]


def _random_chain(x, rng):
    op_indices = torch.randint(len(_ALL_OPS), (OPS_PER_CHAIN,), generator=rng).tolist()
    magnitudes = (torch.rand(OPS_PER_CHAIN, generator=rng) * 2.0 - 1.0).tolist()
    out = x
    for idx, mag in zip(op_indices, magnitudes):
        out = _ALL_OPS[idx](out, mag)
    return out


def augmix_views(x, rng, k=K_CHAINS, alpha=ALPHA):
    """Return list of k augmented views of x (no mixture image needed here -
    we apply the JSD loss directly across the k views + original)."""
    return [_random_chain(x, rng) for _ in range(k)]


def jsd_loss(logits_orig, logits_views):
    all_logits = [logits_orig] + logits_views
    all_probs = [F.softmax(lg, dim=1) for lg in all_logits]
    M = torch.stack(all_probs, dim=0).mean(dim=0)
    log_M = M.clamp(1e-8).log()
    kl_sum = sum(F.kl_div(log_M, p, reduction="batchmean") for p in all_probs)
    return kl_sum / len(all_logits)


# ---- corruption proxies ------------------------------------------------------

def add_gaussian_noise(x, sigma, gen):
    n = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * sigma
    return (x + n).clamp(0, 1)


def gaussian_blur(x, sigma):
    """3x3 Gaussian blur (separable would be overkill for 28x28)."""
    k = 3
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = torch.outer(g, g)
    kernel2d = kernel2d.view(1, 1, k, k).expand(x.size(1), 1, k, k)
    return F.conv2d(x, kernel2d, padding=k // 2, groups=x.size(1))


def brightness_shift(x, shift):
    return (x + shift).clamp(0, 1)


# ---- training ----------------------------------------------------------------

def _make_opt(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)


def train_joint(Xtr, Ytr, seed, lambda_jsd, adv_frac, label):
    """Joint AugMix + PGD-AT training loop.

    Per batch:
      - sample adv_frac fraction of batch to be PGD-attacked (else clean).
      - CE loss is computed on this (mixed clean+adv) batch.
      - If lambda_jsd > 0, also compute JSD consistency loss between model
        on the *clean* originals and K AugMix views of those originals
        (consistency target is the clean input, following Hendrycks 2020).
    """
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    rng = torch.Generator(device=Xtr.device).manual_seed(seed + 17 + int(lambda_jsd * 10))
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            # PGD-AT portion
            if adv_frac > 0.0:
                n_adv = max(1, int(round(adv_frac * xb.size(0))))
                x_adv_in = xb[:n_adv]
                y_adv_in = yb[:n_adv]
                x_adv = C.pgd(model, x_adv_in, y_adv_in, eps=EPS,
                              steps=PGD_STEPS, alpha=PGD_ALPHA)
                x_ce = torch.cat([x_adv, xb[n_adv:]], dim=0)
                y_ce = torch.cat([y_adv_in, yb[n_adv:]], dim=0)
            else:
                x_ce, y_ce = xb, yb

            opt.zero_grad()
            ce = F.cross_entropy(model(x_ce), y_ce)
            loss = ce

            if lambda_jsd > 0.0:
                # JSD consistency between clean originals (xb) and their AugMix views.
                views = augmix_views(xb, rng)
                logits_orig = model(xb)
                logits_views = [model(v) for v in views]
                jsd = jsd_loss(logits_orig, logits_views)
                loss = loss + lambda_jsd * jsd

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- evaluation --------------------------------------------------------------

@torch.no_grad()
def _acc(model, X, Y):
    _, a = C.logits_and_acc(model, X, Y)
    return a


def evaluate(model, Xte, Yte, gen):
    """Return dict of metrics: clean, pgd, gauss, blur, bright."""
    clean = _acc(model, Xte, Yte)
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    pgd_asr = pg["asr"]

    X_g = add_gaussian_noise(Xte, GAUSS_SIGMA, gen)
    acc_g = _acc(model, X_g, Yte)

    X_b = gaussian_blur(Xte, BLUR_SIGMA)
    acc_b = _acc(model, X_b, Yte)

    X_br = brightness_shift(Xte, BRIGHT_SHIFT)
    acc_br = _acc(model, X_br, Yte)

    corr_avg = (acc_g + acc_b + acc_br) / 3.0
    return {"clean": clean, "pgd_asr": pgd_asr,
            "gauss": acc_g, "blur": acc_b, "bright": acc_br,
            "corr_avg": corr_avg}


# ---- main --------------------------------------------------------------------

def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H453  AugMix x PGD-AT joint training (Fashion-MNIST, gap G3)")
    out("=" * 80)
    out("Anchor: Hendrycks et al. (2020) AugMix, ICLR 2020.")
    out("Priors: Modas+ (2022) PRIME ECCV; Kireev+ (2022) AT-vs-corruptions UAI.")
    out("")
    out("Question: do AugMix (corruption-style) and PGD-AT (Linf) STACK, INTERFERE,")
    out("          or TRADE OFF when trained jointly on Fashion-MNIST/SmallCNN?")
    out("")
    out("H433 ran AugMix standalone (G1). This script (G3) sweeps mix-weight x")
    out("adv-weight and evaluates both Linf-PGD AND a corruption proxy")
    out("(Gaussian noise + blur + brightness).")
    out("")
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4)")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} SEED={SEED}")
    out(f"        K_CHAINS={K_CHAINS} ALPHA={ALPHA}")
    out(f"        LAMBDA_JSD_GRID={LAMBDA_JSD_GRID}  ADV_FRAC_GRID={ADV_FRAC_GRID}")
    out(f"        corruption proxies: gauss_sigma={GAUSS_SIGMA}, "
        f"blur_sigma={BLUR_SIGMA}, bright_shift={BRIGHT_SHIFT}")
    out(f"        device={C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    eval_gen = torch.Generator(device=Xte.device).manual_seed(SEED + 999)

    # ---- explicit controls + main grid --------------------------------------
    conditions = []
    conditions.append(("CLEAN_BASE", 0.0, 0.0))    # vanilla CE
    conditions.append(("AUGMIX_ONLY", 12.0, 0.0))  # replicates H433 (control)
    # main grid: 3 mix-weights x 2 adv-weights = 6 conditions
    for lj in LAMBDA_JSD_GRID:
        for af in ADV_FRAC_GRID:
            conditions.append((f"L{int(lj)}_A{af}", lj, af))

    out("conditions to run:")
    for nm, lj, af in conditions:
        tag = ""
        if lj == 0.0 and af == 0.0:
            tag = "  (control: clean CE)"
        elif lj == 0.0 and af == 1.0:
            tag = "  (control: pure PGD-AT)"
        elif lj == 12.0 and af == 0.0:
            tag = "  (control: AugMix-only, ~H433)"
        out(f"  {nm:>14}   lambda_jsd={lj:>5.1f}  adv_frac={af:>4.2f}{tag}")
    out("")

    rows = []
    for nm, lj, af in conditions:
        out(f"[training {nm}] lambda_jsd={lj} adv_frac={af} ...")
        t1 = time.time()
        model = train_joint(Xtr, Ytr, SEED, lambda_jsd=lj, adv_frac=af, label=nm)
        m = evaluate(model, Xte, Yte, eval_gen)
        m.update({"cond": nm, "lambda_jsd": lj, "adv_frac": af,
                  "train_sec": time.time() - t1})
        rows.append(m)
        out(f"  {nm}: clean={m['clean']:.4f} PGD_ASR={m['pgd_asr']:.4f} "
            f"gauss={m['gauss']:.4f} blur={m['blur']:.4f} bright={m['bright']:.4f} "
            f"corr_avg={m['corr_avg']:.4f}  ({time.time()-t0:.0f}s)")
        flush()

    # ---- summary table -------------------------------------------------------
    out("")
    out("=" * 80)
    out("SUMMARY TABLE")
    out("=" * 80)
    hdr = ("{:<14} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8}"
           .format("cond", "lam", "adv", "clean", "PGD_ASR",
                   "gauss", "blur", "corrAvg"))
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<14} {:>7.1f} {:>7.2f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}"
            .format(r["cond"], r["lambda_jsd"], r["adv_frac"],
                    r["clean"], r["pgd_asr"], r["gauss"], r["blur"], r["corr_avg"]))
    out("-" * len(hdr))

    # ---- verdict logic -------------------------------------------------------
    out("")
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)

    def find(lj, af):
        for r in rows:
            if r["lambda_jsd"] == lj and r["adv_frac"] == af:
                return r
        return None

    clean_base = find(0.0, 0.0)
    pgd_at     = find(0.0, 1.0)
    augmix_only = find(12.0, 0.0)
    joint_best = None
    joint_candidates = [r for r in rows if r["lambda_jsd"] > 0.0 and r["adv_frac"] > 0.0]
    if joint_candidates:
        # pick the joint condition with lowest PGD_ASR (primary objective)
        joint_best = min(joint_candidates, key=lambda r: r["pgd_asr"])

    out(f"  CLEAN_BASE   : clean={clean_base['clean']:.4f}  PGD_ASR={clean_base['pgd_asr']:.4f}  "
        f"corr_avg={clean_base['corr_avg']:.4f}")
    out(f"  PGD-AT pure  : clean={pgd_at['clean']:.4f}  PGD_ASR={pgd_at['pgd_asr']:.4f}  "
        f"corr_avg={pgd_at['corr_avg']:.4f}")
    out(f"  AugMix only  : clean={augmix_only['clean']:.4f}  PGD_ASR={augmix_only['pgd_asr']:.4f}  "
        f"corr_avg={augmix_only['corr_avg']:.4f}")
    if joint_best is not None:
        out(f"  BEST JOINT   : {joint_best['cond']}  clean={joint_best['clean']:.4f}  "
            f"PGD_ASR={joint_best['pgd_asr']:.4f}  corr_avg={joint_best['corr_avg']:.4f}")

    # decision rules
    THR = 0.03    # 3pp threshold for "meaningful" change
    pgd_stack    = (joint_best is not None
                    and joint_best["pgd_asr"] + THR < pgd_at["pgd_asr"])
    pgd_match    = (joint_best is not None
                    and abs(joint_best["pgd_asr"] - pgd_at["pgd_asr"]) <= THR)
    pgd_interfere = (joint_best is not None
                     and joint_best["pgd_asr"] > pgd_at["pgd_asr"] + THR)

    corr_stack    = (joint_best is not None
                     and joint_best["corr_avg"] > augmix_only["corr_avg"] + THR)
    corr_match    = (joint_best is not None
                     and abs(joint_best["corr_avg"] - augmix_only["corr_avg"]) <= THR)
    corr_interfere = (joint_best is not None
                      and joint_best["corr_avg"] + THR < augmix_only["corr_avg"])

    out("")
    out(f"  PGD: joint vs PGD-AT delta = {joint_best['pgd_asr']-pgd_at['pgd_asr']:+.4f} "
        f"({'STACK' if pgd_stack else 'INTERFERE' if pgd_interfere else 'MATCH'})")
    out(f"  COR: joint vs AugMix delta = {joint_best['corr_avg']-augmix_only['corr_avg']:+.4f} "
        f"({'STACK' if corr_stack else 'INTERFERE' if corr_interfere else 'MATCH'})")
    out("")

    if pgd_stack and corr_stack:
        verdict = ("REFUTED (positive surprise): joint AugMix+PGD-AT STACKS on BOTH "
                   "Linf-PGD and corruption robustness beyond either alone. "
                   "Hypothesis predicted MAX-not-SUM; observed SUM. Cross-threat "
                   "defences are complementary.")
    elif pgd_match and corr_match:
        verdict = ("CONFIRMED: joint training neither stacks nor interferes - each "
                   "defence saturates its own threat model (PGD~=PGD-AT, "
                   "corr~=AugMix). Joint training is essentially MAX, supporting "
                   "the threat-model-separation prediction.")
    elif pgd_interfere or corr_interfere:
        verdict = ("PARTIAL/INTERFERE: joint training trades off one threat for the "
                   "other - AugMix and PGD-AT optimize incompatible objectives at "
                   "this eps/magnitude. Best joint config does not Pareto-dominate "
                   "the singletons.")
    elif pgd_stack and not corr_stack:
        verdict = ("PARTIAL: joint improves Linf-PGD beyond PGD-AT but does not "
                   "improve corruption robustness beyond AugMix-only. Asymmetric "
                   "stacking: AugMix helps AT but AT does not help corruptions here.")
    elif corr_stack and not pgd_stack:
        verdict = ("PARTIAL: joint improves corruption robustness beyond AugMix-only "
                   "but matches (not beats) PGD-AT on Linf. Asymmetric stacking in "
                   "the other direction: PGD-AT helps corruptions but AugMix does "
                   "not help Linf.")
    else:
        verdict = ("MIXED: see deltas above. Joint training does not cleanly stack, "
                   "interfere, or match on both axes simultaneously.")

    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time()-t0:.1f}s")
    flush()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
