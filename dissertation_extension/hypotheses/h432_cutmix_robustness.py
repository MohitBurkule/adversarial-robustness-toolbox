"""
H432 - CutMix vs Mixup vs Baseline robustness comparison (Fashion-MNIST).

Hypothesis: CutMix's patch-based mixing creates training data on a different
manifold than Mixup's linear interpolation — different robustness profile.
Specifically, CutMix pastes a rectangular region from one image onto another
and mixes labels proportional to patch area; Mixup blends pixels globally with
a scalar lambda. We test whether these geometrically distinct augmentations
yield meaningfully different adversarial robustness.

Reference: Yun et al. (2019). "CutMix: Training strategy where random patches
are cut and pasted among training images." ICCV 2019.

Design:
  1. Baseline  — standard training, no augmentation.
  2. Mixup     — global pixel blend with label mix (Zhang et al. 2018).
  3. CutMix    — rectangular patch swap with area-proportional label mix
                 (Yun et al. 2019). Patch aspect ratio uniformly in [0.3, 3.3],
                 patch area = lambda * H * W where lambda ~ Beta(1, 1) = Uniform.
  Each condition trained from scratch (same seed, same CNN width=32).
  Eval: clean acc, FGSM ASR, PGD ASR on the held-out test set.
  Transfer test: adversarial examples crafted on the Mixup model transferred
  to evaluate against the CutMix model and vice-versa (transfer ASR).

Config: N_TRAIN=6000, N_EVAL=2000, EPOCHS=10, LR=0.05, BATCH=128,
        SGD mom=0.9 wd=5e-4, SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01,
        MIXUP_ALPHA=1.0, CUTMIX_ALPHA=1.0 (both Beta(1,1) = Uniform[0,1]).
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
MIXUP_ALPHA = 1.0    # Beta(alpha, alpha); alpha=1 => Uniform[0,1]
CUTMIX_ALPHA = 1.0   # same

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h432_cutmix_robustness_output.txt",
)

META = {"channels": 1, "size": 28, "n_classes": 10}


# ---- augmentation helpers ----------------------------------------------------

def mixup_batch(xb, yb, alpha, rng):
    """Return mixed (x, ya, yb, lam) for a batch. lam ~ Beta(alpha, alpha)."""
    lam = float(rng.beta(alpha, alpha))
    idx = torch.randperm(xb.size(0), device=xb.device)
    xa, xb2 = xb, xb[idx]
    ya, yb2 = yb, yb[idx]
    x_mix = lam * xa + (1.0 - lam) * xb2
    return x_mix, ya, yb2, lam


def cutmix_batch(xb, yb, alpha, rng):
    """CutMix (Yun et al. 2019). Returns (x_mix, ya, yb, lam_area)."""
    lam = float(rng.beta(alpha, alpha))
    B, C, H, W = xb.shape

    # --- bounding box (Yun et al. eq. 1) ---
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx = rng.integers(0, W)
    cy = rng.integers(0, H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    idx = torch.randperm(B, device=xb.device)
    x_mix = xb.clone()
    x_mix[:, :, y1:y2, x1:x2] = xb[idx, :, y1:y2, x1:x2]

    # actual area ratio of the patch
    lam_area = 1.0 - float((x2 - x1) * (y2 - y1)) / float(H * W)
    ya, yb2 = yb, yb[idx]
    return x_mix, ya, yb2, lam_area


def mixed_loss(logits, ya, yb, lam):
    """Convex combination of cross-entropy losses."""
    return lam * F.cross_entropy(logits, ya) + (1.0 - lam) * F.cross_entropy(logits, yb)


# ---- training ----------------------------------------------------------------

def _make_sgd(model):
    return torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=5e-4,
    )


def train_baseline(Xtr, Ytr, seed):
    """Standard training, no augmentation."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_sgd(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_mixup(Xtr, Ytr, seed):
    """Mixup training (Zhang et al. 2018)."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_sgd(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            x_mix, ya, yb2, lam = mixup_batch(xb, yb, MIXUP_ALPHA, rng)
            opt.zero_grad()
            mixed_loss(model(x_mix), ya, yb2, lam).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_cutmix(Xtr, Ytr, seed):
    """CutMix training (Yun et al. 2019)."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_sgd(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            x_mix, ya, yb2, lam = cutmix_batch(xb, yb, CUTMIX_ALPHA, rng)
            opt.zero_grad()
            mixed_loss(model(x_mix), ya, yb2, lam).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- evaluation --------------------------------------------------------------

def eval_robustness(model, X, Y, label):
    """Return dict with clean_acc, fgsm_asr, pgd_asr."""
    _, acc = C.logits_and_acc(model, X, Y)
    fg = C.attack_success(model, X, Y, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, X, Y, attack="pgd",  eps=EPS, steps=PGD_STEPS)
    return {"label": label, "acc": acc, "fgsm": fg["asr"], "pgd": pg["asr"]}


def transfer_asr(src_model, tgt_model, X, Y, attack="pgd"):
    """Craft adversarial examples on src_model, evaluate against tgt_model."""
    if attack == "pgd":
        adv = C.pgd_attack(src_model, X, Y, eps=EPS, steps=PGD_STEPS,
                           alpha=PGD_ALPHA)
    else:
        adv = C.fgsm_attack(src_model, X, Y, eps=EPS)
    with torch.no_grad():
        preds = tgt_model(adv).argmax(1)
    asr = float((preds != Y).float().mean())
    return asr


# ---- main --------------------------------------------------------------------

def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H432  CutMix vs Mixup vs Baseline robustness (Fashion-MNIST)")
    out("Yun et al. 2019 — CutMix: Training strategy where random patches are")
    out("cut and pasted among training images.")
    out("=" * 80)
    out(f"config: DS={DS} N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} "
        f"LR={LR} BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        MIXUP_ALPHA={MIXUP_ALPHA} CUTMIX_ALPHA={CUTMIX_ALPHA} "
        f"(both Beta(a,a)=Uniform[0,1])")
    out(f"        device={C.DEVICE}")
    out("")
    out("Hypothesis: CutMix's patch-based mixing creates training data on a")
    out("different manifold than Mixup's linear interpolation, yielding a")
    out("different adversarial robustness profile.")
    out("")

    # ---- data ----------------------------------------------------------------
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                         seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    # ---- train three models --------------------------------------------------
    out("\n[1] Training BASELINE (standard, no augmentation)...")
    m_base = train_baseline(Xtr, Ytr, SEED)
    r_base = eval_robustness(m_base, Xte, Yte, "baseline")
    out(f"    clean_acc={r_base['acc']:.4f}  FGSM_ASR={r_base['fgsm']:.4f}  "
        f"PGD_ASR={r_base['pgd']:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    out("\n[2] Training MIXUP (Zhang et al. 2018, alpha=1.0)...")
    m_mix = train_mixup(Xtr, Ytr, SEED)
    r_mix = eval_robustness(m_mix, Xte, Yte, "mixup")
    out(f"    clean_acc={r_mix['acc']:.4f}  FGSM_ASR={r_mix['fgsm']:.4f}  "
        f"PGD_ASR={r_mix['pgd']:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    out("\n[3] Training CUTMIX (Yun et al. 2019, alpha=1.0)...")
    m_cut = train_cutmix(Xtr, Ytr, SEED)
    r_cut = eval_robustness(m_cut, Xte, Yte, "cutmix")
    out(f"    clean_acc={r_cut['acc']:.4f}  FGSM_ASR={r_cut['fgsm']:.4f}  "
        f"PGD_ASR={r_cut['pgd']:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- transfer attacks ----------------------------------------------------
    out("\n[4] Transfer attack (PGD): craft on Mixup -> evaluate on CutMix...")
    t_mix2cut = transfer_asr(m_mix, m_cut, Xte, Yte, attack="pgd")
    out(f"    Mixup->CutMix transfer PGD_ASR = {t_mix2cut:.4f}")

    out("\n[5] Transfer attack (PGD): craft on CutMix -> evaluate on Mixup...")
    t_cut2mix = transfer_asr(m_cut, m_mix, Xte, Yte, attack="pgd")
    out(f"    CutMix->Mixup transfer PGD_ASR = {t_cut2mix:.4f}")

    out("\n[6] Transfer attack (PGD): craft on Baseline -> Mixup...")
    t_base2mix = transfer_asr(m_base, m_mix, Xte, Yte, attack="pgd")
    out(f"    Baseline->Mixup transfer PGD_ASR = {t_base2mix:.4f}")

    out("\n[7] Transfer attack (PGD): craft on Baseline -> CutMix...")
    t_base2cut = transfer_asr(m_base, m_cut, Xte, Yte, attack="pgd")
    out(f"    Baseline->CutMix transfer PGD_ASR = {t_base2cut:.4f}")
    flush_file()

    # ---- summary table -------------------------------------------------------
    out("\n" + "=" * 80)
    out("[8] MAIN TABLE")
    out("=" * 80)
    hdr = "{:<12} {:>10} {:>10} {:>10}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in [r_base, r_mix, r_cut]:
        out("{:<12} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["label"], r["acc"], r["fgsm"], r["pgd"]))
    out("-" * len(hdr))

    out("\n[9] TRANSFER TABLE (PGD, source -> target)")
    thdr = "{:<22} {:>10}".format("transfer", "ASR")
    out(thdr)
    out("-" * len(thdr))
    for label, asr in [
        ("Baseline->Mixup",  t_base2mix),
        ("Baseline->CutMix", t_base2cut),
        ("Mixup->CutMix",    t_mix2cut),
        ("CutMix->Mixup",    t_cut2mix),
    ]:
        out("{:<22} {:>10.4f}".format(label, asr))
    out("-" * len(thdr))

    # ---- verdict -------------------------------------------------------------
    out("\n" + "=" * 80)
    out("[10] VERDICT")
    out("=" * 80)

    pgd_base = r_base["pgd"]
    pgd_mix  = r_mix["pgd"]
    pgd_cut  = r_cut["pgd"]

    mix_gain = pgd_base - pgd_mix    # positive => Mixup more robust
    cut_gain = pgd_base - pgd_cut    # positive => CutMix more robust
    profile_diff = abs(pgd_mix - pgd_cut)
    transfer_asymmetry = abs(t_mix2cut - t_cut2mix)

    out(f"  PGD_ASR: baseline={pgd_base:.4f}  mixup={pgd_mix:.4f}  "
        f"cutmix={pgd_cut:.4f}")
    out(f"  Mixup robustness gain vs baseline  = {mix_gain:+.4f}")
    out(f"  CutMix robustness gain vs baseline = {cut_gain:+.4f}")
    out(f"  |Mixup PGD_ASR - CutMix PGD_ASR|  = {profile_diff:.4f} "
        f"({'DIFFERENT profile (>0.02)' if profile_diff > 0.02 else 'SIMILAR profile (<=0.02)'})")
    out(f"  transfer asymmetry |mix->cut - cut->mix| = {transfer_asymmetry:.4f}")
    out("")

    # hypothesis evaluation
    different_profile = profile_diff > 0.02
    either_gains = mix_gain > 0.01 or cut_gain > 0.01

    if different_profile and either_gains:
        verdict = ("SUPPORTED: CutMix and Mixup produce distinct robustness profiles "
                   "(|ΔPGD_ASR| > 0.02) and at least one improves over baseline — "
                   "consistent with different training manifolds (Yun et al. 2019).")
    elif different_profile and not either_gains:
        verdict = ("PARTIAL: CutMix and Mixup differ from each other (|ΔPGD_ASR| > 0.02) "
                   "but neither meaningfully improves on baseline — different manifolds, "
                   "no robustness benefit.")
    elif not different_profile and either_gains:
        verdict = ("PARTIAL: Both augmentations improve robustness similarly "
                   "(|ΔPGD_ASR| <= 0.02) — same effect size, no profile distinction.")
    else:
        verdict = ("NOT SUPPORTED: CutMix and Mixup show similar robustness to baseline "
                   "and to each other — no manifold-difference effect detected.")

    out("  VERDICT: " + verdict)
    out("")
    out(f"done in {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
