"""
H454 - CutMix-AT joint training (Fashion-MNIST).

Hypothesis (seed): "CutMix-AT joint - gap G3 - improves H432 - patch-prob in
{0.25, 0.5, 0.75}".

Critique of the seed and of H432 ----------------------------------------------
H432 found CutMix standalone gives little PGD robustness vs baseline (it is a
clean-acc augmentation, not an adversarial defence). The interesting question
is whether CutMix STACKS with PGD-AT (Madry 2018). The naive recipe -- "do
PGD-AT, and apply CutMix to the clean image before computing the AT loss" --
has a label-mismatch failure mode: CutMix labels are convex combinations
(ya, yb, lam), while PGD-AT's inner-max is usually run against a HARD label.
If the inner max maximises CE(model(x_adv), ya) only, the attack ignores yb and
the defence is effectively only protecting one branch of the mixed label. This
loss-mismatch is precisely the source of the "mixup-AT does not help" finding
in Pang et al. 2022 (Bag of Tricks for AT) and the AdvCutMix proposal
(IIeon / Lee et al. 2021) that the inner attack must use the MIXED label too.

We therefore implement CutMix-AT carefully:
  * Inner max: PGD on the CutMix-mixed image with the MIXED CE loss
    lam*CE(f(x_adv), ya) + (1-lam)*CE(f(x_adv), yb). This is the "mixed-label
    inner" condition.
  * Outer min: same mixed-CE loss on the adversarial mixed image.
  * Ablation: also run inner max with HARD label (ya only) to confirm the
    mismatch hurts.

We additionally introduce a PATCH-PROBABILITY knob p in {0.25, 0.5, 0.75}: a
fraction p of batches use CutMix; the remaining (1-p) are plain PGD-AT. p=0
recovers PGD-AT baseline, p=1 recovers "always CutMix-AT".

Controls
  C0  PGD-AT baseline                            (p=0 in spirit)
  C1  CutMix-only (no AT)                        -- replicates H432 cutmix arm
  C2  CutMix-AT, p=0.25, mixed-label inner       -- main condition
  C3  CutMix-AT, p=0.50, mixed-label inner       -- main condition
  C4  CutMix-AT, p=0.75, mixed-label inner       -- main condition
  C5  CutMix-AT, p=0.50, HARD-label inner        -- ablation (mismatch)

Eval: clean acc, FGSM ASR, PGD ASR (10-step, eps=0.1) on a held-out test set.

Extra papers consulted (beyond Yun 2019 CutMix and Madry 2018 PGD-AT)
  * Lee et al. 2020/2021 "AdvCutMix" / mixup-AT analyses -- argues inner max
    must use the SAME mixed label as the outer loss to avoid bias.
  * Pang et al. 2022 "Bag of Tricks for Adversarial Training" -- finds naive
    mixup/cutmix with AT can REDUCE robustness because the convex labels
    weaken the inner attack signal. Motivates the hard-label ablation here.
  * Zhang et al. 2018 (mixup), Yun et al. 2019 (CutMix) for the augmentations
    themselves.

Verdict rule
  SUPPORTED  : best CutMix-AT (mixed inner) beats PGD-AT baseline PGD ASR by
               >= 0.02 AND clean acc within 0.02 of baseline.
  PARTIAL    : best CutMix-AT improves PGD ASR by 0.005-0.02 OR loses
               <= 0.02 PGD but gains >= 0.01 clean.
  NOT SUPPORTED : best CutMix-AT does not improve PGD ASR vs PGD-AT baseline.
  (Hard-label ablation reported separately; if it loses vs mixed-label inner,
   that confirms the label-mismatch story.)

Config: N_TRAIN=6000, N_EVAL=2000, EPOCHS=10, LR=0.05, BATCH=128,
        SGD(mom=0.9, wd=5e-4), SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01,
        CUTMIX_ALPHA=1.0 (Beta(1,1)=Uniform[0,1]).
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
CUTMIX_ALPHA = 1.0
PATCH_PROBS = [0.25, 0.5, 0.75]

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h454_cutmix_at_joint_output.txt",
)


# ---- helpers -----------------------------------------------------------------

def _make_sgd(model):
    return torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=5e-4,
    )


def cutmix_batch(xb, yb, alpha, rng):
    """CutMix (Yun et al. 2019). Returns (x_mix, ya, yb, lam_area)."""
    lam = float(rng.beta(alpha, alpha))
    B, _, H, W = xb.shape
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx = int(rng.integers(0, W))
    cy = int(rng.integers(0, H))
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)
    idx = torch.randperm(B, device=xb.device)
    x_mix = xb.clone()
    x_mix[:, :, y1:y2, x1:x2] = xb[idx, :, y1:y2, x1:x2]
    lam_area = 1.0 - float((x2 - x1) * (y2 - y1)) / float(H * W)
    ya, yb2 = yb, yb[idx]
    return x_mix, ya, yb2, lam_area


def mixed_ce(logits, ya, yb, lam):
    return lam * F.cross_entropy(logits, ya) + (1.0 - lam) * F.cross_entropy(logits, yb)


def pgd_mixed(model, x, ya, yb, lam, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """PGD inner max using the MIXED-LABEL CE loss."""
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = mixed_ce(model(xa), ya, yb, lam)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def pgd_hard(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """Standard PGD against the dominant hard label only (ablation)."""
    return C.pgd(model, x, y, eps=eps, steps=steps, alpha=alpha)


# ---- training procedures -----------------------------------------------------

def train_pgd_at(Xtr, Ytr, seed):
    """C0: plain PGD-AT baseline (no CutMix)."""
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
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            opt.zero_grad()
            F.cross_entropy(model(xa), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_cutmix_only(Xtr, Ytr, seed):
    """C1: CutMix standalone (no AT) -- replicates H432 cutmix arm."""
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
            mixed_ce(model(x_mix), ya, yb2, lam).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_cutmix_at(Xtr, Ytr, seed, patch_prob, inner="mixed"):
    """C2-C5: CutMix-AT joint.

    With prob `patch_prob` a batch is CutMix-mixed and PGD-attacked using the
    `inner` loss ("mixed" = mixed-label CE, "hard" = ya-only CE). With prob
    (1 - patch_prob) the batch is plain PGD-AT on the original (xb, yb).
    """
    assert inner in ("mixed", "hard")
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
            use_cutmix = rng.random() < patch_prob
            if use_cutmix:
                x_mix, ya, yb2, lam = cutmix_batch(xb, yb, CUTMIX_ALPHA, rng)
                if inner == "mixed":
                    xa = pgd_mixed(model, x_mix, ya, yb2, lam)
                else:  # "hard" -- attack only against the dominant label ya
                    xa = pgd_hard(model, x_mix, ya)
                opt.zero_grad()
                # Outer min always uses mixed CE on the (now adversarial) mix.
                mixed_ce(model(xa), ya, yb2, lam).backward()
                opt.step()
            else:
                xa = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
                opt.zero_grad()
                F.cross_entropy(model(xa), yb).backward()
                opt.step()
        sched.step()
    model.eval()
    return model


# ---- evaluation --------------------------------------------------------------

def eval_robustness(model, X, Y, label):
    _, acc = C.logits_and_acc(model, X, Y)
    fg = C.attack_success(model, X, Y, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, X, Y, attack="pgd",  eps=EPS, steps=PGD_STEPS)
    return {"label": label, "acc": acc, "fgsm": fg["asr"], "pgd": pg["asr"]}


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
    out("H454  CutMix-AT joint training (Fashion-MNIST)")
    out("Yun 2019 CutMix x Madry 2018 PGD-AT; inner-loss ablation per")
    out("Lee 2021 AdvCutMix / Pang 2022 Bag-of-Tricks.")
    out("=" * 80)
    out(f"config: DS={DS} N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} "
        f"LR={LR} BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} "
        f"CUTMIX_ALPHA={CUTMIX_ALPHA}")
    out(f"        PATCH_PROBS={PATCH_PROBS}")
    out(f"        device={C.DEVICE}")
    out("")
    out("Hypothesis: CutMix STACKS with PGD-AT only when the inner max uses the")
    out("MIXED label; without that, label mismatch nullifies or hurts AT.")
    out("Patch-prob in {0.25, 0.5, 0.75} controls how often CutMix is applied.")
    out("")

    # ---- data ----------------------------------------------------------------
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    flush_file()

    results = []

    # ---- C0: PGD-AT baseline -------------------------------------------------
    out("\n[C0] PGD-AT baseline (no CutMix)...")
    m0 = train_pgd_at(Xtr, Ytr, SEED)
    r0 = eval_robustness(m0, Xte, Yte, "pgd_at_baseline")
    results.append(r0)
    out(f"    clean_acc={r0['acc']:.4f}  FGSM_ASR={r0['fgsm']:.4f}  "
        f"PGD_ASR={r0['pgd']:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- C1: CutMix only (no AT) --------------------------------------------
    out("\n[C1] CutMix only, NO AT (replicates H432 cutmix arm)...")
    m1 = train_cutmix_only(Xtr, Ytr, SEED)
    r1 = eval_robustness(m1, Xte, Yte, "cutmix_only_no_at")
    results.append(r1)
    out(f"    clean_acc={r1['acc']:.4f}  FGSM_ASR={r1['fgsm']:.4f}  "
        f"PGD_ASR={r1['pgd']:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- C2-C4: CutMix-AT, mixed-label inner, patch_prob sweep --------------
    sweep_results = {}
    for p in PATCH_PROBS:
        out(f"\n[CutMix-AT p={p} mixed-label inner]...")
        m = train_cutmix_at(Xtr, Ytr, SEED, patch_prob=p, inner="mixed")
        r = eval_robustness(m, Xte, Yte, f"cutmix_at_p{p:.2f}_mixed")
        results.append(r)
        sweep_results[p] = r
        out(f"    clean_acc={r['acc']:.4f}  FGSM_ASR={r['fgsm']:.4f}  "
            f"PGD_ASR={r['pgd']:.4f}  ({time.time()-t0:.0f}s)")
        flush_file()

    # ---- C5: ablation -- hard-label inner at the middle p -------------------
    p_abl = 0.5
    out(f"\n[C5] CutMix-AT p={p_abl} HARD-label inner (ablation)...")
    m_abl = train_cutmix_at(Xtr, Ytr, SEED, patch_prob=p_abl, inner="hard")
    r_abl = eval_robustness(m_abl, Xte, Yte, f"cutmix_at_p{p_abl:.2f}_hard")
    results.append(r_abl)
    out(f"    clean_acc={r_abl['acc']:.4f}  FGSM_ASR={r_abl['fgsm']:.4f}  "
        f"PGD_ASR={r_abl['pgd']:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- summary table -------------------------------------------------------
    out("\n" + "=" * 80)
    out("MAIN TABLE")
    out("=" * 80)
    hdr = "{:<32} {:>10} {:>10} {:>10}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in results:
        out("{:<32} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["label"], r["acc"], r["fgsm"], r["pgd"]))
    out("-" * len(hdr))

    # ---- analysis ------------------------------------------------------------
    pgd_baseline = r0["pgd"]
    acc_baseline = r0["acc"]
    best_p = min(sweep_results, key=lambda p: sweep_results[p]["pgd"])
    r_best = sweep_results[best_p]
    pgd_gain  = pgd_baseline - r_best["pgd"]
    acc_delta = r_best["acc"] - acc_baseline
    pgd_hard_vs_mixed = r_abl["pgd"] - sweep_results[p_abl]["pgd"]

    out("\n" + "=" * 80)
    out("ANALYSIS")
    out("=" * 80)
    out(f"  PGD-AT baseline:        PGD_ASR={pgd_baseline:.4f}  "
        f"clean={acc_baseline:.4f}")
    out(f"  best CutMix-AT (mixed): p={best_p}  PGD_ASR={r_best['pgd']:.4f}  "
        f"clean={r_best['acc']:.4f}")
    out(f"  CutMix-AT gain vs PGD-AT (PGD_ASR drop) = {pgd_gain:+.4f}")
    out(f"  CutMix-AT clean delta vs PGD-AT         = {acc_delta:+.4f}")
    out(f"  p={p_abl}  hard-inner PGD_ASR - mixed-inner PGD_ASR "
        f"= {pgd_hard_vs_mixed:+.4f}  "
        f"({'hard inner is WORSE -> mismatch confirmed' if pgd_hard_vs_mixed > 0.005 else 'no mismatch effect' if abs(pgd_hard_vs_mixed) <= 0.005 else 'hard inner is BETTER -- surprising'})")

    # ---- verdict -------------------------------------------------------------
    out("\n" + "=" * 80)
    out("VERDICT")
    out("=" * 80)

    clean_ok = abs(acc_delta) <= 0.02
    if pgd_gain >= 0.02 and clean_ok:
        verdict = (f"SUPPORTED: CutMix-AT (mixed-label inner, p={best_p}) reduces PGD "
                   f"ASR by {pgd_gain:+.4f} vs PGD-AT baseline with clean acc "
                   f"within 0.02 -- CutMix stacks with AT when the inner attack "
                   f"uses the mixed label (Lee 2021 AdvCutMix).")
    elif pgd_gain >= 0.005:
        verdict = (f"PARTIAL: CutMix-AT (p={best_p}) reduces PGD ASR by {pgd_gain:+.4f} "
                   f"vs PGD-AT (within 0.005-0.02 band, single seed) -- "
                   f"weak stacking signal, plausibly noise.")
    elif pgd_gain <= -0.005 or acc_delta <= -0.02:
        verdict = (f"NOT SUPPORTED: CutMix-AT does not improve over PGD-AT "
                   f"(PGD_ASR delta={pgd_gain:+.4f}, clean delta={acc_delta:+.4f}). "
                   f"Consistent with Pang 2022 Bag-of-Tricks: mixup/cutmix does "
                   f"not stack with AT at this scale.")
    else:
        verdict = (f"NULL: CutMix-AT ties PGD-AT (PGD_ASR delta={pgd_gain:+.4f}, "
                   f"clean delta={acc_delta:+.4f}). No stacking benefit observed.")
    out("  VERDICT: " + verdict)

    if pgd_hard_vs_mixed > 0.005:
        out("  ABLATION: hard-label inner under-performs mixed-label inner by "
            f"{pgd_hard_vs_mixed:+.4f} PGD_ASR -- label-mismatch story SUPPORTED.")
    elif abs(pgd_hard_vs_mixed) <= 0.005:
        out("  ABLATION: hard- vs mixed-label inner are indistinguishable "
            f"({pgd_hard_vs_mixed:+.4f}) -- label-mismatch story NOT SUPPORTED at this scale.")
    else:
        out("  ABLATION: hard-label inner OUT-performs mixed by "
            f"{-pgd_hard_vs_mixed:+.4f} -- contrary to AdvCutMix prediction.")

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
