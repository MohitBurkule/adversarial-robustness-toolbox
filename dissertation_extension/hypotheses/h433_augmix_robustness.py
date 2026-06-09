"""
H433 - AugMix consistency loss for adversarial robustness (Fashion-MNIST).

Reference: Hendrycks et al. (2020) "AugMix: A Simple Data Processing Method to
Improve Robustness and Uncertainty" (ICLR 2020).

AugMix works by:
  1. Producing k augmentation chains, each a random composition of simple
     augmentation ops (translate, rotate, shear, contrast).
  2. Taking a convex mixture of the k augmented images.
  3. Adding a Jensen-Shannon consistency loss that enforces agreement between
     the model's predictions on the original image and on each augmented mixture.

Hypothesis: "Consistency loss over diverse augmentation chains hardens a model
against natural distribution shifts AND may transfer to L-inf adversarial
perturbations, because it flattens the loss landscape near training points."

Augmentation ops (pure PyTorch, no external library):
  - translate_x / translate_y  (± MAX_TRANSLATE fraction)
  - rotate                     (± MAX_ROTATE degrees)
  - shear_x                    (± MAX_SHEAR degrees)
  - contrast                   (scale pixel intensities toward / away from 0.5)

Experimental conditions (all retrain from scratch, same CNN width=32):
  A. BASELINE  — standard cross-entropy, no augmentation.
  B. AUGONLY   — same augmented images used in AugMix, but NO consistency loss
                 (ablates whether diversity alone drives robustness).
  C. AUGMIX    — full AugMix: cross-entropy on original + JSD consistency loss
                 weighted by LAMBDA_JSD.

Metrics: clean acc, FGSM ASR, PGD ASR (eps=0.1, pgd steps=10).

Config: N_TRAIN=6000, N_EVAL=2000, EPOCHS=15, LR=0.05, BATCH=128,
        SGD(mom=0.9, wd=5e-4), SEED=0, K_CHAINS=3, ALPHA=1.0, LAMBDA_JSD=12.
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
EPOCHS = 15
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01

# AugMix hyper-params
K_CHAINS = 3       # number of augmentation chains per image
ALPHA = 1.0        # Dirichlet / Beta concentration for mixing weights
LAMBDA_JSD = 12.0  # weight on JSD consistency loss (Hendrycks 2020 default)

# Augmentation magnitude limits
MAX_TRANSLATE = 0.15   # fraction of image size
MAX_ROTATE = 30.0      # degrees
MAX_SHEAR = 15.0       # degrees
MAX_CONTRAST = 0.5     # scale factor shift around 0.5

OPS_PER_CHAIN = 3      # number of ops drawn per chain

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h433_augmix_robustness_output.txt")


# ---- pure-PyTorch augmentation ops ------------------------------------------

def _affine(x, angle=0.0, translate=(0.0, 0.0), shear=0.0):
    """Apply affine transform to a batch (N,C,H,W) in [0,1]."""
    N, C, H, W = x.shape
    angle_r = math.radians(angle)
    shear_r = math.radians(shear)
    cos_a, sin_a = math.cos(angle_r), math.sin(angle_r)
    # build 2x3 matrix (rotation + shear + translation)
    m00 = cos_a
    m01 = -sin_a + math.tan(shear_r) * cos_a
    m10 = sin_a
    m11 = cos_a + math.tan(shear_r) * sin_a
    tx = translate[0] * 2.0   # grid_sample uses [-1,1] space
    ty = translate[1] * 2.0
    theta = torch.tensor([[m00, m01, tx],
                           [m10, m11, ty]],
                          dtype=x.dtype, device=x.device).unsqueeze(0).expand(N, -1, -1)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=False)


def aug_translate_x(x, magnitude):
    return _affine(x, translate=(magnitude * MAX_TRANSLATE, 0.0))


def aug_translate_y(x, magnitude):
    return _affine(x, translate=(0.0, magnitude * MAX_TRANSLATE))


def aug_rotate(x, magnitude):
    return _affine(x, angle=magnitude * MAX_ROTATE)


def aug_shear_x(x, magnitude):
    return _affine(x, shear=magnitude * MAX_SHEAR)


def aug_contrast(x, magnitude):
    """Scale pixel intensities toward (magnitude<0) or away from (magnitude>0) 0.5."""
    factor = 1.0 + magnitude * MAX_CONTRAST
    return ((x - 0.5) * factor + 0.5).clamp(0, 1)


_ALL_OPS = [aug_translate_x, aug_translate_y, aug_rotate, aug_shear_x, aug_contrast]


def _random_chain(x, rng: torch.Generator, n_ops: int = OPS_PER_CHAIN):
    """Apply a random chain of n_ops augmentation ops with uniform magnitudes in [-1,1]."""
    op_indices = torch.randint(len(_ALL_OPS), (n_ops,), generator=rng).tolist()
    magnitudes = (torch.rand(n_ops, generator=rng) * 2.0 - 1.0).tolist()
    out = x
    for idx, mag in zip(op_indices, magnitudes):
        out = _ALL_OPS[idx](out, mag)
    return out


def augmix_batch(x, rng: torch.Generator, k: int = K_CHAINS, alpha: float = ALPHA):
    """Return (x_orig, [x_aug_1, ..., x_aug_k], x_mix).

    x_mix is a convex combination of the k augmented images using Dirichlet weights,
    then blended with the original using a Beta(alpha,alpha) coefficient (Hendrycks 2020).
    """
    N = x.size(0)
    # Dirichlet weights via Gamma sampling: w_i = g_i / sum(g_j)
    # Approximate with symmetric Dirichlet(alpha): sample k Gamma(alpha,1) rvs
    gammas = torch.stack(
        [torch._standard_gamma(  # noqa: SLF001  (private but stable)
            torch.full((N,), alpha, device=x.device, dtype=x.dtype)) for _ in range(k)],
        dim=1)  # (N, k)
    w = gammas / gammas.sum(dim=1, keepdim=True)  # (N, k) Dirichlet weights

    aug_views = [_random_chain(x, rng) for _ in range(k)]
    # weighted sum of augmented views
    x_aug_mix = sum(
        w[:, i].view(N, 1, 1, 1) * aug_views[i] for i in range(k))

    # blend coefficient m ~ Beta(alpha, alpha): treat as uniform in [0,1] for simplicity
    m = torch.rand(N, 1, 1, 1, generator=rng, device=x.device, dtype=x.dtype) * 0.5  # [0,0.5]
    x_mix = (1.0 - m) * x + m * x_aug_mix
    return x, aug_views, x_mix.clamp(0, 1)


def jsd_loss(logits_orig, logits_views):
    """Jensen-Shannon divergence consistency loss (Hendrycks 2020 eq. 2).

    M = mean of all distributions (orig + each aug view).
    JSD = (1/(k+1)) * sum_i KL(p_i || M)
    """
    all_logits = [logits_orig] + logits_views  # list of (N, C)
    all_probs = [F.softmax(lg, dim=1) for lg in all_logits]
    M = torch.stack(all_probs, dim=0).mean(dim=0)  # (N, C)
    log_M = M.clamp(1e-8).log()
    kl_sum = sum(F.kl_div(log_M, p, reduction="batchmean") for p in all_probs)
    return kl_sum / len(all_logits)


# ---- training routines -------------------------------------------------------

def _make_opt(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)


def train_baseline(Xtr, Ytr, seed):
    """Standard cross-entropy, no augmentation."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_augonly(Xtr, Ytr, seed):
    """Cross-entropy on augmented images only — no JSD consistency loss.

    For each mini-batch we replace x with x_mix (the AugMix mixture) but compute
    only the standard CE loss, to isolate whether augmentation diversity alone helps.
    """
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    rng = torch.Generator(device=Xtr.device).manual_seed(seed + 1)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            _, _views, x_mix = augmix_batch(xb, rng)
            opt.zero_grad()
            F.cross_entropy(model(x_mix), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_augmix(Xtr, Ytr, seed):
    """Full AugMix: CE on original + lambda * JSD consistency loss (Hendrycks 2020)."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    rng = torch.Generator(device=Xtr.device).manual_seed(seed + 2)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            x_orig, aug_views, _x_mix = augmix_batch(xb, rng)
            opt.zero_grad()
            ce = F.cross_entropy(model(x_orig), yb)
            jsd = jsd_loss(model(x_orig), [model(v) for v in aug_views])
            loss = ce + LAMBDA_JSD * jsd
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_robustness(model, X, Y):
    """Return (clean_acc, fgsm_asr, pgd_asr)."""
    _, acc = C.logits_and_acc(model, X, Y)
    fg = C.attack_success(model, X, Y, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


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
    out("H433  AugMix consistency loss for adversarial robustness (Fashion-MNIST)")
    out("=" * 80)
    out("Ref: Hendrycks et al. (2020) 'AugMix: A Simple Data Processing Method to")
    out("     Improve Robustness and Uncertainty'. ICLR 2020.")
    out("")
    out("Hypothesis: consistency loss on diverse augmentation chains hardens against")
    out("natural distribution shifts AND may transfer to L-inf adversarial perturbations.")
    out("")
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4)")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} SEED={SEED}")
    out(f"        K_CHAINS={K_CHAINS} ALPHA={ALPHA} LAMBDA_JSD={LAMBDA_JSD}")
    out(f"        OPS: translate_x/y (±{MAX_TRANSLATE}), rotate (±{MAX_ROTATE}°), "
        f"shear_x (±{MAX_SHEAR}°), contrast (±{MAX_CONTRAST})")
    out(f"        device={C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    conditions = [
        ("BASELINE", train_baseline),
        ("AUGONLY",  train_augonly),
        ("AUGMIX",   train_augmix),
    ]
    rows = []
    for name, train_fn in conditions:
        out(f"[training {name}] ...")
        model = train_fn(Xtr, Ytr, SEED)
        acc, fgsm, pgd = eval_robustness(model, Xte, Yte)
        rows.append({"cond": name, "acc": acc, "fgsm": fgsm, "pgd": pgd})
        out(f"  {name}: clean_acc={acc:.4f}  FGSM_ASR={fgsm:.4f}  PGD_ASR={pgd:.4f}  "
            f"({time.time()-t0:.0f}s)")
        flush()

    # ---- summary table -------------------------------------------------------
    out("")
    out("=" * 80)
    out("SUMMARY TABLE")
    out("=" * 80)
    hdr = "{:<12} {:>10} {:>10} {:>10}".format("condition", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<12} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["cond"], r["acc"], r["fgsm"], r["pgd"]))
    out("-" * len(hdr))

    # ---- verdict -------------------------------------------------------------
    out("")
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)
    base = rows[0]
    augonly = rows[1]
    augmix = rows[2]

    pgd_gain_augonly = base["pgd"] - augonly["pgd"]
    pgd_gain_augmix  = base["pgd"] - augmix["pgd"]
    acc_drop_augmix  = base["acc"] - augmix["acc"]
    jsd_over_augonly = augonly["pgd"] - augmix["pgd"]   # +ve => JSD helps beyond diversity

    out(f"  AUGONLY  PGD_ASR gain vs BASELINE = {pgd_gain_augonly:+.4f}")
    out(f"  AUGMIX   PGD_ASR gain vs BASELINE = {pgd_gain_augmix:+.4f}")
    out(f"  JSD-only contribution (AUGONLY->AUGMIX) = {jsd_over_augonly:+.4f}")
    out(f"  AUGMIX clean-acc drop vs BASELINE  = {acc_drop_augmix:+.4f}")
    out("")

    # one-line verdict
    THR_ROBUST = 0.03
    THR_ACC = 0.02
    jsd_helps   = pgd_gain_augmix > pgd_gain_augonly + THR_ROBUST / 2
    aug_helps   = pgd_gain_augmix > THR_ROBUST
    acc_ok      = acc_drop_augmix < THR_ACC

    if aug_helps and jsd_helps and acc_ok:
        verdict = ("CONFIRMED: AugMix with JSD consistency loss reduces adversarial ASR "
                   "beyond augmentation diversity alone, with acceptable clean-acc cost. "
                   "Hypothesis supported.")
    elif aug_helps and not jsd_helps and acc_ok:
        verdict = ("PARTIAL: Augmentation diversity alone accounts for robustness gain; "
                   "JSD consistency loss adds little extra. AugMix training helps but "
                   "the consistency loss is not the key driver.")
    elif aug_helps and not acc_ok:
        verdict = ("TRADEOFF: AugMix reduces adversarial ASR but at a significant "
                   "clean-accuracy cost (>{:.0f}pp). Robustness gain is real but "
                   "expensive.".format(THR_ACC * 100))
    else:
        verdict = ("NOT CONFIRMED: AugMix's consistency loss does not transfer "
                   "meaningfully to L-inf adversarial robustness on Fashion-MNIST "
                   "(PGD_ASR gain < {:.0f}pp). Hypothesis not supported.".format(
                       THR_ROBUST * 100))

    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time()-t0:.1f}s")
    flush()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
