"""
H474 - Localised adversarial-patch attack on Fashion-MNIST.

usage: .venv/bin/python hypotheses/h474_patch_attack.py 2>&1 | tee \
       results/fashion_mnist/h474_patch_attack_output.txt

Gap: G1 (threat models barely touched), M2 (eval collapsed to Linf-PGD-10).
Anchor: brown-2017-advpatch ("Adversarial Patch", Brown, Mane, Roy, Abadi,
Gilmer, NeurIPS 2017 workshop). The patch is L0-constrained (small spatial
support) but L_inf-UNBOUNDED inside its support: pixels are free to take any
value in [0,1]. This is *exactly* the threat model that the campaign's
Linf-eps-0.1 PGD evaluation does not measure (M2/M3 in CAMPAIGN_GAP_MAP.md).

EXTRA PAPERS (web-searched literature, cited as comments)
---------------------------------------------------------
- karmon-2018-lavan: Karmon, Zoran, Goldberg, "LaVAN: Localized and Visible
  Adversarial Noise", ICML 2018. Tighter optimisation of *localised* patches
  occupying as little as 2% of the image; demonstrates that on ImageNet a
  carefully optimised patch can hit near-100% targeted ASR with ~2% area,
  which is the *direct* motivation for our 2-8 px sweep (the smallest patch
  here, 2x2=4 px, is ~0.5% of a 28x28 image; 4x4=16 px is ~2%; 8x8=64 px is
  ~8%). LaVAN crucially also shows the importance of *learned placement* -
  fixed-corner patches under-estimate the threat.
- wu-2020-defending-physical-patches: Wu, Tong, Liu, Yu, Yi, Goldstein,
  "Defending Against Physical Adversarial Patches via PatchGuard". Shows
  certifiable defences exist but rely on small-receptive-field architectures
  that cap mask coverage. Our SmallCNN is the OPPOSITE: max-pool stack with
  large effective RF, so we predict it is highly patch-vulnerable.
- xiang-2021-patchguard: Xiang, Bhagoji, Sehwag, Mittal, "PatchGuard: A
  Provably Robust Defense against Adversarial Patches via Small Receptive
  Fields and Masking", USENIX Security 2021. Same authors; canonical
  certified-patch defence.
- salman-2022-smoothed-vit: Salman, Jain, Wong, Madry, "Certified Patch
  Robustness via Smoothed Vision Transformers", CVPR 2022. Smoothing-based
  certified patch defence; not implemented here (out of scope) but cited as
  the modern certified-defence baseline.
- naseer-2019-localgradient: Naseer, Khan, Porikli, "Local Gradients Smoothing:
  Defense against localized adversarial attacks", WACV 2019. Predates
  PatchGuard; relevant prior art for input-side patch defences.

CRITIQUE OF THE NAIVE SEED
--------------------------
The original Brown et al. paper targets ImageNet at 224x224 with a circular
patch ~10% of image area. On a 28x28 Fashion-MNIST image, 4x4 = 16 px is ~2%
of area, which is at LaVAN's lower bound. We therefore sweep patch sizes
{2, 3, 4, 6, 8} (covering 0.5%, 1%, 2%, 4.6%, 8.2% of image area) and
*learn* placement per-image in addition to a fixed-corner baseline.

Additionally Fashion-MNIST has large background regions (zero pixels around a
small clothing silhouette); a randomly placed patch in pure background may
have no effect because the first conv layer simply sees the same all-black
neighbourhood it already saw. We therefore:
  (i)   evaluate fixed-corner placement (worst case for the attack),
  (ii)  evaluate learned/best placement (best case for the attack),
  (iii) compare to a RANDOM-VALUE patch baseline (to show optimisation
        matters, not just occlusion).

CONTROLS BEYOND THE SEED (per task brief)
-----------------------------------------
(1) COMPUTE-MATCHED PGD baseline: an Linf eps=0.1 PGD with the SAME total
    number of model gradient steps as the patch optimisation (so we report
    "ASR per gradient step" parity, not just "ASR at PGD-10 vs unbounded-budget
    patch").
(2) RANDOM-VALUE patch baseline: same L0 support, but pixels sampled
    uniform [0,1] - shows optimisation, not just occlusion, drives ASR.
(3) PER-CLASS breakdown of patch ASR (does the attack concentrate on the
    fashion-class confusion structure - shirt/T-shirt/pullover?).
(4) TRANSFER AUDIT: train PGD-AT eps=0.1 model, then test the patch attack
    against it. Quantify the protection gap (delta_ASR = ASR_CE - ASR_AT).
    The hypothesis predicts delta is small (mismatched threat model).
(5) LOCATION ABLATION: fixed top-left corner vs learned per-image placement
    (argmax over a coarse grid of 5x5 candidate offsets evaluated under the
    learned patch).

EXPECTED OUTCOME (the hypothesis)
---------------------------------
A 4x4 unbounded-magnitude patch with LEARNED placement achieves higher ASR
than Linf-eps-0.1 PGD-10 against a standard-trained SmallCNN, AND PGD-AT
eps=0.1 protects against PGD but NOT against the patch (small delta_ASR).

STANDARD CONFIG (campaign-shared)
---------------------------------
N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9, wd=5e-4),
SEED=0, EPS=0.1, PGD_STEPS=10, SmallCNN width=32 act=relu bn=true.
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
DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
WIDTH = 32

# Patch sweep (square side in pixels). 28x28 image, so areas:
# 2: 0.5%, 3: 1.1%, 4: 2.0%, 6: 4.6%, 8: 8.2%.
PATCH_SIZES = [2, 3, 4, 6, 8]

# Patch optimisation budget (Adam on the patch pixels, random placement
# per training step). Kept SMALL relative to ImageNet practice but matched
# to N_TRAIN=6000 budget.
PATCH_ITERS = 600
PATCH_LR = 0.1
PATCH_BATCH = 128

# Compute-matched PGD step count = PATCH_ITERS  (so the COMPARED full-image
# attack has access to the same total number of model gradient evaluations
# as the patch optimisation - see control (1)).
COMPUTE_MATCHED_PGD_STEPS = 50  # cap (per-sample) for runtime; total gradient
                                 # evals ~ COMPUTE_MATCHED_PGD_STEPS * N_eval/batch
                                 # which is comparable to PATCH_ITERS * 1 patch.

# Location ablation: coarse grid of candidate top-left offsets.
LOC_GRID_STRIDE = 5  # for a 28x28 image with 4-px patch -> offsets 0,5,...,20

# Output file (matches the tee redirect in the usage banner above).
OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h474_patch_attack_output.txt",
)


# ---------------------------------------------------------------------------
# training routines (shared with h473)
# ---------------------------------------------------------------------------
def _make_model():
    return C.build_model("cnn", C.dataset_meta(DS), width=WIDTH, act="relu", bn=True)


def _opt(model):
    return torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)


def train_ce(Xtr, Ytr):
    """Standard cross-entropy training (campaign-baseline recipe)."""
    C.set_seed(SEED)
    model = _make_model()
    opt = _opt(model)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def train_at(Xtr, Ytr, adv_eps=0.1, adv_steps=7):
    """PGD-AT (Madry et al. 2018) under Linf eps=0.1, the campaign winner."""
    C.set_seed(SEED)
    model = _make_model()
    opt = _opt(model)
    n = Xtr.size(0)
    alpha = 2.5 * adv_eps / adv_steps
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.pgd(model, xb, yb, eps=adv_eps, steps=adv_steps, alpha=alpha)
            opt.zero_grad()
            loss = F.cross_entropy(model(xa), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# patch placement helpers
# ---------------------------------------------------------------------------
def _apply_patch_batch(x, patch, rows, cols):
    """Place a single (1,1,P,P) patch at per-sample (rows[i], cols[i]).
    x: (B,1,H,W), patch: (1,1,P,P). Returns (B,1,H,W) in [0,1]."""
    B, _, H, W = x.shape
    P = patch.shape[-1]
    out = x.clone()
    # vectorised assignment using a loop is fine at B<=512
    for i in range(B):
        r, c = int(rows[i]), int(cols[i])
        out[i, :, r:r + P, c:c + P] = patch[0]
    return out


def _apply_patch_fixed(x, patch, r, c):
    """Place patch at the same (r,c) for every image in the batch."""
    B, _, H, W = x.shape
    P = patch.shape[-1]
    out = x.clone()
    out[:, :, r:r + P, c:c + P] = patch[0]
    return out


def _random_offsets(B, H, W, P, generator):
    """Sample uniform random patch placements for a batch of B images."""
    rs = torch.randint(0, H - P + 1, (B,), generator=generator)
    cs = torch.randint(0, W - P + 1, (B,), generator=generator)
    return rs, cs


# ---------------------------------------------------------------------------
# patch optimisation (Brown 2017 style; untargeted, random-placement train)
# ---------------------------------------------------------------------------
def train_patch(model, Xtr, Ytr, P, iters=PATCH_ITERS, lr=PATCH_LR,
                batch=PATCH_BATCH, seed=SEED):
    """Optimise a single (1,1,P,P) patch that MAXIMISES untargeted CE when
    pasted at a UNIFORMLY RANDOM position on a randomly-drawn training batch
    (the EOT-like training procedure from Brown et al. 2017).

    Returns a [0,1]-clamped patch tensor (1,1,P,P) on DEVICE."""
    device = Xtr.device
    n = Xtr.size(0)
    # init to mid-grey to avoid trivial all-zero/all-one fixed points.
    patch = torch.full((1, 1, P, P), 0.5, device=device, requires_grad=True)
    opt = torch.optim.Adam([patch], lr=lr)
    g_cpu = torch.Generator().manual_seed(seed + 1)  # for placement RNG
    H = W = Xtr.shape[-1]
    model.eval()
    for step in range(iters):
        # draw a random training batch
        idx = torch.randint(0, n, (batch,), generator=g_cpu).to(device)
        xb = Xtr[idx]
        yb = Ytr[idx]
        rs, cs = _random_offsets(batch, H, W, P, g_cpu)
        # build adv input WITH the current patch (NOT in-place on xb)
        adv = xb.clone()
        for i in range(batch):
            r, c = int(rs[i]), int(cs[i])
            adv[i, :, r:r + P, c:c + P] = patch[0].clamp(0.0, 1.0)
        opt.zero_grad()
        logits = model(adv)
        # untargeted: maximise CE (so MINIMISE -CE)
        loss = -F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        with torch.no_grad():
            patch.data.clamp_(0.0, 1.0)
        if (step + 1) % max(1, iters // 6) == 0:
            with torch.no_grad():
                acc = (logits.argmax(1) == yb).float().mean().item()
            print(f"    [patch P={P}] step {step+1}/{iters}  "
                  f"-CE={loss.item():+.4f}  train_acc_under_patch={acc:.3f}",
                  flush=True)
    return patch.detach().clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# patch evaluation (4 placement strategies)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _correct_mask(model, X, Y, batch=512):
    model.eval()
    out = []
    for i in range(0, X.size(0), batch):
        out.append((model(X[i:i + batch]).argmax(1) == Y[i:i + batch]).cpu())
    return torch.cat(out)


@torch.no_grad()
def _pred(model, X, batch=512):
    out = []
    for i in range(0, X.size(0), batch):
        out.append(model(X[i:i + batch]).argmax(1).cpu())
    return torch.cat(out)


@torch.no_grad()
def eval_patch_fixed_corner(model, X, Y, patch):
    """Place patch at fixed top-left corner (0,0) for every sample.
    Returns (asr, flipped_mask_over_correct, full_pred)."""
    P = patch.shape[-1]
    adv_chunks = []
    for i in range(0, X.size(0), 256):
        xb = X[i:i + 256]
        adv_chunks.append(_apply_patch_fixed(xb, patch, 0, 0))
    Xadv = torch.cat(adv_chunks, dim=0)
    pred = _pred(model, Xadv).numpy()
    return Xadv, pred


@torch.no_grad()
def eval_patch_random_loc(model, X, Y, patch, seed=SEED + 7):
    """Place patch at a uniform random position per sample. Single draw."""
    H = W = X.shape[-1]
    P = patch.shape[-1]
    g_cpu = torch.Generator().manual_seed(seed)
    rs, cs = _random_offsets(X.size(0), H, W, P, g_cpu)
    adv_chunks = []
    for i in range(0, X.size(0), 256):
        xb = X[i:i + 256]
        rb = rs[i:i + 256]
        cb = cs[i:i + 256]
        adv_chunks.append(_apply_patch_batch(xb, patch, rb, cb))
    Xadv = torch.cat(adv_chunks, dim=0)
    pred = _pred(model, Xadv).numpy()
    return Xadv, pred


@torch.no_grad()
def eval_patch_learned_loc(model, X, Y, patch, stride=LOC_GRID_STRIDE):
    """Per sample, pick the offset (over a coarse grid) that MAXIMISES the
    loss (proxy: minimises the correct-class probability). This is the
    'learned location' upper bound on the threat - aligned with LaVAN's
    optimisation-over-location step."""
    H = W = X.shape[-1]
    P = patch.shape[-1]
    offsets = list(range(0, H - P + 1, stride))
    if (H - P) % stride != 0:
        offsets.append(H - P)
    best_pred = np.full((X.size(0),), -1, dtype=np.int64)
    best_correct_logit = np.full((X.size(0),), np.inf, dtype=np.float32)
    Y_np = Y.cpu().numpy()
    # Iterate over (r,c) grid; keep the placement that gives the LOWEST
    # correct-class logit (proxy for highest loss).
    for r in offsets:
        for c in offsets:
            adv_chunks = []
            for i in range(0, X.size(0), 256):
                xb = X[i:i + 256]
                adv_chunks.append(_apply_patch_fixed(xb, patch, r, c))
            Xadv = torch.cat(adv_chunks, dim=0)
            preds = []
            corr_logit = []
            for i in range(0, Xadv.size(0), 512):
                lg = model(Xadv[i:i + 512])
                preds.append(lg.argmax(1).cpu().numpy())
                yb = Y[i:i + 512].cpu().numpy()
                corr_logit.append(lg[np.arange(yb.shape[0]), yb].cpu().numpy())
            preds = np.concatenate(preds)
            corr_logit = np.concatenate(corr_logit)
            mask = corr_logit < best_correct_logit
            best_correct_logit[mask] = corr_logit[mask]
            best_pred[mask] = preds[mask]
    return best_pred


@torch.no_grad()
def eval_random_value_patch(model, X, Y, P, seed=SEED + 13):
    """Control (2): a RANDOM-VALUE patch (uniform [0,1] pixels) placed at a
    uniform random position per sample. Same L0 support, no optimisation."""
    device = X.device
    H = W = X.shape[-1]
    g_cpu = torch.Generator().manual_seed(seed)
    g_gpu = torch.Generator(device=device).manual_seed(seed + 1)
    patch = torch.rand((1, 1, P, P), generator=g_gpu, device=device)
    rs, cs = _random_offsets(X.size(0), H, W, P, g_cpu)
    adv_chunks = []
    for i in range(0, X.size(0), 256):
        xb = X[i:i + 256]
        rb = rs[i:i + 256]
        cb = cs[i:i + 256]
        adv_chunks.append(_apply_patch_batch(xb, patch, rb, cb))
    Xadv = torch.cat(adv_chunks, dim=0)
    pred = _pred(model, Xadv).numpy()
    return pred


# ---------------------------------------------------------------------------
# compute-matched full-image Linf PGD (control 1)
# ---------------------------------------------------------------------------
def eval_pgd_compute_matched(model, X, Y, eps=EPS, steps=COMPUTE_MATCHED_PGD_STEPS):
    """Run PGD with `steps` gradient steps per sample. Returns adversarial
    predictions over the whole eval set."""
    preds = []
    model.eval()
    for i in range(0, X.size(0), 256):
        xb = X[i:i + 256]
        yb = Y[i:i + 256]
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            preds.append(model(xa).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def eval_pgd_standard(model, X, Y, eps=EPS, steps=PGD_STEPS):
    """PGD-10 baseline (matches campaign tables)."""
    preds = []
    model.eval()
    for i in range(0, X.size(0), 256):
        xb = X[i:i + 256]
        yb = Y[i:i + 256]
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            preds.append(model(xa).argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# ASR helpers
# ---------------------------------------------------------------------------
def _asr(pred, Y_np, corr_mask):
    """Fraction of originally-correct samples whose adv prediction != y."""
    if corr_mask.sum() == 0:
        return float("nan")
    flipped = pred != Y_np
    return float(flipped[corr_mask].mean())


def _per_class_asr(pred, Y_np, corr_mask, ncls=10):
    out = {}
    for c in range(ncls):
        sel = (Y_np == c) & corr_mask
        if sel.sum() == 0:
            out[c] = float("nan")
        else:
            out[c] = float((pred[sel] != Y_np[sel]).mean())
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=2000, seed=SEED)
    Y_np = Yte.cpu().numpy()

    # ---- train CE and AT defences ------------------------------------------
    print("\n=== training CE (standard) ===", flush=True)
    t0 = time.time()
    model_ce = train_ce(Xtr, Ytr)
    t_ce = time.time() - t0
    print(f"  trained CE in {t_ce:.1f}s", flush=True)

    print("\n=== training PGD-AT (Linf eps=0.1, 7 steps) ===", flush=True)
    t0 = time.time()
    model_at = train_at(Xtr, Ytr, adv_eps=EPS, adv_steps=7)
    t_at = time.time() - t0
    print(f"  trained AT in {t_at:.1f}s", flush=True)

    models = {"CE": model_ce, "AT": model_at}

    # ---- baseline: clean acc + standard PGD-10 -----------------------------
    print("\n=== baselines: clean acc + PGD-10 ===", flush=True)
    baselines = {}
    for name, model in models.items():
        _, clean_acc = C.logits_and_acc(model, Xte, Yte)
        corr = _correct_mask(model, Xte, Yte).numpy().astype(bool)
        pred_pgd = eval_pgd_standard(model, Xte, Yte, eps=EPS, steps=PGD_STEPS)
        pgd_asr = _asr(pred_pgd, Y_np, corr)
        # control (1): compute-matched PGD
        pred_pgd_cm = eval_pgd_compute_matched(model, Xte, Yte, eps=EPS,
                                               steps=COMPUTE_MATCHED_PGD_STEPS)
        pgd_cm_asr = _asr(pred_pgd_cm, Y_np, corr)
        baselines[name] = dict(
            clean_acc=float(clean_acc),
            pgd10_asr=pgd_asr,
            pgd_cm_asr=pgd_cm_asr,
            corr=corr,
        )
        print(f"  [{name}] clean={clean_acc:.4f}  pgd10_asr={pgd_asr:.4f}  "
              f"pgd{COMPUTE_MATCHED_PGD_STEPS}_asr={pgd_cm_asr:.4f}", flush=True)

    # ---- patch attack: sweep over sizes, two location strategies ----------
    print("\n=== patch attack sweep ===", flush=True)
    patch_results = []  # (defence, P, mode, asr, train_s)
    per_class_4x4 = {}  # control (3): per-class table at P=4 learned loc
    patches_by_defP = {}  # cache patch tensors per (defence, P)
    for name, model in models.items():
        for P in PATCH_SIZES:
            print(f"\n  --- patch P={P} on {name} ---", flush=True)
            t0 = time.time()
            patch = train_patch(model, Xtr, Ytr, P=P, iters=PATCH_ITERS,
                                lr=PATCH_LR, batch=PATCH_BATCH, seed=SEED)
            dt_train = time.time() - t0
            patches_by_defP[(name, P)] = patch
            corr = baselines[name]["corr"]

            # mode A: fixed top-left corner
            _, pred_fc = eval_patch_fixed_corner(model, Xte, Yte, patch)
            asr_fc = _asr(pred_fc, Y_np, corr)
            # mode B: single uniform-random location per sample
            _, pred_rl = eval_patch_random_loc(model, Xte, Yte, patch,
                                               seed=SEED + 100 + P)
            asr_rl = _asr(pred_rl, Y_np, corr)
            # mode C: learned (best-of-grid) location per sample
            pred_ll = eval_patch_learned_loc(model, Xte, Yte, patch,
                                             stride=LOC_GRID_STRIDE)
            asr_ll = _asr(pred_ll, Y_np, corr)
            # control (2): random-value patch (same L0, no optimisation)
            pred_rv = eval_random_value_patch(model, Xte, Yte, P=P,
                                              seed=SEED + 200 + P)
            asr_rv = _asr(pred_rv, Y_np, corr)

            patch_results.append((name, P, "fixed_corner",  asr_fc, dt_train))
            patch_results.append((name, P, "random_loc",    asr_rl, dt_train))
            patch_results.append((name, P, "learned_loc",   asr_ll, dt_train))
            patch_results.append((name, P, "random_value",  asr_rv, 0.0))

            print(f"    [{name} P={P}] fixed_corner ASR={asr_fc:.4f}  "
                  f"random_loc ASR={asr_rl:.4f}  learned_loc ASR={asr_ll:.4f}  "
                  f"random_value ASR={asr_rv:.4f}  ({dt_train:.1f}s train)",
                  flush=True)

            # per-class breakdown at P=4 learned location (control 3)
            if P == 4:
                per_class_4x4[name] = _per_class_asr(pred_ll, Y_np, corr)

    # ---- write report ------------------------------------------------------
    fmnist_classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]

    lines = []
    lines.append("H474 Localised adversarial-patch attack on Fashion-MNIST")
    lines.append("=" * 72)
    lines.append("Anchor: brown-2017-advpatch (NeurIPS 2017 workshop).")
    lines.append("Aux:    karmon-2018-lavan, wu-2020-defending-physical-patches,")
    lines.append("        xiang-2021-patchguard, salman-2022-smoothed-vit,")
    lines.append("        naseer-2019-localgradient.")
    lines.append("Threat: L0-constrained, L_inf-unbounded patch in [0,1].")
    lines.append("")
    lines.append(f"Dataset={DS}  N_train={N_TRAIN}  Epochs={EPOCHS}  "
                 f"LR={LR}  Batch={BATCH}  Seed={SEED}  EPS(Linf)={EPS}")
    lines.append(f"Patch: sizes={PATCH_SIZES}  iters={PATCH_ITERS}  "
                 f"lr={PATCH_LR}  batch={PATCH_BATCH}")
    lines.append(f"Compute-matched PGD steps={COMPUTE_MATCHED_PGD_STEPS} "
                 f"(vs standard PGD-{PGD_STEPS})")
    lines.append(f"Location grid stride={LOC_GRID_STRIDE} (learned-location"
                 f" ablation)")
    lines.append("")

    # baselines table
    lines.append("Defence baselines (clean / standard PGD-10 / compute-matched PGD):")
    lines.append("-" * 72)
    lines.append(f"{'defence':<8} {'clean_acc':>10} {'pgd10_asr':>10} "
                 f"{'pgd_cm_asr':>11} {'train_s':>9}")
    train_s = {"CE": t_ce, "AT": t_at}
    for name in ("CE", "AT"):
        b = baselines[name]
        lines.append(f"{name:<8} {b['clean_acc']:>10.4f} {b['pgd10_asr']:>10.4f} "
                     f"{b['pgd_cm_asr']:>11.4f} {train_s[name]:>9.1f}")
    lines.append("")

    # patch ASR table
    lines.append("Patch ASR table (fraction of originally-correct flipped):")
    lines.append("-" * 72)
    lines.append(f"{'defence':<8} {'P':>3} {'mode':<14} {'asr':>8} {'train_s':>9}")
    for name, P, mode, asr, dt in patch_results:
        lines.append(f"{name:<8} {P:>3d} {mode:<14} {asr:>8.4f} {dt:>9.1f}")
    lines.append("")

    # per-class table (P=4 learned loc)
    lines.append("Per-class patch ASR at P=4, learned location (control 3):")
    lines.append("-" * 72)
    lines.append(f"{'class':<14} {'CE_asr':>8} {'AT_asr':>8}")
    for c in range(10):
        ce_v = per_class_4x4.get("CE", {}).get(c, float("nan"))
        at_v = per_class_4x4.get("AT", {}).get(c, float("nan"))
        lines.append(f"{fmnist_classes[c]:<14} {ce_v:>8.4f} {at_v:>8.4f}")
    lines.append("")

    # transfer / protection-gap audit (control 4)
    def _best_over_modes(name, P, modes):
        vals = [asr for (n2, P2, m, asr, _) in patch_results
                if n2 == name and P2 == P and m in modes]
        return max(vals) if vals else float("nan")

    lines.append("Protection-gap audit (control 4): CE vs AT under each attack")
    lines.append("-" * 72)
    lines.append(f"{'attack':<28} {'CE_asr':>8} {'AT_asr':>8} {'delta':>8}")
    ce_pgd10 = baselines["CE"]["pgd10_asr"]
    at_pgd10 = baselines["AT"]["pgd10_asr"]
    lines.append(f"{'Linf PGD-10 eps=0.1':<28} {ce_pgd10:>8.4f} "
                 f"{at_pgd10:>8.4f} {ce_pgd10 - at_pgd10:>+8.4f}")
    ce_pgd_cm = baselines["CE"]["pgd_cm_asr"]
    at_pgd_cm = baselines["AT"]["pgd_cm_asr"]
    lines.append(f"{'Linf PGD-' + str(COMPUTE_MATCHED_PGD_STEPS) + ' eps=0.1':<28} "
                 f"{ce_pgd_cm:>8.4f} {at_pgd_cm:>8.4f} "
                 f"{ce_pgd_cm - at_pgd_cm:>+8.4f}")
    for P in PATCH_SIZES:
        ce_v = _best_over_modes("CE", P, ("learned_loc",))
        at_v = _best_over_modes("AT", P, ("learned_loc",))
        lines.append(f"{f'Patch P={P} learned_loc':<28} "
                     f"{ce_v:>8.4f} {at_v:>8.4f} {ce_v - at_v:>+8.4f}")
    lines.append("")

    # auto verdict
    lines.append("Verdict (auto):")
    # primary claim: 4x4 unbounded patch (learned loc) > PGD-10 Linf eps=0.1 on CE
    ce_patch4_learned = _best_over_modes("CE", 4, ("learned_loc",))
    primary = ce_patch4_learned > ce_pgd10
    # AT mismatch: AT's patch ASR at P=4 - AT's PGD-10 ASR (large => AT failed to
    # transfer to patch threat model)
    at_patch4_learned = _best_over_modes("AT", 4, ("learned_loc",))
    mismatch = at_patch4_learned - at_pgd10
    # optimisation-vs-occlusion: learned >> random_value at same P
    rv4_ce = _best_over_modes("CE", 4, ("random_value",))
    opt_matters = (ce_patch4_learned - rv4_ce) > 0.10
    # location matters: learned_loc > fixed_corner by >0.05 at P=4
    fc4_ce = _best_over_modes("CE", 4, ("fixed_corner",))
    loc_matters = (ce_patch4_learned - fc4_ce) > 0.05

    lines.append(f"  primary (4x4 patch learned-loc ASR > Linf PGD-10 ASR on CE): "
                 f"{ce_patch4_learned:.4f} vs {ce_pgd10:.4f}  -> "
                 f"{'SUPPORTED' if primary else 'NOT SUPPORTED'}")
    lines.append(f"  AT mismatch (AT 4x4-patch ASR - AT PGD-10 ASR): "
                 f"{mismatch:+.4f}  -> "
                 f"{'AT FAILS to transfer' if mismatch > 0.10 else 'AT partly transfers'}")
    lines.append(f"  optimisation matters (learned-patch ASR - random-value ASR @ P=4 CE):"
                 f" {ce_patch4_learned - rv4_ce:+.4f} -> "
                 f"{'YES' if opt_matters else 'WEAK'}")
    lines.append(f"  placement matters (learned-loc - fixed-corner @ P=4 CE):"
                 f" {ce_patch4_learned - fc4_ce:+.4f} -> "
                 f"{'YES' if loc_matters else 'WEAK'}")
    lines.append("")

    # reading & limitations
    lines.append("Reading:")
    lines.append("  * If primary SUPPORTED and AT FAILS to transfer: the Linf")
    lines.append("    eps=0.1 evaluation in CAMPAIGN_GAP_MAP.md M2/M3 is")
    lines.append("    GENUINELY missing a stronger threat (Brown 2017, LaVAN);")
    lines.append("    'robustness' numbers in the campaign are not robust to a")
    lines.append("    threat-model shift to localised L0-unbounded patches.")
    lines.append("  * If 'placement matters' is YES: fixed-corner evaluations")
    lines.append("    (e.g. H60) UNDER-estimate the patch threat - matches LaVAN.")
    lines.append("  * Per-class skew: expect shirt/T-shirt/pullover/coat (the")
    lines.append("    'upper-body' cluster) to soak up most flips because the")
    lines.append("    patch can mimic class-distinguishing texture local to a")
    lines.append("    small spatial region.")
    lines.append("")
    lines.append("Limitations:")
    lines.append("  * SmallCNN has a large effective receptive field; a small-RF")
    lines.append("    architecture (PatchGuard-style, Xiang 2021) would likely")
    lines.append("    reduce ASR substantially. Not implemented here.")
    lines.append("  * Single seed (SEED=0), N_TRAIN=6000 (M1, M2 still in play).")
    lines.append("  * Untargeted only; no targeted-patch sweep.")
    lines.append("  * 'Learned location' is best-of-grid (stride %d), not full"
                 " continuous search;" % LOC_GRID_STRIDE)
    lines.append("    this is a LOWER bound on the true learned-placement ASR.")
    lines.append("  * No adaptive defence (no patch-AT, no PatchGuard).")
    lines.append("")

    # HEADLINE one-line verdict
    headline_outcome = (
        "SUPPORTED" if (primary and mismatch > 0.10) else
        "PARTIAL"   if (primary or mismatch > 0.10) else
        "NOT_SUPPORTED"
    )
    lines.append("=" * 72)
    lines.append(
        f"HEADLINE: H474 {headline_outcome} - 4x4 patch (learned-loc) ASR="
        f"{ce_patch4_learned:.3f} on CE vs Linf PGD-10 ASR={ce_pgd10:.3f}; "
        f"AT(eps=0.1) PGD-10 ASR={at_pgd10:.3f} but patch ASR={at_patch4_learned:.3f} "
        f"(mismatch +{mismatch:.3f})."
    )
    lines.append("=" * 72)

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
        f.flush()
    print(f"\nSaved to {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
