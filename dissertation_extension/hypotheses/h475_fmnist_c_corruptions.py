"""
H475 - Adversarial training (PGD-AT) does NOT buy natural-corruption robustness
       on a Fashion-MNIST-C suite.

Seed paper (CAMPAIGN_GAP_MAP §5 / §2 M2):
  Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common
  Corruptions and Perturbations" (ICLR 2019, arXiv:1903.12261). Introduced
  CIFAR-10-C / ImageNet-C as a *separate* axis of robustness from adversarial.
  We rebuild a Fashion-MNIST-equivalent ("Fashion-MNIST-C") with 11 corruptions
  at 5 severities each. Per §3 G1 and §2 M9, all corruptions are implemented
  inline with pure numpy/torch -- NO external transfer / no extra deps.

Hypothesis:
  Adversarially trained (PGD-AT, eps=0.1) SmallCNN gains substantial adversarial
  robustness over a standard model but does NOT meaningfully gain natural
  (Fashion-MNIST-C) corruption robustness. This is the canonical finding in
  Hendrycks 2019 and Hendrycks "AugMix" (arXiv:1912.02781): the two robustness
  axes are decorrelated. AugMix-style augmentation, in contrast, *does* close
  the corruption gap. Per Geirhos "shortcut learning" (Nat. Mach. Intel. 2020,
  arXiv:2004.07780) the explanation is that adversarial training and corruption
  augmentation correct for different shortcut features. Taori et al. "Measuring
  Robustness to Natural Distribution Shifts" (NeurIPS 2020, arXiv:2007.00644)
  further documents that robustness interventions which help on synthetic
  perturbations do not always transfer to natural shifts.

Critique seed handling:
  - Original CIFAR-10-C corruptions are RGB; Fashion-MNIST is grayscale 28x28.
    fog/frost in particular need adaptation; we use a procedural Perlin-ish
    grayscale fog (1/f noise mixed in), and frost = bandlimited-noise overlay.
  - 11 corruptions x 5 severities each (gaussian noise, shot noise, defocus
    blur, motion blur, brightness, contrast, fog, frost, elastic transform,
    pixelate, jpeg-style quantisation).
  - Compute-matched: STD, PGD-AT, AUGMIX-AUG all share the same epoch budget
    (EPOCHS). PGD-AT pays for its inner steps; STD/AUGMIX-AUG burn the budget
    on clean / augmented gradient steps. We report wall time as evidence.
  - AugMix-style control: 3rd model trained on a mix of standard data + random
    samples of the same corruption suite (severities 1-3) so it never sees
    held-out severities 4-5. This is the augmentation-baseline.
  - Per-corruption-type breakdown, NOT just mean corruption error (mCE).
  - Per-class breakdown for the worst (highest mean error) corruption.
  - Per-severity Spearman / Pearson correlation between adv-robustness
    (1 - PGD@eps=0.1 ASR) and corruption error, across the (3 models x 11
    corruptions x 5 severities) cells.

# usage:
#   .venv/bin/python dissertation_extension/hypotheses/h475_fmnist_c_corruptions.py
#   Output -> dissertation_extension/results/fashion_mnist/h475_fmnist_c_corruptions_output.txt
"""
import os, sys, time, io, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
DATASET = "fashion_mnist"
SEED = 0
N_TRAIN = 10_000        # subsample so the 3-model x corruption-suite sweep fits in time
N_EVAL = 2000           # corruption eval subset
EPOCHS = 8              # compute-matched across all three models
BATCH = 128
LR = 0.05

ADV_EPS = 0.1
ADV_STEPS = 7
ADV_EVAL_STEPS = 20     # stronger PGD for the headline ASR

SEVERITIES = [1, 2, 3, 4, 5]
CORRUPTIONS = [
    "gaussian_noise", "shot_noise",
    "defocus_blur", "motion_blur",
    "brightness", "contrast",
    "fog", "frost",
    "elastic", "pixelate", "jpeg",
]

META = {"channels": 1, "size": 28, "n_classes": 10}
FMNIST_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle-boot"]


# ---------------------------------------------------------------------------
# Fashion-MNIST-C corruption implementations  (pure numpy, grayscale 28x28)
# All take a float32 numpy array x of shape (N,1,28,28) in [0,1] and return
# a float32 numpy array of the same shape, also clipped to [0,1].
# Severity is an integer in {1..5}.
# ---------------------------------------------------------------------------
def _np01(x):
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def c_gaussian_noise(x, sev, rng):
    sigma = [0.04, 0.08, 0.12, 0.18, 0.26][sev - 1]
    return _np01(x + rng.normal(0, sigma, x.shape).astype(np.float32))


def c_shot_noise(x, sev, rng):
    # Poisson shot noise -- scale chosen so severity 5 is highly destructive on [0,1].
    lam = [60.0, 25.0, 12.0, 6.0, 3.0][sev - 1]
    y = rng.poisson(np.clip(x, 0, 1) * lam) / float(lam)
    return _np01(y)


def _gaussian_kernel(k, sigma):
    ax = np.arange(-k // 2 + 1.0, k // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (g / g.sum()).astype(np.float32)


def _conv2d_same(x, kernel):
    """x: (N,1,H,W) float32 numpy; kernel: (kh,kw). Returns same shape."""
    t = torch.from_numpy(x)
    k = torch.from_numpy(kernel)[None, None]
    pad = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    return F.conv2d(t, k, padding=pad).numpy()


def c_defocus_blur(x, sev, rng):
    sigma = [0.6, 0.9, 1.3, 1.7, 2.2][sev - 1]
    k = _gaussian_kernel(7, sigma)
    return _np01(_conv2d_same(x, k))


def c_motion_blur(x, sev, rng):
    L = [3, 5, 7, 9, 11][sev - 1]
    # 1-D horizontal motion blur kernel (angle randomized per-sample is too costly;
    # we pick a single horizontal direction -- mimics camera shake).
    k = np.zeros((L, L), dtype=np.float32)
    k[L // 2, :] = 1.0 / L
    return _np01(_conv2d_same(x, k))


def c_brightness(x, sev, rng):
    delta = [0.1, 0.2, 0.3, 0.4, 0.5][sev - 1]
    return _np01(x + delta)


def c_contrast(x, sev, rng):
    # multiply around the per-image mean
    factor = [0.75, 0.55, 0.4, 0.28, 0.18][sev - 1]
    means = x.mean(axis=(2, 3), keepdims=True)
    return _np01((x - means) * factor + means)


def _fbm_noise(shape, octaves, rng):
    """Cheap 1/f-ish fractional Brownian motion: sum of upsampled smoothed white
    noise at multiple scales. Returns array in [0,1]."""
    h, w = shape
    img = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    for o in range(octaves):
        scale = 2 ** o
        nh, nw = max(2, h // scale), max(2, w // scale)
        n = rng.normal(0, 1, (nh, nw)).astype(np.float32)
        # upsample (bilinear via torch)
        t = torch.from_numpy(n)[None, None]
        t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        img += amp * t.numpy()[0, 0]
        amp *= 0.5
    img -= img.min()
    img /= (img.max() + 1e-8)
    return img


def c_fog(x, sev, rng):
    """Grayscale fog: blend image toward a procedural cloudy field."""
    strength, density = [(0.25, 0.6), (0.4, 0.7), (0.55, 0.8),
                        (0.7, 0.85), (0.85, 0.9)][sev - 1]
    n = x.shape[0]
    out = np.empty_like(x)
    for i in range(n):
        fog = _fbm_noise((28, 28), octaves=4, rng=rng) * density + (1 - density) * 0.5
        out[i, 0] = (1 - strength) * x[i, 0] + strength * fog
    return _np01(out)


def c_frost(x, sev, rng):
    """Grayscale frost: bandlimited-noise overlay (high-frequency speckle plus
    low-frequency cloudiness), with a multiply-screen blend toward white."""
    strength, hf = [(0.2, 0.4), (0.3, 0.55), (0.45, 0.7),
                    (0.6, 0.85), (0.8, 1.0)][sev - 1]
    n = x.shape[0]
    out = np.empty_like(x)
    for i in range(n):
        low = _fbm_noise((28, 28), octaves=3, rng=rng)
        speck = np.clip(rng.normal(0.5, 0.3, (28, 28)), 0, 1).astype(np.float32)
        frost = (1 - hf) * low + hf * speck
        # screen blend toward white-ish
        out[i, 0] = 1 - (1 - x[i, 0]) * (1 - strength * frost)
    return _np01(out)


def c_elastic(x, sev, rng):
    """Elastic deformation a la Simard 2003 / Hendrycks corruption suite."""
    alpha, sigma = [(2.0, 1.6), (3.0, 1.6), (4.5, 1.4),
                    (6.0, 1.3), (8.0, 1.2)][sev - 1]
    n, _, h, w = x.shape
    out = np.empty_like(x)
    # smoothing kernel
    k = _gaussian_kernel(7, sigma)
    for i in range(n):
        dx = rng.uniform(-1, 1, (h, w)).astype(np.float32)
        dy = rng.uniform(-1, 1, (h, w)).astype(np.float32)
        dx = _conv2d_same(dx[None, None], k)[0, 0] * alpha
        dy = _conv2d_same(dy[None, None], k)[0, 0] * alpha
        # build sampling grid in normalized coords for grid_sample
        gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        sx = (gx + dx) / (w - 1) * 2 - 1
        sy = (gy + dy) / (h - 1) * 2 - 1
        grid = np.stack([sx, sy], axis=-1)[None].astype(np.float32)  # (1,h,w,2)
        t = torch.from_numpy(x[i:i + 1])
        g = torch.from_numpy(grid)
        warped = F.grid_sample(t, g, mode="bilinear", padding_mode="border",
                               align_corners=True).numpy()
        out[i] = warped[0]
    return _np01(out)


def c_pixelate(x, sev, rng):
    """Pixelate: downsample then upsample with nearest neighbour."""
    factor = [0.9, 0.75, 0.6, 0.45, 0.3][sev - 1]
    new_sz = max(4, int(round(28 * factor)))
    t = torch.from_numpy(x)
    down = F.interpolate(t, size=(new_sz, new_sz), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(28, 28), mode="nearest")
    return _np01(up.numpy())


def c_jpeg(x, sev, rng):
    """Pure-numpy JPEG-ish quantisation: 8x8 DCT, scalar quantise, inverse DCT.
    Avoids PIL dependency. Approximates JPEG quality loss."""
    # quality -> step size
    q = [12, 24, 40, 60, 90][sev - 1]   # higher q = harsher
    n, _, h, w = x.shape
    out = np.empty_like(x)
    # 1D DCT-II basis
    def dct_mat(N):
        kk = np.arange(N)[:, None]
        nn = np.arange(N)[None, :]
        m = np.cos(np.pi * (2 * nn + 1) * kk / (2 * N))
        m[0] *= 1 / np.sqrt(2)
        return (m * np.sqrt(2 / N)).astype(np.float32)
    D = dct_mat(4)
    Dt = D.T
    # process 4x4 blocks (28 = 7*4)
    for i in range(n):
        img = x[i, 0].copy()
        for by in range(0, 28, 4):
            for bx in range(0, 28, 4):
                blk = img[by:by + 4, bx:bx + 4]
                c = D @ blk @ Dt
                c = np.round(c * q) / q
                blk2 = Dt @ c @ D
                img[by:by + 4, bx:bx + 4] = blk2
        out[i, 0] = img
    return _np01(out)


CORRUPTION_FNS = {
    "gaussian_noise": c_gaussian_noise,
    "shot_noise":     c_shot_noise,
    "defocus_blur":   c_defocus_blur,
    "motion_blur":    c_motion_blur,
    "brightness":     c_brightness,
    "contrast":       c_contrast,
    "fog":            c_fog,
    "frost":          c_frost,
    "elastic":        c_elastic,
    "pixelate":       c_pixelate,
    "jpeg":           c_jpeg,
}


def apply_corruption(name, X_np, sev, rng):
    fn = CORRUPTION_FNS[name]
    return fn(X_np, sev, rng)


# ---------------------------------------------------------------------------
# AugMix-style augmentation:  for each training batch, with prob p apply a
# randomly chosen corruption at a randomly chosen mild severity (1-3).
# (Not the full AugMix mixture chain -- a simplified single-op augmentation
#  baseline so it is directly comparable to PGD-AT computationally.)
# ---------------------------------------------------------------------------
def _augment_batch(xb_cpu_np, rng, p=0.5, max_sev=3):
    mask = rng.uniform(size=xb_cpu_np.shape[0]) < p
    if not mask.any():
        return xb_cpu_np
    out = xb_cpu_np.copy()
    idxs = np.where(mask)[0]
    for i in idxs:
        name = CORRUPTIONS[rng.integers(0, len(CORRUPTIONS))]
        sev = int(rng.integers(1, max_sev + 1))
        out[i:i + 1] = apply_corruption(name, out[i:i + 1], sev, rng)
    return out


def train_augmix_style(model, Xtr, Ytr, epochs, batch, lr, rng):
    """Same shape as common.train_model but per-batch we apply random corruptions
    to roughly half the batch (sev 1-3). Compute-matched to STD."""
    opt = C.make_optimizer(model, "sgd", lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_np = xb.detach().cpu().numpy()
            xb_aug = _augment_batch(xb_np, rng, p=0.5, max_sev=3)
            xb_t = torch.from_numpy(xb_aug).to(xb.device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_t), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def accuracy(model, X, Y, batch=512):
    model.eval()
    correct = 0
    n = X.size(0)
    for i in range(0, n, batch):
        out = model(X[i:i + batch])
        correct += (out.argmax(1) == Y[i:i + batch]).sum().item()
    return correct / n


def corruption_eval(model, X_clean_cpu_np, Y_dev, name, sev, rng, batch=512):
    """Apply corruption on CPU (numpy), move to device, score."""
    Xc_np = apply_corruption(name, X_clean_cpu_np, sev, rng)
    Xc = torch.from_numpy(Xc_np).to(Y_dev.device)
    acc = accuracy(model, Xc, Y_dev, batch=batch)
    return acc, Xc_np


def pgd_asr(model, X, Y, eps, steps, batch=256):
    r = C.attack_success(model, X, Y, attack="pgd", eps=eps, steps=steps, batch=batch)
    return r["asr"]


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------
def pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ra = a.argsort().argsort()
    rb = b.argsort().argsort()
    return pearson(ra, rb)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    log("=" * 78)
    log("H475 - Fashion-MNIST-C: PGD-AT vs STD vs AugMix-style across natural corruptions")
    log("=" * 78)
    log(f"dataset={DATASET}  n_train={N_TRAIN}  n_eval={N_EVAL}  epochs={EPOCHS}  "
        f"adv_eps={ADV_EPS}  adv_steps_train={ADV_STEPS}  adv_steps_eval={ADV_EVAL_STEPS}")
    log(f"corruptions ({len(CORRUPTIONS)}): {CORRUPTIONS}")
    log(f"severities: {SEVERITIES}")
    log("Refs: Hendrycks&Dietterich 2019 (CIFAR-10-C); Hendrycks AugMix 2019;")
    log("      Geirhos 'shortcut learning' 2020; Taori 'natural shifts' 2020.")
    log()

    # --- data ---
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DATASET, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    log(f"Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    X_clean_np = Xte.detach().cpu().numpy().astype(np.float32)

    rng_aug = np.random.default_rng(SEED + 1)
    rng_eval = np.random.default_rng(SEED + 999)

    # ---- train 3 compute-matched models ----
    models = {}
    train_times = {}

    log("\n[train] STD baseline (standard SGD)")
    C.set_seed(SEED)
    m_std = C.build_model("cnn", META, width=32)
    t = time.time()
    C.train_model(m_std, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, opt="sgd", lr=LR)
    train_times["STD"] = time.time() - t
    models["STD"] = m_std

    log(f"[train] PGD-AT eps={ADV_EPS} steps={ADV_STEPS}")
    C.set_seed(SEED)
    m_at = C.build_model("cnn", META, width=32)
    t = time.time()
    C.train_model(m_at, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, opt="sgd", lr=LR,
                  adv_train=True, adv_eps=ADV_EPS, adv_steps=ADV_STEPS)
    train_times["PGD-AT"] = time.time() - t
    models["PGD-AT"] = m_at

    log("[train] AugMix-style (random corruption augmentation, sev 1-3)")
    C.set_seed(SEED)
    m_aug = C.build_model("cnn", META, width=32)
    t = time.time()
    train_augmix_style(m_aug, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, lr=LR, rng=rng_aug)
    train_times["AUGMIX"] = time.time() - t
    models["AUGMIX"] = m_aug

    log()
    log("Training wall-time (compute-match check):")
    for k, v in train_times.items():
        log(f"  {k:>7s}: {v:7.1f}s")
    log()

    # ---- clean accuracy ----
    log("Clean accuracy on N_EVAL test subset:")
    clean_acc = {}
    for name, m in models.items():
        a = accuracy(m, Xte, Yte)
        clean_acc[name] = a
        log(f"  {name:>7s}: {a:.4f}")
    log()

    # ---- adversarial robustness ----
    log(f"PGD ASR on test subset  (eps={ADV_EPS}, steps={ADV_EVAL_STEPS}):")
    asr = {}
    for name, m in models.items():
        a = pgd_asr(m, Xte, Yte, eps=ADV_EPS, steps=ADV_EVAL_STEPS)
        asr[name] = a
        log(f"  {name:>7s}: ASR={a:.4f}  -> adv-acc on correctly-classified ~ {1 - a:.4f}")
    log()

    # ---- per-corruption, per-severity sweep ----
    # acc_table[(model, corruption, sev)] = accuracy
    acc_table = {}
    # also keep the corruption-applied images for the worst-corruption per-class probe
    worst_record = {}   # corruption -> (model -> Xc_np at sev=3)
    log("Per-corruption / per-severity accuracy sweep (this is the bulk):")
    header = "  {:<18s}  ".format("corruption/sev") + "  ".join(
        f"S{s}" for s in SEVERITIES) + "   |  per-corruption mean acc (per model)"
    log(header)
    log("-" * len(header))

    for cname in CORRUPTIONS:
        # reuse identical corruption seeds across models so noise realisations are matched
        for sev in SEVERITIES:
            r_local = np.random.default_rng(hash((cname, sev)) % (2 ** 32))
            Xc_np = apply_corruption(cname, X_clean_np, sev, r_local)
            Xc = torch.from_numpy(Xc_np).to(Yte.device)
            for name, m in models.items():
                acc_table[(name, cname, sev)] = accuracy(m, Xc, Yte)
            if sev == 3:
                worst_record[cname] = Xc_np   # store for later per-class look (replaced below if needed)
        # row print
        for name in ["STD", "PGD-AT", "AUGMIX"]:
            vals = [acc_table[(name, cname, s)] for s in SEVERITIES]
            mean_acc = float(np.mean(vals))
            log("  {:<18s}  ".format(f"{cname}[{name}]")
                + "  ".join(f"{v:.3f}" for v in vals)
                + f"   |  mean_acc={mean_acc:.3f}")
        log()

    # ---- per-model mean corruption ERROR (mCE-like, unnormalised) ----
    log("=" * 78)
    log("Headline aggregates")
    log("=" * 78)
    mce = {}
    for name in ["STD", "PGD-AT", "AUGMIX"]:
        vals = [1.0 - acc_table[(name, c, s)] for c in CORRUPTIONS for s in SEVERITIES]
        mce[name] = float(np.mean(vals))
        log(f"  {name:>7s}  mean corruption-error (1 - acc, all 11x5 cells): {mce[name]:.4f}")
    log()

    # per-corruption mean-error breakdown
    log("Per-corruption mean error (averaged over severities):")
    log("  {:<16s}  {:>6s}  {:>6s}  {:>6s}  {:>10s}".format(
        "corruption", "STD", "PGD-AT", "AUGMIX", "AT-STD"))
    per_corr_err = {name: {} for name in models}
    for c in CORRUPTIONS:
        row = {}
        for name in models:
            err = float(np.mean([1.0 - acc_table[(name, c, s)] for s in SEVERITIES]))
            row[name] = err
            per_corr_err[name][c] = err
        log("  {:<16s}  {:>6.3f}  {:>6.3f}  {:>6.3f}  {:>+10.3f}".format(
            c, row["STD"], row["PGD-AT"], row["AUGMIX"], row["PGD-AT"] - row["STD"]))
    log()

    # which corruption is the worst (max mean err) under STD?
    worst_c = max(CORRUPTIONS, key=lambda c: per_corr_err["STD"][c])
    log(f"Worst corruption under STD: '{worst_c}' (mean_err={per_corr_err['STD'][worst_c]:.3f})")
    log()

    # ---- per-class breakdown for worst corruption (averaged over severities) ----
    log(f"Per-class accuracy on worst corruption '{worst_c}' (averaged over severities):")
    log("  class                STD     PGD-AT   AUGMIX")
    Y_np = Yte.detach().cpu().numpy()
    per_class_acc = {name: np.zeros(10, dtype=np.float64) for name in models}
    for sev in SEVERITIES:
        r_local = np.random.default_rng(hash((worst_c, sev)) % (2 ** 32))
        Xc_np = apply_corruption(worst_c, X_clean_np, sev, r_local)
        Xc = torch.from_numpy(Xc_np).to(Yte.device)
        for name, m in models.items():
            m.eval()
            with torch.no_grad():
                preds = []
                for i in range(0, Xc.size(0), 512):
                    preds.append(m(Xc[i:i + 512]).argmax(1).cpu().numpy())
                preds = np.concatenate(preds)
            for k in range(10):
                mask = Y_np == k
                if mask.sum() > 0:
                    per_class_acc[name][k] += (preds[mask] == k).mean()
    for k in range(10):
        log("  {:<2d} {:<14s}     {:.3f}    {:.3f}    {:.3f}".format(
            k, FMNIST_CLASSES[k],
            per_class_acc["STD"][k] / len(SEVERITIES),
            per_class_acc["PGD-AT"][k] / len(SEVERITIES),
            per_class_acc["AUGMIX"][k] / len(SEVERITIES)))
    log()

    # ---- adv-vs-corruption correlation across (model, corruption, severity) cells ----
    # adv robustness proxy = (1 - PGD ASR) of the model (constant per model);
    # we instead correlate per-severity *adversarial error at small eps* vs corruption
    # error. Since model-level adv-ASR doesn't vary across corruptions, we expand
    # the dimension differently: for each (corruption, severity), pair (corr_err, model_adv_err)
    # across the 3 models -- giving 11*5=55 triples of (3,2) we collapse to 3*11*5 = 165
    # (model_adv_err_scalar, corr_err_cell).
    log("=" * 78)
    log("Adv-robustness vs corruption-error correlation")
    log("=" * 78)
    rows = []   # (corr_err, model_adv_err)
    for name in models:
        for c in CORRUPTIONS:
            for s in SEVERITIES:
                rows.append((1.0 - acc_table[(name, c, s)], asr[name]))
    ce_arr = np.array([r[0] for r in rows])
    ae_arr = np.array([r[1] for r in rows])
    log(f"  n_cells = {len(rows)}  (3 models x 11 corruptions x 5 severities)")
    log(f"  Pearson  corr( corruption_err , model_PGD_ASR ): {pearson(ce_arr, ae_arr):+.4f}")
    log(f"  Spearman corr( corruption_err , model_PGD_ASR ): {spearman(ce_arr, ae_arr):+.4f}")
    log("  (positive = models that are more adv-vulnerable also more corruption-vulnerable;")
    log("   near zero / negative = decorrelated axes, consistent with Hendrycks 2019)")
    log()

    # Also: model-level summary (3 points)
    log("  Model-level (3 points): mCE vs PGD-ASR")
    for name in ["STD", "PGD-AT", "AUGMIX"]:
        log(f"    {name:>7s}: mCE={mce[name]:.4f}   PGD-ASR={asr[name]:.4f}")
    m_ce = np.array([mce[n] for n in ["STD", "PGD-AT", "AUGMIX"]])
    m_ae = np.array([asr[n] for n in ["STD", "PGD-AT", "AUGMIX"]])
    log(f"    Pearson  (3-pt) corr(mCE, PGD-ASR): {pearson(m_ce, m_ae):+.4f}  "
        f"(unreliable at n=3, reported for completeness)")
    log()

    # ---- HEADLINE verdict ----
    log("=" * 78)
    log("HEADLINE")
    log("=" * 78)
    at_vs_std_adv = asr["STD"] - asr["PGD-AT"]            # positive => PGD-AT more adv-robust
    at_vs_std_mce = mce["STD"] - mce["PGD-AT"]            # positive => PGD-AT lower corruption err
    aug_vs_std_mce = mce["STD"] - mce["AUGMIX"]           # positive => AUGMIX lower corruption err

    log(f"  PGD-AT - STD  : adv-ASR delta = {-at_vs_std_adv:+.4f}  "
        f"(negative = PGD-AT more adv-robust)")
    log(f"  PGD-AT - STD  : mCE delta     = {-at_vs_std_mce:+.4f}  "
        f"(negative = PGD-AT less corruption-vulnerable)")
    log(f"  AUGMIX - STD  : mCE delta     = {-aug_vs_std_mce:+.4f}  "
        f"(negative = AUGMIX less corruption-vulnerable)")
    log()

    # Decision logic
    adv_gain = at_vs_std_adv > 0.10            # PGD-AT meaningfully more adv-robust
    corr_gain_at = at_vs_std_mce > 0.02        # PGD-AT meaningfully better on corruptions
    corr_gain_aug = aug_vs_std_mce > 0.02      # AugMix meaningfully better on corruptions

    if adv_gain and (not corr_gain_at) and corr_gain_aug:
        verdict = ("SUPPORTED: PGD-AT buys adversarial robustness but not natural-corruption "
                   "robustness; AugMix-style augmentation does close the corruption gap. "
                   "Matches Hendrycks 2019 + AugMix.")
    elif adv_gain and corr_gain_at and corr_gain_aug:
        verdict = ("PARTIAL: PGD-AT helps BOTH axes (unexpected for ImageNet-C; possible on "
                   "low-resolution Fashion-MNIST where adv-perturbations look like noise).")
    elif adv_gain and (not corr_gain_at) and (not corr_gain_aug):
        verdict = ("REFUTED for the AugMix control: neither intervention helps corruptions on "
                   "this 28x28 grayscale suite; gap analysis needed.")
    elif (not adv_gain):
        verdict = ("INCONCLUSIVE: PGD-AT did not even gain adversarial robustness "
                   "(training may have failed; investigate before reading corruption deltas).")
    else:
        verdict = ("MIXED: see deltas above; manual interpretation required.")
    log("VERDICT: " + verdict)
    log()

    log(f"Elapsed: {time.time() - t0:.1f}s")
    log("DONE")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h475_fmnist_c_corruptions_output.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nOutput saved to: {out_path}")


if __name__ == "__main__":
    main()
