"""
H411 - VOneNet-style fixed Gabor V1 front-end as an adversarial defense.

VOneNet (Dapello et al. 2020) prepends a biologically-inspired V1 block to a CNN:
a FIXED bank of Gabor filters (simple + complex cells) followed by a STOCHASTIC
neuronal-noise layer. The claim is that a Gabor front-end + neural noise improves
adversarial robustness. Here we test, on Fashion-MNIST, how much (if any) robustness
the Gabor front-end buys vs a plain SmallCNN, and decompose the gain into:
  - the fixed Gabor filtering itself (sigma_neuron = 0, deterministic), vs
  - the stochastic neuronal noise (sigma_neuron > 0).
And we check whether any stochastic gain is REAL or just gradient masking, by
comparing standard PGD (single noise draw per step) against EOT-PGD (gradient
averaged over multiple noise draws) on the best stochastic model.

VOneBlock design (FIXED, requires_grad=False):
  - Gabor bank: 8 orientations x 2 spatial frequencies x 2 phases (0, pi/2) = 32
    quadrature filters. 9x9 kernels, stride 2, padding 4 (28x28 -> 14x14).
  - Simple cells   = half-rectified raw Gabor responses (32 ch).
  - Complex cells  = phase-invariant magnitude sqrt(g_0^2 + g_pi/2^2) over each
    quadrature pair (16 ch).  Front-end output = concat -> 48 channels.
  - Stochastic layer: additive Gaussian noise with std sigma_neuron, ACTIVE at
    BOTH train and inference (it is the stochastic defense).
  - Then a trainable conv backbone + head -> 10 classes (mirrors SmallCNN head).

Conditions (same training budget):
  1. baseline plain SmallCNN
  2. VOneNet, sigma_neuron = 0     (deterministic Gabor only)
  3. VOneNet, sigma_neuron = 0.1   (small noise)
  4. VOneNet, sigma_neuron = 0.25  (larger noise)

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD mom=0.9 wd=5e-4,
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01. CNN width=32.
"""
import os
import sys
import math
import time

import numpy as np
import torch
import torch.nn as nn
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
WIDTH = 32

# VOneBlock
N_ORIENT = 8
FREQS = [0.20, 0.35]        # cycles/pixel
KSIZE = 9
STRIDE = 2
PADDING = 4
EOT_SAMPLES = 8             # noise draws per EOT-PGD step

SIGMAS = [0.0, 0.1, 0.25]

META = {"channels": 1, "size": 28, "n_classes": 10}


# ---------------------------------------------------------------------------
# Gabor kernel generation (analytic)
# ---------------------------------------------------------------------------
def gabor_kernel(ksize, theta, freq, phase, sigma):
    """Standard Gabor = Gaussian envelope * oriented sinusoid. Returns ksize x ksize."""
    half = (ksize - 1) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(ksize, dtype=torch.float32) - half,
        torch.arange(ksize, dtype=torch.float32) - half,
        indexing="ij")
    # rotate coords
    xr = xs * math.cos(theta) + ys * math.sin(theta)
    yr = -xs * math.sin(theta) + ys * math.cos(theta)
    envelope = torch.exp(-(xr ** 2 + yr ** 2) / (2.0 * sigma ** 2))
    carrier = torch.cos(2.0 * math.pi * freq * xr + phase)
    g = envelope * carrier
    g = g - g.mean()                      # zero-mean (remove DC)
    g = g / (g.norm() + 1e-8)             # unit L2 norm
    return g


def build_gabor_bank(ksize, n_orient, freqs):
    """Return weight tensor (n_pairs*2, 1, k, k) ordered so that for each
    (orientation, freq) pair the two phases (0, pi/2) are ADJACENT, plus the
    number of quadrature pairs. Gaussian sigma scales with wavelength."""
    kernels = []
    for f in freqs:
        wavelength = 1.0 / f
        sigma = 0.56 * wavelength          # ~1.4 octave bandwidth (VOneNet-like)
        for o in range(n_orient):
            theta = math.pi * o / n_orient
            k0 = gabor_kernel(ksize, theta, f, 0.0, sigma)
            k90 = gabor_kernel(ksize, theta, f, math.pi / 2.0, sigma)
            kernels.append(k0)
            kernels.append(k90)
    W = torch.stack(kernels, dim=0).unsqueeze(1)   # (P*2, 1, k, k)
    n_pairs = len(freqs) * n_orient
    return W, n_pairs


class VOneBlock(nn.Module):
    """Fixed Gabor simple+complex cells + stochastic Gaussian neuronal noise."""
    def __init__(self, in_ch=1, ksize=KSIZE, n_orient=N_ORIENT, freqs=FREQS,
                 stride=STRIDE, padding=PADDING, sigma_neuron=0.0):
        super().__init__()
        assert in_ch == 1, "Gabor bank built for single-channel input"
        W, n_pairs = build_gabor_bank(ksize, n_orient, freqs)   # (2P,1,k,k)
        self.n_pairs = n_pairs
        self.sigma_neuron = float(sigma_neuron)
        self.conv = nn.Conv2d(1, W.size(0), ksize, stride=stride,
                              padding=padding, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(W)
        self.conv.weight.requires_grad_(False)
        # simple cells = 2P channels (half-rect), complex = P channels (magnitude)
        self.out_channels = 2 * n_pairs + n_pairs

    def forward(self, x):
        g = self.conv(x)                                   # (B, 2P, H, W)
        # quadrature pairs are adjacent: even = phase 0, odd = phase pi/2
        g0 = g[:, 0::2]                                    # (B, P, H, W)
        g90 = g[:, 1::2]                                   # (B, P, H, W)
        simple = F.relu(g)                                 # half-rectified, 2P ch
        complex_ = torch.sqrt(g0 ** 2 + g90 ** 2 + 1e-6)   # phase-invariant, P ch
        feat = torch.cat([simple, complex_], dim=1)        # (B, 3P, H, W)
        if self.sigma_neuron > 0:
            # stochastic neuronal noise, active in BOTH train and eval modes
            feat = feat + self.sigma_neuron * torch.randn_like(feat)
        return feat


class VOneNet(nn.Module):
    """VOneBlock -> trainable conv backbone + head (mirrors SmallCNN head sizing)."""
    def __init__(self, meta, width=WIDTH, sigma_neuron=0.0):
        super().__init__()
        ch, sz, ncls = meta["channels"], meta["size"], meta["n_classes"]
        self.vone = VOneBlock(in_ch=ch, sigma_neuron=sigma_neuron)
        c_in = self.vone.out_channels
        # front-end maps 28x28 -> 14x14 (stride 2). Backbone: two pooled blocks.
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(
            *block(c_in, width * 2),       # 14 -> 7
            *block(width * 2, width * 4))  # 7  -> 3
        feat = (sz // 2) // 4              # 14//4 = 3
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 4 * feat * feat, 256), nn.ReLU(),
            nn.Linear(256, ncls))

    def forward(self, x):
        return self.head(self.features(self.vone(x)))


# ---------------------------------------------------------------------------
# training (config SGD mom=0.9 wd=5e-4)
# ---------------------------------------------------------------------------
def train_net(model, Xtr, Ytr, seed):
    C.set_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# EOT-PGD: average the loss gradient over multiple noise draws per step.
# ---------------------------------------------------------------------------
def eot_pgd(model, x, y, eps, steps, alpha, n_samples, random_start=True):
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        xa = (xa + torch.empty_like(xa).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        grad = torch.zeros_like(xa)
        for _s in range(n_samples):
            loss = F.cross_entropy(model(xa), y)
            g, = torch.autograd.grad(loss, xa, retain_graph=False)
            grad = grad + g
            xa.requires_grad_(True)
        grad = grad / n_samples
        xa = xa.detach() + alpha * grad.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


@torch.no_grad()
def _flip_eval(model, X, Y, Xadv, batch=256):
    flips, corr = [], []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        xa = Xadv[i:i + batch]
        corr.append((model(x).argmax(1) == y).cpu())
        flips.append((model(xa).argmax(1) != y).cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() else float("nan")


def eot_pgd_asr(model, X, Y, batch=256):
    model.eval()
    advs = []
    for i in range(0, X.size(0), batch):
        advs.append(eot_pgd(model, X[i:i + batch], Y[i:i + batch], EPS,
                            PGD_STEPS, PGD_ALPHA, EOT_SAMPLES))
    Xadv = torch.cat(advs, dim=0)
    return _flip_eval(model, X, Y, Xadv)


def eval_robustness(model, X, Y):
    """clean acc (averaged over a few noise draws for stochastic models),
    FGSM ASR, PGD ASR."""
    _, acc = C.logits_and_acc(model, X, Y)
    fg = C.attack_success(model, X, Y, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h411_vonenet_gabor_frontend_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H411  VOneNet-style fixed Gabor V1 front-end as a defense (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} "
        f"width={WIDTH}")
    out(f"        VOneBlock: {N_ORIENT} orient x {len(FREQS)} freq x 2 phase, "
        f"k={KSIZE} stride={STRIDE} pad={PADDING}, freqs={FREQS}")
    out(f"        sigma_neuron conditions = {SIGMAS}; EOT_SAMPLES={EOT_SAMPLES}")
    out(f"        device = {C.DEVICE}")
    out("")

    # ---- Gabor kernel sanity check ----
    W, n_pairs = build_gabor_bank(KSIZE, N_ORIENT, FREQS)
    out(f"[0] Gabor bank: weight shape={tuple(W.shape)}  quadrature pairs={n_pairs}  "
        f"front-end out channels={3 * n_pairs}")
    out(f"    kernel stats: mean(|mean|)={W.mean(dim=(1,2,3)).abs().mean():.2e} "
        f"(should ~0, DC-removed), per-kernel L2 in "
        f"[{W.flatten(1).norm(dim=1).min():.3f},{W.flatten(1).norm(dim=1).max():.3f}] "
        f"(should ~1.0)")
    # quadrature orthogonality: pair (phase 0 vs pi/2) inner product ~ small
    ip = (W[0::2].flatten(1) * W[1::2].flatten(1)).sum(1)
    out(f"    quadrature pair <phase0,phase90> dot: mean={ip.mean():.3f} "
        f"max|.|={ip.abs().max():.3f} (should be small => near-orthogonal)")
    out("")

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    rows = []

    # ---- condition 1: baseline plain SmallCNN ----
    out("[1] baseline plain SmallCNN ...")
    C.set_seed(SEED)
    base = C.build_model("cnn", META, width=WIDTH)
    base = train_net(base, Xtr, Ytr, SEED)
    b_acc, b_fg, b_pg = eval_robustness(base, Xte, Yte)
    rows.append({"cond": "baseline SmallCNN", "acc": b_acc, "fg": b_fg,
                 "pg": b_pg, "eot": None})
    out(f"    clean_acc={b_acc:.4f}  FGSM_ASR={b_fg:.4f}  PGD_ASR={b_pg:.4f}  "
        f"({time.time()-t0:.0f}s)")
    flush_file()

    # ---- conditions 2..4: VOneNet at each sigma ----
    vone_models = {}
    for sg in SIGMAS:
        out(f"\n[VOneNet] sigma_neuron={sg} ...")
        C.set_seed(SEED)
        model = VOneNet(META, width=WIDTH, sigma_neuron=sg).to(C.DEVICE)
        model = train_net(model, Xtr, Ytr, SEED)
        acc, fg, pg = eval_robustness(model, Xte, Yte)
        cond = f"VOneNet sig={sg}"
        rows.append({"cond": cond, "acc": acc, "fg": fg, "pg": pg, "eot": None})
        vone_models[sg] = model
        out(f"    clean_acc={acc:.4f}  FGSM_ASR={fg:.4f}  PGD_ASR={pg:.4f}  "
            f"d_PGD_ASR vs base={pg-b_pg:+.4f}  ({time.time()-t0:.0f}s)")
        flush_file()

    # ---- EOT-PGD on the best stochastic model (lowest PGD_ASR among sigma>0) ----
    stoch = [r for r in rows if r["cond"].startswith("VOneNet sig=") and
             float(r["cond"].split("=")[1]) > 0]
    out("\n" + "=" * 80)
    out("[EOT] EOT-PGD gradient-masking check on best stochastic model")
    out("=" * 80)
    if stoch:
        best_stoch = min(stoch, key=lambda r: r["pg"])
        best_sig = float(best_stoch["cond"].split("=")[1])
        out(f"    best stochastic (lowest PGD_ASR): {best_stoch['cond']} "
            f"(PGD_ASR={best_stoch['pg']:.4f})")
        out(f"    running EOT-PGD with {EOT_SAMPLES} noise draws/step ...")
        eot_asr = eot_pgd_asr(vone_models[best_sig], Xte, Yte)
        # annotate the matching row
        for r in rows:
            if r["cond"] == best_stoch["cond"]:
                r["eot"] = eot_asr
        out(f"    EOT-PGD_ASR={eot_asr:.4f}  vs PGD_ASR={best_stoch['pg']:.4f}  "
            f"(rise={eot_asr-best_stoch['pg']:+.4f})  ({time.time()-t0:.0f}s)")
    else:
        eot_asr = None
        best_stoch = None
        out("    no stochastic condition available")
    flush_file()

    # ---- MAIN TABLE ----
    out("\n" + "=" * 80)
    out("[TABLE] condition | clean_acc | FGSM_ASR | PGD_ASR | EOT-PGD_ASR")
    out("=" * 80)
    hdr = "{:<22} {:>10} {:>10} {:>9} {:>12}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR", "EOT-PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        eot_s = "{:>12.4f}".format(r["eot"]) if r["eot"] is not None else "{:>12}".format("-")
        out("{:<22} {:>10.4f} {:>10.4f} {:>9.4f} {}".format(
            r["cond"], r["acc"], r["fg"], r["pg"], eot_s))
    out("-" * len(hdr))

    # ---- VERDICT ----
    out("\n" + "=" * 80)
    out("[VERDICT]")
    out("=" * 80)
    base_row = rows[0]
    sig0 = next(r for r in rows if r["cond"] == "VOneNet sig=0.0")
    gabor_gain = base_row["pg"] - sig0["pg"]              # gain from fixed Gabor alone
    best_vone = min((r for r in rows if r["cond"].startswith("VOneNet")),
                    key=lambda r: r["pg"])
    total_gain = base_row["pg"] - best_vone["pg"]
    stoch_gain = sig0["pg"] - best_vone["pg"]             # extra from sigma>0 vs sigma=0

    out(f"  baseline PGD_ASR                = {base_row['pg']:.4f} "
        f"(clean_acc {base_row['acc']:.4f})")
    out(f"  VOneNet sig=0 PGD_ASR           = {sig0['pg']:.4f} "
        f"(clean_acc {sig0['acc']:.4f})")
    out(f"  best VOneNet ({best_vone['cond']}) PGD_ASR = {best_vone['pg']:.4f} "
        f"(clean_acc {best_vone['acc']:.4f})")
    out("")
    out(f"  PGD robustness gain, fixed-Gabor only (base - sig0)   = {gabor_gain:+.4f}")
    out(f"  PGD robustness gain, extra from stochasticity         = {stoch_gain:+.4f}")
    out(f"  PGD robustness gain, total (base - best VOneNet)      = {total_gain:+.4f}")
    if best_stoch is not None and best_stoch["eot"] is not None:
        eot_rise = best_stoch["eot"] - best_stoch["pg"]
        out(f"  EOT-PGD vs PGD on {best_stoch['cond']}: "
            f"{best_stoch['pg']:.4f} -> {best_stoch['eot']:.4f} ({eot_rise:+.4f})")
        masking = eot_rise > 0.05
    else:
        eot_rise = None
        masking = False

    out("")
    # build a one-line verdict
    parts = []
    if total_gain > 0.02:
        parts.append(f"Gabor front-end REDUCES PGD-ASR by {total_gain:.3f} vs baseline")
    elif total_gain < -0.02:
        parts.append(f"Gabor front-end INCREASES PGD-ASR by {-total_gain:.3f} (worse)")
    else:
        parts.append("Gabor front-end gives negligible PGD-ASR change")
    # decompose
    if gabor_gain > 0.02:
        parts.append(f"fixed Gabor (sig=0) contributes {gabor_gain:+.3f}")
    else:
        parts.append(f"fixed Gabor alone (sig=0) contributes little ({gabor_gain:+.3f})")
    if stoch_gain > 0.02:
        parts.append(f"stochasticity adds {stoch_gain:+.3f}")
    else:
        parts.append(f"stochasticity adds little ({stoch_gain:+.3f})")
    # masking judgement
    if eot_rise is not None:
        if masking:
            parts.append(f"but EOT-PGD erases {eot_rise:.3f} of it => stochastic gain "
                         f"is largely GRADIENT MASKING")
        else:
            parts.append(f"EOT-PGD barely changes ASR ({eot_rise:+.3f}) => stochastic "
                         f"gain is mostly REAL (not pure masking)")
    out("  ONE-LINE VERDICT: " + "; ".join(parts) + ".")

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
