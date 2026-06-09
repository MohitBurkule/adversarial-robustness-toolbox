"""
H416 - Scattering Transform Front-End for Adversarial Robustness.

Hypothesis: A fixed 2nd-order wavelet scattering transform (Bruna & Mallat 2013,
"Invariant Scattering Convolution Networks", IEEE TPAMI) applied as a non-trainable
front-end removes high-frequency adversarial perturbations through multi-scale
Morlet-wavelet decomposition, yielding translation/small-deformation invariance and
reduced attack surface — connecting to h411 (Gabor filtering alone insufficient;
stochasticity adds the gain) by asking whether principled multi-scale invariance
(not ad-hoc single-band filtering) closes that gap without stochasticity.

Three conditions compared on Fashion-MNIST:
  A. BASELINE:             SmallCNN trained/evaluated on raw images.
  B. SCATTERING:           Fixed scattering front-end (pure PyTorch, analytical
                           Morlet/Gabor kernels) → SmallCNN backbone. Front-end
                           frozen; only backbone trained.
  C. SCATTERING+NOISE:     Same scattering front-end + stochastic Gaussian noise
                           injected on the scattering coefficients at train AND
                           test time (tests h411's finding that noise helps).

Scattering implementation (pure torch, no external deps):
  - 1st-order: convolve with J scales × L orientations real Morlet wavelets,
    take modulus, average-pool (local averaging = lowpass smoothing S_1).
  - 2nd-order: for each 1st-order envelope, convolve with finer-scale wavelets
    (scale j2 < j1), take modulus, average-pool (S_2).
  - Zeroth order: Gaussian lowpass of original image (S_0).
  - All scattering layers are analytical: kernels are computed once, stored as
    frozen nn.Conv2d weights.  No learnable parameters in the front-end.

Config: DS=fashion_mnist, N_TRAIN=6000, N_EVAL=2000, EPOCHS=10, LR=0.05,
BATCH=128, SGD(mom=0.9,wd=5e-4), SEED=0, EPS=0.1, PGD_STEPS=10,
J=3 (scales), L=4 (orientations), SCAT_NOISE_STD=0.05.
CNN width=32 on top of scattering feature channels.

OUT_FILE: results/fashion_mnist/h416_scattering_transform_frontend_output.txt

References:
  Bruna, J. & Mallat, S. (2013). Invariant Scattering Convolution Networks.
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8), 1872-1886.
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
# Scattering hyper-params
J = 3          # number of dyadic scales (2^1, 2^2, 2^3 pixels)
L = 4          # number of orientations (0..pi in L steps)
SCAT_NOISE_STD = 0.05   # Gaussian noise std on scattering coefficients (cond C)

META = {"channels": 1, "size": 28, "n_classes": 10}
IMG_SIZE = 28


# ---- Morlet / Gabor wavelet kernel builders ----------------------------------

def _morlet_kernel(size: int, scale: float, angle: float) -> torch.Tensor:
    """Return a real-valued Morlet (Gabor envelope) kernel of shape (size, size).

    Morlet wavelet: real part of complex Morlet = Gaussian envelope * cos(wave).
    psi(x) = exp(-|x|^2 / (2*sigma^2)) * cos(xi_0 . x)
    where sigma = scale, xi_0 = (k0/scale) * (cos(angle), sin(angle)), k0=pi.

    The kernel is mean-subtracted to approximate zero DC (band-pass property).
    """
    half = size // 2
    yy, xx = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )
    # trim to exactly (size, size)
    yy = yy[:size, :size]
    xx = xx[:size, :size]

    sigma = scale
    envelope = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    # carrier: wave-vector in direction `angle`
    k0 = math.pi
    kx = (k0 / sigma) * math.cos(angle)
    ky = (k0 / sigma) * math.sin(angle)
    carrier = torch.cos(kx * xx + ky * yy)
    kernel = envelope * carrier
    # zero-mean (removes DC)
    kernel = kernel - kernel.mean()
    # normalise to unit norm
    norm = kernel.norm()
    if norm > 1e-8:
        kernel = kernel / norm
    return kernel  # (size, size)


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Isotropic Gaussian lowpass kernel (size x size), unit-normalised."""
    half = size // 2
    yy, xx = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )
    yy = yy[:size, :size]
    xx = xx[:size, :size]
    g = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


# ---- Scattering transform module ---------------------------------------------

class ScatteringFrontend(nn.Module):
    """Fixed (non-trainable) 2nd-order wavelet scattering front-end.

    Produces scattering coefficients S0, {S1_j1}, {S2_j1_j2_theta2} as a
    multi-channel feature map at the same spatial resolution (via reflect-padding
    and average-pooling to match output size).

    Output: (B, C_scat, H_out, W_out) where H_out = W_out = IMG_SIZE // (2**J)
    and C_scat = 1 + J*L + J*(J-1)//2 * L (zeroth + first + second order).

    All convolutions use frozen weights; no gradients flow through this module.
    """

    def __init__(self, in_channels: int = 1, img_size: int = 28,
                 J: int = 3, L: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.J = J
        self.L = L

        # spatial output after pooling: divide by 2^J
        self.out_spatial = img_size // (2 ** J)

        # build Morlet kernel banks: one per (j, theta)
        # scale at level j: sigma_j = 2^(j+1), kernel size = 4*sigma+1 (clipped)
        self._build_kernel_banks()

        # precompute output channel count
        # S0: 1 channel
        # S1: J * L channels
        # S2: for j2 > j1 (finer scale AFTER coarser), L orientations each
        #     count = sum_{j1=0}^{J-2} (J - j1 - 1) * L
        n_s2_pairs = sum((J - j1 - 1) for j1 in range(J - 1))
        self.n_channels = 1 + J * L + n_s2_pairs * L
        self.s2_pairs = [(j1, j2) for j1 in range(J - 1)
                         for j2 in range(j1 + 1, J)]

    def _build_kernel_banks(self):
        """Register all Morlet and Gaussian kernels as frozen buffers."""
        J, L = self.J, self.L

        # Gaussian lowpass: used to average-pool each order
        # sigma for averaging at scale j: 2^j
        for j in range(J):
            sigma_avg = float(2 ** (j + 1))
            ksz = min(int(4 * sigma_avg) | 1, self.img_size)  # odd
            if ksz % 2 == 0:
                ksz += 1
            g = _gaussian_kernel(ksz, sigma_avg)  # (ksz, ksz)
            # shape (1,1,ksz,ksz) for depthwise conv
            self.register_buffer(f"gauss_{j}", g.unsqueeze(0).unsqueeze(0))

        # Morlet wavelet kernels: for each (j, theta)
        for j in range(J):
            sigma_w = float(2 ** (j + 1))
            ksz = min(int(4 * sigma_w) | 1, self.img_size)
            if ksz % 2 == 0:
                ksz += 1
            for l in range(L):
                angle = math.pi * l / L
                k = _morlet_kernel(ksz, sigma_w, angle)  # (ksz, ksz)
                self.register_buffer(f"morlet_{j}_{l}",
                                     k.unsqueeze(0).unsqueeze(0))

        # Final global average pool to fixed spatial size
        self._pool_size = self.out_spatial

    def _conv_pad(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Reflect-pad and convolve (depthwise-style, single channel)."""
        kH, kW = weight.shape[-2], weight.shape[-1]
        pH, pW = kH // 2, kW // 2
        xp = F.pad(x, (pW, pW, pH, pH), mode="reflect")
        return F.conv2d(xp, weight)  # (B, 1, H, W)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)  in [0,1]
        Returns: (B, n_channels, out_spatial, out_spatial)
        """
        B = x.size(0)
        out_sp = self._pool_size
        channels = []

        # --- S0: lowpass of original at coarsest scale (j=J-1) ---------------
        g_coarse = getattr(self, f"gauss_{self.J - 1}")
        s0 = self._conv_pad(x, g_coarse)  # (B,1,H,W)
        s0 = F.adaptive_avg_pool2d(s0, (out_sp, out_sp))
        channels.append(s0)  # 1 channel

        # --- S1: |x * psi_j1| averaged at scale j1 ---------------------------
        # store envelopes U1 for second-order computation
        U1 = {}  # (j1, l1) -> (B,1,H,W) modulus before averaging
        for j1 in range(self.J):
            g_j1 = getattr(self, f"gauss_{j1}")
            for l1 in range(self.L):
                psi = getattr(self, f"morlet_{j1}_{l1}")
                u = self._conv_pad(x, psi).abs()   # (B,1,H,W)
                U1[(j1, l1)] = u
                s1 = self._conv_pad(u, g_j1)
                s1 = F.adaptive_avg_pool2d(s1, (out_sp, out_sp))
                channels.append(s1)  # J*L channels total

        # --- S2: ||U1_j1 * psi_j2| averaged at scale j2, for j2 > j1 --------
        for (j1, j2) in self.s2_pairs:
            g_j2 = getattr(self, f"gauss_{j2}")
            for l1 in range(self.L):
                u1 = U1[(j1, l1)]  # (B,1,H,W)
                for l2 in range(self.L):
                    psi2 = getattr(self, f"morlet_{j2}_{l2}")
                    u2 = self._conv_pad(u1, psi2).abs()
                    s2 = self._conv_pad(u2, g_j2)
                    s2 = F.adaptive_avg_pool2d(s2, (out_sp, out_sp))
                    channels.append(s2)

        out = torch.cat(channels, dim=1)  # (B, n_channels, out_sp, out_sp)
        return out

    def extra_repr(self):
        return (f"J={self.J}, L={self.L}, "
                f"out_channels={self.n_channels}, "
                f"out_spatial={self.out_spatial}")


# ---- Scattering + SmallCNN composite model -----------------------------------

class ScatteringCNN(nn.Module):
    """Non-trainable ScatteringFrontend -> trainable SmallCNN backbone."""

    def __init__(self, J: int = 3, L: int = 4, img_size: int = 28,
                 n_classes: int = 10, width: int = 32,
                 noise_std: float = 0.0, training_noise: bool = False):
        """
        noise_std:       std of Gaussian noise added to scattering coefficients.
        training_noise:  if True, add noise only during training (stochastic);
                         if False, always add (test-time noise, usually noise_std=0).
        For condition C: noise_std=SCAT_NOISE_STD, training_noise=False (always on).
        """
        super().__init__()
        self.scat = ScatteringFrontend(in_channels=1, img_size=img_size, J=J, L=L)
        self.noise_std = noise_std
        self.training_noise = training_noise

        n_scat_ch = self.scat.n_channels
        out_sp = self.scat.out_spatial  # spatial size after scattering

        # SmallCNN backbone adapted to scattering output size
        # Since out_sp is small (28//8=3 for J=3), use a shallower version
        # to avoid spatial collapse: 2 conv blocks instead of 3.
        act = nn.ReLU
        self.backbone = nn.Sequential(
            nn.Conv2d(n_scat_ch, width, 3, padding=1),
            nn.BatchNorm2d(width),
            act(),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            act(),
            nn.Flatten(),
            nn.Linear(width * 2 * out_sp * out_sp, 256),
            act(),
            nn.Linear(256, n_classes),
        )

    def _add_noise(self, feat: torch.Tensor) -> torch.Tensor:
        if self.noise_std <= 0.0:
            return feat
        if self.training_noise and not self.training:
            return feat  # noise only at train time
        return feat + torch.randn_like(feat) * self.noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.scat(x)
        feat = self._add_noise(feat)
        return self.backbone(feat)


# ---- training helpers --------------------------------------------------------

def _make_sgd(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def train_model(model, Xtr, Ytr, seed):
    C.set_seed(seed)
    model.train()
    opt = _make_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
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


def eval_all(model, Xte, Yte):
    """Returns dict: clean_acc, fgsm_asr, pgd_asr."""
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return {"clean_acc": acc, "fgsm_asr": fg["asr"], "pgd_asr": pg["asr"]}


# ---- main --------------------------------------------------------------------

def main():
    t0 = time.time()
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "fashion_mnist",
        "h416_scattering_transform_frontend_output.txt",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H416  Scattering Transform Front-End  (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: DS={DS} N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} "
        f"LR={LR} BATCH={BATCH}")
    out(f"        SGD(mom=0.9,wd=5e-4) SEED={SEED} EPS={EPS} "
        f"PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        J={J} L={L} SCAT_NOISE_STD={SCAT_NOISE_STD}")
    out(f"        device={C.DEVICE}")
    out("")
    out("Reference: Bruna & Mallat (2013) Invariant Scattering Convolution "
        "Networks, IEEE TPAMI 35(8).")
    out("Connection: h411 showed Gabor filtering alone does not help; "
        "stochasticity does. H416 tests principled 2nd-order scattering.")
    out("")

    # ---- data ----------------------------------------------------------------
    Xtr, Ytr, Xte, Yte = C.load_dataset(
        DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    # ---- probe scattering shape ----------------------------------------------
    scat_probe = ScatteringFrontend(in_channels=1, img_size=IMG_SIZE, J=J, L=L)
    scat_probe.to(C.DEVICE)
    with torch.no_grad():
        probe_out = scat_probe(Xtr[:2])
    out(f"ScatteringFrontend output shape: {tuple(probe_out.shape)}  "
        f"(channels={scat_probe.n_channels}, spatial={scat_probe.out_spatial}x"
        f"{scat_probe.out_spatial})")
    out("")
    del scat_probe, probe_out

    results = {}

    # =========================================================================
    # CONDITION A: BASELINE (raw SmallCNN, no scattering)
    # =========================================================================
    out("-" * 60)
    out("[A] BASELINE: raw SmallCNN (no scattering)")
    out("-" * 60)
    C.set_seed(SEED)
    baseline = C.SmallCNN(in_ch=1, size=28, n_classes=10, width=32,
                          act="relu", bn=True).to(C.DEVICE)
    train_model(baseline, Xtr, Ytr, SEED)
    res_a = eval_all(baseline, Xte, Yte)
    results["A_baseline"] = res_a
    out(f"  clean_acc={res_a['clean_acc']:.4f}  "
        f"FGSM_ASR={res_a['fgsm_asr']:.4f}  "
        f"PGD_ASR={res_a['pgd_asr']:.4f}")
    out(f"  elapsed: {time.time()-t0:.0f}s")
    flush_file()
    out("")

    # =========================================================================
    # CONDITION B: SCATTERING FRONT-END (no noise)
    # =========================================================================
    out("-" * 60)
    out("[B] SCATTERING front-end only (noise_std=0)")
    out("-" * 60)
    C.set_seed(SEED)
    scat_model = ScatteringCNN(
        J=J, L=L, img_size=IMG_SIZE, n_classes=10, width=32,
        noise_std=0.0).to(C.DEVICE)
    out(f"  scattering channels in backbone: {scat_model.scat.n_channels}")
    train_model(scat_model, Xtr, Ytr, SEED)
    res_b = eval_all(scat_model, Xte, Yte)
    results["B_scattering"] = res_b
    out(f"  clean_acc={res_b['clean_acc']:.4f}  "
        f"FGSM_ASR={res_b['fgsm_asr']:.4f}  "
        f"PGD_ASR={res_b['pgd_asr']:.4f}")
    out(f"  delta vs baseline: "
        f"clean={res_b['clean_acc']-res_a['clean_acc']:+.4f}  "
        f"FGSM_ASR={res_b['fgsm_asr']-res_a['fgsm_asr']:+.4f}  "
        f"PGD_ASR={res_b['pgd_asr']-res_a['pgd_asr']:+.4f}")
    out(f"  elapsed: {time.time()-t0:.0f}s")
    flush_file()
    out("")

    # =========================================================================
    # CONDITION C: SCATTERING + STOCHASTIC NOISE on coefficients
    # =========================================================================
    out("-" * 60)
    out(f"[C] SCATTERING + noise on coefficients (noise_std={SCAT_NOISE_STD}, "
        f"always-on train+test)")
    out("-" * 60)
    C.set_seed(SEED)
    scat_noise_model = ScatteringCNN(
        J=J, L=L, img_size=IMG_SIZE, n_classes=10, width=32,
        noise_std=SCAT_NOISE_STD, training_noise=False).to(C.DEVICE)
    train_model(scat_noise_model, Xtr, Ytr, SEED)
    res_c = eval_all(scat_noise_model, Xte, Yte)
    results["C_scattering_noise"] = res_c
    out(f"  clean_acc={res_c['clean_acc']:.4f}  "
        f"FGSM_ASR={res_c['fgsm_asr']:.4f}  "
        f"PGD_ASR={res_c['pgd_asr']:.4f}")
    out(f"  delta vs baseline: "
        f"clean={res_c['clean_acc']-res_a['clean_acc']:+.4f}  "
        f"FGSM_ASR={res_c['fgsm_asr']-res_a['fgsm_asr']:+.4f}  "
        f"PGD_ASR={res_c['pgd_asr']-res_a['pgd_asr']:+.4f}")
    out(f"  delta vs scattering-only (B): "
        f"clean={res_c['clean_acc']-res_b['clean_acc']:+.4f}  "
        f"FGSM_ASR={res_c['fgsm_asr']-res_b['fgsm_asr']:+.4f}  "
        f"PGD_ASR={res_c['pgd_asr']-res_b['pgd_asr']:+.4f}")
    out(f"  elapsed: {time.time()-t0:.0f}s")
    flush_file()
    out("")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    out("=" * 80)
    out("SUMMARY TABLE")
    out("=" * 80)
    hdr = "{:<30} {:>10} {:>10} {:>10}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for label, res in [
        ("A baseline (raw CNN)", res_a),
        ("B scattering only", res_b),
        ("C scattering+noise", res_c),
    ]:
        out("{:<30} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            label, res["clean_acc"], res["fgsm_asr"], res["pgd_asr"]))
    out("-" * len(hdr))
    out("")

    # =========================================================================
    # VERDICT
    # =========================================================================
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)

    # Is scattering-only better than baseline on PGD?
    scat_pgd_gain = res_a["pgd_asr"] - res_b["pgd_asr"]
    noise_pgd_gain_over_base = res_a["pgd_asr"] - res_c["pgd_asr"]
    noise_pgd_gain_over_scat = res_b["pgd_asr"] - res_c["pgd_asr"]
    acc_drop_b = res_a["clean_acc"] - res_b["clean_acc"]
    acc_drop_c = res_a["clean_acc"] - res_c["clean_acc"]

    out(f"  Scattering-only PGD_ASR reduction vs baseline: {scat_pgd_gain:+.4f} "
        f"({'positive=more robust' if scat_pgd_gain > 0 else 'negative=worse'})")
    out(f"  Scattering+noise PGD_ASR reduction vs baseline: {noise_pgd_gain_over_base:+.4f}")
    out(f"  Noise gain on top of scattering (B->C): {noise_pgd_gain_over_scat:+.4f}")
    out(f"  Clean-acc cost: B={-acc_drop_b:+.4f}  C={-acc_drop_c:+.4f}")
    out("")

    # Classify findings
    THRESH = 0.03
    ACC_TOL = 0.03

    if scat_pgd_gain > THRESH and acc_drop_b < ACC_TOL:
        verdict_b = ("SUPPORTED: scattering front-end meaningfully reduces PGD "
                     "attack success without clean-accuracy cost.")
    elif scat_pgd_gain > THRESH and acc_drop_b >= ACC_TOL:
        verdict_b = ("PARTIAL: scattering reduces PGD ASR but at significant "
                     "clean-accuracy cost.")
    elif scat_pgd_gain <= 0:
        verdict_b = ("REFUTED: scattering front-end does not reduce (and may "
                     "increase) PGD attack success vs raw CNN.")
    else:
        verdict_b = ("WEAK: scattering gives a small PGD reduction below "
                     f"threshold={THRESH} or incurs accuracy cost.")

    if noise_pgd_gain_over_scat > THRESH:
        verdict_c = (f"CONFIRMED (h411 link): adding noise to scattering "
                     f"coefficients gives additional PGD gain ({noise_pgd_gain_over_scat:+.4f}), "
                     "consistent with h411 finding that stochasticity — not "
                     "filtering alone — drives robustness.")
    else:
        verdict_c = (f"NOT CONFIRMED: noise on scattering coefficients does not "
                     f"add meaningful PGD gain ({noise_pgd_gain_over_scat:+.4f}) "
                     "over scattering-only; stochasticity effect may be "
                     "front-end-dependent.")

    out(f"  [B] Scattering-only: {verdict_b}")
    out(f"  [C] Scattering+noise: {verdict_c}")
    out("")
    out(f"done in {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
