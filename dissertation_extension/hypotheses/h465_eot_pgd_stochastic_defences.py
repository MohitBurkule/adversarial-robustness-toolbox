"""
H465 - EOT-PGD audit on EVERY stochastic defence in the campaign.

Gap filled  : M12 (EOT / stochastic-defence audits incomplete).
Anchor      : Athalye, Carlini & Wagner 2018, "Obfuscated Gradients Give a
              False Sense of Security" (ICML 2018). Standard PGD with one
              forward pass per step yields a noisy gradient estimate when the
              defence is stochastic; the attacker should instead average the
              loss-gradient over K fresh samples of the randomness (EOT-PGD).
Extra refs  :
  - Carlini, Athalye et al. 2019 "On Evaluating Adversarial Robustness" -
    explicit checklist item: report ASR as a function of EOT K, a defence
    that is genuine should plateau as K grows; one that obfuscates gradients
    via stochasticity will see ASR climb monotonically.
  - Gao et al. 2022 "On the Limitations of Stochastic Pre-processing
    Defenses" (NeurIPS 2022) - shows most published stochastic defences have
    too little entropy to actually resist a moderate-K EOT attack; the K=1
    vs K=40 gap is the diagnostic.

Critique of standard PGD on stochastic defences
-----------------------------------------------
The campaign trained several stochastic defences:
  * H134 BNN / dropout-eval-on            (MC-dropout active at test time)
  * H132 / H362 snapshot / diversity ens. (random subset of members at eval)
  * H411 VOneNet                          (Gabor V1 + neuronal Gaussian noise)
  * H423 sparse MoE                       (top-k gating noise via aux noise)
  * H421 stochastic depth                 (Bernoulli layer-drop at eval)
  * H262-H272 noise-aug models            (test-time input Gaussian noise)
H397 audited only the noise-aug family. With K=1 PGD the attacker's gradient
samples one realisation of the randomness; whether the step is useful is
itself stochastic. As K grows, the gradient estimator's variance shrinks like
1/sqrt(K); if a defence's reported robustness was buying noise in the
gradient estimator rather than genuine flatness of the worst-case loss, the
ASR will rise as K increases. A *genuine* stochastic defence will exhibit
ASR plateauing for K small (e.g. K=10 already captures the expectation).

Protocol
--------
1. Light retrain of 6 representative stochastic defences plus a deterministic
   baseline:
     baseline_det      - SmallCNN, no stochasticity (reference plateau).
     noise_aug         - input Gaussian noise sigma=0.15 at eval.
     mc_dropout        - dropout p=0.3 active at eval (BNN proxy).
     stochastic_depth  - Bernoulli skip on the 2nd conv block, p_keep=0.5 eval.
     vone_noise        - VOneNet-style Gabor + neuronal noise sigma=0.25.
     sparse_moe        - sparse MoE top-1 with stochastic gate (Gumbel noise).
     ensemble_random   - 4 SmallCNN members, eval picks 2 at random per batch.

2. For each defence, run PGD with EOT samples K in {1, 10, 20, 40} at
   EPS=0.1, 10 PGD steps, alpha=2.5*EPS/steps. The attacker averages the
   *loss* gradient over K fresh forward passes of the defence at the same
   iterate. Defence stochasticity is freshly resampled on every forward
   (attack and final eval). ASR is computed over originally-correct samples,
   with eval averaged over 5 stochastic inference reps for stability.

3. Verdict per defence:
     ASR(K=40) - ASR(K=1) <= +0.03  =>  GENUINE  (plateau)
     ASR(K=40) - ASR(K=1) in (0.03, 0.10] => SUSPECT (mild masking)
     ASR(K=40) - ASR(K=1) > +0.10  =>  MASKED   (Athalye-style obfuscation)
   Also report the SLOPE between K=1 and K=10 to disambiguate quick-saturation
   genuine defences from slow-rising masked ones.

Standard config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128,
SGD(mom=0.9, wd=5e-4), SEED=0, EPS=0.1. ASCII only.
"""
import math
import os
import sys
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
N_EVAL = 1000             # smaller to keep K=40 budget manageable
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 2.5 * EPS / PGD_STEPS    # = 0.025
WIDTH = 32

K_LIST = [1, 10, 20, 40]
EVAL_REPS = 5             # stochastic inference reps for ASR estimate

NOISE_SIGMA = 0.15        # for noise_aug
DROPOUT_P = 0.3           # for mc_dropout
SD_P_KEEP = 0.5           # for stochastic_depth
VONE_SIGMA = 0.25         # for vone_noise
MOE_GUMBEL_TAU = 1.0      # Gumbel temperature for stochastic gate
N_ENS_MEMBERS = 4
N_ENS_SAMPLE = 2          # picked per forward at eval

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "fashion_mnist",
                   "h465_eot_pgd_stochastic_defences_output.txt")

# ---- logging -------------------------------------------------------------
_LINES = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LINES.append(s)


def flush():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(_LINES) + "\n")


# ---------------------------------------------------------------------------
# Stochastic defence modules
# ---------------------------------------------------------------------------
class NoiseAugWrap(nn.Module):
    """Adds fresh Gaussian noise on every forward (train + eval)."""
    def __init__(self, model, sigma=NOISE_SIGMA):
        super().__init__()
        self.model = model
        self.sigma = float(sigma)

    def forward(self, x):
        return self.model((x + self.sigma * torch.randn_like(x)).clamp(0, 1))


class MCDropoutCNN(nn.Module):
    """SmallCNN with always-on dropout (BNN-style approximation)."""
    def __init__(self, p=DROPOUT_P, width=WIDTH, in_ch=1, size=28, n_classes=10):
        super().__init__()
        self.p = float(p)
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(
            *block(in_ch, width),
            *block(width, width * 2),
            *block(width * 2, width * 4))
        feat = size // 8
        self.fc1 = nn.Linear(width * 4 * feat * feat, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        h = self.features(x).flatten(1)
        # F.dropout with training=True forces dropout even at eval
        h = F.dropout(h, p=self.p, training=True)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=self.p, training=True)
        return self.fc2(h)


class StochasticDepthCNN(nn.Module):
    """SmallCNN where the second conv block is randomly skipped (Huang 2016)."""
    def __init__(self, p_keep=SD_P_KEEP, width=WIDTH, in_ch=1, size=28, n_classes=10):
        super().__init__()
        self.p_keep = float(p_keep)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width),
            nn.ReLU(), nn.MaxPool2d(2))
        # block2 is "stochastic depth" gated; must preserve shape -> add 1x1 conv + pool
        self.block2 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2),
            nn.ReLU(), nn.MaxPool2d(2))
        # identity branch with channel-doubling 1x1 conv + pool (keep shape match)
        self.skip2 = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.BatchNorm2d(width * 4),
            nn.ReLU(), nn.MaxPool2d(2))
        feat = size // 8
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 4 * feat * feat, 256), nn.ReLU(),
            nn.Linear(256, n_classes))

    def forward(self, x):
        h = self.block1(x)
        # Bernoulli per-batch (not per-sample) gate
        keep = (torch.rand(1, device=x.device) < self.p_keep).float()
        h2 = self.block2(h)
        s2 = self.skip2(h)
        # scale by 1/p_keep when kept (inverted dropout style) for unbiasedness
        h = keep * (h2 / self.p_keep) + (1.0 - keep) * s2
        h = self.block3(h)
        return self.head(h)


class VOneNoiseCNN(nn.Module):
    """Compact VOneNet-style stochastic Gabor front-end + trainable backbone."""
    def __init__(self, sigma=VONE_SIGMA, width=WIDTH, in_ch=1, size=28, n_classes=10,
                 ksize=9, n_orient=8, freqs=(0.20, 0.35), stride=2, padding=4):
        super().__init__()
        self.sigma = float(sigma)
        kernels = []
        for f in freqs:
            wavelength = 1.0 / f
            sg = 0.56 * wavelength
            for o in range(n_orient):
                theta = math.pi * o / n_orient
                for phase in (0.0, math.pi / 2.0):
                    half = (ksize - 1) / 2.0
                    ys, xs = torch.meshgrid(
                        torch.arange(ksize, dtype=torch.float32) - half,
                        torch.arange(ksize, dtype=torch.float32) - half,
                        indexing="ij")
                    xr = xs * math.cos(theta) + ys * math.sin(theta)
                    yr = -xs * math.sin(theta) + ys * math.cos(theta)
                    env = torch.exp(-(xr ** 2 + yr ** 2) / (2.0 * sg ** 2))
                    car = torch.cos(2.0 * math.pi * f * xr + phase)
                    g = env * car
                    g = g - g.mean()
                    g = g / (g.norm() + 1e-8)
                    kernels.append(g)
        W = torch.stack(kernels, dim=0).unsqueeze(1)
        n_pairs = len(freqs) * n_orient
        self.n_pairs = n_pairs
        self.conv = nn.Conv2d(1, W.size(0), ksize, stride=stride, padding=padding,
                              bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(W)
        self.conv.weight.requires_grad_(False)
        c_in = 2 * n_pairs + n_pairs   # simple + complex
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(
            *block(c_in, width * 2),       # 14 -> 7
            *block(width * 2, width * 4))  # 7 -> 3
        feat = (size // 2) // 4
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 4 * feat * feat, 256), nn.ReLU(),
            nn.Linear(256, n_classes))

    def _vone(self, x):
        g = self.conv(x)
        g0 = g[:, 0::2]
        g90 = g[:, 1::2]
        simple = F.relu(g)
        complex_ = torch.sqrt(g0 ** 2 + g90 ** 2 + 1e-6)
        feat = torch.cat([simple, complex_], dim=1)
        if self.sigma > 0:
            feat = feat + self.sigma * torch.randn_like(feat)
        return feat

    def forward(self, x):
        return self.head(self.features(self._vone(x)))


class SparseMoEStochastic(nn.Module):
    """Sparse MoE with Gumbel-noise gating (stochastic top-1 at eval)."""
    def __init__(self, n_experts=4, tau=MOE_GUMBEL_TAU, width=WIDTH,
                 in_ch=1, size=28, n_classes=10):
        super().__init__()
        self.n_experts = n_experts
        self.tau = float(tau)
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.trunk = nn.Sequential(
            *block(in_ch, width),
            *block(width, width * 2),
            *block(width * 2, width * 4))
        feat = size // 8
        feat_dim = width * 4 * feat * feat
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(),
                           nn.Linear(256, n_classes)) for _ in range(n_experts)])
        self.gate = nn.Linear(feat_dim, n_experts)

    def forward(self, x):
        h = self.trunk(x).flatten(1)
        scores = self.gate(h)
        # Gumbel noise injection -> stochastic argmax (true randomness at eval)
        gumbel = -torch.log(-torch.log(torch.rand_like(scores).clamp_min(1e-9)).clamp_min(1e-9))
        noisy = (scores + gumbel) / self.tau
        soft = F.softmax(noisy, dim=-1)
        _, idx = soft.max(dim=-1, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(1, idx, 1.0)
        # straight-through for grad
        gate_w = hard + soft - soft.detach()
        outs = torch.stack([e(h) for e in self.experts], dim=2)  # (B, C, E)
        return (outs * gate_w.unsqueeze(1)).sum(2)


class RandomSubsetEnsemble(nn.Module):
    """4 SmallCNN members; eval forward picks N_ENS_SAMPLE at random per call."""
    def __init__(self, n_members=N_ENS_MEMBERS, k_sample=N_ENS_SAMPLE,
                 width=WIDTH, in_ch=1, size=28, n_classes=10):
        super().__init__()
        self.k_sample = k_sample
        self.members = nn.ModuleList(
            [C.SmallCNN(in_ch, size, n_classes, width=width) for _ in range(n_members)])

    def forward(self, x):
        n = len(self.members)
        idx = torch.randperm(n, device=x.device)[:self.k_sample].tolist()
        outs = torch.stack([self.members[i](x) for i in idx], dim=0)
        return outs.mean(0)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _train(model, Xtr, Ytr, epochs=EPOCHS, lr=LR, with_noise_sigma=None):
    """Generic SGD trainer matching campaign defaults."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            if with_noise_sigma is not None:
                xb = (xb + with_noise_sigma * torch.randn_like(xb)).clamp(0, 1)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        sched.step()
    # IMPORTANT: keep model.train()=False for BN behaviour, but the stochastic
    # parts inside the modules (always-on dropout, sd-Bernoulli, Gabor noise,
    # Gumbel gate, ensemble subset) remain active by construction.
    model.eval()
    return model


def train_baseline_det(Xtr, Ytr):
    C.set_seed(SEED)
    m = C.build_model("cnn", META, width=WIDTH).to(C.DEVICE)
    return _train(m, Xtr, Ytr)


def train_noise_aug(Xtr, Ytr):
    C.set_seed(SEED)
    base = C.build_model("cnn", META, width=WIDTH).to(C.DEVICE)
    _train(base, Xtr, Ytr, with_noise_sigma=NOISE_SIGMA)
    return NoiseAugWrap(base, sigma=NOISE_SIGMA).to(C.DEVICE).eval()


def train_mc_dropout(Xtr, Ytr):
    C.set_seed(SEED)
    m = MCDropoutCNN(p=DROPOUT_P).to(C.DEVICE)
    return _train(m, Xtr, Ytr)


def train_stochastic_depth(Xtr, Ytr):
    C.set_seed(SEED)
    m = StochasticDepthCNN(p_keep=SD_P_KEEP).to(C.DEVICE)
    return _train(m, Xtr, Ytr)


def train_vone_noise(Xtr, Ytr):
    C.set_seed(SEED)
    m = VOneNoiseCNN(sigma=VONE_SIGMA).to(C.DEVICE)
    return _train(m, Xtr, Ytr)


def train_sparse_moe(Xtr, Ytr):
    C.set_seed(SEED)
    m = SparseMoEStochastic(n_experts=4, tau=MOE_GUMBEL_TAU).to(C.DEVICE)
    return _train(m, Xtr, Ytr)


def train_random_ensemble(Xtr, Ytr):
    C.set_seed(SEED)
    m = RandomSubsetEnsemble(n_members=N_ENS_MEMBERS, k_sample=N_ENS_SAMPLE).to(C.DEVICE)
    # train each member with independent batch perms -- simplest: train shared loss
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    m.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            # train each member separately (deterministic, all members)
            loss = sum(F.cross_entropy(member(xb), yb) for member in m.members) / N_ENS_MEMBERS
            loss.backward()
            opt.step()
        sched.step()
    m.eval()
    return m


# ---------------------------------------------------------------------------
# EOT-PGD attack
# ---------------------------------------------------------------------------
def eot_pgd(model, x, y, K, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """EOT-PGD: at each step, average the loss gradient over K fresh forward
    passes (each resampling whatever internal randomness the model has).
    K=1 reduces to a single-draw PGD attack against the stochastic defence."""
    x0 = x.clone().detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        grad_acc = torch.zeros_like(xa)
        for _k in range(K):
            xa_k = xa.detach().requires_grad_(True)
            loss = F.cross_entropy(model(xa_k), y)
            g, = torch.autograd.grad(loss, xa_k, retain_graph=False)
            grad_acc = grad_acc + g.detach()
        g_mean = grad_acc / K
        xa = xa.detach() + alpha * g_mean.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


@torch.no_grad()
def _eval_correct(model, X, Y, batch=256, reps=EVAL_REPS):
    """Per-sample mean-correctness over `reps` stochastic forward passes."""
    correct = torch.zeros(X.size(0))
    for _ in range(reps):
        parts = []
        for i in range(0, X.size(0), batch):
            parts.append((model(X[i:i + batch]).argmax(1) == Y[i:i + batch]).float().cpu())
        correct += torch.cat(parts)
    return correct / reps


def asr_eot(model, X, Y, K, batch=128):
    """Craft EOT-PGD adversarials at K samples; report ASR over originally-
    correct samples (majority over EVAL_REPS reps)."""
    model.eval()
    # clean correctness (majority)
    clean = _eval_correct(model, X, Y, batch=batch, reps=EVAL_REPS).numpy()
    corr = clean > 0.5
    # craft
    advs = []
    for i in range(0, X.size(0), batch):
        advs.append(eot_pgd(model, X[i:i + batch], Y[i:i + batch], K=K))
    Xadv = torch.cat(advs, dim=0)
    flip_mean = 1.0 - _eval_correct(model, Xadv, Y, batch=batch, reps=EVAL_REPS).numpy()
    asr = float(flip_mean[corr].mean()) if corr.sum() > 0 else float("nan")
    return asr, int(corr.sum())


def clean_acc_mean(model, X, Y):
    return float(_eval_correct(model, X, Y, reps=EVAL_REPS).mean())


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def verdict_for(delta40):
    if delta40 <= 0.03:
        return "GENUINE"
    if delta40 <= 0.10:
        return "SUSPECT"
    return "MASKED"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log("=" * 88)
    log("H465  EOT-PGD audit on every stochastic defence (Fashion-MNIST)")
    log("=" * 88)
    log(f"  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}  EPOCHS={EPOCHS}  LR={LR}  BATCH={BATCH}")
    log(f"  SGD(mom=0.9, wd=5e-4)  SEED={SEED}  EPS={EPS}  PGD_STEPS={PGD_STEPS}")
    log(f"  alpha={PGD_ALPHA}  K_LIST={K_LIST}  EVAL_REPS={EVAL_REPS}")
    log(f"  device={C.DEVICE}")
    log(f"  anchor: Athalye-Carlini-Wagner 2018 'Obfuscated Gradients';")
    log(f"          Gao 2022 'Limitations of Stochastic Pre-processing Defenses';")
    log(f"          Carlini 2019 'On Evaluating Adversarial Robustness'.")
    log("=" * 88)

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    log(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    log("")

    defences = [
        ("baseline_det",      train_baseline_det,      "deterministic SmallCNN (control)"),
        ("noise_aug",         train_noise_aug,         f"input Gaussian sigma={NOISE_SIGMA}"),
        ("mc_dropout",        train_mc_dropout,        f"always-on dropout p={DROPOUT_P}"),
        ("stochastic_depth",  train_stochastic_depth,  f"Bernoulli block-skip p_keep={SD_P_KEEP}"),
        ("vone_noise",        train_vone_noise,        f"VOneNet Gabor + noise sigma={VONE_SIGMA}"),
        ("sparse_moe_stoch",  train_sparse_moe,        f"sparse MoE k=1 + Gumbel tau={MOE_GUMBEL_TAU}"),
        ("ensemble_random",   train_random_ensemble,   f"random {N_ENS_SAMPLE}-of-{N_ENS_MEMBERS} subset ensemble"),
    ]

    rows = {}
    for name, trainer, desc in defences:
        log("-" * 88)
        log(f"[{name}] {desc}")
        ts = time.time()
        C.set_seed(SEED)
        model = trainer(Xtr, Ytr)
        cacc = clean_acc_mean(model, Xte, Yte)
        log(f"  trained in {time.time()-ts:.1f}s  clean_acc_mean={cacc:.4f}")
        flush()

        asr_by_K = {}
        ncorr = None
        for K in K_LIST:
            tsK = time.time()
            C.set_seed(SEED + 100 + K)   # fix attack RNG per K for reproducibility
            a, nC = asr_eot(model, Xte, Yte, K=K)
            asr_by_K[K] = a
            ncorr = nC
            log(f"  EOT-PGD  K={K:<3d}  ASR={a:.4f}   (n_corr={nC}, {time.time()-tsK:.1f}s)")
            flush()

        d_1_10 = asr_by_K[10] - asr_by_K[1]
        d_1_40 = asr_by_K[40] - asr_by_K[1]
        d_10_40 = asr_by_K[40] - asr_by_K[10]
        v = verdict_for(d_1_40)
        rows[name] = dict(desc=desc, cacc=cacc, ncorr=ncorr, asr=asr_by_K,
                          d_1_10=d_1_10, d_1_40=d_1_40, d_10_40=d_10_40,
                          verdict=v)
        log(f"  delta(K=1 -> K=10)  = {d_1_10:+.4f}")
        log(f"  delta(K=1 -> K=40)  = {d_1_40:+.4f}")
        log(f"  delta(K=10 -> K=40) = {d_10_40:+.4f}    --> VERDICT: {v}")
        # free GPU memory between defences
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        flush()

    # ----- summary table ----------------------------------------------------
    log("")
    log("=" * 88)
    log("SUMMARY: ASR(K) curve per defence  (eps=0.1, PGD-10, alpha=0.025)")
    log("=" * 88)
    hdr = "{:<18} {:>8} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8} {:>10}".format(
        "defence", "clean", "K=1", "K=10", "K=20", "K=40", "d1->10", "d1->40", "verdict")
    log(hdr)
    log("-" * len(hdr))
    for name, _, _ in defences:
        r = rows[name]
        log("{:<18} {:>8.4f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>+8.3f} {:>+8.3f} {:>10}".format(
            name, r["cacc"], r["asr"][1], r["asr"][10], r["asr"][20], r["asr"][40],
            r["d_1_10"], r["d_1_40"], r["verdict"]))
    log("-" * len(hdr))

    # ----- interpretive notes ----------------------------------------------
    log("")
    log("INTERPRETATION")
    log("-" * 88)
    log("  GENUINE  : ASR plateaus once K is moderate (K=10..40 within 0.03 of K=1).")
    log("             Stochasticity is NOT obfuscating gradient; defence is real.")
    log("  SUSPECT  : Moderate climb (0.03..0.10). Mild gradient masking; further")
    log("             diagnostics (transfer, BPDA, more K) warranted.")
    log("  MASKED   : Steep climb (>0.10). Athalye-2018 obfuscated-gradient pattern;")
    log("             reported robustness was a single-draw evaluation artefact.")
    log("")
    log("  Control: baseline_det should be GENUINE by construction (no stochasticity")
    log("  inside the model). If it isn't, the eval pipeline is biased.")
    log("")

    # explicit list per verdict
    by_verdict = {"GENUINE": [], "SUSPECT": [], "MASKED": []}
    for name in [n for n, _, _ in defences]:
        by_verdict[rows[name]["verdict"]].append(name)
    for v in ("GENUINE", "SUSPECT", "MASKED"):
        log(f"  {v:<8s}: {', '.join(by_verdict[v]) if by_verdict[v] else '(none)'}")
    log("")
    log(f"total runtime {time.time()-t0:.1f}s")
    log(f"saved -> {OUT}")
    flush()


if __name__ == "__main__":
    main()
