"""
H413 - Perceptual (brain-like) threat-model adversarial training.

Concept
-------
Adversarial examples are defined relative to a THREAT MODEL. Conventional AT
uses an L-inf pixel ball, but that is NOT the threat model biological vision
is robust to. The brain's threat model is closer to "perceptually
imperceptible" perturbations (a perceptual / feature-space metric), not raw
pixels. This operationalizes Laidlaw et al. 2020 "Perceptual Adversarial
Robustness" (LPIPS-bounded AT, arXiv:2006.12655) in a pure-torch setting
(no external pretrained nets / no extra deps).

We build a PERCEPTUAL METRIC two pure-torch ways and use one to drive AT:
  (A) Random-feature LPIPS proxy: a FROZEN randomly-initialized small conv
      feature extractor; perceptual distance = normalized L2 between its
      (channel-unit-normalized) feature maps -> "self-supervised-free LPIPS".
  (B) Fixed Gabor / wavelet feature space: a frozen Gabor filter bank (like a
      VOneNet front-end); perceptual distance = L2 in Gabor-response space.
      A perturbation big in pixels but invisible in Gabor space is
      "perceptually small".
We report which metric is used to drive Perceptual-AT (default A).

Conditions (same backbone = SmallCNN width=32, same epochs/budget)
  1. Standard       : no AT.
  2. L-inf AT       : ordinary PGD-AT in pixel space (eps=0.1, conventional).
  3. Perceptual AT  : LPA-style PGD adversarial training where the inner-max
      perturbation is constrained in the PERCEPTUAL metric. We implement a
      "Fast-LPA"-style attack: PGD that maximizes CE loss and after each step
      PROJECTS the perturbation back so its perceptual distance <= bound
      (bound = perceptual distance of an L-inf eps ball, calibrated on a small
      batch). This keeps perturbations bounded in feature space, not pixels.

Cross-threat-model evaluation (the core measurement)
  Each of the 3 models is attacked by MULTIPLE threat models:
    - L-inf PGD  (eps=0.1)
    - L2   PGD   (eps=2.0)
    - Perceptual PGD (bounded in the perceptual metric)
    - Gaussian blur (natural-shift proxy; not an attack but a corruption)
  Report clean_acc + ASR for every (model x threat) cell. KEY question: does
  Perceptual-AT generalize across the ROW (robust to L-inf AND L2 AND
  perceptual) better than L-inf-AT, which is known to overfit its own norm?

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD mom=0.9 wd=5e-4,
SEED=0, eps=0.1, PGD_STEPS=10. ASR = 1 - adv_acc (lower=better).
Set SMOKE=1 for a fast smoke path.
"""
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
SMOKE = os.environ.get("SMOKE", "0") == "1"
DS = "fashion_mnist"
N_TRAIN = 1000 if SMOKE else 6000
N_EVAL = 500 if SMOKE else 2000
EPOCHS = 2 if SMOKE else 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1                  # L-inf budget (pixels)
L2_EPS = 2.0               # L2 budget for L2-PGD eval
PGD_STEPS = 4 if SMOKE else 10
PERC_METRIC = "random"     # "random" | "gabor" -> which metric DRIVES perc-AT
ATTACK_BATCH = 256

META = {"channels": 1, "size": 28, "n_classes": 10}


# ===========================================================================
# perceptual metrics (pure torch, frozen)
# ===========================================================================
class RandomFeatureLPIPS(nn.Module):
    """Frozen randomly-initialized conv feature extractor; perceptual distance
    = mean over layers of L2 between channel-unit-normalized feature maps
    (the LPIPS recipe with random instead of pretrained weights)."""
    def __init__(self, in_ch=1, width=32, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.c1 = nn.Conv2d(in_ch, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.2 if p.dim() > 1
                        else torch.zeros(p.shape))
            p.requires_grad_(False)

    def _norm(self, f):
        # unit-normalize across channels (per spatial location), like LPIPS
        return f / (f.norm(dim=1, keepdim=True) + 1e-8)

    def feats(self, x):
        f1 = F.relu(self.c1(x))
        f2 = F.relu(self.c2(f1))
        f3 = F.relu(self.c3(f2))
        return [self._norm(f1), self._norm(f2), self._norm(f3)]

    def dist(self, x, y):
        """Per-sample perceptual distance, shape (N,)."""
        d = 0.0
        for fx, fy in zip(self.feats(x), self.feats(y)):
            d = d + ((fx - fy) ** 2).flatten(1).mean(1)
        return d.sqrt()


class GaborLPIPS(nn.Module):
    """Frozen Gabor filter bank (VOneNet-style front-end). Perceptual distance
    = L2 in the Gabor-response space."""
    def __init__(self, in_ch=1, ksize=7, n_orient=8, sigmas=(1.5, 2.5)):
        super().__init__()
        filters = []
        for sigma in sigmas:
            lam = sigma * 2.0
            for o in range(n_orient):
                theta = np.pi * o / n_orient
                k = self._gabor(ksize, sigma, theta, lam)
                filters.append(k)
        W = torch.stack(filters).unsqueeze(1)        # (F,1,ksize,ksize)
        if in_ch > 1:
            W = W.repeat(1, in_ch, 1, 1) / in_ch
        self.register_buffer("weight", W)
        self.pad = ksize // 2

    @staticmethod
    def _gabor(ksize, sigma, theta, lam, gamma=0.5):
        half = ksize // 2
        y, x = torch.meshgrid(torch.arange(-half, half + 1),
                              torch.arange(-half, half + 1), indexing="ij")
        x = x.float(); y = y.float()
        xr = x * np.cos(theta) + y * np.sin(theta)
        yr = -x * np.sin(theta) + y * np.cos(theta)
        env = torch.exp(-(xr ** 2 + (gamma ** 2) * yr ** 2) / (2 * sigma ** 2))
        carrier = torch.cos(2 * np.pi * xr / lam)
        k = env * carrier
        k = k - k.mean()
        return k / (k.norm() + 1e-8)

    def feats(self, x):
        return F.relu(F.conv2d(x, self.weight, padding=self.pad))

    def dist(self, x, y):
        d = (self.feats(x) - self.feats(y)) ** 2
        return d.flatten(1).mean(1).sqrt()


def build_perc(metric):
    if metric == "random":
        return RandomFeatureLPIPS(in_ch=META["channels"], seed=SEED).to(C.DEVICE)
    if metric == "gabor":
        return GaborLPIPS(in_ch=META["channels"]).to(C.DEVICE)
    raise ValueError(metric)


# ===========================================================================
# attacks
# ===========================================================================
def pgd_l2(model, x, y, eps, steps, alpha=None, random_start=True):
    """L2-bounded PGD."""
    if alpha is None:
        alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        n = torch.randn_like(xa)
        n = n / (n.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
        xa = (xa + 0.5 * eps * n).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        gn = g / (g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
        xa = xa.detach() + alpha * gn
        delta = xa - x0
        dn = delta.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        factor = (eps / (dn + 1e-12)).clamp(max=1.0)
        xa = (x0 + delta * factor).clamp(0, 1)
    return xa.detach()


def pgd_perc(model, x, y, perc, bound, steps, alpha=None, random_start=True):
    """Perceptual PGD (LPA-style): maximize CE, then project the perturbation
    back so perc.dist(x_adv, x0) <= bound. Projection is a scalar shrink of the
    perturbation toward x0 (valid because perceptual dist ~ monotone in
    perturbation scale for a fixed direction)."""
    if alpha is None:
        alpha = 2.5 * EPS / steps        # step size in pixel space
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        xa = (xa + torch.empty_like(xa).uniform_(-EPS, EPS)).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        xa = (xa.detach() + alpha * g.sign()).clamp(0, 1)
        xa = _perc_project(xa, x0, perc, bound)
    return xa.detach()


@torch.no_grad()
def _perc_project(xa, x0, perc, bound):
    """Shrink (xa - x0) by a per-sample scalar so perceptual dist <= bound.
    Binary search on the scale factor s in [0,1] for samples that violate."""
    d = perc.dist(xa, x0)
    viol = d > bound
    if not viol.any():
        return xa
    delta = xa - x0
    lo = torch.zeros(xa.size(0), device=xa.device)
    hi = torch.ones(xa.size(0), device=xa.device)
    for _ in range(8):
        mid = (lo + hi) / 2
        cand = (x0 + delta * mid.view(-1, 1, 1, 1)).clamp(0, 1)
        dm = perc.dist(cand, x0)
        too_big = dm > bound
        hi = torch.where(too_big, mid, hi)
        lo = torch.where(too_big, lo, mid)
    s = lo.view(-1, 1, 1, 1)
    proj = (x0 + delta * s).clamp(0, 1)
    # only replace violators; keep compliant samples as-is
    return torch.where(viol.view(-1, 1, 1, 1), proj, xa)


@torch.no_grad()
def calibrate_perc_bound(perc, X, eps=EPS, n=256):
    """Bound = median perceptual distance produced by a random L-inf eps ball,
    so the perceptual threat budget is comparable to the pixel one."""
    Xs = X[:n]
    noise = torch.empty_like(Xs).uniform_(-eps, eps)
    Xp = (Xs + noise).clamp(0, 1)
    return float(perc.dist(Xp, Xs).median())


def gaussian_blur(x, ksize=5, sigma=1.0):
    half = ksize // 2
    ax = torch.arange(-half, half + 1, device=x.device).float()
    g1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    k2 = (g1[:, None] * g1[None, :]).view(1, 1, ksize, ksize)
    k2 = k2.repeat(x.size(1), 1, 1, 1)
    return F.conv2d(x, k2, padding=half, groups=x.size(1)).clamp(0, 1)


# ===========================================================================
# eval helpers
# ===========================================================================
@torch.no_grad()
def _acc_on(model, X, Y):
    _, acc = C.logits_and_acc(model, X, Y)
    return acc


def asr_under(model, X, Y, attack_fn):
    """ASR over originally-correct samples = fraction flipped."""
    model.eval()
    flips, corr = [], []
    for i in range(0, X.size(0), ATTACK_BATCH):
        x, y = X[i:i + ATTACK_BATCH], Y[i:i + ATTACK_BATCH]
        with torch.no_grad():
            c = model(x).argmax(1) == y
        xa = attack_fn(model, x, y)
        with torch.no_grad():
            f = model(xa).argmax(1) != y
        flips.append(f.cpu()); corr.append(c.cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() > 0 else float("nan")


# ===========================================================================
# training
# ===========================================================================
def _opt(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)


def train_model_at(mode, Xtr, Ytr, perc=None, perc_bound=None, seed=SEED):
    """mode in {'std','linf','perc'}."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            if mode == "linf":
                xb = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS)
            elif mode == "perc":
                xb = pgd_perc(model, xb, yb, perc, perc_bound, steps=PGD_STEPS)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ===========================================================================
# main
# ===========================================================================
def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h413_perceptual_threat_model_AT_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H413  Perceptual (brain-like) threat-model adversarial training (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: SMOKE={SMOKE} N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} "
        f"LR={LR} BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS(Linf)={EPS} L2_EPS={L2_EPS} PGD_STEPS={PGD_STEPS} "
        f"perc_metric_driving_AT={PERC_METRIC}")
    out(f"        ASR = 1 - adv_acc (lower = better);  device = {C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    # ---- perceptual metrics ----
    perc_random = build_perc("random")
    perc_gabor = build_perc("gabor")
    perc_drive = perc_random if PERC_METRIC == "random" else perc_gabor
    bound_random = calibrate_perc_bound(perc_random, Xtr)
    bound_gabor = calibrate_perc_bound(perc_gabor, Xtr)
    bound_drive = bound_random if PERC_METRIC == "random" else bound_gabor
    out(f"\nperceptual bounds (median dist of L-inf eps={EPS} ball):")
    out(f"    random-feature LPIPS bound = {bound_random:.4f}")
    out(f"    gabor-feature   LPIPS bound = {bound_gabor:.4f}")
    out(f"    -> Perceptual-AT driven by '{PERC_METRIC}', bound={bound_drive:.4f}")

    # ---- threat-model attack functions for eval ----
    def atk_linf(m, x, y):
        return C.pgd(m, x, y, eps=EPS, steps=PGD_STEPS)

    def atk_l2(m, x, y):
        return pgd_l2(m, x, y, eps=L2_EPS, steps=PGD_STEPS)

    def atk_perc_r(m, x, y):
        return pgd_perc(m, x, y, perc_random, bound_random, steps=PGD_STEPS)

    def atk_perc_g(m, x, y):
        return pgd_perc(m, x, y, perc_gabor, bound_gabor, steps=PGD_STEPS)

    def atk_blur(m, x, y):
        return gaussian_blur(x, ksize=5, sigma=1.0)

    threats = [
        ("Linf_PGD(eps=0.1)", atk_linf),
        ("L2_PGD(eps=2.0)", atk_l2),
        ("Perc_PGD(random)", atk_perc_r),
        ("Perc_PGD(gabor)", atk_perc_g),
        ("Gauss_blur", atk_blur),
    ]

    # ---- train the 3 models ----
    out("\n" + "=" * 80)
    out("[1] TRAINING 3 MODELS (same backbone, same budget)")
    out("=" * 80)
    models = {}
    for name, mode in [("Standard", "std"), ("Linf-AT", "linf"),
                       (f"Perc-AT({PERC_METRIC})", "perc")]:
        out(f"  training {name} ...")
        m = train_model_at(mode, Xtr, Ytr, perc=perc_drive,
                            perc_bound=bound_drive)
        models[name] = m
        out(f"    done ({time.time()-t0:.0f}s)  clean_acc={_acc_on(m, Xte, Yte):.4f}")
        flush_file()

    # ---- cross-threat-model matrix ----
    out("\n" + "=" * 80)
    out("[2] CROSS-THREAT-MODEL ASR MATRIX  (rows=model, cols=threat; ASR=1-adv_acc)")
    out("=" * 80)
    results = {}     # model -> {threat: asr}
    cleans = {}
    for name, m in models.items():
        cleans[name] = _acc_on(m, Xte, Yte)
        results[name] = {}
        for tname, tfn in threats:
            asr = asr_under(m, Xte, Yte, tfn)
            results[name][tname] = asr
            out(f"    {name:<16} {tname:<20} ASR={asr:.4f}  ({time.time()-t0:.0f}s)")
            flush_file()

    # ---- pretty matrix ----
    out("\n" + "=" * 80)
    out("[3] MATRIX")
    out("=" * 80)
    tnames = [t[0] for t in threats]
    hdr = "{:<18} {:>9}".format("model", "clean")
    for tn in tnames:
        hdr += " {:>18}".format(tn)
    out(hdr)
    out("-" * len(hdr))
    for name in models:
        row = "{:<18} {:>9.4f}".format(name, cleans[name])
        for tn in tnames:
            row += " {:>18.4f}".format(results[name][tn])
        out(row)
    out("-" * len(hdr))

    # ---- aggregate cross-threat robustness (mean ASR over the 3 PGD threats) ----
    pgd_threats = ["Linf_PGD(eps=0.1)", "L2_PGD(eps=2.0)",
                   "Perc_PGD(random)", "Perc_PGD(gabor)"]
    out("\nmean ASR over PGD threat models (lower = broader robustness):")
    mean_asr = {}
    for name in models:
        vals = [results[name][t] for t in pgd_threats]
        mean_asr[name] = float(np.mean(vals))
        worst = max(pgd_threats, key=lambda t: results[name][t])
        out(f"    {name:<18} mean_PGD_ASR={mean_asr[name]:.4f}  "
            f"worst={worst}({results[name][worst]:.4f})  clean={cleans[name]:.4f}")

    # ---- verdict ----
    out("\n" + "=" * 80)
    out("[4] VERDICT")
    out("=" * 80)
    std_n = "Standard"
    linf_n = "Linf-AT"
    perc_n = f"Perc-AT({PERC_METRIC})"

    out(f"  clean acc: Standard={cleans[std_n]:.4f}  Linf-AT={cleans[linf_n]:.4f}  "
        f"{perc_n}={cleans[perc_n]:.4f}")
    out(f"  mean PGD ASR: Standard={mean_asr[std_n]:.4f}  "
        f"Linf-AT={mean_asr[linf_n]:.4f}  {perc_n}={mean_asr[perc_n]:.4f}")

    # does perc-AT generalize across threats better than linf-AT?
    perc_better_mean = mean_asr[perc_n] < mean_asr[linf_n] - 0.01
    # check the L2 and perceptual columns specifically (where Linf-AT overfits)
    l2_better = results[perc_n]["L2_PGD(eps=2.0)"] < results[linf_n]["L2_PGD(eps=2.0)"] - 0.01
    pr_better = results[perc_n]["Perc_PGD(random)"] < results[linf_n]["Perc_PGD(random)"] - 0.01
    acc_cost = cleans[perc_n] - cleans[linf_n]

    out(f"\n  Perc-AT vs Linf-AT  ->  mean_PGD_ASR {mean_asr[linf_n]:.4f}->"
        f"{mean_asr[perc_n]:.4f} ({mean_asr[perc_n]-mean_asr[linf_n]:+.4f})")
    out(f"  L2 column:   Linf-AT={results[linf_n]['L2_PGD(eps=2.0)']:.4f}  "
        f"{perc_n}={results[perc_n]['L2_PGD(eps=2.0)']:.4f}")
    out(f"  Perc column: Linf-AT={results[linf_n]['Perc_PGD(random)']:.4f}  "
        f"{perc_n}={results[perc_n]['Perc_PGD(random)']:.4f}")
    out(f"  clean-acc difference (Perc-AT - Linf-AT) = {acc_cost:+.4f}")

    if perc_better_mean and (l2_better or pr_better):
        one = ("YES: a perceptual (brain-like) threat model gives BROADER "
               "cross-threat-model robustness than pixel-L-inf AT (lower mean "
               f"PGD ASR), at clean-acc cost {acc_cost:+.4f} vs Linf-AT.")
    elif mean_asr[perc_n] <= mean_asr[linf_n] + 0.01:
        one = ("MIXED: Perceptual-AT is roughly comparable to L-inf AT on mean "
               "cross-threat ASR; no clear broad-robustness win here.")
    else:
        one = ("NO: in this setting L-inf AT generalizes across threat models "
               "at least as well as Perceptual-AT; the perceptual threat model "
               "did not buy broader robustness.")
    out("\n  ONE-LINE VERDICT: " + one)

    out("")
    out(f"done in {time.time()-t0:.1f}s")
    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
