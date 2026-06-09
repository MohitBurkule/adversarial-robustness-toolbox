"""
H452 - AT with diffusion-generated data (gap G10).

Anchor: wang-2023-betterdm "Better Diffusion Models Further Improve Adversarial
Training" (ICML 2023). On CIFAR-10 / CIFAR-100, Wang et al. show that adding ~1M
samples from a state-of-the-art DDPM/EDM teacher boosts AutoAttack robust accuracy
by roughly +5pp over the previous SoTA. The classical motivation is the
sample-complexity gap for robust learning identified by Schmidt et al. 2018
("Adversarially Robust Generalization Requires More Data", NeurIPS 2018), and
the line of work showing extra (unlabelled or generated) samples help:
  - Carmon et al. 2019 "Unlabeled Data Improves Adversarial Robustness" (NeurIPS
    2019) - +5pp on CIFAR-10 with 500k unlabelled tiny-images via robust self-
    training.
  - Rebuffi/Gowal et al. 2021 "Fixing Data Augmentation to Improve Adversarial
    Robustness" (NeurIPS 2021) - +7pp robust acc using a generative model to
    inflate the training set.

OPEN QUESTION FOR THIS HYPOTHESIS
Wang/Carmon/Gowal all operate in the regime where the GENERATOR is trained on a
*larger* dataset than the AT classifier (50k CIFAR-10 + a strong off-the-shelf
DDPM, or 50k labelled + 500k unlabelled). The interesting sub-scale regime is
when real data is itself the bottleneck and the diffusion teacher must be trained
on the SAME small set the classifier sees. Does a tiny DDPM trained on the 6k
real Fashion-MNIST subset distil any extra robustness-relevant signal, or does
it merely memorise / re-emit those 6k points? If the latter, synthetic data is
free augmentation at best and a distractor at worst; if the former, the
diffusion prior is doing real work.

DESIGN
  Stage 1: train a tiny DDPM (~1-2M params) on the 6k real training images
    (10 classes, 28x28x1). Class-conditional, simple cosine noise schedule,
    fixed 100-step training schedule, 50-step DDPM sampling at inference.
  Stage 2: sample 24k labelled synthetic images from the DDPM (balanced across
    classes via conditional sampling).
  Stage 3: run PGD-AT on FIVE conditions (CNN width=32, EPS=0.1, PGD steps=10):
    C1. 6k REAL only         (canonical baseline)
    C2. 12k SYNTH only       (no real)
    C3. 6k REAL + 6k SYNTH   (1:1 mix)
    C4. 6k REAL + 12k SYNTH  (1:2 mix; Wang's regime in spirit)
    C5. 6k REAL + 24k SYNTH  (1:4 mix; aggressive)
  Stage 4: report clean acc + PGD ASR for each condition on the same 2000-sample
    real test set; also report the DDPM nearest-neighbour leakage stats
    (mean L2 from each synthetic sample to its closest training image) so the
    reader can tell whether synthetic samples are essentially copies.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9,wd=5e-4),
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01. DDPM uses Adam(lr=2e-4) for
8000 steps over the 6k real set.

VERDICT criteria
  YES: at least one mixed condition improves PGD_ASR by >= 0.02 vs C1 without
       dropping clean acc by more than 0.02.
  PARTIAL: PGD_ASR improves by >= 0.02 but clean acc drops by > 0.02, OR
           clean acc improves while PGD_ASR is within +/- 0.02 of baseline.
  NO: no mixed condition beats C1 by >= 0.02 on PGD_ASR.
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

META = {"channels": 1, "size": 28, "n_classes": 10}

# DDPM hyperparams (kept tiny)
DDPM_T = 100             # diffusion timesteps used for training schedule
DDPM_SAMPLE_T = 50       # sampling steps (sub-sampled, DDIM-style with eta=0 -> DDPM-equivalent on this schedule)
DDPM_STEPS = 8000        # gradient steps on the 6k real set
DDPM_BATCH = 128
DDPM_LR = 2e-4
DDPM_CH = 48             # base channel width of the U-Net
N_SYNTH_MAX = 24000      # we sample once at 24k then take prefixes
SYNTH_SAMPLE_BATCH = 256
SYNTH_DROP_DUPE_L2 = 1e-3  # below this L2-from-closest-real we flag a synth as a copy


# -------------------------------------------------------------------------
# Tiny class-conditional U-Net (~1-2M params for 28x28x1 single-channel input).
# -------------------------------------------------------------------------
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class CondBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=DDPM_CH, n_classes=10, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.label_emb = nn.Embedding(n_classes + 1, emb_dim)  # +1 = "unconditional" token (unused but reserved)
        # 28 -> 14 -> 7 (then back)
        self.d1 = CondBlock(in_ch, base_ch, emb_dim)
        self.d2 = CondBlock(base_ch, base_ch * 2, emb_dim)
        self.mid = CondBlock(base_ch * 2, base_ch * 2, emb_dim)
        self.u2 = CondBlock(base_ch * 4, base_ch, emb_dim)
        self.u1 = CondBlock(base_ch * 2, base_ch, emb_dim)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t, y):
        # x: (B,1,28,28) padded to 32 for clean /2 /2
        emb = timestep_embedding(t, self.emb_dim) + self.label_emb(y)
        emb = self.time_mlp(emb)
        x32 = F.pad(x, (2, 2, 2, 2))  # 28 -> 32
        h1 = self.d1(x32, emb)         # 32
        h2 = self.d2(self.down(h1), emb)  # 16
        m = self.mid(self.down(h2), emb)  # 8
        u2 = self.u2(torch.cat([self.up(m), h2], dim=1), emb)  # 16
        u1 = self.u1(torch.cat([self.up(u2), h1], dim=1), emb)  # 32
        out32 = self.out_conv(u1)
        return out32[:, :, 2:-2, 2:-2]  # back to 28


# -------------------------------------------------------------------------
# DDPM: linear beta schedule, x in [-1,1] during training.
# -------------------------------------------------------------------------
def make_schedule(T, device):
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar}


def ddpm_train(Xtr_01, Ytr, steps=DDPM_STEPS, batch=DDPM_BATCH, lr=DDPM_LR, seed=SEED, log_every=1000):
    """Train tiny U-Net to predict epsilon. Xtr_01 in [0,1] -> internal [-1,1]."""
    device = Xtr_01.device
    C.set_seed(seed)
    net = TinyUNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = make_schedule(DDPM_T, device)
    alpha_bar = sched["alpha_bar"]
    Xtr = Xtr_01 * 2.0 - 1.0  # [-1,1]
    n = Xtr.size(0)
    nparams = sum(p.numel() for p in net.parameters())
    net.train()
    losses = []
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        x0 = Xtr[idx]
        y = Ytr[idx]
        t = torch.randint(0, DDPM_T, (batch,), device=device)
        ab = alpha_bar[t].view(-1, 1, 1, 1)
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        pred = net(xt, t, y)
        loss = F.mse_loss(pred, eps)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            print(f"    [ddpm] step {step+1}/{steps}  loss={np.mean(losses[-200:]):.4f}", flush=True)
    net.eval()
    return net, sched, nparams


@torch.no_grad()
def ddpm_sample(net, sched, n_per_class, ncls=10, batch=SYNTH_SAMPLE_BATCH,
                T_sample=DDPM_SAMPLE_T, device=None, seed=SEED + 7):
    """Class-conditional sampling. Uses sub-sampled schedule (T_sample timesteps
    drawn evenly from the trained T schedule), reverse-time DDPM step.
    Returns X in [0,1] and labels Y."""
    device = device or next(net.parameters()).device
    alpha_bar = sched["alpha_bar"]
    # sub-sample T_sample evenly-spaced timesteps from [0,DDPM_T-1]
    ts = torch.linspace(DDPM_T - 1, 0, T_sample, device=device).round().long()
    total = n_per_class * ncls
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xs = torch.empty(total, 1, 28, 28, device=device)
    Ys = torch.empty(total, dtype=torch.long, device=device)
    cursor = 0
    for cls in range(ncls):
        remaining = n_per_class
        while remaining > 0:
            b = min(batch, remaining)
            x = torch.randn(b, 1, 28, 28, generator=g).to(device)
            y = torch.full((b,), cls, dtype=torch.long, device=device)
            for i in range(T_sample):
                t_i = ts[i].expand(b)
                ab_t = alpha_bar[t_i].view(-1, 1, 1, 1)
                eps_pred = net(x, t_i, y)
                # predict x0
                x0_pred = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt().clamp(min=1e-8)
                x0_pred = x0_pred.clamp(-1, 1)
                if i < T_sample - 1:
                    t_next = ts[i + 1].expand(b)
                    ab_next = alpha_bar[t_next].view(-1, 1, 1, 1)
                    # DDIM eta=0 step (deterministic; equivalent to a DDPM step on the
                    # sub-schedule, fine for our scale)
                    x = ab_next.sqrt() * x0_pred + (1 - ab_next).sqrt() * eps_pred
                else:
                    x = x0_pred
            Xs[cursor:cursor + b] = ((x + 1.0) * 0.5).clamp(0, 1)
            Ys[cursor:cursor + b] = y
            cursor += b
            remaining -= b
    # shuffle so class order is not preserved
    perm = torch.randperm(total, device=device, generator=None)
    return Xs[perm], Ys[perm]


@torch.no_grad()
def nearest_real_l2(Xs, Xtr_real, batch=512):
    """For each Xs row return min L2 distance to any row of Xtr_real."""
    n = Xs.size(0)
    out = torch.empty(n, device=Xs.device)
    Xtr_flat = Xtr_real.flatten(1)
    for i in range(0, n, batch):
        xs = Xs[i:i + batch].flatten(1)
        # pairwise dists via expansion
        d = torch.cdist(xs, Xtr_flat, p=2.0)
        out[i:i + batch] = d.min(dim=1).values
    return out


# -------------------------------------------------------------------------
# PGD-AT training (mirrors common.train_model adv_train=True with our config).
# -------------------------------------------------------------------------
def _make_optimizer_sgd(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def train_pgd_at(Xtr, Ytr, seed):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_adv = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_clean_pgd(model, X, Y):
    _, acc = C.logits_and_acc(model, X, Y)
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, pg["asr"]


# -------------------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h452_diffusion_generated_at_data_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H452  AT with diffusion-generated data (Fashion-MNIST, sub-scale)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        DDPM: T={DDPM_T} sample_T={DDPM_SAMPLE_T} steps={DDPM_STEPS} "
        f"batch={DDPM_BATCH} lr={DDPM_LR} base_ch={DDPM_CH}")
    out(f"        synth pool size = {N_SYNTH_MAX}")
    out(f"        device = {C.DEVICE}")
    out("")

    # ---- real data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: real Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    flush_file()

    # ---- Stage 1: train tiny DDPM on the 6k real set ----
    out("\n[1] training tiny class-conditional DDPM on 6k real images...")
    ddpm_net, ddpm_sched, ddpm_nparams = ddpm_train(Xtr, Ytr, seed=SEED)
    out(f"    DDPM #params = {ddpm_nparams/1e6:.2f}M  ({ddpm_nparams})")
    out(f"    DDPM train wall = {time.time()-t0:.0f}s")
    flush_file()

    # ---- Stage 2: sample 24k synthetic (2400 per class) ----
    out("\n[2] sampling 24k class-balanced synthetic images from DDPM...")
    Xs_all, Ys_all = ddpm_sample(ddpm_net, ddpm_sched, n_per_class=N_SYNTH_MAX // 10,
                                 ncls=10, T_sample=DDPM_SAMPLE_T, seed=SEED + 7)
    out(f"    synth pool: X={tuple(Xs_all.shape)}  Y dist={torch.bincount(Ys_all).tolist()}")

    # leakage / collapse check: nearest-neighbour L2 to real train
    nn_d = nearest_real_l2(Xs_all, Xtr).cpu().numpy()
    out(f"    synth->real nearest-L2: min={nn_d.min():.4f}  med={np.median(nn_d):.4f}  "
        f"max={nn_d.max():.4f}  mean={nn_d.mean():.4f}")
    n_dupes = int((nn_d < SYNTH_DROP_DUPE_L2).sum())
    out(f"    synth essentially copying a real image (L2<{SYNTH_DROP_DUPE_L2}): {n_dupes}/{len(nn_d)}")
    # pixel stats sanity
    out(f"    synth pixel stats: mean={Xs_all.mean().item():.3f}  std={Xs_all.std().item():.3f}  "
        f"(real: mean={Xtr.mean().item():.3f} std={Xtr.std().item():.3f})")
    flush_file()

    # ---- Stage 3: PGD-AT on five conditions ----
    out("\n[3] PGD-AT training under five data conditions...")
    rows = []

    def run(cond_name, X, Y):
        out(f"\n  -> condition: {cond_name}   (n={X.size(0)})")
        m = train_pgd_at(X, Y, SEED)
        acc, pgd_asr = eval_clean_pgd(m, Xte, Yte)
        out(f"     clean_acc={acc:.4f}  PGD_ASR={pgd_asr:.4f}   "
            f"({time.time()-t0:.0f}s elapsed)")
        rows.append({"cond": cond_name, "n": X.size(0), "acc": acc, "pgd": pgd_asr})
        flush_file()

    # C1: 6k real only
    run("C1: 6k REAL only", Xtr, Ytr)
    # C2: 12k synth only
    X_s12, Y_s12 = Xs_all[:12000], Ys_all[:12000]
    run("C2: 12k SYNTH only", X_s12, Y_s12)
    # C3: 6k real + 6k synth
    X_s6, Y_s6 = Xs_all[:6000], Ys_all[:6000]
    run("C3: 6k REAL + 6k SYNTH", torch.cat([Xtr, X_s6]), torch.cat([Ytr, Y_s6]))
    # C4: 6k real + 12k synth
    run("C4: 6k REAL + 12k SYNTH", torch.cat([Xtr, X_s12]), torch.cat([Ytr, Y_s12]))
    # C5: 6k real + 24k synth
    run("C5: 6k REAL + 24k SYNTH", torch.cat([Xtr, Xs_all]), torch.cat([Ytr, Ys_all]))

    # ---- Stage 4: main table ----
    out("\n" + "=" * 80)
    out("[4] MAIN TABLE")
    out("=" * 80)
    hdr = "{:<28} {:>8} {:>10} {:>10}".format("condition", "n_train", "clean_acc", "PGD_ASR")
    out(hdr); out("-" * len(hdr))
    base = rows[0]
    for r in rows:
        d_acc = r["acc"] - base["acc"]
        d_pgd = r["pgd"] - base["pgd"]
        marker = ""
        if r is not base:
            marker = "  (d_acc={:+.3f}, d_PGD={:+.3f})".format(d_acc, d_pgd)
        out("{:<28} {:>8} {:>10.4f} {:>10.4f}{}".format(
            r["cond"], r["n"], r["acc"], r["pgd"], marker))
    out("-" * len(hdr))

    # ---- VERDICT ----
    out("\n" + "=" * 80)
    out("[5] VERDICT")
    out("=" * 80)
    mixed = [r for r in rows if r["cond"].startswith(("C3", "C4", "C5"))]
    best = min(mixed, key=lambda r: r["pgd"])
    gain_pgd = base["pgd"] - best["pgd"]
    d_acc_best = best["acc"] - base["acc"]
    out(f"  baseline (C1): clean_acc={base['acc']:.4f}  PGD_ASR={base['pgd']:.4f}")
    out(f"  best mixed cond = {best['cond']}: clean_acc={best['acc']:.4f}  "
        f"PGD_ASR={best['pgd']:.4f}")
    out(f"  PGD_ASR robustness gain vs baseline = {gain_pgd:+.4f}  "
        f"(positive => more robust)")
    out(f"  clean_acc change vs baseline        = {d_acc_best:+.4f}")
    out(f"  DDPM #params = {ddpm_nparams/1e6:.2f}M  "
        f"nearest-real L2 med = {np.median(nn_d):.4f}  "
        f"n_dupes(L2<{SYNTH_DROP_DUPE_L2}) = {n_dupes}")

    if gain_pgd >= 0.02 and d_acc_best >= -0.02:
        one = ("YES: diffusion-generated data trained on the same 6k subset adds robustness "
               "at sub-scale without breaking clean accuracy - the diffusion prior carries "
               "robustness-relevant signal beyond memorisation.")
    elif gain_pgd >= 0.02 and d_acc_best < -0.02:
        one = ("PARTIAL: synthetic data improves PGD robustness but clean acc drops >0.02 - "
               "the prior helps the inner-max but distorts the data distribution.")
    elif gain_pgd < 0.02 and d_acc_best >= 0.02:
        one = ("PARTIAL: clean acc improves but PGD robustness is unchanged - synth acts as "
               "regularisation, not a robustness source.")
    else:
        one = ("NO: at sub-scale (6k real, tiny DDPM) synthetic data does not transfer "
               "robustness; the generator cannot exceed the information in its training set.")
    out("  ONE-LINE VERDICT: " + one)

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
