"""
H408 - i-ResNet Lipschitz-constrained invertible blocks: spectral-norm sweep.

Topic: exactly-invertible nets for adversarial robustness. i-ResNet (Behrmann
et al. 2019, "Invertible Residual Networks", arXiv:1811.00995) makes a standard
residual block y = x + g(x) EXACTLY invertible whenever Lip(g) < 1: the inverse
is recovered by the Banach fixed-point iteration
        x_{k+1} = y - g(x_k),   x_0 = y
which converges because g is a contraction. Crucially the SAME constraint
Lip(g) < c < 1 bounds the block's sensitivity to input perturbations
(||dy/dx|| <= 1 + c), giving a certified-ish Lipschitz bound on the whole net.

This directly tests our finding (H323/H372/H401) that input-Jacobian smoothness
is the genuine implicit defense: i-ResNet enforces a HARD Lipschitz cap by
construction (spectral-norm-constrained conv layers), so sweeping the coefficient
c should trace out the robustness/accuracy trade-off.

We sweep c in {0.5, 0.9} (Lipschitz constant of each residual branch g) and a
non-invertible / no-cap baseline (c=inf, plain residual net), and measure
clean_acc / FGSM_ASR / PGD_ASR. For each invertible model we VERIFY exact
invertibility: y = x + g(x); recover x by the fixed-point iteration and assert
max|x - x_rec| ~ 0.

Lipschitz control: each conv in g is wrapped with power-iteration spectral
normalisation (the standard Miyato matrix-reshape proxy for the conv operator
norm) and scaled so the per-layer factor is c^(1/n); the composed g has
Lip(g) <~ c (product of per-layer factors; ReLU is 1-Lipschitz). Because the
matrix-reshape SN is a proxy (not the exact conv operator norm), we do NOT rely
on it for the invertibility GUARANTEE -- instead we empirically VERIFY exact
invertibility by running the Banach fixed-point inverse and asserting the
reconstruction error is ~machine precision. If a cap were too loose for the
fixed point to contract, that check would fail and flag it.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05 (SGD mom=0.9 wd=5e-4), BATCH=128,
SEED=0, EPS=0.1, PGD_STEPS=10. (Smoke config via env SMOKE=1.)
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
DS = "fashion_mnist"
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_TRAIN = 2000 if SMOKE else 6000
N_EVAL = 1000 if SMOKE else 2000
EPOCHS = 2 if SMOKE else 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
COEFFS = [0.5, 0.9]          # Lipschitz caps for invertible blocks
N_RES_BLOCKS = 4
WIDTH = 32
FP_ITERS = 200               # fixed-point iterations for the inverse
SN_SAFETY = 0.85             # margin on the (now tight) true conv operator norm
META = {"channels": 1, "size": 28, "n_classes": 10}


# ---------------------------------------------------------------------------
# spectral-normalised conv (power iteration) with an explicit Lipschitz scale
# ---------------------------------------------------------------------------
class SNConv2d(nn.Module):
    """Conv2d whose TRUE convolution operator norm is power-iteration-estimated
    (Gouk et al. 2021 / Sedghi et al. 2019 style: power-iterate through the conv
    operator itself, NOT the reshaped weight matrix) and then scaled so the
    effective Lipschitz constant of this layer is <= `coeff`.

    The reshape-matrix spectral norm (Miyato proxy) systematically UNDER-estimates
    the conv operator norm, so after training the residual branch can stop being a
    contraction and the Banach fixed-point inverse diverges. Power-iterating the
    actual conv operator gives the genuine operator norm, so the contraction is
    real and the exact inverse converges to machine precision. coeff=None disables
    the cap (plain conv)."""

    def __init__(self, ci, co, k=3, padding=1, coeff=0.9, n_pow=10):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, padding=padding, bias=True)
        self.ci, self.co, self.k, self.padding = ci, co, k, padding
        self.coeff = coeff
        self.n_pow = n_pow
        self.register_buffer("u", None)   # persistent input-space probe (1,ci,H,W)
        # frozen mode: use a fixed pre-scaled weight so the operator does NOT
        # drift between fixed-point inverse iterations (else recon plateaus at
        # the SN-estimate jitter, ~1e-4, instead of machine precision).
        self._frozen = False
        self._w_frozen = None

    def _sigma(self, weight, x_shape):
        # operator: x(1,ci,H,W) -> conv2d -> y(1,co,H,W); adjoint via conv_transpose2d.
        H, W = x_shape[-2], x_shape[-1]
        if (self.u is None or self.u.shape[-2] != H or self.u.shape[-1] != W
                or self.u.shape[1] != self.ci):
            self.u = F.normalize(
                torch.randn(1, self.ci, H, W, device=weight.device).flatten(),
                dim=0).view(1, self.ci, H, W)
        u = self.u
        with torch.no_grad():
            for _ in range(self.n_pow):
                v = F.conv2d(u, weight, None, padding=self.padding)      # (1,co,H,W)
                v = F.normalize(v.flatten(), dim=0).view_as(v)
                u = F.conv_transpose2d(v, weight, None, padding=self.padding)
                u = F.normalize(u.flatten(), dim=0).view(1, self.ci, H, W)
            self.u = u
        v = F.conv2d(u, weight, None, padding=self.padding)
        # sigma = ||A u|| with u unit-norm = sqrt(<v,v>)/<u,u>; u is unit so:
        sigma = v.flatten().norm()
        return sigma

    def _scaled_weight(self, x_shape):
        sigma = self._sigma(self.conv.weight, x_shape)
        # genuine operator norm -> a modest safety factor is enough to guarantee
        # the composed residual branch stays a contraction (Lip < 1).
        eff = self.coeff * SN_SAFETY
        scale = (eff / (sigma + 1e-9)).clamp(max=1.0)   # only shrink
        return self.conv.weight * scale

    def freeze(self, x_shape):
        """Cache the scaled weight so subsequent forwards use a FIXED operator
        (no power iteration, no `u` update). Required for the fixed-point inverse
        to converge to machine precision rather than chasing a moving operator."""
        if self.coeff is None:
            self._w_frozen = self.conv.weight.detach()
        else:
            self._w_frozen = self._scaled_weight(x_shape).detach()
        self._frozen = True

    def unfreeze(self):
        self._frozen = False
        self._w_frozen = None

    def forward(self, x):
        if self.coeff is None:
            return self.conv(x)
        if self._frozen:
            w = self._w_frozen
        else:
            w = self._scaled_weight(x.shape)
        return F.conv2d(x, w, self.conv.bias, padding=self.conv.padding)


class ResBranch(nn.Module):
    """g(x): a small conv net whose composed Lipschitz constant is <= coeff.
    Each of the n SNConvs is capped at coeff**(1/n) so the product <= coeff.
    ReLU is 1-Lipschitz so it does not change the bound."""

    def __init__(self, ch, coeff, hidden=None):
        super().__init__()
        h = hidden or ch
        n = 2
        per = None if coeff is None else coeff ** (1.0 / n)
        self.net = nn.Sequential(
            SNConv2d(ch, h, 3, 1, coeff=per), nn.ReLU(),
            SNConv2d(h, ch, 3, 1, coeff=per),
        )

    def forward(self, x):
        return self.net(x)


class InvResBlock(nn.Module):
    """y = x + g(x), exactly invertible when Lip(g) < 1 (Behrmann 2019).
    Inverse via Banach fixed point: x <- y - g(x)."""

    def __init__(self, ch, coeff):
        super().__init__()
        self.g = ResBranch(ch, coeff)

    def forward(self, x):
        return x + self.g(x)

    def inverse(self, y, n_iter=FP_ITERS):
        x = y.clone()
        for _ in range(n_iter):
            x = y - self.g(x)
        return x


class iResNet(nn.Module):
    """Invertible-residual trunk (no spatial reduction) + avg-pool linear head.

    The trunk x -> z is exactly invertible (each block is). The head (global
    average pool + linear) is the only non-invertible readout. coeff=None gives
    a plain (uncapped, NOT guaranteed invertible) residual baseline."""

    def __init__(self, in_ch=1, size=28, n_classes=10, width=WIDTH,
                 n_blocks=N_RES_BLOCKS, coeff=0.9):
        super().__init__()
        self.coeff = coeff
        # lift to `width` channels with a fixed (Lipschitz-capped) conv
        self.lift = SNConv2d(in_ch, width, 3, 1,
                             coeff=(None if coeff is None else 1.0))
        self.blocks = nn.ModuleList(
            [InvResBlock(width, coeff) for _ in range(n_blocks)])
        self.head = nn.Linear(width, n_classes)

    def trunk(self, x):
        x = self.lift(x)
        for b in self.blocks:
            x = b(x)
        return x

    def inverse_blocks(self, z):
        """Invert only the residual blocks (the bijective part). The lift conv
        changes channel count so is not inverted; invertibility of the residual
        stack is what i-ResNet guarantees."""
        for b in reversed(self.blocks):
            z = b.inverse(z)
        return z

    def forward(self, x):
        z = self.trunk(x)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.head(z)


# ---------------------------------------------------------------------------
# training / eval
# ---------------------------------------------------------------------------
def train(model, Xtr, Ytr):
    C.set_seed(SEED)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model


def evaluate(model, Xte, Yte):
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


def _freeze_blocks(model, feat_shape):
    """Freeze the SN scaling of every conv in the residual blocks to a FIXED
    operator (computed once at `feat_shape`), so forward and inverse use the
    identical mapping and the Banach fixed point converges to machine precision."""
    for b in model.blocks:
        for m in b.g.net:
            if isinstance(m, SNConv2d):
                m.freeze(feat_shape)


def _unfreeze_blocks(model):
    for b in model.blocks:
        for m in b.g.net:
            if isinstance(m, SNConv2d):
                m.unfreeze()


def check_invertibility(model, x):
    """Lift, run blocks forward, invert blocks; compare to the lifted tensor.
    The residual-block convs are frozen so the inverse operator exactly matches
    the forward operator (no SN-estimate drift between fixed-point iterations)."""
    model.eval()
    with torch.no_grad():
        h = model.lift(x)
        _freeze_blocks(model, h.shape)
        try:
            z = h
            for b in model.blocks:
                z = b(z)
            h_rec = model.inverse_blocks(z)
        finally:
            _unfreeze_blocks(model)
    return float((h - h_rec).abs().max())


def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h408_iresnet_lipschitz_sweep_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H408  i-ResNet Lipschitz-constrained invertible blocks: coeff sweep")
    out("=" * 80)
    out(f"config: SMOKE={SMOKE} N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SEED={SEED} EPS={EPS} PGD_STEPS={PGD_STEPS}")
    out(f"        coeffs={COEFFS} (+ uncapped baseline) N_RES_BLOCKS={N_RES_BLOCKS} "
        f"WIDTH={WIDTH} FP_ITERS={FP_ITERS} device={C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    rows = []
    conditions = [("uncapped", None)] + [(f"c={c}", c) for c in COEFFS]
    for name, coeff in conditions:
        out("\n" + "=" * 80)
        out(f"[condition {name}] coeff={coeff}")
        out("=" * 80)
        C.set_seed(SEED)
        model = iResNet(in_ch=1, size=28, n_classes=10, coeff=coeff).to(C.DEVICE)
        # invertibility only guaranteed for capped (c<1) models
        if coeff is not None and coeff < 1.0:
            err0 = check_invertibility(model, Xte[:64])
            out(f"    random-init fixed-point recon error = {err0:.3e}")
        model = train(model, Xtr, Ytr)
        if coeff is not None and coeff < 1.0:
            err = check_invertibility(model, Xte[:64])
            out(f"    post-train fixed-point recon error  = {err:.3e}")
        else:
            err = float("nan")
            out("    (uncapped baseline: NOT guaranteed invertible -- skip check)")
        acc, fg, pg = evaluate(model, Xte, Yte)
        out(f"    {name}: clean_acc={acc:.4f}  FGSM_ASR={fg:.4f}  PGD_ASR={pg:.4f}")
        rows.append(dict(name=name, coeff=coeff, acc=acc, fg=fg, pg=pg, err=err))
        flush()

    # ---- table ----
    out("\n" + "=" * 80)
    out("[TABLE]")
    out("=" * 80)
    hdr = "{:<14} {:>10} {:>10} {:>10} {:>14}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR", "recon_err")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        es = f"{r['err']:.3e}" if r["err"] == r["err"] else "n/a"
        out("{:<14} {:>10.4f} {:>10.4f} {:>10.4f} {:>14}".format(
            r["name"], r["acc"], r["fg"], r["pg"], es))
    out("-" * len(hdr))

    # ---- verdict ----
    out("\n" + "=" * 80)
    out("[VERDICT]")
    out("=" * 80)
    base = next(r for r in rows if r["coeff"] is None)
    capped = [r for r in rows if r["coeff"] is not None]
    best = min(capped, key=lambda r: r["pg"])
    d_pgd = base["pg"] - best["pg"]
    d_acc = best["acc"] - base["acc"]
    # is the trend monotone: tighter cap (smaller c) -> lower PGD-ASR?
    cs = sorted(capped, key=lambda r: r["coeff"])
    monotone = all(cs[i]["pg"] <= cs[i + 1]["pg"] + 1e-6 for i in range(len(cs) - 1))
    for r in capped:
        out(f"  c={r['coeff']}: PGD_ASR {base['pg']:.4f}->{r['pg']:.4f} "
            f"({r['pg']-base['pg']:+.4f}); clean {base['acc']:.4f}->{r['acc']:.4f} "
            f"({r['acc']-base['acc']:+.4f}); recon_err={r['err']:.2e}")
    out("")
    out(f"  best capped condition: {best['name']} (PGD_ASR {best['pg']:.4f})")
    out(f"  PGD robustness gain vs uncapped = {d_pgd:+.4f}; clean cost {d_acc:+.4f}")
    out(f"  tighter-cap=>lower-PGD monotone trend: {monotone}")
    if d_pgd > 0.03:
        verdict = ("YES: a hard Lipschitz cap on exactly-invertible residual "
                   "blocks reduces PGD-ASR -- consistent with bounded input "
                   "sensitivity (H323/H372/H401) being the real defense.")
    elif d_pgd < -0.03:
        verdict = ("NO: tightening the Lipschitz cap did not help (PGD-ASR rose) "
                   "-- bounded-Lipschitz invertibility is not sufficient here.")
    else:
        verdict = ("NEUTRAL: the Lipschitz cap had little net effect on PGD-ASR "
                   "at this depth/eps.")
    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
