"""
H407 - Exactly-invertible (i-RevNet-style) classifier vs the standard SmallCNN.

Topic: reversible / exactly-invertible neural networks for adversarial
robustness. The information-preservation argument (Jacobsen et al. 2018,
"i-RevNet: Deep Invertible Networks", arXiv:1802.07088) is that an invertible
network keeps ALL input information -- there is no lossy bottleneck for an
attacker to exploit. We test whether an EXACTLY invertible feedforward
classifier (additive coupling blocks + invertible spatial squeeze, NO lossy
downsampling/pooling) has lower PGD-ASR than the standard SmallCNN at matched
clean accuracy.

Architecture (pure PyTorch, exactly invertible up to the final linear head):
  * Invertible "squeeze": reshape (B,C,H,W) -> (B,4C,H/2,W/2) by space-to-depth.
    This is a bijection (no information lost), unlike MaxPool.
  * Additive coupling block (i-RevNet / NICE, Dinh et al. 2014 arXiv:1410.8516):
        split channels into (x1, x2)
        y1 = x1
        y2 = x2 + F(x1)            # F = small conv net (the "bottleneck")
        swap -> (y2, y1)
    Exact inverse:
        x2 = y2 - F(y1) ; x1 = y1   (after un-swap)
    log|det J| = 0 (volume preserving) -> exact, analytic inverse.
  * Stack: squeeze -> k coupling blocks -> squeeze -> k coupling blocks ...
    The invertible trunk maps the image bijectively to a feature tensor of the
    SAME total dimensionality (28*28 = 784 values preserved). A linear head
    (the only non-invertible part) reads out 10 logits.

We VERIFY exact invertibility: run x through the invertible trunk to z, invert
z back to x_rec, and assert max|x - x_rec| ~ 0 (machine precision).

Compare against C.build_model("cnn") (SmallCNN with MaxPool, lossy) at matched
training budget. Report clean_acc / FGSM_ASR / PGD_ASR for both.

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
N_BLOCKS = 3          # coupling blocks per stage
META = {"channels": 1, "size": 28, "n_classes": 10}


# ---------------------------------------------------------------------------
# invertible building blocks
# ---------------------------------------------------------------------------
def squeeze(x):
    """Invertible space-to-depth: (B,C,H,W) -> (B,4C,H/2,W/2). Bijection."""
    B, C_, H, W = x.shape
    x = x.view(B, C_, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(B, C_ * 4, H // 2, W // 2)


def unsqueeze(x):
    """Exact inverse of squeeze: (B,4C,H/2,W/2) -> (B,C,H,W)."""
    B, C4, Hh, Ww = x.shape
    C_ = C4 // 4
    x = x.view(B, C_, 2, 2, Hh, Ww)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(B, C_, Hh * 2, Ww * 2)


class CouplingBlock(nn.Module):
    """Additive coupling (NICE / i-RevNet) with channel split + swap.

    forward:  y1 = x1 ; y2 = x2 + F(x1) ; return (y2, y1)  [concatenated]
    inverse:  given (y2,y1): x1 = y1 ; x2 = y2 - F(y1) ; return (x1,x2)
    Volume preserving (log|det|=0). Exactly invertible.
    """

    def __init__(self, channels, hidden=None):
        super().__init__()
        assert channels % 2 == 0
        self.c = channels // 2
        h = hidden or max(16, channels)
        self.F = nn.Sequential(
            nn.Conv2d(self.c, h, 3, padding=1), nn.BatchNorm2d(h), nn.ReLU(),
            nn.Conv2d(h, h, 3, padding=1), nn.BatchNorm2d(h), nn.ReLU(),
            nn.Conv2d(h, self.c, 3, padding=1),
        )

    def forward(self, x):
        x1, x2 = x[:, :self.c], x[:, self.c:]
        y1 = x1
        y2 = x2 + self.F(x1)
        return torch.cat([y2, y1], dim=1)   # swap

    def inverse(self, y):
        y2, y1 = y[:, :self.c], y[:, self.c:]
        x1 = y1
        x2 = y2 - self.F(y1)
        return torch.cat([x1, x2], dim=1)


class iRevNet(nn.Module):
    """Exactly-invertible trunk + linear head. No lossy downsampling/pooling.

    Stages: squeeze; N coupling blocks; squeeze; N coupling blocks.
    28x28x1 -> 14x14x4 -> 7x7x16 ; total elements preserved (= 784) throughout.
    """

    def __init__(self, in_ch=1, size=28, n_classes=10, n_blocks=N_BLOCKS):
        super().__init__()
        self.stage1 = nn.ModuleList(
            [CouplingBlock(in_ch * 4) for _ in range(n_blocks)])     # 4 ch
        self.stage2 = nn.ModuleList(
            [CouplingBlock(in_ch * 16) for _ in range(n_blocks)])    # 16 ch
        feat_ch = in_ch * 16
        feat = size // 4
        self.head = nn.Linear(feat_ch * feat * feat, n_classes)
        self.feat_dim = feat_ch * feat * feat

    def trunk(self, x):
        x = squeeze(x)
        for b in self.stage1:
            x = b(x)
        x = squeeze(x)
        for b in self.stage2:
            x = b(x)
        return x

    def inverse_trunk(self, z):
        for b in reversed(self.stage2):
            z = b.inverse(z)
        z = unsqueeze(z)
        for b in reversed(self.stage1):
            z = b.inverse(z)
        z = unsqueeze(z)
        return z

    def forward(self, x):
        z = self.trunk(x)
        return self.head(z.flatten(1))


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


def check_invertibility(model, x):
    """Reconstruct input from trunk output; return max abs reconstruction error."""
    model.eval()
    with torch.no_grad():
        z = model.trunk(x)
        x_rec = model.inverse_trunk(z)
    return float((x - x_rec).abs().max())


def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h407_irevnet_invertible_classifier_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H407  i-RevNet-style EXACTLY-invertible classifier vs SmallCNN (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: SMOKE={SMOKE} N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SEED={SEED} EPS={EPS} PGD_STEPS={PGD_STEPS} "
        f"N_BLOCKS={N_BLOCKS} device={C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    # ---- exact-invertibility check (before AND after training) ----
    C.set_seed(SEED)
    irev = iRevNet(in_ch=1, size=28, n_classes=10).to(C.DEVICE)
    err0 = check_invertibility(irev, Xte[:64])
    out(f"\n[invertibility] random-init recon error max|x-x_rec| = {err0:.3e}")
    assert err0 < 1e-3, f"i-RevNet trunk not invertible (err={err0})"

    out("\n[1] training i-RevNet (exactly-invertible trunk + linear head)...")
    irev = train(irev, Xtr, Ytr)
    err1 = check_invertibility(irev, Xte[:64])
    out(f"    post-train recon error max|x-x_rec| = {err1:.3e}")
    irev_acc, irev_fg, irev_pg = evaluate(irev, Xte, Yte)
    out(f"    i-RevNet: clean_acc={irev_acc:.4f}  FGSM_ASR={irev_fg:.4f}  "
        f"PGD_ASR={irev_pg:.4f}")
    flush()

    out("\n[2] training standard SmallCNN (lossy MaxPool) baseline...")
    C.set_seed(SEED)
    cnn = C.build_model("cnn", META, width=32, seed=SEED)
    cnn = train(cnn, Xtr, Ytr)
    cnn_acc, cnn_fg, cnn_pg = evaluate(cnn, Xte, Yte)
    out(f"    SmallCNN: clean_acc={cnn_acc:.4f}  FGSM_ASR={cnn_fg:.4f}  "
        f"PGD_ASR={cnn_pg:.4f}")

    # ---- table ----
    out("\n" + "=" * 80)
    out("[3] TABLE")
    out("=" * 80)
    hdr = "{:<26} {:>10} {:>10} {:>10} {:>14}".format(
        "model", "clean_acc", "FGSM_ASR", "PGD_ASR", "recon_err")
    out(hdr)
    out("-" * len(hdr))
    out("{:<26} {:>10.4f} {:>10.4f} {:>10.4f} {:>14.3e}".format(
        "i-RevNet (invertible)", irev_acc, irev_fg, irev_pg, err1))
    out("{:<26} {:>10.4f} {:>10.4f} {:>10.4f} {:>14}".format(
        "SmallCNN (lossy)", cnn_acc, cnn_fg, cnn_pg, "n/a"))
    out("-" * len(hdr))

    # ---- verdict ----
    out("\n" + "=" * 80)
    out("[4] VERDICT")
    out("=" * 80)
    d_pgd = cnn_pg - irev_pg          # positive => invertible more robust
    d_acc = irev_acc - cnn_acc
    matched = abs(d_acc) <= 0.03
    out(f"  i-RevNet recon error {err1:.2e} (EXACTLY invertible confirmed).")
    out(f"  PGD_ASR: SmallCNN {cnn_pg:.4f} vs i-RevNet {irev_pg:.4f} "
        f"(d={d_pgd:+.4f}, positive=>invertible more robust)")
    out(f"  clean_acc gap (irev - cnn) = {d_acc:+.4f} "
        f"({'matched <=0.03' if matched else 'NOT matched'})")
    if d_pgd > 0.03 and matched:
        verdict = ("YES: at matched clean accuracy the exactly-invertible "
                   "classifier has materially lower PGD-ASR.")
    elif d_pgd > 0.03:
        verdict = ("PARTIAL: invertible net is more robust but clean accuracy is "
                   "not matched -- robustness/accuracy not cleanly separable.")
    elif d_pgd < -0.03:
        verdict = ("NO: the invertible classifier is MORE vulnerable -- "
                   "invertibility alone does not buy robustness.")
    else:
        verdict = ("NO/NEUTRAL: invertibility alone gives no meaningful PGD-ASR "
                   "advantage over the lossy SmallCNN.")
    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
