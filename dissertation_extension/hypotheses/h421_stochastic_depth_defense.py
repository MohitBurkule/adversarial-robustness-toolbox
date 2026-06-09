"""
H421 - Stochastic Depth as an adversarial defense.

Hypothesis: Random layer-skipping during inference (Huang et al. 2016 "Deep
Networks with Stochastic Depth") creates an implicit ensemble-of-paths effect
that makes gradient computation harder for the attacker, thereby improving
robustness — especially under Expectation over Transformations PGD (EOT-PGD).

Reference: Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016).
"Deep Networks with Stochastic Depth." ECCV 2016. arXiv:1603.09382.

Mechanism tested
  - SD trains each residual block with survival probability p_l (linearly
    decayed from 1 at the first block to p_final at the last).
  - At test-time inference, blocks may also be randomly skipped (non-standard
    but the ensemble hypothesis demands it).
  - Under BPDA / EOT-PGD the attacker either uses the *expected* (deterministic)
    forward pass as surrogate (BPDA) or averages gradients across K random
    forward passes (EOT). We measure robustness under both.

Conditions (4):
  A. baseline        — plain ResNet (no stochastic depth), eval deterministic
  B. SD_train_only   — SD during training, deterministic eval (p_l=1 at test)
  C. SD_train+infer  — SD during both training and evaluation
  D. SD_infer_only   — deterministic training, SD only at evaluation

Attacks:
  1. PGD (standard, single forward pass)
  2. EOT-PGD (K=8 forward passes per step, averages gradients) — only
     meaningful for conditions C and D where inference is stochastic.

Config: Fashion-MNIST, N_TRAIN=6000, N_EVAL=2000, EPOCHS=12, LR=0.05,
BATCH=128, SGD(mom=0.9,wd=1e-4), SEED=0, EPS=0.1, PGD_STEPS=10,
EOT_K=8, p_final=0.5 (survival probability of deepest block).
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

# ---- config -----------------------------------------------------------------
DS          = "fashion_mnist"
N_TRAIN     = 6000
N_EVAL      = 2000
EPOCHS      = 12
LR          = 0.05
BATCH       = 128
SEED        = 0
EPS         = 0.1
PGD_STEPS   = 10
PGD_ALPHA   = 2.5 * EPS / PGD_STEPS
EOT_K       = 8        # forward passes per PGD step for EOT
P_FINAL     = 0.5      # survival probability at the deepest block
N_BLOCKS    = 6        # total residual blocks (3 stages x 2)

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h421_stochastic_depth_defense_output.txt",
)


# ---- model ------------------------------------------------------------------

class ResBlock(nn.Module):
    """Basic residual block with stochastic depth support."""

    def __init__(self, channels, survival_prob: float = 1.0):
        super().__init__()
        self.survival_prob = survival_prob
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x, use_sd: bool = False):
        """use_sd: apply stochastic depth (random skip) if True."""
        if use_sd and self.training:
            # drop entire block with probability (1 - survival_prob)
            if torch.rand(1).item() > self.survival_prob:
                return x
        elif use_sd and not self.training:
            # inference-time stochastic skip (ensemble-of-paths effect)
            if torch.rand(1).item() > self.survival_prob:
                return x
        # residual branch
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class SDResNet(nn.Module):
    """Small ResNet with stochastic depth for Fashion-MNIST (1x28x28 -> 10)."""

    def __init__(self, p_final: float = 0.5, n_blocks: int = 6, width: int = 32):
        super().__init__()
        # survival probs: linearly decay from 1 (block 0) to p_final (block n-1)
        self.survival_probs = [
            1.0 - (1.0 - p_final) * i / max(n_blocks - 1, 1)
            for i in range(n_blocks)
        ]
        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        # stage 1: 28x28, width channels, 2 blocks
        self.stage1 = nn.ModuleList([
            ResBlock(width, self.survival_probs[i]) for i in range(2)
        ])
        # downsample 28->14
        self.down1 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(),
        )
        # stage 2: 14x14, width*2 channels, 2 blocks
        self.stage2 = nn.ModuleList([
            ResBlock(width * 2, self.survival_probs[2 + i]) for i in range(2)
        ])
        # downsample 14->7
        self.down2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4), nn.ReLU(),
        )
        # stage 3: 7x7, width*4 channels, 2 blocks
        self.stage3 = nn.ModuleList([
            ResBlock(width * 4, self.survival_probs[4 + i]) for i in range(2)
        ])
        # head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, 10),
        )
        # SD mode flags (set externally)
        self.sd_train = False
        self.sd_infer = False

    def forward(self, x):
        use_sd = self.sd_train if self.training else self.sd_infer
        h = self.stem(x)
        for blk in self.stage1:
            h = blk(h, use_sd)
        h = self.down1(h)
        for blk in self.stage2:
            h = blk(h, use_sd)
        h = self.down2(h)
        for blk in self.stage3:
            h = blk(h, use_sd)
        return self.head(h)


# ---- training ---------------------------------------------------------------

def train_model(model: SDResNet, Xtr, Ytr):
    opt   = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
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


# ---- attacks ----------------------------------------------------------------

def pgd_standard(model, x, y, eps, steps, alpha):
    """Standard PGD: single forward pass per step."""
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        xa = (xa.detach() + alpha * g.sign())
        xa = torch.max(torch.min(xa, x0 + eps), x0 - eps).clamp(0, 1)
    return xa.detach()


def pgd_eot(model, x, y, eps, steps, alpha, k_eot):
    """EOT-PGD: average gradients over k_eot stochastic forward passes per step."""
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        # accumulate gradient over k_eot random forward passes
        g_sum = torch.zeros_like(xa)
        for _ in range(k_eot):
            loss = F.cross_entropy(model(xa), y)
            g, = torch.autograd.grad(loss, xa, retain_graph=False)
            g_sum = g_sum + g.detach()
        g_avg = g_sum / k_eot
        xa = (xa.detach() + alpha * g_avg.sign())
        xa = torch.max(torch.min(xa, x0 + eps), x0 - eps).clamp(0, 1)
    return xa.detach()


@torch.no_grad()
def clean_acc(model, X, Y, batch=512):
    correct = 0
    for i in range(0, X.size(0), batch):
        correct += (model(X[i:i+batch]).argmax(1) == Y[i:i+batch]).sum().item()
    return correct / Y.size(0)


def asr_pgd(model, X, Y, batch=256, eot=False):
    """Attack-success rate (fraction of clean-correct samples flipped)."""
    model.eval()
    flips, corr = [], []
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i+batch], Y[i:i+batch]
        with torch.no_grad():
            ok = model(xb).argmax(1) == yb
        if eot:
            xa = pgd_eot(model, xb, yb, EPS, PGD_STEPS, PGD_ALPHA, EOT_K)
        else:
            xa = pgd_standard(model, xb, yb, EPS, PGD_STEPS, PGD_ALPHA)
        with torch.no_grad():
            flipped = model(xa).argmax(1) != yb
        flips.append(flipped.cpu()); corr.append(ok.cpu())
    flips = torch.cat(flips).numpy(); corr = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() > 0 else float("nan")


# ---- main -------------------------------------------------------------------

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
    out("H421  Stochastic Depth as adversarial defense (Fashion-MNIST)")
    out("=" * 80)
    out("Reference: Huang et al. (2016) 'Deep Networks with Stochastic Depth'")
    out("           ECCV 2016 / arXiv:1603.09382")
    out("")
    out("Hypothesis: random layer-skipping at inference creates an ensemble-of-paths")
    out("  effect that hardens gradients and provides EOT-resistant robustness.")
    out("")
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=1e-4)")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA:.4f} "
        f"EOT_K={EOT_K} P_FINAL={P_FINAL} N_BLOCKS={N_BLOCKS}")
    out(f"        SEED={SEED}  device={C.DEVICE}")
    out("")

    # ---- data ----------------------------------------------------------------
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    # ---- conditions ----------------------------------------------------------
    conditions = [
        # (label,        sd_train, sd_infer)
        ("A_baseline",        False, False),
        ("B_SD_train_only",   True,  False),
        ("C_SD_train+infer",  True,  True),
        ("D_SD_infer_only",   False, True),
    ]

    rows = []
    for cond_label, sd_train, sd_infer in conditions:
        out("-" * 70)
        out(f"Condition: {cond_label}  (sd_train={sd_train}, sd_infer={sd_infer})")

        C.set_seed(SEED)
        model = SDResNet(p_final=P_FINAL, n_blocks=N_BLOCKS, width=32).to(C.DEVICE)
        model.sd_train = sd_train
        model.sd_infer = sd_infer

        t1 = time.time()
        train_model(model, Xtr, Ytr)
        out(f"  training done in {time.time()-t1:.0f}s")

        # clean accuracy (deterministic for A/B; stochastic for C/D — average 8 passes)
        if sd_infer:
            accs = [clean_acc(model, Xte, Yte) for _ in range(8)]
            acc = float(np.mean(accs))
        else:
            acc = clean_acc(model, Xte, Yte)
        out(f"  clean acc = {acc:.4f}")

        # standard PGD
        asr_std = asr_pgd(model, Xte, Yte, eot=False)
        out(f"  PGD-ASR (standard) = {asr_std:.4f}")

        # EOT-PGD (meaningful for C/D; included for all to compare)
        asr_eot = asr_pgd(model, Xte, Yte, eot=True)
        out(f"  PGD-ASR (EOT K={EOT_K}) = {asr_eot:.4f}")

        rows.append({
            "cond": cond_label, "sd_train": sd_train, "sd_infer": sd_infer,
            "acc": acc, "asr_std": asr_std, "asr_eot": asr_eot,
        })
        flush()

    # ---- summary table -------------------------------------------------------
    out("")
    out("=" * 80)
    out("SUMMARY TABLE")
    out("=" * 80)
    hdr = "{:<22} {:>10} {:>12} {:>12} {:>12} {:>12}".format(
        "condition", "sd_train", "sd_infer", "clean_acc", "PGD_ASR", "EOT_PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<22} {:>10} {:>12} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            r["cond"], str(r["sd_train"]), str(r["sd_infer"]),
            r["acc"], r["asr_std"], r["asr_eot"]))
    out("-" * len(hdr))

    # ---- verdict -------------------------------------------------------------
    out("")
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)
    base  = rows[0]
    best_std = min(rows, key=lambda r: r["asr_std"])
    best_eot = min(rows, key=lambda r: r["asr_eot"])

    for r in rows[1:]:
        out(f"  {r['cond']}: PGD {base['asr_std']:.4f}->{r['asr_std']:.4f} "
            f"({r['asr_std']-base['asr_std']:+.4f}); "
            f"EOT-PGD {base['asr_eot']:.4f}->{r['asr_eot']:.4f} "
            f"({r['asr_eot']-base['asr_eot']:+.4f}); "
            f"acc {base['acc']:.4f}->{r['acc']:.4f} "
            f"({r['acc']-base['acc']:+.4f})")
    out("")

    # check if any SD condition beats baseline on EOT-PGD without large acc drop
    eot_gains = [(r["asr_eot"] - base["asr_eot"], r) for r in rows[1:]]
    best_eot_gain, best_eot_row = min(eot_gains, key=lambda x: x[0])
    acc_ok = best_eot_row["acc"] >= base["acc"] - 0.02

    sd_infer_rows = [r for r in rows if r["sd_infer"]]
    eot_vs_std = [(r["asr_eot"] - r["asr_std"]) for r in sd_infer_rows]

    out(f"  Best PGD-ASR (std):  {best_std['cond']} = {best_std['asr_std']:.4f}")
    out(f"  Best EOT-PGD-ASR:    {best_eot['cond']} = {best_eot['asr_eot']:.4f}")
    out(f"  EOT-PGD gain vs baseline (best): {best_eot_gain:+.4f} "
        f"({'IMPROVED' if best_eot_gain < -0.01 else 'NO IMPROVEMENT'})")
    out(f"  EOT harder than std PGD for SD-infer conditions: "
        f"{['%.4f' % d for d in eot_vs_std]} "
        f"({'YES – EOT closes gap' if any(d > 0.02 for d in eot_vs_std) else 'NO'})")
    out("")

    if best_eot_gain < -0.05 and acc_ok:
        verdict = ("YES: stochastic depth at inference provides meaningful EOT-resistant "
                   "robustness — the ensemble-of-paths effect hampers gradient-based attacks.")
    elif best_eot_gain < -0.02 and acc_ok:
        verdict = ("PARTIAL: stochastic depth at inference yields modest EOT-PGD robustness "
                   "improvement without major accuracy cost.")
    elif best_eot_gain < -0.02 and not acc_ok:
        verdict = ("PARTIAL-COSTLY: some EOT-PGD robustness gain but clean accuracy drops "
                   "> 0.02 — not a free defense.")
    elif any(d > 0.02 for d in eot_vs_std):
        verdict = ("NO-BUT-ADAPTIVE: stochastic depth does not reduce overall EOT-PGD ASR, "
                   "but EOT is observably harder than standard PGD for SD-infer models, "
                   "confirming the gradient-obfuscation hypothesis partially.")
    else:
        verdict = ("NO: stochastic depth (train and/or infer) provides no meaningful "
                   "robustness; EOT-PGD is as effective as standard PGD.")

    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time()-t0:.1f}s")
    flush()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
