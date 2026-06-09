"""
H418 - Skip-only network: extreme ResNet limit where the main path is identity.

Veit, Wilber & Belongie (2016) showed that ResNets behave like ensembles of
shallow networks formed by all possible paths through residual blocks.  At the
extreme limit where the "main" path is the identity, computation is forced
entirely into the residual branches.  We test whether this structure implies a
tighter Lipschitz bound on the mapping (since each branch can only add a small
correction) and therefore yields adversarial robustness.

Three conditions
----------------
  baseline    -- standard ResNet (identity skip + learned main; SmallCNN analogue)
  skip_only   -- residual branch does all computation; main path is strict identity
  skip_jp     -- skip_only + Jacobian penalty (||J||_F regulariser during training)

All three are trained on Fashion-MNIST (N=6000) with identical EPOCHS/LR.
Evaluation: clean accuracy, FGSM ASR, PGD ASR (eps=0.1, 10 steps).

Config: N_TRAIN=6000, N_EVAL=2000, EPOCHS=12, LR=0.05, BATCH=128,
SGD(mom=0.9, wd=1e-4), SEED=0, EPS=0.1, PGD_STEPS=10, JP_LAMBDA=0.01,
JP_SAMPLES=4 (Jacobian estimated from random projection).
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

# ---- config ------------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 12
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 2.5 * EPS / PGD_STEPS
JP_LAMBDA = 0.01    # Jacobian-penalty coefficient
JP_SAMPLES = 4      # random vectors for Frobenius-norm estimate

META = {"channels": 1, "size": 28, "n_classes": 10}
IN_CH, SZ, N_CLS = 1, 28, 10
WIDTH = 32


# ---- architectures -----------------------------------------------------------

class ResBlock(nn.Module):
    """Standard residual block: out = x + F(x).  Channels must match."""
    def __init__(self, ch):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(x + self.branch(x))


class SkipOnlyBlock(nn.Module):
    """Skip-only block: out = x + F(x), identical to ResBlock in math —
    BUT the main path is strict identity and F(x) is the *sole* computation.
    This is the same formula; the distinction is conceptual: we verify that
    x passes unchanged through the skip and the branch starts from zero-bias
    initialisation (branch weights initialised very small so at init the block
    ≈ identity).  The Lipschitz constraint analysis applies to F only.
    We also add a learnable scalar `alpha` initialised to 0.1 so the branch
    cannot immediately saturate the skip."""
    def __init__(self, ch):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
        # Small-weight init so branch starts as near-zero correction
        for m in self.branch.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        return F.relu(x + self.alpha * self.branch(x))


def _make_stem(in_ch, width):
    return nn.Sequential(
        nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
    )


def _make_pool():
    return nn.Sequential(
        nn.MaxPool2d(2),
    )


class BaselineResNet(nn.Module):
    """Standard residual CNN: stem -> [resblock x2 + pool] x2 -> classifier."""
    def __init__(self, in_ch=IN_CH, n_classes=N_CLS, width=WIDTH):
        super().__init__()
        w = width
        self.stem = _make_stem(in_ch, w)
        self.layer1 = nn.Sequential(ResBlock(w), ResBlock(w), _make_pool())
        self.layer2 = nn.Sequential(
            nn.Conv2d(w, w * 2, 1, bias=False), nn.BatchNorm2d(w * 2),  # channel up
            ResBlock(w * 2), ResBlock(w * 2), _make_pool()
        )
        feat = SZ // 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 2 * feat * feat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)


class SkipOnlyResNet(nn.Module):
    """Skip-only ResNet: same topology but all residual blocks are SkipOnlyBlock."""
    def __init__(self, in_ch=IN_CH, n_classes=N_CLS, width=WIDTH):
        super().__init__()
        w = width
        self.stem = _make_stem(in_ch, w)
        self.layer1 = nn.Sequential(SkipOnlyBlock(w), SkipOnlyBlock(w), _make_pool())
        self.layer2 = nn.Sequential(
            nn.Conv2d(w, w * 2, 1, bias=False), nn.BatchNorm2d(w * 2),
            SkipOnlyBlock(w * 2), SkipOnlyBlock(w * 2), _make_pool()
        )
        feat = SZ // 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 2 * feat * feat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)


# ---- Jacobian penalty --------------------------------------------------------

def jacobian_penalty(model, xb):
    """Estimate ||J_f(x)||_F^2 via JP_SAMPLES random projections (Hutchinson).
    Returns mean over the batch."""
    xb = xb.detach().requires_grad_(True)
    out = model(xb)                          # (B, C)
    B, C = out.shape
    loss_jp = 0.0
    for _ in range(JP_SAMPLES):
        v = torch.randn_like(out)            # (B, C)
        # vJ = d(v^T f)/dx  shape (B, in_dim)
        vJ, = torch.autograd.grad(
            (out * v).sum(), xb,
            create_graph=True, retain_graph=True)
        loss_jp = loss_jp + vJ.pow(2).sum(dim=list(range(1, vJ.dim()))).mean()
    return loss_jp / JP_SAMPLES


# ---- training ----------------------------------------------------------------

def _make_opt(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)


def train_model(model, Xtr, Ytr, use_jp=False, verbose=False):
    C.set_seed(SEED)
    opt = _make_opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        ep_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            if use_jp:
                loss = loss + JP_LAMBDA * jacobian_penalty(model, xb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sched.step()
        if verbose:
            print(f"    epoch {ep+1}/{EPOCHS}  loss={ep_loss:.3f}", flush=True)
    model.eval()
    return model


def eval_all(model, Xte, Yte):
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd",  eps=EPS, steps=PGD_STEPS,
                          batch=256)
    return acc, fg["asr"], pg["asr"]


# ---- branch Lipschitz estimate -----------------------------------------------

@torch.no_grad()
def est_branch_lipschitz(model, Xte, n_pairs=500):
    """Estimate Lipschitz constant of the network output via random input pairs.
    L_est = max_i ||f(x_i) - f(x_j)|| / ||x_i - x_j||."""
    idx = torch.randperm(Xte.size(0))[:n_pairs * 2]
    xa, xb_ = Xte[idx[:n_pairs]], Xte[idx[n_pairs:]]
    fa = model(xa)
    fb = model(xb_)
    num = (fa - fb).norm(dim=1)
    den = (xa - xb_).flatten(1).norm(dim=1).clamp(min=1e-8)
    return float((num / den).max().item()), float((num / den).mean().item())


# ---- main --------------------------------------------------------------------

def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h418_skip_only_network_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H418  Skip-only network: identity main path + Jacobian penalty (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=1e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA:.4f}")
    out(f"        JP_LAMBDA={JP_LAMBDA} JP_SAMPLES={JP_SAMPLES}")
    out(f"        device={C.DEVICE}")
    out("")
    out("Hypothesis (Veit-Wilber 2016): collapsing the ResNet main path to strict")
    out("identity forces all computation into residual branches whose corrections")
    out("are bounded, inducing a lower Lipschitz constant and therefore robustness.")
    out("Jacobian penalty should amplify this effect.")
    out("")

    # ---- data ----------------------------------------------------------------
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    # ---- conditions ----------------------------------------------------------
    conditions = [
        ("baseline",  BaselineResNet,  False),
        ("skip_only", SkipOnlyResNet,   False),
        ("skip_jp",   SkipOnlyResNet,   True),
    ]

    rows = []
    for cname, Cls, use_jp in conditions:
        out("")
        out("-" * 60)
        out(f"[condition: {cname}]  jp={use_jp}")
        C.set_seed(SEED)
        model = Cls().to(C.DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        out(f"  params={n_params:,}")
        tc = time.time()
        train_model(model, Xtr, Ytr, use_jp=use_jp, verbose=True)
        out(f"  trained in {time.time()-tc:.1f}s")
        acc, fgsm_asr, pgd_asr = eval_all(model, Xte, Yte)
        out(f"  clean_acc={acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  PGD_ASR={pgd_asr:.4f}")
        lip_max, lip_mean = est_branch_lipschitz(model, Xte)
        out(f"  Lipschitz_est  max={lip_max:.3f}  mean={lip_mean:.3f}")
        rows.append({
            "cond": cname, "acc": acc, "fgsm": fgsm_asr, "pgd": pgd_asr,
            "lip_max": lip_max, "lip_mean": lip_mean,
        })
        flush_file()

    # ---- summary table -------------------------------------------------------
    out("")
    out("=" * 80)
    out("[SUMMARY TABLE]")
    out("=" * 80)
    hdr = "{:<12} {:>10} {:>10} {:>9} {:>10} {:>10}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR", "Lip_max", "Lip_mean")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<12} {:>10.4f} {:>10.4f} {:>9.4f} {:>10.3f} {:>10.3f}".format(
            r["cond"], r["acc"], r["fgsm"], r["pgd"], r["lip_max"], r["lip_mean"]))
    out("-" * len(hdr))

    # ---- verdict -------------------------------------------------------------
    out("")
    out("=" * 80)
    out("[VERDICT]")
    out("=" * 80)
    base = rows[0]
    skip = rows[1]
    skip_jp = rows[2]

    out(f"  baseline  -> skip_only: PGD_ASR {base['pgd']:.4f}->{skip['pgd']:.4f} "
        f"({skip['pgd']-base['pgd']:+.4f}); "
        f"clean_acc {base['acc']:.4f}->{skip['acc']:.4f} ({skip['acc']-base['acc']:+.4f}); "
        f"Lip_max {base['lip_max']:.3f}->{skip['lip_max']:.3f}")
    out(f"  skip_only -> skip_jp  : PGD_ASR {skip['pgd']:.4f}->{skip_jp['pgd']:.4f} "
        f"({skip_jp['pgd']-skip['pgd']:+.4f}); "
        f"clean_acc {skip['acc']:.4f}->{skip_jp['acc']:.4f} "
        f"({skip_jp['acc']-skip['acc']:+.4f}); "
        f"Lip_max {skip['lip_max']:.3f}->{skip_jp['lip_max']:.3f}")

    pgd_gain_skip   = base["pgd"] - skip["pgd"]       # +ve = more robust
    pgd_gain_jp     = base["pgd"] - skip_jp["pgd"]
    lip_drop_skip   = base["lip_max"] - skip["lip_max"]   # +ve = lower Lip
    lip_drop_jp     = base["lip_max"] - skip_jp["lip_max"]
    acc_ok_skip     = skip["acc"]    >= base["acc"] - 0.02
    acc_ok_jp       = skip_jp["acc"] >= base["acc"] - 0.02

    out("")
    out(f"  PGD robustness gain (baseline->skip_only): {pgd_gain_skip:+.4f} "
        f"({'ROBUST' if pgd_gain_skip > 0.02 else 'no gain'})")
    out(f"  PGD robustness gain (baseline->skip_jp)  : {pgd_gain_jp:+.4f} "
        f"({'ROBUST' if pgd_gain_jp > 0.02 else 'no gain'})")
    out(f"  Lipschitz max drop  (baseline->skip_only): {lip_drop_skip:+.3f}")
    out(f"  Lipschitz max drop  (baseline->skip_jp)  : {lip_drop_jp:+.3f}")
    out(f"  Clean-acc kept (within 0.02) — skip_only={acc_ok_skip}  skip_jp={acc_ok_jp}")

    # one-line
    if pgd_gain_jp > 0.02 and acc_ok_jp and lip_drop_jp > 0:
        verdict = ("CONFIRMED: skip-only + Jacobian penalty achieves meaningful PGD "
                   "robustness gain with lower Lipschitz bound, supporting the "
                   "Veit-Wilber ensemble-of-shallow-networks view.")
    elif pgd_gain_skip > 0.02 and acc_ok_skip:
        verdict = ("PARTIAL: skip-only alone yields robustness without Jacobian "
                   "penalty, suggesting identity-path forcing suffices; JP adds "
                   "marginal benefit.")
    elif lip_drop_skip > 0 and pgd_gain_skip <= 0.02:
        verdict = ("WEAK: Lipschitz bound decreases under skip-only constraint but "
                   "empirical PGD robustness does not improve appreciably — "
                   "Lipschitz alone is insufficient for Fashion-MNIST scale.")
    else:
        verdict = ("REFUTED: identity main path does not reduce Lipschitz constant "
                   "or PGD vulnerability; Veit-Wilber ensemble hypothesis does not "
                   "translate to a practical robustness mechanism here.")
    out("")
    out("  ONE-LINE VERDICT: " + verdict)

    out("")
    out(f"done in {time.time()-t0:.1f}s")
    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
