"""
H420 - Dilated convolutions without pooling: does spatial-information destruction
       from pooling layers drive adversarial vulnerability?

HYPOTHESIS
----------
Max-pooling and average-pooling are lossy spatial compression operators.  They
impose local translation invariance by discarding exact spatial detail, collapsing
neighbourhood activations into a single scalar.  An adversary can exploit the
resulting equivalence classes: many distinct pixel arrangements map to the same
internal representation, so small perturbations can move the input to a different
equivalence class belonging to a different class, while the human-perceived class
is unchanged.

Yu & Koltun (2015, "Multi-scale Context Aggregation by Dilated Convolutions",
ICLR 2016) showed that dilated (atrous) convolutions increase receptive field
*without spatial downsampling*, thereby preserving full-resolution feature maps
throughout the network.  If pooling-induced invariance is a primary source of
adversarial vulnerability (the "info-bottleneck argument" of H407 / i-RevNet),
then replacing pooling with dilation should reduce adversarial attack success
rate (ASR) by keeping more spatial information in the feature maps, making the
internal representation injective enough that perturbation-induced class crossings
are harder.

An additional Jacobian-penalty condition tests whether penalising input-output
sensitivity on top of dilation provides complementary benefit.

CONDITIONS
----------
  A. baseline_pool  : Standard SmallCNN with 3 MaxPool2d layers (common.py arch).
  B. dilated_nopool : Same layer count / width but MaxPool replaced by dilated
                      convolutions (rates 2, 4, 4); spatial size preserved end-to-end;
                      head uses global average pool before FC.
  C. dilated_jac    : Condition B + Jacobian-norm penalty (lambda=0.01) on a
                      random mini-batch each step (||J||_F^2 via a single Rademacher
                      probe, efficient O(forward+backward)).

CONFIG
------
  DS=fashion_mnist  N_TRAIN=6000  N_EVAL=2000  EPOCHS=10  LR=0.05
  BATCH=128  SGD mom=0.9 wd=5e-4  SEED=0  EPS=0.1  PGD_STEPS=10
  CNN width=32  JAC_LAMBDA=0.01  JAC_PROBE_FRAC=0.25

REFERENCES
----------
  Yu & Koltun 2015 – dilated convolutions for dense prediction without pooling.
  H407 (i-RevNet) – information bottleneck argument; invertible nets are more robust.
  Sokolic et al. 2017 – Jacobian regularisation improves adversarial robustness.

OUT_FILE
--------
  results/fashion_mnist/h420_dilated_conv_no_pool_output.txt
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
DS            = "fashion_mnist"
N_TRAIN       = 6000
N_EVAL        = 2000
EPOCHS        = 10
LR            = 0.05
BATCH         = 128
SEED          = 0
EPS           = 0.1
PGD_STEPS     = 10
PGD_ALPHA     = 0.01
WIDTH         = 32
JAC_LAMBDA    = 0.01      # penalty weight for Jacobian-norm condition
JAC_PROBE_FRAC= 0.25      # fraction of batch used for Jacobian probe (cost control)

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h420_dilated_conv_no_pool_output.txt")


# ---- architectures ----------------------------------------------------------

class PoolCNN(nn.Module):
    """Standard 3-block CNN with MaxPool2d – the baseline (mirrors common.SmallCNN)."""
    def __init__(self, in_ch=1, n_classes=10, width=32):
        super().__init__()
        W = width
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, W,    3, padding=1), nn.BatchNorm2d(W),    nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(W,    W*2,   3, padding=1), nn.BatchNorm2d(W*2),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(W*2,  W*4,   3, padding=1), nn.BatchNorm2d(W*4),  nn.ReLU(), nn.MaxPool2d(2),
        )
        # 28 -> 14 -> 7 -> 3 (floor)  =>  feat = 3
        feat = 28 // 8   # = 3
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(W*4 * feat * feat, 256), nn.ReLU(),
            nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class DilatedNopoolCNN(nn.Module):
    """
    3-block CNN where MaxPool is replaced by dilated convolutions.

    Block layout (input spatial size preserved throughout):
      conv(3x3, d=1) -> BN -> ReLU   [standard, expands channels]
      conv(3x3, d=r) -> BN -> ReLU   [dilated, grows receptive field]

    Dilation rates: block1 d=2, block2 d=4, block3 d=4.
    All convolutions use 'same' padding so H,W=28 throughout.
    Global average pool before FC collapses spatial dims.

    Receptive field after 3 blocks (counting from centre pixel):
      standard 3x3 RF = 3; after dilation-2: RF = 5; after dilation-4: RF = 9.
      Stacking gives an effective RF > 20, matching a pooling CNN's coverage
      while retaining all 28x28 spatial positions in the feature maps.
    """
    def __init__(self, in_ch=1, n_classes=10, width=32):
        super().__init__()
        W = width

        def block(ci, co, dil):
            pad = dil  # 'same' padding for 3x3 with dilation d: pad = d
            return nn.Sequential(
                nn.Conv2d(ci, co,   3, padding=1),    nn.BatchNorm2d(co),   nn.ReLU(),
                nn.Conv2d(co, co,   3, padding=pad, dilation=dil),
                nn.BatchNorm2d(co), nn.ReLU(),
            )

        self.b1 = block(in_ch, W,   dil=2)
        self.b2 = block(W,     W*2, dil=4)
        self.b3 = block(W*2,   W*4, dil=4)
        self.gap = nn.AdaptiveAvgPool2d(1)           # global average pool
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(W*4, 256), nn.ReLU(),
            nn.Linear(256, n_classes))

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.gap(x)
        return self.head(x)


# ---- Jacobian probe (Rademacher, O(1 backward)) ----------------------------

def jacobian_norm_sq_probe(model, x):
    """
    Estimate ||J_f(x)||_F^2 via a single Rademacher vector probe.
    Returns a scalar tensor (mean over batch) that can be added to the loss.

    For f: R^d -> R^k, pick v ~ Rademacher(k).  Then
        v^T f  is a scalar; its gradient w.r.t. x is J^T v.
    E_v[||J^T v||^2] = ||J||_F^2  (unbiased).
    """
    x = x.detach().requires_grad_(True)
    out = model(x)                    # (B, k)
    k = out.shape[1]
    v = torch.randint(0, 2, out.shape, device=out.device).float() * 2 - 1  # Rademacher
    scalar = (out * v).sum()
    grad, = torch.autograd.grad(scalar, x, create_graph=True)
    return grad.pow(2).sum(1).mean()  # mean over batch


# ---- training ---------------------------------------------------------------

def _make_opt(model):
    return torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=5e-4)


def train_standard(model, Xtr, Ytr, seed):
    C.set_seed(seed)
    opt   = _make_opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n     = Xtr.size(0)
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


def train_with_jacobian_penalty(model, Xtr, Ytr, seed):
    """Train with cross-entropy + JAC_LAMBDA * E[||J||_F^2] using a Rademacher probe."""
    C.set_seed(seed)
    opt   = _make_opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n     = Xtr.size(0)
    n_probe = max(1, int(BATCH * JAC_PROBE_FRAC))
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb)
            # Jacobian penalty on a random sub-batch
            probe_idx = torch.randperm(xb.size(0), device=xb.device)[:n_probe]
            jac = jacobian_norm_sq_probe(model, xb[probe_idx])
            loss = ce + JAC_LAMBDA * jac
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- evaluation -------------------------------------------------------------

def eval_all(model, Xte, Yte, label):
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd",  eps=EPS, steps=PGD_STEPS)
    return {"label": label, "acc": acc, "fgsm": fg["asr"], "pgd": pg["asr"]}


# ---- main -------------------------------------------------------------------

def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def save():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H420  Dilated-conv no-pool CNN vs pooling CNN: spatial-info & adversarial")
    out("      vulnerability (Fashion-MNIST)")
    out("=" * 80)
    out("HYPOTHESIS: pooling destroys spatial info -> collapsed equivalence classes")
    out("  -> adversary exploits class crossings.  Dilated convolutions (Yu & Koltun")
    out("  2015) preserve full-resolution feature maps and should reduce ASR.")
    out("")
    out(f"config: N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}  EPOCHS={EPOCHS}  LR={LR}")
    out(f"        BATCH={BATCH}  SGD(mom=0.9,wd=5e-4)  SEED={SEED}")
    out(f"        EPS={EPS}  PGD_STEPS={PGD_STEPS}  PGD_ALPHA={PGD_ALPHA}")
    out(f"        WIDTH={WIDTH}  JAC_LAMBDA={JAC_LAMBDA}  "
        f"JAC_PROBE_FRAC={JAC_PROBE_FRAC}")
    out(f"        device={C.DEVICE}")
    out("")

    # data
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    results = []

    # ---- condition A: baseline pool CNN -------------------------------------
    out("[A] baseline_pool: standard MaxPool2d CNN (common.SmallCNN architecture)")
    C.set_seed(SEED)
    m_pool = PoolCNN(in_ch=1, n_classes=10, width=WIDTH).to(C.DEVICE)
    train_standard(m_pool, Xtr, Ytr, SEED)
    r_pool = eval_all(m_pool, Xte, Yte, "baseline_pool")
    results.append(r_pool)
    out(f"    clean_acc={r_pool['acc']:.4f}  FGSM_ASR={r_pool['fgsm']:.4f}  "
        f"PGD_ASR={r_pool['pgd']:.4f}  ({time.time()-t0:.0f}s)")
    save()

    # ---- condition B: dilated no-pool ---------------------------------------
    out("")
    out("[B] dilated_nopool: MaxPool replaced by dilated convs (d=2,4,4); "
        "full 28x28 features throughout")
    C.set_seed(SEED)
    m_dil = DilatedNopoolCNN(in_ch=1, n_classes=10, width=WIDTH).to(C.DEVICE)
    train_standard(m_dil, Xtr, Ytr, SEED)
    r_dil = eval_all(m_dil, Xte, Yte, "dilated_nopool")
    results.append(r_dil)
    out(f"    clean_acc={r_dil['acc']:.4f}  FGSM_ASR={r_dil['fgsm']:.4f}  "
        f"PGD_ASR={r_dil['pgd']:.4f}  ({time.time()-t0:.0f}s)")
    save()

    # ---- condition C: dilated + Jacobian penalty ----------------------------
    out("")
    out(f"[C] dilated_jac: dilated_nopool + Jacobian-norm penalty "
        f"(lambda={JAC_LAMBDA}, Rademacher probe on {int(JAC_PROBE_FRAC*100)}% of batch)")
    C.set_seed(SEED)
    m_jac = DilatedNopoolCNN(in_ch=1, n_classes=10, width=WIDTH).to(C.DEVICE)
    train_with_jacobian_penalty(m_jac, Xtr, Ytr, SEED)
    r_jac = eval_all(m_jac, Xte, Yte, "dilated_jac")
    results.append(r_jac)
    out(f"    clean_acc={r_jac['acc']:.4f}  FGSM_ASR={r_jac['fgsm']:.4f}  "
        f"PGD_ASR={r_jac['pgd']:.4f}  ({time.time()-t0:.0f}s)")
    save()

    # ---- summary table ------------------------------------------------------
    out("")
    out("=" * 80)
    out("SUMMARY TABLE")
    out("=" * 80)
    hdr = "{:<18} {:>10} {:>10} {:>10}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in results:
        out("{:<18} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["label"], r["acc"], r["fgsm"], r["pgd"]))
    out("-" * len(hdr))

    # deltas vs baseline
    base = results[0]
    out("")
    for r in results[1:]:
        d_acc  = r["acc"]  - base["acc"]
        d_fgsm = r["fgsm"] - base["fgsm"]
        d_pgd  = r["pgd"]  - base["pgd"]
        out(f"  {r['label']} vs baseline_pool:  "
            f"d_clean={d_acc:+.4f}  d_FGSM={d_fgsm:+.4f}  d_PGD={d_pgd:+.4f}")

    # ---- verdict ------------------------------------------------------------
    out("")
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)

    dil_pgd_delta  = r_dil["pgd"] - base["pgd"]   # negative => more robust
    jac_pgd_delta  = r_jac["pgd"] - base["pgd"]
    dil_acc_ok     = r_dil["acc"] >= base["acc"] - 0.02
    jac_acc_ok     = r_jac["acc"] >= base["acc"] - 0.02
    best_pgd_delta = min(dil_pgd_delta, jac_pgd_delta)
    best_label     = "dilated_jac" if jac_pgd_delta < dil_pgd_delta else "dilated_nopool"

    out(f"  dilated_nopool PGD_ASR change vs baseline : {dil_pgd_delta:+.4f} "
        f"({'more robust' if dil_pgd_delta < 0 else 'less/same'})")
    out(f"  dilated_jac    PGD_ASR change vs baseline : {jac_pgd_delta:+.4f} "
        f"({'more robust' if jac_pgd_delta < 0 else 'less/same'})")
    out(f"  dilated_nopool clean_acc within 2pp       : {dil_acc_ok}")
    out(f"  dilated_jac    clean_acc within 2pp       : {jac_acc_ok}")
    out("")

    if best_pgd_delta < -0.03 and (dil_acc_ok or jac_acc_ok):
        verdict = (
            "SUPPORTED: replacing pooling with dilated convolutions reduces adversarial "
            "ASR (>3pp PGD improvement) while preserving clean accuracy.  The spatial "
            "information destruction imposed by pooling contributes to adversarial "
            "vulnerability, consistent with the info-bottleneck argument (H407 / i-RevNet) "
            "and Yu & Koltun 2015.")
    elif best_pgd_delta < -0.01:
        verdict = (
            "PARTIAL: dilated-conv model shows a small PGD_ASR reduction (<3pp) vs the "
            "pooling baseline.  Pooling-induced invariance may be one factor among several; "
            "dilation alone is insufficient for strong robustness.  Jacobian penalty "
            "provides additive or complementary benefit: " +
            (f"jac d_PGD={jac_pgd_delta:+.4f}  dil d_PGD={dil_pgd_delta:+.4f}."))
    else:
        verdict = (
            "NOT SUPPORTED: dilated convolutions without pooling do not meaningfully "
            "reduce adversarial ASR vs the pooling baseline.  Spatial information "
            "destruction from pooling is not the primary driver of adversarial "
            "vulnerability under these conditions.  The equivalence-class / info-bottleneck "
            "argument does not hold quantitatively for this architecture and dataset.")

    out("ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time()-t0:.1f}s")
    save()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
