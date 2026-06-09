"""
H417 - Multi-scale Feature Pyramid for Adversarial Robustness (Fashion-MNIST).

Hypothesis: perturbations exploit single-scale features; multi-scale aggregation
forces the attacker to flip multiple scales simultaneously, raising attack cost.
A Feature Pyramid Network (FPN)-style architecture (Lin et al. 2017,
https://arxiv.org/abs/1612.03144) adds top-down pathways and lateral skip
connections so the classifier aggregates representations from three spatial
scales. We test whether this structural inductive bias reduces PGD attack
success rate (ASR) versus a matched-width baseline SmallCNN.

Reference: T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, S. Belongie,
"Feature Pyramid Networks for Object Detection," CVPR 2017.

Conditions:
  1. baseline      - SmallCNN (3-block, width=32), standard training
  2. fpn_std       - FPN-style multiscale CNN, standard training
  3. fpn_at        - FPN-style multiscale CNN, PGD adversarial training

Design:
  - FPN backbone: 3 conv blocks (stride-2 MaxPool each) -> P3, P2, P1 maps.
  - Top-down path: upsample coarser maps + lateral 1x1 conv from each level.
  - Each merged feature map is global-avg-pooled; concatenate -> shared head.
  - Same number of base channels (width=32) as SmallCNN so parameter counts
    are broadly comparable.
  - Standard training: SGD+cosine, 15 epochs.
  - Adversarial training (AT): PGD inner loop eps=0.1, steps=7 during training.
  - Evaluation: clean accuracy + PGD ASR (eps=0.1, steps=20) on 2000 test samples.

Config: DS=fashion_mnist, N_TRAIN=6000, N_EVAL=2000, EPOCHS=15, LR=0.05,
BATCH=128, SGD(mom=0.9,wd=1e-4), SEED=0, EPS=0.1, PGD_STEPS=20, width=32.

OUT_FILE: results/fashion_mnist/h417_multiscale_feature_pyramid_output.txt
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config ------------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 15
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 20
PGD_ALPHA = 2.5 * EPS / PGD_STEPS
WIDTH = 32
META = {"channels": 1, "size": 28, "n_classes": 10}


# ---- FPN-style model ---------------------------------------------------------
class FPNSmallCNN(nn.Module):
    """FPN-style 3-level feature pyramid for 28x28 grayscale input.

    Bottom-up: 3 conv blocks each followed by MaxPool(2) ->
      P1: (W, 14, 14), P2: (2W, 7, 7), P3: (4W, 3, 3)  [floor division]
    Top-down with lateral connections:
      M3 = P3 (coarsest, already 3x3)
      M2 = lateral(P2) + upsample(M3)
      M1 = lateral(P1) + upsample(M2)
    Each Mi is global-avg-pooled to a vector; concatenate and classify.
    """

    def __init__(self, in_ch=1, n_classes=10, width=32):
        super().__init__()
        W = width
        # ---- bottom-up backbone ----
        def _block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.layer1 = _block(in_ch, W)       # -> (W,  14, 14)
        self.layer2 = _block(W, W * 2)       # -> (2W,  7,  7)
        self.layer3 = _block(W * 2, W * 4)   # -> (4W,  3,  3)

        # ---- lateral 1x1 projections (to common fpn_dim channels) ----
        fpn_dim = W * 2
        self.lat1 = nn.Conv2d(W, fpn_dim, 1)
        self.lat2 = nn.Conv2d(W * 2, fpn_dim, 1)
        self.lat3 = nn.Conv2d(W * 4, fpn_dim, 1)

        # ---- after merge: 3x3 conv to smooth ----
        self.merge2 = nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
                                    nn.ReLU(inplace=True))
        self.merge1 = nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
                                    nn.ReLU(inplace=True))

        # ---- head: concat pooled vectors from all 3 levels ----
        self.head = nn.Sequential(
            nn.Linear(fpn_dim * 3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )
        self._fpn_dim = fpn_dim

    def forward(self, x):
        # bottom-up
        c1 = self.layer1(x)   # (B, W,  14, 14)
        c2 = self.layer2(c1)  # (B, 2W,  7,  7)
        c3 = self.layer3(c2)  # (B, 4W,  3,  3)

        # top-down
        m3 = self.lat3(c3)                                         # (B, fpn, 3, 3)
        m2 = self.merge2(
            self.lat2(c2) +
            F.interpolate(m3, size=c2.shape[-2:], mode="nearest")  # (B, fpn, 7, 7)
        )
        m1 = self.merge1(
            self.lat1(c1) +
            F.interpolate(m2, size=c1.shape[-2:], mode="nearest")  # (B, fpn, 14, 14)
        )

        # global avg pool each level
        p3 = m3.flatten(2).mean(2)  # (B, fpn)
        p2 = m2.flatten(2).mean(2)
        p1 = m1.flatten(2).mean(2)

        feat = torch.cat([p1, p2, p3], dim=1)  # (B, 3*fpn)
        return self.head(feat)


# ---- training helpers --------------------------------------------------------
def _make_sgd(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)


def train_standard(model, Xtr, Ytr):
    opt = _make_sgd(model)
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


def train_adversarial(model, Xtr, Ytr):
    """PGD adversarial training (eps=0.1, steps=7)."""
    opt = _make_sgd(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    at_alpha = 2.5 * EPS / 7
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_adv = C.pgd(model, xb, yb, eps=EPS, steps=7, alpha=at_alpha)
            opt.zero_grad()
            F.cross_entropy(model(xb_adv), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_all(model, Xte, Yte):
    _, acc = C.logits_and_acc(model, Xte, Yte)
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    return acc, fg["asr"], pg["asr"]


# ---- main --------------------------------------------------------------------
def main():
    t0 = time.time()
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "fashion_mnist",
        "h417_multiscale_feature_pyramid_output.txt",
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
    out("H417  Multi-scale Feature Pyramid (FPN-style) Adversarial Robustness")
    out("      Dataset: Fashion-MNIST")
    out("=" * 80)
    out("Hypothesis: perturbations exploit single-scale features; FPN-style")
    out("multi-scale aggregation forces attacker to flip multiple scales")
    out("simultaneously, raising attack cost (Lin et al. 2017 FPN).")
    out("")
    out(f"config: DS={DS} N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS}")
    out(f"        LR={LR} BATCH={BATCH} SGD(mom=0.9,wd=1e-4) WIDTH={WIDTH}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA:.4f}")
    out(f"        SEED={SEED}  device={C.DEVICE}")
    out("")

    # ---- data ----------------------------------------------------------------
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    results = []

    # ---- condition 1: baseline SmallCNN ------------------------------------
    out("[1/3] BASELINE - SmallCNN (width=32), standard training")
    C.set_seed(SEED)
    base = C.build_model("cnn", META, width=WIDTH).to(C.DEVICE)
    train_standard(base, Xtr, Ytr)
    acc, fgsm_asr, pgd_asr = eval_all(base, Xte, Yte)
    out(f"      clean_acc={acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  PGD_ASR={pgd_asr:.4f}")
    out(f"      elapsed={time.time()-t0:.0f}s")
    results.append({"cond": "baseline_cnn", "acc": acc,
                    "fgsm_asr": fgsm_asr, "pgd_asr": pgd_asr})
    flush_file()
    out("")

    # ---- condition 2: FPN standard training --------------------------------
    out("[2/3] FPN-STYLE - FPNSmallCNN (width=32), standard training")
    C.set_seed(SEED)
    fpn_std = FPNSmallCNN(in_ch=1, n_classes=10, width=WIDTH).to(C.DEVICE)
    train_standard(fpn_std, Xtr, Ytr)
    acc2, fgsm2, pgd2 = eval_all(fpn_std, Xte, Yte)
    out(f"      clean_acc={acc2:.4f}  FGSM_ASR={fgsm2:.4f}  PGD_ASR={pgd2:.4f}")
    out(f"      d_pgd_asr vs baseline = {pgd2 - pgd_asr:+.4f}")
    out(f"      elapsed={time.time()-t0:.0f}s")
    results.append({"cond": "fpn_std", "acc": acc2,
                    "fgsm_asr": fgsm2, "pgd_asr": pgd2})
    flush_file()
    out("")

    # ---- condition 3: FPN + adversarial training ---------------------------
    out("[3/3] FPN + AT - FPNSmallCNN (width=32), PGD adversarial training")
    C.set_seed(SEED)
    fpn_at = FPNSmallCNN(in_ch=1, n_classes=10, width=WIDTH).to(C.DEVICE)
    train_adversarial(fpn_at, Xtr, Ytr)
    acc3, fgsm3, pgd3 = eval_all(fpn_at, Xte, Yte)
    out(f"      clean_acc={acc3:.4f}  FGSM_ASR={fgsm3:.4f}  PGD_ASR={pgd3:.4f}")
    out(f"      d_pgd_asr vs baseline = {pgd3 - pgd_asr:+.4f}")
    out(f"      d_pgd_asr vs fpn_std  = {pgd3 - pgd2:+.4f}")
    out(f"      elapsed={time.time()-t0:.0f}s")
    results.append({"cond": "fpn_at", "acc": acc3,
                    "fgsm_asr": fgsm3, "pgd_asr": pgd3})
    flush_file()
    out("")

    # ---- summary table -------------------------------------------------------
    out("=" * 80)
    out("SUMMARY TABLE")
    out("=" * 80)
    hdr = "{:<20} {:>10} {:>10} {:>10} {:>12} {:>12}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR",
        "d_FGSM_ASR", "d_PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    base_r = results[0]
    for r in results:
        df = r["fgsm_asr"] - base_r["fgsm_asr"]
        dp = r["pgd_asr"] - base_r["pgd_asr"]
        df_s = f"{df:+.4f}" if r["cond"] != "baseline_cnn" else "---"
        dp_s = f"{dp:+.4f}" if r["cond"] != "baseline_cnn" else "---"
        out("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>12} {:>12}".format(
            r["cond"], r["acc"], r["fgsm_asr"], r["pgd_asr"], df_s, dp_s))
    out("-" * len(hdr))
    out("")

    # ---- verdict -------------------------------------------------------------
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)

    fpn_std_r = results[1]
    fpn_at_r = results[2]

    arch_gain = base_r["pgd_asr"] - fpn_std_r["pgd_asr"]   # positive = FPN more robust
    at_gain   = base_r["pgd_asr"] - fpn_at_r["pgd_asr"]    # positive = FPN+AT more robust
    at_arch_delta = fpn_std_r["pgd_asr"] - fpn_at_r["pgd_asr"]  # AT on top of FPN
    acc_ok = fpn_std_r["acc"] >= base_r["acc"] - 0.02

    out(f"  Architecture-only PGD robustness gain (FPN_std vs base): {arch_gain:+.4f}")
    out(f"  FPN+AT PGD robustness gain vs base:                      {at_gain:+.4f}")
    out(f"  Additional AT gain on top of FPN architecture:           {at_arch_delta:+.4f}")
    out(f"  FPN_std clean-acc within 0.02 of baseline:               {acc_ok}")
    out("")

    if arch_gain > 0.02 and acc_ok:
        verdict = ("SUPPORTED: FPN multi-scale architecture alone reduces PGD ASR by "
                   f"{arch_gain:.3f} while preserving clean accuracy.")
    elif arch_gain > 0.02 and not acc_ok:
        verdict = ("PARTIAL: FPN reduces PGD ASR but at a clean-accuracy cost > 0.02.")
    elif arch_gain <= 0.02 and at_gain > 0.05:
        verdict = ("NOT BY ARCH ALONE: FPN architecture gives minimal robustness gain "
                   "without AT; FPN+AT does help.")
    else:
        verdict = ("NOT SUPPORTED: neither FPN architecture alone nor FPN+AT substantially "
                   "reduces PGD ASR beyond baseline.")

    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
