"""
H426 - Rotation-prediction SSL pretraining (Gidaris et al. 2018).

Hypothesis: Self-supervised pretraining via rotation prediction (predict which
of four 90-degree rotations was applied to an input image) produces more
robust feature representations than random initialisation. Finetuning a
linear head (or full network) on top of rotation-SSL features should yield
lower adversarial attack success rates than the standard fully-supervised
baseline trained from scratch.

Motivation
----------
Gidaris et al. (2018) "Unsupervised Representation Learning by Predicting
Image Rotations" (ICLR 2018) showed that rotation prediction is a strong
pretext task that captures semantic structure. Hendrycks et al. (2019)
"Using Pre-Training Can Improve Model Robustness and Uncertainty" (ICML 2019)
demonstrated that SSL pretraining improves corruption and adversarial
robustness. This hypothesis tests whether that benefit holds on Fashion-MNIST
with a small CNN.

Design
------
Phase 1 – SSL pretraining (10 epochs, all 60 000 unlabelled training images):
  * For each image apply a random rotation from {0, 90, 180, 270} degrees.
  * A 4-way rotation-prediction head is attached to the CNN backbone.
  * Train with cross-entropy on the rotation label (no class labels used).

Phase 2 – Supervised finetuning (10 epochs, N_TRAIN=6000 labelled images):
  * Replace the 4-way head with a 10-way classification head.
  * Two finetuning conditions:
    a) finetune_all  – update the entire network.
    b) finetune_head – freeze backbone; only train the classifier head.

Baseline:
  * SmallCNN trained from scratch with standard supervised learning (10 epochs).

Evaluation: clean accuracy, FGSM ASR, PGD ASR (eps=0.1, steps=10).

Reported rows: baseline | pretrain+finetune_all | pretrain+finetune_head

Config: N_TRAIN=6000, N_EVAL=2000, EPOCHS_SSL=10, EPOCHS_FT=10, BATCH=128,
        LR_SSL=0.05, LR_FT=0.05, SGD mom=0.9 wd=1e-4, SEED=0, EPS=0.1,
        PGD_STEPS=10, PGD_ALPHA=0.01.

References
----------
- Gidaris et al. (2018) "Unsupervised Representation Learning by Predicting
  Image Rotations." ICLR 2018. arXiv:1803.07728.
- Hendrycks et al. (2019) "Using Pre-Training Can Improve Model Robustness
  and Uncertainty." ICML 2019. arXiv:1901.09960.
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
EPOCHS_SSL = 10      # rotation-prediction pretraining epochs
EPOCHS_FT = 10       # supervised finetuning epochs
BATCH = 128
LR_SSL = 0.05
LR_FT = 0.05
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
N_ROTATIONS = 4      # {0, 90, 180, 270}

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h426_rotation_prediction_pretrain_output.txt",
)


# ---- model -------------------------------------------------------------------

class SmallCNNBackbone(nn.Module):
    """Backbone extracted from C.build_model('cnn', ..., width=32).
    Returns a flat feature vector; no final classifier."""

    def __init__(self, channels=1, width=32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(channels, w, 3, padding=1), nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1),        nn.ReLU(),
            nn.MaxPool2d(2),                       # 14x14
            nn.Conv2d(w, 2 * w, 3, padding=1),    nn.ReLU(),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 7x7
        )
        self.feat_dim = 2 * w * 7 * 7

    def forward(self, x):
        return self.features(x).flatten(1)


class RotationNet(nn.Module):
    """Backbone + 4-way rotation-prediction head."""

    def __init__(self, channels=1, width=32):
        super().__init__()
        self.backbone = SmallCNNBackbone(channels, width)
        self.head = nn.Linear(self.backbone.feat_dim, N_ROTATIONS)

    def forward(self, x):
        return self.head(self.backbone(x))


class ClassifierNet(nn.Module):
    """Backbone + 10-way classification head."""

    def __init__(self, backbone: SmallCNNBackbone, n_classes=10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.feat_dim, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# ---- helpers -----------------------------------------------------------------

def _make_sgd(params, lr):
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)


def rotate_batch(x, k):
    """Rotate all images in x by k*90 degrees (k in {0,1,2,3})."""
    # torch.rot90 rotates counter-clockwise in the HW plane
    return torch.rot90(x, k=k, dims=[-2, -1])


def make_rotation_dataset(X):
    """Apply one of the 4 rotations uniformly at random to each sample.
    Returns (X_rot, Y_rot) where Y_rot in {0,1,2,3}."""
    n = X.size(0)
    labels = torch.randint(0, N_ROTATIONS, (n,), device=X.device)
    X_rot = X.clone()
    for k in range(N_ROTATIONS):
        mask = labels == k
        if mask.any():
            X_rot[mask] = rotate_batch(X[mask], k)
    return X_rot, labels


# ---- training routines -------------------------------------------------------

def pretrain_ssl(X_all):
    """Phase 1: rotation-prediction SSL on X_all (no labels used)."""
    C.set_seed(SEED)
    net = RotationNet(channels=META["channels"]).to(C.DEVICE)
    opt = _make_sgd(net.parameters(), LR_SSL)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_SSL)
    n = X_all.size(0)
    net.train()
    for ep in range(EPOCHS_SSL):
        perm = torch.randperm(n, device=X_all.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = X_all[idx]
            xb_rot, yb_rot = make_rotation_dataset(xb)
            opt.zero_grad()
            loss = F.cross_entropy(net(xb_rot), yb_rot)
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    return net.backbone   # return backbone only


def finetune(backbone, Xtr, Ytr, freeze_backbone):
    """Phase 2: supervised finetuning with or without backbone freezing."""
    C.set_seed(SEED + 1)
    model = ClassifierNet(backbone, n_classes=META["n_classes"]).to(C.DEVICE)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        params = model.head.parameters()
    else:
        params = model.parameters()
    opt = _make_sgd(list(params), LR_FT)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_FT)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS_FT):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(True)
    model.eval()
    return model


def train_baseline(Xtr, Ytr):
    """Standard supervised training from scratch (10 epochs)."""
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, seed=SEED)
    opt = _make_sgd(list(model.parameters()), LR_FT)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_FT)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS_FT):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_model(model, Xte, Yte):
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd",  eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


# ---- main --------------------------------------------------------------------

def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H426  Rotation-prediction SSL pretraining (Gidaris 2018) — Fashion-MNIST")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} "
        f"EPOCHS_SSL={EPOCHS_SSL} EPOCHS_FT={EPOCHS_FT}")
    out(f"        BATCH={BATCH} LR_SSL={LR_SSL} LR_FT={LR_FT} SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        device={C.DEVICE}")
    out("")
    out("References:")
    out("  Gidaris et al. (2018) 'Unsupervised Representation Learning by")
    out("    Predicting Image Rotations.' ICLR 2018. arXiv:1803.07728.")
    out("  Hendrycks et al. (2019) 'Using Pre-Training Can Improve Model")
    out("    Robustness and Uncertainty.' ICML 2019. arXiv:1901.09960.")
    out("")

    # ---- data ----------------------------------------------------------------
    # Load full training split for SSL (60 000); labelled subset for finetuning
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    # For SSL use all available training data (no label needed)
    X_all_train, _, _, _ = C.load_dataset(DS, n_train=60000, n_eval=0, seed=SEED)
    out(f"data: X_all_ssl={tuple(X_all_train.shape)}  "
        f"Xtr_labelled={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    # ---- [1] Baseline --------------------------------------------------------
    out("[1] Training BASELINE (standard supervised, from scratch, 10 epochs)...")
    baseline = train_baseline(Xtr, Ytr)
    b_acc, b_fgsm, b_pgd = eval_model(baseline, Xte, Yte)
    out(f"    baseline: clean_acc={b_acc:.4f}  FGSM_ASR={b_fgsm:.4f}  "
        f"PGD_ASR={b_pgd:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- [2] SSL pretraining -------------------------------------------------
    out(f"\n[2] SSL PRETRAINING — rotation prediction ({EPOCHS_SSL} epochs, "
        f"all {X_all_train.size(0)} training images, no class labels)...")
    backbone = pretrain_ssl(X_all_train)
    out(f"    pretraining done  ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- [3] Finetuning: full network ----------------------------------------
    out(f"\n[3] FINETUNING (finetune_all — full network, {EPOCHS_FT} epochs)...")
    # finetune_all needs its own backbone copy to not clobber finetune_head's
    import copy
    backbone_a = copy.deepcopy(backbone)
    ft_all = finetune(backbone_a, Xtr, Ytr, freeze_backbone=False)
    fa_acc, fa_fgsm, fa_pgd = eval_model(ft_all, Xte, Yte)
    out(f"    finetune_all: clean_acc={fa_acc:.4f}  FGSM_ASR={fa_fgsm:.4f}  "
        f"PGD_ASR={fa_pgd:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- [4] Finetuning: head only -------------------------------------------
    out(f"\n[4] FINETUNING (finetune_head — backbone frozen, {EPOCHS_FT} epochs)...")
    backbone_h = copy.deepcopy(backbone)
    ft_head = finetune(backbone_h, Xtr, Ytr, freeze_backbone=True)
    fh_acc, fh_fgsm, fh_pgd = eval_model(ft_head, Xte, Yte)
    out(f"    finetune_head: clean_acc={fh_acc:.4f}  FGSM_ASR={fh_fgsm:.4f}  "
        f"PGD_ASR={fh_pgd:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- [5] Main table ------------------------------------------------------
    rows = [
        {"label": "baseline (from scratch)",     "acc": b_acc,  "fgsm": b_fgsm,  "pgd": b_pgd},
        {"label": "SSL-pretrain + finetune_all",  "acc": fa_acc, "fgsm": fa_fgsm, "pgd": fa_pgd},
        {"label": "SSL-pretrain + finetune_head", "acc": fh_acc, "fgsm": fh_fgsm, "pgd": fh_pgd},
    ]

    out("\n" + "=" * 80)
    out("[5] MAIN TABLE")
    out("=" * 80)
    hdr = "{:<38} {:>10} {:>10} {:>10}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<38} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["label"], r["acc"], r["fgsm"], r["pgd"]))
    out("-" * len(hdr))

    # ---- [6] Verdict ---------------------------------------------------------
    out("\n" + "=" * 80)
    out("[6] VERDICT")
    out("=" * 80)

    best_ssl = min(rows[1:], key=lambda r: r["pgd"])
    gain_pgd = b_pgd - best_ssl["pgd"]          # positive => SSL is more robust
    gain_fgsm = b_fgsm - best_ssl["fgsm"]
    acc_drop = b_acc - best_ssl["acc"]           # positive => accuracy declined

    out(f"  Baseline PGD_ASR                  = {b_pgd:.4f}")
    out(f"  Best SSL condition PGD_ASR         = {best_ssl['pgd']:.4f}  "
        f"({best_ssl['label']})")
    out(f"  PGD robustness gain (positive=better) = {gain_pgd:+.4f}")
    out(f"  FGSM robustness gain                  = {gain_fgsm:+.4f}")
    out(f"  Clean-acc change vs baseline          = {-acc_drop:+.4f}  "
        f"({'DROP' if acc_drop > 0.02 else 'OK'})")

    out(f"\n  finetune_all  vs baseline: PGD_ASR {b_pgd:.4f} -> {fa_pgd:.4f} "
        f"({fa_pgd - b_pgd:+.4f})")
    out(f"  finetune_head vs baseline: PGD_ASR {b_pgd:.4f} -> {fh_pgd:.4f} "
        f"({fh_pgd - b_pgd:+.4f})")

    robust_gain = gain_pgd > 0.02
    acc_ok = acc_drop <= 0.02

    if robust_gain and acc_ok:
        verdict = ("YES: rotation-SSL pretraining improves adversarial robustness "
                   "vs the supervised baseline with no clean-accuracy penalty.")
    elif robust_gain and not acc_ok:
        verdict = ("PARTIAL: rotation-SSL improves robustness but at a clean-accuracy cost.")
    elif not robust_gain and fa_pgd < fh_pgd:
        verdict = ("NO: rotation-SSL pretraining does not improve adversarial robustness "
                   "over the supervised baseline; full finetuning beats frozen-head.")
    else:
        verdict = ("NO: rotation-SSL pretraining does not yield meaningful robustness "
                   "improvement over the supervised from-scratch baseline.")

    out("\n  ONE-LINE VERDICT: " + verdict)
    out(f"\ndone in {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
