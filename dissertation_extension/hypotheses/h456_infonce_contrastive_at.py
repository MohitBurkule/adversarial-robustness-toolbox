"""
H456 - InfoNCE Contrastive Adversarial Training (fixes H348 SupCon-AT).

Anchor: this hypothesis fills gap G3 in CAMPAIGN_GAP_MAP.md section 5
("Contrastive AT beyond H348 (which collapsed). InfoNCE-AT, CLAW (Yu 2022)").

Diagnosis of H348 (SupCon-AT, RESULTS_SUMMARY clean ~= 0.077):
  H348 trained the backbone with SupCon ONLY (no CE), then tried to read out
  classifications from the final Linear head INSIDE the backbone's `head`
  branch. That linear head never received any gradient signal from labels:
  SupCon flows through model.encode() / model.proj, which is a separate
  branch. With no label-conditioned objective on the classifier, the
  pre-softmax weights are random, so accuracy collapses near 1/10. This is
  not "SupCon doesn't help" -- the experiment never actually trained a
  classifier. The Khosla 2020 SupCon paper itself uses a *two-stage*
  recipe (contrastive pretraining, then a frozen-backbone linear probe
  trained with CE) precisely to avoid this failure mode.

Fix (this script):
  Train ONE network jointly with two loss terms in every minibatch:
    (a) InfoNCE (van den Oord 2018) on L2-normalised projection-head
        outputs, treating (clean x, PGD-adversarial x) of the SAME class as
        the POSITIVE pair (different-class instances in the batch as
        negatives). This generalises Kim 2020 RoCL's instance-level
        positives to a CLASS-conditioned positive pair and matches the
        Jiang 2020 ACL motivation that "clean and its adversarial twin
        must occupy the same representation".
    (b) Cross-entropy on the linear classifier head, applied to the
        ADVERSARIAL example (Madry-style adversarial training). The
        classifier head is explicitly trained on labels -- the H348 bug
        cannot recur.
  PGD attacks the CLASSIFIER head (not the projection head), so attack
  shape exactly matches Madry / PGD-AT.

Hypothesis: at the campaign's small-scale (N=6000, 10 ep) joint
InfoNCE+CE-AT should match PGD-AT on clean acc and at least tie its PGD
ASR, and should improve the worst-class ASR by virtue of the
class-conditioned InfoNCE pulling rare classes' adversarial twins back
into their cluster (Khosla 2020 motivation).

Conditions:
  A. standard-CE baseline (also TRANSFER surrogate)
  B. PGD-AT baseline (CE only on adversarial; matches campaign winner)
  C. InfoNCE-only (alpha=1, beta=0) -- ablation: classifier never trained
     with CE. This is the H348 failure mode replicated for confirmation.
  D. InfoNCE+CE joint, temperature tau=0.1, alpha (CE) = 1.0, beta (NCE) = 1.0
  E. InfoNCE+CE joint, tau=0.5, alpha=1.0, beta=1.0
  F. InfoNCE+CE joint, tau=1.0, alpha=1.0, beta=1.0
  G. InfoNCE+CE joint, tau=0.5, beta=0.5 (lower NCE weight)
  H. AT + InfoNCE on (clean,clean) positives instead of (clean,adv)
     (ablation: which positive pair drives the gain).
  All conditions D-H attack the LINEAR HEAD with PGD-10 in training.

Each condition reports:
  clean_acc, FGSM ASR, PGD-10 ASR, mean margin,
  TRANSFER ASR (PGD adversarials crafted on the STANDARD baseline,
                eval'd on this model -- masking check),
  per-class PGD ASR + worst-class PGD ASR, wall-clock seconds.

Papers consulted (verified via WebSearch):
  - `oord-2018-cpc` van den Oord et al. arXiv 1807.03748 "Representation
    Learning with Contrastive Predictive Coding" -- introduces InfoNCE.
  - `khosla-2020-supcon` Khosla et al. NeurIPS 2020 arXiv 2004.11362
    "Supervised Contrastive Learning" -- two-stage recipe; we adapt it to
    JOINT training with a CE head to avoid the H348 readout bug.
  - `kim-2020-rocl` Kim, Tack, Hwang ICML 2020 arXiv 2006.07589
    "Adversarial Self-Supervised Contrastive Learning" -- treats
    (x, x+delta) as a positive pair WITHOUT labels; we make the pair
    class-conditioned and add a label-supervised head.
  - `jiang-2020-acl` Jiang, Chen, Chen, Wang NeurIPS 2020 arXiv 2010.13337
    "Robust Pre-Training by Adversarial Contrastive Learning" -- ACL
    pre-training + finetune; we collapse to a single training stage.
  - `fan-2021-advcl` Fan, Liu, Liang, Hjelm, Yang, Wang ICCV 2021 arXiv
    2104.01494 "When Does Contrastive Learning Preserve Adversarial
    Robustness from Pretraining to Finetuning" -- finds that contrastive
    pretraining alone does NOT preserve robustness without explicit adv
    training; their AdvCL uses pseudo-supervision via clustering. This
    motivates our condition C (InfoNCE-only) as a NEGATIVE control.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9, wd=5e-4),
SEED=0, EPS=0.1, PGD_STEPS=10 (eval) / 10 (inner), PGD_ALPHA=0.01.
SmallCNN width=32 backbone with a projection head (2-layer MLP -> 64-d).
ASCII-only output, flushed after each condition.
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

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
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
N_CLASSES = 10

INNER_STEPS = 10
INNER_ALPHA = 2.5 * EPS / INNER_STEPS

PROJ_HIDDEN = 128
PROJ_OUT = 64

# (label, tau, alpha_ce, beta_nce, positives, description)
# positives in {"clean_adv", "clean_clean"}
CONDS = [
    ("C_nce_only_t0p5",    0.5, 0.0, 1.0, "clean_adv",   "InfoNCE-only (replicate H348 bug; no CE head training)"),
    ("D_nce_at_t0p1",      0.1, 1.0, 1.0, "clean_adv",   "InfoNCE+CE-AT, tau=0.1"),
    ("E_nce_at_t0p5",      0.5, 1.0, 1.0, "clean_adv",   "InfoNCE+CE-AT, tau=0.5"),
    ("F_nce_at_t1p0",      1.0, 1.0, 1.0, "clean_adv",   "InfoNCE+CE-AT, tau=1.0"),
    ("G_nce_at_t0p5_lo",   0.5, 1.0, 0.5, "clean_adv",   "InfoNCE+CE-AT, tau=0.5, beta=0.5"),
    ("H_nce_at_clean_pos", 0.5, 1.0, 1.0, "clean_clean", "InfoNCE+CE-AT, tau=0.5, positives=(clean,clean)"),
]

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h456_infonce_contrastive_at_output.txt"
)


# ---------------------------------------------------------------------------
# logging helper (flush per condition)
# ---------------------------------------------------------------------------
_LINES = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LINES.append(s)


def flush_file():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(_LINES) + "\n")


# ---------------------------------------------------------------------------
# model: SmallCNN backbone + projection head + linear classifier
# ---------------------------------------------------------------------------
class CNNWithProj(nn.Module):
    """SmallCNN (campaign-standard) with TWO heads sharing the same backbone:
       - classifier: Flatten -> Linear -> ReLU -> Linear(n_classes)  (PGD attacks this)
       - projection: Flatten -> Linear -> ReLU -> Linear(PROJ_OUT)   (L2-normed, InfoNCE)
    """

    def __init__(self, in_ch=1, size=28, n_classes=10, width=32,
                 proj_hidden=PROJ_HIDDEN, proj_out=PROJ_OUT):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(
            *block(in_ch, width),
            *block(width, width * 2),
            *block(width * 2, width * 4),
        )
        feat = size // 8
        self.feat_dim = width * 4 * feat * feat
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 256), nn.ReLU(),
            nn.Linear(256, n_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(self.feat_dim, proj_hidden), nn.ReLU(),
            nn.Linear(proj_hidden, proj_out),
        )

    def features_flat(self, x):
        return self.flatten(self.features(x))

    def forward(self, x):
        # Default forward is the CLASSIFIER path; PGD uses this.
        return self.classifier(self.features_flat(x))

    def project(self, x):
        z = self.projection(self.features_flat(x))
        return F.normalize(z, dim=1)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def info_nce_class_positive(z1, z2, y, tau):
    """Supervised InfoNCE with a single positive per anchor.

    z1, z2: (B, D) L2-normalised projections (paired views: e.g. clean and adv
            of the SAME sample i in row i). y: (B,) labels.

    For each anchor z1[i], we treat z2[i] as its positive. The denominator
    sums exp(sim) over all z2[j] for j != i (in-batch negatives), and we
    REMOVE same-class negatives (j != i with y[j] == y[i]) from the
    denominator so that other same-class samples are not penalised as
    negatives -- this is the class-conditioned InfoNCE (Khosla 2020 in
    spirit, but with a SINGLE explicit positive pair, matching Kim 2020
    RoCL's instance-level form).
    """
    B = z1.size(0)
    sim = torch.mm(z1, z2.t()) / tau     # (B, B)
    # mask out same-class j != i from negatives (we don't penalise them)
    y_eq = (y.unsqueeze(1) == y.unsqueeze(0))     # (B, B) bool
    diag = torch.eye(B, device=z1.device, dtype=torch.bool)
    # keep diagonal (positive) and all j with y[j] != y[i]
    keep = diag | (~y_eq)
    sim = sim.masked_fill(~keep, -1e9)
    # positive index for row i is i (the diagonal)
    targets = torch.arange(B, device=z1.device)
    return F.cross_entropy(sim, targets)


def pgd_inner(model, x, y, eps=EPS, steps=INNER_STEPS, alpha=INNER_ALPHA):
    """Standard CE-PGD inner attack against the CLASSIFIER head."""
    model.eval()
    x0 = x.detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    model.train()
    return xa.detach()


# ---------------------------------------------------------------------------
# training routines
# ---------------------------------------------------------------------------
def _opt_sched(model):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return opt, sched


def train_standard(meta, Xtr, Ytr):
    """Condition A: standard CE baseline (also transfer-attack surrogate)."""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=SEED)
    opt, sched = _opt_sched(model)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
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


def train_pgd_at(meta, Xtr, Ytr):
    """Condition B: standard PGD-AT (CE only on adversarial)."""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=SEED)
    opt, sched = _opt_sched(model)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            x_adv = pgd_inner(model, xb, yb)
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(x_adv), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_infonce_at(meta, Xtr, Ytr, tau, alpha_ce, beta_nce, positives):
    """Conditions C-H: joint InfoNCE + CE-AT training.

    For each batch:
      1. Craft x_adv with PGD against the CLASSIFIER head (fixes H348 bug).
      2. CE loss = CE(model(x_adv), y)                  [classifier trained!]
      3. NCE loss = info_nce_class_positive(z(clean), z(adv_or_clean2), y, tau)
         positives == "clean_adv":   pair (clean, adv) of same sample
         positives == "clean_clean": pair (clean, clean_again_augmented) of same sample
                                     -- for our purposes "augment" is the same
                                     image (i.e. pure clean-clean baseline);
                                     this isolates whether the (clean,adv)
                                     positive PAIR is what helps.
      4. total = alpha_ce * CE + beta_nce * NCE.
    """
    C.set_seed(SEED)
    model = CNNWithProj(in_ch=meta["channels"], size=meta["size"],
                        n_classes=meta["n_classes"], width=32).to(C.DEVICE)
    opt, sched = _opt_sched(model)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            x_adv = pgd_inner(model, xb, yb)
            model.train()
            opt.zero_grad()

            logits_adv = model(x_adv)
            loss_ce = F.cross_entropy(logits_adv, yb)

            z_clean = model.project(xb)
            if positives == "clean_adv":
                z_pair = model.project(x_adv)
            else:  # "clean_clean": use the SAME clean image as the pair
                # (degenerate-pair ablation: information-free positive)
                z_pair = z_clean
            loss_nce = info_nce_class_positive(z_clean, z_pair, yb, tau)

            loss = alpha_ce * loss_ce + beta_nce * loss_nce
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# evaluation (transfer + masking audit)
# ---------------------------------------------------------------------------
def pgd_batched(model, X, Y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA, batch=256):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(C.pgd(model, X[i:i + batch], Y[i:i + batch],
                          eps=eps, steps=steps, alpha=alpha, random_start=True))
    return torch.cat(outs)


def fgsm_batched(model, X, Y, eps=EPS, batch=256):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(C.fgsm(model, X[i:i + batch], Y[i:i + batch], eps))
    return torch.cat(outs)


@torch.no_grad()
def _preds(model, X, batch=512):
    parts = []
    for i in range(0, X.size(0), batch):
        parts.append(model(X[i:i + batch]).argmax(1).cpu())
    return torch.cat(parts)


def eval_full(model, Xte, Yte, surrogate_pgd_advx=None):
    """clean_acc, FGSM ASR, PGD-10 ASR, mean margin, transfer ASR,
    per-class PGD ASR + worst-class PGD ASR."""
    for p in model.parameters():
        p.requires_grad_(True)
    Yte_cpu = Yte.cpu()

    clean_pred = _preds(model, Xte)
    correct_mask = (clean_pred == Yte_cpu).numpy().astype(bool)
    clean_acc = float(correct_mask.mean())

    X_fg = fgsm_batched(model, Xte, Yte)
    fg_pred = _preds(model, X_fg)
    fg_flips = (fg_pred != Yte_cpu).numpy()
    fgsm_asr = float(fg_flips[correct_mask].mean()) if correct_mask.any() else float("nan")

    X_pgd = pgd_batched(model, Xte, Yte)
    pgd_pred = _preds(model, X_pgd)
    pgd_flips = (pgd_pred != Yte_cpu).numpy()
    pgd_asr = float(pgd_flips[correct_mask].mean()) if correct_mask.any() else float("nan")

    Yc = Yte_cpu.numpy()
    per_cls = []
    for c in range(N_CLASSES):
        m = correct_mask & (Yc == c)
        if m.sum() == 0:
            per_cls.append(float("nan"))
        else:
            per_cls.append(float(pgd_flips[m].mean()))
    worst_cls = float(np.nanmax(per_cls)) if any(not np.isnan(v) for v in per_cls) else float("nan")

    if surrogate_pgd_advx is not None:
        tr_pred = _preds(model, surrogate_pgd_advx)
        tr_flips = (tr_pred != Yte_cpu).numpy()
        transfer_asr = float(tr_flips[correct_mask].mean()) if correct_mask.any() else float("nan")
    else:
        transfer_asr = float("nan")

    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))

    return dict(
        clean_acc=clean_acc, fgsm_asr=fgsm_asr, pgd_asr=pgd_asr,
        transfer_asr=transfer_asr, mean_margin=mean_margin,
        per_class_pgd_asr=per_cls, worst_class_pgd_asr=worst_cls,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    log("=" * 80)
    log("H456  InfoNCE Contrastive AT (fix H348 SupCon-AT collapse) on Fashion-MNIST")
    log("=" * 80)
    log(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    log(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} "
        f"INNER_STEPS={INNER_STEPS}")
    log(f"        PROJ_HIDDEN={PROJ_HIDDEN} PROJ_OUT={PROJ_OUT}")
    log(f"        device = {C.DEVICE}")
    log("anchor: oord-2018-cpc (InfoNCE); khosla-2020-supcon; kim-2020-rocl;")
    log("        jiang-2020-acl; fan-2021-advcl")
    log("diagnosis of H348: classifier head was NOT trained with CE on labels;")
    log("                   gradients only flowed through the projection branch.")
    log("                   Result: random-classifier readout, clean ~= 0.077.")
    log("fix: joint training. PGD attacks the CLASSIFIER head;")
    log("     CE is applied to the adversarial logits;")
    log("     InfoNCE on L2-normed projections with (clean,adv) same-class positives.")
    log("")
    flush_file()

    # ---- data ----
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    log(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    flush_file()

    rows = []

    # ---- (A) standard baseline (also transfer surrogate) ----
    log("\n[A] training STANDARD baseline (also the transfer surrogate) ...")
    ts = time.time()
    std_model = train_standard(meta, Xtr, Ytr)
    std_time = time.time() - ts
    log(f"    standard trained in {std_time:.1f}s; generating surrogate PGD advs ...")
    surrogate_pgd_advx = pgd_batched(std_model, Xte, Yte)
    std_metrics = eval_full(std_model, Xte, Yte, surrogate_pgd_advx=None)
    std_metrics["time_s"] = std_time
    rows.append(("standard", std_metrics))
    log(f"    standard: clean={std_metrics['clean_acc']:.4f} "
        f"FGSM_ASR={std_metrics['fgsm_asr']:.4f} "
        f"PGD_ASR={std_metrics['pgd_asr']:.4f} "
        f"margin={std_metrics['mean_margin']:.4f} "
        f"worst_cls_PGD={std_metrics['worst_class_pgd_asr']:.4f} "
        f"({std_time:.1f}s)")
    flush_file()

    # ---- (B) PGD-AT baseline ----
    log("\n[B] training PGD-AT baseline ...")
    ts = time.time()
    at_model = train_pgd_at(meta, Xtr, Ytr)
    at_time = time.time() - ts
    at_metrics = eval_full(at_model, Xte, Yte, surrogate_pgd_advx=surrogate_pgd_advx)
    at_metrics["time_s"] = at_time
    rows.append(("pgd_at", at_metrics))
    log(f"    pgd_at  : clean={at_metrics['clean_acc']:.4f} "
        f"FGSM_ASR={at_metrics['fgsm_asr']:.4f} "
        f"PGD_ASR={at_metrics['pgd_asr']:.4f} "
        f"transfer_ASR={at_metrics['transfer_asr']:.4f} "
        f"margin={at_metrics['mean_margin']:.4f} "
        f"worst_cls_PGD={at_metrics['worst_class_pgd_asr']:.4f} "
        f"({at_time:.1f}s)")
    flush_file()

    # ---- (C-H) InfoNCE-AT conditions ----
    nce_results = {}
    for label, tau, a_ce, b_nce, positives, desc in CONDS:
        log(f"\n[{label}] {desc}  (tau={tau} alpha_ce={a_ce} beta_nce={b_nce} pos={positives})")
        ts = time.time()
        model = train_infonce_at(meta, Xtr, Ytr, tau, a_ce, b_nce, positives)
        dt = time.time() - ts
        met = eval_full(model, Xte, Yte, surrogate_pgd_advx=surrogate_pgd_advx)
        met["time_s"] = dt
        met["tau"] = tau
        met["alpha_ce"] = a_ce
        met["beta_nce"] = b_nce
        met["positives"] = positives
        nce_results[label] = met
        rows.append((label, met))
        log(f"    {label}: clean={met['clean_acc']:.4f} "
            f"FGSM_ASR={met['fgsm_asr']:.4f} "
            f"PGD_ASR={met['pgd_asr']:.4f} "
            f"transfer_ASR={met['transfer_asr']:.4f} "
            f"margin={met['mean_margin']:.4f} "
            f"worst_cls_PGD={met['worst_class_pgd_asr']:.4f} "
            f"({dt:.1f}s)")
        flush_file()

    # ---- MAIN TABLE ----
    log("\n" + "=" * 80)
    log("[1] MAIN TABLE")
    log("=" * 80)
    hdr = "{:<22} {:>9} {:>9} {:>9} {:>10} {:>8} {:>10} {:>8}".format(
        "condition", "clean", "FGSM", "PGD", "transfer", "margin", "worst_PGD", "time_s")
    log(hdr)
    log("-" * len(hdr))

    def row(name, m):
        log("{:<22} {:>9.4f} {:>9.4f} {:>9.4f} {:>10.4f} {:>8.3f} {:>10.4f} {:>8.1f}".format(
            name, m["clean_acc"], m["fgsm_asr"], m["pgd_asr"],
            m["transfer_asr"], m["mean_margin"], m["worst_class_pgd_asr"],
            m["time_s"]))

    for name, m in rows:
        row(name, m)
    log("")
    flush_file()

    # ---- PER-CLASS BREAKDOWN ----
    log("=" * 80)
    log("[2] PER-CLASS PGD ASR (10 classes)")
    log("=" * 80)
    cls_hdr = "{:<22} ".format("condition") + " ".join(f"c{c}" for c in range(N_CLASSES))
    log(cls_hdr)
    for name, m in rows:
        cells = " ".join(f"{v:.2f}" if not np.isnan(v) else " nan" for v in m["per_class_pgd_asr"])
        log(f"{name:<22} {cells}")
    log("")
    flush_file()

    # ---- DIAGNOSTIC: did we actually fix H348? ----
    log("=" * 80)
    log("[3] H348 BUG REPLICATION CHECK")
    log("=" * 80)
    nce_only = nce_results.get("C_nce_only_t0p5")
    if nce_only is not None:
        log(f"  InfoNCE-only (no CE on classifier): clean_acc={nce_only['clean_acc']:.4f}")
        if nce_only["clean_acc"] < 0.20:
            log("    --> CONFIRMED: classifier head untrained => random-readout collapse,")
            log("        reproducing the H348 symptom (clean ~ 1/n_classes).")
        else:
            log("    --> classifier somehow learned despite alpha_ce=0; investigate.")
    log("")

    # ---- VERDICT logic ----
    log("=" * 80)
    log("[4] VERDICT")
    log("=" * 80)

    # candidate winners: any joint InfoNCE+CE-AT condition (D..H, i.e. alpha_ce>0)
    joint = [(name, m) for name, m in nce_results.items() if m["alpha_ce"] > 0]
    if not joint:
        log("  no joint condition trained -- cannot conclude.")
    else:
        best = min(joint, key=lambda kv: kv[1]["pgd_asr"])
        bname, bmet = best
        d_pgd_vs_at = at_metrics["pgd_asr"] - bmet["pgd_asr"]            # +ve = better
        d_worst_vs_at = at_metrics["worst_class_pgd_asr"] - bmet["worst_class_pgd_asr"]
        d_clean_vs_at = bmet["clean_acc"] - at_metrics["clean_acc"]      # +ve = better
        d_clean_vs_h348 = bmet["clean_acc"] - (nce_only["clean_acc"] if nce_only else 0.0)

        # masking audit (cf. H391, H434)
        mask_signals = []
        for name, m in nce_results.items():
            if m["alpha_ce"] == 0:
                continue
            if m["pgd_asr"] < 0.5 and not np.isnan(m["transfer_asr"]):
                gap = m["transfer_asr"] - m["pgd_asr"]
                if gap > 0.03:
                    mask_signals.append(
                        f"{name}: transfer_ASR {m['transfer_asr']:.3f} > "
                        f"white-box PGD {m['pgd_asr']:.3f} (+{gap:.3f}) => POSSIBLE MASKING")
            if m["pgd_asr"] - m["fgsm_asr"] > 0.15 and m["fgsm_asr"] < 0.5 and m["pgd_asr"] < 0.7:
                mask_signals.append(
                    f"{name}: FGSM {m['fgsm_asr']:.3f} << PGD {m['pgd_asr']:.3f} => FGSM unreliable")

        log(f"  best joint condition: {bname}")
        log(f"  H348-fix delta (clean) : {nce_only['clean_acc'] if nce_only else float('nan'):.4f} -> "
            f"{bmet['clean_acc']:.4f} ({d_clean_vs_h348:+.4f})")
        log(f"  vs PGD-AT, PGD ASR     : {at_metrics['pgd_asr']:.4f} -> {bmet['pgd_asr']:.4f} "
            f"({-d_pgd_vs_at:+.4f}; +ve number = WORSE)")
        log(f"  vs PGD-AT, worst-cls   : {at_metrics['worst_class_pgd_asr']:.4f} -> "
            f"{bmet['worst_class_pgd_asr']:.4f} ({-d_worst_vs_at:+.4f})")
        log(f"  vs PGD-AT, clean acc   : {at_metrics['clean_acc']:.4f} -> "
            f"{bmet['clean_acc']:.4f} ({d_clean_vs_at:+.4f})")
        log("")
        log("  masking-signal audit:")
        if mask_signals:
            for s in mask_signals:
                log("    " + s)
        else:
            log("    (none -- transfer ASR <= white-box ASR + 0.03 and FGSM/PGD consistent)")
        log("")

        # one-line verdict
        clean_ok = d_clean_vs_at > -0.02
        h348_fixed = bmet["clean_acc"] >= 0.50
        if mask_signals:
            verdict = ("MASKING-FLAGGED: InfoNCE+CE-AT shows transfer/FGSM-PGD inconsistencies. "
                       "White-box PGD ASR not trustworthy here.")
        elif not h348_fixed:
            verdict = ("BUG NOT FIXED: best joint InfoNCE+CE-AT still has clean_acc < 0.50; "
                       "the H348 readout pathology persists -- design flaw.")
        elif d_pgd_vs_at > 0.02 and clean_ok:
            verdict = ("SUPPORTED: joint InfoNCE+CE-AT improves on PGD-AT by > 2pp mean PGD ASR "
                       "without clean-accuracy loss > 2pp. Class-conditioned contrastive coupling "
                       "of (clean, adv) pairs ADDS to standard adv training.")
        elif abs(d_pgd_vs_at) <= 0.02 and clean_ok and d_worst_vs_at > 0.05:
            verdict = ("PARTIAL: joint InfoNCE+CE-AT ties PGD-AT on mean PGD ASR but lowers "
                       "worst-class ASR by > 5pp -- gain is class-distributional, not aggregate.")
        elif abs(d_pgd_vs_at) <= 0.02 and clean_ok:
            verdict = ("FIXES H348 BUT TIES AT: joint InfoNCE+CE-AT restores classifier "
                       "(clean ~ AT level) and matches PGD-AT on PGD ASR -- but does not add "
                       "robustness over standard AT. Contrastive auxiliary is neutral here.")
        elif d_pgd_vs_at < -0.02:
            verdict = ("NOT SUPPORTED: joint InfoNCE+CE-AT is WORSE than PGD-AT by > 2pp PGD ASR. "
                       "Contrastive head competes with the classifier and hurts robustness.")
        else:
            verdict = ("INCONCLUSIVE: PGD ASR within +-2pp of PGD-AT but clean-acc trade is "
                       "unfavourable. Single seed; deltas inside campaign noise (M1).")

        log("  ONE-LINE VERDICT:")
        log("    " + verdict)
        log("")

    # ---- caveats (campaign standard) ----
    log("=" * 80)
    log("[5] CAVEATS")
    log("=" * 80)
    log("    - single seed (SEED=0); deltas < 0.02 PGD ASR inside campaign noise (M1).")
    log("    - N_TRAIN=6000 is sub-scale (M2); contrastive auxiliaries typically show")
    log("      LARGER gains with more data and longer schedules (Khosla 2020 used >>")
    log("      400 epochs at full ImageNet scale).")
    log("    - PGD-10 only; H391-style PGD-50 + restarts would be the genuine-robustness")
    log("      audit. Transfer ASR (surrogate=std baseline) is the cheap masking check.")
    log("    - in-batch negatives only -- no memory bank (He 2020 MoCo). At B=128 we have")
    log("      ~12.8 same-class negatives per anchor on average; small but Khosla-2020")
    log("      experiments show this regime still works.")
    log("    - condition H (positives=clean,clean) is a degenerate self-pair; if it ties")
    log("      condition E (clean,adv), the adversarial positive provides no extra signal.")

    log("")
    log(f"total time: {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
