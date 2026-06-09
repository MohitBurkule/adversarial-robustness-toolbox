"""
H435 - GAIRAT: Geometry-Aware Instance-Reweighted Adversarial Training.

Anchor paper: Zhang et al., "Geometry-aware Instance-reweighted Adversarial
Training", ICLR 2021 (oral) [zhang-2021-gairat].

Idea
----
Standard PGD-AT minimises the mean adversarial CE loss across the batch with
equal weight per sample. GAIRAT replaces the uniform mean with a per-sample
weight w_i = omega(kappa_i, K) that is non-increasing in kappa_i, where
kappa_i is the geometric distance to the decision boundary, approximated by
"the number of PGD steps required to FLIP sample i out of K total steps".
Samples that are easy to flip (kappa small) sit near the boundary and get
HEAVY weight; samples that resist the full K-step attack (kappa = K) get
LOW weight. We use the original tanh-shaped reweight (Eq. 6 in the paper):

    w_i = (1 + tanh(kappa_param * (1 - 2 * kappa_i / K))) / 2
    w_i = w_i / mean(w)             # batch-normalise so total loss-mass = 1

with kappa_param in {0.5, 1.0, 2.0} (the design knob; sharper -> harder
re-weighting). kappa_i is computed CHEAPLY during the same inner PGD: at
each inner step we check whether the current x_adv already fools the
classifier and store the FIRST step-index that did, otherwise kappa_i = K.

Critique (and how this script addresses it)
-------------------------------------------
The headline GAIRAT claim was undermined by Hitaj et al. 2021,
"Evaluating the Robustness of Geometry-Aware Instance-Reweighted Adversarial
Training" (arXiv:2103.01914): GAIRAT biases the model toward boundary-close
samples, leaving the OVERALL margin distribution lop-sided and inviting two
specific failure modes -

  (1) LOGIT-SCALING / AutoAttack-style attacks (esp. APGD-DLR and CW-margin
      losses) that the surrogate CE-PGD-used-during-training does not see;
  (2) Adaptive PGD that, instead of using a fixed step budget, recomputes
      its budget from the defender's K. (Liu et al. 2021 NeurIPS,
      "Probabilistic Margins for Instance Reweighting in Adversarial
      Training" - arXiv:2106.07904 - also show kappa is a discontinuous,
      path-dependent surrogate; their PM variant gains 0.5-13pp over GAIRAT
      under PGD/APGD/CW/AA, mainly on the CW/AA side.)

So the headline white-box PGD-10 number for any GAIRAT condition is
*expected* to be optimistic. We therefore evaluate every model under FIVE
attacks rather than one:

  - clean acc
  - FGSM (eps=0.1)
  - white-box PGD-10 (the campaign default)
  - adaptive PGD-30: longer budget, random start, and no kappa-style early
    stop - directly counters the "kappa never saw step > K" surrogate
  - CW-margin PGD-30: same as adaptive PGD but with the Carlini-Wagner
    margin loss (max(z_y - max_{j!=y} z_j, -kappa_cw)) instead of CE.
    This is the cheap stand-in for the AutoAttack DLR/CW failure mode that
    broke GAIRAT in Hitaj 2021.
  - transfer PGD-10 from a separately-trained PGD-AT baseline (M5 in the
    campaign gap map: most defences are never transfer-tested).

We also report per-class PGD ASR for every condition (M6 in the gap map -
nothing in the campaign has done this for an AT-family defence). If
GAIRAT only shaves mean ASR by concentrating gains on already-robust classes
(Trouser, Bag) while leaving Shirt/Pullover untouched, that asymmetry will
show up here.

Controls
--------
  C0: standard training (no AT) - "no defence" floor.
  C1: PGD-AT (uniform weights, otherwise identical inner attack) - the
      apples-to-apples baseline that GAIRAT must beat.
  C2: MART parallel (Wang et al. ICLR 2020) - the OTHER major reweighting
      family. MART weights the KL/BCE term by (1 - p_y(x_clean)) so
      misclassified naturals dominate. Same inner PGD, same K, same epochs.
      Lets us answer "is geometry-reweight better than misclass-reweight
      at matched compute?".
  C3a/b/c: GAIRAT with kappa_param in {0.5, 1.0, 2.0}.

Same outer optimiser and schedule everywhere. The C1 PGD-AT model is also
the source of the transfer attack (item M5 above).

Threat model
------------
Fashion-MNIST, eps=0.1, Linf, K=PGD_STEPS=10 during training. White-box and
adaptive evals at eps=0.1. Pure PyTorch.

Config (standard campaign): N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128,
SGD(mom=0.9, wd=5e-4), SEED=0. ASCII-only output, flushed per condition.
"""
import os
import sys
import time

import numpy as np
import torch
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
PGD_STEPS = 10            # inner-K used in training and the campaign-default eval
PGD_ALPHA = 0.01
ADAPT_STEPS = 30          # longer budget for adaptive eval
CW_KAPPA = 0.0            # CW-margin loss confidence floor
KAPPA_PARAMS = [0.5, 1.0, 2.0]
META = {"channels": 1, "size": 28, "n_classes": 10}
N_CLASSES = META["n_classes"]
F_MNIST_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle-boot"]

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h435_gairat_output.txt")


# ---- attacks (extras beyond common.pgd / common.fgsm) --------------------
def pgd_with_kappa(model, x, y, eps, steps, alpha):
    """PGD-K with random start, also returning kappa_i = the first step (1..K)
    at which sample i was misclassified (or K if never flipped during the
    inner attack). kappa is computed on the perturbed input AT THE END OF
    EACH STEP, so kappa in [1, K].
    """
    model.eval()  # so BN stats stay fixed during the inner attack
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1).detach()
    kappa = torch.full((x.size(0),), steps, dtype=torch.long, device=x.device)
    flipped_already = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    for s in range(1, steps + 1):
        xa.requires_grad_(True)
        logits = model(xa)
        loss = F.cross_entropy(logits, y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
        with torch.no_grad():
            pred = model(xa).argmax(1)
            now = (pred != y) & (~flipped_already)
            kappa[now] = s
            flipped_already = flipped_already | (pred != y)
    model.train()
    return xa.detach(), kappa.detach()


def pgd_cw_margin(model, x, y, eps, steps, alpha, cw_kappa=0.0):
    """PGD on the CW-margin loss instead of CE.

    Loss to MAXIMISE per sample: max(z_y - max_{j!=y} z_j, -cw_kappa)
    so the attacker tries to push z_y BELOW the max other logit. This is
    invariant to logit re-scaling (the failure mode Hitaj 2021 exploits
    against GAIRAT)."""
    model.eval()
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1).detach()
    n = x.size(0)
    idx = torch.arange(n, device=x.device)
    for _ in range(steps):
        xa.requires_grad_(True)
        logits = model(xa)
        z_y = logits[idx, y]
        masked = logits.clone()
        masked[idx, y] = float("-inf")
        z_other = masked.max(dim=1).values
        # we want to MINIMISE z_y - z_other to flip the prediction;
        # equivalently maximise -(z_y - z_other) = z_other - z_y
        per_sample = torch.clamp(z_y - z_other, min=-cw_kappa)
        loss = -per_sample.sum()           # maximise z_other - z_y
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    model.train()
    return xa.detach()


# ---- training routines ---------------------------------------------------
def _make_opt(model):
    p = [q for q in model.parameters() if q.requires_grad]
    return torch.optim.SGD(p, lr=LR, momentum=0.9, weight_decay=5e-4)


def _cosine(opt):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)


def train_standard(Xtr, Ytr, seed):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_opt(model)
    sch = _cosine(opt)
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
        sch.step()
    model.eval()
    return model


def train_pgd_at(Xtr, Ytr, seed):
    """Uniform-weight PGD-AT baseline (same inner attack as GAIRAT, just w=1)."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_opt(model)
    sch = _cosine(opt)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(xa), yb).backward()
            opt.step()
        sch.step()
    model.eval()
    return model


def gairat_weights(kappa, K, kappa_param):
    """Eq. 6 (paper): w = (1 + tanh(kappa_param * (1 - 2*kappa/K))) / 2.
    Then batch-normalise so weights average to 1 (so total loss-mass per
    batch matches the uniform case)."""
    k = kappa.float() / float(K)
    w = 0.5 * (1.0 + torch.tanh(kappa_param * (1.0 - 2.0 * k)))
    w = w / (w.mean() + 1e-12)
    return w


def train_gairat(Xtr, Ytr, seed, kappa_param):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_opt(model)
    sch = _cosine(opt)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xa, kappa = pgd_with_kappa(model, xb, yb, eps=EPS,
                                       steps=PGD_STEPS, alpha=PGD_ALPHA)
            w = gairat_weights(kappa, PGD_STEPS, kappa_param)
            model.train()
            opt.zero_grad()
            per = F.cross_entropy(model(xa), yb, reduction="none")
            (per * w).mean().backward()
            opt.step()
        sch.step()
    model.eval()
    return model


def train_mart(Xtr, Ytr, seed, beta=5.0):
    """MART (Wang et al. ICLR 2020) parallel:
        L = BCE-style margin term on adv + beta * (1 - p_y(clean)) * KL(p_adv || p_clean)
    The (1 - p_y(clean)) weight focuses regularisation on misclassified naturals.
    Standard hyperparameter beta=5 (paper default); kept fixed (not the design
    knob for this hypothesis).
    """
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_opt(model)
    sch = _cosine(opt)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            model.train()
            opt.zero_grad()
            logits_adv = model(xa)
            logits_cle = model(xb)
            p_adv = F.softmax(logits_adv, dim=1)
            p_cle = F.softmax(logits_cle, dim=1).detach()
            # BCE-style "boost the runner-up" term from MART paper
            true_p = p_adv[torch.arange(p_adv.size(0)), yb]
            tmp = p_adv.clone()
            tmp[torch.arange(tmp.size(0)), yb] = 0.0
            run_up_p = tmp.max(dim=1).values
            bce_adv = -torch.log(true_p.clamp_min(1e-12)) \
                      - torch.log((1.0 - run_up_p).clamp_min(1e-12))
            kl = F.kl_div(F.log_softmax(logits_adv, dim=1), p_cle,
                          reduction="none").sum(dim=1)
            misclass_w = (1.0 - p_cle[torch.arange(p_cle.size(0)), yb])
            loss = bce_adv.mean() + beta * (misclass_w * kl).mean()
            loss.backward()
            opt.step()
        sch.step()
    model.eval()
    return model


# ---- evaluation ---------------------------------------------------------
def per_class_pgd_asr(model, X, Y, eps, steps, alpha):
    """Returns array of shape (N_CLASSES,) of PGD-ASR over originally-correct
    samples within each class."""
    out = np.full(N_CLASSES, np.nan)
    for c in range(N_CLASSES):
        mask = (Y == c)
        if mask.sum().item() < 2:
            continue
        Xc, Yc = X[mask], Y[mask]
        res = C.attack_success(model, Xc, Yc, attack="pgd",
                               eps=eps, steps=steps)
        out[c] = res["asr"]
    return out


def transfer_pgd_asr(target_model, source_model, X, Y, eps, steps, alpha):
    """Craft PGD on source_model, evaluate flip-rate on target_model
    (over samples target_model originally classified correctly)."""
    target_model.eval()
    flips, corr = [], []
    bsz = 256
    for i in range(0, X.size(0), bsz):
        x, y = X[i:i + bsz], Y[i:i + bsz]
        with torch.no_grad():
            ok = (target_model(x).argmax(1) == y)
        xa = C.pgd(source_model, x, y, eps=eps, steps=steps, alpha=alpha)
        with torch.no_grad():
            bad = (target_model(xa).argmax(1) != y)
        flips.append(bad.cpu()); corr.append(ok.cpu())
    flips = torch.cat(flips).numpy(); corr = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() else float("nan")


def adapt_pgd_asr(model, X, Y, eps, steps, alpha):
    """Stronger adaptive PGD: longer budget, random start, CE loss; no kappa
    early-stop. Reports ASR over originally-correct samples."""
    flips, corr = [], []
    bsz = 256
    model.eval()
    for i in range(0, X.size(0), bsz):
        x, y = X[i:i + bsz], Y[i:i + bsz]
        with torch.no_grad():
            ok = (model(x).argmax(1) == y)
        xa = C.pgd(model, x, y, eps=eps, steps=steps, alpha=alpha,
                   random_start=True)
        with torch.no_grad():
            bad = (model(xa).argmax(1) != y)
        flips.append(bad.cpu()); corr.append(ok.cpu())
    flips = torch.cat(flips).numpy(); corr = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() else float("nan")


def cw_pgd_asr(model, X, Y, eps, steps, alpha):
    flips, corr = [], []
    bsz = 256
    model.eval()
    for i in range(0, X.size(0), bsz):
        x, y = X[i:i + bsz], Y[i:i + bsz]
        with torch.no_grad():
            ok = (model(x).argmax(1) == y)
        xa = pgd_cw_margin(model, x, y, eps=eps, steps=steps, alpha=alpha,
                           cw_kappa=CW_KAPPA)
        with torch.no_grad():
            bad = (model(xa).argmax(1) != y)
        flips.append(bad.cpu()); corr.append(ok.cpu())
    flips = torch.cat(flips).numpy(); corr = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() else float("nan")


def eval_all(model, Xte, Yte, source_model_for_transfer):
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)["asr"]
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS,
                          steps=PGD_STEPS)["asr"]
    ad = adapt_pgd_asr(model, Xte, Yte, eps=EPS, steps=ADAPT_STEPS,
                       alpha=PGD_ALPHA)
    cw = cw_pgd_asr(model, Xte, Yte, eps=EPS, steps=ADAPT_STEPS,
                    alpha=PGD_ALPHA)
    if source_model_for_transfer is not None and source_model_for_transfer is not model:
        tr = transfer_pgd_asr(model, source_model_for_transfer, Xte, Yte,
                              eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    else:
        tr = float("nan")
    pc = per_class_pgd_asr(model, Xte, Yte, eps=EPS, steps=PGD_STEPS,
                           alpha=PGD_ALPHA)
    mm = float(np.mean(C.margin(model, Xte, Yte)))
    return {"clean_acc": float(acc), "fgsm_asr": float(fg),
            "pgd_asr": float(pg), "adapt_pgd_asr": float(ad),
            "cw_pgd_asr": float(cw), "transfer_pgd_asr": float(tr),
            "mean_margin": mm, "per_class_pgd_asr": pc}


# ---- main ---------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    t0 = time.time()
    out("=" * 80)
    out("H435  GAIRAT geometry-aware instance-reweighted AT  (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} (inner K and white-box eval) "
        f"PGD_ALPHA={PGD_ALPHA}")
    out(f"        ADAPT_STEPS={ADAPT_STEPS} (adaptive + CW-margin eval)")
    out(f"        kappa_param sweep = {KAPPA_PARAMS}")
    out(f"        device = {C.DEVICE}")
    out("")
    out("Conditions: C0 std, C1 PGD-AT, C2 MART, C3a..C3c GAIRAT(kappa_param).")
    out("Eval per model: clean / FGSM / PGD-10 / adaptive-PGD-30 / "
        "CW-margin-PGD-30 / transfer-PGD-10 (source = C1 PGD-AT) / "
        "per-class PGD-10.")
    out("")

    # ---- data ----
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                        seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")
    flush()

    rows = []
    models = {}

    # ---- C0: standard training ----
    out("-" * 80)
    out("[C0] standard training (no AT)")
    out("-" * 80)
    t = time.time()
    m_std = train_standard(Xtr, Ytr, SEED)
    out(f"    trained in {time.time()-t:.1f}s")
    models["C0_std"] = m_std

    # ---- C1: PGD-AT (uniform-weight) -- needed as transfer source ----
    out("")
    out("-" * 80)
    out("[C1] PGD-AT (uniform weights, K=10)")
    out("-" * 80)
    t = time.time()
    m_at = train_pgd_at(Xtr, Ytr, SEED)
    out(f"    trained in {time.time()-t:.1f}s")
    models["C1_pgdat"] = m_at

    source = m_at  # transfer attacks always crafted on this model

    # eval C0 and C1 now that we have the transfer source
    for tag in ["C0_std", "C1_pgdat"]:
        res = eval_all(models[tag], Xte, Yte, source_model_for_transfer=source)
        rows.append({"cond": tag, **res})
        out(f"    [{tag}] clean={res['clean_acc']:.4f}  "
            f"FGSM={res['fgsm_asr']:.4f}  PGD-10={res['pgd_asr']:.4f}  "
            f"adapt-PGD-30={res['adapt_pgd_asr']:.4f}  "
            f"CW-PGD-30={res['cw_pgd_asr']:.4f}  "
            f"transfer={res['transfer_pgd_asr']:.4f}  "
            f"margin={res['mean_margin']:.3f}")
        flush()

    # ---- C2: MART parallel ----
    out("")
    out("-" * 80)
    out("[C2] MART (Wang 2020) parallel control - misclass-weighted")
    out("-" * 80)
    t = time.time()
    m_mart = train_mart(Xtr, Ytr, SEED, beta=5.0)
    out(f"    trained in {time.time()-t:.1f}s")
    models["C2_mart"] = m_mart
    res = eval_all(m_mart, Xte, Yte, source_model_for_transfer=source)
    rows.append({"cond": "C2_mart", **res})
    out(f"    [C2_mart] clean={res['clean_acc']:.4f}  "
        f"FGSM={res['fgsm_asr']:.4f}  PGD-10={res['pgd_asr']:.4f}  "
        f"adapt-PGD-30={res['adapt_pgd_asr']:.4f}  "
        f"CW-PGD-30={res['cw_pgd_asr']:.4f}  "
        f"transfer={res['transfer_pgd_asr']:.4f}  "
        f"margin={res['mean_margin']:.3f}")
    flush()

    # ---- C3a/b/c: GAIRAT sweep ----
    for kp in KAPPA_PARAMS:
        tag = f"C3_gairat_kp{kp:g}"
        out("")
        out("-" * 80)
        out(f"[{tag}] GAIRAT (kappa_param={kp})")
        out("-" * 80)
        t = time.time()
        m = train_gairat(Xtr, Ytr, SEED, kappa_param=kp)
        out(f"    trained in {time.time()-t:.1f}s")
        models[tag] = m
        res = eval_all(m, Xte, Yte, source_model_for_transfer=source)
        rows.append({"cond": tag, **res})
        out(f"    [{tag}] clean={res['clean_acc']:.4f}  "
            f"FGSM={res['fgsm_asr']:.4f}  PGD-10={res['pgd_asr']:.4f}  "
            f"adapt-PGD-30={res['adapt_pgd_asr']:.4f}  "
            f"CW-PGD-30={res['cw_pgd_asr']:.4f}  "
            f"transfer={res['transfer_pgd_asr']:.4f}  "
            f"margin={res['mean_margin']:.3f}")
        flush()

    # ---- MAIN TABLE ----
    out("")
    out("=" * 80)
    out("MAIN TABLE")
    out("=" * 80)
    hdr = ("{:<22} {:>9} {:>8} {:>8} {:>11} {:>11} {:>10} {:>8}".format(
        "condition", "clean", "FGSM", "PGD10", "adaptPGD30", "CW-PGD30",
        "transfer", "margin"))
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<22} {:>9.4f} {:>8.4f} {:>8.4f} {:>11.4f} {:>11.4f} {:>10.4f} "
            "{:>8.3f}".format(
                r["cond"], r["clean_acc"], r["fgsm_asr"], r["pgd_asr"],
                r["adapt_pgd_asr"], r["cw_pgd_asr"], r["transfer_pgd_asr"],
                r["mean_margin"]))
    out("-" * len(hdr))
    out("")
    out("All ASR figures are over originally-correctly-classified samples.")
    out("CW-PGD30 uses logit-margin loss (Hitaj-2021-style logit-scale audit).")
    out("transfer = PGD-10 crafted on C1 PGD-AT, evaluated on target row.")

    # ---- PER-CLASS PGD ASR TABLE ----
    out("")
    out("=" * 80)
    out("PER-CLASS PGD-10 ASR  (Fashion-MNIST classes)")
    out("=" * 80)
    out("{:<22} ".format("condition") +
        " ".join(f"{c[:5]:>6}" for c in F_MNIST_CLASSES) +
        f"  {'worst':>6}  {'spread':>6}")
    out("-" * 110)
    for r in rows:
        pc = r["per_class_pgd_asr"]
        worst = float(np.nanmax(pc))
        spread = float(np.nanmax(pc) - np.nanmin(pc))
        out("{:<22} ".format(r["cond"]) +
            " ".join(f"{v:>6.3f}" for v in pc) +
            f"  {worst:>6.3f}  {spread:>6.3f}")

    # ---- VERDICT ----
    out("")
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)
    base = next(r for r in rows if r["cond"] == "C1_pgdat")
    std = next(r for r in rows if r["cond"] == "C0_std")
    gair_rows = [r for r in rows if r["cond"].startswith("C3_gairat")]
    mart = next(r for r in rows if r["cond"] == "C2_mart")

    out(f"  baseline std PGD ASR     = {std['pgd_asr']:.4f}  "
        f"clean={std['clean_acc']:.4f}")
    out(f"  PGD-AT (C1) PGD ASR      = {base['pgd_asr']:.4f}  "
        f"clean={base['clean_acc']:.4f}  CW={base['cw_pgd_asr']:.4f}  "
        f"adapt={base['adapt_pgd_asr']:.4f}")
    out(f"  MART (C2) PGD ASR        = {mart['pgd_asr']:.4f}  "
        f"clean={mart['clean_acc']:.4f}  CW={mart['cw_pgd_asr']:.4f}  "
        f"adapt={mart['adapt_pgd_asr']:.4f}")
    for r in gair_rows:
        out(f"  {r['cond']:<22}  PGD ASR={r['pgd_asr']:.4f}  "
            f"clean={r['clean_acc']:.4f}  CW={r['cw_pgd_asr']:.4f}  "
            f"adapt={r['adapt_pgd_asr']:.4f}")

    # define "best GAIRAT" by white-box PGD-10 (the headline metric)
    best_g = min(gair_rows, key=lambda r: r["pgd_asr"])
    d_pgd = base["pgd_asr"] - best_g["pgd_asr"]            # +ve => GAIRAT helps
    d_cw = base["cw_pgd_asr"] - best_g["cw_pgd_asr"]       # +ve => GAIRAT helps under CW
    d_adapt = base["adapt_pgd_asr"] - best_g["adapt_pgd_asr"]
    d_clean = best_g["clean_acc"] - base["clean_acc"]
    d_pgd_vs_mart = mart["pgd_asr"] - best_g["pgd_asr"]    # +ve => GAIRAT beats MART
    gairat_pgd_dip = (d_pgd > 0.01)
    gairat_holds_under_cw = (d_cw >= -0.01)                # not WORSE than C1 by >1pp
    gairat_holds_under_adapt = (d_adapt >= -0.01)
    out("")
    out(f"  best GAIRAT condition         = {best_g['cond']}")
    out(f"  d(PGD-10  vs C1 PGD-AT)       = {d_pgd:+.4f}  "
        f"(positive => GAIRAT improves the headline)")
    out(f"  d(CW-PGD-30 vs C1)            = {d_cw:+.4f}  "
        f"(negative => Hitaj-2021-style logit-scale audit worsens GAIRAT)")
    out(f"  d(adapt-PGD-30 vs C1)         = {d_adapt:+.4f}")
    out(f"  d(clean acc vs C1)            = {d_clean:+.4f}")
    out(f"  d(PGD-10 vs MART, +ve = GAIRAT wins) = {d_pgd_vs_mart:+.4f}")

    # one-line verdict logic
    if gairat_pgd_dip and gairat_holds_under_cw and gairat_holds_under_adapt:
        verdict = ("SUPPORTED: GAIRAT lowers PGD-10 ASR vs PGD-AT and the gain "
                   "survives BOTH the CW-margin and the longer-budget adaptive "
                   "audit. Geometry-reweight is a genuine improvement at this "
                   "scale.")
    elif gairat_pgd_dip and not gairat_holds_under_cw:
        verdict = ("PARTIAL / GRADIENT-MASKING: GAIRAT lowers PGD-10 ASR but "
                   "the CW-margin attack erases the gain - matches the Hitaj "
                   "2021 finding that geometric reweighting biases the CE "
                   "surface and is fragile to logit-scale / margin attacks.")
    elif gairat_pgd_dip and not gairat_holds_under_adapt:
        verdict = ("PARTIAL: GAIRAT lowers PGD-10 ASR but loses to a longer-"
                   "budget adaptive PGD - inner-K=10 surrogate over-fits the "
                   "campaign-default attack.")
    elif not gairat_pgd_dip:
        verdict = ("NOT SUPPORTED: GAIRAT does not beat uniform-weight PGD-AT "
                   "even on the white-box PGD-10 headline at this scale (6k / "
                   "10 epochs). Geometric reweight gives nothing here.")
    else:
        verdict = "INCONCLUSIVE."
    out("")
    out("  ONE-LINE VERDICT: " + verdict)

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
