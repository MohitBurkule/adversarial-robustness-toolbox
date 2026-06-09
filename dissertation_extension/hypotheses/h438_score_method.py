"""
H438 - SCORE method (Pang et al., ICML 2022: "Robustness and Accuracy Could Be
Reconcilable by (Proper) Definition") on Fashion-MNIST.

Gap filled: G2 (defence families barely touched).
Anchor: pang-2022-score.
Knob: alpha (trade-off weight on the robust-equivariance term).

----------------------------------------------------------------------------
CRITIQUE (what we are actually testing)

TRADES (Zhang 2019) minimises
    L_TRADES = CE(f(x), y) + beta * KL( p(x) || p(x_adv) )
with x_adv chosen to MAXIMISE that KL. Pang et al. argue this imposes a local
*invariance* prior (push p(x_adv) to *match* p(x)) which conflicts with the
true label whenever p(x) is mis-calibrated -> baked-in accuracy/robustness
trade-off. SCORE replaces KL with a symmetric distance (squared-L2 on the
softmax probability vectors), giving local *equivariance*:

    L_SCORE = CE(f(x), y) + alpha * || softmax(f(x_adv)) - softmax(f(x)) ||_2^2

Inner step: PGD that MAXIMISES the same squared-L2 distance (not KL).

Confounds the seed asks us to verify:
  * At alpha -> 0  SCORE collapses to standard ERM (baseline).
  * At very large alpha SCORE collapses to "force adv = clean", which is the
    same failure mode TRADES has at beta -> infty.
  * The DISTINGUISHING claim is in the MIDDLE alpha range: the squared-L2
    surrogate should buy more clean accuracy at matched PGD ASR than KL.
  * Masking risk: a model whose softmax is uniform satisfies SCORE for free;
    we check this with (a) a transfer attack from a standard surrogate model
    and (b) FGSM ASR (transfer-style cheap probe).

----------------------------------------------------------------------------
EXTRA PAPERS (from WebSearch, beyond the anchor)

* Cui et al. 2024 "Rethinking Invariance Regularization in Adversarial
  Training to Improve Robustness-Accuracy Trade-off" (arXiv 2402.14648) -
  follow-up critique: invariance-style regularisers are *strictly worse*
  than equivariance under mis-calibration; supports SCORE's framing but
  argues the empirical gap is small at small data scales.
* Yang et al. 2025 "Bridging Symmetry and Robustness: On the Role of
  Equivariance in Enhancing Adversarial Robustness" (arXiv 2510.16171) -
  generalises SCORE's local-equivariance argument to group-equivariant
  layers; relevant background for why a squared-L2 KL replacement is not
  just a numerical trick.
* Pang's own SCORE repo (github.com/P2333/SCORE) confirms the
  squared-L2-on-softmax + same-distance PGD inner step that we implement
  below.

----------------------------------------------------------------------------
CONTROLS (per seed brief)

  1. Standard ERM baseline (alpha=0; same backbone, same epochs).
  2. PGD-AT baseline (Madry-style; outer min on CE, inner max on CE).
  3. TRADES baseline (extends H304, beta=6, KL surrogate).
  4. SCORE at alpha in {0.1, 1.0, 6.0} - low/mid/high knob sweep.
  5. Transfer-attack masking check: attack the standard ERM surrogate
     (condition 1) with PGD and apply those adversarials to every other
     model; report transfer ASR.
  6. Per-class PGD ASR (10 Fashion-MNIST classes) on every condition.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9, wd=5e-4),
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01. CNN width=32 (default).
ASCII output only.
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
PGD_STEPS = 10
PGD_ALPHA = 0.01
ALPHAS = [0.1, 1.0, 6.0]
TRADES_BETA = 6.0

META = {"channels": 1, "size": 28, "n_classes": 10}
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankleboot"]


# ---- shared helpers ------------------------------------------------------
def _make_optimizer_sgd(model, lr):
    """SGD mom=0.9 wd=5e-4 (campaign config)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def pgd_on_ce(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """Standard Madry PGD that maximises CE; identical to common.pgd
    but with an explicit local copy for clarity / determinism here."""
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


def pgd_on_kl(model, x, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """TRADES inner step: PGD that maximises KL( softmax(f(x)) || softmax(f(x_adv)) )."""
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)
    x0 = x.detach()
    xa = x0 + 0.001 * torch.randn_like(x0)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        logp_adv = F.log_softmax(model(xa), dim=1)
        kl = F.kl_div(logp_adv, p_clean, reduction="batchmean")
        g, = torch.autograd.grad(kl, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    model.train()
    return xa.detach()


def pgd_on_l2_softmax(model, x, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """SCORE inner step: PGD that maximises ||softmax(f(x_adv)) - softmax(f(x))||_2^2."""
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)
    x0 = x.detach()
    xa = x0 + 0.001 * torch.randn_like(x0)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        p_adv = F.softmax(model(xa), dim=1)
        dist = ((p_adv - p_clean) ** 2).sum(dim=1).mean()
        g, = torch.autograd.grad(dist, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    model.train()
    return xa.detach()


# ---- training functions -------------------------------------------------
def train_standard(Xtr, Ytr, seed=SEED):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
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


def train_pgd_at(Xtr, Ytr, seed=SEED):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            x_adv = pgd_on_ce(model, xb, yb)
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(x_adv), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_trades(Xtr, Ytr, beta=TRADES_BETA, seed=SEED):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            x_adv = pgd_on_kl(model, xb)
            model.train()
            opt.zero_grad()
            out_clean = model(xb)
            out_adv = model(x_adv)
            loss_ce = F.cross_entropy(out_clean, yb)
            loss_kl = F.kl_div(
                F.log_softmax(out_adv, dim=1),
                F.softmax(out_clean, dim=1),
                reduction="batchmean")
            loss = loss_ce + beta * loss_kl
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_score(Xtr, Ytr, alpha, seed=SEED):
    """SCORE: CE on clean + alpha * ||softmax(adv) - softmax(clean)||_2^2.
    Inner PGD maximises the same squared-L2 distance."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            x_adv = pgd_on_l2_softmax(model, xb)
            model.train()
            opt.zero_grad()
            out_clean = model(xb)
            out_adv = model(x_adv)
            p_clean = F.softmax(out_clean, dim=1)
            p_adv = F.softmax(out_adv, dim=1)
            loss_ce = F.cross_entropy(out_clean, yb)
            loss_eq = ((p_adv - p_clean) ** 2).sum(dim=1).mean()
            loss = loss_ce + alpha * loss_eq
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- evaluation helpers --------------------------------------------------
@torch.no_grad()
def _acc(model, X, Y):
    _, a = C.logits_and_acc(model, X, Y)
    return float(a)


def eval_white_box(model, Xte, Yte):
    """Clean acc, FGSM ASR, white-box PGD ASR."""
    for p in model.parameters():
        p.requires_grad_(True)
    clean_acc = _acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return clean_acc, fg["asr"], pg["asr"]


def transfer_asr(model, X_adv_surrogate, Yte, batch=256):
    """ASR (over originally-correct samples) when feeding pre-computed
    surrogate-generated adversarials to `model`."""
    model.eval()
    n = Yte.size(0)
    correct = torch.empty(n, dtype=torch.bool)
    flipped = torch.empty(n, dtype=torch.bool)
    # need clean predictions to define "originally correct"
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(i + batch, n)
            xa = X_adv_surrogate[i:j]
            # originally-correct from clean (we approximate with the model's
            # clean predictions matching the label)
            # actually: we need clean inputs here; surrogate sample passes
            # original eval set images NOT surrogate-perturbed.
            raise RuntimeError("internal: use transfer_asr_pair instead")


def transfer_asr_pair(model, X_clean, X_adv, Y, batch=256):
    """Standard transfer ASR: fraction of originally-correct samples whose
    model prediction flips when given X_adv (built on a SURROGATE)."""
    model.eval()
    flips, corr = [], []
    with torch.no_grad():
        for i in range(0, Y.size(0), batch):
            xc = X_clean[i:i + batch]
            xa = X_adv[i:i + batch]
            yb = Y[i:i + batch]
            pc = model(xc).argmax(1)
            pa = model(xa).argmax(1)
            corr.append((pc == yb).cpu())
            flips.append((pa != yb).cpu())
    corr = torch.cat(corr).numpy().astype(bool)
    flips = torch.cat(flips).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() > 0 else float("nan")


def per_class_pgd_asr(model, Xte, Yte, n_classes=10):
    """Per-class PGD ASR (over originally-correct samples in that class)."""
    for p in model.parameters():
        p.requires_grad_(True)
    out = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    flips = out["flips"].astype(bool)
    corr = out["correct"].astype(bool)
    y = Yte.cpu().numpy()
    asrs = []
    for c in range(n_classes):
        mc = (y == c) & corr
        if mc.sum() == 0:
            asrs.append(float("nan"))
        else:
            asrs.append(float(flips[mc].mean()))
    return asrs


# ---- main ----------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h438_score_method_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H438  SCORE method (Pang et al. ICML 2022) on Fashion-MNIST")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        SCORE alphas = {ALPHAS}   TRADES beta = {TRADES_BETA}")
    out(f"        device = {C.DEVICE}")
    out("")
    out("hypothesis: SCORE's squared-L2-on-softmax surrogate reconciles")
    out("            robustness and accuracy better than TRADES (KL) at")
    out("            matched PGD budget, and does not collapse to TRADES")
    out("            at the alpha extremes tested here.")
    out("")
    flush_file()

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    flush_file()

    # ---- container ----
    rows = []
    surrogate_pgd_adv = None  # filled after we train the standard baseline

    def run_one(name, train_fn, kw):
        nonlocal surrogate_pgd_adv
        t_a = time.time()
        out("")
        out("-" * 80)
        out(f"training {name} ...")
        model = train_fn(Xtr, Ytr, **kw)
        t_b = time.time()
        clean, fgsm, pgd_asr = eval_white_box(model, Xte, Yte)
        per_class = per_class_pgd_asr(model, Xte, Yte)
        worst_cls = int(np.nanargmax(per_class))
        worst_asr = float(per_class[worst_cls])
        # generate surrogate adversarials once (from the first model = standard ERM)
        if surrogate_pgd_adv is None and name == "standard":
            out("  caching surrogate (standard) PGD adversarials for transfer attack...")
            for p in model.parameters():
                p.requires_grad_(True)
            surrogate_pgd_adv = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS,
                                      alpha=PGD_ALPHA)
        # transfer ASR (skip for the surrogate itself; defined as white-box pgd)
        if name == "standard":
            trans_asr = pgd_asr
        else:
            trans_asr = transfer_asr_pair(model, Xte, surrogate_pgd_adv, Yte)
        row = {"name": name, "clean": clean, "fgsm": fgsm,
               "pgd": pgd_asr, "transfer": trans_asr,
               "worst_cls": worst_cls, "worst_asr": worst_asr,
               "per_class": per_class,
               "train_s": round(t_b - t_a, 1)}
        rows.append(row)
        out(f"  RESULT {name}: clean={clean:.4f} FGSM_ASR={fgsm:.4f} "
            f"PGD_ASR={pgd_asr:.4f} transfer_ASR={trans_asr:.4f} "
            f"worst_cls={CLASS_NAMES[worst_cls]}({worst_asr:.3f}) "
            f"(train={row['train_s']}s)")
        pc_str = " ".join(f"{a:.2f}" for a in per_class)
        out(f"           per-class PGD ASR: {pc_str}")
        out(f"           elapsed total {time.time() - t0:.0f}s")
        flush_file()
        return model

    # 1. standard ERM
    run_one("standard", train_standard, {})

    # 2. PGD-AT baseline
    run_one("pgd_at", train_pgd_at, {})

    # 3. TRADES baseline (beta=6)
    run_one(f"trades_b{int(TRADES_BETA)}", train_trades, {"beta": TRADES_BETA})

    # 4. SCORE alpha sweep
    for a in ALPHAS:
        run_one(f"score_a{a}", train_score, {"alpha": a})

    # ---- MAIN TABLE ----
    out("")
    out("=" * 80)
    out("[MAIN TABLE]  white-box and transfer ASR (lower = more robust)")
    out("=" * 80)
    hdr = ("{:<14} {:>9} {:>9} {:>9} {:>10} {:>14} {:>9}".format(
        "condition", "clean", "FGSM_ASR", "PGD_ASR", "transfer", "worst_class",
        "train_s"))
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<14} {:>9.4f} {:>9.4f} {:>9.4f} {:>10.4f} {:>14} {:>9}".format(
            r["name"], r["clean"], r["fgsm"], r["pgd"], r["transfer"],
            f"{CLASS_NAMES[r['worst_cls']]}({r['worst_asr']:.2f})",
            r["train_s"]))
    out("-" * len(hdr))

    out("")
    out("[PER-CLASS PGD ASR]  one row per condition, 10 Fashion-MNIST classes")
    out("classes: " + " ".join(CLASS_NAMES))
    for r in rows:
        pc_str = " ".join(f"{a:.3f}" for a in r["per_class"])
        out(f"  {r['name']:<14} {pc_str}")

    # ---- VERDICT ----
    out("")
    out("=" * 80)
    out("[VERDICT]")
    out("=" * 80)

    def find(name_prefix):
        for r in rows:
            if r["name"].startswith(name_prefix):
                return r
        return None

    std = find("standard")
    at = find("pgd_at")
    tr = find("trades")
    sc_rows = [r for r in rows if r["name"].startswith("score_")]
    sc_best = min(sc_rows, key=lambda r: r["pgd"])
    sc_low = find("score_a0.1")
    sc_high = find("score_a6.0")

    out(f"  baseline PGD ASR (standard ERM)        : {std['pgd']:.4f}")
    out(f"  PGD-AT PGD ASR                          : {at['pgd']:.4f}")
    out(f"  TRADES(beta={int(TRADES_BETA)}) PGD ASR              : {tr['pgd']:.4f}")
    for r in sc_rows:
        out(f"  SCORE({r['name']}) PGD ASR             : {r['pgd']:.4f}   "
            f"clean={r['clean']:.4f}  transfer={r['transfer']:.4f}")
    out("")
    out(f"  best SCORE alpha = {sc_best['name']}  PGD_ASR={sc_best['pgd']:.4f}  "
        f"clean={sc_best['clean']:.4f}")

    # masking sanity:  if white-box PGD >> transfer PGD => obfuscated gradient
    mask_gap = sc_best["transfer"] - sc_best["pgd"]
    masking = mask_gap > 0.10
    out(f"  masking gap (transfer - white_box PGD)  : {mask_gap:+.4f} "
        f"({'SUSPECT MASKING' if masking else 'no masking signal'})")

    # collapse to extremes:
    near_erm = abs(sc_low["pgd"] - std["pgd"]) < 0.05 and abs(sc_low["clean"] - std["clean"]) < 0.03
    near_trades = abs(sc_high["pgd"] - tr["pgd"]) < 0.05 and abs(sc_high["clean"] - tr["clean"]) < 0.05
    out(f"  collapse check: alpha=0.1 ~ ERM ?       : "
        f"{'YES' if near_erm else 'NO'}  "
        f"(d_pgd={sc_low['pgd']-std['pgd']:+.3f}, d_clean={sc_low['clean']-std['clean']:+.3f})")
    out(f"  collapse check: alpha=6.0 ~ TRADES ?    : "
        f"{'YES' if near_trades else 'NO'}  "
        f"(d_pgd={sc_high['pgd']-tr['pgd']:+.3f}, d_clean={sc_high['clean']-tr['clean']:+.3f})")

    # reconciliation: SCORE beats TRADES if (clean higher OR pgd lower) by >= 0.01
    score_beats_trades = (
        (sc_best["clean"] >= tr["clean"] + 0.01 and sc_best["pgd"] <= tr["pgd"] + 0.01)
        or
        (sc_best["pgd"] <= tr["pgd"] - 0.01 and sc_best["clean"] >= tr["clean"] - 0.01))
    score_beats_at = (
        (sc_best["clean"] >= at["clean"] + 0.01 and sc_best["pgd"] <= at["pgd"] + 0.01)
        or
        (sc_best["pgd"] <= at["pgd"] - 0.01 and sc_best["clean"] >= at["clean"] - 0.01))

    out("")
    if masking:
        verdict = ("INCONCLUSIVE: SCORE shows a transfer-vs-whitebox gap > 0.10, "
                   "consistent with gradient masking. White-box PGD numbers are "
                   "not trustworthy.")
    elif score_beats_trades and score_beats_at:
        verdict = ("SUPPORTED: SCORE's local-equivariance surrogate Pareto-improves "
                   "over both PGD-AT and TRADES at this scale (no masking).")
    elif score_beats_trades:
        verdict = ("PARTIAL: SCORE beats TRADES on the accuracy/robustness frontier "
                   "but ties or trails PGD-AT.")
    elif near_trades and not score_beats_trades:
        verdict = ("NOT SUPPORTED (collapse): at large alpha SCORE behaves like "
                   "TRADES and does not give an extra robustness/accuracy gain.")
    else:
        verdict = ("NOT SUPPORTED: SCORE does not Pareto-improve over the TRADES "
                   "or PGD-AT baselines at N_TRAIN=6000 / 10 epochs.")
    out("  ONE-LINE VERDICT: " + verdict)

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
