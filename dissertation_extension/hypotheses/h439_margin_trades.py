"""
H439 - Margin-TRADES: replace KL surrogate in TRADES with a margin-style surrogate.

Hypothesis (seed): "Margin-TRADES - gap G3 - extends `zhang-2019-trades`
- margin-based KL surrogate". The idea is to swap the KL(p(x) || p(x_adv))
robustness regulariser in TRADES for a margin-style surrogate on the adv
example (correct-class logit/prob minus best-other logit/prob), keeping the
clean CE term untouched.

Critique (written before running):
  Margin-TRADES is suspiciously close to "CW-loss inside TRADES". The
  Carlini-Wagner f6 surrogate (Carlini & Wagner 2017) is exactly
  max(z_other) - z_y; using it as the inner-max objective and as the outer
  regulariser turns TRADES into CW-AT with a clean-CE consistency term.
  Plausible outcomes:
    (a) ties PGD-AT (and PGD-AT-CW); the campaign's bottleneck is the
        6k-sample/10-epoch budget (CAMPAIGN_GAP_MAP M2), not the AT loss.
    (b) collapses if beta or PGD-on-margin produces vanishing/saturating
        signal (logit-margin can swing wildly when |z| grows; softmax-margin
        saturates near +-1).
    (c) shows gradient masking - margin-PGD inner loop may not actually find
        Linf-eps adversaries that the standard CE-PGD evaluator at test time
        will easily find. Transfer-attack check below.
  Prior art (>=2 papers picked beyond zhang-2019-trades):
    - Carlini & Wagner 2017 ("Towards Evaluating the Robustness of Neural
      Networks") - the f6 margin loss this script borrows.
    - Ding et al. 2018 ("MMA Training: Direct Input Space Margin
      Maximization") - direct margin maximisation as an AT objective.
    - Yu et al. 2022 ("Boosting Adversarial Robustness From the Perspective
      of Effective Margin Regularization") - margin-style regularisers
      stacked on AT.
    - Pang et al. 2020 ("Boosting Adversarial Training with Hypersphere
      Embedding") - softmax-margin variants for AT.

Design (controls cover the obvious "this is just CW-AT" failure modes):
  C0  Clean baseline (no adv loss).
  C1  PGD-AT (standard, CE inner+outer)         - reference robust baseline.
  C2  PGD-AT-CW (CW/margin loss inner+outer)    - the "is Margin-TRADES
      just CW-AT in disguise?" control.
  C3  TRADES-KL beta=6 (h304 replication)       - TRADES baseline.
  C4  Margin-TRADES (softmax-margin surrogate)  - beta sweep {1, 5, 10}.
  C5  Margin-TRADES (logit-margin surrogate)    - beta sweep {1, 5, 10}.
  Plus a transfer-masking check: for each margin-TRADES model, run a
  PGD-CE attack crafted against the PGD-AT (C1) model and transferred to
  the margin-TRADES model. If transfer ASR >> white-box ASR -> masking.

Output: results/fashion_mnist/h439_margin_trades_output.txt
Verdict policy:
  SUPPORTED       if best Margin-TRADES PGD_ASR <= PGD-AT - 0.02 AND no masking
                  (transfer ASR not >5pp higher than white-box).
  TIED            if within +-0.02 of PGD-AT, clean >= PGD-AT - 0.02, no masking.
  MASKING         if white-box PGD_ASR << transfer PGD_ASR (>0.05 gap).
  NOT-SUPPORTED   otherwise.

Config: standard campaign budget (N_TRAIN=6000, EPOCHS=10, LR=0.05,
BATCH=128, SGD mom=0.9 wd=5e-4, SEED=0, EPS=0.1, PGD_STEPS=10, ALPHA=0.01).
Pure torch. No ART.
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
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
BETAS_TRADES = [1.0, 5.0, 10.0]
BETA_TRADES_KL = 6.0     # standard TRADES baseline (matches h304's beta=6)
CW_KAPPA = 0.0           # CW confidence margin; 0 = "just push it over"

META = {"channels": 1, "size": 28, "n_classes": 10}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _opt(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)


def _other_max(logits, y):
    """Return (best_other_logit, correct_logit). Masks out the correct class."""
    n, k = logits.shape
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[torch.arange(n, device=logits.device), y] = True
    correct = logits[mask]
    other = logits.masked_fill(mask, float("-inf")).max(dim=1).values
    return other, correct


def cw_margin_loss(logits, y, kappa=0.0):
    """CW f6 surrogate, batch-mean. Larger value => attacker happier.

    f6(x,y) = max( max_{i!=y} z_i - z_y, -kappa )

    Maximising f6 over x pushes the classifier toward misclassification.
    Used here both as inner-PGD objective and as outer-loss term (with sign
    chosen so the *trainer* minimises misclassification).
    """
    other, correct = _other_max(logits, y)
    f = other - correct
    f = torch.clamp(f, min=-kappa)
    return f.mean()


def softmax_margin_loss(logits, y, kappa=0.0):
    """Softmax-margin surrogate: p_other_max - p_y. Bounded in [-1, 1].

    Less sensitive to logit scale than logit-margin (no exploding-norm
    pathology), but saturates near +-1 when the model is very confident.
    """
    p = F.softmax(logits, dim=1)
    other, correct = _other_max(p, y)
    f = other - correct
    f = torch.clamp(f, min=-kappa)
    return f.mean()


# --------------------------------------------------------------------------
# inner-PGD variants (all return adv tensor; model state untouched)
# --------------------------------------------------------------------------
def pgd_ce(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA, rs=True):
    return C.pgd(model, x, y, eps=eps, steps=steps, alpha=alpha, random_start=rs)


def pgd_cw(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA, rs=True,
           kappa=CW_KAPPA):
    """PGD ascending the CW f6 margin (logit-margin)."""
    x0 = x.clone().detach()
    xa = x0.clone()
    if rs:
        xa = (xa + torch.empty_like(xa).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        logits = model(xa)
        loss = cw_margin_loss(logits, y, kappa=kappa)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def pgd_softmax_margin(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA,
                        rs=True, kappa=CW_KAPPA):
    """PGD ascending the softmax-margin surrogate."""
    x0 = x.clone().detach()
    xa = x0.clone()
    if rs:
        xa = (xa + torch.empty_like(xa).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        logits = model(xa)
        loss = softmax_margin_loss(logits, y, kappa=kappa)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def pgd_kl(model, x, steps=PGD_STEPS, eps=EPS, alpha=PGD_ALPHA):
    """TRADES inner step: maximise KL(p(x_adv) || p(x))."""
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)
    x0 = x.detach()
    xa = (x0 + 0.001 * torch.randn_like(x0)).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        kl = F.kl_div(F.log_softmax(model(xa), dim=1), p_clean,
                      reduction='batchmean')
        g, = torch.autograd.grad(kl, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


# --------------------------------------------------------------------------
# training routines
# --------------------------------------------------------------------------
def train_clean(Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
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


def train_pgd_at(Xtr, Ytr, inner="ce"):
    """Standard PGD-AT. inner in {'ce', 'cw'}."""
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            if inner == "ce":
                xa = pgd_ce(model, xb, yb)
            elif inner == "cw":
                xa = pgd_cw(model, xb, yb)
            else:
                raise ValueError(inner)
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(xa), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_trades_kl(Xtr, Ytr, beta=BETA_TRADES_KL):
    """Standard TRADES (KL surrogate). Replicates h304's beta=6 setting."""
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = pgd_kl(model, xb)
            model.train()
            opt.zero_grad()
            out_clean = model(xb)
            out_adv = model(xa)
            loss_ce = F.cross_entropy(out_clean, yb)
            loss_kl = F.kl_div(F.log_softmax(out_adv, dim=1),
                               F.softmax(out_clean, dim=1),
                               reduction='batchmean')
            loss = loss_ce + beta * loss_kl
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_margin_trades(Xtr, Ytr, beta, surrogate):
    """Margin-TRADES:
        loss = CE(model(x), y) + beta * margin_surrogate(model(x_adv), y)
    where x_adv is found by PGD ascending the SAME margin surrogate.
    surrogate in {'softmax', 'logit'}.
    """
    assert surrogate in ("softmax", "logit")
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            if surrogate == "softmax":
                xa = pgd_softmax_margin(model, xb, yb)
            else:
                xa = pgd_cw(model, xb, yb)
            model.train()
            opt.zero_grad()
            out_clean = model(xb)
            out_adv = model(xa)
            loss_ce = F.cross_entropy(out_clean, yb)
            if surrogate == "softmax":
                loss_marg = softmax_margin_loss(out_adv, yb)
            else:
                loss_marg = cw_margin_loss(out_adv, yb)
            loss = loss_ce + beta * loss_marg
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def eval_whitebox(model, Xte, Yte):
    """clean acc, FGSM ASR, PGD-CE ASR, PGD-CW ASR, mean margin."""
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS,
                          steps=PGD_STEPS)
    # CW-PGD evaluation
    cw_flips = []
    corr_mask = []
    for i in range(0, Xte.size(0), 256):
        xb = Xte[i:i + 256]
        yb = Yte[i:i + 256]
        with torch.no_grad():
            correct = (model(xb).argmax(1) == yb)
        xa = pgd_cw(model, xb, yb)
        with torch.no_grad():
            flipped = (model(xa).argmax(1) != yb)
        cw_flips.append(flipped.cpu())
        corr_mask.append(correct.cpu())
    cw_flips = torch.cat(cw_flips).numpy()
    corr = torch.cat(corr_mask).numpy().astype(bool)
    cw_asr = float(cw_flips[corr].mean()) if corr.sum() > 0 else float("nan")
    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))
    return dict(clean_acc=float(acc), fgsm_asr=float(fg["asr"]),
                pgd_asr=float(pg["asr"]), cw_asr=cw_asr,
                mean_margin=mean_margin)


def transfer_asr(source_model, target_model, Xte, Yte):
    """Craft PGD-CE adv examples on source_model, eval on target_model.

    Returns ASR restricted to examples target_model originally classified
    correctly (matches attack_success convention)."""
    target_model.eval()
    flips = []
    corr = []
    for i in range(0, Xte.size(0), 256):
        xb = Xte[i:i + 256]
        yb = Yte[i:i + 256]
        with torch.no_grad():
            correct = (target_model(xb).argmax(1) == yb)
        xa = pgd_ce(source_model, xb, yb)
        with torch.no_grad():
            flipped = (target_model(xa).argmax(1) != yb)
        flips.append(flipped.cpu())
        corr.append(correct.cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr).numpy().astype(bool)
    asr = float(flips[corr].mean()) if corr.sum() > 0 else float("nan")
    return asr


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h439_margin_trades_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H439  Margin-TRADES (margin-style surrogate in place of TRADES KL)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        Margin-TRADES betas = {BETAS_TRADES}; TRADES-KL beta = {BETA_TRADES_KL}")
    out(f"        device = {C.DEVICE}")
    out("")

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                        seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    # ---- containers ----
    rows = []        # list of dict result rows
    models = {}      # keep PGD-AT around for transfer-attack check

    # ---- C0: clean baseline ----
    out("-" * 80)
    out("[C0] CLEAN baseline (no adv loss)")
    out("-" * 80)
    t = time.time()
    m = train_clean(Xtr, Ytr)
    r = eval_whitebox(m, Xte, Yte)
    r.update(cond="C0 clean", beta=0.0, time_s=round(time.time() - t, 1))
    out(f"    clean={r['clean_acc']:.4f}  FGSM={r['fgsm_asr']:.4f}  "
        f"PGD={r['pgd_asr']:.4f}  CW={r['cw_asr']:.4f}  "
        f"margin={r['mean_margin']:.3f}  t={r['time_s']}s")
    rows.append(r)
    models["C0_clean"] = m
    flush_file()

    # ---- C1: PGD-AT (CE inner+outer) -- reference robust baseline ----
    out("")
    out("-" * 80)
    out("[C1] PGD-AT  (CE inner+outer, the canonical robust baseline)")
    out("-" * 80)
    t = time.time()
    m = train_pgd_at(Xtr, Ytr, inner="ce")
    r = eval_whitebox(m, Xte, Yte)
    r.update(cond="C1 PGD-AT-CE", beta=0.0, time_s=round(time.time() - t, 1))
    out(f"    clean={r['clean_acc']:.4f}  FGSM={r['fgsm_asr']:.4f}  "
        f"PGD={r['pgd_asr']:.4f}  CW={r['cw_asr']:.4f}  "
        f"margin={r['mean_margin']:.3f}  t={r['time_s']}s")
    rows.append(r)
    models["C1_pgd_at"] = m
    flush_file()

    # ---- C2: PGD-AT-CW ----
    out("")
    out("-" * 80)
    out("[C2] PGD-AT-CW  (CW/margin inner+outer) - 'is Margin-TRADES just this?'")
    out("-" * 80)
    t = time.time()
    m = train_pgd_at(Xtr, Ytr, inner="cw")
    r = eval_whitebox(m, Xte, Yte)
    r.update(cond="C2 PGD-AT-CW", beta=0.0, time_s=round(time.time() - t, 1))
    out(f"    clean={r['clean_acc']:.4f}  FGSM={r['fgsm_asr']:.4f}  "
        f"PGD={r['pgd_asr']:.4f}  CW={r['cw_asr']:.4f}  "
        f"margin={r['mean_margin']:.3f}  t={r['time_s']}s")
    rows.append(r)
    models["C2_pgd_at_cw"] = m
    flush_file()

    # ---- C3: TRADES-KL (replication of h304 at beta=6) ----
    out("")
    out("-" * 80)
    out(f"[C3] TRADES-KL beta={BETA_TRADES_KL}  (h304 replication)")
    out("-" * 80)
    t = time.time()
    m = train_trades_kl(Xtr, Ytr, beta=BETA_TRADES_KL)
    r = eval_whitebox(m, Xte, Yte)
    r.update(cond=f"C3 TRADES-KL b={BETA_TRADES_KL}",
             beta=BETA_TRADES_KL, time_s=round(time.time() - t, 1))
    out(f"    clean={r['clean_acc']:.4f}  FGSM={r['fgsm_asr']:.4f}  "
        f"PGD={r['pgd_asr']:.4f}  CW={r['cw_asr']:.4f}  "
        f"margin={r['mean_margin']:.3f}  t={r['time_s']}s")
    rows.append(r)
    models["C3_trades_kl"] = m
    flush_file()

    # ---- C4: Margin-TRADES, softmax-margin, beta sweep ----
    out("")
    out("-" * 80)
    out(f"[C4] Margin-TRADES (SOFTMAX-margin surrogate) beta in {BETAS_TRADES}")
    out("-" * 80)
    for beta in BETAS_TRADES:
        t = time.time()
        m = train_margin_trades(Xtr, Ytr, beta=beta, surrogate="softmax")
        r = eval_whitebox(m, Xte, Yte)
        cond = f"C4 mTRADES-sm b={beta}"
        r.update(cond=cond, beta=beta, time_s=round(time.time() - t, 1))
        out(f"    beta={beta}: clean={r['clean_acc']:.4f}  FGSM={r['fgsm_asr']:.4f}  "
            f"PGD={r['pgd_asr']:.4f}  CW={r['cw_asr']:.4f}  "
            f"margin={r['mean_margin']:.3f}  t={r['time_s']}s")
        rows.append(r)
        models[cond] = m
        flush_file()

    # ---- C5: Margin-TRADES, logit-margin (CW), beta sweep ----
    out("")
    out("-" * 80)
    out(f"[C5] Margin-TRADES (LOGIT-margin / CW surrogate) beta in {BETAS_TRADES}")
    out("-" * 80)
    for beta in BETAS_TRADES:
        t = time.time()
        m = train_margin_trades(Xtr, Ytr, beta=beta, surrogate="logit")
        r = eval_whitebox(m, Xte, Yte)
        cond = f"C5 mTRADES-lo b={beta}"
        r.update(cond=cond, beta=beta, time_s=round(time.time() - t, 1))
        out(f"    beta={beta}: clean={r['clean_acc']:.4f}  FGSM={r['fgsm_asr']:.4f}  "
            f"PGD={r['pgd_asr']:.4f}  CW={r['cw_asr']:.4f}  "
            f"margin={r['mean_margin']:.3f}  t={r['time_s']}s")
        rows.append(r)
        models[cond] = m
        flush_file()

    # ---- TRANSFER-ATTACK MASKING CHECK ----
    out("")
    out("=" * 80)
    out("[6] TRANSFER-ATTACK MASKING CHECK")
    out("=" * 80)
    out("Craft PGD-CE adv examples on C1 PGD-AT-CE and evaluate them on every")
    out("margin-TRADES model. If transfer ASR >> white-box PGD ASR -> masking.")
    out("")
    source = models["C1_pgd_at"]
    transfer_rows = []
    targets = [k for k in models if k.startswith("C4") or k.startswith("C5")]
    for tgt in targets:
        wb_pgd = next(r["pgd_asr"] for r in rows if r["cond"] == tgt)
        t_asr = transfer_asr(source, models[tgt], Xte, Yte)
        gap = t_asr - wb_pgd
        flag = "MASKING?" if gap > 0.05 else "ok"
        out(f"    {tgt:>26s}  whitebox PGD={wb_pgd:.4f}  "
            f"transfer PGD={t_asr:.4f}  gap={gap:+.4f}  [{flag}]")
        transfer_rows.append(dict(cond=tgt, whitebox=wb_pgd, transfer=t_asr,
                                   gap=gap, flag=flag))
    flush_file()

    # ---- MAIN TABLE ----
    out("")
    out("=" * 80)
    out("[7] MAIN TABLE")
    out("=" * 80)
    hdr = ("{:<26} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "condition", "beta", "clean", "FGSM", "PGD", "CW", "margin"))
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<26} {:>6.1f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.3f}".format(
            r["cond"], r["beta"], r["clean_acc"], r["fgsm_asr"], r["pgd_asr"],
            r["cw_asr"], r["mean_margin"]))
    out("-" * len(hdr))

    # ---- VERDICT ----
    out("")
    out("=" * 80)
    out("[8] VERDICT")
    out("=" * 80)
    pgd_at = next(r for r in rows if r["cond"] == "C1 PGD-AT-CE")
    mtrades_rows = [r for r in rows if r["cond"].startswith("C4") or r["cond"].startswith("C5")]
    best = min(mtrades_rows, key=lambda r: r["pgd_asr"])
    delta_pgd = best["pgd_asr"] - pgd_at["pgd_asr"]
    delta_clean = best["clean_acc"] - pgd_at["clean_acc"]
    masked_any = any(tr["flag"] == "MASKING?" for tr in transfer_rows)
    best_tr = next(tr for tr in transfer_rows if tr["cond"] == best["cond"])

    out(f"  best Margin-TRADES: {best['cond']}  "
        f"PGD={best['pgd_asr']:.4f}  CW={best['cw_asr']:.4f}  clean={best['clean_acc']:.4f}")
    out(f"  PGD-AT-CE reference: PGD={pgd_at['pgd_asr']:.4f}  "
        f"CW={pgd_at['cw_asr']:.4f}  clean={pgd_at['clean_acc']:.4f}")
    out(f"  delta(PGD vs PGD-AT) = {delta_pgd:+.4f}  "
        f"delta(clean vs PGD-AT) = {delta_clean:+.4f}")
    out(f"  best Margin-TRADES transfer-gap = {best_tr['gap']:+.4f}  "
        f"any-masking-flag = {masked_any}")

    if masked_any and (best_tr["gap"] > 0.05):
        verdict = ("MASKING: Margin-TRADES looks robust under white-box PGD-CE but "
                   "gives way under transfer attack -- the margin loss is masking "
                   "gradients, not buying real robustness.")
    elif delta_pgd <= -0.02 and delta_clean >= -0.02 and not masked_any:
        verdict = ("SUPPORTED: Margin-TRADES beats PGD-AT-CE on PGD ASR by >=0.02 "
                   "with no clean-accuracy regression and no transfer masking.")
    elif abs(delta_pgd) < 0.02 and delta_clean >= -0.02 and not masked_any:
        verdict = ("TIED: Margin-TRADES is statistically indistinguishable from "
                   "PGD-AT-CE at this budget. Consistent with CAMPAIGN_GAP_MAP M2 "
                   "(the AT loss is not the bottleneck at 6k/10ep).")
    else:
        verdict = ("NOT-SUPPORTED: Margin-TRADES does not improve on PGD-AT-CE "
                   "(or costs clean acc / shows masking).")
    out("")
    out("  ONE-LINE VERDICT: " + verdict)

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
