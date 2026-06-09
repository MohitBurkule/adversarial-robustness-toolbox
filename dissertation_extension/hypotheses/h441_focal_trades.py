"""
H441 - Focal-TRADES: focal weighting inside the TRADES KL term.

Seed (CAMPAIGN_GAP_MAP.md S5):
    h441: Focal-TRADES - gap G3 - focal weighting inside TRADES KL term -
          gamma in {0,1,2}.

Idea
----
TRADES (Zhang et al. 2019, H304/H346) trains with:
    L = CE(f(x), y) + beta * KL(f(x) || f(x_adv))
where x_adv = argmax KL(f(x)||f(x+delta)) inside an Linf eps-ball.

Focal loss (Lin et al. 2017) reweights per-sample CE by w = (1 - p_t)^gamma
so easy (high-confidence-correct) samples are down-weighted and boundary
samples keep their gradient. We port that idea INSIDE the TRADES KL term:

    w_i  = (1 - p_correct_i)^gamma     (computed on CLEAN logits, no grad)
    L    = CE(f(x), y)
         + beta * mean_i [ w_i * KL( f(x_i) || f(x_adv_i) ) ]

Rationale: TRADES already up-weights samples whose clean prediction is unstable
under perturbation, but it does so symmetrically across confidence levels.
Focal weighting concentrates the KL pressure on samples the model is *currently
unsure about*, which are also the H252 / H404 "vulnerable" samples (low margin).

Critique (must keep up front; do not bury)
------------------------------------------
1. Confidence-compression masking. Focal-like reweighting on clean p_correct
   pushes the model toward softer, more uniform softmax outputs on hard
   samples. That can SHRINK the input-space loss gradient and mimic the
   classical gradient-masking failure mode (Athalye 2018 obfuscated grads,
   Tramer 2020 adaptive attacks). H325 (confidence reg) and H429 (focal CE
   without AT) BOTH showed in this campaign that confidence-flattening can
   give *fake* PGD ASR gains. We therefore MUST run an adaptive / transfer
   probe and check FGSM-vs-PGD gap and margin-CDF for signs of masking.
2. Confound with focal CE itself. Any gain at gamma>0 might come from the
   focal weighting moving the *outer* CE objective (because the model sees
   a different gradient field), not from the KL reweight. We therefore run a
   focal-CE-AT ablation (focal weighting on the CE term only, no KL) to
   isolate the focal-x-KL interaction.
3. Single seed, N=6000 (M1/M2). Gains < 0.02 PGD ASR are inside campaign noise.
   The gamma=0 condition IS the TRADES baseline and any gamma>0 condition
   must beat it by >= 0.02 to count as PARTIAL, and survive the transfer
   probe to count as YES.
4. MART (Wang et al. 2020) is the published prior art for "misclassification
   weighting of TRADES KL"; Focal-TRADES is a strictly softer, continuous
   variant. We do NOT claim novelty over MART, only test the specific
   focal-weighting form at our fixed scale.

Extra prior art (>=2 papers, via WebSearch)
-------------------------------------------
- `wang-2020-mart` Wang et al., "Improving Adversarial Robustness Requires
  Revisiting Misclassified Examples", ICLR 2020. The canonical prior art:
  weights the KL term by (1 - p_correct) and adds BCE on misclassified
  examples. Focal-TRADES with gamma=1 is essentially MART's KL weight
  without the BCE-on-misclassified piece -> sanity anchor.
- `liu-2022-bregman` Liu et al., "Lower Difficulty and Better Robustness: A
  Bregman Divergence Perspective for Adversarial Training", arXiv 2208.12511.
  Argues TRADES' KL is harder to optimise than alternative Bregman
  surrogates -> per-sample reweighting could plausibly help, but can also
  smooth the loss in ways that mimic masking.
- `cui-2025-gkl` "Generalized Kullback-Leibler Divergence Loss" (arXiv
  2503.08038, surveyed via WebSearch). Decoupled / weighted KL inside
  TRADES is a 2024-2025 active topic; uses class-wise weighting that
  generalises focal-style sample weighting.

Controls / ablations (must-haves)
---------------------------------
A. TRADES baseline (gamma=0, focal weighting disabled): the canonical
   reference. This is *not* a CE baseline - it is TRADES with beta=BETA.
B. Standard-CE baseline (no AT, no KL): so the "TRADES helps" baseline gap
   is visible alongside the gamma sweep.
C. PGD-AT baseline (Madry): so we can see whether TRADES variants ever beat
   PGD-AT at this scale (H304/H346 typically find them comparable).
D. Focal-CE-AT ablation: focal weighting on the OUTER CE term only, no KL.
   Isolates the focal-vs-KL contribution.
E. Gamma sweep {0, 1, 2, 5}: 0 is the TRADES baseline; 1 is MART-like; 2
   is Lin et al.'s default; 5 is the H429 most-aggressive setting.
F. Transfer-attack probe: take adversarial examples crafted on the
   standard-CE baseline and feed them to each Focal-TRADES model. If
   Focal-TRADES claims white-box robustness but TRANSFER ASR is much
   higher, that is the gradient-masking signature (Tramer 2020, H391).
G. Per-class breakdown: report 10-class PGD ASR vector so we can see
   whether any gain is concentrated in one class (H252-style) instead of
   broadly improving robustness.
H. Margin CDF anchor (low-tail): report 5th-percentile margin so that
   confidence compression (which lowers mean margin) is visible if it
   happens.

Design
------
- Pure-torch, no ART; reuses campaign.common (build_model, pgd, fgsm,
  attack_success, logits_and_acc, margin).
- All conditions train a fresh SmallCNN (width=32) from scratch on the same
  6000-sample subset (N_TRAIN), same SEED=0.
- TRADES inner attack: 10-step PGD maximising KL(f(x)||f(x+delta)) with
  Linf eps=0.1, alpha=0.01, random start (matches H346).
- All hard-coded knobs match the standard project config.

Config (project standard)
-------------------------
DS=fashion_mnist, N_TRAIN=6000, N_EVAL=2000, EPOCHS=10, LR=0.05, BATCH=128,
SGD(mom=0.9, wd=5e-4), SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01,
BETA=6 (TRADES coupling, H346 mid-grid), GAMMAS=[0, 1, 2, 5].

Output
------
results/fashion_mnist/h441_focal_trades_output.txt
Flushed after every condition so partial progress is durable.

Do NOT execute this script from the main session; delegate to a background
agent per project workflow.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config (project standard) --------------------------------------------
DS        = "fashion_mnist"
N_TRAIN   = 6000
N_EVAL    = 2000
EPOCHS    = 10
LR        = 0.05
BATCH     = 128
SEED      = 0
EPS       = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
BETA      = 6                 # TRADES KL coupling (H346 mid-grid)
GAMMAS    = [0, 1, 2, 5]      # 0 = pure TRADES; >=1 = focal weighting on KL

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h441_focal_trades_output.txt",
)


# ---- losses / inner attacks -----------------------------------------------
def pgd_kl(model, x, eps, steps, alpha):
    """TRADES inner attack: PGD maximising KL(f(x) || f(x+delta))."""
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1).detach()
    with torch.no_grad():
        p_clean = F.softmax(model(x0), dim=1)
    for _ in range(steps):
        xa.requires_grad_(True)
        log_p_adv = F.log_softmax(model(xa), dim=1)
        kl = F.kl_div(log_p_adv, p_clean, reduction="batchmean")
        g, = torch.autograd.grad(kl, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def focal_weight_from_clean(model, x, y, gamma):
    """w_i = (1 - p_correct_i)^gamma computed on clean logits, no grad."""
    if gamma == 0:
        return None
    with torch.no_grad():
        p = F.softmax(model(x), dim=1)
        p_t = p.gather(1, y.unsqueeze(1)).squeeze(1).clamp(0.0, 1.0)
        w = (1.0 - p_t).pow(gamma)
    return w.detach()


def focal_ce(logits, targets, gamma):
    """Standard focal CE: -(1-p_t)^gamma * log p_t (per-sample mean)."""
    if gamma == 0:
        return F.cross_entropy(logits, targets)
    log_p = F.log_softmax(logits, dim=1)
    log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = log_p_t.exp()
    w = (1.0 - p_t).pow(gamma)
    return -(w * log_p_t).mean()


# ---- training routines ----------------------------------------------------
def _new_model_and_opt(seed):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=5e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return model, opt, sched


def train_std(Xtr, Ytr, seed=SEED):
    """Standard CE training (no AT, no KL)."""
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
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


def train_pgd_at(Xtr, Ytr, seed=SEED):
    """Madry PGD-AT baseline: CE on PGD-attacked inputs."""
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            xa = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(xa), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_focal_trades(Xtr, Ytr, gamma, seed=SEED):
    """Focal-TRADES: CE on clean + beta * mean( w_i * KL(f(x)||f(x_adv)) ).
       gamma=0 recovers vanilla TRADES (H346)."""
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            # inner attack maximises KL (in eval mode for clean BN stats)
            model.eval()
            x_adv = pgd_kl(model, xb, EPS, PGD_STEPS, PGD_ALPHA)
            model.train()

            opt.zero_grad()
            out_clean = model(xb)
            loss_ce = F.cross_entropy(out_clean, yb)

            # focal weight from CLEAN p_correct (no grad through weights)
            w = focal_weight_from_clean(model, xb, yb, gamma)

            log_p_clean = F.log_softmax(out_clean, dim=1).detach()
            p_clean = log_p_clean.exp()
            log_p_adv = F.log_softmax(model(x_adv), dim=1)
            # per-sample KL(p_clean || p_adv) = sum p_clean * (log p_clean - log p_adv)
            per_sample_kl = (p_clean * (log_p_clean - log_p_adv)).sum(dim=1)
            if w is None:
                kl_loss = per_sample_kl.mean()
            else:
                # normalise weights to mean 1 so BETA scale is comparable to gamma=0
                w_n = w / (w.mean() + 1e-8)
                kl_loss = (w_n * per_sample_kl).mean()

            loss = loss_ce + BETA * kl_loss
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_focal_ce_at(Xtr, Ytr, gamma, seed=SEED):
    """Focal-CE-AT ablation: focal CE on PGD-attacked inputs, NO KL term.
       Isolates the focal-vs-KL contribution by stripping TRADES' KL out."""
    model, opt, sched = _new_model_and_opt(seed)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            xa = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            model.train()
            opt.zero_grad()
            focal_ce(model(xa), yb, gamma).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- eval helpers ---------------------------------------------------------
def eval_robustness(model, Xte, Yte):
    """clean acc, FGSM ASR, PGD ASR, mean+5th-pct margin."""
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    fgsm = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)["asr"]
    pgd  = C.attack_success(model, Xte, Yte, attack="pgd",
                            eps=EPS, steps=PGD_STEPS)["asr"]
    mg = C.margin(model, Xte, Yte)
    return {
        "clean_acc": float(clean_acc),
        "fgsm_asr":  float(fgsm),
        "pgd_asr":   float(pgd),
        "mg_mean":   float(np.mean(mg)),
        "mg_p05":    float(np.percentile(mg, 5)),
    }


def transfer_asr_pair(victim_model, Xte, Yte, X_adv):
    """Restrict to victim-clean-correct samples, compute fraction flipped under
       a transferred adversarial set X_adv (same indexing as Xte)."""
    victim_model.eval()
    with torch.no_grad():
        clean_pred = victim_model(Xte).argmax(1)
        corr = (clean_pred == Yte)
        adv_pred = victim_model(X_adv).argmax(1)
        flipped = (adv_pred != Yte) & corr
    denom = int(corr.sum().item())
    if denom == 0:
        return float("nan")
    return float(flipped.sum().item()) / denom


def per_class_pgd_asr(model, Xte, Yte, n_classes=10):
    """Per-class PGD ASR over all test samples (no correct-mask conditioning;
       reports class-conditional flip rate of label y -> something != y)."""
    model.eval()
    asr = [float("nan")] * n_classes
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    with torch.no_grad():
        pred = model(Xpgd).argmax(1)
    for c in range(n_classes):
        m = (Yte == c)
        if int(m.sum().item()) == 0:
            continue
        asr[c] = float(((pred != Yte) & m).sum().item() / int(m.sum().item()))
    return asr


def build_baseline_adv(model_std, Xte, Yte):
    """PGD adv examples crafted on the standard-CE baseline -> transfer source."""
    return C.pgd(model_std, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)


# ---- main -----------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H441  Focal-TRADES: focal weighting inside TRADES KL term (Fashion-MNIST)")
    out("=" * 80)
    out("seed: h441 - gap G3 - focal weighting inside TRADES KL term - gamma in {0,1,2,5}")
    out("anchors: zhang-2019-trades, lin-2017-focal, wang-2020-mart,")
    out("         liu-2022-bregman (arXiv 2208.12511),")
    out("         cui-2025-gkl     (arXiv 2503.08038)")
    out("")
    out(f"config: DS={DS} N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} "
        f"LR={LR} BATCH={BATCH}")
    out(f"        SGD(mom=0.9,wd=5e-4) SEED={SEED} EPS={EPS} PGD_STEPS={PGD_STEPS} "
        f"PGD_ALPHA={PGD_ALPHA}")
    out(f"        BETA={BETA}  GAMMAS={GAMMAS}  device={C.DEVICE}")
    out("")

    # ---- data ------------------------------------------------------------
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    # ---- (B) standard-CE baseline + transfer source ----------------------
    out("[1/6] training STD baseline (CE, no AT)...")
    m_std = train_std(Xtr, Ytr, SEED)
    r_std = eval_robustness(m_std, Xte, Yte)
    out(f"   STD : clean={r_std['clean_acc']:.4f} fgsm={r_std['fgsm_asr']:.4f} "
        f"pgd={r_std['pgd_asr']:.4f} mg_mean={r_std['mg_mean']:.4f} "
        f"mg_p05={r_std['mg_p05']:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    out("   building transfer-attack source (PGD-10 on STD model)...")
    X_adv_std = build_baseline_adv(m_std, Xte, Yte)
    out(f"   transfer source ready: X_adv_std shape={tuple(X_adv_std.shape)}")
    out("")
    flush_file()

    # ---- (C) PGD-AT reference -------------------------------------------
    out("[2/6] training PGD-AT baseline (Madry)...")
    m_at = train_pgd_at(Xtr, Ytr, SEED)
    r_at = eval_robustness(m_at, Xte, Yte)
    t_at = transfer_asr_pair(m_at, Xte, Yte, X_adv_std)
    out(f"   PGD-AT: clean={r_at['clean_acc']:.4f} fgsm={r_at['fgsm_asr']:.4f} "
        f"pgd={r_at['pgd_asr']:.4f} transfer={t_at:.4f} "
        f"mg_mean={r_at['mg_mean']:.4f} mg_p05={r_at['mg_p05']:.4f}  "
        f"({time.time()-t0:.0f}s)")
    out("")
    flush_file()

    # ---- (A,E) Focal-TRADES gamma sweep (gamma=0 IS TRADES baseline) ----
    rows = []
    rows.append({
        "name": "STD (no AT)",
        "kind": "ref",
        **r_std,
        "transfer": float("nan"),
    })
    rows.append({
        "name": "PGD-AT (Madry)",
        "kind": "ref",
        **r_at,
        "transfer": t_at,
    })

    out("[3/6] Focal-TRADES gamma sweep (gamma=0 = vanilla TRADES baseline)")
    per_class_dump = {}
    for gi, gamma in enumerate(GAMMAS):
        tag = f"Focal-TRADES gamma={gamma}" + ("  [=TRADES baseline]" if gamma == 0 else "")
        out(f"   [{gi+1}/{len(GAMMAS)}] training {tag} ...")
        m = train_focal_trades(Xtr, Ytr, gamma, SEED)
        r = eval_robustness(m, Xte, Yte)
        tr = transfer_asr_pair(m, Xte, Yte, X_adv_std)
        per_class_dump[f"FT_g{gamma}"] = per_class_pgd_asr(m, Xte, Yte)
        out(f"      clean={r['clean_acc']:.4f} fgsm={r['fgsm_asr']:.4f} "
            f"pgd={r['pgd_asr']:.4f} transfer={tr:.4f} "
            f"mg_mean={r['mg_mean']:.4f} mg_p05={r['mg_p05']:.4f}  "
            f"({time.time()-t0:.0f}s)")
        rows.append({
            "name": f"Focal-TRADES g={gamma}",
            "kind": "ft",
            "gamma": gamma,
            **r,
            "transfer": tr,
        })
        flush_file()
    out("")

    # ---- (D) Focal-CE-AT ablation (no KL) -------------------------------
    out("[4/6] Focal-CE-AT ablation (no KL term) - isolates focal contribution")
    for gi, gamma in enumerate(GAMMAS):
        tag = f"Focal-CE-AT gamma={gamma}" + ("  [=PGD-AT baseline]" if gamma == 0 else "")
        out(f"   [{gi+1}/{len(GAMMAS)}] training {tag} ...")
        m = train_focal_ce_at(Xtr, Ytr, gamma, SEED)
        r = eval_robustness(m, Xte, Yte)
        tr = transfer_asr_pair(m, Xte, Yte, X_adv_std)
        per_class_dump[f"FCEAT_g{gamma}"] = per_class_pgd_asr(m, Xte, Yte)
        out(f"      clean={r['clean_acc']:.4f} fgsm={r['fgsm_asr']:.4f} "
            f"pgd={r['pgd_asr']:.4f} transfer={tr:.4f} "
            f"mg_mean={r['mg_mean']:.4f} mg_p05={r['mg_p05']:.4f}  "
            f"({time.time()-t0:.0f}s)")
        rows.append({
            "name": f"Focal-CE-AT g={gamma}",
            "kind": "fceat",
            "gamma": gamma,
            **r,
            "transfer": tr,
        })
        flush_file()
    out("")

    # ---- (5) MAIN TABLE -------------------------------------------------
    out("=" * 80)
    out("[5/6] MAIN TABLE")
    out("=" * 80)
    hdr = "{:<26} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "condition", "clean", "FGSM", "PGD", "transfer", "mg_mean", "mg_p05")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        tr = r["transfer"]
        tr_s = "{:>9.4f}".format(tr) if not (isinstance(tr, float) and np.isnan(tr)) else "{:>9}".format("nan")
        out("{:<26} {:>9.4f} {:>9.4f} {:>9.4f} {} {:>9.4f} {:>9.4f}".format(
            r["name"], r["clean_acc"], r["fgsm_asr"], r["pgd_asr"], tr_s,
            r["mg_mean"], r["mg_p05"]))
    out("-" * len(hdr))
    out("")

    # ---- per-class breakdown (G) ----------------------------------------
    out("Per-class PGD ASR (classes 0..9):")
    pc_hdr = "{:<22} ".format("model") + " ".join("{:>5}".format(c) for c in range(10))
    out(pc_hdr)
    out("-" * len(pc_hdr))
    for name, arr in per_class_dump.items():
        out("{:<22} ".format(name) + " ".join(
            "{:>5.2f}".format(v) if not np.isnan(v) else "  nan" for v in arr))
    out("")

    # ---- (6) MASKING / TRANSFER PROBE & VERDICT -------------------------
    out("=" * 80)
    out("[6/6] MASKING PROBE  +  VERDICT")
    out("=" * 80)
    # TRADES baseline = Focal-TRADES gamma=0
    trades_row = next(r for r in rows if r.get("gamma", -1) == 0 and r["kind"] == "ft")
    ft_rows = [r for r in rows if r["kind"] == "ft" and r.get("gamma", 0) > 0]
    fceat_rows = [r for r in rows if r["kind"] == "fceat"]
    best_ft = min(ft_rows, key=lambda r: r["pgd_asr"]) if ft_rows else None

    out(f"TRADES baseline (gamma=0): PGD={trades_row['pgd_asr']:.4f}  "
        f"transfer={trades_row['transfer']:.4f}  "
        f"clean={trades_row['clean_acc']:.4f}")
    if best_ft is not None:
        d_pgd = best_ft["pgd_asr"] - trades_row["pgd_asr"]
        d_tr  = best_ft["transfer"] - trades_row["transfer"]
        d_acc = best_ft["clean_acc"] - trades_row["clean_acc"]
        gap_fgsm_pgd = best_ft["fgsm_asr"] - best_ft["pgd_asr"]
        out(f"best Focal-TRADES gamma>0 -> {best_ft['name']}")
        out(f"   d_PGD_ASR    = {d_pgd:+.4f}  (vs TRADES gamma=0)")
        out(f"   d_transfer   = {d_tr:+.4f}   (positive -> transfer got HARDER too)")
        out(f"   d_clean_acc  = {d_acc:+.4f}")
        out(f"   FGSM-PGD gap = {gap_fgsm_pgd:+.4f}  (negative is normal; large "
            f"positive is a masking flag)")

        # masking signal: white-box PGD ASR drops but transfer ASR doesn't
        # i.e. d_pgd << 0 (better white-box) but d_tr ~= 0 or > 0.
        white_box_gain = -d_pgd                                  # >0 means improved
        transfer_gain  = trades_row["transfer"] - best_ft["transfer"]  # >0 means harder to transfer
        masking_flag = (white_box_gain > 0.02) and (transfer_gain < 0.5 * white_box_gain)
        acc_ok = d_acc >= -0.02

        out(f"   white_box_gain = {white_box_gain:+.4f}   transfer_gain = "
            f"{transfer_gain:+.4f}")
        if masking_flag:
            verdict = ("MASKING-SUSPECT: Focal-TRADES improves white-box PGD ASR but "
                       "transfer ASR is barely affected. Consistent with H325/H429 "
                       "confidence-compression masking. Treat any white-box gain as "
                       "unreliable until an adaptive / EOT / AutoAttack probe is run.")
        elif white_box_gain > 0.02 and acc_ok:
            verdict = ("YES: Focal-TRADES (best gamma) lowers PGD ASR by >0.02 vs the "
                       "TRADES gamma=0 baseline AND transfer ASR moves with it, so the "
                       "gain is unlikely to be pure masking. Clean accuracy preserved.")
        elif white_box_gain > 0.02 and not acc_ok:
            verdict = ("PARTIAL: Focal-TRADES lowers PGD ASR but clean accuracy drops "
                       ">0.02 vs TRADES baseline. Not a free lunch.")
        elif -0.02 <= white_box_gain <= 0.02:
            verdict = ("NEUTRAL: Focal weighting on the TRADES KL term does not "
                       "meaningfully change PGD ASR at N_train=6000 / 10 epochs. "
                       "Consistent with the campaign's broader finding that TRADES "
                       "variants tie vanilla AT at this scale.")
        else:
            verdict = ("NO: Focal weighting on TRADES KL hurts robustness vs the "
                       "gamma=0 TRADES baseline.")
        out("")
        out("ONE-LINE VERDICT: " + verdict)
    else:
        out("ONE-LINE VERDICT: NO FOCAL-TRADES CONDITIONS RAN")

    # focal-vs-KL isolation note
    if ft_rows and fceat_rows:
        out("")
        out("Focal-vs-KL isolation (best gamma>0 in each family):")
        best_fceat = min((r for r in fceat_rows if r["gamma"] > 0),
                        key=lambda r: r["pgd_asr"], default=None)
        if best_ft and best_fceat:
            out(f"   best Focal-TRADES : {best_ft['name']}  PGD={best_ft['pgd_asr']:.4f}")
            out(f"   best Focal-CE-AT  : {best_fceat['name']}  PGD={best_fceat['pgd_asr']:.4f}")
            d = best_ft["pgd_asr"] - best_fceat["pgd_asr"]
            out(f"   PGD diff (FT - FCEAT) = {d:+.4f}  "
                f"(negative => KL channel adds something beyond focal CE-AT)")

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
