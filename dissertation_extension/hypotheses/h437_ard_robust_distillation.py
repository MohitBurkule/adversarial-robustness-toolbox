"""
H437 - ARD (Adversarially Robust Distillation, Goldblum et al. AAAI 2020).

Anchor:
  Goldblum, Fowl, Feizi, Goldstein, "Adversarially Robust Distillation",
  AAAI 2020. Distill a *robust* teacher into a student via a KL term computed
  on ADVERSARIAL examples crafted against the student. The student inherits
  the teacher's robust soft-label structure.

Critique (gap G3):
  ARD's claimed gain over plain PGD-AT may partly be (a) extra effective
  compute (the inner-PGD plus teacher forward), and (b) a "gradient masking"
  artefact: distilling soft labels from a frozen robust teacher can flatten
  the student's logit landscape locally without making the student genuinely
  robust to attacks crafted on the *teacher* (transfer). Two follow-ups in
  the literature also pull on this thread:
    - Zi et al. 2021 RSLAD (ICCV): student is fully driven by the teacher's
      soft labels on adv examples (no hard-label CE); reports better
      robust accuracy than ARD on CIFAR-10/100.
    - Zhao et al. MTARD (Multi-Teacher AT-Distillation, AAAI 2022): mix a
      clean teacher and a robust teacher to balance natural/robust acc.
  Both motivate the same audit: is robustness transferred or merely masked?

Hypothesis (H437):
  Distilling a PGD-AT or TRADES teacher into a fresh same-architecture
  student via ARD gives the student measurable PGD robustness, but the
  student is MORE vulnerable to adversarials transferred from its TEACHER
  than to its own white-box PGD: i.e. ARD partially launders robustness
  into local gradient flatness.

Design (this script):
  Teacher menu:
    Tv  - PGD-AT teacher (10-step PGD inner, eps=0.1, 10 epochs)
    Tt  - TRADES teacher (beta=3, 10-step KL-PGD inner, 10 epochs)
  Student conditions (same SmallCNN width=32):
    S0  - standard CE training (baseline reference, no AT)
    S1  - PGD-AT student trained directly (compute-matched reference)
    S2a - ARD(T=Tv, temperature=2)
    S2b - ARD(T=Tv, temperature=4)
    S2c - ARD(T=Tv, temperature=10)
    S3  - ARD(T=Tt, temperature=4)        # check whether TRADES teacher helps
  Compute matching:
    ARD inner-loop uses PGD-10 to craft x_adv against the student, exactly
    like the S1 PGD-AT reference; both run for EPOCHS=10. The teacher is
    trained ONCE up-front. Per-epoch wall-clock of S1 and the ARD students
    is reported so the reader can confirm matched adv-PGD compute.

ARD objective (per Goldblum 2020):
    x_adv = PGD_student(x, y; eps, steps)
    L_ARD = alpha * T^2 * KL( log_softmax(student(x_adv)/T) ||
                              softmax(teacher(x_adv)/T) )
            + (1 - alpha) * CE(student(x_adv), y)
  Default alpha = 1.0 (pure distillation on adv), matches the headline
  ARD setting in the paper.

Evaluation (all on test set):
  - clean acc
  - FGSM ASR (eps=0.1)
  - white-box PGD ASR (steps=10, eps=0.1, alpha=0.01)
  - white-box PGD ASR with 5 random restarts (worst-case per sample)
  - PGD ASR with steps=50 (curve plateau check)
  - TRANSFER from teacher: PGD adversarials crafted on the teacher used
    against the student. Compare to white-box ASR. transfer > white-box
    by margin > 0.03 is a masking flag (cf. H391).
  - mean logit margin
  - per-model masking VERDICT: GENUINE if claims robust (wb<0.5) AND
    (transfer <= wb + 0.03) AND (multi-restart <= wb + 0.05) AND
    (curve_50 <= curve_10 + 0.05); otherwise LIKELY-MASKING. Standard CE
    reference is N/A.

Config (campaign-standard):
  N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9, wd=5e-4),
  SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.

Output: results/fashion_mnist/h437_ard_robust_distillation_output.txt
ASCII only. No emojis. Flush per stage.

DO NOT EXECUTE from this file (orchestrator will run later).
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
TRADES_BETA = 3.0
ARD_ALPHA = 1.0                 # weight on KL term; (1-alpha) on hard-label CE
TEMPS = [2.0, 4.0, 10.0]
N_RESTARTS = 5                  # masking probe (worst-case over 5 restarts)
STEPS_PLATEAU = 50              # masking probe (does ASR keep rising?)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "fashion_mnist",
                   "h437_ard_robust_distillation_output.txt")

_LINES = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LINES.append(s)


def flush_file():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(_LINES) + "\n")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _opt(model):
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                           lr=LR, momentum=0.9, weight_decay=5e-4)


def _iter_batches(Xtr, Ytr):
    n = Xtr.size(0)
    perm = torch.randperm(n, device=Xtr.device)
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        yield Xtr[idx], Ytr[idx]


def pgd_ce(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA,
           random_start=True):
    return C.pgd(model, x, y, eps=eps, steps=steps, alpha=alpha,
                 random_start=random_start)


def pgd_kl(model, x, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """TRADES inner step: maximise KL( p_clean || p_adv ) w.r.t. x_adv."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)
    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(0, 1)
    x0 = x.detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        kl = F.kl_div(F.log_softmax(model(x_adv), dim=1), p_clean,
                      reduction="batchmean")
        g, = torch.autograd.grad(kl, x_adv)
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1)
    if was_training:
        model.train()
    return x_adv.detach()


# ---------------------------------------------------------------------------
# teacher trainers
# ---------------------------------------------------------------------------
def train_pgd_at_teacher(meta, Xtr, Ytr):
    """PGD-AT teacher Tv (vanilla AT, Madry)."""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=SEED)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    model.train()
    for ep in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            x_adv = pgd_ce(model, xb, yb)
            opt.zero_grad()
            loss = F.cross_entropy(model(x_adv), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_trades_teacher(meta, Xtr, Ytr, beta=TRADES_BETA):
    """TRADES teacher Tt."""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=SEED)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    model.train()
    for ep in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            x_adv = pgd_kl(model, xb)
            model.train()
            opt.zero_grad()
            out_c = model(xb)
            out_a = model(x_adv)
            loss = F.cross_entropy(out_c, yb) + beta * F.kl_div(
                F.log_softmax(out_a, dim=1),
                F.softmax(out_c, dim=1),
                reduction="batchmean")
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# student trainers
# ---------------------------------------------------------------------------
def train_standard_student(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=SEED)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    model.train()
    for ep in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_pgd_at_student(meta, Xtr, Ytr):
    """Compute-matched reference: PGD-AT trained student (same as teacher Tv)."""
    return train_pgd_at_teacher(meta, Xtr, Ytr)


def train_ard_student(meta, Xtr, Ytr, teacher, temperature, alpha=ARD_ALPHA):
    """ARD (Goldblum 2020).

        x_adv = PGD on student loss (CE w.r.t. y)
        L = alpha * T^2 * KL( log_softmax(s(x_adv)/T) || softmax(t(x_adv)/T) )
            + (1-alpha) * CE(s(x_adv), y)
    """
    C.set_seed(SEED)
    student = C.build_model("cnn", meta, width=32, seed=SEED)
    opt = _opt(student)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    T = float(temperature)
    student.train()
    for ep in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            # inner: craft x_adv against the STUDENT (matches Goldblum eq.)
            x_adv = pgd_ce(student, xb, yb)
            student.train()
            opt.zero_grad()
            s_logits = student(x_adv)
            with torch.no_grad():
                t_logits = teacher(x_adv)
            kl = F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction="batchmean") * (T * T)
            ce = F.cross_entropy(s_logits, yb)
            loss = alpha * kl + (1.0 - alpha) * ce
            loss.backward()
            opt.step()
        sched.step()
    student.eval()
    return student


# ---------------------------------------------------------------------------
# eval / masking battery (mirrors H391 conventions)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _correct_mask(model, X, Y, batch=256):
    parts = []
    for i in range(0, X.size(0), batch):
        parts.append((model(X[i:i + batch]).argmax(1) == Y[i:i + batch]).cpu())
    return torch.cat(parts)


def _asr_from_advx(model, Xadv, Y, corr, batch=256):
    flips = []
    with torch.no_grad():
        for i in range(0, Xadv.size(0), batch):
            pred = model(Xadv[i:i + batch]).argmax(1)
            flips.append((pred != Y[i:i + batch]).cpu())
    flips = torch.cat(flips).numpy()
    corr_np = corr.numpy().astype(bool)
    return float(flips[corr_np].mean()) if corr_np.sum() > 0 else float("nan")


def pgd_batched(model, X, Y, eps, steps, alpha, random_start=True, batch=256):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(C.pgd(model, X[i:i + batch], Y[i:i + batch], eps=eps,
                          steps=steps, alpha=alpha, random_start=random_start))
    return torch.cat(outs)


def fgsm_batched(model, X, Y, eps, batch=256):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(C.fgsm(model, X[i:i + batch], Y[i:i + batch], eps))
    return torch.cat(outs)


def multi_restart_asr(model, X, Y, corr, restarts=N_RESTARTS, batch=256):
    corr_np = corr.numpy().astype(bool)
    flipped_any = np.zeros(X.size(0), dtype=bool)
    for _ in range(restarts):
        adv = pgd_batched(model, X, Y, EPS, PGD_STEPS, PGD_ALPHA,
                          random_start=True)
        with torch.no_grad():
            f = []
            for i in range(0, adv.size(0), batch):
                f.append((model(adv[i:i + batch]).argmax(1)
                          != Y[i:i + batch]).cpu())
            f = torch.cat(f).numpy()
        flipped_any |= f
    return float(flipped_any[corr_np].mean()) if corr_np.sum() > 0 else float("nan")


def evaluate_full(model, Xte, Yte, transfer_advx_by_teacher=None):
    """Return a dict of clean acc, FGSM ASR, white-box PGD-10 ASR,
    multi-restart ASR, PGD-50 ASR, transfer-from-teacher ASR (if given),
    and mean margin."""
    corr = _correct_mask(model, Xte, Yte)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    fg_adv = fgsm_batched(model, Xte, Yte, EPS)
    fg_asr = _asr_from_advx(model, fg_adv, Yte, corr)
    wb_adv = pgd_batched(model, Xte, Yte, EPS, PGD_STEPS, PGD_ALPHA,
                         random_start=True)
    wb_asr = _asr_from_advx(model, wb_adv, Yte, corr)
    mr_asr = multi_restart_asr(model, Xte, Yte, corr)
    pl_adv = pgd_batched(model, Xte, Yte, EPS, STEPS_PLATEAU,
                         2.5 * EPS / STEPS_PLATEAU, random_start=True)
    pl_asr = _asr_from_advx(model, pl_adv, Yte, corr)
    transfer = {}
    if transfer_advx_by_teacher is not None:
        for tname, tadv in transfer_advx_by_teacher.items():
            transfer[tname] = _asr_from_advx(model, tadv, Yte, corr)
    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))
    return {
        "n_correct": int(corr.sum()),
        "clean_acc": float(clean_acc),
        "fgsm_asr": fg_asr,
        "wb_pgd_asr": wb_asr,
        "mr_pgd_asr": mr_asr,
        "pgd50_asr": pl_asr,
        "transfer": transfer,
        "mean_margin": mean_margin,
    }


def masking_verdict(r, robust_claim=0.5):
    """LIKELY-MASKING if model claims robustness AND any signal trips."""
    wb = r["wb_pgd_asr"]
    claims = wb < robust_claim
    sigs = []
    # transfer-from-teacher > wb
    for tname, tasr in r["transfer"].items():
        if tasr > wb + 0.03:
            sigs.append(f"transfer({tname})={tasr:.3f} > wb={wb:.3f}")
    # multi-restart >> single
    if r["mr_pgd_asr"] > wb + 0.05:
        sigs.append(f"5xrestart={r['mr_pgd_asr']:.3f} >> wb={wb:.3f}")
    # plateau (steps=50 vs steps=10)
    rise = r["pgd50_asr"] - wb
    if rise > 0.05:
        sigs.append(f"steps10->50 rises +{rise:.3f}")
    if not claims:
        return "N/A (not claiming robustness; wb=%.3f)" % wb, sigs
    if sigs:
        return "LIKELY-MASKING", sigs
    return "GENUINE", sigs


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log("=" * 78)
    log("H437  ARD (Adversarially Robust Distillation) - masking audit")
    log("       anchor: goldblum-2020-ard (AAAI 2020)")
    log("       gap: G3 (distillation-based AT not replicated post H173)")
    log("=" * 78)
    log(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9, wd=5e-4) SEED={SEED}")
    log(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} "
        f"TRADES_BETA={TRADES_BETA}")
    log(f"        ARD_ALPHA={ARD_ALPHA} TEMPS={TEMPS} N_RESTARTS={N_RESTARTS} "
        f"STEPS_PLATEAU={STEPS_PLATEAU}")
    log(f"        device={C.DEVICE}")
    log("")

    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                        seed=SEED)
    log(f"data: Xtr={tuple(Xtr.shape)} Xte={tuple(Xte.shape)}")

    # -----------------------------------------------------------------------
    # [1] train both teachers
    # -----------------------------------------------------------------------
    log("\n[1] training teachers ...")
    ts = time.time()
    Tv = train_pgd_at_teacher(meta, Xtr, Ytr)
    t_Tv = time.time() - ts
    _, acc_Tv = C.logits_and_acc(Tv, Xte, Yte)
    log(f"    Tv  (PGD-AT teacher, 10-step inner)   clean_acc={acc_Tv:.4f}  "
        f"({t_Tv:.1f}s)")
    flush_file()

    ts = time.time()
    Tt = train_trades_teacher(meta, Xtr, Ytr, beta=TRADES_BETA)
    t_Tt = time.time() - ts
    _, acc_Tt = C.logits_and_acc(Tt, Xte, Yte)
    log(f"    Tt  (TRADES teacher, beta={TRADES_BETA})   "
        f"clean_acc={acc_Tt:.4f}  ({t_Tt:.1f}s)")
    flush_file()

    # teacher transfer adversarials (crafted on teacher, used against student)
    log("\n[1b] crafting teacher transfer adversarials (PGD-10 on each teacher) ...")
    Tv_adv = pgd_batched(Tv, Xte, Yte, EPS, PGD_STEPS, PGD_ALPHA,
                         random_start=True)
    Tt_adv = pgd_batched(Tt, Xte, Yte, EPS, PGD_STEPS, PGD_ALPHA,
                         random_start=True)
    teacher_advs = {"Tv": Tv_adv, "Tt": Tt_adv}
    log("    done.")
    flush_file()

    # -----------------------------------------------------------------------
    # [2] train students (S0 std, S1 PGD-AT, S2{a,b,c} ARD(Tv,T), S3 ARD(Tt,T=4))
    # -----------------------------------------------------------------------
    log("\n[2] training students ...")
    students = {}
    times = {}

    ts = time.time()
    students["S0_std"] = train_standard_student(meta, Xtr, Ytr)
    times["S0_std"] = time.time() - ts
    log(f"    S0_std         done ({times['S0_std']:.1f}s)")
    flush_file()

    ts = time.time()
    students["S1_pgd_at"] = train_pgd_at_student(meta, Xtr, Ytr)
    times["S1_pgd_at"] = time.time() - ts
    log(f"    S1_pgd_at      done ({times['S1_pgd_at']:.1f}s)  "
        f"(compute-matched reference)")
    flush_file()

    for T in TEMPS:
        name = f"S2_ard_Tv_T{int(T)}"
        ts = time.time()
        students[name] = train_ard_student(meta, Xtr, Ytr, Tv, T)
        times[name] = time.time() - ts
        log(f"    {name:18s} done ({times[name]:.1f}s)")
        flush_file()

    ts = time.time()
    students["S3_ard_Tt_T4"] = train_ard_student(meta, Xtr, Ytr, Tt, 4.0)
    times["S3_ard_Tt_T4"] = time.time() - ts
    log(f"    S3_ard_Tt_T4   done ({times['S3_ard_Tt_T4']:.1f}s)")
    flush_file()

    # -----------------------------------------------------------------------
    # [3] full evaluation per student
    # -----------------------------------------------------------------------
    log("\n[3] evaluating students (clean, FGSM, wb-PGD-10, 5xrestart, PGD-50, "
        "transfer-from-Tv, transfer-from-Tt) ...")
    results = {}
    for name, model in students.items():
        ts = time.time()
        r = evaluate_full(model, Xte, Yte, transfer_advx_by_teacher=teacher_advs)
        results[name] = r
        log(f"    {name:18s} done ({time.time()-ts:.1f}s)")
        flush_file()

    # also evaluate the teachers themselves for reference
    log("\n[3b] evaluating teachers for reference ...")
    teacher_results = {}
    for name, model in [("Tv", Tv), ("Tt", Tt)]:
        ts = time.time()
        r = evaluate_full(model, Xte, Yte, transfer_advx_by_teacher=None)
        teacher_results[name] = r
        log(f"    teacher {name}       done ({time.time()-ts:.1f}s)")
        flush_file()

    # -----------------------------------------------------------------------
    # [4] main table
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[4] MAIN TABLE")
    log("=" * 78)
    hdr = ("{:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7}"
           .format("model", "n_ok", "clean", "fgsm", "wb_pgd",
                   "5x_pgd", "pgd_50", "tx_Tv", "tx_Tt", "margin"))
    log(hdr)
    log("-" * len(hdr))

    def row(name, r):
        tx_v = r["transfer"].get("Tv", float("nan"))
        tx_t = r["transfer"].get("Tt", float("nan"))
        log("{:<18} {:>6d} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} "
            "{:>8.4f} {:>8.4f} {:>7.3f}".format(
                name, r["n_correct"], r["clean_acc"], r["fgsm_asr"],
                r["wb_pgd_asr"], r["mr_pgd_asr"], r["pgd50_asr"],
                tx_v, tx_t, r["mean_margin"]))

    # teachers first
    for tn in ("Tv", "Tt"):
        tr = teacher_results[tn]
        log("{:<18} {:>6d} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} "
            "{:>8} {:>8} {:>7.3f}".format(
                "teacher_" + tn, tr["n_correct"], tr["clean_acc"],
                tr["fgsm_asr"], tr["wb_pgd_asr"], tr["mr_pgd_asr"],
                tr["pgd50_asr"], "-", "-", tr["mean_margin"]))

    for name in ["S0_std", "S1_pgd_at",
                 "S2_ard_Tv_T2", "S2_ard_Tv_T4", "S2_ard_Tv_T10",
                 "S3_ard_Tt_T4"]:
        row(name, results[name])
    log("-" * len(hdr))

    # -----------------------------------------------------------------------
    # [5] per-student masking verdicts
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[5] MASKING VERDICTS (per student)")
    log("=" * 78)
    log("  signals checked: transfer-from-teacher > wb+0.03,")
    log("                   5x-restart > wb+0.05, pgd_50 > wb+0.05.")
    log("  verdict only meaningful if wb_pgd_asr < 0.50 (model claims robust).")
    log("")
    for name in ["S0_std", "S1_pgd_at",
                 "S2_ard_Tv_T2", "S2_ard_Tv_T4", "S2_ard_Tv_T10",
                 "S3_ard_Tt_T4"]:
        r = results[name]
        verdict, sigs = masking_verdict(r)
        log(f"  {name:18s} wb={r['wb_pgd_asr']:.3f}  -> {verdict}")
        if sigs:
            for s in sigs:
                log(f"        signal: {s}")
        else:
            log("        (no masking signals)")

    # -----------------------------------------------------------------------
    # [6] compute-matched comparison (S1 vs ARD students)
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[6] COMPUTE-MATCHED COMPARISON (S1 PGD-AT vs ARD students)")
    log("=" * 78)
    log("  every student does PGD-10 per batch x EPOCHS=10. teacher fwd cost in")
    log("  ARD is ~1 extra forward per batch on top of the matched PGD-AT cost.")
    log("")
    log("  per-student total wall-clock (training only):")
    for name in ["S0_std", "S1_pgd_at", "S2_ard_Tv_T2", "S2_ard_Tv_T4",
                 "S2_ard_Tv_T10", "S3_ard_Tt_T4"]:
        log(f"    {name:18s} {times[name]:7.1f}s")
    log("")
    s1 = results["S1_pgd_at"]
    best_ard = min(
        ["S2_ard_Tv_T2", "S2_ard_Tv_T4", "S2_ard_Tv_T10", "S3_ard_Tt_T4"],
        key=lambda n: results[n]["wb_pgd_asr"])
    bard = results[best_ard]
    log(f"  S1 (PGD-AT reference) : clean={s1['clean_acc']:.4f}  "
        f"wb_pgd={s1['wb_pgd_asr']:.4f}  5x={s1['mr_pgd_asr']:.4f}")
    log(f"  best ARD = {best_ard:18s}: clean={bard['clean_acc']:.4f}  "
        f"wb_pgd={bard['wb_pgd_asr']:.4f}  5x={bard['mr_pgd_asr']:.4f}")
    d_wb = bard["wb_pgd_asr"] - s1["wb_pgd_asr"]
    d_mr = bard["mr_pgd_asr"] - s1["mr_pgd_asr"]
    d_acc = bard["clean_acc"] - s1["clean_acc"]
    log(f"  delta vs S1: clean {d_acc:+.4f}  wb_pgd {d_wb:+.4f}  "
        f"5x_pgd {d_mr:+.4f}")

    # -----------------------------------------------------------------------
    # [7] FINAL VERDICT
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[7] VERDICT")
    log("=" * 78)
    # ARD beats PGD-AT on white-box?
    ard_beats_at = d_wb < -0.02
    # ARD survives masking probes (best ARD)?
    bv, bsigs = masking_verdict(bard)
    ard_genuine = (bv == "GENUINE")
    # transfer-from-teacher worse than white-box (the hypothesis)?
    bard_tx_Tv = bard["transfer"].get("Tv", float("nan"))
    bard_tx_Tt = bard["transfer"].get("Tt", float("nan"))
    teacher_used = "Tt" if best_ard.endswith("Tt_T4") else "Tv"
    relevant_tx = bard_tx_Tt if teacher_used == "Tt" else bard_tx_Tv
    teacher_transfer_worse = relevant_tx > bard["wb_pgd_asr"] + 0.03

    log(f"  ARD best = {best_ard}")
    log(f"  ARD vs PGD-AT (S1) on wb_pgd_asr: delta={d_wb:+.4f}  "
        f"{'(ARD beats AT by >0.02)' if ard_beats_at else '(no >0.02 gain over AT)'}")
    log(f"  ARD masking verdict: {bv}")
    log(f"  transfer from used teacher ({teacher_used}): "
        f"{relevant_tx:.4f} vs wb {bard['wb_pgd_asr']:.4f}  "
        f"{'(teacher transfer > wb)' if teacher_transfer_worse else '(teacher transfer <= wb)'}")

    if ard_genuine and ard_beats_at:
        one = ("YES: ARD genuinely improves PGD robustness over compute-matched "
               "PGD-AT, no masking signals tripped.")
    elif ard_genuine and not ard_beats_at:
        one = ("PARTIAL: ARD ties PGD-AT (no meaningful wb gain) but is GENUINE "
               "by the masking battery; distillation buys little here at "
               "N=6000/10 epochs.")
    elif (not ard_genuine) and teacher_transfer_worse:
        one = ("MASKED: ARD lowers white-box PGD ASR but adversarials from the "
               "TEACHER transfer to the student more effectively than its own "
               "white-box attack - hypothesis SUPPORTED, ARD partially launders "
               "robustness into local gradient flatness.")
    else:
        one = ("MASKED: ARD shows masking signals (multi-restart or "
               "increasing-steps) without matched gain over PGD-AT.")
    log("  ONE-LINE VERDICT: " + one)

    log(f"\ndone in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
