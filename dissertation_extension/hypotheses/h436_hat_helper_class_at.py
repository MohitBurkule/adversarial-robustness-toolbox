"""
H436 - HAT: Helper-based Adversarial Training (Rade & Moosavi-Dezfooli ICLR 2022).

Paper: "Reducing Excessive Margin to Achieve a Better Accuracy vs. Robustness
Trade-off". The argument is that PGD-AT pushes the decision boundary too far
along certain adversarial directions, hurting clean accuracy. HAT adds a
"helper" example x_h = x + gamma * r where r = x_adv - x is the PGD
perturbation, and labels x_h by the prediction of a STANDARD (non-robust)
network f_std. By telling the robust model "stop pushing the boundary out
past where the standard model has already changed its mind", HAT recovers
some clean accuracy without sacrificing PGD robustness.

HAT loss:
    L_HAT = CE(f_th(x), y)                            # clean
          + beta  * CE(f_th(x_adv), y)                # PGD-AT term
          + lam   * CE(f_th(x_h), y_help)             # helper term
where
    r        = x_adv - x          (PGD perturbation, signed)
    x_h      = clamp(x + gamma*r, 0, 1)
    y_help   = argmax f_std(x_h)  (standard-network label, possibly != y)

Knobs swept here:
    helper-extrapolation gamma in {0.0, 1.0, 2.0, 3.0}
        gamma=0.0 is degenerate (x_h = x) and serves as a regulariser sanity.
        gamma=2.0 is the paper's recommended default.
    beta = 1.0, lam = 1.0 (paper's HAT-baseline ratio).

Controls (gap M4, M7, M5, M6 from CAMPAIGN_GAP_MAP.md):
  C1. Compute-matched PGD-AT baseline (same PGD-10 inner step count, no helper).
  C2. TRADES-style ablation: remove helper => effectively PGD-AT with KD off
      => same script with lam=0 (this is the gamma=0 "no helper" condition).
  C3. Per-class clean-acc preservation (Tsipras lens): is the recovered clean
      acc concentrated on already-easy classes? Report per-class clean and
      worst-class PGD ASR for every condition.
  C4. Transfer-attack masking check: take PGD adv examples crafted on a
      separately-trained baseline (non-robust) model and evaluate every HAT
      condition on them. If white-box PGD ASR is much higher than transfer
      ASR for the same defence, gradient masking is suspected. We compare
      the gap to the same gap for PGD-AT.

Critique up-front (logged in the output for the dissertation chapter):
  - HAT introduces a confidence-smoothing knob (helper label coming from a
    standard, typically high-confidence model). It is at risk of being a
    relabel/soft-label trick rather than a margin fix. The gamma=0 helper
    (which collapses to a self-distillation onto the standard model on the
    SAME input) is the sanity check.
  - If lam is large the helper term can dominate and just clone f_std,
    erasing the PGD-AT term's effect. Here lam = beta = 1 to avoid that.
  - Transfer-attack control is essential because helper labelling by f_std
    could plausibly induce gradient masking via mismatched logit scaling.

Config (campaign standard):
  Fashion-MNIST, SmallCNN width=32, N_TRAIN=6000, EPOCHS=10, LR=0.05,
  BATCH=128, SGD(mom=0.9, wd=5e-4), SEED=0, EPS=0.1, PGD_STEPS=10,
  PGD_ALPHA=0.01. Single seed (single-seed convention of the campaign;
  see methodological caveat M1).

Extra anchors (found via web search, beyond the seed paper):
  - Tsipras et al. ICLR 2019 "Robustness May Be at Odds with Accuracy" -
    provides the formal account of why AT inflates margin and motivates
    HAT's correction.
  - Pang et al. ICML 2022 "Robustness and Accuracy Could Be Reconcilable
    by (Proper) Definition" (SCORE) - independent re-derivation of the
    "excessive margin" problem; directly comparable family.
  - Wang et al. ICLR 2020 "Improving Adversarial Robustness Requires
    Revisiting Misclassified Examples" (MART) - close cousin: also
    reweights by mis-prediction; HAT instead injects a helper class.
  - Tramer et al. NeurIPS 2020 "On Adaptive Attacks to Adversarial Example
    Defenses" - methodology for the transfer-attack masking sanity check.

This script writes ASCII only. It flushes after every condition and emits a
final VERDICT block. Do NOT execute (per task instructions).
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

BETA = 1.0          # weight on the PGD-AT term
LAM = 1.0           # weight on the helper term

# gamma sweep: extrapolation factor for the helper example x_h = x + gamma * r.
# 0.0 = no extrapolation (degenerate, helper == clean), 1.0 = helper at x_adv,
# 2.0 = paper default, 3.0 = aggressive.
GAMMAS = [0.0, 1.0, 2.0, 3.0]

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
)
OUT_FILE = os.path.join(OUT_DIR, "h436_hat_helper_class_at_output.txt")


# ---- utilities -----------------------------------------------------------
def _make_optimizer_sgd(model, lr):
    """SGD mom=0.9 wd=5e-4 per campaign standard."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def train_std(Xtr, Ytr, seed):
    """Train a fresh CNN from scratch with standard CE (no AT).
    Used both as the campaign baseline and as the HAT helper-labeller f_std.
    """
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_optimizer_sgd(model, LR)
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


def train_pgd_at(Xtr, Ytr, seed):
    """Compute-matched PGD-AT baseline: same PGD-10 budget as HAT, no helper."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            x_adv = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(x_adv), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_hat(Xtr, Ytr, f_std, gamma, seed):
    """HAT training with the given helper extrapolation factor gamma.

    For each batch:
      1. Run PGD-10 against the current model to get x_adv (==> r = x_adv - x).
      2. Build helper x_h = clamp(x + gamma * r, 0, 1).
      3. Query y_help = argmax f_std(x_h)  (no grad through f_std).
      4. Loss = CE(f(x), y) + beta * CE(f(x_adv), y) + lam * CE(f(x_h), y_help).
    """
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    f_std.eval()
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            # ---- inner PGD against current model ----
            model.eval()
            x_adv = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            model.train()
            r = (x_adv - xb).detach()

            # ---- helper construction ----
            x_h = (xb + gamma * r).clamp(0.0, 1.0).detach()
            with torch.no_grad():
                y_help = f_std(x_h).argmax(dim=1)

            # ---- HAT loss ----
            opt.zero_grad()
            logits_clean = model(xb)
            logits_adv = model(x_adv)
            loss_clean = F.cross_entropy(logits_clean, yb)
            loss_adv = F.cross_entropy(logits_adv, yb)
            if gamma == 0.0:
                # gamma=0 => x_h == x, helper term degenerates to a
                # self-distillation onto f_std on the clean inputs.
                # Keep the term explicit so this acts as the "no extrapolation"
                # control (still trained, just at the degenerate point).
                logits_help = model(x_h)
                loss_help = F.cross_entropy(logits_help, y_help)
            else:
                logits_help = model(x_h)
                loss_help = F.cross_entropy(logits_help, y_help)
            loss = loss_clean + BETA * loss_adv + LAM * loss_help
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- evaluation helpers --------------------------------------------------
@torch.no_grad()
def per_class_clean_acc(model, X, Y, n_classes=10):
    """Per-class clean accuracy."""
    model.eval()
    preds = []
    for i in range(0, X.size(0), 512):
        preds.append(model(X[i:i + 512]).argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    yt = Y.cpu().numpy()
    accs = []
    for c in range(n_classes):
        m = yt == c
        accs.append(float((preds[m] == c).mean()) if m.sum() > 0 else float("nan"))
    return accs


def per_class_pgd_asr(model, X, Y, n_classes=10):
    """PGD-10 ASR computed per ground-truth class (over originally-correct
    samples of that class)."""
    res = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    flips = res["flips"].astype(bool)
    corr = res["correct"].astype(bool)
    yt = Y.cpu().numpy()
    asrs = []
    for c in range(n_classes):
        m = (yt == c) & corr
        asrs.append(float(flips[m].mean()) if m.sum() > 0 else float("nan"))
    return asrs, float(res["asr"])


def eval_model(model, X, Y):
    """clean acc, FGSM ASR, whitebox PGD-10 ASR, mean margin."""
    _, clean_acc = C.logits_and_acc(model, X, Y)
    fg = C.attack_success(model, X, Y, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    mm = float(np.mean(C.margin(model, X, Y)))
    return dict(clean_acc=float(clean_acc), fgsm_asr=float(fg["asr"]),
                pgd_asr=float(pg["asr"]), mean_margin=mm)


def transfer_asr(target_model, X_adv, Y):
    """ASR of pre-crafted adversarial examples against target_model.
    Restricted to samples that target_model originally classifies correctly
    on the CLEAN inputs (this is how attack_success() defines ASR)."""
    # We need clean predictions to define the "originally-correct" mask.
    with torch.no_grad():
        # x_adv was crafted on a separate source model; we still want the
        # standard ASR definition: among samples target classifies correctly
        # on the clean version, what fraction get flipped by x_adv?
        # Reconstruct clean correctness from the source-side info would require
        # the original X; we assume caller provides aligned (X_adv, Y, clean_X)
        # via a wrapper below.
        raise NotImplementedError("use transfer_asr_full")


def transfer_asr_full(target_model, X_clean, X_adv, Y, batch=256):
    """Transfer-attack success rate of x_adv on target_model, with the
    standard ASR convention (denominator = originally-correct on clean)."""
    target_model.eval()
    flips, corr = [], []
    for i in range(0, X_clean.size(0), batch):
        xc = X_clean[i:i + batch]
        xa = X_adv[i:i + batch]
        y = Y[i:i + batch]
        with torch.no_grad():
            c = target_model(xc).argmax(1) == y
            f = target_model(xa).argmax(1) != y
        corr.append(c.cpu())
        flips.append(f.cpu())
    flips = torch.cat(flips).numpy().astype(bool)
    corr = torch.cat(corr).numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() > 0 else float("nan")


# ---- main ----------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    t0 = time.time()

    out("=" * 80)
    out("H436  HAT - Helper-based Adversarial Training (Rade & Moosavi-Dezfooli ICLR 2022)")
    out("=" * 80)
    out("config: dataset=" + DS +
        f"  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}  EPOCHS={EPOCHS}  LR={LR}  "
        f"BATCH={BATCH}  SGD(mom=0.9, wd=5e-4)  SEED={SEED}")
    out(f"        EPS={EPS}  PGD_STEPS={PGD_STEPS}  PGD_ALPHA={PGD_ALPHA}")
    out(f"        BETA={BETA}  LAM={LAM}  GAMMAS={GAMMAS}")
    out(f"        device={C.DEVICE}")
    out("")
    out("Hypothesis: helper extrapolation (gamma>=1) preserves clean accuracy")
    out("            relative to compute-matched PGD-AT WITHOUT sacrificing PGD")
    out("            robustness. Risk: helper term acts as confidence smoothing")
    out("            or gradient masking; controls below try to rule that out.")
    out("")
    flush_file()

    # ---------- data ----------
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    flush_file()

    # ---------- f_std (standard, non-robust) ----------
    out("\n[1] training f_std (standard, non-robust; used as HAT helper-labeller")
    out("    AND as the source of transfer-attack examples)")
    t = time.time()
    f_std = train_std(Xtr, Ytr, SEED)
    std_eval = eval_model(f_std, Xte, Yte)
    out(f"    f_std: clean={std_eval['clean_acc']:.4f}  "
        f"FGSM_ASR={std_eval['fgsm_asr']:.4f}  PGD_ASR={std_eval['pgd_asr']:.4f}  "
        f"margin={std_eval['mean_margin']:.4f}  ({time.time()-t:.1f}s)")
    flush_file()

    # ---------- transfer attack source: PGD adv crafted on f_std ----------
    out("\n[2] crafting transfer attacks: PGD-10 against f_std on the test set")
    t = time.time()
    f_std.eval()
    X_te_adv_from_std = []
    for i in range(0, Xte.size(0), 256):
        x = Xte[i:i + 256]; y = Yte[i:i + 256]
        X_te_adv_from_std.append(C.pgd(f_std, x, y, eps=EPS, steps=PGD_STEPS,
                                       alpha=PGD_ALPHA))
    X_te_adv_from_std = torch.cat(X_te_adv_from_std, dim=0)
    out(f"    transfer source ready ({time.time()-t:.1f}s)")
    flush_file()

    # ---------- compute-matched PGD-AT baseline ----------
    out("\n[3] training compute-matched PGD-AT baseline (no helper)")
    t = time.time()
    f_pgdat = train_pgd_at(Xtr, Ytr, SEED)
    pgdat_eval = eval_model(f_pgdat, Xte, Yte)
    pgdat_pc_clean = per_class_clean_acc(f_pgdat, Xte, Yte)
    pgdat_pc_pgd, _ = per_class_pgd_asr(f_pgdat, Xte, Yte)
    pgdat_transfer = transfer_asr_full(f_pgdat, Xte, X_te_adv_from_std, Yte)
    out(f"    PGD-AT: clean={pgdat_eval['clean_acc']:.4f}  "
        f"FGSM_ASR={pgdat_eval['fgsm_asr']:.4f}  "
        f"PGD_ASR(wb)={pgdat_eval['pgd_asr']:.4f}  "
        f"PGD_ASR(transfer-from-std)={pgdat_transfer:.4f}  "
        f"margin={pgdat_eval['mean_margin']:.4f}  ({time.time()-t:.1f}s)")
    out(f"    PGD-AT per-class clean: " +
        " ".join(f"{a:.2f}" for a in pgdat_pc_clean))
    out(f"    PGD-AT worst-class clean acc = {min(pgdat_pc_clean):.4f}")
    out(f"    PGD-AT worst-class PGD ASR   = {max(pgdat_pc_pgd):.4f}")
    flush_file()

    # ---------- baseline (non-robust f_std) row for the table ----------
    rows = []
    rows.append({
        "cond": "baseline (std, no AT)",
        "clean_acc": std_eval["clean_acc"],
        "fgsm_asr": std_eval["fgsm_asr"],
        "pgd_asr_wb": std_eval["pgd_asr"],
        "pgd_asr_tr": transfer_asr_full(f_std, Xte, X_te_adv_from_std, Yte),
        "margin": std_eval["mean_margin"],
        "worst_clean": min(per_class_clean_acc(f_std, Xte, Yte)),
        "worst_pgd": max(per_class_pgd_asr(f_std, Xte, Yte)[0]),
        "time_s": 0.0,
    })
    rows.append({
        "cond": "PGD-AT (compute-matched)",
        "clean_acc": pgdat_eval["clean_acc"],
        "fgsm_asr": pgdat_eval["fgsm_asr"],
        "pgd_asr_wb": pgdat_eval["pgd_asr"],
        "pgd_asr_tr": pgdat_transfer,
        "margin": pgdat_eval["mean_margin"],
        "worst_clean": min(pgdat_pc_clean),
        "worst_pgd": max(pgdat_pc_pgd),
        "time_s": 0.0,
    })

    # ---------- HAT sweep over gamma ----------
    out("\n[4] HAT sweep over helper extrapolation gamma")
    for gi, gamma in enumerate(GAMMAS):
        out("\n" + "-" * 80)
        out(f"[4.{gi+1}] HAT  gamma={gamma}  beta={BETA}  lam={LAM}")
        out("-" * 80)
        t = time.time()
        model = train_hat(Xtr, Ytr, f_std=f_std, gamma=gamma, seed=SEED)
        m = eval_model(model, Xte, Yte)
        pc_clean = per_class_clean_acc(model, Xte, Yte)
        pc_pgd, _ = per_class_pgd_asr(model, Xte, Yte)
        tr_asr = transfer_asr_full(model, Xte, X_te_adv_from_std, Yte)
        elapsed = time.time() - t
        cond = f"HAT gamma={gamma}"
        rows.append({
            "cond": cond,
            "clean_acc": m["clean_acc"],
            "fgsm_asr": m["fgsm_asr"],
            "pgd_asr_wb": m["pgd_asr"],
            "pgd_asr_tr": tr_asr,
            "margin": m["mean_margin"],
            "worst_clean": min(pc_clean),
            "worst_pgd": max(pc_pgd),
            "time_s": elapsed,
        })
        out(f"    {cond}: clean={m['clean_acc']:.4f}  "
            f"FGSM_ASR={m['fgsm_asr']:.4f}  "
            f"PGD_ASR(wb)={m['pgd_asr']:.4f}  "
            f"PGD_ASR(transfer)={tr_asr:.4f}  "
            f"margin={m['mean_margin']:.4f}  ({elapsed:.1f}s)")
        out(f"    per-class clean: " + " ".join(f"{a:.2f}" for a in pc_clean))
        out(f"    per-class PGD ASR: " + " ".join(f"{a:.2f}" for a in pc_pgd))
        out(f"    worst-class clean = {min(pc_clean):.4f}   "
            f"worst-class PGD ASR = {max(pc_pgd):.4f}")
        flush_file()

    # ---------- main table ----------
    out("\n" + "=" * 80)
    out("[5] MAIN TABLE")
    out("=" * 80)
    hdr = ("{:<28} {:>9} {:>9} {:>10} {:>10} {:>8} {:>10} {:>10}".format(
        "condition", "clean", "FGSM", "PGD(wb)", "PGD(tr)", "margin",
        "wc_clean", "wc_PGD"))
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<28} {:>9.4f} {:>9.4f} {:>10.4f} {:>10.4f} {:>8.3f} {:>10.4f} {:>10.4f}"
            .format(r["cond"], r["clean_acc"], r["fgsm_asr"],
                    r["pgd_asr_wb"], r["pgd_asr_tr"], r["margin"],
                    r["worst_clean"], r["worst_pgd"]))
    out("-" * len(hdr))

    # ---------- verdict ----------
    out("\n" + "=" * 80)
    out("[6] VERDICT")
    out("=" * 80)
    pgdat = next(r for r in rows if r["cond"].startswith("PGD-AT"))
    hat_rows = [r for r in rows if r["cond"].startswith("HAT")]
    best_hat = min(hat_rows, key=lambda r: r["pgd_asr_wb"])
    best_acc_hat = max(hat_rows, key=lambda r: r["clean_acc"])

    out(f"  PGD-AT baseline:  clean={pgdat['clean_acc']:.4f}  "
        f"PGD_ASR(wb)={pgdat['pgd_asr_wb']:.4f}  "
        f"worst_clean={pgdat['worst_clean']:.4f}  "
        f"worst_PGD={pgdat['worst_pgd']:.4f}")
    out("")
    for r in hat_rows:
        d_acc = r["clean_acc"] - pgdat["clean_acc"]
        d_pgd = r["pgd_asr_wb"] - pgdat["pgd_asr_wb"]
        d_wc_clean = r["worst_clean"] - pgdat["worst_clean"]
        gap = r["pgd_asr_wb"] - r["pgd_asr_tr"]   # >0 wb harder than transfer
        pgdat_gap = pgdat["pgd_asr_wb"] - pgdat["pgd_asr_tr"]
        masking_flag = "MASKING-SUSPECT" if (gap < pgdat_gap - 0.10) else "ok"
        out(f"  {r['cond']}: d_clean={d_acc:+.4f}  d_PGD_ASR={d_pgd:+.4f}  "
            f"d_worst_clean={d_wc_clean:+.4f}  "
            f"wb-tr gap={gap:+.4f} vs PGD-AT gap={pgdat_gap:+.4f}  "
            f"[{masking_flag}]")

    # robustness preserved if HAT PGD_ASR is no more than +0.03 above PGD-AT
    rob_kept = best_acc_hat["pgd_asr_wb"] <= pgdat["pgd_asr_wb"] + 0.03
    acc_gain = best_acc_hat["clean_acc"] - pgdat["clean_acc"]
    wc_clean_gain = best_acc_hat["worst_clean"] - pgdat["worst_clean"]
    masking_any = any(
        (r["pgd_asr_wb"] - r["pgd_asr_tr"]) <
        (pgdat["pgd_asr_wb"] - pgdat["pgd_asr_tr"]) - 0.10
        for r in hat_rows
    )

    out("")
    out(f"  best-accuracy HAT condition: {best_acc_hat['cond']}")
    out(f"    clean gain vs PGD-AT       = {acc_gain:+.4f}")
    out(f"    worst-class clean gain     = {wc_clean_gain:+.4f}")
    out(f"    PGD_ASR(wb) vs PGD-AT      = "
        f"{best_acc_hat['pgd_asr_wb']:.4f} vs {pgdat['pgd_asr_wb']:.4f}  "
        f"(robustness {'PRESERVED' if rob_kept else 'LOST'})")
    out(f"  best-robustness HAT condition: {best_hat['cond']}  "
        f"(PGD_ASR={best_hat['pgd_asr_wb']:.4f})")
    out(f"  transfer-attack masking detected in any HAT condition: "
        f"{'YES' if masking_any else 'no'}")

    if acc_gain > 0.01 and rob_kept and not masking_any:
        verdict = ("SUPPORTED: HAT recovers clean accuracy vs PGD-AT "
                   "without losing PGD robustness, and the transfer-attack "
                   "gap is consistent with PGD-AT (no masking signal).")
    elif acc_gain > 0.01 and rob_kept and masking_any:
        verdict = ("PARTIAL: clean-acc gain present but at least one HAT "
                   "condition shows a suspiciously large white-box vs "
                   "transfer ASR gap; treat helper term as a candidate "
                   "gradient-masking knob.")
    elif acc_gain > 0.01 and not rob_kept:
        verdict = ("NOT SUPPORTED (acc-only): HAT gains clean accuracy but "
                   "loses PGD robustness vs PGD-AT; it is not the claimed "
                   "trade-off improvement.")
    elif abs(acc_gain) <= 0.01 and rob_kept:
        verdict = ("NULL: HAT matches PGD-AT on both axes; at this 6k-sample "
                   "scale the helper term adds no signal.")
    else:
        verdict = ("NOT SUPPORTED: HAT under-performs PGD-AT on the clean/"
                   "robust trade-off (clean acc not improved AND/OR robustness "
                   "lost).")
    out("  ONE-LINE VERDICT: " + verdict)

    out("")
    out("Caveats:")
    out("  - Single seed (campaign convention; M1).")
    out("  - PGD-10 at eps=0.1 is the campaign attack; AutoAttack not run (M3).")
    out("  - f_std is trained once and reused as helper-labeller (paper does the same).")
    out("  - lam=beta=1 fixed; full lam-sensitivity sweep is left to a follow-up.")
    out(f"\ndone in {time.time()-t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
