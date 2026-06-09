"""
H440 - Class-Balanced TRADES: fix worst-class PGD ASR via per-class beta.

Gaps addressed: G3 (TRADES variants missing) + M6 (mean-only reporting, no
worst-class).

Background (campaign-internal):
  H252 establishes per-class FGSM ASR on Fashion-MNIST is highly asymmetric
  (Shirt ~1.00, Trouser ~0.30) on a standard CNN. Mean ASR is the headline
  number everywhere in H173-H413; nothing reports worst-class. PGD-ASR is
  often saturated at eps=0.1 (M9) on STD models, but TRADES/AT models bring
  it down to ~0.33 mean and there per-class signal returns.

Hypothesis:
  A class-balanced TRADES that puts MORE robustness pressure on currently
  under-robust classes (via a per-class beta in the KL term) will REDUCE
  worst-class PGD-ASR for free or with a tolerable hit to mean-ASR. Reporting
  only mean-ASR (campaign default) hides this lever entirely.

Conditions (all PGD-10 inner, eps=0.1, alpha=0.01, KL-on-adv outer):
  C0  baseline           : standard CE training (beta=0)
  C1  TRADES-uniform     : per-class beta_c = 6 for all c (replicates H304)
  C2  CB-TRADES-invfreq  : beta_c proportional to inverse class frequency in
                            Xtr; the rarer the class in the (sub-sampled) 6k
                            train set, the higher its KL weight. Normalised so
                            mean(beta_c) = 6.
  C3  CB-TRADES-online   : every epoch, refresh beta_c from current PGD-ASR
                            per class on a small probe set. Higher PGD-ASR
                            -> higher beta_c. Smoothing alpha=0.5 between
                            epochs. Mean(beta_c) re-normalised to 6 each epoch.

Per-class beta is applied by multiplying the per-sample KL term by
beta[y_i] before the batch mean, i.e. the TRADES objective becomes
    CE(f(x), y) + mean_i( beta[y_i] * KL( f(x_adv_i) || f(x_i) ) )
where x_adv_i is the standard TRADES inner-max sample (PGD on KL).

Reporting (all conditions):
  - mean clean accuracy
  - mean FGSM ASR
  - mean PGD ASR
  - per-class clean acc, FGSM ASR, PGD ASR (10 classes Fashion-MNIST)
  - worst-class PGD ASR (the headline metric for this hypothesis)
  - max-min gap = (worst PGD ASR) - (best PGD ASR)  (lower = fairer)
  - transfer-attack masking check: take PGD adversarials crafted on the
    BASELINE model (C0) and evaluate them on each defended model. If a
    "defended" model only resists its OWN PGD but not transferred PGD, it
    is masking gradients rather than being robust (Athalye 2018).

Standard config (campaign defaults):
  N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9, wd=5e-4),
  SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.

Verdict (both thresholds must be met for YES):
  - mean PGD-ASR not worse than TRADES-uniform by more than +0.05, AND
  - worst-class PGD-ASR strictly lower than TRADES-uniform by >= 0.05.

External papers anchoring the design:
  - Zhang et al. 2019 (TRADES, ICML)            -- uniform-beta baseline.
  - Xu et al. 2021 "To be Robust or to be Fair" -- robust-fairness problem;
    per-class variance arg.
  - Wei et al. 2023 CFA (CVPR)                  -- per-class calibrated AT.
  - Li & Liu 2023 WAT (AAAI)                    -- worst-class min-max AT.
  - Sun et al. 2023 BAT                          -- dynamic per-class strength.

ASCII output only. Single GPU, deterministic seed. No execution here.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# -------------------- config --------------------
DS         = "fashion_mnist"
N_TRAIN    = 6000
N_EVAL     = 2000
EPOCHS     = 10
LR         = 0.05
BATCH      = 128
SEED       = 0
EPS        = 0.1
PGD_STEPS  = 10
PGD_ALPHA  = 0.01
BETA_MEAN  = 6.0           # target mean of per-class betas (matches H304 beta=6)
N_CLASSES  = 10
PROBE_PER_CLASS = 32        # tiny probe used to refresh online betas each epoch
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot"]
META = {"channels": 1, "size": 28, "n_classes": N_CLASSES}

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h440_class_balanced_trades_output.txt",
)


# -------------------- TRADES inner-max (KL) --------------------
def pgd_on_kl(model, x, steps=PGD_STEPS, eps=EPS, alpha=PGD_ALPHA):
    """Standard TRADES inner step: PGD that maximises KL( f(x_adv) || f(x) )."""
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)
    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(0, 1)
    x0 = x.detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logp_adv = F.log_softmax(model(x_adv), dim=1)
        kl = F.kl_div(logp_adv, p_clean, reduction="batchmean")
        g, = torch.autograd.grad(kl, x_adv)
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1)
    if model_was_training:
        model.train()
    return x_adv.detach()


# -------------------- training: per-class beta TRADES --------------------
def make_opt(model):
    return torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=5e-4,
    )


def per_class_pgd_asr(model, Xp, Yp):
    """PGD ASR on a small probe set, computed per class. Returns np(N_CLASSES,)."""
    model.eval()
    Xa = C.pgd(model, Xp, Yp, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    with torch.no_grad():
        pa = model(Xa).argmax(1).cpu().numpy()
    yp = Yp.cpu().numpy()
    flip = (pa != yp).astype(np.float64)
    out = np.zeros(N_CLASSES, dtype=np.float64)
    for c in range(N_CLASSES):
        m = (yp == c)
        out[c] = flip[m].mean() if m.sum() else 0.0
    return out


def renormalise_betas(beta_c, target_mean=BETA_MEAN, floor=0.5):
    """Floor + renormalise so mean(beta_c) == target_mean."""
    b = np.maximum(beta_c.astype(np.float64), floor)
    b = b * (target_mean / b.mean())
    return b


def build_probe_set(Xtr, Ytr, per_class=PROBE_PER_CLASS, seed=SEED):
    """Take up to `per_class` examples from each class out of Xtr/Ytr."""
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    yn = Ytr.cpu().numpy()
    idx_all = []
    for c in range(N_CLASSES):
        ids = np.where(yn == c)[0]
        if len(ids) == 0:
            continue
        take = min(per_class, len(ids))
        perm = torch.randperm(len(ids), generator=g).numpy()[:take]
        idx_all.append(ids[perm])
    idx = np.concatenate(idx_all)
    idx_t = torch.as_tensor(idx, device=Xtr.device, dtype=torch.long)
    return Xtr[idx_t], Ytr[idx_t]


def train_cb_trades(Xtr, Ytr, mode, log_fn=print):
    """
    mode: one of
      "baseline" - standard CE
      "uniform"  - TRADES with constant beta = BETA_MEAN
      "invfreq"  - per-class beta proportional to 1 / class-frequency
      "online"   - per-class beta refreshed each epoch from current PGD ASR
    """
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, seed=SEED)
    opt = make_opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # initial per-class beta
    if mode == "baseline":
        beta_c = np.zeros(N_CLASSES, dtype=np.float64)
    elif mode == "uniform":
        beta_c = np.full(N_CLASSES, BETA_MEAN, dtype=np.float64)
    elif mode == "invfreq":
        counts = np.bincount(Ytr.cpu().numpy(), minlength=N_CLASSES).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        inv = 1.0 / counts
        beta_c = renormalise_betas(inv, target_mean=BETA_MEAN)
    elif mode == "online":
        beta_c = np.full(N_CLASSES, BETA_MEAN, dtype=np.float64)
    else:
        raise ValueError(mode)

    log_fn(f"    initial beta_c = {np.round(beta_c, 3).tolist()}")

    # probe set for online beta updates
    Xp, Yp = build_probe_set(Xtr, Ytr) if mode == "online" else (None, None)
    beta_history = [beta_c.copy()]

    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            if mode == "baseline":
                loss = F.cross_entropy(model(xb), yb)
            else:
                # TRADES inner-max
                x_adv = pgd_on_kl(model, xb, steps=PGD_STEPS, eps=EPS, alpha=PGD_ALPHA)
                model.train()
                out_clean = model(xb)
                out_adv = model(x_adv)
                loss_ce = F.cross_entropy(out_clean, yb)
                # per-sample KL ( f(x_adv_i) || f(x_i) ) -- not reduction="batchmean"
                logp_adv = F.log_softmax(out_adv, dim=1)
                p_clean = F.softmax(out_clean, dim=1)
                kl_per = (p_clean * (p_clean.clamp_min(1e-12).log() - logp_adv)).sum(dim=1)
                # per-sample weight = beta[y_i]
                w = torch.as_tensor(
                    beta_c[yb.cpu().numpy()],
                    device=xb.device, dtype=kl_per.dtype,
                )
                loss_kl = (w * kl_per).mean()
                loss = loss_ce + loss_kl
            loss.backward()
            opt.step()
        sched.step()

        # online beta update (uses last-epoch model state)
        if mode == "online" and ep < EPOCHS - 1:
            with torch.enable_grad():
                # need grads through model for PGD
                for p in model.parameters():
                    p.requires_grad_(True)
                asr_c = per_class_pgd_asr(model, Xp, Yp)
            # ema between previous beta and new beta-from-asr; smoothing 0.5
            raw = asr_c + 1e-3       # avoid zeros
            new_b = renormalise_betas(raw, target_mean=BETA_MEAN)
            beta_c = 0.5 * beta_c + 0.5 * new_b
            beta_c = renormalise_betas(beta_c, target_mean=BETA_MEAN)
            beta_history.append(beta_c.copy())
            log_fn(f"    [online] ep{ep+1} probe PGD_ASR per class = "
                   f"{np.round(asr_c, 3).tolist()}")
            log_fn(f"    [online] ep{ep+1} new beta_c             = "
                   f"{np.round(beta_c, 3).tolist()}")

    model.eval()
    return model, beta_c, beta_history


# -------------------- evaluation --------------------
def eval_full(model, Xte, Yte):
    """Clean / FGSM / PGD ASR overall AND per-class. Returns dict."""
    # ensure grad-enabled params for attacks
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()

    # clean predictions
    with torch.no_grad():
        clean_logits = []
        for i in range(0, Xte.size(0), 256):
            clean_logits.append(model(Xte[i:i + 256]).cpu())
        clean_logits = torch.cat(clean_logits)
    clean_pred = clean_logits.argmax(1).numpy()

    # FGSM adv
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    with torch.no_grad():
        fgsm_logits = []
        for i in range(0, Xfgsm.size(0), 256):
            fgsm_logits.append(model(Xfgsm[i:i + 256]).cpu())
        fgsm_logits = torch.cat(fgsm_logits)
    fgsm_pred = fgsm_logits.argmax(1).numpy()

    # PGD adv
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    with torch.no_grad():
        pgd_logits = []
        for i in range(0, Xpgd.size(0), 256):
            pgd_logits.append(model(Xpgd[i:i + 256]).cpu())
        pgd_logits = torch.cat(pgd_logits)
    pgd_pred = pgd_logits.argmax(1).numpy()

    y = Yte.cpu().numpy()
    clean_correct = (clean_pred == y)
    fgsm_flip = (fgsm_pred != y).astype(np.float64)
    pgd_flip = (pgd_pred != y).astype(np.float64)

    per_class = []
    for c in range(N_CLASSES):
        m = (y == c)
        n = int(m.sum())
        if n == 0:
            per_class.append({"n": 0, "clean": float("nan"),
                              "fgsm_asr": float("nan"), "pgd_asr": float("nan")})
            continue
        per_class.append({
            "n": n,
            "clean":    float(clean_correct[m].mean()),
            "fgsm_asr": float(fgsm_flip[m].mean()),
            "pgd_asr":  float(pgd_flip[m].mean()),
        })

    return {
        "clean_acc": float(clean_correct.mean()),
        "fgsm_asr":  float(fgsm_flip.mean()),
        "pgd_asr":   float(pgd_flip.mean()),
        "per_class": per_class,
        "Xpgd":      Xpgd,    # kept so we can run transfer check later
    }


def eval_transfer(model, X_adv_from_baseline, Yte):
    """Evaluate the model on adversarial inputs CRAFTED ON the baseline.
    Returns ASR (fraction flipped vs original Yte)."""
    model.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, X_adv_from_baseline.size(0), 256):
            logits.append(model(X_adv_from_baseline[i:i + 256]).cpu())
        logits = torch.cat(logits)
    pred = logits.argmax(1).numpy()
    y = Yte.cpu().numpy()
    return float((pred != y).mean())


# -------------------- runner --------------------
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(str(s))

    def flush_file():
        with open(OUT_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")

    t0 = time.time()
    out("=" * 80)
    out("H440  Class-Balanced TRADES (per-class beta) -- Fashion-MNIST")
    out("=" * 80)
    out(f"config: DS={DS} N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} "
        f"LR={LR} BATCH={BATCH}")
    out(f"        SGD(mom=0.9, wd=5e-4) SEED={SEED}  EPS={EPS} "
        f"PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        BETA_MEAN={BETA_MEAN}  N_CLASSES={N_CLASSES}  "
        f"PROBE_PER_CLASS={PROBE_PER_CLASS}")
    out(f"        device={C.DEVICE}")
    out("")

    # --- data ---
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data:   Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    counts = np.bincount(Ytr.cpu().numpy(), minlength=N_CLASSES)
    out(f"        train class counts = {counts.tolist()}")
    out("")

    conditions = [
        ("baseline", "C0 baseline (CE only, beta=0)"),
        ("uniform",  "C1 TRADES uniform (beta=6)"),
        ("invfreq",  "C2 CB-TRADES inv-frequency"),
        ("online",   "C3 CB-TRADES online per-class PGD-ASR"),
    ]

    results = {}            # mode -> eval dict
    final_beta = {}         # mode -> np.array
    baseline_Xpgd = None     # for transfer check

    for mode, label in conditions:
        out("-" * 80)
        out(f"[train] {label}")
        out("-" * 80)
        ts = time.time()
        model, beta_c, beta_hist = train_cb_trades(Xtr, Ytr, mode, log_fn=out)
        train_time = time.time() - ts
        out(f"    final beta_c = {np.round(beta_c, 3).tolist()}")
        out(f"    train time   = {train_time:.1f}s")

        out(f"[eval]  {label}")
        ev = eval_full(model, Xte, Yte)
        results[mode] = ev
        final_beta[mode] = beta_c
        if mode == "baseline":
            baseline_Xpgd = ev["Xpgd"]

        out(f"    clean_acc={ev['clean_acc']:.4f}  "
            f"FGSM_ASR={ev['fgsm_asr']:.4f}  PGD_ASR={ev['pgd_asr']:.4f}")
        per_pgd = [pc["pgd_asr"] for pc in ev["per_class"]]
        out(f"    per-class PGD_ASR = "
            f"{[round(v, 3) for v in per_pgd]}")
        out(f"    worst-class PGD_ASR = {max(per_pgd):.4f} "
            f"({CLASS_NAMES[int(np.argmax(per_pgd))]})")
        out(f"    best-class  PGD_ASR = {min(per_pgd):.4f} "
            f"({CLASS_NAMES[int(np.argmin(per_pgd))]})")
        out(f"    max-min gap         = {max(per_pgd) - min(per_pgd):.4f}")
        out("")
        flush_file()       # persist after every condition

    # ---------- per-class table ----------
    out("=" * 80)
    out("[per-class table]  PGD ASR (eps=0.1, 10 steps)")
    out("=" * 80)
    hdr = "{:<14}".format("class") + "".join(
        " {:>10}".format(c.split()[0]) for c, _ in [(m, l) for m, l in conditions]
    )
    # cleaner header
    hdr = "{:<14}".format("class") + "".join(
        " {:>10}".format(mode) for mode, _ in conditions
    )
    out(hdr)
    out("-" * len(hdr))
    for c in range(N_CLASSES):
        row = "{:<14}".format(CLASS_NAMES[c])
        for mode, _ in conditions:
            v = results[mode]["per_class"][c]["pgd_asr"]
            row += " {:>10.4f}".format(v)
        out(row)
    out("-" * len(hdr))
    row = "{:<14}".format("MEAN")
    for mode, _ in conditions:
        row += " {:>10.4f}".format(results[mode]["pgd_asr"])
    out(row)
    row = "{:<14}".format("WORST")
    for mode, _ in conditions:
        per_pgd = [pc["pgd_asr"] for pc in results[mode]["per_class"]]
        row += " {:>10.4f}".format(max(per_pgd))
    out(row)
    row = "{:<14}".format("BEST")
    for mode, _ in conditions:
        per_pgd = [pc["pgd_asr"] for pc in results[mode]["per_class"]]
        row += " {:>10.4f}".format(min(per_pgd))
    out(row)
    row = "{:<14}".format("MAX-MIN")
    for mode, _ in conditions:
        per_pgd = [pc["pgd_asr"] for pc in results[mode]["per_class"]]
        row += " {:>10.4f}".format(max(per_pgd) - min(per_pgd))
    out(row)
    out("")

    # ---------- per-class clean & FGSM tables ----------
    for metric_key, metric_label in [("clean", "CLEAN ACC"),
                                     ("fgsm_asr", "FGSM ASR")]:
        out("=" * 80)
        out(f"[per-class table]  {metric_label}")
        out("=" * 80)
        out(hdr)
        out("-" * len(hdr))
        for c in range(N_CLASSES):
            row = "{:<14}".format(CLASS_NAMES[c])
            for mode, _ in conditions:
                v = results[mode]["per_class"][c][metric_key]
                row += " {:>10.4f}".format(v)
            out(row)
        out("")

    # ---------- transfer-attack masking check ----------
    out("=" * 80)
    out("[transfer]  PGD adversarials crafted on BASELINE evaluated on each model")
    out("=" * 80)
    out(f"{'model':<14} {'self_PGD_ASR':>14} {'transfer_PGD_ASR':>18} "
        f"{'flag':>22}")
    out("-" * 72)
    for mode, _ in conditions:
        self_asr = results[mode]["pgd_asr"]
        trans_asr = eval_transfer(_dummy_load(results[mode]), baseline_Xpgd, Yte) \
            if False else None
        # the model object is gone; we kept Xpgd. Just compute transfer using the
        # baseline adv inputs and the per-mode model's predictions, which means
        # we need the model. Workaround: store predictions during eval_full.
        # -> we recompute transfer_asr inline since `results[mode]` does not
        # carry the model. Cleaner: store transfer asr in eval_full by passing
        # baseline_Xpgd in. We do that here instead with a re-eval branch.
        trans_asr = -1.0
        # placeholder, will overwrite in next block
        out(f"{mode:<14} {self_asr:>14.4f} {'(see below)':>18} {'':>22}")
    out("")

    # Re-run a clean transfer-eval that does have the model. We retrain
    # quickly is too expensive; instead we kept Xpgd from baseline and we
    # have the eval_full output 'Xpgd' per mode. Transfer asr is computed by
    # feeding BASELINE Xpgd into each defended model -- but we no longer hold
    # the defended models. To avoid that loss we restructure: keep models
    # in `keep_models` during the main loop.
    # (The block above is left for narrative; the real numbers come from the
    # second pass below, which is what gets reported.)
    pass

    # ---------- VERDICT ----------
    out("=" * 80)
    out("[VERDICT]")
    out("=" * 80)
    uni = results["uniform"]
    uni_per = [pc["pgd_asr"] for pc in uni["per_class"]]
    uni_worst = max(uni_per)
    uni_mean  = uni["pgd_asr"]

    for mode in ("invfreq", "online"):
        per = [pc["pgd_asr"] for pc in results[mode]["per_class"]]
        worst = max(per)
        mean_ = results[mode]["pgd_asr"]
        d_mean = mean_ - uni_mean       # negative = better
        d_worst = worst - uni_worst      # negative = better
        cond_mean_ok  = (d_mean <= 0.05)
        cond_worst_ok = (d_worst <= -0.05)
        verdict = "YES" if (cond_mean_ok and cond_worst_ok) else (
            "PARTIAL" if cond_worst_ok else "NO"
        )
        out(f"  {mode}: mean PGD_ASR {uni_mean:.4f} -> {mean_:.4f}  "
            f"({d_mean:+.4f})  | worst PGD_ASR {uni_worst:.4f} -> {worst:.4f}  "
            f"({d_worst:+.4f})  -> {verdict}")

    out("")
    out("thresholds: mean must not worsen by > +0.05; worst must improve by "
        ">= 0.05 to count as YES.")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_PATH}")


def _dummy_load(_):
    # placeholder so the file parses; real transfer step is performed in main()
    return None


# -------------------- patched main with model retention + transfer ----------
def main_with_transfer():
    """Replaces main() but retains models across conditions so we can run a
    cross-model transfer-attack check. This is the entry point invoked by
    __main__.
    """
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(str(s))

    def flush_file():
        with open(OUT_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")

    t0 = time.time()
    out("=" * 80)
    out("H440  Class-Balanced TRADES (per-class beta) -- Fashion-MNIST")
    out("=" * 80)
    out(f"config: DS={DS} N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} "
        f"LR={LR} BATCH={BATCH}")
    out(f"        SGD(mom=0.9, wd=5e-4) SEED={SEED}  EPS={EPS} "
        f"PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        BETA_MEAN={BETA_MEAN}  N_CLASSES={N_CLASSES}  "
        f"PROBE_PER_CLASS={PROBE_PER_CLASS}")
    out(f"        device={C.DEVICE}")
    out("")
    out("conditions:")
    out("  C0 baseline           - standard CE (beta=0)")
    out("  C1 TRADES uniform     - beta_c = 6 for all c")
    out("  C2 CB-TRADES invfreq  - beta_c ~ 1/freq(c) (mean=6)")
    out("  C3 CB-TRADES online   - beta_c refreshed from per-class PGD ASR")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data:   Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    counts = np.bincount(Ytr.cpu().numpy(), minlength=N_CLASSES)
    out(f"        train class counts = {counts.tolist()}")
    out("")

    conditions = [
        ("baseline", "C0 baseline (CE only, beta=0)"),
        ("uniform",  "C1 TRADES uniform (beta=6)"),
        ("invfreq",  "C2 CB-TRADES inv-frequency"),
        ("online",   "C3 CB-TRADES online per-class PGD-ASR"),
    ]

    models = {}
    results = {}
    final_beta = {}

    for mode, label in conditions:
        out("-" * 80)
        out(f"[train] {label}")
        out("-" * 80)
        ts = time.time()
        model, beta_c, beta_hist = train_cb_trades(Xtr, Ytr, mode, log_fn=out)
        train_time = time.time() - ts
        out(f"    final beta_c = {np.round(beta_c, 3).tolist()}")
        out(f"    train time   = {train_time:.1f}s")

        out(f"[eval]  {label}")
        ev = eval_full(model, Xte, Yte)
        results[mode] = ev
        final_beta[mode] = beta_c
        models[mode] = model

        per_pgd = [pc["pgd_asr"] for pc in ev["per_class"]]
        out(f"    clean_acc={ev['clean_acc']:.4f}  "
            f"FGSM_ASR={ev['fgsm_asr']:.4f}  PGD_ASR={ev['pgd_asr']:.4f}")
        out(f"    per-class PGD_ASR = {[round(v, 3) for v in per_pgd]}")
        out(f"    worst-class PGD_ASR = {max(per_pgd):.4f} "
            f"({CLASS_NAMES[int(np.argmax(per_pgd))]})")
        out(f"    best-class  PGD_ASR = {min(per_pgd):.4f} "
            f"({CLASS_NAMES[int(np.argmin(per_pgd))]})")
        out(f"    max-min gap         = {max(per_pgd) - min(per_pgd):.4f}")
        out("")
        flush_file()

    baseline_Xpgd = results["baseline"]["Xpgd"]

    # ----- per-class PGD table -----
    out("=" * 80)
    out("[per-class table]  PGD ASR (eps=0.1, 10 steps)")
    out("=" * 80)
    hdr = "{:<14}".format("class") + "".join(
        " {:>10}".format(mode) for mode, _ in conditions
    )
    out(hdr)
    out("-" * len(hdr))
    for c in range(N_CLASSES):
        row = "{:<14}".format(CLASS_NAMES[c])
        for mode, _ in conditions:
            row += " {:>10.4f}".format(results[mode]["per_class"][c]["pgd_asr"])
        out(row)
    out("-" * len(hdr))
    for stat_name, fn in [
        ("MEAN",    lambda mode: results[mode]["pgd_asr"]),
        ("WORST",   lambda mode: max(pc["pgd_asr"] for pc in results[mode]["per_class"])),
        ("BEST",    lambda mode: min(pc["pgd_asr"] for pc in results[mode]["per_class"])),
        ("MAX-MIN", lambda mode: (max(pc["pgd_asr"] for pc in results[mode]["per_class"])
                                  - min(pc["pgd_asr"] for pc in results[mode]["per_class"]))),
    ]:
        row = "{:<14}".format(stat_name)
        for mode, _ in conditions:
            row += " {:>10.4f}".format(fn(mode))
        out(row)
    out("")

    # ----- per-class CLEAN ACC and FGSM ASR tables -----
    for metric_key, metric_label in [("clean", "CLEAN ACC"),
                                     ("fgsm_asr", "FGSM ASR")]:
        out("=" * 80)
        out(f"[per-class table]  {metric_label}")
        out("=" * 80)
        out(hdr)
        out("-" * len(hdr))
        for c in range(N_CLASSES):
            row = "{:<14}".format(CLASS_NAMES[c])
            for mode, _ in conditions:
                v = results[mode]["per_class"][c][metric_key]
                row += " {:>10.4f}".format(v)
            out(row)
        row = "{:<14}".format("MEAN")
        for mode, _ in conditions:
            vals = [pc[metric_key] for pc in results[mode]["per_class"]
                    if not (isinstance(pc[metric_key], float) and np.isnan(pc[metric_key]))]
            row += " {:>10.4f}".format(float(np.mean(vals)))
        out(row)
        out("")

    # ----- final beta_c table -----
    out("=" * 80)
    out("[final beta_c] per condition")
    out("=" * 80)
    out("{:<14}".format("class") + "".join(
        " {:>10}".format(mode) for mode, _ in conditions))
    out("-" * len(hdr))
    for c in range(N_CLASSES):
        row = "{:<14}".format(CLASS_NAMES[c])
        for mode, _ in conditions:
            row += " {:>10.3f}".format(float(final_beta[mode][c]))
        out(row)
    out("")

    # ----- transfer check: PGD-on-baseline evaluated on every model -----
    out("=" * 80)
    out("[transfer-attack masking check]")
    out("=" * 80)
    out("PGD adversarials are crafted on the BASELINE model (C0). They are then")
    out("evaluated on each model. If self_PGD_ASR << transfer_PGD_ASR the model")
    out("is masking gradients (white-box only); we expect honest defences to")
    out("have transfer_PGD_ASR ~ self_PGD_ASR.")
    out("")
    out(f"{'model':<14} {'self_PGD_ASR':>14} {'transfer_PGD_ASR':>18} "
        f"{'gap (self - tr)':>18}  flag")
    out("-" * 80)
    for mode, _ in conditions:
        self_asr = results[mode]["pgd_asr"]
        tr_asr = eval_transfer(models[mode], baseline_Xpgd, Yte)
        gap = self_asr - tr_asr
        # heuristic flag: if self << transfer (gap very negative), masking
        flag = "MASKING?" if gap < -0.10 else "ok"
        out(f"{mode:<14} {self_asr:>14.4f} {tr_asr:>18.4f} {gap:>18.4f}  {flag}")
    out("")

    # ----- VERDICT -----
    out("=" * 80)
    out("[VERDICT]")
    out("=" * 80)
    uni_per = [pc["pgd_asr"] for pc in results["uniform"]["per_class"]]
    uni_worst = max(uni_per)
    uni_mean  = results["uniform"]["pgd_asr"]
    out(f"  reference: TRADES-uniform  mean PGD_ASR = {uni_mean:.4f}  "
        f"worst = {uni_worst:.4f}")

    summary_lines = []
    for mode in ("invfreq", "online"):
        per = [pc["pgd_asr"] for pc in results[mode]["per_class"]]
        worst = max(per)
        mean_ = results[mode]["pgd_asr"]
        d_mean  = mean_ - uni_mean
        d_worst = worst - uni_worst
        cond_mean_ok  = (d_mean <= 0.05)
        cond_worst_ok = (d_worst <= -0.05)
        if cond_mean_ok and cond_worst_ok:
            verdict = "YES"
        elif cond_worst_ok:
            verdict = "PARTIAL (worst improved, mean degraded > 0.05)"
        elif cond_mean_ok and d_worst < 0:
            verdict = "WEAK (worst improved < 0.05)"
        else:
            verdict = "NO"
        summary_lines.append(
            f"  {mode}: mean {uni_mean:.4f} -> {mean_:.4f} ({d_mean:+.4f}) | "
            f"worst {uni_worst:.4f} -> {worst:.4f} ({d_worst:+.4f}) -> {verdict}"
        )
    for s in summary_lines:
        out(s)
    out("")
    out("decision rule:")
    out("  YES     := mean PGD_ASR does not worsen by > +0.05 AND")
    out("            worst-class PGD_ASR improves by >= 0.05 vs TRADES-uniform.")
    out("  PARTIAL := worst-class improves >= 0.05 but mean worsens > 0.05.")
    out("  WEAK    := worst-class improves but by < 0.05.")
    out("  NO      := worst-class does not improve.")
    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main_with_transfer()
