"""
H404 - Vulnerable-sample noise hardening: extend H403's single-sample noise
augmentation to MANY samples, targeting the MOST adversarially vulnerable
training images, and ask whether this buys robustness WITHOUT the clean-accuracy
cost of adversarial training (AT).

Background (H403): augmenting training with K noisy copies (L-inf eps=0.1,
true-class labelled) of a SINGLE image pushed that image's nearest adversarial
~1.8x further away while clean test acc stayed ~0.88 (no degradation). True
PGD-AT gets strong robustness (PGD ASR ~0.33) but costs ~11% clean accuracy
(0.88 -> 0.77).

Hypothesis: apply noise-copy hardening to MANY samples -- specifically the most
adversarially vulnerable ones (lowest input-space margin) -- to get a model that
is BOTH robust AND generalised (keeps high clean accuracy, unlike AT).

Design:
  1. Train a BASE model from scratch; record clean acc, FGSM ASR, PGD ASR.
  2. Rank TRAINING samples by vulnerability = per-sample margin (lowest = most
     vulnerable).
  3. Noise helper: K copies = (x + 0.1*(2*rand-1)).clamp(0,1), true-class label.
  4. MAIN SWEEP: fix K=50, vary n_target in {0,250,500,1000,2000} most-vulnerable
     training samples. Evaluate clean/FGSM/PGD ASR on test.
  5. CONTROL: n_target=1000 but pick the 1000 SAFEST (highest-margin) samples.
  6. MATCHED-BUDGET UNIFORM: distribute the largest budget (2000*50=100000 extra
     copies) UNIFORMLY across all 6000 train samples.
  7. TRUE AT reference: adv_train=True, adv_eps=0.1, adv_steps=7.
  8. VULNERABLE TEST SUBSET: 25% lowest-margin TEST samples (by base model);
     report FGSM/PGD ASR restricted to them, baseline vs best targeted condition.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD mom=0.9 wd=5e-4,
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01. CNN width=32.
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
K_COPIES = 50               # copies per targeted sample in the main sweep
N_TARGETS = [0, 250, 500, 1000, 2000]
AT_EPS = 0.1
AT_STEPS = 7
VULN_TEST_FRAC = 0.25       # lowest-margin fraction of test for the subset metric

META = {"channels": 1, "size": 28, "n_classes": 10}


def _make_optimizer_sgd(model, lr):
    """SGD mom=0.9 wd=5e-4 (config); common.make_optimizer uses wd=1e-4."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def train_std(Xtr, Ytr, seed):
    """Train a fresh CNN from scratch, standard SGD, EPOCHS epochs."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
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


def train_at(Xtr, Ytr, seed):
    """True PGD adversarial training via common.train_model."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    # common.train_model builds SGD with wd=1e-4 (the 'sgd' opt); use it as the
    # canonical AT reference per the task spec.
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, opt="sgd",
                         lr=LR, adv_train=True, adv_eps=AT_EPS, adv_steps=AT_STEPS)


def make_noisy_copies_for(Xsrc, Ysrc, k, eps, seed):
    """For each row of Xsrc make k noisy copies = (x + eps*(2*rand-1)).clamp(0,1)
    with fresh noise per copy; label = that source's true label.
    Returns (Xc, Yc) on the source device. If k==0 or Xsrc empty -> (None,None)."""
    n = Xsrc.size(0)
    if k == 0 or n == 0:
        return None, None
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = Xsrc.detach().cpu().repeat_interleave(k, dim=0)   # (n*k, C,H,W)
    noise = (2.0 * torch.rand(base.shape, generator=g) - 1.0) * eps
    Xc = (base + noise).clamp(0, 1).to(Xsrc.device)
    Yc = Ysrc.detach().repeat_interleave(k, dim=0).to(Xsrc.device)
    return Xc, Yc


def make_noisy_copies_total(Xsrc, Ysrc, total, eps, seed):
    """Make exactly `total` noisy copies spread uniformly (round-robin) over all
    rows of Xsrc. label = source's true label. Returns (Xc, Yc)."""
    n = Xsrc.size(0)
    if total == 0 or n == 0:
        return None, None
    sel = (torch.arange(total) % n)                          # round-robin indices
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = Xsrc.detach().cpu()[sel]                          # (total, C,H,W)
    noise = (2.0 * torch.rand(base.shape, generator=g) - 1.0) * eps
    Xc = (base + noise).clamp(0, 1).to(Xsrc.device)
    Yc = Ysrc.detach().cpu()[sel].to(Xsrc.device)
    return Xc, Yc


def eval_robustness(model, X, Y):
    """clean acc, FGSM ASR, PGD ASR on (X,Y)."""
    _, acc = C.logits_and_acc(model, X, Y)
    fg = C.attack_success(model, X, Y, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


def eval_subset_asr(model, X, Y, mask):
    """FGSM/PGD ASR restricted to the boolean subset `mask`."""
    Xs, Ys = X[mask], Y[mask]
    fg = C.attack_success(model, Xs, Ys, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xs, Ys, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return fg["asr"], pg["asr"]


def main():
    t0 = time.time()
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 80)
    out("H404  Vulnerable-sample noise hardening (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} "
        f"K_COPIES={K_COPIES}")
    out(f"        n_target sweep = {N_TARGETS}")
    out(f"        AT: adv_eps={AT_EPS} adv_steps={AT_STEPS}")
    out(f"        vulnerable-test subset = lowest {int(VULN_TEST_FRAC*100)}% margin")
    out(f"        device = {C.DEVICE}")
    out("")

    # ---- data & base model ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    out("\n[1] training BASE model (standard, no augmentation)...")
    base = train_std(Xtr, Ytr, SEED)
    base_acc, base_fgsm, base_pgd = eval_robustness(base, Xte, Yte)
    out(f"    base clean acc={base_acc:.4f}  FGSM_ASR={base_fgsm:.4f}  "
        f"PGD_ASR={base_pgd:.4f}")

    # ---- rank training samples by vulnerability (margin) ----
    out("\n[2] ranking TRAINING samples by vulnerability (input-space margin)...")
    tr_margin = C.margin(base, Xtr, Ytr)                    # numpy (N,)
    order_low = np.argsort(tr_margin)                       # ascending: most vuln first
    order_high = order_low[::-1].copy()                     # descending: safest first
    out(f"    train margin: min={tr_margin.min():.3f} median="
        f"{np.median(tr_margin):.3f} max={tr_margin.max():.3f}")
    out(f"    most-vulnerable (lowest margin) sample margins[:5]: "
        f"{np.sort(tr_margin)[:5].round(3).tolist()}")

    # ---- vulnerable TEST subset (by base model) ----
    te_margin = C.margin(base, Xte, Yte)
    n_vuln_te = int(round(VULN_TEST_FRAC * Xte.size(0)))
    vuln_te_idx = np.argsort(te_margin)[:n_vuln_te]
    vuln_te_mask = np.zeros(Xte.size(0), dtype=bool)
    vuln_te_mask[vuln_te_idx] = True
    out(f"    vulnerable test subset: {n_vuln_te} samples "
        f"(margin <= {np.sort(te_margin)[n_vuln_te-1]:.3f})")

    # ---- container for the main results table ----
    # each entry: dict(condition, n_target, aug_size, acc, fgsm, pgd, model_tag)
    rows = []
    # keep models we need later for subset eval
    keep_models = {}

    def add_row(cond, n_target, aug_size, acc, fgsm, pgd):
        rows.append({"cond": cond, "n_target": n_target, "aug_size": aug_size,
                     "acc": acc, "fgsm": fgsm, "pgd": pgd})

    # ---- MAIN SWEEP over n_target (vulnerable) ----
    out("\n[3] MAIN SWEEP: K=50 noisy copies of the n_target most-vulnerable "
        "training samples")
    for nt in N_TARGETS:
        if nt == 0:
            acc, fgsm, pgd = base_acc, base_fgsm, base_pgd
            aug = N_TRAIN
            cond = "baseline"
            keep_models["baseline"] = base
        else:
            sel_idx = order_low[:nt]
            Xsrc = Xtr[torch.as_tensor(sel_idx, device=Xtr.device, dtype=torch.long)]
            Ysrc = Ytr[torch.as_tensor(sel_idx, device=Xtr.device, dtype=torch.long)]
            nseed = SEED * 1_000_000 + nt
            Xc, Yc = make_noisy_copies_for(Xsrc, Ysrc, K_COPIES, EPS, nseed)
            Xaug = torch.cat([Xtr, Xc], dim=0)
            Yaug = torch.cat([Ytr, Yc], dim=0)
            aug = Xaug.size(0)
            model = train_std(Xaug, Yaug, SEED)
            acc, fgsm, pgd = eval_robustness(model, Xte, Yte)
            cond = f"n={nt} (vulnerable)"
            keep_models[f"vuln_{nt}"] = model
        add_row(cond, nt, aug, acc, fgsm, pgd)
        out(f"    {cond:<22} aug_size={aug:>7}  acc={acc:.4f}  "
            f"FGSM={fgsm:.4f}  PGD={pgd:.4f}  ({time.time()-t0:.0f}s)")

    # ---- CONTROL: n=1000 safest (highest margin) ----
    out("\n[4] CONTROL: K=50 copies of the 1000 SAFEST (highest-margin) samples")
    nt = 1000
    sel_idx = order_high[:nt]
    Xsrc = Xtr[torch.as_tensor(sel_idx, device=Xtr.device, dtype=torch.long)]
    Ysrc = Ytr[torch.as_tensor(sel_idx, device=Xtr.device, dtype=torch.long)]
    Xc, Yc = make_noisy_copies_for(Xsrc, Ysrc, K_COPIES, EPS, SEED * 1_000_000 + 99999)
    Xaug = torch.cat([Xtr, Xc], dim=0)
    Yaug = torch.cat([Ytr, Yc], dim=0)
    aug_ctrl = Xaug.size(0)
    model_ctrl = train_std(Xaug, Yaug, SEED)
    acc_c, fgsm_c, pgd_c = eval_robustness(model_ctrl, Xte, Yte)
    add_row("n=1000 (safest, control)", nt, aug_ctrl, acc_c, fgsm_c, pgd_c)
    out(f"    n=1000 (safest)         aug_size={aug_ctrl:>7}  acc={acc_c:.4f}  "
        f"FGSM={fgsm_c:.4f}  PGD={pgd_c:.4f}  ({time.time()-t0:.0f}s)")

    # ---- MATCHED-BUDGET UNIFORM ----
    out("\n[5] MATCHED-BUDGET UNIFORM: same total extra copies (2000*50=100000) "
        "spread over ALL 6000 train samples")
    total_budget = max(N_TARGETS) * K_COPIES                # 100000
    Xc, Yc = make_noisy_copies_total(Xtr, Ytr, total_budget, EPS,
                                     SEED * 1_000_000 + 77777)
    Xaug = torch.cat([Xtr, Xc], dim=0)
    Yaug = torch.cat([Ytr, Yc], dim=0)
    aug_unif = Xaug.size(0)
    model_unif = train_std(Xaug, Yaug, SEED)
    acc_u, fgsm_u, pgd_u = eval_robustness(model_unif, Xte, Yte)
    add_row("uniform matched-budget", None, aug_unif, acc_u, fgsm_u, pgd_u)
    out(f"    uniform matched-budget  aug_size={aug_unif:>7}  acc={acc_u:.4f}  "
        f"FGSM={fgsm_u:.4f}  PGD={pgd_u:.4f}  ({time.time()-t0:.0f}s)")

    # ---- TRUE AT reference ----
    out("\n[6] TRUE AT reference (PGD adversarial training)...")
    model_at = train_at(Xtr, Ytr, SEED)
    acc_at, fgsm_at, pgd_at = eval_robustness(model_at, Xte, Yte)
    add_row("true AT (PGD-AT)", None, N_TRAIN, acc_at, fgsm_at, pgd_at)
    keep_models["at"] = model_at
    out(f"    true AT                 aug_size={N_TRAIN:>7}  acc={acc_at:.4f}  "
        f"FGSM={fgsm_at:.4f}  PGD={pgd_at:.4f}  ({time.time()-t0:.0f}s)")

    # ---- MAIN TABLE ----
    out("\n" + "=" * 80)
    out("[7] MAIN TABLE")
    out("=" * 80)
    hdr = "{:<26} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "condition", "n_target", "aug_size", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        nt_s = "-" if r["n_target"] is None else str(r["n_target"])
        out("{:<26} {:>9} {:>9} {:>9.4f} {:>9.4f} {:>9.4f}".format(
            r["cond"], nt_s, r["aug_size"], r["acc"], r["fgsm"], r["pgd"]))
    out("-" * len(hdr))

    # ---- best targeted condition = lowest PGD ASR among vulnerable n>0 ----
    vuln_rows = [r for r in rows if "(vulnerable)" in r["cond"] and r["n_target"] > 0]
    best = min(vuln_rows, key=lambda r: r["pgd"])
    best_tag = f"vuln_{best['n_target']}"
    best_model = keep_models[best_tag]
    out(f"\nbest targeted (vulnerable) condition by PGD_ASR: {best['cond']} "
        f"(PGD={best['pgd']:.4f}, acc={best['acc']:.4f})")

    # ---- HEADLINE CURVE ----
    out("\n[8] HEADLINE CURVE (vulnerable-targeting, K=50): n_target vs ASR & clean acc")
    out("  {:>9} {:>10} {:>10} {:>10}".format("n_target", "clean_acc", "FGSM_ASR", "PGD_ASR"))
    for r in [x for x in rows if "vulnerable" in x["cond"] or x["cond"] == "baseline"]:
        out("  {:>9} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["n_target"], r["acc"], r["fgsm"], r["pgd"]))
    base_row = rows[0]
    biggest = max(vuln_rows, key=lambda r: r["n_target"])
    d_pgd = biggest["pgd"] - base_row["pgd"]
    d_acc = biggest["acc"] - base_row["acc"]
    out(f"  -> from n=0 to n={biggest['n_target']}: PGD_ASR {base_row['pgd']:.4f}"
        f"->{biggest['pgd']:.4f} ({d_pgd:+.4f}); "
        f"clean_acc {base_row['acc']:.4f}->{biggest['acc']:.4f} ({d_acc:+.4f})")
    acc_held = abs(biggest["acc"] - base_row["acc"]) <= 0.02
    out(f"  -> clean acc held within +-0.02 of baseline? {acc_held} "
        f"(baseline {base_row['acc']:.4f})")

    # ---- EXPLICIT COMPARISONS ----
    out("\n[9] EXPLICIT COMPARISONS")
    # (a) vulnerable vs safest at n=1000
    vuln1000 = next(r for r in rows if r["cond"] == "n=1000 (vulnerable)")
    safe1000 = next(r for r in rows if r["cond"] == "n=1000 (safest, control)")
    out("  (a) n=1000 vulnerable vs safest control:")
    out(f"      vulnerable: acc={vuln1000['acc']:.4f} FGSM={vuln1000['fgsm']:.4f} "
        f"PGD={vuln1000['pgd']:.4f}")
    out(f"      safest    : acc={safe1000['acc']:.4f} FGSM={safe1000['fgsm']:.4f} "
        f"PGD={safe1000['pgd']:.4f}")
    out(f"      delta PGD_ASR (vuln - safe) = {vuln1000['pgd']-safe1000['pgd']:+.4f} "
        f"(negative => targeting vulnerable helps MORE)")
    # (b) best vulnerable vs uniform matched-budget
    unif = next(r for r in rows if r["cond"] == "uniform matched-budget")
    out("  (b) best vulnerable-targeting vs uniform matched-budget:")
    out(f"      best vuln ({best['cond']}): acc={best['acc']:.4f} "
        f"FGSM={best['fgsm']:.4f} PGD={best['pgd']:.4f}")
    out(f"      uniform matched-budget    : acc={unif['acc']:.4f} "
        f"FGSM={unif['fgsm']:.4f} PGD={unif['pgd']:.4f}")
    out(f"      delta PGD_ASR (best - uniform) = {best['pgd']-unif['pgd']:+.4f} "
        f"(negative => targeting beats uniform)")
    # (c) best vulnerable vs true AT
    at = next(r for r in rows if r["cond"] == "true AT (PGD-AT)")
    out("  (c) best vulnerable-targeting vs true AT (robustness/clean-acc tradeoff):")
    out(f"      best vuln ({best['cond']}): acc={best['acc']:.4f} PGD={best['pgd']:.4f}")
    out(f"      true AT                   : acc={at['acc']:.4f} PGD={at['pgd']:.4f}")
    out(f"      AT clean-acc cost vs baseline = {at['acc']-base_row['acc']:+.4f}")
    out(f"      best-vuln clean-acc cost vs baseline = {best['acc']-base_row['acc']:+.4f}")
    out(f"      PGD gap (best-vuln - AT) = {best['pgd']-at['pgd']:+.4f} "
        f"(positive => AT still more robust)")

    # ---- VULNERABLE TEST SUBSET ----
    out("\n[10] VULNERABLE TEST SUBSET (lowest 25% margin by base model)")
    b_fg, b_pg = eval_subset_asr(base, Xte, Yte, vuln_te_mask)
    bt_fg, bt_pg = eval_subset_asr(best_model, Xte, Yte, vuln_te_mask)
    out(f"   subset size = {n_vuln_te}")
    out("   {:<26} {:>10} {:>10}".format("condition", "FGSM_ASR", "PGD_ASR"))
    out("   {:<26} {:>10.4f} {:>10.4f}".format("baseline", b_fg, b_pg))
    out("   {:<26} {:>10.4f} {:>10.4f}".format(
        f"best vuln ({best['cond']})", bt_fg, bt_pg))
    out(f"   delta on vulnerable subset: FGSM {bt_fg-b_fg:+.4f}  PGD {bt_pg-b_pg:+.4f}")

    # ---- VERDICT ----
    out("\n" + "=" * 80)
    out("[11] VERDICT")
    out("=" * 80)
    robust_gain = base_row["pgd"] - best["pgd"]             # positive = more robust
    acc_kept = best["acc"] >= base_row["acc"] - 0.02
    beats_unif = best["pgd"] < unif["pgd"] - 1e-4
    at_acc_cost = base_row["acc"] - at["acc"]
    verdict = (
        f"Targeted vulnerable-sample noise hardening (best={best['cond']}): "
        f"PGD_ASR {base_row['pgd']:.3f}->{best['pgd']:.3f} (gain {robust_gain:+.3f}), "
        f"clean_acc {base_row['acc']:.3f}->{best['acc']:.3f} "
        f"({'KEPT' if acc_kept else 'DROPPED'}); "
        f"AT reaches PGD {at['pgd']:.3f} but costs {at_acc_cost:.3f} clean acc. "
        f"Beats uniform matched-budget on PGD? {beats_unif} "
        f"(best {best['pgd']:.3f} vs uniform {unif['pgd']:.3f})."
    )
    out("  " + verdict)
    # crisp one-liner
    if robust_gain > 0.02 and acc_kept and beats_unif:
        one = ("YES: targeting vulnerable samples buys meaningful robustness while "
               "keeping clean accuracy, and beats uniform matched-budget augmentation.")
    elif robust_gain > 0.02 and acc_kept and not beats_unif:
        one = ("PARTIAL: targeting keeps clean acc and adds robustness, but does NOT "
               "beat uniform matched-budget augmentation (targeting not special).")
    elif robust_gain <= 0.02 and acc_kept:
        one = ("NO: noise-copy hardening preserves clean accuracy but gives little "
               "robustness gain at this scale; AT remains the only strong-robustness route.")
    else:
        one = ("MIXED: see numbers above; robustness/clean-acc tradeoff did not cleanly "
               "beat the AT baseline.")
    out("  ONE-LINE VERDICT: " + one)

    out("")
    out(f"done in {time.time() - t0:.1f}s")

    # ---- save report ----
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h404_vulnerable_sample_noise_hardening_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
