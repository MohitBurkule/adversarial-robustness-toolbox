"""
H405 - High-K vulnerable-sample noise hardening: follow-up to H404.

H404 used K=50 noisy copies per targeted vulnerable training sample and found
little robustness gain. H403's single-image sweet spot was K=1000 copies. Here
we test whether the hardening effect survives at H403-level multiplicity (K=1000)
when applied to a SMALL set of the most adversarially vulnerable training samples.

Design (mirrors H404 method, simplified to two targeted conditions):
  1. Train a BASE model from scratch; record clean acc, FGSM ASR, PGD ASR
     (baseline = K=0, n=0).
  2. Rank TRAINING samples by vulnerability = per-sample margin (lowest = most
     vulnerable).
  3. Noise helper: K=1000 copies = (x + 0.1*(2*rand-1)).clamp(0,1), true label.
  4. Condition A: n_target=30  most-vulnerable -> aug_size = 6000 + 30*1000 = 36000.
  5. Condition B: n_target=300 most-vulnerable -> aug_size = 6000 + 300*1000 = 306000.
  Print/save the n=30 block BEFORE starting n=300 so partial output is visible.
  6. VULNERABLE TEST SUBSET: 25% lowest-margin TEST samples (by base model);
     report FGSM/PGD ASR restricted to them, baseline vs each condition.

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
K_COPIES = 1000             # copies per targeted sample (H403 sweet spot)
N_TARGETS = [30, 300]       # two sequential conditions, n=30 then n=300
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
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h405_highK_vulnerable_hardening_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H405  High-K (K=1000) vulnerable-sample noise hardening (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} "
        f"K_COPIES={K_COPIES}")
    out(f"        n_target conditions = {N_TARGETS} (run sequentially, n=30 first)")
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

    # baseline subset ASR
    b_fg, b_pg = eval_subset_asr(base, Xte, Yte, vuln_te_mask)

    # ---- results container ----
    rows = []
    rows.append({"cond": "baseline (K=0,n=0)", "n_target": 0, "aug_size": N_TRAIN,
                 "acc": base_acc, "fgsm": base_fgsm, "pgd": base_pgd,
                 "sub_fg": b_fg, "sub_pg": b_pg})
    keep_models = {"baseline": base}

    # ---- TARGETED CONDITIONS (K=1000), n=30 then n=300, sequential ----
    for ci, nt in enumerate(N_TARGETS):
        out("\n" + "=" * 80)
        out(f"[3.{ci+1}] CONDITION n_target={nt}  (K={K_COPIES} copies of the "
            f"{nt} most-vulnerable training samples)")
        out("=" * 80)
        sel_idx = order_low[:nt]
        Xsrc = Xtr[torch.as_tensor(sel_idx, device=Xtr.device, dtype=torch.long)]
        Ysrc = Ytr[torch.as_tensor(sel_idx, device=Xtr.device, dtype=torch.long)]
        nseed = SEED * 1_000_000 + nt
        Xc, Yc = make_noisy_copies_for(Xsrc, Ysrc, K_COPIES, EPS, nseed)
        Xaug = torch.cat([Xtr, Xc], dim=0)
        Yaug = torch.cat([Ytr, Yc], dim=0)
        aug = Xaug.size(0)
        out(f"    aug_size = {aug}  (= {N_TRAIN} + {nt}*{K_COPIES})")
        out(f"    retraining from scratch (seed={SEED})...")
        model = train_std(Xaug, Yaug, SEED)
        acc, fgsm, pgd = eval_robustness(model, Xte, Yte)
        sub_fg, sub_pg = eval_subset_asr(model, Xte, Yte, vuln_te_mask)
        cond = f"n={nt} (K={K_COPIES})"
        keep_models[cond] = model
        rows.append({"cond": cond, "n_target": nt, "aug_size": aug,
                     "acc": acc, "fgsm": fgsm, "pgd": pgd,
                     "sub_fg": sub_fg, "sub_pg": sub_pg})
        out(f"    RESULT {cond}: clean_acc={acc:.4f}  FGSM_ASR={fgsm:.4f}  "
            f"PGD_ASR={pgd:.4f}")
        out(f"           vulnerable-subset FGSM_ASR={sub_fg:.4f}  "
            f"PGD_ASR={sub_pg:.4f}")
        out(f"           d_clean_acc={acc-base_acc:+.4f}  "
            f"d_PGD_ASR={pgd-base_pgd:+.4f} (vs baseline)  "
            f"({time.time()-t0:.0f}s)")
        # save partial output (esp. so n=30 block is on disk before n=300 starts)
        flush_file()

    # ---- MAIN TABLE ----
    out("\n" + "=" * 80)
    out("[4] MAIN TABLE")
    out("=" * 80)
    hdr = ("{:<22} {:>9} {:>10} {:>9} {:>9} {:>11} {:>11}".format(
        "condition", "aug_size", "clean_acc", "FGSM_ASR", "PGD_ASR",
        "vuln_FGSM", "vuln_PGD"))
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<22} {:>9} {:>10.4f} {:>9.4f} {:>9.4f} {:>11.4f} {:>11.4f}".format(
            r["cond"], r["aug_size"], r["acc"], r["fgsm"], r["pgd"],
            r["sub_fg"], r["sub_pg"]))
    out("-" * len(hdr))

    # ---- VERDICT ----
    out("\n" + "=" * 80)
    out("[5] VERDICT")
    out("=" * 80)
    base_row = rows[0]
    targeted = [r for r in rows if r["n_target"] > 0]
    best = min(targeted, key=lambda r: r["pgd"])   # most robust targeted condition
    robust_gain = base_row["pgd"] - best["pgd"]    # positive => more robust
    acc_cost = best["acc"] - base_row["acc"]       # negative => acc dropped
    acc_kept = best["acc"] >= base_row["acc"] - 0.02
    sub_gain = base_row["sub_pg"] - best["sub_pg"]

    for r in targeted:
        out(f"  {r['cond']}: PGD_ASR {base_row['pgd']:.4f}->{r['pgd']:.4f} "
            f"({r['pgd']-base_row['pgd']:+.4f}); clean_acc "
            f"{base_row['acc']:.4f}->{r['acc']:.4f} ({r['acc']-base_row['acc']:+.4f}); "
            f"vuln_PGD {base_row['sub_pg']:.4f}->{r['sub_pg']:.4f} "
            f"({r['sub_pg']-base_row['sub_pg']:+.4f})")

    out("")
    out(f"  best (lowest PGD_ASR) targeted condition: {best['cond']}")
    out(f"  whole-test PGD robustness gain vs baseline = {robust_gain:+.4f} "
        f"(positive => more robust)")
    out(f"  vulnerable-subset PGD gain vs baseline    = {sub_gain:+.4f}")
    out(f"  clean-acc change vs baseline              = {acc_cost:+.4f} "
        f"({'KEPT within 0.02' if acc_kept else 'DROPPED >0.02'})")

    if robust_gain > 0.02 and acc_kept:
        one = ("YES: high-K (1000) on a small vulnerable set recovers meaningful "
               "robustness while keeping clean accuracy.")
    elif robust_gain > 0.02 and not acc_kept:
        one = ("PARTIAL: high-K (1000) recovers robustness but at a clean-accuracy "
               "cost (acc dropped > 0.02).")
    elif robust_gain <= 0.02 and acc_kept:
        one = ("NO: even at H403's K=1000 multiplicity, targeting a small vulnerable "
               "set gives little whole-test robustness; clean acc is preserved.")
    else:
        one = ("NO+COST: high-K on the small vulnerable set neither recovers robustness "
               "nor preserves clean accuracy.")
    out("  ONE-LINE VERDICT: " + one)

    out("")
    out(f"done in {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
