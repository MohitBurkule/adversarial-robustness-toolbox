"""
H414 - Free Adversarial Training (Shafahi et al., 2019, arXiv:1904.12843)

Hypothesis: Free Adversarial Training (Free-AT) — which amortises the adversarial
perturbation update by replaying each mini-batch m times and accumulating the
gradient for both the model and the perturbation simultaneously — achieves
PGD-AT-level robustness at a fraction of the wall-clock cost on Fashion-MNIST
SmallCNN.

Design:
  1. Sweep m (mini-batch replays) in {1, 4, 8}.
     Free-AT effective epochs = EPOCHS // m (so total gradient steps ~= EPOCHS*batches).
  2. PGD-AT baseline: EPOCHS=10, PGD inner steps=10, same SGD config.
     Wall-time is recorded for both so cost comparison is explicit.
  3. Eval on test set: clean_acc, FGSM_ASR, PGD_ASR (eps=0.1, PGD steps=10).

Reference: Shafahi A. et al. "Adversarial Training for Free!" NeurIPS 2019.
           arXiv:1904.12843.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128,
        SGD(mom=0.9, wd=5e-4), SEED=0, eps=0.1, PGD_STEPS=10.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config ----------------------------------------------------------------
DS         = "fashion_mnist"
N_TRAIN    = 6000
N_EVAL     = 2000
EPOCHS     = 10
LR         = 0.05
BATCH      = 128
SEED       = 0
EPS        = 0.1
PGD_STEPS  = 10
M_VALUES   = [1, 4, 8]   # mini-batch replay counts to sweep

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h414_free_adversarial_training_output.txt",
)


# ---- helpers ---------------------------------------------------------------

def _make_sgd(model, lr=LR):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def eval_robustness(model, Xte, Yte):
    """Return (clean_acc, fgsm_asr, pgd_asr)."""
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd",  eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


# ---- Free Adversarial Training (Shafahi 2019) ------------------------------

def train_free(Xtr, Ytr, m, seed):
    """
    Free-AT: each mini-batch is replayed m times.
    The perturbation delta is accumulated across replays (global per batch).
    Effective epochs = ceil(EPOCHS / m) so total gradient steps ≈ EPOCHS * n_batches.

    Algorithm (per mini-batch b):
        delta <- carry-over from previous batch (initialised 0, same shape as one batch)
        for k in range(m):
            xa = clamp(xb + delta, 0, 1)
            forward + backward w.r.t. loss(model(xa), yb)
            delta <- clamp(delta + EPS * sign(grad_x), -EPS, EPS)   # simultaneous update
            model params <- SGD step using grad_model accumulated in backward
    """
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32)
    opt   = _make_sgd(model)
    eff_epochs = max(1, EPOCHS // m)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=eff_epochs)
    n      = Xtr.size(0)

    # global perturbation buffer (one batch worth; reused across batches)
    # shape: (BATCH, C, H, W)
    delta_buf = torch.zeros(BATCH, *Xtr.shape[1:], device=C.DEVICE)

    model.train()
    for ep in range(eff_epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx  = perm[i:i + BATCH]
            xb   = Xtr[idx]   # (bs, C, H, W)
            yb   = Ytr[idx]
            bs   = xb.size(0)

            # carry-over delta for this batch (trim / zero-pad to match batch size)
            if bs < BATCH:
                delta = delta_buf[:bs].clone()
            else:
                delta = delta_buf.clone()

            for _ in range(m):
                xa = (xb + delta).clamp(0.0, 1.0).requires_grad_(True)
                logits = model(xa)
                loss   = F.cross_entropy(logits, yb)
                opt.zero_grad()
                loss.backward()

                # update perturbation using gradient w.r.t. xa
                with torch.no_grad():
                    g_x   = xa.grad.detach()
                    delta  = (delta + EPS * g_x.sign()).clamp(-EPS, EPS)

                # model step (grad already computed by backward)
                opt.step()

            # write back the perturbation for carry-over
            if bs < BATCH:
                delta_buf[:bs] = delta.detach()
            else:
                delta_buf[:] = delta.detach()

        sched.step()

    model.eval()
    return model


# ---- PGD-AT baseline -------------------------------------------------------

def train_pgdat(Xtr, Ytr, seed):
    """Standard PGD adversarial training: EPOCHS epochs, PGD_STEPS inner steps."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32)
    opt   = _make_sgd(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n     = Xtr.size(0)
    alpha = 2.5 * EPS / PGD_STEPS

    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb  = Xtr[idx]
            yb  = Ytr[idx]

            # generate adversarial examples with PGD
            xadv = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=alpha)

            opt.zero_grad()
            loss = F.cross_entropy(model(xadv), yb)
            loss.backward()
            opt.step()

        sched.step()

    model.eval()
    return model


# ---- standard (clean) baseline ---------------------------------------------

def train_clean(Xtr, Ytr, seed):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32)
    opt   = _make_sgd(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n     = Xtr.size(0)

    model.train()
    for ep in range(EPOCHS):
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


# ---- main ------------------------------------------------------------------

def main():
    t_global = time.time()
    lines    = []
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H414  Free Adversarial Training (Shafahi et al., arXiv:1904.12843)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS}  m_sweep={M_VALUES}")
    out(f"        device={C.DEVICE}")
    out("")

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    rows = []

    # ---- [1] clean baseline ----
    out("[1] Training CLEAN baseline ...")
    t0 = time.time()
    model_clean = train_clean(Xtr, Ytr, SEED)
    t_clean = time.time() - t0
    acc, fgsm_asr, pgd_asr = eval_robustness(model_clean, Xte, Yte)
    out(f"    clean_acc={acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  PGD_ASR={pgd_asr:.4f}  "
        f"wall={t_clean:.1f}s")
    rows.append(dict(method="clean", m="-", eff_epochs=EPOCHS,
                     acc=acc, fgsm=fgsm_asr, pgd=pgd_asr, wall=t_clean))
    flush_file()

    # ---- [2] PGD-AT baseline ----
    out("")
    out("[2] Training PGD-AT baseline (EPOCHS=10, PGD_STEPS=10) ...")
    t0 = time.time()
    model_pgdat = train_pgdat(Xtr, Ytr, SEED)
    t_pgdat = time.time() - t0
    acc, fgsm_asr, pgd_asr = eval_robustness(model_pgdat, Xte, Yte)
    out(f"    clean_acc={acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  PGD_ASR={pgd_asr:.4f}  "
        f"wall={t_pgdat:.1f}s")
    rows.append(dict(method="PGD-AT", m="-", eff_epochs=EPOCHS,
                     acc=acc, fgsm=fgsm_asr, pgd=pgd_asr, wall=t_pgdat))
    flush_file()

    # ---- [3] Free-AT sweep over m ----
    for m in M_VALUES:
        eff_ep = max(1, EPOCHS // m)
        out("")
        out(f"[3] Free-AT  m={m}  (eff_epochs={eff_ep}, total_replays~"
            f"{eff_ep * (N_TRAIN // BATCH + 1) * m}) ...")
        t0 = time.time()
        model_free = train_free(Xtr, Ytr, m, SEED)
        t_free = time.time() - t0
        acc, fgsm_asr, pgd_asr = eval_robustness(model_free, Xte, Yte)
        speedup = t_pgdat / t_free if t_free > 0 else float("nan")
        out(f"    clean_acc={acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  PGD_ASR={pgd_asr:.4f}  "
            f"wall={t_free:.1f}s  speedup_vs_pgdat={speedup:.2f}x")
        rows.append(dict(method=f"Free-AT(m={m})", m=m, eff_epochs=eff_ep,
                         acc=acc, fgsm=fgsm_asr, pgd=pgd_asr, wall=t_free))
        flush_file()

    # ---- [4] Summary table ----
    out("")
    out("=" * 80)
    out("[4] SUMMARY TABLE")
    out("=" * 80)
    hdr = ("{:<20} {:>4} {:>10} {:>10} {:>9} {:>9} {:>8} {:>10}".format(
        "method", "m", "eff_epochs", "clean_acc", "FGSM_ASR", "PGD_ASR",
        "wall(s)", "speedup"))
    out(hdr)
    out("-" * len(hdr))
    pgdat_row = next(r for r in rows if r["method"] == "PGD-AT")
    for r in rows:
        spd = pgdat_row["wall"] / r["wall"] if r["wall"] > 0 else float("nan")
        out("{:<20} {:>4} {:>10} {:>10.4f} {:>9.4f} {:>9.4f} {:>8.1f} {:>10.2f}".format(
            r["method"], str(r["m"]), str(r["eff_epochs"]),
            r["acc"], r["fgsm"], r["pgd"], r["wall"], spd))
    out("-" * len(hdr))

    # ---- [5] Verdict ----
    out("")
    out("=" * 80)
    out("[5] VERDICT")
    out("=" * 80)

    pgdat_pgd  = pgdat_row["pgd"]
    pgdat_wall = pgdat_row["wall"]

    for r in rows:
        if not str(r["method"]).startswith("Free-AT"):
            continue
        delta_pgd  = r["pgd"]  - pgdat_pgd
        delta_acc  = r["acc"]  - pgdat_row["acc"]
        speedup    = pgdat_wall / r["wall"] if r["wall"] > 0 else float("nan")
        robust_ok  = abs(delta_pgd) <= 0.05   # within 5% of PGD-AT
        fast_ok    = speedup >= float(r["m"]) * 0.5   # at least half of theoretical gain
        out(f"  {r['method']}: PGD_ASR_delta={delta_pgd:+.4f} vs PGD-AT  "
            f"clean_acc_delta={delta_acc:+.4f}  speedup={speedup:.2f}x  "
            f"robust_ok={'YES' if robust_ok else 'NO'}  fast_ok={'YES' if fast_ok else 'NO'}")

    # best Free-AT by PGD_ASR
    free_rows = [r for r in rows if str(r["method"]).startswith("Free-AT")]
    if free_rows:
        best = min(free_rows, key=lambda r: r["pgd"])
        fastest_robust = min(
            (r for r in free_rows if abs(r["pgd"] - pgdat_pgd) <= 0.05),
            key=lambda r: r["wall"],
            default=None,
        )
        out("")
        out(f"  Best Free-AT (lowest PGD_ASR): {best['method']}  "
            f"PGD_ASR={best['pgd']:.4f}  (PGD-AT: {pgdat_pgd:.4f})")
        if fastest_robust:
            spd = pgdat_wall / fastest_robust["wall"]
            out(f"  Fastest Free-AT within 5% of PGD-AT robustness: "
                f"{fastest_robust['method']}  speedup={spd:.2f}x")
            verdict = (
                f"YES: {fastest_robust['method']} matches PGD-AT robustness "
                f"(PGD_ASR delta={fastest_robust['pgd']-pgdat_pgd:+.4f}) "
                f"at {spd:.1f}x wall-time speedup."
            )
        else:
            verdict = (
                "NO: No Free-AT configuration achieves PGD-AT-level robustness "
                "(within 5% PGD_ASR) on this setting."
            )
        out("  ONE-LINE VERDICT: " + verdict)

    out("")
    out(f"total wall time: {time.time() - t_global:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
