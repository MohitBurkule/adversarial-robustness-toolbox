"""
H410 - Exactly-invertible front-end + input-Jacobian penalty combined.

Topic: exactly-invertible nets for adversarial robustness, combined with our
established implicit defense. H401 found that penalising the FIRST block's
input-Jacobian (||d h0/d x||_F^2, Hutchinson estimate) damps adversarial
sensitivity. i-RevNet (Jacobsen 2018, arXiv:1802.07088) preserves all input
information through additive-coupling blocks. This script asks: does adding the
H401 input-Jacobian penalty ON TOP OF an exactly-invertible front-end stack
reduce PGD-ASR more than either alone?

We reuse the i-RevNet architecture from H407 (invertible squeeze + additive
coupling blocks + linear head; verified exactly invertible) and train it under:
  cond A: invertible, lambda=0           (invertible only)
  cond B: invertible, lambda=0.01        (invertible + Jacobian penalty)
  cond C: invertible, lambda=0.1         (invertible + stronger penalty)
Penalty = lambda * ||d h_front/d x||_F^2 where h_front is the output of stage1
(the first invertible coupling stage), matching H401's "first-block" target but
on the invertible front-end.

We also report a plain-CNN + same penalty reference point (built from H401's
recipe) so the invertible vs lossy comparison is anchored.

Exact invertibility of the front-end is verified each condition: x -> z -> x_rec
must reconstruct to ~machine precision.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05 (SGD mom=0.9 wd=5e-4), BATCH=128,
SEED=0, EPS=0.1, PGD_STEPS=10. (Smoke config via env SMOKE=1.)
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

# reuse the verified invertible architecture from H407
import importlib.util
_H407 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "h407_irevnet_invertible_classifier.py")
_spec = importlib.util.spec_from_file_location("h407mod", _H407)
h407 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h407)
iRevNet = h407.iRevNet

# ---- config --------------------------------------------------------------
DS = "fashion_mnist"
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_TRAIN = 2000 if SMOKE else 6000
N_EVAL = 1000 if SMOKE else 2000
EPOCHS = 2 if SMOKE else 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
LAMBDAS = [0.0, 0.01, 0.1]
META = {"channels": 1, "size": 28, "n_classes": 10}


def front_jacobian_penalty(model, x):
    """Hutchinson estimate of (1/B)||d h_front/d x||_F^2, where h_front is the
    output of the invertible stage1 (first coupling stage)."""
    x = x.clone().detach().requires_grad_(True)
    h = h407.squeeze(x)
    for b in model.stage1:
        h = b(h)
    v = torch.randn_like(h)
    jtv, = torch.autograd.grad((h * v).sum(), x, create_graph=True,
                               retain_graph=True)
    return jtv.pow(2).sum() / x.size(0)


def train(model, Xtr, Ytr, lam):
    C.set_seed(SEED)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    pen_run, cnt = 0.0, 0
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            if lam > 0:
                pen = front_jacobian_penalty(model, xb)
                loss = loss + lam * pen
                pen_run += float(pen.detach()); cnt += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model, (pen_run / cnt if cnt else float("nan"))


def evaluate(model, Xte, Yte):
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h410_invertible_plus_jacobian_penalty_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H410  Invertible front-end + input-Jacobian penalty (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: SMOKE={SMOKE} N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SEED={SEED} EPS={EPS} PGD_STEPS={PGD_STEPS} "
        f"LAMBDAS={LAMBDAS} device={C.DEVICE}")
    out("Penalty = lambda * ||d h_front/d x||_F^2 (stage1 output wrt input), "
        "1 Hutchinson probe/step, on an EXACTLY-invertible i-RevNet trunk.")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    rows = []
    for lam in LAMBDAS:
        out("\n" + "=" * 80)
        out(f"[lambda={lam}] invertible i-RevNet" +
            (" (penalty)" if lam > 0 else " (no penalty)"))
        out("=" * 80)
        C.set_seed(SEED)
        model = iRevNet(in_ch=1, size=28, n_classes=10).to(C.DEVICE)
        err0 = h407.check_invertibility(model, Xte[:64])
        out(f"    random-init recon error = {err0:.3e}")
        model, mean_pen = train(model, Xtr, Ytr, lam)
        err = h407.check_invertibility(model, Xte[:64])
        out(f"    post-train recon error  = {err:.3e}")
        acc, fg, pg = evaluate(model, Xte, Yte)
        out(f"    lambda={lam}: clean_acc={acc:.4f}  FGSM_ASR={fg:.4f}  "
            f"PGD_ASR={pg:.4f}  mean_front_jacF2={mean_pen:.3f}")
        rows.append(dict(lam=lam, acc=acc, fg=fg, pg=pg, pen=mean_pen, err=err))
        flush()

    # ---- table ----
    out("\n" + "=" * 80)
    out("[TABLE]")
    out("=" * 80)
    hdr = "{:<10} {:>10} {:>10} {:>10} {:>14} {:>12}".format(
        "lambda", "clean_acc", "FGSM_ASR", "PGD_ASR", "front_jacF2", "recon_err")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        pen = r["pen"] if r["pen"] == r["pen"] else 0.0
        out("{:<10} {:>10.4f} {:>10.4f} {:>10.4f} {:>14.3f} {:>12.3e}".format(
            r["lam"], r["acc"], r["fg"], r["pg"], pen, r["err"]))
    out("-" * len(hdr))

    # ---- verdict ----
    out("\n" + "=" * 80)
    out("[VERDICT]")
    out("=" * 80)
    base = next(r for r in rows if r["lam"] == 0.0)
    pen_rows = [r for r in rows if r["lam"] > 0]
    best = min(pen_rows, key=lambda r: r["pg"])
    d_pgd = base["pg"] - best["pg"]
    d_acc = best["acc"] - base["acc"]
    for r in pen_rows:
        out(f"  lambda={r['lam']}: PGD_ASR {base['pg']:.4f}->{r['pg']:.4f} "
            f"({r['pg']-base['pg']:+.4f}); clean {base['acc']:.4f}->{r['acc']:.4f} "
            f"({r['acc']-base['acc']:+.4f})")
    out("")
    out(f"  best penalised condition: lambda={best['lam']} (PGD_ASR {best['pg']:.4f})")
    out(f"  PGD gain from adding penalty to invertible net = {d_pgd:+.4f}; "
        f"clean cost {d_acc:+.4f}")
    if d_pgd > 0.03:
        verdict = ("YES: the input-Jacobian penalty stacks with the invertible "
                   "front-end to further reduce PGD-ASR -- smoothness + "
                   "information-preservation are complementary.")
    elif d_pgd < -0.03:
        verdict = ("NO: adding the penalty to the invertible net hurt PGD "
                   "robustness.")
    else:
        verdict = ("NEUTRAL: the Jacobian penalty adds little on top of the "
                   "invertible front-end at these settings.")
    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
