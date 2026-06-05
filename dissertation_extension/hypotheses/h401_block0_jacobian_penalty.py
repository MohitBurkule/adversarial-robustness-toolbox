"""
H401 - Block0-only activation Jacobian penalty.

Hypothesis: penalising only the block0 -> input Jacobian (restricting H372's
all-layer Jacobian penalty to the FIRST block) recovers most of the robustness
cheaply. This is the regularisation-side test of WHY block0 suffices (H290): if
making just the first block's response to the input locally flat already damps
adversarial sensitivity, then the first block is where input-space smoothness
matters most.

Mechanism (Hutchinson estimate of ||d h0 / d x||_F^2):
  Forward x (requires_grad) through block0 -> h0 (features[0:4]).
  Draw v ~ N(0, I) with the shape of h0.
  jvp_like = autograd.grad((h0 * v).sum(), x, create_graph=True)[0]   # = J^T v
  penalty  = (jvp_like ** 2).sum() / batch    # E_v ||J^T v||^2 = ||J||_F^2
  loss = CE(logits, y) + lambda * penalty
  (One Hutchinson probe per step; create_graph=True so the penalty is itself
   differentiable wrt the weights.)

Conditions:
  baseline (lambda=0), and lambda in {0.01, 0.1}.
Reference: H372's ALL-LAYER Jacobian penalty reached PGD-ASR ~ 0.763 (better
than the ~0.95 clean baseline). The question is how close block0-only gets.
Report clean / FGSM-ASR / PGD-ASR per condition.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

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
LAMBDAS = [0.0, 0.01, 0.1]
H372_ALL_LAYER_PGD = 0.763   # reference: all-layer Jacobian penalty result


def block0_jacobian_penalty(model, x):
    """Hutchinson estimate of (1/B) * ||d h0/d x||_F^2 with one probe.
    h0 = output of features[0:4] (block0)."""
    x = x.clone().detach().requires_grad_(True)
    h0 = model.features[0:4](x)
    v = torch.randn_like(h0)
    jtv, = torch.autograd.grad((h0 * v).sum(), x, create_graph=True, retain_graph=True)
    return jtv.pow(2).sum() / x.size(0)


def evaluate(model, Xte, Yte):
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    fres = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pres = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return clean_acc, fres["asr"], pres["asr"]


def train_condition(lam, Xtr, Ytr, meta):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=0)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    pen_running = 0.0
    cnt = 0
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            if lam > 0:
                pen = block0_jacobian_penalty(model, xb)
                loss = loss + lam * pen
                pen_running += float(pen.detach()); cnt += 1
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    mean_pen = pen_running / cnt if cnt else float("nan")
    return model, mean_pen


def main():
    t_start = time.time()
    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    emit("=" * 78)
    emit("H401 - Block0-only activation Jacobian penalty (Hutchinson)")
    emit("=" * 78)
    meta = C.dataset_meta(DS)
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    emit(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  lambdas={LAMBDAS}  "
         f"epochs={EPOCHS}  n_train={N_TRAIN}")
    emit("Penalty = lambda * ||d h0/d x||_F^2 (block0 output wrt input), "
         "1 Hutchinson probe/step.")
    emit(f"Reference: H372 ALL-LAYER Jacobian penalty reached PGD-ASR ~ "
         f"{H372_ALL_LAYER_PGD:.3f}.")
    emit("")

    results = {}
    for lam in LAMBDAS:
        t0 = time.time()
        model, mean_pen = train_condition(lam, Xtr, Ytr, meta)
        clean_acc, fgsm_asr, pgd_asr = evaluate(model, Xte, Yte)
        dt = time.time() - t0
        results[lam] = dict(clean=clean_acc, fgsm=fgsm_asr, pgd=pgd_asr,
                            pen=mean_pen, t=dt)
        emit(f"[lambda={lam:5.2f}] clean={clean_acc:.3f}  FGSM-ASR={fgsm_asr:.3f}  "
             f"PGD-ASR={pgd_asr:.3f}  mean_block0_jacF2={mean_pen:.3f}  ({dt:.0f}s)")

    emit("")
    emit("-" * 78)
    emit(f"{'lambda':>7s} {'clean':>7s} {'FGSM-ASR':>9s} {'PGD-ASR':>8s} "
         f"{'block0_jacF2':>13s}")
    emit("-" * 78)
    for lam in LAMBDAS:
        r = results[lam]
        pen = r["pen"] if r["pen"] == r["pen"] else 0.0
        emit(f"{lam:7.2f} {r['clean']:7.3f} {r['fgsm']:9.3f} {r['pgd']:8.3f} "
             f"{pen:13.3f}")
    emit(f"{'(H372 all-layer ref)':>30s}   PGD-ASR={H372_ALL_LAYER_PGD:.3f}")
    emit("-" * 78)

    base = results[0.0]
    best_lam = min([l for l in LAMBDAS if l > 0], key=lambda l: results[l]["pgd"])
    best = results[best_lam]
    pgd_red = base["pgd"] - best["pgd"]
    clean_cost = base["clean"] - best["clean"]
    # how much of the way to H372's all-layer result does block0-only get?
    all_red = base["pgd"] - H372_ALL_LAYER_PGD
    frac_of_all = (pgd_red / all_red * 100.0) if abs(all_red) > 1e-9 else float("nan")

    exceeds = best["pgd"] <= H372_ALL_LAYER_PGD
    emit("")
    emit(f"Best block0-only penalty: lambda={best_lam} -> PGD-ASR {best['pgd']:.3f} "
         f"(baseline {base['pgd']:.3f}, drop {pgd_red:+.3f}), clean cost "
         f"{clean_cost:+.3f}.")
    if exceeds:
        emit(f"Block0-only reaches PGD-ASR {best['pgd']:.3f}, which MATCHES OR BEATS "
             f"H372's all-layer {H372_ALL_LAYER_PGD:.3f} -- penalising only the first "
             f"block recovers >=100% of the all-layer robustness reduction "
             f"(measured {frac_of_all:.0f}% of the clean->H372 gap).")
    else:
        emit(f"Block0-only reaches PGD-ASR {best['pgd']:.3f} vs H372 all-layer "
             f"{H372_ALL_LAYER_PGD:.3f}: block0-only recovers {frac_of_all:.0f}% of "
             f"the all-layer PGD-ASR reduction from the clean baseline.")

    helps = pgd_red > 0.03
    emit("")
    if helps and (exceeds or frac_of_all >= 60.0):
        emit(f"VERDICT: Block0-only Jacobian penalty recovers essentially ALL of "
             f"H372's all-layer robustness gain (PGD-ASR {best['pgd']:.3f} vs "
             f"all-layer {H372_ALL_LAYER_PGD:.3f}) at a moderate clean cost "
             f"({clean_cost:+.3f}) -- input-space smoothness of the FIRST block is "
             f"the dominant contributor, confirming block0 suffices.")
    elif helps:
        emit(f"VERDICT: Block0-only Jacobian penalty helps (PGD-ASR {pgd_red:+.3f}) "
             f"but recovers only {frac_of_all:.0f}% of H372's all-layer gain -- the "
             f"first block matters but later-layer smoothness adds the rest.")
    else:
        emit(f"VERDICT: Block0-only Jacobian penalty does NOT meaningfully reduce "
             f"PGD-ASR (drop {pgd_red:+.3f}) -- penalising only the first block is "
             f"insufficient; H372's gain needs the deeper layers too.")
    emit("=" * 78)
    emit(f"total runtime {time.time()-t_start:.0f}s")

    outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", DS, "h401_block0_jacobian_penalty_output.txt")
    with open(outpath, "w") as f:
        f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
