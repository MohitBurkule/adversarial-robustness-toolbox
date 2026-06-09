"""
H399: Combined Jacobian (first-order) + Hessian spectral (second-order) penalty.

Hypothesis: jointly bounding first-order (Jacobian Frobenius) and second-order
(Hessian spectral) sensitivity beats either alone, since FGSM is first-order
and PGD exploits curvature.

Implementation:
  loss = CE + l1 * ||J(x)||_F^2  +  l2 * lambda_max(Hess_x L)

  Jacobian Frobenius via Hutchinson:
    v ~ N(0,I) shaped like logits f(x)
    jvp = autograd.grad((f(x)*v).sum(), x, create_graph=True, retain_graph=True)[0]
    penalty_J = (jvp**2).sum() / batch   (unbiased estimate of ||J||_F^2)

  Hessian spectral via 1-step power iteration (as H388) on the CE loss.

  Grid (l1, l2): (0,0) baseline, (0.01,0), (0,0.01), (0.01,0.01), (0.1,0.01).
  Verdict: does the combo beat each alone?
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "campaign"))
import common as C

import numpy as np
import torch
import torch.nn.functional as F

# ---- config ---------------------------------------------------------------
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
META = {"channels": 1, "size": 28, "n_classes": 10}
GRID = [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (0.01, 0.01), (0.1, 0.01)]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "fashion_mnist", "h399_curvature_lipschitz_combined_output.txt")


def jacobian_frob_sq(model, xb):
    """Hutchinson estimate of ||J(x)||_F^2 (sum over logits, mean over batch).

    Returns scalar tensor with grad to params.
    """
    xb = xb.clone().detach().requires_grad_(True)
    out = model(xb)
    v = torch.randn_like(out)
    jvp = torch.autograd.grad((out * v).sum(), xb, create_graph=True, retain_graph=True)[0]
    return (jvp ** 2).flatten(1).sum(dim=1).mean()


def hessian_spectral(model, xb, yb):
    """1-step power-iteration estimate of top input-Hessian eigenvalue of CE loss."""
    xb = xb.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xb), yb)
    g = torch.autograd.grad(loss, xb, create_graph=True, retain_graph=True)[0]
    v = g.detach()
    flat = v.flatten(1)
    norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    v = (flat / norm).view_as(v)
    Hv = torch.autograd.grad((g * v).sum(), xb, retain_graph=True, create_graph=True)[0]
    num = (v * Hv).flatten(1).sum(dim=1)
    den = (v * v).flatten(1).sum(dim=1).clamp_min(1e-12)
    return (num / den).mean()


def train_combo(l1, l2):
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, seed=0)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
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
            if l1 > 0:
                loss = loss + l1 * jacobian_frob_sq(model, xb)
            if l2 > 0:
                loss = loss + l2 * hessian_spectral(model, xb, yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def evaluate(model):
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    fgsm_correct = pgd_correct = total = 0
    for i in range(0, Xte.size(0), 256):
        xb, yb = Xte[i:i + 256], Yte[i:i + 256]
        xa_f = C.fgsm(model, xb, yb, EPS)
        xa_p = C.pgd(model, xb, yb, EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
        with torch.no_grad():
            fgsm_correct += int((model(xa_f).argmax(1) == yb).sum())
            pgd_correct += int((model(xa_p).argmax(1) == yb).sum())
        total += xb.size(0)
    fgsm_asr = 1.0 - fgsm_correct / total
    pgd_asr = 1.0 - pgd_correct / total
    mg = float(np.mean(C.margin(model, Xte, Yte)))
    return clean_acc, fgsm_asr, pgd_asr, mg


def main():
    global Xtr, Ytr, Xte, Yte
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    rows = []
    for l1, l2 in GRID:
        t0 = time.time()
        model = train_combo(l1, l2)
        clean_acc, fgsm_asr, pgd_asr, mg = evaluate(model)
        dt = time.time() - t0
        rows.append((l1, l2, clean_acc, fgsm_asr, pgd_asr, mg))
        print(f"l1={l1:<5} l2={l2:<5} clean={clean_acc:.4f} FGSM_ASR={fgsm_asr:.4f} "
              f"PGD_ASR={pgd_asr:.4f} margin={mg:.4f}  ({dt:.1f}s)")

    header = f"{'l1(Jac)':>8} | {'l2(Hess)':>8} | {'clean':>7} | {'FGSM_ASR':>8} | {'PGD_ASR':>8} | {'margin':>8}"
    sep = "-" * len(header)
    lines = ["H399: Combined Jacobian-Frobenius + Hessian-spectral penalty",
             "loss = CE + l1*||J(x)||_F^2 + l2*lambda_max(Hess_x L)",
             "", header, sep]
    by_key = {(l1, l2): (ca, fa, pa, mg) for (l1, l2, ca, fa, pa, mg) in rows}
    for l1, l2, ca, fa, pa, mg in rows:
        tag = ""
        if (l1, l2) == (0.0, 0.0):
            tag = "  (baseline)"
        elif l2 == 0.0:
            tag = "  (Jac only)"
        elif l1 == 0.0:
            tag = "  (Hess only)"
        else:
            tag = "  (combo)"
        lines.append(f"{l1:>8} | {l2:>8} | {ca:>7.4f} | {fa:>8.4f} | {pa:>8.4f} | {mg:>8.4f}{tag}")
    lines.append(sep)

    base = by_key[(0.0, 0.0)]
    jac = by_key[(0.01, 0.0)]
    hess = by_key[(0.0, 0.01)]
    combo = by_key[(0.01, 0.01)]
    combo_strong = by_key[(0.1, 0.01)]
    best_combo = min([combo, combo_strong], key=lambda r: r[2])  # placeholder; pick by PGD below
    best_combo = min([combo, combo_strong], key=lambda r: r[2])
    # choose combo with lower PGD_ASR
    best_combo = combo if combo[2] <= combo_strong[2] else combo_strong
    best_combo_lbl = "(0.01,0.01)" if best_combo is combo else "(0.1,0.01)"

    beats_jac = best_combo[2] < jac[2] - 0.005
    beats_hess = best_combo[2] < hess[2] - 0.005
    verdict = (
        f"VERDICT: PGD_ASR -- baseline={base[2]:.4f}, Jac-only={jac[2]:.4f}, "
        f"Hess-only={hess[2]:.4f}, best-combo {best_combo_lbl}={best_combo[2]:.4f}. "
        + ("Combo beats BOTH single penalties." if (beats_jac and beats_hess) else
           ("Combo beats one but not both." if (beats_jac or beats_hess) else
            "Combo does NOT beat either single penalty.")))
    lines.append("")
    lines.append(verdict)

    table = "\n".join(lines)
    print()
    print(table)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
