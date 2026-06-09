"""
H388: Input-Hessian spectral-norm (top eigenvalue) penalty.

Hypothesis: penalising the LARGEST input-Hessian eigenvalue (worst-case
curvature) beats the trace penalty (prior H339/H287 trace penalties
failed/weak) because adversarial directions exploit max-curvature.

Implementation:
  estimate lambda_max(Hess_x L) via 1-step power iteration using
  Hessian-vector products.
    g  = autograd.grad(loss, xb, create_graph=True, retain_graph=True)[0]
    init v = g.detach() normalized per-sample  (warm start)
    Hv = autograd.grad((g*v).sum(), xb, retain_graph=True)[0]
    lambda_max ~= Rayleigh quotient (v . Hv) / (v . v)   (v normalized => denom=1)
  Penalty = lambda_max_estimate.  loss_total = CE + lambda * lambda_max.
  Grid lambda in {0, 0.001, 0.01, 0.05}.
"""
import os
import sys

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
LAMBDAS = [0.0, 0.001, 0.01, 0.05]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "fashion_mnist", "h388_input_hessian_spectral_penalty_output.txt")


def lambda_max_estimate(model, xb, yb):
    """1-step power iteration estimate of top input-Hessian eigenvalue.

    Returns a scalar tensor (mean over batch of Rayleigh quotient), with grad
    flowing back to model params.
    """
    xb = xb.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xb), yb)
    g = torch.autograd.grad(loss, xb, create_graph=True, retain_graph=True)[0]
    # warm-start v = normalized gradient direction (per-sample)
    v = g.detach()
    flat = v.flatten(1)
    norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    v = (flat / norm).view_as(v)
    # Hessian-vector product
    Hv = torch.autograd.grad((g * v).sum(), xb, retain_graph=True, create_graph=True)[0]
    # Rayleigh quotient per-sample; v normalized so denom == 1
    num = (v * Hv).flatten(1).sum(dim=1)
    den = (v * v).flatten(1).sum(dim=1).clamp_min(1e-12)
    rq = num / den
    return rq.mean()


def train_with_penalty(lam):
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
            ce = F.cross_entropy(model(xb), yb)
            if lam > 0:
                pen = lambda_max_estimate(model, xb, yb)
                loss = ce + lam * pen
            else:
                loss = ce
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def evaluate(model):
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    # FGSM
    fgsm_correct = 0
    pgd_correct = 0
    total = 0
    for i in range(0, Xte.size(0), 256):
        xb, yb = Xte[i:i + 256], Yte[i:i + 256]
        xa_f = C.fgsm(model, xb, yb, EPS)
        xa_p = C.pgd(model, xb, yb, EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
        with torch.no_grad():
            fgsm_correct += int((model(xa_f).argmax(1) == yb).sum())
            pgd_correct += int((model(xa_p).argmax(1) == yb).sum())
        total += xb.size(0)
    fgsm_acc = fgsm_correct / total
    pgd_acc = pgd_correct / total
    fgsm_asr = 1.0 - fgsm_acc
    pgd_asr = 1.0 - pgd_acc
    mg = float(np.mean(C.margin(model, Xte, Yte)))
    return clean_acc, fgsm_asr, pgd_asr, mg


def main():
    global Xtr, Ytr, Xte, Yte
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    rows = []
    for lam in LAMBDAS:
        t0 = __import__("time").time()
        model = train_with_penalty(lam)
        clean_acc, fgsm_asr, pgd_asr, mg = evaluate(model)
        dt = __import__("time").time() - t0
        rows.append((lam, clean_acc, fgsm_asr, pgd_asr, mg))
        print(f"lambda={lam:<7} clean={clean_acc:.4f} FGSM_ASR={fgsm_asr:.4f} "
              f"PGD_ASR={pgd_asr:.4f} margin={mg:.4f}  ({dt:.1f}s)")

    # build table
    header = f"{'lambda':>8} | {'clean_acc':>9} | {'FGSM_ASR':>8} | {'PGD_ASR':>8} | {'margin':>8}"
    sep = "-" * len(header)
    lines = ["H388: Input-Hessian spectral-norm (top eigenvalue) penalty",
             "loss = CE + lambda * lambda_max(Hess_x L); 1-step power iteration",
             "", header, sep]
    base = rows[0]
    for lam, ca, fa, pa, mg in rows:
        tag = "  (baseline)" if lam == 0.0 else ""
        lines.append(f"{lam:>8} | {ca:>9.4f} | {fa:>8.4f} | {pa:>8.4f} | {mg:>8.4f}{tag}")
    lines.append(sep)

    # verdict
    base_pgd = base[3]
    best = min(rows[1:], key=lambda r: r[3])
    delta = base_pgd - best[3]
    verdict = (f"VERDICT: baseline PGD_ASR={base_pgd:.4f}; best penalized "
               f"lambda={best[0]} PGD_ASR={best[3]:.4f} (delta={delta:+.4f}, "
               f"clean {base[1]:.4f}->{best[1]:.4f}). "
               + ("Spectral penalty HELPS." if delta > 0.01 else
                  "Spectral penalty does NOT meaningfully help."))
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
