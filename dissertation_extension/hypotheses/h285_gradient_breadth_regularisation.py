"""
H285 - Gradient breadth regularisation.

If broad gradients = more robust, can we train for gradient breadth explicitly?

Add a regularisation term that encourages HIGH entropy of the input gradient
distribution:
    Loss = CE(f(x), y) - λ * H(|∇_x CE|)

where H is the Shannon entropy of the normalised absolute gradient values.
Maximising gradient entropy = encouraging the model to spread sensitivity
evenly across all pixels = harder to attack with a concentrated perturbation.

This requires second-order gradients (gradient of the gradient entropy w.r.t.
weights). We use torch.autograd.grad with create_graph=True.

Three models:
  a) baseline
  b) gradient entropy regularisation λ=0.1
  c) gradient entropy regularisation λ=1.0

N_train=3000 (second-order is expensive), 10 epochs.
Metrics: clean_acc, FGSM_ASR, PGD_ASR, mean_margin, mean gradient entropy.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS        = "fashion_mnist"
N_TRAIN   = 3000
EPOCHS    = 10
EPS       = 0.1
PGD_STEPS = 10
BATCH     = 64      # smaller batch for second-order stability
SEED      = 0
LAMBDAS   = [0.0, 0.1, 1.0]

OUT_PATH  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h285_gradient_breadth_regularisation_output.txt"
)


def gradient_entropy(model, xb, yb):
    """
    Compute H(|∇_x CE|) with create_graph=True so the result is differentiable
    w.r.t. model parameters.

    Returns a scalar entropy averaged over the batch.
    """
    xb_req = xb.clone().requires_grad_(True)
    logits = model(xb_req)
    ce     = F.cross_entropy(logits, yb, reduction="sum")
    # create_graph=True: retain computational graph through the gradient
    g      = torch.autograd.grad(ce, xb_req, create_graph=True)[0]
    g_abs  = g.abs().reshape(g.shape[0], -1)          # (B, D)
    s      = g_abs.sum(dim=1, keepdim=True).clamp(min=1e-10)
    p      = g_abs / s                                  # normalised
    p      = p.clamp(min=1e-10)
    H      = -(p * p.log()).sum(dim=1).mean()           # scalar, still in graph
    return H


def train_with_entropy_reg(model, Xtr, Ytr, lam, epochs=EPOCHS, batch=BATCH):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N   = Xtr.shape[0]
    model.train()
    for ep in range(epochs):
        perm     = torch.randperm(N)
        ep_ce    = 0.0
        ep_H     = 0.0
        for i in range(0, N, batch):
            idx = perm[i: i + batch]
            xb  = Xtr[idx].to(C.DEVICE)
            yb  = Ytr[idx].to(C.DEVICE)

            if lam > 0:
                opt.zero_grad()
                xb_req = xb.clone().requires_grad_(True)
                logits = model(xb_req)
                ce     = F.cross_entropy(logits, yb)
                g      = torch.autograd.grad(ce, xb_req, create_graph=True)[0]
                g_abs  = g.abs().reshape(g.shape[0], -1)
                s      = g_abs.sum(dim=1, keepdim=True).clamp(min=1e-10)
                p      = (g_abs / s).clamp(min=1e-10)
                H      = -(p * p.log()).sum(dim=1).mean()
                # maximise entropy = minimise -H
                loss   = ce - lam * H
                ep_H  += H.item()
            else:
                opt.zero_grad()
                logits = model(xb)
                ce     = F.cross_entropy(logits, yb)
                loss   = ce
                ep_H  += 0.0

            loss.backward()
            opt.step()
            ep_ce += ce.item()

        if (ep + 1) % 5 == 0 or ep == 0:
            nb = max(1, N // batch)
            print(f"    ep {ep+1}/{epochs}  CE={ep_ce/nb:.4f}  H={ep_H/nb:.4f}")
    model.eval()


def mean_gradient_entropy(model, X, Y, batch=64):
    """Mean gradient entropy over dataset (no graph needed, just evaluation)."""
    Hs = []
    N  = X.shape[0]
    model.eval()
    for i in range(0, N, batch):
        xb = X[i: i + batch].clone().detach().to(C.DEVICE).requires_grad_(True)
        yb = Y[i: i + batch].to(C.DEVICE)
        ce = F.cross_entropy(model(xb), yb, reduction="sum")
        g  = torch.autograd.grad(ce, xb)[0].detach().cpu()
        g_abs = g.abs().reshape(g.shape[0], -1)
        s  = g_abs.sum(dim=1, keepdim=True).clamp(min=1e-10)
        p  = (g_abs / s).clamp(min=1e-10)
        H  = -(p * p.log()).sum(dim=1)
        Hs.append(H)
    return torch.cat(Hs).mean().item()


def evaluate(model, Xte, Yte, label, lines):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    margins = C.margin(model, Xte, Yte)

    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    with torch.no_grad():
        fgsm_asr = float((model(Xfgsm).argmax(1) != Yte).float().mean())

    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS)
    with torch.no_grad():
        pgd_asr = float((model(Xpgd).argmax(1) != Yte).float().mean())

    H_mean = mean_gradient_entropy(model, Xte, Yte)

    lines.append(f"\n  [{label}]")
    lines.append(f"    clean_acc={clean_acc:.4f}  fgsm_asr={fgsm_asr:.4f}  "
                 f"pgd_asr={pgd_asr:.4f}")
    lines.append(f"    mean_margin={margins.mean():.4f}  "
                 f"mean_grad_entropy={H_mean:.4f}")
    return {
        "label":             label,
        "clean_acc":         round(clean_acc, 4),
        "fgsm_asr":          round(fgsm_asr, 4),
        "pgd_asr":           round(pgd_asr, 4),
        "mean_margin":       round(float(margins.mean()), 4),
        "mean_grad_entropy": round(H_mean, 4),
    }


def main():
    C.set_seed(SEED)
    lines = []
    lines.append("=" * 74)
    lines.append("H285 - Gradient breadth regularisation")
    lines.append("=" * 74)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_train={N_TRAIN}  "
                 f"epochs={EPOCHS}  eps={EPS}  lambdas={LAMBDAS}")
    lines.append("Regularisation: Loss = CE - λ * H(|∇_x CE|)  (maximise entropy)")

    t0_all = time.time()
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    Xtr, Ytr = Xtr[:N_TRAIN], Ytr[:N_TRAIN]
    Xte, Yte = Xte.to(C.DEVICE), Yte.to(C.DEVICE)

    results = []
    for lam in LAMBDAS:
        label = f"λ={lam}"
        print(f"\nTraining {label} ...")
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, seed=SEED)
        model.to(C.DEVICE)
        train_with_entropy_reg(model, Xtr, Ytr, lam=lam)
        r = evaluate(model, Xte, Yte, label, lines)
        r["runtime_s"] = round(time.time() - t0, 1)
        results.append(r)

    lines.append("\n" + "=" * 74)
    lines.append("Summary table:")
    lines.append(f"  {'λ':>6}  {'clean_acc':>10}  {'fgsm_asr':>9}  "
                 f"{'pgd_asr':>8}  {'margin':>8}  {'grad_H':>8}")
    for r in results:
        lines.append(f"  {r['label']:>6}  {r['clean_acc']:10.4f}  "
                     f"{r['fgsm_asr']:9.4f}  {r['pgd_asr']:8.4f}  "
                     f"{r['mean_margin']:8.4f}  {r['mean_grad_entropy']:8.4f}")

    lines.append(f"\nTotal runtime: {time.time() - t0_all:.1f}s")
    lines.append("=" * 74)
    lines.append("Interpretation: if λ>0 models have higher mean_grad_entropy AND lower")
    lines.append("FGSM/PGD ASR, then explicitly regularising gradient breadth improves")
    lines.append("robustness. If clean_acc drops substantially, there is a cost to")
    lines.append("'flattening' the sensitivity map. This is a novel regulariser connecting")
    lines.append("gradient geometry to adversarial robustness.")
    lines.append("=" * 74)

    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
