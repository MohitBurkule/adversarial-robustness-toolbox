"""
H281 - Input gradient norm regularisation (Ross & Doshi-Velez 2018).

Adds a penalty term to training loss that discourages large input gradient
norms:

    Loss = CE(f(x), y) + λ * ||∇_x CE(f(x), y)||²

This forces the model to have a smoother loss landscape, which should impede
gradient-based attacks.

Four models are trained:
  a) baseline  (λ=0)
  b) λ=0.01
  c) λ=0.1
  d) λ=1.0

Metrics: clean_acc, FGSM_ASR, PGD_ASR, mean_margin, mean input gradient norm.
Key question: does penalising gradient norm improve robustness, and at what λ
does clean accuracy degrade?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS       = "fashion_mnist"
N_TRAIN  = 6000
EPOCHS   = 10
EPS      = 0.1
PGD_STEPS= 10
BATCH    = 128
SEED     = 0
LAMBDAS  = [0.0, 0.01, 0.1, 1.0]

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h281_gradient_norm_regularisation_output.txt"
)


def train_with_gnorm_reg(model, Xtr, Ytr, lam, epochs, batch=BATCH):
    """Training loop with optional input gradient norm regularisation."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N   = Xtr.shape[0]
    model.train()
    for ep in range(epochs):
        perm  = torch.randperm(N)
        ep_ce = 0.0
        ep_pen= 0.0
        for i in range(0, N, batch):
            idx = perm[i: i + batch]
            xb  = Xtr[idx].clone().detach().to(C.DEVICE)
            yb  = Ytr[idx].to(C.DEVICE)

            if lam > 0:
                xb.requires_grad_(True)
                logits = model(xb)
                ce     = F.cross_entropy(logits, yb)
                # gradient of CE w.r.t. input — need create_graph=True so that
                # the penalty is differentiable w.r.t. model parameters
                g = torch.autograd.grad(ce, xb, create_graph=True)[0]
                penalty = (g.reshape(g.shape[0], -1).pow(2).sum(dim=1)).mean()
                loss = ce + lam * penalty
                ep_pen += penalty.item()
            else:
                logits = model(xb)
                ce     = F.cross_entropy(logits, yb)
                loss   = ce
                ep_pen = 0.0

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_ce += ce.item()

        if (ep + 1) % 5 == 0 or ep == 0:
            n_batches = max(1, N // batch)
            print(f"    ep {ep+1}/{epochs}  CE={ep_ce/n_batches:.4f}  "
                  f"pen={ep_pen/n_batches:.4f}")
    model.eval()


def mean_input_grad_norm(model, X, Y, batch=128):
    """Mean L2 norm of ∇_x CE over samples."""
    norms = []
    N = X.shape[0]
    model.eval()
    for i in range(0, N, batch):
        xb = X[i: i + batch].clone().detach().to(C.DEVICE).requires_grad_(True)
        yb = Y[i: i + batch].to(C.DEVICE)
        logits = model(xb)
        ce     = F.cross_entropy(logits, yb, reduction="sum")
        g      = torch.autograd.grad(ce, xb)[0]
        norms.append(g.detach().reshape(g.shape[0], -1)
                      .pow(2).sum(dim=1).sqrt().cpu())
    return torch.cat(norms).mean().item()


def evaluate(model, Xte, Yte):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    margins = C.margin(model, Xte, Yte)

    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    with torch.no_grad():
        fgsm_pred = model(Xfgsm).argmax(1).cpu()
    fgsm_asr = float((fgsm_pred != Yte.cpu()).float().mean())

    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS)
    with torch.no_grad():
        pgd_pred = model(Xpgd).argmax(1).cpu()
    pgd_asr = float((pgd_pred != Yte.cpu()).float().mean())

    gn = mean_input_grad_norm(model, Xte, Yte)
    return {
        "clean_acc":        round(clean_acc, 4),
        "fgsm_asr":         round(fgsm_asr, 4),
        "pgd_asr":          round(pgd_asr, 4),
        "mean_margin":      round(float(margins.mean()), 4),
        "mean_grad_norm":   round(gn, 4),
    }


def main():
    C.set_seed(SEED)
    lines = []
    lines.append("=" * 74)
    lines.append("H281 - Input gradient norm regularisation")
    lines.append("=" * 74)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_train={N_TRAIN}  "
                 f"epochs={EPOCHS}  eps={EPS}")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    Xtr, Ytr = Xtr[:N_TRAIN], Ytr[:N_TRAIN]
    Xte, Yte = Xte.to(C.DEVICE), Yte.to(C.DEVICE)

    results = []
    for lam in LAMBDAS:
        t0 = time.time()
        print(f"\nTraining λ={lam} ...")
        C.set_seed(SEED)
        model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, seed=SEED)
        model.to(C.DEVICE)
        train_with_gnorm_reg(model, Xtr, Ytr, lam=lam, epochs=EPOCHS)
        metrics = evaluate(model, Xte, Yte)
        metrics["lambda"]     = lam
        metrics["runtime_s"]  = round(time.time() - t0, 1)
        results.append(metrics)
        lines.append(f"\n  λ={lam}  ({metrics['runtime_s']}s)")
        lines.append(f"    clean_acc={metrics['clean_acc']:.4f}  "
                     f"fgsm_asr={metrics['fgsm_asr']:.4f}  "
                     f"pgd_asr={metrics['pgd_asr']:.4f}")
        lines.append(f"    mean_margin={metrics['mean_margin']:.4f}  "
                     f"mean_grad_norm={metrics['mean_grad_norm']:.4f}")

    lines.append("\n" + "=" * 74)
    lines.append("Summary table:")
    lines.append(f"  {'λ':>8}  {'clean_acc':>10}  {'fgsm_asr':>10}  "
                 f"{'pgd_asr':>9}  {'margin':>9}  {'grad_norm':>10}")
    for r in results:
        lines.append(f"  {r['lambda']:8.2f}  {r['clean_acc']:10.4f}  "
                     f"{r['fgsm_asr']:10.4f}  {r['pgd_asr']:9.4f}  "
                     f"{r['mean_margin']:9.4f}  {r['mean_grad_norm']:10.4f}")

    lines.append("=" * 74)
    lines.append("Interpretation: increasing λ should reduce mean_grad_norm and")
    lines.append("lower FGSM/PGD ASR at some cost to clean accuracy. The λ at which")
    lines.append("clean_acc begins to fall relative to the robustness gain marks the")
    lines.append("Pareto frontier of gradient-norm regularisation.")
    lines.append("=" * 74)

    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
