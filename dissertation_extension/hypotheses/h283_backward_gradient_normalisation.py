"""
H283 - Backward gradient normalisation for adversarial training.

User's insight: BatchNorm normalises forward activations — what if we normalise
the BACKWARD input gradient?

Four training regimes:
  a) baseline         — no adversarial training
  b) FGSM-AT          — standard FGSM AT: x_adv = x + eps * sign(∇_x L)
  c) Smooth-FGSM-AT   — "smooth" AT: x_adv = x + eps * (∇_x L / ||∇_x L||₂)
  d) L2-PGD-AT        — PGD using L2-normalised gradient steps (L2-ball projection)

Evaluation:
  - clean_acc, FGSM_ASR (L∞), PGD_ASR (L∞), L2-PGD ASR

Key question: does training against L2-normalised adversaries produce different
robustness geometry than sign-based training?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS        = "fashion_mnist"
N_TRAIN   = 6000
EPOCHS    = 10
EPS       = 0.1
PGD_STEPS = 10
BATCH     = 128
SEED      = 0
EPS_L2    = 1.0    # L2-ball radius for L2-PGD evaluation

OUT_PATH  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h283_backward_gradient_normalisation_output.txt"
)


# ── L2-PGD helper (for both training and evaluation) ─────────────────────────
def pgd_l2(model, X, Y, eps, steps=10, alpha=None):
    """
    PGD with L2 ball constraint. Each step: move along normalised gradient,
    project back onto the L2 ball of radius `eps`.
    """
    if alpha is None:
        alpha = eps * 2 / steps
    X   = X.clone().detach().to(C.DEVICE)
    Y   = Y.to(C.DEVICE)
    B, C_, H, W = X.shape
    delta = torch.zeros_like(X)
    for _ in range(steps):
        delta.requires_grad_(True)
        logits = model(torch.clamp(X + delta, 0, 1))
        loss   = F.cross_entropy(logits, Y)
        g      = torch.autograd.grad(loss, delta)[0].detach()
        g_flat = g.reshape(B, -1)
        g_norm = g_flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
        g_flat = g_flat / g_norm                  # normalised gradient direction
        delta  = (delta.detach() + alpha * g_flat.reshape(B, C_, H, W)).detach()
        # project onto L2 ball
        d_flat = delta.reshape(B, -1)
        d_norm = d_flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
        excess = d_norm > eps
        d_flat = torch.where(excess, d_flat * eps / d_norm, d_flat)
        delta  = d_flat.reshape(B, C_, H, W)
    return torch.clamp(X + delta, 0, 1).detach()


# ── training loops ────────────────────────────────────────────────────────────
def train_baseline(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N   = Xtr.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(N)
        ep_loss = 0.0
        for i in range(0, N, batch):
            idx = perm[i: i + batch]
            xb  = Xtr[idx].to(C.DEVICE)
            yb  = Ytr[idx].to(C.DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    ep {ep+1}/{epochs}  loss={ep_loss / max(1, N // batch):.4f}")
    model.eval()


def train_fgsm_at(model, Xtr, Ytr, epochs=EPOCHS, eps=EPS, batch=BATCH):
    """Standard FGSM adversarial training."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N   = Xtr.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(N)
        ep_loss = 0.0
        for i in range(0, N, batch):
            idx = perm[i: i + batch]
            xb  = Xtr[idx].clone().detach().to(C.DEVICE).requires_grad_(True)
            yb  = Ytr[idx].to(C.DEVICE)
            logits = model(xb)
            loss   = F.cross_entropy(logits, yb)
            g      = torch.autograd.grad(loss, xb)[0].detach()
            xadv   = torch.clamp(xb.detach() + eps * g.sign(), 0, 1)
            model.train()
            opt.zero_grad()
            loss_adv = F.cross_entropy(model(xadv), yb)
            loss_adv.backward()
            opt.step()
            ep_loss += loss_adv.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    ep {ep+1}/{epochs}  loss={ep_loss / max(1, N // batch):.4f}")
    model.eval()


def train_smooth_fgsm(model, Xtr, Ytr, epochs=EPOCHS, eps=EPS, batch=BATCH):
    """Smooth FGSM: adversarial direction = grad/||grad||₂ (L2-normalised)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N   = Xtr.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(N)
        ep_loss = 0.0
        for i in range(0, N, batch):
            idx   = perm[i: i + batch]
            xb    = Xtr[idx].clone().detach().to(C.DEVICE).requires_grad_(True)
            yb    = Ytr[idx].to(C.DEVICE)
            logits= model(xb)
            loss  = F.cross_entropy(logits, yb)
            g     = torch.autograd.grad(loss, xb)[0].detach()
            B     = xb.shape[0]
            g_flat= g.reshape(B, -1)
            g_norm= g_flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
            g_dir = (g_flat / g_norm).reshape_as(g)
            xadv  = torch.clamp(xb.detach() + eps * g_dir, 0, 1)
            model.train()
            opt.zero_grad()
            loss_adv = F.cross_entropy(model(xadv), yb)
            loss_adv.backward()
            opt.step()
            ep_loss += loss_adv.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    ep {ep+1}/{epochs}  loss={ep_loss / max(1, N // batch):.4f}")
    model.eval()


def train_l2pgd_at(model, Xtr, Ytr, epochs=EPOCHS, eps=EPS, steps=PGD_STEPS,
                   batch=BATCH):
    """L2-PGD adversarial training."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N   = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N)
        ep_loss = 0.0
        model.eval()
        for i in range(0, N, batch):
            idx  = perm[i: i + batch]
            xb   = Xtr[idx].to(C.DEVICE)
            yb   = Ytr[idx].to(C.DEVICE)
            xadv = pgd_l2(model, xb, yb, eps=eps, steps=steps)
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(xadv), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    ep {ep+1}/{epochs}  loss={ep_loss / max(1, N // batch):.4f}")
    model.eval()


def evaluate(model, Xte, Yte, label, lines):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    _, clean_acc = C.logits_and_acc(model, Xte, Yte)

    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    with torch.no_grad():
        fgsm_asr = float((model(Xfgsm).argmax(1) != Yte).float().mean())

    Xpgd_linf = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS)
    with torch.no_grad():
        pgd_linf_asr = float((model(Xpgd_linf).argmax(1) != Yte).float().mean())

    Xpgd_l2 = pgd_l2(model, Xte, Yte, eps=EPS_L2, steps=PGD_STEPS)
    with torch.no_grad():
        pgd_l2_asr = float((model(Xpgd_l2).argmax(1) != Yte).float().mean())

    lines.append(f"\n  [{label}]")
    lines.append(f"    clean_acc={clean_acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  "
                 f"PGD(Linf)_ASR={pgd_linf_asr:.4f}  PGD(L2)_ASR={pgd_l2_asr:.4f}")
    return {
        "label":         label,
        "clean_acc":     round(clean_acc, 4),
        "fgsm_asr":      round(fgsm_asr, 4),
        "pgd_linf_asr":  round(pgd_linf_asr, 4),
        "pgd_l2_asr":    round(pgd_l2_asr, 4),
    }


def main():
    C.set_seed(SEED)
    lines = []
    lines.append("=" * 74)
    lines.append("H283 - Backward gradient normalisation for adversarial training")
    lines.append("=" * 74)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_train={N_TRAIN}  "
                 f"epochs={EPOCHS}  eps_linf={EPS}  eps_l2={EPS_L2}")

    t0_all = time.time()
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    Xtr, Ytr = Xtr[:N_TRAIN], Ytr[:N_TRAIN]
    Xte, Yte = Xte.to(C.DEVICE), Yte.to(C.DEVICE)

    configs = [
        ("a_baseline",     train_baseline),
        ("b_FGSM-AT",      train_fgsm_at),
        ("c_Smooth-FGSM",  train_smooth_fgsm),
        ("d_L2-PGD-AT",    train_l2pgd_at),
    ]

    results = []
    for label, train_fn in configs:
        print(f"\nTraining {label} ...")
        t0 = time.time()
        C.set_seed(SEED)
        model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, seed=SEED)
        model.to(C.DEVICE)
        train_fn(model, Xtr, Ytr)
        r = evaluate(model, Xte, Yte, label, lines)
        r["runtime_s"] = round(time.time() - t0, 1)
        results.append(r)

    lines.append("\n" + "=" * 74)
    lines.append("Summary table:")
    lines.append(f"  {'model':22s}  {'clean_acc':>10}  {'fgsm_asr':>9}  "
                 f"{'pgd_linf':>9}  {'pgd_l2':>8}")
    for r in results:
        lines.append(f"  {r['label']:22s}  {r['clean_acc']:10.4f}  "
                     f"{r['fgsm_asr']:9.4f}  {r['pgd_linf_asr']:9.4f}  "
                     f"{r['pgd_l2_asr']:8.4f}")

    lines.append(f"\nTotal runtime: {time.time() - t0_all:.1f}s")
    lines.append("=" * 74)
    lines.append("Interpretation: smooth-FGSM and L2-PGD training use a L2-normalised")
    lines.append("gradient direction instead of the sign (Linf direction). If they yield")
    lines.append("better L2 robustness at a similar clean-acc cost, backward gradient")
    lines.append("normalisation IS the key difference — confirming the user's insight that")
    lines.append("NOTHING normalises backward gradients by default, and doing so changes")
    lines.append("the robustness geometry.")
    lines.append("=" * 74)

    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
