"""
H293 - Gradient Acceleration Penalty.

Penalises second-order change in gradient trajectory:

  acceleration_t = g_t - 2*g_{t-1} + g_{t-2}
  penalty = ||acceleration_t||^2

Momentum damps gradient oscillations implicitly via exponential averaging.
This makes the smoothness requirement explicit and tunable.

Comparison conditions:
  (a) SGD no momentum, no penalty (λ=0)          — raw baseline
  (b) SGD + momentum=0.9, no penalty              — standard SGD
  (c) SGD no momentum + acceleration penalty (λ)  — penalty replaces momentum
  (d) SGD + momentum=0.9 + acceleration penalty   — penalty on top of momentum

Lambda grid: {0.001, 0.01, 0.1} for (c) and (d).

This isolates whether the penalty adds anything beyond what momentum already does.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
BATCH = 128
LR = 0.05
SEED = 0
EPS_FGSM = 0.1
EPS_PGD = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01

LAMBDAS = [0.001, 0.01, 0.1]

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h293_gradient_acceleration_penalty_output.txt"
)


def get_flat_grad(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().cpu().reshape(-1))
    return torch.cat(grads) if grads else None


def train(model, Xtr, Ytr, use_momentum, lam):
    mom = 0.9 if use_momentum else 0.0
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=mom, weight_decay=5e-4)
    n = len(Xtr)
    buf = deque(maxlen=3)   # need 3 to compute acceleration

    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n - BATCH, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr[idx].to(C.DEVICE)
            yb = Ytr[idx].to(C.DEVICE)

            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()

            g = get_flat_grad(model)
            if g is not None:
                buf.append(g)

            # Acceleration penalty: ||g_t - 2*g_{t-1} + g_{t-2}||^2
            if lam > 0 and len(buf) == 3:
                g0, g1, g2 = list(buf)
                accel = g2 - 2 * g1 + g0
                penalty_val = (accel ** 2).mean().item()
                # Scale existing gradients proportionally
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(1.0 + lam * penalty_val)

            opt.step()


def eval_model(model, Xte, Yte):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS_FGSM)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS_PGD, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))
    return dict(
        clean_acc=float(clean_acc),
        fgsm_asr=1.0 - float(acc_fgsm),
        pgd_asr=1.0 - float(acc_pgd),
        mean_margin=mean_margin,
    )


def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    idx = torch.randperm(len(Xtr))[:N_TRAIN]
    Xtr, Ytr = Xtr[idx], Ytr[idx]

    results = []

    # (a) No momentum, no penalty
    print("=== (a) SGD no momentum, λ=0 ===")
    C.set_seed(SEED)
    model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, width=32, seed=SEED)
    model.to(C.DEVICE)
    t0 = time.time()
    train(model, Xtr, Ytr, use_momentum=False, lam=0.0)
    m = eval_model(model, Xte, Yte)
    m.update(label="SGD no-mom λ=0", time_s=time.time()-t0)
    results.append(m)
    print(f"  clean={m['clean_acc']:.4f}  FGSM_ASR={m['fgsm_asr']:.4f}  PGD_ASR={m['pgd_asr']:.4f}")

    # (b) Momentum, no penalty
    print("\n=== (b) SGD momentum=0.9, λ=0 ===")
    C.set_seed(SEED)
    model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, width=32, seed=SEED)
    model.to(C.DEVICE)
    t0 = time.time()
    train(model, Xtr, Ytr, use_momentum=True, lam=0.0)
    m = eval_model(model, Xte, Yte)
    m.update(label="SGD mom=0.9 λ=0", time_s=time.time()-t0)
    results.append(m)
    print(f"  clean={m['clean_acc']:.4f}  FGSM_ASR={m['fgsm_asr']:.4f}  PGD_ASR={m['pgd_asr']:.4f}")

    # (c) No momentum + acceleration penalty
    for lam in LAMBDAS:
        print(f"\n=== (c) SGD no-mom + accel penalty λ={lam} ===")
        C.set_seed(SEED)
        model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, width=32, seed=SEED)
        model.to(C.DEVICE)
        t0 = time.time()
        train(model, Xtr, Ytr, use_momentum=False, lam=lam)
        m = eval_model(model, Xte, Yte)
        m.update(label=f"no-mom+accel λ={lam}", time_s=time.time()-t0)
        results.append(m)
        print(f"  clean={m['clean_acc']:.4f}  FGSM_ASR={m['fgsm_asr']:.4f}  PGD_ASR={m['pgd_asr']:.4f}")

    # (d) Momentum + acceleration penalty
    for lam in LAMBDAS:
        print(f"\n=== (d) SGD mom=0.9 + accel penalty λ={lam} ===")
        C.set_seed(SEED)
        model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, width=32, seed=SEED)
        model.to(C.DEVICE)
        t0 = time.time()
        train(model, Xtr, Ytr, use_momentum=True, lam=lam)
        m = eval_model(model, Xte, Yte)
        m.update(label=f"mom+accel λ={lam}", time_s=time.time()-t0)
        results.append(m)
        print(f"  clean={m['clean_acc']:.4f}  FGSM_ASR={m['fgsm_asr']:.4f}  PGD_ASR={m['pgd_asr']:.4f}")

    lines = [
        "H293 Gradient Acceleration Penalty vs Momentum",
        "=" * 65,
        f"Dataset: {DS}  N_train={N_TRAIN}  Epochs={EPOCHS}  Seed={SEED}",
        f"EPS_FGSM={EPS_FGSM}  EPS_PGD={EPS_PGD}  PGD_STEPS={PGD_STEPS}",
        "",
        f"{'Condition':<28}  {'clean_acc':>10}  {'fgsm_asr':>9}  {'pgd_asr':>8}  {'margin':>8}  {'time_s':>7}",
        "-" * 75,
    ]
    for r in results:
        lines.append(
            f"{r['label']:<28}  {r['clean_acc']:>10.4f}  {r['fgsm_asr']:>9.4f}  "
            f"{r['pgd_asr']:>8.4f}  {r['mean_margin']:>8.4f}  {r['time_s']:>7.1f}"
        )
    lines += [
        "",
        "Key comparisons:",
        "  (a) vs (b): does momentum alone improve robustness?",
        "  (a) vs (c): does acceleration penalty alone improve robustness?",
        "  (b) vs (d): does acceleration penalty add value ON TOP of momentum?",
        "  (c) vs (b): can penalty substitute for momentum?",
        "  If (c)~=(b): penalty and momentum are equivalent mechanisms.",
        "  If (d)>(b): penalty adds orthogonal signal beyond momentum.",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
