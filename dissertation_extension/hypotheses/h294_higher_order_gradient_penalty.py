"""
H294 - Higher-order input gradient penalties.

H288 showed input gradient norm penalty (||∇_x L||^2) is best implicit method.
This extends to higher orders:

  Order 1: penalise ||∇_x L||^2                     (H288 winner)
  Order 2: penalise ||∇_x (||∇_x L||^2)||^2         (sensitivity of sensitivity)
  Combined: penalise order-1 + order-2 jointly

Order-2 penalises how much the gradient norm changes w.r.t. the input —
i.e., it smooths the gradient field rather than just its magnitude.
Gradient field smoothness is directly linked to transferability of adversarial
examples (smoother gradient field = harder to find universal perturbations).

Implementation:
  Order 1: compute grad_x = autograd.grad(loss, x, create_graph=True)
           penalty_1 = (grad_x ** 2).sum()
  Order 2: penalty_2 = (autograd.grad(penalty_1, x)[0] ** 2).sum()
  total = loss + lam1 * penalty_1 + lam2 * penalty_2

Lambda grid: lam1 in {0, 0.01}, lam2 in {0, 0.001, 0.01}
(lam1=0, lam2>0 = order-2 only; lam1>0, lam2=0 = order-1 only as baseline)

N_train=6000, 10 epochs, Fashion-MNIST.
Note: order-2 requires 3 backward passes (expensive).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS_FGSM = 0.1
EPS_PGD = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01

# (lam1, lam2) conditions
CONDITIONS = [
    (0.0,  0.0,   "baseline"),
    (0.01, 0.0,   "order1 only (λ1=0.01)"),
    (0.0,  0.001, "order2 only (λ2=0.001)"),
    (0.0,  0.01,  "order2 only (λ2=0.01)"),
    (0.01, 0.001, "order1+2 (λ1=0.01, λ2=0.001)"),
    (0.01, 0.01,  "order1+2 (λ1=0.01, λ2=0.01)"),
]

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h294_higher_order_gradient_penalty_output.txt"
)


def train(model, Xtr, Ytr, lam1, lam2):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    n = len(Xtr)

    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n - BATCH, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr[idx].to(C.DEVICE).requires_grad_(True)
            yb = Ytr[idx].to(C.DEVICE)

            opt.zero_grad()

            if lam1 == 0.0 and lam2 == 0.0:
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
            elif lam2 == 0.0:
                # Order-1 only
                loss = F.cross_entropy(model(xb), yb)
                grad_x = torch.autograd.grad(loss, xb, create_graph=True)[0]
                penalty_1 = (grad_x ** 2).sum()
                total = loss + lam1 * penalty_1
                total.backward()
            else:
                # Order-2 (optionally + order-1)
                loss = F.cross_entropy(model(xb), yb)
                grad_x = torch.autograd.grad(loss, xb, create_graph=True, retain_graph=True)[0]
                penalty_1 = (grad_x ** 2).sum()
                grad2_x = torch.autograd.grad(penalty_1, xb, create_graph=False, retain_graph=True)[0]
                penalty_2 = (grad2_x ** 2).sum()
                total = loss + lam1 * penalty_1 + lam2 * penalty_2
                total.backward()

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
    for lam1, lam2, label in CONDITIONS:
        print(f"\n=== {label} ===")
        C.set_seed(SEED)
        model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, width=32, seed=SEED)
        model.to(C.DEVICE)
        t0 = time.time()
        train(model, Xtr, Ytr, lam1=lam1, lam2=lam2)
        elapsed = time.time() - t0
        m = eval_model(model, Xte, Yte)
        m.update(lam1=lam1, lam2=lam2, label=label, time_s=elapsed)
        results.append(m)
        print(f"  clean={m['clean_acc']:.4f}  FGSM_ASR={m['fgsm_asr']:.4f}  "
              f"PGD_ASR={m['pgd_asr']:.4f}  margin={m['mean_margin']:.4f}  time={elapsed:.1f}s")

    lines = [
        "H294 Higher-Order Input Gradient Penalties",
        "=" * 70,
        f"Dataset: {DS}  N_train={N_TRAIN}  Epochs={EPOCHS}  Seed={SEED}",
        f"EPS_FGSM={EPS_FGSM}  EPS_PGD={EPS_PGD}  PGD_STEPS={PGD_STEPS}",
        "",
        f"{'Condition':<35}  {'clean_acc':>10}  {'fgsm_asr':>9}  {'pgd_asr':>8}  {'margin':>8}  {'time_s':>7}",
        "-" * 80,
    ]
    for r in results:
        lines.append(
            f"{r['label']:<35}  {r['clean_acc']:>10.4f}  {r['fgsm_asr']:>9.4f}  "
            f"{r['pgd_asr']:>8.4f}  {r['mean_margin']:>8.4f}  {r['time_s']:>7.1f}"
        )
    lines += [
        "",
        "Analysis:",
        "- Order-1 only = H288 winner (input grad norm penalty)",
        "- Order-2 only = penalises how much sensitivity varies w.r.t. input",
        "- Order-1+2 = joint penalty on gradient magnitude AND field smoothness",
        "- Lower PGD_ASR with order-2 → gradient field smoothness helps beyond magnitude",
        "- Compute cost: order-2 needs 3 backward passes per step",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
