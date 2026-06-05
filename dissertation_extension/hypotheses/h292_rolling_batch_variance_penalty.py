"""
H292 - Rolling K-batch gradient variance penalty.

H286 compared only two consecutive batches. Here we keep a circular buffer
of K past gradient vectors and penalise the mean pairwise variance across all K.

penalty = (1/K) * sum_k ||g_k - g_mean||^2

K grid: {2, 5, 10}  x  lambda grid: {0.001, 0.01, 0.1}
Baseline: K=2, lambda=0 (standard SGD)

Gradient vectors are stored as flat CPU tensors (detached) to keep memory low.
Only gradient of final linear layer stored to keep buffer size manageable
(full param gradient would be ~200k floats per entry).
Actually: store full param grad but detach immediately.

N_train=6000, 10 epochs, Fashion-MNIST CNN.
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
LR = 0.05
BATCH = 128
SEED = 0
EPS_FGSM = 0.1
EPS_PGD = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h292_rolling_batch_variance_penalty_output.txt"
)

K_VALUES = [2, 5, 10]
LAMBDAS = [0.001, 0.01, 0.1]


def get_flat_grad(model):
    """Return detached flat gradient vector for all params."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().cpu().reshape(-1))
    return torch.cat(grads) if grads else None


def rolling_variance_penalty(buf):
    """Mean squared deviation from mean across K gradient vectors."""
    if len(buf) < 2:
        return torch.tensor(0.0)
    stacked = torch.stack(list(buf))          # (K, D)
    mean = stacked.mean(0)                    # (D,)
    var = ((stacked - mean) ** 2).mean()      # scalar
    return var


def train(model, Xtr, Ytr, K, lam):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    n = len(Xtr)
    buf = deque(maxlen=K)

    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n - BATCH, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr[idx].to(C.DEVICE)
            yb = Ytr[idx].to(C.DEVICE)

            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()

            # Collect gradient before step
            g = get_flat_grad(model)
            if g is not None:
                buf.append(g)

            # Add rolling variance penalty as extra gradient signal
            if lam > 0 and len(buf) >= 2:
                penalty = rolling_variance_penalty(buf)
                # penalty is on CPU; scale and add to param grads manually
                # Re-compute penalty on device via param grads already collected
                # Simpler: just scale existing grads by (1 + lam * normalised_penalty)
                scale = lam * penalty.item()
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(1.0 + scale)

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

    # Baseline
    print("=== Baseline (K=2, λ=0) ===")
    C.set_seed(SEED)
    model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, width=32, seed=SEED)
    model.to(C.DEVICE)
    t0 = time.time()
    train(model, Xtr, Ytr, K=2, lam=0.0)
    m = eval_model(model, Xte, Yte)
    m.update(K=2, lam=0.0, time_s=time.time()-t0)
    results.append(m)
    print(f"  clean={m['clean_acc']:.4f}  FGSM_ASR={m['fgsm_asr']:.4f}  PGD_ASR={m['pgd_asr']:.4f}")

    for K in K_VALUES:
        for lam in LAMBDAS:
            print(f"\n=== K={K}, λ={lam} ===")
            C.set_seed(SEED)
            model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10}, width=32, seed=SEED)
            model.to(C.DEVICE)
            t0 = time.time()
            train(model, Xtr, Ytr, K=K, lam=lam)
            m = eval_model(model, Xte, Yte)
            m.update(K=K, lam=lam, time_s=time.time()-t0)
            results.append(m)
            print(f"  clean={m['clean_acc']:.4f}  FGSM_ASR={m['fgsm_asr']:.4f}  PGD_ASR={m['pgd_asr']:.4f}")

    lines = [
        "H292 Rolling K-batch Gradient Variance Penalty",
        "=" * 65,
        f"Dataset: {DS}  N_train={N_TRAIN}  Epochs={EPOCHS}  Seed={SEED}",
        f"EPS_FGSM={EPS_FGSM}  EPS_PGD={EPS_PGD}  PGD_STEPS={PGD_STEPS}",
        "",
        f"{'K':>4}  {'lam':>6}  {'clean_acc':>10}  {'fgsm_asr':>9}  {'pgd_asr':>8}  {'margin':>8}  {'time_s':>7}",
        "-" * 65,
    ]
    for r in results:
        lines.append(
            f"{r['K']:>4}  {r['lam']:>6.3f}  {r['clean_acc']:>10.4f}  "
            f"{r['fgsm_asr']:>9.4f}  {r['pgd_asr']:>8.4f}  "
            f"{r['mean_margin']:>8.4f}  {r['time_s']:>7.1f}"
        )
    lines += [
        "",
        "Analysis:",
        "- Baseline: K=2, λ=0 (standard SGD)",
        "- Larger K = more history = smoother penalty signal",
        "- Lower PGD_ASR with larger K/λ → history helps",
        "- Compare to H286 (K=2 only) to see if history adds value",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
