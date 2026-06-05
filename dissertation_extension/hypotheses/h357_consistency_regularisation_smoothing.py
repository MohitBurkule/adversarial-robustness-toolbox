"""
H357 - Consistency Regularisation for Randomized Smoothing (NeurIPS 2020)

Loss = CE + λ * Var(f(x+η_k) across k), η_k ~ N(0, σ²), K=4.
Forces classifier to be consistent under Gaussian noise = better certified radii.
σ=0.1, λ grid: [0, 0.1, 1.0, 10.0].
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
SIGMA = 0.1
K_NOISE = 4
LAMBDAS = [0, 0.1, 1.0, 10.0]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h357_consistency_regularisation_smoothing_output.txt")


def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1 - float(acc_fgsm),
                pgd_asr=1 - float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))


def smoothed_acc(model, Xte, Yte, sigma, K=100):
    """Informal smoothed accuracy: majority vote over K noise samples."""
    model.eval()
    n = Xte.size(0)
    votes = torch.zeros(n, 10, device=Xte.device)
    with torch.no_grad():
        for _ in range(K):
            noise = torch.randn_like(Xte) * sigma
            out = model((Xte + noise).clamp(0, 1))
            preds = out.argmax(1)
            for i in range(n):
                votes[i, preds[i]] += 1
    smoothed_preds = votes.argmax(1)
    return float((smoothed_preds == Yte).float().mean())


def train_consistency(meta, Xtr, Ytr, lam):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            opt.zero_grad()
            # CE on clean
            loss_ce = F.cross_entropy(model(xb), yb)

            if lam > 0:
                # K noise samples -> softmax outputs
                outputs = []
                for k in range(K_NOISE):
                    noise = torch.randn_like(xb) * SIGMA
                    x_noisy = (xb + noise).clamp(0, 1)
                    out_k = F.softmax(model(x_noisy), dim=1)
                    outputs.append(out_k)
                # Stack: (K, B, C)
                stacked = torch.stack(outputs, dim=0)  # K x B x C
                # Variance across K noise samples, mean over B and C
                var_loss = stacked.var(dim=0).mean()
                loss = loss_ce + lam * var_loss
            else:
                loss = loss_ce

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("H357 - Consistency Regularisation for Randomized Smoothing (NeurIPS 2020)")
    lines.append("=" * 72)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_TRAIN={N_TRAIN}  sigma={SIGMA}  K={K_NOISE}")
    lines.append(f"lambda grid: {LAMBDAS}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    for lam in LAMBDAS:
        t0 = time.time()
        model = train_consistency(meta, Xtr, Ytr, lam)
        metrics = eval_model(model, Xte, Yte)
        # Also compute informal smoothed accuracy
        sm_acc = smoothed_acc(model, Xte, Yte, SIGMA, K=50)
        metrics["lambda"] = lam
        metrics["smoothed_acc"] = sm_acc
        metrics["runtime_s"] = round(time.time() - t0, 1)
        line = (f"  lambda={lam:<5}  clean={metrics['clean_acc']:.3f}  "
                f"smoothed={sm_acc:.3f}  "
                f"fgsm_asr={metrics['fgsm_asr']:.3f}  pgd_asr={metrics['pgd_asr']:.3f}  "
                f"margin={metrics['mean_margin']:.3f}  ({metrics['runtime_s']}s)")
        lines.append(line)
        print(line)

    lines.append("")
    lines.append("Interpretation: Consistency regularisation forces output distributions")
    lines.append("to be stable under Gaussian noise, which improves certified robustness")
    lines.append("via randomized smoothing. Higher lambda should raise smoothed_acc while")
    lines.append("also improving adversarial robustness (noise and adversarial correlation).")
    lines.append("=" * 72)

    out = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(out + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
