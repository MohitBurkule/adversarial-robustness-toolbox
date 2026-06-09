"""
H352 - Spectral Alignment Regularisation via FFT

Penalise distance between FFT spectral representations of clean and adversarial inputs.
dct_clean = fft2(x).abs(), dct_adv = fft2(x_adv).abs().
penalty = ||dct_clean - dct_adv||². x_adv via FGSM(eps=0.1).
λ grid: [0, 0.01, 0.1, 1.0].
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
EPS = 0.1
LAMBDAS = [0, 0.01, 0.1, 1.0]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h352_spectral_alignment_regularisation_output.txt")


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


def spectral_penalty(x_clean, x_adv):
    """||FFT(x_clean).abs() - FFT(x_adv).abs()||^2 (mean over batch)."""
    fft_clean = torch.fft.fft2(x_clean).abs()
    fft_adv = torch.fft.fft2(x_adv).abs()
    return ((fft_clean - fft_adv) ** 2).mean()


def train_spectral(meta, Xtr, Ytr, lam):
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
            if lam > 0:
                model.eval()
                x_adv = C.fgsm(model, xb, yb, eps=EPS)
                model.train()
                loss_ce = F.cross_entropy(model(xb), yb)
                penalty = spectral_penalty(xb, x_adv)
                loss = loss_ce + lam * penalty
            else:
                loss = F.cross_entropy(model(xb), yb)

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("H352 - Spectral Alignment Regularisation via FFT")
    lines.append("=" * 72)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_TRAIN={N_TRAIN}  eps={EPS}")
    lines.append(f"lambda grid: {LAMBDAS}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    for lam in LAMBDAS:
        t0 = time.time()
        model = train_spectral(meta, Xtr, Ytr, lam)
        metrics = eval_model(model, Xte, Yte)
        metrics["lambda"] = lam
        metrics["runtime_s"] = round(time.time() - t0, 1)
        line = (f"  lambda={lam:<5}  clean={metrics['clean_acc']:.3f}  "
                f"fgsm_asr={metrics['fgsm_asr']:.3f}  pgd_asr={metrics['pgd_asr']:.3f}  "
                f"margin={metrics['mean_margin']:.3f}  ({metrics['runtime_s']}s)")
        lines.append(line)
        print(line)

    lines.append("")
    lines.append("Interpretation: Spectral alignment penalises frequency-domain differences")
    lines.append("between clean and adversarial inputs during training, encouraging the model")
    lines.append("to be invariant to high-frequency perturbations introduced by FGSM.")
    lines.append("=" * 72)

    out = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(out + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
