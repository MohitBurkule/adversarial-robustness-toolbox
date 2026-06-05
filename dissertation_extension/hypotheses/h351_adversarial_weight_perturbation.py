"""
H351 - Adversarial Weight Perturbation (AWP, Wu et al. NeurIPS 2020)

Double perturbation: perturb inputs (FGSM) AND weights adversarially.
δ_w = γ * ∇_w L(x_adv) / ||∇_w L||. Apply δ_w, compute loss, backprop, restore weights.
γ grid: [0, 0.001, 0.005, 0.01].
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
GAMMAS = [0, 0.001, 0.005, 0.01]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h351_adversarial_weight_perturbation_output.txt")


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


def train_awp(meta, Xtr, Ytr, gamma):
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

            # Step 1: FGSM on input
            model.eval()
            x_adv = C.fgsm(model, xb, yb, eps=EPS)
            model.train()

            if gamma > 0:
                # Step 2: Compute weight gradient on adversarial examples
                opt.zero_grad()
                loss_adv = F.cross_entropy(model(x_adv), yb)
                loss_adv.backward()
                # Collect gradients and compute weight perturbation
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5 + 1e-8

                # Apply weight perturbation
                saved_params = []
                for p in model.parameters():
                    saved_params.append(p.data.clone())
                    if p.grad is not None:
                        p.data.add_(gamma / total_norm * p.grad.data)

                # Step 3: Compute final loss with perturbed weights
                opt.zero_grad()
                loss = F.cross_entropy(model(x_adv), yb)
                loss.backward()

                # Restore original weights
                for p, orig in zip(model.parameters(), saved_params):
                    p.data.copy_(orig)
            else:
                # Baseline: FGSM-AT without weight perturbation
                opt.zero_grad()
                loss = F.cross_entropy(model(x_adv), yb)
                loss.backward()

            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("H351 - Adversarial Weight Perturbation (AWP, Wu et al. NeurIPS 2020)")
    lines.append("=" * 72)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_TRAIN={N_TRAIN}  eps={EPS}")
    lines.append(f"gamma grid: {GAMMAS}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    for gamma in GAMMAS:
        t0 = time.time()
        model = train_awp(meta, Xtr, Ytr, gamma)
        metrics = eval_model(model, Xte, Yte)
        metrics["gamma"] = gamma
        metrics["runtime_s"] = round(time.time() - t0, 1)
        line = (f"  gamma={gamma:<6}  clean={metrics['clean_acc']:.3f}  "
                f"fgsm_asr={metrics['fgsm_asr']:.3f}  pgd_asr={metrics['pgd_asr']:.3f}  "
                f"margin={metrics['mean_margin']:.3f}  ({metrics['runtime_s']}s)")
        lines.append(line)
        print(line)

    lines.append("")
    lines.append("Interpretation: AWP perturbs both inputs and weights adversarially,")
    lines.append("seeking flatter loss landscapes around adversarial examples. gamma=0")
    lines.append("is baseline FGSM-AT; larger gamma adds weight perturbation benefit.")
    lines.append("=" * 72)

    out = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(out + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
