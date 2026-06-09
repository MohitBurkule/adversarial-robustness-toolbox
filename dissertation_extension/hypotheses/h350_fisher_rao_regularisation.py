"""
H350 - Fisher-Rao Regularisation (FIRE, Picot et al. 2021)

Penalise geodesic distance (Fisher-Rao metric) between clean/adversarial distributions.
Approximate: d_FR(p, q) ≈ arccos(sum(sqrt(p*q))) (Bhattacharyya arc distance).
λ grid: [0, 0.1, 1.0, 10.0]. x_adv via FGSM(eps=0.1).
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
LAMBDAS = [0, 0.1, 1.0, 10.0]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h350_fisher_rao_regularisation_output.txt")


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


def bhattacharyya_arc_distance(p, q):
    """Fisher-Rao approx: 1 - Bhattacharyya coefficient (stable, bounded [0,1])."""
    bc = (p.clamp(min=1e-8) * q.clamp(min=1e-8)).sqrt().sum(dim=1)
    return (1.0 - bc).mean()


def train_fire(meta, Xtr, Ytr, lam):
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

            if lam > 0:
                model.eval()
                x_adv = C.fgsm(model, xb, yb, eps=EPS)
                model.train()
                opt.zero_grad()
                out_clean = model(xb)
                loss_ce = F.cross_entropy(out_clean, yb)
                with torch.no_grad():
                    p_adv = F.softmax(model(x_adv), dim=1)
                p_clean = F.softmax(out_clean, dim=1)
                penalty = bhattacharyya_arc_distance(p_clean, p_adv.detach())
                loss = loss_ce + lam * penalty
            else:
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb)

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("H350 - Fisher-Rao Regularisation (FIRE, Picot et al. 2021)")
    lines.append("=" * 72)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_TRAIN={N_TRAIN}  eps={EPS}")
    lines.append(f"lambda grid: {LAMBDAS}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    for lam in LAMBDAS:
        t0 = time.time()
        model = train_fire(meta, Xtr, Ytr, lam)
        metrics = eval_model(model, Xte, Yte)
        metrics["lambda"] = lam
        metrics["runtime_s"] = round(time.time() - t0, 1)
        line = (f"  lambda={lam:<5}  clean={metrics['clean_acc']:.3f}  "
                f"fgsm_asr={metrics['fgsm_asr']:.3f}  pgd_asr={metrics['pgd_asr']:.3f}  "
                f"margin={metrics['mean_margin']:.3f}  ({metrics['runtime_s']}s)")
        lines.append(line)
        print(line)

    lines.append("")
    lines.append("Interpretation: Fisher-Rao distance is a geodesic on the statistical")
    lines.append("manifold. Penalising d_FR(p_clean, p_adv) encourages distributional")
    lines.append("stability under FGSM perturbations. Compare to KL-based TRADES (H346).")
    lines.append("=" * 72)

    out = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(out + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
