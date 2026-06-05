"""
H355 - OT + Jacobian Regularisation (OTJR, CVPR 2023)

penalty = SW(f_features(x), f_features(x_adv)) + λ_J * ||J||_F²
SW approximated via K=20 random 1D projections. x_adv via FGSM.
λ_OT grid: [0, 0.1, 1.0]; λ_J=0.001 fixed.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
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
K_PROJ = 20
LAM_J = 0.001
LAMBDAS_OT = [0, 0.1, 1.0]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h355_ot_jacobian_regularisation_output.txt")


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


class CNNWithFeatures(nn.Module):
    """SmallCNN that also exposes intermediate feature maps."""
    def __init__(self, meta, width=32):
        super().__init__()
        ch, sz, ncls = meta["channels"], meta["size"], meta["n_classes"]
        self.features = nn.Sequential(
            nn.Conv2d(ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width*2, 3, padding=1), nn.BatchNorm2d(width*2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width*2, width*4, 3, padding=1), nn.BatchNorm2d(width*4), nn.ReLU(), nn.MaxPool2d(2),
        )
        feat = sz // 8
        feat_dim = width * 4 * feat * feat
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(feat_dim, 256), nn.ReLU(),
                                  nn.Linear(256, ncls))

    def forward(self, x):
        return self.head(self.features(x))

    def get_features(self, x):
        return self.features(x).flatten(1)


def sliced_wasserstein(f_clean, f_adv, K=20):
    """Sliced Wasserstein distance via K random projections."""
    d = f_clean.size(1)
    projs = F.normalize(torch.randn(K, d, device=f_clean.device), dim=1)
    sw = torch.tensor(0.0, device=f_clean.device)
    for k in range(K):
        p = projs[k]  # (d,)
        proj_c = f_clean @ p  # (N,)
        proj_a = f_adv @ p    # (N,)
        proj_c_sorted, _ = proj_c.sort()
        proj_a_sorted, _ = proj_a.sort()
        sw = sw + (proj_c_sorted - proj_a_sorted).pow(2).mean()
    return sw / K


def jacobian_frobenius(model, x):
    """||J||_F² via sum of squared gradients (one-hot output trick)."""
    x_req = x.clone().detach().requires_grad_(True)
    out = model(x_req)
    ncls = out.size(1)
    jf2 = torch.tensor(0.0, device=x.device)
    for c in range(ncls):
        g = torch.autograd.grad(out[:, c].sum(), x_req, retain_graph=True,
                                create_graph=True)[0]
        jf2 = jf2 + g.flatten(1).pow(2).sum(dim=1).mean()
    return jf2


def train_otjr(meta, Xtr, Ytr, lam_ot):
    C.set_seed(SEED)
    model = CNNWithFeatures(meta).to(C.DEVICE)
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
            if lam_ot > 0 or LAM_J > 0:
                model.eval()
                x_adv = C.fgsm(model, xb, yb, eps=EPS)
                model.train()

                loss_ce = F.cross_entropy(model(xb), yb)
                penalty = torch.tensor(0.0, device=xb.device)

                if lam_ot > 0:
                    f_clean = model.get_features(xb)
                    f_adv = model.get_features(x_adv)
                    sw = sliced_wasserstein(f_clean, f_adv.detach(), K_PROJ)
                    penalty = penalty + lam_ot * sw

                if LAM_J > 0:
                    jf = jacobian_frobenius(model, xb)
                    penalty = penalty + LAM_J * jf

                loss = loss_ce + penalty
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
    lines.append("H355 - OT + Jacobian Regularisation (OTJR, CVPR 2023)")
    lines.append("=" * 72)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_TRAIN={N_TRAIN}  eps={EPS}")
    lines.append(f"lambda_OT grid: {LAMBDAS_OT}  lambda_J={LAM_J}  K_proj={K_PROJ}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    for lam_ot in LAMBDAS_OT:
        t0 = time.time()
        model = train_otjr(meta, Xtr, Ytr, lam_ot)
        metrics = eval_model(model, Xte, Yte)
        metrics["lambda_ot"] = lam_ot
        metrics["runtime_s"] = round(time.time() - t0, 1)
        line = (f"  lam_ot={lam_ot:<4}  clean={metrics['clean_acc']:.3f}  "
                f"fgsm_asr={metrics['fgsm_asr']:.3f}  pgd_asr={metrics['pgd_asr']:.3f}  "
                f"margin={metrics['mean_margin']:.3f}  ({metrics['runtime_s']}s)")
        lines.append(line)
        print(line)

    lines.append("")
    lines.append("Interpretation: Sliced Wasserstein aligns feature distributions for")
    lines.append("clean/adversarial inputs; Jacobian Frobenius term constrains sensitivity.")
    lines.append("Combined (OTJR) should outperform each alone.")
    lines.append("=" * 72)

    out = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(out + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
