"""
H348 - Supervised Contrastive Adversarial Training (arXiv 2412.19747, Dec 2024)

Replace CE with SupCon loss on adversarial examples. Temperature τ=0.1.
Conditions: CE baseline, FGSM-AT (CE), SupCon on clean, SupCon on adversarial.
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
TAU = 0.1

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h348_supervised_contrastive_adversarial_output.txt")


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


class ModelWithHead(nn.Module):
    """CNN with a projection head for contrastive training + linear head for eval."""
    def __init__(self, meta, proj_dim=128):
        super().__init__()
        ch, sz, ncls = meta["channels"], meta["size"], meta["n_classes"]
        width = 32
        self.features = nn.Sequential(
            nn.Conv2d(ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width*2, 3, padding=1), nn.BatchNorm2d(width*2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width*2, width*4, 3, padding=1), nn.BatchNorm2d(width*4), nn.ReLU(), nn.MaxPool2d(2),
        )
        feat = sz // 8
        feat_dim = width * 4 * feat * feat
        self.flatten = nn.Flatten()
        self.proj = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, proj_dim))
        self.head = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, ncls))

    def forward(self, x):
        h = self.flatten(self.features(x))
        return self.head(h)

    def encode(self, x):
        h = self.flatten(self.features(x))
        z = self.proj(h)
        return F.normalize(z, dim=1)


def supcon_loss(z, y, tau=0.1):
    """Supervised Contrastive Loss."""
    n = z.size(0)
    # Cosine similarity matrix
    sim = torch.mm(z, z.t()) / tau
    # Mask diagonal
    mask_self = torch.eye(n, device=z.device).bool()
    sim = sim.masked_fill(mask_self, -1e9)
    # Positive mask: same class, not self
    y_col = y.unsqueeze(1)
    pos_mask = (y_col == y_col.t()) & ~mask_self
    # For numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    exp_sim = torch.exp(sim - sim_max.detach())
    log_prob = sim - sim_max.detach() - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    # Mean over positives
    n_pos = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos
    return loss.mean()


def train_condition(meta, Xtr, Ytr, condition):
    C.set_seed(SEED)
    if condition in ("supcon_clean", "supcon_adv"):
        model = ModelWithHead(meta).to(C.DEVICE)
    else:
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
            if condition == "ce_baseline":
                loss = F.cross_entropy(model(xb), yb)
            elif condition == "fgsm_at":
                model.eval()
                x_adv = C.fgsm(model, xb, yb, eps=EPS)
                model.train()
                loss = F.cross_entropy(model(x_adv), yb)
            elif condition == "supcon_clean":
                z = model.encode(xb)
                loss = supcon_loss(z, yb, TAU)
            elif condition == "supcon_adv":
                model.eval()
                x_adv = C.fgsm(model, xb, yb, eps=EPS)
                model.train()
                z = model.encode(x_adv)
                loss = supcon_loss(z, yb, TAU)

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("H348 - Supervised Contrastive Adversarial Training (arXiv 2412.19747)")
    lines.append("=" * 72)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_TRAIN={N_TRAIN}  eps={EPS}  tau={TAU}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    conditions = ["ce_baseline", "fgsm_at", "supcon_clean", "supcon_adv"]
    for cond in conditions:
        t0 = time.time()
        model = train_condition(meta, Xtr, Ytr, cond)
        metrics = eval_model(model, Xte, Yte)
        metrics["condition"] = cond
        metrics["runtime_s"] = round(time.time() - t0, 1)
        line = (f"  {cond:<18}  clean={metrics['clean_acc']:.3f}  "
                f"fgsm_asr={metrics['fgsm_asr']:.3f}  pgd_asr={metrics['pgd_asr']:.3f}  "
                f"margin={metrics['mean_margin']:.3f}  ({metrics['runtime_s']}s)")
        lines.append(line)
        print(line)

    lines.append("")
    lines.append("Interpretation: SupCon on adversarial examples should improve robustness")
    lines.append("by clustering same-class adv representations and separating classes.")
    lines.append("SupCon-clean vs SupCon-adv isolates the adversarial training effect.")
    lines.append("=" * 72)

    out = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(out + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
