"""
H356 - Lipschitz-Proportional Stochastic Depth (arXiv 2509.10298)

Drop layers with probability proportional to estimated Lipschitz constant.
L_k = ||∇_x h_k||_F (per-block via random projections).
p_drop_k = L_k / max(L_k) * p_max. p_max grid: [0, 0.1, 0.3].
Compare to fixed stochastic depth.
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
P_MAX_GRID = [0, 0.1, 0.3]
FIXED_P = 0.1  # for fixed stochastic depth comparison

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h356_lipschitz_stochastic_depth_output.txt")


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


class StochasticDepthCNN(nn.Module):
    """CNN with 3 conv blocks; supports Lipschitz-proportional or fixed stochastic depth."""
    def __init__(self, meta, width=32):
        super().__init__()
        ch, sz, ncls = meta["channels"], meta["size"], meta["n_classes"]
        self.block1 = nn.Sequential(
            nn.Conv2d(ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(width, width*2, 3, padding=1), nn.BatchNorm2d(width*2), nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(width*2, width*4, 3, padding=1), nn.BatchNorm2d(width*4), nn.ReLU(), nn.MaxPool2d(2))
        feat = sz // 8
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width*4*feat*feat, 256), nn.ReLU(),
                                  nn.Linear(256, ncls))
        self.drop_probs = [0.0, 0.0, 0.0]  # set externally per batch

    def forward(self, x):
        h = x
        for k, block in enumerate([self.block1, self.block2, self.block3]):
            h_block = block(h)
            if self.training and self.drop_probs[k] > 0:
                # Bernoulli drop: zero out this block's contribution for whole batch
                keep = (torch.rand(1, device=x.device).item() >= self.drop_probs[k])
                if not keep:
                    h_block = torch.zeros_like(h_block)
            h = h_block
        return self.head(h)


def estimate_lipschitz(model, xb, block_idx):
    """Estimate Lipschitz constant of block_idx via gradient Frobenius norm."""
    xb = xb.clone().detach().requires_grad_(True)
    blocks = [model.block1, model.block2, model.block3]
    h = xb
    for k in range(block_idx + 1):
        h = blocks[k](h)
    # Random projection in output space
    v = torch.randn_like(h)
    scalar = (h * v).sum()
    g = torch.autograd.grad(scalar, xb)[0]
    return g.norm().item()


def train_condition(meta, Xtr, Ytr, mode, p_max=0.0):
    C.set_seed(SEED)
    model = StochasticDepthCNN(meta).to(C.DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            if mode == "lipschitz" and p_max > 0:
                # Estimate per-block Lipschitz
                model.eval()
                lk = [estimate_lipschitz(model, xb[:16], k) for k in range(3)]
                model.train()
                max_lk = max(lk) + 1e-8
                model.drop_probs = [l / max_lk * p_max for l in lk]
            elif mode == "fixed":
                model.drop_probs = [FIXED_P] * 3
            else:
                model.drop_probs = [0.0] * 3

            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.drop_probs = [0.0, 0.0, 0.0]
    model.eval()
    return model


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("H356 - Lipschitz-Proportional Stochastic Depth (arXiv 2509.10298)")
    lines.append("=" * 72)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_TRAIN={N_TRAIN}")
    lines.append(f"p_max grid: {P_MAX_GRID}  fixed_p={FIXED_P}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    # Baseline (no drop) + fixed stochastic depth + Lipschitz-proportional for p_max in grid
    conditions = [("no_drop", "baseline", 0)] + \
                 [("fixed_p0.1", "fixed", FIXED_P)] + \
                 [(f"lipschitz_p{p}", "lipschitz", p) for p in P_MAX_GRID if p > 0]

    for name, mode, p_max in conditions:
        t0 = time.time()
        model = train_condition(meta, Xtr, Ytr, mode, p_max)
        metrics = eval_model(model, Xte, Yte)
        metrics["runtime_s"] = round(time.time() - t0, 1)
        line = (f"  {name:<20}  clean={metrics['clean_acc']:.3f}  "
                f"fgsm_asr={metrics['fgsm_asr']:.3f}  pgd_asr={metrics['pgd_asr']:.3f}  "
                f"margin={metrics['mean_margin']:.3f}  ({metrics['runtime_s']}s)")
        lines.append(line)
        print(line)

    lines.append("")
    lines.append("Interpretation: Dropping high-Lipschitz blocks more often regularises")
    lines.append("the model towards lower sensitivity layers, potentially improving")
    lines.append("adversarial robustness beyond fixed-probability stochastic depth.")
    lines.append("=" * 72)

    out = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(out + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
