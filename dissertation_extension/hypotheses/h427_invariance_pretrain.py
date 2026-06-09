"""
Hypothesis H427: Enforced augmentation-invariance as soft adversarial training.

Hypothesis: Adding an explicit invariance penalty — L2 distance between embeddings
of an original image and its augmented counterpart (translate, flip, small rotation,
Gaussian noise) — during supervised training regularises the representation geometry
in a way that overlaps with the effect of L∞-AT, giving broader robustness without
an explicit adversary.

Motivation: Worrall et al. (2017, "Harmonic Networks: Deep Translation and Rotation
Equivariance") and later Lyle et al. (2020, "On the Benefits of Invariance in Neural
Networks") show that representations invariant to natural transforms occupy smoother
regions of input space.  If the invariance constraint compresses the directions most
exploited by gradient-based attacks, the model should be harder to fool even without
seeing adversarial examples.  The claim is not equivalence with AT but rather that
invariance acts as a *soft* AT across a different, transform-defined threat model.

Three conditions:
  (A) Baseline       — standard cross-entropy, no augmentation penalty.
  (B) +InvPenalty    — cross-entropy + λ * L2(embed(x), embed(aug(x))).
  (C) +InvPenalty+AT — condition B with added L∞-PGD inner loop (standard AT).

Metrics: clean accuracy, FGSM-flip rate, PGD-10-flip rate, mean min-eps to flip.
Three seeds, Fashion-MNIST.

Cite: Worrall et al. (CVPR 2017); Lyle et al. (NeurIPS 2020, "On the Benefits of
Invariance in Neural Networks").
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 12
BATCH = 128
EPS = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = 2.0 / 255.0
INV_LAMBDA = 1.0   # weight for invariance penalty
SEEDS = [0, 1, 2]
N_CLASSES = 10

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE = os.path.join(
    BASE_DIR, "results", "fashion_mnist", "h427_invariance_pretrain_output.txt"
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    """Small CNN — same architecture used throughout the campaign."""

    def __init__(self, n=N_CLASSES):
        super().__init__()
        self.c1  = nn.Conv2d(1, 32, 3)
        self.c2  = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def embed(self, x):
        """Return pre-logit embedding (128-d)."""
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.do2(self.embed(x)))


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment(x: torch.Tensor) -> torch.Tensor:
    """
    Apply a stack of natural transforms to a batch:
      - Random horizontal flip (p=0.5)
      - Random translation up to 4px (via affine)
      - Random rotation ±15 degrees
      - Additive Gaussian noise σ=0.05
    Returns clipped [0,1] tensor of same shape.
    """
    out = []
    for img in x:
        # img: (1, 28, 28)
        if torch.rand(1).item() > 0.5:
            img = TF.hflip(img)
        angle = (torch.rand(1).item() * 2 - 1) * 15.0
        tx    = int((torch.rand(1).item() * 2 - 1) * 4)
        ty    = int((torch.rand(1).item() * 2 - 1) * 4)
        img = TF.affine(img, angle=angle, translate=(tx, ty),
                        scale=1.0, shear=0)
        img = img + 0.05 * torch.randn_like(img)
        img = img.clamp(0.0, 1.0)
        out.append(img)
    return torch.stack(out)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def fgsm(model, x, y, eps=EPS):
    xr = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xr), y)
    loss.backward()
    with torch.no_grad():
        return (xr + eps * xr.grad.sign()).clamp(0, 1)


def pgd(model, x, y, eps=EPS, alpha=PGD_ALPHA, steps=PGD_STEPS):
    xa = x.clone().detach()
    for _ in range(steps):
        xa.requires_grad_()
        loss = F.cross_entropy(model(xa), y)
        loss.backward()
        with torch.no_grad():
            xa = (xa + alpha * xa.grad.sign()).clamp(
                x - eps, x + eps).clamp(0, 1)
    return xa.detach()


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=8):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    for _ in range(iters):
        mid = (lo + hi) / 2
        xa  = x.clone().detach()
        alp = mid / 4.0
        for _ in range(5):
            xa.requires_grad_()
            loss = F.cross_entropy(model(xa), y)
            loss.backward()
            with torch.no_grad():
                xa = (xa + alp.view(-1, 1, 1, 1) * xa.grad.sign()).clamp(
                    x - mid.view(-1, 1, 1, 1),
                    x + mid.view(-1, 1, 1, 1)).clamp(0, 1)
        with torch.no_grad():
            flipped = model(xa).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(train_loader, mode: str, seed: int) -> nn.Module:
    """
    mode: 'baseline' | 'inv' | 'inv_at'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CNN().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if mode == "inv_at":
                # AT inner loop first, then invariance on original
                x_adv = pgd(model, x, y)
                logits = model(x_adv)
            else:
                logits = model(x)

            ce_loss = F.cross_entropy(logits, y)

            if mode in ("inv", "inv_at"):
                x_aug   = augment(x).to(DEVICE)
                emb_orig = model.embed(x)
                with torch.no_grad() if mode == "inv" else torch.enable_grad():
                    pass  # placeholder — always compute aug embedding with grad
                emb_aug = model.embed(x_aug)
                inv_loss = F.mse_loss(emb_orig, emb_aug)
                loss = ce_loss + INV_LAMBDA * inv_loss
            else:
                loss = ce_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader):
    model.eval()
    results = {
        "clean_correct": 0, "fgsm_correct": 0,
        "pgd_correct": 0, "total": 0,
        "min_eps": [],
    }
    with torch.no_grad():
        pass  # used below for clean pass only

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        n = x.size(0)
        results["total"] += n

        with torch.no_grad():
            results["clean_correct"] += (model(x).argmax(1) == y).sum().item()

        xf = fgsm(model, x, y)
        with torch.no_grad():
            results["fgsm_correct"] += (model(xf).argmax(1) == y).sum().item()

        xp = pgd(model, x, y)
        with torch.no_grad():
            results["pgd_correct"] += (model(xp).argmax(1) == y).sum().item()

        me = min_eps_to_flip(model, x, y)
        results["min_eps"].append(me.cpu())

    N = results["total"]
    return {
        "clean_acc":   results["clean_correct"] / N,
        "fgsm_acc":    results["fgsm_correct"]  / N,
        "pgd_acc":     results["pgd_correct"]   / N,
        "fgsm_flip":   1.0 - results["fgsm_correct"] / N,
        "pgd_flip":    1.0 - results["pgd_correct"]  / N,
        "mean_min_eps": torch.cat(results["min_eps"]).mean().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.FashionMNIST(
        os.path.join(BASE_DIR, "data"), train=True,  download=True, transform=transform)
    test_set  = datasets.FashionMNIST(
        os.path.join(BASE_DIR, "data"), train=False, download=True, transform=transform)

    # Use a fixed 2000-sample subset for evaluation speed
    test_subset = torch.utils.data.Subset(
        test_set, list(range(2000)))
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)

    modes = ["baseline", "inv", "inv_at"]
    agg   = {m: {k: [] for k in
                 ["clean_acc", "fgsm_acc", "pgd_acc",
                  "fgsm_flip", "pgd_flip", "mean_min_eps"]}
             for m in modes}

    lines = []
    t0 = time.time()

    for seed in SEEDS:
        train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                                  num_workers=2, pin_memory=True)
        for mode in modes:
            lines.append(f"\n--- seed={seed}  mode={mode} ---")
            model = train(train_loader, mode=mode, seed=seed)
            res   = evaluate(model, test_loader)
            for k, v in res.items():
                agg[mode][k].append(v)
            lines.append(
                f"  clean={res['clean_acc']:.3f}  "
                f"fgsm_flip={res['fgsm_flip']:.3f}  "
                f"pgd_flip={res['pgd_flip']:.3f}  "
                f"mean_min_eps={res['mean_min_eps']:.4f}"
            )

    lines.append("\n=== Aggregated results (mean ± std across seeds) ===")
    for mode in modes:
        lines.append(f"\nMode: {mode}")
        for k, vals in agg[mode].items():
            arr = np.array(vals)
            lines.append(f"  {k:20s}: {arr.mean():.4f} ± {arr.std():.4f}")

    lines.append(f"\nElapsed: {time.time() - t0:.1f}s")

    # Interpret
    lines.append("\n=== Interpretation ===")
    b_flip  = np.mean(agg["baseline"]["pgd_flip"])
    iv_flip = np.mean(agg["inv"]["pgd_flip"])
    ia_flip = np.mean(agg["inv_at"]["pgd_flip"])
    lines.append(
        f"PGD flip: baseline={b_flip:.3f}  +inv={iv_flip:.3f}  +inv+AT={ia_flip:.3f}"
    )
    if iv_flip < b_flip - 0.02:
        lines.append(
            "SUPPORTED: invariance penalty alone reduces PGD-flip rate, "
            "consistent with the soft-AT hypothesis (Worrall et al. 2017; "
            "Lyle et al. 2020)."
        )
    else:
        lines.append(
            "NOT SUPPORTED: invariance penalty does not substantially reduce "
            "adversarial vulnerability; explicit AT (condition C) may still help."
        )

    report = "\n".join(lines)
    print(report)
    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
