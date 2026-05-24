"""
h78_label_smoothing.py

Hypothesis: Label-smoothing training affects per-sample vulnerability
predictability. Specifically: training a victim with label smoothing alters
the per-sample geometry around the decision boundary, and may therefore
change which simple, model-free or model-light features (margin, mean pixel,
std pixel, sobel mean) best predict per-sample adversarial vulnerability.

Procedure (self-contained):
  1. Train two victims on Fashion-MNIST for 10 epochs each, small CNN
     matching diagnostic_test.py:
        - Victim V:    vanilla cross-entropy
        - Victim S:    label smoothing with epsilon = 0.1
  2. For each victim, compute four per-sample features on the test set:
        - margin:       (top1 - top2) logit gap of the victim's final model
        - mean_pix:     mean pixel value of the input
        - std_pix:      std of pixel values of the input
        - sobel_mean:   mean magnitude of the Sobel gradient of the input
  3. For each victim, compute three vulnerability targets on samples the
     victim classifies correctly:
        - FGSM_flip at eps = 15/255 (binary)
        - PGD_flip  at eps = 15/255, 20 steps, step = 2/255 (binary)
        - min_eps:  smallest L_inf eps that flips an FGSM attack (continuous,
                    binarised at its median to give an AUROC-friendly target)
  4. Compute univariate AUROC of each feature against each binary target,
     for each victim. Report feature rankings per victim and the change in
     ranking induced by label smoothing.

This file only DEFINES the experiment. It does not run automatically; the
__main__ block calls main() but is gated so importing the module is cheap.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_STEP = 2.0 / 255.0
N_CLASSES = 10


# ---------------------------------------------------------------------------
# Model (matches diagnostic_test.py)
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, n: int = 10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_victim(seed: int, train_set, label_smoothing: float = 0.0) -> CNN:
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        print(f"   [ls={label_smoothing}] epoch {ep+1}/{EPOCHS} ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def batched_logits(model: CNN, x: torch.Tensor, batch: int = 512) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            out.append(model(x[i:i + batch]))
    return torch.cat(out, 0)


def sobel_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean Sobel-gradient magnitude per sample. x in [B,1,H,W]."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag.mean(dim=(1, 2, 3))


def compute_features(model: CNN, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    logits = batched_logits(model, x)
    s, _ = logits.sort(1, descending=True)
    margin = (s[:, 0] - s[:, 1]).detach()
    mp = x.mean(dim=(1, 2, 3)).detach()
    sp = x.std(dim=(1, 2, 3)).detach()
    sm = sobel_mean(x).detach()
    return {"margin": margin, "mean_pix": mp, "std_pix": sp, "sobel_mean": sm}


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def fgsm_grad(model: CNN, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model: CNN, x: torch.Tensor, y: torch.Tensor,
              eps: float = EPS_TEST) -> torch.Tensor:
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model: CNN, x: torch.Tensor, y: torch.Tensor,
             eps: float = EPS_TEST, steps: int = PGD_STEPS,
             step_size: float = PGD_STEP) -> torch.Tensor:
    # untargeted L_inf PGD with random start
    delta = torch.empty_like(x).uniform_(-eps, eps)
    adv = (x + delta).clamp(0, 1)
    for _ in range(steps):
        adv = adv.clone().detach().requires_grad_(True)
        F.cross_entropy(model(adv), y).backward()
        with torch.no_grad():
            adv = adv + step_size * adv.grad.sign()
            adv = torch.max(torch.min(adv, x + eps), x - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model: CNN, x: torch.Tensor, y: torch.Tensor,
                    eps_max: float = 0.3, iters: int = 15) -> torch.Tensor:
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# ---------------------------------------------------------------------------
# AUROC helpers
# ---------------------------------------------------------------------------
def auroc(feature: np.ndarray, target: np.ndarray) -> float:
    if np.unique(target).size < 2:
        return float("nan")
    a = roc_auc_score(target, feature)
    return max(a, 1 - a)


def evaluate_victim(name: str, model: CNN, test_x: torch.Tensor,
                    test_y: torch.Tensor) -> Dict:
    print(f"\n=== Evaluating victim: {name} ===")
    feats = compute_features(model, test_x)
    with torch.no_grad():
        preds = batched_logits(model, test_x).argmax(1)
    correct = preds == test_y
    print(f"  clean acc = {correct.float().mean().item():.4f}")
    idx = correct.nonzero(as_tuple=True)[0]
    xc = test_x[idx]
    yc = test_y[idx]
    feats_c = {k: v[idx] for k, v in feats.items()}

    # targets, batched
    fgsm_chunks, pgd_chunks, me_chunks = [], [], []
    for i in range(0, xc.size(0), 256):
        xb = xc[i:i + 256]; yb = yc[i:i + 256]
        fgsm_chunks.append(fgsm_flip(model, xb, yb))
        pgd_chunks.append(pgd_flip(model, xb, yb))
        me_chunks.append(min_eps_to_flip(model, xb, yb))
    fgsm = torch.cat(fgsm_chunks).cpu().numpy().astype(int)
    pgd  = torch.cat(pgd_chunks).cpu().numpy().astype(int)
    me   = torch.cat(me_chunks).cpu().numpy()
    me_bin = (me <= np.median(me)).astype(int)  # 1 = more vulnerable

    print(f"  FGSM flip rate = {fgsm.mean():.4f}")
    print(f"  PGD  flip rate = {pgd.mean():.4f}")
    print(f"  median min_eps = {np.median(me):.4f}")

    feat_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]
    target_names = ["FGSM", "PGD", "min_eps_below_median"]
    targets = {"FGSM": fgsm, "PGD": pgd, "min_eps_below_median": me_bin}

    aurocs: Dict[str, Dict[str, float]] = {}
    for tn in target_names:
        aurocs[tn] = {}
        for fn in feat_names:
            f_np = feats_c[fn].cpu().numpy()
            aurocs[tn][fn] = auroc(f_np, targets[tn])

    # print AUROCs and rankings
    for tn in target_names:
        print(f"\n  target = {tn}")
        ranked = sorted(aurocs[tn].items(), key=lambda kv: -kv[1])
        for i, (fn, a) in enumerate(ranked):
            print(f"    rank {i+1}  {fn:<12} AUROC={a:.4f}")
    return {"name": name, "aurocs": aurocs, "feat_names": feat_names,
            "target_names": target_names}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_rankings(res_vanilla: Dict, res_smooth: Dict) -> None:
    print("\n========== Ranking comparison (vanilla vs label-smoothing 0.1) ==========")
    feat_names = res_vanilla["feat_names"]
    target_names = res_vanilla["target_names"]
    for tn in target_names:
        va = res_vanilla["aurocs"][tn]
        sa = res_smooth["aurocs"][tn]
        rank_v = {fn: i for i, (fn, _) in enumerate(
            sorted(va.items(), key=lambda kv: -kv[1]))}
        rank_s = {fn: i for i, (fn, _) in enumerate(
            sorted(sa.items(), key=lambda kv: -kv[1]))}
        print(f"\n  target = {tn}")
        print(f"    {'feature':<12} {'AUROC_van':>10} {'rank_van':>9} "
              f"{'AUROC_smo':>10} {'rank_smo':>9} {'dAUROC':>8} {'drank':>6}")
        for fn in feat_names:
            d_auc = sa[fn] - va[fn]
            d_rank = rank_s[fn] - rank_v[fn]
            print(f"    {fn:<12} {va[fn]:>10.4f} {rank_v[fn]+1:>9d} "
                  f"{sa[fn]:>10.4f} {rank_s[fn]+1:>9d} {d_auc:>+8.4f} "
                  f"{d_rank:>+6d}")
        # qualitative summary
        order_v = [fn for fn, _ in sorted(va.items(), key=lambda kv: -kv[1])]
        order_s = [fn for fn, _ in sorted(sa.items(), key=lambda kv: -kv[1])]
        same = order_v == order_s
        print(f"    ranking unchanged? {same}")
        print(f"    vanilla order:  {order_v}")
        print(f"    smoothing order:{order_s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True,
                                      transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True,
                                     transform=tf)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print(">>> Training vanilla victim (label_smoothing=0.0)")
    victim_v = train_victim(seed=0, train_set=train_set, label_smoothing=0.0)
    print("\n>>> Training label-smoothing victim (label_smoothing=0.1)")
    victim_s = train_victim(seed=0, train_set=train_set, label_smoothing=0.1)

    res_v = evaluate_victim("vanilla", victim_v, test_x, test_y)
    res_s = evaluate_victim("label_smoothing_0.1", victim_s, test_x, test_y)
    compare_rankings(res_v, res_s)


if __name__ == "__main__":
    main()
