"""
H99: BIM with CW loss (margin-based) flips different samples than BIM with
cross-entropy. Per-sample disagreement reveals which samples are vulnerable to
weaker vs stronger losses.

Pipeline:
  1. Train small CNN (matching diagnostic_test.py) on Fashion-MNIST for 10 epochs.
  2. Run two attacks (10 steps, eps=15/255, alpha=eps/4):
        - BIM-CE: BIM with cross-entropy loss
        - BIM-CW: BIM with the Carlini-Wagner margin loss
            f(x) = max(Z(x)_y - max_{i!=y} Z(x)_i, -kappa)
  3. Features: margin, mean_pix, std_pix, sobel_mean.
  4. Targets: flipped_BIM_CE, flipped_BIM_CW.
  5. Cross-attack disagreement: samples flipped ONLY by CW (and only by CE).
  6. Univariate AUROC per feature per attack.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 15.0 / 255.0
STEPS = 10
ALPHA = EPS / 4.0
EPOCHS = 10
BATCH = 128
KAPPA = 0.0


class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2); x = self.do1(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.do2(x); return self.fc2(x)


def train_model(train_set, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done")
    model.eval()
    return model


def cw_margin_loss(logits, y, kappa=KAPPA):
    """Carlini-Wagner f(x) for untargeted attack.
    We want to MAXIMISE (max_{i!=y} Z_i - Z_y), i.e. minimise (Z_y - max_other).
    Returns a per-sample scalar to be SUMMED then we ascend its gradient
    relative to (Z_y - max_other). We return mean of (Z_y - max_other) clipped
    by -kappa; backward through this and add sign(grad) to x.
    """
    n_classes = logits.size(1)
    one_hot = F.one_hot(y, n_classes).bool()
    true_logit = logits[one_hot]
    other_max = logits.masked_fill(one_hot, -1e9).max(1).values
    # we want to descend (true_logit - other_max), so loss = (true_logit - other_max)
    f = torch.clamp(true_logit - other_max, min=-kappa)
    return f.mean()


def bim_attack(model, x, y, loss_kind, eps=EPS, alpha=ALPHA, steps=STEPS):
    """BIM / I-FGSM with either CE or CW loss. Untargeted, L_inf."""
    x0 = x.clone().detach()
    adv = x.clone().detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        logits = model(adv)
        if loss_kind == "ce":
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, adv)[0]
            # ascend CE -> add sign(grad)
            adv = adv.detach() + alpha * grad.sign()
        elif loss_kind == "cw":
            loss = cw_margin_loss(logits, y)
            grad = torch.autograd.grad(loss, adv)[0]
            # we want to DECREASE (true - max_other), so descend -> subtract sign
            adv = adv.detach() - alpha * grad.sign()
        else:
            raise ValueError(loss_kind)
        # project to L_inf ball and valid pixel range
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps)
        adv = adv.clamp(0.0, 1.0).detach()
    return adv


def batched_attack(model, x, y, loss_kind, batch=256):
    out = []
    for i in range(0, x.size(0), batch):
        out.append(bim_attack(model, x[i:i+batch], y[i:i+batch], loss_kind))
    return torch.cat(out, 0)


def batched_predict(model, x, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            out.append(model(x[i:i+batch]).argmax(1))
    return torch.cat(out, 0)


def batched_logits(model, x, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            out.append(model(x[i:i+batch]))
    return torch.cat(out, 0)


def sobel_mean(x):
    """Mean absolute Sobel response per image. x: (N,1,H,W) in [0,1]."""
    kx = torch.tensor([[[[-1., 0., 1.],
                          [-2., 0., 2.],
                          [-1., 0., 1.]]]], device=x.device)
    ky = torch.tensor([[[[-1., -2., -1.],
                          [ 0.,  0.,  0.],
                          [ 1.,  2.,  1.]]]], device=x.device)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return g.mean(dim=(1, 2, 3))


def compute_features(model, x, y):
    logits = batched_logits(model, x)
    n_classes = logits.size(1)
    one_hot = F.one_hot(y, n_classes).bool()
    true_logit = logits[one_hot]
    other_max = logits.masked_fill(one_hot, -1e9).max(1).values
    margin = (true_logit - other_max).detach()
    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))
    sob = sobel_mean(x)
    feats = torch.stack([margin, mean_pix, std_pix, sob], dim=1)
    return feats


def univariate_auroc(feats_np, y, names):
    print(f"  positive rate = {y.mean():.4f}  (n_pos={int(y.sum())}, n={len(y)})")
    if y.std() == 0:
        print("   target degenerate; skipping AUROC")
        return
    for i, n in enumerate(names):
        x_i = feats_np[:, i]
        a = roc_auc_score(y, x_i)
        a = max(a, 1 - a)
        print(f"   univariate AUROC  {n:<12} {a:.4f}")


def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("Training CNN on Fashion-MNIST (10 epochs)...")
    t0 = time.time()
    model = train_model(train_set, seed=0)
    print(f"  trained in {time.time() - t0:.1f}s")

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # restrict to correctly-classified samples
    pred0 = batched_predict(model, test_x)
    correct = pred0 == test_y
    x_c, y_c = test_x[correct], test_y[correct]
    print(f"correctly classified clean: {int(correct.sum())} / {len(test_y)}")

    print("Running BIM-CE attack...")
    t0 = time.time()
    adv_ce = batched_attack(model, x_c, y_c, "ce")
    pred_ce = batched_predict(model, adv_ce)
    flipped_ce = (pred_ce != y_c).cpu().numpy().astype(int)
    print(f"  BIM-CE flip rate: {flipped_ce.mean():.4f}  ({time.time() - t0:.1f}s)")

    print("Running BIM-CW attack...")
    t0 = time.time()
    adv_cw = batched_attack(model, x_c, y_c, "cw")
    pred_cw = batched_predict(model, adv_cw)
    flipped_cw = (pred_cw != y_c).cpu().numpy().astype(int)
    print(f"  BIM-CW flip rate: {flipped_cw.mean():.4f}  ({time.time() - t0:.1f}s)")

    # Cross-attack disagreement
    only_cw = ((flipped_cw == 1) & (flipped_ce == 0)).astype(int)
    only_ce = ((flipped_ce == 1) & (flipped_cw == 0)).astype(int)
    both = ((flipped_ce == 1) & (flipped_cw == 1)).astype(int)
    neither = ((flipped_ce == 0) & (flipped_cw == 0)).astype(int)
    n = len(flipped_ce)
    print("\n--- cross-attack disagreement ---")
    print(f"  both flipped       : {both.sum():>5d} ({both.mean():.4f})")
    print(f"  only BIM-CE flips  : {only_ce.sum():>5d} ({only_ce.mean():.4f})")
    print(f"  only BIM-CW flips  : {only_cw.sum():>5d} ({only_cw.mean():.4f})")
    print(f"  neither            : {neither.sum():>5d} ({neither.mean():.4f})")
    if (flipped_ce.sum() + flipped_cw.sum()) > 0:
        union = ((flipped_ce | flipped_cw) == 1).sum()
        inter = both.sum()
        jac = inter / max(union, 1)
        print(f"  Jaccard(CE, CW)    : {jac:.4f}")
        # McNemar-ish: disagreement count
        disagree = only_ce.sum() + only_cw.sum()
        print(f"  total disagreement : {disagree} / {n} ({disagree / n:.4f})")

    # Features
    feats = compute_features(model, x_c, y_c).cpu().numpy()
    names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    print("\n--- univariate AUROC: target = flipped_BIM_CE ---")
    univariate_auroc(feats, flipped_ce, names)

    print("\n--- univariate AUROC: target = flipped_BIM_CW ---")
    univariate_auroc(feats, flipped_cw, names)

    print("\n--- univariate AUROC: target = only_CW (flipped by CW but not CE) ---")
    univariate_auroc(feats, only_cw, names)

    print("\n--- univariate AUROC: target = only_CE (flipped by CE but not CW) ---")
    univariate_auroc(feats, only_ce, names)

    print("\nDone.")


if __name__ == "__main__":
    main()
