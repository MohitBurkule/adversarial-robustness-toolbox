"""
H175 - Margin AUROC at MATCHED attack-success rate (saturation-artefact control).

Motivation (advisor critique, Paper 1):
  Margin AUROC ~ 0.97 on Fashion-MNIST at eps=15/255 partly reflects that ~96% of samples are
  successfully attacked, leaving a tiny, near-degenerate ROC problem against an almost-constant
  label. The benchmark needs AUROC reported at MATCHED attack-success rates across datasets, not
  at a constant eps. This script makes the control explicit:

  For a trained vanilla CNN, it binary-searches the L-inf budget eps so that the PGD-10 ASR on the
  evaluation set hits a target (30%, 50%, 70%), then recomputes the margin AUROC at each matched
  difficulty. It contrasts these with the AUROC at the original saturated eps=15/255. If margin
  dominance is real (not a saturation artefact), the AUROC should remain high at ASR~50% where the
  label is balanced and the ROC problem is hardest.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
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
EPOCHS = 10
BATCH = 128
EPS_SAT = 15.0 / 255.0
N_CLASSES = 10
EVAL_N = 1000
SEED = 0
TARGET_ASRS = [0.30, 0.50, 0.70]


class CNN(nn.Module):
    def __init__(self, n=10):
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


def train(train_set, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return model


def pgd_flip(model, x, y, eps, steps=10):
    """PGD-10 at given eps; alpha scales with eps. Returns boolean flip tensor."""
    model.eval()
    alpha = max(eps / 4.0, 1e-4)
    adv = (x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        with torch.no_grad():
            adv = (adv + alpha * adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def asr_at_eps(model, x, y, eps):
    flips = []
    for i in range(0, x.size(0), 256):
        flips.append(pgd_flip(model, x[i:i+256], y[i:i+256], eps).cpu())
    f = torch.cat(flips).numpy().astype(int)
    return f.mean(), f


def find_eps_for_asr(model, x, y, target, lo=0.0, hi=0.5, iters=12):
    for _ in range(iters):
        mid = (lo + hi) / 2
        asr, _ = asr_at_eps(model, x, y, mid)
        if asr < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def safe_auroc(label, score):
    label = np.asarray(label).astype(int)
    if label.std() == 0:
        return float("nan")
    a = roc_auc_score(label, np.asarray(score))
    return max(a, 1 - a)


def main():
    print("=" * 74)
    print("H175 - Margin AUROC at matched attack-success rate")
    print("=" * 74)
    print(f"Device={DEVICE}  saturated eps={EPS_SAT:.4f}  EVAL_N={EVAL_N}")

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n--- Training vanilla CNN ---"); t0 = time.time()
    model = train(train_set); print(f"  {time.time()-t0:.1f}s")
    model.eval()
    with torch.no_grad():
        correct = (model(test_x).argmax(1) == test_y)
    idx = correct.nonzero(as_tuple=True)[0][:EVAL_N]
    x, y = test_x[idx], test_y[idx]
    with torch.no_grad():
        s, _ = model(x).sort(1, descending=True)
        margin = (s[:, 0] - s[:, 1]).cpu().numpy()

    print(f"\n  {'condition':<22} {'eps':>8} {'ASR':>7} {'margin AUROC':>14}")
    print("  " + "-" * 54)
    # saturated reference
    asr_sat, f_sat = asr_at_eps(model, x, y, EPS_SAT)
    print(f"  {'saturated (15/255)':<22} {EPS_SAT:>8.4f} {asr_sat:>7.3f} {safe_auroc(f_sat, -margin):>14.4f}")
    # matched-ASR conditions
    for tgt in TARGET_ASRS:
        eps = find_eps_for_asr(model, x, y, tgt)
        asr, f = asr_at_eps(model, x, y, eps)
        print(f"  {('matched ASR='+str(int(tgt*100))+'%'):<22} {eps:>8.4f} {asr:>7.3f} {safe_auroc(f, -margin):>14.4f}")

    print("\n" + "=" * 74)
    print("If margin AUROC stays high at ASR~50% (balanced label, hardest ROC), margin dominance")
    print("is genuine. If it falls sharply from the saturated number, the high AUROC was inflated")
    print("by label saturation -- the artefact the advisor flagged.")
    print("=" * 74)


if __name__ == "__main__":
    main()
