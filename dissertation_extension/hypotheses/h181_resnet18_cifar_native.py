"""
H181 - ResNet-18 on NATIVE 32x32 RGB CIFAR-10 (external-validity check).

Motivation (advisor meta-review, the second most important missing experiment):
  Every result in the dissertation extension bottoms out on the same 5-layer CNN at 28x28
  grayscale -- even CIFAR-10 and Imagenette are downsampled and greyscaled to fit it. The single
  greatest threat to external validity is that none of the headline claims has been reproduced on
  a modern architecture at native resolution. This script trains a CIFAR-style ResNet-18 on
  native 32x32 RGB CIFAR-10 to a realistic accuracy, then re-tests the three load-bearing claims:

    (A) Margin dominance: is the logit margin still the dominant per-sample vulnerability
        predictor (vs input-gradient-norm and softmax confidence)?
    (B) Structural / universal vulnerability: what fraction of samples are vulnerable to ALL of
        {FGSM, PGD, BIM, MIM}, and what is the chance baseline at matched ASR?
    (C) Attack-invariance: Jaccard overlap of the vulnerable sets across the four attacks.

This script is self-contained and does NOT use the FashionMNIST monkey-patch harness (it needs
native RGB). It downloads CIFAR-10 to ./data. Set TORCH_HOME to a large disk if needed.
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
EPOCHS = 40
BATCH = 128
EPS = 8.0 / 255.0          # standard CIFAR-10 L-inf budget
ALPHA = 2.0 / 255.0
N_CLASSES = 10
EVAL_N = 1000
SEED = 0
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------------------
# CIFAR ResNet-18 (3x3 stem, no maxpool) -- standard for 32x32
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 2, 1)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)
        self.linear = nn.Linear(512, n)

    def _make(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


# ---------------------------------------------------------------------------
# Normalisation wrapper so attacks operate in [0,1] pixel space
# ---------------------------------------------------------------------------

class Normalized(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


# ---------------------------------------------------------------------------
# Attacks (operate in [0,1] space; return boolean flip tensors)
# ---------------------------------------------------------------------------

def fgsm_flip(model, x, y, eps=EPS):
    xr = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xr), y)
    g, = torch.autograd.grad(loss, xr)
    adv = (x + eps * g.sign()).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def iterative_flip(model, x, y, eps=EPS, alpha=ALPHA, steps=10, momentum=0.0, random_start=True):
    """Generic L-inf iterative attack. momentum=0 -> BIM (no random start) / PGD (random start);
    momentum>0 -> MIM (Dong et al. 2018)."""
    model.eval()
    adv = x.clone().detach()
    if random_start:
        adv = (adv + torch.empty_like(adv).uniform_(-eps, eps)).clamp(0, 1)
    g_acc = torch.zeros_like(x)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        g, = torch.autograd.grad(loss, adv)
        with torch.no_grad():
            if momentum > 0:
                g = g / g.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
                g_acc = momentum * g_acc + g
                step_dir = g_acc.sign()
            else:
                step_dir = g.sign()
            adv = (adv + alpha * step_dir).clamp(x - eps, x + eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def input_grad_norm(model, x, y):
    xr = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xr), y)
    g, = torch.autograd.grad(loss, xr)
    return g.flatten(1).norm(dim=1)


def safe_auroc(label, score):
    label = np.asarray(label).astype(int)
    if label.std() == 0:
        return float("nan")
    a = roc_auc_score(label, np.asarray(score))
    return max(a, 1 - a)


def train(model, loader, adversarial=False):
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, epochs=EPOCHS,
                                                steps_per_epoch=len(loader))
    for ep in range(EPOCHS):
        model.train()
        tot = correct = n = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if adversarial:
                model.eval()
                adv = x.clone().detach()
                adv = (adv + torch.empty_like(adv).uniform_(-EPS, EPS)).clamp(0, 1)
                for _ in range(7):
                    adv = adv.detach().requires_grad_(True)
                    l = F.cross_entropy(model(adv), y)
                    g, = torch.autograd.grad(l, adv)
                    adv = (adv + ALPHA * g.sign()).clamp(x - EPS, x + EPS).clamp(0, 1)
                x = adv.detach()
                model.train()
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
            correct += (out.argmax(1) == y).sum().item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:>2}/{EPOCHS}  loss={tot/n:.4f}  train_acc={correct/n:.4f}")
    return model


def main():
    print("=" * 74)
    print("H181 - ResNet-18 on NATIVE 32x32 RGB CIFAR-10")
    print("=" * 74)
    print(f"Device={DEVICE}  EPOCHS={EPOCHS}  EPS={EPS:.4f} (8/255)  EVAL_N={EVAL_N}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()
    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=4, drop_last=True)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n--- Training ResNet-18 (standard) ---")
    t0 = time.time()
    model = Normalized(ResNet18(N_CLASSES), CIFAR_MEAN, CIFAR_STD).to(DEVICE)
    train(model, loader, adversarial=False)
    print(f"  trained in {time.time()-t0:.1f}s")

    model.eval()
    with torch.no_grad():
        clean_acc = (model(test_x).argmax(1) == test_y).float().mean().item()
    print(f"\n  Clean test accuracy: {clean_acc:.4f}")

    # eval subset: correctly-classified
    with torch.no_grad():
        correct = (model(test_x).argmax(1) == test_y)
    idx = correct.nonzero(as_tuple=True)[0][:EVAL_N]
    x, y = test_x[idx], test_y[idx]
    N = x.size(0)
    with torch.no_grad():
        logits = model(x)
        s, _ = logits.sort(1, descending=True)
        margin = (s[:, 0] - s[:, 1]).cpu().numpy()
        conf = F.softmax(logits, 1).max(1).values.cpu().numpy()

    # four attacks (eps-parameterised)
    def run_attacks(eps):
        attack_fns = {
            "FGSM": lambda bx, by: fgsm_flip(model, bx, by, eps=eps),
            "PGD":  lambda bx, by: iterative_flip(model, bx, by, eps=eps, alpha=max(eps / 4.0, 1e-4), steps=10, momentum=0.0, random_start=True),
            "BIM":  lambda bx, by: iterative_flip(model, bx, by, eps=eps, alpha=max(eps / 4.0, 1e-4), steps=10, momentum=0.0, random_start=False),
            "MIM":  lambda bx, by: iterative_flip(model, bx, by, eps=eps, alpha=max(eps / 4.0, 1e-4), steps=10, momentum=1.0, random_start=False),
        }
        out_flips = {}
        for name, fn in attack_fns.items():
            out = []
            for i in range(0, N, 250):
                out.append(fn(x[i:i+250], y[i:i+250]).cpu())
            out_flips[name] = torch.cat(out).numpy().astype(int)
        return out_flips

    gnorm = []
    for i in range(0, N, 250):
        gnorm.append(input_grad_norm(model, x[i:i+250], y[i:i+250]).detach().cpu())
    gnorm = torch.cat(gnorm).numpy()

    # --- saturated reference (8/255) for context ---
    sat_flips = run_attacks(EPS)
    print(f"\n  Per-attack ASR at saturated eps=8/255 (n={N}):")
    for name in sat_flips:
        print(f"    {name:<5} ASR = {sat_flips[name].mean():.4f}")

    # --- binary-search eps so PGD ASR ~ 0.5 (matched-ASR, balanced label, hardest ROC) ---
    def pgd_asr(eps):
        out = []
        for i in range(0, N, 250):
            out.append(iterative_flip(model, x[i:i+250], y[i:i+250], eps=eps,
                                      alpha=max(eps / 4.0, 1e-4), steps=10,
                                      momentum=0.0, random_start=True).cpu())
        return torch.cat(out).numpy().astype(int).mean()

    lo, hi = 0.0, EPS
    for _ in range(12):
        mid = (lo + hi) / 2
        if pgd_asr(mid) < 0.5:
            lo = mid
        else:
            hi = mid
    eps_m = (lo + hi) / 2
    print(f"\n  Matched-ASR eps (PGD~0.5): {eps_m:.5f}  ({eps_m*255:.2f}/255)")

    flips = run_attacks(eps_m)
    attacks = flips  # name iterable for downstream loops
    print(f"  Per-attack ASR at matched eps (n={N}):")
    for name in flips:
        print(f"    {name:<5} ASR = {flips[name].mean():.4f}")

    # (A) margin dominance (at matched-ASR eps, balanced label)
    print("\n(A) Per-sample vulnerability predictors (AUROC vs PGD flip @ matched eps):")
    pgd = flips["PGD"]
    print(f"    {'margin':<18} {safe_auroc(pgd, -margin):.4f}")
    print(f"    {'input_grad_norm':<18} {safe_auroc(pgd, gnorm):.4f}")
    print(f"    {'softmax_confidence':<18} {safe_auroc(pgd, -conf):.4f}")

    # (B) universal vulnerability + chance baseline
    stacked = np.stack([flips[a] for a in attacks], axis=1)  # (N,4)
    universal = (stacked.sum(1) == 4).mean()
    asrs = [flips[a].mean() for a in attacks]
    chance_universal = float(np.prod(asrs))  # independence null
    print("\n(B) Universal vulnerability (flipped by ALL 4 attacks):")
    print(f"    observed  = {universal:.4f}")
    print(f"    chance (independent, matched ASR) = {chance_universal:.4f}")
    print(f"    excess over chance = {universal - chance_universal:+.4f}")

    # (C) attack-invariance: pairwise Jaccard + chance Jaccard
    print("\n(C) Pairwise Jaccard overlap of vulnerable sets (observed / chance):")
    names = list(attacks.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = flips[names[i]], flips[names[j]]
            inter = ((a == 1) & (b == 1)).sum()
            union = ((a == 1) | (b == 1)).sum()
            jac = inter / union if union > 0 else float("nan")
            pa, pb = a.mean(), b.mean()
            # chance Jaccard under independence: P(A&B)/P(A|B) = pa*pb/(pa+pb-pa*pb)
            cj = (pa * pb) / (pa + pb - pa * pb) if (pa + pb - pa * pb) > 0 else float("nan")
            print(f"    {names[i]:<4}-{names[j]:<4}  J={jac:.4f}  chance={cj:.4f}")

    print("\n" + "=" * 74)
    print("External-validity verdict: do margin-dominance, universal vulnerability (vs chance),")
    print("and attack-invariance survive on a real ResNet-18 at native 32x32 RGB?")
    print("=" * 74)


if __name__ == "__main__":
    main()
