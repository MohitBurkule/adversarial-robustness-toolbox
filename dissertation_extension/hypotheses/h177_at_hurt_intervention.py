"""
H177 - Targeted intervention on the AT-hurt set.

Motivation (advisor critique, Paper 8):
  Paper 8 ("double jeopardy") shows the vanilla model's margin predicts, BEFORE any adversarial
  training, which samples PGD-AT will newly misclassify (the "AT-hurt set", AUROC ~0.935). The
  obvious missing experiment is the intervention: if low-margin samples are the ones AT breaks,
  does treating them differently during AT shrink the AT-hurt set and recover clean accuracy?

This script:
  1. Trains a vanilla CNN; records per-TRAIN-sample clean logit margin (top1 - top2).
  2. Flags the bottom-decile-margin training samples as "fragile".
  3. Trains four PGD-AT models that differ only in how fragile samples are handled:
       - AT-uniform   : standard PGD-AT on all samples (the baseline).
       - AT-downweight : fragile samples' loss x0.25 (de-emphasise hard samples).
       - AT-upweight   : fragile samples' loss x4.0  (advisor also asks this direction).
       - AT-exclude    : fragile samples are trained on CLEAN images (no adversarial augmentation),
                         all others get standard PGD adversarial examples.
  4. Evaluates every model on a fixed held-out test set:
       - clean accuracy
       - PGD-10 attack success rate (robustness)
       - mean min-eps
       - |AT-hurt set| = #(vanilla-correct AND this-model-wrong)
       - |newly-right set| = #(vanilla-wrong AND this-model-right)
  5. Reports, per intervention, the change in clean accuracy, robustness, and AT-hurt size
     relative to AT-uniform. Also profiles the vanilla margin of the newly-right samples (the
     symmetric question Paper 8 never analysed: what predicts a GAIN from AT?).

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10
N_TEST = 2000
SEED = 0
FRAGILE_FRAC = 0.10
DOWN_W = 0.25
UP_W = 4.0


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


class IndexedDataset(Dataset):
    """Wrap a dataset so __getitem__ also returns the sample index."""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def pgd_attack(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=7):
    model.eval()
    adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    adv = adv.clamp(0, 1)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        with torch.no_grad():
            adv = (adv + alpha * adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return adv.detach()


def train_vanilla(loader):
    set_seed(SEED)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        model.train()
        for x, y, _ in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return model


@torch.no_grad()
def train_margins(model, loader, n):
    """Per-sample clean logit margin over the whole training set."""
    model.eval()
    margins = torch.zeros(n)
    for x, y, idx in loader:
        x = x.to(DEVICE)
        logits = model(x)
        s, _ = logits.sort(1, descending=True)
        m = (s[:, 0] - s[:, 1]).cpu()
        margins[idx] = m
    return margins


def train_at_intervention(loader, mode, fragile_mask):
    """mode in {uniform, downweight, upweight, exclude}. fragile_mask: bool tensor over train idx."""
    set_seed(SEED)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    fragile_mask = fragile_mask.to(DEVICE)
    for _ in range(EPOCHS):
        model.train()
        for x, y, idx in loader:
            x, y, idx = x.to(DEVICE), y.to(DEVICE), idx.to(DEVICE)
            frag = fragile_mask[idx]

            if mode == "exclude":
                # fragile samples trained clean; others adversarial
                x_adv_all = pgd_attack(model, x, y)
                x_in = torch.where(frag.view(-1, 1, 1, 1), x, x_adv_all)
                model.train()
                opt.zero_grad()
                F.cross_entropy(model(x_in), y).backward()
                opt.step()
                continue

            x_adv = pgd_attack(model, x, y)
            model.train()
            opt.zero_grad()
            loss_vec = F.cross_entropy(model(x_adv), y, reduction="none")
            if mode == "uniform":
                w = torch.ones_like(loss_vec)
            elif mode == "downweight":
                w = torch.where(frag, torch.full_like(loss_vec, DOWN_W), torch.ones_like(loss_vec))
            elif mode == "upweight":
                w = torch.where(frag, torch.full_like(loss_vec, UP_W), torch.ones_like(loss_vec))
            else:
                raise ValueError(mode)
            (loss_vec * w).mean().backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, x):
    model.eval()
    return model(x).argmax(1)


def pgd_asr(model, x, y, steps=10):
    flips = []
    for i in range(0, x.size(0), 256):
        xa = pgd_attack(model, x[i:i+256], y[i:i+256], steps=steps)
        with torch.no_grad():
            flips.append((model(xa).argmax(1) != y[i:i+256]).cpu())
    return torch.cat(flips).float().mean().item()


def mean_min_eps(model, x, y, iters=12, eps_max=0.3):
    vals = []
    for i in range(0, x.size(0), 256):
        bx, by = x[i:i+256], y[i:i+256]
        xr = bx.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(xr), by)
        g, = torch.autograd.grad(loss, xr)
        sign = g.sign()
        lo = torch.zeros(bx.size(0), device=DEVICE)
        hi = torch.full((bx.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (bx + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
            with torch.no_grad():
                fl = (model(adv).argmax(1) != by)
            hi = torch.where(fl, mid, hi)
            lo = torch.where(fl, lo, mid)
        vals.append(hi.cpu())
    return torch.cat(vals).mean().item()


def main():
    print("=" * 74)
    print("H177 - Targeted intervention on the AT-hurt set")
    print("=" * 74)
    print(f"Device={DEVICE}  EPOCHS={EPOCHS}  EPS={EPS:.4f}  fragile_frac={FRAGILE_FRAC}")

    tf = transforms.ToTensor()
    train_base = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_base = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    train_set = IndexedDataset(train_base)
    n_train = len(train_set)

    rng = np.random.RandomState(SEED)
    test_idx = rng.choice(len(test_base), N_TEST, replace=False)
    test_x = torch.stack([test_base[i][0] for i in test_idx]).to(DEVICE)
    test_y = torch.tensor([test_base[i][1] for i in test_idx]).to(DEVICE)

    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    eval_loader = DataLoader(train_set, 512, shuffle=False, num_workers=2)

    print("\n--- Train vanilla + compute train margins ---")
    t0 = time.time()
    vanilla = train_vanilla(loader)
    margins = train_margins(vanilla, eval_loader, n_train)
    k = int(FRAGILE_FRAC * n_train)
    thresh = margins.kthvalue(k).values.item()
    fragile_mask = margins <= thresh
    print(f"  vanilla trained in {time.time()-t0:.1f}s; "
          f"fragile (bottom {FRAGILE_FRAC:.0%}) margin <= {thresh:.4f}, n={int(fragile_mask.sum())}")

    vanilla_test_pred = predict(vanilla, test_x)
    vanilla_correct = (vanilla_test_pred == test_y)
    print(f"  vanilla test acc = {vanilla_correct.float().mean().item():.4f}")

    modes = ["uniform", "downweight", "upweight", "exclude"]
    results = {}
    for mode in modes:
        print(f"\n--- Train AT-{mode} ---")
        t0 = time.time()
        model = train_at_intervention(loader, mode, fragile_mask)
        pred = predict(model, test_x)
        correct = (pred == test_y)
        hurt = (vanilla_correct & ~correct)
        gained = (~vanilla_correct & correct)
        results[mode] = {
            "clean_acc": correct.float().mean().item(),
            "pgd_asr": pgd_asr(model, test_x, test_y),
            "min_eps": mean_min_eps(model, test_x, test_y),
            "n_hurt": int(hurt.sum()),
            "n_gained": int(gained.sum()),
            "hurt_mask": hurt.cpu().numpy(),
            "gained_mask": gained.cpu().numpy(),
        }
        print(f"  {time.time()-t0:.1f}s  cleanAcc={results[mode]['clean_acc']:.4f}  "
              f"PGD-ASR={results[mode]['pgd_asr']:.4f}  minEps={results[mode]['min_eps']:.4f}  "
              f"hurt={results[mode]['n_hurt']}  gained={results[mode]['n_gained']}")

    # vanilla test margins (for profiling hurt/gained groups)
    with torch.no_grad():
        vlog = vanilla(test_x)
        vs, _ = vlog.sort(1, descending=True)
        vtest_margin = (vs[:, 0] - vs[:, 1]).cpu().numpy()

    print("\n" + "=" * 74)
    print("Intervention comparison (delta vs AT-uniform)")
    print("=" * 74)
    base = results["uniform"]
    hdr = f"{'mode':<12} {'cleanAcc':>9} {'dClean':>8} {'PGD-ASR':>8} {'minEps':>8} {'hurt':>6} {'dHurt':>7} {'gained':>7}"
    print(hdr); print("-" * len(hdr))
    for mode in modes:
        r = results[mode]
        print(f"{mode:<12} {r['clean_acc']:>9.4f} {r['clean_acc']-base['clean_acc']:>+8.4f} "
              f"{r['pgd_asr']:>8.4f} {r['min_eps']:>8.4f} {r['n_hurt']:>6d} "
              f"{r['n_hurt']-base['n_hurt']:>+7d} {r['n_gained']:>7d}")

    print("\n--- Profile of hurt vs gained samples (AT-uniform), by vanilla test margin ---")
    hm = base["hurt_mask"]; gm = base["gained_mask"]
    pres = vanilla_correct.cpu().numpy() & ~hm
    if hm.sum() > 0:
        print(f"  hurt    (vanilla-correct, AT-wrong):  n={hm.sum():>4d}  "
              f"vanilla margin mean={vtest_margin[hm].mean():.4f}")
    if pres.sum() > 0:
        print(f"  preserved (vanilla-correct, AT-right): n={pres.sum():>4d}  "
              f"vanilla margin mean={vtest_margin[pres].mean():.4f}")
    if gm.sum() > 0:
        print(f"  gained  (vanilla-wrong, AT-right):    n={gm.sum():>4d}  "
              f"vanilla margin mean={vtest_margin[gm].mean():.4f}  "
              f"(symmetry probe: do gains have distinctive pre-AT margin?)")
    print("=" * 74)


if __name__ == "__main__":
    main()
