"""
Phase A: re-run learning-epoch / 2nd-best / first-wrong analysis with LONGER training (20 epochs).
         Also bucket by FINAL LOGIT MARGIN (runtime-usable proxy for difficulty).
Phase B: curriculum adversarial training.
         Train 3 models with same total adv-budget:
            (1) vanilla       (no adv samples)
            (2) FGSM-all      (50% adv ratio, standard)
            (3) FGSM-hardonly (only inject adv samples for low-margin samples - same total count)
         Evaluate clean, FGSM, PGD, BIM accuracy.
Phase C: new analyses
   - per-sample minimum eps to flip with FGSM (binary search) vs learning_epoch / margin
   - is logit margin a usable proxy for learning_epoch?
"""
import time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda")
EPS = 15.0 / 255.0
BATCH = 128
SEED = 0
N_FIXED = 2000
torch.manual_seed(SEED); np.random.seed(SEED)


class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*12*12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2); x = self.do1(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.do2(x); return self.fc2(x)


def fgsm(model, x, y, eps=EPS):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps=EPS, steps=10):
    alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        F.cross_entropy(model(x_adv), y).backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)
    return x_adv.detach()


def bim(model, x, y, eps=EPS, steps=10):
    alpha = eps / steps
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        F.cross_entropy(model(x_adv), y).backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)
    return x_adv.detach()


def evaluate(model, loader, attack=None):
    model.eval(); c = t = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if attack is not None:
            x = attack(model, x, y)
        with torch.no_grad():
            c += (model(x).argmax(1) == y).sum().item(); t += y.size(0)
    return c / t


def get_data():
    tf = transforms.ToTensor()
    train = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    return train, test


# =============== Phase A ===============
def phase_A(epochs=20):
    print(f"\n=== Phase A: {epochs} epochs, tracking learning trajectory ===")
    train, test = get_data()
    train_loader = DataLoader(train, BATCH, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, 256, shuffle=False, num_workers=2)

    idx = torch.randperm(len(test))[:N_FIXED]
    fx = torch.stack([test[i][0] for i in idx]).to(DEVICE)
    fy = torch.tensor([test[i][1] for i in idx]).to(DEVICE)
    n_cls = 10

    model = CNN().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    per_epoch_pred = []
    confusion = torch.zeros(N_FIXED, n_cls, device=DEVICE)
    learning_ep = torch.full((N_FIXED,), -1, device=DEVICE, dtype=torch.long)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(fx).argmax(1)
        per_epoch_pred.append(p.clone())
        wrong = p != fy
        if wrong.any():
            confusion[torch.arange(N_FIXED, device=DEVICE)[wrong], p[wrong]] += 1
        right = (p == fy) & (learning_ep == -1)
        learning_ep[right] = ep
        if (ep+1) % 5 == 0:
            print(f"  ep {ep+1}/{epochs}  ({time.time()-t0:.1f}s)")
    per_epoch_pred = torch.stack(per_epoch_pred)

    # final stats
    model.eval()
    with torch.no_grad():
        logits = model(fx)
        final_pred = logits.argmax(1)
        sorted_logits, _ = logits.sort(1, descending=True)
        margin = (sorted_logits[:, 0] - sorted_logits[:, 1])
        final_2nd = logits.masked_fill(F.one_hot(fy, n_cls).bool(), -1e9).argmax(1)

    x_adv = fgsm(model, fx, fy)
    with torch.no_grad():
        fgsm_pred = model(x_adv).argmax(1)
    fooled = (fgsm_pred != fy) & (final_pred == fy)

    first_wrong_cls = torch.full((N_FIXED,), -1, device=DEVICE, dtype=torch.long)
    for e in range(epochs):
        miss = (per_epoch_pred[e] != fy) & (first_wrong_cls == -1)
        first_wrong_cls[miss] = per_epoch_pred[e][miss]
    top_conf = confusion.argmax(1); top_conf[confusion.sum(1) == 0] = -1

    def frac(m):
        return m.float().mean().item() if m.numel() else float("nan")

    print(f"\n  clean acc on fixed: {(final_pred==fy).float().mean():.4f}, fooled: {fooled.sum().item()}/{N_FIXED}")
    print("\n  --- bucket by learning_epoch ---")
    print("   le  | n    | FGSM->2nd | FGSM->firstwrong | FGSM->topconf | margin")
    for le in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, -1]:
        m = (learning_ep == le)
        if m.sum() < 5: continue
        mf = m & fooled
        if mf.sum() == 0: continue
        f2nd = frac(fgsm_pred[mf] == final_2nd[mf])
        ffw = frac((first_wrong_cls[mf] >= 0) & (fgsm_pred[mf] == first_wrong_cls[mf]))
        ftc = frac((top_conf[mf] >= 0) & (fgsm_pred[mf] == top_conf[mf]))
        mm = margin[m].mean().item()
        label = "never" if le == -1 else f"ep{le}"
        print(f"   {label:>5} | {int(m.sum()):4d} | {f2nd:.2%}   | {ffw:.2%}        | {ftc:.2%}    | {mm:.2f}")

    print("\n  --- bucket by final logit MARGIN (runtime proxy) ---")
    qs = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=DEVICE)
    boundaries = torch.quantile(margin, qs)
    print("   margin bin     | n    | FGSM->2nd | FGSM->firstwrong | mean_le")
    for i in range(len(qs)-1):
        lo, hi = boundaries[i], boundaries[i+1]
        m = (margin >= lo) & (margin <= hi if i == len(qs)-2 else margin < hi)
        if m.sum() == 0: continue
        mf = m & fooled
        mean_le = learning_ep[m].float().mean().item()
        if mf.sum() == 0:
            print(f"   [{lo:5.2f},{hi:5.2f}] | {int(m.sum()):4d} | n/a       | n/a              | {mean_le:.2f}")
            continue
        f2nd = frac(fgsm_pred[mf] == final_2nd[mf])
        ffw = frac((first_wrong_cls[mf] >= 0) & (fgsm_pred[mf] == first_wrong_cls[mf]))
        print(f"   [{lo:5.2f},{hi:5.2f}] | {int(m.sum()):4d} | {f2nd:.2%}   | {ffw:.2%}        | {mean_le:.2f}")

    # Phase C: minimum eps to flip via binary search (per-sample) on a 256-sample subset
    print("\n  --- min-eps-to-flip vs learning_epoch (256 samples) ---")
    sub = torch.arange(N_FIXED, device=DEVICE)[final_pred == fy][:256]
    sub_x, sub_y = fx[sub], fy[sub]
    sub_le = learning_ep[sub]
    sub_margin = margin[sub]
    # binary search epsilon for each sample
    lo = torch.zeros(sub.size(0), device=DEVICE)
    hi = torch.full((sub.size(0),), 0.3, device=DEVICE)
    for _ in range(15):
        mid = (lo + hi) / 2
        # need batched FGSM with per-sample eps
        xa = sub_x.clone().detach().requires_grad_(True)
        F.cross_entropy(model(xa), sub_y).sum().backward()
        adv = (sub_x + mid.view(-1, 1, 1, 1) * xa.grad.sign()).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != sub_y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    min_eps = hi
    # bucket by learning_epoch
    print("   le   | n   | mean_min_eps  | mean_margin")
    for le in range(0, epochs):
        m = (sub_le == le)
        if m.sum() < 3: continue
        print(f"   ep{le:<3}| {int(m.sum()):3d} | {min_eps[m].mean():.4f}      | {sub_margin[m].mean():.2f}")
    # correlation
    import math
    cor_le = float(torch.corrcoef(torch.stack([min_eps, sub_le.float()]))[0, 1])
    cor_mg = float(torch.corrcoef(torch.stack([min_eps, sub_margin]))[0, 1])
    print(f"   corr(min_eps, learning_epoch) = {cor_le:.3f}")
    print(f"   corr(min_eps, final_margin)   = {cor_mg:.3f}")

    return model, margin.detach().cpu(), learning_ep.detach().cpu()


# =============== Phase B: curriculum adversarial training ===============
def difficulty_scores(model, train_loader_eval):
    """Return per-sample final-logit margin (lower = harder) for the training set."""
    model.eval()
    margins = []
    with torch.no_grad():
        for x, y in train_loader_eval:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            top2 = logits.topk(2, dim=1).values
            margins.append((top2[:, 0] - top2[:, 1]).cpu())
    return torch.cat(margins)


def train_with_adv(model, train_loader, epochs, adv_mask=None, adv_ratio=1.0):
    """If adv_mask is None: standard training. If provided (bool[N_train]), only inject adv
       on samples in mask; adv_ratio = fraction of in-mask samples that get adv augmentation."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # we need a way to know dataset indices in the batch. train_loader must yield (idx, x, y) or
    # we use a wrapped Dataset.
    for ep in range(epochs):
        model.train()
        for idx, x, y in train_loader:
            idx = idx.to(DEVICE); x = x.to(DEVICE); y = y.to(DEVICE)
            if adv_mask is not None:
                in_mask = adv_mask.to(DEVICE)[idx]
                if in_mask.any():
                    model.eval()
                    x_adv = fgsm(model, x[in_mask], y[in_mask])
                    model.train()
                    x = torch.cat([x, x_adv]); y = torch.cat([y, y[in_mask]])
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
    model.eval()


class IndexedTensorDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y = self.ds[i]; return i, x, y


def phase_B(adv_epochs=8):
    print(f"\n=== Phase B: curriculum adv training ({adv_epochs} epochs each) ===")
    train, test = get_data()
    idx_train = IndexedTensorDataset(train)
    train_loader = DataLoader(idx_train, BATCH, shuffle=True, num_workers=2)
    # plain loader for difficulty scoring
    eval_train_loader = DataLoader(train, 512, shuffle=False, num_workers=2)
    test_loader = DataLoader(test, 256, shuffle=False, num_workers=2)

    # (1) Vanilla model
    print("  [1/3] vanilla...")
    m_van = CNN().to(DEVICE)
    train_with_adv(m_van, train_loader, adv_epochs)
    # difficulty scores from this vanilla model
    margins_train = difficulty_scores(m_van, eval_train_loader)
    hard_mask = margins_train < margins_train.median()   # bottom 50% margin = "hard"

    # (2) Standard FGSM adv on ALL samples
    print("  [2/3] adv_all (every sample gets FGSM augmentation)...")
    m_all = CNN().to(DEVICE)
    all_mask = torch.ones(len(train), dtype=torch.bool)
    train_with_adv(m_all, train_loader, adv_epochs, adv_mask=all_mask)

    # (3) FGSM adv on HARD samples only (= half the dataset)
    print("  [3/3] adv_hard (only bottom-50% margin samples)...")
    m_hard = CNN().to(DEVICE)
    train_with_adv(m_hard, train_loader, adv_epochs, adv_mask=hard_mask)

    print("\n  evaluating...")
    results = {}
    for name, m in [("vanilla", m_van), ("adv_all", m_all), ("adv_hard_only", m_hard)]:
        clean = evaluate(m, test_loader)
        a_fgsm = evaluate(m, test_loader, lambda mm, x, y: fgsm(mm, x, y))
        a_pgd = evaluate(m, test_loader, lambda mm, x, y: pgd(mm, x, y))
        a_bim = evaluate(m, test_loader, lambda mm, x, y: bim(mm, x, y))
        results[name] = dict(clean=clean, fgsm=a_fgsm, pgd=a_pgd, bim=a_bim,
                             adv_samples_per_epoch=int(
                                 (all_mask if name == "adv_all" else
                                  (hard_mask if name == "adv_hard_only" else
                                   torch.zeros_like(all_mask))).sum()))
        print(f"   {name:<14}: clean={clean:.4f} fgsm={a_fgsm:.4f} pgd={a_pgd:.4f} bim={a_bim:.4f}")
    return results


if __name__ == "__main__":
    t0 = time.time()
    phase_A(epochs=20)
    res = phase_B(adv_epochs=8)
    with open("long_curriculum_results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nTotal: {time.time()-t0:.1f}s")
