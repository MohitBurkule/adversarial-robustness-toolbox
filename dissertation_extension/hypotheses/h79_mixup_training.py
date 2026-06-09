"""
H79: Mixup training (Zhang et al. 2018) changes which features predict adversarial
vulnerability.

Setup:
  - Train two victims on Fashion-MNIST, 10 epochs each, small CNN matching
    diagnostic_test.py:
        (a) vanilla cross-entropy training
        (b) mixup training with alpha=1.0
  - Per victim, compute four per-sample image / model features:
        margin     : final-model logit margin (top1 - top2)
        mean_pix   : per-image mean pixel intensity
        std_pix    : per-image pixel std
        sobel_mean : mean Sobel gradient magnitude (edge energy)
  - Per victim, compute three vulnerability targets on the test set:
        FGSM     : flipped by FGSM at eps=15/255
        PGD      : flipped by PGD-20 at eps=15/255
        min_eps  : binary-searched smallest L_inf FGSM eps that flips
  - Per-feature AUROC (binary targets) and |Spearman| for min_eps.
  - Report whether mixup changes the ranking of features. Expectation:
    mixup smooths the margin so its dominance over raw image stats may shrink,
    promoting mean_pix/std_pix/sobel_mean in the ranking.

This file is self-contained: it trains both victims, attacks them, computes
features, evaluates, and prints a side-by-side ranking comparison.
DO NOT RUN automatically — kept here for the dissertation pipeline.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0
MIXUP_ALPHA = 1.0


# ---------------------------------------------------------------------------
# Model: identical small CNN from diagnostic_test.py
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_vanilla(seed, train_set, n_classes=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    model.eval()
    return model


def train_mixup(seed, train_set, alpha=MIXUP_ALPHA, n_classes=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            lam = float(rng.beta(alpha, alpha))
            perm = torch.randperm(x.size(0), device=DEVICE)
            x_mix = lam * x + (1 - lam) * x[perm]
            y_a, y_b = y, y[perm]
            opt.zero_grad()
            logits = model(x_mix)
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            loss.backward()
            opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def margin_feature(model, x, batch=512):
    margins = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            logits = model(x[i:i + batch])
            srt, _ = logits.sort(1, descending=True)
            margins.append((srt[:, 0] - srt[:, 1]).cpu())
    return torch.cat(margins).numpy()


def mean_pix_feature(x):
    return x.view(x.size(0), -1).mean(1).cpu().numpy()


def std_pix_feature(x):
    return x.view(x.size(0), -1).std(1).cpu().numpy()


def sobel_mean_feature(x, batch=1024):
    """Mean Sobel gradient magnitude per image. x is (N,1,H,W) in [0,1]."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device)
    ky = kx.t()
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            xb = x[i:i + batch]
            gx = F.conv2d(xb, kx, padding=1)
            gy = F.conv2d(xb, ky, padding=1)
            mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
            out.append(mag.view(mag.size(0), -1).mean(1).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    adv = (x0 + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=x.device)
    hi = torch.full((x.size(0),), eps_max, device=x.device)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def batched_attack(fn, model, x, y, batch=512, **kw):
    out = []
    for i in range(0, x.size(0), batch):
        out.append(fn(model, x[i:i + batch], y[i:i + batch], **kw))
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["margin", "mean_pix", "std_pix", "sobel_mean"]


def compute_all_features(model, x):
    return {
        "margin":     margin_feature(model, x),
        "mean_pix":   mean_pix_feature(x),
        "std_pix":    std_pix_feature(x),
        "sobel_mean": sobel_mean_feature(x),
    }


def directional_auroc(y_bin, x_score):
    a = roc_auc_score(y_bin, x_score)
    return max(a, 1 - a)


def evaluate_victim(tag, model, x, y):
    print(f"\n===== victim: {tag} =====")
    feats = compute_all_features(model, x)

    print("  computing FGSM attack ...")
    fgsm_succ = batched_attack(fgsm_flip, model, x, y).cpu().numpy().astype(int)
    print(f"   FGSM success rate = {fgsm_succ.mean():.3f}")

    print("  computing PGD attack ...")
    pgd_succ = batched_attack(pgd_flip, model, x, y).cpu().numpy().astype(int)
    print(f"   PGD  success rate = {pgd_succ.mean():.3f}")

    print("  computing min_eps_to_flip ...")
    me = []
    for i in range(0, x.size(0), 512):
        me.append(min_eps_to_flip(model, x[i:i + 512], y[i:i + 512]))
    min_eps = torch.cat(me).cpu().numpy()
    print(f"   mean min_eps      = {min_eps.mean():.4f}")

    binary_targets = {"FGSM": fgsm_succ, "PGD": pgd_succ}
    auroc_table = {}
    for tname, ybin in binary_targets.items():
        if ybin.std() == 0:
            print(f"   target {tname} degenerate; skipping AUROC")
            continue
        row = {}
        for fname in FEATURE_NAMES:
            row[fname] = directional_auroc(ybin, feats[fname])
        auroc_table[tname] = row

    # min_eps: continuous target, use |Spearman| as the ranking statistic
    spear_row = {}
    for fname in FEATURE_NAMES:
        rho, _ = spearmanr(feats[fname], min_eps)
        spear_row[fname] = abs(rho) if not np.isnan(rho) else 0.0
    auroc_table["min_eps_|spearman|"] = spear_row

    # print
    for tname, row in auroc_table.items():
        ranked = sorted(row.items(), key=lambda kv: -kv[1])
        print(f"\n  target {tname}:")
        for r, (fname, v) in enumerate(ranked, 1):
            print(f"    rank {r}: {fname:<11} {v:.4f}")
    return auroc_table


def compare_rankings(vanilla_tab, mixup_tab):
    print("\n========== ranking comparison: vanilla vs mixup ==========")
    print(f"{'target':<22} {'feature':<12} {'vanilla':>9} {'mixup':>9} {'delta':>9}")
    for tname in vanilla_tab:
        if tname not in mixup_tab:
            continue
        v_row = vanilla_tab[tname]
        m_row = mixup_tab[tname]
        for fname in FEATURE_NAMES:
            v = v_row[fname]
            m = m_row[fname]
            print(f"{tname:<22} {fname:<12} {v:>9.4f} {m:>9.4f} {m - v:>+9.4f}")

        v_rank = [f for f, _ in sorted(v_row.items(), key=lambda kv: -kv[1])]
        m_rank = [f for f, _ in sorted(m_row.items(), key=lambda kv: -kv[1])]
        changed = v_rank != m_rank
        print(f"  ranking [{tname}]")
        print(f"    vanilla order : {v_rank}")
        print(f"    mixup   order : {m_rank}")
        print(f"    ranking changed: {changed}")
        # Spearman rank correlation between the AUROC vectors
        v_vec = np.array([v_row[f] for f in FEATURE_NAMES])
        m_vec = np.array([m_row[f] for f in FEATURE_NAMES])
        rho, _ = spearmanr(v_vec, m_vec)
        print(f"    Spearman(rank_vanilla, rank_mixup) over features: {rho:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True,  download=True, transform=tf)
    test_set  = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("Training vanilla victim ...")
    t0 = time.time()
    model_vanilla = train_vanilla(seed=0, train_set=train_set)
    print(f"  done ({time.time() - t0:.1f}s)")

    print(f"Training mixup victim (alpha={MIXUP_ALPHA}) ...")
    t0 = time.time()
    model_mixup = train_mixup(seed=0, train_set=train_set, alpha=MIXUP_ALPHA)
    print(f"  done ({time.time() - t0:.1f}s)")

    # Restrict to samples each victim classifies correctly (matches diagnostic_test.py
    # convention: vulnerability is only meaningful for currently-correct samples).
    with torch.no_grad():
        v_pred = []
        m_pred = []
        for i in range(0, test_x.size(0), 512):
            v_pred.append(model_vanilla(test_x[i:i + 512]).argmax(1))
            m_pred.append(model_mixup(test_x[i:i + 512]).argmax(1))
        v_pred = torch.cat(v_pred)
        m_pred = torch.cat(m_pred)

    v_mask = (v_pred == test_y)
    m_mask = (m_pred == test_y)
    print(f"\nvanilla clean acc: {v_mask.float().mean().item():.4f}")
    print(f"mixup   clean acc: {m_mask.float().mean().item():.4f}")

    vanilla_tab = evaluate_victim("vanilla", model_vanilla,
                                  test_x[v_mask], test_y[v_mask])
    mixup_tab   = evaluate_victim(f"mixup(alpha={MIXUP_ALPHA})", model_mixup,
                                  test_x[m_mask], test_y[m_mask])

    compare_rankings(vanilla_tab, mixup_tab)

    print("\nInterpretation hint:")
    print("  Expected: mixup smooths the margin, so its dominance shrinks while")
    print("  raw image stats (mean_pix / std_pix / sobel_mean) may rise in rank.")
    print("  A changed ranking or strong drop in margin-AUROC supports H79.")


if __name__ == "__main__":
    main()
