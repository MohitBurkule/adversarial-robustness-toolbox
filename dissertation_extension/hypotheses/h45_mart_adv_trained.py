"""
H45: MART (Wang et al. ICLR 2020) adversarial training reweights misclassified
examples more heavily via a (1 - p_y)-weighted KL term. This is a different
inductive bias than TRADES. We test whether margin and image stats still
dominate as vulnerability predictors after MART.

MART loss (per-sample):
    L = BCE_misclass(f(adv), y) + lambda * (1 - p_y_clean) * KL(softmax(f(adv)) || softmax(f(clean)))
where BCE_misclass uses a "boosted CE" that explicitly upweights misclassified
samples: -log(p_y^adv) - log(1 - max_{k != y} p_k^adv).

Pipeline:
  1. Train two CNN victims on Fashion-MNIST (10 epochs):
       - vanilla
       - MART adversarially trained (PGD-AT inner loop)
  2. For each victim, on the test set, compute:
       - per-sample features: margin, mean_pix, std_pix, sobel_mean
       - per-sample targets:  flipped_FGSM, flipped_PGD, FGSM_min_eps
  3. Per-victim univariate AUROC of each feature against each binary target,
     plus Spearman corr against the continuous FGSM_min_eps target.
  4. Print AUROC ranking of features for each victim, so we can see whether
     MART changes the ordering compared to vanilla.
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
EPOCHS = 10
BATCH = 128
EPS_TRAIN = 0.1
ALPHA_TRAIN = 0.01
PGD_STEPS_TRAIN = 7
EPS_TEST = 15.0 / 255.0
PGD_STEPS_TEST = 20
ALPHA_TEST = EPS_TEST / 4.0
MART_LAMBDA = 5.0
DATA_DIR = "/tmp/data"


# ---------------------------------------------------------------- CNN
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


# ---------------------------------------------------------------- Attacks
def pgd_attack(model, x, y, eps, alpha, steps, random_start=True):
    model_was_training = model.training
    model.eval()
    x_adv = x.clone().detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    if model_was_training:
        model.train()
    return x_adv.detach()


def fgsm_attack(model, x, y, eps):
    model.eval()
    x_a = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_a), y)
    grad = torch.autograd.grad(loss, x_a)[0]
    return (x_a + eps * grad.sign()).clamp(0, 1).detach()


def fgsm_sign(model, x, y):
    model.eval()
    x_a = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_a), y)
    grad = torch.autograd.grad(loss, x_a)[0]
    return grad.sign().detach()


# ---------------------------------------------------------------- Losses
def mart_loss(model, x, y, eps, alpha, steps, lam=MART_LAMBDA):
    """MART loss (Wang et al. ICLR 2020).

      L = BCE_boost(f(x_adv), y) + lam * (1 - p_y_clean) * KL(softmax(f(x_adv)) || softmax(f(x)))

    where BCE_boost = -log(softmax(adv)_y) - log(1 - max_{k!=y} softmax(adv)_k)
    """
    x_adv = pgd_attack(model, x, y, eps, alpha, steps, random_start=True)
    model.train()
    logits_adv = model(x_adv)
    logits_clean = model(x)
    p_adv = F.softmax(logits_adv, dim=1)
    p_clean = F.softmax(logits_clean, dim=1)
    log_p_adv = F.log_softmax(logits_adv, dim=1)

    n = x.size(0)
    idx = torch.arange(n, device=x.device)
    # boosted CE: standard CE + log(1 - top non-true prob)
    ce = F.cross_entropy(logits_adv, y, reduction="none")
    # mask true class to find max over k != y
    mask = F.one_hot(y, num_classes=logits_adv.size(1)).bool()
    p_adv_notrue = p_adv.masked_fill(mask, 0.0)
    top_other = p_adv_notrue.max(dim=1).values
    boosted = ce - torch.log(1.0 - top_other + 1e-12)

    # (1 - p_y_clean) reweighted KL
    p_y_clean = p_clean[idx, y].detach()
    # KL(softmax(adv) || softmax(clean)) using log_softmax + softmax via F.kl_div
    log_p_clean = F.log_softmax(logits_clean, dim=1)
    # per-sample KL(adv || clean) = sum_k p_adv * (log p_adv - log p_clean)
    kl = (p_adv * (log_p_adv - log_p_clean)).sum(dim=1)
    weight = (1.0 - p_y_clean)

    loss = boosted.mean() + lam * (weight * kl).mean()
    return loss


# ---------------------------------------------------------------- Training
def train_vanilla(seed, train_loader):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        print(f"  [vanilla] epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    return model


def train_mart(seed, train_loader):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = mart_loss(model, x, y, EPS_TRAIN, ALPHA_TRAIN, PGD_STEPS_TRAIN)
            loss.backward(); opt.step()
        print(f"  [MART]    epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    return model


# ---------------------------------------------------------------- Features
def sobel_mean(x):
    """x: (N,1,H,W) in [0,1]. Returns per-image mean |sobel| magnitude."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = (gx ** 2 + gy ** 2).sqrt()
    return mag.mean(dim=(1, 2, 3))


def compute_margin(model, x, y, batch=512):
    model.eval()
    out_margin = []
    out_pred = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            logits = model(x[i:i+batch])
            yb = y[i:i+batch]
            true_logit = logits[torch.arange(logits.size(0)), yb]
            logits_no_true = logits.masked_fill(
                F.one_hot(yb, logits.size(1)).bool(), -1e9)
            second = logits_no_true.max(dim=1).values
            out_margin.append(true_logit - second)
            out_pred.append(logits.argmax(1))
    return torch.cat(out_margin), torch.cat(out_pred)


# ---------------------------------------------------------------- Targets
def fgsm_flip(model, x, y, eps=EPS_TEST, batch=512):
    flips = []
    for i in range(0, x.size(0), batch):
        adv = fgsm_attack(model, x[i:i+batch], y[i:i+batch], eps)
        with torch.no_grad():
            flips.append(model(adv).argmax(1) != y[i:i+batch])
    return torch.cat(flips)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=ALPHA_TEST,
             steps=PGD_STEPS_TEST, batch=256):
    flips = []
    for i in range(0, x.size(0), batch):
        adv = pgd_attack(model, x[i:i+batch], y[i:i+batch],
                         eps, alpha, steps, random_start=True)
        with torch.no_grad():
            flips.append(model(adv).argmax(1) != y[i:i+batch])
    return torch.cat(flips)


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=15, batch=512):
    """Binary search smallest eps for FGSM to flip. Direction = sign(grad) at x."""
    results = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        sign = fgsm_sign(model, xb, yb)
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xb + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
            with torch.no_grad():
                flipped = model(adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        results.append(hi)
    return torch.cat(results)


# ---------------------------------------------------------------- Eval
def eval_features_targets(name, feats_np, feat_names, bin_targets,
                          bin_target_names, cont_target, cont_target_name):
    print(f"\n========== victim: {name} ==========")
    print(f"  test-set clean acc-mask size: {feats_np.shape[0]}")

    rankings = {}
    for tn, y_bin in zip(bin_target_names, bin_targets):
        print(f"\n  -- binary target: {tn}  pos rate={y_bin.mean():.3f}")
        if y_bin.std() == 0:
            print("     (degenerate target, skipping)")
            continue
        aucs = []
        for i, fn in enumerate(feat_names):
            x_i = feats_np[:, i]
            try:
                a = roc_auc_score(y_bin, x_i)
                a = max(a, 1 - a)
            except Exception:
                a = float("nan")
            aucs.append((fn, a))
        aucs.sort(key=lambda t: -t[1])
        for fn, a in aucs:
            print(f"     {fn:<14}  AUROC = {a:.4f}")
        rankings[tn] = [fn for fn, _ in aucs]

    # continuous
    print(f"\n  -- continuous target: {cont_target_name} (Spearman) --")
    cors = []
    for i, fn in enumerate(feat_names):
        try:
            rho, _ = spearmanr(feats_np[:, i], cont_target)
        except Exception:
            rho = float("nan")
        cors.append((fn, rho))
    cors.sort(key=lambda t: -abs(t[1]) if not np.isnan(t[1]) else 0)
    for fn, rho in cors:
        print(f"     {fn:<14}  Spearman = {rho:+.4f}")
    rankings[cont_target_name] = [fn for fn, _ in cors]
    return rankings


# ---------------------------------------------------------------- Main
def run_victim(name, model, test_x, test_y, feat_names):
    print(f"\n>>>>> evaluating victim: {name}")
    margin, pred = compute_margin(model, test_x, test_y)
    correct = pred == test_y
    print(f"  clean test acc: {correct.float().mean().item():.4f}")
    # restrict to correctly classified
    idx = torch.nonzero(correct, as_tuple=True)[0]
    x_c = test_x[idx]; y_c = test_y[idx]; margin_c = margin[idx]

    mean_pix = x_c.mean(dim=(1, 2, 3))
    std_pix = x_c.std(dim=(1, 2, 3))
    sob = sobel_mean(x_c)
    feats = torch.stack([margin_c, mean_pix, std_pix, sob], dim=1).cpu().numpy()

    print("  computing FGSM flips ...")
    f_fgsm = fgsm_flip(model, x_c, y_c).cpu().numpy().astype(int)
    print("  computing PGD flips ...")
    f_pgd = pgd_flip(model, x_c, y_c).cpu().numpy().astype(int)
    print("  computing FGSM min_eps ...")
    me = fgsm_min_eps(model, x_c, y_c).cpu().numpy()

    return eval_features_targets(
        name, feats, feat_names,
        bin_targets=[f_fgsm, f_pgd],
        bin_target_names=["flipped_FGSM", "flipped_PGD"],
        cont_target=me,
        cont_target_name="FGSM_min_eps",
    )


def main():
    print(f"device: {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n--- training vanilla victim ---")
    m_van = train_vanilla(seed=0, train_loader=train_loader)
    print("\n--- training MART victim ---")
    m_mart = train_mart(seed=0, train_loader=train_loader)

    feat_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    r_van = run_victim("vanilla", m_van, test_x, test_y, feat_names)
    r_mart = run_victim("MART", m_mart, test_x, test_y, feat_names)

    print("\n========== RANKING COMPARISON ==========")
    for tn in r_van:
        print(f"\n target: {tn}")
        print(f"   vanilla ranking: {r_van[tn]}")
        print(f"   MART    ranking: {r_mart.get(tn, [])}")
        same = r_van[tn] == r_mart.get(tn, [])
        print(f"   identical? {same}")


if __name__ == "__main__":
    main()
