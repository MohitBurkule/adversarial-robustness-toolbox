"""
H81: Gradient regularization training (Ross & Doshi-Velez 2018) changes which
features predict adversarial vulnerability.

Train two victims on Fashion-MNIST, 10 epochs each, using a small CNN matching
diagnostic_test.py:
    - vanilla: standard cross-entropy
    - grad-reg: cross-entropy + lambda * ||grad_x L||^2  with lambda=0.01

For each victim, compute per-sample features:
    margin, mean_pix, std_pix, sobel_mean, input_grad_norm

Targets:
    FGSM_flip (eps=15/255), PGD_flip (eps=15/255, 10 steps), min_eps (binary
    search for smallest L_inf eps that flips FGSM).

Compare per-feature AUROC across the two victims.
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
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
LAMBDA = 0.01
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0


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


def train_model(seed, train_set, grad_reg=False, lam=LAMBDA):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if grad_reg:
                x.requires_grad_(True)
                logits = model(x)
                ce = F.cross_entropy(logits, y)
                # ||grad_x L||^2 penalty; create_graph for double-backprop
                gx = torch.autograd.grad(ce, x, create_graph=True)[0]
                penalty = gx.pow(2).flatten(1).sum(1).mean()
                loss = ce + lam * penalty
            else:
                loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done")
    return model


def stack_test(test_set):
    x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    y = torch.tensor([test_set[i][1] for i in range(len(test_set))])
    return x.to(DEVICE), y.to(DEVICE)


def model_logits(model, x, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i+bs]))
    return torch.cat(out, 0)


def input_grad(model, x, y, bs=256):
    """Return per-sample input grad of CE loss (not in eval mode w.r.t. dropout,
    but we switch to eval to remove stochasticity)."""
    model.eval()
    grads = []
    for i in range(0, x.size(0), bs):
        xb = x[i:i+bs].clone().detach().requires_grad_(True)
        yb = y[i:i+bs]
        loss = F.cross_entropy(model(xb), yb, reduction="sum")
        g = torch.autograd.grad(loss, xb)[0]
        grads.append(g.detach())
    return torch.cat(grads, 0)


def sobel_mean(x):
    """Mean absolute Sobel gradient magnitude per sample. x in [B,1,H,W]."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag.flatten(1).mean(1)


def compute_features(model, x, y):
    logits = model_logits(model, x)
    sorted_l, _ = logits.sort(1, descending=True)
    margin = (sorted_l[:, 0] - sorted_l[:, 1]).detach()
    pred = logits.argmax(1)

    mean_pix = x.flatten(1).mean(1)
    std_pix = x.flatten(1).std(1)
    sob = sobel_mean(x)

    g = input_grad(model, x, y)
    ign = g.flatten(1).norm(dim=1)

    feats = torch.stack([margin, mean_pix, std_pix, sob, ign], 1)
    return feats, pred


def fgsm_attack(model, x, y, eps=EPS_TEST):
    model.eval()
    xa = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xa), y)
    g = torch.autograd.grad(loss, xa)[0]
    adv = (xa + eps * g.sign()).clamp(0, 1).detach()
    return adv


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    model.eval()
    adv = x.clone().detach()
    # random start
    adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
    adv = adv.clamp(0, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        g = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * g.sign()
        adv = torch.max(torch.min(adv, x + eps), x - eps).clamp(0, 1)
    return adv.detach()


def batched_attack_flip(model, x, y, attack_fn, bs=256, **kw):
    flips = []
    for i in range(0, x.size(0), bs):
        adv = attack_fn(model, x[i:i+bs], y[i:i+bs], **kw)
        with torch.no_grad():
            p = model(adv).argmax(1)
        flips.append(p != y[i:i+bs])
    return torch.cat(flips, 0)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15, bs=256):
    """Per-sample binary search on FGSM eps using fixed sign direction."""
    model.eval()
    out = []
    for i in range(0, x.size(0), bs):
        xb = x[i:i+bs]; yb = y[i:i+bs]
        xa = xb.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(xa), yb)
        sign = torch.autograd.grad(loss, xa)[0].sign().detach()
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xb + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
            with torch.no_grad():
                flipped = model(adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        out.append(hi)
    return torch.cat(out, 0)


def auroc(y, score):
    y = np.asarray(y).astype(int)
    if y.std() == 0:
        return float("nan")
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def evaluate_victim(tag, model, x, y, feat_names):
    print(f"\n========== victim: {tag} ==========")
    feats, pred = compute_features(model, x, y)
    correct = pred == y
    print(f"  clean accuracy: {correct.float().mean().item():.4f}")
    x_c = x[correct]; y_c = y[correct]; feats_c = feats[correct]

    print("  computing FGSM ...")
    fgsm_flip = batched_attack_flip(model, x_c, y_c, fgsm_attack)
    print(f"    FGSM flip rate: {fgsm_flip.float().mean().item():.4f}")

    print("  computing PGD ...")
    pgd_flip = batched_attack_flip(model, x_c, y_c, pgd_attack)
    print(f"    PGD flip rate:  {pgd_flip.float().mean().item():.4f}")

    print("  computing min_eps ...")
    t0 = time.time()
    min_eps = min_eps_to_flip(model, x_c, y_c)
    print(f"    done ({time.time()-t0:.1f}s)  mean min_eps={min_eps.mean().item():.4f}")

    feats_np = feats_c.detach().cpu().numpy()
    fgsm_np = fgsm_flip.cpu().numpy().astype(int)
    pgd_np = pgd_flip.cpu().numpy().astype(int)
    meps_np = min_eps.detach().cpu().numpy()

    # binarize min_eps at median for AUROC vs a binary "easy to flip" target
    meps_bin = (meps_np <= np.median(meps_np)).astype(int)

    targets = {"FGSM_flip": fgsm_np, "PGD_flip": pgd_np, "min_eps<=median": meps_bin}
    results = {}
    for t_name, t_arr in targets.items():
        print(f"\n  target: {t_name}  (pos rate = {t_arr.mean():.3f})")
        results[t_name] = {}
        for i, fn in enumerate(feat_names):
            a = auroc(t_arr, feats_np[:, i])
            results[t_name][fn] = a
            print(f"    AUROC  {fn:<18} {a:.4f}")
        # also raw correlation w/ min_eps (continuous)
    print(f"\n  Continuous Pearson corr with min_eps:")
    for i, fn in enumerate(feat_names):
        c = np.corrcoef(feats_np[:, i], meps_np)[0, 1]
        print(f"    corr({fn:<18}, min_eps) = {c:+.4f}")
    return results


def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    x, y = stack_test(test_set)

    feat_names = ["margin", "mean_pix", "std_pix", "sobel_mean", "input_grad_norm"]

    print("##### training vanilla victim #####")
    t0 = time.time()
    model_vanilla = train_model(seed=0, train_set=train_set, grad_reg=False)
    print(f"  done ({time.time()-t0:.1f}s)")

    print(f"\n##### training grad-reg victim (lambda={LAMBDA}) #####")
    t0 = time.time()
    model_gr = train_model(seed=0, train_set=train_set, grad_reg=True, lam=LAMBDA)
    print(f"  done ({time.time()-t0:.1f}s)")

    res_v = evaluate_victim("vanilla", model_vanilla, x, y, feat_names)
    res_g = evaluate_victim(f"grad_reg(lambda={LAMBDA})", model_gr, x, y, feat_names)

    print("\n===== SIDE-BY-SIDE per-feature AUROC =====")
    targets = list(res_v.keys())
    for t in targets:
        print(f"\n  target: {t}")
        print(f"    {'feature':<18} {'vanilla':>9} {'grad_reg':>9} {'delta':>9}")
        for fn in feat_names:
            av = res_v[t][fn]; ag = res_g[t][fn]
            print(f"    {fn:<18} {av:>9.4f} {ag:>9.4f} {ag-av:>+9.4f}")


if __name__ == "__main__":
    main()
