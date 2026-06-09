"""
Hypothesis H56: Per-sample 1D loss landscape along the FGSM-gradient direction
has shape statistics (peak position, slope, curvature) that predict vulnerability
better than the gradient magnitude alone.

Procedure:
  - Train small CNN victim on Fashion-MNIST (10 epochs).
  - For each correctly-classified test sample, take s = sign(grad_x L(x,y)).
  - Walk x + eps * s for eps in {0.01, 0.02, ..., 0.30} (30 points), record loss.
  - Extract shape statistics from the per-sample loss curve:
        loss_at_eps15, max_loss, eps_at_max, slope0 (1st FD), curv0 (2nd FD),
        integral (trapezoidal).
  - Baselines: victim_margin, input_grad_L2_norm.
  - Targets: flipped_FGSM (eps=15/255), flipped_PGD (10-step, eps=15/255),
             FGSM_min_eps (continuous; binary "easy to flip" target derived).
  - Univariate AUROC; multivariate logistic regression to test if shape stats
    add over (margin + grad_norm).
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
EPS_GRID = torch.linspace(0.01, 0.30, 30)  # 30 points
PGD_ITERS = 10
PGD_ALPHA = (15.0 / 255.0) / 4.0


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


def train_victim(seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)

    model = CNN(10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done")
    model.eval()

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    return model, test_x, test_y


def compute_grad_and_sign(model, x, y):
    """Returns (sign(grad), grad_L2_norm per sample)."""
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    # per-sample loss
    loss = F.cross_entropy(logits, y, reduction="sum")
    loss.backward()
    g = x.grad.detach()
    sign = g.sign()
    g_norm = g.flatten(1).norm(dim=1)
    return sign, g_norm


def per_sample_losses(model, x_adv, y):
    """Return per-sample cross-entropy loss (no reduction)."""
    with torch.no_grad():
        logits = model(x_adv)
        return F.cross_entropy(logits, y, reduction="none")


def victim_margin(model, x, y):
    with torch.no_grad():
        logits = model(x)
        true_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        other = logits.masked_fill(F.one_hot(y, logits.size(1)).bool(), -1e9).max(1).values
        return true_logit - other


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign, _ = compute_grad_and_sign(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, iters=PGD_ITERS):
    x_orig = x.clone().detach()
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta = (x_orig + delta).clamp(0, 1) - x_orig
    for _ in range(iters):
        delta = delta.detach().requires_grad_(True)
        logits = model(x_orig + delta)
        loss = F.cross_entropy(logits, y, reduction="sum")
        g = torch.autograd.grad(loss, delta)[0]
        delta = delta.detach() + alpha * g.sign()
        delta = delta.clamp(-eps, eps)
        delta = ((x_orig + delta).clamp(0, 1) - x_orig).detach()
    with torch.no_grad():
        return (model(x_orig + delta).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    sign, _ = compute_grad_and_sign(model, x, y)
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def loss_landscape(model, x, y, eps_grid):
    """For each sample, compute loss along x + eps*sign for each eps in eps_grid.
    Returns tensor [N, len(eps_grid)] and also the sign and L2-norm of the input grad.
    """
    sign, g_norm = compute_grad_and_sign(model, x, y)
    N = x.size(0); K = eps_grid.numel()
    curve = torch.zeros(N, K, device=DEVICE)
    for k, eps in enumerate(eps_grid.tolist()):
        adv = (x + eps * sign).clamp(0, 1)
        curve[:, k] = per_sample_losses(model, adv, y)
    return curve, sign, g_norm


def shape_features(curve, eps_grid, l0_per_sample):
    """Extract shape statistics from per-sample loss curve.

    curve: [N, K]   loss at each eps along sign direction (eps > 0)
    eps_grid: [K]   the eps values (must be uniform spacing assumed for FD)
    l0_per_sample:  [N]  loss at eps=0 (clean input)

    Returns dict of feature_name -> [N] tensor.
    """
    eps = eps_grid.to(curve.device)
    K = eps.numel()
    deps = float(eps[1] - eps[0])  # uniform 0.01

    # loss_at_eps=15/255 ~ eps idx where eps closest to 15/255
    idx15 = int(torch.argmin(torch.abs(eps - EPS_TEST)).item())
    loss_at_eps15 = curve[:, idx15]

    max_loss, argmax_idx = curve.max(dim=1)
    eps_at_max = eps[argmax_idx]

    # slope at eps=0: forward finite difference using L0 and curve[:,0]
    slope0 = (curve[:, 0] - l0_per_sample) / deps

    # second derivative at eps=0: central-ish 2nd FD using L0, curve[0], curve[1]
    # f''(0) ~ (f(2h) - 2 f(h) + f(0)) / h^2
    curv0 = (curve[:, 1] - 2.0 * curve[:, 0] + l0_per_sample) / (deps ** 2)

    # integral: trapezoidal across [0, eps_max] including L0 at eps=0
    # build full curve including L0
    full = torch.cat([l0_per_sample.unsqueeze(1), curve], dim=1)  # [N, K+1]
    full_eps = torch.cat([torch.zeros(1, device=eps.device), eps])
    integral = torch.trapz(full, full_eps, dim=1)

    return {
        "loss_at_eps15": loss_at_eps15,
        "max_loss": max_loss,
        "eps_at_max": eps_at_max,
        "slope0": slope0,
        "curv0": curv0,
        "integral": integral,
    }


def auroc(scores, labels):
    s = scores.detach().cpu().numpy() if torch.is_tensor(scores) else scores
    y = labels.detach().cpu().numpy().astype(int) if torch.is_tensor(labels) else labels.astype(int)
    if y.std() == 0:
        return float("nan")
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def multivariate_auc(feat_mat, y, names_used):
    if y.std() == 0:
        return float("nan")
    Xs = StandardScaler().fit_transform(feat_mat)
    lr = LogisticRegression(max_iter=2000).fit(Xs, y)
    p = lr.predict_proba(Xs)[:, 1]
    return roc_auc_score(y, p)


def main():
    print(f"device: {DEVICE}")
    print("training victim CNN on Fashion-MNIST ...")
    t0 = time.time()
    model, test_x, test_y = train_victim(seed=0)
    print(f"  done ({time.time()-t0:.1f}s)")

    # restrict to correctly classified samples
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct = preds == test_y
    x_c = test_x[correct]; y_c = test_y[correct]
    print(f"using {x_c.size(0)} correctly classified samples")

    # baseline: margin
    print("computing victim margin ...")
    margins = []
    for i in range(0, x_c.size(0), 512):
        margins.append(victim_margin(model, x_c[i:i+512], y_c[i:i+512]))
    margin = torch.cat(margins)

    # clean per-sample loss (eps=0)
    l0_chunks = []
    for i in range(0, x_c.size(0), 512):
        l0_chunks.append(per_sample_losses(model, x_c[i:i+512], y_c[i:i+512]))
    l0 = torch.cat(l0_chunks)

    # loss landscape along sign direction
    print(f"sweeping loss landscape over {EPS_GRID.numel()} eps values ...")
    t0 = time.time()
    curve_chunks = []; gnorm_chunks = []
    BATCH_LS = 256
    for i in range(0, x_c.size(0), BATCH_LS):
        c, _, gn = loss_landscape(model, x_c[i:i+BATCH_LS], y_c[i:i+BATCH_LS], EPS_GRID)
        curve_chunks.append(c); gnorm_chunks.append(gn)
    curve = torch.cat(curve_chunks, 0)
    g_norm = torch.cat(gnorm_chunks, 0)
    print(f"  done ({time.time()-t0:.1f}s)  curve shape={tuple(curve.shape)}")

    feats = shape_features(curve, EPS_GRID, l0)
    feats["victim_margin"] = margin
    feats["input_grad_L2_norm"] = g_norm

    # ----- TARGETS -----
    print("computing FGSM (eps=15/255) flip target ...")
    fgsm_chunks = []
    for i in range(0, x_c.size(0), 512):
        fgsm_chunks.append(fgsm_flip(model, x_c[i:i+512], y_c[i:i+512]))
    t_fgsm = torch.cat(fgsm_chunks)

    print(f"computing PGD-{PGD_ITERS} (eps=15/255) flip target ...")
    pgd_chunks = []
    for i in range(0, x_c.size(0), 512):
        pgd_chunks.append(pgd_flip(model, x_c[i:i+512], y_c[i:i+512]))
    t_pgd = torch.cat(pgd_chunks)

    print("computing FGSM min-eps ...")
    me_chunks = []
    for i in range(0, x_c.size(0), 512):
        me_chunks.append(min_eps_to_flip(model, x_c[i:i+512], y_c[i:i+512]))
    min_eps = torch.cat(me_chunks)
    # binary "easy to flip": min_eps below median => more vulnerable
    median_eps = float(min_eps.median().item())
    t_min_eps_bin = (min_eps < median_eps)

    targets = {
        "flipped_FGSM_eps15": t_fgsm,
        "flipped_PGD_eps15": t_pgd,
        "FGSM_min_eps_below_median": t_min_eps_bin,
    }
    print(f"  positive rates: "
          f"FGSM={t_fgsm.float().mean().item():.3f}, "
          f"PGD={t_pgd.float().mean().item():.3f}, "
          f"min_eps_bin={t_min_eps_bin.float().mean().item():.3f}, "
          f"median_min_eps={median_eps:.4f}")

    # ----- UNIVARIATE AUROC -----
    feat_order = ["loss_at_eps15", "max_loss", "eps_at_max", "slope0", "curv0",
                  "integral", "victim_margin", "input_grad_L2_norm"]
    print("\n========== univariate AUROC ==========")
    print(f"{'feature':<26} " + " ".join([f"{t:>22}" for t in targets]))
    for fn in feat_order:
        row = [f"{fn:<26}"]
        for t_name, t in targets.items():
            a = auroc(feats[fn], t)
            row.append(f"{a:>22.4f}")
        print(" ".join(row))

    # ----- MULTIVARIATE: does shape add over (margin + grad_norm)? -----
    print("\n========== multivariate AUROC ==========")
    shape_names = ["loss_at_eps15", "max_loss", "eps_at_max", "slope0", "curv0", "integral"]
    base_names = ["victim_margin", "input_grad_L2_norm"]

    def stack(names):
        return torch.stack([feats[n] for n in names], dim=1).detach().cpu().numpy()

    M_base = stack(base_names)
    M_shape = stack(shape_names)
    M_full = stack(base_names + shape_names)

    print(f"{'target':<32} {'base':>8} {'shape':>8} {'full':>8} {'delta(full-base)':>18}")
    for t_name, t in targets.items():
        y = t.detach().cpu().numpy().astype(int)
        if y.std() == 0:
            print(f"  {t_name}: degenerate target")
            continue
        a_base = multivariate_auc(M_base, y, base_names)
        a_shape = multivariate_auc(M_shape, y, shape_names)
        a_full = multivariate_auc(M_full, y, base_names + shape_names)
        print(f"{t_name:<32} {a_base:>8.4f} {a_shape:>8.4f} {a_full:>8.4f} {a_full-a_base:>+18.4f}")

    # standardised coefficients for full model on FGSM target (most relevant)
    print("\n=== standardised coefficients (target=flipped_FGSM_eps15, full model) ===")
    y = targets["flipped_FGSM_eps15"].detach().cpu().numpy().astype(int)
    Xs = StandardScaler().fit_transform(M_full)
    lr = LogisticRegression(max_iter=2000).fit(Xs, y)
    for n, c in zip(base_names + shape_names, lr.coef_.flatten()):
        print(f"  {n:<26} {c:+.4f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
