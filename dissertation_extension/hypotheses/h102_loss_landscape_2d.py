"""
H102: 2D loss landscape geometry predicts adversarial vulnerability.

Hypothesis:
  The local 2D loss landscape geometry around a test sample — sampled along the
  span of two random orthogonal unit-L2 directions in input space — provides a
  per-sample sharpness measure that correlates with adversarial vulnerability.

  For each test sample we:
    1. Draw two random directions in input space (shape [1,28,28]).
    2. Orthogonalise via Gram-Schmidt and normalise both to unit L2 norm.
    3. Evaluate the cross-entropy loss on the 5x5 grid
            x' = x + eps_a * d_a + eps_b * d_b
       with eps_a, eps_b in linspace(-0.1, 0.1, 5) (so the centre cell is the
       clean loss).
    4. Derive three sharpness features:
         - mean_loss_increase : mean over the 24 off-centre cells of
                                (loss(cell) - loss(centre))
         - max_loss_increase  : max  over the 24 off-centre cells of the same
         - asymmetry          : (max_off_centre_loss) / (min_off_centre_loss),
                                clamped to avoid division blow-ups

We additionally include classic baselines:
    - margin     : z_true - z_2nd
    - mean_pix   : mean pixel intensity
    - std_pix    : pixel std

Targets (per-sample, binary unless noted):
    - FGSM @ eps=15/255 self
    - PGD  @ eps=15/255 self (10-step, alpha=eps/4)
    - min_eps to flip with FGSM (continuous; binary-search); also binarised
      at its median for AUROC.

Model: small CNN matching diagnostic_test.py, trained 10 epochs on Fashion-MNIST.
Analysis: univariate AUROC of each feature against each binary target, and
Pearson correlation against min_eps.

This file is self-contained: it does not import from the rest of the project.
Run with:  python h102_loss_landscape_2d.py
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
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0
N_CLASSES = 10
SEED = 0

# loss-landscape grid configuration
GRID_N = 5
GRID_EPS_MIN = -0.1
GRID_EPS_MAX = 0.1
LANDSCAPE_BATCH = 64    # samples processed at once for landscape evaluation
DIR_SEED = 1234


# --------------------- model: same CNN as diagnostic_test.py ----------------
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


# --------------------------------- training ---------------------------------
def train(seed, train_set):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# --------------------------- 2D loss landscape ------------------------------
def _orthonormal_pair(shape, n, generator):
    """Return two tensors d_a, d_b each of shape [n, *shape], such that for
    every sample i: ||d_a[i]||_2 = ||d_b[i]||_2 = 1 and <d_a[i], d_b[i]> = 0.
    """
    flat_dim = int(np.prod(shape))
    a = torch.randn(n, flat_dim, generator=generator, device=DEVICE)
    b = torch.randn(n, flat_dim, generator=generator, device=DEVICE)
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
    # remove a-component from b, then renormalise
    proj = (a * b).sum(dim=1, keepdim=True)
    b = b - proj * a
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return a.view(n, *shape), b.view(n, *shape)


def compute_landscape_features(model, x, y):
    """Per-sample 2D loss landscape features.

    Returns dict with tensors of length N:
        mean_loss_increase, max_loss_increase, asymmetry
    """
    N = x.size(0)
    shape = tuple(x.shape[1:])  # (1, 28, 28)
    eps_grid = torch.linspace(GRID_EPS_MIN, GRID_EPS_MAX, GRID_N, device=DEVICE)
    # full 2D grid as flat list of (eps_a, eps_b)
    ea, eb = torch.meshgrid(eps_grid, eps_grid, indexing="ij")
    ea = ea.flatten()  # [G*G]
    eb = eb.flatten()  # [G*G]
    G2 = ea.numel()
    # find centre index (eps_a == 0 and eps_b == 0)
    centre_mask = (ea == 0) & (eb == 0)
    centre_idx = int(torch.nonzero(centre_mask, as_tuple=False).item())

    mean_inc = torch.zeros(N, device=DEVICE)
    max_inc = torch.zeros(N, device=DEVICE)
    asym = torch.zeros(N, device=DEVICE)

    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(DIR_SEED)

    for i in range(0, N, LANDSCAPE_BATCH):
        xb = x[i:i+LANDSCAPE_BATCH]
        yb = y[i:i+LANDSCAPE_BATCH]
        nb = xb.size(0)
        d_a, d_b = _orthonormal_pair(shape, nb, gen)  # [nb, 1, 28, 28]

        # build [nb, G2, 1, 28, 28]
        # x_pert[i, k] = xb[i] + ea[k]*d_a[i] + eb[k]*d_b[i]
        ea_v = ea.view(1, G2, 1, 1, 1)
        eb_v = eb.view(1, G2, 1, 1, 1)
        d_a_v = d_a.unsqueeze(1)  # [nb, 1, 1, 28, 28]
        d_b_v = d_b.unsqueeze(1)
        xb_v = xb.unsqueeze(1)    # [nb, 1, 1, 28, 28]
        x_pert = xb_v + ea_v * d_a_v + eb_v * d_b_v
        x_pert = x_pert.clamp(0.0, 1.0)
        # flatten batch dim for model: [nb*G2, 1, 28, 28]
        x_flat = x_pert.reshape(nb * G2, *shape)
        y_flat = yb.unsqueeze(1).expand(nb, G2).reshape(nb * G2)

        with torch.no_grad():
            # process in sub-chunks to limit memory
            losses = []
            sub = 1024
            for j in range(0, x_flat.size(0), sub):
                logits = model(x_flat[j:j+sub])
                ll = F.cross_entropy(logits, y_flat[j:j+sub], reduction="none")
                losses.append(ll)
            losses = torch.cat(losses).view(nb, G2)

        centre_loss = losses[:, centre_idx:centre_idx+1]  # [nb, 1]
        # off-centre cells
        off_mask = torch.ones(G2, dtype=torch.bool, device=DEVICE)
        off_mask[centre_idx] = False
        off_losses = losses[:, off_mask]  # [nb, G2-1]
        delta = off_losses - centre_loss  # increases (can be negative)

        mean_inc[i:i+nb] = delta.mean(dim=1)
        max_inc[i:i+nb] = delta.max(dim=1).values
        # asymmetry: ratio of max off-centre loss to min off-centre loss
        # losses are non-negative (cross-entropy); clamp denominator
        max_l = off_losses.max(dim=1).values
        min_l = off_losses.min(dim=1).values.clamp_min(1e-8)
        asym[i:i+nb] = max_l / min_l

    return {
        "mean_loss_increase": mean_inc,
        "max_loss_increase": max_inc,
        "asymmetry": asym,
    }


# ------------------------------ baseline features ---------------------------
def compute_margin(model, x, y):
    N = x.size(0)
    with torch.no_grad():
        chunks = []
        for i in range(0, N, 512):
            chunks.append(model(x[i:i+512]))
        logits = torch.cat(chunks, 0)
    C = logits.size(1)
    z_true = logits[torch.arange(N, device=logits.device), y]
    masked = logits.masked_fill(F.one_hot(y, C).bool(), float("-inf"))
    z_2nd = masked.max(dim=1).values
    return z_true - z_2nd, logits


def compute_pixel_features(x):
    flat = x.flatten(1)
    return {"mean_pix": flat.mean(dim=1), "std_pix": flat.std(dim=1)}


# --------------------------------- attacks ----------------------------------
def fgsm_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign = fgsm_sign(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    delta = torch.zeros_like(x0).uniform_(-eps, eps)
    adv = (x0 + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    """Binary search per sample for smallest L_inf eps that flips FGSM."""
    sign = fgsm_sign(model, x, y)
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


def batched_attack(model, x, y, fn, **kw):
    out = []
    for i in range(0, x.size(0), 512):
        out.append(fn(model, x[i:i+512], y[i:i+512], **kw))
    return torch.cat(out)


# ------------------------------ AUROC reporting -----------------------------
def univariate_auroc(feats_np, names, y_bin):
    print(f"  positive rate = {y_bin.mean():.4f}")
    for i, n in enumerate(names):
        v = feats_np[:, i]
        if not np.isfinite(v).all() or v.std() == 0:
            print(f"    {n:<32} AUROC = NaN (degenerate)")
            continue
        try:
            a = roc_auc_score(y_bin, v)
        except ValueError:
            print(f"    {n:<32} AUROC = NaN (degenerate)")
            continue
        a_dir = max(a, 1 - a)
        print(f"    {n:<32} AUROC = {a_dir:.4f}  (raw={a:.4f})")


def correlations(feats_np, names, y_cont):
    print(f"  mean min_eps = {y_cont.mean():.4f}  std = {y_cont.std():.4f}")
    for i, n in enumerate(names):
        v = feats_np[:, i]
        if not np.isfinite(v).all() or v.std() == 0:
            print(f"    {n:<32} corr = NaN (degenerate)")
            continue
        c = np.corrcoef(v, y_cont)[0, 1]
        print(f"    {n:<32} corr(min_eps) = {c:+.4f}")


# ----------------------------------- main -----------------------------------
def main():
    print(f"device = {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(f"training CNN for {EPOCHS} epochs on FashionMNIST (seed={SEED})...")
    t0 = time.time()
    model = train(SEED, train_set)
    print(f"training done in {time.time()-t0:.1f}s")

    # stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # predictions / correctness filter
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct = preds == test_y
    print(f"clean accuracy = {correct.float().mean().item():.4f}")
    x_c, y_c = test_x[correct], test_y[correct]
    print(f"using {int(correct.sum())} correctly-classified samples")

    # --- features ---
    print("computing 2D loss-landscape features ...")
    t0 = time.time()
    land = compute_landscape_features(model, x_c, y_c)
    print(f"  done ({time.time()-t0:.1f}s)")

    print("computing margin and pixel features ...")
    margin_c, _ = compute_margin(model, x_c, y_c)
    pix = compute_pixel_features(x_c)

    feat_names = [
        "mean_loss_increase",
        "max_loss_increase",
        "asymmetry",
        "margin",
        "mean_pix",
        "std_pix",
    ]
    feat_cols = [
        land["mean_loss_increase"],
        land["max_loss_increase"],
        land["asymmetry"],
        margin_c,
        pix["mean_pix"],
        pix["std_pix"],
    ]
    feats = torch.stack(feat_cols, dim=1)
    feats_np = feats.detach().cpu().numpy()

    # --- attacks ---
    print("\ncomputing FGSM self-flip @ eps=15/255 ...")
    t0 = time.time()
    fgsm = batched_attack(model, x_c, y_c, fgsm_flip, eps=EPS_TEST)
    print(f"  done ({time.time()-t0:.1f}s)  flip rate = {fgsm.float().mean():.4f}")

    print("computing PGD self-flip @ eps=15/255, 10-step ...")
    t0 = time.time()
    pgd = batched_attack(model, x_c, y_c, pgd_flip, eps=EPS_TEST,
                         alpha=PGD_ALPHA, steps=PGD_STEPS)
    print(f"  done ({time.time()-t0:.1f}s)  flip rate = {pgd.float().mean():.4f}")

    print("computing min_eps_FGSM (binary search) ...")
    t0 = time.time()
    me = batched_attack(model, x_c, y_c, min_eps_fgsm)
    print(f"  done ({time.time()-t0:.1f}s)  mean min_eps = {me.mean():.4f}")

    fgsm_np = fgsm.cpu().numpy().astype(int)
    pgd_np = pgd.cpu().numpy().astype(int)
    me_np = me.cpu().numpy()

    # --- analysis ---
    print("\n========== univariate AUROC: FGSM_self_flip ==========")
    univariate_auroc(feats_np, feat_names, fgsm_np)

    print("\n========== univariate AUROC: PGD_self_flip ==========")
    univariate_auroc(feats_np, feat_names, pgd_np)

    print("\n========== Pearson correlation: min_eps_FGSM ==========")
    correlations(feats_np, feat_names, me_np)

    # Also AUROC for min_eps treated as binary at its median (vulnerable = below median)
    median = float(np.median(me_np))
    y_me_bin = (me_np <= median).astype(int)
    print(f"\n========== univariate AUROC: min_eps <= median ({median:.4f}) ==========")
    univariate_auroc(feats_np, feat_names, y_me_bin)


if __name__ == "__main__":
    main()
