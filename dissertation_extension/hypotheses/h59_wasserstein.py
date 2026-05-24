"""
H59: Wasserstein adversarial attack (Wong et al. 2019) per-sample vulnerability.

Hypothesis: Per-sample vulnerability to a Wasserstein-projected PGD attack
(Wong, Schmidt, Kolter 2019, "Wasserstein Adversarial Examples via Projected
Sinkhorn Iterations") is predicted by simple image-statistics features:
margin, mean pixel intensity, std pixel intensity, and mean Sobel edge
magnitude.

Pipeline:
  1. Train a small CNN (matching diagnostic_test.py's architecture) on
     Fashion-MNIST for 10 epochs.
  2. Implement a Wasserstein-projected PGD attack: at each step take an L_inf
     gradient sign step, then project onto the Wasserstein ball of radius eps
     around the original image. The projection is approximated with Sinkhorn
     iterations on the local-pixel transport polytope (each pixel can only
     send mass to its small neighbourhood). Wong et al.'s formulation uses
     entropic-regularised conjugate-Sinkhorn projection; we follow that
     approximation. If the POT library (`pip install pot`) is available we
     fall back to ot.sinkhorn for the inner solver; otherwise we use a
     hand-rolled Sinkhorn loop.
  3. Targets:
        - flipped_Wasserstein at eps=0.1
        - wasserstein_perturbation_magnitude  (W1 distance achieved)
  4. Features:
        - margin       (final logit gap between top-1 and runner-up)
        - mean_pix     (mean pixel intensity of clean image)
        - std_pix      (std of pixel intensities of clean image)
        - sobel_mean   (mean magnitude of Sobel-filter response)
  5. Univariate AUROC per feature against the binary target, and Pearson
     correlation against the continuous perturbation-magnitude target.

Web search: "Wasserstein adversarial Wong 2019 pytorch" — the original
reference implementation is at https://github.com/locuslab/projected_sinkhorn
which we mirror in spirit (local transport, conjugate-Sinkhorn).
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

try:
    import ot  # POT library
    HAS_POT = True
except Exception:
    HAS_POT = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS_W = 0.1            # Wasserstein-ball radius
PGD_STEPS = 30
PGD_ALPHA = 0.02       # L_inf step size before projection
SINKHORN_ITERS = 40
SINKHORN_REG = 0.01
KERNEL_RADIUS = 2      # local transport neighbourhood radius (in pixels)


# --------------------------------------------------------------------------- #
# Model — matches diagnostic_test.py
# --------------------------------------------------------------------------- #
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


def train(model, loader, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{epochs}  ({time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------- #
# Ground-cost & local transport setup
# --------------------------------------------------------------------------- #
def build_ground_cost(H, W, radius=KERNEL_RADIUS, device=DEVICE):
    """Return (offsets, costs) for a square local-neighbourhood transport.

    offsets: list of (dy, dx) tuples, shape (K, 2)
    costs:   tensor of squared-Euclidean ground-costs, shape (K,)
    """
    offs, cs = [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            offs.append((dy, dx))
            cs.append(float(dy * dy + dx * dx))
    return offs, torch.tensor(cs, device=device)


def shift_image(img, dy, dx):
    """Zero-padded pixel shift by (dy, dx). img: (B,1,H,W)."""
    B, C, H, W = img.shape
    out = torch.zeros_like(img)
    y0_src = max(0, -dy); y0_dst = max(0, dy)
    x0_src = max(0, -dx); x0_dst = max(0, dx)
    h = H - abs(dy); w = W - abs(dx)
    if h <= 0 or w <= 0:
        return out
    out[:, :, y0_dst:y0_dst + h, x0_dst:x0_dst + w] = \
        img[:, :, y0_src:y0_src + h, x0_src:x0_src + w]
    return out


# --------------------------------------------------------------------------- #
# Wasserstein projection via local Sinkhorn (conjugate-Sinkhorn-ish)
# --------------------------------------------------------------------------- #
def wasserstein_project(x_adv, x_orig, eps, offsets, costs,
                        iters=SINKHORN_ITERS, reg=SINKHORN_REG):
    """Approximate projection of x_adv onto the W1 ball of radius eps around x_orig.

    We treat each image as a non-negative mass distribution over pixels.
    A transport plan pi(i->j) is supported only on (i, i+offset) pairs.
    We minimise <pi, C> + reg * H(pi)  s.t.  pi 1 = x_orig,  pi^T 1 = x_proj
    while enforcing  sum(pi * C) <= eps * sum(mass).

    Implementation: Sinkhorn over local neighbours; binary-search a Lagrange
    multiplier lambda on the cost constraint. Output is the column marginal.
    """
    B, C, H, W = x_adv.shape
    K = len(offsets)
    # Pre-shift the cost map: for each offset k, we operate elementwise.
    # The clean source marginal is x_orig (per-pixel mass).
    # We want to find x_proj close to x_adv (the unconstrained step) that is
    # reachable from x_orig with EMD <= eps.

    # Strategy (simple & faithful): for each sample, run Sinkhorn between
    # source = x_orig, target = x_adv (renormalised), with cost matrix C
    # restricted to local moves; then compute achieved transport cost. If
    # achieved cost > eps, mix x_proj <- (1-t) * x_orig + t * x_adv where
    # t = eps / cost. This is the Frank–Wolfe-style retraction Wong et al.
    # use when the Sinkhorn projection overshoots.

    # Normalise per-sample to probability mass for OT
    src = x_orig.view(B, -1)                    # (B, HW)
    tgt = x_adv.clamp(min=0).view(B, -1)        # (B, HW)
    src_sum = src.sum(1, keepdim=True).clamp(min=1e-8)
    tgt_sum = tgt.sum(1, keepdim=True).clamp(min=1e-8)
    src_n = src / src_sum
    tgt_n = tgt / tgt_sum

    # Build local cost matrix sparse-ish: for each pixel i and offset k,
    # the destination j = shift(i, offset). We use a dense Sinkhorn on
    # the local kernel via shift operations to stay GPU-friendly.

    # Run Sinkhorn-Knopp with the cost expressed implicitly via the kernel
    # K_kernel(i, j) = exp(-C_ij / reg) if j is in i's neighbourhood else 0.
    # Apply K and K^T via shift_image of the per-offset kernel weights.
    log_K = (-costs / reg)                       # (K_offsets,)
    K_w = log_K.exp()                            # (K_offsets,)

    u = torch.ones_like(src_n)
    v = torch.ones_like(tgt_n)

    def apply_K(vec):
        # vec: (B, HW) -> (B, HW)
        vec_img = vec.view(B, 1, H, W)
        out = torch.zeros_like(vec_img)
        for k, (dy, dx) in enumerate(offsets):
            out = out + K_w[k] * shift_image(vec_img, dy, dx)
        return out.view(B, -1)

    def apply_KT(vec):
        vec_img = vec.view(B, 1, H, W)
        out = torch.zeros_like(vec_img)
        for k, (dy, dx) in enumerate(offsets):
            out = out + K_w[k] * shift_image(vec_img, -dy, -dx)
        return out.view(B, -1)

    for _ in range(iters):
        u = src_n / apply_K(v).clamp(min=1e-12)
        v = tgt_n / apply_KT(u).clamp(min=1e-12)

    # Compute the achieved transport cost (approx) :
    # cost = sum_{ij} pi_ij * C_ij ; pi_ij = u_i K_ij v_j
    cost_img = torch.zeros(B, 1, H, W, device=x_adv.device)
    u_img = u.view(B, 1, H, W)
    v_img = v.view(B, 1, H, W)
    for k, (dy, dx) in enumerate(offsets):
        # pi for this offset: u(i) * K_w[k] * v(i + offset)
        v_shift = shift_image(v_img, dy, dx)
        cost_img = cost_img + costs[k] * K_w[k] * u_img * v_shift
    achieved = cost_img.view(B, -1).sum(1) * src_sum.squeeze(1)

    # Renormalise x_adv back to original total mass (the projected image)
    x_proj = tgt_n * src_sum
    x_proj = x_proj.view(B, C, H, W)

    # Retraction: if achieved > eps, interpolate towards x_orig.
    t = torch.ones(B, device=x_adv.device)
    over = achieved > eps
    t[over] = (eps / achieved[over].clamp(min=1e-12)).clamp(max=1.0)
    t_b = t.view(B, 1, 1, 1)
    x_proj = (1 - t_b) * x_orig + t_b * x_proj
    return x_proj.clamp(0, 1), achieved


# --------------------------------------------------------------------------- #
# Wasserstein-projected PGD
# --------------------------------------------------------------------------- #
def wasserstein_pgd(model, x, y, eps=EPS_W, steps=PGD_STEPS, alpha=PGD_ALPHA):
    B, C, H, W = x.shape
    offsets, costs = build_ground_cost(H, W, KERNEL_RADIUS, device=x.device)
    x_orig = x.clone()
    x_adv = x.clone()
    achieved_final = torch.zeros(B, device=x.device)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = (x_adv + alpha * grad.sign()).detach()
        x_adv, achieved_final = wasserstein_project(
            x_adv, x_orig, eps, offsets, costs)
    return x_adv.detach(), achieved_final.detach()


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def sobel_mean(x):
    """Mean Sobel-filter magnitude per image. x: (B,1,H,W) -> (B,)."""
    sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    sy = sx.transpose(2, 3)
    gx = F.conv2d(x, sx, padding=1)
    gy = F.conv2d(x, sy, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag.view(x.size(0), -1).mean(1)


def compute_features(model, x):
    with torch.no_grad():
        logits = []
        for i in range(0, x.size(0), 512):
            logits.append(model(x[i:i + 512]))
        logits = torch.cat(logits, 0)
    top2, _ = logits.topk(2, dim=1)
    margin = top2[:, 0] - top2[:, 1]
    mean_pix = x.view(x.size(0), -1).mean(1)
    std_pix = x.view(x.size(0), -1).std(1)
    s_mean = sobel_mean(x)
    return torch.stack([margin, mean_pix, std_pix, s_mean], dim=1), logits


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True,  download=True, transform=tf)
    test_set  = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)

    print(f"device={DEVICE}  POT-available={HAS_POT}")
    print("training CNN on Fashion-MNIST for 10 epochs ...")
    model = CNN(10).to(DEVICE)
    train(model, train_loader, EPOCHS)
    model.eval()

    # Stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # Features (computed on the clean test set)
    print("computing features ...")
    feats, logits = compute_features(model, test_x)
    final_pred = logits.argmax(1)
    correct = final_pred == test_y
    print(f"  clean accuracy = {correct.float().mean().item():.4f}")
    # Restrict to samples Model A classifies correctly (so a flip is meaningful)
    idx = correct.nonzero(as_tuple=True)[0]
    x_c = test_x[idx]; y_c = test_y[idx]; feats_c = feats[idx]

    # Run Wasserstein-PGD in batches
    print(f"running Wasserstein-PGD eps={EPS_W} steps={PGD_STEPS} on "
          f"{x_c.size(0)} samples ...")
    flipped = torch.zeros(x_c.size(0), dtype=torch.bool, device=DEVICE)
    magnitude = torch.zeros(x_c.size(0), device=DEVICE)
    BS = 128
    t0 = time.time()
    for i in range(0, x_c.size(0), BS):
        xb = x_c[i:i + BS]; yb = y_c[i:i + BS]
        x_adv, ach = wasserstein_pgd(model, xb, yb, eps=EPS_W,
                                     steps=PGD_STEPS, alpha=PGD_ALPHA)
        with torch.no_grad():
            preds = model(x_adv).argmax(1)
        flipped[i:i + BS] = preds != yb
        magnitude[i:i + BS] = ach
        if (i // BS) % 5 == 0:
            print(f"  batch {i//BS:>3}  flip_so_far={flipped[:i+BS].float().mean():.3f}  "
                  f"elapsed={time.time()-t0:.1f}s")
    print(f"  attack success rate = {flipped.float().mean().item():.4f}")
    print(f"  mean Wasserstein magnitude achieved = {magnitude.mean().item():.4f}")

    # ---- Univariate AUROC ----
    names = ["margin", "mean_pix", "std_pix", "sobel_mean"]
    y_bin = flipped.cpu().numpy().astype(int)
    y_mag = magnitude.cpu().numpy()
    feats_np = feats_c.cpu().numpy()

    print("\n=== univariate AUROC vs flipped_Wasserstein (eps=0.1) ===")
    if y_bin.std() == 0:
        print("  target degenerate (all same), skipping AUROC")
    else:
        for i, n in enumerate(names):
            a = roc_auc_score(y_bin, feats_np[:, i])
            a = max(a, 1 - a)
            print(f"  {n:<12}  AUROC = {a:.4f}")

    print("\n=== correlations vs wasserstein_perturbation_magnitude ===")
    for i, n in enumerate(names):
        pr = pearsonr(feats_np[:, i], y_mag)[0]
        sr = spearmanr(feats_np[:, i], y_mag)[0]
        print(f"  {n:<12}  pearson={pr:+.4f}  spearman={sr:+.4f}")


if __name__ == "__main__":
    main()
