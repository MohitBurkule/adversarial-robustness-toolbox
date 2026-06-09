"""
H528: Radial Feature Geometry — representation determines adversarial fragility.

Core idea: A concentric-ring binary classification dataset has a circular true decision
boundary that is trivially separable in polar (r, theta) coordinates but non-linearly
hard in Cartesian (x1, x2) coordinates. Training the same MLP architecture in the
"wrong" feature space (Cartesian) makes it more fragile to adversarial perturbations
even though the underlying data distribution is identical.

Three models:
  1. Cartesian-STD  — trained on (x1, x2) with standard CE loss
  2. Polar-STD      — trained on (r, theta/pi) with standard CE loss
  3. Cartesian-AT   — trained on (x1, x2) with PGD adversarial training (eps=0.3)

Metrics evaluated at FGSM eps=0.3 and PGD-7 eps=0.3:
  - Clean accuracy, FGSM ASR, PGD ASR
  - Polar-space displacement (Delta_r, Delta_theta) under FGSM perturbation
  - Boundary smoothness: std of predicted class along the true circular boundary

PASS: Polar-STD FGSM_ASR < Cartesian-STD FGSM_ASR
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
BASE = "/run/media/mohit/NewVolume1/adversarial-robustness-toolbox/dissertation_extension"
OUT_PNG = os.path.join(BASE, "results/fashion_mnist/h528_radial_feature_geometry.png")
OUT_TXT = os.path.join(BASE, "results/fashion_mnist/h528_radial_feature_geometry_output.txt")
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)


# Tee stdout to file
class Tee:
    def __init__(self, path):
        self._file = open(path, "w")
        self._stdout = sys.stdout

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


sys.stdout = Tee(OUT_TXT)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
INNER_R = 1.0
OUTER_R = 2.2
NOISE = 0.25
N = 1200  # per class


def make_rings(n, inner_r=INNER_R, outer_r=OUTER_R, noise=NOISE, seed=42):
    rng = np.random.default_rng(seed)
    # Class 0: inner ring
    theta0 = rng.uniform(0, 2 * np.pi, n)
    r0 = inner_r + rng.normal(0, noise, n)
    X0 = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1).astype(np.float32)
    # Class 1: outer ring
    theta1 = rng.uniform(0, 2 * np.pi, n)
    r1 = outer_r + rng.normal(0, noise, n)
    X1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1).astype(np.float32)
    X = np.vstack([X0, X1])
    Y = np.array([0] * n + [1] * n, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


def to_polar(X):
    """Convert Cartesian (x1, x2) → polar (r, theta/pi) features."""
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    theta = np.arctan2(X[:, 1], X[:, 0]) / np.pi  # normalise to [-1, 1]
    return np.stack([r, theta], axis=1).astype(np.float32)


# Build train/test splits
X_all, Y_all = make_rings(N, seed=42)
X_test_raw, Y_test_raw = make_rings(N // 3, seed=99)

X_train_cart = torch.from_numpy(X_all).to(DEVICE)
Y_train = torch.from_numpy(Y_all).to(DEVICE)
X_test_cart = torch.from_numpy(X_test_raw).to(DEVICE)
Y_test = torch.from_numpy(Y_test_raw).to(DEVICE)

X_train_polar = torch.from_numpy(to_polar(X_all)).to(DEVICE)
X_test_polar = torch.from_numpy(to_polar(X_test_raw)).to(DEVICE)

# ---------------------------------------------------------------------------
# Model: 2 → 64 → 64 → 2 MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Attacks (L-inf)
# ---------------------------------------------------------------------------

def fgsm(model, x, y, eps=0.3):
    """Fast Gradient Sign Method."""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    with torch.no_grad():
        x_adv = x + eps * x_adv.grad.sign()
    return x_adv.detach()


def pgd(model, x, y, eps=0.3, alpha=0.1, steps=7):
    """Projected Gradient Descent attack."""
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            # Project back into eps-ball around original x
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    return x_adv.detach()


def pgd_at_step(model, x, y, eps=0.3, alpha=0.1, steps=7):
    """PGD inner loop for adversarial training (no clamp to [0,1] — unbounded input)."""
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    return x_adv.detach()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS = 500
LR = 1e-3
BATCH = 256
AT_EPS = 0.3
AT_STEPS = 7
AT_ALPHA = 0.1


def train_standard(X_train, Y_train, seed=0):
    torch.manual_seed(seed)
    model = MLP().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    n = len(X_train)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = X_train[idx], Y_train[idx]
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    return model


def train_at(X_train, Y_train, eps=AT_EPS, steps=AT_STEPS, alpha=AT_ALPHA, seed=0):
    torch.manual_seed(seed)
    model = MLP().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    n = len(X_train)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = X_train[idx], Y_train[idx]
            model.eval()
            xb_adv = pgd_at_step(model, xb, yb, eps=eps, alpha=alpha, steps=steps)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(xb_adv), yb).backward()
            opt.step()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test, Y_test, eps=0.3):
    """Returns (clean_acc, fgsm_asr, pgd_asr, fgsm_adv_x)."""
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(1)
        clean_acc = (preds == Y_test).float().mean().item()

    # Only evaluate attack success on correctly classified samples
    correct_mask = (preds == Y_test)
    X_correct = X_test[correct_mask]
    Y_correct = Y_test[correct_mask]

    # FGSM
    X_fgsm = fgsm(model, X_correct, Y_correct, eps=eps)
    with torch.no_grad():
        fgsm_preds = model(X_fgsm).argmax(1)
    fgsm_asr = (fgsm_preds != Y_correct).float().mean().item()

    # PGD-7
    X_pgd = pgd(model, X_correct, Y_correct, eps=eps, steps=7)
    with torch.no_grad():
        pgd_preds = model(X_pgd).argmax(1)
    pgd_asr = (pgd_preds != Y_correct).float().mean().item()

    return clean_acc, fgsm_asr, pgd_asr, X_correct.cpu().numpy(), X_fgsm.cpu().numpy(), Y_correct.cpu().numpy(), fgsm_preds.cpu().numpy()


def boundary_smoothness(model, feature_type="cart", n_points=360, boundary_r=None):
    """
    Sample n_points on the true circular boundary (r = midpoint between inner/outer rings),
    get model predictions, return std of predicted class probabilities.
    A perfectly rotationally-symmetric model returns std ≈ 0.
    """
    if boundary_r is None:
        boundary_r = (INNER_R + OUTER_R) / 2.0
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x1 = (boundary_r * np.cos(thetas)).astype(np.float32)
    x2 = (boundary_r * np.sin(thetas)).astype(np.float32)
    X_circ_cart = np.stack([x1, x2], axis=1)

    if feature_type == "polar":
        X_input = torch.from_numpy(to_polar(X_circ_cart)).to(DEVICE)
    else:
        X_input = torch.from_numpy(X_circ_cart).to(DEVICE)

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(X_input), dim=1)[:, 1].cpu().numpy()
    return float(np.std(probs))


# ---------------------------------------------------------------------------
# Train all three models
# ---------------------------------------------------------------------------
print("Training Cartesian-STD ...")
model_cart_std = train_standard(X_train_cart, Y_train, seed=0)

print("Training Polar-STD ...")
model_polar_std = train_standard(X_train_polar, Y_train, seed=0)

print("Training Cartesian-AT ...")
model_cart_at = train_at(X_train_cart, Y_train, seed=0)

print("All models trained.\n")

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
print("Evaluating Cartesian-STD ...")
cart_clean, cart_fgsm_asr, cart_pgd_asr, X_corr_cart, X_fgsm_cart, Y_corr_cart, fgsm_preds_cart = evaluate(
    model_cart_std, X_test_cart, Y_test, eps=0.3
)
cart_smooth = boundary_smoothness(model_cart_std, feature_type="cart")

print("Evaluating Polar-STD ...")
pol_clean, pol_fgsm_asr, pol_pgd_asr, X_corr_pol, X_fgsm_pol, Y_corr_pol, fgsm_preds_pol = evaluate(
    model_polar_std, X_test_polar, Y_test, eps=0.3
)
pol_smooth = boundary_smoothness(model_polar_std, feature_type="polar")

print("Evaluating Cartesian-AT ...")
at_clean, at_fgsm_asr, at_pgd_asr, X_corr_at, X_fgsm_at, Y_corr_at, fgsm_preds_at = evaluate(
    model_cart_at, X_test_cart, Y_test, eps=0.3
)
at_smooth = boundary_smoothness(model_cart_at, feature_type="cart")

# Print summary
print("\n" + "=" * 60)
print(f"{'Model':<20} {'CleanAcc':>10} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'BndSmooth':>12}")
print("-" * 60)
print(f"{'Cartesian-STD':<20} {cart_clean:>10.4f} {cart_fgsm_asr:>10.4f} {cart_pgd_asr:>10.4f} {cart_smooth:>12.4f}")
print(f"{'Polar-STD':<20} {pol_clean:>10.4f} {pol_fgsm_asr:>10.4f} {pol_pgd_asr:>10.4f} {pol_smooth:>12.4f}")
print(f"{'Cartesian-AT':<20} {at_clean:>10.4f} {at_fgsm_asr:>10.4f} {at_pgd_asr:>10.4f} {at_smooth:>12.4f}")
print("=" * 60)

# PASS check
if pol_fgsm_asr < cart_fgsm_asr:
    print("\nPASS: Polar-STD FGSM_ASR < Cartesian-STD FGSM_ASR "
          f"({pol_fgsm_asr:.4f} < {cart_fgsm_asr:.4f})")
else:
    print("\nFAIL: Polar-STD FGSM_ASR >= Cartesian-STD FGSM_ASR "
          f"({pol_fgsm_asr:.4f} >= {cart_fgsm_asr:.4f})")

# ---------------------------------------------------------------------------
# Polar-space displacement analysis for Cartesian-STD
# ---------------------------------------------------------------------------
# Delta_r and Delta_theta under FGSM perturbation
r_orig = np.sqrt(X_corr_cart[:, 0] ** 2 + X_corr_cart[:, 1] ** 2)
theta_orig = np.arctan2(X_corr_cart[:, 1], X_corr_cart[:, 0])

r_adv = np.sqrt(X_fgsm_cart[:, 0] ** 2 + X_fgsm_cart[:, 1] ** 2)
theta_adv = np.arctan2(X_fgsm_cart[:, 1], X_fgsm_cart[:, 0])

delta_r = np.abs(r_adv - r_orig)
# Angular difference (wrapped to [-pi, pi])
d_theta_raw = theta_adv - theta_orig
d_theta_wrapped = (d_theta_raw + np.pi) % (2 * np.pi) - np.pi
delta_theta = np.abs(d_theta_wrapped)

print(f"\nPolar displacement under FGSM (Cartesian-STD):")
print(f"  mean |Delta_r|     = {delta_r.mean():.4f}  std = {delta_r.std():.4f}")
print(f"  mean |Delta_theta| = {delta_theta.mean():.4f}  std = {delta_theta.std():.4f}")

# FGSM flipped mask for Cartesian model
fgsm_flipped_cart = (fgsm_preds_cart != Y_corr_cart)

# ---------------------------------------------------------------------------
# Decision boundary grid helper
# ---------------------------------------------------------------------------

def decision_boundary_grid(model, feature_type="cart", resolution=200):
    x1_range = np.linspace(-3.5, 3.5, resolution)
    x2_range = np.linspace(-3.5, 3.5, resolution)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    X_grid_cart = np.stack([xx1.ravel(), xx2.ravel()], axis=1).astype(np.float32)
    if feature_type == "polar":
        X_input = torch.from_numpy(to_polar(X_grid_cart)).to(DEVICE)
    else:
        X_input = torch.from_numpy(X_grid_cart).to(DEVICE)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(X_input), dim=1)[:, 1].cpu().numpy()
    return xx1, xx2, probs.reshape(resolution, resolution)


# ---------------------------------------------------------------------------
# Radial and angular ASR profiles for Cartesian-STD FGSM
# ---------------------------------------------------------------------------
N_BINS = 12

# Radial ASR profile
r_bins = np.linspace(r_orig.min(), r_orig.max(), N_BINS + 1)
r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
radial_asr = []
for lo, hi in zip(r_bins[:-1], r_bins[1:]):
    mask = (r_orig >= lo) & (r_orig < hi)
    if mask.sum() > 0:
        radial_asr.append(fgsm_flipped_cart[mask].mean())
    else:
        radial_asr.append(np.nan)
radial_asr = np.array(radial_asr)

# Angular ASR profile
theta_bins = np.linspace(-np.pi, np.pi, N_BINS + 1)
theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
angular_asr = []
for lo, hi in zip(theta_bins[:-1], theta_bins[1:]):
    mask = (theta_orig >= lo) & (theta_orig < hi)
    if mask.sum() > 0:
        angular_asr.append(fgsm_flipped_cart[mask].mean())
    else:
        angular_asr.append(np.nan)
angular_asr = np.array(angular_asr)

# ---------------------------------------------------------------------------
# Figure: 3×3 panels
# ---------------------------------------------------------------------------
print("\nGenerating figure ...")

fig, axes = plt.subplots(3, 3, figsize=(15, 14))
fig.suptitle("H528: Radial Feature Geometry — Representation Determines Adversarial Fragility",
             fontsize=13, fontweight="bold", y=0.98)

# Colour palette
C0, C1 = "#4477AA", "#EE6677"
FLIP_COLOR = "#CC3311"
cmap_boundary = "RdBu"

# Test data in Cartesian for plotting
X_test_np = X_test_cart.cpu().numpy()
Y_test_np = Y_test.cpu().numpy()

# --- Row 0: Decision boundaries (Cartesian space) ---
models_row0 = [
    (model_cart_std, "cart", "Cartesian-STD"),
    (model_polar_std, "polar", "Polar-STD"),
    (model_cart_at, "cart", "Cartesian-AT"),
]
for col, (mdl, ftype, title) in enumerate(models_row0):
    ax = axes[0, col]
    xx1, xx2, probs = decision_boundary_grid(mdl, feature_type=ftype)
    ax.contourf(xx1, xx2, probs, levels=50, cmap=cmap_boundary, alpha=0.6, vmin=0, vmax=1)
    ax.contour(xx1, xx2, probs, levels=[0.5], colors="k", linewidths=1.2)
    ax.scatter(X_test_np[Y_test_np == 0, 0], X_test_np[Y_test_np == 0, 1],
               s=5, c=C0, alpha=0.5, label="Class 0")
    ax.scatter(X_test_np[Y_test_np == 1, 0], X_test_np[Y_test_np == 1, 1],
               s=5, c=C1, alpha=0.5, label="Class 1")
    # Flipped test points (Cartesian-STD only)
    if col == 0 and fgsm_flipped_cart.sum() > 0:
        ax.scatter(X_corr_cart[fgsm_flipped_cart, 0], X_corr_cart[fgsm_flipped_cart, 1],
                   marker="x", s=25, c=FLIP_COLOR, linewidths=1.2, label="FGSM flipped", zorder=5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    if col == 0:
        ax.legend(fontsize=7, loc="upper left")

# --- Row 1, col 0: Polar feature space scatter ---
ax = axes[1, 0]
X_test_pol_np = X_test_polar.cpu().numpy()
# Compute FGSM outcome on full test set for Cartesian model
model_cart_std.eval()
with torch.no_grad():
    preds_all = model_cart_std(X_test_cart).argmax(1).cpu().numpy()
correct_all = (preds_all == Y_test_np)
X_corr_all_cart = X_test_np[correct_all]
Y_corr_all = Y_test_np[correct_all]
X_fgsm_all = fgsm(model_cart_std, torch.from_numpy(X_corr_all_cart).to(DEVICE),
                  torch.from_numpy(Y_corr_all).to(DEVICE), eps=0.3).cpu().numpy()
fgsm_preds_all = model_cart_std(torch.from_numpy(X_fgsm_all).to(DEVICE)).argmax(1).cpu().numpy()
fgsm_flipped_all = (fgsm_preds_all != Y_corr_all)

# Polar coords for correct test points
pol_all = to_polar(X_corr_all_cart)
ax.scatter(pol_all[~fgsm_flipped_all, 0], pol_all[~fgsm_flipped_all, 1],
           s=6, c=C0, alpha=0.4, label="Robust")
ax.scatter(pol_all[fgsm_flipped_all, 0], pol_all[fgsm_flipped_all, 1],
           s=12, c=FLIP_COLOR, alpha=0.7, marker="x", label="FGSM flipped", zorder=5)
ax.set_xlabel("$r$")
ax.set_ylabel(r"$\theta / \pi$")
ax.set_title("Polar space: FGSM outcomes\n(Cartesian-STD)", fontsize=10)
ax.legend(fontsize=8)
# Mark boundary r
boundary_r = (INNER_R + OUTER_R) / 2.0
ax.axvline(boundary_r, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Boundary r")

# --- Row 1, col 1: Radial ASR profile ---
ax = axes[1, 1]
valid = ~np.isnan(radial_asr)
ax.bar(r_centers[valid], radial_asr[valid],
       width=(r_centers[1] - r_centers[0]) * 0.8, color="#CC6677", alpha=0.8)
ax.axvline(INNER_R, color=C0, linestyle="--", linewidth=1.2, label=f"inner r={INNER_R}")
ax.axvline(OUTER_R, color=C1, linestyle="--", linewidth=1.2, label=f"outer r={OUTER_R}")
ax.axvline(boundary_r, color="gray", linestyle=":", linewidth=1.2, label=f"boundary r={boundary_r:.2f}")
ax.set_xlabel("$r$ (radial distance)")
ax.set_ylabel("FGSM ASR")
ax.set_title("Radial ASR profile\n(Cartesian-STD)", fontsize=10)
ax.legend(fontsize=7)
ax.set_ylim(0, 1)

# --- Row 1, col 2: Angular ASR profile ---
ax = axes[1, 2]
valid_th = ~np.isnan(angular_asr)
ax.bar(theta_centers[valid_th], angular_asr[valid_th],
       width=(theta_centers[1] - theta_centers[0]) * 0.8, color="#DDAA33", alpha=0.8)
ax.axhline(cart_fgsm_asr, color="k", linestyle="--", linewidth=1, label=f"mean ASR={cart_fgsm_asr:.3f}")
ax.set_xlabel(r"$\theta$ (radians)")
ax.set_ylabel("FGSM ASR")
ax.set_title("Angular ASR profile\n(Cartesian-STD)", fontsize=10)
ax.legend(fontsize=8)
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(0, 1)

# --- Row 2, col 0: Histogram of Delta_r and Delta_theta ---
ax = axes[2, 0]
ax.hist(delta_r, bins=40, alpha=0.7, color=C0, label=r"$|\Delta r|$", density=True)
ax.hist(delta_theta, bins=40, alpha=0.7, color=C1, label=r"$|\Delta\theta|$", density=True)
ax.set_xlabel("Polar displacement magnitude")
ax.set_ylabel("Density")
ax.set_title(r"$|\Delta r|$ and $|\Delta\theta|$ under FGSM" + "\n(Cartesian eps=0.3)", fontsize=10)
ax.legend(fontsize=9)

# --- Row 2, col 1: Scatter: Cartesian pert magnitude vs Delta_r ---
ax = axes[2, 1]
cart_pert_mag = np.sqrt(((X_fgsm_cart - X_corr_cart) ** 2).sum(axis=1))
# Colour by r_orig (small r → large Delta_theta)
sc = ax.scatter(cart_pert_mag, delta_r, c=r_orig, cmap="viridis", s=8, alpha=0.5)
plt.colorbar(sc, ax=ax, label="$r$ (original)")
ax.set_xlabel("Cartesian perturbation magnitude")
ax.set_ylabel(r"$|\Delta r|$")
ax.set_title("Cartesian pert magnitude vs $|\\Delta r|$\n(coloured by original r)", fontsize=10)

# --- Row 2, col 2: Summary table ---
ax = axes[2, 2]
ax.axis("off")
table_data = [
    ["Model", "CleanAcc", "FGSM ASR", "PGD ASR", "BndSmooth"],
    ["Cartesian-STD", f"{cart_clean:.3f}", f"{cart_fgsm_asr:.3f}", f"{cart_pgd_asr:.3f}", f"{cart_smooth:.3f}"],
    ["Polar-STD",     f"{pol_clean:.3f}",  f"{pol_fgsm_asr:.3f}",  f"{pol_pgd_asr:.3f}",  f"{pol_smooth:.3f}"],
    ["Cartesian-AT",  f"{at_clean:.3f}",   f"{at_fgsm_asr:.3f}",   f"{at_pgd_asr:.3f}",   f"{at_smooth:.3f}"],
]
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               cellLoc="center", loc="center", bbox=[0, 0.2, 1, 0.7])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
# Highlight header
for j in range(5):
    tbl[(0, j)].set_facecolor("#DDDDDD")
    tbl[(0, j)].set_text_props(fontweight="bold")
# Highlight Polar-STD row (lower ASR)
for j in range(5):
    tbl[(2, j)].set_facecolor("#DDEEFF")

pass_str = "PASS" if pol_fgsm_asr < cart_fgsm_asr else "FAIL"
pass_color = "green" if pass_str == "PASS" else "red"
ax.text(0.5, 0.08, f"{pass_str}: Polar FGSM ASR ({pol_fgsm_asr:.3f}) {'<' if pol_fgsm_asr < cart_fgsm_asr else '>='} Cart FGSM ASR ({cart_fgsm_asr:.3f})",
        transform=ax.transAxes, ha="center", va="center", fontsize=10,
        fontweight="bold", color=pass_color)
ax.set_title("Summary", fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {OUT_PNG}")

sys.stdout.flush()
print("\nDone.")


if __name__ == "__main__":
    pass  # All code runs at module level above (standard pattern for this repo)
