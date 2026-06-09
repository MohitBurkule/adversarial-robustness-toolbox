"""
h520_seed_stability_2d_viz.py
Multi-seed decision boundary stability visualization (2D synthetic dataset).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)
N = 600
z = RNG.normal(0, 1, N)
x1 = 0.98 * z + 0.20 * RNG.normal(0, 1, N)
x2 = 0.92 * z + 0.39 * RNG.normal(0, 1, N)
y = (z > 0).astype(int)
X = np.stack([x1, x2], axis=1).astype(np.float32)
Y = y.astype(np.int64)

X_t = torch.from_numpy(X)
Y_t = torch.from_numpy(Y)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_mlp(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(2, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 2),
    )

# ---------------------------------------------------------------------------
# PGD attack helper
# ---------------------------------------------------------------------------

def pgd_attack(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor,
               eps: float = 0.3, steps: int = 7, alpha: float = None) -> torch.Tensor:
    if alpha is None:
        alpha = eps * 2 / steps
    delta = torch.zeros_like(xb).uniform_(-eps, eps)
    delta.requires_grad_(True)
    for _ in range(steps):
        loss = nn.CrossEntropyLoss()(model(xb + delta), yb)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + alpha * delta.grad.sign()).clamp(-eps, eps)
        delta.grad.zero_()
    return (xb + delta).detach()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_standard(seed: int, epochs: int = 200) -> nn.Module:
    model = make_mlp(seed)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        ce(model(X_t), Y_t).backward()
        opt.step()
    model.eval()
    return model


def train_pgd_at(seed: int, epochs: int = 200,
                 eps: float = 0.3, steps: int = 7) -> nn.Module:
    model = make_mlp(seed)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        xadv = pgd_attack(model, X_t, Y_t, eps=eps, steps=steps)
        opt.zero_grad()
        ce(model(xadv), Y_t).backward()
        opt.step()
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

GRID_RES = 50
gx = np.linspace(-4, 4, GRID_RES)
gy = np.linspace(-4, 4, GRID_RES)
GX, GY = np.meshgrid(gx, gy)
grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1).astype(np.float32)
grid_t = torch.from_numpy(grid_pts)


def get_prob_map(model: nn.Module) -> np.ndarray:
    with torch.no_grad():
        logits = model(grid_t)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    return probs.reshape(GRID_RES, GRID_RES)


def get_boundary_contour(model: nn.Module, ax, color, alpha=0.5):
    pm = get_prob_map(model)
    ax.contour(GX, GY, pm, levels=[0.5], colors=[color], linewidths=0.8, alpha=alpha)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

N_SEEDS = 10
SEEDS_FOR_LINES = list(range(5))

print("Training standard models…")
std_models = [train_standard(s) for s in range(N_SEEDS)]

print("Training PGD-AT models…")
at_models = [train_pgd_at(s) for s in range(N_SEEDS)]

# Aggregate probability maps
std_maps = np.stack([get_prob_map(m) for m in std_models], axis=0)  # (10, 50, 50)
at_maps  = np.stack([get_prob_map(m) for m in at_models],  axis=0)

std_mean = std_maps.mean(axis=0)
at_mean  = at_maps.mean(axis=0)

# Disagreement = fraction of grid points where 0.2 < mean_p < 0.8
std_disagree = float(((std_mean > 0.2) & (std_mean < 0.8)).mean())
at_disagree  = float(((at_mean  > 0.2) & (at_mean  < 0.8)).mean())

print(f"STD disagreement fraction: {std_disagree:.4f}")
print(f"AT  disagreement fraction: {at_disagree:.4f}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
LINE_COLORS = plt.cm.tab10(np.linspace(0, 0.5, 5))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, mean_map, models, label, disagree in [
    (axes[0], std_mean, std_models, "Standard Training", std_disagree),
    (axes[1], at_mean,  at_models,  "PGD-AT",            at_disagree),
]:
    im = ax.pcolormesh(GX, GY, mean_map, cmap="RdBu", vmin=0, vmax=1, shading="auto")
    plt.colorbar(im, ax=ax, label="Fraction predicting class 1")

    # Data points
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], c="red",  s=4, alpha=0.35, label="Class 0")
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], c="blue", s=4, alpha=0.35, label="Class 1")

    # Individual decision boundary lines (5 seeds)
    for i, seed in enumerate(SEEDS_FOR_LINES):
        get_boundary_contour(models[seed], ax, color=LINE_COLORS[i], alpha=0.8)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(f"{label}\nBoundary disagreement = {disagree:.3f}")
    ax.legend(loc="upper left", fontsize=7, markerscale=2)

fig.suptitle("Multi-Seed Decision Boundary Stability (10 seeds)", fontsize=13, fontweight="bold")
plt.tight_layout()

out_path = os.path.join(
    os.path.dirname(__file__),
    "../results/fashion_mnist/h520_seed_stability_2d_viz.png"
)
out_path = os.path.normpath(out_path)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
