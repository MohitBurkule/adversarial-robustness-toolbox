"""
H527 - Single-example ripple: how much can one training point shift the decision
boundary and change ASR across the test set?

Concept:
  A single training example — poisoned (bad) or optimally chosen (good) — can
  shift the decision boundary of a small model enough to flip multiple test
  predictions and measurably change adversarial success rate.  We quantify
  this "ripple" under four scenarios:

    A — Label-flipped poison: add 1 point near the class-0/class-1 boundary
        but labelled as class 1 (wrong label).
    B — Worst-case adversarial poison: grid search → pick the candidate that
        maximally increases FGSM ASR (influence-function proxy).
    C — Best-case helpful point: grid search → pick the candidate that most
        reduces FGSM ASR.
    D — Ripple vs distance: add a single (correctly labelled) point at
        distances [0.1, 0.3, 0.5, 1.0, 2.0] from the true boundary (x=0
        line separating class 0 and class 1); measure acc/ASR change.

Dataset: 4-class synthetic 2-D, corners at (±1.5, ±1.5), std=0.5.
Model:   2-layer MLP, 50 training examples per class (200 total) so that
         single examples have genuine leverage.

Figure: 2×3 panels
  (0,0)  Base model decision boundary + training data
  (0,1)  Boundary after adding worst poison point  (red star)
  (0,2)  Boundary after adding best helpful point  (green star)
  (1,0)  Heatmap of ASR delta over 2-D candidate grid
  (1,1)  Line plot — distance from boundary vs ASR change  (Scenario D)
  (1,2)  Bar chart — n_test_predictions_changed per scenario

PASS: worst poison causes >5 % ASR increase OR >2 % accuracy drop.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)

FIG_PATH = os.path.join(RESULTS_DIR, "h527_single_example_ripple.png")
OUT_PATH = os.path.join(RESULTS_DIR, "h527_single_example_ripple_output.txt")

# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")          # 2-D toy; CPU is fine
N_PER_CLASS = 50                      # small so single points have leverage
N_TEST_PER_CLASS = 200
EPOCHS = 300
LR = 5e-3
EPS = 0.1                             # FGSM epsilon (2-D feature space, ~unit scale)
SEED = 42

# Grid for Scenario B/C search and heatmap
GRID_RES = 20                         # 20×20 = 400 candidate points
GRID_LO, GRID_HI = -2.5, 2.5

# Distances for Scenario D
DISTANCES = [0.1, 0.3, 0.5, 1.0, 2.0]

CLASS_CENTERS = [(-1.5, -1.5), (+1.5, -1.5), (-1.5, +1.5), (+1.5, +1.5)]
N_CLASSES = 4
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
MARKERS = ["o", "s", "^", "D"]

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def make_4class(n_per_class, seed=42, std=0.5):
    rng = np.random.default_rng(seed)
    Xs, Ys = [], []
    for ci, (cx, cy) in enumerate(CLASS_CENTERS):
        pts = rng.normal([cx, cy], std, (n_per_class, 2)).astype(np.float32)
        Xs.append(pts)
        Ys.extend([ci] * n_per_class)
    X = np.vstack(Xs)
    Y = np.array(Ys, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


def to_tensors(X, Y):
    return (torch.tensor(X, dtype=torch.float32, device=DEVICE),
            torch.tensor(Y, dtype=torch.long, device=DEVICE))


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def train_model(X_np, Y_np, seed=SEED):
    torch.manual_seed(seed)
    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt, Yt = to_tensors(X_np, Y_np)
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(Xt), Yt).backward()
        opt.step()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# FGSM
# ──────────────────────────────────────────────────────────────────────────────

def fgsm_asr(model, Xt, Yt):
    """Return (ASR float, per-sample bool tensor flipped)."""
    model.eval()
    Xt = Xt.detach().requires_grad_(True)
    loss = F.cross_entropy(model(Xt), Yt)
    loss.backward()
    with torch.no_grad():
        adv = (Xt + EPS * Xt.grad.sign()).detach()
        preds = model(adv).argmax(1)
        flipped = (preds != Yt)
    return flipped.float().mean().item(), flipped


def clean_acc(model, Xt, Yt):
    model.eval()
    with torch.no_grad():
        return (model(Xt).argmax(1) == Yt).float().mean().item()


def get_preds(model, Xt):
    model.eval()
    with torch.no_grad():
        return model(Xt).argmax(1)


# ──────────────────────────────────────────────────────────────────────────────
# Decision boundary helpers
# ──────────────────────────────────────────────────────────────────────────────

def boundary_grid(lo=GRID_LO, hi=GRID_HI, res=300):
    xs = np.linspace(lo, hi, res)
    ys = np.linspace(lo, hi, res)
    xx, yy = np.meshgrid(xs, ys)
    pts = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32,
                       device=DEVICE)
    return xx, yy, pts


def predict_grid(model, pts):
    model.eval()
    with torch.no_grad():
        return model(pts).argmax(1).cpu().numpy()


def mean_boundary_shift(model_a, model_b):
    """Mean absolute difference in grid-class label (0/1 = same/different class)."""
    _, _, pts = boundary_grid(res=200)
    pa = predict_grid(model_a, pts)
    pb = predict_grid(model_b, pts)
    return (pa != pb).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Augment dataset with one extra point
# ──────────────────────────────────────────────────────────────────────────────

def augment(X_np, Y_np, point_xy, label):
    X_aug = np.vstack([X_np, np.array(point_xy, dtype=np.float32).reshape(1, 2)])
    Y_aug = np.append(Y_np, label)
    return X_aug, Y_aug


def scenario_metrics(base_model, aug_model, Xt_test, Yt_test):
    """Return dict with clean_acc_delta, asr_delta, n_predictions_changed, boundary_shift."""
    base_acc = clean_acc(base_model, Xt_test, Yt_test)
    aug_acc = clean_acc(aug_model, Xt_test, Yt_test)
    base_asr, _ = fgsm_asr(base_model, Xt_test.clone(), Yt_test)
    aug_asr, _ = fgsm_asr(aug_model, Xt_test.clone(), Yt_test)

    base_preds = get_preds(base_model, Xt_test)
    aug_preds = get_preds(aug_model, Xt_test)
    n_changed = int((base_preds != aug_preds).sum().item())
    bs = mean_boundary_shift(base_model, aug_model)

    return {
        "clean_acc_delta": (aug_acc - base_acc) * 100,
        "asr_delta": (aug_asr - base_asr) * 100,
        "n_predictions_changed": n_changed,
        "boundary_shift": bs,
        "base_acc": base_acc * 100,
        "aug_acc": aug_acc * 100,
        "base_asr": base_asr * 100,
        "aug_asr": aug_asr * 100,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Grid search for Scenarios B and C
# ──────────────────────────────────────────────────────────────────────────────

def grid_search_candidates(X_train, Y_train, Xt_test, Yt_test, base_asr):
    """
    For each point on a GRID_RES×GRID_RES grid, add it with its nearest
    correct class label, retrain, measure ASR delta.
    Returns:
        grid_asr_delta  (GRID_RES, GRID_RES) numpy array
        worst_point     (x, y) np array
        best_point      (x, y) np array
    """
    xs = np.linspace(GRID_LO, GRID_HI, GRID_RES)
    ys = np.linspace(GRID_LO, GRID_HI, GRID_RES)
    asr_deltas = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)

    worst_delta = -np.inf
    best_delta = np.inf
    worst_pt = None
    best_pt = None

    for i, gx in enumerate(xs):
        for j, gy in enumerate(ys):
            # assign nearest class center label
            dists = [np.hypot(gx - cx, gy - cy) for cx, cy in CLASS_CENTERS]
            label = int(np.argmin(dists))
            X_aug, Y_aug = augment(X_train, Y_train, [gx, gy], label)
            aug_model = train_model(X_aug, Y_aug)
            aug_asr, _ = fgsm_asr(aug_model, Xt_test.clone(), Yt_test)
            delta = (aug_asr - base_asr) * 100
            asr_deltas[j, i] = delta   # row=y, col=x for imshow

            if delta > worst_delta:
                worst_delta = delta
                worst_pt = np.array([gx, gy])
            if delta < best_delta:
                best_delta = delta
                best_pt = np.array([gx, gy])

    return asr_deltas, worst_pt, best_pt


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_boundary(ax, model, X_train, Y_train, title, extra_point=None,
                  extra_color="red", extra_marker="*", extra_label=None):
    xx, yy, pts = boundary_grid(res=300)
    Z = predict_grid(model, pts).reshape(xx.shape)
    cmap = mcolors.ListedColormap(["#d0e8ff", "#ffe0c0", "#d0ffd8", "#ffd0d0"])
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=cmap, alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5, 1.5, 2.5], colors="k", linewidths=0.8, alpha=0.5)
    for ci in range(N_CLASSES):
        mask = Y_train == ci
        ax.scatter(X_train[mask, 0], X_train[mask, 1],
                   c=COLORS[ci], marker=MARKERS[ci], s=18, alpha=0.7,
                   label=f"cls {ci}" if ci == 0 else None, edgecolors="none")
    if extra_point is not None:
        ax.scatter(extra_point[0], extra_point[1], c=extra_color,
                   marker=extra_marker, s=200, zorder=10, label=extra_label,
                   edgecolors="k", linewidths=0.7)
        if extra_label:
            ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(GRID_LO, GRID_HI)
    ax.set_ylim(GRID_LO, GRID_HI)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 72)
    log("H527 — Single-Example Ripple")
    log("=" * 72)

    # ── Data ─────────────────────────────────────────────────────────────────
    X_train, Y_train = make_4class(N_PER_CLASS, seed=SEED)
    X_test, Y_test = make_4class(N_TEST_PER_CLASS, seed=SEED + 1)
    Xt_test, Yt_test = to_tensors(X_test, Y_test)

    log(f"Train: {len(X_train)} pts  |  Test: {len(X_test)} pts")
    log(f"FGSM eps={EPS:.3f}  |  MLP epochs={EPOCHS}  |  device={DEVICE}")

    # ── Base model ────────────────────────────────────────────────────────────
    base_model = train_model(X_train, Y_train)
    base_acc = clean_acc(base_model, Xt_test, Yt_test)
    base_asr, _ = fgsm_asr(base_model, Xt_test.clone(), Yt_test)
    log(f"\nBase model  clean_acc={base_acc*100:.2f}%  FGSM_ASR={base_asr*100:.2f}%")

    # ── Scenario A: label-flipped poison near x=0 boundary ───────────────────
    log("\n─── Scenario A: label-flipped poison ───")
    # Point near the class-0/class-1 boundary: x≈0, y≈-1.5 (between the two
    # bottom-row classes), labelled as class 1 instead of class 0.
    poison_xy = [0.05, -1.5]
    poison_label = 1   # wrong label (true label would be 0)
    X_A, Y_A = augment(X_train, Y_train, poison_xy, poison_label)
    model_A = train_model(X_A, Y_A)
    m_A = scenario_metrics(base_model, model_A, Xt_test, Yt_test)
    log(f"  Added point {poison_xy} with label {poison_label} (true label: 0)")
    log(f"  clean_acc_delta={m_A['clean_acc_delta']:+.2f}%  "
        f"ASR_delta={m_A['asr_delta']:+.2f}%  "
        f"n_predictions_changed={m_A['n_predictions_changed']}  "
        f"boundary_shift={m_A['boundary_shift']:.4f}")

    # ── Scenarios B & C: grid search ─────────────────────────────────────────
    log(f"\n─── Scenarios B & C: grid search ({GRID_RES}×{GRID_RES} = "
        f"{GRID_RES**2} candidates) ───")
    log("  (this may take a minute …)")
    asr_delta_grid, worst_pt, best_pt = grid_search_candidates(
        X_train, Y_train, Xt_test, Yt_test, base_asr)

    # Worst-case (Scenario B)
    dists_worst = [np.hypot(worst_pt[0]-cx, worst_pt[1]-cy) for cx, cy in CLASS_CENTERS]
    label_worst = int(np.argmin(dists_worst))
    X_B, Y_B = augment(X_train, Y_train, worst_pt, label_worst)
    model_B = train_model(X_B, Y_B)
    m_B = scenario_metrics(base_model, model_B, Xt_test, Yt_test)

    log(f"\n  [B] Worst-case poison at {worst_pt.round(3)}, label={label_worst}")
    log(f"  clean_acc_delta={m_B['clean_acc_delta']:+.2f}%  "
        f"ASR_delta={m_B['asr_delta']:+.2f}%  "
        f"n_predictions_changed={m_B['n_predictions_changed']}  "
        f"boundary_shift={m_B['boundary_shift']:.4f}")

    # Best-case helpful (Scenario C)
    dists_best = [np.hypot(best_pt[0]-cx, best_pt[1]-cy) for cx, cy in CLASS_CENTERS]
    label_best = int(np.argmin(dists_best))
    X_C, Y_C = augment(X_train, Y_train, best_pt, label_best)
    model_C = train_model(X_C, Y_C)
    m_C = scenario_metrics(base_model, model_C, Xt_test, Yt_test)

    log(f"\n  [C] Best helpful point at {best_pt.round(3)}, label={label_best}")
    log(f"  clean_acc_delta={m_C['clean_acc_delta']:+.2f}%  "
        f"ASR_delta={m_C['asr_delta']:+.2f}%  "
        f"n_predictions_changed={m_C['n_predictions_changed']}  "
        f"boundary_shift={m_C['boundary_shift']:.4f}")

    # ── Scenario D: distance sweep ────────────────────────────────────────────
    log(f"\n─── Scenario D: distance sweep from boundary (x=0, y=-1.5) ───")
    dist_asr_changes = []
    dist_acc_changes = []
    for dist in DISTANCES:
        pt_D = [dist, -1.5]          # moves right of x=0 into class-1 territory
        label_D = 0                  # correctly labelled as class 0 (slightly wrong side)
        X_D, Y_D = augment(X_train, Y_train, pt_D, label_D)
        model_D = train_model(X_D, Y_D)
        m_D = scenario_metrics(base_model, model_D, Xt_test, Yt_test)
        dist_asr_changes.append(m_D["asr_delta"])
        dist_acc_changes.append(m_D["clean_acc_delta"])
        log(f"  dist={dist:.1f}  clean_acc_delta={m_D['clean_acc_delta']:+.2f}%  "
            f"ASR_delta={m_D['asr_delta']:+.2f}%  "
            f"n_changed={m_D['n_predictions_changed']}  "
            f"boundary_shift={m_D['boundary_shift']:.4f}")

    # ── PASS/FAIL ─────────────────────────────────────────────────────────────
    log("\n" + "─" * 72)
    worst_asr_increase = m_B["asr_delta"]
    worst_acc_drop = -m_B["clean_acc_delta"]   # positive if accuracy decreased
    passed = worst_asr_increase > 5.0 or worst_acc_drop > 2.0
    verdict = "PASS" if passed else "FAIL"
    log(f"PASS condition: worst-poison ASR increase > 5% OR acc drop > 2%")
    log(f"  worst_poison_asr_delta = {worst_asr_increase:+.2f}%")
    log(f"  worst_poison_acc_drop  = {worst_acc_drop:+.2f}%")
    log(f"VERDICT: {verdict}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("H527 — Single-Example Ripple: How Much Does One Training Point Move the Boundary?",
                 fontsize=11, fontweight="bold")

    # (0,0) Base model boundary
    plot_boundary(axes[0, 0], base_model, X_train, Y_train,
                  f"(0,0) Base model\nacc={base_acc*100:.1f}%  ASR={base_asr*100:.1f}%")

    # (0,1) After worst poison
    plot_boundary(axes[0, 1], model_B, X_train, Y_train,
                  f"(0,1) Worst poison point [B]\nASR Δ={m_B['asr_delta']:+.1f}%  "
                  f"n_changed={m_B['n_predictions_changed']}",
                  extra_point=worst_pt, extra_color="red", extra_marker="*",
                  extra_label="Poison (B)")

    # (0,2) After best helpful point
    plot_boundary(axes[0, 2], model_C, X_train, Y_train,
                  f"(0,2) Best helpful point [C]\nASR Δ={m_C['asr_delta']:+.1f}%  "
                  f"n_changed={m_C['n_predictions_changed']}",
                  extra_point=best_pt, extra_color="green", extra_marker="*",
                  extra_label="Helpful (C)")

    # (1,0) ASR delta heatmap
    ax = axes[1, 0]
    xs_g = np.linspace(GRID_LO, GRID_HI, GRID_RES)
    ys_g = np.linspace(GRID_LO, GRID_HI, GRID_RES)
    vmax = max(abs(asr_delta_grid).max(), 0.01)
    im = ax.imshow(asr_delta_grid, extent=[GRID_LO, GRID_HI, GRID_LO, GRID_HI],
                   origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    fig.colorbar(im, ax=ax, label="ASR delta (%)")
    ax.scatter(*worst_pt, c="red", marker="*", s=200, zorder=10,
               edgecolors="k", linewidths=0.7, label="Worst (B)")
    ax.scatter(*best_pt, c="green", marker="*", s=200, zorder=10,
               edgecolors="k", linewidths=0.7, label="Best (C)")
    ax.scatter(*poison_xy, c="orange", marker="P", s=120, zorder=10,
               edgecolors="k", linewidths=0.7, label="Poison A")
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("(1,0) Heatmap: ASR delta over candidate grid", fontsize=9)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)

    # (1,1) Scenario D: distance vs ASR change
    ax = axes[1, 1]
    ax.plot(DISTANCES, dist_asr_changes, "o-", color="steelblue", linewidth=1.8,
            markersize=6, label="ASR delta")
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(DISTANCES, dist_acc_changes, "s--", color="firebrick", linewidth=1.5,
             markersize=5, label="Acc delta")
    ax2.set_ylabel("Clean acc delta (%)", fontsize=8, color="firebrick")
    ax2.tick_params(labelsize=7, colors="firebrick")
    ax.set_xlabel("Distance from boundary (x=0)", fontsize=8)
    ax.set_ylabel("FGSM ASR delta (%)", fontsize=8, color="steelblue")
    ax.tick_params(labelsize=7, colors="steelblue")
    ax.set_title("(1,1) Scenario D: distance vs ASR/acc change", fontsize=9)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    # (1,2) Bar chart — n_predictions_changed per scenario
    ax = axes[1, 2]
    scenario_labels = ["A\n(flip-poison)", "B\n(worst)", "C\n(helpful)"]
    n_changed_vals = [m_A["n_predictions_changed"],
                      m_B["n_predictions_changed"],
                      m_C["n_predictions_changed"]]
    bar_colors = ["orange", "red", "green"]
    bars = ax.bar(scenario_labels, n_changed_vals, color=bar_colors, alpha=0.75,
                  edgecolor="k", linewidth=0.7)
    for bar, val in zip(bars, n_changed_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# test predictions changed", fontsize=8)
    ax.set_title("(1,2) Predictions changed by one added point", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, max(n_changed_vals) * 1.2 + 1)

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"\nFigure saved → {FIG_PATH}")

    # ── Write output file ─────────────────────────────────────────────────────
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Output saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
