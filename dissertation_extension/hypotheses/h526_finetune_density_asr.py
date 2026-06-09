"""
H526 - Fine-tune density vs ASR: radial/polar orthogonal boundary transfer.

Design:
  Feature space: polar coordinates (r, theta/pi) — all data lives on rings.

  Source task (pretrain): radial — inner ring (r~1.0) vs outer ring (r~2.2).
    Boundary at r=1.6.  The relevant feature is r; theta is irrelevant.
    N=600/class, 600 epochs.

  Target task (fine-tune): angular — left half (theta~-pi/2) vs right half (theta~+pi/2).
    Both classes share the same radial distribution (r~N(1.6, 0.5)).
    Boundary at theta=0.  The relevant feature is theta/pi; r is irrelevant.
    A pretrained radial model predicts ~50% on this task.

  Fine-tune density control (angular spread in theta):
    Dense:  theta std = 0.2  -> tight clusters, strong signal, ceiling ~82%
    Sparse: theta std = 1.0  -> wide overlap near theta=0, ceiling ~70%

  N_per_class sweep: [5, 10, 20, 50, 100, 200, 400]

  4 conditions: dense, sparse, AT-dense, AT-sparse
  FGSM adversarial training during fine-tune: eps=0.2

PASS criterion (at N=20):
  dense_acc > sparse_acc + 0.03
  OR dense_FGSM_ASR < sparse_FGSM_ASR - 0.03

Figure (2x4):
  (0,0) Source model decision boundary in (r, theta/pi) — vertical stripe at r~1.6.
        Overlay angular task test data. Shows mismatch: model uses r, task needs theta.
  (0,1) Dense FT N=50 boundary — rotates toward horizontal (theta-based).
  (0,2) Sparse FT N=50 — partially rotated, noisier boundary.
  (0,3) N_finetune vs clean accuracy (4 conditions). Dashed = source-on-target baseline.
  (1,0) N_finetune vs FGSM ASR (4 conditions).
  (1,1) Dense FT N=200 boundary.
  (1,2) Sparse FT N=200 boundary.
  (1,3) Bar chart FGSM ASR at N=20 for all 4 conditions.

Output paths:
  results/fashion_mnist/h526_finetune_density_asr.png
  results/fashion_mnist/h526_finetune_density_asr_output.txt  (Tee stdout)
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Tee: mirror stdout to file
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "w")
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # delegate attribute lookups to the real stdout so nothing breaks
    def __getattr__(self, name):
        return getattr(self._stdout, name)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
PRETRAIN_N = 600          # per class
PRETRAIN_EPOCHS = 600
FINETUNE_EPOCHS = 300
LR = 1e-3
FGSM_EPS = 0.2
N_SWEEP = [5, 10, 20, 50, 100, 200, 400]
OUT_DIR = "results/fashion_mnist"
FIG_PATH = os.path.join(OUT_DIR, "h526_finetune_density_asr.png")
TXT_PATH = os.path.join(OUT_DIR, "h526_finetune_density_asr_output.txt")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def make_polar_dataset(n, task="radial", spread=0.6, seed=42):
    """
    Generate n samples PER CLASS in polar feature space (r, theta/pi).

    task="radial":
        Class 0: inner ring  r~N(1.0, 0.30), theta ~ Uniform(-pi, pi)
        Class 1: outer ring  r~N(2.2, 0.30), theta ~ Uniform(-pi, pi)
        Boundary at r=1.6.

    task="angular":
        Class 0: left half   theta ~ N(-pi/2, spread), r~N(1.6, 0.50)
        Class 1: right half  theta ~ N(+pi/2, spread), r~N(1.6, 0.50)
        Boundary at theta=0.
        `spread` controls how much the classes overlap near theta=0.

    Features returned: (r, theta/pi) — theta normalised to [-1, 1].
    """
    rng = np.random.default_rng(seed)

    if task == "radial":
        r0 = rng.normal(1.0, 0.30, n).astype(np.float32)
        r1 = rng.normal(2.2, 0.30, n).astype(np.float32)
        t0 = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
        t1 = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    else:  # angular
        r0 = rng.normal(1.6, 0.50, n).astype(np.float32)
        r1 = rng.normal(1.6, 0.50, n).astype(np.float32)
        t0 = rng.normal(-np.pi / 2, spread, n).astype(np.float32)
        t1 = rng.normal(+np.pi / 2, spread, n).astype(np.float32)

    X0 = np.stack([r0, t0 / np.pi], axis=1)
    X1 = np.stack([r1, t1 / np.pi], axis=1)
    X = np.vstack([X0, X1]).astype(np.float32)
    Y = np.array([0] * n + [1] * n, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


def to_tensors(X, Y):
    return (torch.tensor(X, dtype=torch.float32).to(DEVICE),
            torch.tensor(Y, dtype=torch.long).to(DEVICE))


# ---------------------------------------------------------------------------
# Model: 2->32->32->2
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def fgsm(model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
         eps: float = FGSM_EPS) -> torch.Tensor:
    model.eval()
    x_adv = X.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_adv), Y).backward()
    return (X + eps * x_adv.grad.sign()).detach()


def compute_asr(model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    """FGSM attack success rate on correctly-classified examples."""
    model.eval()
    with torch.no_grad():
        correct = model(X).argmax(1) == Y
    if correct.sum() == 0:
        return 0.0
    Xc, Yc = X[correct], Y[correct]
    x_adv = fgsm(model, Xc, Yc)
    with torch.no_grad():
        return (model(x_adv).argmax(1) != Yc).float().mean().item()


def compute_acc(model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == Y).float().mean().item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
                epochs: int, lr: float = LR, at: bool = False) -> nn.Module:
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        if at:
            x_adv = fgsm(model, X, Y)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(x_adv), Y).backward()
        else:
            F.cross_entropy(model(X), Y).backward()
        opt.step()
    return model


def finetune(pretrained: nn.Module, X: torch.Tensor, Y: torch.Tensor,
             epochs: int = FINETUNE_EPOCHS, lr: float = LR,
             at: bool = False) -> nn.Module:
    m = copy.deepcopy(pretrained)
    return train_model(m, X, Y, epochs=epochs, lr=lr, at=at)


# ---------------------------------------------------------------------------
# Decision boundary plot in (r, theta/pi) space
# ---------------------------------------------------------------------------
def plot_boundary(ax, model: nn.Module, X: np.ndarray, Y: np.ndarray,
                  title: str,
                  xlim=(0.0, 3.5), ylim=(-1.05, 1.05)):
    """
    xlim covers r range, ylim covers theta/pi range.
    The angular boundary should appear as a horizontal line at theta/pi=0.
    The radial boundary should appear as a vertical line at r=1.6.
    """
    rr = np.linspace(xlim[0], xlim[1], 300)
    tt = np.linspace(ylim[0], ylim[1], 300)
    xx, yy = np.meshgrid(rr, tt)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                        dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        zz = model(grid).argmax(1).cpu().numpy().reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                colors=["#aec6cf", "#f4a460"], alpha=0.35)
    for ci, (col, mk) in enumerate(zip(["#1f4e79", "#7b2d00"], ["o", "s"])):
        mask = Y == ci
        ax.scatter(X[mask, 0], X[mask, 1], s=14, alpha=0.65,
                   color=col, marker=mk, label=f"class {ci}", linewidths=0)
    # reference lines
    ax.axvline(1.6, color="gray", ls=":", lw=1, alpha=0.6, label="r=1.6")
    ax.axhline(0.0, color="silver", ls="--", lw=1, alpha=0.6, label="θ=0")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("r", fontsize=8)
    ax.set_ylabel("θ/π", fontsize=8)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tee = Tee(TXT_PATH)
    sys.stdout = tee

    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        print("=" * 70)
        print("H526 – Polar/Radial Orthogonal Transfer: Density vs ASR")
        print("=" * 70)

        # ------------------------------------------------------------------
        # 1. Pretrain source model (radial task)
        # ------------------------------------------------------------------
        print("\n[1] Pretraining source model (radial task, N=600/class, 600 epochs)...")
        Xsrc, Ysrc = make_polar_dataset(PRETRAIN_N, task="radial", seed=SEED)
        Xsrc_t, Ysrc_t = to_tensors(Xsrc, Ysrc)

        torch.manual_seed(SEED)
        src_model = MLP().to(DEVICE)
        src_model = train_model(src_model, Xsrc_t, Ysrc_t,
                                epochs=PRETRAIN_EPOCHS, lr=LR, at=False)
        src_train_acc = compute_acc(src_model, Xsrc_t, Ysrc_t)
        print(f"   Source train acc (radial): {src_train_acc:.3f}  (expect ~0.90)")

        # ------------------------------------------------------------------
        # 2. Evaluate source model on angular task
        # ------------------------------------------------------------------
        print("\n[2] Source model evaluated on angular task (dense, N=400 test)...")
        # Use dense spread for the test set so results are consistent
        Xtgt_eval, Ytgt_eval = make_polar_dataset(400, task="angular",
                                                   spread=0.2, seed=SEED + 1)
        Xtgt_t, Ytgt_t = to_tensors(Xtgt_eval, Ytgt_eval)
        src_on_tgt = compute_acc(src_model, Xtgt_t, Ytgt_t)
        print(f"   Source model acc on angular task: {src_on_tgt:.3f}  (expect ~0.50)")

        # ------------------------------------------------------------------
        # 3. Fine-tune sweep
        # ------------------------------------------------------------------
        print("\n[3] Fine-tune sweep...")
        print(f"    N sweep: {N_SWEEP}")
        print(f"    Dense:  angular theta spread=0.2 (tight clusters)")
        print(f"    Sparse: angular theta spread=1.0 (heavy overlap near theta=0)")
        print(f"    AT: FGSM eps={FGSM_EPS}, {FINETUNE_EPOCHS} fine-tune epochs")

        # Conditions: (spread, at, random_init)
        # rand-dense: same data as dense but NO pretrained weights (random init)
        # This separates pretrain transfer benefit from data-density effect.
        conditions = {
            "dense":      dict(spread=0.2, at=False, random_init=False),
            "sparse":     dict(spread=1.0, at=False, random_init=False),
            "AT-dense":   dict(spread=0.2, at=True,  random_init=False),
            "rand-dense": dict(spread=0.2, at=False, random_init=True),
        }
        results = {c: {"clean_acc": [], "fgsm_asr": []}
                   for c in conditions}

        for N in N_SWEEP:
            print(f"\n  N={N}:")
            for cname, cp in conditions.items():
                Xft, Yft = make_polar_dataset(N, task="angular",
                                              spread=cp["spread"],
                                              seed=SEED + N + 100)
                Xft_t, Yft_t = to_tensors(Xft, Yft)
                # Evaluation set matched to condition spread
                Xeval, Yeval = make_polar_dataset(400, task="angular",
                                                  spread=cp["spread"],
                                                  seed=SEED + N + 200)
                Xeval_t, Yeval_t = to_tensors(Xeval, Yeval)
                if cp.get("random_init", False):
                    # Train from scratch — no pretrained weights
                    torch.manual_seed(SEED + N + 77)
                    rand_model = MLP().to(DEVICE)
                    ft = train_model(rand_model, Xft_t, Yft_t,
                                     epochs=FINETUNE_EPOCHS, lr=LR, at=False)
                else:
                    ft = finetune(src_model, Xft_t, Yft_t,
                                  epochs=FINETUNE_EPOCHS, lr=LR, at=cp["at"])
                acc  = compute_acc(ft, Xeval_t, Yeval_t)
                fasr = compute_asr(ft, Xeval_t, Yeval_t)
                results[cname]["clean_acc"].append(acc)
                results[cname]["fgsm_asr"].append(fasr)
                print(f"    {cname:12s}  acc={acc:.3f}  FGSM_ASR={fasr:.3f}")

        # ------------------------------------------------------------------
        # 4. PASS criterion
        # ------------------------------------------------------------------
        print("\n[4] PASS criterion (at N=20):")
        i20 = N_SWEEP.index(20)
        d_acc  = results["dense"]["clean_acc"][i20]
        s_acc  = results["sparse"]["clean_acc"][i20]
        d_fasr = results["dense"]["fgsm_asr"][i20]
        s_fasr = results["sparse"]["fgsm_asr"][i20]
        crit_acc = d_acc  > s_acc  + 0.03
        crit_asr = d_fasr < s_fasr - 0.03
        print(f"   dense  clean_acc={d_acc:.3f}  FGSM_ASR={d_fasr:.3f}")
        print(f"   sparse clean_acc={s_acc:.3f}  FGSM_ASR={s_fasr:.3f}")
        print(f"   Acc gap  dense-sparse = {d_acc - s_acc:+.3f}  "
              f"(need >+0.03): {'OK' if crit_acc else 'FAIL'}")
        print(f"   ASR gap  sparse-dense = {s_fasr - d_fasr:+.3f}  "
              f"(need >+0.03): {'OK' if crit_asr else 'FAIL'}")
        passed = crit_acc or crit_asr
        # Also print rand-dense vs dense comparison (pretrain benefit)
        r_acc  = results["rand-dense"]["clean_acc"][i20]
        r_fasr = results["rand-dense"]["fgsm_asr"][i20]
        print(f"   rand-dense clean_acc={r_acc:.3f}  FGSM_ASR={r_fasr:.3f}")
        print(f"   Pretrain benefit (dense vs rand-dense) acc  = {d_acc - r_acc:+.3f}")
        print(f"   Pretrain benefit (dense vs rand-dense) ASR  = {r_fasr - d_fasr:+.3f}")
        print(f"\n   => OVERALL: {'PASS' if passed else 'FAIL'}")

        # ------------------------------------------------------------------
        # 5. Build boundary-panel models (dense spread for visual clarity)
        # ------------------------------------------------------------------
        def build_ft(n, spread, random_init=False):
            Xft, Yft = make_polar_dataset(n, task="angular",
                                          spread=spread, seed=SEED + n + 100)
            Xft_t, Yft_t = to_tensors(Xft, Yft)
            if random_init:
                torch.manual_seed(SEED + n + 77)
                m = MLP().to(DEVICE)
                return train_model(m, Xft_t, Yft_t, epochs=FINETUNE_EPOCHS, lr=LR), Xft, Yft
            return (finetune(src_model, Xft_t, Yft_t,
                             epochs=FINETUNE_EPOCHS, lr=LR),
                    Xft, Yft)

        ft_d50,  Xd50,  Yd50  = build_ft(50,  0.2)
        ft_s50,  Xs50,  Ys50  = build_ft(50,  1.0)
        ft_d200, Xd200, Yd200 = build_ft(200, 0.2)
        ft_r50,  Xr50,  Yr50  = build_ft(50,  0.2, random_init=True)

        # ------------------------------------------------------------------
        # 6. Figure (2 rows x 4 cols)
        # ------------------------------------------------------------------
        print("\n[5] Saving figure...")
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle(
            "H526: Polar/Radial Orthogonal Transfer — Density vs ASR",
            fontsize=13, fontweight="bold"
        )

        # --- Row 0 ---
        # (0,0) Source model boundary with angular test data overlaid
        plot_boundary(axes[0, 0], src_model, Xtgt_eval, Ytgt_eval,
                      "Source model (radial boundary)\noverlaid with angular task data")

        # (0,1) Dense FT N=50
        plot_boundary(axes[0, 1], ft_d50, Xd50, Yd50,
                      "Dense FT N=50 (spread=0.2)\nboundary rotates toward θ-axis")

        # (0,2) Sparse FT N=50
        plot_boundary(axes[0, 2], ft_s50, Xs50, Ys50,
                      "Sparse FT N=50 (spread=1.0)\npartially rotated, noisy boundary")

        # (0,3) N vs clean accuracy
        col_map = {"dense": "#1f4e79", "sparse": "#7b2d00",
                   "AT-dense": "#2e86ab", "rand-dense": "#6a0dad"}
        ls_map  = {"dense": "o-", "sparse": "s--",
                   "AT-dense": "^-.", "rand-dense": "D:"}
        ax = axes[0, 3]
        for cname in conditions:
            ax.plot(N_SWEEP, results[cname]["clean_acc"],
                    ls_map[cname], color=col_map[cname],
                    label=cname, linewidth=1.5)
        ax.axhline(src_on_tgt, color="gray", ls=":", lw=1.2,
                   label=f"source baseline ({src_on_tgt:.2f})")
        ax.set_xlabel("N per class", fontsize=9)
        ax.set_ylabel("Clean accuracy", fontsize=9)
        ax.set_title("N_finetune vs Clean Accuracy", fontsize=9)
        ax.set_xscale("log")
        ax.legend(fontsize=7.5)
        ax.set_ylim(0.35, 1.05)
        ax.tick_params(labelsize=7)

        # --- Row 1 ---
        # (1,0) N vs FGSM ASR
        ax = axes[1, 0]
        for cname in conditions:
            ax.plot(N_SWEEP, results[cname]["fgsm_asr"],
                    ls_map[cname], color=col_map[cname],
                    label=cname, linewidth=1.5)
        ax.set_xlabel("N per class", fontsize=9)
        ax.set_ylabel("FGSM ASR", fontsize=9)
        ax.set_title("N_finetune vs FGSM ASR", fontsize=9)
        ax.set_xscale("log")
        ax.legend(fontsize=7.5)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)

        # (1,1) Dense FT N=200 (pretrained)
        plot_boundary(axes[1, 1], ft_d200, Xd200, Yd200,
                      "Dense FT N=200 (pretrained)\nnear-correct θ boundary")

        # (1,2) Rand-dense N=50 — same data, no pretrained weights
        plot_boundary(axes[1, 2], ft_r50, Xr50, Yr50,
                      "Rand-init dense N=50 (no pretrain)\nbaseline: no transfer benefit")

        # (1,3) Bar chart: FGSM ASR at N=20
        ax = axes[1, 3]
        i20_bar = N_SWEEP.index(20)
        bar_vals   = [results[c]["fgsm_asr"][i20_bar] for c in conditions]
        bar_colors = [col_map[c] for c in conditions]
        bars = ax.bar(list(conditions.keys()), bar_vals, color=bar_colors,
                      alpha=0.8, edgecolor="black", linewidth=0.8)
        ax.set_ylabel("FGSM ASR", fontsize=9)
        ax.set_title("FGSM ASR at N=20 (all conditions)", fontsize=9)
        ax.set_ylim(0, 1.15)
        for bar, v in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)

        plt.tight_layout()
        plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Figure: {FIG_PATH}")
        print(f"   Text:   {TXT_PATH}")
        print("\nDone.")

    finally:
        sys.stdout = tee._stdout
        tee.close()


if __name__ == "__main__":
    main()
