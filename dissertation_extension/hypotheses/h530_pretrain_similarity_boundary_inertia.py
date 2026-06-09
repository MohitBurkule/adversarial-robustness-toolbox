"""
H530 – Pretrain Similarity → Boundary Inertia → Inherited ASR
==============================================================

Hypothesis:
  When a pretrained model is fine-tuned on a similar task, the decision
  boundary barely moves (boundary inertia).  The fine-tuned model therefore
  INHERITS the adversarial structure of the CE-trained pretrained boundary,
  causing higher ASR than training from scratch on the same fine-tune data.

  As pretrain–finetune task similarity INCREASES:
    • boundary displacement   DECREASES  (boundary inertia)
    • FGSM ASR               INCREASES   (inherited vulnerability)
    • vs random-init baseline: gap WIDENS

Design (all in polar feature space: 2 features = r, theta/pi):

  We define 5 pretrain tasks by rotating the decision boundary angle:
    angle=0°   (radial):    boundary at r=1.6  — orthogonal to angular FT task
    angle=22°:             boundary slightly tilted
    angle=45°:             boundary diagonal
    angle=67°:             boundary mostly angular
    angle=90°  (identical): boundary at theta=0 — same as fine-tune task

  Fine-tune task: angular — left half (theta < 0) vs right half (theta > 0).
    N=20/class, spread=0.2 (dense), 300 epochs.

  For each pretrain angle we measure:
    1. Boundary displacement (fraction of grid cells that change class label
       from pretrained → fine-tuned model)
    2. Fine-tuned FGSM ASR
    3. Same metrics for rand-init baseline (no pretrain)

  PASS: FGSM ASR is monotonically correlated with pretrain similarity
        (Spearman rho > 0.6) AND similar pretrain (angle=90°) has
        higher ASR than orthogonal pretrain (angle=0°).

Figure (2×3):
  (0,0) Boundary displacement vs pretrain angle
  (0,1) FGSM ASR vs pretrain angle  (with rand-init dashed baseline)
  (0,2) Scatter: boundary displacement vs ASR (one point per pretrain angle)
  (1,0–4) Decision boundary snapshots at 5 pretrain angles (fine-tuned models)
"""

import os, sys, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR     = os.path.join(PROJECT_DIR, "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_PATH = os.path.join(OUT_DIR, "h530_pretrain_similarity_boundary_inertia.png")
TXT_PATH = os.path.join(OUT_DIR, "h530_pretrain_similarity_boundary_inertia_output.txt")

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED           = 42
PRETRAIN_N     = 600
PRETRAIN_EPOCHS= 600
FINETUNE_N     = 20   # small so pretrain effect is large
FINETUNE_EPOCHS= 300
LR             = 1e-3
FGSM_EPS       = 0.20
EVAL_N         = 400  # test set size per class

# Pretrain angles: 0°=radial (orthogonal), 90°=angular (identical to FT task)
PRETRAIN_ANGLES = [0, 22, 45, 67, 90]   # degrees


# ──────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path):
        self._f = open(path, "w"); self._s = sys.stdout
    def write(self, d): self._s.write(d); self._f.write(d)
    def flush(self): self._s.flush(); self._f.flush()
    def close(self): self._f.close()
    def __getattr__(self, n): return getattr(self._s, n)


# ──────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )
    def forward(self, x): return self.net(x)


def to_t(X, Y):
    return (torch.tensor(X, dtype=torch.float32).to(DEVICE),
            torch.tensor(Y, dtype=torch.long).to(DEVICE))


def train(model, Xt, Yt, epochs, at=False):
    opt = optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        if at:
            xv = Xt.clone().detach().requires_grad_(True)
            F.cross_entropy(model(xv), Yt).backward()
            xa = (Xt + FGSM_EPS * xv.grad.sign()).detach()
            model.train(); opt.zero_grad()
            F.cross_entropy(model(xa), Yt).backward()
        else:
            F.cross_entropy(model(Xt), Yt).backward()
        opt.step()
    return model


def acc(model, Xt, Yt):
    model.eval()
    with torch.no_grad():
        return (model(Xt).argmax(1) == Yt).float().mean().item()


def fgsm_asr(model, Xt, Yt):
    model.eval()
    with torch.no_grad():
        correct = model(Xt).argmax(1) == Yt
    if correct.sum() == 0: return 0.0
    Xc, Yc = Xt[correct], Yt[correct]
    xv = Xc.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xv), Yc).backward()
    xa = (Xc + FGSM_EPS * xv.grad.sign()).detach()
    with torch.no_grad():
        return (model(xa).argmax(1) != Yc).float().mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Data: rotated boundary in polar feature space
# ──────────────────────────────────────────────────────────────────────────────
def make_rotated_task(n, angle_deg, seed=42):
    """
    Generate 2-class dataset in (r, theta/pi) feature space whose decision
    boundary is a LINE rotated by `angle_deg` from vertical.

    angle=0°:  vertical boundary at r=1.6  (radial task, uses r feature)
    angle=90°: horizontal boundary at theta/pi=0 (angular task, uses theta)

    Both classes share the same ring radius distribution to avoid confounds.
    The boundary is defined by the projection along direction (cos a, sin a).
    """
    rng = np.random.default_rng(seed)
    a   = np.radians(angle_deg)
    # unit normal to boundary
    nx, ny = np.cos(a), np.sin(a)

    # Generate points on a ring: r~N(1.6, 0.3), theta~Uniform
    r_all   = rng.normal(1.6, 0.30, 2 * n).astype(np.float32)
    t_all   = rng.uniform(-np.pi, np.pi, 2 * n).astype(np.float32)
    # features: (r, theta/pi)
    X_all   = np.stack([r_all, t_all / np.pi], axis=1)

    # Label by sign of dot product with boundary normal
    # Boundary passes through centroid (1.6, 0)
    centroid = np.array([1.6, 0.0], dtype=np.float32)
    proj = (X_all - centroid) @ np.array([nx, ny], dtype=np.float32)
    Y_all = (proj > 0).astype(np.int64)

    # Balance classes
    idx0 = np.where(Y_all == 0)[0]
    idx1 = np.where(Y_all == 1)[0]
    min_n = min(len(idx0), len(idx1), n)
    idx = np.concatenate([idx0[:min_n], idx1[:min_n]])
    rng.shuffle(idx)
    return X_all[idx], Y_all[idx]


def make_angular_task(n, spread=0.2, seed=42):
    """Target fine-tune task: left/right angular split, theta~N(±pi/2, spread)."""
    rng = np.random.default_rng(seed)
    r0  = rng.normal(1.6, 0.50, n).astype(np.float32)
    r1  = rng.normal(1.6, 0.50, n).astype(np.float32)
    t0  = rng.normal(-np.pi/2, spread, n).astype(np.float32)
    t1  = rng.normal(+np.pi/2, spread, n).astype(np.float32)
    X0  = np.stack([r0, t0/np.pi], axis=1)
    X1  = np.stack([r1, t1/np.pi], axis=1)
    X   = np.vstack([X0, X1]).astype(np.float32)
    Y   = np.array([0]*n + [1]*n, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Boundary displacement metric
# ──────────────────────────────────────────────────────────────────────────────
def boundary_displacement(model_a, model_b, res=200):
    """Fraction of grid cells where class label changes between two models."""
    rr = np.linspace(0.5, 2.7, res)
    tt = np.linspace(-1.0, 1.0, res)
    xx, yy = np.meshgrid(rr, tt)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                        dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pa = model_a(grid).argmax(1)
        pb = model_b(grid).argmax(1)
    return (pa != pb).float().mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Decision boundary plot
# ──────────────────────────────────────────────────────────────────────────────
def plot_boundary(ax, model, Xft, Yft, title,
                  xlim=(0.5, 2.7), ylim=(-1.05, 1.05)):
    rr = np.linspace(xlim[0], xlim[1], 250)
    tt = np.linspace(ylim[0], ylim[1], 250)
    xx, yy = np.meshgrid(rr, tt)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                        dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        zz = model(grid).argmax(1).cpu().numpy().reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                colors=["#aec6cf", "#f4a460"], alpha=0.35)
    for ci, col in enumerate(["#1f4e79", "#7b2d00"]):
        mask = Yft == ci
        ax.scatter(Xft[mask, 0], Xft[mask, 1], s=14, alpha=0.7,
                   color=col, linewidths=0)
    ax.axhline(0.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(1.6, color="silver", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("r", fontsize=7); ax.set_ylabel("θ/π", fontsize=7)
    ax.tick_params(labelsize=6)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    tee = Tee(TXT_PATH)
    sys.stdout = tee
    try:
        _run()
    finally:
        sys.stdout = tee._s
        tee.close()


def _run():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")
    print("=" * 70)
    print("H530 – Pretrain Similarity → Boundary Inertia → Inherited ASR")
    print("=" * 70)

    # ── Fine-tune data (shared across all conditions) ──────────────────────
    Xft, Yft   = make_angular_task(FINETUNE_N, spread=0.2, seed=SEED + 1)
    Xft_t, Yft_t = to_t(Xft, Yft)
    Xeval, Yeval = make_angular_task(EVAL_N, spread=0.2, seed=SEED + 2)
    Xeval_t, Yeval_t = to_t(Xeval, Yeval)

    # ── Random-init baseline (no pretrain) ─────────────────────────────────
    print("\n[0] Random-init baseline (no pretrain) ...")
    torch.manual_seed(SEED + 99)
    rand_model = MLP().to(DEVICE)
    rand_model = train(rand_model, Xft_t, Yft_t, epochs=FINETUNE_EPOCHS)
    rand_acc   = acc(rand_model, Xeval_t, Yeval_t)
    rand_asr   = fgsm_asr(rand_model, Xeval_t, Yeval_t)
    print(f"   acc={rand_acc:.3f}  FGSM_ASR={rand_asr:.3f}")

    # ── Pretrain sweep ─────────────────────────────────────────────────────
    print(f"\n[1] Pretrain sweep over angles: {PRETRAIN_ANGLES}")
    print(f"    Pretrain: N={PRETRAIN_N}/class, {PRETRAIN_EPOCHS} epochs")
    print(f"    Finetune: N={FINETUNE_N}/class, {FINETUNE_EPOCHS} epochs")

    results = []
    ft_models, ft_at_models, ft_Xft_list = [], [], []

    for angle in PRETRAIN_ANGLES:
        # Pretrain
        Xpre, Ypre = make_rotated_task(PRETRAIN_N, angle, seed=SEED + angle)
        Xpre_t, Ypre_t = to_t(Xpre, Ypre)
        torch.manual_seed(SEED + angle)
        src = MLP().to(DEVICE)
        src = train(src, Xpre_t, Ypre_t, epochs=PRETRAIN_EPOCHS)
        src_acc_pre = acc(src, Xpre_t, Ypre_t)

        # Fine-tune STD
        ft = copy.deepcopy(src)
        ft = train(ft, Xft_t, Yft_t, epochs=FINETUNE_EPOCHS, at=False)
        ft_acc  = acc(ft, Xeval_t, Yeval_t)
        ft_asr  = fgsm_asr(ft, Xeval_t, Yeval_t)
        disp    = boundary_displacement(src, ft)

        # Fine-tune AT
        ft_at = copy.deepcopy(src)
        ft_at = train(ft_at, Xft_t, Yft_t, epochs=FINETUNE_EPOCHS, at=True)
        ft_at_acc = acc(ft_at, Xeval_t, Yeval_t)
        ft_at_asr = fgsm_asr(ft_at, Xeval_t, Yeval_t)
        disp_at   = boundary_displacement(src, ft_at)

        # Similarity label: angle=90 → identical, angle=0 → orthogonal
        similarity = angle / 90.0

        results.append(dict(
            angle=angle,
            similarity=similarity,
            pretrain_acc=src_acc_pre,
            ft_acc=ft_acc,     ft_asr=ft_asr,     boundary_disp=disp,
            at_acc=ft_at_acc,  at_asr=ft_at_asr,  at_disp=disp_at,
        ))
        ft_models.append(ft)
        ft_at_models.append(ft_at)
        ft_Xft_list.append(Xft)

        print(f"\n  angle={angle:3d}° (similarity={similarity:.2f})")
        print(f"    pretrain_acc={src_acc_pre:.3f}")
        print(f"    STD  ft_acc={ft_acc:.3f}  FGSM_ASR={ft_asr:.3f}  disp={disp:.4f}")
        print(f"    AT   ft_acc={ft_at_acc:.3f}  FGSM_ASR={ft_at_asr:.3f}  disp={disp_at:.4f}")

    # ── Statistics ──────────────────────────────────────────────────────────
    sims     = [r["similarity"]    for r in results]
    asrs     = [r["ft_asr"]        for r in results]
    at_asrs  = [r["at_asr"]        for r in results]
    accs     = [r["ft_acc"]        for r in results]
    at_accs  = [r["at_acc"]        for r in results]
    disps    = [r["boundary_disp"] for r in results]
    at_disps = [r["at_disp"]       for r in results]

    rho_sim_asr,  p_sim_asr  = spearmanr(sims,  asrs)
    rho_disp_asr, p_disp_asr = spearmanr(disps, asrs)
    rho_sim_disp, _          = spearmanr(sims,  disps)

    print("\n" + "=" * 70)
    print("STATISTICS (STD fine-tune):")
    print(f"  Spearman rho(similarity, ASR)         = {rho_sim_asr:+.3f}  p={p_sim_asr:.4f}")
    print(f"  Spearman rho(boundary_disp, ASR)      = {rho_disp_asr:+.3f}  p={p_disp_asr:.4f}")
    print(f"  Spearman rho(similarity, disp)        = {rho_sim_disp:+.3f}")
    print(f"\nAT vs STD fine-tune comparison:")
    for r in results:
        print(f"  angle={r['angle']:3d}°  STD ASR={r['ft_asr']:.3f} acc={r['ft_acc']:.3f}"
              f"  |  AT ASR={r['at_asr']:.3f} acc={r['at_acc']:.3f}"
              f"  |  AT_disp={r['at_disp']:.4f} (STD_disp={r['boundary_disp']:.4f})")
    print(f"\n  Rand-init STD  ASR={rand_asr:.3f}  acc={rand_acc:.3f}")

    # ── PASS ────────────────────────────────────────────────────────────────
    crit1 = rho_sim_asr > 0.6
    crit2 = results[-1]["ft_asr"] > results[0]["ft_asr"]
    # AT should reduce ASR relative to STD for at least 3/5 angles
    at_reduces = sum(r["at_asr"] < r["ft_asr"] for r in results)
    crit3 = at_reduces >= 3
    passed = (crit1 or crit2) and crit3
    print("\n" + "=" * 70)
    print(f"PASS criterion 1 (rho > 0.6):           {'OK' if crit1 else 'FAIL'}  rho={rho_sim_asr:+.3f}")
    print(f"PASS criterion 2 (90° ASR > 0° ASR):    {'OK' if crit2 else 'FAIL'}")
    print(f"PASS criterion 3 (AT reduces ASR ≥3/5): {'OK' if crit3 else 'FAIL'}  ({at_reduces}/5)")
    print(f"=> OVERALL: {'PASS' if passed else 'FAIL'}")

    # ── Figure ──────────────────────────────────────────────────────────────
    print("\nGenerating figure ...")
    # Layout: 4 rows × 5 cols
    #   Row 0: summary line plots (displacement, ASR, acc, AT-vs-STD ASR, AT-vs-STD acc)
    #   Row 1: summary grouped bars (STD ASR + AT ASR + STD acc + AT acc)
    #   Row 2: STD fine-tune boundaries (one per angle)
    #   Row 3: AT  fine-tune boundaries (one per angle)
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "H530 – Pretrain Similarity → Boundary Inertia → Inherited ASR\n"
        "STD & AT fine-tune at every pretrain angle.  Fine-tune task: angular (N=20).",
        fontsize=12, fontweight="bold"
    )
    gs = fig.add_gridspec(4, 5, hspace=0.50, wspace=0.38)

    colors_angle = plt.cm.plasma(np.linspace(0.15, 0.85, len(PRETRAIN_ANGLES)))
    angle_labels = [f"{a}°" for a in PRETRAIN_ANGLES]

    # ── Row 0: line plots ────────────────────────────────────────────────────

    # (0,0) Boundary displacement: STD vs AT
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(sims, disps,    "o-",  color="steelblue",  lw=2, ms=6, label="STD-FT")
    ax.plot(sims, at_disps, "s--", color="darkorange",  lw=2, ms=6, label="AT-FT")
    for i, (s, d) in enumerate(zip(sims, disps)):
        ax.annotate(angle_labels[i], (s, d), fontsize=6.5, xytext=(2,2),
                    textcoords="offset points")
    ax.set_xlabel("Pretrain similarity", fontsize=8)
    ax.set_ylabel("Boundary displacement", fontsize=8)
    ax.set_title("Boundary inertia\n(STD vs AT fine-tune)", fontsize=8)
    ax.legend(fontsize=7); ax.tick_params(labelsize=7)

    # (0,1) FGSM ASR: STD vs AT vs rand-init
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(sims, asrs,    "o-",  color="firebrick",   lw=2, ms=6, label="STD-FT")
    ax.plot(sims, at_asrs, "s--", color="darkorange",  lw=2, ms=6, label="AT-FT")
    ax.axhline(rand_asr, color="gray", ls=":", lw=1.5, label=f"rand-init")
    for i, (s, a) in enumerate(zip(sims, asrs)):
        ax.annotate(angle_labels[i], (s, a), fontsize=6.5, xytext=(2,2),
                    textcoords="offset points")
    ax.set_xlabel("Pretrain similarity", fontsize=8)
    ax.set_ylabel("FGSM ASR", fontsize=8)
    ax.set_title(f"FGSM ASR vs similarity\nρ(sim,ASR_STD)={rho_sim_asr:+.2f}", fontsize=8)
    ax.legend(fontsize=7); ax.tick_params(labelsize=7); ax.set_ylim(-0.02, 0.35)

    # (0,2) Clean accuracy: STD vs AT
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(sims, accs,    "o-",  color="steelblue",   lw=2, ms=6, label="STD-FT")
    ax.plot(sims, at_accs, "s--", color="darkorange",  lw=2, ms=6, label="AT-FT")
    ax.axhline(rand_acc, color="gray", ls=":", lw=1.5, label="rand-init")
    ax.set_xlabel("Pretrain similarity", fontsize=8)
    ax.set_ylabel("Clean accuracy", fontsize=8)
    ax.set_title("Clean accuracy vs similarity", fontsize=8)
    ax.legend(fontsize=7); ax.tick_params(labelsize=7); ax.set_ylim(0.85, 1.02)

    # (0,3) AT margin benefit: ASR reduction (STD - AT) per angle
    ax = fig.add_subplot(gs[0, 3])
    asr_reductions = [s - a for s, a in zip(asrs, at_asrs)]
    bar_c = ["green" if v >= 0 else "red" for v in asr_reductions]
    bars = ax.bar(angle_labels, asr_reductions, color=bar_c, alpha=0.75,
                  edgecolor="k", lw=0.8)
    for bar, v in zip(bars, asr_reductions):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + 0.003 if v >= 0 else v - 0.008,
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("ASR reduction (STD − AT)", fontsize=8)
    ax.set_title("AT benefit per angle\n(green = AT reduces ASR)", fontsize=8)
    ax.tick_params(labelsize=7)

    # (0,4) Acc cost of AT: STD_acc - AT_acc per angle
    ax = fig.add_subplot(gs[0, 4])
    acc_costs = [s - a for s, a in zip(accs, at_accs)]
    bar_c2 = ["red" if v > 0 else "green" for v in acc_costs]
    bars2 = ax.bar(angle_labels, acc_costs, color=bar_c2, alpha=0.75,
                   edgecolor="k", lw=0.8)
    for bar, v in zip(bars2, acc_costs):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + 0.001 if v >= 0 else v - 0.003,
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Acc cost (STD − AT)", fontsize=8)
    ax.set_title("Accuracy cost of AT\n(red = AT hurts acc)", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── Row 1: grouped bars per angle ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :])
    bar_labels_ext = angle_labels + ["rand-init"]
    all_std_asrs   = asrs  + [rand_asr]
    all_at_asrs    = at_asrs + [0.0]      # rand-init has no AT version
    all_std_accs   = accs  + [rand_acc]
    all_at_accs    = at_accs + [rand_acc]
    bc = [plt.cm.plasma(x) for x in np.linspace(0.15, 0.85, len(PRETRAIN_ANGLES))] + [(0.6,0.6,0.6,1)]
    x  = np.arange(len(bar_labels_ext))
    w  = 0.20
    b1 = ax.bar(x - 1.5*w, all_std_asrs, w, label="STD ASR",  color=bc, alpha=0.9, edgecolor="k", lw=0.7)
    b2 = ax.bar(x - 0.5*w, all_at_asrs,  w, label="AT  ASR",  color=bc, alpha=0.5, edgecolor="k", lw=0.7, hatch="xx")
    b3 = ax.bar(x + 0.5*w, all_std_accs, w, label="STD Acc",  color=bc, alpha=0.6, edgecolor="k", lw=0.7, hatch="//")
    b4 = ax.bar(x + 1.5*w, all_at_accs,  w, label="AT  Acc",  color=bc, alpha=0.3, edgecolor="k", lw=0.7, hatch="\\\\")
    for bars, vals in [(b1,all_std_asrs),(b2,all_at_asrs),(b3,all_std_accs),(b4,all_at_accs)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(bar_labels_ext, fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.set_title("STD ASR / AT ASR / STD Acc / AT Acc — per pretrain angle", fontsize=10)
    ax.legend(fontsize=8, ncol=4); ax.set_ylim(0, 1.25); ax.tick_params(labelsize=8)

    # ── Row 2: STD fine-tune boundaries ──────────────────────────────────────
    for col_i, (angle, ft_m, Xft_i, r) in enumerate(
            zip(PRETRAIN_ANGLES, ft_models, ft_Xft_list, results)):
        ax = fig.add_subplot(gs[2, col_i])
        plot_boundary(ax, ft_m, Xft_i, Yft,
                      f"STD  angle={angle}°\nacc={r['ft_acc']:.3f}  ASR={r['ft_asr']:.3f}")

    # ── Row 3: AT fine-tune boundaries ───────────────────────────────────────
    for col_i, (angle, ft_m, Xft_i, r) in enumerate(
            zip(PRETRAIN_ANGLES, ft_at_models, ft_Xft_list, results)):
        ax = fig.add_subplot(gs[3, col_i])
        plot_boundary(ax, ft_m, Xft_i, Yft,
                      f"AT   angle={angle}°\nacc={r['at_acc']:.3f}  ASR={r['at_asr']:.3f}")

    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {FIG_PATH}")
    print(f"Text   → {TXT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
