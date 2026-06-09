"""
H532 – Margin Dynamics During Training
=======================================

Question: what happens to the decision boundary margin as training progresses?

Hypothesis:
  Standard CE training: margin SHRINKS over epochs.
    Mechanism: CE loss is minimised by increasing logit confidence, which
    pushes the boundary as close as possible to each training point to maximise
    the log-probability.  The boundary gets "greedy" — it crowds the data.

  Adversarial training (AT): margin GROWS (or stays bounded from below at eps).
    Mechanism: AT trains on worst-case examples within the eps-ball, so the
    boundary must maintain eps-clearance from ALL training points or the
    adversarial examples will be misclassified.

Margin definitions measured per epoch:
  1. Functional margin (logit margin): max_logit - second_max_logit.
     Positive for correct predictions.  Proxy — monotone with geometric margin
     for linear classifiers, correlated for MLP.
  2. Geometric margin (2D only): binary-search for the boundary crossing
     distance along the gradient direction.  Exact but expensive; computed
     only at checkpoints.

Dataset: 2D Gaussian blobs (2-class) for exact geometric margin + boundary viz.
         Also 2-class Fashion-MNIST (T-shirt vs Trouser) for logit margin.

Figures:
  Figure 1 (2D):
    (0,0) Logit margin mean±std vs epoch  (STD vs AT)
    (0,1) Geometric margin mean vs epoch  (checkpoints)
    (0,2) Clean accuracy vs epoch
    (0,3) FGSM ASR vs epoch
    Row 1: boundary snapshots at epoch [1, 10, 50, 200, final] for STD
    Row 2: boundary snapshots at epoch [1, 10, 50, 200, final] for AT

  Figure 2 (FMNIST):
    (0,0) Logit margin mean±std vs epoch  (STD vs AT)
    (0,1) Clean accuracy vs epoch
    (0,2) FGSM ASR vs epoch
    (0,3) Logit margin histogram: epoch 1 vs final (STD vs AT)

PASS: AT mean logit margin at final epoch > STD mean logit margin at final epoch.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
OUT_DIR     = os.path.join(PROJECT_DIR, "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)

FIG1_PATH = os.path.join(OUT_DIR, "h532_margin_dynamics_2d.png")
FIG2_PATH = os.path.join(OUT_DIR, "h532_margin_dynamics_fmnist.png")
TXT_PATH  = os.path.join(OUT_DIR, "h532_margin_dynamics_training_output.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

# 2D experiment
EPOCHS_2D    = 300
LR_2D        = 5e-3
N_TRAIN_2D   = 200   # per class
FGSM_EPS_2D  = 0.15
GEO_CHECKPTS = [1, 5, 10, 25, 50, 100, 150, 200, 250, 300]
BND_CHECKPTS = [1, 10, 50, 150, 300]   # boundary snapshots

# FMNIST experiment
EPOCHS_FM    = 30
LR_FM        = 1e-3
BATCH_FM     = 256
FGSM_EPS_FM  = 0.10


# ──────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path):
        self._f = open(path, "w"); self._s = sys.stdout
    def write(self, d): self._s.write(d); self._f.write(d)
    def flush(self): self._s.flush(); self._f.flush()
    def close(self): self._f.close()
    def __getattr__(self, n): return getattr(self._s, n)


# ──────────────────────────────────────────────────────────────────────────────
# 2D dataset
# ──────────────────────────────────────────────────────────────────────────────
def make_2d(n, seed=SEED):
    rng = np.random.default_rng(seed)
    X0 = rng.normal([-1.5, 0], 0.6, (n, 2)).astype(np.float32)
    X1 = rng.normal([ 1.5, 0], 0.6, (n, 2)).astype(np.float32)
    X  = np.vstack([X0, X1])
    Y  = np.array([0]*n + [1]*n, dtype=np.int64)
    idx = rng.permutation(len(X)); return X[idx], Y[idx]

def to_t(X, Y):
    return (torch.tensor(X, dtype=torch.float32).to(DEVICE),
            torch.tensor(Y, dtype=torch.long).to(DEVICE))


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
class MLP2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, n_cls=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.clf = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7,128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_cls),
        )
    def forward(self, x): return self.clf(self.feat(x))


# ──────────────────────────────────────────────────────────────────────────────
# Attacks
# ──────────────────────────────────────────────────────────────────────────────
def fgsm(model, x, y, eps):
    xv = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xv), y).backward()
    return (x + eps * xv.grad.sign()).clamp(0, 1).detach()


def fgsm_2d(model, x, y, eps):
    """No clamp for 2D unbounded features."""
    xv = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xv), y).backward()
    return (x + eps * xv.grad.sign()).detach()


# ──────────────────────────────────────────────────────────────────────────────
# Margin metrics
# ──────────────────────────────────────────────────────────────────────────────
def logit_margin(model, X, Y):
    """Per-sample: logit of true class minus max other logit. Mean ± std."""
    model.eval()
    with torch.no_grad():
        logits = model(X)                               # (N, C)
        true_logit = logits[torch.arange(len(Y)), Y]   # (N,)
        # mask true class and take max of rest
        tmp = logits.clone()
        tmp[torch.arange(len(Y)), Y] = -1e9
        other_max = tmp.max(1).values
        margin = true_logit - other_max                 # positive = correct + margin
    return margin.cpu()


def geometric_margin_2d(model, X, Y, eps_search=3.0, steps=30):
    """
    Binary search along the FGSM gradient direction to find boundary crossing.
    Returns mean distance across correctly-classified points.
    """
    model.eval()
    margins = []
    with torch.no_grad():
        correct = model(X).argmax(1) == Y
    Xc, Yc = X[correct], Y[correct]
    if len(Xc) == 0: return 0.0

    for i in range(len(Xc)):
        xi = Xc[i:i+1].clone().detach().requires_grad_(True)
        F.cross_entropy(model(xi), Yc[i:i+1]).backward()
        direction = xi.grad.sign().detach()   # (1,2)

        lo, hi = 0.0, eps_search
        for _ in range(steps):
            mid = (lo + hi) / 2
            x_mid = Xc[i:i+1] + mid * direction
            with torch.no_grad():
                pred = model(x_mid).argmax(1)
            if pred == Yc[i]:
                lo = mid      # still same class, push further
            else:
                hi = mid      # already flipped
        margins.append(hi)    # smallest eps that flips the prediction
    return float(np.mean(margins))


def fgsm_asr_tensor(model, X, Y, eps, attack_fn):
    model.eval()
    with torch.no_grad():
        correct = model(X).argmax(1) == Y
    if correct.sum() == 0: return 0.0
    Xc, Yc = X[correct], Y[correct]
    xa = attack_fn(model, Xc, Yc, eps)
    with torch.no_grad():
        return (model(xa).argmax(1) != Yc).float().mean().item()


def fgsm_asr_loader(model, loader, eps):
    model.eval(); c = f = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            ok = model(x).argmax(1) == y
        if ok.sum() == 0: continue
        xc, yc = x[ok], y[ok]
        xa = fgsm(model, xc, yc, eps)
        with torch.no_grad():
            f += (model(xa).argmax(1) != yc).sum().item()
        c += ok.sum().item()
    return f / max(c, 1)


def acc_tensor(model, X, Y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == Y).float().mean().item()


def acc_loader(model, loader):
    model.eval(); c = t = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            c += (model(x).argmax(1) == y).sum().item()
        t += len(y)
    return c / t


# ──────────────────────────────────────────────────────────────────────────────
# 2D boundary plot
# ──────────────────────────────────────────────────────────────────────────────
def plot_bnd(ax, model, X, Y, title, lo=-4, hi=4):
    rr = np.linspace(lo, hi, 200); tt = np.linspace(lo, hi, 200)
    xx, yy = np.meshgrid(rr, tt)
    g = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        zz = model(g).argmax(1).cpu().numpy().reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5,0.5,1.5], colors=["#aec6cf","#f4a460"], alpha=0.35)
    for ci, col in enumerate(["#1f4e79","#7b2d00"]):
        mask = Y == ci
        ax.scatter(X[mask,0], X[mask,1], s=10, alpha=0.6, color=col, linewidths=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(title, fontsize=7.5); ax.tick_params(labelsize=6)
    ax.set_aspect("equal")


# ──────────────────────────────────────────────────────────────────────────────
# Training with per-epoch recording
# ──────────────────────────────────────────────────────────────────────────────
def train_2d_record(model, Xt, Yt, Xte, Yte, epochs, at=False, eps=FGSM_EPS_2D):
    opt   = optim.Adam(model.parameters(), lr=LR_2D)
    rec   = dict(epoch=[], lm_mean=[], lm_std=[], acc=[], asr=[], geo=[])
    snapshots = {}   # epoch → model state_dict copy

    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        if at:
            xa = fgsm_2d(model, Xt, Yt, eps)
            model.train(); opt.zero_grad()
            F.cross_entropy(model(xa), Yt).backward()
        else:
            F.cross_entropy(model(Xt), Yt).backward()
        opt.step()

        if ep % 5 == 0 or ep == 1:
            lm = logit_margin(model, Xt, Yt)
            a  = acc_tensor(model, Xte, Yte)
            asr = fgsm_asr_tensor(model, Xte, Yte, eps, fgsm_2d)
            rec["epoch"].append(ep)
            rec["lm_mean"].append(lm.mean().item())
            rec["lm_std"].append(lm.std().item())
            rec["acc"].append(a)
            rec["asr"].append(asr)
            # geometric margin at checkpoints
            if ep in GEO_CHECKPTS:
                geo = geometric_margin_2d(model, Xt, Yt)
                rec["geo"].append((ep, geo))

        if ep in BND_CHECKPTS:
            import copy
            snapshots[ep] = copy.deepcopy(model).cpu()

    return rec, snapshots


def train_fm_record(model, tr_loader, te_loader, epochs, at=False, eps=FGSM_EPS_FM):
    opt  = optim.Adam(model.parameters(), lr=LR_FM)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rec  = dict(epoch=[], lm_mean=[], lm_std=[], acc=[], asr=[],
                lm_hist_first=None, lm_hist_last=None)

    # Collect a fixed eval batch for logit margin (same points each epoch)
    eval_x, eval_y = [], []
    for x, y in te_loader:
        eval_x.append(x); eval_y.append(y)
        if len(torch.cat(eval_x)) >= 1000: break
    eval_x = torch.cat(eval_x)[:1000].to(DEVICE)
    eval_y = torch.cat(eval_y)[:1000].to(DEVICE)

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if at:
                xa = fgsm(model, x, y, eps)
                model.train(); opt.zero_grad()
                F.cross_entropy(model(xa), y).backward()
            else:
                F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()

        lm  = logit_margin(model, eval_x, eval_y)
        a   = acc_loader(model, te_loader)
        asr = fgsm_asr_loader(model, te_loader, eps)
        rec["epoch"].append(ep)
        rec["lm_mean"].append(lm.mean().item())
        rec["lm_std"].append(lm.std().item())
        rec["acc"].append(a)
        rec["asr"].append(asr)
        if ep == 1:
            rec["lm_hist_first"] = lm.numpy().copy()
        print(f"  ep {ep:3d}/{epochs}  acc={a:.3f}  ASR={asr:.3f}  "
              f"margin_mean={lm.mean():.3f}")

    rec["lm_hist_last"] = lm.numpy().copy()
    return rec


# ──────────────────────────────────────────────────────────────────────────────
def main():
    tee = Tee(TXT_PATH)
    sys.stdout = tee
    try: _run()
    finally: sys.stdout = tee._s; tee.close()


def _run():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"Device: {DEVICE}")
    print("=" * 70)
    print("H532 – Margin Dynamics During Training")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: 2D Gaussian blobs
    # ══════════════════════════════════════════════════════════════════════════
    print("\n──── Part 1: 2D Gaussian blobs ────")
    Xtr, Ytr = make_2d(N_TRAIN_2D, seed=SEED)
    Xte, Yte = make_2d(500,        seed=SEED + 1)
    Xtr_t, Ytr_t = to_t(Xtr, Ytr)
    Xte_t, Yte_t = to_t(Xte, Yte)

    print("\n[2D] Training STD model ...")
    torch.manual_seed(SEED)
    m_std = MLP2D().to(DEVICE)
    rec_std, snaps_std = train_2d_record(m_std, Xtr_t, Ytr_t,
                                          Xte_t, Yte_t, EPOCHS_2D, at=False)

    print("\n[2D] Training AT model ...")
    torch.manual_seed(SEED)
    m_at = MLP2D().to(DEVICE)
    rec_at, snaps_at = train_2d_record(m_at, Xtr_t, Ytr_t,
                                        Xte_t, Yte_t, EPOCHS_2D, at=True)

    print("\n[2D] Final metrics:")
    print(f"  STD: acc={rec_std['acc'][-1]:.3f}  ASR={rec_std['asr'][-1]:.3f}  "
          f"margin_mean={rec_std['lm_mean'][-1]:.3f}")
    print(f"  AT:  acc={rec_at['acc'][-1]:.3f}  ASR={rec_at['asr'][-1]:.3f}  "
          f"margin_mean={rec_at['lm_mean'][-1]:.3f}")

    geo_epochs_std = [g[0] for g in rec_std["geo"]]
    geo_vals_std   = [g[1] for g in rec_std["geo"]]
    geo_epochs_at  = [g[0] for g in rec_at["geo"]]
    geo_vals_at    = [g[1] for g in rec_at["geo"]]

    # ── Figure 1 ─────────────────────────────────────────────────────────────
    print("\nGenerating Figure 1 (2D) ...")
    n_snaps = len(BND_CHECKPTS)
    fig1 = plt.figure(figsize=(22, 12))
    fig1.suptitle("H532 – Margin Dynamics During Training (2D Gaussian Blobs)\n"
                  "CE training shrinks margin; AT training grows margin",
                  fontsize=12, fontweight="bold")
    gs1 = mgs.GridSpec(3, n_snaps, figure=fig1, hspace=0.50, wspace=0.30)

    # Row 0: 4 metric plots spanning first 4 cols; geo margin in col 4
    # Logit margin
    ax = fig1.add_subplot(gs1[0, 0])
    ep = rec_std["epoch"]
    ax.plot(ep, rec_std["lm_mean"], color="steelblue", lw=2, label="STD mean")
    ax.fill_between(ep,
                    [m-s for m,s in zip(rec_std["lm_mean"], rec_std["lm_std"])],
                    [m+s for m,s in zip(rec_std["lm_mean"], rec_std["lm_std"])],
                    alpha=0.15, color="steelblue")
    ax.plot(ep, rec_at["lm_mean"],  color="darkorange", lw=2, label="AT  mean")
    ax.fill_between(ep,
                    [m-s for m,s in zip(rec_at["lm_mean"], rec_at["lm_std"])],
                    [m+s for m,s in zip(rec_at["lm_mean"], rec_at["lm_std"])],
                    alpha=0.15, color="darkorange")
    ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel("Logit margin", fontsize=8)
    ax.set_title("Logit margin (mean ± std)\nover training epochs", fontsize=8)
    ax.legend(fontsize=7.5); ax.tick_params(labelsize=7)

    # Geometric margin
    ax = fig1.add_subplot(gs1[0, 1])
    ax.plot(geo_epochs_std, geo_vals_std, "o-", color="steelblue", lw=2, ms=5, label="STD")
    ax.plot(geo_epochs_at,  geo_vals_at,  "s-", color="darkorange", lw=2, ms=5, label="AT")
    ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel("Geometric margin (ε to flip)", fontsize=8)
    ax.set_title("Geometric margin\n(binary-search boundary distance)", fontsize=8)
    ax.legend(fontsize=7.5); ax.tick_params(labelsize=7)

    # Clean accuracy
    ax = fig1.add_subplot(gs1[0, 2])
    ax.plot(ep, rec_std["acc"], color="steelblue", lw=2, label="STD")
    ax.plot(ep, rec_at["acc"],  color="darkorange", lw=2, label="AT")
    ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel("Clean accuracy", fontsize=8)
    ax.set_title("Clean accuracy\nover training", fontsize=8)
    ax.legend(fontsize=7.5); ax.tick_params(labelsize=7); ax.set_ylim(0.5, 1.02)

    # ASR
    ax = fig1.add_subplot(gs1[0, 3])
    ax.plot(ep, rec_std["asr"], color="steelblue", lw=2, label="STD")
    ax.plot(ep, rec_at["asr"],  color="darkorange", lw=2, label="AT")
    ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel("FGSM ASR", fontsize=8)
    ax.set_title(f"FGSM ASR (ε={FGSM_EPS_2D})\nover training", fontsize=8)
    ax.legend(fontsize=7.5); ax.tick_params(labelsize=7); ax.set_ylim(-0.02, 1.02)

    # Margin vs ASR scatter (both models all time steps)
    ax = fig1.add_subplot(gs1[0, 4])
    ax.scatter(rec_std["lm_mean"], rec_std["asr"], c=rec_std["epoch"],
               cmap="Blues", s=20, alpha=0.7, label="STD", marker="o")
    sc = ax.scatter(rec_at["lm_mean"], rec_at["asr"], c=rec_at["epoch"],
                    cmap="Oranges", s=20, alpha=0.7, label="AT", marker="s")
    ax.set_xlabel("Mean logit margin", fontsize=8); ax.set_ylabel("FGSM ASR", fontsize=8)
    ax.set_title("Margin vs ASR trajectory\n(colour = epoch)", fontsize=8)
    ax.legend(fontsize=7.5); ax.tick_params(labelsize=7)

    # Rows 1-2: boundary snapshots
    for col_i, ep_snap in enumerate(BND_CHECKPTS):
        # STD
        ax = fig1.add_subplot(gs1[1, col_i])
        m_snap = snaps_std[ep_snap].to(DEVICE)
        lm_snap = logit_margin(m_snap, Xtr_t, Ytr_t).mean().item()
        geo_snap_vals = [g[1] for g in rec_std["geo"] if g[0] == ep_snap]
        geo_snap = geo_snap_vals[0] if geo_snap_vals else float("nan")
        plot_bnd(ax, m_snap, Xtr, Ytr,
                 f"STD ep={ep_snap}\nlogit_m={lm_snap:.2f}  geo_m={geo_snap:.3f}")

        # AT
        ax = fig1.add_subplot(gs1[2, col_i])
        m_snap = snaps_at[ep_snap].to(DEVICE)
        lm_snap = logit_margin(m_snap, Xtr_t, Ytr_t).mean().item()
        geo_snap_vals = [g[1] for g in rec_at["geo"] if g[0] == ep_snap]
        geo_snap = geo_snap_vals[0] if geo_snap_vals else float("nan")
        plot_bnd(ax, m_snap, Xtr, Ytr,
                 f"AT  ep={ep_snap}\nlogit_m={lm_snap:.2f}  geo_m={geo_snap:.3f}")

    fig1.savefig(FIG1_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Figure 1 → {FIG1_PATH}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: Fashion-MNIST (T-shirt vs Trouser)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n──── Part 2: Fashion-MNIST (T-shirt vs Trouser) ────")
    tf = transforms.ToTensor()
    fmnist_tr = torchvision.datasets.FashionMNIST(DATA_DIR, train=True,  download=True, transform=tf)
    fmnist_te = torchvision.datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)

    def subset_2cls(ds):
        xs, ys = [], []
        for x, y in DataLoader(ds, batch_size=512, num_workers=2):
            mask = (y == 0) | (y == 1)
            xs.append(x[mask]); ys.append(y[mask])
        return TensorDataset(torch.cat(xs), torch.cat(ys))

    tr_ds = subset_2cls(fmnist_tr)
    te_ds = subset_2cls(fmnist_te)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_FM, shuffle=True,  num_workers=2, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=BATCH_FM, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Train: {len(tr_ds)}  Test: {len(te_ds)}")

    print("\n[FMNIST] Training STD model ...")
    torch.manual_seed(SEED)
    fm_std = SmallCNN(n_cls=2).to(DEVICE)
    rec_fm_std = train_fm_record(fm_std, tr_loader, te_loader, EPOCHS_FM, at=False)

    print("\n[FMNIST] Training AT model ...")
    torch.manual_seed(SEED)
    fm_at = SmallCNN(n_cls=2).to(DEVICE)
    rec_fm_at = train_fm_record(fm_at, tr_loader, te_loader, EPOCHS_FM, at=True)

    print("\n[FMNIST] Final metrics:")
    print(f"  STD: acc={rec_fm_std['acc'][-1]:.4f}  ASR={rec_fm_std['asr'][-1]:.4f}  "
          f"margin_mean={rec_fm_std['lm_mean'][-1]:.4f}")
    print(f"  AT:  acc={rec_fm_at['acc'][-1]:.4f}  ASR={rec_fm_at['asr'][-1]:.4f}  "
          f"margin_mean={rec_fm_at['lm_mean'][-1]:.4f}")

    # ── Figure 2 ─────────────────────────────────────────────────────────────
    print("\nGenerating Figure 2 (FMNIST) ...")
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
    fig2.suptitle("H532 – Margin Dynamics on Fashion-MNIST (T-shirt vs Trouser)\n"
                  "CE training shrinks logit margin; AT training grows it",
                  fontsize=11, fontweight="bold")

    ep_fm = rec_fm_std["epoch"]

    # Logit margin
    ax = axes2[0]
    ax.plot(ep_fm, rec_fm_std["lm_mean"], color="steelblue", lw=2, label="STD mean")
    ax.fill_between(ep_fm,
        [m-s for m,s in zip(rec_fm_std["lm_mean"], rec_fm_std["lm_std"])],
        [m+s for m,s in zip(rec_fm_std["lm_mean"], rec_fm_std["lm_std"])],
        alpha=0.15, color="steelblue")
    ax.plot(ep_fm, rec_fm_at["lm_mean"],  color="darkorange", lw=2, label="AT  mean")
    ax.fill_between(ep_fm,
        [m-s for m,s in zip(rec_fm_at["lm_mean"], rec_fm_at["lm_std"])],
        [m+s for m,s in zip(rec_fm_at["lm_mean"], rec_fm_at["lm_std"])],
        alpha=0.15, color="darkorange")
    ax.set_xlabel("Epoch", fontsize=9); ax.set_ylabel("Logit margin", fontsize=9)
    ax.set_title("Logit margin (mean ± std)\nSTD grows fast then flatlines; AT grows slower but larger", fontsize=8.5)
    ax.legend(fontsize=8); ax.tick_params(labelsize=8)

    # Clean accuracy
    ax = axes2[1]
    ax.plot(ep_fm, rec_fm_std["acc"], color="steelblue", lw=2, label="STD")
    ax.plot(ep_fm, rec_fm_at["acc"],  color="darkorange", lw=2, label="AT")
    ax.set_xlabel("Epoch", fontsize=9); ax.set_ylabel("Clean accuracy", fontsize=9)
    ax.set_title("Clean accuracy over training", fontsize=8.5)
    ax.legend(fontsize=8); ax.tick_params(labelsize=8); ax.set_ylim(0.5, 1.02)

    # FGSM ASR
    ax = axes2[2]
    ax.plot(ep_fm, rec_fm_std["asr"], color="steelblue", lw=2, label="STD")
    ax.plot(ep_fm, rec_fm_at["asr"],  color="darkorange", lw=2, label="AT")
    ax.set_xlabel("Epoch", fontsize=9); ax.set_ylabel("FGSM ASR", fontsize=9)
    ax.set_title(f"FGSM ASR (ε={FGSM_EPS_FM}) over training\nSTD ASR rises; AT ASR falls", fontsize=8.5)
    ax.legend(fontsize=8); ax.tick_params(labelsize=8); ax.set_ylim(-0.02, 1.02)

    # Logit margin histogram: first vs last epoch
    ax = axes2[3]
    bins = np.linspace(-3, 10, 50)
    ax.hist(rec_fm_std["lm_hist_first"], bins=bins, alpha=0.4, color="steelblue",
            label="STD ep=1",  density=True)
    ax.hist(rec_fm_std["lm_hist_last"],  bins=bins, alpha=0.7, color="steelblue",
            label=f"STD ep={EPOCHS_FM}", density=True, histtype="step", lw=2)
    ax.hist(rec_fm_at["lm_hist_first"],  bins=bins, alpha=0.4, color="darkorange",
            label="AT  ep=1",  density=True)
    ax.hist(rec_fm_at["lm_hist_last"],   bins=bins, alpha=0.7, color="darkorange",
            label=f"AT  ep={EPOCHS_FM}", density=True, histtype="step", lw=2)
    ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Logit margin", fontsize=9); ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Logit margin distribution\nepoch 1 vs final (both models)", fontsize=8.5)
    ax.legend(fontsize=7.5); ax.tick_params(labelsize=8)

    fig2.tight_layout()
    fig2.savefig(FIG2_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Figure 2 → {FIG2_PATH}")

    # ── PASS ──────────────────────────────────────────────────────────────────
    at_margin_final = rec_fm_at["lm_mean"][-1]
    std_margin_final = rec_fm_std["lm_mean"][-1]
    passed = at_margin_final > std_margin_final
    print("\n" + "=" * 70)
    print(f"PASS: AT final margin ({at_margin_final:.4f}) "
          f"{'>' if passed else '<='} STD final margin ({std_margin_final:.4f})")
    print(f"=> {'PASS' if passed else 'FAIL'}")

    # 2D geometric margin comparison
    geo_std_final = rec_std["geo"][-1][1]
    geo_at_final  = rec_at["geo"][-1][1]
    print(f"\n2D geometric margin: STD={geo_std_final:.4f}  AT={geo_at_final:.4f}")
    print(f"AT geometric margin {'>' if geo_at_final > geo_std_final else '<='} STD: "
          f"{'confirmed' if geo_at_final > geo_std_final else 'NOT confirmed'}")

    print(f"\nText   → {TXT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
