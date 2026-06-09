"""
H509 - Information Bottleneck view of adversarial training (§5 / §3 G6).

Hypothesis (Tishby's IB lens, applied to robustness):
    Adversarial training (PGD-AT) compresses irrelevant input variance, i.e. it
    REDUCES I(X; Z) (input-feature mutual information) relative to a standard
    (STD) model, WHILE preserving I(Z; Y) (feature-label MI).  Equivalently, AT
    moves the representation toward the IB-optimal corner: less about the input,
    just as much about the label.

We probe this on a Fashion-MNIST SmallCNN with two estimators (each known to be
biased on continuous high-dim variables; we use them as cross-checks, not
ground-truth):

  (A) Plug-in / binning estimator.
      Penultimate features Z (256-d after first FC+ReLU) are reduced to 4-d via
      PCA fit on a held-out probe set, then equal-frequency-binned to 8 bins per
      axis (=> 4096-cell joint histogram).  Inputs X are reduced the same way
      (PCA on flattened pixels -> 4-d -> 8 bins). Labels Y are categorical.
      I(X;Z) and I(Z;Y) are computed by plug-in entropy on the resulting
      discrete codebooks. This is the binning approach Tishby and Saxe et al.
      use in their toy IB analyses.

  (B) MINE neural estimator (Belghazi et al. 2018, ICML).
      Small statistics network T(x, z) trained with the Donsker-Varadhan
      lower bound on KL(p(x,z) || p(x) p(z))   =>   I_MINE = sup_T E_p[T] -
      log E_{p_x p_z}[exp(T)].  We use the same penultimate Z (no PCA on the
      MINE side, so we get a higher-fidelity estimate that is not bottlenecked
      by 4-d PCA) and X flattened. Tested with 600 SGD steps and EMA bias
      correction on the denominator gradient (Belghazi §3.2).

Controls (§3 G6):
  (1) STD vs PGD-AT (matched architecture / opt / epochs).
  (2) Penultimate features -> PCA dim 4 for the binning estimator.
  (3) Plug-in entropy on the joint codebook for I(X;Z) and I(Z;Y).
  (4) MINE neural estimator as cross-check on un-PCA'd features.
  (5) Epoch-wise correlation of I(X;Z) (binning) with PGD attack-success rate
      across the AT training trajectory (we snapshot the AT model every epoch
      and report the rank correlation).

Citations:
  * Tishby & Zaslavsky, "Deep Learning and the Information Bottleneck Principle",
    ITW 2015 (arXiv:1503.02406) - the IB-of-DNN claim being tested.
  * Belghazi et al., "MINE: Mutual Information Neural Estimation", ICML 2018
    (arXiv:1801.04062) - the neural estimator.
  * Achille & Soatto, "Emergence of Invariance and Disentanglement in Deep
    Representations", JMLR 2018 (arXiv:1706.01350) - links low I(X;Z) /
    "minimality" of representation to invariance & robustness, motivating the
    hypothesis that adversarial robustness should track a smaller I(X;Z).
  * (Also relevant: Saxe et al. 2018, "On the Information Bottleneck Theory of
    Deep Learning"; Goldfeld et al. 2019 on estimator sensitivity. We treat
    both estimators as approximations and emphasise the AT-vs-STD CONTRAST.)

Limitations (documented explicitly in the printed report):
  * MI in continuous high-dim spaces is notoriously hard to estimate; binning
    with 8 bins per 4 PCA axes throws away most of the geometry and is biased
    upward by the small sample. We therefore report DIFFERENCES (AT - STD), not
    absolute nats, and require both estimators to agree in sign.
  * Deterministic ReLU networks have I(X; Z) = H(X) in theory (Z is a
    deterministic function of X). Empirically the binned estimate is finite
    because quantisation injects noise; this is the standard caveat (Saxe 2018).
  * The epoch correlation uses 6 AT epochs; this is a low-N test.

DO NOT RUN. Caller will dispatch this via the campaign harness.
"""
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
EPS = 0.1
PGD_STEPS = 10
EPOCHS = 6
N_TRAIN = 6000
N_EVAL = 2000
N_PROBE = 1500            # how many samples to use for MI estimation
PCA_DIM = 4
N_BINS = 8                # 8 bins per axis => 8^4 = 4096 cells
MINE_STEPS = 600
MINE_LR = 5e-4
MINE_BATCH = 256
MINE_HIDDEN = 128
RESULT_PATH = "results/fashion_mnist/h509_info_bottleneck_output.txt"


# ---------------------------------------------------------------------------
# Penultimate-feature extraction for SmallCNN (256-d after first FC+ReLU).
# SmallCNN.head = Sequential(Flatten, Linear(., 256), A(), Linear(256, ncls)).
# We want the activation AFTER the activation A() and BEFORE the final Linear.
# ---------------------------------------------------------------------------
@torch.no_grad()
def penultimate(model, X, batch=512):
    model.eval()
    outs = []
    for i in range(0, X.size(0), batch):
        x = X[i:i + batch]
        h = model.features(x)
        # model.head = Sequential(Flatten, Linear, Act, Linear); run all but last
        z = h
        for layer in list(model.head.children())[:-1]:
            z = layer(z)
        outs.append(z.detach().cpu())
    return torch.cat(outs).numpy()


# ---------------------------------------------------------------------------
# PCA (centred, top-k components) - NumPy SVD, deterministic.
# ---------------------------------------------------------------------------
def pca_fit_transform(X_train, X_eval, dim):
    mu = X_train.mean(0, keepdims=True)
    Xc = X_train - mu
    # economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:dim].T                # (D, dim)
    return (X_train - mu) @ W, (X_eval - mu) @ W


# ---------------------------------------------------------------------------
# Equal-frequency binning (per axis) -> integer codebook id.
# ---------------------------------------------------------------------------
def quantise(X, nbins):
    """Per-axis equal-frequency bins; return integer code per sample
    (flat index over the per-axis bin tuple). X: (N, d)."""
    N, d = X.shape
    codes = np.zeros((N, d), dtype=np.int64)
    for j in range(d):
        # quantile edges; use np.searchsorted on the column
        qs = np.quantile(X[:, j], np.linspace(0, 1, nbins + 1)[1:-1])
        codes[:, j] = np.searchsorted(qs, X[:, j], side="right")
    # collapse multi-axis bin id to a single integer
    flat = np.zeros(N, dtype=np.int64)
    mult = 1
    for j in range(d):
        flat += codes[:, j] * mult
        mult *= nbins
    return flat


def _entropy_from_counts(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())   # nats


def mi_plugin(a, b):
    """Plug-in MI between two discrete-coded arrays a, b (1-D ints)."""
    # joint histogram
    ua = np.unique(a, return_inverse=True)[1]
    ub = np.unique(b, return_inverse=True)[1]
    Na, Nb = ua.max() + 1, ub.max() + 1
    J = np.zeros((Na, Nb), dtype=np.int64)
    np.add.at(J, (ua, ub), 1)
    pa = J.sum(1)
    pb = J.sum(0)
    Ha = _entropy_from_counts(pa)
    Hb = _entropy_from_counts(pb)
    Hab = _entropy_from_counts(J.ravel())
    return Ha + Hb - Hab


# ---------------------------------------------------------------------------
# MINE estimator (Belghazi et al., ICML 2018) with EMA bias correction.
# Returns the final running-average lower bound on I(X; Z) in nats.
# ---------------------------------------------------------------------------
class MineNet(nn.Module):
    def __init__(self, dx, dz, hidden=MINE_HIDDEN):
        super().__init__()
        self.fx = nn.Linear(dx, hidden)
        self.fz = nn.Linear(dz, hidden)
        self.out = nn.Sequential(nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x, z):
        h = self.fx(x) + self.fz(z)
        return self.out(h).squeeze(-1)


def mine_estimate(X, Z, steps=MINE_STEPS, lr=MINE_LR, batch=MINE_BATCH, seed=0):
    """X: (N, dx) torch on device, Z: (N, dz) torch on device.
    Standardise inputs per-feature for numerical stability."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    N = Xs.size(0)
    net = MineNet(Xs.size(1), Zs.size(1)).to(Xs.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ema = None
    ema_decay = 0.99
    last = []
    for t in range(steps):
        i = torch.randint(0, N, (batch,), generator=g, device=Xs.device)
        j = torch.randint(0, N, (batch,), generator=g, device=Xs.device)
        x_j, z_j = Xs[i], Zs[i]                    # joint samples
        z_m = Zs[j]                                # marginal Z (shuffled)
        t_joint = net(x_j, z_j)
        t_marg = net(x_j, z_m)
        # DV lower bound: E[T] - log E[exp(T)]
        et = torch.exp(t_marg)
        if ema is None:
            ema = et.mean().detach()
        else:
            ema = ema_decay * ema + (1 - ema_decay) * et.mean().detach()
        # bias-corrected gradient (Belghazi §3.2): use ema in the denominator
        loss = -(t_joint.mean() - (et.mean() / (ema + 1e-8)).log() * ema.detach()
                 / (ema.detach() + 1e-8) * 1.0)
        # simpler & equivalent in practice: use the standard DV objective for the
        # *value* and the EMA only for the gradient. We approximate by:
        mi_val = t_joint.mean() - torch.log(et.mean() + 1e-8)
        # actual gradient step on the EMA-corrected loss:
        loss_grad = -(t_joint.mean() - et.mean() / (ema + 1e-8))
        opt.zero_grad()
        loss_grad.backward()
        opt.step()
        last.append(float(mi_val.detach().cpu()))
    # report mean over last 100 steps to reduce variance
    tail = last[-100:] if len(last) >= 100 else last
    return float(np.mean(tail))


# ---------------------------------------------------------------------------
# MI block: given an evaluated model, return I(X;Z) [binning + MINE] and I(Z;Y).
# ---------------------------------------------------------------------------
def estimate_mi(model, X_probe, Y_probe, seed):
    """X_probe: (N, C, H, W) torch on device. Y_probe: (N,) torch on device."""
    Z = penultimate(model, X_probe)                                  # np (N, 256)
    Xflat = X_probe.detach().cpu().numpy().reshape(X_probe.size(0), -1)
    Y = Y_probe.detach().cpu().numpy()

    # split probe set into "fit" (for PCA basis) and "eval" (for MI counting)
    N = Z.shape[0]
    half = N // 2
    Z_fit, Z_ev = Z[:half], Z[half:]
    X_fit, X_ev = Xflat[:half], Xflat[half:]
    Y_ev = Y[half:]

    # PCA to PCA_DIM
    _, Zp_ev = pca_fit_transform(Z_fit, Z_ev, PCA_DIM)
    _, Xp_ev = pca_fit_transform(X_fit, X_ev, PCA_DIM)

    z_code = quantise(Zp_ev, N_BINS)
    x_code = quantise(Xp_ev, N_BINS)

    I_xz_bin = mi_plugin(x_code, z_code)
    I_zy_bin = mi_plugin(z_code, Y_ev)

    # MINE cross-check on the same eval half, using un-PCA'd Z and flattened X
    dev = X_probe.device
    Zt = torch.tensor(Z_ev, device=dev, dtype=torch.float32)
    Xt = torch.tensor(X_ev, device=dev, dtype=torch.float32)
    I_xz_mine = mine_estimate(Xt, Zt, seed=seed)

    return {
        "I_xz_bin": I_xz_bin,
        "I_zy_bin": I_zy_bin,
        "I_xz_mine": I_xz_mine,
    }


# ---------------------------------------------------------------------------
# Spearman rank correlation (small N safe, no SciPy dep)
# ---------------------------------------------------------------------------
def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3:
        return float("nan")
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = (np.sqrt((ra * ra).sum() * (rb * rb).sum())) + 1e-12
    return float((ra * rb).sum() / denom)


# ---------------------------------------------------------------------------
# Per-epoch trajectory for AT model: train one epoch, snapshot MI + PGD ASR.
# We re-implement a manual epoch-by-epoch loop because train_model() takes
# total epochs as an argument; this exposes the trajectory.
# ---------------------------------------------------------------------------
def train_one_epoch(model, Xtr, Ytr, opt_, ncls, adv=False, adv_eps=EPS,
                    adv_steps=7, batch=128):
    model.train()
    n = Xtr.size(0)
    perm = torch.randperm(n, device=Xtr.device)
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        xb, yb = Xtr[idx], Ytr[idx]
        if adv:
            xb = C.pgd(model, xb, yb, eps=adv_eps, steps=adv_steps,
                       alpha=2.5 * adv_eps / adv_steps)
        opt_.zero_grad()
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        loss.backward()
        opt_.step()
    model.eval()


def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)

    # probe set for MI: held-out, fixed across STD / AT
    Xp = Xte[:N_PROBE]
    Yp = Yte[:N_PROBE]

    # ---- STD model ----
    std = C.build_model("cnn", meta, seed=seed)
    C.train_model(std, Xtr, Ytr, epochs=EPOCHS, opt="sgd", lr=0.05,
                  ncls=meta["n_classes"])
    std_mi = estimate_mi(std, Xp, Yp, seed=seed)
    std_pgd = C.attack_success(std, Xte, Yte, attack="pgd", eps=EPS,
                               steps=PGD_STEPS)["asr"]
    _, std_clean = C.logits_and_acc(std, Xte, Yte)

    # ---- AT model with per-epoch trajectory ----
    at = C.build_model("cnn", meta, seed=seed)
    opt_at = C.make_optimizer(at, "sgd", 0.05)
    traj = []
    for ep in range(EPOCHS):
        train_one_epoch(at, Xtr, Ytr, opt_at, meta["n_classes"], adv=True,
                        adv_eps=EPS, adv_steps=7)
        mi_ep = estimate_mi(at, Xp, Yp, seed=seed * 100 + ep)
        asr_ep = C.attack_success(at, Xte, Yte, attack="pgd", eps=EPS,
                                  steps=PGD_STEPS)["asr"]
        traj.append({"epoch": ep + 1, "I_xz_bin": mi_ep["I_xz_bin"],
                     "I_zy_bin": mi_ep["I_zy_bin"],
                     "I_xz_mine": mi_ep["I_xz_mine"],
                     "pgd_asr": asr_ep})
    at_mi = {k: traj[-1][k] for k in ("I_xz_bin", "I_zy_bin", "I_xz_mine")}
    at_pgd = traj[-1]["pgd_asr"]
    _, at_clean = C.logits_and_acc(at, Xte, Yte)

    # within-trajectory rank corr (epochs as units)
    rho = spearman([t["I_xz_bin"] for t in traj], [t["pgd_asr"] for t in traj])

    return {
        "seed": seed,
        "std": {"I_xz_bin": std_mi["I_xz_bin"], "I_zy_bin": std_mi["I_zy_bin"],
                "I_xz_mine": std_mi["I_xz_mine"], "pgd_asr": std_pgd,
                "clean_acc": std_clean},
        "at": {"I_xz_bin": at_mi["I_xz_bin"], "I_zy_bin": at_mi["I_zy_bin"],
               "I_xz_mine": at_mi["I_xz_mine"], "pgd_asr": at_pgd,
               "clean_acc": at_clean},
        "delta": {"I_xz_bin": at_mi["I_xz_bin"] - std_mi["I_xz_bin"],
                  "I_zy_bin": at_mi["I_zy_bin"] - std_mi["I_zy_bin"],
                  "I_xz_mine": at_mi["I_xz_mine"] - std_mi["I_xz_mine"]},
        "trajectory": traj,
        "epoch_rho_Ixz_asr": rho,
    }


def main():
    print("=" * 78)
    print("H509 - Information Bottleneck view of adversarial training")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  epochs={EPOCHS}")
    print(f"probe_N={N_PROBE}  PCA_dim={PCA_DIM}  bins/axis={N_BINS}  "
          f"MINE_steps={MINE_STEPS}")
    print("Citations: Tishby & Zaslavsky 2015 (IB); Belghazi+ 2018 (MINE);")
    print("           Achille & Soatto 2018 (invariance / disentanglement).")
    print()

    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"\n[seed {s}]  ({r['runtime_s']}s)")
        print(f"  STD : I(X;Z)_bin={r['std']['I_xz_bin']:.3f}  "
              f"I(Z;Y)_bin={r['std']['I_zy_bin']:.3f}  "
              f"I(X;Z)_MINE={r['std']['I_xz_mine']:.3f}  "
              f"PGD_ASR={r['std']['pgd_asr']:.3f}  "
              f"clean={r['std']['clean_acc']:.3f}")
        print(f"  AT  : I(X;Z)_bin={r['at']['I_xz_bin']:.3f}  "
              f"I(Z;Y)_bin={r['at']['I_zy_bin']:.3f}  "
              f"I(X;Z)_MINE={r['at']['I_xz_mine']:.3f}  "
              f"PGD_ASR={r['at']['pgd_asr']:.3f}  "
              f"clean={r['at']['clean_acc']:.3f}")
        print(f"  AT-STD: dI(X;Z)_bin={r['delta']['I_xz_bin']:+.3f}  "
              f"dI(Z;Y)_bin={r['delta']['I_zy_bin']:+.3f}  "
              f"dI(X;Z)_MINE={r['delta']['I_xz_mine']:+.3f}")
        print(f"  AT epoch trajectory (epoch, I(X;Z)_bin, I(Z;Y)_bin, "
              f"I(X;Z)_MINE, PGD_ASR):")
        for t in r["trajectory"]:
            print(f"     ep{t['epoch']}: {t['I_xz_bin']:.3f}  "
                  f"{t['I_zy_bin']:.3f}  {t['I_xz_mine']:.3f}  "
                  f"{t['pgd_asr']:.3f}")
        print(f"  epoch rank-corr(I(X;Z)_bin , PGD_ASR) over AT trajectory: "
              f"rho={r['epoch_rho_Ixz_asr']:+.3f}")

    # ---- aggregate ----
    def m(side, key):
        v = [r[side][key] for r in rows]
        return sum(v) / len(v)

    def md(key):
        v = [r["delta"][key] for r in rows]
        return sum(v) / len(v)

    print("\n" + "=" * 78)
    print("MEANS across seeds")
    print(f"  STD : I(X;Z)_bin={m('std', 'I_xz_bin'):.3f}  "
          f"I(Z;Y)_bin={m('std', 'I_zy_bin'):.3f}  "
          f"I(X;Z)_MINE={m('std', 'I_xz_mine'):.3f}  "
          f"PGD_ASR={m('std', 'pgd_asr'):.3f}  "
          f"clean={m('std', 'clean_acc'):.3f}")
    print(f"  AT  : I(X;Z)_bin={m('at', 'I_xz_bin'):.3f}  "
          f"I(Z;Y)_bin={m('at', 'I_zy_bin'):.3f}  "
          f"I(X;Z)_MINE={m('at', 'I_xz_mine'):.3f}  "
          f"PGD_ASR={m('at', 'pgd_asr'):.3f}  "
          f"clean={m('at', 'clean_acc'):.3f}")
    print(f"  delta(AT - STD): I(X;Z)_bin={md('I_xz_bin'):+.3f}  "
          f"I(Z;Y)_bin={md('I_zy_bin'):+.3f}  "
          f"I(X;Z)_MINE={md('I_xz_mine'):+.3f}")
    rhos = [r["epoch_rho_Ixz_asr"] for r in rows
            if r["epoch_rho_Ixz_asr"] == r["epoch_rho_Ixz_asr"]]
    print(f"  mean epoch rank-corr(I(X;Z)_bin , PGD_ASR) = "
          f"{(sum(rhos) / len(rhos) if rhos else float('nan')):+.3f}")
    print("=" * 78)
    print("HEADLINE VERDICT (to be filled in by the actual run):")
    print("  The IB hypothesis predicts:")
    print("    (i)  dI(X;Z) = I(X;Z)_AT - I(X;Z)_STD  <  0   (AT compresses input)")
    print("    (ii) dI(Z;Y) ~ 0  or  > 0                     (AT preserves label info)")
    print("    (iii) within the AT trajectory, lower I(X;Z) tracks lower PGD_ASR")
    print("          => positive rank correlation epoch-wise (both shrinking).")
    print("  We require AGREEMENT between the binning estimator and MINE on the")
    print("  SIGN of dI(X;Z) before claiming support; either estimator alone is")
    print("  not trustworthy in absolute nats.")
    print("=" * 78)
    print("LIMITATIONS (Saxe 2018; Goldfeld 2019):")
    print("  * Deterministic ReLU nets have I(X;Z)=H(X) in theory; both estimators")
    print("    treat the discretised / approximated version, so absolute numbers")
    print("    are quantisation artefacts. Only the AT-vs-STD CONTRAST is reported.")
    print("  * 4-d PCA + 8 bins discards most of the 256-d geometry; MINE on the")
    print("    full features is the cross-check.")
    print("  * MINE is itself a biased lower bound (DV) and depends on the")
    print("    capacity of the statistics network; we use 600 SGD steps + EMA")
    print("    bias correction per Belghazi 2018 §3.2.")
    print("  * Epoch correlation uses 6 AT epochs -> very low-N test, treat rho")
    print("    as descriptive only.")
    print("=" * 78)


if __name__ == "__main__":
    main()
