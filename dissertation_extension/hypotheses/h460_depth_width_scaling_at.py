"""
H460 - Depth-width scaling under adversarial training (gap G4).

Background / critique
---------------------
Bubeck & Sellke (NeurIPS 2021, "A Universal Law of Robustness via
Isoperimetry") proved that smooth Lipschitz interpolation of n points in
d dimensions with a model of p parameters requires p >= n * d, i.e.
robust fits demand overparameterisation by a factor of d versus standard
interpolation. Madry et al. (ICLR 2018, "Towards Deep Learning Models
Resistant to Adversarial Attacks") empirically confirmed that PGD-AT
robust accuracy on CIFAR-10 rises monotonically with capacity (depth/
width). Singh et al. (NeurIPS 2023, "Revisiting Adversarial Training for
ImageNet: Architectures, Training and Generalization across Threat
Models") report modern scaling laws for ImageNet AT and show width tends
to dominate depth at fixed FLOPs. Wu & Xia (2021) also studied this axis
on CIFAR.

Open question at H460: does the depth-vs-width scaling story still appear
at the campaign's Fashion-MNIST / N_train=6000 sub-scale regime, or does
the curve flatten/saturate because (a) the task is too easy and (b)
overparameterisation is hard to reach when n is small (Bubeck-Sellke
predicts the *threshold* is at p ~ n*d). H217 already swept 6 (depth,
width) settings at ~constant params on a CNN; this script runs the full
4x4 grid on a small MLP under PGD-AT and reports params + clean acc +
PGD ASR + train time per cell, so the depth-vs-width axis can be read
along the anti-diagonal (constant-params slice).

Why an MLP and not the CNN: 16 cells * 10 epochs of PGD-10 AT is heavy.
A small MLP at depth in {2,4,6,8} and width in {8,16,32,64} keeps the
biggest cell to ~200K params, well inside RTX 4090 budget and avoids
having any cell drown the runtime.

Setup
-----
- Dataset: Fashion-MNIST, N_train=6000, N_eval=2000.
- Optimizer: SGD(lr=0.05, mom=0.9, wd=5e-4), cosine schedule, 10 epochs,
  batch 128.
- AT: PGD-10, eps=0.1, alpha=0.01, random start.
- Architecture: MLP with `depth` hidden layers, hidden width `width`,
  ReLU, no batch-norm (keeps the scaling law clean - BN under AT is a
  separate axis, see H458).
- Grid: depth in {2,4,6,8} x width in {8,16,32,64} (16 cells).
- Constant-params slice = anti-diagonal: depth*width^2 = const, e.g.
  (d=8,w=8) ~ 4*64=256K params? Reported per cell. The natural
  constant-p anti-diagonal is roughly (d=2,w=64), (d=4,w~45),
  (d=8,w=32). We report ALL 16 cells; the analysis section reads the
  trends both along constant-depth rows (width effect) and constant-
  width columns (depth effect).
- Reported per cell: total params, clean acc, FGSM ASR, PGD ASR,
  train time, robust margin mean. Flushed to disk after each cell so a
  partial run is recoverable.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config --------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01

DEPTHS = [2, 4, 6, 8]
WIDTHS = [8, 16, 32, 64]

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h460_depth_width_scaling_at_output.txt",
)


# ---------------------------------------------------------------------------
# small MLP: `depth` hidden layers each of `width`, ReLU, no BN.
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, depth, width):
        super().__init__()
        layers = [nn.Flatten()]
        d_in = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.ReLU(inplace=True)]
            d_in = width
        layers.append(nn.Linear(d_in, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# PGD-AT trainer (mirrors common.train_model but with explicit wd=5e-4
# per the standard campaign config; cosine schedule).
# ---------------------------------------------------------------------------
def train_pgd_at(model, Xtr, Ytr, epochs, lr, batch, eps, steps, alpha):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.pgd(model, xb, yb, eps=eps, steps=steps, alpha=alpha)
            opt.zero_grad()
            loss = F.cross_entropy(model(xa), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H460  Depth-width scaling under PGD-AT (Fashion-MNIST)")
    out("=" * 80)
    out("anchor: Bubeck-Sellke 2021 (NeurIPS); Singh 2023 (NeurIPS);")
    out("        Madry 2018 (capacity-vs-robustness); Wu-Xia 2021.")
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR}")
    out(f"        BATCH={BATCH} SGD(mom=0.9,wd=5e-4) cosine SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        arch = MLP, depths={DEPTHS}, widths={WIDTHS} (16 cells)")
    out(f"        device = {C.DEVICE}")
    out("")

    # ---- data ----
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    in_dim = META["channels"] * META["size"] * META["size"]

    # ---- pre-print the param grid so partial output already conveys design ----
    out("")
    out("[1] parameter counts per (depth, width) cell (before training):")
    hdr = "depth\\width" + "".join(f"{w:>10}" for w in WIDTHS)
    out("    " + hdr)
    for d in DEPTHS:
        row = f"    d={d:<8}"
        for w in WIDTHS:
            m = MLP(in_dim, META["n_classes"], d, w)
            row += f"{count_params(m):>10,}"
        out(row)
    out("")

    # ---- run grid ----
    cells = []  # list of dict
    out("[2] training PGD-AT per cell (flushed after each cell):")
    out("    {:<10} {:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "depth", "width", "params", "clean", "FGSM", "PGD", "mgn", "time(s)"))
    out("    " + "-" * 88)

    for d in DEPTHS:
        for w in WIDTHS:
            cell_t = time.time()
            C.set_seed(SEED)
            model = MLP(in_dim, META["n_classes"], d, w).to(C.DEVICE)
            p = count_params(model)
            train_pgd_at(model, Xtr, Ytr, EPOCHS, LR, BATCH,
                         EPS, PGD_STEPS, PGD_ALPHA)
            _, clean_acc = C.logits_and_acc(model, Xte, Yte)
            fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
            pg = C.attack_success(model, Xte, Yte, attack="pgd",
                                  eps=EPS, steps=PGD_STEPS)
            mgn = C.margin(model, Xte, Yte)
            dt = time.time() - cell_t

            cells.append({
                "depth": d, "width": w, "params": p,
                "clean": clean_acc, "fgsm": fg["asr"], "pgd": pg["asr"],
                "mgn_mean": float(mgn.mean()), "time_s": dt,
            })
            out("    {:<10} {:<10} {:>10,} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.3f} {:>10.1f}".format(
                d, w, p, clean_acc, fg["asr"], pg["asr"],
                float(mgn.mean()), dt))
            flush_file()

    # ---- PGD ASR table ----
    out("")
    out("[3] PGD ASR grid (lower = more robust):")
    hdr = "depth\\width" + "".join(f"{w:>10}" for w in WIDTHS)
    out("    " + hdr)
    pgd_grid = np.full((len(DEPTHS), len(WIDTHS)), np.nan)
    clean_grid = np.full((len(DEPTHS), len(WIDTHS)), np.nan)
    param_grid = np.full((len(DEPTHS), len(WIDTHS)), np.nan)
    for c in cells:
        i = DEPTHS.index(c["depth"])
        j = WIDTHS.index(c["width"])
        pgd_grid[i, j] = c["pgd"]
        clean_grid[i, j] = c["clean"]
        param_grid[i, j] = c["params"]
    for i, d in enumerate(DEPTHS):
        row = f"    d={d:<8}"
        for j, w in enumerate(WIDTHS):
            row += f"{pgd_grid[i,j]:>10.4f}"
        out(row)

    out("")
    out("[4] Clean acc grid:")
    out("    " + hdr)
    for i, d in enumerate(DEPTHS):
        row = f"    d={d:<8}"
        for j, w in enumerate(WIDTHS):
            row += f"{clean_grid[i,j]:>10.4f}"
        out(row)

    # ---- analysis: width effect at fixed depth, depth effect at fixed width ----
    out("")
    out("[5] AXIS-WISE TRENDS")
    out("-" * 80)
    out("Width effect at fixed depth (PGD ASR, lower = better):")
    for i, d in enumerate(DEPTHS):
        deltas = pgd_grid[i, -1] - pgd_grid[i, 0]
        out(f"  d={d}: PGD@w={WIDTHS[0]:<3}={pgd_grid[i,0]:.3f}  "
            f"PGD@w={WIDTHS[-1]:<3}={pgd_grid[i,-1]:.3f}  "
            f"delta(w_max - w_min)={deltas:+.3f}")
    out("")
    out("Depth effect at fixed width (PGD ASR, lower = better):")
    for j, w in enumerate(WIDTHS):
        deltad = pgd_grid[-1, j] - pgd_grid[0, j]
        out(f"  w={w}: PGD@d={DEPTHS[0]:<3}={pgd_grid[0,j]:.3f}  "
            f"PGD@d={DEPTHS[-1]:<3}={pgd_grid[-1,j]:.3f}  "
            f"delta(d_max - d_min)={deltad:+.3f}")

    # ---- constant-params slice analysis ----
    out("")
    out("[6] CONSTANT-PARAM SLICE  (depth*width^2 ~ const, Singh-2023 axis)")
    out("-" * 80)
    # bucket cells by log-param into ~3 buckets and within each, see if
    # higher width (=> lower depth) wins.
    logp = np.log10(param_grid.flatten())
    pmin, pmax = logp.min(), logp.max()
    edges = np.linspace(pmin, pmax + 1e-6, 4)  # 3 buckets
    out("Buckets by log10(params):")
    for b in range(3):
        lo, hi = 10 ** edges[b], 10 ** edges[b + 1]
        out(f"  bucket {b+1}: params in [{lo:.0f}, {hi:.0f}]")
        in_bucket = [c for c in cells
                     if 10 ** edges[b] <= c["params"] < 10 ** edges[b + 1] + 1]
        in_bucket.sort(key=lambda c: c["width"])  # wider last
        for c in in_bucket:
            out(f"    d={c['depth']} w={c['width']:<3} params={c['params']:>7,}  "
                f"clean={c['clean']:.3f}  PGD={c['pgd']:.3f}")
        if len(in_bucket) >= 2:
            # is wider-shallower better than narrower-deeper inside the bucket?
            wide = max(in_bucket, key=lambda c: c["width"])
            narrow = min(in_bucket, key=lambda c: c["width"])
            if wide["width"] != narrow["width"]:
                out(f"    => within-bucket width-vs-depth PGD delta "
                    f"(wide d={wide['depth']},w={wide['width']} - "
                    f"narrow d={narrow['depth']},w={narrow['width']}) = "
                    f"{wide['pgd']-narrow['pgd']:+.3f}  "
                    f"(negative = wider-shallower is more robust)")

    # ---- VERDICT ----
    out("")
    out("=" * 80)
    out("[7] VERDICT")
    out("=" * 80)
    # Find best (lowest PGD) and worst cell.
    best = min(cells, key=lambda c: c["pgd"])
    worst = max(cells, key=lambda c: c["pgd"])
    span = worst["pgd"] - best["pgd"]
    out(f"  best  cell: d={best['depth']} w={best['width']} "
        f"params={best['params']:,}  clean={best['clean']:.4f} "
        f"PGD={best['pgd']:.4f}")
    out(f"  worst cell: d={worst['depth']} w={worst['width']} "
        f"params={worst['params']:,}  clean={worst['clean']:.4f} "
        f"PGD={worst['pgd']:.4f}")
    out(f"  grid PGD span = {span:.4f}")

    # Mean within-row (width effect) vs within-column (depth effect) range.
    width_ranges = pgd_grid.max(axis=1) - pgd_grid.min(axis=1)
    depth_ranges = pgd_grid.max(axis=0) - pgd_grid.min(axis=0)
    mean_width_range = float(width_ranges.mean())
    mean_depth_range = float(depth_ranges.mean())
    out(f"  mean PGD range across width-at-fixed-depth = {mean_width_range:.4f}")
    out(f"  mean PGD range across depth-at-fixed-width = {mean_depth_range:.4f}")

    # Verdict logic.
    SCALING_THRESH = 0.05  # one-shot single-seed noise band
    if span < SCALING_THRESH:
        verdict = ("NO: at Fashion-MNIST / N_train=6000 / 10 epochs, the PGD ASR "
                   "grid is flat (span < 0.05). Depth-width scaling laws saturate "
                   "below the campaign's resolution.")
    elif mean_width_range > mean_depth_range + 0.02:
        verdict = ("YES (width-dominant): widening at fixed depth changes PGD "
                   "more than deepening at fixed width. Matches Singh 2023's "
                   "ImageNet finding that width is the more efficient robust-"
                   "capacity axis.")
    elif mean_depth_range > mean_width_range + 0.02:
        verdict = ("YES (depth-dominant): deepening at fixed width changes PGD "
                   "more than widening at fixed depth. Opposite of Singh 2023, "
                   "consistent with Fashion-MNIST being a low-dimensional task "
                   "where extra depth still helps representation.")
    else:
        verdict = ("PARTIAL: scaling is visible (PGD span >= 0.05) but neither "
                   "axis dominates clearly. Bubeck-Sellke overparam threshold is "
                   "probably crossed by every cell tested.")
    out("  ONE-LINE VERDICT: " + verdict)

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
