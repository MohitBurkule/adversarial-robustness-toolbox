"""
H508 - PGD perturbations align with the top input-NTK eigenvector; AT decorrelates.

Seed (paper-6 review, sections 5 / 3 G6 / G8): the Neural Tangent Kernel theory
predicts that, for a sufficiently wide / lazy network, function-space dynamics
are governed by the NTK J(x)J(x)^T whose top eigenvectors are the directions
along which the network is most sensitive (Jacot et al. 2018, NeurIPS).
Tsilivis & Kempe ("What Can the Neural Tangent Kernel Tell Us About Adversarial
Robustness?", NeurIPS 2022) make this concrete: adversarial examples in the
NTK regime concentrate on a low-rank "robust/non-robust" decomposition of the
kernel, so a single-step gradient attack should align strongly with the leading
NTK eigen-direction. Loo et al. ("Evolution of the NTK during training",
NeurIPS 2022) show that the after-training NTK preserves this structure on
realistic small CNNs even though the kernel itself drifts.

If this picture is correct on a Fashion-MNIST SmallCNN, then:

  (H1)  Standard training:    cos( PGD direction , top input-NTK eigenvector )
                              > 0.5 on average.
  (H2)  PGD adversarial train: AT decorrelates the attack direction from the
                              dominant kernel direction => cos < 0.3.
  (H3)  Higher cos-sim => smaller min-eps-to-flip (samples whose PGD direction
                              is well aligned with the top eigenvector are the
                              easiest to flip): negative Spearman/Pearson on
                              the STD model.

We measure all three on the SAME 500 test samples for both models, plus a
per-class breakdown. The input-Jacobian-kernel here is the per-sample
linearisation K_x = J(x) J(x)^T where J(x) = dlogits/dx (shape C x D); its top
eigenvector is computed by 20-step power iteration over the D-dim input space
via the rank-C identity v <- J^T J v = J^T (J v) (no need to form K_x).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
EPS = 0.1
PGD_STEPS = 10
N_PROBE = 500          # samples used for alignment / min-eps analysis
POWER_ITERS = 20
EPS_GRID = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3]
COS_HI = 0.5           # H1 threshold for STD
COS_LO = 0.3           # H2 threshold for AT


# ---------------------------------------------------------------------------
# input-Jacobian-kernel top eigenvector via power iteration
# ---------------------------------------------------------------------------
def _jvp(model, x, v):
    """J(x) v  where J = d logits / d x, returned as a (C,) vector. Single sample."""
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        out = model(x.unsqueeze(0)).squeeze(0)          # (C,)
        # J v  = grad_x ( out . s ) . v  is awkward; use jvp via double-backward trick:
        # easier: compute J explicitly for small C (=10) -> still cheap.
    return None  # placeholder; we use explicit Jacobian below


def _input_jacobian(model, x):
    """Full input Jacobian d logits/dx for ONE sample x (shape [C, D])."""
    x = x.detach().clone().requires_grad_(True)
    out = model(x.unsqueeze(0)).squeeze(0)               # (C,)
    C_out = out.numel()
    Js = []
    for c in range(C_out):
        g, = torch.autograd.grad(out[c], x, retain_graph=(c < C_out - 1))
        Js.append(g.flatten())
    return torch.stack(Js, dim=0)                        # (C, D)


def top_eigvec_input_kernel(model, x, iters=POWER_ITERS, seed=0):
    """Top eigenvector of K_x = J(x)^T J(x) in INPUT space (D-dim).

    Note: K_x = J^T J shares non-zero spectrum with J J^T; the top eigenvector
    in input-space is the direction the network is most sensitive to. Power
    iteration uses the rank-C operator v <- J^T (J v).
    """
    J = _input_jacobian(model, x)                        # (C, D)
    D = J.size(1)
    g = torch.Generator(device=J.device).manual_seed(seed)
    v = torch.randn(D, generator=g, device=J.device)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        Jv = J @ v                                       # (C,)
        v = J.t() @ Jv                                   # (D,)
        nrm = v.norm()
        if nrm < 1e-20:
            break
        v = v / nrm
    return v.detach()                                    # (D,)


# ---------------------------------------------------------------------------
# per-sample PGD direction
# ---------------------------------------------------------------------------
def pgd_direction(model, x, y, eps=EPS, steps=PGD_STEPS):
    """Return the NORMALISED perturbation delta = (x_adv - x) / ||x_adv - x||,
    flattened to 1-D, for a single sample.
    """
    xa = C.pgd(model, x.unsqueeze(0), y.unsqueeze(0), eps=eps, steps=steps)
    delta = (xa.squeeze(0) - x).flatten()
    n = delta.norm()
    if n < 1e-20:
        return torch.zeros_like(delta)
    return delta / n


# ---------------------------------------------------------------------------
# min-eps-to-flip via the PGD attack (binary-ish search over a fixed grid)
# ---------------------------------------------------------------------------
def min_eps_to_flip(model, X, Y, eps_grid=EPS_GRID, steps=PGD_STEPS, batch=256):
    """For each sample, the smallest eps in `eps_grid` whose PGD flips the
    sample's clean prediction; +inf-stand-in (eps_grid[-1] * 1.5) if never flipped.
    Vectorised over batch.
    """
    model.eval()
    N = X.size(0)
    out = np.full(N, eps_grid[-1] * 1.5, dtype=np.float32)
    found = np.zeros(N, dtype=bool)
    with torch.no_grad():
        clean = model(X).argmax(1)
    for eps in eps_grid:
        # skip already-flipped
        idx_todo = np.where(~found)[0]
        if idx_todo.size == 0:
            break
        for i0 in range(0, idx_todo.size, batch):
            sub = idx_todo[i0:i0 + batch]
            xs = X[sub]
            ys = Y[sub]
            xa = C.pgd(model, xs, ys, eps=eps, steps=steps)
            with torch.no_grad():
                pred = model(xa).argmax(1)
            ref = clean[sub]
            flipped = (pred != ref).cpu().numpy()
            for k, gidx in enumerate(sub):
                if flipped[k] and not found[gidx]:
                    out[gidx] = eps
                    found[gidx] = True
    return out, found


# ---------------------------------------------------------------------------
# correlation helpers (no scipy dependency)
# ---------------------------------------------------------------------------
def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return float("nan")
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(np.float64)
    rb = np.argsort(np.argsort(b[m])).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12: return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# per-model probe
# ---------------------------------------------------------------------------
def probe_model(model, Xp, Yp, seed=0):
    model.eval()
    N = Xp.size(0)
    cos = np.zeros(N, dtype=np.float32)
    # PGD direction (vectorised) + min-eps (vectorised over batches)
    # but cos-sim is per-sample because each sample has its own eigenvector.
    # We use clean predictions as reference labels for PGD direction.
    with torch.no_grad():
        clean = model(Xp).argmax(1)
    for i in range(N):
        x = Xp[i]
        y_ref = clean[i]
        d = pgd_direction(model, x, y_ref, eps=EPS, steps=PGD_STEPS)
        v = top_eigvec_input_kernel(model, x, iters=POWER_ITERS, seed=seed + i)
        # cosine sim: take absolute value (eigenvector sign is arbitrary)
        denom = (d.norm() * v.norm()).clamp_min(1e-20)
        cs = float((d @ v) / denom)
        cos[i] = abs(cs)
    min_eps, found = min_eps_to_flip(model, Xp, Yp, eps_grid=EPS_GRID, steps=PGD_STEPS)
    return {
        "cos": cos,
        "min_eps": min_eps,
        "found": found,
        "clean_pred": clean.cpu().numpy(),
    }


def per_class_cos(cos, labels, ncls=10):
    out = []
    for c in range(ncls):
        m = labels == c
        out.append(float(cos[m].mean()) if m.sum() else float("nan"))
    return out


# ---------------------------------------------------------------------------
# per-seed
# ---------------------------------------------------------------------------
def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=2000, seed=seed)

    # standard model
    m_std = C.build_model("cnn", meta, seed=seed)
    C.train_model(m_std, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05, ncls=meta["n_classes"])

    # PGD-AT model (training PGD-7 @ eps=EPS, per common.train_model defaults)
    m_at = C.build_model("cnn", meta, seed=seed)
    C.train_model(m_at, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05, ncls=meta["n_classes"],
                  adv_train=True, adv_eps=EPS, adv_steps=7)

    # 500-sample probe set: deterministic prefix of the eval split.
    n = min(N_PROBE, Xte.size(0))
    Xp, Yp = Xte[:n], Yte[:n]

    res_std = probe_model(m_std, Xp, Yp, seed=seed)
    res_at = probe_model(m_at, Xp, Yp, seed=seed + 10_000)

    labs = Yp.cpu().numpy()
    out = {"seed": seed, "n": n}
    for tag, r in [("std", res_std), ("at", res_at)]:
        out[tag] = {
            "cos_mean": float(np.mean(r["cos"])),
            "cos_median": float(np.median(r["cos"])),
            "cos_std": float(np.std(r["cos"])),
            "cos_frac_gt_0p5": float(np.mean(r["cos"] > 0.5)),
            "cos_frac_lt_0p3": float(np.mean(r["cos"] < 0.3)),
            "min_eps_mean": float(np.mean(r["min_eps"])),
            "frac_flipped_at_max_eps": float(np.mean(r["found"])),
            "pearson_cos_vs_min_eps": _pearson(r["cos"], r["min_eps"]),
            "spearman_cos_vs_min_eps": _spearman(r["cos"], r["min_eps"]),
            "per_class_cos_mean": per_class_cos(r["cos"], labs, meta["n_classes"]),
        }
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("H508 - PGD direction vs top input-NTK eigenvector alignment (STD vs PGD-AT)")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  pgd_steps={PGD_STEPS}  "
          f"power_iters={POWER_ITERS}  n_probe={N_PROBE}")
    print("Refs: Jacot 2018 (NTK), Tsilivis & Kempe 2022 (NTK x adv. robustness),")
    print("      Loo et al. 2022 (NTK evolution during training).")

    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"\n[seed {s}] ({r['runtime_s']}s)  n={r['n']}")
        for tag in ("std", "at"):
            d = r[tag]
            print(f"  {tag.upper():3s}: cos mean={d['cos_mean']:.3f}  median={d['cos_median']:.3f}  "
                  f"std={d['cos_std']:.3f}  >.5={d['cos_frac_gt_0p5']:.2f}  <.3={d['cos_frac_lt_0p3']:.2f}")
            print(f"       min_eps_mean={d['min_eps_mean']:.3f}  "
                  f"frac_flipped(grid)={d['frac_flipped_at_max_eps']:.2f}  "
                  f"r(cos,min_eps)={d['pearson_cos_vs_min_eps']:+.3f}  "
                  f"rho={d['spearman_cos_vs_min_eps']:+.3f}")
            pc = d["per_class_cos_mean"]
            print(f"       per-class cos: " + " ".join(f"{v:.2f}" for v in pc))

    # aggregate across seeds
    def m(tag, key):
        v = [r[tag][key] for r in rows if r[tag][key] == r[tag][key]]
        return sum(v) / len(v) if v else float("nan")

    print("\n" + "=" * 78)
    print("MEANS across seeds")
    for tag in ("std", "at"):
        print(f"  {tag.upper():3s}: cos_mean={m(tag,'cos_mean'):.3f}  "
              f">.5={m(tag,'cos_frac_gt_0p5'):.2f}  <.3={m(tag,'cos_frac_lt_0p3'):.2f}  "
              f"min_eps_mean={m(tag,'min_eps_mean'):.3f}  "
              f"r(cos,min_eps)={m(tag,'pearson_cos_vs_min_eps'):+.3f}  "
              f"rho={m(tag,'spearman_cos_vs_min_eps'):+.3f}")
    print("=" * 78)

    std_cos = m("std", "cos_mean")
    at_cos = m("at", "cos_mean")
    std_rho = m("std", "spearman_cos_vs_min_eps")

    h1 = std_cos > COS_HI
    h2 = at_cos < COS_LO
    h3 = std_rho < -0.1            # negative correlation: higher align => easier flip

    print("HEADLINE")
    print(f"  H1 (STD cos > {COS_HI}):              {std_cos:.3f}  =>  {'SUPPORTED' if h1 else 'NOT SUPPORTED'}")
    print(f"  H2 (AT  cos < {COS_LO}):              {at_cos:.3f}  =>  {'SUPPORTED' if h2 else 'NOT SUPPORTED'}")
    print(f"  H3 (STD rho(cos,min_eps) < -0.1): {std_rho:+.3f}  =>  {'SUPPORTED' if h3 else 'NOT SUPPORTED'}")
    if h1 and h2:
        verdict = ("PGD attacks ride the top input-NTK eigen-direction on STD models; "
                   "adversarial training decorrelates them. Consistent with the "
                   "Tsilivis-Kempe (2022) NTK picture of robustness.")
    elif h1 and not h2:
        verdict = ("STD model shows kernel-aligned attacks but AT does NOT decorrelate "
                   "them on this small CNN -- attack direction may be feature- rather "
                   "than kernel-controlled after AT.")
    elif not h1 and h2:
        verdict = ("STD model already misaligned: the NTK linearisation does not "
                   "predict the attack direction on a non-lazy SmallCNN, so the "
                   "NTK-robustness story (Tsilivis-Kempe) does not transfer here.")
    else:
        verdict = ("Neither STD nor AT match the predicted alignment regimes -- the "
                   "SmallCNN is far from the lazy/NTK regime and the top input-Jacobian "
                   "eigen-direction is not informative about PGD direction.")
    print(f"  VERDICT: {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
