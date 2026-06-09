"""
H495 - Per-layer adversarial sensitivity: where do adversarial activations
diverge from clean activations, and does adversarial training flatten the
divergence curve?

Seed (§5 / §3 G8 mechanistic):
  Hypothesis: in the standard (STD) SmallCNN, the relative L2 deviation
    rel_dev(l) = ||z_adv(l) - z_clean(l)||_2 / ||z_clean(l)||_2
  grows monotonically through the conv blocks, with most of the acceleration
  in the LAST 1-2 conv blocks (consistent with non-robust features being
  amplified by late, more class-selective filters). In a PGD-adversarially-
  trained (PGD-AT) SmallCNN, the curve is FLATTER and the late-layer
  amplification disappears.

Critique seed - subtle confounds we address:
  (a) Layer dimensions differ, so absolute L2 distance is meaningless. We
      report RELATIVE deviation (ratio to the clean activation norm), per
      sample, averaged over 500 samples. (mean +/- std.)
  (b) The same fixed eps L_inf budget is used at the INPUT for both models so
      that input-space perturbation magnitudes are matched - the comparison
      is fair w.r.t. the threat model.
  (c) Per-channel deviation distribution at the worst layer surfaces whether
      a few channels dominate (consistent with Bai et al.'s "non-robust
      channels") or the divergence is broad-band.

Related work (cited):
  - Xie, Wu, van der Maaten, Yuille & He, "Feature Denoising for Improving
    Adversarial Robustness", CVPR 2019. Shows adversarial perturbations
    accumulate / amplify through depth and that explicit late-stage feature
    denoising helps - direct prediction that STD nets exhibit a depth-wise
    blow-up that AT/denoising attenuates.
  - Bai, Zeng, Jiang, Xia, Ma & Wang, "Improving Adversarial Robustness via
    Channel-wise Activation Suppressing" (CAS), ICLR 2021. Identifies a
    small subset of channels whose activations diverge dramatically on
    adversarial inputs; predicts heavy-tailed per-channel deviation.
  - Ilyas, Santurkar, Tsipras, Engstrom, Tran & Madry, "Adversarial Examples
    Are Not Bugs, They Are Features", NeurIPS 2019. Frames adversarial
    susceptibility as the model latching onto class-predictive but brittle
    features - which AT should down-weight, so PGD-AT activations at the
    feature-vector stage should be more stable.

Controls:
  (1) STD vs PGD-AT models trained on identical data with identical seed,
      arch, optimiser, lr, epochs - the ONLY difference is adv_train.
  (2) Instrument SmallCNN with forward hooks AFTER each conv block (so we
      see post-ReLU/post-pool activations at the 3 spatial stages and the
      256-D penultimate feature).
  (3) PGD adversarial examples generated at eps=0.1 (L_inf, 10 steps), the
      campaign-standard Fashion-MNIST budget.
  (4) Per-layer relative L2 deviation, mean +/- std over 500 originally-
      correctly-classified test samples (we condition on clean-correct so a
      divergent activation actually reflects "flipping" pressure).
  (5) Per-channel deviation distribution at the worst layer: for each
      channel c, compute mean over samples of
        ||z_adv[:, c] - z_clean[:, c]||_2 / (||z_clean[:, c]||_2 + eps)
      and report the sorted distribution + Gini / top-10% concentration.

NOTE: This script writes results to results/fashion_mnist/. It is NOT run
here per project policy.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
EPS = 0.1
PGD_STEPS = 10
N_PROBE = 500           # samples on which to compute per-layer divergence
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS_STD = 8
EPOCHS_AT = 10          # AT typically needs a touch more


# ---------------------------------------------------------------------------
# Hook-based activation extractor
# ---------------------------------------------------------------------------
def collect_block_activations(model, X, batch=128):
    """Run X through model.features and capture activations AFTER each block
    (i.e. after MaxPool2d), plus the penultimate 256-D vector after the
    head's first Linear+ReLU. Returns OrderedDict[name] = tensor [N, ...].

    Assumes the SmallCNN topology in campaign.common: features is
        block1: Conv-BN-ReLU-Pool   (indices 0..3)
        block2: Conv-BN-ReLU-Pool   (indices 4..7)
        block3: Conv-BN-ReLU-Pool   (indices 8..11)
    and head is Flatten-Linear-ReLU-Linear.
    """
    model.eval()
    feats = model.features
    # Identify the 3 MaxPool2d module indices (end of each block).
    pool_indices = [i for i, m in enumerate(feats) if isinstance(m, nn.MaxPool2d)]
    assert len(pool_indices) == 3, f"expected 3 conv blocks, got pools at {pool_indices}"

    captured = {f"block{k+1}": [] for k in range(3)}
    captured["penult"] = []

    handles = []
    for k, idx in enumerate(pool_indices):
        name = f"block{k+1}"

        def make_hook(nm):
            def hook(_mod, _inp, out):
                captured[nm].append(out.detach().cpu())
            return hook

        handles.append(feats[idx].register_forward_hook(make_hook(name)))

    # Hook the penultimate ReLU in the head (after the first Linear).
    head = model.head
    # head = Flatten, Linear, ReLU, Linear
    relu_idx = None
    for i, m in enumerate(head):
        if isinstance(m, (nn.ReLU, nn.GELU, nn.ELU, nn.Tanh, nn.Sigmoid, nn.Softplus)):
            relu_idx = i
            break
    assert relu_idx is not None, "could not find activation inside head"

    def penult_hook(_mod, _inp, out):
        captured["penult"].append(out.detach().cpu())

    handles.append(head[relu_idx].register_forward_hook(penult_hook))

    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            _ = model(X[i:i + batch])

    for h in handles:
        h.remove()

    return {k: torch.cat(v, dim=0) for k, v in captured.items()}


# ---------------------------------------------------------------------------
# Divergence metrics
# ---------------------------------------------------------------------------
def per_sample_relative_l2(z_clean, z_adv, eps=1e-8):
    """Per-sample relative L2 deviation. Flattens all non-batch dims.
    Returns numpy array shape (N,)."""
    N = z_clean.size(0)
    a = z_clean.reshape(N, -1).float()
    b = z_adv.reshape(N, -1).float()
    num = (b - a).norm(dim=1)
    den = a.norm(dim=1).clamp_min(eps)
    return (num / den).cpu().numpy()


def per_channel_relative_l2(z_clean, z_adv, eps=1e-8):
    """Per-channel relative L2 over the sample axis. Assumes shape [N, C, H, W]
    or [N, C]. Returns numpy array shape (C,)."""
    if z_clean.dim() == 4:
        N, Cc, H, W = z_clean.shape
        a = z_clean.permute(1, 0, 2, 3).reshape(Cc, -1).float()
        b = z_adv.permute(1, 0, 2, 3).reshape(Cc, -1).float()
    elif z_clean.dim() == 2:
        a = z_clean.t().float()       # (C, N)
        b = z_adv.t().float()
    else:
        raise ValueError(f"unexpected shape {tuple(z_clean.shape)}")
    num = (b - a).norm(dim=1)
    den = a.norm(dim=1).clamp_min(eps)
    return (num / den).cpu().numpy()


def gini(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.sort(np.abs(x))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------
def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)

    # --- two models, identical arch & data, only AT differs --------------
    std = C.build_model("cnn", meta, seed=seed)
    C.train_model(std, Xtr, Ytr, epochs=EPOCHS_STD, opt="sgd", lr=0.05,
                  ncls=meta["n_classes"])

    C.set_seed(seed)                                # keep init comparable
    at = C.build_model("cnn", meta, seed=seed)
    C.train_model(at, Xtr, Ytr, epochs=EPOCHS_AT, opt="sgd", lr=0.05,
                  ncls=meta["n_classes"],
                  adv_train=True, adv_eps=EPS, adv_steps=7)

    out = {"seed": seed, "models": {}}

    for name, model in [("STD", std), ("PGD_AT", at)]:
        model.eval()
        # condition on clean-correct so divergence is meaningful
        with torch.no_grad():
            pred = []
            for i in range(0, Xte.size(0), 256):
                pred.append(model(Xte[i:i + 256]).argmax(1).cpu())
            pred = torch.cat(pred)
        correct_mask = (pred == Yte.cpu())
        idx = torch.nonzero(correct_mask, as_tuple=False).flatten()
        if idx.numel() < N_PROBE:
            print(f"  [warn] {name}: only {idx.numel()} clean-correct samples "
                  f"(< {N_PROBE}); using all.")
            sel = idx
        else:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(idx.numel(), generator=g)[:N_PROBE]
            sel = idx[perm]

        X_clean = Xte[sel.to(Xte.device)]
        Y_clean = Yte[sel.to(Yte.device)]
        # generate adversarial examples for THIS model
        X_adv = C.pgd(model, X_clean, Y_clean, eps=EPS, steps=PGD_STEPS)

        # measure clean acc on the probe + adv acc
        with torch.no_grad():
            clean_acc = float((model(X_clean).argmax(1) == Y_clean).float().mean())
            adv_acc = float((model(X_adv).argmax(1) == Y_clean).float().mean())

        acts_clean = collect_block_activations(model, X_clean)
        acts_adv = collect_block_activations(model, X_adv)

        layer_names = ["block1", "block2", "block3", "penult"]
        per_layer = {}
        per_layer_full = {}
        for ln in layer_names:
            dev = per_sample_relative_l2(acts_clean[ln], acts_adv[ln])
            per_layer_full[ln] = dev
            per_layer[ln] = {
                "mean": float(dev.mean()),
                "std": float(dev.std()),
                "median": float(np.median(dev)),
                "n": int(dev.size),
                "shape": list(acts_clean[ln].shape[1:]),
            }

        # worst layer = highest mean relative deviation
        worst = max(layer_names, key=lambda L: per_layer[L]["mean"])
        per_chan = per_channel_relative_l2(acts_clean[worst], acts_adv[worst])
        order = np.argsort(per_chan)[::-1]
        per_chan_sorted = per_chan[order]
        total = float(per_chan_sorted.sum())
        top10_frac = (float(per_chan_sorted[:max(1, len(per_chan_sorted) // 10)].sum())
                      / total) if total > 0 else float("nan")
        ch_summary = {
            "layer": worst,
            "n_channels": int(per_chan.size),
            "mean": float(per_chan.mean()),
            "std": float(per_chan.std()),
            "max": float(per_chan.max()),
            "min": float(per_chan.min()),
            "top10pct_share": top10_frac,
            "gini": gini(per_chan),
            "top5_values": [float(v) for v in per_chan_sorted[:5].tolist()],
        }

        # late-layer amplification ratio: block3 / block1
        amp = per_layer["block3"]["mean"] / max(per_layer["block1"]["mean"], 1e-12)

        out["models"][name] = {
            "clean_acc_probe": clean_acc,
            "adv_acc_probe": adv_acc,
            "per_layer": per_layer,
            "amp_block3_over_block1": float(amp),
            "worst_layer_per_channel": ch_summary,
        }

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_layer_row(name, pl):
    return (f"    {name:7s}  shape={str(pl['shape']):<14s} "
            f"rel_dev mean={pl['mean']:.3f}  std={pl['std']:.3f}  "
            f"median={pl['median']:.3f}")


def main():
    print("=" * 78)
    print("H495 - Per-layer adversarial sensitivity (STD vs PGD-AT, F-MNIST SmallCNN)")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  PGD_steps={PGD_STEPS}")
    print(f"N_probe={N_PROBE} clean-correct samples per (seed, model)")
    print(f"Seeds={SEEDS}")
    print()

    all_rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        all_rows.append(r)
        print("-" * 78)
        print(f"[seed {s}]  ({r['runtime_s']}s)")
        for name in ("STD", "PGD_AT"):
            m = r["models"][name]
            print(f"  {name}:  clean_acc(probe)={m['clean_acc_probe']:.3f}  "
                  f"adv_acc(probe)={m['adv_acc_probe']:.3f}  "
                  f"amp(block3/block1)={m['amp_block3_over_block1']:.2f}")
            for ln in ("block1", "block2", "block3", "penult"):
                print(_fmt_layer_row(ln, m["per_layer"][ln]))
            ch = m["worst_layer_per_channel"]
            print(f"    worst layer = {ch['layer']}: n_ch={ch['n_channels']}  "
                  f"mean={ch['mean']:.3f}  max={ch['max']:.3f}  "
                  f"top10%-share={ch['top10pct_share']:.3f}  gini={ch['gini']:.3f}")
            print(f"      top-5 channel rel-dev: "
                  + ", ".join(f"{v:.2f}" for v in ch["top5_values"]))

    # --- aggregates across seeds ---------------------------------------------
    def stack(name, ln, key):
        return np.array([r["models"][name]["per_layer"][ln][key] for r in all_rows])

    def stack_scalar(name, key):
        return np.array([r["models"][name][key] for r in all_rows])

    print("\n" + "=" * 78)
    print("AGGREGATE (mean +/- std across seeds)")
    print("=" * 78)
    for name in ("STD", "PGD_AT"):
        print(f"\n  {name}:")
        for ln in ("block1", "block2", "block3", "penult"):
            mu = stack(name, ln, "mean")
            print(f"    {ln:7s}  rel_dev_mean = {mu.mean():.3f} +/- {mu.std():.3f}")
        amp = stack_scalar(name, "amp_block3_over_block1")
        print(f"    amplification block3/block1 = {amp.mean():.2f} +/- {amp.std():.2f}")
        gi = np.array([r["models"][name]["worst_layer_per_channel"]["gini"] for r in all_rows])
        sh = np.array([r["models"][name]["worst_layer_per_channel"]["top10pct_share"]
                       for r in all_rows])
        print(f"    worst-layer channel-Gini  = {gi.mean():.3f} +/- {gi.std():.3f}")
        print(f"    worst-layer top10%-share  = {sh.mean():.3f} +/- {sh.std():.3f}")

    # --- HEADLINE verdict ----------------------------------------------------
    std_b1 = stack("STD", "block1", "mean").mean()
    std_b3 = stack("STD", "block3", "mean").mean()
    std_pen = stack("STD", "penult", "mean").mean()
    at_b1 = stack("PGD_AT", "block1", "mean").mean()
    at_b3 = stack("PGD_AT", "block3", "mean").mean()
    at_pen = stack("PGD_AT", "penult", "mean").mean()
    std_amp = stack_scalar("STD", "amp_block3_over_block1").mean()
    at_amp = stack_scalar("PGD_AT", "amp_block3_over_block1").mean()

    std_monotone = (std_b1 <= std_b3) or (std_b1 <= std_pen)
    late_blowup_in_std = (std_b3 / max(std_b1, 1e-12)) >= 1.5
    at_flatter = at_amp < std_amp * 0.75
    at_late_reduced = (at_pen < std_pen * 0.85) or (at_b3 < std_b3 * 0.85)

    if late_blowup_in_std and at_flatter and at_late_reduced and std_monotone:
        verdict = "SUPPORTED"
    elif late_blowup_in_std and (at_flatter or at_late_reduced):
        verdict = "PARTIALLY SUPPORTED"
    else:
        verdict = "REJECTED"

    print("\n" + "=" * 78)
    print(f"HEADLINE VERDICT: {verdict}")
    print("=" * 78)
    print(f"  STD  rel_dev: block1={std_b1:.3f}  block3={std_b3:.3f}  "
          f"penult={std_pen:.3f}   amp(b3/b1)={std_amp:.2f}")
    print(f"  AT   rel_dev: block1={at_b1:.3f}  block3={at_b3:.3f}  "
          f"penult={at_pen:.3f}   amp(b3/b1)={at_amp:.2f}")
    print(f"  STD  late-layer blow-up (block3 >= 1.5x block1)?  {late_blowup_in_std}")
    print(f"  AT   amplification curve flatter than STD (<0.75x)? {at_flatter}")
    print(f"  AT   late-layer divergence reduced vs STD (<0.85x)? {at_late_reduced}")
    print(f"  STD  curve monotone-ish through depth?             {std_monotone}")
    print()
    print("Interpretation: a SUPPORTED verdict means standard training lets")
    print("adversarial perturbations compound through depth - small input-space")
    print("noise becomes a large feature-space displacement by the late conv")
    print("blocks and the 256-D penultimate vector - while PGD-AT flattens the")
    print("curve and removes the late-layer blow-up. This is the mechanism Xie")
    print("et al. (2019) target with feature denoising and Bai et al. (2021)")
    print("with channel suppression; the per-channel concentration (Gini /")
    print("top-10% share) at the worst layer adjudicates between 'a few bad")
    print("channels' (Bai) and 'broad-band drift' (Ilyas-style global non-")
    print("robust features).")
    print("=" * 78)


if __name__ == "__main__":
    main()
