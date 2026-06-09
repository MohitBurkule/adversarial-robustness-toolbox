"""
H484 - Robust features live in a small set of conv channels (PGD-AT SmallCNN).

CAMPAIGN_GAP_MAP §5 / §3 G8 mechanistic seed: identify *which* conv channels in
an adversarially-trained SmallCNN carry the "robust" signal. The strong form of
the hypothesis is that a small fraction (<30%) of conv channels is responsible
for the majority of the robust signal: zero-ablating the top-k robust-important
channels collapses PGD robust accuracy, whereas zero-ablating a random k of the
same size, or the top-k by L2 weight norm, does not.

Operational definition (per advisor critique - "robust feature" is overloaded):
  A channel c is "robust-important" iff zero-ablating it raises PGD-ASR by at
  least delta (we report the full ASR-delta distribution; ranking is by ASR
  delta, NOT by an a-priori notion of "robustness").

This gives us a behavioural / causal definition (Ilyas et al. 2019 "Adversarial
Examples are Not Bugs, They Are Features" treat robust vs non-robust features
as data-distributional; we instead probe which *learned* channels causally
encode that signal). Engstrom et al. 2019 "Robustness May Be at Odds with
Accuracy" (appendix) shows AT models learn qualitatively different features and
gradients - this hypothesis tests whether those features are *channel-localised*
rather than spread uniformly. Allen-Zhu & Li 2021 "Feature Purification" gives a
theoretical motivation: AT acts as a purifier, projecting features onto a small
"clean" subspace, which would predict sparsity of robust channels.

Controls:
  (1) train STD and PGD-AT SmallCNNs (same arch, same seed).
  (2) per-channel zero-ablation -> ASR-delta for each of the 32+64+128=224 conv
      channels (PGD-AT model), GLOBAL ranking + per-layer.
  (3) cumulative top-k ablation curve (sorted by ASR-delta).
  (4) baseline: top-k by L2 weight norm of the conv kernel (weight-magnitude
      saliency, the classic "structured pruning" criterion).
  (5) baseline: random-k channels.
  (6) per-class effect of the global top-3 channels (which classes lose most
      robust accuracy when those channels die?).
  (7) repeat (2)-(3) on the STD model: if the effect is AT-specific we expect a
      flatter ablation curve (no concentrated robust subspace to destroy).

Headline verdict at the end. DO NOT execute this from main session.
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
TOPK_FRAC = 0.30           # "small fraction" upper bound from the hypothesis
PER_CLASS_TOPN = 3         # how many global-top channels to probe per class
RANDOM_REPEATS = 5         # random-baseline replicates per k

# Indices of the three Conv2d layers inside SmallCNN.features (block = [Conv,
# BN, ReLU, MaxPool]); confirmed by reading campaign/common.SmallCNN.
CONV_IDX = [0, 4, 8]


# ---------------------------------------------------------------------------
# ablation hook utilities
# ---------------------------------------------------------------------------
class ChannelAblator:
    """Context manager that zero-ablates a set of (layer, channel) pairs by
    forward-hooks on the corresponding Conv2d modules. Ablation is applied to
    the conv OUTPUT (before BN/activation), which zeros the entire feature
    map for that channel and propagates to downstream layers."""

    def __init__(self, model, ablations):
        # ablations: dict {layer_idx_in_features: list[int channel indices]}
        self.model = model
        self.ablations = {k: list(v) for k, v in ablations.items()}
        self.handles = []

    def __enter__(self):
        for li, chans in self.ablations.items():
            if not chans:
                continue
            mod = self.model.features[li]
            chans_t = torch.tensor(chans, dtype=torch.long)

            def make_hook(ct):
                def hook(_m, _inp, out):
                    out = out.clone()
                    out[:, ct.to(out.device), :, :] = 0.0
                    return out
                return hook

            self.handles.append(mod.register_forward_hook(make_hook(chans_t)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


def conv_layer_widths(model):
    return [model.features[li].out_channels for li in CONV_IDX]


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def clean_acc(model, X, Y, batch=512):
    _, a = C.logits_and_acc(model, X, Y, batch=batch)
    return a


def pgd_asr(model, X, Y, eps=EPS, steps=PGD_STEPS, batch=256):
    """ASR over originally-correct samples wrt the CURRENT model (i.e. with any
    hooks active). We re-evaluate correctness under the hooked model so the ASR
    delta isolates the change in adversarial behaviour, not just clean-acc loss.
    """
    model.eval()
    flips, corr = [], []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        with torch.no_grad():
            c = model(x).argmax(1) == y
        xa = C.pgd(model, x, y, eps, steps)
        with torch.no_grad():
            f = model(xa).argmax(1) != y
        flips.append(f.cpu()); corr.append(c.cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr).numpy().astype(bool)
    asr = float(flips[corr].mean()) if corr.sum() > 0 else float("nan")
    return asr, corr, flips


# ---------------------------------------------------------------------------
# per-channel ablation sweep
# ---------------------------------------------------------------------------
def per_channel_asr_deltas(model, X, Y, base_asr):
    """For every (layer, channel), zero-ablate it alone and record (asr - base_asr).
    Returns a list of dicts with global ordering metadata."""
    rows = []
    widths = conv_layer_widths(model)
    for li, w in zip(CONV_IDX, widths):
        for c in range(w):
            with ChannelAblator(model, {li: [c]}):
                asr, _, _ = pgd_asr(model, X, Y)
            rows.append({
                "layer_idx": li, "layer_pos": CONV_IDX.index(li),
                "channel": c, "width": w,
                "asr": asr, "asr_delta": asr - base_asr,
            })
    return rows


def l2_weight_norms(model):
    """L2 norm of each conv kernel (per output channel)."""
    rows = []
    for li in CONV_IDX:
        W = model.features[li].weight.detach()        # [out, in, kh, kw]
        norms = W.flatten(1).norm(dim=1).cpu().numpy()
        for c, v in enumerate(norms):
            rows.append({"layer_idx": li, "layer_pos": CONV_IDX.index(li),
                         "channel": c, "l2": float(v)})
    return rows


# ---------------------------------------------------------------------------
# cumulative top-k ablation curves
# ---------------------------------------------------------------------------
def ablate_set(model, X, Y, channel_list):
    """channel_list: list of (layer_idx, channel)."""
    bucket = {}
    for li, c in channel_list:
        bucket.setdefault(li, []).append(c)
    with ChannelAblator(model, bucket):
        asr, _, _ = pgd_asr(model, X, Y)
        cln = clean_acc(model, X, Y)
    return asr, cln


def topk_curve(model, X, Y, ranked_channels, ks, base_asr, base_cln):
    """ranked_channels: list of (layer_idx, channel) sorted by importance, most
    important FIRST. Returns list of dicts for each k in ks."""
    out = []
    for k in ks:
        sel = ranked_channels[:k]
        asr, cln = ablate_set(model, X, Y, sel)
        out.append({"k": k, "asr": asr, "asr_delta": asr - base_asr,
                    "clean_acc": cln, "clean_delta": cln - base_cln})
    return out


def random_curve(model, X, Y, all_channels, ks, base_asr, base_cln, repeats, rng):
    """Random-k ablation; averaged over `repeats` independent draws."""
    out = []
    for k in ks:
        accs, asrs = [], []
        for _ in range(repeats):
            idx = rng.choice(len(all_channels), size=k, replace=False)
            sel = [all_channels[i] for i in idx]
            asr, cln = ablate_set(model, X, Y, sel)
            asrs.append(asr); accs.append(cln)
        out.append({"k": k, "asr_mean": float(np.mean(asrs)),
                    "asr_std": float(np.std(asrs)),
                    "asr_delta_mean": float(np.mean(asrs)) - base_asr,
                    "clean_acc_mean": float(np.mean(accs)),
                    "clean_delta_mean": float(np.mean(accs)) - base_cln})
    return out


# ---------------------------------------------------------------------------
# per-class probe
# ---------------------------------------------------------------------------
def per_class_robust_acc(model, X, Y, ncls):
    """Robust accuracy (1 - per-sample-flip on originally-correct) bucketed by
    true class. Returns array shape (ncls,) of robust acc per class."""
    asr, corr, flips = pgd_asr(model, X, Y)
    Y_np = Y.cpu().numpy()
    out = np.full(ncls, np.nan)
    for c in range(ncls):
        m = (Y_np == c)
        if m.sum() == 0:
            continue
        # robust = correct AND not flipped
        rob = (corr & ~flips)[m]
        out[c] = float(rob.mean())
    return out


# ---------------------------------------------------------------------------
# main per-seed routine
# ---------------------------------------------------------------------------
def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    ncls = meta["n_classes"]
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=2000, seed=seed)

    # --- STD model ---
    std = C.build_model("cnn", meta, seed=seed)
    C.train_model(std, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05, ncls=ncls)

    # --- PGD-AT model ---
    at = C.build_model("cnn", meta, seed=seed)
    C.train_model(at, Xtr, Ytr, epochs=8, opt="sgd", lr=0.05, ncls=ncls,
                  adv_train=True, adv_eps=EPS, adv_steps=7)

    rep = {"seed": seed}
    for tag, model in [("std", std), ("at", at)]:
        base_cln = clean_acc(model, Xte, Yte)
        base_asr, _, _ = pgd_asr(model, Xte, Yte)
        rep[f"{tag}_base_clean"] = base_cln
        rep[f"{tag}_base_asr"] = base_asr

        # (2) per-channel ASR deltas
        per_ch = per_channel_asr_deltas(model, Xte, Yte, base_asr)
        # rank globally by ASR-delta (descending)
        ranked = sorted(per_ch, key=lambda r: r["asr_delta"], reverse=True)
        ranked_pairs = [(r["layer_idx"], r["channel"]) for r in ranked]
        rep[f"{tag}_top10_channels"] = [
            {"layer_pos": r["layer_pos"], "channel": r["channel"],
             "asr_delta": r["asr_delta"]} for r in ranked[:10]]

        # per-layer top-1 ASR-delta and means
        per_layer = {p: [] for p in range(len(CONV_IDX))}
        for r in per_ch:
            per_layer[r["layer_pos"]].append(r["asr_delta"])
        rep[f"{tag}_per_layer_mean_asr_delta"] = {
            p: float(np.mean(v)) for p, v in per_layer.items()}
        rep[f"{tag}_per_layer_max_asr_delta"] = {
            p: float(np.max(v)) for p, v in per_layer.items()}

        # (3) cumulative top-k curve by ASR-delta
        total = sum(conv_layer_widths(model))
        ks = sorted(set([1, 2, 3, 5, 10, 20,
                         max(1, int(0.05 * total)),
                         max(1, int(0.10 * total)),
                         max(1, int(0.20 * total)),
                         max(1, int(TOPK_FRAC * total)),
                         max(1, int(0.50 * total))]))
        topk = topk_curve(model, Xte, Yte, ranked_pairs, ks, base_asr, base_cln)
        rep[f"{tag}_topk_asr_curve"] = topk

        # (4) baseline: rank by L2 weight norm (largest first)
        l2 = l2_weight_norms(model)
        l2_ranked = sorted(l2, key=lambda r: r["l2"], reverse=True)
        l2_pairs = [(r["layer_idx"], r["channel"]) for r in l2_ranked]
        rep[f"{tag}_l2_topk_curve"] = topk_curve(
            model, Xte, Yte, l2_pairs, ks, base_asr, base_cln)

        # (5) random-k baseline
        rng = np.random.RandomState(1000 + seed)
        all_pairs = [(r["layer_idx"], r["channel"]) for r in per_ch]
        rep[f"{tag}_random_topk_curve"] = random_curve(
            model, Xte, Yte, all_pairs, ks, base_asr, base_cln,
            repeats=RANDOM_REPEATS, rng=rng)

        # (6) per-class effect of global top-PER_CLASS_TOPN
        base_per_class = per_class_robust_acc(model, Xte, Yte, ncls)
        with ChannelAblator(model, _group(ranked_pairs[:PER_CLASS_TOPN])):
            abl_per_class = per_class_robust_acc(model, Xte, Yte, ncls)
        rep[f"{tag}_per_class_robust_acc_base"] = base_per_class.tolist()
        rep[f"{tag}_per_class_robust_acc_top{PER_CLASS_TOPN}_ablated"] = abl_per_class.tolist()
        rep[f"{tag}_per_class_robust_acc_drop"] = (
            base_per_class - abl_per_class).tolist()

        rep[f"{tag}_total_channels"] = total
        rep[f"{tag}_ks"] = ks

    return rep


def _group(pairs):
    out = {}
    for li, c in pairs:
        out.setdefault(li, []).append(c)
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _fmt_curve(curve, key_asr="asr", key_cln="clean_acc"):
    lines = []
    for r in curve:
        if "asr_mean" in r:
            lines.append(f"    k={r['k']:>3}  asr={r['asr_mean']:.3f}+-{r['asr_std']:.3f}"
                         f"  clean={r['clean_acc_mean']:.3f}"
                         f"  (asr-delta={r['asr_delta_mean']:+.3f})")
        else:
            lines.append(f"    k={r['k']:>3}  asr={r[key_asr]:.3f}  clean={r[key_cln]:.3f}"
                         f"  (asr-delta={r['asr_delta']:+.3f})")
    return "\n".join(lines)


def main():
    print("=" * 78)
    print("H484 - Robust features concentrated in a small set of conv channels?")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  pgd_steps={PGD_STEPS}")
    print(f"Hypothesis: in a PGD-AT SmallCNN, <{int(TOPK_FRAC*100)}% of conv channels")
    print(f"carry the majority of the robust signal. Operational definition: a")
    print(f"channel is robust-important if zero-ablating it raises PGD-ASR by delta.")
    print()

    all_reps = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        all_reps.append(r)

        print("-" * 78)
        print(f"[seed {s}]  ({r['runtime_s']}s)  total_conv_channels (AT)={r['at_total_channels']}")
        for tag in ["std", "at"]:
            print(f"\n  {tag.upper()} model:")
            print(f"    base clean acc = {r[f'{tag}_base_clean']:.3f}   "
                  f"base PGD-ASR = {r[f'{tag}_base_asr']:.3f}")
            print(f"    top-10 channels by ASR-delta (zero-one-channel):")
            for h in r[f"{tag}_top10_channels"]:
                print(f"      layer{h['layer_pos']}.ch{h['channel']:>3}   "
                      f"asr_delta={h['asr_delta']:+.3f}")
            print(f"    per-layer mean ASR-delta: "
                  f"{ {k: round(v, 4) for k, v in r[f'{tag}_per_layer_mean_asr_delta'].items()} }")
            print(f"    per-layer max  ASR-delta: "
                  f"{ {k: round(v, 4) for k, v in r[f'{tag}_per_layer_max_asr_delta'].items()} }")
            print(f"    cumulative top-k ablation curve (by ASR-delta ranking):")
            print(_fmt_curve(r[f"{tag}_topk_asr_curve"]))
            print(f"    L2-weight-norm-ranked top-k (CONTROL):")
            print(_fmt_curve(r[f"{tag}_l2_topk_curve"]))
            print(f"    random-k baseline (mean +- std over {RANDOM_REPEATS} draws):")
            print(_fmt_curve(r[f"{tag}_random_topk_curve"]))
            base_pc = r[f"{tag}_per_class_robust_acc_base"]
            abl_pc = r[f"{tag}_per_class_robust_acc_top{PER_CLASS_TOPN}_ablated"]
            drop = r[f"{tag}_per_class_robust_acc_drop"]
            print(f"    per-class robust acc, base -> ablate top-{PER_CLASS_TOPN}:")
            for c in range(len(base_pc)):
                print(f"      class {c}: {base_pc[c]:.3f} -> {abl_pc[c]:.3f}   "
                      f"(drop {drop[c]:+.3f})")

    # -------- aggregate verdict --------
    def mean_curve(curves):
        # list of curves (one per seed); each is list of dicts with same ks
        out = []
        for j in range(len(curves[0])):
            ks = curves[0][j]["k"]
            ad_vals = []
            for cv in curves:
                ad_vals.append(cv[j].get("asr_delta", cv[j].get("asr_delta_mean", float("nan"))))
            out.append((ks, float(np.nanmean(ad_vals))))
        return out

    print()
    print("=" * 78)
    print("AGGREGATE across seeds  (mean ASR-delta vs k for AT model)")
    print("=" * 78)
    top_curves_at = [r["at_topk_asr_curve"] for r in all_reps]
    l2_curves_at = [r["at_l2_topk_curve"] for r in all_reps]
    rnd_curves_at = [r["at_random_topk_curve"] for r in all_reps]
    top_curves_std = [r["std_topk_asr_curve"] for r in all_reps]

    mc_at_top = mean_curve(top_curves_at)
    mc_at_l2 = mean_curve(l2_curves_at)
    mc_at_rnd = mean_curve(rnd_curves_at)
    mc_std_top = mean_curve(top_curves_std)

    print(f"{'k':>5} | {'AT top-asr':>11} | {'AT top-L2':>10} | {'AT random':>10} | {'STD top-asr':>11}")
    for i, (k, _) in enumerate(mc_at_top):
        print(f"{k:>5} | {mc_at_top[i][1]:>+11.3f} | {mc_at_l2[i][1]:>+10.3f} | "
              f"{mc_at_rnd[i][1]:>+10.3f} | {mc_std_top[i][1]:>+11.3f}")

    total_at = all_reps[0]["at_total_channels"]
    k_at_30 = max(1, int(TOPK_FRAC * total_at))
    # find index of the k closest to TOPK_FRAC*total
    ks_at = all_reps[0]["at_ks"]
    j30 = min(range(len(ks_at)), key=lambda j: abs(ks_at[j] - k_at_30))
    at_top_30 = mc_at_top[j30][1]
    at_l2_30 = mc_at_l2[j30][1]
    at_rnd_30 = mc_at_rnd[j30][1]

    print()
    print("=" * 78)
    print("HEADLINE")
    print("=" * 78)
    confirmed = (at_top_30 > 0.10) and (at_top_30 - at_rnd_30 > 0.05) and (at_top_30 - at_l2_30 > 0.05)
    verdict = "CONFIRMED" if confirmed else "NOT CONFIRMED"
    print(f"At k = {ks_at[j30]} (~{int(TOPK_FRAC*100)}% of {total_at} conv channels) in PGD-AT:")
    print(f"  ASR-delta from top-ASR-rank ablation : {at_top_30:+.3f}")
    print(f"  ASR-delta from top-L2-norm ablation  : {at_l2_30:+.3f}  (weight-mag control)")
    print(f"  ASR-delta from random-k ablation     : {at_rnd_30:+.3f}  (random control)")
    print(f"VERDICT: {verdict} - robust-feature concentration "
          f"({'small subset suffices' if confirmed else 'no concentrated subset'}).")
    print()
    print("Interpretation: a strongly positive ASR-delta for the top-ranked")
    print("subset, with both random-k and L2-rank controls remaining near zero,")
    print("indicates the AT model has learned a *channel-localised* robust")
    print("subspace - consistent with Allen-Zhu & Li (Feature Purification) and")
    print("with Engstrom et al. (AT learns qualitatively different features).")
    print("A flatter STD-model curve at the same k indicates the effect is AT-")
    print("specific rather than a generic pruning artefact, supporting Ilyas et")
    print("al.'s framing that robust features are a distinct subset.")
    print("=" * 78)


if __name__ == "__main__":
    main()
