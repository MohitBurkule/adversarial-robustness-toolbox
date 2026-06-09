"""
H497 - Feature-space distance between adversarial and clean inputs (seed §5 / §3 G8).

Hypothesis
----------
PGD adversarial training (PGD-AT) compresses the *feature-space* distance between
adversarial and clean versions of the same input by >2x relative to a standard
(STD) classifier, WITHOUT collapsing class structure. That is, AT should
simultaneously:

  (a) reduce per-sample ||f(x) - f(x_adv)|| (penultimate features), AND
  (b) preserve (or not catastrophically shrink) the between-class feature
      distance relative to the within-class feature spread.

Equivalently, the *normalised* adversarial drift
        D_ratio = E_x ||f(x) - f(x_adv)|| / mean_c std_within(c)
should be >2x smaller for PGD-AT than for STD, while the between/within
class-distance ratio (a class-separability proxy) stays comparable.

Critique-driven controls
------------------------
The naive seed metric ||f(x) - f(x_adv)|| is highly dependent on layer choice
and on the absolute scale of the penultimate representation: a model whose
features happen to be smaller in norm trivially has a smaller adversarial drift.
We therefore:
  * fix the layer to the penultimate (post-ReLU 256-d) representation of
    SmallCNN, which is consistent across STD/AT models, AND
  * normalise the adversarial drift by the mean within-class feature spread of
    the *same* model, giving a scale-invariant ratio.

Prior art (cited)
-----------------
  * Mao et al., "Metric Learning for Adversarial Robustness" (NeurIPS 2019) -
    formalises adversarial training as a feature-space metric-learning problem:
    pull (x, x_adv) together while keeping inter-class samples apart. Direct
    motivation for measuring per-sample adv drift AND between-class distance
    together.
  * Engstrom et al., "Adversarial Robustness as a Prior for Learned
    Representations" (2019/2020) - shows AT representations are more
    perceptually aligned and that nearest-neighbour structure in feature space
    is more semantically meaningful than for STD models; supports using
    penultimate features as the natural readout.
  * Pang et al., "Boosting Adversarial Training with Hypersphere Embedding"
    (NeurIPS 2020) - explicitly argues AT benefits from compact within-class
    feature clusters and large inter-class angular margins, justifying our
    within/between class-spread normalisation.

Design (controls 1-6)
---------------------
  (1) Train two SmallCNN models on Fashion-MNIST: STD and PGD-AT (eps=0.1,
      7-step PGD inner loop).
  (2) Extract penultimate features for 1000 clean test samples and 1000
      matched adversarial samples (PGD, eps=0.1, 10 steps).
  (3) Per-sample adv drift d_i = ||f(x_i) - f(x_adv_i)||_2.
  (4) Within-class mean feature spread: for each class c, sigma_c = mean over
      samples of class c of ||f(x_i) - mu_c||_2 where mu_c is the clean class
      centroid. Report mean_c sigma_c.
  (5) Between-class mean feature distance: mean over c != c' of ||mu_c - mu_c'||_2.
  (6) Scale-invariant ratio R = mean(d) / mean_c(sigma_c). Headline test:
      R_STD / R_AT > 2.

We run 3 seeds and report means.

Output
------
  Results path: dissertation_extension/results/fashion_mnist/h497_feature_space_adv_distance_output.txt
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
EPS = 0.1
PGD_STEPS = 10
N_CLEAN = 1000
N_ADV = 1000  # matched: same indices as clean
TRAIN_N = 6000


def _penultimate_features(model: C.SmallCNN, X: torch.Tensor, batch: int = 256) -> torch.Tensor:
    """Return the post-ReLU 256-d penultimate features of a SmallCNN.

    SmallCNN.head = Sequential(Flatten, Linear(*, 256), ReLU, Linear(256, ncls)).
    So head[:3] is Flatten -> Linear -> ReLU = the penultimate representation.
    """
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            xb = X[i:i + batch]
            h = model.features(xb)
            # head[0] = Flatten, head[1] = Linear(*, 256), head[2] = activation
            z = model.head[0](h)
            z = model.head[1](z)
            z = model.head[2](z)
            feats.append(z.cpu())
    return torch.cat(feats, dim=0)


def _within_class_spread(F_clean: torch.Tensor, Y: torch.Tensor, ncls: int):
    """Return (mean_within_spread, per_class_spread, centroids)."""
    spreads = []
    centroids = torch.zeros(ncls, F_clean.size(1))
    for c in range(ncls):
        mask = (Y == c)
        if mask.sum() < 2:
            centroids[c] = F_clean.mean(0)
            spreads.append(float("nan"))
            continue
        Fc = F_clean[mask]
        mu = Fc.mean(0)
        centroids[c] = mu
        d = (Fc - mu).norm(dim=1)
        spreads.append(float(d.mean()))
    finite = [s for s in spreads if s == s]
    mean_within = float(np.mean(finite)) if finite else float("nan")
    return mean_within, spreads, centroids


def _between_class_distance(centroids: torch.Tensor):
    ncls = centroids.size(0)
    dists = []
    for i in range(ncls):
        for j in range(i + 1, ncls):
            dists.append(float((centroids[i] - centroids[j]).norm()))
    return float(np.mean(dists)) if dists else float("nan")


def run_seed(seed: int):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    ncls = meta["n_classes"]
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=TRAIN_N, n_eval=max(N_CLEAN, 2000), seed=seed)

    # Take a fixed 1000-sample evaluation slice
    Xe = Xte[:N_CLEAN]
    Ye = Yte[:N_CLEAN]

    out = {"seed": seed}

    for tag, adv_train in [("STD", False), ("AT", True)]:
        model = C.build_model("cnn", meta, seed=seed)
        C.train_model(
            model, Xtr, Ytr,
            epochs=6, opt="sgd", lr=0.05, ncls=ncls,
            adv_train=adv_train, adv_eps=EPS, adv_steps=7,
        )

        # clean accuracy + adv accuracy as sanity
        _, clean_acc = C.logits_and_acc(model, Xe, Ye)
        Xadv = C.pgd(model, Xe, Ye, eps=EPS, steps=PGD_STEPS)
        _, adv_acc = C.logits_and_acc(model, Xadv, Ye)

        F_clean = _penultimate_features(model, Xe)
        F_adv = _penultimate_features(model, Xadv)

        # (3) per-sample adv drift
        drift = (F_clean - F_adv).norm(dim=1)
        mean_drift = float(drift.mean())
        median_drift = float(drift.median())

        # (4) within-class mean spread
        mean_within, per_cls_spread, centroids = _within_class_spread(F_clean, Ye.cpu(), ncls)

        # (5) between-class mean centroid distance
        mean_between = _between_class_distance(centroids)

        # (6) scale-invariant ratio
        ratio = mean_drift / mean_within if mean_within and mean_within == mean_within else float("nan")
        sep = mean_between / mean_within if mean_within and mean_within == mean_within else float("nan")

        out[tag] = {
            "clean_acc": clean_acc,
            "adv_acc": adv_acc,
            "mean_drift": mean_drift,
            "median_drift": median_drift,
            "mean_within_spread": mean_within,
            "mean_between_dist": mean_between,
            "drift_over_within": ratio,
            "between_over_within": sep,
            "per_class_spread": per_cls_spread,
        }

    # cross-model summary metrics
    r_std = out["STD"]["drift_over_within"]
    r_at = out["AT"]["drift_over_within"]
    out["ratio_STD_over_AT"] = (r_std / r_at) if (r_at and r_at == r_at and r_at > 0) else float("nan")
    out["raw_drift_STD_over_AT"] = (
        out["STD"]["mean_drift"] / out["AT"]["mean_drift"]
        if out["AT"]["mean_drift"] > 0 else float("nan")
    )
    out["sep_preserved"] = (
        out["AT"]["between_over_within"] / out["STD"]["between_over_within"]
        if out["STD"]["between_over_within"] > 0 else float("nan")
    )
    return out


def main():
    print("=" * 78)
    print("H497 - Feature-space adversarial distance: STD vs PGD-AT (penultimate)")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  pgd_steps={PGD_STEPS}")
    print(f"N_clean=N_adv={N_CLEAN}  train_n={TRAIN_N}  seeds={SEEDS}")
    print("Prior art: Mao 2019 (metric learning AT), Engstrom 2019 (AT as repr prior),")
    print("           Pang 2020 (hypersphere embedding AT).")
    print()

    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"[seed {s}]  ({r['runtime_s']}s)")
        for tag in ("STD", "AT"):
            d = r[tag]
            print(f"  {tag}: clean={d['clean_acc']:.3f}  adv={d['adv_acc']:.3f}"
                  f"  drift={d['mean_drift']:.3f}  within={d['mean_within_spread']:.3f}"
                  f"  between={d['mean_between_dist']:.3f}"
                  f"  d/within={d['drift_over_within']:.3f}"
                  f"  bet/within={d['between_over_within']:.3f}")
        print(f"  -> ratio_STD/AT (normalised drift)  : {r['ratio_STD_over_AT']:.3f}")
        print(f"  -> raw drift STD/AT                  : {r['raw_drift_STD_over_AT']:.3f}")
        print(f"  -> separation preserved (AT/STD)     : {r['sep_preserved']:.3f}")
        print()

    def mean(key_path):
        vals = []
        for r in rows:
            cur = r
            for k in key_path:
                cur = cur[k]
            if cur == cur:
                vals.append(float(cur))
        return float(np.mean(vals)) if vals else float("nan")

    print("=" * 78)
    print("MEANS across seeds")
    print("=" * 78)
    for tag in ("STD", "AT"):
        print(f"  {tag}: clean={mean([tag, 'clean_acc']):.3f}  adv={mean([tag, 'adv_acc']):.3f}"
              f"  drift={mean([tag, 'mean_drift']):.3f}"
              f"  within={mean([tag, 'mean_within_spread']):.3f}"
              f"  between={mean([tag, 'mean_between_dist']):.3f}"
              f"  d/within={mean([tag, 'drift_over_within']):.3f}"
              f"  bet/within={mean([tag, 'between_over_within']):.3f}")
    m_ratio = mean(["ratio_STD_over_AT"])
    m_raw = mean(["raw_drift_STD_over_AT"])
    m_sep = mean(["sep_preserved"])
    print(f"  ratio (normalised drift, STD/AT) : {m_ratio:.3f}   [hypothesis: > 2.0]")
    print(f"  ratio (raw drift,        STD/AT) : {m_raw:.3f}")
    print(f"  separation preserved (AT/STD)    : {m_sep:.3f}   [>= ~0.7 = preserved]")

    print()
    print("=" * 78)
    print("HEADLINE")
    print("=" * 78)
    compress = m_ratio > 2.0
    preserved = m_sep == m_sep and m_sep >= 0.7
    if compress and preserved:
        verdict = ("SUPPORTED: PGD-AT compresses the (within-class-normalised) feature-space "
                   "adv distance by >2x vs STD AND preserves between/within class separation. "
                   "Consistent with Mao 2019 / Engstrom 2019 / Pang 2020.")
    elif compress and not preserved:
        verdict = ("PARTIAL: AT compresses normalised adv drift by >2x but class separability "
                   "(between/within) shrinks notably -- compression bought partly via class "
                   "collapse, contra the strict Mao/Pang reading.")
    elif (not compress) and preserved:
        verdict = ("REJECTED (magnitude): AT preserves class separability but does not deliver "
                   ">2x compression of normalised feature-space adv distance.")
    else:
        verdict = ("REJECTED: neither >2x normalised compression nor preserved class separability.")
    print(verdict)
    print("=" * 78)


if __name__ == "__main__":
    main()
