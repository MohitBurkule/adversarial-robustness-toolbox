"""
H504 - Robust feature decomposition: spectral structure of penultimate features
       differs between STD and PGD-AT models.

Seed (G6, §5 / §3): take the SVD/PCA of penultimate-layer feature
representations and ask whether adversarial training spreads representational
capacity over more directions than standard training. We summarise the spectrum
with three width-aware quantities:

  * effective_rank   = (Sum sigma_i)^2 / Sum sigma_i^2   (participation ratio
                       of the squared singular values; aka stable rank when
                       written this way -- see Roy & Vetterli 2007)
  * energy90_rank    = smallest k s.t. cumulative sigma_i^2 / total >= 0.9
  * top_k_energy     = cumulative sigma_i^2 / total for k in {1,3,5,10,32}
  * within-class variance breakdown (per-class trace of centred Gram)

Hypothesis: penultimate-feature effective rank of the PGD-AT model is >50%
larger than that of the STD model. The advisor critique is "depends on net
width" -- we therefore report BOTH effective rank and the 90%-energy rank, and
normalise every spectral quantity by the embedding width (256 for SmallCNN).

Extra papers (we cite):
  * Engstrom et al. "Adversarial Robustness as a Prior for Learned
    Representations" (arXiv:1906.00945) - AT features are perceptually
    aligned and more directly invertible, suggesting richer/more distributed
    representations.
  * Allen-Zhu & Li, "Feature Purification: How Adversarial Training Performs
    Robust Deep Learning" (arXiv:2005.10190) - AT removes "dense mixtures"
    of features from individual neurons; a priori this should change the
    feature-covariance spectrum.
  * Salman et al. "Do Adversarially Robust ImageNet Models Transfer Better?"
    (NeurIPS 2020, arXiv:2007.08489) - robust features transfer better, which
    is empirically associated with more useful (often higher-rank) feature
    spaces.

Controls:
  (1) STD vs PGD-AT models (same SmallCNN, same data, same seed list);
  (2) collect penultimate features over 2000 test samples;
  (3) SVD -> effective rank, 90% energy rank, participation ratio;
  (4) energy in top-k directions for k in {1,3,5,10,32};
  (5) per-class within-class feature variance breakdown.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
EPS = 0.1               # L-inf budget for PGD-AT
PGD_STEPS = 7
N_EVAL = 2000           # penultimate features collected on this many test points
TOPK = [1, 3, 5, 10, 32]
EMBED_DIM = 256         # width of the penultimate layer in SmallCNN (see common.py)


# ---------------------------------------------------------------------------
# Penultimate-feature extractor.
# SmallCNN.head = Sequential(Flatten, Linear(*->256), Act, Linear(256, ncls)).
# The penultimate features are the output of the activation (index 2 in head),
# i.e. the 256-d post-activation embedding feeding the classifier.
# ---------------------------------------------------------------------------
def _penult_features(model, X, batch=256):
    model.eval()
    feats = []
    # head = [Flatten, Linear(..,256), Act, Linear(256, ncls)]
    head_pre = nn.Sequential(*list(model.head.children())[:-1])  # up to & incl Act
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            xb = X[i:i + batch]
            h = model.features(xb)
            f = head_pre(h)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0).numpy()  # (N, 256)


# ---------------------------------------------------------------------------
# Spectral summaries.
# ---------------------------------------------------------------------------
def _spectrum_stats(F):
    """F: (N, D) feature matrix. Returns dict of spectral quantities.

    We centre F (mean-subtract) so the spectrum reflects feature *covariance*
    rather than a rank-1 mean dominating.
    """
    F = F - F.mean(axis=0, keepdims=True)
    # economy SVD; sigma are the singular values of F (length min(N,D)).
    sigma = np.linalg.svd(F, full_matrices=False, compute_uv=False)
    sig2 = sigma ** 2
    total = float(sig2.sum()) + 1e-12

    # participation ratio / effective rank in the (Sum s)^2 / Sum s^2 sense
    eff_rank_pr = float((sigma.sum() ** 2) / (sig2.sum() + 1e-12))
    # entropy-based effective rank (Roy-Vetterli 2007), for cross-check
    p = sig2 / total
    p = np.clip(p, 1e-20, 1.0)
    eff_rank_entropy = float(np.exp(-(p * np.log(p)).sum()))

    cum = np.cumsum(sig2) / total
    energy90 = int(np.searchsorted(cum, 0.90) + 1)
    energy99 = int(np.searchsorted(cum, 0.99) + 1)

    topk = {}
    for k in TOPK:
        kk = min(k, len(sig2))
        topk[k] = float(sig2[:kk].sum() / total)

    return {
        "eff_rank_pr": eff_rank_pr,
        "eff_rank_entropy": eff_rank_entropy,
        "eff_rank_pr_normwidth": eff_rank_pr / EMBED_DIM,
        "energy90_rank": energy90,
        "energy99_rank": energy99,
        "topk_energy": topk,
        "spectral_total_energy": total,
    }


def _per_class_within_var(F, Y, ncls):
    """Per-class within-class variance = trace of class-centred covariance.

    Returns dict {class: within_trace} plus the ratio of mean within-class
    variance to the total (centred) variance -- a Fisher-like compactness
    score: small ratio = tight class clusters.
    """
    F = np.asarray(F)
    Y = np.asarray(Y)
    total_centred = F - F.mean(axis=0, keepdims=True)
    total_trace = float((total_centred ** 2).sum() / max(1, F.shape[0]))
    per = {}
    within_sum = 0.0
    n_total = 0
    for c in range(ncls):
        mask = Y == c
        nc = int(mask.sum())
        if nc < 2:
            per[c] = float("nan")
            continue
        Fc = F[mask]
        Fc = Fc - Fc.mean(axis=0, keepdims=True)
        tr = float((Fc ** 2).sum() / nc)
        per[c] = tr
        within_sum += tr * nc
        n_total += nc
    mean_within = within_sum / max(1, n_total)
    return per, mean_within, total_trace


# ---------------------------------------------------------------------------
# per-seed run
# ---------------------------------------------------------------------------
def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_EVAL, seed=seed)
    ncls = meta["n_classes"]

    # ----- STD model -----
    std = C.build_model("cnn", meta, seed=seed)
    C.train_model(std, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05, ncls=ncls)

    # ----- PGD-AT model -----
    at = C.build_model("cnn", meta, seed=seed)
    C.train_model(at, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05, ncls=ncls,
                  adv_train=True, adv_eps=EPS, adv_steps=PGD_STEPS)

    # ----- sanity: clean + adv accuracy -----
    _, std_clean = C.logits_and_acc(std, Xte, Yte)
    _, at_clean = C.logits_and_acc(at, Xte, Yte)
    std_asr = C.attack_success(std, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)["asr"]
    at_asr = C.attack_success(at, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)["asr"]

    # ----- collect penultimate features (test set, clean inputs) -----
    F_std = _penult_features(std, Xte)
    F_at = _penult_features(at, Xte)
    Y_np = Yte.cpu().numpy()

    s_std = _spectrum_stats(F_std)
    s_at = _spectrum_stats(F_at)

    per_std, mw_std, tt_std = _per_class_within_var(F_std, Y_np, ncls)
    per_at, mw_at, tt_at = _per_class_within_var(F_at, Y_np, ncls)

    return {
        "seed": seed,
        "std_clean_acc": std_clean,
        "at_clean_acc": at_clean,
        "std_pgd_asr": std_asr,
        "at_pgd_asr": at_asr,
        "std_spectrum": s_std,
        "at_spectrum": s_at,
        "std_per_class_within": per_std,
        "at_per_class_within": per_at,
        "std_mean_within": mw_std,
        "at_mean_within": mw_at,
        "std_total_trace": tt_std,
        "at_total_trace": tt_at,
    }


def main():
    print("=" * 78)
    print("H504 - Robust feature decomposition (penultimate-feature SVD: STD vs PGD-AT)")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  pgd_steps={PGD_STEPS}  "
          f"n_eval={N_EVAL}  embed_dim={EMBED_DIM}")
    print("Refs: Engstrom 2019 (arXiv:1906.00945); Allen-Zhu & Li 2020 (arXiv:2005.10190);")
    print("      Salman et al. NeurIPS 2020 (arXiv:2007.08489); Roy & Vetterli 2007 (eff. rank).")
    print()

    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        rows.append(r)

        ss, sa = r["std_spectrum"], r["at_spectrum"]
        print(f"[seed {s}]  ({r['runtime_s']}s)")
        print(f"  clean acc       STD={r['std_clean_acc']:.3f}  AT={r['at_clean_acc']:.3f}")
        print(f"  PGD@{EPS} ASR   STD={r['std_pgd_asr']:.3f}  AT={r['at_pgd_asr']:.3f}")
        print(f"  eff rank (PR)   STD={ss['eff_rank_pr']:.2f}  AT={sa['eff_rank_pr']:.2f}  "
              f"(/width: {ss['eff_rank_pr_normwidth']:.3f} vs {sa['eff_rank_pr_normwidth']:.3f})")
        print(f"  eff rank (ent)  STD={ss['eff_rank_entropy']:.2f}  AT={sa['eff_rank_entropy']:.2f}")
        print(f"  90% energy rk   STD={ss['energy90_rank']}      AT={sa['energy90_rank']}")
        print(f"  99% energy rk   STD={ss['energy99_rank']}      AT={sa['energy99_rank']}")
        for k in TOPK:
            print(f"    top-{k:>2} energy  STD={ss['topk_energy'][k]:.3f}  AT={sa['topk_energy'][k]:.3f}")
        print(f"  mean within-class var   STD={r['std_mean_within']:.3f}  AT={r['at_mean_within']:.3f}")
        print(f"  total feature trace     STD={r['std_total_trace']:.3f}  AT={r['at_total_trace']:.3f}")
        # Fisher-like compactness (smaller = tighter clusters)
        comp_std = r["std_mean_within"] / max(1e-9, r["std_total_trace"])
        comp_at = r["at_mean_within"] / max(1e-9, r["at_total_trace"])
        print(f"  within/total ratio      STD={comp_std:.3f}  AT={comp_at:.3f}")
        print()

    # ------- means across seeds -------
    def avg(get):
        vs = [get(r) for r in rows]
        vs = [v for v in vs if v == v]
        return float(np.mean(vs)) if vs else float("nan")

    m_eff_std = avg(lambda r: r["std_spectrum"]["eff_rank_pr"])
    m_eff_at = avg(lambda r: r["at_spectrum"]["eff_rank_pr"])
    m_e90_std = avg(lambda r: r["std_spectrum"]["energy90_rank"])
    m_e90_at = avg(lambda r: r["at_spectrum"]["energy90_rank"])
    m_ent_std = avg(lambda r: r["std_spectrum"]["eff_rank_entropy"])
    m_ent_at = avg(lambda r: r["at_spectrum"]["eff_rank_entropy"])
    m_mw_std = avg(lambda r: r["std_mean_within"])
    m_mw_at = avg(lambda r: r["at_mean_within"])
    m_tt_std = avg(lambda r: r["std_total_trace"])
    m_tt_at = avg(lambda r: r["at_total_trace"])

    rel_eff = (m_eff_at - m_eff_std) / max(1e-9, m_eff_std)
    rel_e90 = (m_e90_at - m_e90_std) / max(1e-9, m_e90_std)

    print("=" * 78)
    print("MEANS across seeds")
    print(f"  eff rank (PR)      STD={m_eff_std:.2f}  AT={m_eff_at:.2f}  "
          f"(AT/STD - 1 = {rel_eff*100:+.1f}%)")
    print(f"  eff rank (entropy) STD={m_ent_std:.2f}  AT={m_ent_at:.2f}")
    print(f"  90% energy rank    STD={m_e90_std:.2f}  AT={m_e90_at:.2f}  "
          f"(AT/STD - 1 = {rel_e90*100:+.1f}%)")
    print(f"  mean within-class var  STD={m_mw_std:.3f}  AT={m_mw_at:.3f}")
    print(f"  total feature trace    STD={m_tt_std:.3f}  AT={m_tt_at:.3f}")
    print("=" * 78)
    threshold = 0.50
    verdict = "SUPPORTED" if rel_eff > threshold else ("PARTIAL" if rel_eff > 0.10 else "REJECTED")
    print(f"HEADLINE  ({DS}, SmallCNN, eps={EPS}):")
    print(f"  Hypothesis -- PGD-AT effective rank >50% larger than STD -- {verdict}.")
    print(f"  AT / STD effective-rank gain = {rel_eff*100:+.1f}% (PR); "
          f"90%-energy-rank gain = {rel_e90*100:+.1f}%.")
    print(f"  Width-normalised eff rank: STD={m_eff_std/EMBED_DIM:.3f}, "
          f"AT={m_eff_at/EMBED_DIM:.3f} (embed dim = {EMBED_DIM}).")
    print("Caveat (advisor critique): both quantities scale with embedding width;")
    print("we therefore report the same numbers normalised by the 256-d penultimate")
    print("layer and the 90%-energy rank as a cross-check.  Compactness")
    print("(within/total feature variance) is also reported per seed so the reader")
    print("can disentangle 'AT spreads the spectrum' from 'AT just makes features bigger'.")
    print("Connects to: Engstrom 2019 (richer / invertible AT features),")
    print("Allen-Zhu & Li 2020 (feature purification removes dense mixtures),")
    print("Salman 2020 (transfer of robust features).")
    print("=" * 78)


if __name__ == "__main__":
    main()
