"""
H492 - Example-forgetting under PGD-AT predicts the low-robust-margin tail.

Seed (Toneva et al. 2019, "An Empirical Study of Example Forgetting During
Deep Learning", ICLR'19): during SGD training some examples are *unforgettable*
(once learned, never misclassified again) while others get repeatedly forgotten;
the forgetting count is a stable, learning-dynamics-based notion of "difficulty".

We port that notion from clean training to PGD adversarial training (Madry et al.
2018). Hypothesis: in PGD-AT, the SAMPLES THAT GET FORGOTTEN UNDER ADV LOSS
correlate with the LOW ROBUST-MARGIN TAIL at the end of training (Spearman
rho > 0.3, forgetting-count vs -robust_margin, within the AT model).

Critique-driven design choices (advisor):
  * N=6000, 10 epochs -> forgetting events are sparse on FashionMNIST. We use
    the original Toneva definition: a "forgetting event" at epoch t means the
    sample was correctly classified at the end of epoch t-1 (under PGD-perturbed
    input for the AT model) and is misclassified at the end of epoch t. We track
    the count over the 10 epochs and also report the fraction of "never-learned"
    samples (a Toneva-style unforgettable / never-forgotten split).
  * We score per-sample status under the SAME loss being optimised: PGD-AT is
    tracked under adversarial-input correctness, STD under clean-input
    correctness. This is the apples-to-apples notion of "forgetting under the
    training loss" used by Toneva.
  * "Final margin" = robust margin for the AT model (margin on a fresh PGD
    example) and clean margin for the STD model.

Extra prior art (cited >= 2):
  * Toneva, Sordoni, Combes, Trischler, Bengio, Gordon. "An Empirical Study of
    Example Forgetting During Deep Learning." ICLR 2019. arXiv:1812.05159.
  * Maini, Garg, Lipton, Kolter. "Characterizing Datapoints via Second-Split
    Forgetting." NeurIPS 2022. arXiv:2210.15031. (Forgetting time as a
    learning-dynamics proxy for mislabelled vs rare-but-clean samples; analogue
    of our "low robust margin tail = repeatedly forgotten" claim.)
  * Carlini, Erlingsson, Papernot. "Distribution Density, Tails, and Outliers in
    Machine Learning: Five New Methods." arXiv:1910.13427. (Multiple
    dynamics-based outlier scores including learning-speed and forgetting-style
    signals; motivates correlating one such score with adversarial margin.)
  * Madry, Makelov, Schmidt, Tsipras, Vladu. "Towards Deep Learning Models
    Resistant to Adversarial Attacks." ICLR 2018. arXiv:1706.06083.

Controls / outputs (advisor):
  (1) One PGD-AT training; per-sample forgetting count tracked over the 10
      epochs (status = correct-under-PGD at end-of-epoch).
  (2) One STD training comparator; same forgetting bookkeeping under clean
      correctness.
  (3) Per-sample final robust margin (AT) and clean margin (STD).
  (4) Spearman correlation forgetting-count vs -final-margin WITHIN each method.
  (5) Class-level comparison: per-class mean forgetting count vs per-class
      robust accuracy for the AT model (i.e., do classes with high-forgetting
      samples also have low robust acc?).

HEADLINE: rho_AT(forgetting, -robust_margin) > 0.3 supports the hypothesis.

Output: results/fashion_mnist/h492_forgetting_at_output.txt
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
BATCH = 128
LR = 0.05

EPS = 0.1
PGD_STEPS_TRAIN = 7
PGD_STEPS_EVAL = 20  # stronger PGD for final-margin / status evaluation

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h492_forgetting_at_output.txt",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _correct_clean(model, X, Y, batch=512):
    """Per-sample correctness on clean inputs."""
    model.eval()
    out = []
    for i in range(0, X.size(0), batch):
        pred = model(X[i:i + batch]).argmax(1)
        out.append((pred == Y[i:i + batch]).cpu())
    return torch.cat(out).numpy().astype(np.int8)


def _correct_pgd(model, X, Y, eps, steps, batch=256):
    """Per-sample correctness on PGD-perturbed inputs (robust status)."""
    model.eval()
    out = []
    for i in range(0, X.size(0), batch):
        xb = X[i:i + batch]
        yb = Y[i:i + batch]
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            pred = model(xa).argmax(1)
        out.append((pred == yb).cpu())
    return torch.cat(out).numpy().astype(np.int8)


@torch.no_grad()
def _logits(model, X, batch=512):
    model.eval()
    parts = []
    for i in range(0, X.size(0), batch):
        parts.append(model(X[i:i + batch]).cpu())
    return torch.cat(parts)


def _margin_from_logits(logits, Y):
    Y = Y.cpu()
    correct = logits.gather(1, Y[:, None]).squeeze(1)
    tmp = logits.clone()
    tmp[torch.arange(tmp.size(0)), Y] = -1e9
    other = tmp.max(1).values
    return (correct - other).numpy()


def _robust_margin(model, X, Y, eps, steps, batch=256):
    """Margin on a fresh PGD-perturbed copy of each sample (true-label margin)."""
    parts = []
    model.eval()
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            lg = model(xa).cpu()
        parts.append(lg)
    logits = torch.cat(parts)
    return _margin_from_logits(logits, Y)


def _spearman(a, b):
    """Spearman rho. Returns NaN if either input is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


# ---------------------------------------------------------------------------
# training loop with per-epoch per-sample status bookkeeping
# ---------------------------------------------------------------------------
def train_and_track(model, Xtr, Ytr, adv_train, epochs=EPOCHS, batch=BATCH, lr=LR,
                    eps=EPS, pgd_steps_train=PGD_STEPS_TRAIN,
                    pgd_steps_eval=PGD_STEPS_EVAL, verbose=False):
    """Train for `epochs` epochs; at the END of each epoch, score every training
    sample's correctness under the SAME loss (PGD-perturbed for AT, clean for STD).
    Returns a (epochs, N) int8 array of per-sample correctness.
    """
    opt = C.make_optimizer(model, "sgd", lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    status = np.zeros((epochs, n), dtype=np.int8)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            if adv_train:
                xb = C.pgd(model, xb, yb, eps=eps, steps=pgd_steps_train,
                           alpha=2.5 * eps / pgd_steps_train)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        sched.step()

        # End-of-epoch per-sample status under the matched loss.
        if adv_train:
            status[ep] = _correct_pgd(model, Xtr, Ytr, eps=eps, steps=pgd_steps_eval)
        else:
            status[ep] = _correct_clean(model, Xtr, Ytr)

        if verbose:
            print(f"    epoch {ep+1}/{epochs}  status_mean={status[ep].mean():.3f}")

    model.eval()
    return status


def forgetting_counts(status):
    """Toneva-style forgetting events: at epoch t, sample was correct at t-1
    and incorrect at t. We also count "never learned" samples (status==0 in
    every epoch) separately.

    Returns: (counts shape (N,), never_learned mask shape (N,)).
    """
    # status: (epochs, N) of {0,1}
    transitions = (status[:-1] == 1) & (status[1:] == 0)
    counts = transitions.sum(axis=0).astype(np.int32)  # per-sample forgetting count
    never_learned = (status.sum(axis=0) == 0)
    return counts, never_learned


# ---------------------------------------------------------------------------
# per-seed experiment
# ---------------------------------------------------------------------------
def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)
    ncls = meta["n_classes"]

    # ---- (1) PGD-AT training with per-epoch tracking ----
    m_at = C.build_model("cnn", meta, seed=seed)
    status_at = train_and_track(m_at, Xtr, Ytr, adv_train=True)
    counts_at, never_at = forgetting_counts(status_at)

    # ---- (2) STD training with per-epoch tracking (clean status) ----
    m_std = C.build_model("cnn", meta, seed=seed)
    status_std = train_and_track(m_std, Xtr, Ytr, adv_train=False)
    counts_std, never_std = forgetting_counts(status_std)

    # ---- (3) Final per-sample margins on the TRAIN set under matched loss ----
    rob_margin_at = _robust_margin(m_at, Xtr, Ytr, eps=EPS, steps=PGD_STEPS_EVAL)
    clean_margin_std = _margin_from_logits(_logits(m_std, Xtr), Ytr)

    # ---- (4) Spearman correlation within each method ----
    # hypothesis: high forgetting count <-> low (robust) margin -> rho(count, -margin) > 0
    mask_at = ~never_at  # drop never-learned (no signal in counts)
    mask_std = ~never_std
    rho_at = _spearman(counts_at[mask_at], -rob_margin_at[mask_at])
    rho_std = _spearman(counts_std[mask_std], -clean_margin_std[mask_std])

    # Also rank-correlate ACROSS the two methods (do hard samples agree?):
    rho_cross_counts = _spearman(counts_at, counts_std)
    rho_cross_margin = _spearman(rob_margin_at, clean_margin_std)

    # ---- (5) Class-level: per-class mean forgetting count (AT) vs per-class
    #         robust accuracy (AT) on the TEST set ----
    rob_correct_te = _correct_pgd(m_at, Xte, Yte, eps=EPS, steps=PGD_STEPS_EVAL)
    per_class = []
    for c in range(ncls):
        mtr = (Ytr.cpu().numpy() == c)
        mte = (Yte.cpu().numpy() == c)
        mean_count = float(counts_at[mtr].mean()) if mtr.any() else float("nan")
        nl_frac = float(never_at[mtr].mean()) if mtr.any() else float("nan")
        rob_acc_te = float(rob_correct_te[mte].mean()) if mte.any() else float("nan")
        per_class.append({
            "class": c,
            "mean_forgetting_count_at": mean_count,
            "never_learned_frac_at": nl_frac,
            "test_robust_acc_at": rob_acc_te,
            "n_train": int(mtr.sum()),
            "n_test": int(mte.sum()),
        })
    pc_counts = np.array([pc["mean_forgetting_count_at"] for pc in per_class])
    pc_rob = np.array([pc["test_robust_acc_at"] for pc in per_class])
    rho_class = _spearman(pc_counts, -pc_rob)

    # global summary stats
    return {
        "seed": seed,
        "n_train": int(N_TRAIN),
        "epochs": int(EPOCHS),
        "eps": EPS,
        # AT
        "at_mean_forgetting": float(counts_at.mean()),
        "at_frac_unforgettable": float(((status_at.sum(axis=0) == EPOCHS)).mean()),
        "at_frac_never_learned": float(never_at.mean()),
        "at_train_robust_acc_final": float(status_at[-1].mean()),
        "at_mean_robust_margin": float(np.mean(rob_margin_at)),
        "rho_at_forgetting_vs_neg_robust_margin": rho_at,
        # STD
        "std_mean_forgetting": float(counts_std.mean()),
        "std_frac_unforgettable": float(((status_std.sum(axis=0) == EPOCHS)).mean()),
        "std_frac_never_learned": float(never_std.mean()),
        "std_train_clean_acc_final": float(status_std[-1].mean()),
        "std_mean_clean_margin": float(np.mean(clean_margin_std)),
        "rho_std_forgetting_vs_neg_clean_margin": rho_std,
        # cross-method
        "rho_cross_forgetting_counts": rho_cross_counts,
        "rho_cross_final_margin": rho_cross_margin,
        # class-level
        "per_class": per_class,
        "rho_class_meanforget_vs_neg_robustacc": rho_class,
    }


# ---------------------------------------------------------------------------
# pretty-printer
# ---------------------------------------------------------------------------
def _fmt_row(r, fh):
    print(f"\n[seed {r['seed']}]", file=fh)
    print(f"  PGD-AT   mean_forget={r['at_mean_forgetting']:.3f}"
          f"  unforgettable={r['at_frac_unforgettable']:.3f}"
          f"  never_learned={r['at_frac_never_learned']:.3f}"
          f"  train_rob_acc={r['at_train_robust_acc_final']:.3f}"
          f"  mean_rob_margin={r['at_mean_robust_margin']:+.3f}",
          file=fh)
    print(f"  STD      mean_forget={r['std_mean_forgetting']:.3f}"
          f"  unforgettable={r['std_frac_unforgettable']:.3f}"
          f"  never_learned={r['std_frac_never_learned']:.3f}"
          f"  train_clean_acc={r['std_train_clean_acc_final']:.3f}"
          f"  mean_clean_margin={r['std_mean_clean_margin']:+.3f}",
          file=fh)
    print(f"  rho(forgetting, -margin)  AT={r['rho_at_forgetting_vs_neg_robust_margin']:+.3f}"
          f"   STD={r['rho_std_forgetting_vs_neg_clean_margin']:+.3f}",
          file=fh)
    print(f"  cross-method rho   counts={r['rho_cross_forgetting_counts']:+.3f}"
          f"   final_margin={r['rho_cross_final_margin']:+.3f}",
          file=fh)
    print(f"  per-class AT (mean_forget / never_learned / test_robust_acc):", file=fh)
    for pc in r["per_class"]:
        print(f"    class {pc['class']}: forget={pc['mean_forgetting_count_at']:.3f}"
              f"  nl={pc['never_learned_frac_at']:.3f}"
              f"  test_rob_acc={pc['test_robust_acc_at']:.3f}"
              f"  (n_tr={pc['n_train']}, n_te={pc['n_test']})",
              file=fh)
    print(f"  rho_class (mean_forget vs -test_robust_acc)"
          f" = {r['rho_class_meanforget_vs_neg_robustacc']:+.3f}",
          file=fh)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fh = open(OUT_PATH, "w")

    def out(s=""):
        print(s)
        print(s, file=fh)

    out("=" * 78)
    out("H492 - Example forgetting under PGD-AT predicts the low-robust-margin tail")
    out("=" * 78)
    out(f"Device={C.DEVICE}  dataset={DS}  N_train={N_TRAIN}  epochs={EPOCHS}")
    out(f"eps={EPS}  pgd_steps_train={PGD_STEPS_TRAIN}  pgd_steps_eval={PGD_STEPS_EVAL}")
    out("Refs: Toneva 2019 (1812.05159); Maini 2022 (2210.15031);")
    out("      Carlini 2019 (1910.13427); Madry 2018 (1706.06083).")
    out("")
    out("Per-sample forgetting event (Toneva): correct at end of epoch t-1, wrong")
    out("at end of epoch t, under the SAME loss as training (PGD for AT, clean for")
    out("STD). 'Never learned' samples are excluded from the within-method rho.")

    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        rows.append(r)
        out(f"\n(seed {s} done in {r['runtime_s']}s)")
        _fmt_row(r, fh)
        # also echo to stdout (the table is long; keep stdout terse)
        print(f"  AT rho(forget, -robust_margin) = "
              f"{r['rho_at_forgetting_vs_neg_robust_margin']:+.3f}"
              f"   STD rho = {r['rho_std_forgetting_vs_neg_clean_margin']:+.3f}"
              f"   class-level AT rho = {r['rho_class_meanforget_vs_neg_robustacc']:+.3f}")

    # ---- aggregate ----
    def mean_key(k):
        v = [r[k] for r in rows if isinstance(r[k], float) and r[k] == r[k]]
        return sum(v) / len(v) if v else float("nan")

    rho_at_mean = mean_key("rho_at_forgetting_vs_neg_robust_margin")
    rho_std_mean = mean_key("rho_std_forgetting_vs_neg_clean_margin")
    rho_class_mean = mean_key("rho_class_meanforget_vs_neg_robustacc")
    rho_cross_counts_mean = mean_key("rho_cross_forgetting_counts")
    rho_cross_margin_mean = mean_key("rho_cross_final_margin")

    out("\n" + "=" * 78)
    out("MEANS across seeds")
    out("-" * 78)
    out(f"  AT  rho(forgetting_count, -robust_margin)     = {rho_at_mean:+.3f}")
    out(f"  STD rho(forgetting_count, -clean_margin)      = {rho_std_mean:+.3f}")
    out(f"  per-class AT  rho(mean_forget, -robust_acc)   = {rho_class_mean:+.3f}")
    out(f"  cross-method rho (forgetting counts AT vs STD)= {rho_cross_counts_mean:+.3f}")
    out(f"  cross-method rho (final margin AT vs STD)     = {rho_cross_margin_mean:+.3f}")
    out("")
    out(f"  AT  mean_forgetting={mean_key('at_mean_forgetting'):.3f}"
        f"  unforgettable={mean_key('at_frac_unforgettable'):.3f}"
        f"  never_learned={mean_key('at_frac_never_learned'):.3f}")
    out(f"  STD mean_forgetting={mean_key('std_mean_forgetting'):.3f}"
        f"  unforgettable={mean_key('std_frac_unforgettable'):.3f}"
        f"  never_learned={mean_key('std_frac_never_learned'):.3f}")

    # ---- HEADLINE ----
    thresh = 0.30
    if rho_at_mean == rho_at_mean and rho_at_mean > thresh:
        verdict = (f"SUPPORTED: AT rho(forgetting, -robust_margin) = {rho_at_mean:+.3f} "
                   f"> {thresh:.2f}. Samples that the PGD-AT model repeatedly forgets "
                   f"during training do end up with the lowest robust margins, in line "
                   f"with the Toneva learning-dynamics view of difficulty extended to "
                   f"the adversarial regime.")
    elif rho_at_mean == rho_at_mean and rho_at_mean > 0:
        verdict = (f"WEAK / PARTIAL: AT rho = {rho_at_mean:+.3f} is positive but below "
                   f"the pre-registered 0.30 threshold. Direction matches Toneva-style "
                   f"forgetting predicting low margin, but the effect is too small to "
                   f"call supported with N={N_TRAIN}, {EPOCHS} epochs.")
    else:
        verdict = (f"NOT SUPPORTED: AT rho = {rho_at_mean:+.3f}. Under-trained / sparse "
                   f"forgetting events do not predict the low robust-margin tail at "
                   f"this scale; would need more epochs or a denser status-tracking "
                   f"schedule (per-iter, not per-epoch).")

    out("")
    out("=" * 78)
    out("HEADLINE VERDICT")
    out("-" * 78)
    out(verdict)
    out("=" * 78)
    fh.close()


if __name__ == "__main__":
    main()
