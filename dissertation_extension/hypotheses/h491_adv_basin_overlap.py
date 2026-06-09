"""
H491 - Adversarial loss-landscape basin overlap between defenses (F-MNIST SmallCNN).

Seed (Paper 5 / Paper 3 G8). The "robust basin" found by adversarial training
is not a single object: different defenses may converge to qualitatively
different sets of robust features. We probe this by asking which test inputs
each defended model is *vulnerable to* under a fixed PGD attack and measure
how strongly two defended models share the same vulnerabilities.

Hypothesis
----------
  H1 (within-method overlap).  Two PGD-AT models that differ only in seed share
      MOST of their adversarially-flipped test samples:
          mean Jaccard( PGD-AT_i , PGD-AT_j )  >  0.6   for i != j.

  H2 (between-method overlap). PGD-AT and TRADES, despite reaching comparable
      robust accuracy, find qualitatively different robust basins:
          mean Jaccard( PGD-AT_i , TRADES_j )  <  0.5.

  H3 (margin-rank).  Within a method, per-sample robust-margin rankings are
      strongly correlated across seeds (Spearman > 0.6), implying the
      vulnerability ordering is a property of the *method* not the seed.

Operational definition
----------------------
"Shared basin" = the set of test samples successfully attacked at eps=0.1 by
PGD-10 (untargeted, CE loss). Each model gives a binary vector
 success[i] in {0,1} over test samples. Jaccard is computed on these binary
vectors; "successfully attacked" means the clean prediction was correct AND
the PGD prediction is wrong.

Controls
--------
  (1) 3 seeds  x  3 methods  =  9 trained SmallCNNs.
       methods: STD (no AT), PGD-AT (Madry), TRADES (Zhang et al. 2019).
  (2) for each model, build success_i = (clean-correct) AND (PGD-flipped).
  (3) compute the 9x9 Jaccard matrix.
  (4) summarise: within-method mean Jaccard vs between-method mean Jaccard.
  (5) within-method margin-rank correlation (Spearman) on the *adversarial*
      margins (margin under PGD-10 adversarial inputs).

Extra references
----------------
  * Madry et al. 2018  "Towards Deep Learning Models Resistant to Adversarial
        Attacks"  (arXiv:1706.06083) - PGD-AT baseline.
  * Zhang et al. 2019  "Theoretically Principled Trade-off between Robustness
        and Accuracy"  (arXiv:1901.08573) - TRADES objective.
  * Wang et al. 2020  "Improving Adversarial Robustness Requires Revisiting
        Misclassified Examples"  (ICLR; MART) - argues hard / misclassified
        examples define the robust frontier; supports H1 (within-method, the
        same hard examples appear) and H2 (different weightings can move that
        frontier).
  * Kim et al. 2021  "Bridging Adversarial Robustness and Gradient
        Interpretability"  (arXiv:2009.12669 / "Bridging Adversarial
        Robustness") - distinct AT objectives produce qualitatively different
        gradient geometry, motivating H2.

These four works jointly imply: defenses sharing the same inner attack should
produce overlapping vulnerability sets, but objective changes (CE vs KL) can
re-shape *which* hard examples remain hard.

NOTE: this script is intentionally DO-NOT-RUN here; it is dispatched and run
by a background agent. Outputs go to:
   results/fashion_mnist/h491_adv_basin_overlap_output.txt
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
DS         = "fashion_mnist"
N_TRAIN    = 6000           # subset for tractable 9-model training
N_EVAL     = 1500           # test subset for basin comparison
EPOCHS     = 10
BATCH      = 128
LR         = 0.05
EPS        = 0.1
AT_STEPS   = 7              # inner PGD steps during AT / TRADES
EVAL_STEPS = 10             # PGD-10 for evaluation (per critique seed)
ALPHA_EVAL = 2.5 * EPS / EVAL_STEPS
TRADES_BETA = 6.0
SEEDS      = [0, 1, 2]
METHODS    = ["STD", "PGD-AT", "TRADES"]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h491_adv_basin_overlap_output.txt")


# ---------------------------------------------------------------------------
# training: STD / PGD-AT use C.train_model; TRADES needs KL inner attack
# ---------------------------------------------------------------------------
def pgd_kl(model, x, eps, steps, alpha):
    """Inner PGD maximising KL(f(x) || f(x+delta)) for TRADES."""
    x0 = x.clone().detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1).detach()
    with torch.no_grad():
        p_clean = F.softmax(model(x0), dim=1)
    for _ in range(steps):
        xa.requires_grad_(True)
        log_p_adv = F.log_softmax(model(xa), dim=1)
        kl = F.kl_div(log_p_adv, p_clean, reduction="batchmean")
        g, = torch.autograd.grad(kl, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def train_trades(meta, Xtr, Ytr, seed, beta=TRADES_BETA):
    C.set_seed(seed)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    alpha_in = 2.5 * EPS / AT_STEPS
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            x_adv = pgd_kl(model, xb, EPS, AT_STEPS, alpha_in)
            model.train()
            opt.zero_grad()
            out_clean = model(xb)
            loss_ce = F.cross_entropy(out_clean, yb)
            p_clean = F.softmax(out_clean, dim=1).detach()
            log_p_adv = F.log_softmax(model(x_adv), dim=1)
            kl_loss = F.kl_div(log_p_adv, p_clean, reduction="batchmean")
            loss = loss_ce + beta * kl_loss
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_one(method, meta, Xtr, Ytr, seed):
    """Dispatch to the appropriate training routine for `method` at `seed`."""
    if method == "STD":
        C.set_seed(seed)
        m = C.build_model("cnn", meta)
        C.train_model(m, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, opt="sgd", lr=LR,
                      adv_train=False)
        return m
    if method == "PGD-AT":
        C.set_seed(seed)
        m = C.build_model("cnn", meta)
        C.train_model(m, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, opt="sgd", lr=LR,
                      adv_train=True, adv_eps=EPS, adv_steps=AT_STEPS)
        return m
    if method == "TRADES":
        return train_trades(meta, Xtr, Ytr, seed)
    raise ValueError(method)


# ---------------------------------------------------------------------------
# evaluation: success vector + adversarial margin
# ---------------------------------------------------------------------------
@torch.no_grad()
def _clean_correct(model, X, Y, batch=256):
    out = []
    for i in range(0, X.size(0), batch):
        out.append((model(X[i:i + batch]).argmax(1) == Y[i:i + batch]).cpu())
    return torch.cat(out).numpy().astype(bool)


def success_and_advmargin(model, X, Y, eps=EPS, steps=EVAL_STEPS, alpha=ALPHA_EVAL,
                          batch=256):
    """Return (success_vec[N] bool, adv_margin[N] float).

    success[i] = clean_correct[i] AND (pgd_pred[i] != y[i])
    adv_margin[i] = margin of model on x_adv (correct - max-other logit).
    """
    model.eval()
    N = X.size(0)
    success = np.zeros(N, dtype=bool)
    adv_margin = np.zeros(N, dtype=np.float32)
    clean_correct = _clean_correct(model, X, Y, batch=batch)
    for i in range(0, N, batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        xa = C.pgd(model, x, y, eps=eps, steps=steps, alpha=alpha)
        with torch.no_grad():
            logits = model(xa).cpu()
        adv_pred = logits.argmax(1).numpy()
        y_cpu = y.cpu().numpy()
        flipped = adv_pred != y_cpu
        success[i:i + batch] = clean_correct[i:i + batch] & flipped
        # adv margin: correct-class logit minus max-other
        adv_margin[i:i + batch] = C.margin_of(logits, y.cpu())
    return success, adv_margin, clean_correct


def jaccard(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    def log(s=""):
        print(s); lines.append(s)

    log("=" * 78)
    log("H491 - Adversarial-basin overlap across defenses (STD / PGD-AT / TRADES)")
    log("=" * 78)
    log(f"Device={C.DEVICE}  dataset={DS}  N_train={N_TRAIN}  N_eval={N_EVAL}")
    log(f"eps={EPS}  AT_inner_steps={AT_STEPS}  eval_steps={EVAL_STEPS}  "
        f"TRADES_beta={TRADES_BETA}")
    log(f"methods={METHODS}  seeds={SEEDS}")
    log("")

    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=0)
    log(f"Loaded F-MNIST: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    log("")

    # ---- train 9 models ---------------------------------------------------
    models = {}    # (method, seed) -> model
    train_times = {}
    for method in METHODS:
        for s in SEEDS:
            tt = time.time()
            log(f"Training {method:<7} seed={s} ...")
            m = train_one(method, meta, Xtr, Ytr, s)
            models[(method, s)] = m
            train_times[(method, s)] = time.time() - tt
            log(f"  done in {train_times[(method, s)]:.1f}s")
    log("")

    # ---- per-model success vectors + adv margins --------------------------
    log("Computing PGD-10 success vectors and adversarial margins ...")
    success_map = {}
    advmargin_map = {}
    clean_acc_map = {}
    asr_map = {}
    for key, m in models.items():
        sv, am, cc = success_and_advmargin(m, Xte, Yte)
        success_map[key] = sv
        advmargin_map[key] = am
        clean_acc_map[key] = float(cc.mean())
        # ASR among originally-correct samples (matches campaign convention)
        denom = max(int(cc.sum()), 1)
        asr_map[key] = float(sv.sum()) / float(denom)

    log("")
    log("Per-model metrics (clean acc, PGD-10 ASR on correct samples,"
        " #flipped):")
    log(f"  {'model':<14} {'clean':>7} {'pgd_asr':>9} {'#flipped':>10}")
    keys_ordered = [(mth, s) for mth in METHODS for s in SEEDS]
    for k in keys_ordered:
        log(f"  {k[0]+'/s'+str(k[1]):<14} "
            f"{clean_acc_map[k]:>7.3f} {asr_map[k]:>9.3f} "
            f"{int(success_map[k].sum()):>10d}")
    log("")

    # ---- 9x9 Jaccard matrix ------------------------------------------------
    K = len(keys_ordered)
    J = np.zeros((K, K), dtype=float)
    for i, ki in enumerate(keys_ordered):
        for j, kj in enumerate(keys_ordered):
            J[i, j] = jaccard(success_map[ki], success_map[kj])

    log("9x9 Jaccard matrix (rows/cols in the order printed above):")
    header = "        " + " ".join(f"{ki[0][:3]}{ki[1]}".rjust(6) for ki in keys_ordered)
    log(header)
    for i, ki in enumerate(keys_ordered):
        row = f"{ki[0][:3]}{ki[1]:<3} " + " ".join(f"{J[i,j]:6.3f}" for j in range(K))
        log(row)
    log("")

    # ---- within-method vs between-method mean Jaccard ---------------------
    def mean_block(method_a, method_b):
        vals = []
        for ki in keys_ordered:
            if ki[0] != method_a:
                continue
            for kj in keys_ordered:
                if kj[0] != method_b or kj == ki:
                    continue
                vals.append(jaccard(success_map[ki], success_map[kj]))
        return float(np.mean(vals)) if vals else float("nan"), vals

    log("Block-mean Jaccard summary (off-diagonal only within same block):")
    block_means = {}
    for a in METHODS:
        for b in METHODS:
            mu, vals = mean_block(a, b)
            block_means[(a, b)] = mu
            log(f"  mean_jaccard({a:<7}, {b:<7}) = {mu:.3f}  (n={len(vals)})")
    log("")

    within_pgdat = block_means[("PGD-AT", "PGD-AT")]
    between_pgdat_trades = 0.5 * (block_means[("PGD-AT", "TRADES")] +
                                  block_means[("TRADES", "PGD-AT")])
    within_trades = block_means[("TRADES", "TRADES")]
    within_std = block_means[("STD", "STD")]

    log("Headline quantities:")
    log(f"  within-method Jaccard, PGD-AT  = {within_pgdat:.3f}")
    log(f"  within-method Jaccard, TRADES  = {within_trades:.3f}")
    log(f"  within-method Jaccard, STD     = {within_std:.3f}")
    log(f"  between-method Jaccard, PGD-AT vs TRADES = {between_pgdat_trades:.3f}")
    log("")

    # ---- within-method margin-rank Spearman correlations ------------------
    log("Within-method margin-rank Spearman (on adversarial margin vectors):")
    margin_rho = {}
    for method in METHODS:
        ms = [advmargin_map[(method, s)] for s in SEEDS]
        rhos = []
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                r, _ = spearmanr(ms[i], ms[j])
                rhos.append(r)
        margin_rho[method] = float(np.mean(rhos))
        log(f"  mean Spearman({method:<7}) = {margin_rho[method]:.3f}  "
            f"(pairs={len(rhos)})")
    log("")

    # ---- verdict ----------------------------------------------------------
    h1 = within_pgdat > 0.6
    h2 = between_pgdat_trades < 0.5
    h3 = margin_rho["PGD-AT"] > 0.6 and margin_rho["TRADES"] > 0.6

    log("=" * 78)
    log("HEADLINE VERDICT")
    log("=" * 78)
    log(f"  H1  within-PGDAT Jaccard > 0.60 :  observed {within_pgdat:.3f}"
        f"   -> {'PASS' if h1 else 'FAIL'}")
    log(f"  H2  PGDAT-vs-TRADES Jaccard < 0.50: observed "
        f"{between_pgdat_trades:.3f}   -> {'PASS' if h2 else 'FAIL'}")
    log(f"  H3  within-method Spearman > 0.60 for PGD-AT and TRADES: "
        f"PGD-AT={margin_rho['PGD-AT']:.3f}  TRADES={margin_rho['TRADES']:.3f}"
        f"   -> {'PASS' if h3 else 'FAIL'}")
    log("")

    if h1 and h2:
        verdict = ("SUPPORTED: PGD-AT seeds share most of their adversarial "
                   "vulnerabilities, while PGD-AT and TRADES occupy "
                   "qualitatively different robust basins.")
    elif h1 and not h2:
        verdict = ("PARTIAL: PGD-AT seeds agree (H1), but TRADES and PGD-AT "
                   "share more than expected -- the inner attack dominates "
                   "the basin geometry regardless of CE-vs-KL outer loss.")
    elif (not h1) and h2:
        verdict = ("PARTIAL: defenses disagree across method (H2) but PGD-AT "
                   "seeds also disagree -- robust basin is seed-fragile, "
                   "weakening the 'method-specific basin' claim.")
    else:
        verdict = ("REJECTED: neither within-method consistency nor "
                   "between-method divergence holds at the stated thresholds.")
    log(verdict)
    log("")
    log(f"Total runtime: {time.time() - t0:.1f}s")
    log("=" * 78)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
