"""
H480 - Per-class worst-case robust accuracy: do AT defenses hide a vulnerable
class behind a healthy-looking mean?

Seed (paper §5 / §2 M6 "no per-class / worst-case"). Aggregate robust-acc /
mean ASR is the dominant defense metric in the campaign (and the wider
literature), but it can be propped up by 9 strong classes while one class
collapses. The classic robust-fairness papers show this is the rule rather
than the exception:

  - Xu, Liu, Wang, Han, Liu (NeurIPS 2021), "To be Robust or to be Fair:
    Towards Fairness in Adversarial Training" - PGD-AT systematically inflates
    accuracy *disparity* across classes; a "Fair Robust Learning" (FRL) penalty
    is needed to close it.
  - Tian, Cui, Liu, Liu (CVPR 2021), "Analysis and Applications of Class-wise
    Robustness in Adversarial Training" - per-class robust accuracy under
    PGD-AT / TRADES on CIFAR-10 has min-class < 0.4 * mean-class for several
    schedules.
  - Benz, Zhang, Karjauv, Kweon (ICML-W 2021 / arXiv 2006.13726), "Robustness
    May Be at Odds with Fairness: an Empirical Study on Class-wise Accuracy" -
    even on simple datasets, AT exaggerates the worst-class gap.

Hypothesis (HEADLINE):  PGD-AT and TRADES give large *mean* gains in robust
accuracy on Fashion-MNIST, but worst-class robust accuracy is <= 60% of mean
robust accuracy across multiple AT variants - i.e. the defense looks strong on
average but masks a fully vulnerable class. We further predict class 6
(SHIRT) - notorious for confusion with T-shirt/Pullover/Coat - is the
worst-class on most models.

Critique seed (advisor): per-class metrics ARE common in the campaign already
(e.g. H188, H53) but they're rarely worst-case. Our contribution here is to
explicitly compute (i) min-class clean-acc and min-class robust-acc, (ii) the
"robustness gap" = mean_class - min_class, both as fractions of mean, AND to
ask whether successful PGD flips are *targeted toward* the worst class
(i.e. the worst class is also an adversarial attractor, not just a victim).

Controls / experiments
======================
(1) Train FOUR victims at matched compute:
      STD      - standard cross-entropy SGD.
      PGDAT    - Madry-style PGD adversarial training (eps=0.1, 7 steps).
      TRADES   - Zhang et al. 2019, KL-regularised AT, beta=6, eps=0.1, 7 steps.
      FASTAT   - "free-AT-style" fast variant: FGSM-AT with random init,
                 eps=0.1 (Wong et al. ICLR 2020, "Fast is Better than Free");
                 the open-source proxy for Shafahi et al.'s Free-AT that
                 reuses the inner gradient inside one SGD step.
   (If FASTAT collapses to ~0% robust acc - the known "catastrophic
   overfitting" failure mode - we still report it; that *is* a per-class
   robustness story.)
(2) Per-class clean accuracy + per-class robust accuracy at eps=0.1 under
   10-step PGD. We report both the mean across classes (the headline number)
   AND the min across classes (the worst-case number this paper exists for).
(3) "Robustness gap" = mean_class_robust_acc - min_class_robust_acc, and
   its relative form min/mean, for each model. The seed predicts min/mean
   <= 0.6 for every AT model.
(4) Confusion matrix of *successful* adversarial flips on each model: for
   each (true_class, attacked_class) pair, count flips. Then per-target-class
   "attractor rate" = fraction of all flips that land in that class. Test
   whether class 6 (SHIRT) over-receives flips.
(5) Per-class margin distribution (mean / std / q10 / q90 over correctly-
   classified samples) on each model, to localise WHERE the worst class is
   thin.

Output: results/fashion_mnist/h480_per_class_worst_case_asr_output.txt
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config -----------------------------------------------------------------
DS         = "fashion_mnist"
SEED       = 0
EPS        = 0.1
PGD_STEPS_EVAL   = 10        # attack budget at eval time
PGD_STEPS_TRAIN  = 7         # inner AT budget (Madry default for MNIST/FMNIST)
EPOCHS     = 6               # matched compute across victims
BATCH      = 128
N_TRAIN    = 12000           # campaign-typical fast budget
N_EVAL     = 2000            # full per-class breakdown still has ~200/class
BETA_TRADES = 6.0            # Zhang et al. 2019 default

CLASS_NAMES = ["T-shirt","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle-boot"]
NCLS = 10

OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "fashion_mnist")
OUT_FILE = os.path.join(OUT_DIR, "h480_per_class_worst_case_asr_output.txt")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- buffered logger --------------------------------------------------------
_LINES = []
def log(s=""):
    print(s)
    _LINES.append(s)


# ============================================================================
# TRADES inner adversary + custom training loops not covered by common.py
# ============================================================================
def _trades_inner_adv(model, x, eps, steps, alpha):
    """Find x_adv that maximises KL(p(x_adv) || p(x)). Standard TRADES inner."""
    model.eval()
    with torch.no_grad():
        p_nat = F.softmax(model(x), dim=1)
    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logp_adv = F.log_softmax(model(x_adv), dim=1)
        loss_kl = F.kl_div(logp_adv, p_nat, reduction="batchmean")
        g, = torch.autograd.grad(loss_kl, x_adv)
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    return x_adv.detach()


def train_trades(model, Xtr, Ytr, epochs, batch, eps, steps, alpha, beta, lr=0.05):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            x_adv = _trades_inner_adv(model, xb, eps, steps, alpha)
            model.train()
            opt.zero_grad()
            logits_nat = model(xb)
            logits_adv = model(x_adv)
            loss_natural = F.cross_entropy(logits_nat, yb)
            loss_robust = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_nat, dim=1),
                reduction="batchmean",
            )
            loss = loss_natural + beta * loss_robust
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_fastat(model, Xtr, Ytr, epochs, batch, eps, lr=0.05):
    """Wong et al. ICLR 2020 'Fast is better than free': single-step FGSM-AT
    with uniform random init in [-eps, eps]. This is the open-source proxy for
    Shafahi et al. Free-AT - same compute as standard training.
    """
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    alpha = 1.25 * eps   # Wong et al. recommend alpha = 1.25 * eps for FGSM-AT
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            # random init in eps-ball
            delta = torch.empty_like(xb).uniform_(-eps, eps)
            xa = (xb + delta).clamp(0, 1).detach().requires_grad_(True)
            loss = F.cross_entropy(model(xa), yb)
            g, = torch.autograd.grad(loss, xa)
            xa = (xa.detach() + alpha * g.sign())
            xa = torch.min(torch.max(xa, xb - eps), xb + eps).clamp(0, 1)
            opt.zero_grad()
            loss2 = F.cross_entropy(model(xa), yb)
            loss2.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ============================================================================
# Per-class evaluation
# ============================================================================
def eval_per_class(model, X, Y, eps, steps):
    """Return a dict with per-class clean/robust accuracy, per-class margin
    distribution stats, and the confusion matrix of SUCCESSFUL flips (rows =
    true class, cols = attacked class on the adversarial example).
    """
    model.eval()
    # clean predictions + margins
    logits_clean, clean_acc = C.logits_and_acc(model, X, Y)
    preds_clean = logits_clean.argmax(1).cpu().numpy()
    Y_np = Y.cpu().numpy()
    margins = C.margin_of(logits_clean, Y)   # numpy (N,)

    # PGD attack batched (use common.pgd via attack_success-style loop)
    pgd_preds = np.empty_like(Y_np)
    B = 256
    for i in range(0, X.size(0), B):
        x, y = X[i:i + B], Y[i:i + B]
        xa = C.pgd(model, x, y, eps=eps, steps=steps)
        with torch.no_grad():
            pgd_preds[i:i + B] = model(xa).argmax(1).cpu().numpy()

    per = []
    confusion = np.zeros((NCLS, NCLS), dtype=np.int64)   # true x adversarial_pred
    for c in range(NCLS):
        idx = np.where(Y_np == c)[0]
        n = len(idx)
        if n == 0:
            per.append({"n": 0})
            continue
        correct_mask = preds_clean[idx] == c
        n_correct = int(correct_mask.sum())
        clean_acc_c = float(correct_mask.mean())

        # robust acc on this class = fraction of class-c samples where the
        # PGD adversarial is *still* predicted as c. (Standard convention -
        # mirrors the robust-acc number cited in Madry/Trades/Xu papers.)
        robust_mask_all = pgd_preds[idx] == c
        robust_acc_c = float(robust_mask_all.mean())

        # margin stats over correctly-classified-only (more meaningful)
        mc = margins[idx[correct_mask]] if n_correct > 0 else np.array([np.nan])
        per.append({
            "n": n,
            "n_correct_clean": n_correct,
            "clean_acc": clean_acc_c,
            "robust_acc": robust_acc_c,
            "margin_mean": float(np.nanmean(mc)) if n_correct > 0 else float("nan"),
            "margin_std":  float(np.nanstd(mc))  if n_correct > 0 else float("nan"),
            "margin_q10":  float(np.nanquantile(mc, 0.10)) if n_correct > 0 else float("nan"),
            "margin_q90":  float(np.nanquantile(mc, 0.90)) if n_correct > 0 else float("nan"),
        })

        # confusion of SUCCESSFUL flips: clean-correct AND adv-flipped
        flipped_idx = idx[correct_mask & (pgd_preds[idx] != c)]
        for j in flipped_idx:
            confusion[c, pgd_preds[j]] += 1

    return {
        "clean_acc_overall": float(clean_acc),
        "robust_acc_overall": float((pgd_preds == Y_np).mean()),
        "per_class": per,
        "confusion": confusion,
    }


# ============================================================================
# main
# ============================================================================
def main():
    t_all = time.time()
    log("=" * 78)
    log("H480  Per-class WORST-CASE robust accuracy: do AT defenses hide a vulnerable class?")
    log("=" * 78)
    log(f"device={C.DEVICE}  dataset={DS}  eps={EPS}  pgd_eval_steps={PGD_STEPS_EVAL}")
    log(f"epochs={EPOCHS}  batch={BATCH}  n_train={N_TRAIN}  n_eval={N_EVAL}  seed={SEED}")
    log("Refs: Xu+ NeurIPS'21 'To be Robust or to be Fair'; Tian+ CVPR'21 'Class-wise")
    log("      Robustness'; Benz+ 'Robustness May Be at Odds with Fairness'.")
    log("")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    log(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    log("")

    # ---- train the four victims ---------------------------------------------
    victims = {}
    alpha_train = 2.5 * EPS / PGD_STEPS_TRAIN

    # STD
    log("[train] STD - standard cross-entropy SGD")
    C.set_seed(SEED)
    m_std = C.build_model("cnn", meta, width=32, seed=SEED)
    t0 = time.time()
    C.train_model(m_std, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                  opt="sgd", lr=0.05, ncls=NCLS, adv_train=False)
    log(f"        done in {time.time()-t0:.1f}s")
    victims["STD"] = m_std

    # PGDAT
    log("[train] PGDAT - Madry PGD-AT (eps=0.1, 7 steps)")
    C.set_seed(SEED)
    m_pgd = C.build_model("cnn", meta, width=32, seed=SEED)
    t0 = time.time()
    C.train_model(m_pgd, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                  opt="sgd", lr=0.05, ncls=NCLS,
                  adv_train=True, adv_eps=EPS, adv_steps=PGD_STEPS_TRAIN)
    log(f"        done in {time.time()-t0:.1f}s")
    victims["PGDAT"] = m_pgd

    # TRADES
    log("[train] TRADES - KL-regularised AT (beta=6, eps=0.1, 7 steps)")
    C.set_seed(SEED)
    m_tr = C.build_model("cnn", meta, width=32, seed=SEED)
    t0 = time.time()
    train_trades(m_tr, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                 eps=EPS, steps=PGD_STEPS_TRAIN, alpha=alpha_train, beta=BETA_TRADES, lr=0.05)
    log(f"        done in {time.time()-t0:.1f}s")
    victims["TRADES"] = m_tr

    # FASTAT
    log("[train] FASTAT - Wong+ ICLR'20 fast FGSM-AT (rand-init, alpha=1.25*eps)")
    C.set_seed(SEED)
    m_fast = C.build_model("cnn", meta, width=32, seed=SEED)
    t0 = time.time()
    train_fastat(m_fast, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, eps=EPS, lr=0.05)
    log(f"        done in {time.time()-t0:.1f}s")
    victims["FASTAT"] = m_fast

    # ---- per-class eval -----------------------------------------------------
    results = {}
    for name, m in victims.items():
        log("")
        log(f"[eval] {name}")
        t0 = time.time()
        r = eval_per_class(m, Xte, Yte, EPS, PGD_STEPS_EVAL)
        log(f"       clean_acc_overall={r['clean_acc_overall']:.3f}   "
            f"robust_acc_overall={r['robust_acc_overall']:.3f}   "
            f"({time.time()-t0:.1f}s)")
        results[name] = r

    # ---- per-class breakdown table ------------------------------------------
    log("")
    log("=" * 78)
    log("PER-CLASS CLEAN ACC / ROBUST ACC (eps=0.1, PGD-10)")
    log("=" * 78)
    hdr = f"{'class':<11} " + "  ".join(f"{n:>16}" for n in victims.keys())
    log(hdr)
    log(f"{'':<11} " + "  ".join(f"{'clean / robust':>16}" for _ in victims.keys()))
    log("-" * len(hdr))
    for c in range(NCLS):
        row = [f"{c}:{CLASS_NAMES[c]:<8}"]
        for name in victims.keys():
            p = results[name]["per_class"][c]
            row.append(f"{p['clean_acc']:>7.3f} /{p['robust_acc']:>7.3f}")
        log("  ".join(row))
    log("-" * len(hdr))

    # ---- mean / min / robustness gap ----------------------------------------
    log("")
    log("=" * 78)
    log("AGGREGATES vs WORST-CASE")
    log("=" * 78)
    log(f"{'model':<8} {'mean_clean':>11} {'min_clean':>10} {'worst_clean_cls':<18}"
        f" {'mean_rob':>10} {'min_rob':>9} {'worst_rob_cls':<18} {'gap':>7} {'min/mean':>9}")
    summary = {}
    for name in victims.keys():
        pc = results[name]["per_class"]
        cleans = np.array([p["clean_acc"] for p in pc])
        robs   = np.array([p["robust_acc"] for p in pc])
        mc, mr = float(cleans.mean()), float(robs.mean())
        argmin_c = int(cleans.argmin()); argmin_r = int(robs.argmin())
        ratio = robs.min() / mr if mr > 1e-9 else float("nan")
        gap = mr - float(robs.min())
        log(f"{name:<8} {mc:>11.3f} {cleans.min():>10.3f} "
            f"{CLASS_NAMES[argmin_c]+' ('+str(argmin_c)+')':<18} "
            f"{mr:>10.3f} {robs.min():>9.3f} "
            f"{CLASS_NAMES[argmin_r]+' ('+str(argmin_r)+')':<18} "
            f"{gap:>7.3f} {ratio:>9.3f}")
        summary[name] = {
            "mean_clean": mc, "min_clean": float(cleans.min()), "argmin_clean": argmin_c,
            "mean_rob": mr,   "min_rob":   float(robs.min()),   "argmin_rob":   argmin_r,
            "ratio_min_over_mean": ratio, "gap": gap,
        }
    log("")
    log("Seed prediction: min_rob / mean_rob <= 0.60 for every AT variant")
    log("                 (i.e. defenses look strong on average but mask a vulnerable class).")

    # ---- attractor analysis: do flips concentrate on a particular class? ----
    log("")
    log("=" * 78)
    log("FLIP-DESTINATION (ATTRACTOR) RATES per model")
    log("=" * 78)
    log("For each model: of all successful PGD flips, what fraction land in each class?")
    log("(High value for a class = it is an adversarial attractor, not just a victim.)")
    log("")
    log(f"{'class':<11} " + "  ".join(f"{n:>10}" for n in victims.keys()))
    attractor = {}
    for name in victims.keys():
        conf = results[name]["confusion"]
        total = conf.sum()
        attractor[name] = conf.sum(axis=0) / max(int(total), 1)
    for c in range(NCLS):
        row = [f"{c}:{CLASS_NAMES[c]:<8}"]
        for name in victims.keys():
            row.append(f"{attractor[name][c]:>10.3f}")
        log("  ".join(row))

    # which class dominates flips? does class 6 (shirt) win?
    log("")
    log("Top attractor class per model:")
    for name in victims.keys():
        top = int(np.argmax(attractor[name]))
        log(f"  {name:<8}  top_attractor={CLASS_NAMES[top]} ({top})  "
            f"frac={attractor[name][top]:.3f}   "
            f"class6(Shirt)_frac={attractor[name][6]:.3f}")

    # ---- per-class margin distribution table --------------------------------
    log("")
    log("=" * 78)
    log("PER-CLASS MARGIN DISTRIBUTION (correctly-classified only)")
    log("=" * 78)
    for name in victims.keys():
        log(f"\n-- {name} --")
        log(f"{'class':<11} {'n_corr':>7} {'mean':>9} {'std':>9} {'q10':>9} {'q90':>9}")
        for c in range(NCLS):
            p = results[name]["per_class"][c]
            log(f"{c}:{CLASS_NAMES[c]:<8} {p.get('n_correct_clean',0):>7d} "
                f"{p['margin_mean']:>9.3f} {p['margin_std']:>9.3f} "
                f"{p['margin_q10']:>9.3f} {p['margin_q90']:>9.3f}")

    # ---- HEADLINE verdict ---------------------------------------------------
    log("")
    log("=" * 78)
    log("HEADLINE VERDICT")
    log("=" * 78)
    # Check whether all AT variants satisfy min_rob/mean_rob <= 0.60
    at_models = [k for k in victims.keys() if k != "STD"]
    ratios = {k: summary[k]["ratio_min_over_mean"] for k in at_models}
    worst_classes = {k: summary[k]["argmin_rob"] for k in at_models}
    shirt_wins = sum(1 for k in at_models if worst_classes[k] == 6)

    log("Hypothesis 1 (worst-class <= 60% of mean for every AT variant):")
    for k in at_models:
        verdict = "YES" if ratios[k] <= 0.60 else "NO"
        log(f"   {k:<8}  min_rob/mean_rob = {ratios[k]:.3f}   -> {verdict}")
    hyp1 = all(ratios[k] <= 0.60 for k in at_models)
    log(f"   H1 SUPPORTED ACROSS ALL AT VARIANTS: {hyp1}")

    log("")
    log("Hypothesis 2 (class 6 'Shirt' is the worst class on most AT models):")
    log(f"   Worst-robust class per AT model: " +
        ", ".join(f"{k}={CLASS_NAMES[worst_classes[k]]}" for k in at_models))
    log(f"   shirt_is_worst on {shirt_wins}/{len(at_models)} AT models  -> "
        f"{'SUPPORTED' if shirt_wins > len(at_models)//2 else 'NOT SUPPORTED'}")

    log("")
    log("Implication (advisor critique): aggregate robust accuracy alone is NOT a")
    log("complete robustness report. Worst-class robust acc is the headline number")
    log("a deployment should see; the gap mean-min flags fairness failure, and the")
    log("attractor table flags which class to target.")
    log(f"\nTotal runtime: {time.time()-t_all:.1f}s")
    log("=" * 78)

    # ---- write output -------------------------------------------------------
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(_LINES) + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
