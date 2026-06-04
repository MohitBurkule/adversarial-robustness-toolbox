"""
H186 - Probability-space margin vs logit-space margin as predictors of
       per-sample adversarial vulnerability.

Paper 2411.15210 (Probability Margin Attack, PMA) argues that the gap in
*softmax probability* space — prob_margin = softmax[top1] - softmax[top2] —
better captures a model's genuine confidence than the raw logit gap
logit_margin = logit[top1] - logit[top2].

We test: for Fashion-MNIST, which margin better predicts FGSM / PGD success
(attack flips the prediction)?

Metric: AUROC where the binary label = (attack succeeded) and the score =
-margin (lower margin => should be easier to attack => higher AUROC).

We also compute the Spearman rank correlation between the two margin
variants to understand how much they diverge in practice.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C
from scipy.stats import spearmanr

DS = "fashion_mnist"
EPS = 0.1
PGD_STEPS = 10
N_EVAL = 500
SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def compute_margins(model, X, Y, batch=256):
    """Return logit_margin and prob_margin arrays (numpy, length N)."""
    logit_list, prob_list = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            xb = X[i:i + batch]
            lg = model(xb).cpu()
            pb = F.softmax(lg, dim=1)
            logit_list.append(lg)
            prob_list.append(pb)
    logits = torch.cat(logit_list)   # (N, C)
    probs  = torch.cat(prob_list)    # (N, C)
    Y_cpu  = Y.cpu()

    # logit margin: correct - best_other
    logit_margin = C.margin_of(logits, Y_cpu)   # numpy (N,)

    # prob margin: top1 prob - top2 prob
    sorted_p, _ = probs.sort(dim=1, descending=True)
    prob_margin  = (sorted_p[:, 0] - sorted_p[:, 1]).numpy()   # numpy (N,)

    return logit_margin, prob_margin


def attack_success(model, X, Y, attack="fgsm", eps=EPS, steps=PGD_STEPS):
    """Return binary array: 1 = attack succeeded (prediction flipped), 0 = held."""
    model.eval()
    if attack == "fgsm":
        Xadv = C.fgsm(model, X, Y, eps)
    else:
        Xadv = C.pgd(model, X, Y, eps, steps=steps)
    with torch.no_grad():
        preds = model(Xadv).argmax(1).cpu()
    return (preds != Y.cpu()).numpy().astype(int)


def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    _, _, Xte, Yte = C.load_dataset(DS)
    Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]

    model = C.build_model("cnn", meta, seed=seed)
    C.train_model(model, *C.load_dataset(DS)[:2], epochs=10,
                  opt="sgd", lr=0.05, ncls=meta["n_classes"])

    # keep only correctly-classified samples (attacking wrong predictions is noise)
    model.eval()
    with torch.no_grad():
        clean_preds = model(Xte).argmax(1).cpu()
    correct_mask = (clean_preds == Yte.cpu())
    Xc = Xte[correct_mask.to(Xte.device)]
    Yc = Yte[correct_mask.to(Yte.device)]

    logit_m, prob_m = compute_margins(model, Xc, Yc)

    fgsm_succ = attack_success(model, Xc, Yc, attack="fgsm")
    pgd_succ  = attack_success(model, Xc, Yc, attack="pgd")

    # AUROC: score = -margin (low margin => easier to attack => should be 1)
    auroc_logit_fgsm = C.safe_auroc(fgsm_succ, -logit_m)
    auroc_logit_pgd  = C.safe_auroc(pgd_succ,  -logit_m)
    auroc_prob_fgsm  = C.safe_auroc(fgsm_succ, -prob_m)
    auroc_prob_pgd   = C.safe_auroc(pgd_succ,  -prob_m)

    spear_r, spear_p = spearmanr(logit_m, prob_m)

    return {
        "seed": seed,
        "n_correct": int(correct_mask.sum()),
        "fgsm_success_rate": float(fgsm_succ.mean()),
        "pgd_success_rate":  float(pgd_succ.mean()),
        "auroc_logit_fgsm":  auroc_logit_fgsm,
        "auroc_logit_pgd":   auroc_logit_pgd,
        "auroc_prob_fgsm":   auroc_prob_fgsm,
        "auroc_prob_pgd":    auroc_prob_pgd,
        "spearman_r":        float(spear_r),
        "spearman_p":        float(spear_p),
    }


def main():
    print("=" * 74)
    print("H186 - Probability-space vs logit-space margin: adversarial AUROC")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  "
          f"pgd_steps={PGD_STEPS}  n_eval={N_EVAL}")
    print()

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        r = run_seed(seed)
        r["runtime_s"] = round(time.time() - t0, 1)
        rows.append(r)

        print(f"[seed {seed}]  n_correct={r['n_correct']}  "
              f"fgsm_succ={r['fgsm_success_rate']:.3f}  "
              f"pgd_succ={r['pgd_success_rate']:.3f}  "
              f"({r['runtime_s']}s)")
        print(f"  AUROC logit margin : FGSM={r['auroc_logit_fgsm']:.4f}  "
              f"PGD={r['auroc_logit_pgd']:.4f}")
        print(f"  AUROC prob  margin : FGSM={r['auroc_prob_fgsm']:.4f}  "
              f"PGD={r['auroc_prob_pgd']:.4f}")
        print(f"  Spearman(logit,prob): r={r['spearman_r']:.4f}  "
              f"p={r['spearman_p']:.2e}")
        print()

    def m(k):
        vals = [r[k] for r in rows if r[k] == r[k]]
        return float(np.mean(vals)) if vals else float("nan")

    print("=" * 74)
    print("MEAN across seeds")
    print(f"  AUROC logit margin : FGSM={m('auroc_logit_fgsm'):.4f}  "
          f"PGD={m('auroc_logit_pgd'):.4f}")
    print(f"  AUROC prob  margin : FGSM={m('auroc_prob_fgsm'):.4f}  "
          f"PGD={m('auroc_prob_pgd'):.4f}")
    print(f"  Spearman r (mean)  : {m('spearman_r'):.4f}")
    print()

    # determine winner
    delta_fgsm = m('auroc_prob_fgsm') - m('auroc_logit_fgsm')
    delta_pgd  = m('auroc_prob_pgd')  - m('auroc_logit_pgd')
    print("FINDING")
    if delta_fgsm > 0.005 and delta_pgd > 0.005:
        winner = "PROBABILITY margin"
    elif delta_fgsm < -0.005 and delta_pgd < -0.005:
        winner = "LOGIT margin"
    else:
        winner = "NEITHER (tie / inconsistent across attacks)"
    print(f"  Better predictor   : {winner}")
    print(f"  AUROC delta (prob-logit): FGSM={delta_fgsm:+.4f}  PGD={delta_pgd:+.4f}")
    print()
    print("Interpretation:")
    print("  AUROC >> 0.5 means margin reliably separates vulnerable from robust")
    print("  samples. A margin with higher AUROC is a better per-sample oracle.")
    print("  High Spearman r indicates the two margins rank samples similarly;")
    print("  low r would mean softmax nonlinearity genuinely reorders vulnerability.")
    print("=" * 74)


if __name__ == "__main__":
    main()
