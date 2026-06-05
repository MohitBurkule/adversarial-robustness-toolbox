"""
H386 - Jacobian Frobenius Penalty + Confidence Weighting.

Hypothesis: combining a Jacobian Frobenius penalty with per-sample confidence
weighting (penalise high-confidence samples more) beats either alone.

Implementation:
  - Estimate ||J(x)||_F² via a single-sample Hutchinson estimator:
        v ~ N(0, I) with the shape of the logits
        jvp = autograd.grad((f(x) * v).sum(), x, create_graph=True)[0]
        penalty_per_sample = (jvp**2).flatten(1).sum(1)        # ≈ ||J^T v||²
    Over v~N(0,I), E[||J^T v||²] = ||J||_F², so one sample is an unbiased estimate.
  - Weight each sample by conf(x) = max softmax prob, raised to power p.
        loss = CE + λ · mean( conf^p · penalty_per_sample )
    p=0 reduces to uniform weighting (plain unweighted Jacobian penalty).

Grid:
  λ ∈ {0.01, 0.1}, p ∈ {1, 2}, plus baselines:
    (λ=0)                — plain CE
    (uniform, p=0)       — unweighted Jacobian penalty at each λ
Compare confidence-weighted cells to the unweighted Jacobian penalty.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH_SIZE = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist"
)
OUT_FILE = os.path.join(RESULTS_DIR, "h386_jacobian_confidence_weighted_output.txt")


def _make_sgd(model, lr=LR):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


def train_jacobian(model, Xtr, Ytr, epochs, lam, p, lr=LR):
    """L = CE + lam * mean( conf^p * ||J^T v||² ),  v~N(0,I) (1 Hutchinson sample).

    p == 0 -> uniform weighting (conf^0 = 1), i.e. unweighted Jacobian penalty.
    lam == 0 -> plain CE.
    """
    opt = _make_sgd(model, lr)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - BATCH_SIZE, BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            yb = Ytr[idx].to(C.DEVICE)

            opt.zero_grad()
            xb = Xtr[idx].to(C.DEVICE).clone().requires_grad_(True)
            logits = model(xb)
            ce = F.cross_entropy(logits, yb)
            total = ce

            if lam > 0.0:
                v = torch.randn_like(logits)
                jvp = torch.autograd.grad(
                    (logits * v).sum(), xb, create_graph=True, retain_graph=True
                )[0]
                pen_per_sample = jvp.pow(2).flatten(1).sum(1)   # (B,)

                if p == 0:
                    weight = torch.ones_like(pen_per_sample)
                else:
                    conf = F.softmax(logits, dim=1).max(dim=1).values.detach()
                    weight = conf.pow(p)

                pen = (weight * pen_per_sample).mean()
                total = total + lam * pen

            total.backward()
            opt.step()
    model.eval()
    return model


def eval_model(model, Xte, Yte):
    for q in model.parameters():
        q.requires_grad_(True)

    _, clean_acc = C.logits_and_acc(model, Xte, Yte)

    Xadv_fgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, acc_fgsm = C.logits_and_acc(model, Xadv_fgsm, Yte)
    fgsm_asr = 1.0 - acc_fgsm

    Xadv_pgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, acc_pgd = C.logits_and_acc(model, Xadv_pgd, Yte)
    pgd_asr = 1.0 - acc_pgd

    margins = C.margin(model, Xte, Yte)
    mean_margin = float(np.mean(margins))

    return {
        "clean_acc": float(clean_acc),
        "fgsm_asr": float(fgsm_asr),
        "pgd_asr": float(pgd_asr),
        "mean_margin": mean_margin,
    }


# (key, label, lam, p)
CONDITIONS = [
    ("baseline",       "Baseline (CE only) λ=0",            0.0,  0),
    ("uniform_l001",   "Jacobian uniform p=0 λ=0.01",       0.01, 0),
    ("uniform_l01",    "Jacobian uniform p=0 λ=0.1",        0.1,  0),
    ("conf1_l001",     "Conf-weighted p=1 λ=0.01",          0.01, 1),
    ("conf1_l01",      "Conf-weighted p=1 λ=0.1",           0.1,  1),
    ("conf2_l001",     "Conf-weighted p=2 λ=0.01",          0.01, 2),
    ("conf2_l01",      "Conf-weighted p=2 λ=0.1",           0.1,  2),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    C.set_seed(SEED)

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    results = []
    for key, label, lam, p in CONDITIONS:
        print(f"\n{'='*60}\nCondition: {label}")
        C.set_seed(SEED)
        model = C.build_model(
            "cnn", {"channels": 1, "size": 28, "n_classes": 10}, width=32, seed=SEED
        )
        t0 = time.time()
        model = train_jacobian(model, Xtr, Ytr, EPOCHS, lam, p)
        dt = time.time() - t0
        m = eval_model(model, Xte, Yte)
        m["condition"] = label
        m["lam"] = lam
        m["p"] = p
        m["train_time_s"] = dt
        results.append(m)
        print(
            f"  clean_acc={m['clean_acc']:.4f}  fgsm_asr={m['fgsm_asr']:.4f}  "
            f"pgd_asr={m['pgd_asr']:.4f}  margin={m['mean_margin']:.4f}  ({dt:.1f}s)"
        )

    by_key = {c[0]: r for (c, r) in zip(CONDITIONS, results)}

    # Compare confidence-weighted vs unweighted Jacobian at matched λ.
    comparisons = []
    for lam, uni_key in [(0.01, "uniform_l001"), (0.1, "uniform_l01")]:
        uni = by_key[uni_key]
        for ck in [k for (k, lbl, l, pp) in CONDITIONS if l == lam and pp > 0]:
            cw = by_key[ck]
            comparisons.append((lam, uni, cw, uni["pgd_asr"] - cw["pgd_asr"]))

    # best confidence-weighted vs best uniform on PGD ASR
    best_uniform = min((by_key["uniform_l001"], by_key["uniform_l01"]),
                       key=lambda r: r["pgd_asr"])
    cw_keys = ["conf1_l001", "conf1_l01", "conf2_l001", "conf2_l01"]
    best_cw = min((by_key[k] for k in cw_keys), key=lambda r: r["pgd_asr"])

    cw_beats = best_cw["pgd_asr"] < best_uniform["pgd_asr"] - 1e-4
    # also require it beats best single component (uniform Jacobian is the "either alone")
    if cw_beats:
        verdict = "SUPPORTED"
    elif best_cw["pgd_asr"] <= best_uniform["pgd_asr"] + 0.01:
        verdict = "PARTIAL"
    else:
        verdict = "NOT SUPPORTED"

    lines = [
        "H386 Jacobian Frobenius Penalty + Confidence Weighting",
        "=" * 70,
        f"Dataset: {DS}  N_train={N_TRAIN}  N_eval={N_EVAL}  Epochs={EPOCHS}  Seed={SEED}",
        f"EPS={EPS}  PGD_STEPS={PGD_STEPS}  PGD_ALPHA={PGD_ALPHA}",
        "Hutchinson: 1 sample/step, v~N(0,I); penalty=||J^T v||^2 (unbiased ||J||_F^2)",
        "loss = CE + lambda * mean( conf^p * penalty );  p=0 -> uniform weighting",
        "",
    ]
    col_w = 30
    header = (f"{'Condition':<{col_w}}  {'clean_acc':>10}  {'fgsm_asr':>9}  "
              f"{'pgd_asr':>8}  {'margin':>8}  {'time_s':>7}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"{r['condition']:<{col_w}}  {r['clean_acc']:>10.4f}  {r['fgsm_asr']:>9.4f}  "
            f"{r['pgd_asr']:>8.4f}  {r['mean_margin']:>8.4f}  {r['train_time_s']:>7.1f}"
        )

    lines += ["", "Matched-lambda comparisons (uniform vs confidence-weighted):"]
    for lam, uni, cw, delta in comparisons:
        lines.append(
            f"  λ={lam}: {cw['condition']:<26} PGD_ASR={cw['pgd_asr']:.4f}  "
            f"vs uniform {uni['pgd_asr']:.4f}  (Δ={delta:+.4f}, +=cw better)"
        )

    lines += [
        "",
        "Analysis:",
        f"- Baseline PGD_ASR = {by_key['baseline']['pgd_asr']:.4f}",
        f"- Best unweighted Jacobian: {best_uniform['condition']}  "
        f"PGD_ASR={best_uniform['pgd_asr']:.4f}  clean_acc={best_uniform['clean_acc']:.4f}",
        f"- Best confidence-weighted: {best_cw['condition']}  "
        f"PGD_ASR={best_cw['pgd_asr']:.4f}  clean_acc={best_cw['clean_acc']:.4f}",
        f"- Best CW improvement vs best uniform (PGD ASR): "
        f"{(best_uniform['pgd_asr'] - best_cw['pgd_asr']):+.4f}",
        "- Rationale: high-confidence samples sit deep in their basin; penalising their",
        "  Jacobian harder should flatten the decision surface where the model is overconfident.",
        "",
        f"VERDICT: {verdict} — confidence weighting "
        + ("beats" if cw_beats else "does not clearly beat")
        + " the unweighted Jacobian penalty on PGD ASR "
        f"(best CW {best_cw['pgd_asr']:.4f} vs best uniform {best_uniform['pgd_asr']:.4f}).",
    ]

    report = "\n".join(lines)
    print("\n" + report)
    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
