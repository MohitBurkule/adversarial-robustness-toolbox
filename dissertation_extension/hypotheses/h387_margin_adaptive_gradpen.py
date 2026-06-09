"""
H387 - Margin-Adaptive Input-Gradient Penalty Weighting.

Hypothesis: weighting the input-grad penalty by inverse margin (low-margin
samples weighted higher) improves the robustness-accuracy frontier vs the
uniform input-grad penalty (H288).

Implementation:
  Per batch, compute per-sample margin from logits:
      margin_i = correct_logit_i - max_{j != y} logit_{i,j}
  Adaptive weights via a temperature-softmax over the negated margin:
      w_i = softmax(-margin_i / T)            (sum_i w_i = 1 over the batch)
  so low-margin (more vulnerable) samples receive larger weight.

  Per-sample input-grad norm:
      Take the total CE loss (sum over batch); grad wrt x gives per-sample grad
      maps (since each sample's loss depends only on its own input row), so
      ||grad_x L_i||² = (grad_x[i]**2).sum(). Square-sum per sample, then weight:
      loss = CE + lambda * sum_i w_i * ||grad_x L_i||²

Grid:
  T ∈ {0.5, 1, 2}, λ ∈ {0.01, 0.1}, plus baselines:
    uniform-weight (== H288 input-grad penalty) at each λ
    λ=0 (plain CE)
Verdict: does adaptive weighting beat uniform at matched λ?
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
OUT_FILE = os.path.join(RESULTS_DIR, "h387_margin_adaptive_gradpen_output.txt")


def _make_sgd(model, lr=LR):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


def _per_sample_margin(logits, yb):
    """margin_i = correct logit - max other-class logit. Returns (B,) tensor."""
    correct = logits.gather(1, yb[:, None]).squeeze(1)
    tmp = logits.clone()
    tmp[torch.arange(tmp.size(0), device=tmp.device), yb] = -1e9
    other = tmp.max(dim=1).values
    return correct - other


def train_adaptive(model, Xtr, Ytr, epochs, lam, T, uniform, lr=LR):
    """L = CE + lam * sum_i w_i * ||grad_x L_i||².

    uniform=True  -> w_i = 1/B  (matches H288 mean input-grad penalty)
    uniform=False -> w_i = softmax(-margin_i / T)  (low-margin weighted higher)
    lam == 0      -> plain CE.
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
                # grad of the *total* CE wrt x -> per-sample grad maps.
                grad_x = torch.autograd.grad(
                    ce, xb, create_graph=True, retain_graph=True
                )[0]
                gnorm2 = grad_x.pow(2).flatten(1).sum(1)   # ||grad_x L_i||² (B,)

                B = gnorm2.size(0)
                if uniform:
                    w = torch.full((B,), 1.0 / B, device=gnorm2.device)
                else:
                    m = _per_sample_margin(logits, yb).detach()
                    w = F.softmax(-m / T, dim=0)            # sums to 1

                pen = (w * gnorm2).sum()
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


# (key, label, lam, T, uniform)
CONDITIONS = [
    ("baseline",      "Baseline (CE only) λ=0",          0.0,  1.0, True),
    ("uniform_l001",  "Uniform (H288) λ=0.01",           0.01, 1.0, True),
    ("uniform_l01",   "Uniform (H288) λ=0.1",            0.1,  1.0, True),
    ("adapt_T05_l001", "Adaptive T=0.5 λ=0.01",          0.01, 0.5, False),
    ("adapt_T1_l001",  "Adaptive T=1 λ=0.01",            0.01, 1.0, False),
    ("adapt_T2_l001",  "Adaptive T=2 λ=0.01",            0.01, 2.0, False),
    ("adapt_T05_l01",  "Adaptive T=0.5 λ=0.1",           0.1,  0.5, False),
    ("adapt_T1_l01",   "Adaptive T=1 λ=0.1",             0.1,  1.0, False),
    ("adapt_T2_l01",   "Adaptive T=2 λ=0.1",             0.1,  2.0, False),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    C.set_seed(SEED)

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    results = []
    for key, label, lam, T, uniform in CONDITIONS:
        print(f"\n{'='*60}\nCondition: {label}")
        C.set_seed(SEED)
        model = C.build_model(
            "cnn", {"channels": 1, "size": 28, "n_classes": 10}, width=32, seed=SEED
        )
        t0 = time.time()
        model = train_adaptive(model, Xtr, Ytr, EPOCHS, lam, T, uniform)
        dt = time.time() - t0
        m = eval_model(model, Xte, Yte)
        m["condition"] = label
        m["lam"] = lam
        m["T"] = T
        m["uniform"] = uniform
        m["train_time_s"] = dt
        results.append(m)
        print(
            f"  clean_acc={m['clean_acc']:.4f}  fgsm_asr={m['fgsm_asr']:.4f}  "
            f"pgd_asr={m['pgd_asr']:.4f}  margin={m['mean_margin']:.4f}  ({dt:.1f}s)"
        )

    by_key = {c[0]: r for (c, r) in zip(CONDITIONS, results)}

    # Matched-lambda comparisons: each adaptive cell vs uniform at same λ.
    comparisons = []
    for lam, uni_key, adapt_keys in [
        (0.01, "uniform_l001", ["adapt_T05_l001", "adapt_T1_l001", "adapt_T2_l001"]),
        (0.1,  "uniform_l01",  ["adapt_T05_l01", "adapt_T1_l01", "adapt_T2_l01"]),
    ]:
        uni = by_key[uni_key]
        for ak in adapt_keys:
            a = by_key[ak]
            comparisons.append((lam, uni, a, uni["pgd_asr"] - a["pgd_asr"]))

    # Best adaptive vs best uniform (by PGD ASR).
    best_uniform = min((by_key["uniform_l001"], by_key["uniform_l01"]),
                       key=lambda r: r["pgd_asr"])
    adapt_keys_all = ["adapt_T05_l001", "adapt_T1_l001", "adapt_T2_l001",
                      "adapt_T05_l01", "adapt_T1_l01", "adapt_T2_l01"]
    best_adapt = min((by_key[k] for k in adapt_keys_all), key=lambda r: r["pgd_asr"])

    # Count how many matched-lambda comparisons adaptive wins.
    wins = sum(1 for (_, _, _, d) in comparisons if d > 1e-4)

    adapt_beats = best_adapt["pgd_asr"] < best_uniform["pgd_asr"] - 1e-4
    if adapt_beats and wins >= len(comparisons) / 2:
        verdict = "SUPPORTED"
    elif adapt_beats or wins > 0:
        verdict = "PARTIAL"
    else:
        verdict = "NOT SUPPORTED"

    lines = [
        "H387 Margin-Adaptive Input-Gradient Penalty Weighting",
        "=" * 70,
        f"Dataset: {DS}  N_train={N_TRAIN}  N_eval={N_EVAL}  Epochs={EPOCHS}  Seed={SEED}",
        f"EPS={EPS}  PGD_STEPS={PGD_STEPS}  PGD_ALPHA={PGD_ALPHA}",
        "w_i = softmax(-margin_i / T) over batch (low-margin weighted higher)",
        "loss = CE + lambda * sum_i w_i * ||grad_x L_i||^2 ; uniform == H288",
        "",
    ]
    col_w = 28
    header = (f"{'Condition':<{col_w}}  {'clean_acc':>10}  {'fgsm_asr':>9}  "
              f"{'pgd_asr':>8}  {'margin':>8}  {'time_s':>7}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"{r['condition']:<{col_w}}  {r['clean_acc']:>10.4f}  {r['fgsm_asr']:>9.4f}  "
            f"{r['pgd_asr']:>8.4f}  {r['mean_margin']:>8.4f}  {r['train_time_s']:>7.1f}"
        )

    lines += ["", "Matched-lambda comparisons (uniform vs adaptive):"]
    for lam, uni, a, delta in comparisons:
        lines.append(
            f"  λ={lam}: {a['condition']:<24} PGD_ASR={a['pgd_asr']:.4f}  "
            f"vs uniform {uni['pgd_asr']:.4f}  (Δ={delta:+.4f}, +=adaptive better)"
        )

    lines += [
        "",
        "Analysis:",
        f"- Baseline PGD_ASR = {by_key['baseline']['pgd_asr']:.4f}",
        f"- Best uniform (H288): {best_uniform['condition']}  "
        f"PGD_ASR={best_uniform['pgd_asr']:.4f}  clean_acc={best_uniform['clean_acc']:.4f}",
        f"- Best adaptive:       {best_adapt['condition']}  "
        f"PGD_ASR={best_adapt['pgd_asr']:.4f}  clean_acc={best_adapt['clean_acc']:.4f}",
        f"- Best adaptive improvement vs best uniform (PGD ASR): "
        f"{(best_uniform['pgd_asr'] - best_adapt['pgd_asr']):+.4f}",
        f"- Adaptive wins {wins}/{len(comparisons)} matched-lambda comparisons.",
        "- Rationale: focusing the smoothing budget on low-margin (near-boundary) samples",
        "  should push the robustness-accuracy frontier outward vs spending it uniformly.",
        "",
        f"VERDICT: {verdict} — adaptive weighting "
        + ("beats" if adapt_beats else "does not clearly beat")
        + " uniform on PGD ASR "
        f"(best adaptive {best_adapt['pgd_asr']:.4f} vs best uniform {best_uniform['pgd_asr']:.4f}).",
    ]

    report = "\n".join(lines)
    print("\n" + report)
    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
