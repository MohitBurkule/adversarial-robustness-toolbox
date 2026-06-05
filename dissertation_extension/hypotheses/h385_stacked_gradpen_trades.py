"""
H385 - Stacked Input-Gradient Penalty + TRADES-KL.

Hypothesis: the best implicit method (input-grad penalty ||∇_x L||²) and TRADES-KL
attack robustness via *different routes* and should stack below either alone (i.e.
the combination beats both single components on PGD ASR).

Implementation:
    L = CE(f(x), y)
        + β · KL(softmax(f(x)) || softmax(f(x_adv)))      (TRADES-KL term)
        + λ · ||∇_x L||²                                  (input-grad penalty)

  x_adv is obtained from a short PGD (steps=7, eps=0.1) inside the loop run on
  the CE loss. Following standard TRADES, the *clean* logits are the reference
  distribution: KL(p_clean || p_adv), with p_clean detached as the target.

Grid:
  β ∈ {1, 3}, λ ∈ {0.01, 0.1}, plus baselines:
    (β=0, λ=0)  — plain CE
    (β only)    — TRADES-KL alone (β=1,3 ; λ=0)
    (λ only)    — input-grad penalty alone (λ=0.01,0.1 ; β=0)

Report each cell's clean_acc / FGSM_ASR / PGD_ASR / mean_margin.
Verdict: does the stack beat the best single component (on PGD ASR)?
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

# inner PGD used to make x_adv for the TRADES-KL term during training
TRADES_PGD_STEPS = 7
TRADES_PGD_EPS = 0.1

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist"
)
OUT_FILE = os.path.join(RESULTS_DIR, "h385_stacked_gradpen_trades_output.txt")


def _make_sgd(model, lr=LR):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


def _inner_pgd(model, x, y, eps, steps):
    """Short PGD on the CE loss to build x_adv for the TRADES term.
    Returns a detached adversarial input. Runs under no grad-graph for params.
    """
    alpha = 2.5 * eps / steps
    x0 = x.detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def train_stacked(model, Xtr, Ytr, epochs, beta, lam, lr=LR):
    """Train with L = CE + beta*KL(p_clean||p_adv) + lam*||grad_x L||²."""
    opt = _make_sgd(model, lr)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - BATCH_SIZE, BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            yb = Ytr[idx].to(C.DEVICE)

            # Build x_adv first (no param-graph needed) if TRADES term active
            x_clean = Xtr[idx].to(C.DEVICE)
            if beta > 0.0:
                model.eval()
                x_adv = _inner_pgd(model, x_clean, yb, TRADES_PGD_EPS, TRADES_PGD_STEPS)
                model.train()

            opt.zero_grad()
            xb = x_clean.clone().requires_grad_(True)
            logits_clean = model(xb)
            ce = F.cross_entropy(logits_clean, yb)

            total = ce

            if lam > 0.0:
                grad_x = torch.autograd.grad(
                    ce, xb, create_graph=True, retain_graph=True
                )[0]
                grad_pen = grad_x.pow(2).sum(dim=(1, 2, 3)).mean()
                total = total + lam * grad_pen

            if beta > 0.0:
                logits_adv = model(x_adv)
                # KL(p_clean || p_adv): clean logits are the reference (target),
                # detached so the gradient flows through the adv branch.
                p_clean = F.softmax(logits_clean, dim=1).detach()
                logp_adv = F.log_softmax(logits_adv, dim=1)
                logp_clean = torch.log(p_clean + 1e-12)
                kl = (p_clean * (logp_clean - logp_adv)).sum(dim=1).mean()
                total = total + beta * kl

            total.backward()
            opt.step()
    model.eval()
    return model


def eval_model(model, Xte, Yte):
    for p in model.parameters():
        p.requires_grad_(True)

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


# (key, label, beta, lam)
CONDITIONS = [
    ("baseline",     "Baseline (CE only) β=0 λ=0",          0.0, 0.0),
    ("trades_b1",    "TRADES-KL only β=1",                  1.0, 0.0),
    ("trades_b3",    "TRADES-KL only β=3",                  3.0, 0.0),
    ("grad_l001",    "Input-grad only λ=0.01",              0.0, 0.01),
    ("grad_l01",     "Input-grad only λ=0.1",               0.0, 0.1),
    ("stack_b1_l001", "Stack β=1 λ=0.01",                   1.0, 0.01),
    ("stack_b1_l01",  "Stack β=1 λ=0.1",                    1.0, 0.1),
    ("stack_b3_l001", "Stack β=3 λ=0.01",                   3.0, 0.01),
    ("stack_b3_l01",  "Stack β=3 λ=0.1",                    3.0, 0.1),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    C.set_seed(SEED)

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    results = []
    for key, label, beta, lam in CONDITIONS:
        print(f"\n{'='*60}\nCondition: {label}")
        C.set_seed(SEED)
        model = C.build_model(
            "cnn", {"channels": 1, "size": 28, "n_classes": 10}, width=32, seed=SEED
        )
        t0 = time.time()
        model = train_stacked(model, Xtr, Ytr, EPOCHS, beta, lam)
        dt = time.time() - t0
        m = eval_model(model, Xte, Yte)
        m["condition"] = label
        m["beta"] = beta
        m["lam"] = lam
        m["train_time_s"] = dt
        results.append(m)
        print(
            f"  clean_acc={m['clean_acc']:.4f}  fgsm_asr={m['fgsm_asr']:.4f}  "
            f"pgd_asr={m['pgd_asr']:.4f}  margin={m['mean_margin']:.4f}  ({dt:.1f}s)"
        )

    # ---- analysis: stack vs best single component (by PGD ASR) -------------
    by_key = {c[0]: r for (c, r) in zip(CONDITIONS, results)}
    best_single_keys = ["trades_b1", "trades_b3", "grad_l001", "grad_l01"]
    best_single = min((by_key[k] for k in best_single_keys), key=lambda r: r["pgd_asr"])
    stack_keys = ["stack_b1_l001", "stack_b1_l01", "stack_b3_l001", "stack_b3_l01"]
    best_stack = min((by_key[k] for k in stack_keys), key=lambda r: r["pgd_asr"])

    stack_beats = best_stack["pgd_asr"] < best_single["pgd_asr"] - 1e-4
    if stack_beats:
        verdict = "SUPPORTED"
    elif abs(best_stack["pgd_asr"] - best_single["pgd_asr"]) <= 1e-4 \
            or best_stack["pgd_asr"] <= best_single["pgd_asr"] + 0.01:
        verdict = "PARTIAL"
    else:
        verdict = "NOT SUPPORTED"

    # ---- write table -------------------------------------------------------
    lines = [
        "H385 Stacked Input-Gradient Penalty + TRADES-KL",
        "=" * 70,
        f"Dataset: {DS}  N_train={N_TRAIN}  N_eval={N_EVAL}  Epochs={EPOCHS}  Seed={SEED}",
        f"EPS={EPS}  PGD_STEPS={PGD_STEPS}  PGD_ALPHA={PGD_ALPHA}",
        f"Inner TRADES PGD: steps={TRADES_PGD_STEPS}  eps={TRADES_PGD_EPS}",
        "",
        "L = CE + beta*KL(p_clean || p_adv) + lambda*||grad_x L||^2",
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

    lines += [
        "",
        "Analysis:",
        f"- Baseline PGD_ASR = {by_key['baseline']['pgd_asr']:.4f}",
        f"- Best single component: {best_single['condition']}  "
        f"PGD_ASR={best_single['pgd_asr']:.4f}  clean_acc={best_single['clean_acc']:.4f}",
        f"- Best stacked cell:    {best_stack['condition']}  "
        f"PGD_ASR={best_stack['pgd_asr']:.4f}  clean_acc={best_stack['clean_acc']:.4f}",
        f"- Stack improvement vs best single (PGD ASR): "
        f"{(best_single['pgd_asr'] - best_stack['pgd_asr']):+.4f}",
        "- Routes differ: input-grad penalty smooths the loss surface locally (1st-order),",
        "  while TRADES-KL explicitly matches clean/adv output distributions. If they were",
        "  redundant the stack would not improve on the better of the two.",
        "",
        f"VERDICT: {verdict} — the stack "
        + ("beats" if stack_beats else "does not clearly beat")
        + " the best single component on PGD ASR "
        f"(stack {best_stack['pgd_asr']:.4f} vs single {best_single['pgd_asr']:.4f}).",
    ]

    report = "\n".join(lines)
    print("\n" + report)
    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
