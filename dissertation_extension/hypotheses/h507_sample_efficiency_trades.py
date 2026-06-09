"""
H507 - Sample efficiency of TRADES vs PGD-AT (Fashion-MNIST SmallCNN).

Seed (paper-§5 / §3 G6 critique): Zhang et al. "Theoretically Principled Trade-off
between Robustness and Accuracy" (TRADES, ICML 2019) decomposes the robust risk
into a natural-error term and a KL boundary term and gives generalization bounds
on the robust risk that, under mild assumptions, are *tighter* than the
straight empirical-risk bound for vanilla PGD adversarial training (Madry 2018).
Schmidt et al. "Adversarially Robust Generalization Requires More Data" (NeurIPS
2018) prove that robust generalization has a strictly higher sample complexity
than standard generalization, so any method that better controls the
boundary/KL term should pay off most when training data is SCARCE and the gap
should shrink as N grows. Carmon et al. "Unlabeled Data Improves Adversarial
Robustness" (NeurIPS 2019) operationalise the same intuition (data is the
bottleneck for AT), and Najafi et al. "Robustness to Adversarial Perturbations
in Learning from Incomplete Data" (NeurIPS 2019) give a distributionally-robust
bound that shrinks with N at a rate that depends on the divergence term TRADES
explicitly minimises.

Critique seed: the original TRADES paper does NOT study sample efficiency
empirically; its experiments are at full CIFAR-10 / MNIST size. The hypothesis
- that TRADES dominates PGD-AT in the low-data regime and the gap closes as N
grows - is a NOVEL claim consistent with, but not made by, Zhang 2019.

Hypothesis (pre-registered):
  H1: at N=2000, TRADES (beta=6) PGD robust-accuracy beats PGD-AT by >= 3 pp.
  H2: at N >= 6000, the robust-accuracy gap |TRADES - PGD-AT| < 2 pp (closure).

Controls (paper-§5):
  (1) Train PGD-AT and TRADES (beta=6) at N in {1000, 2000, 4000, 6000, 12000}.
  (2) At each N, both clean-acc and PGD attack-success / robust-acc.
  (3) Sample-efficiency curve (robust-acc vs N for each method).
  (4) Gap-closure point: smallest N at which |gap| < 2 pp.
  (5) Per-class robust-acc to check the gain is broad, not driven by 1-2 classes.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
N_GRID = [1000, 2000, 4000, 6000, 12000]
EPS = 0.1
PGD_STEPS_TRAIN = 7
PGD_STEPS_EVAL = 20
TRADES_BETA = 6.0
EPOCHS = 10
BATCH = 128
LR = 0.05
N_EVAL = 2000

GAP_LOW_N = 2000      # H1 reference point
GAP_LOW_TARGET = 0.03 # TRADES - PGD-AT >= 3 pp at N=2000
GAP_HIGH_N = 6000     # H2 reference point
GAP_HIGH_TARGET = 0.02 # |TRADES - PGD-AT| < 2 pp at N>=6000


# ---------------------------------------------------------------------------
# TRADES training (KL-divergence boundary loss; Zhang et al. 2019)
# ---------------------------------------------------------------------------
def _trades_inner(model, x_nat, eps, steps, alpha):
    """Generate TRADES-style adversarial x' maximising KL(p(x_nat) || p(x'))."""
    x_adv = x_nat.detach() + 0.001 * torch.randn_like(x_nat)
    x_adv = x_adv.clamp(0, 1)
    with torch.no_grad():
        p_nat = F.softmax(model(x_nat), dim=1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logp_adv = F.log_softmax(model(x_adv), dim=1)
        loss_kl = F.kl_div(logp_adv, p_nat, reduction="batchmean")
        g, = torch.autograd.grad(loss_kl, x_adv)
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x_nat - eps), x_nat + eps).clamp(0, 1)
    return x_adv.detach()


def train_trades(model, Xtr, Ytr, epochs, batch, lr, eps, steps, beta):
    """TRADES loss: CE(clean) + beta * KL(clean || adv)."""
    alpha = 2.5 * eps / steps
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            x_adv = _trades_inner(model, xb, eps=eps, steps=steps, alpha=alpha)
            model.train()
            opt.zero_grad()
            logits_nat = model(xb)
            logits_adv = model(x_adv)
            loss_nat = F.cross_entropy(logits_nat, yb)
            loss_rob = F.kl_div(F.log_softmax(logits_adv, dim=1),
                                F.softmax(logits_nat, dim=1),
                                reduction="batchmean")
            loss = loss_nat + beta * loss_rob
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def robust_acc_and_per_class(model, X, Y, eps, steps, ncls, batch=256):
    """Returns (clean_acc, robust_acc, per_class_robust_acc[ncls])."""
    model.eval()
    n = X.size(0)
    clean_correct = torch.zeros(n, dtype=torch.bool)
    rob_correct = torch.zeros(n, dtype=torch.bool)
    for i in range(0, n, batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        with torch.no_grad():
            cpred = model(xb).argmax(1)
        clean_correct[i:i + batch] = (cpred == yb).cpu()
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            apred = model(xa).argmax(1)
        rob_correct[i:i + batch] = (apred == yb).cpu()
    Ycpu = Y.cpu()
    clean_acc = float(clean_correct.float().mean())
    rob_acc = float(rob_correct.float().mean())
    per_class = []
    for c in range(ncls):
        m = (Ycpu == c)
        per_class.append(float(rob_correct[m].float().mean()) if m.sum() > 0 else float("nan"))
    return clean_acc, rob_acc, per_class


# ---------------------------------------------------------------------------
# one (N, seed, method) cell
# ---------------------------------------------------------------------------
def run_cell(method, n_train, seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=n_train, n_eval=N_EVAL, seed=seed)

    model = C.build_model("cnn", meta, seed=seed)
    if method == "pgd_at":
        C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, opt="sgd", lr=LR,
                      ncls=meta["n_classes"],
                      adv_train=True, adv_eps=EPS, adv_steps=PGD_STEPS_TRAIN)
    elif method == "trades":
        train_trades(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, lr=LR,
                     eps=EPS, steps=PGD_STEPS_TRAIN, beta=TRADES_BETA)
    else:
        raise ValueError(method)

    clean, rob, per_class = robust_acc_and_per_class(
        model, Xte, Yte, eps=EPS, steps=PGD_STEPS_EVAL, ncls=meta["n_classes"])
    return {"method": method, "n_train": n_train, "seed": seed,
            "clean_acc": clean, "robust_acc": rob, "per_class_robust": per_class}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("H507 - Sample efficiency: TRADES vs PGD-AT (F-MNIST SmallCNN)")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  pgd_eval_steps={PGD_STEPS_EVAL}")
    print(f"N grid={N_GRID}  seeds={SEEDS}  TRADES beta={TRADES_BETA}  epochs={EPOCHS}")
    print(f"Refs: Zhang TRADES 2019; Schmidt 2018; Carmon 2019; Najafi 2019")
    print("-" * 74)

    rows = []
    for n in N_GRID:
        for method in ("pgd_at", "trades"):
            for s in SEEDS:
                t0 = time.time()
                r = run_cell(method, n, s)
                r["runtime_s"] = round(time.time() - t0, 1)
                rows.append(r)
                print(f"[N={n:5d} {method:7s} s{s}] clean={r['clean_acc']:.3f}  "
                      f"rob={r['robust_acc']:.3f}  ({r['runtime_s']}s)")

    # aggregate
    def agg(method, n, key):
        vals = [r[key] for r in rows if r["method"] == method and r["n_train"] == n]
        return float(np.mean(vals)) if vals else float("nan")

    def agg_per_class(method, n):
        mats = [r["per_class_robust"] for r in rows if r["method"] == method and r["n_train"] == n]
        if not mats:
            return [float("nan")] * 10
        arr = np.array(mats)  # (seeds, ncls)
        return arr.mean(axis=0).tolist()

    print("\n" + "=" * 74)
    print("SAMPLE-EFFICIENCY CURVE  (mean over seeds)")
    print("-" * 74)
    print(f"{'N_train':>8s} | {'PGD-AT clean':>13s} {'PGD-AT rob':>11s} | "
          f"{'TRADES clean':>13s} {'TRADES rob':>11s} | {'gap (T - P)':>12s}")
    gap_table = {}
    for n in N_GRID:
        pc = agg("pgd_at", n, "clean_acc")
        pr = agg("pgd_at", n, "robust_acc")
        tc = agg("trades", n, "clean_acc")
        tr = agg("trades", n, "robust_acc")
        gap = tr - pr
        gap_table[n] = gap
        print(f"{n:>8d} | {pc:>13.3f} {pr:>11.3f} | {tc:>13.3f} {tr:>11.3f} | {gap:>+12.3f}")

    # gap-closure point
    closure_n = None
    for n in N_GRID:
        if abs(gap_table[n]) < GAP_HIGH_TARGET:
            closure_n = n
            break

    print("\n" + "=" * 74)
    print("PER-CLASS ROBUST ACC  (mean over seeds; F-MNIST classes 0..9)")
    print("-" * 74)
    cls_names = ["T-sh", "Trou", "Pull", "Drs ", "Coat", "Sand", "Shrt", "Snk ", "Bag ", "Boot"]
    print("           " + " ".join(f"{c:>5s}" for c in cls_names))
    for n in N_GRID:
        pgd_pc = agg_per_class("pgd_at", n)
        tr_pc = agg_per_class("trades", n)
        print(f"N={n:<5d} P:" + " ".join(f"{v:5.2f}" for v in pgd_pc))
        print(f"        T:" + " ".join(f"{v:5.2f}" for v in tr_pc))

    # verdicts
    gap_low = gap_table.get(GAP_LOW_N, float("nan"))
    h1_pass = gap_low >= GAP_LOW_TARGET
    high_ns = [n for n in N_GRID if n >= GAP_HIGH_N]
    h2_pass = all(abs(gap_table[n]) < GAP_HIGH_TARGET for n in high_ns) if high_ns else False

    print("\n" + "=" * 74)
    print("HEADLINE VERDICT")
    print("-" * 74)
    print(f"  H1 (TRADES - PGD-AT >= +{GAP_LOW_TARGET:.2f} at N={GAP_LOW_N}): "
          f"observed gap = {gap_low:+.3f}  ->  {'PASS' if h1_pass else 'FAIL'}")
    print(f"  H2 (|gap| < {GAP_HIGH_TARGET:.2f} for all N >= {GAP_HIGH_N}): "
          f"gaps = " + ", ".join(f"N={n}:{gap_table[n]:+.3f}" for n in high_ns)
          + f"  ->  {'PASS' if h2_pass else 'FAIL'}")
    print(f"  Gap-closure N (smallest N with |gap|<{GAP_HIGH_TARGET:.2f}): "
          f"{closure_n if closure_n is not None else 'never within grid'}")
    if h1_pass and h2_pass:
        verdict = "CONFIRMED: TRADES is more sample-efficient and the advantage decays with N."
    elif h1_pass and not h2_pass:
        verdict = "PARTIAL: TRADES wins at low N but the gap does not close in our range."
    elif (not h1_pass) and h2_pass:
        verdict = "PARTIAL: methods match at high N but no low-N TRADES advantage observed."
    else:
        verdict = "REJECTED: no clear sample-efficiency advantage of TRADES over PGD-AT."
    print(f"  Overall: {verdict}")
    print("=" * 74)
    print("Interpretation: a confirmed H1+H2 pattern is consistent with TRADES'")
    print("tighter robust-risk bound (Zhang 2019) being most valuable when robust")
    print("sample complexity (Schmidt 2018) bites hardest, i.e. in the small-N")
    print("regime. As N grows, the empirical PGD-AT bound also tightens (Najafi")
    print("2019) and the explicit KL boundary term loses its statistical edge -")
    print("the same data-bottleneck story told by Carmon 2019 from the other side.")
    print("=" * 74)


if __name__ == "__main__":
    main()
