"""
H403 - Single-sample noise hardening: can replicating ONE input as many noisy
copies push its nearest adversarial further away?

Focused SINGLE-SAMPLE study. Take ONE clean input x0. Its nearest adversarial
lives at some L-inf distance (min-eps-to-flip, found by binary search over PGD).
We then augment the TRAINING set with K noisy copies of THAT SINGLE image
(uniform L-inf noise at the SAME magnitude eps=0.1 as the adversarial budget,
all labelled with x0's true class y0), retrain from scratch, and re-measure the
nearest adversarial. Question: does min-eps-to-flip GROW with K (attacker forced
to spend a larger perturbation), and at what K does it meaningfully move?

We run this per-target over the first 5 correctly-classified test images and
also report the aggregate (mean) curve.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD mom=0.9 wd=5e-4,
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01. CNN width=32.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config --------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
BSEARCH_STEPS = 20          # PGD steps used inside the binary search
BSEARCH_ITERS = 8           # binary-search iterations over [0, 0.3]
EPS_HI = 0.3                # upper bound of the min-eps search
N_TARGETS = 5
K_VALUES = [0, 1, 10, 100, 1000, 5000]

META = {"channels": 1, "size": 28, "n_classes": 10}


def _make_optimizer_sgd(model, lr):
    """SGD mom=0.9 wd=5e-4 as the config requires (common.make_optimizer uses
    wd=1e-4, so we build the optimizer explicitly here)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def train_std(Xtr, Ytr, seed):
    """Train a fresh CNN from scratch, standard SGD, EPOCHS epochs."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def _pred(model, x):
    return int(model(x).argmax(1).item())


def flips_at_eps(model, x0, y0, eps, steps=BSEARCH_STEPS):
    """Run PGD at budget eps on the single image; return True if it flips."""
    if eps <= 0:
        return False
    y = torch.tensor([y0], device=x0.device)
    xa = C.pgd(model, x0, y, eps=eps, steps=steps, alpha=None, random_start=True)
    return _pred(model, xa) != y0


def min_eps_to_flip(model, x0, y0, lo=0.0, hi=EPS_HI, iters=BSEARCH_ITERS):
    """Binary-search smallest eps in [lo,hi] such that PGD flips x0.
    Returns the eps (upper end of the bracket). If even hi does not flip,
    returns hi (capped / "not found within budget")."""
    if not flips_at_eps(model, x0, y0, hi):
        return hi, False  # never flips within budget
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if flips_at_eps(model, x0, y0, mid):
            hi = mid
        else:
            lo = mid
    return hi, True


def make_noisy_copies(x0, y0, k, eps, seed):
    """K noisy copies of x0: x0 + eps*(2*rand-1), clamped to [0,1], fresh noise
    per copy, all labelled y0."""
    if k == 0:
        return None, None
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = x0.detach().cpu().repeat(k, 1, 1, 1)
    noise = (2.0 * torch.rand(base.shape, generator=g) - 1.0) * eps
    Xc = (base + noise).clamp(0, 1).to(x0.device)
    Yc = torch.full((k,), y0, dtype=torch.long, device=x0.device)
    return Xc, Yc


def main():
    t0 = time.time()
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 78)
    out("H403  Single-sample noise hardening (Fashion-MNIST)")
    out("=" * 78)
    out(f"config: N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} LR={LR} BATCH={BATCH} "
        f"SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        binary-search: [0,{EPS_HI}] x {BSEARCH_ITERS} iters, "
        f"PGD steps={BSEARCH_STEPS}")
    out(f"        K sweep = {K_VALUES}   targets = first {N_TARGETS} "
        f"correctly-classified test images")
    out(f"        device = {C.DEVICE}")
    out("")

    # ---- data & base model ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    out("\n[1] training BASE model (K=0, standard, no augmentation)...")
    base = train_std(Xtr, Ytr, SEED)
    _, base_acc = C.logits_and_acc(base, Xte, Yte)
    out(f"    base clean test acc = {base_acc:.4f}")

    # ---- pick targets: first N correctly-classified test images ----
    out("\n[2] selecting targets (first correctly-classified test images)...")
    targets = []  # list of dicts: idx, x0 (1,1,28,28), y0
    with torch.no_grad():
        preds = base(Xte).argmax(1)
    for i in range(Xte.size(0)):
        if len(targets) >= N_TARGETS:
            break
        if int(preds[i].item()) == int(Yte[i].item()):
            x0 = Xte[i:i + 1].clone()
            y0 = int(Yte[i].item())
            targets.append({"idx": i, "x0": x0, "y0": y0})
    out(f"    selected {len(targets)} targets:")
    for t in targets:
        out(f"      target idx={t['idx']:4d}  true_label={t['y0']}")

    # ---- base-model adversarial characterisation per target ----
    out("\n[3] BASE model adversarial characterisation per target")
    out("    (PGD eps=0.1 steps=10 adv distance, + min-eps-to-flip binary search)")
    out("    " + "-" * 70)
    out("    {:>5} {:>5} | {:>9} {:>9} {:>7} | {:>14}".format(
        "tgt", "y0", "L2_dist", "Linf", "flip?", "base_min_eps"))
    out("    " + "-" * 70)
    for t in targets:
        x0, y0 = t["x0"], t["y0"]
        y = torch.tensor([y0], device=x0.device)
        xadv = C.pgd(base, x0, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA,
                     random_start=True)
        diff = (xadv - x0).flatten()
        l2 = float(diff.norm(p=2).item())
        linf = float(diff.abs().max().item())
        flipped = _pred(base, xadv) != y0
        meps, found = min_eps_to_flip(base, x0, y0)
        t["xadv_base"] = xadv.detach()        # store the base-model adversarial
        t["base_min_eps"] = meps
        t["base_min_eps_found"] = found
        cap = "" if found else " (cap)"
        out("    {:>5} {:>5} | {:>9.4f} {:>9.4f} {:>7} | {:>10.4f}{}".format(
            t["idx"], y0, l2, linf, str(bool(flipped)), meps, cap))
    out("    " + "-" * 70)
    out("    note: 'base_min_eps' = L-inf distance to nearest adversarial "
        "(robustness measure).")

    # ---- K sweep ----
    out("\n[4] K sweep: augment training with K noisy copies of EACH target, "
        "retrain, re-measure")
    out("    (one retraining per (K, target); K=0 reuses the base model)")

    # results[target_idx][K] = dict(min_eps, found, margin, orig_adv_flips, clean_acc)
    results = {t["idx"]: {} for t in targets}

    n_train_total = len(targets) * (len(K_VALUES))
    done = 0
    for K in K_VALUES:
        for t in targets:
            x0, y0, idx = t["x0"], t["y0"], t["idx"]
            done += 1
            if K == 0:
                model = base
                clean_acc = base_acc
            else:
                # per (K, target) noise seed so copies differ but are reproducible
                noise_seed = SEED * 1_000_000 + idx * 1000 + K
                Xc, Yc = make_noisy_copies(x0, y0, K, EPS, noise_seed)
                Xaug = torch.cat([Xtr, Xc], dim=0)
                Yaug = torch.cat([Ytr, Yc], dim=0)
                model = train_std(Xaug, Yaug, SEED)
                _, clean_acc = C.logits_and_acc(model, Xte, Yte)

            meps, found = min_eps_to_flip(model, x0, y0)
            marg = float(C.margin(model, x0, torch.tensor([y0]))[0])
            # does the ORIGINAL base-model adversarial still flip on this model?
            orig_adv_flips = _pred(model, t["xadv_base"]) != y0

            results[idx][K] = {
                "min_eps": meps,
                "found": found,
                "margin": marg,
                "orig_adv_flips": bool(orig_adv_flips),
                "clean_acc": clean_acc,
            }
            out(f"    [{done:2d}/{n_train_total}] K={K:>5} target={idx:<4d} "
                f"min_eps={meps:.4f}{'' if found else '(cap)'} "
                f"margin={marg:+.3f} orig_adv_flips={orig_adv_flips} "
                f"clean_acc={clean_acc:.4f}")

    # ---- per-target tables ----
    out("\n[5] PER-TARGET RESULTS")
    for t in targets:
        idx, y0 = t["idx"], t["y0"]
        out("")
        out(f"  target idx={idx}  true_label={y0}  "
            f"base_min_eps={t['base_min_eps']:.4f}")
        out("  " + "-" * 74)
        out("  {:>6} | {:>13} | {:>11} | {:>17} | {:>13}".format(
            "K", "min_eps_flip", "margin@x0", "orig_adv_flips", "clean_test_acc"))
        out("  " + "-" * 74)
        for K in K_VALUES:
            r = results[idx][K]
            cap = "(cap)" if not r["found"] else ""
            out("  {:>6} | {:>10.4f}{:<3} | {:>+11.3f} | {:>17} | {:>13.4f}".format(
                K, r["min_eps"], cap, r["margin"],
                str(r["orig_adv_flips"]), r["clean_acc"]))
        out("  " + "-" * 74)
        # per-target delta
        e0 = results[idx][0]["min_eps"]
        eK = results[idx][K_VALUES[-1]]["min_eps"]
        out(f"  delta min_eps (K={K_VALUES[-1]} vs K=0): "
            f"{eK - e0:+.4f}  ({'pushed FURTHER' if eK > e0 + 1e-4 else 'no meaningful change' if abs(eK - e0) <= 1e-4 else 'pulled CLOSER'})")

    # ---- aggregate (mean) curve ----
    out("\n[6] AGGREGATE (mean over targets) curve")
    out("  " + "-" * 70)
    out("  {:>6} | {:>16} | {:>13} | {:>16} | {:>10}".format(
        "K", "mean_min_eps", "mean_margin", "frac_orig_flips", "mean_acc"))
    out("  " + "-" * 70)
    mean_curve = {}
    for K in K_VALUES:
        eps_vals = [results[t["idx"]][K]["min_eps"] for t in targets]
        marg_vals = [results[t["idx"]][K]["margin"] for t in targets]
        flip_vals = [results[t["idx"]][K]["orig_adv_flips"] for t in targets]
        acc_vals = [results[t["idx"]][K]["clean_acc"] for t in targets]
        me = float(np.mean(eps_vals))
        mean_curve[K] = me
        out("  {:>6} | {:>16.4f} | {:>+13.3f} | {:>16.2f} | {:>10.4f}".format(
            K, me, float(np.mean(marg_vals)),
            float(np.mean(flip_vals)), float(np.mean(acc_vals))))
    out("  " + "-" * 70)

    # ---- headline ----
    out("\n[7] HEADLINE")
    base_mean = mean_curve[0]
    out(f"  mean base_min_eps (K=0) = {base_mean:.4f}")
    # find first K where mean min_eps moves meaningfully (>= 5% relative or +0.005 abs)
    move_thresh = max(0.005, 0.05 * base_mean)
    first_move = None
    for K in K_VALUES[1:]:
        if mean_curve[K] - base_mean >= move_thresh:
            first_move = K
            break
    out(f"  meaningful-move threshold (mean min_eps) = +{move_thresh:.4f}")
    if first_move is not None:
        out(f"  -> mean nearest-adversarial distance first moves FURTHER at K={first_move} "
            f"(mean_min_eps {base_mean:.4f} -> {mean_curve[first_move]:.4f})")
    else:
        # report final direction even if below threshold
        final = mean_curve[K_VALUES[-1]]
        direction = "increased slightly" if final > base_mean else ("decreased" if final < base_mean else "unchanged")
        out(f"  -> NO meaningful move within K<= {K_VALUES[-1]}; mean min_eps "
            f"{base_mean:.4f} -> {final:.4f} ({direction})")

    # one-line answer
    out("")
    if first_move is not None:
        ans = (f"It takes ~K={first_move} noisy copies of the single image before the "
               f"nearest adversarial is pushed meaningfully further away "
               f"(mean min-eps {base_mean:.4f} -> {mean_curve[first_move]:.4f}; "
               f"at K={K_VALUES[-1]} -> {mean_curve[K_VALUES[-1]]:.4f}).")
    else:
        ans = (f"Even at K={K_VALUES[-1]} noisy copies the nearest adversarial is NOT "
               f"pushed meaningfully further away (mean min-eps {base_mean:.4f} -> "
               f"{mean_curve[K_VALUES[-1]]:.4f}).")
    out("  ONE-LINE ANSWER: " + ans)

    out("")
    out(f"done in {time.time() - t0:.1f}s")

    # ---- save report ----
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h403_single_sample_noise_hardening_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
