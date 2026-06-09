"""
H202 - Curriculum Adversarial Training: easy-to-hard sample scheduling.

Motivated by 2512.22069 (Scaling AT via Data Selection): margin-based selection
at 50% compute matches full AT. We test a 3-phase curriculum:
  Phase 1 (epochs 1-4):  AT only top-50% margin samples (easiest half)
  Phase 2 (epochs 5-7):  AT top-75%
  Phase 3 (epochs 8-10): AT all samples

Compare to uniform AT (all samples in every epoch).

Margin = correct-class logit minus max other-class logit (higher = easier).
Selection targets EASY samples first (high margin), expanding to harder over time.
Samples not selected in a phase are skipped entirely (not trained clean either).

Key metrics:
  - Clean accuracy and PGD-10 ASR on Xte[:500]
  - Per-quartile ASR split by vanilla model margin quartiles on Xte
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS          = "fashion_mnist"
SEEDS       = [0, 1, 2]
EPS         = 0.1
AT_STEPS    = 3       # PGD steps during training (fast)
AT_ALPHA    = 0.033   # step size during training
EVAL_STEPS  = 10      # PGD steps during eval
EVAL_N      = 500     # first 500 test samples
EPOCHS      = 10
BATCH       = 128

# Phase thresholds: fraction of training set to include (by margin rank, easiest first)
PHASES = {
    1: (1,  4,  0.50),   # epochs 1-4: top 50%
    2: (5,  7,  0.75),   # epochs 5-7: top 75%
    3: (8, 10,  1.00),   # epochs 8-10: all
}


def compute_margins(model, X, Y, batch=256):
    """Return margin array (correct logit - max other logit) for each sample."""
    model.eval()
    logits_list = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            logits_list.append(model(X[i:i+batch]).cpu())
    logits = torch.cat(logits_list)
    margins = C.margin_of(logits, Y.cpu())
    return margins  # numpy array, shape (N,)


def get_phase_mask(margins, top_frac):
    """Return boolean mask: True = sample is in the easy top_frac (highest margin)."""
    n = len(margins)
    k = max(1, int(n * top_frac))
    threshold = np.sort(margins)[-k]   # k-th largest value
    return margins >= threshold


def train_uniform_at(meta, Xtr, Ytr, seed):
    """Standard AT: all samples, every epoch, PGD adversarials."""
    C.set_seed(seed)
    model = C.build_model("cnn", meta)
    opt = C.make_optimizer(model, "sgd", 0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xadv = C.pgd(model, xb, yb, eps=EPS, steps=AT_STEPS, alpha=AT_ALPHA)
            opt.zero_grad()
            loss = F.cross_entropy(model(xadv), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_curriculum_at(meta, Xtr, Ytr, seed):
    """Curriculum AT: 3-phase easy-to-hard sample scheduling."""
    C.set_seed(seed)
    model = C.build_model("cnn", meta)
    opt = C.make_optimizer(model, "sgd", 0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)

    # Pre-compute phase boundaries
    # At start of each phase, recompute margins and select subset
    phase_boundaries = sorted(set([1, 5, 8]))  # epoch (1-indexed) where each phase starts
    current_mask = None
    current_top_frac = None

    for ep in range(1, EPOCHS + 1):
        # Check if we're entering a new phase
        for phase_id, (start_ep, end_ep, top_frac) in PHASES.items():
            if ep == start_ep:
                current_top_frac = top_frac
                # Recompute margins with current model
                margins = compute_margins(model, Xtr, Ytr)
                current_mask = get_phase_mask(margins, top_frac)
                break

        # Indices of samples in current phase
        phase_indices = torch.where(torch.from_numpy(current_mask))[0].to(Xtr.device)
        np_phase = phase_indices.size(0)

        model.train()
        perm = torch.randperm(np_phase, device=Xtr.device)
        for i in range(0, np_phase, BATCH):
            idx = phase_indices[perm[i:i+BATCH]]
            xb, yb = Xtr[idx], Ytr[idx]
            xadv = C.pgd(model, xb, yb, eps=EPS, steps=AT_STEPS, alpha=AT_ALPHA)
            opt.zero_grad()
            loss = F.cross_entropy(model(xadv), yb)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    return model


def eval_model(model, Xte, Yte):
    """Clean acc + PGD-10 ASR on Xte[:EVAL_N]."""
    X, Y = Xte[:EVAL_N], Yte[:EVAL_N]
    with torch.no_grad():
        logits = model(X)
    clean_acc = float((logits.argmax(1) == Y).float().mean())
    res = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=EVAL_STEPS)
    return clean_acc, res["asr"]


def compute_quartile_asr(model, Xte, Yte, vanilla_margins):
    """Per-quartile ASR split by vanilla model margins on Xte[:EVAL_N]."""
    X, Y = Xte[:EVAL_N], Yte[:EVAL_N]
    margins = vanilla_margins[:EVAL_N]
    q_boundaries = np.percentile(margins, [25, 50, 75])

    quartile_asr = {}
    for q in range(4):
        if q == 0:
            mask = margins <= q_boundaries[0]
        elif q == 1:
            mask = (margins > q_boundaries[0]) & (margins <= q_boundaries[1])
        elif q == 2:
            mask = (margins > q_boundaries[1]) & (margins <= q_boundaries[2])
        else:
            mask = margins > q_boundaries[2]

        mask_t = torch.from_numpy(mask).to(X.device)
        Xq, Yq = X[mask_t], Y[mask_t]
        if Xq.size(0) == 0:
            quartile_asr[q+1] = float("nan")
            continue
        res = C.attack_success(model, Xq, Yq, attack="pgd", eps=EPS, steps=EVAL_STEPS)
        quartile_asr[q+1] = res["asr"]
    return quartile_asr


def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)

    print(f"\n[seed {seed}] Training vanilla model for quartile boundaries...")
    t0 = time.time()
    # Vanilla model (no AT) for quartile definition
    C.set_seed(seed)
    vanilla = C.build_model("cnn", meta)
    C.train_model(vanilla, Xtr, Ytr, epochs=8, opt="sgd", lr=0.05, ncls=meta["n_classes"])
    vanilla_logits, _ = C.logits_and_acc(vanilla, Xte[:EVAL_N], Yte[:EVAL_N])
    vanilla_margins = C.margin_of(vanilla_logits, Yte[:EVAL_N].cpu())
    print(f"  vanilla done ({time.time()-t0:.1f}s)")

    # --- Uniform AT ---
    print(f"[seed {seed}] Training uniform AT...")
    t1 = time.time()
    unif_model = train_uniform_at(meta, Xtr, Ytr, seed)
    unif_clean, unif_asr = eval_model(unif_model, Xte, Yte)
    unif_qasr = compute_quartile_asr(unif_model, Xte, Yte, vanilla_margins)
    print(f"  uniform AT done ({time.time()-t1:.1f}s): clean={unif_clean:.3f} ASR={unif_asr:.3f}")

    # --- Curriculum AT ---
    print(f"[seed {seed}] Training curriculum AT...")
    t2 = time.time()
    curr_model = train_curriculum_at(meta, Xtr, Ytr, seed)
    curr_clean, curr_asr = eval_model(curr_model, Xte, Yte)
    curr_qasr = compute_quartile_asr(curr_model, Xte, Yte, vanilla_margins)
    print(f"  curriculum AT done ({time.time()-t2:.1f}s): clean={curr_clean:.3f} ASR={curr_asr:.3f}")

    return {
        "seed": seed,
        "unif_clean": unif_clean, "unif_asr": unif_asr,
        "unif_q1": unif_qasr[1], "unif_q2": unif_qasr[2],
        "unif_q3": unif_qasr[3], "unif_q4": unif_qasr[4],
        "curr_clean": curr_clean, "curr_asr": curr_asr,
        "curr_q1": curr_qasr[1], "curr_q2": curr_qasr[2],
        "curr_q3": curr_qasr[3], "curr_q4": curr_qasr[4],
        "delta_q1": curr_qasr[1] - unif_qasr[1] if not (
            curr_qasr[1] != curr_qasr[1] or unif_qasr[1] != unif_qasr[1]) else float("nan"),
        "runtime_s": round(time.time() - t0, 1),
    }


def main():
    print("=" * 74)
    print("H202 - Curriculum Adversarial Training (easy-to-hard scheduling)")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  eval_n={EVAL_N}")
    print(f"AT PGD: steps={AT_STEPS} alpha={AT_ALPHA}  Eval PGD: steps={EVAL_STEPS}")
    print(f"Phases: 1-4 (top 50%), 5-7 (top 75%), 8-10 (all samples)")
    print()

    rows = []
    for s in SEEDS:
        r = run_seed(s)
        rows.append(r)
        print(f"\n  [seed {s}] ({r['runtime_s']}s)")
        print(f"  {'Condition':<20} {'Clean Acc':>10} {'PGD ASR':>10} "
              f"{'Q1 ASR':>10} {'Q2 ASR':>10} {'Q3 ASR':>10} {'Q4 ASR':>10}")
        print(f"  {'-'*80}")
        print(f"  {'Uniform AT':<20} {r['unif_clean']:>10.3f} {r['unif_asr']:>10.3f} "
              f"{r['unif_q1']:>10.3f} {r['unif_q2']:>10.3f} {r['unif_q3']:>10.3f} {r['unif_q4']:>10.3f}")
        print(f"  {'Curriculum AT':<20} {r['curr_clean']:>10.3f} {r['curr_asr']:>10.3f} "
              f"{r['curr_q1']:>10.3f} {r['curr_q2']:>10.3f} {r['curr_q3']:>10.3f} {r['curr_q4']:>10.3f}")
        q1_delta = r["delta_q1"]
        sign = "lower" if q1_delta < 0 else "higher"
        print(f"\n  Q1 (hardest) ASR: curriculum {sign} by {abs(q1_delta):.3f}")

    def m(k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return sum(v) / len(v) if v else float("nan")

    print("\n" + "=" * 74)
    print("MEANS across seeds")
    print(f"  {'Condition':<20} {'Clean Acc':>10} {'PGD ASR':>10} "
          f"{'Q1 ASR':>10} {'Q2 ASR':>10} {'Q3 ASR':>10} {'Q4 ASR':>10}")
    print(f"  {'-'*80}")
    print(f"  {'Uniform AT':<20} {m('unif_clean'):>10.3f} {m('unif_asr'):>10.3f} "
          f"{m('unif_q1'):>10.3f} {m('unif_q2'):>10.3f} {m('unif_q3'):>10.3f} {m('unif_q4'):>10.3f}")
    print(f"  {'Curriculum AT':<20} {m('curr_clean'):>10.3f} {m('curr_asr'):>10.3f} "
          f"{m('curr_q1'):>10.3f} {m('curr_q2'):>10.3f} {m('curr_q3'):>10.3f} {m('curr_q4'):>10.3f}")

    delta_q1 = m("delta_q1")
    delta_clean = m("curr_clean") - m("unif_clean")
    delta_asr = m("curr_asr") - m("unif_asr")

    print("\n" + "=" * 74)
    print("KEY FINDINGS")
    print(f"  Net Q1 (hardest) ASR delta (curriculum - uniform): {delta_q1:+.3f}")
    print(f"  Net clean acc delta (curriculum - uniform):        {delta_clean:+.3f}")
    print(f"  Net overall ASR delta (curriculum - uniform):      {delta_asr:+.3f}")
    print()

    # Answer the hypothesis questions
    q1_better = delta_q1 < -0.02   # curriculum meaningfully reduces Q1 ASR
    overall_comparable = abs(delta_asr) <= 0.05

    if q1_better and overall_comparable:
        verdict = "SUPPORTED: curriculum reduces Q1 (hardest) ASR while overall ASR is comparable."
    elif q1_better and not overall_comparable:
        verdict = ("PARTIALLY SUPPORTED: curriculum reduces Q1 ASR but overall robustness differs."
                   if delta_asr < 0 else
                   "MIXED: curriculum helps Q1 but hurts overall robustness.")
    elif not q1_better and overall_comparable:
        verdict = "NOT SUPPORTED: no meaningful Q1 benefit; curriculum provides no advantage on hard samples."
    else:
        verdict = "NOT SUPPORTED: curriculum is worse on both Q1 and overall metrics."

    print(f"  Hypothesis verdict: {verdict}")
    print()
    print("Interpretation:")
    print("  Q1 = hardest 25% of test samples (lowest vanilla-model margin).")
    print("  A supported hypothesis means curriculum AT, by warming up on easy samples,")
    print("  builds robustness on hard samples more effectively than uniform AT,")
    print("  consistent with the margin-based selection findings of 2512.22069.")
    print("=" * 74)


if __name__ == "__main__":
    main()
