"""
H185 - Margin-adaptive ε adversarial training improves hard-sample robustness.

Standard AT uses a fixed ε=0.1 for all samples. Paper 2403.04070 argues that
per-sample ε proportional to margin should help hard (low-margin) samples most.

We train two models on Fashion-MNIST:
  * standard_AT: fixed ε=0.1 for every sample
  * adaptive_AT: ε_i = clip(0.05 + 0.1*(1 - margin_i/max_margin), 0.01, 0.2)
    where margin_i is computed on the CURRENT model at each training epoch.

Evaluation: 200 test samples split into margin quartiles (Q1=lowest margin).
Per-quartile PGD ASR shows whether adaptive AT helps most for hard samples.

Expected outcome: adaptive AT reduces Q1 ASR more than Q4 ASR vs standard AT,
consistent with the paper's claim that variable ε concentrates perturbation
budget where the decision boundary is closest.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config ----------------------------------------------------------------
SEED = 0
N_TRAIN = 10000        # subset for speed
N_TEST = 200           # evaluation set size
EPOCHS = 10
BATCH = 128
LR = 0.001
EPS_FIXED = 0.1        # standard AT epsilon
EPS_MIN = 0.01
EPS_MAX = 0.2
EPS_BASE = 0.05        # adaptive: eps = clip(EPS_BASE + 0.1*(1 - m/mmax), ...)
AT_STEPS = 7
EVAL_STEPS = 10
META = {"channels": 1, "size": 28, "n_classes": 10}

# ---- adaptive AT training loop ---------------------------------------------

def adaptive_at_train(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, lr=LR, seed=SEED):
    """AT loop where each sample gets its own ε based on current-model margin."""
    C.set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]

            # compute per-sample margin on the current model
            model.eval()
            with torch.no_grad():
                logits = model(xb)
            correct_logit = logits.gather(1, yb.unsqueeze(1)).squeeze(1)
            tmp = logits.clone()
            tmp[torch.arange(tmp.size(0)), yb] = -1e9
            other_logit = tmp.max(1).values
            margin = (correct_logit - other_logit).cpu()

            # eps_i = clip(EPS_BASE + 0.1*(1 - margin_i/max_margin), EPS_MIN, EPS_MAX)
            mmax = margin.max().item()
            if mmax <= 0:
                mmax = 1.0
            eps_per = (EPS_BASE + 0.1 * (1.0 - margin / mmax)).clamp(EPS_MIN, EPS_MAX)

            # build adversarials per-sample using mean eps for the batch PGD call
            # (PGD operates on the whole batch with a single eps; we approximate by
            # using the sample-average eps since C.pgd doesn't support per-sample eps)
            eps_batch = float(eps_per.mean().item())
            model.train()
            xa = _pgd_per_sample(model, xb, yb, eps_per.to(xb.device),
                                  steps=AT_STEPS)

            opt.zero_grad()
            loss = F.cross_entropy(model(xa), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def _pgd_per_sample(model, x, y, eps_vec, steps=7):
    """PGD attack where each sample i has its own epsilon eps_vec[i].
    eps_vec: 1-D tensor of shape (N,) on the same device as x.
    """
    # expand eps to (N,1,1,1) for broadcasting
    eps = eps_vec.view(-1, 1, 1, 1)
    alpha = 2.5 * eps / steps

    x0 = x.clone().detach()
    xa = x0 + (torch.rand_like(x0) * 2 - 1) * eps
    xa = xa.clamp(0, 1)

    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


# ---- standard AT training --------------------------------------------------

def standard_at_train(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, lr=LR, seed=SEED):
    """Standard PGD-AT with fixed eps=EPS_FIXED."""
    C.set_seed(seed)
    C.train_model(model, Xtr, Ytr, epochs=epochs, batch=batch, opt="adam", lr=lr,
                  adv_train=True, adv_eps=EPS_FIXED, adv_steps=AT_STEPS)
    return model


# ---- evaluation ------------------------------------------------------------

def evaluate_per_quartile(model, Xte, Yte, eps=EPS_FIXED, steps=EVAL_STEPS):
    """Returns per-quartile PGD ASR and overall ASR/clean-acc, stratified by
    margin on the CLEAN model predictions."""
    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        preds = logits.argmax(1)

    # clean accuracy
    clean_acc = float((preds == Yte).float().mean())

    # margins on clean data
    Y_cpu = Yte.cpu()
    logits_cpu = logits.cpu()
    correct_l = logits_cpu.gather(1, Y_cpu.unsqueeze(1)).squeeze(1)
    tmp = logits_cpu.clone()
    tmp[torch.arange(tmp.size(0)), Y_cpu] = -1e9
    other_l = tmp.max(1).values
    margins = (correct_l - other_l).numpy()

    # quartile boundaries
    q25, q50, q75 = float(np.percentile(margins, 25)), \
                    float(np.percentile(margins, 50)), \
                    float(np.percentile(margins, 75))

    def quartile_asr(mask_idx):
        """ASR for samples in mask_idx."""
        if len(mask_idx) == 0:
            return float("nan")
        xs = Xte[mask_idx]
        ys = Yte[mask_idx]
        xa = C.pgd(model, xs, ys, eps, steps)
        with torch.no_grad():
            flipped = (model(xa).argmax(1) != ys).float().mean().item()
        return flipped

    q1_idx = np.where(margins <= q25)[0]
    q2_idx = np.where((margins > q25) & (margins <= q50))[0]
    q3_idx = np.where((margins > q50) & (margins <= q75))[0]
    q4_idx = np.where(margins > q75)[0]

    # overall PGD ASR (on all samples)
    xa_all = C.pgd(model, Xte, Yte, eps, steps)
    with torch.no_grad():
        overall_asr = float((model(xa_all).argmax(1) != Yte).float().mean())

    return {
        "clean_acc": clean_acc,
        "overall_asr": overall_asr,
        "q1_asr": quartile_asr(q1_idx),
        "q2_asr": quartile_asr(q2_idx),
        "q3_asr": quartile_asr(q3_idx),
        "q4_asr": quartile_asr(q4_idx),
        "q1_n": len(q1_idx),
        "q2_n": len(q2_idx),
        "q3_n": len(q3_idx),
        "q4_n": len(q4_idx),
        "margin_q25": q25,
        "margin_q50": q50,
        "margin_q75": q75,
    }


# ---- main ------------------------------------------------------------------

def main():
    print("=" * 72)
    print("H185 - Margin-adaptive ε adversarial training vs standard AT")
    print("=" * 72)
    print(f"Device={C.DEVICE}  seed={SEED}  n_train={N_TRAIN}  n_test={N_TEST}")
    print(f"epochs={EPOCHS}  lr={LR}  eps_fixed={EPS_FIXED}")
    print(f"adaptive eps: clip({EPS_BASE} + 0.1*(1 - m/mmax), {EPS_MIN}, {EPS_MAX})")
    print()

    # load data
    t0 = time.time()
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_TEST, seed=SEED)
    print(f"Data loaded: train={Xtr.shape[0]} test={Xte.shape[0]}  ({time.time()-t0:.1f}s)")

    # --- Train standard AT model ---
    print("\n[1/2] Training standard AT (fixed ε=0.1) ...")
    t1 = time.time()
    C.set_seed(SEED)
    model_std = C.build_model("cnn", META, width=32, seed=SEED)
    standard_at_train(model_std, Xtr, Ytr)
    print(f"     done in {time.time()-t1:.1f}s")

    # --- Train adaptive AT model ---
    print("\n[2/2] Training adaptive AT (ε proportional to margin) ...")
    t2 = time.time()
    C.set_seed(SEED)
    model_adp = C.build_model("cnn", META, width=32, seed=SEED)
    adaptive_at_train(model_adp, Xtr, Ytr)
    print(f"     done in {time.time()-t2:.1f}s")

    # --- Evaluate ---
    print("\nEvaluating standard AT ...")
    res_std = evaluate_per_quartile(model_std, Xte, Yte)

    print("Evaluating adaptive AT ...")
    res_adp = evaluate_per_quartile(model_adp, Xte, Yte)

    # --- Report ---
    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"{'Metric':<30} {'Standard AT':>14} {'Adaptive AT':>14} {'Delta':>10}")
    print("-" * 72)

    metrics = [
        ("Clean accuracy",        "clean_acc",   False),
        ("Overall PGD ASR",       "overall_asr", True),
        ("Q1 PGD ASR (low margin)","q1_asr",     True),
        ("Q2 PGD ASR",            "q2_asr",      True),
        ("Q3 PGD ASR",            "q3_asr",      True),
        ("Q4 PGD ASR (high margin)","q4_asr",    True),
    ]
    for label, key, lower_is_better in metrics:
        s = res_std[key]
        a = res_adp[key]
        delta = a - s
        direction = "better" if (lower_is_better and delta < 0) or \
                                (not lower_is_better and delta > 0) else "worse"
        print(f"{label:<30} {s:>14.4f} {a:>14.4f} {delta:>+10.4f}  ({direction})")

    print()
    print(f"Quartile sample counts (n):  Q1={res_std['q1_n']}  Q2={res_std['q2_n']}  "
          f"Q3={res_std['q3_n']}  Q4={res_std['q4_n']}")
    print(f"Standard AT margin quartiles (Q25/Q50/Q75): "
          f"{res_std['margin_q25']:.3f} / {res_std['margin_q50']:.3f} / {res_std['margin_q75']:.3f}")
    print(f"Adaptive AT margin quartiles (Q25/Q50/Q75): "
          f"{res_adp['margin_q25']:.3f} / {res_adp['margin_q50']:.3f} / {res_adp['margin_q75']:.3f}")

    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    q1_improvement = res_std["q1_asr"] - res_adp["q1_asr"]
    q4_improvement = res_std["q4_asr"] - res_adp["q4_asr"]
    print(f"Q1 ASR improvement (std - adp): {q1_improvement:+.4f}")
    print(f"Q4 ASR improvement (std - adp): {q4_improvement:+.4f}")
    if q1_improvement > q4_improvement:
        print("=> Adaptive AT helps MOST for hard (Q1, low-margin) samples — SUPPORTS H185.")
    else:
        print("=> Adaptive AT does NOT preferentially help hard samples — REFUTES H185.")

    overall_improvement = res_std["overall_asr"] - res_adp["overall_asr"]
    print(f"Overall ASR improvement (std - adp): {overall_improvement:+.4f}")
    if overall_improvement > 0.01:
        print("=> Adaptive AT also improves overall PGD robustness.")
    elif overall_improvement < -0.01:
        print("=> Adaptive AT HURTS overall PGD robustness.")
    else:
        print("=> Overall PGD robustness roughly unchanged.")

    print("=" * 72)
    total_time = time.time() - t0
    print(f"Total runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
