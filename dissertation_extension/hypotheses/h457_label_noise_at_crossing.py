"""
H457 - Label-noise x Adversarial-Training crossing on Fashion-MNIST.

Gaps targeted: G3 (loss/training objectives missing - label-noise AT) and
G7 (per-sample structure / learning dynamics - memorisation under AT).

Hypothesis seed (Section 5):
  "h457: Label-noise + AT crossing - symmetric noise rate x AT epsilon - what
   does AT do under 20% / 40% label noise."

Question
--------
Standard cross-entropy with deep nets memorises noisy labels (Zhang 2017).
Does PGD-AT, under symmetric label noise, (a) become *more* robust because
the noise smooths the boundary (Sanyal et al. 2021's "noisy labels are one
source of adversarial vulnerability"), or (b) memorise the noisy labels
even harder via the adversarial loss (Dong et al. 2022 "Label Noise in
Adversarial Training" - argue the adv. label is itself noisy w.r.t. the
distorted true distribution, driving robust overfitting)?

Critique
--------
- The Sanyal et al. (ICLR 2021) story: standard SGD finds simple boundaries
  that pass through label-noisy points - boundaries close to clean data ->
  small adversarial margin. Removing noise alone does not buy robustness;
  AT does. So under additional injected noise, AT should *still* be better
  than CE at robustness, possibly with a widening gap.
- The Dong et al. (NeurIPS 2022) story: AT inherits the clean label for an
  adversarially perturbed input, which is itself distorted relative to the
  true conditional - this is structural label noise. Adding more (symmetric)
  noise on top should *amplify* robust overfitting: high noisy-train acc,
  low robust test acc, larger generalisation gap. This predicts that at
  high noise rates AT's robust-test performance degrades faster than its
  clean train performance suggests.
- Wei et al. (NeurIPS 2021 "Understanding (Generalized) Label Smoothing
  When Learning with Noisy Labels") finds soft targets help under noise -
  AT's KL-style smoothness (via TRADES-like behaviour) is a related lens.
- Memorisation diagnostic: compare TRAIN accuracy on the (originally) clean
  subset vs the (relabelled) noisy subset. A model that memorises noisy
  labels has high accuracy on the noisy subset's *noisy* labels and low
  accuracy on those samples' *true* labels.

Reference prior art (WebSearch):
  Sanyal, Dokania, Kanade & Torr (ICLR 2021) "How Benign is Benign
    Overfitting?" - arXiv:2007.04028.
  Dong, Xu, Yang, et al. (NeurIPS 2022) "Label Noise in Adversarial Training:
    A Novel Perspective to Study Robust Overfitting" - arXiv:2110.03135.
  Wei, Liu, Mei et al. (NeurIPS 2021) "Understanding Generalized Label
    Smoothing When Learning with Noisy Labels".

Design (6 conditions, single seed, standard config)
---------------------------------------------------
  CE  + 0%  noise            (clean baseline)
  CE  + 20% symmetric noise  (CE memorisation control)
  CE  + 40% symmetric noise  (heavier memorisation control)
  PGD-AT + 0%  noise         (clean AT baseline)
  PGD-AT + 20% symmetric noise
  PGD-AT + 40% symmetric noise

Symmetric noise: with prob p, replace label with a uniform draw over the
OTHER 9 classes. Seed fixed so the same training indices are flipped
across the CE and AT runs at each rate -> clean vs AT directly comparable.

Metrics
-------
  clean_test_acc      - test accuracy on clean test labels
  pgd_asr             - PGD-10 attack-success rate at eps=0.1 on test set
  train_acc_clean_subset_true  - train acc on (originally clean) subset,
                                 measured against the TRUE label
  train_acc_noisy_subset_noisy - train acc on (label-flipped) subset,
                                 measured against the NOISY label that the
                                 model was trained on (memorisation signal)
  train_acc_noisy_subset_true  - train acc on flipped subset, against the
                                 TRUE label (generalisation through noise)
  mem_gap = train_acc_noisy_subset_noisy - train_acc_noisy_subset_true

A large mem_gap = model is memorising the flips. Sanyal-style smoothing
should show *lower* mem_gap under AT than under CE; Dong-style amplified
noise should show *higher* mem_gap.

Config
------
N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128,
SGD(mom=0.9, wd=5e-4), SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.
SmallCNN, width=32. ASCII output, periodic flush.
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
NOISE_RATES = [0.0, 0.20, 0.40]
META = {"channels": 1, "size": 28, "n_classes": 10}

CONDITIONS = []
for p in NOISE_RATES:
    for adv in (False, True):
        tag = ("PGD-AT" if adv else "CE   ") + f" + {int(p*100):>2d}% noise"
        CONDITIONS.append({"adv": adv, "noise": p, "label": tag})

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h457_label_noise_at_crossing_output.txt",
)


def inject_symmetric_noise(Y, rate, ncls, seed):
    """Symmetric label noise: with prob `rate`, replace label with uniform
    over the OTHER classes. Returns (Y_noisy, flip_mask numpy bool)."""
    if rate <= 0.0:
        return Y.clone(), np.zeros(Y.size(0), dtype=bool)
    g = np.random.RandomState(seed)
    n = Y.size(0)
    y_np = Y.detach().cpu().numpy().copy()
    flip = g.rand(n) < rate
    for i in np.where(flip)[0]:
        choices = [c for c in range(ncls) if c != y_np[i]]
        y_np[i] = int(g.choice(choices))
    return torch.as_tensor(y_np, dtype=Y.dtype, device=Y.device), flip


def make_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)


def train(Xtr, Ytr_train, adv, seed):
    """Train CNN from scratch. Xtr float in [0,1] on DEVICE; Ytr_train holds
    the (possibly noisy) labels the model is trained against."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = make_optimizer(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr_train[idx]
            if adv:
                xb = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_clean_and_pgd(model, X, Y):
    """clean test acc + PGD ASR (fraction of originally-correct flipped)."""
    _, clean_acc = C.logits_and_acc(model, X, Y)
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return float(clean_acc), float(pg["asr"])


def train_subset_accs(model, X, Y_true, Y_noisy, flip_mask):
    """Return dict of train-set diagnostic accuracies.
    All accuracies computed in eval mode."""
    # full train preds
    _, full_clean_acc_true = C.logits_and_acc(model, X, Y_true)
    _, full_train_acc_noisy = C.logits_and_acc(model, X, Y_noisy)
    flip_t = torch.as_tensor(flip_mask, device=X.device)
    keep_t = ~flip_t
    out = {
        "train_acc_full_true":  float(full_clean_acc_true),
        "train_acc_full_noisy": float(full_train_acc_noisy),
    }
    if keep_t.any():
        _, a_clean_true = C.logits_and_acc(model, X[keep_t], Y_true[keep_t])
        out["train_acc_clean_subset_true"] = float(a_clean_true)
    else:
        out["train_acc_clean_subset_true"] = float("nan")
    if flip_t.any():
        _, a_noisy_noisy = C.logits_and_acc(model, X[flip_t], Y_noisy[flip_t])
        _, a_noisy_true = C.logits_and_acc(model, X[flip_t], Y_true[flip_t])
        out["train_acc_noisy_subset_noisy"] = float(a_noisy_noisy)
        out["train_acc_noisy_subset_true"] = float(a_noisy_true)
        out["mem_gap"] = float(a_noisy_noisy - a_noisy_true)
    else:
        out["train_acc_noisy_subset_noisy"] = float("nan")
        out["train_acc_noisy_subset_true"] = float("nan")
        out["mem_gap"] = float("nan")
    return out


def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H457  Label-noise x Adversarial-Training crossing  (Fashion-MNIST)")
    out("=" * 80)
    out("Gaps: G3 (loss/objective: label-noise AT)  G7 (per-sample dynamics)")
    out("Refs: Sanyal 2021 (How Benign is Benign Overfitting?, ICLR);")
    out("      Dong 2022 (Label Noise in Adversarial Training, NeurIPS);")
    out("      Wei 2021 (Understanding Generalized Label Smoothing, NeurIPS).")
    out("")
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        noise rates = {NOISE_RATES}  (symmetric, over OTHER classes)")
    out(f"        device = {C.DEVICE}")
    out("")

    # ---- data ----
    Xtr, Ytr_true, Xte, Yte = C.load_dataset(
        DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED
    )
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")
    flush_file()

    # Pre-compute noisy label vectors per rate so that CE and AT
    # at the same rate see the SAME flipped samples.
    noisy_cache = {}
    for p in NOISE_RATES:
        Y_noisy, flip_mask = inject_symmetric_noise(
            Ytr_true, p, META["n_classes"], seed=SEED * 7919 + int(p * 1000)
        )
        noisy_cache[p] = (Y_noisy, flip_mask)
        n_flipped = int(flip_mask.sum())
        out(f"  rate={p:.2f}: flipped {n_flipped}/{N_TRAIN} train labels "
            f"(actual rate={n_flipped/N_TRAIN:.3f})")
    out("")
    flush_file()

    # ---- run conditions ----
    rows = []
    for ci, cond in enumerate(CONDITIONS):
        p = cond["noise"]
        adv = cond["adv"]
        label = cond["label"]
        Y_noisy, flip_mask = noisy_cache[p]
        out("-" * 80)
        out(f"[{ci+1}/{len(CONDITIONS)}] {label}")
        out("-" * 80)
        ts = time.time()
        model = train(Xtr, Y_noisy, adv=adv, seed=SEED)
        train_time = time.time() - ts
        clean_acc, pgd_asr = eval_clean_and_pgd(model, Xte, Yte)
        diag = train_subset_accs(model, Xtr, Ytr_true, Y_noisy, flip_mask)
        row = {
            "label": label,
            "adv": adv,
            "noise": p,
            "clean_test_acc": clean_acc,
            "pgd_asr": pgd_asr,
            "train_time_s": train_time,
        }
        row.update(diag)
        rows.append(row)
        out(f"  clean_test_acc            = {clean_acc:.4f}")
        out(f"  pgd_asr (eps=0.1, 10-step)= {pgd_asr:.4f}")
        out(f"  train_acc_full_true       = {row['train_acc_full_true']:.4f}")
        out(f"  train_acc_full_noisy      = {row['train_acc_full_noisy']:.4f}")
        out(f"  train_acc_clean_subset_true   = "
            f"{row['train_acc_clean_subset_true']:.4f}")
        out(f"  train_acc_noisy_subset_noisy  = "
            f"{row['train_acc_noisy_subset_noisy']:.4f}  "
            f"(memorisation: high => mem)")
        out(f"  train_acc_noisy_subset_true   = "
            f"{row['train_acc_noisy_subset_true']:.4f}  "
            f"(generalisation through noise)")
        out(f"  mem_gap (noisy.noisy - noisy.true) = {row['mem_gap']:.4f}")
        out(f"  train_time = {train_time:.1f}s  (elapsed = {time.time()-t0:.0f}s)")
        out("")
        flush_file()

    # ---- main tables ----
    out("=" * 80)
    out("[A] ROBUSTNESS TABLE")
    out("=" * 80)
    hdr = "{:<26} {:>10} {:>10} {:>10}".format(
        "condition", "clean_acc", "pgd_asr", "time_s")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<26} {:>10.4f} {:>10.4f} {:>10.1f}".format(
            r["label"], r["clean_test_acc"], r["pgd_asr"], r["train_time_s"]))
    out("-" * len(hdr))
    out("")

    out("=" * 80)
    out("[B] MEMORISATION TABLE  (train-set diagnostics)")
    out("=" * 80)
    hdr2 = "{:<26} {:>10} {:>10} {:>10} {:>9}".format(
        "condition", "clean.true", "noisy.noisy", "noisy.true", "mem_gap")
    out(hdr2)
    out("-" * len(hdr2))
    for r in rows:
        out("{:<26} {:>10.4f} {:>10.4f} {:>10.4f} {:>9.4f}".format(
            r["label"],
            r["train_acc_clean_subset_true"],
            r["train_acc_noisy_subset_noisy"],
            r["train_acc_noisy_subset_true"],
            r["mem_gap"]))
    out("-" * len(hdr2))
    out("  Legend: clean.true = train acc on un-flipped samples vs TRUE label;")
    out("          noisy.noisy = train acc on FLIPPED samples vs the NOISY")
    out("                        label the model was trained on (memorisation);")
    out("          noisy.true  = train acc on FLIPPED samples vs TRUE label")
    out("                        (generalisation despite noise);")
    out("          mem_gap     = noisy.noisy - noisy.true. Large => memorised.")
    out("")
    flush_file()

    # ---- VERDICT ----
    out("=" * 80)
    out("[C] VERDICT")
    out("=" * 80)

    def find(adv, noise):
        for r in rows:
            if r["adv"] == adv and abs(r["noise"] - noise) < 1e-9:
                return r
        return None

    ce0 = find(False, 0.0)
    ce2 = find(False, 0.20)
    ce4 = find(False, 0.40)
    at0 = find(True,  0.0)
    at2 = find(True,  0.20)
    at4 = find(True,  0.40)

    # ASR gaps: smaller gap under AT-with-noise vs CE-with-noise = AT helps under noise.
    def fmt_pair(rce, rat, tag):
        out(f"  {tag}:")
        out(f"    CE     clean={rce['clean_test_acc']:.4f}  "
            f"pgd_asr={rce['pgd_asr']:.4f}  mem_gap={rce['mem_gap']:.4f}")
        out(f"    PGD-AT clean={rat['clean_test_acc']:.4f}  "
            f"pgd_asr={rat['pgd_asr']:.4f}  mem_gap={rat['mem_gap']:.4f}")
        out(f"    AT-vs-CE: d_clean={rat['clean_test_acc']-rce['clean_test_acc']:+.4f}  "
            f"d_pgd_asr={rat['pgd_asr']-rce['pgd_asr']:+.4f}  "
            f"d_mem_gap={rat['mem_gap']-rce['mem_gap']:+.4f}")

    fmt_pair(ce0, at0, "0%  noise")
    fmt_pair(ce2, at2, "20% noise")
    fmt_pair(ce4, at4, "40% noise")
    out("")

    # robust-degradation slopes (PGD ASR at 40% minus at 0%)
    ce_deg = ce4["pgd_asr"] - ce0["pgd_asr"]
    at_deg = at4["pgd_asr"] - at0["pgd_asr"]
    ce_acc_deg = ce4["clean_test_acc"] - ce0["clean_test_acc"]
    at_acc_deg = at4["clean_test_acc"] - at0["clean_test_acc"]
    ce_mem_deg = ce4["mem_gap"] - ce0["mem_gap"]
    at_mem_deg = at4["mem_gap"] - at0["mem_gap"]

    out(f"  PGD-ASR degradation 0%->40%:   CE {ce_deg:+.4f}   AT {at_deg:+.4f}")
    out(f"  clean-acc degradation 0%->40%: CE {ce_acc_deg:+.4f}   AT {at_acc_deg:+.4f}")
    out(f"  mem_gap   degradation 0%->40%: CE {ce_mem_deg:+.4f}   AT {at_mem_deg:+.4f}")
    out("")

    # one-line verdict logic
    at_helps_under_noise = (
        (at4["pgd_asr"] < ce4["pgd_asr"] - 0.05)
        and (at4["clean_test_acc"] > ce4["clean_test_acc"] - 0.02)
    )
    at_memorises_more = at_mem_deg > ce_mem_deg + 0.05
    at_acc_collapses = at_acc_deg < -0.10

    if at_helps_under_noise and not at_memorises_more:
        verdict = ("SANYAL-LIKE: PGD-AT remains the dominant lever for robustness "
                   "under label noise; AT does NOT memorise the noisy labels more "
                   "than CE on Fashion-MNIST at this scale.")
    elif at_helps_under_noise and at_memorises_more:
        verdict = ("MIXED: PGD-AT still beats CE on PGD-ASR under noise, but it "
                   "memorises the noisy labels harder (mem_gap grows faster) - "
                   "consistent with Dong 2022's structural-label-noise view.")
    elif (not at_helps_under_noise) and at_acc_collapses:
        verdict = ("AT COLLAPSES UNDER NOISE: clean accuracy and robustness both "
                   "degrade so heavily that AT is no longer the right lever at "
                   "40% symmetric noise.")
    else:
        verdict = ("NULL: at this scale (N=6000, 10 epochs) the AT-vs-CE gap is "
                   "stable across noise rates; no strong evidence either way.")
    out("  ONE-LINE VERDICT: " + verdict)

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
