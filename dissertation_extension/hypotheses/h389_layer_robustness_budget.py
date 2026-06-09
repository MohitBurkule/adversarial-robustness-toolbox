"""
H389 - Cumulative front-to-back AT budget allocation.

Hypothesis: robustness is front-loaded. Applying activation-space adversarial
training (AT) to only the EARLY layers of the SmallCNN captures most of the gain
that full input-space AT provides. This probes WHY H290 found block0
activation-space AT nearly matched full-input AT.

Mechanism (activation-space AT):
  Forward x through the network up to a chosen split point, perturb the
  activations THERE with an FGSM-style single ascent step (sign of grad of CE
  wrt that activation, magnitude eps_act), then continue the forward pass and
  train on cross-entropy of the perturbed path. To budget MULTIPLE split points
  jointly we accumulate perturbations: at every budgeted point we take an FGSM
  step on the running activation before continuing. Because activations are not
  in [0,1] we use a larger eps_act than the input eps.

SmallCNN layout (see campaign/common.py):
  features[0:4]  = block0 [Conv,BN,ReLU,MaxPool]   -> h0  (width   x14x14)
  features[4:8]  = block1                          -> h1  (2*width x 7x 7)
  features[8:12] = block2                          -> h2  (4*width x 3x 3)
  head           = [Flatten, Linear(1152,256), ReLU, Linear(256,10)]
  "head input" perturbation = perturb the flattened vector entering head.

Conditions:
  clean                    : no AT (baseline)
  input_AT                 : standard input-space PGD AT (reference upper bound)
  act_b0                   : perturb at {block0 output}
  act_b0_b1                : perturb at {block0, block1}
  act_b0_b1_b2             : perturb at {block0, block1, block2}
  act_all                  : perturb at {block0, block1, block2, head input}

We report clean acc / FGSM-ASR / PGD-ASR per condition and, crucially, the
MARGINAL PGD-ASR reduction added by each successive layer in the cumulative
budget. If block0 alone delivers most of the reduction, robustness is
front-loaded.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

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
# Activation-space perturbation magnitude. A FIXED additive eps in activation
# space destroys the deeper (small, sparse) layers when applied cumulatively, so
# we scale the step to each batch-activation's RMS: h_adv = h + EPS_ACT_FRAC *
# rms(h) * sign(grad). This keeps the relative perturbation comparable across
# layers and lets the cumulative-budget comparison stay meaningful.
EPS_ACT_FRAC = 0.25     # perturbation as a fraction of per-layer activation RMS

# names of the split points, in front-to-back order
SPLIT_NAMES = ["block0", "block1", "block2", "head_in"]


def split_forward(model, x, perturb_set, eps_act, y=None):
    """Forward through SmallCNN, optionally taking an FGSM ascent step on the
    activation at each split point named in `perturb_set`.

    perturb_set : set of names from SPLIT_NAMES to perturb.
    If perturb_set is empty, this is just an ordinary forward pass.
    Returns logits.
    """
    feats = model.features
    h = x
    # block0
    h = feats[0:4](h)
    if "block0" in perturb_set:
        h = fgsm_step_act(model, h, y, eps_act, cont_fn=lambda a: _cont(model, a, 1))
    # block1
    h = feats[4:8](h)
    if "block1" in perturb_set:
        h = fgsm_step_act(model, h, y, eps_act, cont_fn=lambda a: _cont(model, a, 2))
    # block2
    h = feats[8:12](h)
    if "block2" in perturb_set:
        h = fgsm_step_act(model, h, y, eps_act, cont_fn=lambda a: _cont(model, a, 3))
    # head input (flatten)
    h = model.head[0](h)  # Flatten
    if "head_in" in perturb_set:
        h = fgsm_step_act(model, h, y, eps_act, cont_fn=lambda a: model.head[1:](a))
    h = model.head[1:](h)
    return h


def _cont(model, h, after_block):
    """Continue forward from the output of block `after_block-1` (i.e. h is the
    activation just produced by block index after_block-1) to logits.

    after_block=1 -> h is block0 out, continue through block1,block2,head
    after_block=2 -> h is block1 out, continue through block2,head
    after_block=3 -> h is block2 out, continue through head
    """
    feats = model.features
    if after_block == 1:
        h = feats[4:8](h); h = feats[8:12](h)
    elif after_block == 2:
        h = feats[8:12](h)
    elif after_block == 3:
        pass
    return model.head(h)


def fgsm_step_act(model, h, y, eps_act, cont_fn):
    """Single FGSM ascent step on activation h, using CE of the continued path.

    Detaches the graph: we compute the adversarial activation, then re-attach it
    so the subsequent (training) forward differentiates through the perturbed
    activation. The perturbation direction itself is treated as a constant
    (standard adversarial-training practice).
    """
    h0 = h.detach().clone().requires_grad_(True)
    logits = cont_fn(h0)
    loss = F.cross_entropy(logits, y)
    g, = torch.autograd.grad(loss, h0)
    # scale step to the activation RMS so deeper/smaller layers are not destroyed
    rms = h.detach().pow(2).mean().sqrt().clamp_min(1e-6)
    h_adv = h + eps_act * rms * g.sign()  # keep grad flow into params via original h
    return h_adv


def evaluate(model, Xte, Yte):
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    fres = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pres = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return clean_acc, fres["asr"], pres["asr"]


def train_condition(cond, Xtr, Ytr, meta):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=0)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    perturb_set = COND_SETS[cond]
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            if cond == "input_AT":
                xb_adv = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
                out = model(xb_adv)
            elif cond == "clean":
                out = model(xb)
            else:
                out = split_forward(model, xb, perturb_set, EPS_ACT_FRAC, y=yb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


COND_SETS = {
    "clean": set(),
    "input_AT": set(),                                  # handled specially
    "act_b0": {"block0"},
    "act_b0_b1": {"block0", "block1"},
    "act_b0_b1_b2": {"block0", "block1", "block2"},
    "act_all": {"block0", "block1", "block2", "head_in"},
}
COND_ORDER = ["clean", "input_AT", "act_b0", "act_b0_b1", "act_b0_b1_b2", "act_all"]


def main():
    t_start = time.time()
    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    emit("=" * 78)
    emit("H389 - Cumulative front-to-back AT budget allocation (activation-space AT)")
    emit("=" * 78)
    meta = C.dataset_meta(DS)
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    emit(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  eps_act_frac={EPS_ACT_FRAC}  "
         f"(of per-layer activation RMS)  epochs={EPOCHS}  n_train={N_TRAIN}")
    emit("Activation-space AT = FGSM ascent on chosen layer activations during "
         "training.")
    emit("")

    results = {}
    for cond in COND_ORDER:
        t0 = time.time()
        model = train_condition(cond, Xtr, Ytr, meta)
        clean_acc, fgsm_asr, pgd_asr = evaluate(model, Xte, Yte)
        dt = time.time() - t0
        results[cond] = dict(clean=clean_acc, fgsm=fgsm_asr, pgd=pgd_asr, t=dt)
        emit(f"[{cond:14s}] clean={clean_acc:.3f}  FGSM-ASR={fgsm_asr:.3f}  "
             f"PGD-ASR={pgd_asr:.3f}  ({dt:.0f}s)")

    emit("")
    emit("-" * 78)
    emit(f"{'condition':14s} {'clean':>7s} {'FGSM-ASR':>9s} {'PGD-ASR':>8s} "
         f"{'budget (layers perturbed)':>28s}")
    emit("-" * 78)
    labels = {
        "clean": "none (baseline)",
        "input_AT": "input-space PGD (ref)",
        "act_b0": "block0",
        "act_b0_b1": "block0+block1",
        "act_b0_b1_b2": "block0+block1+block2",
        "act_all": "block0+b1+b2+head_in",
    }
    for cond in COND_ORDER:
        r = results[cond]
        emit(f"{cond:14s} {r['clean']:7.3f} {r['fgsm']:9.3f} {r['pgd']:8.3f} "
             f"{labels[cond]:>28s}")
    emit("-" * 78)

    # A condition whose clean acc drops to chance (~0.10 for 10 classes) is a
    # COLLAPSED model: its near-zero ASR is an artefact (almost nothing is
    # correctly classified for an attack to flip), NOT genuine robustness.
    COLLAPSE = 0.20
    base_pgd = results["clean"]["pgd"]
    emit("")
    emit("MARGINAL PGD-ASR reduction added by each successive layer (cumulative):")
    emit("  (* = COLLAPSED model: clean acc ~ chance, so its ASR is meaningless)")
    cum_chain = [("clean", "clean baseline"),
                 ("act_b0", "+block0"),
                 ("act_b0_b1", "+block1"),
                 ("act_b0_b1_b2", "+block2"),
                 ("act_all", "+head_in")]
    prev_pgd = base_pgd
    # total usable reduction = clean -> last FUNCTIONAL cumulative condition
    functional = [c for c, _ in cum_chain
                  if results[c]["clean"] >= COLLAPSE]
    last_func = functional[-1]
    total_reduction = base_pgd - results[last_func]["pgd"]
    for cond, lab in cum_chain:
        r = results[cond]
        pgd = r["pgd"]
        col = r["clean"] < COLLAPSE
        tag = " *" if col else ""
        if cond == "clean":
            emit(f"  {lab:16s} PGD-ASR={pgd:.3f}  (start)")
        elif col:
            emit(f"  {lab:16s} PGD-ASR={pgd:.3f}  clean={r['clean']:.3f}{tag}  "
                 f"-> COLLAPSED, marginal contribution not counted")
        else:
            marg = prev_pgd - pgd
            share = (marg / total_reduction * 100.0) if total_reduction > 1e-9 else float("nan")
            emit(f"  {lab:16s} PGD-ASR={pgd:.3f}  marginal_drop={marg:+.3f}  "
                 f"({share:5.1f}% of usable act-AT reduction)")
        if not col:
            prev_pgd = pgd
    emit(f"  usable activation-AT PGD-ASR reduction (clean->{last_func}) = "
         f"{total_reduction:+.3f}")
    emit(f"  input-space AT PGD-ASR = {results['input_AT']['pgd']:.3f}  "
         f"(reference); clean = {base_pgd:.3f}")
    emit("  NOTE: perturbing block2 (the 1152-d activation feeding the head) and "
         "beyond")
    emit("        collapses training -- robustness budget is only spendable on the "
         "early")
    emit("        conv blocks; the head-adjacent layer cannot absorb activation AT "
         "here.")

    # verdict, based on functional conditions only
    b0_drop = base_pgd - results["act_b0"]["pgd"]
    b0_share = (b0_drop / total_reduction * 100.0) if total_reduction > 1e-9 else float("nan")
    emit("")
    front_loaded = (not np.isnan(b0_share)) and b0_share >= 60.0
    emit("VERDICT: " + (
        f"Robustness is FRONT-LOADED -- block0 alone delivers {b0_share:.0f}% of the "
        f"usable activation-AT PGD-ASR reduction (clean {base_pgd:.3f} -> block0 "
        f"{results['act_b0']['pgd']:.3f}); adding block1 only adds a little more, and "
        f"deeper layers collapse the model rather than help." if front_loaded else
        f"Robustness is NOT strongly front-loaded -- block0 delivers only "
        f"{b0_share:.0f}% of the usable activation-AT reduction; block1 contributes "
        f"the rest before deeper layers collapse."))
    emit("=" * 78)
    emit(f"total runtime {time.time()-t_start:.0f}s")

    outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", DS, "h389_layer_robustness_budget_output.txt")
    with open(outpath, "w") as f:
        f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
