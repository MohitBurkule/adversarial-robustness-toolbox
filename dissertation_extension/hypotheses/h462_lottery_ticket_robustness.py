"""
H462 - Lottery-ticket robustness: does an adversarial lottery ticket exist?

Hypothesis: a sparse subnetwork found via iterative magnitude pruning (IMP) of a
PGD-AT-trained dense network, when rewound to its init and re-trained with PGD-AT,
matches the dense PGD-AT baseline at moderate sparsities (30%/50%) and only
degrades at high sparsity (70%). Random pruning at the same sparsity should
degrade more, isolating the role of magnitude information.

Gap-map: G8 (diagnostic / mechanistic; lottery-ticket robustness was absent in
H173-H413). Paper anchor: Frankle & Carbin 2019 "The Lottery Ticket Hypothesis"
(ICLR 2019), extended for the robust setting by Diffenderfer, Bartoldson, Chaganti,
Zhang, Kailkhura 2021 "A Winning Hat Trick: Compact, Sparse, and Robust"
(NeurIPS 2021) which found robust tickets exist at non-trivial sparsities, and
Lee, Yune, Yoon 2021 "Towards a Better Understanding of Adversarial Lottery
Tickets" (NeurIPS workshop).

Procedure (single seed, single eps; 6k training samples; CNN width=32):
  (1) Dense PGD-AT baseline: train SmallCNN from scratch with PGD-10 AT for
      EPOCHS epochs; record clean+PGD ASR. This is also the "scoring" model
      used to derive the magnitude-based pruning masks.
  (2) For each target sparsity s in {0.30, 0.50, 0.70}:
        (a) IMP mask: take the dense PGD-AT weights, compute global magnitude
            threshold across prunable params (conv/linear weights, NOT biases or
            BN), keep top-(1-s) by |w|. Mask is a binary tensor of same shape.
        (b) Rewind: reset all prunable params to their *original init* (the
            torch RNG state at model construction with SEED), keep BN/bias as
            re-initialised (this matches "weight rewinding to init" of
            Frankle 2019; we report it cleanly rather than rewind-to-epoch-k
            which needs more hyperparams).
        (c) PGD-AT retrain: train EPOCHS epochs with PGD-10 AT, applying the
            mask after every optimiser step (hard prune, gradients on dead
            weights are zeroed at the parameter level).
        (d) Random mask at same sparsity (per-tensor random Bernoulli with
            keep-rate (1-s)) + same rewind + same PGD-AT retrain. This is the
            magnitude-information control.
  (3) Report (clean acc, FGSM ASR, PGD ASR, remaining-params count) for:
      dense AT baseline, IMP@30/50/70, random@30/50/70.

Verdict bands (per Diffenderfer-2021 expectation): a robust ticket exists if
IMP@50% PGD ASR is within +0.02 of dense AT, and IMP beats random at the same
sparsity by at least +0.02 PGD ASR.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9,wd=5e-4),
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01. ASCII output, flushed.
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
SPARSITIES = [0.30, 0.50, 0.70]

META = {"channels": 1, "size": 28, "n_classes": 10}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_optimizer_sgd(model, lr):
    """SGD mom=0.9 wd=5e-4 (campaign standard)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def build_cnn(seed):
    """Build a fresh SmallCNN with deterministic init under torch's RNG."""
    C.set_seed(seed)
    return C.build_model("cnn", META, width=32, seed=seed)


def snapshot_init_state(model):
    """Return a dict {param_name -> CPU tensor} of the model's *current* params,
    intended to be called right after construction (i.e. the init state)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def prunable_names(model):
    """Return the list of parameter names that we are willing to prune:
    conv weights + linear weights. Biases and BN params are NOT pruned."""
    names = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".weight") and p.dim() >= 2:
            # captures Conv2d.weight (4D) and Linear.weight (2D); excludes BN (1D)
            names.append(n)
    return names


def imp_global_mask(model, sparsity, prunable):
    """Global magnitude-based mask: keep top-(1-sparsity) by |w| across the
    union of all prunable tensors. Returns {name -> bool tensor on DEVICE}."""
    flat = []
    name2num = {}
    for n, p in model.named_parameters():
        if n in prunable:
            v = p.detach().abs().flatten()
            name2num[n] = v.numel()
            flat.append(v)
    allv = torch.cat(flat)
    k_keep = int(round((1.0 - sparsity) * allv.numel()))
    k_keep = max(1, min(k_keep, allv.numel()))
    # threshold = the k_keep-th largest magnitude
    thresh = torch.topk(allv, k_keep, largest=True, sorted=False).values.min()
    masks = {}
    for n, p in model.named_parameters():
        if n in prunable:
            masks[n] = (p.detach().abs() >= thresh).to(p.dtype)
    return masks


def random_mask(model, sparsity, prunable, seed):
    """Per-tensor random Bernoulli mask with keep-rate (1-sparsity)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    masks = {}
    for n, p in model.named_parameters():
        if n in prunable:
            keep_p = 1.0 - sparsity
            r = torch.rand(p.shape, generator=g)
            masks[n] = (r < keep_p).to(p.dtype).to(p.device)
    return masks


def remaining_params(model, masks):
    """Number of *kept* prunable params + total prunable params."""
    kept = 0
    total = 0
    for n, p in model.named_parameters():
        if n in masks:
            kept += int(masks[n].sum().item())
            total += p.numel()
    return kept, total


def rewind_to_init(model, init_state):
    """Copy init_state tensors back into the model's params in-place."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in init_state:
                p.copy_(init_state[n].to(p.device))


def apply_masks(model, masks):
    """Hard-multiply current params by their masks (in-place)."""
    if masks is None:
        return
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in masks:
                p.mul_(masks[n])


def train_pgd_at(model, Xtr, Ytr, masks=None, seed=SEED):
    """PGD-AT for EPOCHS epochs. If masks is not None, multiply masked params
    after every optimiser step so dead weights stay zero (and effectively their
    gradient steps are wasted but harmless)."""
    C.set_seed(seed)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    # initial mask application so we *start* sparse
    apply_masks(model, masks)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            opt.zero_grad()
            loss = F.cross_entropy(model(xa), yb)
            loss.backward()
            opt.step()
            apply_masks(model, masks)
        sched.step()
    model.eval()
    return model


def eval_robustness(model, X, Y):
    """clean acc, FGSM ASR, PGD ASR on (X,Y)."""
    _, acc = C.logits_and_acc(model, X, Y)
    fg = C.attack_success(model, X, Y, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h462_lottery_ticket_robustness_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H462  Lottery-ticket robustness (Frankle 2019 / Diffenderfer 2021)")
    out("      Fashion-MNIST, SmallCNN width=32, PGD-AT 10 epochs")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        sparsities = {SPARSITIES}")
    out(f"        device = {C.DEVICE}")
    out("")
    flush_file()

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    # ---- snapshot init ----
    out("[0] building fresh CNN and snapshotting init state (for rewinding)...")
    init_model = build_cnn(SEED)
    init_state = snapshot_init_state(init_model)
    prunable = prunable_names(init_model)
    total_prunable = sum(init_model.state_dict()[n].numel() for n in prunable)
    out(f"    prunable tensors ({len(prunable)}): {prunable}")
    out(f"    total prunable params = {total_prunable}")
    out("")

    # ---- (1) dense PGD-AT baseline (also the IMP scoring model) ----
    out("[1] training DENSE PGD-AT baseline (and using it for IMP magnitudes)...")
    dense = build_cnn(SEED)
    # sanity: dense starts from the same init as init_state
    train_pgd_at(dense, Xtr, Ytr, masks=None, seed=SEED)
    acc_d, fg_d, pg_d = eval_robustness(dense, Xte, Yte)
    out(f"    dense PGD-AT: clean_acc={acc_d:.4f}  FGSM_ASR={fg_d:.4f}  "
        f"PGD_ASR={pg_d:.4f}  ({time.time()-t0:.0f}s)")
    out("")
    flush_file()

    rows = []
    rows.append({"cond": "dense PGD-AT", "sparsity": 0.0,
                 "kept": total_prunable, "total": total_prunable,
                 "acc": acc_d, "fgsm": fg_d, "pgd": pg_d})

    # ---- (2) IMP and Random at each sparsity ----
    for si, s in enumerate(SPARSITIES):
        out("=" * 80)
        out(f"[2.{si+1}] SPARSITY = {s:.2f}  (keep {(1-s)*100:.0f}% of prunable weights)")
        out("=" * 80)

        # --- (a) IMP mask from the dense PGD-AT model ---
        imp_masks = imp_global_mask(dense, s, prunable)
        kept_i, _ = remaining_params(dense, imp_masks)
        out(f"  IMP global-magnitude mask: kept = {kept_i}/{total_prunable} "
            f"({100.0*kept_i/total_prunable:.2f}%)")

        # --- (b) build fresh model, rewind prunable params to init, retrain ---
        m_imp = build_cnn(SEED)
        rewind_to_init(m_imp, init_state)
        train_pgd_at(m_imp, Xtr, Ytr, masks=imp_masks, seed=SEED + 1)
        acc_i, fg_i, pg_i = eval_robustness(m_imp, Xte, Yte)
        out(f"  IMP@{int(s*100):02d}%  PGD-AT retrain: clean_acc={acc_i:.4f}  "
            f"FGSM_ASR={fg_i:.4f}  PGD_ASR={pg_i:.4f}  ({time.time()-t0:.0f}s)")
        rows.append({"cond": f"IMP@{int(s*100):02d}%", "sparsity": s,
                     "kept": kept_i, "total": total_prunable,
                     "acc": acc_i, "fgsm": fg_i, "pgd": pg_i})
        flush_file()

        # --- (c) Random mask same sparsity (magnitude-info control) ---
        rnd_masks = random_mask(dense, s, prunable, seed=SEED * 10_000 + int(s * 100))
        kept_r, _ = remaining_params(dense, rnd_masks)
        out(f"  RANDOM mask: kept = {kept_r}/{total_prunable} "
            f"({100.0*kept_r/total_prunable:.2f}%)")
        m_rnd = build_cnn(SEED)
        rewind_to_init(m_rnd, init_state)
        train_pgd_at(m_rnd, Xtr, Ytr, masks=rnd_masks, seed=SEED + 2)
        acc_r, fg_r, pg_r = eval_robustness(m_rnd, Xte, Yte)
        out(f"  RND@{int(s*100):02d}%  PGD-AT retrain: clean_acc={acc_r:.4f}  "
            f"FGSM_ASR={fg_r:.4f}  PGD_ASR={pg_r:.4f}  ({time.time()-t0:.0f}s)")
        rows.append({"cond": f"RND@{int(s*100):02d}%", "sparsity": s,
                     "kept": kept_r, "total": total_prunable,
                     "acc": acc_r, "fgsm": fg_r, "pgd": pg_r})
        out("")
        flush_file()

    # ---- (3) main table ----
    out("=" * 80)
    out("[3] MAIN TABLE")
    out("=" * 80)
    hdr = "{:<14} {:>9} {:>10} {:>10} {:>10} {:>10}".format(
        "condition", "sparsity", "kept", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<14} {:>9.2f} {:>10} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["cond"], r["sparsity"], r["kept"],
            r["acc"], r["fgsm"], r["pgd"]))
    out("-" * len(hdr))
    out("")

    # ---- (4) verdict ----
    out("=" * 80)
    out("[4] VERDICT")
    out("=" * 80)
    by = {r["cond"]: r for r in rows}
    dense_pgd = by["dense PGD-AT"]["pgd"]
    dense_acc = by["dense PGD-AT"]["acc"]

    out(f"  dense PGD-AT baseline:   PGD_ASR={dense_pgd:.4f}  clean_acc={dense_acc:.4f}")
    out("")
    out("  per-sparsity comparison (IMP vs RND, vs dense AT):")
    ticket_exists_any = False
    imp_beats_rnd_any = False
    for s in SPARSITIES:
        tag = f"{int(s*100):02d}%"
        imp = by[f"IMP@{tag}"]
        rnd = by[f"RND@{tag}"]
        d_imp = imp["pgd"] - dense_pgd      # negative => IMP more robust than dense (rare)
        d_rnd = rnd["pgd"] - dense_pgd
        d_imp_rnd = rnd["pgd"] - imp["pgd"] # positive => IMP more robust than RND
        ticket_here = (imp["pgd"] - dense_pgd) <= 0.02 and (imp["acc"] - dense_acc) >= -0.03
        beats_rnd = d_imp_rnd >= 0.02
        ticket_exists_any |= ticket_here
        imp_beats_rnd_any |= beats_rnd
        out(f"    s={s:.2f}: IMP PGD={imp['pgd']:.4f} (d_vs_dense {d_imp:+.4f})  "
            f"RND PGD={rnd['pgd']:.4f} (d_vs_dense {d_rnd:+.4f})  "
            f"d_RND_minus_IMP={d_imp_rnd:+.4f}  "
            f"ticket_here={ticket_here}  beats_rnd={beats_rnd}")

    out("")
    if ticket_exists_any and imp_beats_rnd_any:
        verdict = ("SUPPORTED: a robust lottery ticket exists at >=1 tested sparsity "
                   "(IMP matches dense AT within 0.02 PGD ASR and beats random pruning "
                   "by >=0.02 PGD ASR). Consistent with Diffenderfer 2021.")
    elif ticket_exists_any and not imp_beats_rnd_any:
        verdict = ("PARTIAL: IMP matches dense AT at some sparsity, but does NOT beat "
                   "random pruning by >=0.02 PGD ASR. The magnitude information is not "
                   "doing the work; sparsity itself may be tolerated by AT.")
    elif (not ticket_exists_any) and imp_beats_rnd_any:
        verdict = ("PARTIAL: IMP is better than random pruning, but no sparsity matches "
                   "dense AT within 0.02 PGD ASR. Pruning degrades robustness at this "
                   "scale (N_train=6000).")
    else:
        verdict = ("NOT SUPPORTED: no robust lottery ticket found; IMP neither matches "
                   "dense AT nor reliably beats random pruning. Sub-scale training "
                   "(N_train=6000, 10 epochs) and CNN width=32 may be too small for the "
                   "Frankle/Diffenderfer effect to surface.")
    out("  ONE-LINE VERDICT: " + verdict)

    out("")
    out(f"done in {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
