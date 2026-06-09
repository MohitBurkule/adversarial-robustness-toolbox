"""
H498 - IMP-on-PGD-AT (no rewind): does adversarial-training carve robust
magnitudes that survive pruning + finetuning?

Hypothesis (Sec.5 / Sec.3 G8):
  Starting from a fully-trained PGD-AT model and applying *iterative magnitude
  pruning* (IMP) followed by short PGD-AT finetuning at each round (NO weight
  rewinding to init):
    (i)  at 50% global sparsity, IMP-pruned PGD-AT retains >=95% of dense
         PGD-AT's robust accuracy (PGD-10);
    (ii) at 80% sparsity, robust accuracy drops below 80% of dense (the
         robustness budget breaks);
    (iii) random pruning at 50% sparsity loses >=10 percentage points of
          robust accuracy vs the dense baseline -- showing that the AT-learned
          magnitudes carry a robustness signal (not just any 50%-sparse
          subnet works).

DISTINCTION FROM H462 (must be kept clean):
  H462 is the LOTTERY-TICKET-HYPOTHESIS framing: derive an IMP mask from the
  end-of-AT model, *rewind* the surviving weights to their init values, then
  RE-train from scratch with PGD-AT (Frankle & Carbin 2019; Diffenderfer 2021).
  H498 here is the COMPLEMENTARY PRUNE-AND-FINETUNE framing: keep the
  end-of-AT magnitudes, prune the smallest, FINETUNE the survivors with
  PGD-AT (Han 2015 "Deep Compression" recipe, in the robust setting of
  Sehwag 2020 "HYDRA"). No rewinding, no re-initialisation -- the model
  carries forward the magnitudes learned by adversarial training.

  This is a separate question from H462: H462 asks "does a sparse robust
  subnet *exist* at init", H498 asks "do the AT-learned magnitudes survive
  compression". Both can be true, both can be false, independently.

Papers anchored:
  * Frankle & Carbin 2019, "The Lottery Ticket Hypothesis: Finding Sparse,
    Trainable Neural Networks", ICLR 2019.
  * Sehwag, Wang, Mittal, Jana 2020, "HYDRA: Pruning Adversarially Robust
    Neural Networks", NeurIPS 2020 -- learns importance scores for robust
    pruning; we use the simpler magnitude proxy as in their ablations.
  * Cosentino, Zaiter, Pei, Zhu 2019, "The Search for Sparse, Robust Neural
    Networks", arXiv:1912.02386 -- studies how robust accuracy degrades with
    sparsity, predicts a sharp cliff above ~70-80% sparsity for small CNNs.
  * Han, Pool, Tran, Dally 2015, "Learning both Weights and Connections for
    Efficient Neural Networks", NeurIPS 2015 -- the canonical prune-and-
    finetune recipe (no rewind), which we adapt here.

Procedure (single seed):
  (1) Train DENSE PGD-AT baseline (SmallCNN width=32, 10 epochs).
  (2) IMP-finetune loop for target sparsities S in {0.20, 0.50, 0.80}, in
      ascending order, carrying weights forward each round:
        cur_sparsity = 0
        for s in S:
            # 3 IMP rounds between cur_sparsity and s (geometric schedule)
            for round in 1..3:
                p = cur_sparsity + (s - cur_sparsity) * round / 3
                build global-magnitude mask at sparsity p from current weights
                apply mask
                PGD-AT finetune for FT_EPOCHS epochs (mask re-applied each step)
            cur_sparsity = s
            evaluate (clean acc, PGD ASR, mean margin, per-class robust acc)
  (3) Random-pruning controls at the same sparsities S, also followed by the
      same FT_EPOCHS * 3 PGD-AT finetuning budget -- the only difference is
      WHICH weights are zeroed. (Random mask is drawn once at the start of the
      target sparsity, applied to the dense AT model, then finetuned.)
  (4) Per condition we record clean acc, PGD-10 ASR, mean margin, and
      per-class robust accuracy.
  (5) Verdict checks (i)-(iii) above.

Config: N_TRAIN=6000, N_EVAL=2000, EPOCHS=10, FT_EPOCHS=2, IMP_ROUNDS=3,
BATCH=128, LR=0.05 (cosine; finetune uses LR/5), SGD(mom=0.9, wd=5e-4),
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
FT_EPOCHS = 2
IMP_ROUNDS = 3
LR = 0.05
LR_FT = 0.01            # finetune LR (LR/5)
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
SPARSITIES = [0.20, 0.50, 0.80]

META = {"channels": 1, "size": 28, "n_classes": 10}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sgd(model, lr):
    """SGD mom=0.9 wd=5e-4 (campaign standard)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def build_cnn(seed):
    C.set_seed(seed)
    return C.build_model("cnn", META, width=32, seed=seed)


def prunable_names(model):
    """Conv2d.weight + Linear.weight (>=2D). NOT BN, NOT biases."""
    names = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".weight") and p.dim() >= 2:
            names.append(n)
    return names


def imp_global_mask(model, sparsity, prunable, existing_mask=None):
    """Global magnitude-based mask: keep top-(1-sparsity) by |w| across all
    prunable tensors. If existing_mask is given, dead weights (mask==0) are
    treated as having magnitude -inf so they STAY pruned (monotone IMP).

    Returns {name -> float tensor in {0,1} on each param's device}."""
    flat = []
    for n, p in model.named_parameters():
        if n in prunable:
            v = p.detach().abs().flatten().clone()
            if existing_mask is not None and n in existing_mask:
                v = v * existing_mask[n].flatten()  # dead weights become 0
            flat.append(v)
    allv = torch.cat(flat)
    k_keep = int(round((1.0 - sparsity) * allv.numel()))
    k_keep = max(1, min(k_keep, allv.numel()))
    thresh = torch.topk(allv, k_keep, largest=True, sorted=False).values.min()
    # to break ties cleanly when many zeros exist, use strict ">" if thresh==0
    use_strict = float(thresh.item()) <= 0.0
    masks = {}
    for n, p in model.named_parameters():
        if n in prunable:
            mag = p.detach().abs()
            if existing_mask is not None and n in existing_mask:
                mag = mag * existing_mask[n]
            if use_strict:
                m = (mag > 0).to(p.dtype)
            else:
                m = (mag >= thresh).to(p.dtype)
            masks[n] = m
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
    kept = 0
    total = 0
    for n, p in model.named_parameters():
        if n in masks:
            kept += int(masks[n].sum().item())
            total += p.numel()
    return kept, total


def apply_masks(model, masks):
    if masks is None:
        return
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in masks:
                p.mul_(masks[n])


def pgd_at_train(model, Xtr, Ytr, epochs, lr, masks=None, seed=SEED):
    """Run PGD-AT for `epochs`, reapplying `masks` after every optimiser step.
    Cosine schedule over the local epoch budget."""
    C.set_seed(seed)
    opt = _sgd(model, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    n = Xtr.size(0)
    apply_masks(model, masks)
    model.train()
    for ep in range(epochs):
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


def clone_model_with_state(src):
    """Build a fresh CNN with the same arch and copy src's state_dict into it.
    Used to branch the dense AT model into independent IMP / random arms."""
    dst = build_cnn(SEED)
    dst.load_state_dict(src.state_dict())
    return dst


def eval_robustness(model, X, Y):
    """clean acc, PGD ASR, mean margin (correct-class minus runner-up)."""
    logits, acc = C.logits_and_acc(model, X, Y)
    marg = float(C.margin_of(logits, Y).mean())
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, pg["asr"], marg, pg


def per_class_robust_acc(model, X, Y, ncls=10):
    """Per-class accuracy under PGD-10 attack (i.e. 1 - per-class ASR over the
    class members, restricted to originally-correct samples is messy at small
    scale; instead we report straight robust accuracy = P(model(adv)==y))."""
    model.eval()
    pgd_pred = []
    for i in range(0, X.size(0), 256):
        x, y = X[i:i + 256], Y[i:i + 256]
        xa = C.pgd(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
        with torch.no_grad():
            pgd_pred.append(model(xa).argmax(1).cpu())
    pred = torch.cat(pgd_pred)
    y_cpu = Y.cpu()
    out = []
    for c in range(ncls):
        m = (y_cpu == c)
        if m.sum() == 0:
            out.append(float("nan"))
        else:
            out.append(float((pred[m] == c).float().mean()))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h498_imp_pruning_robust_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H498  IMP-on-PGD-AT (no rewind) -- prune-and-finetune robustness")
    out("      Fashion-MNIST, SmallCNN width=32, PGD-AT 10 ep + IMP finetune")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}  EPOCHS={EPOCHS}  "
        f"FT_EPOCHS={FT_EPOCHS}  IMP_ROUNDS={IMP_ROUNDS}")
    out(f"        LR={LR} LR_FT={LR_FT}  BATCH={BATCH}  SGD(mom=0.9,wd=5e-4)  SEED={SEED}")
    out(f"        EPS={EPS}  PGD_STEPS={PGD_STEPS}  PGD_ALPHA={PGD_ALPHA}")
    out(f"        sparsities = {SPARSITIES}   device = {C.DEVICE}")
    out("")
    out("Distinction from H462: H462 = LTH (rewind to init, retrain from scratch).")
    out("                       H498 = prune-and-finetune (no rewind, keep AT weights).")
    out("")
    flush_file()

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    # ---- (1) dense PGD-AT baseline ----
    out("[1] training DENSE PGD-AT baseline ...")
    dense = build_cnn(SEED)
    pgd_at_train(dense, Xtr, Ytr, epochs=EPOCHS, lr=LR, masks=None, seed=SEED)
    acc_d, asr_d, marg_d, _ = eval_robustness(dense, Xte, Yte)
    rob_d = 1.0 - asr_d  # robust accuracy on originally-correct samples
    pc_d = per_class_robust_acc(dense, Xte, Yte, META["n_classes"])
    prunable = prunable_names(dense)
    total_prunable = sum(dense.state_dict()[n].numel() for n in prunable)
    out(f"    dense PGD-AT: clean_acc={acc_d:.4f}  PGD_ASR={asr_d:.4f}  "
        f"PGD_robust_acc={rob_d:.4f}  mean_margin={marg_d:+.3f}  "
        f"({time.time()-t0:.0f}s)")
    out(f"    total prunable params = {total_prunable} across {len(prunable)} tensors")
    out("")
    flush_file()

    rows = []
    pc_table = {}

    rows.append({"cond": "dense PGD-AT", "sparsity": 0.0,
                 "kept": total_prunable, "total": total_prunable,
                 "acc": acc_d, "asr": asr_d, "rob": rob_d, "margin": marg_d})
    pc_table["dense PGD-AT"] = pc_d

    # ---- (2) IMP-finetune (no rewind), monotone schedule across sparsities ----
    out("=" * 80)
    out("[2] IMP prune-and-finetune (no rewind), cumulative across sparsities")
    out("=" * 80)
    imp_model = clone_model_with_state(dense)  # start from the dense AT model
    imp_mask = None
    cur_sparsity = 0.0
    for s_target in SPARSITIES:
        out("")
        out(f"-- target sparsity = {s_target:.2f} --")
        # IMP_ROUNDS rounds: geometric / linear sweep between cur_sparsity and s_target
        for r in range(1, IMP_ROUNDS + 1):
            p_r = cur_sparsity + (s_target - cur_sparsity) * (r / IMP_ROUNDS)
            imp_mask = imp_global_mask(imp_model, p_r, prunable,
                                       existing_mask=imp_mask)
            apply_masks(imp_model, imp_mask)
            kept_r, _ = remaining_params(imp_model, imp_mask)
            out(f"    round {r}/{IMP_ROUNDS}: target_sparsity={p_r:.4f}  "
                f"kept={kept_r}/{total_prunable} ({100.0*kept_r/total_prunable:.2f}%)")
            pgd_at_train(imp_model, Xtr, Ytr, epochs=FT_EPOCHS, lr=LR_FT,
                         masks=imp_mask, seed=SEED + 100 + r)
        # evaluate at this sparsity
        acc_i, asr_i, marg_i, _ = eval_robustness(imp_model, Xte, Yte)
        rob_i = 1.0 - asr_i
        kept_i, _ = remaining_params(imp_model, imp_mask)
        out(f"    IMP@{int(s_target*100):02d}%: clean_acc={acc_i:.4f}  "
            f"PGD_ASR={asr_i:.4f}  PGD_robust_acc={rob_i:.4f}  "
            f"mean_margin={marg_i:+.3f}  ({time.time()-t0:.0f}s)")
        rows.append({"cond": f"IMP@{int(s_target*100):02d}%", "sparsity": s_target,
                     "kept": kept_i, "total": total_prunable,
                     "acc": acc_i, "asr": asr_i, "rob": rob_i, "margin": marg_i})
        pc_table[f"IMP@{int(s_target*100):02d}%"] = per_class_robust_acc(
            imp_model, Xte, Yte, META["n_classes"])
        cur_sparsity = s_target
        flush_file()

    # ---- (3) Random pruning controls at the same sparsities ----
    out("")
    out("=" * 80)
    out("[3] RANDOM pruning controls (one-shot random mask + same finetune budget)")
    out("=" * 80)
    total_ft = IMP_ROUNDS * FT_EPOCHS  # same finetune-epoch budget as IMP path
    for s_target in SPARSITIES:
        out("")
        out(f"-- random sparsity = {s_target:.2f} --")
        m_rnd = clone_model_with_state(dense)
        rnd_msk = random_mask(m_rnd, s_target, prunable,
                              seed=SEED * 10_000 + int(s_target * 100))
        apply_masks(m_rnd, rnd_msk)
        kept_r, _ = remaining_params(m_rnd, rnd_msk)
        out(f"    random mask: kept={kept_r}/{total_prunable} "
            f"({100.0*kept_r/total_prunable:.2f}%)")
        pgd_at_train(m_rnd, Xtr, Ytr, epochs=total_ft, lr=LR_FT,
                     masks=rnd_msk, seed=SEED + 500 + int(s_target * 100))
        acc_r, asr_r, marg_r, _ = eval_robustness(m_rnd, Xte, Yte)
        rob_r = 1.0 - asr_r
        out(f"    RND@{int(s_target*100):02d}%: clean_acc={acc_r:.4f}  "
            f"PGD_ASR={asr_r:.4f}  PGD_robust_acc={rob_r:.4f}  "
            f"mean_margin={marg_r:+.3f}  ({time.time()-t0:.0f}s)")
        rows.append({"cond": f"RND@{int(s_target*100):02d}%", "sparsity": s_target,
                     "kept": kept_r, "total": total_prunable,
                     "acc": acc_r, "asr": asr_r, "rob": rob_r, "margin": marg_r})
        pc_table[f"RND@{int(s_target*100):02d}%"] = per_class_robust_acc(
            m_rnd, Xte, Yte, META["n_classes"])
        flush_file()

    # ---- (4) main tables ----
    out("")
    out("=" * 80)
    out("[4] MAIN TABLE  (PGD-10 robust accuracy = 1 - ASR over correct samples)")
    out("=" * 80)
    hdr = "{:<14} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "condition", "sparsity", "kept", "clean_acc", "PGD_ASR", "rob_acc", "margin")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<14} {:>9.2f} {:>10} {:>10.4f} {:>10.4f} {:>10.4f} {:>+10.3f}".format(
            r["cond"], r["sparsity"], r["kept"],
            r["acc"], r["asr"], r["rob"], r["margin"]))
    out("-" * len(hdr))
    out("")

    # ---- (4b) per-class robust accuracy table ----
    out("=" * 80)
    out("[4b] PER-CLASS robust accuracy (P(argmax(model(adv)) == y) per class)")
    out("=" * 80)
    pc_hdr = "{:<14}".format("condition") + "".join(
        ["{:>7}".format(f"c{c}") for c in range(META["n_classes"])])
    out(pc_hdr)
    out("-" * len(pc_hdr))
    cond_order = ["dense PGD-AT"]
    for s in SPARSITIES:
        cond_order.append(f"IMP@{int(s*100):02d}%")
    for s in SPARSITIES:
        cond_order.append(f"RND@{int(s*100):02d}%")
    for cond in cond_order:
        vals = pc_table[cond]
        out("{:<14}".format(cond) + "".join(["{:>7.3f}".format(v) for v in vals]))
    out("-" * len(pc_hdr))

    # per-class drop vs dense (for IMP@50% and RND@50% specifically)
    out("")
    out("  per-class drop in robust acc vs dense PGD-AT:")
    base = np.array(pc_table["dense PGD-AT"])
    for cond in cond_order[1:]:
        cur = np.array(pc_table[cond])
        d = cur - base
        worst = int(np.nanargmin(d))
        out("    {:<14} drop_mean={:+.3f}  drop_worst_class=c{} ({:+.3f})".format(
            cond, float(np.nanmean(d)), worst, float(d[worst])))
    out("")
    flush_file()

    # ---- (5) verdict ----
    out("=" * 80)
    out("[5] VERDICT")
    out("=" * 80)
    by = {r["cond"]: r for r in rows}
    dense_rob = by["dense PGD-AT"]["rob"]
    out(f"  dense PGD-AT baseline: PGD_robust_acc = {dense_rob:.4f}")
    out("")

    # check (i): IMP@50% >= 95% of dense robust acc
    imp50 = by["IMP@50%"]["rob"]
    ratio50 = imp50 / max(dense_rob, 1e-9)
    cond_i = ratio50 >= 0.95
    out(f"  (i)  IMP@50% PGD_robust_acc = {imp50:.4f}  "
        f"({100*ratio50:.1f}% of dense)  -- target >=95%  -> {cond_i}")

    # check (ii): IMP@80% robust acc drops below 80% of dense
    imp80 = by["IMP@80%"]["rob"]
    ratio80 = imp80 / max(dense_rob, 1e-9)
    cond_ii = ratio80 < 0.80
    out(f"  (ii) IMP@80% PGD_robust_acc = {imp80:.4f}  "
        f"({100*ratio80:.1f}% of dense)  -- target <80%   -> {cond_ii}")

    # check (iii): RND@50% loses >=10 pp robust acc vs dense
    rnd50 = by["RND@50%"]["rob"]
    drop_rnd = dense_rob - rnd50
    cond_iii = drop_rnd >= 0.10
    out(f"  (iii) RND@50% PGD_robust_acc = {rnd50:.4f}  "
        f"(drop = {drop_rnd:+.4f})  -- target drop >=0.10 -> {cond_iii}")

    # gap between IMP@50% and RND@50% (the "magnitudes carry robust signal" claim)
    gap50 = imp50 - rnd50
    out("")
    out(f"  IMP@50% - RND@50% robust-acc gap = {gap50:+.4f}  "
        f"(positive => AT magnitudes beat random pruning at same sparsity)")
    out("")

    supported = cond_i and cond_ii and cond_iii
    if supported:
        verdict = ("SUPPORTED: IMP on the end-of-AT model survives 50% sparsity "
                   "with >=95% of dense robust accuracy, breaks at 80%, and "
                   "random pruning at 50% loses >=10pp. AT-learned magnitudes "
                   "carry robust signal that finetuning preserves (Sehwag 2020; "
                   "Cosentino 2019).")
    elif cond_i and cond_iii and not cond_ii:
        verdict = ("PARTIAL: IMP@50% preserves robustness and beats random "
                   "pruning at 50% (magnitudes do carry signal), but the 80% "
                   "cliff predicted by Cosentino 2019 is not reached at this "
                   "scale -- finetuning recovers even at 80%.")
    elif cond_i and cond_ii and not cond_iii:
        verdict = ("PARTIAL: IMP@50% retains robustness and 80% breaks it, but "
                   "random pruning at 50% does NOT lose >=10pp -- at this scale "
                   "of CNN, sparsity itself is tolerated and the magnitude "
                   "ranking is not what is doing the work.")
    elif (not cond_i) and cond_iii:
        verdict = ("NOT SUPPORTED for (i): IMP@50% does not retain >=95% of "
                   "dense robust accuracy; pruning at this scale damages "
                   "robustness even with finetuning. The random-pruning gap "
                   "is large though, so magnitudes still help.")
    else:
        verdict = ("NOT SUPPORTED: the H498 prune-and-finetune story does not "
                   "hold at this scale (N_train=6000, width=32). Either the "
                   "model is too small for IMP to find a robust subnet, or "
                   "the finetune budget is too short.")
    out("  ONE-LINE VERDICT: " + verdict)

    out("")
    out(f"done in {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
