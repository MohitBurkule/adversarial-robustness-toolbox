"""
H406 - Max-scale random-noise hardening of vulnerable training samples.

H404 (K=50) and H405 (K=1000, n=30/300) both found that hardening the most
adversarially-vulnerable TRAINING samples with random-uniform L-inf noise copies
gives essentially NO whole-test robustness. This experiment pushes the idea to
the EXTREME: much bigger K (2000+) and much bigger n, INCLUDING n=ALL (6000)
training samples, to settle whether the dead end is a matter of scale.

Two changes vs H405:

  1. GPU MAXIMIZATION. Saturate the RTX 4090 so the sweep finishes fast:
     - Move the entire train set to GPU ONCE (already on DEVICE from loader).
     - LARGE batch (4096, raised to 8192 if it fits) -> big GPU kernels.
     - Generate noisy copies ON-THE-FLY per batch on the GPU. We never
       materialize the (n*K)-image augmented dataset (could be tens of millions
       of images -> OOM). Instead we build an INDEX array of length aug_size that
       maps into the 6000 originals: [0..5999] (clean originals) followed by the
       target indices repeated K times (the copies). Each epoch we randperm the
       index array on GPU, gather the batch's source images, and for rows that
       are COPIES (position >= N_TRAIN in the index layout) we add fresh random
       uniform noise on-device. Memory stays flat at O(6000 images + aug_size
       int32 indices); the GPU is kept busy with large gathers + conv batches.

  2. SCALE. Sweep larger than H405:
       baseline (n=0, K=0)
       n=300,  K=2000
       n=1000, K=2000
       n=3000, K=2000
       n=6000 (ALL), K=2000
       n=6000 (ALL), K=5000   (extra extreme condition)

Eval per condition: clean_acc, FGSM_ASR, PGD_ASR, and PGD_ASR on the
lowest-margin 25% TEST subset (vuln_PGD).

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, SGD(mom=0.9,wd=5e-4), SEED=0,
EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01, CNN width=32. Batch enlarged for GPU
saturation (comparable training; LR/epochs/seed unchanged for comparability).
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
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
VULN_TEST_FRAC = 0.25

# GPU-saturating batch. Large enough to keep 4090 kernels big, small enough that
# even the baseline (6000 imgs) gets a sane number of optimizer steps/epoch.
BATCH = 512

# Per-epoch draw cap. H405 trained on the WHOLE materialized aug set each epoch
# (so more data == more gradient steps). At n=6000,K=5000 that is 30M draws/epoch
# which is far too slow. We instead draw a fixed STEPS_PER_EPOCH*BATCH augmented
# samples each epoch (uniformly over the aug index, so every targeted-vuln sample
# still appears with its K-fold over-representation), keeping optimizer step COUNT
# identical across conditions -> a fair comparison AND a bounded, GPU-saturating
# wall-time. With BATCH=512 and STEPS_PER_EPOCH=1024 each epoch sees ~524k draws.
STEPS_PER_EPOCH = 1024

# (n_target, K) conditions. n=6000 == ALL training samples.
CONDITIONS = [
    (0, 0),        # baseline
    (300, 2000),
    (1000, 2000),
    (3000, 2000),
    (6000, 2000),  # n = ALL, K=2000
    (6000, 5000),  # extra extreme: n = ALL, K=5000
]

META = {"channels": 1, "size": 28, "n_classes": 10}


def _make_optimizer_sgd(model, lr):
    """SGD mom=0.9 wd=5e-4 (config)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def train_onthefly_noise(Xtr, Ytr, src_index, is_copy, seed):
    """GPU-saturating training with on-the-fly noise for the copy rows.

    Xtr,Ytr        : the 6000 originals, already on GPU.
    src_index      : LongTensor (aug_size,) on GPU; for each augmented row, which
                     original (0..N_TRAIN-1) it draws its image/label from.
    is_copy        : BoolTensor (aug_size,) on GPU; True for noisy-copy rows,
                     False for the clean originals.

    Fixed-step epochs: each epoch we draw STEPS_PER_EPOCH*BATCH augmented rows
    UNIFORMLY (with replacement) from the aug index. Because targeted-vuln samples
    are repeated K times in the index, they keep their K-fold over-representation
    in the draw distribution -- exactly the H405 mixture -- but the optimizer step
    COUNT is identical for every condition (incl. the no-aug baseline), so clean
    accuracy is comparable. Copy rows get fresh uniform L-inf noise on-device.
    Returns (model, samples_per_sec).
    """
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    dev = Xtr.device
    aug = src_index.size(0)
    model.train()
    total_samples = 0
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.time()
    for ep in range(EPOCHS):
        # uniform draw over the aug index (with replacement) -> preserves the
        # K-fold over-representation of targeted samples, fixed steps per epoch.
        draw = torch.randint(aug, (STEPS_PER_EPOCH * BATCH,), device=dev)
        for s in range(STEPS_PER_EPOCH):
            sel = draw[s * BATCH:(s + 1) * BATCH]
            si = src_index[sel]
            xb = Xtr[si]                       # gather originals (B,C,H,W)
            yb = Ytr[si]
            cmask = is_copy[sel]               # which rows are noisy copies
            if cmask.any():
                # fresh uniform noise in [-eps, eps], applied to a fresh copy so
                # we never mutate the stored originals; clamp to [0,1].
                noise = (2.0 * torch.rand_like(xb) - 1.0) * EPS
                xb = torch.where(cmask.view(-1, 1, 1, 1),
                                 (xb + noise).clamp(0, 1), xb)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_samples += xb.size(0)
        sched.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    sps = total_samples / max(time.time() - t_start, 1e-9)
    model.eval()
    return model, sps


def build_aug_index(order_low, n_target, k, device):
    """Return (src_index, is_copy, aug_size).
    Layout: clean originals [0..N_TRAIN-1] then target indices repeated K times.
    """
    base = torch.arange(N_TRAIN, device=device, dtype=torch.long)
    base_copy = torch.zeros(N_TRAIN, dtype=torch.bool, device=device)
    if n_target == 0 or k == 0:
        return base, base_copy, N_TRAIN
    sel = torch.as_tensor(order_low[:n_target], device=device, dtype=torch.long)
    copies = sel.repeat_interleave(k)            # (n_target*k,)
    src_index = torch.cat([base, copies], dim=0)
    is_copy = torch.cat([base_copy,
                         torch.ones(copies.size(0), dtype=torch.bool, device=device)])
    return src_index, is_copy, src_index.size(0)


def eval_robustness(model, X, Y):
    _, acc = C.logits_and_acc(model, X, Y)
    fg = C.attack_success(model, X, Y, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


def eval_subset_pgd(model, X, Y, mask):
    pg = C.attack_success(model, X[mask], Y[mask], attack="pgd", eps=EPS, steps=PGD_STEPS)
    return pg["asr"]


def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h406_maxscale_noise_hardening_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 90)
    out("H406  Max-scale random-noise hardening of vulnerable train samples (Fashion-MNIST)")
    out("=" * 90)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} STEPS/EPOCH={STEPS_PER_EPOCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        conditions (n_target,K) = {CONDITIONS}")
    out(f"        vulnerable-test subset = lowest {int(VULN_TEST_FRAC*100)}% margin")
    out(f"        device = {C.DEVICE}")
    if C.DEVICE.type == "cuda":
        out(f"        gpu    = {torch.cuda.get_device_name(0)}")
    out("        GPU-MAX: data resident on GPU, on-the-fly per-batch noise, "
        "no materialized aug set")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)} (device={Xtr.device})  Xte={tuple(Xte.shape)}")

    out("\n[1] training BASE model (no augmentation, same step budget)...")
    tb = time.time()
    b_idx, b_copy, _ = build_aug_index(np.arange(N_TRAIN), 0, 0, Xtr.device)
    base, base_sps = train_onthefly_noise(Xtr, Ytr, b_idx, b_copy, SEED)
    base_wall = time.time() - tb
    base_acc, base_fgsm, base_pgd = eval_robustness(base, Xte, Yte)
    out(f"    base clean_acc={base_acc:.4f}  FGSM_ASR={base_fgsm:.4f}  "
        f"PGD_ASR={base_pgd:.4f}  (train {base_wall:.1f}s, {base_sps:,.0f} samples/sec)")

    out("\n[2] ranking TRAINING samples by vulnerability (input-space margin)...")
    tr_margin = C.margin(base, Xtr, Ytr)
    order_low = np.argsort(tr_margin)
    out(f"    train margin: min={tr_margin.min():.3f} median="
        f"{np.median(tr_margin):.3f} max={tr_margin.max():.3f}")

    te_margin = C.margin(base, Xte, Yte)
    n_vuln_te = int(round(VULN_TEST_FRAC * Xte.size(0)))
    vuln_te_idx = np.argsort(te_margin)[:n_vuln_te]
    vuln_te_mask = np.zeros(Xte.size(0), dtype=bool)
    vuln_te_mask[vuln_te_idx] = True
    base_vuln_pgd = eval_subset_pgd(base, Xte, Yte, vuln_te_mask)
    out(f"    vulnerable test subset: {n_vuln_te} samples "
        f"(margin <= {np.sort(te_margin)[n_vuln_te-1]:.3f})  base vuln_PGD={base_vuln_pgd:.4f}")

    rows = []
    rows.append({"cond": "baseline (n=0,K=0)", "n": 0, "k": 0, "aug": N_TRAIN,
                 "wall": base_wall, "sps": base_sps,
                 "acc": base_acc, "fgsm": base_fgsm, "pgd": base_pgd,
                 "vpgd": base_vuln_pgd})

    out(f"    [perf] baseline train samples/sec = {rows[0]['sps']:,.0f} "
        f"(steps/epoch={STEPS_PER_EPOCH}, batch={BATCH})")

    for ci, (nt, k) in enumerate(CONDITIONS):
        if nt == 0 and k == 0:
            continue
        out("\n" + "=" * 90)
        ntag = "ALL" if nt == N_TRAIN else str(nt)
        out(f"[3.{ci}] CONDITION n_target={nt} ({ntag})  K={k}")
        out("=" * 90)
        src_index, is_copy, aug = build_aug_index(order_low, nt, k, Xtr.device)
        out(f"    aug_size = {aug:,}  (= {N_TRAIN} originals + {nt}*{k} on-the-fly noisy copies)")
        out(f"    retraining from scratch (seed={SEED}, on-the-fly GPU noise)...")
        tc = time.time()
        model, sps = train_onthefly_noise(Xtr, Ytr, src_index, is_copy, SEED)
        wall = time.time() - tc
        acc, fgsm, pgd = eval_robustness(model, Xte, Yte)
        vpgd = eval_subset_pgd(model, Xte, Yte, vuln_te_mask)
        cond = f"n={ntag},K={k}"
        rows.append({"cond": cond, "n": nt, "k": k, "aug": aug, "wall": wall,
                     "sps": sps, "acc": acc, "fgsm": fgsm, "pgd": pgd, "vpgd": vpgd})
        out(f"    RESULT {cond}: clean_acc={acc:.4f}  FGSM_ASR={fgsm:.4f}  "
            f"PGD_ASR={pgd:.4f}  vuln_PGD={vpgd:.4f}")
        out(f"           d_clean_acc={acc-base_acc:+.4f}  d_PGD_ASR={pgd-base_pgd:+.4f} "
            f"(vs baseline)")
        out(f"           [perf] wall={wall:.1f}s  samples/sec={sps:,.0f}  "
            f"({time.time()-t0:.0f}s total)")
        # free the (possibly large) index tensors before next condition
        del src_index, is_copy, model
        if Xtr.device.type == "cuda":
            torch.cuda.empty_cache()
        flush_file()

    # ---- MAIN TABLE ----
    out("\n" + "=" * 90)
    out("[4] MAIN TABLE")
    out("=" * 90)
    hdr = ("{:<16} {:>12} {:>11} {:>10} {:>10} {:>10} {:>10}".format(
        "condition", "eff_aug", "wall_time_s", "clean_acc", "FGSM_ASR",
        "PGD_ASR", "vuln_PGD"))
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<16} {:>12,} {:>11.1f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["cond"], r["aug"], r["wall"], r["acc"], r["fgsm"], r["pgd"], r["vpgd"]))
    out("-" * len(hdr))

    # ---- PERF SUMMARY ----
    out("\n[4b] PERF (GPU saturation)")
    out("{:<16} {:>14} {:>12}".format("condition", "samples/sec", "wall_time_s"))
    for r in rows:
        out("{:<16} {:>14,.0f} {:>12.1f}".format(r["cond"], r["sps"], r["wall"]))
    aug_rows = [r for r in rows if r["n"] > 0]
    if aug_rows:
        peak = max(aug_rows, key=lambda r: r["sps"])
        out(f"  peak augmented throughput: {peak['sps']:,.0f} samples/sec "
            f"({peak['cond']}, aug={peak['aug']:,})")

    # ---- VERDICT ----
    out("\n" + "=" * 90)
    out("[5] VERDICT")
    out("=" * 90)
    base_row = rows[0]
    targeted = [r for r in rows if r["n"] > 0]
    best = min(targeted, key=lambda r: r["pgd"])
    robust_gain = base_row["pgd"] - best["pgd"]      # positive => more robust
    vuln_gain = base_row["vpgd"] - best["vpgd"]
    acc_cost = best["acc"] - base_row["acc"]
    acc_kept = best["acc"] >= base_row["acc"] - 0.02

    for r in targeted:
        out(f"  {r['cond']}: PGD_ASR {base_row['pgd']:.4f}->{r['pgd']:.4f} "
            f"({r['pgd']-base_row['pgd']:+.4f}); clean_acc "
            f"{base_row['acc']:.4f}->{r['acc']:.4f} ({r['acc']-base_row['acc']:+.4f}); "
            f"vuln_PGD {base_row['vpgd']:.4f}->{r['vpgd']:.4f} "
            f"({r['vpgd']-base_row['vpgd']:+.4f})")

    out("")
    out(f"  best (lowest PGD_ASR) targeted condition: {best['cond']} (aug={best['aug']:,})")
    out(f"  whole-test PGD robustness gain vs baseline = {robust_gain:+.4f} "
        f"(positive => more robust)")
    out(f"  vulnerable-subset PGD gain vs baseline     = {vuln_gain:+.4f}")
    out(f"  clean-acc change vs baseline               = {acc_cost:+.4f} "
        f"({'KEPT within 0.02' if acc_kept else 'DROPPED >0.02'})")

    if robust_gain > 0.02 and acc_kept:
        one = ("YES: at extreme scale (n=ALL, K>=2000) random-noise multiplicity DOES "
               "recover meaningful whole-test robustness while keeping clean accuracy. "
               "This OVERTURNS H404/H405.")
    elif robust_gain > 0.02 and not acc_kept:
        one = ("PARTIAL: extreme scale recovers some robustness but at a clean-accuracy "
               "cost (>0.02 drop).")
    else:
        one = ("NO: even at the EXTREME (n=ALL training samples, K up to 5000) random "
               "uniform L-inf noise multiplicity gives no whole-test robustness. This "
               "CONFIRMS H404/H405: random-noise hardening is a dead end regardless of "
               "scale; PGD curvature is not addressed by axis-aligned random noise.")
    out("  ONE-LINE VERDICT: " + one)

    # ---- GPU speedup note ----
    out("")
    if aug_rows:
        peak = max(aug_rows, key=lambda r: r["sps"])
        out(f"  GPU speedup: peak {peak['sps']:,.0f} samples/sec with data GPU-resident + "
            f"on-the-fly noise + BATCH={BATCH}.")
        out(f"  Largest condition (aug={max(r['aug'] for r in aug_rows):,}) trained in "
            f"{max(r['wall'] for r in aug_rows):.1f}s; whole sweep in {time.time()-t0:.1f}s.")
        out("  H405's per-batch host->device copies + materialized aug set are eliminated; "
            "memory stays flat (only int index grows).")

    out("")
    out(f"done in {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
