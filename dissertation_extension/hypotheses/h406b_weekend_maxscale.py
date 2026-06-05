"""
H406b - WEEKEND EXTREME-SCALE random-noise hardening of ALL training samples.

This is the long, robust, weekend-spanning sibling of h406. H404 (K=50),
H405 (K=1000) and H406 (K up to 5000) all found that hardening adversarially
vulnerable TRAINING samples with random-uniform L-inf noise copies gives
essentially NO whole-test robustness. This run pushes the idea to the absolute
extreme to settle the question of scale once and for all:

  ONE condition: n = ALL 6000 training samples, K = 500,000 noisy copies per
  sample  ->  6000 * 500000 = 3.0e9 effective noisy presentations.
  Plus a baseline (n=0) reference trained once at the start.

Noise (identical to h406): xnoisy = (x + 0.1*(2*rand_like(x)-1)).clamp(0,1),
labeled with the true class. Fashion-MNIST, SmallCNN width=32, eps=0.1 eval
(clean_acc, FGSM_ASR, PGD_ASR, vuln-subset PGD_ASR).

MEMORY: we never build a 3-billion-length index. Training is implemented as
random sampling: each step draws a random batch of indices uniformly from the
6000 originals, gathers, adds FRESH uniform noise on the GPU, and trains. With
total_steps = ceil(6000 * K / batch) each original is presented ~K times in
expectation while memory stays flat (only the 6000 originals live on GPU).

ROBUSTNESS (this is a multi-day unattended run):
  * OOM-safe batch auto-tuning: starts at BATCH_START (4096); on a CUDA OOM the
    step is retried with batch halved (floor BATCH_MIN=128); the safe batch is
    persisted into the checkpoint so resume uses it.
  * Single OVERWRITING checkpoint (storage-limited): {model, optimizer, RNG
    states, global_step, current_batch, accumulated metrics} -> ONE .ckpt file,
    written atomically (tmp + os.replace) every CKPT_EVERY_SEC (wall clock) AND
    at every K-fraction milestone. Never keeps multiple checkpoints.
  * Auto-resume: on startup, if the checkpoint exists, load it and continue from
    global_step with the saved batch. Otherwise start fresh (computing the
    baseline reference first and storing it in the metrics).
  * Periodic result flushing: every checkpoint interval a quick eval is run and
    the FULL human-readable running report is rewritten to the output .txt so
    the latest snapshot survives a total process death.
  * On clean completion writes a final summary block and touches a .DONE flag.

SMOKE=1 overrides K to a tiny value (SMOKE_K, default 2000) and shrinks the
checkpoint interval so the full lifecycle (train -> checkpoint -> resume ->
flush -> done) can be exercised in seconds. Useful env overrides:
  SMOKE=1            tiny K, fast checkpoints (smoke test)
  K=<int>            override copies-per-sample
  BATCH_START=<int>  override starting batch
  CKPT_EVERY_SEC=<n> override checkpoint/flush wall-clock interval
  MAX_SECONDS=<n>    stop cleanly after n seconds WITHOUT touching .DONE
                     (so a later relaunch resumes) -- used by the smoke test.
"""
import os
import sys
import time
import math

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config --------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
LR = 0.05
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
VULN_TEST_FRAC = 0.25

# baseline trained with a bounded step budget identical in spirit to h406 so its
# clean accuracy is a fair reference (not part of the 3e9 sampling run).
BASELINE_EPOCHS = 10
BASELINE_BATCH = 512
BASELINE_STEPS_PER_EPOCH = 1024

SMOKE = os.environ.get("SMOKE", "0") == "1"

# extreme-scale settings
K = int(os.environ.get("K", "2000" if SMOKE else "4000000"))
BATCH_START = int(os.environ.get("BATCH_START", "256" if SMOKE else "4096"))
BATCH_MIN = 128
# checkpoint/flush wall-clock interval (seconds)
CKPT_EVERY_SEC = float(os.environ.get("CKPT_EVERY_SEC", "2" if SMOKE else "300"))
# optional clean stop after N seconds WITHOUT writing .DONE (resume-test hook)
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "0"))  # 0 == run to completion
# number of K-fraction milestones at which we force a checkpoint+flush
N_MILESTONES = 100

META = {"channels": 1, "size": 28, "n_classes": 10}

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist")
CKPT_PATH = os.path.join(_RESULTS_DIR, "h406b_weekend.ckpt")
OUT_PATH = os.path.join(_RESULTS_DIR, "h406b_weekend_maxscale_output.txt")
DONE_PATH = os.path.join(_RESULTS_DIR, "h406b_weekend.DONE")


def _make_optimizer_sgd(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


# ---------------------------------------------------------------------------
# baseline (no augmentation) -- bounded step budget, run once.
# ---------------------------------------------------------------------------
def train_baseline(Xtr, Ytr, seed):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BASELINE_EPOCHS)
    dev = Xtr.device
    n = Xtr.size(0)
    model.train()
    for ep in range(BASELINE_EPOCHS):
        draw = torch.randint(n, (BASELINE_STEPS_PER_EPOCH * BASELINE_BATCH,), device=dev)
        for s in range(BASELINE_STEPS_PER_EPOCH):
            si = draw[s * BASELINE_BATCH:(s + 1) * BASELINE_BATCH]
            xb, yb = Xtr[si], Ytr[si]
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_robustness(model, X, Y):
    _, acc = C.logits_and_acc(model, X, Y)
    fg = C.attack_success(model, X, Y, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


def eval_subset_pgd(model, X, Y, mask):
    pg = C.attack_success(model, X[mask], Y[mask], attack="pgd", eps=EPS, steps=PGD_STEPS)
    return pg["asr"]


# ---------------------------------------------------------------------------
# checkpoint I/O (single overwriting file, atomic)
# ---------------------------------------------------------------------------
def save_checkpoint(state):
    tmp = CKPT_PATH + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, CKPT_PATH)


def rng_state_dump(dev):
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
        "numpy": np.random.get_state(),
    }


def rng_state_load(rng, dev):
    if rng is None:
        return
    # torch.load(map_location=cuda) may have moved the saved ByteTensor RNG states
    # onto the GPU and/or changed dtype; force them back to CPU ByteTensors.
    cpu_state = rng["torch_cpu"]
    if not torch.is_tensor(cpu_state):
        cpu_state = torch.as_tensor(cpu_state)
    torch.set_rng_state(cpu_state.to("cpu", torch.uint8))
    if dev.type == "cuda" and rng.get("torch_cuda") is not None:
        cuda_states = [s.to("cpu", torch.uint8) for s in rng["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])


# ---------------------------------------------------------------------------
# report flushing -- rewrite the whole human-readable file each time.
# ---------------------------------------------------------------------------
def write_report(meta, snapshots, final=False):
    """meta: dict of run-level info. snapshots: list of per-flush metric dicts."""
    L = []
    def w(s=""):
        L.append(s)
    w("=" * 96)
    w("H406b  WEEKEND EXTREME-SCALE random-noise hardening of ALL train samples (Fashion-MNIST)")
    w("=" * 96)
    w(f"condition: n = ALL {N_TRAIN} train samples,  K = {meta['K']:,} noisy copies/sample")
    w(f"           effective presentations target = {meta['total_presentations']:,} (~{meta['total_presentations']/1e9:.3f}e9)")
    w(f"           total_steps (at batch {meta['batch_ref']}) = {meta['total_steps']:,}")
    w(f"noise:     (x + {EPS}*(2*rand-1)).clamp(0,1), labeled true class  |  EPS={EPS} PGD_STEPS={PGD_STEPS}")
    w(f"config:    SmallCNN width=32, SGD(lr={LR},mom=0.9,wd=5e-4), SEED={SEED}, device={meta['device']}")
    if meta.get("gpu"):
        w(f"           gpu = {meta['gpu']}")
    w(f"           BATCH_START={meta['batch_start']} BATCH_MIN={BATCH_MIN}  CKPT_EVERY_SEC={meta['ckpt_every']}")
    w("")
    w("BASELINE reference (n=0, no augmentation):")
    b = meta["baseline"]
    w(f"   clean_acc={b['acc']:.4f}  FGSM_ASR={b['fgsm']:.4f}  PGD_ASR={b['pgd']:.4f}  "
      f"vuln_PGD={b['vpgd']:.4f}")
    w(f"   vulnerable test subset = lowest {int(VULN_TEST_FRAC*100)}% margin "
      f"({meta['n_vuln_te']} samples)")
    w("")
    w("PROGRESS SNAPSHOTS (each = a periodic eval during the long run):")
    hdr = ("{:>14} {:>8} {:>7} {:>10} {:>10} {:>10} {:>10} {:>11}".format(
        "global_step", "pct", "batch", "clean_acc", "FGSM_ASR", "PGD_ASR",
        "vuln_PGD", "elapsed_s"))
    w(hdr)
    w("-" * len(hdr))
    for s in snapshots:
        w("{:>14,} {:>7.2f}% {:>7} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>11.1f}".format(
            s["step"], s["pct"], s["batch"], s["acc"], s["fgsm"], s["pgd"],
            s["vpgd"], s["elapsed"]))
    w("-" * len(hdr))
    if snapshots:
        last = snapshots[-1]
        w("")
        w(f"latest: step {last['step']:,}/{meta['total_steps']:,} "
          f"({last['pct']:.2f}%)  batch={last['batch']}  "
          f"clean_acc={last['acc']:.4f}  PGD_ASR={last['pgd']:.4f}  "
          f"d_PGD_ASR={last['pgd']-b['pgd']:+.4f}  d_clean={last['acc']-b['acc']:+.4f}")
        if last["elapsed"] > 0 and last["step"] > 0:
            sps = last["step"] * last["batch"] / last["elapsed"]
            w(f"        throughput ~{sps:,.0f} presentations/sec (instantaneous est.)")
    if final:
        w("")
        w("=" * 96)
        w("[FINAL] run completed")
        w("=" * 96)
        last = snapshots[-1]
        robust_gain = b["pgd"] - last["pgd"]
        vuln_gain = b["vpgd"] - last["vpgd"]
        acc_cost = last["acc"] - b["acc"]
        acc_kept = last["acc"] >= b["acc"] - 0.02
        w(f"  final clean_acc={last['acc']:.4f}  FGSM_ASR={last['fgsm']:.4f}  "
          f"PGD_ASR={last['pgd']:.4f}  vuln_PGD={last['vpgd']:.4f}")
        w(f"  whole-test PGD robustness gain vs baseline = {robust_gain:+.4f} "
          f"(positive => more robust)")
        w(f"  vulnerable-subset PGD gain vs baseline     = {vuln_gain:+.4f}")
        w(f"  clean-acc change vs baseline               = {acc_cost:+.4f} "
          f"({'KEPT within 0.02' if acc_kept else 'DROPPED >0.02'})")
        if robust_gain > 0.02 and acc_kept:
            one = ("YES: at K=%d (3e9-scale) random-noise multiplicity on ALL samples DOES "
                   "recover meaningful whole-test robustness while keeping clean accuracy. "
                   "OVERTURNS H404/H405/H406." % meta["K"])
        elif robust_gain > 0.02 and not acc_kept:
            one = "PARTIAL: extreme scale recovers some robustness but at a clean-accuracy cost (>0.02 drop)."
        else:
            one = ("NO: even at the absolute EXTREME (n=ALL, K=%d, ~3e9 presentations) random "
                   "uniform L-inf noise gives no whole-test robustness. CONFIRMS H404/H405/H406: "
                   "random-noise hardening is a dead end regardless of scale; PGD curvature is "
                   "not addressed by axis-aligned random noise." % meta["K"])
        w("  ONE-LINE VERDICT: " + one)
    w("")
    w(f"(report written {time.strftime('%Y-%m-%d %H:%M:%S')})")
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(L) + "\n")
    os.replace(tmp, OUT_PATH)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    t_wall0 = time.time()
    dev = C.DEVICE

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    total_presentations = N_TRAIN * K
    # total_steps defined against the START batch (presentations are batch-invariant;
    # this is just the reference count printed in the report).
    total_steps = math.ceil(total_presentations / BATCH_START)
    milestone_every = max(1, total_steps // N_MILESTONES)

    run_meta = {
        "K": K,
        "batch_start": BATCH_START,
        "batch_ref": BATCH_START,
        "total_presentations": total_presentations,
        "total_steps": total_steps,
        "ckpt_every": CKPT_EVERY_SEC,
        "device": str(dev),
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
    }

    # ---- resume or fresh start --------------------------------------------
    snapshots = []
    if os.path.exists(CKPT_PATH):
        print(f"[resume] loading checkpoint {CKPT_PATH}", flush=True)
        ck = torch.load(CKPT_PATH, map_location=dev, weights_only=False)
        model = C.build_model("cnn", META, width=32, seed=SEED)
        model.load_state_dict(ck["model"])
        opt = _make_optimizer_sgd(model, LR)
        opt.load_state_dict(ck["opt"])
        global_step = ck["global_step"]
        batch = ck["batch"]
        baseline = ck["baseline"]
        vuln_te_mask = ck["vuln_te_mask"]
        snapshots = ck.get("snapshots", [])
        rng_state_load(ck.get("rng"), dev)
        print(f"[resume] global_step={global_step:,} batch={batch} "
              f"({len(snapshots)} prior snapshots)", flush=True)
    else:
        print("[fresh] no checkpoint -> computing baseline reference first", flush=True)
        base_model = train_baseline(Xtr, Ytr, SEED)
        b_acc, b_fgsm, b_pgd = eval_robustness(base_model, Xte, Yte)
        # vulnerable test subset by base-model input-space margin
        te_margin = C.margin(base_model, Xte, Yte)
        n_vuln_te = int(round(VULN_TEST_FRAC * Xte.size(0)))
        vuln_idx = np.argsort(te_margin)[:n_vuln_te]
        vuln_te_mask = np.zeros(Xte.size(0), dtype=bool)
        vuln_te_mask[vuln_idx] = True
        b_vpgd = eval_subset_pgd(base_model, Xte, Yte, vuln_te_mask)
        baseline = {"acc": b_acc, "fgsm": b_fgsm, "pgd": b_pgd, "vpgd": b_vpgd}
        run_meta["n_vuln_te"] = n_vuln_te
        print(f"[fresh] baseline clean_acc={b_acc:.4f} FGSM_ASR={b_fgsm:.4f} "
              f"PGD_ASR={b_pgd:.4f} vuln_PGD={b_vpgd:.4f}", flush=True)
        del base_model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        # the augmented model trains from scratch (seed=SEED) on the noisy stream
        C.set_seed(SEED)
        model = C.build_model("cnn", META, width=32, seed=SEED)
        opt = _make_optimizer_sgd(model, LR)
        global_step = 0
        batch = BATCH_START

    run_meta["baseline"] = baseline
    run_meta["n_vuln_te"] = int(vuln_te_mask.sum())

    # cosine LR schedule over the whole long run, recomputed from global_step so
    # resume picks up the right LR (LambdaLR-free: set lr each step).
    def lr_at(step):
        frac = min(1.0, step / max(1, total_steps))
        return 0.5 * LR * (1.0 + math.cos(math.pi * frac))

    # ---- training loop ----------------------------------------------------
    model.train()
    last_ckpt_t = time.time()
    n = Xtr.size(0)

    def do_eval_and_flush():
        model.eval()
        acc, fgsm, pgd = eval_robustness(model, Xte, Yte)
        vpgd = eval_subset_pgd(model, Xte, Yte, vuln_te_mask)
        pct = 100.0 * global_step / max(1, total_steps)
        snap = {"step": global_step, "pct": pct, "batch": batch, "acc": acc,
                "fgsm": fgsm, "pgd": pgd, "vpgd": vpgd,
                "elapsed": time.time() - t_wall0}
        snapshots.append(snap)
        write_report(run_meta, snapshots, final=False)
        model.train()
        return snap

    def checkpoint():
        save_checkpoint({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "global_step": global_step,
            "batch": batch,
            "baseline": baseline,
            "vuln_te_mask": vuln_te_mask,
            "snapshots": snapshots,
            "rng": rng_state_dump(dev),
        })

    print(f"[train] starting/resuming at step {global_step:,}/{total_steps:,} "
          f"batch={batch}", flush=True)

    while global_step < total_steps:
        # ---- one optimizer step with OOM-safe auto-downsize ----
        ok = False
        while not ok:
            try:
                si = torch.randint(n, (batch,), device=dev)
                xb = Xtr[si]
                yb = Ytr[si]
                noise = (2.0 * torch.rand_like(xb) - 1.0) * EPS
                xb = (xb + noise).clamp(0, 1)
                for g in opt.param_groups:
                    g["lr"] = lr_at(global_step)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()
                ok = True
            except torch.cuda.OutOfMemoryError:
                ok = False
                if batch <= BATCH_MIN:
                    print(f"[OOM] at batch={batch} (== BATCH_MIN); cannot shrink "
                          f"further. retrying after empty_cache.", flush=True)
                    torch.cuda.empty_cache()
                    time.sleep(1.0)
                    continue
                new_batch = max(BATCH_MIN, batch // 2)
                print(f"[OOM] batch {batch} -> {new_batch}; empty_cache + checkpoint",
                      flush=True)
                batch = new_batch
                torch.cuda.empty_cache()
                checkpoint()  # persist safe batch so a relaunch uses it
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    ok = False
                    if batch <= BATCH_MIN:
                        torch.cuda.empty_cache()
                        time.sleep(1.0)
                        continue
                    new_batch = max(BATCH_MIN, batch // 2)
                    print(f"[OOM-rt] batch {batch} -> {new_batch}", flush=True)
                    batch = new_batch
                    torch.cuda.empty_cache()
                    checkpoint()
                else:
                    raise

        global_step += 1

        # ---- periodic checkpoint + flush (wall clock OR milestone) ----
        now = time.time()
        is_milestone = (global_step % milestone_every == 0)
        if (now - last_ckpt_t >= CKPT_EVERY_SEC) or is_milestone or global_step >= total_steps:
            snap = do_eval_and_flush()
            checkpoint()
            last_ckpt_t = now
            print(f"[ckpt] step {global_step:,}/{total_steps:,} "
                  f"({snap['pct']:.2f}%) batch={batch} acc={snap['acc']:.4f} "
                  f"PGD_ASR={snap['pgd']:.4f} elapsed={snap['elapsed']:.0f}s",
                  flush=True)

        # ---- optional clean stop WITHOUT .DONE (resume-test hook) ----
        if MAX_SECONDS > 0 and (time.time() - t_wall0) >= MAX_SECONDS \
                and global_step < total_steps:
            print(f"[stop] MAX_SECONDS={MAX_SECONDS} reached at step "
                  f"{global_step:,}; checkpointing and exiting (NO .DONE) so a "
                  f"relaunch resumes.", flush=True)
            do_eval_and_flush()
            checkpoint()
            return

    # ---- clean completion -------------------------------------------------
    if not snapshots or snapshots[-1]["step"] != global_step:
        do_eval_and_flush()
    checkpoint()
    write_report(run_meta, snapshots, final=True)
    with open(DONE_PATH, "w") as f:
        f.write(f"done {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"step={global_step} elapsed={time.time()-t_wall0:.1f}s\n")
    print(f"[done] completed {global_step:,} steps in "
          f"{time.time()-t_wall0:.1f}s -> {DONE_PATH}", flush=True)


if __name__ == "__main__":
    main()
