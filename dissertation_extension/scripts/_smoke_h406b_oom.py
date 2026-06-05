"""Targeted test of the OOM auto-downsize path in h406b_weekend_maxscale.

We cannot reliably force a real CUDA OOM at small scale, so we monkeypatch the
model's forward to raise torch.cuda.OutOfMemoryError on the FIRST few calls,
then succeed. We assert that:
  * the batch size is halved on OOM (4096 -> 2048 -> ... down toward BATCH_MIN),
  * a checkpoint with the SHRUNKEN batch is written (so a relaunch resumes safe),
  * training then proceeds and progress (global_step) advances past the OOMs.

Run: cd <proj> && SMOKE=1 .venv/bin/python scripts/_smoke_h406b_oom.py
"""
import os
import sys

os.environ["SMOKE"] = "1"
os.environ["K"] = "100"
os.environ["BATCH_START"] = "4096"
os.environ["CKPT_EVERY_SEC"] = "100000"   # disable wall-clock flush noise
os.environ["MAX_SECONDS"] = "0"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results", "fashion_mnist")
CKPT = os.path.join(RES, "h406b_weekend.ckpt")
OUT = os.path.join(RES, "h406b_weekend_maxscale_output.txt")
DONE = os.path.join(RES, "h406b_weekend.DONE")

import torch
import torch.nn.functional as F
import importlib
import hypotheses.h406b_weekend_maxscale as H
importlib.reload(H)

# reset
for f in [CKPT, CKPT + ".tmp", OUT, OUT + ".tmp", DONE]:
    if os.path.exists(f):
        os.remove(f)


def fail(m):
    print("OOM-TEST FAIL:", m)
    sys.exit(1)


# Patch build_model so the returned model raises OOM on its first N forward
# calls, then behaves normally. Track observed batch sizes via checkpoint reads.
_orig_build = H.C.build_model
N_OOM = {"left": 3}
CALLS = {"n": 0}


def patched_build(arch, meta, **kw):
    model = _orig_build(arch, meta, **kw)
    CALLS["n"] += 1
    # call #1 is the baseline (no OOM handling there); only inject OOMs into the
    # augmented training model (call #2).
    if CALLS["n"] < 2:
        return model
    real_forward = model.forward

    def forward(x):
        if N_OOM["left"] > 0:
            N_OOM["left"] -= 1
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return real_forward(x)

    model.forward = forward
    return model


H.C.build_model = patched_build

# We also want the run to STOP quickly after surviving the OOMs so we can inspect
# the shrunken batch. Patch MAX_SECONDS via re-reading: set module constant.
H.MAX_SECONDS = 5.0  # stop cleanly after 5s without .DONE

print("=" * 70)
print("OOM-TEST: BATCH_START=4096, forcing 3 OOMs at the start")
print("=" * 70)
try:
    H.main()
except SystemExit:
    raise
except Exception as e:
    # main() may finish or hit MAX_SECONDS stop; either is fine. Re-raise real bugs.
    fail(f"unexpected exception: {type(e).__name__}: {e}")

if not os.path.exists(CKPT):
    fail("no checkpoint written after OOM path")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
batch = ck["batch"]
step = ck["global_step"]
print(f"[result] after OOM path: checkpoint batch={batch} global_step={step}")

# 3 OOMs from 4096 -> 2048 -> 1024 -> 512 (each OOM halves; the 3rd retry at 512
# succeeds because N_OOM is exhausted). So we expect batch <= 512 and < 4096.
if batch >= H.BATCH_START:
    fail(f"batch did not shrink (still {batch} >= {H.BATCH_START})")
if batch < H.BATCH_MIN:
    fail(f"batch shrank below BATCH_MIN ({batch} < {H.BATCH_MIN})")
print(f"[assert] batch shrank {H.BATCH_START} -> {batch} on OOM (PASS)")
print(f"[assert] checkpoint persisted shrunken batch for safe resume (PASS)")
if step < 1:
    fail("no training progress after surviving OOMs")
print(f"[assert] training advanced past OOMs: global_step={step} (PASS)")

# cleanup so the OOM test does not leave a bad checkpoint behind
for f in [CKPT, CKPT + ".tmp", OUT, OUT + ".tmp", DONE]:
    if os.path.exists(f):
        os.remove(f)
print("[cleanup] removed OOM-test artifacts")
print("\nOOM-TEST PASS")
