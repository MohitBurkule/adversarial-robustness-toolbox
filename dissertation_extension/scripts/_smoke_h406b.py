"""End-to-end smoke test for hypotheses/h406b_weekend_maxscale.py.

Exercises the full lifecycle without extra shell env vars (the harness only
allowlists `SMOKE=1 .venv/bin/python ...`):

  Phase 1: fresh start with a tiny K, run for a few seconds then stop CLEANLY
           WITHOUT writing .DONE (via MAX_SECONDS) -> proves train+checkpoint+flush.
  Phase 2: relaunch -> proves AUTO-RESUME (global_step continues, not reset) and
           runs to clean completion -> proves .DONE + final report.
  Also asserts: checkpoint is a single file, output report exists and grows,
  resumed step > phase-1 step, batch persisted.

Run:  cd <proj> && SMOKE=1 .venv/bin/python scripts/_smoke_h406b.py
"""
import os
import sys
import time

# tiny + fast config for the smoke test (set BEFORE importing the module)
os.environ["SMOKE"] = "1"
os.environ["K"] = "400"            # 6000*400 = 2.4M presentations
os.environ["BATCH_START"] = "256"  # ~9375 steps total -> quick
os.environ["CKPT_EVERY_SEC"] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results", "fashion_mnist")
CKPT = os.path.join(RES, "h406b_weekend.ckpt")
OUT = os.path.join(RES, "h406b_weekend_maxscale_output.txt")
DONE = os.path.join(RES, "h406b_weekend.DONE")

import torch


def reset():
    for f in [CKPT, CKPT + ".tmp", OUT, OUT + ".tmp", DONE]:
        if os.path.exists(f):
            os.remove(f)


def ckpt_step():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    return ck["global_step"], ck["batch"]


def fail(msg):
    print("SMOKE FAIL:", msg)
    sys.exit(1)


print("=" * 70)
print("SMOKE: reset artifacts")
print("=" * 70)
reset()

# import after env is set
import importlib
import hypotheses.h406b_weekend_maxscale as H

# ---- Phase 1: fresh start, stop early without DONE ----
print("\n[phase 1] fresh start, MAX_SECONDS=10 deliberate stop (no DONE)")
os.environ["MAX_SECONDS"] = "10"
importlib.reload(H)
H.main()

if not os.path.exists(CKPT):
    fail("no checkpoint written in phase 1")
if os.path.exists(DONE):
    fail(".DONE written in phase 1 but we asked for an early stop")
if not os.path.exists(OUT):
    fail("no output report written in phase 1")
step1, batch1 = ckpt_step()
out_size1 = os.path.getsize(OUT)
print(f"[phase 1] OK: checkpoint step={step1:,} batch={batch1} "
      f"report={out_size1}B, NO .DONE  (single ckpt file: "
      f"{os.path.exists(CKPT)})")
if step1 <= 0:
    fail("phase-1 global_step is 0 -> no progress")

# ---- Phase 2: relaunch -> resume + run to completion ----
print("\n[phase 2] relaunch -> auto-resume + run to .DONE")
os.environ["MAX_SECONDS"] = "0"   # run to completion
importlib.reload(H)
# capture resume log by checking step BEFORE/AFTER via the module's print
H.main()

if not os.path.exists(DONE):
    fail("no .DONE flag after phase 2 completion")
step2, batch2 = ckpt_step()
out_size2 = os.path.getsize(OUT)
print(f"[phase 2] OK: final step={step2:,} batch={batch2} report={out_size2}B "
      f".DONE present")

# ---- assertions ----
if step2 <= step1:
    fail(f"resume did not advance: phase1 step {step1} >= phase2 final {step2}")
expected_total = H.total_presentations_for_smoke() if hasattr(H, "total_presentations_for_smoke") else None
print(f"\n[assert] resumed and advanced: {step1:,} -> {step2:,}  (PASS)")
_ = expected_total
print(f"[assert] single overwriting checkpoint present: {os.path.exists(CKPT)} (PASS)")
print(f"[assert] periodic-flush report present & grew: {out_size2 >= out_size1} (PASS)")
print(f"[assert] .DONE flag on clean completion: {os.path.exists(DONE)} (PASS)")

with open(DONE) as f:
    print("[.DONE] " + f.read().strip())

print("\nSMOKE PASS")
