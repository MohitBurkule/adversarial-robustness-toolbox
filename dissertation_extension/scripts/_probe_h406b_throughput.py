"""Probe real-run config + measured throughput at BATCH=4096 (no checkpoint).

Prints total_steps for the full K=500000 run and measures presentations/sec by
timing N pure train steps (gather + fresh noise + fwd/bwd/step) at batch 4096.
Then estimates how far ~48h gets through the 3e9 target. Writes/reads nothing
persistent.
"""
import os
import sys
import math
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
K = 500000
BATCH = 4096
EPS = 0.1
LR = 0.05
META = {"channels": 1, "size": 28, "n_classes": 10}

dev = C.DEVICE
Xtr, Ytr, _, _ = C.load_dataset(DS, n_train=N_TRAIN, n_eval=2000, seed=0)

total_presentations = N_TRAIN * K
total_steps = math.ceil(total_presentations / BATCH)
print(f"total_presentations = {total_presentations:,} (~{total_presentations/1e9:.3f}e9)")
print(f"total_steps @ batch {BATCH} = {total_steps:,}")

C.set_seed(0)
model = C.build_model("cnn", META, width=32, seed=0)
opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                      lr=LR, momentum=0.9, weight_decay=5e-4)
model.train()
n = Xtr.size(0)

# warmup
for _ in range(10):
    si = torch.randint(n, (BATCH,), device=dev)
    xb = (Xtr[si] + (2 * torch.rand_like(Xtr[si]) - 1) * EPS).clamp(0, 1)
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(xb), Ytr[si]).backward()
    opt.step()
torch.cuda.synchronize()

NSTEP = 200
t0 = time.time()
for _ in range(NSTEP):
    si = torch.randint(n, (BATCH,), device=dev)
    xb = (Xtr[si] + (2 * torch.rand_like(Xtr[si]) - 1) * EPS).clamp(0, 1)
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(xb), Ytr[si]).backward()
    opt.step()
torch.cuda.synchronize()
dt = time.time() - t0

steps_per_sec = NSTEP / dt
pres_per_sec = steps_per_sec * BATCH
print(f"measured: {steps_per_sec:,.1f} steps/sec  ->  {pres_per_sec:,.0f} presentations/sec")
print(f"peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

for hours in (48,):
    done = pres_per_sec * hours * 3600
    print(f"~{hours}h -> {done:,.0f} presentations "
          f"({100*done/total_presentations:.1f}% of {total_presentations/1e9:.2f}e9; "
          f"~{done/total_presentations:.2f}x... effective K reached "
          f"~{done/N_TRAIN:,.0f})")
full_h = total_presentations / pres_per_sec / 3600
print(f"full 3e9 run ETA at this rate: ~{full_h:.1f} h ({full_h/24:.1f} days)")
