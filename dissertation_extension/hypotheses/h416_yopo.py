"""
H416 - YOPO: You Only Propagate Once (Zhang et al., NeurIPS 2019, arXiv:1905.00877)

YOPO exploits Pontryagin's Maximum Principle to show that the adversarial
perturbation update is only coupled with the FIRST LAYER of the network.
Therefore, during the inner adversarial loop, we only need to backprop through
the first layer rather than the full network, dramatically reducing cost.

YOPO-m-n algorithm:
  - Outer loop: n full forward+backward passes to update network weights (n model updates)
  - Inner loop: m perturbation steps, each backpropping only through the first layer
    (re-using the cached gradient of the first-layer output w.r.t. delta)

Hypothesis: YOPO-m-n achieves comparable robustness to full PGD-AT at matched
wall-clock time (because fewer full backprop passes per effective perturbation step).

Design:
  - SmallCNN split at the first conv block (features[0:4]) vs rest
  - Inner loop: forward only through block0, backprop gradient of loss w.r.t delta
    using chain rule: grad_delta = grad_h0 @ Jacobian(h0, delta)
    where grad_h0 is fixed for the m inner steps (reuse from first computation)
  - Sweep: YOPO-5-3, YOPO-3-5, YOPO-10-3 vs baseline PGD-AT (m=10, n=1 full)
  - Wall-time matched comparison: run each for EPOCHS epochs, report time
  - Eval: clean acc, FGSM_ASR, PGD_ASR

Config: Fashion-MNIST, SmallCNN width=32, N_TRAIN=6000, EPOCHS=10, EPS=0.1,
PGD_STEPS=10, BATCH=128, LR=0.05, SGD mom=0.9 wd=5e-4, SEED=0.
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

# ---- config ---------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = EPS * 2.5 / PGD_STEPS

META = {"channels": 1, "size": 28, "n_classes": 10}

# YOPO sweep: (m, n) pairs — m inner perturbation steps, n outer weight updates per batch
YOPO_CONFIGS = [
    (5, 3),   # YOPO-5-3 (paper default)
    (3, 5),   # YOPO-3-5 (more weight updates)
    (10, 3),  # YOPO-10-3 (more perturbation steps)
]


# ---- YOPO split model -----------------------------------------------------
class YOPOSmallCNN(nn.Module):
    """SmallCNN split into block0 (first conv block) and rest.

    block0: Conv -> BN -> ReLU -> MaxPool  (features[0:4])
    rest:   features[4:] + head

    For YOPO inner loop we only pass through block0 to get gradient w.r.t. delta.
    """

    def __init__(self, in_ch=1, size=28, n_classes=10, width=32):
        super().__init__()
        # First conv block
        self.block0 = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Remaining conv blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        feat = size // 8  # 3 maxpool2d halving
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * feat * feat, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward_full(self, x):
        h = self.block0(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.head(h)

    def forward_from_h0(self, h0):
        """Forward pass starting from block0 output (for YOPO inner loop)."""
        h = self.block1(h0)
        h = self.block2(h)
        return self.head(h)

    def forward(self, x):
        return self.forward_full(x)


def _make_sgd(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


# ---- YOPO adversarial training step ---------------------------------------
def yopo_train_epoch(model, Xtr, Ytr, opt, sched, m, n, eps, alpha):
    """One epoch of YOPO-m-n training.

    Algorithm (per mini-batch):
      1. Sample mini-batch (x, y).
      2. Random-start delta in [-eps, eps].
      3. Forward x+delta through block0 -> h0 (detached from delta for reuse).
         Simultaneously get grad_h0: gradient of loss w.r.t h0 by doing one
         full forward+backward (this counts as 1 of the n outer updates).
      4. Inner loop (m steps): update delta using only block0 backprop:
           grad_delta = autograd(block0(x+delta), delta, grad_output=grad_h0)
         delta <- delta + alpha * sign(grad_delta)
         delta <- clamp(delta, -eps, eps); x+delta in [0,1]
         No re-computation of grad_h0 (fixed across all m inner steps).
      5. After m inner steps, do remaining (n-1) full forward+backward weight updates
         on the final x+delta.

    Total full backprops per batch: n (as opposed to m*n for naive PGD-AT with m*n steps).
    """
    model.train()
    N = Xtr.size(0)
    perm = torch.randperm(N, device=Xtr.device)
    total_loss = 0.0

    for i in range(0, N, BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = Xtr[idx], Ytr[idx]

        # Random start
        delta = torch.empty_like(xb).uniform_(-eps, eps)
        delta.requires_grad_(True)

        # --- Step 3: one full forward+backward to get grad_h0 and do first weight update ---
        xa = (xb + delta).clamp(0, 1)
        h0 = model.block0(xa)             # (B, width, 14, 14)
        logits = model.forward_from_h0(h0)
        loss = F.cross_entropy(logits, yb)

        opt.zero_grad()
        # We need grad w.r.t. delta AND network params simultaneously
        # Use retain_graph to also get delta grad
        loss.backward(retain_graph=False)

        # grad w.r.t. delta from this first step
        if delta.grad is not None:
            delta_grad = delta.grad.detach().clone()
        else:
            # fallback: compute separately (should not happen)
            delta_grad = torch.zeros_like(delta)

        opt.step()   # weight update #1

        # We also need grad_h0 for the inner loop.
        # Re-do block0 forward to get grad_h0 via a fresh computation.
        # This is the "reuse" trick: compute grad_h0 once and fix it for m inner steps.
        delta = delta.detach()
        delta.requires_grad_(False)

        with torch.no_grad():
            xa_cur = (xb + delta).clamp(0, 1)

        # Compute grad_h0: need h0 as a leaf requiring grad
        xa_for_h0 = xa_cur.detach()
        h0_leaf = model.block0(xa_for_h0)
        h0_leaf = h0_leaf.detach().requires_grad_(True)

        logits_for_grad = model.forward_from_h0(h0_leaf)
        loss_for_grad = F.cross_entropy(logits_for_grad, yb)
        loss_for_grad.backward()
        grad_h0 = h0_leaf.grad.detach()   # fixed across m inner steps

        # --- Step 4: m inner perturbation steps using only block0 backprop ---
        delta = delta.detach()
        for _inner in range(m):
            delta.requires_grad_(True)
            xa_inner = (xb + delta).clamp(0, 1)
            h0_inner = model.block0(xa_inner)
            # grad of dot(grad_h0, h0_inner) w.r.t. delta
            # = chain rule: grad_delta = (block0 Jacobian)^T @ grad_h0
            surrogate = (grad_h0 * h0_inner).sum()
            surrogate.backward()
            with torch.no_grad():
                delta_step = delta.grad.sign()
                delta = (delta.detach() + alpha * delta_step)
                delta = torch.min(torch.max(delta, -eps * torch.ones_like(delta)),
                                  eps * torch.ones_like(delta))
                delta = torch.clamp(xb + delta, 0, 1) - xb  # project to [0,1]

        delta = delta.detach()

        # --- Step 5: remaining (n-1) full forward+backward weight updates ---
        for _outer in range(n - 1):
            xa_final = (xb + delta).clamp(0, 1)
            logits_final = model.forward_full(xa_final)
            loss_final = F.cross_entropy(logits_final, yb)
            opt.zero_grad()
            loss_final.backward()
            opt.step()

        total_loss += loss.item()

    sched.step()
    return total_loss


# ---- Standard PGD-AT training epoch (baseline) ----------------------------
def pgdat_train_epoch(model, Xtr, Ytr, opt, sched, pgd_steps, eps, alpha):
    """Full PGD adversarial training epoch (standard, no YOPO shortcut)."""
    model.train()
    N = Xtr.size(0)
    perm = torch.randperm(N, device=Xtr.device)
    total_loss = 0.0

    for i in range(0, N, BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = Xtr[idx], Ytr[idx]
        # PGD inner loop
        xa = xb.clone().detach()
        xa = xa + torch.empty_like(xa).uniform_(-eps, eps)
        xa = xa.clamp(0, 1)
        for _ in range(pgd_steps):
            xa.requires_grad_(True)
            loss_adv = F.cross_entropy(model(xa), yb)
            loss_adv.backward()
            with torch.no_grad():
                xa = xa.detach() + alpha * xa.grad.sign()
                xa = torch.min(torch.max(xa, xb - eps), xb + eps).clamp(0, 1)
        opt.zero_grad()
        logits = model(xa.detach())
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    sched.step()
    return total_loss


# ---- eval helpers ----------------------------------------------------------
def eval_robustness(model, X, Y, label):
    _, acc = C.logits_and_acc(model, X, Y)
    fg = C.attack_success(model, X, Y, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return {"label": label, "clean_acc": acc, "fgsm_asr": fg["asr"], "pgd_asr": pg["asr"]}


# ---- training driver -------------------------------------------------------
def run_yopo(Xtr, Ytr, Xte, Yte, m, n, seed, out):
    label = f"YOPO-{m}-{n}"
    out(f"\n{'='*70}")
    out(f"Training {label}  (m={m} inner steps, n={n} outer weight updates/batch)")
    out(f"{'='*70}")
    C.set_seed(seed)
    model = YOPOSmallCNN(in_ch=1, size=28, n_classes=10, width=32).to(C.DEVICE)
    opt = _make_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    alpha = EPS * 2.5 / max(m, 1)

    t0 = time.time()
    for ep in range(EPOCHS):
        ep_loss = yopo_train_epoch(model, Xtr, Ytr, opt, sched, m, n, EPS, alpha)
        if (ep + 1) % 5 == 0 or ep == 0:
            out(f"  epoch {ep+1}/{EPOCHS}  loss={ep_loss:.3f}  "
                f"elapsed={time.time()-t0:.1f}s")
    wall = time.time() - t0

    model.eval()
    res = eval_robustness(model, Xte, Yte, label)
    res["wall_s"] = wall
    out(f"  {label}: clean_acc={res['clean_acc']:.4f}  "
        f"FGSM_ASR={res['fgsm_asr']:.4f}  PGD_ASR={res['pgd_asr']:.4f}  "
        f"wall={wall:.1f}s")
    return res


def run_pgdat(Xtr, Ytr, Xte, Yte, pgd_steps, seed, out):
    label = f"PGD-AT (steps={pgd_steps})"
    out(f"\n{'='*70}")
    out(f"Training {label}  (full PGD-AT, standard baseline)")
    out(f"{'='*70}")
    C.set_seed(seed)
    model = YOPOSmallCNN(in_ch=1, size=28, n_classes=10, width=32).to(C.DEVICE)
    opt = _make_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    t0 = time.time()
    for ep in range(EPOCHS):
        ep_loss = pgdat_train_epoch(model, Xtr, Ytr, opt, sched, pgd_steps, EPS, PGD_ALPHA)
        if (ep + 1) % 5 == 0 or ep == 0:
            out(f"  epoch {ep+1}/{EPOCHS}  loss={ep_loss:.3f}  "
                f"elapsed={time.time()-t0:.1f}s")
    wall = time.time() - t0

    model.eval()
    res = eval_robustness(model, Xte, Yte, label)
    res["wall_s"] = wall
    out(f"  {label}: clean_acc={res['clean_acc']:.4f}  "
        f"FGSM_ASR={res['fgsm_asr']:.4f}  PGD_ASR={res['pgd_asr']:.4f}  "
        f"wall={wall:.1f}s")
    return res


def run_clean(Xtr, Ytr, Xte, Yte, seed, out):
    """Standard clean training (no adversarial augmentation) — baseline reference."""
    label = "Clean (no AT)"
    out(f"\n{'='*70}")
    out(f"Training {label}")
    out(f"{'='*70}")
    C.set_seed(seed)
    model = YOPOSmallCNN(in_ch=1, size=28, n_classes=10, width=32).to(C.DEVICE)
    opt = _make_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    t0 = time.time()
    N = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(N, device=Xtr.device)
        ep_loss = 0.0
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            out(f"  epoch {ep+1}/{EPOCHS}  loss={ep_loss:.3f}  "
                f"elapsed={time.time()-t0:.1f}s")
    wall = time.time() - t0
    model.eval()
    res = eval_robustness(model, Xte, Yte, label)
    res["wall_s"] = wall
    out(f"  {label}: clean_acc={res['clean_acc']:.4f}  "
        f"FGSM_ASR={res['fgsm_asr']:.4f}  PGD_ASR={res['pgd_asr']:.4f}  "
        f"wall={wall:.1f}s")
    return res


# ---- main ------------------------------------------------------------------
def main():
    t_global = time.time()
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "fashion_mnist", "h416_yopo_output.txt",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H416  YOPO — You Only Propagate Once (Zhang et al., NeurIPS 2019)")
    out("      arXiv:1905.00877 — Accelerating Adversarial Training via Maximal Principle")
    out("=" * 80)
    out(f"config: DS={DS}  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}  EPOCHS={EPOCHS}")
    out(f"        LR={LR}  BATCH={BATCH}  EPS={EPS}  PGD_STEPS={PGD_STEPS}")
    out(f"        SEED={SEED}  device={C.DEVICE}")
    out(f"        YOPO configs (m inner steps, n outer weight updates): {YOPO_CONFIGS}")
    out("")
    out("Key idea: PMP shows adversarial delta is only coupled with first-layer weights.")
    out("  Inner loop backprops only through block0 (first conv block) using fixed grad_h0.")
    out("  YOPO-m-n: m cheap inner steps + n full weight updates vs m*n full backprops.")
    out("")

    # Load data
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    results = []

    # 1. Clean (no AT)
    res = run_clean(Xtr, Ytr, Xte, Yte, SEED, out)
    results.append(res)
    flush_file()

    # 2. Full PGD-AT baseline
    res = run_pgdat(Xtr, Ytr, Xte, Yte, pgd_steps=PGD_STEPS, seed=SEED, out=out)
    results.append(res)
    flush_file()

    # 3. YOPO variants
    for m, n in YOPO_CONFIGS:
        res = run_yopo(Xtr, Ytr, Xte, Yte, m, n, SEED, out)
        results.append(res)
        flush_file()

    # ---- summary table ----
    out("")
    out("=" * 80)
    out("SUMMARY TABLE")
    out("=" * 80)
    hdr = "{:<22} {:>10} {:>10} {:>10} {:>10}".format(
        "method", "clean_acc", "FGSM_ASR", "PGD_ASR", "wall_s"
    )
    out(hdr)
    out("-" * len(hdr))
    for r in results:
        out("{:<22} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.1f}".format(
            r["label"], r["clean_acc"], r["fgsm_asr"], r["pgd_asr"], r["wall_s"]
        ))
    out("-" * len(hdr))

    # ---- wall-time efficiency analysis ----
    out("")
    out("WALL-TIME EFFICIENCY ANALYSIS")
    pgdat_res = next(r for r in results if r["label"].startswith("PGD-AT"))
    out(f"PGD-AT wall time: {pgdat_res['wall_s']:.1f}s  "
        f"PGD_ASR={pgdat_res['pgd_asr']:.4f}")
    for r in results:
        if not r["label"].startswith("YOPO"):
            continue
        speedup = pgdat_res["wall_s"] / max(r["wall_s"], 0.1)
        pgd_delta = r["pgd_asr"] - pgdat_res["pgd_asr"]
        clean_delta = r["clean_acc"] - pgdat_res["clean_acc"]
        out(f"  {r['label']}: wall={r['wall_s']:.1f}s  speedup={speedup:.2f}x  "
            f"ΔPGD_ASR={pgd_delta:+.4f}  Δclean={clean_delta:+.4f} (vs PGD-AT)")

    # ---- verdict ----
    out("")
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)

    yopo_results = [r for r in results if r["label"].startswith("YOPO")]
    best_yopo = min(yopo_results, key=lambda r: r["pgd_asr"])
    fastest_yopo = min(yopo_results, key=lambda r: r["wall_s"])

    pgd_gain = pgdat_res["pgd_asr"] - best_yopo["pgd_asr"]   # positive = YOPO more robust
    speedup_best = pgdat_res["wall_s"] / max(fastest_yopo["wall_s"], 0.1)

    out(f"  Best YOPO (lowest PGD_ASR): {best_yopo['label']}")
    out(f"    PGD_ASR: {pgdat_res['pgd_asr']:.4f} (PGD-AT) -> "
        f"{best_yopo['pgd_asr']:.4f} ({best_yopo['label']})  "
        f"delta={pgd_gain:+.4f} (negative=YOPO worse)")
    out(f"  Fastest YOPO: {fastest_yopo['label']}  speedup={speedup_best:.2f}x vs PGD-AT")

    # Robustness within tolerance
    TOLERANCE = 0.03
    yopo_comparable = [r for r in yopo_results
                       if abs(r["pgd_asr"] - pgdat_res["pgd_asr"]) <= TOLERANCE]
    speedups_comparable = [(pgdat_res["wall_s"] / max(r["wall_s"], 0.1), r["label"])
                           for r in yopo_comparable]

    if yopo_comparable:
        best_sp, best_sp_label = max(speedups_comparable, key=lambda x: x[0])
        if best_sp > 1.2:
            verdict = (f"CONFIRMED: {best_sp_label} achieves comparable robustness "
                       f"(within {TOLERANCE} PGD_ASR) at {best_sp:.2f}x speedup vs PGD-AT. "
                       f"YOPO's first-layer-only backprop acceleration works on SmallCNN/FashionMNIST.")
        else:
            verdict = (f"PARTIAL: {best_sp_label} achieves comparable robustness "
                       f"but only {best_sp:.2f}x speedup — may be dominated by data loading "
                       f"overhead at small N_TRAIN.")
    else:
        # Check if YOPO is faster even if robustness differs
        if any(pgdat_res["wall_s"] / max(r["wall_s"], 0.1) > 1.2 for r in yopo_results):
            verdict = (f"SPEED-ONLY: YOPO is faster but PGD_ASR gap > {TOLERANCE} — "
                       f"first-layer approximation loses robustness at this scale. "
                       f"Larger networks/datasets may close the gap.")
        else:
            verdict = (f"INCONCLUSIVE: YOPO not faster nor more robust than PGD-AT at "
                       f"this small scale (N_TRAIN={N_TRAIN}, SmallCNN). "
                       f"YOPO advantages emerge on larger models (ResNet) and datasets.")

    out(f"  ONE-LINE VERDICT: {verdict}")

    out("")
    out(f"Total elapsed: {time.time() - t_global:.1f}s")
    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
