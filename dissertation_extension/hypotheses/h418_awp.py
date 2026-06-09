"""
H418 - Adversarial Weight Perturbation (AWP) on PGD-AT (Wu et al., 2020).

Reference: Wu, D., Xia, S.-T., & Wang, Y. (2020). Adversarial weight perturbation
helps robust generalisation. arXiv:2004.05884.

AWP augments PGD adversarial training: at each mini-batch step, after computing
adversarial examples, the model weights are *perturbed* in the direction that
maximises adversarial loss (scaled by gamma * ||w||), the loss is computed under
perturbed weights, then weights are restored before the actual gradient update.
This flattens the adversarial loss landscape around weight space and improves
robust generalisation.

Sweep: gamma_awp ∈ {0, 0.005, 0.01}
  gamma_awp = 0 → standard PGD-AT (no weight perturbation)
  gamma_awp > 0 → AWP on top of PGD-AT

Config: Fashion-MNIST, SmallCNN width=32, N_TRAIN=6000, EPOCHS=10, BATCH=128,
        EPS=0.1, PGD_STEPS=10 (inner), SEED=0.
Eval: clean accuracy, FGSM_ASR, PGD_ASR on 2000 test samples.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config ------------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 2.5 * EPS / PGD_STEPS   # standard 2.5*eps/steps rule
GAMMAS = [0, 0.005, 0.01]           # AWP strength sweep
META = {"channels": 1, "size": 28, "n_classes": 10}


# ---- AWP weight perturbation helpers ----------------------------------------

def _weight_norm(model):
    """Frobenius norm of all trainable parameters (flattened)."""
    with torch.no_grad():
        total = sum(p.norm() ** 2 for p in model.parameters() if p.requires_grad)
    return total.sqrt()


def _perturb_weights(model, delta_dict):
    """Add delta to each parameter in-place."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and name in delta_dict:
                p.add_(delta_dict[name])


def _restore_weights(model, delta_dict):
    """Subtract delta from each parameter in-place (restore)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and name in delta_dict:
                p.sub_(delta_dict[name])


def compute_awp_delta(model, xadv, y, gamma):
    """Compute the AWP weight perturbation delta_w = gamma * ||w|| * grad_w / ||grad_w||.

    Steps (following Wu 2020 §3):
      1. Forward + backward on adversarial loss to get d_loss / d_w.
      2. Normalise gradient to unit norm.
      3. Scale by gamma * ||w||.
    Returns a dict {name: delta_tensor} without modifying model.
    """
    model.zero_grad()
    loss = F.cross_entropy(model(xadv), y)
    loss.backward()

    with torch.no_grad():
        # collect gradients; compute their joint norm
        grads = {name: p.grad.clone()
                 for name, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None}

        g_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads.values())) + 1e-12
        w_norm = _weight_norm(model)
        scale = gamma * w_norm / g_norm

        delta = {name: scale * g for name, g in grads.items()}
    model.zero_grad()
    return delta


# ---- training ----------------------------------------------------------------

def train_pgd_awp(Xtr, Ytr, gamma, seed):
    """PGD-AT + AWP (gamma=0 → standard PGD-AT).

    For each mini-batch:
      1. Generate adversarial examples x_adv via PGD (inner attack).
      2. If gamma > 0: compute AWP delta, perturb weights.
      3. Compute loss on x_adv under (perturbed) weights; backward.
      4. Restore weights; apply optimiser step to original (restored) weights.
    """
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            # --- step 1: generate adversarial examples (PGD inner attack) ---
            model.eval()   # BN in eval for stable adversarial generation
            xadv = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            model.train()

            # --- step 2 (AWP): perturb weights ---
            if gamma > 0:
                delta = compute_awp_delta(model, xadv, yb, gamma)
                _perturb_weights(model, delta)

            # --- step 3: compute loss under (perturbed) weights ---
            opt.zero_grad()
            loss = F.cross_entropy(model(xadv), yb)
            loss.backward()

            # --- step 4: restore weights, then update ---
            if gamma > 0:
                _restore_weights(model, delta)

            opt.step()
        sched.step()

    model.eval()
    return model


# ---- eval --------------------------------------------------------------------

def eval_robustness(model, Xte, Yte):
    """Return (clean_acc, fgsm_asr, pgd_asr)."""
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


# ---- main --------------------------------------------------------------------

def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "fashion_mnist", "h418_awp_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H418  Adversarial Weight Perturbation (AWP) on PGD-AT  (Fashion-MNIST)")
    out("Reference: Wu et al. 2020, arXiv:2004.05884")
    out("=" * 80)
    out(f"config: DS={DS}  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}  EPOCHS={EPOCHS}")
    out(f"        BATCH={BATCH}  LR={LR}  SGD(mom=0.9,wd=1e-4)  SEED={SEED}")
    out(f"        EPS={EPS}  PGD_STEPS={PGD_STEPS}  PGD_ALPHA={PGD_ALPHA:.4f}")
    out(f"        gamma_awp sweep = {GAMMAS}  (0 = standard PGD-AT)")
    out(f"        device = {C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    rows = []
    for gamma in GAMMAS:
        label = "PGD-AT (baseline)" if gamma == 0 else f"AWP gamma={gamma}"
        out("-" * 60)
        out(f"[training]  {label} ...")
        t1 = time.time()
        model = train_pgd_awp(Xtr, Ytr, gamma, SEED)
        acc, fgsm_asr, pgd_asr = eval_robustness(model, Xte, Yte)
        elapsed = time.time() - t1
        out(f"  clean_acc={acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  PGD_ASR={pgd_asr:.4f}"
            f"  ({elapsed:.0f}s)")
        rows.append({"gamma": gamma, "label": label,
                     "acc": acc, "fgsm": fgsm_asr, "pgd": pgd_asr})
        flush_file()

    # ---- MAIN TABLE ----------------------------------------------------------
    out("")
    out("=" * 80)
    out("MAIN TABLE")
    out("=" * 80)
    hdr = "{:<26} {:>7} {:>10} {:>9} {:>9}".format(
        "condition", "gamma", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    baseline = rows[0]
    for r in rows:
        out("{:<26} {:>7} {:>10.4f} {:>9.4f} {:>9.4f}".format(
            r["label"], r["gamma"], r["acc"], r["fgsm"], r["pgd"]))
    out("-" * len(hdr))

    # ---- DELTAS vs baseline --------------------------------------------------
    out("")
    out("Deltas vs PGD-AT baseline (gamma=0):")
    for r in rows[1:]:
        d_acc = r["acc"] - baseline["acc"]
        d_fgsm = r["fgsm"] - baseline["fgsm"]
        d_pgd = r["pgd"] - baseline["pgd"]
        out(f"  gamma={r['gamma']}: d_clean={d_acc:+.4f}  "
            f"d_FGSM_ASR={d_fgsm:+.4f}  d_PGD_ASR={d_pgd:+.4f}")

    # ---- VERDICT -------------------------------------------------------------
    out("")
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)

    awp_rows = rows[1:]
    best = min(awp_rows, key=lambda r: r["pgd"])
    robust_gain = baseline["pgd"] - best["pgd"]   # positive = more robust
    acc_cost = best["acc"] - baseline["acc"]       # negative = acc dropped
    acc_kept = best["acc"] >= baseline["acc"] - 0.02

    out(f"  best AWP condition: gamma={best['gamma']}")
    out(f"  PGD_ASR:  {baseline['pgd']:.4f} -> {best['pgd']:.4f}  "
        f"(gain={robust_gain:+.4f}, positive => more robust)")
    out(f"  FGSM_ASR: {baseline['fgsm']:.4f} -> {best['fgsm']:.4f}  "
        f"(delta={best['fgsm']-baseline['fgsm']:+.4f})")
    out(f"  clean_acc: {baseline['acc']:.4f} -> {best['acc']:.4f}  "
        f"(delta={acc_cost:+.4f}, "
        f"{'kept within 0.02' if acc_kept else 'dropped >0.02'})")

    if robust_gain > 0.02 and acc_kept:
        verdict = ("YES: AWP materially reduces PGD-ASR over PGD-AT while preserving "
                   "clean accuracy, confirming Wu 2020's finding on this small-scale setting.")
    elif robust_gain > 0.02 and not acc_kept:
        verdict = ("PARTIAL: AWP reduces PGD-ASR but at a clean-accuracy cost > 0.02; "
                   "trade-off depends on application.")
    elif robust_gain > 0 and acc_kept:
        verdict = ("WEAK: AWP shows marginal PGD-ASR reduction (<= 0.02) with clean acc "
                   "preserved; effect may not be practically significant at this scale.")
    else:
        verdict = ("NO: AWP does not materially improve over PGD-AT at this scale / "
                   "gamma sweep; possible that small N_TRAIN limits the benefit.")
    out("")
    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"total elapsed: {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
