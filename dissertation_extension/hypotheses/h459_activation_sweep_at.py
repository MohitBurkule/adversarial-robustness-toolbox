"""
H459: Activation Sweep under Adversarial Training.

Gap (CAMPAIGN_GAP_MAP.md, G4): activation choice beyond H366 (SiLU only,
standalone) is unexplored under AT. H366 found ReLU vs SiLU NULL under
*standard* training. The conjecture for H459 is: PGD inner-max sees
input-gradients that depend on activation smoothness; under PGD-AT the
*outer*-min loss is taken at adv points where ReLU's piecewise-linear
gradient is informative but unstable, whereas smooth activations
(SiLU/GELU/Mish/Softplus) provide better-behaved gradient flow to the
attacker (potentially harder to defend) AND to the defender (smoother
loss landscape). Two ways this could go:

  (i)  Smooth helps AT: Gowal 2020 "Uncovering the Limits of AT" (DeepMind)
       and Xie 2020 "Smooth Adversarial Training" both report ~+1.5 to +2.5
       pp robust accuracy on CIFAR-10/ImageNet when ReLU is replaced by a
       smooth activation under AT (Xie ParamSoftplus, SiLU, GELU).
  (ii) Smooth hurts AT: smoother gradients give the inner-max a cleaner
       direction, so PGD finds stronger advs => same outer model can be
       *more* attacked at eval. Smoothness can also mask gradients
       (Athalye 2018) — must check.

Design:
  - 5 activations: ReLU, GELU, SiLU, Mish, Softplus.
  - 2 training modes: CE (standard) and PGD-AT (10-step PGD, eps=0.1).
  - 5 x 2 = 10 conditions, single seed (campaign default).
  - SmallCNN, width 32, 10 epochs, SGD(mom=0.9, wd=5e-4), LR=0.05, BATCH=128.
  - Report clean acc, FGSM-ASR, PGD-10-ASR, mean margin, and an
    input-gradient smoothness diagnostic: ||grad_x L_adv||_2 averaged
    over test samples (computed on adv points). Smaller norms suggest
    smoother loss in input space — relevant to the gradient-masking
    audit and to the smoothness narrative.

Verdict rule (pre-registered):
  * SUPPORTED if at least one smooth activation drops PGD-ASR by >=0.03
    over ReLU+PGD-AT AND clean accuracy is within 0.02 of ReLU+PGD-AT.
  * NULL if all smooth activations stay within +-0.02 PGD-ASR of ReLU+PGD-AT.
  * NEGATIVE (smooth hurts) if every smooth activation increases PGD-ASR
    by >=0.03 over ReLU+PGD-AT.

Caveats (single-seed, 6k samples, eps=0.1): a 0.02 PGD swing is inside
plausible seed noise (campaign M1). Report deltas but interpret with care.

Outputs: results/fashion_mnist/h459_activation_sweep_at_output.txt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import campaign.common as C

N_TRAIN    = 6000
EPOCHS     = 10
LR         = 0.05
BATCH      = 128
SEED       = 0
EPS        = 0.1
PGD_STEPS  = 10
PGD_ALPHA  = 0.01

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h459_activation_sweep_at_output.txt")


# ---------------------------------------------------------------------------
# Activation library. We post-build-replace nn.ReLU modules so we can reuse
# campaign.common.build_model("cnn", ...) with its default ReLU scaffold.
# ---------------------------------------------------------------------------
ACT_FACTORIES = {
    "relu":     lambda: nn.ReLU(),
    "gelu":     lambda: nn.GELU(),
    "silu":     lambda: nn.SiLU(),
    "mish":     lambda: nn.Mish(),
    "softplus": lambda: nn.Softplus(beta=1.0),
}


def replace_relu(model, factory):
    """Recursively replace every nn.ReLU module with factory()."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, factory())
        else:
            replace_relu(module, factory)
    return model


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(model, Xtr, Ytr, adv_train):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            if adv_train:
                xb = C.pgd(model, xb, yb, eps=EPS,
                           steps=PGD_STEPS, alpha=PGD_ALPHA)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def adv_grad_norm(model, X, Y, eps=EPS, batch=256):
    """Mean L2 norm of grad of CE-loss wrt input, evaluated at PGD-adv points.
    Smoothness diagnostic: smoother loss => smaller / more-stable input grads.
    """
    model.eval()
    norms = []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        xa = C.pgd(model, x, y, eps=eps, steps=PGD_STEPS, alpha=PGD_ALPHA)
        xa = xa.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        # per-sample L2 norm
        flat = g.view(g.size(0), -1)
        norms.append(flat.norm(dim=1).detach().cpu().numpy())
    return float(np.concatenate(norms).mean())


def evaluate(model, Xte, Yte):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean = C.logits_and_acc(model, Xte, Yte)
    Xf = C.fgsm(model, Xte, Yte, eps=EPS)
    _, accf = C.logits_and_acc(model, Xf, Yte)
    Xp = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, accp = C.logits_and_acc(model, Xp, Yte)
    margin = float(np.mean(C.margin(model, Xte, Yte)))
    g_norm = adv_grad_norm(model, Xte, Yte)
    return dict(clean_acc=float(clean),
                fgsm_asr=1.0 - float(accf),
                pgd_asr=1.0 - float(accp),
                mean_margin=margin,
                adv_grad_l2=g_norm)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    lines = []
    def log(s):
        print(s, flush=True)
        lines.append(s)

    log("H459: Activation Sweep under AT (ReLU/GELU/SiLU/Mish/Softplus)")
    log("=" * 64)
    log(f"config: N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} LR={LR} BATCH={BATCH} "
        f"SEED={SEED} EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    log("dataset: fashion_mnist | arch: SmallCNN width=32 (BN on)")
    log("")

    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset(
        "fashion_mnist", n_train=N_TRAIN, seed=SEED)

    activations = ["relu", "gelu", "silu", "mish", "softplus"]
    modes = [("ce", False), ("pgd_at", True)]

    results = {}
    header = ("cond                     | clean  fgsm_asr  pgd_asr  margin  "
              "adv_grad_l2")
    log(header)
    log("-" * len(header))

    for mode_name, adv_train in modes:
        for act in activations:
            cond = f"{act:<8}_{mode_name:<6}"
            C.set_seed(SEED)
            model = C.build_model("cnn", meta, width=32, act="relu", bn=True)
            replace_relu(model, ACT_FACTORIES[act])
            train(model, Xtr, Ytr, adv_train=adv_train)
            res = evaluate(model, Xte, Yte)
            results[(act, mode_name)] = res
            line = (f"{cond:<24} | {res['clean_acc']:.3f}  "
                    f"{res['fgsm_asr']:.3f}     {res['pgd_asr']:.3f}    "
                    f"{res['mean_margin']:+.3f}  {res['adv_grad_l2']:.4f}")
            log(line)
            # flush per-condition
            with open(OUT_FILE, "w") as f:
                f.write("\n".join(lines) + "\n")

    log("")
    log("=" * 64)
    log("Analysis (PGD-AT block):")
    base = results[("relu", "pgd_at")]
    log(f"  baseline ReLU+PGD-AT: clean={base['clean_acc']:.3f} "
        f"pgd_asr={base['pgd_asr']:.3f} grad_l2={base['adv_grad_l2']:.4f}")
    smooth_acts = ["gelu", "silu", "mish", "softplus"]
    deltas = []
    for act in smooth_acts:
        r = results[(act, "pgd_at")]
        d_pgd = r["pgd_asr"] - base["pgd_asr"]
        d_clean = r["clean_acc"] - base["clean_acc"]
        deltas.append((act, d_pgd, d_clean))
        log(f"  {act:<8} d_pgd_asr={d_pgd:+.3f}  d_clean={d_clean:+.3f}  "
            f"grad_l2={r['adv_grad_l2']:.4f}")

    # verdict per pre-registered rule
    helps = [a for a, dp, dc in deltas if dp <= -0.03 and dc >= -0.02]
    hurts_all = all(dp >= 0.03 for _, dp, _ in deltas)
    inside_null = all(abs(dp) < 0.02 for _, dp, _ in deltas)

    log("")
    if helps:
        verdict = ("SUPPORTED: smooth activation(s) "
                   f"{','.join(helps)} reduce PGD-ASR by >=0.03 vs ReLU+AT "
                   "with clean within 0.02.")
    elif hurts_all:
        verdict = ("NEGATIVE: every smooth activation increases PGD-ASR by "
                   ">=0.03 over ReLU+AT (smooth hurts AT here).")
    elif inside_null:
        verdict = ("NULL: all smooth activations stay within +-0.02 PGD-ASR "
                   "of ReLU+AT — replicates H366 under AT.")
    else:
        verdict = ("MIXED: deltas span the null band; no activation meets "
                   "the SUPPORTED threshold and not all meet NEGATIVE. "
                   "Treat as inconclusive at single seed.")
    log("VERDICT: " + verdict)

    # Cross-mode smoothness check: does AT itself shrink the adv-grad norm
    # more for smooth activations than for ReLU?
    log("")
    log("Smoothness diagnostic (adv_grad_l2 ratio CE / PGD-AT):")
    for act in activations:
        ce = results[(act, "ce")]["adv_grad_l2"]
        at = results[(act, "pgd_at")]["adv_grad_l2"]
        ratio = ce / at if at > 0 else float("nan")
        log(f"  {act:<8} ce={ce:.4f} at={at:.4f} ratio={ratio:.2f}")

    log("")
    log("Caveats: single seed (SEED=0), N_TRAIN=6000 (~10% Fashion-MNIST),")
    log("PGD-10 only (no AutoAttack), eps=0.1 (saturating regime). A 0.02")
    log("PGD-ASR swing is plausible single-seed noise — treat sub-0.03")
    log("deltas as not-significant. Smoothness diagnostic is suggestive,")
    log("not a gradient-masking certificate (would need transfer + EOT,")
    log("see H391 protocol).")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
