"""
H431 - Input Mixup training and adversarial robustness on Fashion-MNIST.

Hypothesis: linear interpolation training (Mixup; Zhang et al., 2018) enforces
local linearity of the learned function, which may reduce sensitivity to small
adversarial perturbations by smoothing decision boundaries.  Lamb et al. (2019)
showed Mixup combined with adversarial training further hardens models; here we
isolate the Mixup-only effect and ask whether it provides any free-lunch
robustness even without explicit adversarial training.

References:
  - Zhang H. et al. (2018). "mixup: Beyond Empirical Risk Minimisation." ICLR
    2018. https://arxiv.org/abs/1710.09412
  - Lamb A. et al. (2019). "Interpolated Adversarial Training." NIPS workshop.
    https://arxiv.org/abs/1906.06784

Design:
  1. Train a BASELINE CNN (standard cross-entropy, no Mixup).
  2. Train four MIXUP CNNs with α ∈ {0.1, 0.2, 0.4, 1.0} (λ ~ Beta(α,α)).
  3. Evaluate for each: clean accuracy, FGSM ASR, PGD-10 ASR vs baseline.
  4. TRANSFER-ATTACK MASKING CHECK: generate PGD adversarial examples on the
     baseline model and evaluate their ASR on each Mixup model (and vice versa).
     A drop in transferred ASR indicates gradient masking / obfuscation rather
     than genuine robustness.
  5. Report Δ(PGD_ASR) and Δ(FGSM_ASR) vs baseline, and transferred ASR table.

Config: DS=fashion_mnist, N_TRAIN=6000, N_EVAL=2000, EPOCHS=10, LR=0.05,
BATCH=128, SGD mom=0.9 wd=5e-4, SEED=0, EPS=0.1, PGD_STEPS=10, width=32.
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
DS        = "fashion_mnist"
N_TRAIN   = 6000
N_EVAL    = 2000
EPOCHS    = 10
LR        = 0.05
BATCH     = 128
SEED      = 0
EPS       = 0.1
PGD_STEPS = 10
PGD_ALPHA = 2.5 * EPS / PGD_STEPS
ALPHAS    = [0.1, 0.2, 0.4, 1.0]   # Mixup Beta concentration sweeps
WIDTH     = 32

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h431_mixup_robustness_output.txt",
)


# ---- helpers -----------------------------------------------------------------

def _sgd(model):
    return torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=5e-4,
    )


def _mixup_batch(xb, yb, alpha):
    """Return (mixed_x, ya, yb, lam) with lam ~ Beta(alpha, alpha)."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(xb.size(0), device=xb.device)
    xm = lam * xb + (1.0 - lam) * xb[perm]
    return xm, yb, yb[perm], lam


def train(Xtr, Ytr, seed, alpha=None):
    """Train CNN from scratch. alpha=None -> standard; alpha>0 -> Mixup."""
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=WIDTH, seed=seed)
    opt   = _sgd(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n     = Xtr.size(0)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            if alpha is not None:
                xm, ya, yb2, lam = _mixup_batch(xb, yb, alpha)
                logits = model(xm)
                loss = (lam * F.cross_entropy(logits, ya)
                        + (1.0 - lam) * F.cross_entropy(logits, yb2))
            else:
                loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_all(model, Xte, Yte):
    """Return dict: clean_acc, fgsm_asr, pgd_asr."""
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd",  eps=EPS, steps=PGD_STEPS)
    return {"acc": acc, "fgsm_asr": fg["asr"], "pgd_asr": pg["asr"]}


def gen_pgd_advs(model, Xte, Yte):
    """Generate PGD adversarial examples (full eval set) in batches."""
    advs = []
    for i in range(0, Xte.size(0), 256):
        xb, yb = Xte[i:i + 256], Yte[i:i + 256]
        advs.append(C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA))
    return torch.cat(advs, dim=0)


def transfer_asr(victim_model, adv_X, Yte):
    """ASR of pre-generated adv_X on victim_model (over all samples)."""
    victim_model.eval()
    flips = []
    for i in range(0, adv_X.size(0), 256):
        xa, y = adv_X[i:i + 256], Yte[i:i + 256]
        with torch.no_grad():
            flips.append((victim_model(xa).argmax(1) != y).cpu())
    flips = torch.cat(flips).numpy()
    return float(flips.mean())


# ---- main --------------------------------------------------------------------

def main():
    t0    = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H431  Mixup training & adversarial robustness (Fashion-MNIST)")
    out("=" * 80)
    out(f"  Zhang et al. 2018 (mixup: Beyond ERM) | Lamb et al. 2019 (Interp. AT)")
    out(f"  hypothesis: local-linearity from Mixup reduces adversarial sensitivity")
    out(f"  sweep α ∈ {ALPHAS}   (λ ~ Beta(α,α) per batch)")
    out(f"  config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4)")
    out(f"         EPS={EPS} PGD_STEPS={PGD_STEPS} width={WIDTH} SEED={SEED}")
    out(f"         device={C.DEVICE}")
    out("")

    # ---- data ----------------------------------------------------------------
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    # ---- baseline ------------------------------------------------------------
    out("\n[1] Training BASELINE (standard CE, no Mixup)...")
    base_model  = train(Xtr, Ytr, seed=SEED, alpha=None)
    base_metrics = eval_all(base_model, Xte, Yte)
    out(f"    baseline: clean_acc={base_metrics['acc']:.4f}  "
        f"FGSM_ASR={base_metrics['fgsm_asr']:.4f}  "
        f"PGD_ASR={base_metrics['pgd_asr']:.4f}")

    # pre-generate baseline PGD adversarials for transfer check
    out("    generating baseline PGD adversarials for transfer-attack masking check...")
    base_advs = gen_pgd_advs(base_model, Xte, Yte)

    # ---- Mixup sweep ---------------------------------------------------------
    results   = [{"label": "baseline (α=—)", "alpha": None, **base_metrics,
                  "transfer_from_base_asr": transfer_asr(base_model, base_advs, Yte),
                  "transfer_to_base_asr": base_metrics["pgd_asr"]}]
    mixup_advs_list = []   # store per-α PGD advs for cross-transfer

    for ai, alpha in enumerate(ALPHAS):
        out(f"\n[{ai+2}] Training Mixup model α={alpha}...")
        m = train(Xtr, Ytr, seed=SEED, alpha=alpha)
        mt = eval_all(m, Xte, Yte)

        # how well do baseline adversarials fool this Mixup model?
        xfer_from_base = transfer_asr(m, base_advs, Yte)

        # generate this model's PGD advs; evaluate ASR on baseline
        out(f"    generating Mixup-α={alpha} PGD adversarials for transfer check...")
        mx_advs = gen_pgd_advs(m, Xte, Yte)
        mixup_advs_list.append((alpha, mx_advs))
        xfer_to_base = transfer_asr(base_model, mx_advs, Yte)

        row = {"label": f"Mixup α={alpha}", "alpha": alpha, **mt,
               "transfer_from_base_asr": xfer_from_base,
               "transfer_to_base_asr": xfer_to_base}
        results.append(row)

        out(f"    Mixup α={alpha}: clean_acc={mt['acc']:.4f}  "
            f"FGSM_ASR={mt['fgsm_asr']:.4f}  PGD_ASR={mt['pgd_asr']:.4f}")
        out(f"    transfer masking: base->this={xfer_from_base:.4f}  "
            f"this->base={xfer_to_base:.4f}  "
            f"(white-box PGD_ASR this={mt['pgd_asr']:.4f})")
        out(f"    Δ clean_acc={mt['acc']-base_metrics['acc']:+.4f}  "
            f"Δ PGD_ASR={mt['pgd_asr']-base_metrics['pgd_asr']:+.4f}  "
            f"({time.time()-t0:.0f}s elapsed)")
        flush()

    # ---- main table ----------------------------------------------------------
    out("\n" + "=" * 80)
    out("[MAIN TABLE]")
    out("=" * 80)
    hdr = ("{:<20} {:>9} {:>9} {:>9} {:>18} {:>16}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR",
        "transfer_base→m", "transfer_m→base"))
    out(hdr)
    out("-" * len(hdr))
    for r in results:
        out("{:<20} {:>9.4f} {:>9.4f} {:>9.4f} {:>18.4f} {:>16.4f}".format(
            r["label"], r["acc"], r["fgsm_asr"], r["pgd_asr"],
            r["transfer_from_base_asr"], r["transfer_to_base_asr"]))
    out("-" * len(hdr))

    # ---- masking detection ---------------------------------------------------
    out("\n[TRANSFER-ATTACK MASKING CHECK]")
    out("  Gradient masking indicator: white-box PGD_ASR << transfer PGD_ASR")
    out("  (a masked model appears robust to white-box but remains vulnerable to transfer)")
    for r in results[1:]:
        wb  = r["pgd_asr"]
        xfr = r["transfer_from_base_asr"]
        masked = (xfr > wb + 0.05)
        out(f"  {r['label']}: white-box={wb:.4f}  transfer_from_base={xfr:.4f}  "
            f"masking={'SUSPECTED' if masked else 'unlikely'}")

    # ---- verdict -------------------------------------------------------------
    out("\n" + "=" * 80)
    out("[VERDICT]")
    out("=" * 80)
    base_pgd = base_metrics["pgd_asr"]
    best = min(results[1:], key=lambda r: r["pgd_asr"])
    worst = max(results[1:], key=lambda r: r["pgd_asr"])
    robust_gain = base_pgd - best["pgd_asr"]
    acc_cost    = best["acc"] - base_metrics["acc"]
    acc_ok      = best["acc"] >= base_metrics["acc"] - 0.02

    out(f"  best robustness: {best['label']}  PGD_ASR={best['pgd_asr']:.4f}  "
        f"(baseline={base_pgd:.4f}  gain={robust_gain:+.4f})")
    out(f"  worst robustness: {worst['label']}  PGD_ASR={worst['pgd_asr']:.4f}")
    out(f"  clean-acc change at best: {acc_cost:+.4f}  "
        f"({'within 0.02' if acc_ok else 'DROPPED >0.02'})")

    # check for gradient masking across all Mixup models
    any_masked = any(r["transfer_from_base_asr"] > r["pgd_asr"] + 0.05
                     for r in results[1:])

    if robust_gain > 0.03 and acc_ok and not any_masked:
        verdict = ("SUPPORTED: Mixup training yields genuine robustness "
                   f"(PGD_ASR reduced by {robust_gain:.3f}) with no gradient masking "
                   "and acceptable clean-accuracy cost.")
    elif robust_gain > 0.03 and any_masked:
        verdict = ("MASKED: apparent robustness from Mixup is confounded by gradient "
                   "masking (transfer ASR > white-box ASR); the improvement is not genuine.")
    elif robust_gain > 0.03 and not acc_ok:
        verdict = ("PARTIAL: Mixup reduces PGD_ASR but at a clean-accuracy cost > 0.02; "
                   "no free lunch.")
    else:
        verdict = ("NOT SUPPORTED: Mixup training provides little to no robustness "
                   f"(best PGD_ASR gain = {robust_gain:.3f}); local-linearity alone "
                   "is insufficient against L∞ attacks at this scale.")

    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time()-t0:.1f}s")

    flush()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
