"""
H417 - Curriculum Adversarial Training (Cai et al. 2018).

Cai et al. 2018 "Curriculum Adversarial Training" (arXiv:1805.04807) showed that
gradually increasing attack strength during training leads to better robustness
than jumping straight to the hardest attack — the model can learn incrementally
rather than collapsing to a degenerate solution early on.

Three conditions on Fashion-MNIST SmallCNN:
  A. Standard PGD-AT: fixed eps=0.1, steps=10 (all epochs).
  B. Eps curriculum: eps linearly ramped 0 -> 0.1 over EPOCHS (warm-up on strength).
  C. Steps curriculum: PGD steps linearly ramped 1 -> 10 over EPOCHS (warm-up on
     attack quality), eps fixed at 0.1.

Evaluation: clean accuracy, FGSM ASR, PGD ASR (eps=0.1, 10 steps).

Config: N_TRAIN=6000, EPOCHS=10, BATCH=128, SEED=0, LR=0.05, SGD mom=0.9 wd=5e-4.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config ---------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
BATCH = 128
SEED = 0
LR = 0.05
EPS = 0.1
PGD_STEPS = 10
MAX_STEPS = 10   # curriculum steps end point

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h417_curriculum_at_output.txt"
)


# ---- helpers ---------------------------------------------------------------

def _make_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)


def _linear_schedule(start, end, epoch, total):
    """Linearly interpolate from start to end over total epochs (0-indexed epoch)."""
    if total <= 1:
        return end
    return start + (end - start) * epoch / (total - 1)


def train_pgd_at(Xtr, Ytr, mode="fixed"):
    """
    Train with PGD adversarial training.

    mode:
      'fixed'        -> fixed eps=EPS, steps=PGD_STEPS every epoch
      'eps_curriculum'  -> eps linearly 0->EPS, steps fixed=PGD_STEPS
      'steps_curriculum' -> eps fixed=EPS, steps linearly 1->MAX_STEPS
    """
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, seed=SEED)
    opt = _make_optimizer(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()

    for ep in range(EPOCHS):
        # determine this epoch's eps and steps
        if mode == "fixed":
            ep_eps = EPS
            ep_steps = PGD_STEPS
        elif mode == "eps_curriculum":
            ep_eps = _linear_schedule(0.0, EPS, ep, EPOCHS)
            ep_steps = PGD_STEPS
        elif mode == "steps_curriculum":
            ep_eps = EPS
            ep_steps = max(1, round(_linear_schedule(1, MAX_STEPS, ep, EPOCHS)))
        else:
            raise ValueError(mode)

        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            # Generate adversarial examples for this batch
            if ep_eps > 0 and ep_steps > 0:
                xb = C.pgd(model, xb, yb, eps=ep_eps, steps=ep_steps,
                           alpha=2.5 * ep_eps / ep_steps)

            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

        sched.step()

    model.eval()
    return model


def eval_model(model, Xte, Yte):
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


# ---- main ------------------------------------------------------------------

def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H417  Curriculum Adversarial Training  (Cai et al. 2018, arXiv:1805.04807)")
    out("=" * 80)
    out(f"dataset={DS}  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}  EPOCHS={EPOCHS}  "
        f"BATCH={BATCH}  SEED={SEED}")
    out(f"LR={LR}  SGD(mom=0.9, wd=5e-4)  EPS={EPS}  PGD_STEPS={PGD_STEPS}")
    out(f"device={C.DEVICE}")
    out("")
    out("CONDITIONS:")
    out("  A. fixed      : PGD-AT eps=0.1 steps=10 (all epochs)")
    out("  B. eps_curric : eps 0->0.1 linear over epochs, steps=10")
    out("  C. steps_curric: steps 1->10 linear over epochs, eps=0.1")
    out("")

    # load data
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    conditions = [
        ("A: fixed PGD-AT",          "fixed"),
        ("B: eps curriculum 0->0.1", "eps_curriculum"),
        ("C: steps curriculum 1->10","steps_curriculum"),
    ]

    rows = []
    for label, mode in conditions:
        out(f"--- {label} ---")
        out(f"    training...")
        t1 = time.time()
        model = train_pgd_at(Xtr, Ytr, mode=mode)
        acc, fgsm_asr, pgd_asr = eval_model(model, Xte, Yte)
        elapsed = time.time() - t1
        out(f"    clean_acc={acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  PGD_ASR={pgd_asr:.4f}"
            f"  ({elapsed:.0f}s)")
        rows.append({
            "label": label,
            "mode": mode,
            "acc": acc,
            "fgsm_asr": fgsm_asr,
            "pgd_asr": pgd_asr,
        })
        flush_file()
        out("")

    # ---- MAIN TABLE ----
    out("=" * 80)
    out("MAIN TABLE")
    out("=" * 80)
    hdr = "{:<30} {:>10} {:>10} {:>10}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<30} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["label"], r["acc"], r["fgsm_asr"], r["pgd_asr"]))
    out("-" * len(hdr))
    out("")

    # ---- VERDICT ----
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)

    base = rows[0]  # condition A: fixed PGD-AT
    currics = rows[1:]

    for r in currics:
        d_acc = r["acc"] - base["acc"]
        d_pgd = r["pgd_asr"] - base["pgd_asr"]
        d_fgsm = r["fgsm_asr"] - base["fgsm_asr"]
        out(f"  {r['label']}:")
        out(f"    vs fixed AT -> d_clean={d_acc:+.4f}  d_FGSM_ASR={d_fgsm:+.4f}  "
            f"d_PGD_ASR={d_pgd:+.4f}")

    out("")
    # best curriculum by lowest PGD_ASR
    best = min(currics, key=lambda r: r["pgd_asr"])
    pgd_gain = base["pgd_asr"] - best["pgd_asr"]

    if pgd_gain > 0.02 and best["acc"] >= base["acc"] - 0.02:
        verdict = (
            f"YES: curriculum training ({best['mode']}) reduces PGD_ASR by "
            f"{pgd_gain:.4f} vs fixed AT while maintaining clean accuracy. "
            f"Gradual warm-up helps the model learn more robust features."
        )
    elif pgd_gain > 0.02:
        verdict = (
            f"PARTIAL: curriculum ({best['mode']}) reduces PGD_ASR by {pgd_gain:.4f} "
            f"vs fixed AT but at a clean-accuracy cost > 0.02."
        )
    elif pgd_gain >= -0.02:
        verdict = (
            f"NEUTRAL: curriculum training does not meaningfully change robustness "
            f"vs fixed AT (best PGD_ASR delta = {pgd_gain:+.4f}). "
            f"At 10 epochs / 6000 samples the curriculum warm-up may need more "
            f"training budget to show a benefit."
        )
    else:
        verdict = (
            f"NO: curriculum training HURTS robustness vs fixed AT "
            f"(best PGD_ASR delta = {pgd_gain:+.4f}); stepping straight to full "
            f"strength appears better in this regime."
        )

    out(f"  ONE-LINE VERDICT: {verdict}")
    out("")
    out(f"done in {time.time() - t0:.1f}s")
    out(f"[saved] {OUT_PATH}")

    flush_file()
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
