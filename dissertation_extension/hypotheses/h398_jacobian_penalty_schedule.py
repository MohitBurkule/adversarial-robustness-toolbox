"""
H398 – Annealed Jacobian penalty schedule.
Hypothesis: a strong-early then decayed Jacobian penalty front-loads smoothness
while letting late training recover clean accuracy, beating constant-λ.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import campaign.common as C

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_FILE = os.path.join(RESULTS_DIR, "h398_jacobian_penalty_schedule_output.txt")

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
SEED = 0

META = {"channels": 1, "size": 28, "n_classes": 10}


def jacobian_frob_penalty(model, xb):
    """Hutchinson estimator of Jacobian Frobenius norm squared."""
    xb = xb.requires_grad_(True)
    logits = model(xb)
    v = torch.randn_like(logits)
    jvp = autograd.grad((logits * v).sum(), xb,
                        create_graph=True, retain_graph=True)[0]
    return (jvp ** 2).mean()


def build_lambda_schedule(name, epochs):
    if name == "zero":
        return [0.0] * epochs
    elif name == "constant":
        return [0.1] * epochs
    elif name == "high_to_low":
        return [0.2 - 0.2 * i / (epochs - 1) for i in range(epochs)]
    elif name == "low_to_high":
        return [0.2 * i / (epochs - 1) for i in range(epochs)]
    elif name == "cosine_decay":
        import math
        return [0.2 * 0.5 * (1 + math.cos(math.pi * i / (epochs - 1))) for i in range(epochs)]
    else:
        raise ValueError(name)


def train_with_schedule(model, Xtr, Ytr, lam_schedule):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        lam = lam_schedule[ep]
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            ce_loss = F.cross_entropy(model(xb), yb)
            if lam > 0:
                pen = jacobian_frob_penalty(model, xb)
                loss = ce_loss + lam * pen
            else:
                loss = ce_loss
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def evaluate(model, Xte, Yte):
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    xf = C.fgsm(model, Xte, Yte, eps=EPS)
    _, fgsm_acc = C.logits_and_acc(model, xf, Yte)
    xp = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, pgd_acc = C.logits_and_acc(model, xp, Yte)
    mg = C.margin(model, Xte, Yte).mean()
    return clean_acc, 1 - fgsm_acc, 1 - pgd_acc, float(mg)


def main():
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("H398 – Annealed Jacobian Penalty Schedule")
    log("=" * 60)

    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist",
                                         n_train=N_TRAIN, n_eval=2000, seed=SEED)

    schedules = ["zero", "constant", "high_to_low", "low_to_high", "cosine_decay"]
    results = {}

    for sched_name in schedules:
        log(f"\nTraining schedule: {sched_name}")
        lam_sched = build_lambda_schedule(sched_name, EPOCHS)
        log(f"  λ per epoch: {[round(x,3) for x in lam_sched]}")

        C.set_seed(SEED)
        model = C.build_model("cnn", META, width=32, seed=SEED)
        model = train_with_schedule(model, Xtr, Ytr, lam_sched)
        clean_acc, fgsm_asr, pgd_asr, mg = evaluate(model, Xte, Yte)
        results[sched_name] = (clean_acc, fgsm_asr, pgd_asr, mg)
        log(f"  Clean Acc={clean_acc:.3f}  FGSM ASR={fgsm_asr:.3f}  PGD ASR={pgd_asr:.3f}  Margin={mg:.3f}")

    log("\n" + "=" * 60)
    log("SUMMARY TABLE")
    log(f"{'Schedule':<20} {'Clean':>8} {'FGSM ASR':>10} {'PGD ASR':>9} {'Margin':>8}")
    log("-" * 60)
    for s, (ca, fa, pa, mg) in results.items():
        log(f"{s:<20} {ca:>8.3f} {fa:>10.3f} {pa:>9.3f} {mg:>8.3f}")

    # Verdict
    const_fa = results["constant"][1]
    const_pa = results["constant"][2]
    best_rob = min(results, key=lambda s: results[s][1] + results[s][2])
    log("\nVERDICT:")
    log(f"  Constant λ=0.1: FGSM ASR={const_fa:.3f}, PGD ASR={const_pa:.3f}")
    log(f"  Best robustness schedule: {best_rob} "
        f"(FGSM={results[best_rob][1]:.3f}, PGD={results[best_rob][2]:.3f})")
    if best_rob != "constant" and best_rob != "zero":
        hl = results["high_to_low"]
        log(f"  high→low: FGSM ASR={hl[1]:.3f}, PGD ASR={hl[2]:.3f}")
        improvement = (const_fa + const_pa) - (results[best_rob][1] + results[best_rob][2])
        if improvement > 0.01:
            log(f"  HYPOTHESIS SUPPORTED: {best_rob} beats constant-λ (improvement={improvement:.3f} total ASR)")
        else:
            log(f"  HYPOTHESIS NOT CLEARLY SUPPORTED: improvement marginal ({improvement:.3f})")
    else:
        log("  HYPOTHESIS NOT SUPPORTED: constant or baseline is best")

    output = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(output)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
