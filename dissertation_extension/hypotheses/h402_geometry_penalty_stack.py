"""
H402 – Weight-displacement budget + input-grad penalty stack.
Hypothesis: constraining weight displacement (‖W_t−W_0‖>budget → project back)
WHILE applying input-grad penalty reproduces more of AT's robustness than either alone.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import campaign.common as C

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_FILE = os.path.join(RESULTS_DIR, "h402_geometry_penalty_stack_output.txt")

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
SEED = 0
GRAD_PEN_LAM = 0.1
DISP_BUDGET = 10.0

META = {"channels": 1, "size": 28, "n_classes": 10}


def get_flat_params(model):
    return torch.cat([p.data.flatten() for p in model.parameters()])


def set_flat_params(model, flat):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].view(p.shape))
        offset += n


def project_displacement(model, w0_flat, budget):
    """If ‖W - W0‖ > budget, scale displacement back to budget."""
    wt = get_flat_params(model)
    disp = wt - w0_flat
    norm = disp.norm()
    if norm > budget:
        disp = disp * (budget / norm)
        set_flat_params(model, w0_flat + disp)


def input_grad_penalty(model, xb, yb):
    xb = xb.clone().requires_grad_(True)
    ce = F.cross_entropy(model(xb), yb)
    g = autograd.grad(ce, xb, create_graph=True, retain_graph=True)[0]
    return (g ** 2).mean()


def train_condition(model, Xtr, Ytr, use_grad_pen, use_disp_budget, w0_flat=None):
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
            opt.zero_grad()
            if use_grad_pen:
                pen = input_grad_penalty(model, xb, yb)
                loss = F.cross_entropy(model(xb), yb) + GRAD_PEN_LAM * pen
            else:
                loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            if use_disp_budget and w0_flat is not None:
                with torch.no_grad():
                    project_displacement(model, w0_flat, DISP_BUDGET)
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

    log("H402 – Weight-Displacement Budget + Input-Grad Penalty Stack")
    log("=" * 65)
    log(f"Grad penalty λ={GRAD_PEN_LAM}, Displacement budget={DISP_BUDGET}")

    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist",
                                         n_train=N_TRAIN, n_eval=2000, seed=SEED)

    results = {}

    # (a) baseline
    log("\n[a] Baseline (no penalty, no budget)")
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, seed=SEED)
    model = train_condition(model, Xtr, Ytr, use_grad_pen=False, use_disp_budget=False)
    results["baseline"] = evaluate(model, Xte, Yte)
    log(f"  Clean={results['baseline'][0]:.3f}  FGSM ASR={results['baseline'][1]:.3f}  PGD ASR={results['baseline'][2]:.3f}")

    # (b) grad-penalty only
    log("\n[b] Grad-penalty only (λ=0.1)")
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, seed=SEED)
    model = train_condition(model, Xtr, Ytr, use_grad_pen=True, use_disp_budget=False)
    results["grad_pen"] = evaluate(model, Xte, Yte)
    log(f"  Clean={results['grad_pen'][0]:.3f}  FGSM ASR={results['grad_pen'][1]:.3f}  PGD ASR={results['grad_pen'][2]:.3f}")

    # (c) displacement-budget only
    log(f"\n[c] Displacement-budget only (budget={DISP_BUDGET})")
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, seed=SEED)
    w0_flat = get_flat_params(model).clone()
    model = train_condition(model, Xtr, Ytr, use_grad_pen=False, use_disp_budget=True, w0_flat=w0_flat)
    disp_final = (get_flat_params(model) - w0_flat).norm().item()
    log(f"  Final displacement norm: {disp_final:.2f}")
    results["disp_budget"] = evaluate(model, Xte, Yte)
    log(f"  Clean={results['disp_budget'][0]:.3f}  FGSM ASR={results['disp_budget'][1]:.3f}  PGD ASR={results['disp_budget'][2]:.3f}")

    # (d) both
    log(f"\n[d] Both (grad-penalty + displacement-budget)")
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, seed=SEED)
    w0_flat = get_flat_params(model).clone()
    model = train_condition(model, Xtr, Ytr, use_grad_pen=True, use_disp_budget=True, w0_flat=w0_flat)
    disp_final = (get_flat_params(model) - w0_flat).norm().item()
    log(f"  Final displacement norm: {disp_final:.2f}")
    results["both"] = evaluate(model, Xte, Yte)
    log(f"  Clean={results['both'][0]:.3f}  FGSM ASR={results['both'][1]:.3f}  PGD ASR={results['both'][2]:.3f}")

    # (e) true AT reference
    log("\n[e] True Adversarial Training (PGD-7, eps=0.1)")
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, seed=SEED)
    model = C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, opt="sgd", lr=LR,
                          adv_train=True, adv_eps=EPS, adv_steps=7)
    results["adv_train"] = evaluate(model, Xte, Yte)
    log(f"  Clean={results['adv_train'][0]:.3f}  FGSM ASR={results['adv_train'][1]:.3f}  PGD ASR={results['adv_train'][2]:.3f}")

    log("\n" + "=" * 65)
    log("SUMMARY TABLE")
    log(f"{'Condition':<20} {'Clean':>8} {'FGSM ASR':>10} {'PGD ASR':>9} {'Margin':>8}")
    log("-" * 60)
    for cond, (ca, fa, pa, mg) in results.items():
        log(f"{cond:<20} {ca:>8.3f} {fa:>10.3f} {pa:>9.3f} {mg:>8.3f}")

    # Verdict
    log("\nVERDICT:")
    at_fgsm = results["adv_train"][1]
    at_pgd = results["adv_train"][2]
    both_fgsm = results["both"][1]
    both_pgd = results["both"][2]
    gp_fgsm = results["grad_pen"][1]
    gp_pgd = results["grad_pen"][2]
    db_fgsm = results["disp_budget"][1]
    db_pgd = results["disp_budget"][2]

    stack_beats_each = (both_fgsm + both_pgd) < (gp_fgsm + gp_pgd) and \
                       (both_fgsm + both_pgd) < (db_fgsm + db_pgd)

    at_gap_pgd = both_pgd - at_pgd
    log(f"  Grad-pen only: FGSM={gp_fgsm:.3f}, PGD={gp_pgd:.3f}")
    log(f"  Disp-budget only: FGSM={db_fgsm:.3f}, PGD={db_pgd:.3f}")
    log(f"  Stack (both): FGSM={both_fgsm:.3f}, PGD={both_pgd:.3f}")
    log(f"  True AT: FGSM={at_fgsm:.3f}, PGD={at_pgd:.3f}")
    log(f"  Stack PGD gap vs AT: {at_gap_pgd:+.3f}")

    if stack_beats_each:
        log(f"  STACK BEATS EACH COMPONENT INDIVIDUALLY.")
    else:
        log(f"  Stack does NOT clearly beat both components individually.")

    if at_gap_pgd < 0.05:
        log(f"  Stack approximates AT closely (gap={at_gap_pgd:.3f})")
    else:
        log(f"  Stack lags behind AT by {at_gap_pgd:.3f} PGD ASR")

    output = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(output)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
