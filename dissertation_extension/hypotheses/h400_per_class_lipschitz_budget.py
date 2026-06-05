"""
H400 – Per-class Jacobian budget.
Hypothesis: low-margin classes (Pullover=2, Coat=4, Shirt=6) benefit from
higher Jacobian-penalty weight; class-conditional weighting improves their
robustness without global clean cost.
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
OUT_FILE = os.path.join(RESULTS_DIR, "h400_per_class_lipschitz_budget_output.txt")

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
SEED = 0

META = {"channels": 1, "size": 28, "n_classes": 10}

# Fashion-MNIST class names
CLASS_NAMES = {0:"T-shirt",1:"Trouser",2:"Pullover",3:"Dress",4:"Coat",
               5:"Sandal",6:"Shirt",7:"Sneaker",8:"Bag",9:"Ankle boot"}
VULNERABLE = {2, 4, 6}  # Pullover, Coat, Shirt

# class weight map for condition (b)
# Vulnerable classes get 2x, others 1x; scaled so mean≈0.1 (10 classes, 3 at 2x, 7 at 1x -> mean=1.3x -> scale by 0.1/1.3)
_base = 0.1 / (3 * 2 + 7 * 1) * 10  # = 0.1 / 1.3 * 1 ≈ 0.0769
CLASS_WEIGHTS = {c: (2 * _base if c in VULNERABLE else _base) for c in range(10)}


def jacobian_frob_penalty_batch(model, xb):
    """Per-sample Jacobian Frobenius penalty (Hutchinson), returns shape (B,)."""
    xb = xb.clone().requires_grad_(True)
    logits = model(xb)
    v = torch.randn_like(logits)
    jvp = autograd.grad((logits * v).sum(), xb,
                        create_graph=True, retain_graph=True)[0]
    return (jvp ** 2).view(xb.size(0), -1).mean(1)  # (B,)


def train_condition(model, Xtr, Ytr, mode):
    """mode: 'zero', 'uniform', 'classweighted'"""
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    lam_tensor = torch.tensor([CLASS_WEIGHTS[c] for c in range(10)],
                               dtype=torch.float32, device=C.DEVICE)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            ce_loss = F.cross_entropy(model(xb), yb)
            if mode == "zero":
                loss = ce_loss
            elif mode == "uniform":
                pen = jacobian_frob_penalty_batch(model, xb).mean()
                loss = ce_loss + 0.1 * pen
            else:  # classweighted
                per_sample_pen = jacobian_frob_penalty_batch(model, xb)
                per_sample_lam = lam_tensor[yb]
                loss = ce_loss + (per_sample_lam * per_sample_pen).mean()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def per_class_fgsm_asr(model, Xte, Yte):
    """Returns dict: class -> FGSM ASR."""
    xf = C.fgsm(model, Xte, Yte, eps=EPS)
    with torch.no_grad():
        clean_pred = model(Xte).argmax(1)
        adv_pred = model(xf).argmax(1)
    asr_per_class = {}
    for c in range(10):
        mask = (Yte == c) & (clean_pred == c)
        if mask.sum() > 0:
            flipped = (adv_pred[mask] != Yte[mask]).float().mean().item()
            asr_per_class[c] = flipped
        else:
            asr_per_class[c] = float("nan")
    return asr_per_class


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

    log("H400 – Per-class Lipschitz Budget")
    log("=" * 60)
    log(f"Vulnerable classes (2x weight): {[f'{c}:{CLASS_NAMES[c]}' for c in sorted(VULNERABLE)]}")
    log(f"Class weights (classweighted): { {c: round(CLASS_WEIGHTS[c],4) for c in range(10)} }")

    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist",
                                         n_train=N_TRAIN, n_eval=2000, seed=SEED)

    conditions = ["zero", "uniform", "classweighted"]
    results = {}
    class_asrs = {}

    for cond in conditions:
        log(f"\nCondition: {cond}")
        C.set_seed(SEED)
        model = C.build_model("cnn", META, width=32, seed=SEED)
        model = train_condition(model, Xtr, Ytr, cond)
        clean_acc, fgsm_asr, pgd_asr, mg = evaluate(model, Xte, Yte)
        results[cond] = (clean_acc, fgsm_asr, pgd_asr, mg)
        pc_asr = per_class_fgsm_asr(model, Xte, Yte)
        class_asrs[cond] = pc_asr
        log(f"  Clean={clean_acc:.3f}  FGSM ASR={fgsm_asr:.3f}  PGD ASR={pgd_asr:.3f}  Margin={mg:.3f}")
        log(f"  Per-class FGSM ASR (vulnerable): " +
            ", ".join(f"{CLASS_NAMES[c]}={pc_asr[c]:.3f}" for c in sorted(VULNERABLE)))

    log("\n" + "=" * 60)
    log("SUMMARY TABLE (overall)")
    log(f"{'Condition':<15} {'Clean':>8} {'FGSM ASR':>10} {'PGD ASR':>9} {'Margin':>8}")
    log("-" * 55)
    for cond, (ca, fa, pa, mg) in results.items():
        log(f"{cond:<15} {ca:>8.3f} {fa:>10.3f} {pa:>9.3f} {mg:>8.3f}")

    log("\nPer-class FGSM ASR for vulnerable classes:")
    log(f"{'Class':<12} {'zero':>8} {'uniform':>10} {'classweighted':>14}")
    log("-" * 48)
    for c in sorted(VULNERABLE):
        log(f"{CLASS_NAMES[c]:<12} {class_asrs['zero'][c]:>8.3f} "
            f"{class_asrs['uniform'][c]:>10.3f} {class_asrs['classweighted'][c]:>14.3f}")

    # Verdict
    log("\nVERDICT:")
    vuln_uniform = np.mean([class_asrs["uniform"][c] for c in VULNERABLE])
    vuln_weighted = np.mean([class_asrs["classweighted"][c] for c in VULNERABLE])
    log(f"  Uniform λ: mean ASR on vulnerable classes = {vuln_uniform:.3f}")
    log(f"  Class-weighted: mean ASR on vulnerable classes = {vuln_weighted:.3f}")
    clean_drop = results["uniform"][0] - results["classweighted"][0]
    log(f"  Clean acc change (uniform -> weighted): {clean_drop:+.3f}")
    if vuln_weighted < vuln_uniform - 0.01:
        log(f"  HYPOTHESIS SUPPORTED: class weighting reduces vulnerable ASR by "
            f"{vuln_uniform - vuln_weighted:.3f} (clean delta {clean_drop:+.3f})")
    else:
        log(f"  HYPOTHESIS NOT SUPPORTED: class weighting did not clearly help vulnerable classes")

    output = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(output)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
