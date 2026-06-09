"""
H208 - Two-teacher diversity distillation (DARHT-style).

Hypothesis: distilling from two teachers with different random seeds (diverse
adversarial failure modes) produces a student with PGD-10 ASR 5-8pp lower than
single-teacher ARD, because diverse soft labels cover more of the adversarial
input space.

Grounded in: arXiv:2402.15586 (DARHT, Feb 2024).

Protocol:
  - Train 2 teachers on Fashion-MNIST (n_train=6000, 20 epochs, different seeds).
  - Train 1 student via ARD in two variants:
    (a) Single-teacher ARD: KL(student || teacher_A) on PGD-3 adversarial examples.
    (b) Two-teacher DARHT: 0.5*[KL(student || teacher_A) + KL(student || teacher_B)]
        on PGD-3 adversarial examples.
  - Measure teacher adversarial transferability (PGD-10 on A, eval on B).
  - Evaluate student: clean_acc, FGSM ASR (eps=0.1), PGD-10 ASR (eps=0.1).
  - Test: pgd_asr(two-teacher) < pgd_asr(single-teacher) - 3pp?

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign.common import (
    DEVICE, set_seed, load_dataset, dataset_meta, build_model,
    train_model, pgd, fgsm, attack_success, logits_and_acc,
)

DATASET = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 500
TEACHER_EPOCHS = 20
STUDENT_EPOCHS = 20
ADV_EPS = 0.1
PGD_STEPS_TRAIN = 3
PGD_STEPS_EVAL = 10
TEMPERATURE = 4.0
BATCH = 128


def train_teacher(Xtr, Ytr, meta, seed):
    """Train a standard CNN teacher with a given seed."""
    set_seed(seed)
    model = build_model("cnn", meta, width=32)
    train_model(model, Xtr, Ytr, epochs=TEACHER_EPOCHS, batch=BATCH,
                opt="adam", lr=1e-3, ncls=meta["n_classes"])
    return model


def distill_student(Xtr, Ytr, meta, teachers, mode="single"):
    """Train student via adversarial robustness distillation.

    mode="single": KL(student || teachers[0]) on PGD-3 adv examples.
    mode="darht":  0.5 * sum KL(student || t_i) on PGD-3 adv examples.
    """
    set_seed(99)
    student = build_model("cnn", meta, width=32)
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STUDENT_EPOCHS)
    n = Xtr.size(0)

    for t in teachers:
        t.eval()

    for ep in range(STUDENT_EPOCHS):
        student.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            # generate PGD-3 adversarial examples w.r.t. student
            student.eval()
            xadv = pgd(student, xb, yb, eps=ADV_EPS, steps=PGD_STEPS_TRAIN,
                        alpha=2.5 * ADV_EPS / PGD_STEPS_TRAIN)
            student.train()

            # student logits on adversarial examples
            s_logits = student(xadv)
            s_log_probs = F.log_softmax(s_logits / TEMPERATURE, dim=1)

            # teacher soft targets
            if mode == "single":
                with torch.no_grad():
                    t_probs = F.softmax(teachers[0](xadv) / TEMPERATURE, dim=1)
                loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (TEMPERATURE ** 2)
            else:  # darht
                loss = 0.0
                for t in teachers:
                    with torch.no_grad():
                        t_probs = F.softmax(t(xadv) / TEMPERATURE, dim=1)
                    loss = loss + F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (TEMPERATURE ** 2)
                loss = loss / len(teachers)

            # add CE on clean examples for stability
            ce = F.cross_entropy(student(xb), yb)
            total = 0.7 * loss + 0.3 * ce

            opt.zero_grad()
            total.backward()
            opt.step()
        sched.step()

    student.eval()
    return student


def teacher_transfer_rate(teacher_a, teacher_b, X, Y):
    """Craft PGD-10 on teacher_a, measure % that also fool teacher_b."""
    teacher_a.eval(); teacher_b.eval()
    total_correct_a = 0
    fooled_both = 0
    for i in range(0, X.size(0), BATCH):
        x, y = X[i:i + BATCH], Y[i:i + BATCH]
        with torch.no_grad():
            corr_a = (teacher_a(x).argmax(1) == y)
        xadv = pgd(teacher_a, x, y, eps=ADV_EPS, steps=PGD_STEPS_EVAL)
        with torch.no_grad():
            flip_a = (teacher_a(xadv).argmax(1) != y) & corr_a
            flip_b = (teacher_b(xadv).argmax(1) != y) & corr_a
        total_correct_a += corr_a.sum().item()
        fooled_both += (flip_a & flip_b).sum().item()
    # transfer rate = fraction of A-adversarials that also fool B
    n_flip_a = 0
    for i in range(0, X.size(0), BATCH):
        x, y = X[i:i + BATCH], Y[i:i + BATCH]
        with torch.no_grad():
            corr_a = (teacher_a(x).argmax(1) == y)
        xadv = pgd(teacher_a, x, y, eps=ADV_EPS, steps=PGD_STEPS_EVAL)
        with torch.no_grad():
            flip_a = (teacher_a(xadv).argmax(1) != y) & corr_a
            flip_b = (teacher_b(xadv).argmax(1) != y) & corr_a
            n_flip_a += flip_a.sum().item()
    return fooled_both / max(n_flip_a, 1)


def main():
    print("=" * 74)
    print("H208 - Two-teacher diversity distillation (DARHT-style)")
    print("=" * 74)
    t0 = time.time()

    meta = dataset_meta(DATASET)
    Xtr, Ytr, Xte, Yte = load_dataset(DATASET, n_train=N_TRAIN, n_eval=N_EVAL, seed=42)
    print(f"Dataset: {DATASET}  n_train={Xtr.size(0)}  n_eval={Xte.size(0)}  device={DEVICE}")

    # --- Train two teachers ---
    print("\n--- Training Teacher A (seed=1) ---")
    teacher_a = train_teacher(Xtr, Ytr, meta, seed=1)
    _, acc_a = logits_and_acc(teacher_a, Xte, Yte)
    print(f"  Teacher A clean acc: {acc_a:.4f}")

    print("--- Training Teacher B (seed=2) ---")
    teacher_b = train_teacher(Xtr, Ytr, meta, seed=2)
    _, acc_b = logits_and_acc(teacher_b, Xte, Yte)
    print(f"  Teacher B clean acc: {acc_b:.4f}")

    # --- Teacher diversity (transfer rate) ---
    print("\n--- Teacher adversarial transferability ---")
    tr = teacher_transfer_rate(teacher_a, teacher_b, Xte, Yte)
    print(f"  PGD-10 transfer rate (A->B): {tr:.4f}")
    print(f"  (Low = diverse failure modes)")

    # --- Single-teacher ARD ---
    print("\n--- Training student: single-teacher ARD ---")
    student_single = distill_student(Xtr, Ytr, meta, [teacher_a], mode="single")
    _, clean_single = logits_and_acc(student_single, Xte, Yte)
    fgsm_single = attack_success(student_single, Xte, Yte, attack="fgsm", eps=ADV_EPS, batch=BATCH)
    pgd_single = attack_success(student_single, Xte, Yte, attack="pgd", eps=ADV_EPS, steps=PGD_STEPS_EVAL, batch=BATCH)

    # --- Two-teacher DARHT ---
    print("--- Training student: two-teacher DARHT ---")
    student_darht = distill_student(Xtr, Ytr, meta, [teacher_a, teacher_b], mode="darht")
    _, clean_darht = logits_and_acc(student_darht, Xte, Yte)
    fgsm_darht = attack_success(student_darht, Xte, Yte, attack="fgsm", eps=ADV_EPS, batch=BATCH)
    pgd_darht = attack_success(student_darht, Xte, Yte, attack="pgd", eps=ADV_EPS, steps=PGD_STEPS_EVAL, batch=BATCH)

    # --- Results ---
    elapsed = time.time() - t0
    print(f"\n{'=' * 74}")
    print("RESULTS")
    print(f"{'=' * 74}")
    print(f"  Teacher transfer rate (A->B): {tr:.4f}")
    print()
    print(f"  {'Condition':<22} {'Clean Acc':>10} {'FGSM ASR':>10} {'PGD-10 ASR':>12}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*12}")
    print(f"  {'Single-teacher ARD':<22} {clean_single:>10.4f} {fgsm_single['asr']:>10.4f} {pgd_single['asr']:>12.4f}")
    print(f"  {'Two-teacher DARHT':<22} {clean_darht:>10.4f} {fgsm_darht['asr']:>10.4f} {pgd_darht['asr']:>12.4f}")

    delta = pgd_single["asr"] - pgd_darht["asr"]
    print(f"\n  PGD ASR reduction (single - DARHT): {delta:+.4f} ({delta*100:+.1f}pp)")
    supported = delta > 0.03
    print(f"  Hypothesis supported (>3pp reduction)? {'YES' if supported else 'NO'}")
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
