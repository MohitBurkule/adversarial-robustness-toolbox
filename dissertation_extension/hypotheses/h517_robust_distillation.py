"""
H517 - Adversarial Robustness Distillation: can a small student learn
robustness from a robust teacher WITHOUT seeing adversarial examples?

Paper: "Toward Understanding Adversarial Distillation: Why Robust Teachers Fail"
(arXiv:2605.21999, May 2026). Key finding: overly strong robust teachers can
DEGRADE student robustness due to overfitting to teacher's soft labels. This
suggests a sweet-spot in teacher strength.

Hypothesis: A student CNN trained via knowledge distillation (KD) from a
PGD-AT teacher on CLEAN data only achieves non-trivial adversarial robustness
(>50% of teacher's robustness) despite never seeing adversarial examples during
its own training. However, the student's robustness scales non-monotonically
with teacher AT strength (eps_train): a moderately robust teacher transfers
more robustness than a very robust teacher.

Experiment:
  (1) Train 3 teachers with different AT strengths: eps=0.05, 0.1, 0.2
  (2) For each teacher, train a student via KD on clean data (soft labels
      from teacher, temperature scaling T=4)
  (3) Evaluate all models (teachers + students) under PGD attack at eps=0.15
  (4) Compare: student robustness vs teacher robustness

Controls: 2 seeds, same architecture for student and teacher.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1]
TEACHER_EPS_LIST = [0.05, 0.1, 0.2]
ADV_STEPS_TRAIN = 7
EVAL_EPS = 0.15
EVAL_STEPS = 20
KD_TEMP = 4.0
KD_EPOCHS = 8
N_TRAIN = 6000
N_EVAL = 1000


def kd_train(student, teacher, X_train, Y_train, epochs, temp, lr=0.01, ncls=10):
    """Knowledge distillation: student matches teacher soft labels on clean data."""
    teacher.eval()
    student.train()
    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    bs = 128

    for ep in range(epochs):
        perm = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), bs):
            idx = perm[i:i+bs]
            xb, yb = X_train[idx], Y_train[idx]

            with torch.no_grad():
                teacher_logits = teacher(xb)
                teacher_soft = F.softmax(teacher_logits / temp, dim=1)

            student_logits = student(xb)
            student_log_soft = F.log_softmax(student_logits / temp, dim=1)

            # KD loss: KL divergence on soft labels + hard label CE
            kd_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * (temp ** 2)
            ce_loss = F.cross_entropy(student_logits, yb)
            loss = 0.7 * kd_loss + 0.3 * ce_loss

            opt.zero_grad()
            loss.backward()
            opt.step()


def eval_robustness(model, X, Y, eps, steps):
    """Return clean acc and adversarial acc (PGD)."""
    model.eval()
    with torch.no_grad():
        clean_pred = model(X).argmax(1)
        clean_acc = (clean_pred == Y).float().mean().item()

    X_adv = C.pgd(model, X, Y, eps=eps, steps=steps)
    with torch.no_grad():
        adv_pred = model(X_adv).argmax(1)
        adv_acc = (adv_pred == Y).float().mean().item()

    return clean_acc, adv_acc


def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)

    # STD baseline (no AT, no KD)
    std_model = C.build_model("cnn", meta, seed=seed)
    C.train_model(std_model, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05, ncls=meta["n_classes"])
    std_clean, std_adv = eval_robustness(std_model, Xte, Yte, EVAL_EPS, EVAL_STEPS)

    teacher_results = []
    student_results = []

    for t_eps in TEACHER_EPS_LIST:
        # Train teacher with AT
        teacher = C.build_model("cnn", meta, seed=seed)
        C.train_model(teacher, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05, ncls=meta["n_classes"],
                      adv_train=True, adv_eps=t_eps, adv_steps=ADV_STEPS_TRAIN)
        t_clean, t_adv = eval_robustness(teacher, Xte, Yte, EVAL_EPS, EVAL_STEPS)
        teacher_results.append({"eps_train": t_eps, "clean_acc": t_clean, "adv_acc": t_adv})

        # Train student via KD from this teacher (clean data only!)
        student = C.build_model("cnn", meta, seed=seed)
        kd_train(student, teacher, Xtr, Ytr, epochs=KD_EPOCHS, temp=KD_TEMP, ncls=meta["n_classes"])
        s_clean, s_adv = eval_robustness(student, Xte, Yte, EVAL_EPS, EVAL_STEPS)
        student_results.append({"eps_train": t_eps, "clean_acc": s_clean, "adv_acc": s_adv})

    return {
        "seed": seed,
        "std_baseline": {"clean_acc": std_clean, "adv_acc": std_adv},
        "teachers": teacher_results,
        "students": student_results,
    }


def main():
    t0 = time.time()
    all_results = [run_seed(s) for s in SEEDS]

    print("=" * 72)
    print("H517 — Adversarial Robustness Distillation")
    print("=" * 72)

    for r in all_results:
        print(f"\n--- Seed {r['seed']} ---")
        print(f"  STD baseline: clean={r['std_baseline']['clean_acc']:.3f}  "
              f"adv={r['std_baseline']['adv_acc']:.3f}")
        print(f"  {'Teacher eps':>12} | {'T-clean':>8} {'T-adv':>8} | {'S-clean':>8} {'S-adv':>8} | {'Transfer%':>9}")
        for t, s in zip(r["teachers"], r["students"]):
            transfer = (s["adv_acc"] / max(t["adv_acc"], 1e-6)) * 100
            print(f"  {t['eps_train']:>12.3f} | {t['clean_acc']:>8.3f} {t['adv_acc']:>8.3f} | "
                  f"{s['clean_acc']:>8.3f} {s['adv_acc']:>8.3f} | {transfer:>8.1f}%")

    print("\n" + "=" * 72)
    print("CROSS-SEED SUMMARY")
    print("=" * 72)

    for i, t_eps in enumerate(TEACHER_EPS_LIST):
        t_advs = [r["teachers"][i]["adv_acc"] for r in all_results]
        s_advs = [r["students"][i]["adv_acc"] for r in all_results]
        transfers = [s / max(t, 1e-6) * 100 for s, t in zip(s_advs, t_advs)]
        print(f"  Teacher eps={t_eps:.3f}: "
              f"T-adv={np.mean(t_advs):.3f}±{np.std(t_advs):.3f}  "
              f"S-adv={np.mean(s_advs):.3f}±{np.std(s_advs):.3f}  "
              f"Transfer={np.mean(transfers):.1f}%")

    # Check non-monotonicity
    mean_student_advs = [np.mean([r["students"][i]["adv_acc"] for r in all_results])
                         for i in range(len(TEACHER_EPS_LIST))]
    best_idx = int(np.argmax(mean_student_advs))
    print(f"\n  Best student robustness from teacher eps={TEACHER_EPS_LIST[best_idx]:.3f} "
          f"(adv_acc={mean_student_advs[best_idx]:.3f})")
    if best_idx != len(TEACHER_EPS_LIST) - 1:
        print("  → NON-MONOTONIC: strongest teacher is NOT the best for distillation")
    else:
        print("  → MONOTONIC: strongest teacher produces most robust student")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
