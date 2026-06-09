"""
H419 - Self-distillation with adversarial labels.

Hypothesis: A student trained via soft-label KL-distillation from a PGD-AT
teacher inherits adversarial robustness *without ever seeing ground-truth labels*
-- pure teacher-logit supervision. Does distilling on (clean+adv) pairs beat
distilling on clean pairs only? Does distillation temperature matter?

Design
------
1. Train TEACHER with full PGD adversarial training (10 epochs).
2. For each temperature T in {1, 4, 10} train two fresh students:
   a) student_clean    -- KL on clean inputs only, teacher soft labels
   b) student_clean_adv -- KL on clean + PGD-adv inputs, teacher soft labels
3. Eval all models: clean acc, FGSM_ASR, PGD_ASR.

Conditions reported:
  teacher              (PGD-AT, cross-entropy)
  student_clean_T1/4/10
  student_clean+adv_T1/4/10
  Total = 1 + 2*3 = 7 rows

Config: N_TRAIN=6000, EPOCHS=10, BATCH=128, EPS=0.1, SEED=0, Fashion-MNIST,
        SmallCNN width=32, SGD lr=0.05 cosine, PGD steps=7 for AT / steps=10 eval.

Output: results/fashion_mnist/h419_self_distill_adversarial_labels_output.txt
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config ------------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
BATCH = 128
LR = 0.05
SEED = 0
EPS = 0.1
PGD_STEPS_TRAIN = 7
PGD_STEPS_EVAL = 10
TEMPERATURES = [1, 4, 10]

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h419_self_distill_adversarial_labels_output.txt",
)


# ---- helpers -----------------------------------------------------------------

def make_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)


def train_teacher(Xtr, Ytr):
    """PGD adversarial training with cross-entropy."""
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32)
    opt = make_optimizer(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_adv = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS_TRAIN,
                           alpha=2.5 * EPS / PGD_STEPS_TRAIN)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def kl_loss(student_logits, teacher_logits, T):
    """KL(teacher_soft || student_soft) scaled by T^2."""
    p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p, q, reduction="batchmean") * (T * T)


def train_student(teacher, Xtr, Ytr, temperature, use_adv, seed_offset):
    """Distil into a fresh student using teacher soft labels only (no ground-truth)."""
    C.set_seed(SEED + seed_offset)
    student = C.build_model("cnn", META, width=32)
    opt = make_optimizer(student)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    teacher.eval()
    student.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            with torch.no_grad():
                t_clean = teacher(xb)
            loss = kl_loss(student(xb), t_clean, temperature)
            if use_adv:
                # generate adv wrt student (to pressure student's adv distribution)
                xb_adv = C.pgd(student, xb, yb, eps=EPS, steps=PGD_STEPS_TRAIN,
                               alpha=2.5 * EPS / PGD_STEPS_TRAIN)
                with torch.no_grad():
                    t_adv = teacher(xb_adv)
                loss = loss + kl_loss(student(xb_adv), t_adv, temperature)
                loss = loss / 2.0
            loss.backward()
            opt.step()
        sched.step()
    student.eval()
    return student


def eval_model(model, Xte, Yte):
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS_EVAL)
    return acc, fg["asr"], pg["asr"]


# ---- main --------------------------------------------------------------------

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
    out("H419  Self-distillation with adversarial labels (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} "
        f"BATCH={BATCH} LR={LR} SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS_TRAIN={PGD_STEPS_TRAIN} "
        f"PGD_STEPS_EVAL={PGD_STEPS_EVAL} TEMPERATURES={TEMPERATURES}")
    out(f"        device={C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    # ---- TEACHER ----
    out("\n[1] Training TEACHER (PGD-AT, cross-entropy)...")
    teacher = train_teacher(Xtr, Ytr)
    t_acc, t_fgsm, t_pgd = eval_model(teacher, Xte, Yte)
    out(f"    teacher: clean_acc={t_acc:.4f}  FGSM_ASR={t_fgsm:.4f}  "
        f"PGD_ASR={t_pgd:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    rows = [{"label": "teacher (PGD-AT)",
             "acc": t_acc, "fgsm": t_fgsm, "pgd": t_pgd}]

    # ---- STUDENTS ----
    for ti, T in enumerate(TEMPERATURES):
        for use_adv, kind in [(False, "clean"), (True, "clean+adv")]:
            label = f"student_{kind}_T{T}"
            out(f"\n[2] Training {label}...")
            seed_off = (ti + 1) * 10 + int(use_adv)
            student = train_student(teacher, Xtr, Ytr,
                                    temperature=T,
                                    use_adv=use_adv,
                                    seed_offset=seed_off)
            acc, fgsm, pgd = eval_model(student, Xte, Yte)
            out(f"    {label}: clean_acc={acc:.4f}  FGSM_ASR={fgsm:.4f}  "
                f"PGD_ASR={pgd:.4f}  ({time.time()-t0:.0f}s)")
            rows.append({"label": label, "acc": acc, "fgsm": fgsm, "pgd": pgd})
            flush_file()

    # ---- MAIN TABLE ----
    out("\n" + "=" * 80)
    out("[3] MAIN TABLE")
    out("=" * 80)
    hdr = "{:<30} {:>10} {:>10} {:>10}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<30} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["label"], r["acc"], r["fgsm"], r["pgd"]))
    out("-" * len(hdr))

    # ---- VERDICT ----
    out("\n" + "=" * 80)
    out("[4] VERDICT")
    out("=" * 80)

    teacher_row = rows[0]
    clean_students = [r for r in rows[1:] if "clean+adv" not in r["label"]]
    adv_students   = [r for r in rows[1:] if "clean+adv" in r["label"]]

    best_clean = min(clean_students, key=lambda r: r["pgd"])
    best_adv   = min(adv_students,   key=lambda r: r["pgd"])

    out(f"  Teacher PGD_ASR                      = {teacher_row['pgd']:.4f}")
    out(f"  Best student_clean PGD_ASR (T={best_clean['label'].split('T')[1]}) "
        f"= {best_clean['pgd']:.4f}  "
        f"(delta vs teacher = {best_clean['pgd'] - teacher_row['pgd']:+.4f})")
    out(f"  Best student_clean+adv PGD_ASR (T={best_adv['label'].split('T')[1]}) "
        f"= {best_adv['pgd']:.4f}  "
        f"(delta vs teacher = {best_adv['pgd'] - teacher_row['pgd']:+.4f})")
    out(f"  clean+adv vs clean-only best delta   = "
        f"{best_adv['pgd'] - best_clean['pgd']:+.4f} "
        f"(negative => adv distillation is more robust)")

    # Temperature analysis (within each student type)
    out("\n  Temperature sweep (PGD_ASR by condition x T):")
    out("  {:<30} {:>6} {:>6} {:>6}".format("condition", "T=1", "T=4", "T=10"))
    for prefix, group in [("student_clean", clean_students),
                           ("student_clean+adv", adv_students)]:
        vals = {r["label"].split("_T")[1]: r["pgd"] for r in group}
        out("  {:<30} {:>6.4f} {:>6.4f} {:>6.4f}".format(
            prefix, float(vals.get("1", float("nan"))),
            float(vals.get("4", float("nan"))),
            float(vals.get("10", float("nan")))))

    adv_beats_clean = best_adv["pgd"] < best_clean["pgd"] - 0.01
    T_helps = any(
        [r["pgd"] for r in adv_students if "T10" in r["label"]][0] <
        [r["pgd"] for r in adv_students if "T1"  in r["label"]][0]
        for _ in [1]
    )
    teacher_beats_all = teacher_row["pgd"] < min(r["pgd"] for r in rows[1:])

    if adv_beats_clean and not teacher_beats_all:
        verdict = ("YES: distilling on adversarial pairs transfers more robustness "
                   "than clean-only distillation, and a student matches/beats the teacher.")
    elif adv_beats_clean and teacher_beats_all:
        verdict = ("PARTIAL: adversarial distillation beats clean-only distillation "
                   "but teacher remains the most robust model.")
    elif not adv_beats_clean and not teacher_beats_all:
        verdict = ("NO (unexpected): clean-only distillation is sufficient; "
                   "a student matches/beats the teacher without adversarial pairs.")
    else:
        verdict = ("NO: adversarial distillation does not improve on clean-only "
                   "distillation; teacher is the most robust model.")

    out("\n  ONE-LINE VERDICT: " + verdict)
    out(f"\ndone in {time.time() - t0:.1f}s")

    flush_file()
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
