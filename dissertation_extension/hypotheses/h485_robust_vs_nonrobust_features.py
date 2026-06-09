"""
H485 - Robust vs Non-Robust Features on Fashion-MNIST.

Seed (Paper 6 of the advisor critique, §5 / §3 G8):
  Ilyas et al. 2019, "Adversarial Examples Are Not Bugs, They Are Features"
  (arXiv:1905.02175). They construct two distilled datasets from CIFAR-10:

    D_R  (robust dataset)      - inputs whose representation under a *robust*
                                 (PGD-AT) teacher matches that of a true sample
                                 of class y. A vanilla model trained on D_R
                                 alone inherits PGD-robustness for free.

    D_NR (non-robust dataset)  - inputs that are visually-class-x but were
                                 adversarially perturbed under a *standard*
                                 teacher to be labelled class y. A vanilla
                                 model trained on (x, y) pairs from D_NR
                                 generalises to the real test set (>~70% clean
                                 acc on CIFAR-10) but is trivially attacked
                                 by PGD (~0% robustness).

The headline claim is that non-robust features are real generalising signal
that humans cannot see, and robustness is a property of the *features used*,
not of the architecture or training algorithm.

This hypothesis asks whether the same picture holds on Fashion-MNIST with
our SmallCNN. F-MNIST is a 1-channel 28x28 dataset with much simpler class
structure than CIFAR-10; if the Ilyas dichotomy is universal, training a
fresh student on D_R should give nontrivial PGD-robustness with no adv
training, and a fresh student on D_NR should give clean-acc on the real test
set >70% with ~0% PGD-robustness.

Critique-aware simplifications (the full Ilyas pipeline is heavy):

  * D_R is built by *feature distillation* under the AT teacher. For each
    (x, y) in the training set we initialise from random noise (or, here, a
    different sample x' of same shape) and optimise it so that
    teacher_AT.features(x_R) approx teacher_AT.features(x). Label of x_R is
    set to y (the original label of x). This is the Madry-feature-matching
    construction described in §3 of Ilyas et al.

  * D_NR is built by adversarial-relabel under the STD teacher: pick a
    random target t != y, run PGD on the STD teacher to relabel x -> t,
    and store the resulting (x_NR, t) pair. Per Ilyas, a vanilla model
    fit to this pair *still generalises* to the clean test set, because
    the non-robust features correlate with the assigned label.

  * Why this might fail on F-MNIST: Ilyas-style results were demonstrated
    on CIFAR-10 with deep ResNets. On grayscale 28x28 F-MNIST the
    information-theoretic gap between robust and non-robust features is
    much smaller (lower input dimension -> fewer redundant non-robust
    directions; cf. Tsipras et al. 2019 "There Is No Free Lunch In
    Adversarial Robustness", who give a Gaussian-toy theorem in which
    robust accuracy is bounded by clean accuracy minus a term that shrinks
    with input dimension). So we expect a weaker, possibly null, effect.

Extra papers cited:

  * Tsipras, Santurkar, Engstrom, Turner, Madry 2019, "Robustness May Be
    at Odds with Accuracy" (ICLR) - the accuracy-robustness tradeoff.
  * Tsipras et al. 2019, "There Is No Free Lunch In Adversarial
    Robustness" (formal version of the above) - dimension-dependent gap.
  * Allen-Zhu and Li 2022, "Feature Purification: How Adversarial Training
    Performs Robust Deep Learning" - mechanistic account of why AT erases
    non-robust features layer by layer.
  * Engstrom, Ilyas, Salman, Santurkar, Tsipras 2019, "Adversarial
    Robustness as a Prior for Learned Representations" - the AT-teacher
    feature space is more semantically aligned.

Controls / pipeline:

  (1) Train AT-teacher (PGD-AT, eps=0.1) and STD-teacher on the same F-MNIST
      training split with the same SmallCNN architecture.
  (2) Construct D_R via feature-distillation under the AT-teacher
      (matches teacher penultimate features; label = original y).
  (3) Construct D_NR via PGD adversarial relabel under the STD-teacher
      (perturb x to flip prediction to a random target t; store (x_NR, t)).
  (4) Train a fresh randomly-initialised SmallCNN *student* on each of:
      D_R, D_NR, and a clean baseline (the real F-MNIST train split).
      Measure clean accuracy and PGD-ASR on the real, untouched test set.
  (5) Per-class breakdown (correct / robust / ASR for each F-MNIST class)
      to see whether some classes carry more non-robust signal than others.

HEADLINE verdict: a student trained on D_R should achieve materially
higher PGD-robustness than the clean-baseline student despite NEVER seeing
an adversarial example during training; a student trained on D_NR should
achieve clean accuracy >> 50% on the real test set while having
near-zero PGD-robustness, demonstrating that non-robust features alone
are sufficient to generalise. Failure of either direction on F-MNIST would
be evidence that the Ilyas dichotomy is dataset/dimension-dependent (as
suggested by the Tsipras free-lunch theorem) and not a universal property
of NNs.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
EPS = 0.1                  # PGD L_inf budget (matches campaign default for F-MNIST)
PGD_STEPS = 20             # eval-time PGD
AT_PGD_STEPS = 7           # teacher AT inner-PGD
TEACHER_EPOCHS = 8
STUDENT_EPOCHS = 8
DR_STEPS = 100             # feature-distillation steps for D_R
DR_LR = 0.05               # PGD-style step size used in feature distillation
DNR_STEPS = 40             # PGD steps to build D_NR adversarial-relabel samples
N_TRAIN = 6000             # subset of F-MNIST train used for everything
N_EVAL = 2000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _features(model, x):
    """Penultimate features under SmallCNN: convolutional trunk + flatten +
    first Linear+Act, i.e. the 256-dim representation that feeds the final
    classifier. We pull it by slicing model.head."""
    f = model.features(x)
    flat = model.head[0](f)            # Flatten
    h = model.head[1](flat)            # Linear -> 256
    h = model.head[2](h)               # activation
    return h


def build_DR(teacher_at, X, Y, eps=EPS, steps=DR_STEPS, lr=DR_LR, seed=0):
    """Construct D_R via feature-distillation under the robust teacher.

    For each (x, y), initialise x_R from a *different* training sample of
    a *different* class (so non-robust features cannot leak from x), then
    optimise within [0,1] to minimise ||features_AT(x_R) - features_AT(x)||_2.
    The resulting (x_R, y) pair is the D_R datapoint.
    """
    g = torch.Generator(device=X.device).manual_seed(seed)
    n = X.size(0)
    X_R = torch.empty_like(X)
    # for each i, pick a starting image j with Y[j] != Y[i]
    perm = torch.randperm(n, generator=g, device=X.device)
    for k in range(n):
        if Y[perm[k]] == Y[k]:
            # swap with the next mismatched index (best-effort)
            for q in range(k + 1, n):
                if Y[perm[q]] != Y[k]:
                    perm[k], perm[q] = perm[q].clone(), perm[k].clone()
                    break

    teacher_at.eval()
    batch = 64
    for i in range(0, n, batch):
        x_tgt = X[i:i + batch]
        x_init = X[perm[i:i + batch]].clone()
        with torch.no_grad():
            f_tgt = _features(teacher_at, x_tgt).detach()
        xR = x_init.clone().detach().requires_grad_(True)
        opt = torch.optim.SGD([xR], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            f_cur = _features(teacher_at, xR)
            loss = F.mse_loss(f_cur, f_tgt)
            loss.backward()
            opt.step()
            with torch.no_grad():
                xR.clamp_(0, 1)
        X_R[i:i + batch] = xR.detach()
    return X_R, Y.clone()


def build_DNR(teacher_std, X, Y, ncls, eps=EPS, steps=DNR_STEPS, seed=0):
    """Construct D_NR by adversarial relabel under the standard teacher.

    For each (x, y), pick a random target t != y and run PGD targeted under
    teacher_std to push x into class t. The relabel-target t becomes the new
    label of x_NR; per Ilyas this label is *predictive* on the clean test
    set even though only non-robust features encode it.
    """
    g = torch.Generator(device=X.device).manual_seed(seed)
    n = X.size(0)
    # sample random non-self targets
    rnd = torch.randint(0, ncls, (n,), generator=g, device=X.device)
    clash = (rnd == Y)
    while clash.any():
        rnd[clash] = torch.randint(0, ncls, (int(clash.sum().item()),),
                                   generator=g, device=X.device)
        clash = (rnd == Y)

    teacher_std.eval()
    X_NR = torch.empty_like(X)
    batch = 128
    alpha = 2.5 * eps / steps
    for i in range(0, n, batch):
        x = X[i:i + batch].clone().detach()
        t = rnd[i:i + batch]
        x0 = x.clone()
        xa = x.clone() + torch.empty_like(x).uniform_(-eps, eps)
        xa = xa.clamp(0, 1)
        for _ in range(steps):
            xa.requires_grad_(True)
            # targeted PGD: descend cross-entropy to target t
            loss = F.cross_entropy(teacher_std(xa), t)
            (grad,) = torch.autograd.grad(loss, xa)
            xa = xa.detach() - alpha * grad.sign()
            xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
        X_NR[i:i + batch] = xa.detach()
    return X_NR, rnd


# ---------------------------------------------------------------------------
# main per-seed routine
# ---------------------------------------------------------------------------
def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    ncls = meta["n_classes"]
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)

    # (1) teachers
    t0 = time.time()
    teacher_at = C.build_model("cnn", meta, seed=seed)
    C.train_model(teacher_at, Xtr, Ytr, epochs=TEACHER_EPOCHS, opt="sgd", lr=0.05,
                  ncls=ncls, adv_train=True, adv_eps=EPS, adv_steps=AT_PGD_STEPS)
    teacher_std = C.build_model("cnn", meta, seed=seed)
    C.train_model(teacher_std, Xtr, Ytr, epochs=TEACHER_EPOCHS, opt="sgd", lr=0.05, ncls=ncls)
    t_teachers = time.time() - t0

    # baseline teacher metrics on the real test set
    _, at_clean = C.logits_and_acc(teacher_at, Xte, Yte)
    at_asr = C.attack_success(teacher_at, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)["asr"]
    _, std_clean = C.logits_and_acc(teacher_std, Xte, Yte)
    std_asr = C.attack_success(teacher_std, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)["asr"]

    # (2) D_R via feature-distillation under AT-teacher
    t0 = time.time()
    X_R, Y_R = build_DR(teacher_at, Xtr, Ytr, eps=EPS, steps=DR_STEPS, lr=DR_LR, seed=seed)
    t_dr = time.time() - t0

    # (3) D_NR via adversarial relabel under STD-teacher
    t0 = time.time()
    X_NR, Y_NR = build_DNR(teacher_std, Xtr, Ytr, ncls=ncls, eps=EPS,
                           steps=DNR_STEPS, seed=seed)
    t_dnr = time.time() - t0

    # (4) train three fresh students from scratch
    def fresh_student():
        C.set_seed(seed + 12345)
        return C.build_model("cnn", meta, seed=seed + 12345)

    # student_clean: real (Xtr, Ytr)
    s_clean = fresh_student()
    C.train_model(s_clean, Xtr, Ytr, epochs=STUDENT_EPOCHS, opt="sgd", lr=0.05, ncls=ncls)
    _, s_clean_acc = C.logits_and_acc(s_clean, Xte, Yte)
    s_clean_asr = C.attack_success(s_clean, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)["asr"]

    # student_DR: trained on (X_R, Y_R) only
    s_R = fresh_student()
    C.train_model(s_R, X_R, Y_R, epochs=STUDENT_EPOCHS, opt="sgd", lr=0.05, ncls=ncls)
    _, s_R_acc = C.logits_and_acc(s_R, Xte, Yte)
    s_R_asr = C.attack_success(s_R, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)["asr"]

    # student_DNR: trained on (X_NR, Y_NR) only
    s_NR = fresh_student()
    C.train_model(s_NR, X_NR, Y_NR, epochs=STUDENT_EPOCHS, opt="sgd", lr=0.05, ncls=ncls)
    _, s_NR_acc = C.logits_and_acc(s_NR, Xte, Yte)
    s_NR_asr = C.attack_success(s_NR, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)["asr"]

    # (5) per-class breakdown on the real test set
    def per_class(model):
        rows = []
        for c in range(ncls):
            m = (Yte == c)
            if m.sum() == 0:
                rows.append({"class": c, "n": 0, "clean": float("nan"),
                             "asr": float("nan"), "robust": float("nan")})
                continue
            Xc, Yc = Xte[m], Yte[m]
            _, acc = C.logits_and_acc(model, Xc, Yc)
            d = C.attack_success(model, Xc, Yc, attack="pgd", eps=EPS, steps=PGD_STEPS)
            rows.append({"class": c, "n": int(m.sum()), "clean": acc,
                         "asr": d["asr"], "robust": acc * (1 - d["asr"]) if d["asr"] == d["asr"] else float("nan")})
        return rows

    pc_clean = per_class(s_clean)
    pc_R = per_class(s_R)
    pc_NR = per_class(s_NR)

    return {
        "seed": seed,
        "teacher_AT": {"clean": at_clean, "asr": at_asr},
        "teacher_STD": {"clean": std_clean, "asr": std_asr},
        "student_clean": {"clean": s_clean_acc, "asr": s_clean_asr},
        "student_DR":    {"clean": s_R_acc,    "asr": s_R_asr},
        "student_DNR":   {"clean": s_NR_acc,   "asr": s_NR_asr},
        "per_class_clean": pc_clean,
        "per_class_DR":    pc_R,
        "per_class_DNR":   pc_NR,
        "runtime_s": {"teachers": round(t_teachers, 1),
                      "DR_build": round(t_dr, 1),
                      "DNR_build": round(t_dnr, 1)},
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _fmt_pc(rows, label):
    print(f"  {label:14s}  " + " ".join(f"c{r['class']}:cln={r['clean']:.2f}/asr={r['asr']:.2f}" for r in rows))


def main():
    print("=" * 78)
    print("H485 - Robust vs Non-Robust Features (Ilyas et al. 2019) on Fashion-MNIST")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  PGD_steps={PGD_STEPS}")
    print(f"N_train={N_TRAIN}  N_eval={N_EVAL}  teacher_epochs={TEACHER_EPOCHS}  "
          f"student_epochs={STUDENT_EPOCHS}")
    print(f"D_R: feature-distill steps={DR_STEPS}, lr={DR_LR}")
    print(f"D_NR: PGD steps={DNR_STEPS}, eps={EPS}")
    print()
    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["_wall"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"[seed {s}]  ({r['_wall']}s, teachers={r['runtime_s']['teachers']}s, "
              f"D_R build={r['runtime_s']['DR_build']}s, D_NR build={r['runtime_s']['DNR_build']}s)")
        print(f"  teacher_AT     clean={r['teacher_AT']['clean']:.3f}  PGD-ASR={r['teacher_AT']['asr']:.3f}")
        print(f"  teacher_STD    clean={r['teacher_STD']['clean']:.3f}  PGD-ASR={r['teacher_STD']['asr']:.3f}")
        print(f"  student_clean  clean={r['student_clean']['clean']:.3f}  PGD-ASR={r['student_clean']['asr']:.3f}")
        print(f"  student_D_R    clean={r['student_DR']['clean']:.3f}  PGD-ASR={r['student_DR']['asr']:.3f}"
              f"   <-- trained ONLY on feature-distilled (D_R), never on adversarial examples")
        print(f"  student_D_NR   clean={r['student_DNR']['clean']:.3f}  PGD-ASR={r['student_DNR']['asr']:.3f}"
              f"   <-- trained ONLY on adversarial-relabel pairs (D_NR)")
        print("  per-class summary (real test set):")
        _fmt_pc(r["per_class_clean"], "student_clean")
        _fmt_pc(r["per_class_DR"],    "student_D_R")
        _fmt_pc(r["per_class_DNR"],   "student_D_NR")
        print()

    # means
    def m(getter):
        v = [getter(r) for r in rows]
        v = [x for x in v if x == x]
        return sum(v) / len(v) if v else float("nan")

    print("=" * 78)
    print("MEANS across seeds")
    print(f"  teacher_AT     clean={m(lambda r: r['teacher_AT']['clean']):.3f}  "
          f"PGD-ASR={m(lambda r: r['teacher_AT']['asr']):.3f}")
    print(f"  teacher_STD    clean={m(lambda r: r['teacher_STD']['clean']):.3f}  "
          f"PGD-ASR={m(lambda r: r['teacher_STD']['asr']):.3f}")
    print(f"  student_clean  clean={m(lambda r: r['student_clean']['clean']):.3f}  "
          f"PGD-ASR={m(lambda r: r['student_clean']['asr']):.3f}")
    print(f"  student_D_R    clean={m(lambda r: r['student_DR']['clean']):.3f}  "
          f"PGD-ASR={m(lambda r: r['student_DR']['asr']):.3f}")
    print(f"  student_D_NR   clean={m(lambda r: r['student_DNR']['clean']):.3f}  "
          f"PGD-ASR={m(lambda r: r['student_DNR']['asr']):.3f}")
    print("=" * 78)
    print("HEADLINE: Ilyas et al. 2019 predicts that on Fashion-MNIST the D_R-only")
    print("student should pick up nontrivial PGD-robustness (PGD-ASR materially below")
    print("the clean-baseline student's) WITHOUT ever seeing an adversarial example,")
    print("and the D_NR-only student should still achieve clean accuracy >> chance on")
    print("the real test set while being trivially attacked (PGD-ASR near 1). A null")
    print("result on F-MNIST would corroborate Tsipras et al.'s dimension-dependent")
    print("free-lunch theorem: the robust/non-robust dichotomy may be weaker on")
    print("low-dimensional grayscale data than on CIFAR-10, and the Ilyas claim is")
    print("not universal. Either outcome is informative.")
    print("References: Ilyas et al. 1905.02175; Tsipras et al. 1805.12152 and")
    print("1809.10875; Engstrom et al. 1906.00945; Allen-Zhu & Li 2005.10190.")
    print("=" * 78)


if __name__ == "__main__":
    main()
