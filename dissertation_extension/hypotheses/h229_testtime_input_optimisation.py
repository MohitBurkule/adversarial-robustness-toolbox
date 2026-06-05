"""
H229 - Test-time input optimisation: fixed model, learn the input instead.

Train CNN, freeze it completely. At test time for each sample x_test:
  x_opt = x_test.clone().requires_grad_(True)
  optimizer_input = Adam([x_opt], lr=0.01)
  For T steps: minimise CE(model(x_opt), predicted_label); clamp to [0,1]
  x_final clamped to original ± eps ball

Measure at each T:
  A) Clean test samples: does margin improve with T?
  B) FGSM adversarials: does T steps of input optimisation recover clean prediction?
  C) PGD adversarials: same

Print table: T × experiment -> accuracy, mean_margin, recovery_rate.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_EVAL = 300
SEED = 0
EPS = 0.1
T_VALUES = [0, 5, 10, 20, 50]
INPUT_LR = 0.01

os.makedirs("results/fashion_mnist", exist_ok=True)


def input_optimise(model, x_batch, labels, T, eps, x_orig):
    """Optimise inputs for T steps; clamp to eps-ball around x_orig and [0,1]."""
    device = x_batch.device
    if T == 0:
        return x_batch.clone()

    results = []
    for i in range(x_batch.shape[0]):
        x0 = x_orig[i:i+1].clone().to(device)
        x_opt = x_batch[i:i+1].clone().detach().to(device)
        x_opt.requires_grad_(True)
        opt = torch.optim.Adam([x_opt], lr=INPUT_LR)
        lbl = labels[i:i+1].to(device)

        for _ in range(T):
            opt.zero_grad()
            loss = F.cross_entropy(model(x_opt), lbl)
            loss.backward()
            opt.step()
            with torch.no_grad():
                # project to eps-ball around original
                delta = (x_opt - x0).clamp(-eps, eps)
                x_opt.data = (x0 + delta).clamp(0, 1)

        results.append(x_opt.detach())
    return torch.cat(results, dim=0)


def evaluate(model, x_opt, y_true, x_clean):
    """Return accuracy, mean_margin, recovery_rate vs clean prediction."""
    device = next(model.parameters()).device
    x_opt = x_opt.to(device)
    y_true = y_true.to(device)
    x_clean = x_clean.to(device)

    with torch.no_grad():
        logits_opt = model(x_opt)
        logits_clean = model(x_clean)
        pred_opt = logits_opt.argmax(1)
        pred_clean = logits_clean.argmax(1)

    acc = float((pred_opt == y_true).float().mean())
    margin = C.margin(model, x_opt)
    mean_margin = float(margin.mean())
    # recovery: where was wrong on adversarial, now correct after opt
    recovery = float((pred_opt == y_true).float().mean())  # acc is recovery here
    return acc, mean_margin, recovery


def main():
    t0 = time.time()
    C.set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_EVAL, seed=SEED)
    Xte, Yte = Xte[:N_EVAL].to(device), Yte[:N_EVAL].to(device)
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)

    model = C.build_model("cnn", meta, width=32, seed=SEED)
    model = model.to(device)
    C.train_model(model, Xtr, Ytr, epochs=10)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Get model's own predictions on clean samples (used as optimisation target label)
    with torch.no_grad():
        pred_clean = model(Xte).argmax(1)

    # Generate adversarials
    print("Generating FGSM adversarials...")
    X_fgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    print("Generating PGD adversarials...")
    X_pgd = C.pgd(model, Xte, Yte, eps=EPS, steps=10, alpha=0.01)

    # Baseline accuracies
    with torch.no_grad():
        acc_clean_base = float((model(Xte).argmax(1) == Yte).float().mean())
        acc_fgsm_base = float((model(X_fgsm).argmax(1) == Yte).float().mean())
        acc_pgd_base = float((model(X_pgd).argmax(1) == Yte).float().mean())
    print(f"\nBaseline  clean={acc_clean_base:.3f}  fgsm={acc_fgsm_base:.3f}  pgd={acc_pgd_base:.3f}")

    header = f"\n{'T':>4}  {'Exp':<6}  {'Acc':>6}  {'Margin':>8}  {'RecRate':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for T in T_VALUES:
        for exp_name, X_input, X_orig_for_ball, label_src in [
            ("clean", Xte,    Xte,    pred_clean),
            ("fgsm",  X_fgsm, Xte,    Yte),
            ("pgd",   X_pgd,  Xte,    Yte),
        ]:
            x_opt = input_optimise(model, X_input, label_src, T, EPS, X_orig_for_ball)
            acc, mm, rec = evaluate(model, x_opt, Yte, Xte)
            rows.append((T, exp_name, acc, mm, rec))
            print(f"{T:>4}  {exp_name:<6}  {acc:>6.3f}  {mm:>8.4f}  {rec:>8.3f}")

    # Summary: for FGSM/PGD, show recovery vs T
    print("\n--- Recovery rate (correct prediction after opt) vs T for adversarials ---")
    for exp_name in ["fgsm", "pgd"]:
        print(f"\n{exp_name.upper()} adversarials:")
        for T, en, acc, mm, rec in rows:
            if en == exp_name:
                print(f"  T={T:>2}  acc={acc:.3f}  margin={mm:.4f}")

    # Does margin on clean improve with T?
    print("\n--- Clean margin vs T (does optimisation improve margin?) ---")
    for T, en, acc, mm, rec in rows:
        if en == "clean":
            print(f"  T={T:>2}  acc={acc:.3f}  margin={mm:.4f}")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
