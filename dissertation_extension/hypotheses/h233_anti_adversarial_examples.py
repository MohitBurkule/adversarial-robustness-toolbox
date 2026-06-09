"""
H233 - Anti-adversarial (friendly) examples: train on gradient-direction samples.

Anti-adversarial = x_anti = x - eps * sign(grad_x L(model(x), y))
  [move AWAY from decision boundary = increase margin]

Train 4 models:
  1. baseline:       standard training on x
  2. anti_adversarial: train on x_anti only
  3. mixed:          train on concat(x, x_anti) with equal weight
  4. top20pct:       train only on top-20% highest-margin samples

Evaluate all 4:
  - clean accuracy, FGSM ASR, PGD ASR, mean margin

Hypothesis A: anti-adversarial training improves generalisation but reduces robustness
Hypothesis B: top-20% training has highest clean acc but worst robustness

Also: margin distribution comparison for model 2 vs baseline.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_EVAL = 300
SEED = 0
EPS = 0.1
PGD_STEPS = 10
EPOCHS = 10

os.makedirs("results/fashion_mnist", exist_ok=True)


def compute_anti_adversarial(model, X, Y, eps=EPS):
    """Compute x_anti = x - eps * sign(grad_x L), moving away from boundary."""
    model.eval()
    Xout = []
    batch = 128
    for i in range(0, X.size(0), batch):
        xb = X[i:i + batch].clone().detach().requires_grad_(True)
        yb = Y[i:i + batch]
        loss = F.cross_entropy(model(xb), yb)
        g, = torch.autograd.grad(loss, xb)
        xa = (xb - eps * g.sign()).clamp(0, 1).detach()
        Xout.append(xa)
    return torch.cat(Xout)


def get_top_margin_indices(model, X, Y, top_frac=0.20):
    """Return indices of top fraction by margin (most naturally robust samples)."""
    m = C.margin(model, X, Y)
    k = max(1, int(len(m) * top_frac))
    idx = np.argsort(m)[-k:]
    return idx


def evaluate_model(model, Xte, Yte, n_eval=N_EVAL, eps=EPS, steps=PGD_STEPS):
    model.eval()
    X = Xte[:n_eval]
    Y = Yte[:n_eval]

    with torch.no_grad():
        logits, clean_acc = C.logits_and_acc(model, X, Y)

    fgsm_res = C.attack_success(model, X, Y, attack="fgsm", eps=eps)
    pgd_res = C.attack_success(model, X, Y, attack="pgd", eps=eps, steps=steps)

    mg = C.margin(model, X, Y)
    mean_margin = float(np.mean(mg))

    return {
        "clean_acc": clean_acc,
        "fgsm_asr": fgsm_res["asr"],
        "pgd_asr": pgd_res["asr"],
        "mean_margin": mean_margin,
        "margin_dist": mg,
    }


def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_EVAL, seed=SEED)

    print("H233 - Anti-Adversarial Examples Training")
    print("=" * 70)
    print(f"Dataset: {DS}, N_EVAL={N_EVAL}, EPS={EPS}, PGD_STEPS={PGD_STEPS}, EPOCHS={EPOCHS}")

    # ---- Model 1: Baseline ------------------------------------------------
    print("\n[1/4] Training baseline model...")
    model_base = C.build_model("cnn", meta, width=32, seed=SEED)
    C.train_model(model_base, Xtr, Ytr, epochs=EPOCHS, ncls=meta["n_classes"])
    res_base = evaluate_model(model_base, Xte, Yte)
    print(f"  Baseline: clean={res_base['clean_acc']:.4f}, fgsm_asr={res_base['fgsm_asr']:.4f}, "
          f"pgd_asr={res_base['pgd_asr']:.4f}, margin={res_base['mean_margin']:.4f}")

    # ---- Model 2: Anti-adversarial ----------------------------------------
    print("\n[2/4] Computing anti-adversarial training images...")
    # Need a warm model; use baseline to compute gradient directions
    Xtr_anti = compute_anti_adversarial(model_base, Xtr, Ytr, eps=EPS)

    print("  Training anti-adversarial model on x_anti only...")
    model_anti = C.build_model("cnn", meta, width=32, seed=SEED)
    C.train_model(model_anti, Xtr_anti, Ytr, epochs=EPOCHS, ncls=meta["n_classes"])
    res_anti = evaluate_model(model_anti, Xte, Yte)
    print(f"  Anti-adv:  clean={res_anti['clean_acc']:.4f}, fgsm_asr={res_anti['fgsm_asr']:.4f}, "
          f"pgd_asr={res_anti['pgd_asr']:.4f}, margin={res_anti['mean_margin']:.4f}")

    # ---- Model 3: Mixed ---------------------------------------------------
    print("\n[3/4] Training mixed model on concat(x, x_anti)...")
    Xtr_mixed = torch.cat([Xtr, Xtr_anti], dim=0)
    Ytr_mixed = torch.cat([Ytr, Ytr], dim=0)
    model_mixed = C.build_model("cnn", meta, width=32, seed=SEED)
    C.train_model(model_mixed, Xtr_mixed, Ytr_mixed, epochs=EPOCHS, ncls=meta["n_classes"])
    res_mixed = evaluate_model(model_mixed, Xte, Yte)
    print(f"  Mixed:     clean={res_mixed['clean_acc']:.4f}, fgsm_asr={res_mixed['fgsm_asr']:.4f}, "
          f"pgd_asr={res_mixed['pgd_asr']:.4f}, margin={res_mixed['mean_margin']:.4f}")

    # ---- Model 4: Top-20% margin ------------------------------------------
    print("\n[4/4] Selecting top-20% highest-margin training samples...")
    top_idx = get_top_margin_indices(model_base, Xtr, Ytr, top_frac=0.20)
    Xtr_top = Xtr[top_idx]
    Ytr_top = Ytr[top_idx]
    print(f"  Using {len(top_idx)} samples (from {Xtr.size(0)} total)")
    model_top = C.build_model("cnn", meta, width=32, seed=SEED)
    C.train_model(model_top, Xtr_top, Ytr_top, epochs=EPOCHS, ncls=meta["n_classes"])
    res_top = evaluate_model(model_top, Xte, Yte)
    print(f"  Top20pct:  clean={res_top['clean_acc']:.4f}, fgsm_asr={res_top['fgsm_asr']:.4f}, "
          f"pgd_asr={res_top['pgd_asr']:.4f}, margin={res_top['mean_margin']:.4f}")

    # ---- Summary table ----------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print(f"{'Model':<20}  {'clean_acc':>10}  {'fgsm_asr':>9}  {'pgd_asr':>8}  {'mean_margin':>12}")
    print("-" * 70)
    for name, res in [("baseline", res_base), ("anti_adversarial", res_anti),
                      ("mixed", res_mixed), ("top20pct", res_top)]:
        print(f"{name:<20}  {res['clean_acc']:>10.4f}  {res['fgsm_asr']:>9.4f}  "
              f"{res['pgd_asr']:>8.4f}  {res['mean_margin']:>12.4f}")

    # ---- Hypothesis checks ------------------------------------------------
    print("\nHypothesis A: anti-adversarial training improves clean acc, reduces robustness")
    hyp_a_clean = res_anti["clean_acc"] > res_base["clean_acc"]
    hyp_a_robust = res_anti["pgd_asr"] > res_base["pgd_asr"]
    print(f"  Clean acc: anti={res_anti['clean_acc']:.4f} vs base={res_base['clean_acc']:.4f} -> "
          f"{'HIGHER (Hyp A partial)' if hyp_a_clean else 'NOT higher'}")
    print(f"  PGD ASR:   anti={res_anti['pgd_asr']:.4f} vs base={res_base['pgd_asr']:.4f} -> "
          f"{'higher ASR = less robust (Hyp A partial)' if hyp_a_robust else 'NOT less robust'}")
    print(f"  -> Hypothesis A: {'SUPPORTED' if hyp_a_clean and hyp_a_robust else 'NOT SUPPORTED'}")

    print("\nHypothesis B: top-20% has highest clean acc but worst robustness")
    hyp_b_clean = res_top["clean_acc"] >= max(res_base["clean_acc"], res_anti["clean_acc"], res_mixed["clean_acc"])
    hyp_b_robust = res_top["pgd_asr"] >= max(res_base["pgd_asr"], res_anti["pgd_asr"], res_mixed["pgd_asr"])
    print(f"  Top20pct clean_acc={res_top['clean_acc']:.4f} (highest? {hyp_b_clean})")
    print(f"  Top20pct PGD ASR={res_top['pgd_asr']:.4f} (worst robustness? {hyp_b_robust})")
    print(f"  -> Hypothesis B: {'SUPPORTED' if hyp_b_clean and hyp_b_robust else 'NOT SUPPORTED'}")

    # ---- Margin distribution shift ----------------------------------------
    print("\nMargin distribution shift (model 2 anti_adversarial vs baseline):")
    mb = res_base["margin_dist"]
    ma = res_anti["margin_dist"]
    print(f"  Baseline:       mean={mb.mean():.4f}, std={mb.std():.4f}, "
          f"pct_neg={float((mb < 0).mean()):.4f}")
    print(f"  Anti-adversarial: mean={ma.mean():.4f}, std={ma.std():.4f}, "
          f"pct_neg={float((ma < 0).mean()):.4f}")
    shift = ma.mean() - mb.mean()
    print(f"  Margin shift: {shift:+.4f} ({'positive = anti-adv increases margins' if shift > 0 else 'negative = anti-adv decreases margins'})")


if __name__ == "__main__":
    main()
