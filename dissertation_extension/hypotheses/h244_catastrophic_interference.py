"""
H244 - Catastrophic Interference: fine-tuning on one class degrades others.

Take trained model. For each class c, fine-tune for 1 gradient step on 10 samples
of class c only. Measure margin change on OTHER classes.
margin_change(x_j) = margin_after(x_j) - margin_before(x_j) for x_j not in class c.
Does magnitude of margin_change predict which samples were adversarially vulnerable
in the original model? Repeat for each class c, aggregate interference signal.
"""
import os, sys, time
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

try:
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

SEED = 0
EPS = 0.1
N_EVAL = 300
EPOCHS = 10
N_FINETUNE_SAMPLES = 10
FINETUNE_STEPS = 1
LR_FINETUNE = 0.05

def finetune_one_step(model, X_class, Y_class, lr=LR_FINETUNE):
    """Fine-tune model on class samples for one gradient step. Returns modified copy."""
    model_ft = copy.deepcopy(model)
    model_ft.train()
    optimizer = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)
    for _ in range(FINETUNE_STEPS):
        optimizer.zero_grad()
        logits = model_ft(X_class)
        loss = F.cross_entropy(logits, Y_class)
        loss.backward()
        optimizer.step()
    return model_ft

def main():
    print("=== H244: Catastrophic Interference ===")
    t0 = time.time()

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]
    n_classes = meta['n_classes']

    # [1] Train baseline model
    print("\n[1] Training baseline model...")
    model = C.build_model("cnn", meta, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=EPOCHS)

    # [2] Baseline metrics: margin and PGD vulnerability
    print("[2] Computing baseline margins and PGD vulnerability...")
    margins_base = np.array(C.margin(model, Xte_e))

    Xadv = C.pgd(model, Xte_e, Yte_e, eps=EPS, steps=10, alpha=0.01)
    model.eval()
    with torch.no_grad():
        pgd_succ = (model(Xadv).argmax(1).cpu() != Yte_e.cpu()).numpy().astype(int)
    print(f"    PGD ASR: {pgd_succ.mean():.3f}")

    # [3] For each class c, fine-tune on N_FINETUNE_SAMPLES and measure interference
    print(f"\n[3] Fine-tuning on each class ({FINETUNE_STEPS} step(s), "
          f"{N_FINETUNE_SAMPLES} samples) and measuring margin change...")

    # Aggregate interference: max |margin_change| per test sample across classes
    aggregate_interference = np.zeros(N_EVAL)

    class_results = []
    for c in range(n_classes):
        # Select N_FINETUNE_SAMPLES from training set for class c
        class_mask = (Ytr == c)
        class_idx = torch.where(class_mask)[0]
        n_avail = len(class_idx)
        sel = class_idx[:min(N_FINETUNE_SAMPLES, n_avail)]
        X_c = Xtr[sel]
        Y_c = Ytr[sel]

        # Fine-tune copy
        model_ft = finetune_one_step(model, X_c, Y_c)
        model_ft.eval()

        # Margins after fine-tuning on OTHER classes
        other_mask = (Yte_e != c).cpu().numpy()
        if other_mask.sum() == 0:
            continue

        Xte_other = Xte_e[torch.from_numpy(other_mask).to(Xte_e.device)]
        margins_ft_other = np.array(C.margin(model_ft, Xte_other))
        margins_base_other = margins_base[other_mask]

        delta = margins_ft_other - margins_base_other
        abs_delta = np.abs(delta)

        # Update aggregate interference
        aggregate_interference[other_mask] = np.maximum(
            aggregate_interference[other_mask], abs_delta
        )

        mean_delta = delta.mean()
        mean_abs = abs_delta.mean()
        class_results.append({'class': c, 'mean_delta': mean_delta,
                               'mean_abs_delta': mean_abs})
        print(f"    Class {c}: mean margin change={mean_delta:+.4f}, "
              f"mean |delta|={mean_abs:.4f}")

    # [4] Does aggregate interference predict PGD vulnerability?
    print("\n[4] Correlating aggregate interference with PGD vulnerability...")
    auroc_interf = float('nan')
    rho_interf_margin = float('nan')
    rho_interf_pgd = float('nan')

    if HAS_SKLEARN:
        try:
            if len(np.unique(pgd_succ)) == 2:
                auroc_interf = roc_auc_score(pgd_succ, aggregate_interference)
        except Exception:
            pass
        try:
            rho_interf_margin = spearmanr(aggregate_interference, margins_base).correlation
        except Exception:
            pass
        try:
            rho_interf_pgd = spearmanr(aggregate_interference, pgd_succ).correlation
        except Exception:
            pass

    # Baseline AUROC
    auroc_margin = float('nan')
    if HAS_SKLEARN:
        try:
            if len(np.unique(pgd_succ)) == 2:
                auroc_margin = roc_auc_score(pgd_succ, -margins_base)
        except Exception:
            pass

    print(f"\n--- Summary ---")
    print(f"PGD ASR: {pgd_succ.mean():.3f}")
    print(f"Mean aggregate interference: {aggregate_interference.mean():.4f}")
    print(f"AUROC(interference → PGD success):  {auroc_interf:.3f}")
    print(f"AUROC(margin → PGD success):        {auroc_margin:.3f}  [baseline]")
    print(f"Spearman ρ(interference, margin):   {rho_interf_margin:.3f}")
    print(f"Spearman ρ(interference, pgd_succ): {rho_interf_pgd:.3f}")
    print("Interpretation: high interference = sample near a class boundary "
          "whose margin shifts when a neighbouring class is reinforced, "
          "suggesting original model relied on fragile decision geometry.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
