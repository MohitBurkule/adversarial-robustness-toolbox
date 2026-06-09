"""
H245 - Generalisation Gap & Robustness: does train-test gap rank predict PGD ASR rank?

Train models with varying regularisation: no_reg, dropout=0.3, dropout=0.5,
weight_decay=1e-4, weight_decay=1e-2, early_stop_epoch=5.
Measure generalisation_gap = train_acc - test_acc.
Does generalisation_gap rank predict pgd_asr rank?
Does reducing the gap preserve the per-sample vulnerability ranking?
Measure rank correlation of vulnerability vectors across model pairs.
"""
import os, sys, time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

try:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

SEED = 0
EPS = 0.1
N_EVAL = 300
EPOCHS = 10
BATCH_SIZE = 128

def add_dropout(model, p):
    """Insert dropout after each ReLU in the model (shallow copy approach)."""
    # We rebuild by wrapping the model in a sequential with dropout
    layers = []
    for module in model.children():
        layers.append(module)
        if isinstance(module, nn.ReLU):
            layers.append(nn.Dropout(p=p))
    if layers:
        return nn.Sequential(*layers)
    return model

def train_custom(model, Xtr, Ytr, epochs, weight_decay=0.0, lr=0.01):
    """Custom training loop with configurable weight decay."""
    device = next(model.parameters()).device
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay)
    n = len(Xtr)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]
            optimizer.zero_grad()
            logits = model(xb)
            F.cross_entropy(logits, yb).backward()
            optimizer.step()

def eval_acc(model, X, Y, batch=256):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb, yb = X[i:i+batch], Y[i:i+batch]
            correct += (model(xb).argmax(1).cpu() == yb.cpu()).sum().item()
    return correct / len(X)

def get_vuln_vector(model, X, Y):
    """Return per-sample PGD success indicator."""
    Xadv = C.pgd(model, X, Y, eps=EPS, steps=10, alpha=0.01)
    model.eval()
    with torch.no_grad():
        preds = model(Xadv).argmax(1).cpu()
    return (preds != Y.cpu()).numpy().astype(float)

def main():
    print("=== H245: Generalisation Gap & Robustness ===")
    t0 = time.time()

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]

    configs = [
        ('no_reg',          {'epochs': EPOCHS, 'weight_decay': 0.0}),
        ('dropout_0.3',     {'epochs': EPOCHS, 'weight_decay': 0.0, 'dropout': 0.3}),
        ('dropout_0.5',     {'epochs': EPOCHS, 'weight_decay': 0.0, 'dropout': 0.5}),
        ('wd_1e-4',         {'epochs': EPOCHS, 'weight_decay': 1e-4}),
        ('wd_1e-2',         {'epochs': EPOCHS, 'weight_decay': 1e-2}),
        ('early_stop_e5',   {'epochs': 5,      'weight_decay': 0.0}),
    ]

    results = []
    vuln_vectors = {}

    for name, cfg in configs:
        print(f"\n[Training: {name}]")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, seed=SEED)

        dropout = cfg.get('dropout', None)
        if dropout is not None:
            model = add_dropout(model, dropout)

        train_custom(model, Xtr, Ytr, epochs=cfg['epochs'],
                     weight_decay=cfg['weight_decay'])

        train_acc = eval_acc(model, Xtr[:2000], Ytr[:2000])
        test_acc = eval_acc(model, Xte_e, Yte_e)
        gen_gap = train_acc - test_acc

        vuln = get_vuln_vector(model, Xte_e, Yte_e)
        pgd_asr = vuln.mean()

        auroc = float('nan')
        if HAS_SKLEARN:
            try:
                margins = np.array(C.margin(model, Xte_e))
                if len(np.unique(vuln.astype(int))) == 2:
                    auroc = roc_auc_score(vuln, -margins)
            except Exception:
                pass

        results.append({
            'name': name, 'train_acc': train_acc, 'test_acc': test_acc,
            'gen_gap': gen_gap, 'pgd_asr': pgd_asr, 'auroc': auroc,
        })
        vuln_vectors[name] = vuln
        print(f"  train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, "
              f"gen_gap={gen_gap:.3f}, PGD_ASR={pgd_asr:.3f}, "
              f"margin_AUROC={auroc:.3f}")

    # [Rank correlation: gen_gap → PGD ASR]
    gen_gaps = [r['gen_gap'] for r in results]
    pgd_asrs = [r['pgd_asr'] for r in results]
    rho_gap_asr = float('nan')
    if HAS_SKLEARN:
        try:
            rho_gap_asr = spearmanr(gen_gaps, pgd_asrs).correlation
        except Exception:
            pass

    # [Cross-model vulnerability ranking correlation]
    print("\n[Cross-model Spearman ρ of vulnerability vectors]")
    names = [cfg[0] for cfg in configs]
    header = "           " + "".join(f"{n[:8]:>10}" for n in names)
    print(header)
    rho_matrix = {}
    for n1 in names:
        row = f"  {n1[:8]:>8}:"
        for n2 in names:
            rho = float('nan')
            if HAS_SKLEARN:
                try:
                    rho = spearmanr(vuln_vectors[n1], vuln_vectors[n2]).correlation
                except Exception:
                    pass
            row += f"{rho:>10.3f}"
            rho_matrix[(n1, n2)] = rho
        print(row)

    print(f"\n--- Summary ---")
    print(f"{'Model':>18} | {'gen_gap':>8} | {'PGD_ASR':>8} | {'margin_AUROC':>12}")
    print("-" * 56)
    for r in results:
        print(f"  {r['name']:>16} | {r['gen_gap']:>8.3f} | "
              f"{r['pgd_asr']:>8.3f} | {r['auroc']:>12.3f}")
    print(f"\nSpearman ρ(gen_gap, PGD_ASR): {rho_gap_asr:.3f}")
    print(f"Cross-model vulnerability ρ(no_reg, early_stop_e5): "
          f"{rho_matrix.get(('no_reg','early_stop_e5'), float('nan')):.3f}")
    print("Interpretation: positive ρ(gap,ASR) => overfitting correlates with "
          "vulnerability; high cross-model ρ => same samples always vulnerable.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
