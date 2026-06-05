"""
H243 - Dual Model Disagreement: AT model vs clean model disagreement predicts vulnerability.

Train model_A on clean data, model_B on PGD-AT data (same architecture, same seed).
On clean test samples, measure prediction_disagreement = (argmax_A != argmax_B).
Does disagreement predict adversarial vulnerability of model_A?
Also: soft_disagreement = KL(softmax_A || softmax_B).
Measure: AUROC(disagreement → pgd_success_on_A).
"""
import os, sys, time
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
AT_EPOCHS = 10
BATCH_SIZE = 128

def train_pgd_at(model, Xtr, Ytr, epochs, eps=EPS, pgd_steps=5, lr=0.01):
    """PGD adversarial training."""
    device = next(model.parameters()).device
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    n = len(Xtr)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb = Xtr[idx]
            yb = Ytr[idx]
            # Generate PGD adversarial examples
            xb_adv = C.pgd(model, xb, yb, eps=eps, steps=pgd_steps, alpha=eps/4)
            model.train()
            optimizer.zero_grad()
            logits = model(xb_adv)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0:
            print(f"    AT Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

def kl_divergence(p_logits, q_logits):
    """KL(p || q) per sample. p, q are logit tensors."""
    p = F.softmax(p_logits, dim=1)
    log_p = F.log_softmax(p_logits, dim=1)
    log_q = F.log_softmax(q_logits, dim=1)
    return (p * (log_p - log_q)).sum(dim=1)

def main():
    print("=== H243: Dual Model Disagreement ===")
    t0 = time.time()

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]

    # [1] Train model_A on clean data
    print("\n[1] Training model_A on clean data...")
    model_A = C.build_model("cnn", meta, seed=SEED)
    C.train_model(model_A, Xtr, Ytr, epochs=EPOCHS)
    model_A.eval()
    with torch.no_grad():
        preds_A_clean = model_A(Xte_e).argmax(1).cpu()
    clean_acc_A = (preds_A_clean == Yte_e.cpu()).float().mean().item()
    print(f"    Model_A clean acc: {clean_acc_A:.3f}")

    # [2] Train model_B with PGD adversarial training
    print("\n[2] Training model_B with PGD adversarial training...")
    C.set_seed(SEED)
    model_B = C.build_model("cnn", meta, seed=SEED)
    train_pgd_at(model_B, Xtr, Ytr, epochs=AT_EPOCHS)
    model_B.eval()
    with torch.no_grad():
        preds_B_clean = model_B(Xte_e).argmax(1).cpu()
    clean_acc_B = (preds_B_clean == Yte_e.cpu()).float().mean().item()
    print(f"    Model_B (AT) clean acc: {clean_acc_B:.3f}")

    # [3] Compute disagreement metrics
    print("\n[3] Computing disagreement between model_A and model_B...")
    with torch.no_grad():
        logits_A = model_A(Xte_e)
        logits_B = model_B(Xte_e)

    pred_A = logits_A.argmax(1).cpu().numpy()
    pred_B = logits_B.argmax(1).cpu().numpy()
    hard_disagreement = (pred_A != pred_B).astype(int)

    kl_AB = kl_divergence(logits_A, logits_B).cpu().numpy()
    kl_BA = kl_divergence(logits_B, logits_A).cpu().numpy()
    sym_kl = (kl_AB + kl_BA) / 2

    print(f"    Hard disagreement rate: {hard_disagreement.mean():.3f}")
    print(f"    Symmetric KL: mean={sym_kl.mean():.4f}, "
          f"std={sym_kl.std():.4f}")

    # [4] PGD attack on model_A
    print("\n[4] Running PGD attack on model_A...")
    Xadv = C.pgd(model_A, Xte_e, Yte_e, eps=EPS, steps=10, alpha=0.01)
    model_A.eval()
    with torch.no_grad():
        pgd_succ = (model_A(Xadv).argmax(1).cpu() != Yte_e.cpu()).numpy().astype(int)
    print(f"    PGD ASR on model_A: {pgd_succ.mean():.3f}")

    # [5] AUROC metrics
    print("\n[5] Computing AUROCs...")
    auroc_hard = float('nan')
    auroc_kl = float('nan')
    rho_kl_margin = float('nan')

    margins_A = np.array(C.margin(model_A, Xte_e))
    auroc_margin = float('nan')

    if HAS_SKLEARN:
        try:
            if len(np.unique(pgd_succ)) == 2:
                auroc_hard = roc_auc_score(pgd_succ, hard_disagreement)
                auroc_kl = roc_auc_score(pgd_succ, sym_kl)
                auroc_margin = roc_auc_score(pgd_succ, -margins_A)
        except Exception:
            pass
        try:
            rho_kl_margin = spearmanr(sym_kl, margins_A).correlation
        except Exception:
            pass

    # Breakdown by disagreement
    print(f"\n    Agree (n={( hard_disagreement==0).sum()}): "
          f"PGD ASR={pgd_succ[hard_disagreement==0].mean():.3f}")
    print(f"    Disagree (n={(hard_disagreement==1).sum()}): "
          f"PGD ASR={pgd_succ[hard_disagreement==1].mean():.3f}")

    print(f"\n--- Summary ---")
    print(f"Model_A clean acc: {clean_acc_A:.3f}")
    print(f"Model_B (AT) clean acc: {clean_acc_B:.3f}")
    print(f"Hard disagreement rate: {hard_disagreement.mean():.3f}")
    print(f"PGD ASR on model_A: {pgd_succ.mean():.3f}")
    print(f"AUROC(hard_disagree → PGD success):  {auroc_hard:.3f}")
    print(f"AUROC(sym_KL → PGD success):         {auroc_kl:.3f}")
    print(f"AUROC(margin → PGD success):         {auroc_margin:.3f}  [baseline]")
    print(f"Spearman ρ(sym_KL, margin):          {rho_kl_margin:.3f}")
    print("Interpretation: AUROC>0.6 for disagreement => A/B disagreement "
          "identifies model_A's vulnerable samples.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
