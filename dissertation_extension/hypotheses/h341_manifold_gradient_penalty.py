"""H341: Manifold gradient penalty via PCA projection.

Project input gradient onto data manifold (top-K PCA components).
Penalty = ||g_manifold||² where g_manifold = P(P^T g).
K grid: [10, 50, 100]. λ=0.01.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

OUT = "results/fashion_mnist/h341_manifold_gradient_penalty_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
LAM = 0.01


def compute_pca_projection(Xtr, k):
    """Compute top-k PCA eigenvectors of flattened training data. Returns P: (D, K)."""
    X_flat = Xtr.cpu().float().flatten(1)  # (N, D)
    X_mean = X_flat.mean(0)
    X_centered = X_flat - X_mean
    # SVD: X_centered = U S V^T, top-k right singular vectors are principal components
    # Use torch.linalg.svd with full_matrices=False for efficiency
    _, _, Vt = torch.linalg.svd(X_centered, full_matrices=False)  # Vt: (min(N,D), D)
    P = Vt[:k].T  # (D, K)
    return P.to(C.DEVICE), X_mean.to(C.DEVICE)


def train_with_manifold_penalty(model, Xtr, Ytr, P, lam):
    """P: (D, K) projection matrix."""
    C.set_seed(SEED)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    D = P.shape[0]
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = Xtr[idx].clone().detach().requires_grad_(True)
            yb = Ytr[idx]
            opt.zero_grad()
            logits = model(xb)
            ce_loss = F.cross_entropy(logits, yb)
            if lam > 0:
                g, = torch.autograd.grad(ce_loss, xb, create_graph=True)
                g_flat = g.flatten(1)  # (B, D)
                # Project onto PCA subspace: g_proj = P @ (P^T @ g^T)
                # P: (D, K), g_flat: (B, D) -> g_flat @ P: (B, K)
                g_proj_coords = g_flat @ P  # (B, K)
                g_proj = g_proj_coords @ P.T  # (B, D)
                penalty = lam * (g_proj ** 2).sum(dim=1).mean()
                total_loss = ce_loss + penalty
            else:
                total_loss = ce_loss
            total_loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1-float(acc_fgsm), pgd_asr=1-float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))


def main():
    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    lines = ["H341: Manifold Gradient Penalty (PCA Projection)", "="*60]
    lines.append(f"Settings: N_TRAIN={N_TRAIN}, EPOCHS={EPOCHS}, LR={LR}, BATCH={BATCH}, λ={LAM}")
    lines.append("Penalty = ||P(P^T ∇_x L)||² where P = top-K PCA eigenvectors\n")

    # Baseline (no penalty)
    print("Training baseline (λ=0)...")
    C.set_seed(SEED)
    model_base = C.build_model("cnn", meta)
    P_base, _ = compute_pca_projection(Xtr, k=10)  # dummy P, won't be used
    train_with_manifold_penalty(model_base, Xtr, Ytr, P=P_base, lam=0)
    r_base = eval_model(model_base, Xte, Yte)
    results = {"baseline_k=0": r_base}
    lines.append(f"baseline (λ=0): clean={r_base['clean_acc']:.4f} fgsm_asr={r_base['fgsm_asr']:.4f} pgd_asr={r_base['pgd_asr']:.4f} margin={r_base['mean_margin']:.4f}")

    k_grid = [10, 50, 100]
    for k in k_grid:
        print(f"Computing PCA (K={k})...")
        P, _ = compute_pca_projection(Xtr, k=k)
        print(f"Training K={k}, λ={LAM}...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_with_manifold_penalty(model, Xtr, Ytr, P=P, lam=LAM)
        r = eval_model(model, Xte, Yte)
        key = f"K={k}"
        results[key] = r
        lines.append(f"{key}: clean={r['clean_acc']:.4f} fgsm_asr={r['fgsm_asr']:.4f} pgd_asr={r['pgd_asr']:.4f} margin={r['mean_margin']:.4f}")

    lines.append("\n" + "="*60)
    lines.append("SUMMARY")
    lines.append(f"{'Condition':<15} {'Clean':>8} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Margin':>10}")
    for key, r in results.items():
        lines.append(f"{key:<15} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f}")

    out_text = "\n".join(lines)
    print(out_text)
    with open(OUT, "w") as f:
        f.write(out_text)
    print(f"\nResults written to {OUT}")

if __name__ == "__main__":
    main()
