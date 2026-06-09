"""
H207 - U-shaped adversarial robustness vs PCA dimensionality curve.

Hypothesis: FGSM ASR is minimized at an intermediate PCA dimensionality
k~32-64 (not at full 784), forming a U-shaped curve. The k that minimizes
ASR != the k that maximizes clean accuracy, demonstrating a robustness-
accuracy tradeoff in dimensionality space.

Grounded in: arXiv:2509.21130 (sparse representations + optimal dimensionality),
              arXiv:2502.15017 (PCA alignment correlates with robustness).

Methodology:
  - Fit PCA on Fashion-MNIST training set (n=6000)
  - For k in [4, 8, 16, 32, 64, 128, 256, 784]:
      project to k dims, train logistic regression, evaluate clean acc + FGSM ASR
  - FGSM in PCA space: perturb k-dim input by eps*sign(grad) via PyTorch autograd
  - Test: is fgsm_asr minimized at k in [16,64]? Is argmin(asr) < argmax(clean_acc)?

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign import common as C

SEED = 42
N_TRAIN = 6000
N_EVAL = 1000
EPS = 0.3
K_VALUES = [4, 8, 16, 32, 64, 128, 256, 784]
N_CLASSES = 10
CLASS_NAMES = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
               5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Boot"}


class LogisticRegression(nn.Module):
    """Simple logistic regression in PyTorch for gradient-based FGSM."""
    def __init__(self, in_dim, n_classes=10):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


def train_logreg(X, Y, in_dim, epochs=200, lr=0.1, batch=256):
    """Train logistic regression with SGD."""
    model = LogisticRegression(in_dim, N_CLASSES).to(C.DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=X.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out = model(X[idx])
            loss = F.cross_entropy(out, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def fgsm_pca(model, X, Y, eps):
    """FGSM attack in PCA space: perturb k-dim input."""
    X_adv = X.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(X_adv), Y)
    g, = torch.autograd.grad(loss, X_adv)
    return (X_adv + eps * g.sign()).detach()


def eval_fgsm_asr(model, X, Y, eps, batch=512):
    """FGSM ASR on correctly-classified samples in PCA space."""
    model.eval()
    all_correct = []
    all_flipped = []
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i+batch], Y[i:i+batch]
        with torch.no_grad():
            pred = model(xb).argmax(1)
            correct = pred == yb
        xb_adv = fgsm_pca(model, xb, yb, eps)
        with torch.no_grad():
            pred_adv = model(xb_adv).argmax(1)
            flipped = pred_adv != yb
        all_correct.append(correct.cpu())
        all_flipped.append(flipped.cpu())
    correct = torch.cat(all_correct).numpy().astype(bool)
    flipped = torch.cat(all_flipped).numpy()
    asr = float(flipped[correct].mean()) if correct.sum() > 0 else float("nan")
    return asr


def per_class_asr(model, X, Y, eps):
    """Per-class FGSM ASR."""
    results = {}
    for c in range(N_CLASSES):
        mask = Y == c
        if mask.sum() == 0:
            results[c] = float("nan")
            continue
        xc, yc = X[mask], Y[mask]
        results[c] = eval_fgsm_asr(model, xc, yc, eps)
    return results


def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # Flatten to 784-dim
    Xtr_flat = Xtr.cpu().reshape(Xtr.size(0), -1).numpy()
    Xte_flat = Xte.cpu().reshape(Xte.size(0), -1).numpy()

    print("=" * 74)
    print("H207 - U-shaped adversarial robustness vs PCA dimensionality")
    print("=" * 74)
    print(f"Device={C.DEVICE}  seed={SEED}  eps={EPS}")
    print(f"Train={N_TRAIN}  Eval={N_EVAL}  K values={K_VALUES}")

    # Fit PCA on full training set (keep max components)
    print("\nFitting PCA (784 components)...")
    t0 = time.time()
    pca = PCA(n_components=784, random_state=SEED)
    pca.fit(Xtr_flat)
    print(f"  PCA fit time: {time.time()-t0:.1f}s")

    # Explained variance summary
    for k in K_VALUES:
        ev = pca.explained_variance_ratio_[:k].sum()
        print(f"  k={k:>3}: explained variance = {ev:.4f}")

    results = []
    for k in K_VALUES:
        print(f"\n--- k={k} ---")
        t0 = time.time()

        # Project
        Xtr_k = pca.transform(Xtr_flat)[:, :k]
        Xte_k = pca.transform(Xte_flat)[:, :k]

        # Convert to torch
        Xtr_t = torch.tensor(Xtr_k, dtype=torch.float32, device=C.DEVICE)
        Xte_t = torch.tensor(Xte_k, dtype=torch.float32, device=C.DEVICE)
        Ytr_d = Ytr.to(C.DEVICE)
        Yte_d = Yte.to(C.DEVICE)

        # Train logistic regression
        model = train_logreg(Xtr_t, Ytr_d, k)

        # Clean accuracy
        with torch.no_grad():
            logits = model(Xte_t)
            clean_acc = (logits.argmax(1) == Yte_d).float().mean().item()

        # FGSM ASR in PCA space
        asr = eval_fgsm_asr(model, Xte_t, Yte_d, EPS)

        elapsed = time.time() - t0
        results.append({"k": k, "clean_acc": clean_acc, "fgsm_asr": asr})
        print(f"  clean_acc={clean_acc:.4f}  fgsm_asr={asr:.4f}  ({elapsed:.1f}s)")

    # --- Summary table ---
    print("\n" + "=" * 74)
    print("SUMMARY TABLE")
    print("=" * 74)
    print(f"{'k':>6} {'Clean acc':>10} {'FGSM ASR':>10}")
    print("-" * 30)
    for row in results:
        print(f"{row['k']:>6} {row['clean_acc']:>10.4f} {row['fgsm_asr']:>10.4f}")

    # --- Find optima ---
    clean_accs = [r["clean_acc"] for r in results]
    fgsm_asrs = [r["fgsm_asr"] for r in results]
    ks = [r["k"] for r in results]

    best_clean_k = ks[np.argmax(clean_accs)]
    best_rob_k = ks[np.argmin(fgsm_asrs)]

    print(f"\n  argmax(clean_acc) = k={best_clean_k} (acc={max(clean_accs):.4f})")
    print(f"  argmin(fgsm_asr)  = k={best_rob_k} (asr={min(fgsm_asrs):.4f})")

    # --- Hypothesis tests ---
    print("\n" + "=" * 74)
    print("HYPOTHESIS TESTS")
    print("=" * 74)
    u_shaped = best_rob_k in [16, 32, 64]
    tradeoff = best_rob_k != best_clean_k
    print(f"  ASR minimized at intermediate k (16-64): {'SUPPORTED' if u_shaped else 'NOT SUPPORTED'} (k={best_rob_k})")
    print(f"  argmin(ASR) != argmax(clean_acc): {'SUPPORTED' if tradeoff else 'NOT SUPPORTED'} ({best_rob_k} vs {best_clean_k})")
    print(f"  H207 overall: {'SUPPORTED' if (u_shaped and tradeoff) else 'PARTIALLY SUPPORTED' if (u_shaped or tradeoff) else 'NOT SUPPORTED'}")

    # --- Per-class analysis at optimal k ---
    print(f"\n--- Per-class FGSM ASR at optimal robustness k={best_rob_k} ---")
    Xte_opt = pca.transform(Xte_flat)[:, :best_rob_k]
    Xte_opt_t = torch.tensor(Xte_opt, dtype=torch.float32, device=C.DEVICE)
    model_opt = train_logreg(
        torch.tensor(pca.transform(Xtr_flat)[:, :best_rob_k], dtype=torch.float32, device=C.DEVICE),
        Ytr.to(C.DEVICE), best_rob_k
    )
    pc_asr = per_class_asr(model_opt, Xte_opt_t, Yte.to(C.DEVICE), EPS)

    # PCA variance per class
    print(f"{'Class':<12} {'FGSM ASR':>10}")
    print("-" * 24)
    for c in range(N_CLASSES):
        print(f"{CLASS_NAMES[c]:<12} {pc_asr[c]:>10.4f}")

    # Check if low-variance classes (Trouser=1, Bag=8) have lower ASR than
    # high-variance classes (Shirt=6, Pullover=2)
    low_var = [1, 8]  # Trouser, Bag
    high_var = [2, 6]  # Pullover, Shirt
    low_var_asr = np.mean([pc_asr[c] for c in low_var])
    high_var_asr = np.mean([pc_asr[c] for c in high_var])
    print(f"\n  Low-variance classes (Trouser, Bag) avg ASR: {low_var_asr:.4f}")
    print(f"  High-variance classes (Pullover, Shirt) avg ASR: {high_var_asr:.4f}")
    print(f"  Low-var < High-var: {'YES' if low_var_asr < high_var_asr else 'NO'}")


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "h207_pca_dimensionality_robustness_output.txt")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    text = buf.getvalue()
    print(text)
    with open(outpath, "w") as f:
        f.write(text)
    print(f"\nSaved to {outpath}")
