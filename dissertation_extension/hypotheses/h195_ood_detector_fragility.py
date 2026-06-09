"""
H195 - OOD detector fragility under adversarial attack.

Hypothesis: a max-softmax OOD detector is adversarially fragile -- >50% of
in-distribution samples can be re-classified as OOD with a small L-inf
perturbation (eps=0.1), consistent with arXiv:2406.15104's finding that
detectors drop from ~90% to ~55% AUROC under white-box attack.

Methodology:
  - Hold out class 0 (T-shirt) as "OOD". Train CNN on classes 1-9.
  - Baseline OOD detector: max softmax score < threshold -> "OOD".
    Set threshold at 95% TPR on clean in-distribution data.
  - Measure baseline AUROC (OOD vs in-dist discrimination).
  - Attack in-dist samples: PGD to minimize max-softmax score (push toward OOD).
  - Attack OOD samples: PGD to maximize max-softmax score (push toward in-dist).
  - Report: baseline AUROC, post-attack AUROC, fraction fooled each direction.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH = 128
EPS = 0.1
PGD_STEPS = 20
PGD_ALPHA = 0.01
N_TRAIN = 6000
N_EVAL_INDIST = 500
N_EVAL_OOD = 200
N_CLASSES_TRAIN = 9  # classes 1-9
OOD_CLASS = 0
SEED = 42


class CNN(nn.Module):
    def __init__(self, n=9):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, n)

    def forward(self, x):
        x = F.relu(self.bn1(self.c1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.c2(x)))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_fashion_mnist():
    tf = transforms.ToTensor()
    tr = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
    te = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))])
    Ytr = torch.tensor([tr[i][1] for i in range(len(tr))])
    Xte = torch.stack([te[i][0] for i in range(len(te))])
    Yte = torch.tensor([te[i][1] for i in range(len(te))])
    return Xtr, Ytr, Xte, Yte


def max_softmax_score(model, X, batch=256):
    """Return max softmax probability for each sample."""
    scores = []
    for i in range(0, X.size(0), batch):
        with torch.no_grad():
            logits = model(X[i:i+batch])
            probs = F.softmax(logits, dim=1)
            scores.append(probs.max(dim=1).values.cpu())
    return torch.cat(scores)


def pgd_minimize_max_softmax(model, x, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """PGD attack to MINIMIZE max softmax score (push in-dist toward OOD)."""
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        logits = model(xa)
        # Minimize max softmax = maximize negative of max softmax
        max_prob = F.softmax(logits, dim=1).max(dim=1).values
        loss = max_prob.sum()  # we want to minimize this
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() - alpha * g.sign()  # gradient descent (minimize)
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def pgd_maximize_max_softmax(model, x, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """PGD attack to MAXIMIZE max softmax score (push OOD toward in-dist)."""
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        logits = model(xa)
        max_prob = F.softmax(logits, dim=1).max(dim=1).values
        loss = max_prob.sum()  # we want to maximize this
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()  # gradient ascent (maximize)
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def safe_auroc(labels, scores):
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def main():
    t0 = time.time()
    print("=" * 74)
    print("H195 - OOD detector fragility under adversarial attack")
    print("=" * 74)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Xtr_full, Ytr_full, Xte_full, Yte_full = load_fashion_mnist()

    # Split: in-dist = classes 1-9, OOD = class 0
    tr_indist = Ytr_full != OOD_CLASS
    te_indist = Yte_full != OOD_CLASS
    te_ood = Yte_full == OOD_CLASS

    Xtr_id, Ytr_id = Xtr_full[tr_indist], Ytr_full[tr_indist]
    # Remap labels: class 1->0, 2->1, ..., 9->8
    Ytr_id = Ytr_id - 1

    # Subsample training set
    g = torch.Generator().manual_seed(SEED)
    if N_TRAIN < Xtr_id.size(0):
        idx = torch.randperm(Xtr_id.size(0), generator=g)[:N_TRAIN]
        Xtr_id, Ytr_id = Xtr_id[idx], Ytr_id[idx]

    # Eval sets
    Xte_id, Yte_id = Xte_full[te_indist], Yte_full[te_indist]
    Yte_id = Yte_id - 1  # remap
    Xte_ood = Xte_full[te_ood]

    # Subsample eval
    if N_EVAL_INDIST < Xte_id.size(0):
        idx = torch.randperm(Xte_id.size(0), generator=g)[:N_EVAL_INDIST]
        Xte_id, Yte_id = Xte_id[idx], Yte_id[idx]
    if N_EVAL_OOD < Xte_ood.size(0):
        idx = torch.randperm(Xte_ood.size(0), generator=g)[:N_EVAL_OOD]
        Xte_ood = Xte_ood[idx]

    Xtr_id, Ytr_id = Xtr_id.to(DEVICE), Ytr_id.to(DEVICE)
    Xte_id, Yte_id = Xte_id.to(DEVICE), Yte_id.to(DEVICE)
    Xte_ood = Xte_ood.to(DEVICE)

    print(f"\n  Train: {Xtr_id.size(0)} in-dist samples (classes 1-9)")
    print(f"  Eval:  {Xte_id.size(0)} in-dist + {Xte_ood.size(0)} OOD (class 0)")

    # Train
    print("\n--- Training CNN on classes 1-9 ---")
    model = CNN(n=N_CLASSES_TRAIN).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr_id.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(Xtr_id[idx]), Ytr_id[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()

    with torch.no_grad():
        acc = float((model(Xte_id).argmax(1) == Yte_id).float().mean())
    print(f"  Clean accuracy on in-dist test: {acc:.4f}")

    # Baseline OOD detection
    print("\n--- Baseline OOD detection (max softmax) ---")
    scores_id = max_softmax_score(model, Xte_id).numpy()
    scores_ood = max_softmax_score(model, Xte_ood).numpy()

    # Set threshold at 95% TPR on in-dist (95% of in-dist above threshold)
    threshold = np.percentile(scores_id, 5)  # 5th percentile -> 95% TPR
    print(f"  Threshold (95% TPR on in-dist): {threshold:.4f}")
    print(f"  Mean max-softmax in-dist: {scores_id.mean():.4f}")
    print(f"  Mean max-softmax OOD:     {scores_ood.mean():.4f}")

    # AUROC: label 0 = in-dist (high score), 1 = OOD (low score)
    # For AUROC: we want score to discriminate OOD (1) vs in-dist (0)
    # OOD should have LOWER scores, so negate for AUROC convention
    labels_baseline = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    scores_baseline = np.concatenate([-scores_id, -scores_ood])  # negate: OOD gets higher neg-score
    auroc_baseline = safe_auroc(labels_baseline, scores_baseline)
    print(f"  Baseline AUROC (OOD detection): {auroc_baseline:.4f}")

    baseline_ood_detected = (scores_ood < threshold).mean()
    print(f"  Baseline OOD detected (score < threshold): {baseline_ood_detected:.4f}")

    # Attack 1: push in-dist samples toward OOD (minimize max softmax)
    print("\n--- Attack: push in-dist -> OOD (minimize max softmax) ---")
    attacked_id = []
    for i in range(0, Xte_id.size(0), 256):
        xb = Xte_id[i:i+256]
        xa = pgd_minimize_max_softmax(model, xb)
        attacked_id.append(xa)
    attacked_id = torch.cat(attacked_id)
    scores_id_attacked = max_softmax_score(model, attacked_id).numpy()

    frac_id_fooled = (scores_id_attacked < threshold).mean()
    print(f"  Fraction in-dist fooled (pushed to OOD): {frac_id_fooled:.4f}")
    print(f"  Mean max-softmax after attack: {scores_id_attacked.mean():.4f}")

    # Attack 2: push OOD samples toward in-dist (maximize max softmax)
    print("\n--- Attack: push OOD -> in-dist (maximize max softmax) ---")
    attacked_ood = []
    for i in range(0, Xte_ood.size(0), 256):
        xb = Xte_ood[i:i+256]
        xa = pgd_maximize_max_softmax(model, xb)
        attacked_ood.append(xa)
    attacked_ood = torch.cat(attacked_ood)
    scores_ood_attacked = max_softmax_score(model, attacked_ood).numpy()

    frac_ood_fooled = (scores_ood_attacked >= threshold).mean()
    print(f"  Fraction OOD fooled (pushed to in-dist): {frac_ood_fooled:.4f}")
    print(f"  Mean max-softmax after attack: {scores_ood_attacked.mean():.4f}")

    # Post-attack AUROC
    labels_post = np.concatenate([np.zeros(len(scores_id_attacked)), np.ones(len(scores_ood_attacked))])
    scores_post = np.concatenate([-scores_id_attacked, -scores_ood_attacked])
    auroc_post = safe_auroc(labels_post, scores_post)

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  {'Metric':<40}  {'Value':>10}")
    print(f"  {'Baseline AUROC':<40}  {auroc_baseline:>10.4f}")
    print(f"  {'Post-attack AUROC':<40}  {auroc_post:>10.4f}")
    print(f"  {'AUROC drop':<40}  {auroc_baseline - auroc_post:>10.4f}")
    print(f"  {'Frac in-dist fooled (-> OOD)':<40}  {frac_id_fooled:>10.4f}")
    print(f"  {'Frac OOD fooled (-> in-dist)':<40}  {frac_ood_fooled:>10.4f}")
    print(f"  {'Hypothesis (>50% in-dist fooled)':<40}  {'SUPPORTED' if frac_id_fooled > 0.50 else 'NOT SUPPORTED':>10}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
