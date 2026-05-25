"""
Hypothesis H173: Clean re-implementation of AT variants (SAT, FAT, RSLAD, Margin-weighted).

This script is a definitive, bug-free consolidation of H144–H147. The original scripts
for those hypotheses contained a critical bug: pgd_attack() was called inside the
evaluation loop and its return value (raw adversarial tensor) was appended directly to
flipped_pgd[], which expected a boolean flip tensor. This caused a shape/type mismatch
crash when torch.cat() tried to concatenate mixed tensors.

Fix applied here: all evaluation paths use a dedicated pgd_eval() wrapper that runs
pgd_attack() and immediately returns a boolean flip tensor (model(x_adv).argmax(1) != y),
never the raw adversarial examples.

Five CNN variants are trained and compared:

  1. Vanilla          -- standard cross-entropy on clean images
  2. SAT              -- adversarial training with temperature-scheduled CE loss
                        (T linearly decays 5.0 -> 1.0 over epochs, Maini et al. 2020 style)
  3. FAT              -- friendly adversarial training: early-stopped PGD that halts
                        per-sample once the prediction flips (Zhang et al. ICML 2020)
  4. RSLAD            -- adversarial distillation: teacher = PGD-AT model; student trained
                        with KL(student(x_adv) || teacher(x_clean)) (Zi et al. 2021 style)
  5. MarginWeighted   -- standard PGD-AT with sample losses weighted by 1/(clean_margin+0.1)

Each model is evaluated on 1000 correctly-classified test samples:
  - Clean accuracy
  - FGSM attack success rate (eps=15/255)
  - PGD-10 attack success rate (eps=15/255)
  - Mean min_eps (binary search over FGSM direction)
  - Univariate AUROC of clean margin vs PGD flip, and vs min_eps

A final comparison table is printed at the end.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10
EVAL_N = 1000  # number of correctly-classified samples to evaluate


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    """Small CNN matching diagnostic_test.py architecture."""

    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Attack utilities
# ---------------------------------------------------------------------------

def pgd_attack(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=10, random_start=True):
    """PGD attack — returns adversarial IMAGES (use pgd_eval for flip booleans)."""
    model.eval()
    adv = x.clone().detach()
    if random_start:
        adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
        adv = adv.clamp(0, 1)

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        with torch.no_grad():
            adv = (adv + alpha * adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return adv.detach()


def pgd_eval(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=10):
    """Run PGD and return a BOOLEAN flip tensor — never the raw adversarial images.

    This wrapper was introduced in H173 to fix the bug in H144-H147 where
    pgd_attack()'s tensor return was mistakenly appended to a list of flip booleans.
    """
    x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps)
    with torch.no_grad():
        return (model(x_adv).argmax(1) != y)  # bool tensor, shape (N,)


def fgsm_eval(model, x, y, eps=EPS):
    """FGSM attack — returns boolean flip tensor."""
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    sign = x_adv.grad.sign().detach()
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Binary search (FGSM direction) for minimum eps to flip each sample."""
    model.eval()
    N = x.size(0)
    lo = torch.zeros(N, device=DEVICE)
    hi = torch.full((N,), eps_max, device=DEVICE)

    x_req = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_req), y)
    loss.backward()
    sign = x_req.grad.sign().detach()

    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = (model(adv).argmax(1) != y)
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def fat_attack(model, x, y, eps=EPS, alpha=2.0 / 255.0, max_steps=10):
    """Friendly Adversarial Training attack (Zhang et al. ICML 2020).

    Early-stops PGD per sample once the prediction flips.  Samples that have
    already been fooled are frozen; only the remaining samples continue to be
    perturbed.  This avoids over-adversarial examples and promotes a
    'friendly' perturbation magnitude.
    """
    model.eval()
    x_adv = x.clone().detach()

    for step in range(max_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)

        flipped = (logits.argmax(1) != y)
        if flipped.all():
            break

        active = ~flipped
        loss = F.cross_entropy(logits[active], y[active])
        loss.backward()

        with torch.no_grad():
            grad_sign = x_adv.grad[active].sign()
            x_adv_new = x_adv.clone()
            x_adv_new[active] = (
                x_adv[active] + alpha * grad_sign
            ).clamp(x[active] - eps, x[active] + eps).clamp(0, 1)
            x_adv = x_adv_new

    return x_adv.detach()


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def train_vanilla(train_set, seed=0):
    """Standard cross-entropy training on clean images."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return model


def train_sat(train_set, seed=0):
    """SAT: PGD-AT with temperature-scheduled CE loss (T: 5.0 -> 1.0)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        T = 5.0 - 4.0 * (epoch / max(EPOCHS - 1, 1))
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = pgd_attack(model, x, y, eps=EPS, alpha=5.0 / 255.0, steps=2)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(x_adv) / T, y).backward()
            opt.step()
    return model


def train_fat(train_set, seed=0):
    """FAT: adversarial training with early-stopped (friendly) PGD perturbations."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = fat_attack(model, x, y, eps=EPS, alpha=2.0 / 255.0, max_steps=10)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(x_adv), y).backward()
            opt.step()
    return model


def train_rslad(train_set, seed=0):
    """RSLAD: adversarial distillation — teacher is PGD-AT, student mimics teacher on x_adv.

    Steps:
      1. Train a PGD-AT teacher for EPOCHS.
      2. Train a fresh student CNN for EPOCHS using:
           loss = KL(log_softmax(student(x_adv)), softmax(teacher(x_clean)))
         where x_adv is generated by PGD-7 on the student model.
    """
    # --- Step 1: Train PGD-AT teacher ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    teacher = CNN(N_CLASSES).to(DEVICE)
    opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)

    print("  [RSLAD] Training teacher (PGD-AT)...")
    for epoch in range(EPOCHS):
        teacher.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = pgd_attack(teacher, x, y, eps=EPS, alpha=5.0 / 255.0, steps=2)
            teacher.train()
            opt_t.zero_grad()
            F.cross_entropy(teacher(x_adv), y).backward()
            opt_t.step()
    teacher.eval()

    # --- Step 2: Train student with distillation loss ---
    torch.manual_seed(seed + 1)
    np.random.seed(seed + 1)
    loader2 = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    student = CNN(N_CLASSES).to(DEVICE)
    opt_s = torch.optim.Adam(student.parameters(), lr=1e-3)

    print("  [RSLAD] Training student (distillation)...")
    for epoch in range(EPOCHS):
        student.train()
        for x, y in loader2:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # PGD-7 adversarial examples on the student
            x_adv = pgd_attack(student, x, y, eps=EPS, alpha=2.0 / 255.0, steps=7)
            student.train()
            opt_s.zero_grad()
            # KL divergence: student(x_adv) vs teacher(x_clean)
            with torch.no_grad():
                teacher_soft = F.softmax(teacher(x), dim=1)
            student_log_soft = F.log_softmax(student(x_adv), dim=1)
            loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
            loss.backward()
            opt_s.step()

    return student


def train_margin_weighted_at(train_set, seed=0):
    """Margin-weighted PGD-AT: per-sample adversarial loss weighted by 1/(clean_margin+0.1)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            N = x.size(0)

            # Compute clean margins for weighting (no grad needed)
            with torch.no_grad():
                model.eval()
                logits_clean = model(x)
                correct_logit = logits_clean[torch.arange(N), y]
                mask = torch.ones_like(logits_clean, dtype=torch.bool)
                mask[torch.arange(N), y] = False
                max_other = logits_clean[mask].view(N, -1).max(dim=1).values
                margin = correct_logit - max_other
                w = 1.0 / (margin + 0.1)
                w = w / w.mean()  # normalise so mean weight == 1

            x_adv = pgd_attack(model, x, y, eps=EPS, alpha=5.0 / 255.0, steps=2)
            model.train()
            opt.zero_grad()
            loss_vec = F.cross_entropy(model(x_adv), y, reduction="none")
            (loss_vec * w).mean().backward()
            opt.step()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, x_test, y_test, n_eval=EVAL_N):
    """Evaluate model on up to n_eval correctly-classified test samples.

    Returns a dict with keys:
      clean_acc, fgsm_rate, pgd_rate, mean_min_eps,
      margin_auroc_pgd, margin_auroc_mineps
    """
    model.eval()
    with torch.no_grad():
        all_preds = model(x_test).argmax(1)
        correct_mask = (all_preds == y_test)

    # Limit to n_eval correct samples
    correct_idx = correct_mask.nonzero(as_tuple=True)[0][:n_eval]
    x_c = x_test[correct_idx]
    y_c = y_test[correct_idx]
    N = x_c.size(0)
    clean_acc = correct_mask.float().mean().item()

    # Clean margins on the evaluation subset
    with torch.no_grad():
        logits = model(x_c)
        sorted_l, _ = logits.sort(1, descending=True)
        margin = (sorted_l[:, 0] - sorted_l[:, 1]).cpu().numpy()

    # Attack evaluation (batched to avoid OOM)
    fgsm_flips, pgd_flips, min_eps_vals = [], [], []
    for i in range(0, N, 512):
        bx, by = x_c[i:i + 512], y_c[i:i + 512]
        fgsm_flips.append(fgsm_eval(model, bx, by).cpu())
        pgd_flips.append(pgd_eval(model, bx, by, steps=10).cpu())   # returns bool, not images
        min_eps_vals.append(min_eps_to_flip(model, bx, by).cpu())

    fgsm_arr = torch.cat(fgsm_flips).numpy().astype(int)
    pgd_arr = torch.cat(pgd_flips).numpy().astype(int)
    min_eps_arr = torch.cat(min_eps_vals).numpy()

    # AUROC: margin predicting vulnerability (lower margin -> more vulnerable -> flip=1)
    # Use -margin so higher = more vulnerable, matching flip=1 direction
    def safe_auroc(target, score):
        if target.std() == 0:
            return float("nan")
        a = roc_auc_score(target, score)
        return max(a, 1 - a)

    auroc_pgd = safe_auroc(pgd_arr, -margin)
    auroc_mineps = safe_auroc(pgd_arr, min_eps_arr)  # min_eps as predictor of PGD flip

    return {
        "clean_acc": clean_acc,
        "fgsm_rate": fgsm_arr.mean(),
        "pgd_rate": pgd_arr.mean(),
        "mean_min_eps": min_eps_arr.mean(),
        "margin_auroc_pgd": auroc_pgd,
        "margin_auroc_mineps": auroc_mineps,
        "n_eval": N,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Hypothesis H173: Clean AT Variants Comparison (SAT, FAT, RSLAD, MarginWT)")
    print("Bug fix: pgd_eval() now returns bool flip tensor, not raw adversarial images.")
    print("=" * 70)

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Pre-load test set onto device
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # Define training jobs: (label, train_fn)
    jobs = [
        ("Vanilla",       train_vanilla),
        ("SAT",           train_sat),
        ("FAT",           train_fat),
        ("RSLAD",         train_rslad),
        ("MarginWeighted", train_margin_weighted_at),
    ]

    results = {}
    for name, train_fn in jobs:
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        model = train_fn(train_set)
        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.1f}s")

        print(f"  Evaluating {name} on {EVAL_N} correct test samples...")
        metrics = evaluate_model(model, test_x, test_y)
        results[name] = metrics
        print(f"  CleanAcc={metrics['clean_acc']:.4f}  "
              f"FGSM={metrics['fgsm_rate']:.4f}  "
              f"PGD={metrics['pgd_rate']:.4f}  "
              f"MinEps={metrics['mean_min_eps']:.4f}  "
              f"MarginAUROC-PGD={metrics['margin_auroc_pgd']:.4f}  "
              f"(n={metrics['n_eval']})")

    # Final comparison table
    print("\n")
    print("=" * 70)
    print("Final Comparison Table")
    print("=" * 70)
    header = f"{'Model':<18} {'CleanAcc':>9} {'FGSM%':>7} {'PGD%':>7} {'MinEps':>8} {'MarginAUROC-PGD':>16}"
    print(header)
    print("-" * 70)
    for name, m in results.items():
        auroc_str = f"{m['margin_auroc_pgd']:.4f}" if not np.isnan(m['margin_auroc_pgd']) else "  N/A "
        print(
            f"{name:<18} "
            f"{m['clean_acc']:>9.4f} "
            f"{m['fgsm_rate']:>7.4f} "
            f"{m['pgd_rate']:>7.4f} "
            f"{m['mean_min_eps']:>8.4f} "
            f"{auroc_str:>16}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
