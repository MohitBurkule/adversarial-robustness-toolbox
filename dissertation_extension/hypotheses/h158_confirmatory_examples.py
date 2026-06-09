"""
Hypothesis H158: Anti-Adversarial ("Confirmatory") Examples.

Standard FGSM maximises cross-entropy loss to push an input away from the
correct-class decision region.  The mirror operation — *minimising* cross-
entropy (gradient descent instead of ascent) — pushes the input deeper into
the correct-class region, making the model more confident.  We call these
"confirmatory" or "anti-adversarial" examples.

Four CNN variants are trained on Fashion-MNIST for 10 epochs:
  1. Vanilla          — standard cross-entropy training
  2. Anti-adv aug     — 50/50 mix of clean + anti-adversarial samples per
                        batch (anti-FGSM: x - eps * sign(grad))
  3. PGD-AT           — standard adversarial training with PGD-10
  4. PGD-AT + anti-adv — PGD adversarial training PLUS anti-adversarial
                         augmentation; tests whether confirmatory examples
                         recover the clean-accuracy penalty of PGD-AT

Each model is evaluated on the first 1000 correctly-classified test samples:
  - Clean accuracy
  - FGSM attack success rate  (eps = 15/255, 1 step)
  - PGD  attack success rate  (eps = 15/255, 10 steps)
  - Mean min_eps_to_flip      (binary search, 8 iters, FGSM direction)
  - Univariate AUROC of margin and top1_prob vs PGD flip label

A comparison table is printed at the end.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNN(nn.Module):
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
# Anti-adversarial perturbation
# ---------------------------------------------------------------------------

def anti_fgsm(model, x, y, eps=EPS):
    """One-step anti-adversarial: gradient DESCENT on CE (maximise confidence).

    Returns a *new* tensor with the perturbation applied; does not touch the
    original batch's computation graph.
    """
    model.eval()
    x_anti = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_anti), y)
    loss.backward()
    with torch.no_grad():
        x_anti_out = (x - eps * x_anti.grad.sign()).clamp(0, 1)
    model.train()
    return x_anti_out.detach()


# ---------------------------------------------------------------------------
# PGD attack (used in adversarial training and evaluation)
# ---------------------------------------------------------------------------

def pgd(model, x, y, eps=EPS, steps=10, alpha=None, rand_init=True):
    """PGD adversarial example generator."""
    if alpha is None:
        alpha = eps / 4.0
    model.eval()
    x_adv = x.clone().detach()
    if rand_init:
        x_adv = x_adv + (torch.rand_like(x_adv) * 2 - 1) * eps
        x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)
    model.train()
    return x_adv.detach()


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def train_vanilla(train_set, seed=0):
    """Standard cross-entropy training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2, pin_memory=True)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  [vanilla] epoch {epoch+1}/{EPOCHS}")
    return model


def train_anti_adv(train_set, seed=1):
    """Training with 50/50 mix of clean + anti-adversarial samples."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2, pin_memory=True)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Generate anti-adversarial batch
            x_anti = anti_fgsm(model, x, y, eps=EPS)
            # Mix 50/50
            x_mix = torch.cat([x, x_anti], dim=0)
            y_mix = torch.cat([y, y], dim=0)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(x_mix), y_mix).backward()
            opt.step()
        print(f"  [anti-adv] epoch {epoch+1}/{EPOCHS}")
    return model


def train_pgd_at(train_set, seed=2):
    """Standard PGD adversarial training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2, pin_memory=True)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = pgd(model, x, y, eps=EPS, steps=10)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(x_adv), y).backward()
            opt.step()
        print(f"  [pgd-at]   epoch {epoch+1}/{EPOCHS}")
    return model


def train_pgd_at_plus_anti(train_set, seed=3):
    """PGD adversarial training augmented with anti-adversarial samples.

    Each mini-batch trains on: adversarial examples + anti-adversarial examples.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2, pin_memory=True)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = pgd(model, x, y, eps=EPS, steps=10)
            x_anti = anti_fgsm(model, x, y, eps=EPS)
            x_mix = torch.cat([x_adv, x_anti], dim=0)
            y_mix = torch.cat([y, y], dim=0)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(x_mix), y_mix).backward()
            opt.step()
        print(f"  [pgd+anti] epoch {epoch+1}/{EPOCHS}")
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def get_correct_subset(model, test_set, n=1000):
    """Return the first n correctly-classified test samples."""
    model.eval()
    all_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    all_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    with torch.no_grad():
        preds = model(all_x).argmax(1)
    correct = (preds == all_y).nonzero(as_tuple=True)[0]
    idx = correct[:n]
    return all_x[idx], all_y[idx]


def clean_accuracy(model, test_set):
    """Compute clean test accuracy over the full test set."""
    model.eval()
    loader = DataLoader(test_set, batch_size=512, shuffle=False)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def attack_success_fgsm(model, x, y, eps=EPS):
    """FGSM attack success rate (fraction of inputs misclassified after attack)."""
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    with torch.no_grad():
        x_adv_out = (x + eps * x_adv.grad.sign()).clamp(0, 1)
        flipped = (model(x_adv_out).argmax(1) != y).float()
    return flipped.mean().item(), flipped


def attack_success_pgd(model, x, y, eps=EPS, steps=10, batch_size=256):
    """PGD attack success rate, processed in mini-batches to save memory."""
    model.eval()
    flipped_list = []
    for i in range(0, x.size(0), batch_size):
        xb, yb = x[i:i+batch_size], y[i:i+batch_size]
        x_adv = pgd(model, xb, yb, eps=eps, steps=steps)
        model.eval()
        with torch.no_grad():
            flipped_list.append((model(x_adv).argmax(1) != yb).float())
    flipped = torch.cat(flipped_list)
    return flipped.mean().item(), flipped


def min_eps_to_flip(model, x, y, eps_max=0.5, iters=8, batch_size=256):
    """Per-sample binary search for the minimum epsilon (FGSM direction) that flips the prediction.

    Returns a CPU tensor of shape (N,) with epsilon values.
    """
    model.eval()
    results = []
    for i in range(0, x.size(0), batch_size):
        xb, yb = x[i:i+batch_size], y[i:i+batch_size]
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            # Single FGSM step at eps=mid (per-sample)
            xb_g = xb.clone().detach().requires_grad_(True)
            loss = F.cross_entropy(model(xb_g), yb)
            loss.backward()
            with torch.no_grad():
                grad_sign = xb_g.grad.sign()
                x_adv = (xb + mid.view(-1, 1, 1, 1) * grad_sign).clamp(0, 1)
                flipped = model(x_adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        results.append(hi.cpu())
    return torch.cat(results)


def compute_features(model, x, y):
    """Compute margin and top1_prob for a batch of correctly-classified samples."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        sorted_logits, _ = logits.sort(dim=1, descending=True)
        margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).cpu()
        top1_prob = probs.max(dim=1).values.cpu()
    return margin, top1_prob


def auroc(scores, labels):
    """Univariate AUROC, oriented so that higher score => more vulnerable."""
    s = scores.numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
    l = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    if l.std() == 0:
        return 0.5
    a = roc_auc_score(l, s)
    return max(a, 1.0 - a)


def evaluate_model(name, model, test_set):
    """Run all evaluations for one model and return a results dict."""
    print(f"\n  Evaluating [{name}]...")

    clean_acc = clean_accuracy(model, test_set)
    print(f"    Clean accuracy (full test set): {clean_acc*100:.2f}%")

    x_c, y_c = get_correct_subset(model, test_set, n=1000)
    N = x_c.size(0)
    print(f"    Using {N} correctly-classified samples for adversarial evaluation.")

    # FGSM
    fgsm_rate, fgsm_flipped = attack_success_fgsm(model, x_c, y_c)
    print(f"    FGSM success rate: {fgsm_rate*100:.1f}%")

    # PGD
    pgd_rate, pgd_flipped = attack_success_pgd(model, x_c, y_c)
    print(f"    PGD-10 success rate: {pgd_rate*100:.1f}%")

    # min eps to flip
    print(f"    Computing min_eps_to_flip (binary search, 8 iters)...")
    min_eps = min_eps_to_flip(model, x_c, y_c)
    mean_min_eps = min_eps.mean().item()
    print(f"    Mean min_eps_to_flip: {mean_min_eps:.4f}")

    # Features & AUROC vs PGD flip
    margin, top1_prob = compute_features(model, x_c, y_c)
    pgd_label = pgd_flipped.cpu().int()

    auroc_margin = auroc(-margin, pgd_label)   # lower margin => more vulnerable
    auroc_top1   = auroc(-top1_prob, pgd_label)
    print(f"    AUROC margin (vs PGD flip):   {auroc_margin:.4f}")
    print(f"    AUROC top1_prob (vs PGD flip):{auroc_top1:.4f}")

    return {
        "name": name,
        "clean_acc": clean_acc,
        "fgsm_rate": fgsm_rate,
        "pgd_rate": pgd_rate,
        "mean_min_eps": mean_min_eps,
        "auroc_margin": auroc_margin,
        "auroc_top1": auroc_top1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("Hypothesis H158: Anti-Adversarial (Confirmatory) Examples")
    print("=" * 78)
    print(f"Device: {DEVICE}  |  Epochs: {EPOCHS}  |  Batch: {BATCH}  |  EPS: {EPS:.5f}")

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True,  download=True, transform=tf)
    test_set  = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    results = []

    # 1. Vanilla
    print("\n--- Training: Vanilla ---")
    t0 = time.time()
    model_vanilla = train_vanilla(train_set, seed=0)
    print(f"  Done in {time.time()-t0:.1f}s")
    results.append(evaluate_model("Vanilla", model_vanilla, test_set))

    # 2. Anti-adv augmented
    print("\n--- Training: Anti-Adv Augmented ---")
    t0 = time.time()
    model_anti = train_anti_adv(train_set, seed=1)
    print(f"  Done in {time.time()-t0:.1f}s")
    results.append(evaluate_model("Anti-Adv Aug", model_anti, test_set))

    # 3. PGD-AT
    print("\n--- Training: PGD Adversarial Training ---")
    t0 = time.time()
    model_pgd = train_pgd_at(train_set, seed=2)
    print(f"  Done in {time.time()-t0:.1f}s")
    results.append(evaluate_model("PGD-AT", model_pgd, test_set))

    # 4. PGD-AT + anti-adv
    print("\n--- Training: PGD-AT + Anti-Adv ---")
    t0 = time.time()
    model_pgd_anti = train_pgd_at_plus_anti(train_set, seed=3)
    print(f"  Done in {time.time()-t0:.1f}s")
    results.append(evaluate_model("PGD-AT+Anti", model_pgd_anti, test_set))

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY TABLE — H158 Confirmatory Examples")
    print("=" * 78)
    hdr = f"{'Model':<18} {'CleanAcc':>9} {'FGSM%':>7} {'PGD%':>7} {'MinEps':>8} {'AUROC-M':>8} {'AUROC-P':>8}"
    print(hdr)
    print("-" * 78)
    for r in results:
        print(
            f"{r['name']:<18}"
            f" {r['clean_acc']*100:>8.2f}%"
            f" {r['fgsm_rate']*100:>6.1f}%"
            f" {r['pgd_rate']*100:>6.1f}%"
            f" {r['mean_min_eps']:>8.4f}"
            f" {r['auroc_margin']:>8.4f}"
            f" {r['auroc_top1']:>8.4f}"
        )
    print("=" * 78)

    # Interpretation hints
    vanilla   = results[0]
    anti      = results[1]
    pgd_at    = results[2]
    pgd_anti  = results[3]

    print("\nKey comparisons:")
    delta_clean_anti = anti["clean_acc"] - vanilla["clean_acc"]
    print(f"  Anti-adv aug  vs Vanilla  — clean acc delta: {delta_clean_anti*100:+.2f}%")
    delta_fgsm_anti = anti["fgsm_rate"] - vanilla["fgsm_rate"]
    print(f"  Anti-adv aug  vs Vanilla  — FGSM rate delta: {delta_fgsm_anti*100:+.1f}%")

    delta_clean_pgd = pgd_at["clean_acc"] - vanilla["clean_acc"]
    print(f"  PGD-AT        vs Vanilla  — clean acc delta: {delta_clean_pgd*100:+.2f}%  (robustness-accuracy trade-off)")
    delta_clean_recovery = pgd_anti["clean_acc"] - pgd_at["clean_acc"]
    print(f"  PGD-AT+Anti   vs PGD-AT   — clean acc delta: {delta_clean_recovery*100:+.2f}%  (confirmatory recovery?)")
    delta_rob_recovery = pgd_anti["pgd_rate"] - pgd_at["pgd_rate"]
    print(f"  PGD-AT+Anti   vs PGD-AT   — PGD rate delta:  {delta_rob_recovery*100:+.1f}%  (robustness cost of mixing?)")
    print("=" * 78)


if __name__ == "__main__":
    main()
