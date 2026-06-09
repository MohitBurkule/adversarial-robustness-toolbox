"""
h522_robust_distillation_2d_viz.py
Hypothesis: A student trained with soft labels from a robust (AT) teacher
inherits robustness on clean data only, no adversarial examples needed during
student training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(0)
RNG = np.random.default_rng(42)

# ── dataset ──────────────────────────────────────────────────────────────────
N = 600
z = RNG.normal(0, 1, N)
x1 = 0.98 * z + 0.20 * RNG.normal(0, 1, N)
x2 = 0.92 * z + 0.39 * RNG.normal(0, 1, N)
y  = (z > 0).astype(int)
X  = np.stack([x1, x2], axis=1).astype(np.float32)
Y  = y.astype(np.int64)

# train/test split
idx = RNG.permutation(N)
tr, te = idx[:480], idx[480:]
Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(Y[tr])
Xte, Yte = torch.tensor(X[te]), torch.tensor(Y[te])

DEVICE = torch.device("cpu")

# ── model ────────────────────────────────────────────────────────────────────
def make_mlp():
    return nn.Sequential(
        nn.Linear(2, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 2)
    )

# ── PGD attack ───────────────────────────────────────────────────────────────
def pgd_attack(model, x, y, eps=0.3, steps=7, alpha=None):
    if alpha is None:
        alpha = 2 * eps / steps
    model.eval()
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = x_adv.detach()
    return x_adv

def fgsm_attack(model, x, y, eps=0.3):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    loss.backward()
    with torch.no_grad():
        x_adv = x + eps * x_adv.grad.sign()
    return x_adv.detach()

# ── training helpers ──────────────────────────────────────────────────────────
def train_standard(model, epochs=200, lr=1e-2):
    opt = optim.Adam(model.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        ce(model(Xtr), Ytr).backward()
        opt.step()

def train_at(model, epochs=200, lr=1e-2, eps=0.3, steps=7):
    opt = optim.Adam(model.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        x_adv = pgd_attack(model, Xtr, Ytr, eps=eps, steps=steps)
        model.train()
        opt.zero_grad()
        ce(model(x_adv), Ytr).backward()
        opt.step()

def train_student(student, teacher, epochs=200, lr=1e-2, T=4.0, alpha=0.5):
    opt = optim.Adam(student.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()
    kl  = nn.KLDivLoss(reduction="batchmean")
    teacher.eval()
    for _ in range(epochs):
        student.train()
        opt.zero_grad()
        logits_s = student(Xtr)
        with torch.no_grad():
            logits_t = teacher(Xtr)
        soft_s = torch.log_softmax(logits_s / T, dim=1)
        soft_t = torch.softmax(logits_t / T, dim=1)
        loss_kl = kl(soft_s, soft_t) * (T ** 2)
        loss_ce = ce(logits_s, Ytr)
        loss = alpha * loss_kl + (1 - alpha) * loss_ce
        loss.backward()
        opt.step()

# ── metrics ───────────────────────────────────────────────────────────────────
def test_acc(model):
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1)
    return (preds == Yte).float().mean().item()

def fgsm_asr(model, eps=0.3):
    model.eval()
    x_adv = fgsm_attack(model, Xte, Yte, eps=eps)
    with torch.no_grad():
        preds = model(x_adv).argmax(1)
    flipped = (preds != Yte).float().mean().item()
    return flipped

def pgd_asr(model, eps=0.3, steps=7):
    model.eval()
    x_adv = pgd_attack(model, Xte, Yte, eps=eps, steps=steps)
    with torch.no_grad():
        preds = model(x_adv).argmax(1)
    flipped = (preds != Yte).float().mean().item()
    return flipped

# ── train all models ──────────────────────────────────────────────────────────
print("Training standard teacher …")
std_teacher = make_mlp()
train_standard(std_teacher)

print("Training robust (AT) teacher …")
rob_teacher = make_mlp()
train_at(rob_teacher)

print("Training student from std teacher …")
std_student = make_mlp()
train_student(std_student, std_teacher)

print("Training student from robust teacher …")
rob_student = make_mlp()
train_student(rob_student, rob_teacher)

models = {
    "Std Teacher":    std_teacher,
    "Robust Teacher": rob_teacher,
    "Std Student":    std_student,
    "Rob Student":    rob_student,
}

# ── print metrics ─────────────────────────────────────────────────────────────
print("\n{:<18} {:>10} {:>12} {:>10}".format("Model", "Test Acc", "FGSM ASR", "PGD ASR"))
print("-" * 54)
metrics = {}
for name, m in models.items():
    ta   = test_acc(m)
    fa   = fgsm_asr(m)
    pa   = pgd_asr(m)
    metrics[name] = (ta, fa, pa)
    print(f"{name:<18} {ta:>10.3f} {fa:>12.3f} {pa:>10.3f}")

# ── decision-boundary plot ────────────────────────────────────────────────────
def decision_boundary_ax(ax, model, title, Xte_np, Yte_np, eps=0.3):
    h = 0.04
    x_min, x_max = Xte_np[:, 0].min() - 0.5, Xte_np[:, 0].max() + 0.5
    y_min, y_max = Xte_np[:, 1].min() - 0.5, Xte_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Z = model(grid).argmax(1).numpy().reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.8)

    # clean test points
    colors = ["#e74c3c", "#2980b9"]
    for cls in range(2):
        mask = Yte_np == cls
        ax.scatter(Xte_np[mask, 0], Xte_np[mask, 1],
                   c=colors[cls], s=14, alpha=0.7, zorder=3)

    # FGSM-flipped points
    x_adv = fgsm_attack(model, Xte, Yte, eps=eps)
    model.eval()
    with torch.no_grad():
        adv_preds = model(x_adv).argmax(1)
    flipped_mask = (adv_preds != Yte).numpy()
    x_adv_np = x_adv.numpy()
    ax.scatter(x_adv_np[flipped_mask, 0], x_adv_np[flipped_mask, 1],
               marker="x", c="k", s=30, linewidths=1.2, zorder=4,
               label=f"FGSM flipped ({flipped_mask.sum()})")

    ta, fa, pa = metrics[title]
    ax.set_title(f"{title}\nAcc={ta:.2f}  FGSM-ASR={fa:.2f}  PGD-ASR={pa:.2f}",
                 fontsize=9)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(fontsize=7, loc="upper left")

Xte_np = Xte.numpy()
Yte_np = Yte.numpy()

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle("Robust Knowledge Distillation — 2D Latent-Z Dataset", fontsize=12)

panels = [
    (axes[0, 0], std_teacher, "Std Teacher"),
    (axes[0, 1], rob_teacher, "Robust Teacher"),
    (axes[1, 0], std_student, "Std Student"),
    (axes[1, 1], rob_student, "Rob Student"),
]
for ax, model, title in panels:
    decision_boundary_ax(ax, model, title, Xte_np, Yte_np)

plt.tight_layout()
out = Path("/run/media/mohit/NewVolume1/adversarial-robustness-toolbox/"
           "dissertation_extension/results/fashion_mnist/"
           "h522_robust_distillation_2d_viz.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")
print("DONE")
