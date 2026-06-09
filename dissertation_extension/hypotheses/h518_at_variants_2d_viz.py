"""
h518_at_variants_2d_viz.py
2D visualization of adversarial training variants on latent-z dataset.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Dataset ──────────────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)
N = 600
z = RNG.normal(0, 1, N)
x1 = 0.98 * z + 0.20 * RNG.normal(0, 1, N)
x2 = 0.92 * z + 0.39 * RNG.normal(0, 1, N)
y = (z > 0).astype(int)
X = np.stack([x1, x2], axis=1).astype(np.float32)
Y = y.astype(np.int64)

# train/test split
idx = np.arange(N)
RNG2 = np.random.default_rng(7)
RNG2.shuffle(idx)
train_idx, test_idx = idx[:480], idx[480:]
X_tr, Y_tr = X[train_idx], Y[train_idx]
X_te, Y_te = X[test_idx], Y[test_idx]

X_tr_t = torch.tensor(X_tr)
Y_tr_t = torch.tensor(Y_tr)
X_te_t = torch.tensor(X_te)
Y_te_t = torch.tensor(Y_te)

# ── Model ─────────────────────────────────────────────────────────────────────
def make_mlp():
    return nn.Sequential(
        nn.Linear(2, 16), nn.ReLU(),
        nn.Linear(16, 16), nn.ReLU(),
        nn.Linear(16, 2)
    )

# ── Attack helpers ────────────────────────────────────────────────────────────
def fgsm(model, x, y, eps=0.3):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return (x + eps * x.grad.sign()).detach()

def pgd(model, x, y, eps=0.3, alpha=0.1, steps=7):
    xadv = x.clone().detach() + torch.zeros_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        xadv = xadv.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(xadv), y)
        loss.backward()
        xadv = xadv.detach() + alpha * xadv.grad.sign()
        xadv = torch.max(torch.min(xadv, x + eps), x - eps).detach()
    return xadv

# ── Training routines ─────────────────────────────────────────────────────────
def train_standard(model, epochs=200, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(X_tr_t), Y_tr_t).backward()
        opt.step()

def train_pgdat(model, epochs=200, lr=1e-2, eps=0.3, alpha=0.1, steps=7):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.eval()
        xadv = pgd(model, X_tr_t, Y_tr_t, eps=eps, alpha=alpha, steps=steps)
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(xadv), Y_tr_t).backward()
        opt.step()

def train_trades(model, epochs=200, lr=1e-2, eps=0.3, alpha=0.1, steps=7, beta=6.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.eval()
        # inner: maximise KL divergence
        xadv = X_tr_t.clone().detach() + torch.zeros_like(X_tr_t).uniform_(-eps, eps)
        with torch.no_grad():
            logits_nat = model(X_tr_t)
            p_nat = F.softmax(logits_nat, dim=-1)
        for _ in range(steps):
            xadv = xadv.clone().detach().requires_grad_(True)
            logits_adv = model(xadv)
            kl = F.kl_div(F.log_softmax(logits_adv, dim=-1), p_nat, reduction='batchmean')
            kl.backward()
            xadv = xadv.detach() + alpha * xadv.grad.sign()
            xadv = torch.max(torch.min(xadv, X_tr_t + eps), X_tr_t - eps).detach()
        model.train()
        opt.zero_grad()
        ce = F.cross_entropy(model(X_tr_t), Y_tr_t)
        logits_adv = model(xadv)
        kl = F.kl_div(F.log_softmax(logits_adv, dim=-1),
                       F.softmax(model(X_tr_t).detach(), dim=-1),
                       reduction='batchmean')
        loss = ce + beta * kl
        loss.backward()
        opt.step()

def train_mart(model, epochs=200, lr=1e-2, eps=0.3, alpha=0.1, steps=7, beta=6.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.eval()
        xadv = pgd(model, X_tr_t, Y_tr_t, eps=eps, alpha=alpha, steps=steps)
        model.train()
        opt.zero_grad()
        logits_nat = model(X_tr_t)
        logits_adv = model(xadv)
        # MART: BCE with boosted weight for misclassified
        pred_nat = logits_nat.argmax(dim=1)
        incorrect = (pred_nat != Y_tr_t).float()  # 1 for misclassified
        # cross-entropy on adversarial examples
        ce_adv = F.cross_entropy(logits_adv, Y_tr_t, reduction='none')
        # KL between adv and nat
        kl = F.kl_div(F.log_softmax(logits_adv, dim=-1),
                       F.softmax(logits_nat.detach(), dim=-1),
                       reduction='none').sum(dim=1)
        loss = (ce_adv + beta * kl * (1 + incorrect)).mean()
        loss.backward()
        opt.step()

# ── Metrics ───────────────────────────────────────────────────────────────────
def test_acc(model):
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).argmax(dim=1)
    return (preds == Y_te_t).float().mean().item()

def fgsm_asr(model, eps=0.3):
    model.eval()
    xadv = fgsm(model, X_te_t.clone().detach(), Y_te_t, eps=eps)
    with torch.no_grad():
        preds = model(xadv).argmax(dim=1)
    return (preds != Y_te_t).float().mean().item()

# ── Decision boundary ─────────────────────────────────────────────────────────
def decision_boundary(ax, model, title):
    res = 200
    xs = np.linspace(-4, 4, res)
    ys = np.linspace(-4, 4, res)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32))
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(grid), dim=1)[:, 1].numpy().reshape(res, res)
    ax.contourf(xx, yy, probs, levels=50, cmap="RdBu_r", alpha=0.7, vmin=0, vmax=1)
    ax.contour(xx, yy, probs, levels=[0.5], colors='k', linewidths=1)

    # training points
    ax.scatter(X_tr[Y_tr == 0, 0], X_tr[Y_tr == 0, 1], c='blue', s=8, alpha=0.5, label='class 0')
    ax.scatter(X_tr[Y_tr == 1, 0], X_tr[Y_tr == 1, 1], c='red', s=8, alpha=0.5, label='class 1')

    # FGSM adversarial examples on test set
    model.eval()
    xadv_te = fgsm(model, X_te_t.clone().detach(), Y_te_t, eps=0.3).numpy()
    ax.scatter(xadv_te[Y_te == 0, 0], xadv_te[Y_te == 0, 1], c='blue', marker='x', s=20, alpha=0.7)
    ax.scatter(xadv_te[Y_te == 1, 0], xadv_te[Y_te == 1, 1], c='red', marker='x', s=20, alpha=0.7)

    acc = test_acc(model)
    asr = fgsm_asr(model)
    ax.set_title(f"{title}\nAcc={acc:.3f}  ASR={asr:.3f}", fontsize=9)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return acc, asr

# ── Main ──────────────────────────────────────────────────────────────────────
torch.manual_seed(0)

configs = [
    ("Standard",  train_standard),
    ("PGD-AT",    train_pgdat),
    ("TRADES",    train_trades),
    ("MART",      train_mart),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

summary = {}
for ax, (name, train_fn) in zip(axes, configs):
    print(f"Training {name}...")
    torch.manual_seed(0)
    model = make_mlp()
    train_fn(model)
    acc, asr = decision_boundary(ax, model, name)
    summary[name] = {"acc": acc, "asr": asr}
    print(f"  {name}: Acc={acc:.4f}  FGSM_ASR={asr:.4f}")

fig.suptitle("AT Variants — Decision Boundaries on Latent-Z Dataset\n(× = FGSM adversarial examples)", fontsize=11)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__),
                        "../results/fashion_mnist/h518_at_variants_2d_viz.png")
out_path = os.path.normpath(out_path)
plt.savefig(out_path, dpi=120)
print(f"\nSaved: {out_path}")

print("\n=== Summary ===")
for name, m in summary.items():
    print(f"  {name:12s}  Acc={m['acc']:.4f}  FGSM_ASR={m['asr']:.4f}")
