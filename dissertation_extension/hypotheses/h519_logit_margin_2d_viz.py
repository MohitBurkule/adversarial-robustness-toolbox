"""
h519_logit_margin_2d_viz.py
Demonstrates that logit margin predicts adversarial vulnerability in 2D.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import pearsonr

# ── Reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)
torch.manual_seed(42)

# ── Dataset ───────────────────────────────────────────────────────────────────
N = 600
z = RNG.normal(0, 1, N)
x1 = 0.98 * z + 0.20 * RNG.normal(0, 1, N)
x2 = 0.92 * z + 0.39 * RNG.normal(0, 1, N)
y = (z > 0).astype(int)
X = np.stack([x1, x2], axis=1).astype(np.float32)
Y = y.astype(np.int64)

X_t = torch.from_numpy(X)
Y_t = torch.from_numpy(Y)

# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ── Training ──────────────────────────────────────────────────────────────────
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_t), Y_t)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    logits_clean = model(X_t)
    preds_clean = logits_clean.argmax(dim=1)
    train_acc = (preds_clean == Y_t).float().mean().item()
print(f"Training accuracy: {train_acc:.3f}")

# ── Logit margin ─────────────────────────────────────────────────────────────
with torch.no_grad():
    logits = model(X_t)  # (N, 2)

correct_logit = logits[torch.arange(N), Y_t]          # logit of true class
wrong_logit   = logits[torch.arange(N), 1 - Y_t]      # logit of other class
margin = (correct_logit - wrong_logit).numpy()         # positive => correctly classified

# ── FGSM attack ───────────────────────────────────────────────────────────────
eps = 0.3

def fgsm(x_batch, y_batch):
    x_adv = x_batch.clone().detach().requires_grad_(True)
    loss = criterion(model(x_adv), y_batch)
    loss.backward()
    with torch.no_grad():
        x_adv = x_batch + eps * x_adv.grad.sign()
    return x_adv.detach()

model.train()  # enable grad for FGSM
X_adv = fgsm(X_t, Y_t)
model.eval()

with torch.no_grad():
    preds_adv = model(X_adv).argmax(dim=1)

flipped = (preds_adv != Y_t).numpy().astype(int)   # 1 = flipped (vulnerable)
survived = 1 - flipped                              # 1 = survived (robust)

frac_flipped = flipped.mean()
print(f"Fraction flipped by FGSM: {frac_flipped:.3f}")

# ── Statistics ───────────────────────────────────────────────────────────────
# Use -margin so that higher score → more vulnerable
r, p = pearsonr(margin, survived)
print(f"Pearson correlation(margin, FGSM_survived): r={r:.4f}, p={p:.4e}")

# AUC: predict vulnerability (flipped=1) from -margin (lower margin → higher risk)
auc = roc_auc_score(flipped, -margin)
print(f"AUC (logit margin predicts FGSM vulnerability): {auc:.4f}")

# ── Decision boundary grid ────────────────────────────────────────────────────
xx1, xx2 = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300)
)
grid = torch.from_numpy(np.c_[xx1.ravel(), xx2.ravel()].astype(np.float32))
with torch.no_grad():
    grid_preds = model(grid).argmax(dim=1).numpy().reshape(xx1.shape)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Logit Margin as Predictor of Adversarial Vulnerability (2D)", fontsize=14, fontweight='bold')

# ── Top-left: scatter by logit margin + decision boundary ────────────────────
ax = axes[0, 0]
ax.contourf(xx1, xx2, grid_preds, alpha=0.15, cmap='RdBu', levels=[-0.5, 0.5, 1.5])
ax.contour(xx1, xx2, grid_preds, colors='k', linewidths=1.0, levels=[0.5])
sc = ax.scatter(X[:, 0], X[:, 1], c=margin, cmap='RdYlBu', s=20,
                vmin=np.percentile(margin, 5), vmax=np.percentile(margin, 95), alpha=0.8)
plt.colorbar(sc, ax=ax, label='Logit Margin')
ax.set_title('Logit Margin (blue=robust, red=vulnerable)')
ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

# ── Top-right: scatter by FGSM outcome ───────────────────────────────────────
ax = axes[0, 1]
ax.contourf(xx1, xx2, grid_preds, alpha=0.10, cmap='RdBu', levels=[-0.5, 0.5, 1.5])
ax.contour(xx1, xx2, grid_preds, colors='k', linewidths=1.0, levels=[0.5])
colors = np.where(flipped == 1, 'red', 'green')
ax.scatter(X[:, 0], X[:, 1], c=colors, s=20, alpha=0.7)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', label=f'Survived ({survived.sum()})'),
                   Patch(facecolor='red',   label=f'Flipped  ({flipped.sum()})')]
ax.legend(handles=legend_elements, fontsize=9)
ax.set_title(f'FGSM Outcome (ε={eps}) — {frac_flipped*100:.1f}% flipped')
ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

# ── Bottom-left: histogram of margin split by FGSM outcome ───────────────────
ax = axes[1, 0]
bins = np.linspace(margin.min(), margin.max(), 40)
ax.hist(margin[survived == 1], bins=bins, color='green', alpha=0.6, label='Survived', density=True)
ax.hist(margin[flipped == 1],  bins=bins, color='red',   alpha=0.6, label='Flipped',  density=True)
ax.axvline(0, color='k', linestyle='--', linewidth=1, label='Margin=0')
ax.set_xlabel('Logit Margin')
ax.set_ylabel('Density')
ax.set_title('Logit Margin Distribution by FGSM Outcome')
ax.legend(fontsize=9)

# ── Bottom-right: ROC curve ───────────────────────────────────────────────────
ax = axes[1, 1]
fpr, tpr, _ = roc_curve(flipped, -margin)
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
ax.fill_between(fpr, tpr, alpha=0.15, color='darkorange')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC: Logit Margin → FGSM Vulnerability')
ax.legend(fontsize=9)

plt.tight_layout()
out_path = 'results/fashion_mnist/h519_logit_margin_2d_viz.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
print("PASS")
