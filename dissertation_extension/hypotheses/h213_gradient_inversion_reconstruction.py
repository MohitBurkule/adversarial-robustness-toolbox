"""
H213 - Gradient Inversion: reconstruct input from its gradient, predict vulnerability.

Normal forward+backward:
  x → model → loss → ∇_x (input gradient)

Flipped question:
  Given ∇_x (the gradient of the loss w.r.t. the input), can we reconstruct x?

This is the "gradient inversion" problem (Zhu et al. 2019 "Deep Leakage from
Gradients", federated learning privacy attacks). The key mechanism:
  ∇_x encodes WHERE in input space the model is sensitive — it's a signed
  magnitude map of which pixels matter. For a linear approximation,
  reconstructing x from ∇_x amounts to inverting the Jacobian.

Two experiments:
  (A) RECONSTRUCTION QUALITY: train an inversion network G: ∇_x → x̂.
      Measure MSE and SSIM between x̂ and x. Simple: use an MLP or CNN decoder.
      Success = inversion network can approximate inputs from gradients.

  (B) VULNERABILITY CORRELATION: does reconstruction quality predict vulnerability?
      Intuition: if ∇_x faithfully reconstructs x, the boundary is locally planar
      and FGSM is a good attack direction. High reconstruction quality → easy target.

This is the "flipped" network in the user's framing:
  NORMAL: input x → forward propagation → activations → loss → backprop → ∇_x
  FLIPPED: ∇_x → "forward propagation" (G network) → reconstructed x̂
           i.e. the gradient IS the input and the inversion IS the forward pass.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0
EPS = 0.1
N_EVAL = 300
N_TRAIN_INV = 2000  # samples to train the inversion network
META = {"channels": 1, "size": 28, "n_classes": 10}

torch.manual_seed(SEED)
np.random.seed(SEED)

print("H213: Gradient Inversion — Reconstruct Input from Gradient")
print("=" * 60)
print("Flipped network: gradient is input, inversion is forward pass")
print()

Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Train base classifier ─────────────────────────────────────────────────────
print("[1] Training base classifier...")
model = C.build_model("cnn", META, width=32, seed=SEED)
C.train_model(model, Xtr, Ytr, epochs=10)
model.eval()

# ── Compute input gradients for all training samples ─────────────────────────
print(f"[2] Computing input gradients for {N_TRAIN_INV} training + {N_EVAL} test samples...")

def get_input_gradient(model, X, Y):
    """Returns ∇_x L(f(x), y) for each sample. Shape: same as X."""
    X_req = X.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(X_req), Y, reduction='none')
    # per-sample gradient: sum of individual losses (chain rule holds per-sample)
    loss.sum().backward()
    return X_req.grad.detach().clone()

# Batch gradient computation
def batch_gradients(model, X, Y, batch=64):
    grads = []
    for i in range(0, len(X), batch):
        xb, yb = X[i:i+batch].to(device), Y[i:i+batch].to(device)
        g = get_input_gradient(model, xb, yb)
        grads.append(g.cpu())
    return torch.cat(grads, dim=0)

t0 = time.time()
Xtr_sub, Ytr_sub = Xtr[:N_TRAIN_INV], Ytr[:N_TRAIN_INV]
Xte_sub, Yte_sub = Xte[:N_EVAL], Yte[:N_EVAL]

grad_tr = batch_gradients(model, Xtr_sub, Ytr_sub)  # (N_TRAIN_INV, 1, 28, 28)
grad_te = batch_gradients(model, Xte_sub, Yte_sub)  # (N_EVAL, 1, 28, 28)
print(f"  Gradients computed in {time.time()-t0:.1f}s")
print(f"  Gradient stats: mean={grad_tr.mean():.4f}  std={grad_tr.std():.4f}  "
      f"max={grad_tr.abs().max():.4f}")

# ── Inversion network: ∇_x → x̂ ───────────────────────────────────────────────
print("[3] Training inversion network G: gradient → image...")

class InversionNet(nn.Module):
    """CNN that takes ∇_x (same shape as x) and outputs reconstructed x̂."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()  # output in [0,1]
        )
    def forward(self, grad):
        return self.enc(grad)

G = InversionNet().to(device)
opt_G = torch.optim.Adam(G.parameters(), lr=1e-3)
loader_inv = torch.utils.data.DataLoader(
    list(zip(grad_tr, Xtr_sub)), batch_size=128, shuffle=True)

t0 = time.time()
for epoch in range(20):
    G.train()
    total_loss = 0
    for gb, xb in loader_inv:
        gb, xb = gb.to(device), xb.to(device)
        opt_G.zero_grad()
        xhat = G(gb)
        loss = F.mse_loss(xhat, xb)
        loss.backward()
        opt_G.step()
        total_loss += loss.item()
    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1}/20  MSE={total_loss/len(loader_inv):.4f}")
G.eval()
print(f"  Inversion network trained in {time.time()-t0:.1f}s")

# ── Evaluate reconstruction quality on test set ───────────────────────────────
print("[4] Evaluating reconstruction on test set...")
with torch.no_grad():
    xhat_te = G(grad_te.to(device)).cpu()

mse_per_sample = ((xhat_te - Xte_sub.cpu())**2).mean(dim=(1,2,3)).numpy()
# SSIM proxy: normalised cross-correlation per sample
def ncc(a, b):
    a_flat = a.flatten(1)
    b_flat = b.flatten(1)
    a_norm = a_flat - a_flat.mean(1, keepdim=True)
    b_norm = b_flat - b_flat.mean(1, keepdim=True)
    num = (a_norm * b_norm).sum(1)
    denom = a_norm.norm(dim=1) * b_norm.norm(dim=1) + 1e-8
    return (num / denom).numpy()

ncc_per_sample = ncc(xhat_te, Xte_sub.cpu())

print(f"  Mean reconstruction MSE: {mse_per_sample.mean():.4f} ± {mse_per_sample.std():.4f}")
print(f"  Mean NCC (1=perfect):    {ncc_per_sample.mean():.4f} ± {ncc_per_sample.std():.4f}")

# Baseline: what MSE would a zero-output network get? (predict mean)
baseline_mse = ((Xte_sub - Xte_sub.mean())**2).mean().item()
print(f"  Baseline MSE (predict mean): {baseline_mse:.4f}")
print(f"  Improvement over baseline: {(baseline_mse - mse_per_sample.mean())/baseline_mse*100:.1f}%")

# ── Vulnerability correlation ─────────────────────────────────────────────────
print("[5] Correlating reconstruction quality with adversarial vulnerability...")
margins = C.margin(model, Xte_sub)
margins_np = np.array(margins)

X_fgsm = C.fgsm(model, Xte_sub, Yte_sub, eps=EPS)
with torch.no_grad():
    fgsm_success = (model(X_fgsm).argmax(1) != Yte_sub).cpu().float().numpy()

X_pgd = C.pgd(model, Xte_sub, Yte_sub, eps=EPS, steps=10, alpha=0.01)
with torch.no_grad():
    pgd_success = (model(X_pgd).argmax(1) != Yte_sub).cpu().float().numpy()

# Reconstruction quality (NCC) — higher NCC = better reconstruction
# Hypothesis: higher NCC (more gradient-invertible) = more vulnerable (planar boundary)
try:
    auroc_ncc_fgsm = roc_auc_score(fgsm_success, ncc_per_sample)
    auroc_ncc_pgd  = roc_auc_score(pgd_success,  ncc_per_sample)
    auroc_mse_fgsm = roc_auc_score(fgsm_success, -mse_per_sample)  # low MSE = good recon = vulnerable
    auroc_mse_pgd  = roc_auc_score(pgd_success,  -mse_per_sample)
    auroc_margin   = roc_auc_score(pgd_success,  -margins_np)
except Exception as e:
    auroc_ncc_fgsm = auroc_ncc_pgd = auroc_mse_fgsm = auroc_mse_pgd = float('nan')
    auroc_margin = float('nan')

rho_ncc_margin, _ = spearmanr(ncc_per_sample, margins_np)
rho_mse_margin, _ = spearmanr(-mse_per_sample, margins_np)
rho_grad_norm, _ = spearmanr(grad_te.flatten(1).norm(dim=1).numpy(), margins_np)

print(f"\n  AUROC (NCC  → FGSM success): {auroc_ncc_fgsm:.4f}")
print(f"  AUROC (NCC  → PGD success):  {auroc_ncc_pgd:.4f}")
print(f"  AUROC (-MSE → FGSM success): {auroc_mse_fgsm:.4f}")
print(f"  AUROC (-MSE → PGD success):  {auroc_mse_pgd:.4f}")
print(f"  AUROC (-margin → PGD):       {auroc_margin:.4f}  [baseline]")
print(f"\n  Spearman rho (NCC vs margin): {rho_ncc_margin:.4f}")
print(f"  Spearman rho (MSE vs margin): {rho_mse_margin:.4f}")
print(f"  Spearman rho (grad_norm vs margin): {rho_grad_norm:.4f}")

print("\n--- Gradient norm analysis ---")
grad_norms = grad_te.flatten(1).norm(dim=1).numpy()
print(f"  Gradient norm: mean={grad_norms.mean():.4f}  std={grad_norms.std():.4f}")
try:
    auroc_gnorm_pgd = roc_auc_score(pgd_success, grad_norms)
    print(f"  AUROC (grad_norm → PGD success): {auroc_gnorm_pgd:.4f}")
except: pass

print("\n--- Interpretation ---")
best_recon_auroc = max(auroc_ncc_pgd, auroc_mse_pgd) if not np.isnan(auroc_ncc_pgd) else 0
if best_recon_auroc > auroc_margin + 0.02:
    print("EXCEEDS MARGIN: Reconstruction quality is a BETTER vulnerability predictor than margin!")
    print("Gradient invertibility reveals boundary planarity beyond distance alone.")
elif best_recon_auroc > 0.6:
    print(f"PARTIAL: Reconstruction quality predicts vulnerability (AUROC={best_recon_auroc:.3f})")
    print(f"  Below margin baseline ({auroc_margin:.3f}) but meaningful signal exists.")
    print("  Gradient inversion captures local boundary planarity, correlated with margin.")
else:
    print("NULL: Gradient reconstruction quality does not predict vulnerability.")
    print("  The flipped-network signal is not informative beyond the gradient norm.")
