"""
H186 - Neural Networks are Decision Trees: exact extraction and adversarial implications.

Reproduces and extends Aytekin (arXiv:2210.05189): any ReLU MLP is mathematically
equivalent to a binary oblique decision tree whose splits correspond to ReLU
activation thresholds and whose leaves are the linear maps in each activation region.

Two experiments:

  (A) 2D toy (two-moons, 3 classes):
      Train a tiny MLP (2→8→8→3, ReLU). Enumerate all 2^16 activation patterns,
      prune unreachable ones (via LP feasibility), build the exact decision tree,
      and verify 100% equivalence on a dense test grid.  Optionally plot decision
      boundaries for both MLP and extracted DT (identical by construction).
      Run FGSM on both and confirm identical adversarial regions.

  (B) MNIST / Fashion-MNIST MLP (784→64→32→10):
      Full enumeration is intractable (2^96 patterns).  Instead, do lazy
      sampling-based verification: for 1000 test points, extract the per-sample
      activation pattern, compute the effective affine map for that region, and
      verify MLP output == DT output.  Count unique activation patterns as a
      lower bound on tree size.

Adversarial connection:
  - "Boundary density" = number of ReLU thresholds within epsilon of a test point.
    High boundary density ≈ sample lives near many DT splits ≈ easier to perturb
    into a different activation region ≈ higher adversarial vulnerability.
  - Correlate boundary density with logit margin and PGD attack success rate.

Uses only torch, numpy, sklearn (toy data), scipy (LP feasibility check), matplotlib.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
TOY_HIDDEN = [8, 8]          # 2D toy MLP hidden sizes
TOY_N_SAMPLES = 1200
TOY_EPOCHS = 300
TOY_LR = 1e-2
MNIST_HIDDEN = [64, 32]      # MNIST/FMNIST MLP hidden sizes
MNIST_EPOCHS = 10
MNIST_LR = 1e-3
MNIST_BATCH = 128
EVAL_N = 1000                # number of test points for sampling-based verification
EPS = 0.15                   # L-inf epsilon for adversarial experiments (2D toy)
FMNIST_EPS = 15.0 / 255.0    # L-inf epsilon for Fashion-MNIST
PGD_STEPS = 10
PGD_ALPHA_RATIO = 2.5        # alpha = eps / alpha_ratio


# ---------------------------------------------------------------------------
# MLP with ReLU (generic depth)
# ---------------------------------------------------------------------------
class ReLUMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# Activation pattern extraction
# ---------------------------------------------------------------------------
def get_activation_pattern(model, x):
    """Return binary activation pattern string for input x (batch supported).

    For each hidden ReLU neuron, '1' if pre-activation >= 0, '0' otherwise.
    Returns list of pattern strings, one per sample.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    patterns = []
    h = x
    for layer in model.layers[:-1]:
        pre = layer(h)          # pre-activation
        mask = (pre >= 0).int()  # 1 where ReLU is on
        h = F.relu(pre)
        patterns.append(mask)
    # Concatenate all layer masks into a single binary string per sample
    full = torch.cat(patterns, dim=1)  # (batch, total_hidden)
    strs = []
    for row in full:
        strs.append("".join(str(b.item()) for b in row))
    return strs


def get_effective_affine(model, pattern_str):
    """Given an activation pattern string, compute the effective affine map.

    For a ReLU MLP with pattern p, the network reduces to:
        f(x) = W_eff @ x + b_eff
    where W_eff and b_eff are computed by composing the linear layers with
    diagonal ReLU masks.

    Returns (W_eff, b_eff) as numpy arrays.
    """
    # Parse pattern into per-layer masks
    idx = 0
    masks = []
    for layer in model.layers[:-1]:
        out_features = layer.out_features
        mask = np.array([int(c) for c in pattern_str[idx:idx + out_features]], dtype=np.float64)
        masks.append(mask)
        idx += out_features

    # Compose: start from identity on input
    # After layer i with weight W_i, bias b_i, and diagonal mask D_i:
    #   h_i = D_i (W_i h_{i-1} + b_i)
    # So the cumulative affine is:
    #   h_i = (D_i W_i) h_{i-1} + (D_i b_i)
    # Cumulative: W_cum = D_i W_i W_{cum,prev}, b_cum = D_i W_i b_{cum,prev} + D_i b_i

    W_cum = None  # will be identity implicitly
    b_cum = None

    for i, layer in enumerate(model.layers[:-1]):
        W = layer.weight.detach().cpu().numpy().astype(np.float64)
        b = layer.bias.detach().cpu().numpy().astype(np.float64)
        D = np.diag(masks[i])

        DW = D @ W
        Db = D @ b

        if W_cum is None:
            W_cum = DW
            b_cum = Db
        else:
            b_cum = DW @ b_cum + Db
            W_cum = DW @ W_cum

    # Final layer (no ReLU)
    W_final = model.layers[-1].weight.detach().cpu().numpy().astype(np.float64)
    b_final = model.layers[-1].bias.detach().cpu().numpy().astype(np.float64)

    W_eff = W_final @ W_cum
    b_eff = W_final @ b_cum + b_final
    return W_eff, b_eff


def dt_predict(model, x):
    """Predict using the extracted DT: get activation pattern → effective affine → argmax."""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
        x_t = x
    else:
        x_np = x
        x_t = torch.tensor(x, dtype=torch.float32)

    patterns = get_activation_pattern(model, x_t.to(DEVICE))
    preds = []
    for i, pat in enumerate(patterns):
        W_eff, b_eff = get_effective_affine(model, pat)
        logits = W_eff @ x_np[i] + b_eff
        preds.append(np.argmax(logits))
    return np.array(preds)


# ---------------------------------------------------------------------------
# Boundary density: count ReLU thresholds within epsilon of input
# ---------------------------------------------------------------------------
def boundary_density(model, x, eps):
    """Count how many hidden ReLU neurons have |pre-activation| < eps for input x.

    A neuron with small |pre-activation| means the input is close to that
    neuron's decision hyperplane — a small perturbation can flip it.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    densities = torch.zeros(x.shape[0], device=x.device)
    h = x
    for layer in model.layers[:-1]:
        pre = layer(h)
        near = (pre.abs() < eps).float().sum(dim=1)
        densities += near
        h = F.relu(pre)
    return densities


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    return (x + eps * x.grad.sign()).detach()


def pgd(model, x, y, eps, steps=PGD_STEPS, alpha=None):
    if alpha is None:
        alpha = eps / PGD_ALPHA_RATIO
    adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    adv = adv.clamp(0, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        adv = (adv + alpha * adv.grad.sign()).detach()
        adv = torch.max(torch.min(adv, x + eps), x - eps).clamp(0, 1)
    return adv


# ---------------------------------------------------------------------------
# Full enumeration for 2D toy (tractable for ≤16 neurons)
# ---------------------------------------------------------------------------
def enumerate_reachable_patterns_sampling(model, input_dim, n_samples=200000):
    """Find reachable activation patterns by dense random sampling.

    For the 2D toy case, we sample uniformly from the input bounding box
    and collect all unique activation patterns.  Faster and more robust
    than LP-based enumeration for small input dimensions.
    """
    torch.manual_seed(SEED)
    # Sample uniformly from a generous bounding box
    x = torch.rand(n_samples, input_dim, device=DEVICE) * 6 - 3  # [-3, 3]^d
    patterns = get_activation_pattern(model, x)
    unique = set(patterns)
    return unique


# ---------------------------------------------------------------------------
# Experiment A: 2D Toy
# ---------------------------------------------------------------------------
def experiment_toy():
    from sklearn.datasets import make_moons
    print("=" * 70)
    print("EXPERIMENT A: 2D Toy (Two Moons + third cluster)")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Generate 3-class dataset: two moons + a Gaussian blob
    X_moons, y_moons = make_moons(n_samples=800, noise=0.15, random_state=SEED)
    X_blob = np.random.randn(400, 2) * 0.3 + np.array([0.5, 1.5])
    y_blob = np.full(400, 2)
    X_all = np.vstack([X_moons, X_blob]).astype(np.float32)
    y_all = np.concatenate([y_moons, y_blob])

    # Shuffle
    perm = np.random.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]

    n_train = int(0.8 * len(X_all))
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]

    X_tr = torch.tensor(X_train, device=DEVICE)
    y_tr = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_te = torch.tensor(X_test, device=DEVICE)
    y_te = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

    # Train MLP
    model = ReLUMLP(2, TOY_HIDDEN, 3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=TOY_LR)
    for epoch in range(TOY_EPOCHS):
        model.train()
        logits = model(X_tr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_tr).argmax(1) == y_tr).float().mean().item()
        test_acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    print(f"MLP train acc: {train_acc:.4f}, test acc: {test_acc:.4f}")
    total_neurons = sum(TOY_HIDDEN)
    print(f"Total hidden neurons: {total_neurons}, theoretical max patterns: 2^{total_neurons} = {2**total_neurons}")

    # Enumerate reachable patterns via sampling
    t0 = time.time()
    reachable = enumerate_reachable_patterns_sampling(model, 2)
    enum_time = time.time() - t0
    print(f"Reachable activation patterns (via sampling): {len(reachable)}  (found in {enum_time:.2f}s)")

    # Verify exact equivalence on test set
    with torch.no_grad():
        mlp_preds = model(X_te).argmax(1).cpu().numpy()
    dt_preds = dt_predict(model, X_te)
    equiv = (mlp_preds == dt_preds).mean()
    print(f"MLP vs DT equivalence on test set: {equiv:.6f}  ({(mlp_preds == dt_preds).sum()}/{len(dt_preds)})")

    # Verify on a dense grid
    grid_x = np.linspace(-2, 3, 200)
    grid_y = np.linspace(-1.5, 2.5, 200)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    grid_t = torch.tensor(grid_pts, device=DEVICE)

    with torch.no_grad():
        mlp_grid = model(grid_t).argmax(1).cpu().numpy()
    dt_grid = dt_predict(model, grid_t)
    grid_equiv = (mlp_grid == dt_grid).mean()
    print(f"MLP vs DT equivalence on 40000-point grid: {grid_equiv:.6f}")

    # FGSM attack on MLP — check DT gives same prediction on adversarial inputs
    x_adv = fgsm(model, X_te, y_te, EPS)
    with torch.no_grad():
        mlp_adv_preds = model(x_adv).argmax(1).cpu().numpy()
    dt_adv_preds = dt_predict(model, x_adv)
    adv_equiv = (mlp_adv_preds == dt_adv_preds).mean()
    fgsm_asr = (mlp_adv_preds != y_te.cpu().numpy()).mean()
    print(f"FGSM ASR (eps={EPS}): {fgsm_asr:.4f}")
    print(f"MLP vs DT equivalence on adversarial inputs: {adv_equiv:.6f}")

    # Boundary density analysis
    with torch.no_grad():
        bd = boundary_density(model, X_te, EPS).cpu().numpy()
        margins = []
        logits = model(X_te)
        for i in range(len(X_te)):
            top2 = logits[i].topk(2).values
            margins.append((top2[0] - top2[1]).item())
        margins = np.array(margins)

    # Adversarial path-change example
    print("\n--- Adversarial Path-Change Example ---")
    # Find a sample that FGSM flips
    flipped = np.where(mlp_adv_preds != y_te.cpu().numpy())[0]
    if len(flipped) > 0:
        idx = flipped[0]
        pat_clean = get_activation_pattern(model, X_te[idx:idx+1])[0]
        pat_adv = get_activation_pattern(model, x_adv[idx:idx+1])[0]
        n_flipped_neurons = sum(a != b for a, b in zip(pat_clean, pat_adv))
        print(f"Sample {idx}: true={y_te[idx].item()}, clean_pred={mlp_preds[idx]}, adv_pred={mlp_adv_preds[idx]}")
        print(f"  Activation pattern (clean): {pat_clean}")
        print(f"  Activation pattern (adv):   {pat_adv}")
        print(f"  Number of flipped ReLUs:    {n_flipped_neurons}/{total_neurons}")
        print(f"  Boundary density:           {bd[idx]:.0f}")
        print(f"  Logit margin:               {margins[idx]:.4f}")
    else:
        print("No FGSM-flipped samples found at this epsilon.")

    # Correlation: boundary density vs margin
    from numpy import corrcoef
    corr_bd_margin = corrcoef(bd, margins)[0, 1]
    print(f"\nCorrelation(boundary_density, margin): {corr_bd_margin:.4f}")

    # Correlation: boundary density vs PGD flip
    adv_pgd = pgd(model, X_te, y_te, EPS)
    with torch.no_grad():
        pgd_preds = model(adv_pgd).argmax(1).cpu().numpy()
    pgd_flip = (pgd_preds != y_te.cpu().numpy()).astype(float)
    corr_bd_flip = corrcoef(bd, pgd_flip)[0, 1]
    print(f"Correlation(boundary_density, PGD_flip): {corr_bd_flip:.4f}")
    print(f"PGD ASR: {pgd_flip.mean():.4f}")

    # Boundary density quartile analysis
    quartiles = np.percentile(bd, [25, 50, 75])
    for lo, hi, label in [(0, quartiles[0], "Q1 (lowest BD)"),
                          (quartiles[0], quartiles[1], "Q2"),
                          (quartiles[1], quartiles[2], "Q3"),
                          (quartiles[2], bd.max() + 1, "Q4 (highest BD)")]:
        mask = (bd >= lo) & (bd < hi)
        if mask.sum() == 0:
            continue
        q_asr = pgd_flip[mask].mean()
        q_margin = margins[mask].mean()
        print(f"  {label}: n={mask.sum()}, PGD ASR={q_asr:.4f}, mean margin={q_margin:.4f}")

    # Optional: save decision boundary plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, preds, title in [(axes[0], mlp_grid, "MLP Decision Boundary"),
                                  (axes[1], dt_grid, "Extracted DT Decision Boundary")]:
            ax.contourf(xx, yy, preds.reshape(xx.shape), alpha=0.3, cmap="Set1")
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="Set1",
                       edgecolors="k", s=20, alpha=0.7)
            ax.set_title(title)
            ax.set_xlim(-2, 3)
            ax.set_ylim(-1.5, 2.5)
        plt.tight_layout()
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "results", "fashion_mnist")
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "h186_decision_boundaries.png"), dpi=150)
        plt.close()
        print(f"\nDecision boundary plot saved.")
    except Exception as e:
        print(f"\n(Plot skipped: {e})")

    return {
        "train_acc": train_acc, "test_acc": test_acc,
        "reachable_patterns": len(reachable), "max_patterns": 2 ** total_neurons,
        "test_equiv": equiv, "grid_equiv": grid_equiv, "adv_equiv": adv_equiv,
        "fgsm_asr": fgsm_asr, "pgd_asr": pgd_flip.mean(),
        "corr_bd_margin": corr_bd_margin, "corr_bd_flip": corr_bd_flip,
    }


# ---------------------------------------------------------------------------
# Experiment B: Fashion-MNIST (sampling-based verification)
# ---------------------------------------------------------------------------
def experiment_fmnist():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from campaign import common as C

    print("\n" + "=" * 70)
    print("EXPERIMENT B: Fashion-MNIST MLP (sampling-based DT verification)")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data via campaign API: returns (Xtr, Ytr, Xte, Yte) tensors on DEVICE
    Xtr_raw, Ytr, Xte_raw, Yte = C.load_dataset("fashion_mnist", n_train=6000, n_eval=1000, seed=42)

    # Flatten images for MLP: (N, 1, 28, 28) -> (N, 784)
    Xtr_flat = Xtr_raw.flatten(1)
    Xte_flat = Xte_raw.flatten(1)

    # Build DataLoader from tensors
    train_ds = torch.utils.data.TensorDataset(Xtr_flat, Ytr)
    train_loader = torch.utils.data.DataLoader(train_ds, MNIST_BATCH, shuffle=True)

    input_dim = Xtr_flat.shape[1]  # 784 for 28x28
    print(f"Input dim: {input_dim}, hidden: {MNIST_HIDDEN}, output: 10")
    total_neurons = sum(MNIST_HIDDEN)
    print(f"Total hidden neurons: {total_neurons}, theoretical max patterns: 2^{total_neurons}")

    # Train MLP
    model = ReLUMLP(input_dim, MNIST_HIDDEN, 10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=MNIST_LR)
    for epoch in range(MNIST_EPOCHS):
        model.train()
        correct = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{MNIST_EPOCHS}, train acc: {correct/total:.4f}")

    model.eval()

    # Select EVAL_N correctly-classified test samples
    all_x = Xte_flat  # already on DEVICE
    all_y = Yte

    with torch.no_grad():
        all_preds = model(all_x).argmax(1)
    correct_mask = (all_preds == all_y)
    correct_idx = correct_mask.nonzero(as_tuple=True)[0][:EVAL_N]
    X_eval = all_x[correct_idx]
    y_eval = all_y[correct_idx]
    clean_acc = correct_mask.float().mean().item()
    print(f"Test accuracy (full): {clean_acc:.4f}")
    print(f"Evaluating on {len(X_eval)} correctly-classified test samples")

    # Sampling-based DT equivalence verification
    t0 = time.time()
    with torch.no_grad():
        mlp_logits = model(X_eval)
        mlp_preds = mlp_logits.argmax(1).cpu().numpy()

    patterns_seen = set()
    dt_preds = []
    for i in range(len(X_eval)):
        pat = get_activation_pattern(model, X_eval[i:i+1])[0]
        patterns_seen.add(pat)
        W_eff, b_eff = get_effective_affine(model, pat)
        x_np = X_eval[i].cpu().numpy().astype(np.float64)
        logits_dt = W_eff @ x_np + b_eff
        dt_preds.append(np.argmax(logits_dt))
    dt_preds = np.array(dt_preds)
    verify_time = time.time() - t0

    equiv = (mlp_preds == dt_preds).mean()
    print(f"\nSampling-based verification ({len(X_eval)} points):")
    print(f"  MLP vs DT equivalence: {equiv:.6f}  ({(mlp_preds == dt_preds).sum()}/{len(dt_preds)})")
    print(f"  Unique activation patterns: {len(patterns_seen)}")
    print(f"  Verification time: {verify_time:.2f}s")

    # Logit-level equivalence (not just argmax)
    max_logit_diff = 0.0
    for i in range(min(100, len(X_eval))):
        pat = get_activation_pattern(model, X_eval[i:i+1])[0]
        W_eff, b_eff = get_effective_affine(model, pat)
        x_np = X_eval[i].cpu().numpy().astype(np.float64)
        logits_dt = W_eff @ x_np + b_eff
        logits_mlp = mlp_logits[i].cpu().numpy().astype(np.float64)
        diff = np.max(np.abs(logits_dt - logits_mlp))
        max_logit_diff = max(max_logit_diff, diff)
    print(f"  Max logit difference (first 100 samples): {max_logit_diff:.2e}")

    # Boundary density vs adversarial vulnerability
    print("\n--- Boundary Density vs Adversarial Vulnerability ---")
    with torch.no_grad():
        bd = boundary_density(model, X_eval, FMNIST_EPS).cpu().numpy()
        margins = []
        for i in range(len(X_eval)):
            top2 = mlp_logits[i].topk(2).values
            margins.append((top2[0] - top2[1]).item())
        margins = np.array(margins)

    # PGD attack
    adv = pgd(model, X_eval, y_eval, FMNIST_EPS)
    with torch.no_grad():
        pgd_preds = model(adv).argmax(1).cpu().numpy()
    pgd_flip = (pgd_preds != y_eval.cpu().numpy()).astype(float)
    print(f"PGD ASR: {pgd_flip.mean():.4f}")

    from numpy import corrcoef
    corr_bd_margin = corrcoef(bd, margins)[0, 1]
    corr_bd_flip = corrcoef(bd, pgd_flip)[0, 1]
    print(f"Correlation(boundary_density, margin): {corr_bd_margin:.4f}")
    print(f"Correlation(boundary_density, PGD_flip): {corr_bd_flip:.4f}")

    # Quartile analysis
    quartiles = np.percentile(bd, [25, 50, 75])
    for lo, hi, label in [(0, quartiles[0], "Q1 (lowest BD)"),
                          (quartiles[0], quartiles[1], "Q2"),
                          (quartiles[1], quartiles[2], "Q3"),
                          (quartiles[2], bd.max() + 1, "Q4 (highest BD)")]:
        mask = (bd >= lo) & (bd < hi)
        if mask.sum() == 0:
            continue
        q_asr = pgd_flip[mask].mean()
        q_margin = margins[mask].mean()
        print(f"  {label}: n={mask.sum()}, PGD ASR={q_asr:.4f}, mean margin={q_margin:.4f}")

    # Activation pattern change under adversarial attack
    n_flipped_neurons = []
    for i in range(len(X_eval)):
        pat_clean = get_activation_pattern(model, X_eval[i:i+1])[0]
        pat_adv = get_activation_pattern(model, adv[i:i+1])[0]
        n_flipped_neurons.append(sum(a != b for a, b in zip(pat_clean, pat_adv)))
    n_flipped_neurons = np.array(n_flipped_neurons)
    print(f"\nActivation pattern changes under PGD:")
    print(f"  Mean flipped neurons: {n_flipped_neurons.mean():.2f}/{total_neurons}")
    print(f"  Mean flipped (successful attacks): {n_flipped_neurons[pgd_flip==1].mean():.2f}" if pgd_flip.sum() > 0 else "  (no successful attacks)")
    print(f"  Mean flipped (failed attacks):     {n_flipped_neurons[pgd_flip==0].mean():.2f}" if (1-pgd_flip).sum() > 0 else "  (all attacks succeeded)")
    corr_flipped_flip = corrcoef(n_flipped_neurons, pgd_flip)[0, 1]
    print(f"  Correlation(flipped_neurons, PGD_flip): {corr_flipped_flip:.4f}")

    return {
        "clean_acc": clean_acc, "equiv": equiv,
        "unique_patterns": len(patterns_seen),
        "max_logit_diff": max_logit_diff,
        "pgd_asr": pgd_flip.mean(),
        "corr_bd_margin": corr_bd_margin,
        "corr_bd_flip": corr_bd_flip,
        "mean_flipped_neurons": n_flipped_neurons.mean(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_start = time.time()
    results_a = experiment_toy()
    results_b = experiment_fmnist()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Toy 2D:")
    for k, v in results_a.items():
        print(f"  {k}: {v}")
    print(f"Fashion-MNIST:")
    for k, v in results_b.items():
        print(f"  {k}: {v}")
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
