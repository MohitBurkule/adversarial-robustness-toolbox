"""
H279 - Loss basin geometry: standard-trained vs PGD-AT models.

The hypothesis: adversarially trained models sit in wider, flatter basins in
weight space, and this flatness correlates with input-space robustness.

Basin geometry metrics (for both standard and AT models):
  1. Random-direction sharpness: sample 50 random unit directions in weight
     space, measure loss increase at radii r in {0.01, 0.05, 0.1}.
     Average across directions = "basin width estimate".
  2. Gradient norm at convergence: ||grad_theta L|| on train set.
  3. Hessian trace proxy via Hutchinson estimator: 20 random vectors v,
     compute E[v^T H v] = E[(v . grad(grad(L).v))] -- trace(H) / dim.
  4. Linear interpolation to a random nearby model (same init, different seed):
     measure average loss along the linear path.
  5. Cosine similarity of weight vectors (standard vs AT): how far apart?

Also: per-sample clean margin vs per-sample adversarial vulnerability --
does the basin flatness (measured globally) predict per-sample robustness?

N_train=6000 for speed. N_eval=200 test samples.
"""
import os, sys, time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS          = "fashion_mnist"
SEED        = 0
N_TRAIN     = 6000
EPOCHS      = 10
LR_STD      = 0.05
LR_AT       = 0.01
MOMENTUM    = 0.9
BATCH       = 128
EPS         = 0.1
PGD_STEPS   = 10
PGD_ALPHA   = 0.01
AT_STEPS    = 7
AT_ALPHA    = 0.02
N_DIRECTIONS    = 50
RADII           = [0.01, 0.05, 0.10]
HUTCHINSON_ITERS = 20
N_EVAL      = 200
OUT_FILE    = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h279_robust_basin_geometry_output.txt"
)


# ---------------------------------------------------------------------------
# PGD-AT training (provided spec)
# ---------------------------------------------------------------------------

def pgd_at_train(model, X, Y, eps=EPS, steps=AT_STEPS, alpha=AT_ALPHA,
                 epochs=EPOCHS, lr=LR_AT):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM,
                          weight_decay=5e-4)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = X[idx].to(C.DEVICE), Y[idx].to(C.DEVICE)
            # PGD inner loop
            xadv = xb.clone().detach() + torch.zeros_like(xb).uniform_(-eps, eps)
            for _ in range(steps):
                xadv.requires_grad_(True)
                loss = F.cross_entropy(model(xadv), yb)
                loss.backward()
                xadv = xadv.detach() + alpha * xadv.grad.sign()
                xadv = torch.max(torch.min(xadv, xb + eps), xb - eps).clamp(0, 1).detach()
            opt.zero_grad()
            F.cross_entropy(model(xadv), yb).backward()
            opt.step()
        if (ep + 1) % 5 == 0:
            print(f"    AT epoch {ep+1}/{epochs}")


def train_standard(model, X, Y):
    opt = torch.optim.SGD(model.parameters(), lr=LR_STD, momentum=MOMENTUM,
                          weight_decay=5e-4)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = X[idx].to(C.DEVICE), Y[idx].to(C.DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"    std epoch {ep+1}/{EPOCHS}  loss={total_loss:.3f}")


# ---------------------------------------------------------------------------
# Basin geometry measurements
# ---------------------------------------------------------------------------

def get_param_vector(model):
    """Flatten all parameters into a single 1-D CPU tensor."""
    return torch.cat([p.data.cpu().reshape(-1) for p in model.parameters()])


def set_param_vector(model, vec):
    """Load a flat parameter vector back into a model (in-place)."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[offset:offset + numel].reshape(p.shape).to(p.device))
        offset += numel


def compute_loss(model, X, Y):
    model.eval()
    total, nb = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb, yb = X[i:i+256].to(C.DEVICE), Y[i:i+256].to(C.DEVICE)
            total += F.cross_entropy(model(xb), yb).item()
            nb += 1
    return total / nb


def random_direction_sharpness(model, X, Y, n_dirs=N_DIRECTIONS, radii=RADII):
    """
    Measure average loss increase when moving distance r along n_dirs random
    unit directions in weight space.
    Returns dict {r: (mean_increase, std_increase)}.
    """
    orig_vec = get_param_vector(model)
    dim = orig_vec.numel()
    base_loss = compute_loss(model, X, Y)

    results = {r: [] for r in radii}
    for _ in range(n_dirs):
        d = torch.randn(dim)
        d = d / (d.norm() + 1e-12)
        for r in radii:
            perturbed = orig_vec + r * d
            set_param_vector(model, perturbed)
            loss_r = compute_loss(model, X, Y)
            results[r].append(loss_r - base_loss)
            set_param_vector(model, orig_vec)  # restore

    return {r: (float(np.mean(v)), float(np.std(v))) for r, v in results.items()}


def gradient_norm_at_convergence(model, X, Y):
    """||grad_theta L|| on training set."""
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad.zero_()

    total_loss = 0.0
    nb = 0
    for i in range(0, len(X), 256):
        xb, yb = X[i:i+256].to(C.DEVICE), Y[i:i+256].to(C.DEVICE)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        total_loss += loss.item()
        nb += 1

    grad_norm = float(
        torch.norm(
            torch.stack([p.grad.norm() for p in model.parameters()
                         if p.grad is not None])
        ).item()
    )
    # Zero grads to clean up
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    model.eval()
    return grad_norm


def hutchinson_hessian_trace(model, X, Y, n_iters=HUTCHINSON_ITERS):
    """
    Approximate trace(H) / dim via Hutchinson estimator.
    For each random Rademacher vector v: estimate v^T H v = v . grad(grad(L) . v).
    We use the finite-difference approximation to avoid full second-order:
      v^T H v ≈ (g(theta + eps*v) - g(theta - eps*v)) . v / (2*eps)
    where g = grad_theta L.
    """
    h = 1e-4
    orig_vec = get_param_vector(model)
    dim = orig_vec.numel()
    estimates = []

    for _ in range(n_iters):
        v = torch.randint(0, 2, (dim,)).float() * 2 - 1  # Rademacher

        # Loss at theta + h*v
        set_param_vector(model, orig_vec + h * v)
        loss_plus = compute_loss(model, X, Y)

        # Loss at theta - h*v
        set_param_vector(model, orig_vec - h * v)
        loss_minus = compute_loss(model, X, Y)

        # Base loss (finite diff gradient approx in direction v)
        # v^T H v ≈ (loss(+) - 2*loss(base) + loss(-)) / h^2
        set_param_vector(model, orig_vec)
        loss_base = compute_loss(model, X, Y)

        vHv = (loss_plus - 2 * loss_base + loss_minus) / (h ** 2)
        estimates.append(float(vHv))

    # trace(H) / dim ≈ mean(v^T H v) / (sum_i E[v_i^2]) -- but for Rademacher, E[v_i^2]=1
    set_param_vector(model, orig_vec)
    return float(np.mean(estimates)), float(np.std(estimates))


def linear_interp_path(model_a, model_b, X, Y, n_steps=11):
    """
    Average loss along the linear path from model_a to model_b.
    Returns list of (alpha, loss) pairs.
    """
    va = get_param_vector(model_a)
    vb = get_param_vector(model_b)
    interp = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                           width=32, seed=0)
    interp.to(C.DEVICE)
    path_losses = []
    for step in range(n_steps):
        alpha = step / (n_steps - 1)
        v_interp = (1.0 - alpha) * va + alpha * vb
        set_param_vector(interp, v_interp)
        loss = compute_loss(interp, X, Y)
        path_losses.append((alpha, loss))
    return path_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    C.set_seed(SEED)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    print("Loading data...")
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    idx = torch.randperm(len(Xtr))[:N_TRAIN]
    Xtr, Ytr = Xtr[idx], Ytr[idx]
    Xte, Yte = Xte[:N_EVAL], Yte[:N_EVAL]
    print(f"  train={len(Xtr)}  eval={len(Xte)}")

    # Train standard model
    print("\n--- Training standard model ---")
    C.set_seed(SEED)
    model_std = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                              width=32, seed=SEED)
    model_std.to(C.DEVICE)
    train_standard(model_std, Xtr, Ytr)

    # Train AT model
    print("\n--- Training PGD-AT model ---")
    C.set_seed(SEED)
    model_at = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                             width=32, seed=SEED)
    model_at.to(C.DEVICE)
    pgd_at_train(model_at, Xtr, Ytr)

    # Cosine similarity of weight vectors
    v_std = get_param_vector(model_std)
    v_at  = get_param_vector(model_at)
    cos_sim = float(F.cosine_similarity(v_std.unsqueeze(0), v_at.unsqueeze(0)).item())
    weight_l2 = float((v_std - v_at).norm().item())
    print(f"\nWeight cosine similarity (std vs AT): {cos_sim:.4f}")
    print(f"Weight L2 distance (std vs AT):       {weight_l2:.4f}")

    geo = {}

    for tag, model in [("standard", model_std), ("pgd_at", model_at)]:
        print(f"\n--- Basin geometry: {tag} ---")

        print("  Random-direction sharpness...")
        sharpness = random_direction_sharpness(model, Xtr, Ytr)
        for r, (mn, sd) in sharpness.items():
            print(f"    r={r}: mean_loss_increase={mn:.4f}±{sd:.4f}")

        print("  Gradient norm at convergence...")
        gnorm = gradient_norm_at_convergence(model, Xtr, Ytr)
        print(f"    grad_norm={gnorm:.4f}")

        print(f"  Hutchinson Hessian trace ({HUTCHINSON_ITERS} iters)...")
        h_trace, h_std = hutchinson_hessian_trace(model, Xtr, Ytr)
        print(f"    hessian_trace_proxy={h_trace:.4f}±{h_std:.4f}")

        geo[tag] = dict(sharpness=sharpness, grad_norm=gnorm,
                        h_trace=h_trace, h_std=h_std)

    # Linear interpolation path between std and AT
    print("\n--- Linear interpolation path (std -> AT) ---")
    path = linear_interp_path(model_std, model_at, Xtr, Ytr, n_steps=11)
    for alpha, loss in path:
        print(f"  alpha={alpha:.1f}  train_loss={loss:.4f}")
    path_losses = [l for _, l in path]
    end_max = max(path_losses[0], path_losses[-1])
    mid_max  = max(path_losses[1:-1])
    barrier  = mid_max - end_max

    # Robustness evaluation
    print("\n--- Evaluating robustness ---")
    rob = {}
    for tag, model in [("standard", model_std), ("pgd_at", model_at)]:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)
        _, clean_acc = C.logits_and_acc(model, Xte, Yte)
        Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
        _, fgsm_acc = C.logits_and_acc(model, Xfgsm, Yte)
        Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
        _, pgd_acc = C.logits_and_acc(model, Xpgd, Yte)
        margins = C.margin(model, Xte, Yte)
        mean_margin = float(np.mean(margins))
        fgsm_asr, pgd_asr = 1.0 - fgsm_acc, 1.0 - pgd_acc
        print(f"  [{tag}] clean={clean_acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  "
              f"PGD_ASR={pgd_asr:.4f}  margin={mean_margin:.4f}")
        rob[tag] = dict(clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                        pgd_asr=pgd_asr, mean_margin=mean_margin,
                        margins=margins)

    # Spearman correlation: per-sample margin vs per-sample adversarial vulnerability
    # Vulnerability proxy = 1 if PGD succeeds, else 0 (need per-sample)
    print("\n--- Per-sample margin vs vulnerability correlation ---")
    correlations = {}
    for tag, model in [("standard", model_std), ("pgd_at", model_at)]:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)
        Xpgd_full = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
        with torch.no_grad():
            logits_adv = []
            for i in range(0, len(Xpgd_full), 128):
                logits_adv.append(model(Xpgd_full[i:i+128].to(C.DEVICE)).cpu())
            logits_adv = torch.cat(logits_adv)
        preds_adv = logits_adv.argmax(1).cpu().numpy()
        Yte_np = Yte.cpu().numpy() if hasattr(Yte, 'numpy') else np.array(Yte)
        vulnerable = (preds_adv != Yte_np).astype(float)
        margins_np = rob[tag]["margins"]
        rho, pval = spearmanr(margins_np, vulnerable)
        print(f"  [{tag}] Spearman(margin, pgd_vulnerable): rho={rho:.4f}  p={pval:.4g}")
        correlations[tag] = dict(spearman_rho=rho, spearman_p=pval)

    elapsed = time.time() - t0

    lines = [
        "H279 - Robust Basin Geometry (Standard vs AT)\n",
        "=" * 70 + "\n\n",
        f"N_train={N_TRAIN}  epochs={EPOCHS}  eps={EPS}  pgd_steps={PGD_STEPS}\n",
        f"N_directions={N_DIRECTIONS}  Hutchinson_iters={HUTCHINSON_ITERS}\n\n",
    ]

    lines.append("--- Robustness summary ---\n")
    lines.append(f"{'Model':<12} {'CleanAcc':>10} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Margin':>10}\n")
    lines.append("-" * 52 + "\n")
    for tag in ["standard", "pgd_at"]:
        r = rob[tag]
        lines.append(f"{tag:<12} {r['clean_acc']:>10.4f} {r['fgsm_asr']:>10.4f} "
                     f"{r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f}\n")

    lines.append("\n--- Basin geometry summary ---\n")
    for tag in ["standard", "pgd_at"]:
        g = geo[tag]
        lines.append(f"\n{tag}:\n")
        lines.append(f"  grad_norm={g['grad_norm']:.4f}\n")
        lines.append(f"  hessian_trace_proxy={g['h_trace']:.4f}±{g['h_std']:.4f}\n")
        for r, (mn, sd) in g["sharpness"].items():
            lines.append(f"  sharpness(r={r}): mean_loss_increase={mn:.4f}±{sd:.4f}\n")

    lines.append(f"\nWeight cosine similarity (std vs AT): {cos_sim:.4f}\n")
    lines.append(f"Weight L2 distance (std vs AT):       {weight_l2:.4f}\n")

    lines.append("\n--- Linear interpolation path (std -> AT) ---\n")
    lines.append(f"{'alpha':>6} {'train_loss':>12}\n")
    for alpha, loss in path:
        lines.append(f"{alpha:>6.1f} {loss:>12.4f}\n")
    lines.append(f"Loss barrier (max mid - max endpoint): {barrier:+.4f}\n")
    lines.append(f"  -> {'barrier detected (different basins)' if barrier > 0.05 else 'no significant barrier (linearly connected)'}\n")

    lines.append("\n--- Per-sample margin vs vulnerability ---\n")
    for tag, c in correlations.items():
        lines.append(f"  [{tag}] Spearman rho={c['spearman_rho']:.4f}  p={c['spearman_p']:.4g}\n")

    lines.append(f"\nElapsed: {elapsed:.1f}s\n\n")
    lines.append("ANALYSIS\n--------\n")
    std_geo  = geo["standard"]
    at_geo   = geo["pgd_at"]
    std_sharp = std_geo["sharpness"][0.05][0]
    at_sharp  = at_geo["sharpness"][0.05][0]
    lines.append(f"AT sharpness(r=0.05) vs standard: {at_sharp:.4f} vs {std_sharp:.4f}  "
                 f"(delta={at_sharp-std_sharp:+.4f})\n")
    lines.append(f"AT Hessian trace vs standard: {at_geo['h_trace']:.4f} vs "
                 f"{std_geo['h_trace']:.4f}\n")
    flatter = at_sharp < std_sharp
    lines.append(f"AT model in flatter basin: {flatter}\n")
    lines.append(f"AT model more robust (lower PGD_ASR): "
                 f"{rob['pgd_at']['pgd_asr'] < rob['standard']['pgd_asr']}\n")
    verdict = ("AT models sit in flatter basins AND are more robust => flatness-robustness link supported."
               if flatter and rob["pgd_at"]["pgd_asr"] < rob["standard"]["pgd_asr"]
               else "Evidence for flatness-robustness link is mixed or absent.")
    lines.append(f"Verdict: {verdict}\n")

    with open(OUT_FILE, "w") as f:
        f.writelines(lines)
    print(f"\nResults written to {OUT_FILE}")
