"""
H231 - Adversarial trajectory transfer: apply image A's attack steps to image B.

For 50 source images (Xte[0:50]): compute PGD-20 trajectory storing gradient sign at each step.
  trajectory[i] = [sign(∇_x L) at step t for t in 1..20]  shape (20, 1, 28, 28)

Transfer experiment: for each source image i, apply its trajectory to 20 OTHER test images j:
  x_transferred = x_j
  for t in range(20): x_transferred = clamp(x_transferred + alpha * trajectory[i][t], x_j±eps, 0,1)
  Measure: ASR of x_transferred on classifier

Baselines:
  - From-scratch PGD-20 on each target image (upper bound)
  - Universal adversarial delta: mean of all 50 source trajectories' net displacement
  - Random walk: 20 random sign steps

Print: ASR(trajectory_transfer), ASR(from_scratch), ASR(universal_delta), ASR(random_walk)
Also: Spearman rho between source_image_margin and ASR of its transferred trajectory.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DS = "fashion_mnist"
N_EVAL = 300
SEED = 0
EPS = 0.1
ALPHA = 0.01
N_SOURCE = 50
N_TARGETS_PER_SOURCE = 20
PGD_STEPS = 20

os.makedirs("results/fashion_mnist", exist_ok=True)


def compute_pgd_trajectories(model, X, Y, eps, steps, alpha):
    """
    Returns trajectories: list of (steps, C, H, W) sign tensors, one per sample.
    X: (N, C, H, W) on device
    """
    device = next(model.parameters()).device
    N = X.shape[0]
    trajectories = [[] for _ in range(N)]

    X = X.clone().detach().to(device)
    Y = Y.clone().detach().to(device)

    delta = torch.zeros_like(X).uniform_(-eps, eps)
    delta = delta.clamp(-eps, eps)

    for step in range(steps):
        delta = delta.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(torch.clamp(X + delta, 0, 1)), Y, reduction="sum")
        loss.backward()
        grad_sign = delta.grad.sign().detach().cpu()  # (N, C, H, W)
        for i in range(N):
            trajectories[i].append(grad_sign[i])  # (C, H, W)
        with torch.no_grad():
            delta = (delta + alpha * delta.grad.sign()).clamp(-eps, eps)

    # trajectories[i] is list of `steps` tensors each (C, H, W)
    # stack to (steps, C, H, W)
    trajectories = [torch.stack(t, dim=0) for t in trajectories]
    return trajectories  # list of N tensors, each (steps, C, H, W)


def apply_trajectory(x_target, trajectory, eps, alpha):
    """
    Apply a stored trajectory (steps, C, H, W) to x_target (1, C, H, W).
    Returns perturbed image (1, C, H, W).
    """
    x = x_target.clone().cpu()
    x0 = x.clone()
    steps = trajectory.shape[0]
    for t in range(steps):
        sign = trajectory[t].unsqueeze(0)  # (1, C, H, W)
        x = x + alpha * sign
        delta = (x - x0).clamp(-eps, eps)
        x = (x0 + delta).clamp(0, 1)
    return x


def main():
    t0 = time.time()
    C.set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_EVAL, seed=SEED)
    Xte, Yte = Xte[:N_EVAL].to(device), Yte[:N_EVAL].to(device)
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)

    model = C.build_model("cnn", meta, width=32, seed=SEED)
    model = model.to(device)
    C.train_model(model, Xtr, Ytr, epochs=10)
    model.eval()

    # Clean accuracy
    with torch.no_grad():
        clean_acc = float((model(Xte).argmax(1) == Yte).float().mean())
    print(f"Clean accuracy: {clean_acc:.3f}")

    # Source and target images
    X_src = Xte[:N_SOURCE]
    Y_src = Yte[:N_SOURCE]
    # Use indices N_SOURCE: N_SOURCE + N_TARGETS_PER_SOURCE * N_SOURCE for targets
    # Actually we just pick N_TARGETS_PER_SOURCE different targets for each source
    X_targets = Xte[N_SOURCE:N_SOURCE + N_TARGETS_PER_SOURCE]
    Y_targets = Yte[N_SOURCE:N_SOURCE + N_TARGETS_PER_SOURCE]

    print(f"\nComputing PGD-{PGD_STEPS} trajectories for {N_SOURCE} source images...")
    trajectories = compute_pgd_trajectories(model, X_src, Y_src, EPS, PGD_STEPS, ALPHA)

    # Source image margins
    margins_src_raw = C.margin(model, X_src)
    margins_src = margins_src_raw if isinstance(margins_src_raw, np.ndarray) else margins_src_raw.cpu().numpy()  # (N_SOURCE,)

    # --- Trajectory transfer experiment ---
    print("Running trajectory transfer experiment...")
    transfer_asr_per_source = []
    all_x_transferred = []

    for i in range(N_SOURCE):
        traj_i = trajectories[i]  # (steps, C, H, W)
        transferred = []
        for j in range(N_TARGETS_PER_SOURCE):
            x_t = X_targets[j:j+1].cpu()
            x_adv = apply_trajectory(x_t, traj_i, EPS, ALPHA)
            transferred.append(x_adv)
        X_trans = torch.cat(transferred, dim=0).to(device)  # (N_TARGETS, C, H, W)
        with torch.no_grad():
            preds = model(X_trans).argmax(1).cpu()
        y_tgt_cpu = Y_targets[:N_TARGETS_PER_SOURCE].cpu()
        asr_i = float((preds != y_tgt_cpu).float().mean())
        transfer_asr_per_source.append(asr_i)
        all_x_transferred.append(X_trans.cpu())

    mean_transfer_asr = float(np.mean(transfer_asr_per_source))

    # --- Baseline 1: From-scratch PGD-20 on target images ---
    print("Running from-scratch PGD-20 on target images (upper bound)...")
    X_pgd_scratch = C.pgd(model, X_targets, Y_targets, eps=EPS, steps=PGD_STEPS, alpha=ALPHA)
    with torch.no_grad():
        preds_scratch = model(X_pgd_scratch).argmax(1).cpu()
    asr_scratch = float((preds_scratch != Y_targets.cpu()).float().mean())

    # --- Baseline 2: Universal adversarial delta ---
    print("Computing universal adversarial delta...")
    # Net displacement of each source trajectory
    net_displacements = []
    for i in range(N_SOURCE):
        traj_i = trajectories[i]  # (steps, C, H, W)
        net = traj_i.sum(0) * ALPHA  # sum over steps, (C, H, W)
        net_displacements.append(net)
    universal_delta = torch.stack(net_displacements, dim=0).mean(0, keepdim=True)  # (1, C, H, W)
    universal_delta_clipped = universal_delta.clamp(-EPS, EPS)

    X_universal = []
    for j in range(N_TARGETS_PER_SOURCE):
        x_t = X_targets[j:j+1].cpu()
        x_u = (x_t + universal_delta_clipped).clamp(0, 1)
        X_universal.append(x_u)
    X_universal = torch.cat(X_universal, dim=0).to(device)
    with torch.no_grad():
        preds_univ = model(X_universal).argmax(1).cpu()
    asr_universal = float((preds_univ != Y_targets[:N_TARGETS_PER_SOURCE].cpu()).float().mean())

    # --- Baseline 3: Random walk (20 random sign steps) ---
    print("Running random walk baseline...")
    C.set_seed(SEED + 999)
    X_random = []
    for j in range(N_TARGETS_PER_SOURCE):
        x_t = X_targets[j:j+1].cpu()
        x0 = x_t.clone()
        for _ in range(PGD_STEPS):
            sign = torch.randint(0, 2, x_t.shape).float() * 2 - 1
            x_t = x_t + ALPHA * sign
            delta = (x_t - x0).clamp(-EPS, EPS)
            x_t = (x0 + delta).clamp(0, 1)
        X_random.append(x_t)
    X_random = torch.cat(X_random, dim=0).to(device)
    with torch.no_grad():
        preds_random = model(X_random).argmax(1).cpu()
    asr_random = float((preds_random != Y_targets[:N_TARGETS_PER_SOURCE].cpu()).float().mean())

    # --- Spearman rho: source margin vs ASR of transferred trajectory ---
    if HAS_SCIPY:
        rho, pval = spearmanr(margins_src, transfer_asr_per_source)
        spearman_str = f"rho={rho:.4f}  p={pval:.4f}"
    else:
        # Manual rank correlation
        n = len(margins_src)
        rank_m = np.argsort(np.argsort(margins_src)).astype(float)
        rank_a = np.argsort(np.argsort(transfer_asr_per_source)).astype(float)
        rho = 1 - 6 * np.sum((rank_m - rank_a)**2) / (n * (n**2 - 1))
        spearman_str = f"rho={rho:.4f}  (scipy not available, p-value not computed)"

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nASR(trajectory_transfer)  = {mean_transfer_asr:.3f}")
    print(f"ASR(from_scratch PGD-20)  = {asr_scratch:.3f}  [upper bound]")
    print(f"ASR(universal_delta)      = {asr_universal:.3f}")
    print(f"ASR(random_walk)          = {asr_random:.3f}")
    print(f"\nSpearman rho (source_margin vs transfer_ASR): {spearman_str}")
    print("  Negative rho => higher-margin sources yield lower transfer ASR (expected)")

    # Per-source breakdown
    print(f"\n--- Per-source transfer ASR (source_margin vs ASR) ---")
    print(f"{'Src':>4}  {'Margin':>8}  {'TransferASR':>12}")
    sorted_by_margin = sorted(zip(margins_src, transfer_asr_per_source), key=lambda x: x[0])
    for margin, asr in sorted_by_margin:
        print(f"{'':>4}  {margin:>8.4f}  {asr:>12.3f}")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
