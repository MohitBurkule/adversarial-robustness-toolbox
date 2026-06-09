"""
H185 - Per-sample adversarial boundary manifold sampler.

Concept:
  Standard PGD produces one deterministic adversarial example per input. This
  hypothesis injects isotropic noise into the gradient at each step, so that K
  independent trajectories from the same clean input explore the decision-boundary
  surface laterally rather than converging to a single point:

    delta_{t+1} = clip(delta_t + alpha * sign(grad_delta L + sigma * eps_t), -eps, eps)
    eps_t ~ N(0, I)

  sigma=0 recovers standard PGD; large sigma turns the walk into diffusion across
  the boundary surface. The K resulting perturbations collectively characterise
  the local geometry of the adversarial region around that sample:

    - solid_angle_proxy  = ASR * delta_diversity  (how wide the adversarial cone is)
    - delta_diversity    = mean pairwise cosine distance among adversarial deltas
    - min_norm           = minimum L2 norm among adversarial deltas (approx boundary dist)
    - mean_adv_norm      = mean L2 norm of adversarial deltas

Motivation (Papers 1 / 5 / 6):
  Paper 1 shows that logit margin is the dominant univariate vulnerability
  predictor, but it captures only the DISTANCE to the boundary, not its SHAPE.
  Paper 5 (GradCAM / SmoothGrad) captures gradient saliency but not boundary
  geometry. This experiment tests whether sampling-based boundary geometry stats
  carry additional predictive signal for per-sample PGD vulnerability.

Gap in prior art:
  - Boundary Attack (Brendel et al. 2018): decision-based, converges to ONE
    boundary point, designed for attack success not vulnerability prediction.
  - CURE (Moosavi-Dezfooli et al. 2019): second-order Hessian curvature of the
    loss surface; expensive, not sampling-based.
  - Neither has been used as a per-sample vulnerability *predictor*. Stochastic-
    gradient boundary sampling for vulnerability characterisation is unstudied.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ── defaults ────────────────────────────────────────────────────────────────
EPS       = 0.1
K_TRAJS   = 20
STEPS     = 15
SIGMA     = 0.5
N_EVAL    = 1000
EPOCHS    = 6
SEED      = 0


# ── core sampler ────────────────────────────────────────────────────────────

def boundary_manifold_sample(model, x, y, eps, K, steps, alpha, sigma, seed):
    """Run K stochastic-gradient trajectories from the same clean input.

    Args:
        model: classifier (eval mode expected)
        x: single input tensor [C, H, W]
        y: scalar label tensor
        eps: L-inf budget
        K: number of independent trajectories
        steps: PGD steps per trajectory
        alpha: step size
        sigma: noise scale (0 = standard PGD)
        seed: random seed

    Returns:
        dict with deltas [K, *x.shape], is_adv [K], asr float, adv_deltas.
    """
    model.eval()
    device = x.device
    rng = torch.Generator(device=device).manual_seed(seed)
    x0 = x.unsqueeze(0)                          # [1, C, H, W]
    y0 = y.unsqueeze(0) if y.dim() == 0 else y   # [1]

    deltas = []
    is_adv = []

    for k in range(K):
        # random start within eps-ball
        delta = torch.empty_like(x0).uniform_(-eps, eps, generator=rng)
        for _ in range(steps):
            xa = (x0 + delta).clamp(0, 1).requires_grad_(True)
            loss = F.cross_entropy(model(xa), y0)
            g, = torch.autograd.grad(loss, xa)
            noise = torch.randn_like(g, generator=rng) if sigma > 0 else 0
            delta = delta + alpha * (g + sigma * noise).sign()
            delta = delta.clamp(-eps, eps)
            delta = ((x0 + delta).clamp(0, 1) - x0).detach()

        # check if adversarial
        with torch.no_grad():
            pred = model((x0 + delta).clamp(0, 1)).argmax(1)
        deltas.append(delta.squeeze(0))      # [C, H, W]
        is_adv.append((pred != y0).item())

    deltas = torch.stack(deltas)              # [K, C, H, W]
    is_adv = torch.tensor(is_adv, dtype=torch.bool)
    asr = float(is_adv.float().mean())
    adv_deltas = deltas[is_adv] if is_adv.any() else deltas[:0]

    return {"deltas": deltas, "is_adv": is_adv, "asr": asr,
            "adv_deltas": adv_deltas}


# ── per-sample geometry stats ───────────────────────────────────────────────

def _pairwise_cosine_distance(vecs):
    """Mean pairwise cosine distance among a set of flat vectors.
    Returns 0 if fewer than 2 vectors."""
    if vecs.shape[0] < 2:
        return 0.0
    flat = vecs.reshape(vecs.shape[0], -1)
    flat = F.normalize(flat, dim=1)
    sim = flat @ flat.T                       # [n, n]
    n = sim.shape[0]
    # mean of upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(n, n, device=sim.device, dtype=torch.bool), diagonal=1)
    mean_cos = sim[mask].mean().item()
    return 1.0 - mean_cos                     # cosine distance


def boundary_geometry_stats(model, X, Y, eps, K, steps, alpha, sigma, seed):
    """Run boundary_manifold_sample per sample and compute geometry statistics.

    Returns dict of numpy arrays, one entry per sample:
        mean_adv_norm, delta_diversity, solid_angle_proxy, min_norm, asr.
    """
    model.eval()
    N = X.shape[0]
    mean_adv_norm   = np.zeros(N, dtype=np.float32)
    delta_diversity = np.zeros(N, dtype=np.float32)
    solid_angle     = np.zeros(N, dtype=np.float32)
    min_norm        = np.full(N, np.inf, dtype=np.float32)
    asr_arr         = np.zeros(N, dtype=np.float32)

    for i in range(N):
        res = boundary_manifold_sample(
            model, X[i], Y[i], eps, K, steps, alpha, sigma,
            seed=seed + i)   # vary per sample for independence

        asr_arr[i] = res["asr"]
        adv = res["adv_deltas"]

        if adv.shape[0] > 0:
            norms = adv.reshape(adv.shape[0], -1).norm(dim=1)
            mean_adv_norm[i] = norms.mean().item()
            min_norm[i] = norms.min().item()
        else:
            mean_adv_norm[i] = 0.0
            min_norm[i] = float("inf")

        div = _pairwise_cosine_distance(adv) if adv.shape[0] >= 2 else 0.0
        delta_diversity[i] = div
        solid_angle[i] = res["asr"] * div

    return {"mean_adv_norm": mean_adv_norm, "delta_diversity": delta_diversity,
            "solid_angle_proxy": solid_angle, "min_norm": min_norm,
            "asr": asr_arr}


# ── main experiment ─────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    C.set_seed(SEED)
    device = C.DEVICE
    print(f"Device: {device}")

    # ── data & model ────────────────────────────────────────────────────────
    (Xtr, Ytr), (Xev, Yev) = C.load_dataset("fashion_mnist",
                                              n_train=6000, n_eval=N_EVAL, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    model = C.build_model("cnn", meta, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=EPOCHS, opt="sgd", lr=0.05)

    # ── PGD vulnerability labels (restrict to correctly-classified) ─────────
    atk = C.attack_success(model, Xev, Yev, attack="pgd", eps=EPS, steps=10)
    corr = atk["correct"].astype(bool)
    flips = atk["flips"][corr]
    Xc, Yc = Xev[corr], Yev[corr]
    print(f"Correctly classified: {corr.sum()}/{len(corr)}, PGD ASR: {flips.mean():.4f}")

    # ── margin baseline ─────────────────────────────────────────────────────
    logits, _ = C.logits_and_acc(model, Xc, Yc)
    margin = C.margin_of(logits, Yc)
    margin_auroc = C.safe_auroc(flips, -margin)   # low margin → vulnerable
    print(f"\nMargin AUROC (baseline): {margin_auroc:.4f}")

    # ── sigma sweep ─────────────────────────────────────────────────────────
    alpha = EPS / 8.0
    sigmas = [0.0, 0.1, 0.5, 1.0, 2.0]

    print(f"\n{'sigma':>6} | {'solid_angle':>12} | {'diversity':>10} | "
          f"{'min_norm':>10} | {'mean_norm':>10} | {'sampler_asr':>10}")
    print("-" * 75)

    for sig in sigmas:
        stats = boundary_geometry_stats(model, Xc, Yc, EPS, K_TRAJS, STEPS,
                                        alpha, sig, seed=SEED)
        # AUROCs vs PGD flip
        sa_auroc  = C.safe_auroc(flips, stats["solid_angle_proxy"])
        div_auroc = C.safe_auroc(flips, stats["delta_diversity"])
        # min_norm: lower → more vulnerable → flip sign
        mn_auroc  = C.safe_auroc(flips, -stats["min_norm"])
        mn2_auroc = C.safe_auroc(flips, stats["mean_adv_norm"])
        asr_auroc = C.safe_auroc(flips, stats["asr"])

        print(f"{sig:6.1f} | {sa_auroc:12.4f} | {div_auroc:10.4f} | "
              f"{mn_auroc:10.4f} | {mn2_auroc:10.4f} | {asr_auroc:10.4f}")

    # ── seed stability (sigma=0.5, K=20) ────────────────────────────────────
    print("\n── Seed stability (sigma=0.5, K=20, seeds 0/1/2) ──")
    seeds = [0, 1, 2]
    stat_names = ["solid_angle_proxy", "delta_diversity", "min_norm", "mean_adv_norm", "asr"]
    seed_aurocs = {n: [] for n in stat_names}

    for s in seeds:
        stats = boundary_geometry_stats(model, Xc, Yc, EPS, K_TRAJS, STEPS,
                                        alpha, SIGMA, seed=s)
        for n in stat_names:
            sign = -1 if n == "min_norm" else 1
            auc = C.safe_auroc(flips, sign * stats[n])
            seed_aurocs[n].append(auc)

    print(f"{'stat':>22} | {'mean':>8} | {'std':>8}")
    print("-" * 45)
    for n in stat_names:
        vals = np.array(seed_aurocs[n])
        print(f"{n:>22} | {vals.mean():8.4f} | {vals.std():8.4f}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
