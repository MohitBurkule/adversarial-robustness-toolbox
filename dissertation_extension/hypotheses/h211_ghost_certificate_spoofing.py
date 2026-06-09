"""
H211 - Ghost certificate spoofing attack on randomized smoothing.

Hypothesis: imperceptible adversarial perturbations (L-inf eps=0.1) can cause a
randomized smoothing certifier to issue a large certified radius for the WRONG
class -- a "ghost certificate". This is a fundamentally new attack class distinct
from misclassification attacks.

Grounded in arXiv:2511.14003 (AAAI 2026, "Certified but Fooled! Breaking
Certified Defences with Ghost Certificates").

Background:
  For input x, the certifier adds Gaussian noise N(0, sigma^2 I) N times and
  takes majority vote. The certified class c_A = argmax of vote counts, and the
  certified radius r = sigma * Phi^{-1}(lower_bound_p_A) where p_A = fraction of
  votes for c_A. The ghost certificate attack crafts delta s.t. the perturbed
  input x+delta is misclassified AND the certifier produces a LARGE r for that
  wrong class -- i.e., the certifier "certifies" the wrong prediction with high
  confidence.

Methodology:
  1. Train a standard CNN on Fashion-MNIST (n_train=6000, 15 epochs).
  2. Randomized smoothing: sigma=0.25, N_eval=500 MC samples for final eval,
     N_opt=100 for PGD optimization, M=5 noise samples per PGD step.
  3. For 100 clean test samples:
       - Baseline: smoothed prediction + certified radius for correctly classified.
  4. Ghost certificate attack via PGD-20:
       - Target class t = argmax of base model logits excluding true label y.
       - Maximize fraction of noisy forward passes predicting class t.
       - Constraint: ||delta||_inf <= eps=0.1.
  5. After attack: compute certified radius for ghost class on perturbed input.
  6. Metrics: ghost_rate, mean_ghost_radius vs mean_clean_radius, base_asr.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm, binom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign.common import (DEVICE, set_seed, load_dataset, dataset_meta,
                              build_model, train_model, logits_and_acc)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
EPS = 0.1
SIGMA = 0.25            # smoothing noise std
N_EVAL = 500            # MC samples for final certification
N_OPT = 100             # MC samples during PGD optimization (cheaper)
M_PGD = 5               # noise samples per PGD step
PGD_STEPS = 20
PGD_ALPHA = 2.5 * EPS / PGD_STEPS
N_TEST = 100            # number of test samples to attack
ALPHA_CONF = 0.001      # Clopper-Pearson confidence level


# ---------------------------------------------------------------------------
# Certification helpers
# ---------------------------------------------------------------------------
def certify(model, x, sigma, n_samples, batch_size=200):
    """Randomized smoothing certification for a single input x (1, C, H, W).

    Returns (predicted_class, certified_radius, p_A_lower_bound, counts).
    If abstain (no class gets > 50%), returns (-1, 0.0, 0.0, counts).
    """
    n_classes = 10
    counts = torch.zeros(n_classes, dtype=torch.long, device='cpu')

    for i in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - i)
        # x is (1, C, H, W) -> repeat to (bs, C, H, W)
        x_rep = x.repeat(bs, 1, 1, 1)
        noise = torch.randn_like(x_rep) * sigma
        with torch.no_grad():
            preds = model((x_rep + noise).clamp(0, 1)).argmax(1).cpu()
        for c in range(n_classes):
            counts[c] += (preds == c).sum().item()

    top2 = counts.topk(2)
    c_A = top2.indices[0].item()
    n_A = top2.values[0].item()

    # Clopper-Pearson lower bound on p_A
    p_A_lower = binom.ppf(ALPHA_CONF, n_samples, 1.0 * n_A / n_samples) / n_samples
    # More standard: use the beta distribution / Clopper-Pearson directly
    # Lower bound of Clopper-Pearson CI:
    if n_A == 0:
        p_A_lower = 0.0
    else:
        from scipy.stats import beta as beta_dist
        p_A_lower = beta_dist.ppf(ALPHA_CONF, n_A, n_samples - n_A + 1)

    if p_A_lower <= 0.5:
        return -1, 0.0, p_A_lower, counts

    radius = sigma * norm.ppf(p_A_lower)
    return c_A, radius, p_A_lower, counts


def certify_class(model, x, target_class, sigma, n_samples, batch_size=200):
    """Compute certified radius specifically for a given target_class.

    Returns (radius, p_target_lower, vote_fraction) regardless of whether
    target_class is the majority class.
    """
    n_classes = 10
    counts = torch.zeros(n_classes, dtype=torch.long, device='cpu')

    for i in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - i)
        x_rep = x.repeat(bs, 1, 1, 1)
        noise = torch.randn_like(x_rep) * sigma
        with torch.no_grad():
            preds = model((x_rep + noise).clamp(0, 1)).argmax(1).cpu()
        for c in range(n_classes):
            counts[c] += (preds == c).sum().item()

    n_target = counts[target_class].item()
    vote_frac = n_target / n_samples

    # Clopper-Pearson lower bound
    if n_target == 0:
        p_lower = 0.0
    else:
        from scipy.stats import beta as beta_dist
        p_lower = beta_dist.ppf(ALPHA_CONF, n_target, n_samples - n_target + 1)

    if p_lower <= 0.5:
        radius = 0.0
    else:
        radius = sigma * norm.ppf(p_lower)

    return radius, p_lower, vote_frac, counts


# ---------------------------------------------------------------------------
# Ghost certificate PGD attack
# ---------------------------------------------------------------------------
def ghost_pgd(model, x, target_class, sigma, eps, steps, alpha, m_noise):
    """PGD attack to maximize the fraction of noisy forward passes predicting
    target_class (the ghost class). Returns perturbed x+delta."""
    x0 = x.clone().detach()  # (1, C, H, W)
    delta = torch.empty_like(x0).uniform_(-eps, eps)
    delta = delta.clamp(-(x0), 1 - x0)  # keep x0+delta in [0,1]

    for step in range(steps):
        delta.requires_grad_(True)
        x_adv = (x0 + delta).clamp(0, 1)

        # Sample m_noise Gaussian perturbations, compute loss toward target_class
        x_rep = x_adv.repeat(m_noise, 1, 1, 1)
        noise = torch.randn_like(x_rep) * sigma
        logits = model((x_rep + noise).clamp(0, 1))

        # Maximize cross-entropy toward target class = minimize CE with target
        targets = torch.full((m_noise,), target_class, dtype=torch.long, device=DEVICE)
        loss = -F.cross_entropy(logits, targets)  # negative because we MAXIMIZE

        loss.backward()
        with torch.no_grad():
            grad = delta.grad.sign()
            delta = delta.detach() - alpha * grad  # gradient descent on -loss = ascent on target prob
            delta = delta.clamp(-eps, eps)
            delta = (x0 + delta).clamp(0, 1) - x0

    return (x0 + delta).clamp(0, 1).detach()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("H211 - Ghost certificate spoofing attack on randomized smoothing")
    print("=" * 74)
    print(f"Device={DEVICE}  EPS={EPS}  SIGMA={SIGMA}  N_EVAL={N_EVAL}")
    print(f"PGD_STEPS={PGD_STEPS}  M_PGD={M_PGD}  N_TEST={N_TEST}")
    print(f"SEED={SEED}")
    t_start = time.time()

    set_seed(SEED)
    meta = dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = load_dataset("fashion_mnist", n_train=6000, n_eval=1000, seed=SEED)

    # --- Train base model ---
    print("\n--- Training base CNN (15 epochs) ---")
    model = build_model("cnn", meta, width=32)
    train_model(model, Xtr, Ytr, epochs=15, batch=128, opt="adam", lr=1e-3,
                ncls=meta["n_classes"], verbose=True)
    model.eval()

    _, clean_acc = logits_and_acc(model, Xte, Yte)
    print(f"  Clean accuracy: {clean_acc:.4f}")

    # --- Select N_TEST correctly classified samples ---
    with torch.no_grad():
        preds = model(Xte).argmax(1)
        correct_mask = (preds == Yte)
    correct_idx = correct_mask.nonzero(as_tuple=True)[0][:N_TEST]
    X_eval = Xte[correct_idx]
    Y_eval = Yte[correct_idx]
    n_eval = X_eval.size(0)
    print(f"\n  Selected {n_eval} correctly-classified samples for evaluation")

    # --- Baseline certification on clean samples ---
    print("\n--- Baseline certification (clean samples) ---")
    clean_cert_class = []
    clean_cert_radius = []
    for i in range(n_eval):
        c_A, r, p_low, _ = certify(model, X_eval[i:i+1], SIGMA, N_EVAL)
        clean_cert_class.append(c_A)
        clean_cert_radius.append(r)
        if (i + 1) % 20 == 0:
            print(f"  certified {i+1}/{n_eval} ...")

    clean_cert_class = np.array(clean_cert_class)
    clean_cert_radius = np.array(clean_cert_radius)
    correctly_certified = (clean_cert_class == Y_eval.cpu().numpy())
    cert_rate = correctly_certified.mean()
    mean_clean_r = clean_cert_radius[correctly_certified].mean() if correctly_certified.any() else 0.0
    print(f"  Correctly certified: {correctly_certified.sum()}/{n_eval} ({cert_rate:.4f})")
    print(f"  Mean certified radius (correct): {mean_clean_r:.4f}")

    # --- Choose target class for each sample ---
    with torch.no_grad():
        logits_all = model(X_eval)
    target_classes = []
    for i in range(n_eval):
        l = logits_all[i].clone()
        l[Y_eval[i]] = -1e9
        target_classes.append(l.argmax().item())
    target_classes = np.array(target_classes)

    # --- Ghost certificate attack ---
    print("\n--- Ghost certificate PGD attack ---")
    ghost_radii = []
    ghost_vote_fracs = []
    ghost_succeeded = []
    base_model_flipped = []

    for i in range(n_eval):
        t_cls = int(target_classes[i])
        x_adv = ghost_pgd(model, X_eval[i:i+1], t_cls, SIGMA, EPS,
                           PGD_STEPS, PGD_ALPHA, M_PGD)

        # Check base model prediction on adversarial
        with torch.no_grad():
            adv_pred = model(x_adv).argmax(1).item()
        base_model_flipped.append(adv_pred != Y_eval[i].item())

        # Certify the ghost class on perturbed input
        r_ghost, p_low, vote_frac, counts = certify_class(
            model, x_adv, t_cls, SIGMA, N_EVAL)

        # Also check: is ghost class the majority?
        c_A_adv, r_adv, _, _ = certify(model, x_adv, SIGMA, N_EVAL)
        ghost_success = (c_A_adv == t_cls) and (r_adv > 0)

        ghost_radii.append(r_ghost)
        ghost_vote_fracs.append(vote_frac)
        ghost_succeeded.append(ghost_success)

        if (i + 1) % 20 == 0:
            print(f"  attacked {i+1}/{n_eval} ... ghost_rate so far: "
                  f"{np.mean(ghost_succeeded):.4f}")

    ghost_radii = np.array(ghost_radii)
    ghost_vote_fracs = np.array(ghost_vote_fracs)
    ghost_succeeded = np.array(ghost_succeeded)
    base_model_flipped = np.array(base_model_flipped)

    # --- Metrics ---
    ghost_rate = ghost_succeeded.mean()
    base_asr = base_model_flipped.mean()

    if ghost_succeeded.any():
        mean_ghost_r = ghost_radii[ghost_succeeded].mean()
        max_ghost_r = ghost_radii[ghost_succeeded].max()
    else:
        mean_ghost_r = 0.0
        max_ghost_r = 0.0

    # Compare: for samples that were ghost-certified, what was their clean radius?
    ghost_idx = np.where(ghost_succeeded)[0]
    if len(ghost_idx) > 0:
        clean_r_for_ghosts = clean_cert_radius[ghost_idx]
        mean_clean_r_matched = clean_r_for_ghosts.mean()
    else:
        mean_clean_r_matched = 0.0
        clean_r_for_ghosts = np.array([])

    elapsed = time.time() - t_start

    # --- Report ---
    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    print(f"  Clean accuracy:                   {clean_acc:.4f}")
    print(f"  Correctly certified (clean):      {correctly_certified.sum()}/{n_eval}")
    print(f"  Mean clean certified radius:      {mean_clean_r:.4f}")
    print(f"")
    print(f"  Ghost certificate rate:           {ghost_rate:.4f} ({ghost_succeeded.sum()}/{n_eval})")
    print(f"  Base model ASR (no smoothing):    {base_asr:.4f}")
    print(f"  Mean ghost radius (successful):   {mean_ghost_r:.4f}")
    print(f"  Max ghost radius:                 {max_ghost_r:.4f}")
    print(f"  Mean clean radius (matched):      {mean_clean_r_matched:.4f}")
    print(f"  Mean ghost vote fraction:         {ghost_vote_fracs.mean():.4f}")
    print(f"")

    if ghost_succeeded.any():
        ratio = mean_ghost_r / mean_clean_r_matched if mean_clean_r_matched > 0 else float('inf')
        print(f"  Ghost/Clean radius ratio:         {ratio:.4f}")
        exceeds = (ghost_radii[ghost_succeeded] > clean_cert_radius[ghost_succeeded]).sum()
        print(f"  Ghost r > Clean r (same sample):  {exceeds}/{ghost_succeeded.sum()}")
    else:
        print(f"  (No successful ghost certificates)")

    # --- 5 Example cases ---
    print(f"\n--- Example ghost certificates (up to 5) ---")
    examples = ghost_idx[:5] if len(ghost_idx) >= 5 else ghost_idx
    if len(examples) == 0:
        # show top-5 by ghost vote fraction even if not certified
        top5 = np.argsort(-ghost_vote_fracs)[:5]
        print("  (No certified ghosts; showing top-5 by ghost vote fraction)")
        for rank, idx in enumerate(top5):
            print(f"  [{rank}] sample={idx}  true_class={Y_eval[idx].item()}  "
                  f"ghost_class={target_classes[idx]}  "
                  f"clean_r={clean_cert_radius[idx]:.4f}  "
                  f"ghost_r={ghost_radii[idx]:.4f}  "
                  f"ghost_vote_frac={ghost_vote_fracs[idx]:.4f}  "
                  f"base_flipped={base_model_flipped[idx]}")
    else:
        for rank, idx in enumerate(examples):
            print(f"  [{rank}] sample={idx}  true_class={Y_eval[idx].item()}  "
                  f"ghost_class={target_classes[idx]}  "
                  f"clean_r={clean_cert_radius[idx]:.4f}  "
                  f"ghost_r={ghost_radii[idx]:.4f}  "
                  f"ghost_vote_frac={ghost_vote_fracs[idx]:.4f}")

    # --- Hypothesis test ---
    print(f"\n--- Hypothesis test ---")
    if ghost_succeeded.any() and mean_ghost_r > mean_clean_r_matched:
        print("  SUPPORTED: ghost certificates have LARGER radius than clean certificates")
        print(f"  (mean_ghost_r={mean_ghost_r:.4f} > mean_clean_r_matched={mean_clean_r_matched:.4f})")
    elif ghost_succeeded.any():
        print("  PARTIAL: ghost certificates exist but radius is smaller than clean")
        print(f"  (mean_ghost_r={mean_ghost_r:.4f} <= mean_clean_r_matched={mean_clean_r_matched:.4f})")
    else:
        print("  NOT SUPPORTED: no ghost certificates found at eps=0.1, sigma=0.25")

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
