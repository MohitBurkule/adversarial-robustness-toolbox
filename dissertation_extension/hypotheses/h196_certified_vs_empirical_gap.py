"""
H196 - Certified vs empirical robustness gap.

Hypothesis (arXiv:2410.06895 ICML 2025, arXiv:2504.02412):
  Randomised smoothing (sigma=0.25) certified accuracy at L2 radius 0.25
  underestimates true PGD-L2 empirical robustness for both natural and AT
  models, but the gap is larger for natural models (certified/empirical ratio
  ~0.55-0.70) than AT models (~0.80-0.92).

Protocol:
  - Train two SmallCNNs on Fashion-MNIST (n_train=6000, 15 epochs):
      Model A: standard cross-entropy (natural).
      Model B: adversarial training with PGD-3 L2 eps=0.5.
  - For each model, evaluate 200 test samples:
      (a) Certified accuracy via randomised smoothing:
          - N=200 Gaussian perturbations per sample (sigma=0.25).
          - Majority-vote class count n_A; Clopper-Pearson lower bound p_hat_A
            at alpha=0.001 via Beta distribution.
          - Certified iff p_hat_A > 0.5 AND sigma * Phi^-1(p_hat_A) >= 0.25,
            AND majority class == true label.
      (b) Empirical robust accuracy under PGD-L2:
          - PGD-20, L2 eps=0.25, step_size=0.05, 3 random restarts.
          - Robust accuracy = fraction not fooled among correctly-classified.
  - Report: clean_acc, certified_acc, empirical_robust_acc, gap, ratio per model.
  - Test: ratio_natural < ratio_AT (natural model has larger gap).

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm, beta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign.common import (
    DEVICE, set_seed, load_dataset, dataset_meta,
    build_model, train_model, logits_and_acc,
)

# ── hyperparameters ──────────────────────────────────────────────────────────
DATASET   = "fashion_mnist"
N_TRAIN   = 6000
N_EVAL    = 1000
SEED      = 42
EPOCHS    = 15
CERT_N    = 200       # samples to certify / attack
SIGMA     = 0.25      # smoothing noise std
N_SMOOTH  = 200       # number of noisy forward passes per sample
ALPHA     = 0.001     # confidence level for Clopper-Pearson
CERT_R    = 0.25      # L2 certification radius
PGD_EPS   = 0.25      # L2 epsilon for empirical attack
PGD_STEPS = 20
PGD_ALPHA = 0.05
PGD_RESTARTS = 3
AT_EPS    = 0.5       # L2 epsilon for adversarial training
AT_STEPS  = 3


# ── PGD-L2 attack ───────────────────────────────────────────────────────────
def pgd_l2(model, x, y, eps, steps, alpha, random_start=True):
    """PGD projected onto L2 ball of radius eps around x."""
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        noise = torch.randn_like(xa)
        noise = noise / (noise.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1) + 1e-12) * eps
        xa = (xa + noise).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        # normalise gradient to unit L2
        g_flat = g.flatten(1)
        g_norm = g_flat.norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1) + 1e-12
        xa = xa.detach() + alpha * (g / g_norm)
        # project back onto L2 ball
        delta = xa - x0
        d_flat = delta.flatten(1)
        d_norm = d_flat.norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        factor = torch.clamp(d_norm / eps, min=1.0)
        xa = (x0 + delta / factor).clamp(0, 1)
    return xa.detach()


# ── adversarial training with PGD-L2 ────────────────────────────────────────
def train_model_at_l2(model, Xtr, Ytr, epochs, batch=128, lr=0.05,
                       at_eps=0.5, at_steps=3):
    """Standard adversarial training using PGD-L2 inner maximisation."""
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_adv = pgd_l2(model, xb, yb, eps=at_eps, steps=at_steps,
                            alpha=2.5 * at_eps / max(at_steps, 1), random_start=True)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ── randomised smoothing certification ───────────────────────────────────────
@torch.no_grad()
def certify_batch(model, X, Y, sigma, n_smooth, alpha, radius):
    """Return (certified_correct, clean_correct) boolean arrays of len(X)."""
    N = X.size(0)
    ncls = 10
    certified = np.zeros(N, dtype=bool)
    clean_correct = np.zeros(N, dtype=bool)

    batch_smooth = 100  # forward this many noisy copies at once
    for i in range(N):
        x_i = X[i:i+1]  # (1,C,H,W)
        y_i = Y[i].item()

        # clean prediction
        pred_clean = model(x_i).argmax(1).item()
        clean_correct[i] = (pred_clean == y_i)

        # count votes under noise
        counts = torch.zeros(ncls, dtype=torch.long, device='cpu')
        done = 0
        while done < n_smooth:
            bs = min(batch_smooth, n_smooth - done)
            noisy = x_i.expand(bs, -1, -1, -1) + torch.randn(bs, *x_i.shape[1:], device=DEVICE) * sigma
            noisy = noisy.clamp(0, 1)
            preds = model(noisy).argmax(1).cpu()
            for c in range(ncls):
                counts[c] += (preds == c).sum()
            done += bs

        top_class = counts.argmax().item()
        n_A = counts[top_class].item()

        # Clopper-Pearson lower bound
        if n_A == 0:
            p_lower = 0.0
        else:
            p_lower = beta.ppf(alpha, n_A, n_smooth - n_A + 1)

        if p_lower > 0.5 and top_class == y_i:
            cert_radius = sigma * norm.ppf(p_lower)
            if cert_radius >= radius:
                certified[i] = True

    return certified, clean_correct


# ── empirical PGD-L2 robust accuracy ────────────────────────────────────────
@torch.no_grad()
def _pred(model, x, batch=256):
    preds = []
    for i in range(0, x.size(0), batch):
        preds.append(model(x[i:i+batch]).argmax(1))
    return torch.cat(preds)


def empirical_robust_acc(model, X, Y, eps, steps, alpha, restarts):
    """Fraction of correctly-classified samples that survive all restarts."""
    model.eval()
    with torch.no_grad():
        clean_pred = _pred(model, X)
        correct = (clean_pred == Y)

    if correct.sum() == 0:
        return 0.0, correct.cpu().numpy()

    # track which correct samples survive all restarts
    survived = correct.clone()  # start with all correct

    for r in range(restarts):
        # only attack survived samples (optimisation)
        idx = torch.where(survived)[0]
        if len(idx) == 0:
            break
        x_sub, y_sub = X[idx], Y[idx]
        x_adv = pgd_l2(model, x_sub, y_sub, eps=eps, steps=steps,
                        alpha=alpha, random_start=True)
        with torch.no_grad():
            adv_pred = _pred(model, x_adv)
        fooled = (adv_pred != y_sub)
        survived[idx[fooled]] = False

    n_correct = correct.sum().item()
    n_survived = survived.sum().item()
    return n_survived / n_correct, correct.cpu().numpy()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    set_seed(SEED)
    meta = dataset_meta(DATASET)
    Xtr, Ytr, Xte, Yte = load_dataset(DATASET, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # subset for certification / attack
    Xc, Yc = Xte[:CERT_N], Yte[:CERT_N]

    results = {}

    for label, do_at in [("natural", False), ("AT_L2", True)]:
        print(f"\n{'='*60}")
        print(f"Training {label} model ...")
        set_seed(SEED)
        model = build_model("cnn", meta, width=32)

        if do_at:
            model = train_model_at_l2(model, Xtr, Ytr, epochs=EPOCHS,
                                       at_eps=AT_EPS, at_steps=AT_STEPS)
        else:
            model = train_model(model, Xtr, Ytr, epochs=EPOCHS, opt="sgd", lr=0.05)

        _, clean_acc = logits_and_acc(model, Xte, Yte)
        print(f"  Clean accuracy (full eval): {clean_acc:.4f}")

        # certified accuracy
        print(f"  Computing certified accuracy (N_smooth={N_SMOOTH}, sigma={SIGMA}) ...")
        certified, clean_corr = certify_batch(model, Xc, Yc, SIGMA, N_SMOOTH, ALPHA, CERT_R)
        cert_acc = certified.mean()

        # empirical robust accuracy
        print(f"  Computing empirical robust accuracy (PGD-L2, eps={PGD_EPS}, "
              f"steps={PGD_STEPS}, restarts={PGD_RESTARTS}) ...")
        emp_acc, emp_correct = empirical_robust_acc(
            model, Xc, Yc, eps=PGD_EPS, steps=PGD_STEPS,
            alpha=PGD_ALPHA, restarts=PGD_RESTARTS)

        gap = emp_acc - cert_acc
        ratio = cert_acc / emp_acc if emp_acc > 0 else float('nan')

        results[label] = {
            "clean_acc": clean_acc,
            "certified_acc": float(cert_acc),
            "empirical_robust_acc": float(emp_acc),
            "gap": float(gap),
            "ratio": float(ratio),
        }

        print(f"  Certified acc: {cert_acc:.4f}")
        print(f"  Empirical robust acc: {emp_acc:.4f}")
        print(f"  Gap (emp - cert): {gap:.4f}")
        print(f"  Ratio (cert/emp): {ratio:.4f}")

    # ── test hypothesis ──────────────────────────────────────────────────────
    r_nat = results["natural"]["ratio"]
    r_at  = results["AT_L2"]["ratio"]
    hypothesis_supported = r_nat < r_at

    elapsed = time.time() - t0

    # ── print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("H196 - Certified vs Empirical Robustness Gap")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  dataset={DATASET}, n_train={N_TRAIN}, n_eval={N_EVAL}, seed={SEED}")
    print(f"  cert: sigma={SIGMA}, n_smooth={N_SMOOTH}, alpha={ALPHA}, radius={CERT_R}")
    print(f"  attack: PGD-L2 eps={PGD_EPS}, steps={PGD_STEPS}, restarts={PGD_RESTARTS}")
    print(f"  AT: L2 eps={AT_EPS}, steps={AT_STEPS}")
    print(f"  cert_samples={CERT_N}")

    print(f"\n{'Model':<12} {'Clean':>8} {'Certified':>10} {'Empirical':>10} {'Gap':>8} {'Ratio':>8}")
    print("-" * 60)
    for label in ["natural", "AT_L2"]:
        r = results[label]
        print(f"{label:<12} {r['clean_acc']:>8.4f} {r['certified_acc']:>10.4f} "
              f"{r['empirical_robust_acc']:>10.4f} {r['gap']:>8.4f} {r['ratio']:>8.4f}")

    print(f"\nHypothesis: ratio_natural ({r_nat:.4f}) < ratio_AT ({r_at:.4f})")
    print(f"  → {'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}")
    print(f"\nElapsed: {elapsed:.1f}s")

    # ── save results ─────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h196_certified_vs_empirical_gap_output.txt")
    with open(out_path, "w") as f:
        f.write("H196 - Certified vs Empirical Robustness Gap\n")
        f.write("=" * 60 + "\n")
        f.write(f"dataset={DATASET}, n_train={N_TRAIN}, n_eval={N_EVAL}, seed={SEED}\n")
        f.write(f"cert: sigma={SIGMA}, n_smooth={N_SMOOTH}, alpha={ALPHA}, radius={CERT_R}\n")
        f.write(f"attack: PGD-L2 eps={PGD_EPS}, steps={PGD_STEPS}, restarts={PGD_RESTARTS}\n")
        f.write(f"AT: L2 eps={AT_EPS}, steps={AT_STEPS}, cert_samples={CERT_N}\n\n")
        f.write(f"{'Model':<12} {'Clean':>8} {'Certified':>10} {'Empirical':>10} {'Gap':>8} {'Ratio':>8}\n")
        f.write("-" * 60 + "\n")
        for label in ["natural", "AT_L2"]:
            r = results[label]
            f.write(f"{label:<12} {r['clean_acc']:>8.4f} {r['certified_acc']:>10.4f} "
                    f"{r['empirical_robust_acc']:>10.4f} {r['gap']:>8.4f} {r['ratio']:>8.4f}\n")
        f.write(f"\nHypothesis: ratio_natural ({r_nat:.4f}) < ratio_AT ({r_at:.4f})\n")
        f.write(f"  → {'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}\n")
        f.write(f"\nElapsed: {elapsed:.1f}s\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
