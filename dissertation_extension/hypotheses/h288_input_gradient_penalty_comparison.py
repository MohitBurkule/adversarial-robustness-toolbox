"""
H288 - Input Gradient Penalty Comparison: All Flatness Approaches on Equal Footing.

Definitive comparison of seven training strategies for finding flat/smooth minima
and their effect on adversarial robustness.  The Adam conditions directly test
whether Adam's second-moment normalisation already captures what the inter-batch
variance penalty adds; if (g)≈(f) in robustness gain over baseline, Adam makes
the explicit penalty redundant.

Conditions:
  (a) SGD baseline — standard CE, SGD+momentum
  (b) SGD + input gradient norm penalty (Ross & Doshi-Velez 2018):
          L = CE + λ · ||∇_x L||²
  (c) SGD + inter-batch gradient variance penalty (H286 best λ)
  (d) SGD + Hutchinson Hessian trace penalty (H287 best λ)
  (e) SGD + SAM optimiser (Foret et al. 2021, ρ=0.05)
  (f) Adam baseline — CE only, Adam optimiser
  (g) Adam + inter-batch gradient variance penalty (same λ as condition c)

λ values are chosen to give similar clean accuracy degradation (~2% below baseline).
N_train=6000, 10 epochs each.

Metrics: clean_acc, FGSM_ASR, PGD_ASR, mean_margin, post-hoc sharpness.

Post-hoc sharpness: perturb weights with 20 random unit-sphere vectors scaled
by ρ=0.05; measure mean loss increase relative to the base loss at w*.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH_SIZE = 128
SEED = 0
EPS_FGSM = 0.1
EPS_PGD = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
SAM_RHO = 0.05
SHARPNESS_SAMPLES = 20
SHARPNESS_RHO = 0.05

# λ for each penalised method — tune to ≈ 2% clean-acc drop vs baseline
LAM_INPUT_GRAD = 0.01      # input gradient norm penalty
LAM_BATCH_VAR = 0.01       # inter-batch gradient variance (H286 best)
LAM_HESS = 0.001           # Hutchinson Hessian trace (H287 best)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist"
)
OUT_FILE = os.path.join(RESULTS_DIR, "h288_input_gradient_penalty_comparison_output.txt")


# ===========================================================================
# Training routines
# ===========================================================================

def _make_sgd(model, lr=LR):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


def _make_adam(model, lr=1e-3):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)


def train_baseline(model, Xtr, Ytr, epochs, lr=LR):
    """Standard SGD + cross-entropy (condition a)."""
    opt = _make_sgd(model, lr)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - BATCH_SIZE, BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            xb = Xtr[idx].to(C.DEVICE)
            yb = Ytr[idx].to(C.DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    return model


def train_adam_baseline(model, Xtr, Ytr, epochs):
    """Standard Adam + cross-entropy (condition f)."""
    opt = _make_adam(model)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - BATCH_SIZE, BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            xb = Xtr[idx].to(C.DEVICE)
            yb = Ytr[idx].to(C.DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    return model


def train_adam_batch_variance_penalty(model, Xtr, Ytr, epochs, lam):
    """Adam + inter-batch gradient variance penalty (condition g).

    Same penalty logic as condition (c) but with Adam base optimiser.
    Tests whether Adam's second-moment already subsumes the variance penalty.
    """
    opt = _make_adam(model)
    n = len(Xtr)
    params = list(model.parameters())

    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - 2 * BATCH_SIZE, 2 * BATCH_SIZE):
            idx1 = perm[i: i + BATCH_SIZE]
            idx2 = perm[i + BATCH_SIZE: i + 2 * BATCH_SIZE]
            xb1, yb1 = Xtr[idx1].to(C.DEVICE), Ytr[idx1].to(C.DEVICE)
            xb2, yb2 = Xtr[idx2].to(C.DEVICE), Ytr[idx2].to(C.DEVICE)

            opt.zero_grad()
            F.cross_entropy(model(xb1), yb1).backward()
            g1 = [p.grad.detach().clone() for p in params if p.grad is not None]

            opt.zero_grad()
            F.cross_entropy(model(xb2), yb2).backward()
            g2 = [p.grad.detach().clone() for p in params if p.grad is not None]

            variance = sum((a - b).pow(2).sum() for a, b in zip(g1, g2)).item()

            opt.zero_grad()
            loss_combined = (
                F.cross_entropy(model(xb1), yb1)
                + F.cross_entropy(model(xb2), yb2)
            )
            loss_combined.backward()

            if lam > 0.0:
                scale = min(1.0 + lam * variance, 10.0)
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(scale)

            opt.step()
    return model


def train_input_grad_penalty(model, Xtr, Ytr, epochs, lam, lr=LR):
    """Input gradient norm penalty: L = CE + λ·||∇_x L||² (condition b)."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - BATCH_SIZE, BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            xb = Xtr[idx].to(C.DEVICE).clone().requires_grad_(True)
            yb = Ytr[idx].to(C.DEVICE)

            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            # Input gradient with graph so we can diff through it
            input_grads = torch.autograd.grad(
                loss, xb, create_graph=True, retain_graph=True
            )[0]
            grad_penalty = input_grads.pow(2).sum(dim=(1, 2, 3)).mean()
            total = loss + lam * grad_penalty
            total.backward()
            opt.step()
    return model


def train_batch_variance_penalty(model, Xtr, Ytr, epochs, lam, lr=LR):
    """Inter-batch gradient variance penalty (condition c, H286 approach)."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    n = len(Xtr)
    params = list(model.parameters())

    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - 2 * BATCH_SIZE, 2 * BATCH_SIZE):
            idx1 = perm[i: i + BATCH_SIZE]
            idx2 = perm[i + BATCH_SIZE: i + 2 * BATCH_SIZE]
            xb1, yb1 = Xtr[idx1].to(C.DEVICE), Ytr[idx1].to(C.DEVICE)
            xb2, yb2 = Xtr[idx2].to(C.DEVICE), Ytr[idx2].to(C.DEVICE)

            # Detached gradient from batch 1
            opt.zero_grad()
            F.cross_entropy(model(xb1), yb1).backward()
            g1 = [p.grad.detach().clone() for p in params if p.grad is not None]

            # Detached gradient from batch 2
            opt.zero_grad()
            F.cross_entropy(model(xb2), yb2).backward()
            g2 = [p.grad.detach().clone() for p in params if p.grad is not None]

            variance = sum((a - b).pow(2).sum() for a, b in zip(g1, g2)).item()

            # Combined loss step with variance scaling
            opt.zero_grad()
            loss_combined = (
                F.cross_entropy(model(xb1), yb1)
                + F.cross_entropy(model(xb2), yb2)
            )
            loss_combined.backward()

            if lam > 0.0:
                scale = min(1.0 + lam * variance, 10.0)
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(scale)

            opt.step()
    return model


def train_hessian_trace_penalty(model, Xtr, Ytr, epochs, lam, lr=LR):
    """Hutchinson Hessian trace penalty (condition d, H287 approach)."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    n = len(Xtr)
    params = list(model.parameters())

    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - BATCH_SIZE, BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            xb = Xtr[idx].to(C.DEVICE)
            yb = Ytr[idx].to(C.DEVICE)

            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)

            if lam > 0.0:
                # Pass 1: CE loss backward
                loss = F.cross_entropy(model(xb), yb)
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
                v_list = [
                    torch.randint(0, 2, g.shape, device=g.device).float() * 2 - 1
                    for g in grads
                ]
                gv = sum((g * v).sum() for g, v in zip(grads, v_list))
                Hvp = torch.autograd.grad(gv, params, retain_graph=True)
                hess_trace_est = sum((v * hv).sum() for v, hv in zip(v_list, Hvp))
                total = loss + lam * hess_trace_est
                total.backward()
            else:
                loss.backward()

            opt.step()
    return model


class SAMOptimiser:
    """Sharpness-Aware Minimisation (Foret et al. 2021).

    Wraps an SGD base optimiser. Each step:
      1. Compute gradients at w
      2. Compute perturbation e_w = ρ * grad / ||grad||
      3. Perturb w ← w + e_w (first ascent step)
      4. Recompute gradients at w + e_w
      5. Restore w ← w - e_w; step base optimiser with new gradients
    """

    def __init__(self, params, lr, momentum, weight_decay, rho):
        self.base_opt = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.param_groups = self.base_opt.param_groups
        self.rho = rho

    def zero_grad(self):
        self.base_opt.zero_grad()

    @torch.no_grad()
    def first_step(self):
        """Perturb weights toward sharpest ascent direction."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad.detach() * scale
                p.add_(e_w)  # w ← w + e_w
                # Store perturbation for restoration
                p._sam_e_w = e_w

    @torch.no_grad()
    def second_step(self):
        """Restore weights and apply base optimiser step."""
        for group in self.param_groups:
            for p in group["params"]:
                if not hasattr(p, "_sam_e_w"):
                    continue
                p.sub_(p._sam_e_w)  # w ← w - e_w
        self.base_opt.step()

    def _grad_norm(self):
        all_norms = [
            p.grad.detach().norm(2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.stack(all_norms).norm(2)


def train_sam(model, Xtr, Ytr, epochs, rho=SAM_RHO, lr=LR):
    """SAM optimiser training (condition e)."""
    opt = SAMOptimiser(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, rho=rho
    )
    n = len(Xtr)

    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - BATCH_SIZE, BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            xb = Xtr[idx].to(C.DEVICE)
            yb = Ytr[idx].to(C.DEVICE)

            # First forward-backward at w
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.first_step()

            # Second forward-backward at w + e_w
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.second_step()

    return model


# ===========================================================================
# Post-hoc sharpness
# ===========================================================================

def compute_sharpness(model, X, Y, n_samples=SHARPNESS_SAMPLES, rho=SHARPNESS_RHO):
    """Estimate sharpness: mean loss increase under random weight perturbations.

    Sample n_samples random directions on the unit sphere scaled by ρ,
    perturb w, measure loss, restore w.  Sharpness = mean(ΔLoss).
    """
    model.eval()
    n = len(X)
    idx = torch.randperm(n)[:256]
    xb = X[idx].to(C.DEVICE)
    yb = Y[idx].to(C.DEVICE)

    with torch.no_grad():
        base_loss = F.cross_entropy(model(xb), yb).item()

    params = list(model.parameters())
    deltas = []

    for _ in range(n_samples):
        # Random unit direction
        noise = [torch.randn_like(p) for p in params]
        noise_norm = (sum(v.pow(2).sum() for v in noise) ** 0.5).item()
        scale = rho / (noise_norm + 1e-12)

        # Perturb
        with torch.no_grad():
            for p, n_ in zip(params, noise):
                p.add_(n_ * scale)

        with torch.no_grad():
            pert_loss = F.cross_entropy(model(xb), yb).item()

        # Restore
        with torch.no_grad():
            for p, n_ in zip(params, noise):
                p.sub_(n_ * scale)

        deltas.append(pert_loss - base_loss)

    model.train()
    return float(np.mean(deltas))


# ===========================================================================
# Evaluation
# ===========================================================================

def eval_model(model, Xte, Yte):
    for p in model.parameters():
        p.requires_grad_(True)

    _, clean_acc = C.logits_and_acc(model, Xte, Yte)

    Xadv_fgsm = C.fgsm(model, Xte, Yte, eps=EPS_FGSM)
    _, acc_fgsm = C.logits_and_acc(model, Xadv_fgsm, Yte)
    fgsm_asr = 1.0 - acc_fgsm

    Xadv_pgd = C.pgd(model, Xte, Yte, eps=EPS_PGD, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, acc_pgd = C.logits_and_acc(model, Xadv_pgd, Yte)
    pgd_asr = 1.0 - acc_pgd

    margins = C.margin(model, Xte, Yte)
    mean_margin = float(np.mean(margins))

    return {
        "clean_acc": float(clean_acc),
        "fgsm_asr": float(fgsm_asr),
        "pgd_asr": float(pgd_asr),
        "mean_margin": mean_margin,
    }


# ===========================================================================
# Main
# ===========================================================================

CONDITIONS = [
    ("baseline",          "SGD baseline (CE only)",                          None),
    ("input_grad",        f"SGD + input grad penalty (λ={LAM_INPUT_GRAD})",  LAM_INPUT_GRAD),
    ("batch_var",         f"SGD + batch var penalty (λ={LAM_BATCH_VAR})",    LAM_BATCH_VAR),
    ("hess_trace",        f"SGD + Hessian trace penalty (λ={LAM_HESS})",     LAM_HESS),
    ("sam",               f"SGD + SAM (ρ={SAM_RHO})",                        SAM_RHO),
    ("adam_baseline",     "Adam baseline (CE only)",                         None),
    ("adam_batch_var",    f"Adam + batch var penalty (λ={LAM_BATCH_VAR})",   LAM_BATCH_VAR),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    C.set_seed(SEED)

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    idx = torch.randperm(len(Xtr))[:N_TRAIN]
    Xtr, Ytr = Xtr[idx], Ytr[idx]

    results = []

    for cond_key, cond_label, cond_lam in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"Condition: {cond_label}")

        C.set_seed(SEED)
        model = C.build_model(
            "cnn", {"channels": 1, "size": 28, "n_classes": 10}, width=32, seed=SEED
        )
        model.to(C.DEVICE)

        t_start = time.time()

        if cond_key == "baseline":
            model = train_baseline(model, Xtr, Ytr, epochs=EPOCHS)
        elif cond_key == "input_grad":
            model = train_input_grad_penalty(model, Xtr, Ytr, epochs=EPOCHS, lam=cond_lam)
        elif cond_key == "batch_var":
            model = train_batch_variance_penalty(model, Xtr, Ytr, epochs=EPOCHS, lam=cond_lam)
        elif cond_key == "hess_trace":
            model = train_hessian_trace_penalty(model, Xtr, Ytr, epochs=EPOCHS, lam=cond_lam)
        elif cond_key == "sam":
            model = train_sam(model, Xtr, Ytr, epochs=EPOCHS, rho=cond_lam)
        elif cond_key == "adam_baseline":
            model = train_adam_baseline(model, Xtr, Ytr, epochs=EPOCHS)
        elif cond_key == "adam_batch_var":
            model = train_adam_batch_variance_penalty(model, Xtr, Ytr, epochs=EPOCHS, lam=cond_lam)

        train_time = time.time() - t_start
        print(f"  Training time: {train_time:.1f}s")

        metrics = eval_model(model, Xte, Yte)

        print("  Computing post-hoc sharpness...")
        sharpness = compute_sharpness(model, Xte, Yte)

        metrics["condition"] = cond_label
        metrics["train_time_s"] = train_time
        metrics["sharpness"] = sharpness

        results.append(metrics)
        print(
            f"  clean_acc={metrics['clean_acc']:.4f}  "
            f"fgsm_asr={metrics['fgsm_asr']:.4f}  "
            f"pgd_asr={metrics['pgd_asr']:.4f}  "
            f"mean_margin={metrics['mean_margin']:.4f}  "
            f"sharpness={sharpness:.4f}"
        )

    # Write results
    lines = [
        "H288 Input Gradient Penalty Comparison — Definitive Flatness vs Robustness",
        "=" * 70,
        f"Dataset: {DS}  N_train={N_TRAIN}  Epochs={EPOCHS}  Seed={SEED}",
        f"EPS_FGSM={EPS_FGSM}  EPS_PGD={EPS_PGD}  PGD_STEPS={PGD_STEPS}",
        f"SAM_RHO={SAM_RHO}  Sharpness_rho={SHARPNESS_RHO}  Sharpness_samples={SHARPNESS_SAMPLES}",
        "",
    ]

    col_w = 38
    header = (
        f"{'Condition':<{col_w}}  {'clean_acc':>10}  {'fgsm_asr':>9}  "
        f"{'pgd_asr':>8}  {'margin':>8}  {'sharpness':>10}  {'time_s':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        lines.append(
            f"{r['condition']:<{col_w}}  {r['clean_acc']:>10.4f}  {r['fgsm_asr']:>9.4f}  "
            f"{r['pgd_asr']:>8.4f}  {r['mean_margin']:>8.4f}  "
            f"{r['sharpness']:>10.4f}  {r['train_time_s']:>7.1f}"
        )

    # Identify best condition by PGD ASR
    best = min(results, key=lambda r: r["pgd_asr"])
    baseline_acc = results[0]["clean_acc"]

    lines += [
        "",
        "Summary:",
        f"  Best PGD robustness: {best['condition']}  "
        f"PGD_ASR={best['pgd_asr']:.4f}  clean_acc={best['clean_acc']:.4f}",
        f"  Baseline clean_acc: {baseline_acc:.4f}",
        f"  Clean-acc degradation of best vs baseline: "
        f"{(baseline_acc - best['clean_acc'])*100:.2f}%",
        "",
        "Analysis:",
        "- Sharpness (post-hoc ρ perturbation) should anti-correlate with robustness",
        "- Input gradient penalty (Ross 2018) is a natural adversarial defence by design",
        "- Batch-variance penalty (H286) and Hessian trace (H287) are weight-space flatness methods",
        "- SAM directly minimises worst-case loss in a weight neighbourhood",
        "- Compute cost: baseline < batch_var ≈ SAM < input_grad < hess_trace",
        "- Adam baseline vs SGD baseline reveals optimiser effect independent of penalty",
        "- If Adam+penalty (g) ≈ Adam baseline (f): second moment subsumes variance penalty",
        "- If Adam+penalty (g) > Adam baseline (f) by same margin as (c)>(a): penalty adds value beyond Adam",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
