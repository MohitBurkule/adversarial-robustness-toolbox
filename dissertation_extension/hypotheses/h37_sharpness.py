"""
H37: Per-sample SAM-style sharpness in WEIGHT space predicts adversarial vulnerability.

Hypothesis: Samples whose loss L(x, y; theta) is sensitive to small perturbations
of the model parameters theta -- i.e. "sharp" minima in weight space when viewed
through a single example -- are also adversarially fragile in input space.

We measure per-sample weight-space sharpness in two ways:

  1. Weight-gradient norm:
         g_theta_norm(x, y) = || grad_theta L(x, y; theta) ||_2
     (sum-of-squares norm aggregated over all trainable parameter tensors,
     then sqrt). A larger gradient norm means the loss surface is locally
     steeper in parameter space at this sample.

  2. SAM sharpness probe (Foret et al., ICLR 2020):
         sharp_SAM(x, y) = L(x, y; theta + eps * sign(grad_theta L)) - L(x, y; theta)
     This is the worst-case-direction loss increase under a sign-step ascent in
     weight space of radius eps (here eps = 0.05, applied parameter-wise). The
     reference SAM paper formalises sharpness as max_{||eps||<=rho} L(theta+eps)
     - L(theta) and uses the L2 dual norm to get the optimal direction; the
     sign-step variant is the L_inf relaxation widely used as a probe.

Background: Foret, Kleiner, Mobahi, Neyshabur, "Sharpness-Aware Minimization
for Efficiently Improving Generalization", ICLR 2021 (arXiv:2010.01412, 2020).
SAM links flat minima in parameter space to better generalization. We extend
the question to the *per-sample* level: do individually-sharp samples coincide
with adversarially-fragile samples?

Baselines: victim_margin, mean_pix, std_pix, input_grad_L2_norm.
Targets: flipped_FGSM (eps=15/255), flipped_PGD, FGSM_min_eps (binary search).

Each sample needs its own backward pass through theta, so we chunk samples to
~32 at a time and loop inside the chunk (PyTorch sums grads across the batch
into a single buffer otherwise).

Self-contained: torchvision Fashion-MNIST in /tmp/data, single CUDA device.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0
SAM_EPS = 0.05            # SAM-style weight-space ball radius (L_inf sign step)
SHARP_CHUNK = 32          # per-sample backward chunk size
N_SAMPLES = 2000          # sub-sample test set (per-sample backward is expensive)
SEED = 0


# ---------------------------------------------------------------------------
# Model (matches diagnostic_test.CNN / h21)
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_victim(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        running = 0.0
        n = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  epoch {ep+1}/{EPOCHS}  loss={running/n:.4f}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Attacks (same primitives as h21)
# ---------------------------------------------------------------------------
def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = (x + eps * grad.sign()).clamp(0.0, 1.0).detach()
    return x_adv


def pgd(model, x, y, eps, alpha, steps):
    x0 = x.clone().detach()
    x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
        x_adv = x_adv.clamp(0.0, 1.0).detach()
    return x_adv


def attack_flipped(model, x, y, attack_fn, batch=256):
    flips = []
    for i in range(0, x.size(0), batch):
        xb = x[i:i + batch].to(DEVICE)
        yb = y[i:i + batch].to(DEVICE)
        x_adv = attack_fn(model, xb, yb)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        flips.append((pred != yb).cpu())
    return torch.cat(flips)


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=15, batch=256):
    out = torch.zeros(x.size(0))
    for i in range(0, x.size(0), batch):
        xb = x[i:i + batch].to(DEVICE)
        yb = y[i:i + batch].to(DEVICE)
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        xc = xb.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(xc), yb)
        g = torch.autograd.grad(loss, xc)[0].sign().detach()
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            x_adv = (xb + mid.view(-1, 1, 1, 1) * g).clamp(0.0, 1.0)
            with torch.no_grad():
                flipped = model(x_adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        out[i:i + xb.size(0)] = hi.cpu()
    return out


# ---------------------------------------------------------------------------
# Sharpness features
# ---------------------------------------------------------------------------
def _params_for_grad(model):
    return [p for p in model.parameters() if p.requires_grad]


def _flat_norm(grads):
    """Aggregate L2 norm across a list of grad tensors."""
    sq = 0.0
    for g in grads:
        if g is None:
            continue
        sq = sq + g.detach().pow(2).sum()
    return sq.sqrt()


def sharpness_features(model, x, y, eps=SAM_EPS, chunk=SHARP_CHUNK):
    """
    For each sample i return:
        weight_grad_norm[i] = || grad_theta L(x_i, y_i; theta) ||_2
        sam_sharpness[i]    = L(x_i, y_i; theta + eps*sign(grad_theta L_i))
                              - L(x_i, y_i; theta)
        input_grad_norm[i]  = || grad_x L(x_i, y_i; theta) ||_2  (baseline)
        clean_loss[i]       = L(x_i, y_i; theta)

    Because PyTorch sums gradients across a batch, the SAM-style probe is
    intrinsically per-sample: we loop one sample at a time (within an outer
    chunk that's just for progress reporting). Each iteration does:
        1 forward + 1 backward on theta for grad_theta L_i (sign step direction)
        1 forward + 1 backward on x_i for input-gradient baseline
        1 forward at theta + eps*sign(grad) for the SAM loss
    """
    model.eval()
    N = x.size(0)
    w_gnorm = torch.zeros(N)
    sam_sh = torch.zeros(N)
    x_gnorm = torch.zeros(N)
    cl_loss = torch.zeros(N)

    params = _params_for_grad(model)

    # Save the original parameter snapshot once.
    orig = [p.detach().clone() for p in params]

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        if start % (chunk * 4) == 0:
            print(f"    sharpness {start}/{N}")
        for i in range(start, end):
            xi = x[i:i + 1].to(DEVICE)
            yi = y[i:i + 1].to(DEVICE)

            # ---- weight-space gradient & clean loss ----
            xi_req = xi.clone().detach().requires_grad_(True)
            logits = model(xi_req)
            loss = F.cross_entropy(logits, yi)
            # Need both grad_theta (for SAM) and grad_x (baseline). We compute
            # them separately to keep code simple.
            w_grads = torch.autograd.grad(
                loss, params, retain_graph=True, create_graph=False
            )
            wn = _flat_norm(w_grads).item()
            w_gnorm[i] = wn
            cl_loss[i] = loss.detach().item()

            # input grad (baseline) -- separate call so graph stays clean
            x_grad = torch.autograd.grad(loss, xi_req, retain_graph=False)[0]
            x_gnorm[i] = x_grad.detach().flatten().norm().item()

            # ---- SAM ascent: theta' = theta + eps * sign(grad_theta L_i) ----
            with torch.no_grad():
                for p, gp in zip(params, w_grads):
                    if gp is None:
                        continue
                    p.add_(eps * gp.sign())

                logits_adv = model(xi)
                loss_adv = F.cross_entropy(logits_adv, yi).item()

                # restore parameters
                for p, p0 in zip(params, orig):
                    p.copy_(p0)

            sam_sh[i] = loss_adv - cl_loss[i].item()

            # free
            del logits, loss, w_grads, x_grad

    # Safety: re-restore (in case of exception path).
    with torch.no_grad():
        for p, p0 in zip(params, orig):
            p.copy_(p0)

    return w_gnorm, sam_sh, x_gnorm, cl_loss


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def report_univariate(features: dict, targets: dict):
    print("\n=== Univariate AUROC / Spearman (higher score -> more vulnerable) ===")
    header = f"{'feature':>22s} | " + " | ".join(f"{t:>18s}" for t in targets)
    print(header)
    print("-" * len(header))
    for fname, fvals in features.items():
        row = [fname]
        for tname, tvals in targets.items():
            if tvals.dtype == bool or set(np.unique(tvals)).issubset({0, 1}):
                a = auroc(fvals, tvals.astype(int))
                row.append(f"AUROC={a:.3f}")
            else:
                rho, _ = spearmanr(fvals, tvals)
                row.append(f"rho={rho:+.3f}")
        print(f"{row[0]:>22s} | " + " | ".join(f"{c:>18s}" for c in row[1:]))


def multivariate_ablation(features: dict, target: np.ndarray, name: str):
    """Does weight-gradient-norm / SAM-sharpness add over input-grad-norm + margin?"""
    print(f"\n=== Multivariate ablation on target: {name} ===")
    base_keys = ["input_grad_norm", "victim_margin"]

    def fit_auc(keys):
        X = np.stack([features[k] for k in keys], axis=1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X, target)
        return auroc(clf.predict_proba(X)[:, 1], target), dict(zip(keys, clf.coef_[0].tolist()))

    a_base, c_base = fit_auc(base_keys)
    print(f"  baseline (input_grad_norm + margin): AUROC={a_base:.3f} coefs={c_base}")

    for add_key in ["weight_grad_norm", "sam_sharpness"]:
        a, c = fit_auc(base_keys + [add_key])
        print(f"  + {add_key:<18s}             : AUROC={a:.3f} delta={a - a_base:+.4f} coefs={c}")

    a_all, c_all = fit_auc(base_keys + ["weight_grad_norm", "sam_sharpness"])
    print(f"  + both (wgn + sam)                : AUROC={a_all:.3f} delta={a_all - a_base:+.4f} coefs={c_all}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print("Training victim CNN on Fashion-MNIST...")
    model = train_victim(train_set)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))])

    # Clean accuracy + restrict to correctly classified samples.
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i + 512].to(DEVICE)).argmax(1).cpu())
        preds = torch.cat(preds)
    correct_mask = (preds == test_y)
    print(f"Clean accuracy: {correct_mask.float().mean().item():.4f}")

    idx_correct = torch.where(correct_mask)[0]
    g = torch.Generator().manual_seed(SEED)
    perm = idx_correct[torch.randperm(idx_correct.numel(), generator=g)]
    sub_idx = perm[:N_SAMPLES]
    x_sub = test_x[sub_idx]
    y_sub = test_y[sub_idx]
    print(f"Sharpness sub-sample: {x_sub.size(0)} correctly-classified examples "
          f"(chunk={SHARP_CHUNK}, SAM eps={SAM_EPS}).")

    # ---- sharpness features ----
    print("Computing per-sample sharpness features (this is the expensive step)...")
    w_gnorm, sam_sh, x_gnorm, cl_loss = sharpness_features(
        model, x_sub, y_sub, eps=SAM_EPS, chunk=SHARP_CHUNK
    )

    # victim margin on clean inputs
    with torch.no_grad():
        margins = []
        for i in range(0, x_sub.size(0), 512):
            lg = model(x_sub[i:i + 512].to(DEVICE))
            srt, _ = lg.sort(dim=1, descending=True)
            margins.append((srt[:, 0] - srt[:, 1]).cpu())
        margin = torch.cat(margins)

    mean_pix = x_sub.flatten(1).mean(1)
    std_pix = x_sub.flatten(1).std(1)

    # ---- targets ----
    print("Running FGSM attack...")
    flipped_fgsm = attack_flipped(
        model, x_sub, y_sub, lambda m, x, y: fgsm(m, x, y, EPS_TEST)
    )
    print("Running PGD attack...")
    flipped_pgd = attack_flipped(
        model, x_sub, y_sub,
        lambda m, x, y: pgd(m, x, y, EPS_TEST, PGD_ALPHA, PGD_STEPS),
    )
    print("Binary-searching FGSM min-eps...")
    fgsm_meps = fgsm_min_eps(model, x_sub, y_sub)

    features = {
        "weight_grad_norm": w_gnorm.numpy(),
        "sam_sharpness":    sam_sh.numpy(),
        "input_grad_norm":  x_gnorm.numpy(),
        "clean_loss":       cl_loss.numpy(),
        "victim_margin":    margin.numpy(),
        "neg_margin":       (-margin).numpy(),  # convenience: higher -> more vulnerable
        "mean_pix":         mean_pix.numpy(),
        "std_pix":          std_pix.numpy(),
    }
    targets_binary = {
        "flipped_FGSM": flipped_fgsm.numpy().astype(int),
        "flipped_PGD":  flipped_pgd.numpy().astype(int),
    }
    # For FGSM_min_eps: lower eps -> more vulnerable, so for AUROC sense we
    # report Spearman (sign tells direction). We also build a binary target
    # at the median for an AUROC view.
    median_eps = float(np.median(fgsm_meps.numpy()))
    targets_binary["flipped_under_median_eps"] = (fgsm_meps.numpy() <= median_eps).astype(int)
    targets_cont = {"FGSM_min_eps": fgsm_meps.numpy()}

    print(f"\nAttack success rates: FGSM={flipped_fgsm.float().mean():.3f}  "
          f"PGD={flipped_pgd.float().mean():.3f}")
    print(f"FGSM min-eps  mean={fgsm_meps.mean():.4f}  median={fgsm_meps.median():.4f}")
    print(f"weight_grad_norm  mean={w_gnorm.mean():.4f}  median={w_gnorm.median():.4f}")
    print(f"sam_sharpness     mean={sam_sh.mean():.4f}  median={sam_sh.median():.4f}")

    report_univariate(features, {**targets_binary, **targets_cont})

    for tname, tvals in targets_binary.items():
        multivariate_ablation(features, tvals, tname)

    # Save raw arrays
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h37_sharpness.npz"
    np.savez(
        out_path,
        **features,
        **{f"y_{k}": v for k, v in targets_binary.items()},
        **{f"y_{k}": v for k, v in targets_cont.items()},
        sub_idx=sub_idx.numpy(),
    )
    print(f"\nSaved raw arrays -> {out_path}")

    print("\n=== Headline (H37) ===")
    print("If per-sample WEIGHT-space sharpness predicts adversarial vulnerability, expect:")
    print("  - AUROC > 0.5 for weight_grad_norm / sam_sharpness vs flipped_{FGSM,PGD}")
    print("  - negative Spearman vs FGSM_min_eps (sharper sample <=> smaller min-eps)")
    print("  - positive multivariate delta over [input_grad_norm + victim_margin]")
    print("Caveat: weight-grad-norm is closely related to confidence; if it adds nothing")
    print("        over input_grad_norm + margin, the SAM-style probe is redundant.")


if __name__ == "__main__":
    main()
