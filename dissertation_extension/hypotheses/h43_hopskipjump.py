"""
Hypothesis H43: HopSkipJump attack (Chen et al. 2020, arXiv:1904.02144) is a
state-of-the-art decision-based black-box attack. Its per-sample success /
perturbation magnitude gives a *different* vulnerability label than white-box
gradient attacks (e.g. FGSM).

HopSkipJump uses only model *decisions* (top-1 label) — no gradients, no scores.
It is the strongest decision-based attack and is what a realistic adversary
with API-only access would use. Predicting vulnerability to HSJ is *not* the
same task as predicting vulnerability to FGSM.

Pipeline:
  1. Train a small CNN (matching diagnostic_test.py) on Fashion-MNIST for 10
     epochs.
  2. Subsample 200 correctly-classified test points.
  3. Run HopSkipJump (L_inf constraint, ~50 iterations) per sample. Use foolbox
     if importable; otherwise fall back to a self-contained PyTorch port of
     Algorithm 1 of Chen et al. (2020).
  4. Compute simple per-sample image features:
        victim_margin, mean_pix, std_pix, sobel_mean, jpeg_q75
  5. Targets:
        HSJ_perturbation_L_inf  (continuous)
        HSJ_flipped_at_budget   (binary: did the attack flip within budget?)
        flipped_FGSM            (white-box reference, eps=15/255)
  6. Univariate AUROC of each feature vs each binary target,
     Spearman rho of HSJ_perturbation vs FGSM_min_eps.

This script is self-contained (does not import diagnostic_test.py) but uses an
identical CNN architecture. It is intended to be RUN OFFLINE — it WRITES
nothing other than what `print()` produces. Do not execute as part of CI.
"""
from __future__ import annotations
import io
import math
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from scipy.stats import spearmanr
from scipy.ndimage import sobel
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
N_SUBSAMPLE = 200
HSJ_ITERS = 50
HSJ_INIT_TRIALS = 100         # # random init samples to find an adversarial start
HSJ_INIT_BIN_SEARCH = 10      # binary-search steps to project init onto boundary
HSJ_MAX_GRAD_QUERIES = 200    # max queries to estimate gradient direction
HSJ_GAMMA = 1.0               # step-size schedule constant (Chen et al.)
HSJ_BUDGET_LINF = 15.0 / 255  # "flipped at budget" decision threshold
FGSM_EPS = 15.0 / 255
FGSM_BS_ITERS = 15            # binary-search iters for FGSM min-eps
SEED = 0


# --------------------------------------------------------------------------- #
# Model — same architecture as diagnostic_test.py
# --------------------------------------------------------------------------- #
class CNN(nn.Module):
    def __init__(self, n: int = 10):
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


def train_victim(train_set, test_set) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done")
    model.eval()
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    return model, test_x, test_y


# --------------------------------------------------------------------------- #
# Decision-only oracle wrapper
# --------------------------------------------------------------------------- #
def decision(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Top-1 label only (decision-based oracle)."""
    with torch.no_grad():
        return model(x).argmax(1)


# --------------------------------------------------------------------------- #
# HopSkipJump (L_inf), self-contained.
# Implements Algorithm 1 from Chen, Jordan & Wainwright (arXiv:1904.02144 v4).
# --------------------------------------------------------------------------- #
def hsj_is_adv(model, x_adv, y_true) -> bool:
    return decision(model, x_adv.unsqueeze(0)).item() != y_true


def hsj_find_init(model, x_orig: torch.Tensor, y_true: int) -> torch.Tensor | None:
    """Random uniform init that is already misclassified."""
    for _ in range(HSJ_INIT_TRIALS):
        cand = torch.rand_like(x_orig)
        if hsj_is_adv(model, cand, y_true):
            return cand
    return None


def hsj_project_linf(x_orig, x_adv, alpha):
    """Project x_adv onto x_orig + L_inf ball of radius alpha (used to move
    toward the boundary)."""
    delta = (x_adv - x_orig).clamp(-alpha, alpha)
    return (x_orig + delta).clamp(0.0, 1.0)


def hsj_binary_search_boundary(model, x_orig, x_adv, y_true,
                               steps=HSJ_INIT_BIN_SEARCH) -> torch.Tensor:
    """L_inf binary search: find smallest alpha in [0,1] such that
    project_linf(x_orig, x_adv, alpha * max|x_adv - x_orig|) is still adv."""
    # In L_inf HSJ, threshold by the current L_inf distance.
    high_thresh = (x_adv - x_orig).abs().max().item()
    if high_thresh == 0.0:
        return x_adv
    lo, hi = 0.0, high_thresh
    for _ in range(steps):
        mid = (lo + hi) / 2
        cand = hsj_project_linf(x_orig, x_adv, mid)
        if hsj_is_adv(model, cand, y_true):
            hi = mid
        else:
            lo = mid
    return hsj_project_linf(x_orig, x_adv, hi)


def hsj_estimate_grad(model, x_bdy, y_true, delta, num_queries) -> torch.Tensor:
    """Monte-Carlo gradient-direction estimate at the boundary point x_bdy.

    Eq. (15) in Chen et al.: average of sign(phi(x + delta * u_b)) * u_b for
    i.i.d. Rademacher / Gaussian u_b. We use Gaussian, then sign for L_inf.
    """
    shape = (num_queries,) + tuple(x_bdy.shape)
    u = torch.randn(shape, device=x_bdy.device)
    # normalise per-sample so each direction is unit-norm
    u = u / (u.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
    perturbed = (x_bdy.unsqueeze(0) + delta * u).clamp(0.0, 1.0)
    # decision-based phi: +1 if adversarial, -1 otherwise
    with torch.no_grad():
        labels = model(perturbed).argmax(1)
    phi = torch.where(labels != y_true,
                      torch.ones_like(labels, dtype=torch.float),
                      -torch.ones_like(labels, dtype=torch.float))
    # baseline subtraction (recommended by paper for variance reduction)
    phi = phi - phi.mean()
    grad = (phi.view(-1, 1, 1, 1) * u).mean(0)
    # For L_inf: direction is sign of estimated gradient.
    return grad.sign()


def hopskipjump_linf(model, x_orig: torch.Tensor, y_true: int,
                      max_iters: int = HSJ_ITERS) -> tuple[torch.Tensor | None, float]:
    """Run HopSkipJump under L_inf. Returns (adversarial_example, L_inf_dist).

    If initialisation fails returns (None, inf).
    """
    x_orig = x_orig.to(DEVICE)
    x_adv = hsj_find_init(model, x_orig, y_true)
    if x_adv is None:
        return None, float("inf")
    # Project onto boundary.
    x_bdy = hsj_binary_search_boundary(model, x_orig, x_adv, y_true)
    for t in range(1, max_iters + 1):
        d = int(np.prod(x_orig.shape))
        # Step-size schedules from the paper.
        cur_linf = (x_bdy - x_orig).abs().max().item()
        delta_t = HSJ_GAMMA * cur_linf / (d ** 1.0)  # for L_inf
        # # of queries grows as sqrt(t).
        num_q = min(int(100 * math.sqrt(t)), HSJ_MAX_GRAD_QUERIES)
        grad_dir = hsj_estimate_grad(model, x_bdy, y_true, delta_t, num_q)
        # Geometric step-size search starting from cur_linf / sqrt(t).
        epsilon_t = cur_linf / math.sqrt(t)
        for _ in range(20):
            cand = (x_bdy + epsilon_t * grad_dir).clamp(0.0, 1.0)
            if hsj_is_adv(model, cand, y_true):
                break
            epsilon_t /= 2
        else:
            cand = x_bdy  # could not advance; keep current boundary point
        # Re-project onto boundary along the L_inf line.
        x_bdy = hsj_binary_search_boundary(model, x_orig, cand, y_true)
    return x_bdy, (x_bdy - x_orig).abs().max().item()


# --------------------------------------------------------------------------- #
# Optional foolbox path
# --------------------------------------------------------------------------- #
def hopskipjump_via_foolbox(model, x_orig, y_true):
    import foolbox as fb
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=DEVICE)
    attack = fb.attacks.HopSkipJumpAttack(steps=HSJ_ITERS, constraint="linf",
                                          initial_num_evals=100,
                                          max_num_evals=HSJ_MAX_GRAD_QUERIES)
    x_b = x_orig.unsqueeze(0)
    y_b = torch.tensor([y_true], device=DEVICE)
    raw, clipped, success = attack(fmodel, x_b, y_b, epsilons=None)
    adv = raw[0]
    return adv, (adv - x_orig).abs().max().item()


# --------------------------------------------------------------------------- #
# FGSM reference (white-box)
# --------------------------------------------------------------------------- #
def fgsm_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=FGSM_EPS):
    sign = fgsm_sign(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=FGSM_BS_ITERS):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_sign(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# --------------------------------------------------------------------------- #
# Per-sample features
# --------------------------------------------------------------------------- #
def compute_features(model, x: torch.Tensor, y: torch.Tensor) -> dict[str, np.ndarray]:
    with torch.no_grad():
        logits = []
        for i in range(0, x.size(0), 256):
            logits.append(model(x[i:i+256]))
        logits = torch.cat(logits, 0)
    sorted_l, _ = logits.sort(1, descending=True)
    margin = (sorted_l[:, 0] - sorted_l[:, 1]).cpu().numpy()

    x_np = x.squeeze(1).cpu().numpy()
    mean_pix = x_np.mean(axis=(1, 2))
    std_pix = x_np.std(axis=(1, 2))

    sobel_mean = np.zeros(x_np.shape[0])
    for i in range(x_np.shape[0]):
        gx = sobel(x_np[i], axis=0)
        gy = sobel(x_np[i], axis=1)
        sobel_mean[i] = np.hypot(gx, gy).mean()

    jpeg_q75 = np.zeros(x_np.shape[0])
    for i in range(x_np.shape[0]):
        im = Image.fromarray((x_np[i] * 255).astype(np.uint8), mode="L")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=75)
        jpeg_q75[i] = len(buf.getvalue())

    return {
        "victim_margin": margin,
        "mean_pix": mean_pix,
        "std_pix": std_pix,
        "sobel_mean": sobel_mean,
        "jpeg_q75": jpeg_q75,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print(f"Device: {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print("Training victim CNN ...")
    t0 = time.time()
    model, test_x, test_y = train_victim(train_set, test_set)
    print(f"  trained in {time.time()-t0:.1f}s")

    # Restrict to correctly classified samples.
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds, 0)
    correct_idx = (preds == test_y).nonzero(as_tuple=True)[0]
    print(f"Victim test accuracy: {correct_idx.numel() / test_y.numel():.4f}")

    rng = np.random.default_rng(SEED)
    sel = rng.choice(correct_idx.cpu().numpy(), size=N_SUBSAMPLE, replace=False)
    sel_t = torch.as_tensor(sel, device=DEVICE)
    xs = test_x[sel_t]
    ys = test_y[sel_t]

    # Try foolbox.
    use_foolbox = False
    try:
        import foolbox  # noqa: F401
        use_foolbox = True
        print("foolbox available — using fb.HopSkipJumpAttack")
    except Exception:
        print("foolbox not available — using local PyTorch HSJ implementation")

    # ---- HopSkipJump per sample ----
    hsj_dist = np.zeros(N_SUBSAMPLE)
    hsj_flipped = np.zeros(N_SUBSAMPLE, dtype=np.int64)
    print(f"Running HopSkipJump on {N_SUBSAMPLE} samples ({HSJ_ITERS} iters each)...")
    t0 = time.time()
    for i in range(N_SUBSAMPLE):
        x_i = xs[i]
        y_i = int(ys[i].item())
        try:
            if use_foolbox:
                adv, dist = hopskipjump_via_foolbox(model, x_i, y_i)
            else:
                adv, dist = hopskipjump_linf(model, x_i, y_i)
        except Exception as e:
            print(f"  sample {i}: HSJ failed ({e!r}); recording inf")
            adv, dist = None, float("inf")
        hsj_dist[i] = dist
        # "Flipped at budget": adv exists with L_inf <= budget AND model
        # actually mispredicts on it.
        if adv is not None and dist <= HSJ_BUDGET_LINF:
            with torch.no_grad():
                pred = model(adv.unsqueeze(0)).argmax(1).item()
            hsj_flipped[i] = int(pred != y_i)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{N_SUBSAMPLE}  mean_dist={hsj_dist[:i+1].mean():.4f}  "
                  f"flipped@budget={hsj_flipped[:i+1].mean():.3f}  "
                  f"({time.time()-t0:.1f}s)")
    print(f"HSJ done in {time.time()-t0:.1f}s")

    # ---- FGSM reference (white-box) ----
    print("Computing FGSM reference ...")
    fgsm_flipped = fgsm_flip(model, xs, ys, eps=FGSM_EPS).cpu().numpy().astype(int)
    fgsm_min = fgsm_min_eps(model, xs, ys).cpu().numpy()

    # ---- Features ----
    feats = compute_features(model, xs, ys)
    feat_names = ["victim_margin", "mean_pix", "std_pix", "sobel_mean", "jpeg_q75"]

    # ---- Univariate AUROC ----
    print("\n=========== Univariate AUROC ===========")
    print(f"FGSM positive rate: {fgsm_flipped.mean():.3f}")
    print(f"HSJ-flipped@budget positive rate: {hsj_flipped.mean():.3f}")
    targets = {
        "HSJ_flipped_at_budget": hsj_flipped,
        "FGSM_flipped":          fgsm_flipped,
    }
    for tname, tvec in targets.items():
        if tvec.std() == 0:
            print(f"\n target {tname}: degenerate (rate={tvec.mean():.3f})")
            continue
        print(f"\n target = {tname}")
        for fn in feat_names:
            a = roc_auc_score(tvec, feats[fn])
            a = max(a, 1 - a)
            print(f"   {fn:<16} AUROC = {a:.4f}")

    # Continuous target: HSJ_perturbation_L_inf  (Spearman with features)
    print("\n target = HSJ_perturbation_L_inf  (Spearman rho)")
    finite = np.isfinite(hsj_dist)
    for fn in feat_names:
        rho, p = spearmanr(feats[fn][finite], hsj_dist[finite])
        print(f"   {fn:<16} rho = {rho:+.4f}  (p = {p:.3g})")

    # ---- Cross-attack agreement ----
    print("\n=========== Cross-attack agreement ===========")
    rho_xa, p_xa = spearmanr(hsj_dist[finite], fgsm_min[finite])
    print(f"  Spearman(HSJ_L_inf, FGSM_min_eps) = {rho_xa:+.4f}  (p={p_xa:.3g})")
    # Binary-label agreement (FGSM_flipped vs HSJ_flipped@budget).
    agree = (fgsm_flipped == hsj_flipped).mean()
    print(f"  Binary agreement HSJ@budget vs FGSM: {agree:.3f}")
    # AUROC of FGSM-flip as a predictor of HSJ-flip and vice versa.
    if hsj_flipped.std() > 0 and fgsm_flipped.std() > 0:
        print(f"  AUROC(FGSM_flipped -> HSJ_flipped):  "
              f"{max(roc_auc_score(hsj_flipped, fgsm_flipped), 1 - roc_auc_score(hsj_flipped, fgsm_flipped)):.4f}")
        print(f"  AUROC(HSJ_flipped -> FGSM_flipped):  "
              f"{max(roc_auc_score(fgsm_flipped, hsj_flipped), 1 - roc_auc_score(fgsm_flipped, hsj_flipped)):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
