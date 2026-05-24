"""
H16: Black-box Boundary Attack (Brendel et al. 2018) vulnerability vs white-box FGSM.

Hypothesis
----------
Boundary Attack is a *decision-based* black-box attack: it only uses the model's
argmax label, no gradients, no probabilities. Its vulnerability label (final L2
perturbation after a budget of steps) may have *different* model-free correlates
than white-box FGSM/PGD vulnerability. We test whether the same model-free
features (mean_pix, std_pix, sobel_mean, jpeg_q75) and one white-box feature
(victim_margin) predict it equivalently well, by comparing per-feature AUROC
between Boundary-Attack targets and FGSM-success on the same samples.

Pipeline
--------
1.  Train a small CNN victim on Fashion-MNIST (10 epochs, Adam).
2.  Sample 200 test points the victim classifies correctly.
3.  Run foolbox BoundaryAttack against the victim for ~50 steps each, starting
    from a random image of a different class (decision-based init). Record
    final L2 perturbation. If foolbox is unavailable, fall back to a built-in
    minimal decision-based attack (see `_fallback_boundary_attack`).
4.  Compute per-sample features:
        victim_margin, mean_pix, std_pix, sobel_mean, jpeg_q75
5.  Targets:
        - boundary_L2  (continuous; lower = more vulnerable)
        - boundary_flipped_at_5 (binary: final L2 < 5.0)
        - FGSM_self_flip       (binary white-box reference at eps=15/255)
6.  Report Spearman correlation (continuous) and AUROC (binary) per feature.

Run
---
    .venv/bin/python dissertation_extension/hypotheses/h16_boundary_attack.py

Caveats
-------
- 200 samples * ~50 boundary steps is a *small* budget; reported L2 values are
  upper bounds on the true minimal L2 — many runs will not converge.
- Foolbox's BoundaryAttack does internal step-size adaptation; the "steps"
  argument is the total budget. We set steps=50 explicitly to match the spec.
- Decision-based attacks have stochastic init (random starting image), so
  results vary across seeds; we fix torch & numpy seeds for reproducibility.
- jpeg_q75 requires PIL; sobel uses a manual 3x3 conv (no scipy/cv2 needed).
"""
import io
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
N_SAMPLES = 200
BOUNDARY_STEPS = 50
FGSM_EPS = 15.0 / 255.0
BOUNDARY_L2_THRESH = 5.0   # binary: "vulnerable" if final L2 < 5.0
SEED = 0


# -----------------------------------------------------------------------------
# Victim
# -----------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2); x = self.do1(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.do2(x); return self.fc2(x)


def train_victim(train_set):
    torch.manual_seed(SEED); np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        print(f"  epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------
SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = SOBEL_X.transpose(2, 3)


def sobel_mean(x):
    # x: (N, 1, 28, 28) on DEVICE
    kx = SOBEL_X.to(x.device); ky = SOBEL_Y.to(x.device)
    gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2).flatten(1).mean(1)


def jpeg_q75_size(x_np_uint8):
    """Size in bytes of JPEG-Q75 encoding of a HxW uint8 image (proxy for complexity)."""
    buf = io.BytesIO()
    Image.fromarray(x_np_uint8, mode="L").save(buf, format="JPEG", quality=75)
    return len(buf.getvalue())


def compute_features(model, x, y):
    """x: (N,1,28,28) in [0,1] on DEVICE. y: (N,) labels."""
    N = x.size(0)
    with torch.no_grad():
        logits = model(x)
    sorted_l, _ = logits.sort(1, descending=True)
    victim_margin = (sorted_l[:, 0] - sorted_l[:, 1]).cpu().numpy()

    flat = x.flatten(1)
    mean_pix = flat.mean(1).cpu().numpy()
    std_pix = flat.std(1).cpu().numpy()
    sm = sobel_mean(x).cpu().numpy()

    jpeg = np.zeros(N)
    x_uint8 = (x.squeeze(1).cpu().numpy() * 255.0).astype(np.uint8)
    for i in range(N):
        jpeg[i] = jpeg_q75_size(x_uint8[i])

    feats = np.stack([victim_margin, mean_pix, std_pix, sm, jpeg], axis=1)
    names = ["victim_margin", "mean_pix", "std_pix", "sobel_mean", "jpeg_q75"]
    return feats, names


# -----------------------------------------------------------------------------
# Boundary Attack (foolbox if available, otherwise fallback)
# -----------------------------------------------------------------------------
def _fallback_boundary_attack(model, x, y, x_pool, y_pool, steps=BOUNDARY_STEPS):
    """
    Minimal decision-based attack: for each sample, find a starting image from
    `x_pool` that is classified differently from y, then do a binary-search-like
    walk toward x while staying misclassified.

    Returns final L2 perturbation per sample (tensor of length N on DEVICE).
    """
    N = x.size(0)
    out_l2 = torch.zeros(N, device=DEVICE)
    pool_preds = []
    with torch.no_grad():
        for i in range(0, x_pool.size(0), 512):
            pool_preds.append(model(x_pool[i:i + 512]).argmax(1))
    pool_preds = torch.cat(pool_preds)

    rng = np.random.default_rng(SEED)
    sigma = 0.05
    for i in range(N):
        target_label = int(y[i].item())
        # pick a starting image classified into a different class
        candidates = (pool_preds != target_label).nonzero(as_tuple=True)[0]
        if candidates.numel() == 0:
            out_l2[i] = float("nan"); continue
        start_idx = int(candidates[rng.integers(candidates.numel())].item())
        adv = x_pool[start_idx].clone()
        src = x[i]
        # binary search toward src on the line, finding boundary
        lo, hi = 0.0, 1.0
        for _ in range(10):
            mid = (lo + hi) / 2
            cand = (1 - mid) * adv + mid * src
            with torch.no_grad():
                p = model(cand.unsqueeze(0)).argmax(1).item()
            if p != target_label:
                lo = mid
            else:
                hi = mid
        adv = (1 - lo) * adv + lo * src
        # random walk
        for _ in range(steps):
            noise = torch.randn_like(adv) * sigma
            cand = (adv + noise).clamp(0, 1)
            # bias slightly toward src
            cand = 0.97 * cand + 0.03 * src
            cand = cand.clamp(0, 1)
            with torch.no_grad():
                p = model(cand.unsqueeze(0)).argmax(1).item()
            if p != target_label:
                # accept if closer to src
                if torch.norm(cand - src) < torch.norm(adv - src):
                    adv = cand
        out_l2[i] = torch.norm(adv - src)
    return out_l2


def run_boundary_attack(model, x, y, x_pool, y_pool):
    """Try foolbox; fall back to handwritten attack on failure."""
    try:
        import foolbox as fb
    except ImportError:
        print("  foolbox not installed; using fallback decision-based attack")
        return _fallback_boundary_attack(model, x, y, x_pool, y_pool)

    print(f"  using foolbox {fb.__version__}")
    fmodel = fb.PyTorchModel(model, bounds=(0.0, 1.0), device=DEVICE)
    attack = fb.attacks.BoundaryAttack(steps=BOUNDARY_STEPS)

    # Build starting points: for each sample, an image of a different class that
    # the model classifies into a class != y[i].
    with torch.no_grad():
        pool_preds = []
        for i in range(0, x_pool.size(0), 512):
            pool_preds.append(model(x_pool[i:i + 512]).argmax(1))
        pool_preds = torch.cat(pool_preds)

    N = x.size(0)
    starts = torch.zeros_like(x)
    rng = np.random.default_rng(SEED)
    for i in range(N):
        cand = (pool_preds != y[i]).nonzero(as_tuple=True)[0]
        if cand.numel() == 0:
            # fallback: random noise
            starts[i] = torch.rand_like(x[i])
        else:
            starts[i] = x_pool[int(cand[rng.integers(cand.numel())].item())]

    # foolbox API: attack(fmodel, inputs, criterion, starting_points=..., epsilons=None)
    criterion = fb.criteria.Misclassification(y)
    try:
        _, advs, _ = attack(fmodel, x, criterion,
                            starting_points=starts, epsilons=None)
    except TypeError:
        # older foolbox: attack.run / different signature
        advs = attack.run(fmodel, x, criterion, starting_points=starts)

    diff = (advs - x).flatten(1)
    return diff.norm(dim=1)


# -----------------------------------------------------------------------------
# FGSM
# -----------------------------------------------------------------------------
def fgsm_flip(model, x, y, eps=FGSM_EPS):
    x_ = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_), y).backward()
    adv = (x_ + eps * x_.grad.sign()).clamp(0, 1).detach()
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print(f"device={DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print("training victim...")
    model = train_victim(train_set)

    # load entire test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i + 512]).argmax(1))
        preds = torch.cat(preds)
    correct_idx = (preds == test_y).nonzero(as_tuple=True)[0]
    print(f"victim test acc = {correct_idx.numel() / test_y.numel():.4f}")

    # sample N_SAMPLES from correctly classified
    g = torch.Generator().manual_seed(SEED)
    perm = correct_idx[torch.randperm(correct_idx.numel(), generator=g)[:N_SAMPLES]]
    x_eval, y_eval = test_x[perm], test_y[perm]

    # pool = rest of test set (for starting points)
    print(f"running boundary attack on {N_SAMPLES} samples, {BOUNDARY_STEPS} steps each...")
    t0 = time.time()
    boundary_l2 = run_boundary_attack(model, x_eval, y_eval, test_x, test_y)
    print(f"  done in {time.time()-t0:.1f}s")
    bl2_np = boundary_l2.detach().cpu().numpy()
    print(f"  boundary L2: mean={np.nanmean(bl2_np):.3f}  median={np.nanmedian(bl2_np):.3f}")

    boundary_flip = (boundary_l2 < BOUNDARY_L2_THRESH).cpu().numpy().astype(int)
    print(f"  boundary_flipped_at_{BOUNDARY_L2_THRESH}: pos rate = {boundary_flip.mean():.3f}")

    print("running FGSM (white-box reference)...")
    fgsm = fgsm_flip(model, x_eval, y_eval).cpu().numpy().astype(int)
    print(f"  FGSM_self_flip pos rate = {fgsm.mean():.3f}")

    print("computing features...")
    feats, names = compute_features(model, x_eval, y_eval)

    # ---- report ----
    print("\n=========================================================")
    print(" Per-feature association with each vulnerability target")
    print("=========================================================")
    targets_bin = {
        "boundary_flipped_at_5.0": boundary_flip,
        "FGSM_self_flip":          fgsm,
    }
    print(f"\n  AUROC (binary targets; reported as max(auc, 1-auc))")
    header = f"  {'feature':<16}" + "".join(f"{t:>26}" for t in targets_bin)
    print(header)
    for j, n in enumerate(names):
        row = f"  {n:<16}"
        for t_name, y_bin in targets_bin.items():
            if y_bin.std() == 0:
                row += f"{'N/A':>26}"; continue
            try:
                a = roc_auc_score(y_bin, feats[:, j])
                a = max(a, 1 - a)
                row += f"{a:>26.4f}"
            except Exception as e:
                row += f"{'err':>26}"
        print(row)

    # continuous target: spearman vs boundary_L2
    print(f"\n  Spearman correlation with continuous boundary_L2")
    valid = ~np.isnan(bl2_np)
    for j, n in enumerate(names):
        rho, p = spearmanr(feats[valid, j], bl2_np[valid])
        print(f"    {n:<16}  rho={rho:+.4f}   p={p:.3g}")

    # cross-target agreement
    print(f"\n  Agreement between vulnerability labels:")
    print(f"    pos rate boundary={boundary_flip.mean():.3f}  pos rate FGSM={fgsm.mean():.3f}")
    if boundary_flip.std() and fgsm.std():
        print(f"    Pearson(boundary_flip, FGSM_flip) = "
              f"{np.corrcoef(boundary_flip, fgsm)[0,1]:+.4f}")
        # AUROC of FGSM-flip predicting boundary-flip (i.e. do the same samples
        # rank as vulnerable?)
        try:
            a = roc_auc_score(boundary_flip, fgsm)
            print(f"    AUROC(FGSM_flip -> boundary_flip)  = {max(a,1-a):.4f}")
        except Exception:
            pass

    print("\nDONE.")


if __name__ == "__main__":
    main()
