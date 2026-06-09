"""
H39: Square Attack (Andriushchenko et al. ECCV 2020) — gradient-free black-box.

Hypothesis
----------
Square Attack is a *score-based* black-box attack that uses random search with
square-shaped L_inf perturbations and queries the victim's loss (no gradients).
It is immune to gradient masking and is widely regarded as the strongest sanity-
check black-box attack. We hypothesise that vulnerability to Square has
*different* correlates than vulnerability to FGSM (white-box, gradient-based).
In particular: if any kind of gradient masking is present, some samples will be
robust to FGSM but flippable by Square. We further check whether the standard
features (victim_margin, mean_pix, std_pix, sobel_mean, jpeg_q75) predict Square
success differently than FGSM success.

Pipeline
--------
1. Train a small CNN victim on Fashion-MNIST (10 epochs, Adam) — same arch as
   diagnostic_test.py / h16.
2. Sample N test points the victim classifies correctly.
3. Run Square Attack (algorithm 1 of arXiv:1912.00049) at eps=15/255, ~500
   queries per sample. Try torchattacks.SquareAttack first; otherwise use the
   built-in implementation.
4. Compute features: victim_margin, mean_pix, std_pix, sobel_mean, jpeg_q75.
5. Targets:
       flipped_Square (binary)
       flipped_FGSM   (binary)
       queries_to_flip (continuous; N_QUERIES+1 if never flipped)
6. Univariate AUROC per feature per binary target. Spearman vs queries_to_flip.
   Cross-tabulate flipped_FGSM vs flipped_Square: samples robust to FGSM but
   flipped by Square would suggest gradient masking.

Run
---
    .venv/bin/python dissertation_extension/hypotheses/h39_square_attack.py
"""
import io
import sys
import time
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
EPS = 15.0 / 255.0
N_QUERIES = 500
P_INIT = 0.05            # initial fraction of image area used as square side^2/area
SEED = 0


# -----------------------------------------------------------------------------
# Victim (same as h16 / diagnostic_test.py)
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
        model.train(); t0 = time.time()
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
    kx = SOBEL_X.to(x.device); ky = SOBEL_Y.to(x.device)
    gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2).flatten(1).mean(1)


def jpeg_q75_size(x_np_uint8):
    buf = io.BytesIO()
    Image.fromarray(x_np_uint8, mode="L").save(buf, format="JPEG", quality=75)
    return len(buf.getvalue())


def compute_features(model, x, y):
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
# Square Attack (arXiv:1912.00049, Alg. 1 — L_inf version)
# -----------------------------------------------------------------------------
def _p_selection(p_init, it, n_iters):
    """Piecewise-constant schedule for square-area fraction (from the paper)."""
    it_norm = int(it / n_iters * 10000)
    if   10 < it_norm <= 50:    p = p_init / 2
    elif 50 < it_norm <= 200:   p = p_init / 4
    elif 200 < it_norm <= 500:  p = p_init / 8
    elif 500 < it_norm <= 1000: p = p_init / 16
    elif 1000 < it_norm <= 2000:p = p_init / 32
    elif 2000 < it_norm <= 4000:p = p_init / 64
    elif 4000 < it_norm <= 6000:p = p_init / 128
    elif 6000 < it_norm <= 8000:p = p_init / 256
    elif 8000 < it_norm:        p = p_init / 512
    else:                        p = p_init
    return p


def _margin_loss(logits, y):
    """Untargeted margin loss: z_y - max_{j!=y} z_j. Lower = more adversarial."""
    N = logits.size(0)
    z_y = logits[torch.arange(N), y]
    logits_other = logits.clone()
    logits_other[torch.arange(N), y] = -1e9
    z_other = logits_other.max(dim=1).values
    return z_y - z_other


def square_attack_linf(model, x, y, eps=EPS, n_iters=N_QUERIES, p_init=P_INIT,
                       seed=SEED):
    """
    Vectorised L_inf Square Attack. Returns (adv, queries_to_flip, success).
      queries_to_flip[i] = iteration at which sample i first changed argmax;
                          n_iters+1 if never flipped.
    """
    rng = np.random.default_rng(seed)
    N, C, H, W = x.shape
    assert C == 1, "this implementation assumes single-channel inputs"
    # init: vertical stripes of +/- eps (paper's L_inf init)
    init_delta = torch.from_numpy(
        rng.choice([-eps, eps], size=(N, C, 1, W)).astype(np.float32)
    ).to(x.device).repeat(1, 1, H, 1)
    x_best = (x + init_delta).clamp(0.0, 1.0)

    with torch.no_grad():
        logits = model(x_best)
        loss_best = _margin_loss(logits, y)
        pred = logits.argmax(1)
    flipped_at = torch.full((N,), n_iters + 1, dtype=torch.long, device=x.device)
    flipped_at[pred != y] = 0

    active = (pred == y).clone()  # only keep attacking still-correct samples

    for it in range(n_iters):
        if not active.any():
            break
        idx = active.nonzero(as_tuple=True)[0]
        x_curr = x[idx]; x_best_curr = x_best[idx]; y_curr = y[idx]
        loss_curr = loss_best[idx]
        n_act = idx.numel()

        p = _p_selection(p_init, it, n_iters)
        s = max(1, int(round(np.sqrt(p * H * W))))
        s = min(s, H - 1, W - 1)
        # one random square per active sample
        deltas = (x_best_curr - x_curr).clone()
        for k in range(n_act):
            r = rng.integers(0, H - s + 1)
            c = rng.integers(0, W - s + 1)
            sign = rng.choice([-eps, eps])
            # ensure we actually change the patch: paper resamples until any
            # pixel differs by 2*eps from current best
            tries = 0
            while tries < 10:
                cur_patch = deltas[k, 0, r:r + s, c:c + s]
                if torch.any(cur_patch != sign):
                    break
                sign = rng.choice([-eps, eps])
                tries += 1
            deltas[k, 0, r:r + s, c:c + s] = sign

        x_new = (x_curr + deltas).clamp(0.0, 1.0)
        with torch.no_grad():
            new_logits = model(x_new)
            new_loss = _margin_loss(new_logits, y_curr)
            new_pred = new_logits.argmax(1)

        improved = new_loss < loss_curr
        if improved.any():
            sel = idx[improved]
            x_best[sel] = x_new[improved]
            loss_best[sel] = new_loss[improved]

        # check flips (only newly flipped)
        flipped_now = (new_pred != y_curr) & improved
        if flipped_now.any():
            sel = idx[flipped_now]
            # only set queries_to_flip for samples not previously flipped
            unset = flipped_at[sel] > n_iters
            sel_set = sel[unset]
            flipped_at[sel_set] = it + 1
            active[sel_set] = False

    success = (flipped_at <= n_iters)
    return x_best, flipped_at, success


def run_square_attack(model, x, y):
    """Try torchattacks.SquareAttack first; otherwise use built-in."""
    try:
        import torchattacks
        print(f"  using torchattacks {torchattacks.__version__} SquareAttack")
        atk = torchattacks.SquareAttack(model, norm="Linf", eps=EPS,
                                        n_queries=N_QUERIES, p_init=P_INIT,
                                        seed=SEED, verbose=False)
        # torchattacks doesn't easily expose per-iter queries; we run it then
        # supplement with a binary flag and use our own attack to get queries.
        # To keep the script self-contained and give us queries_to_flip, we
        # always run the built-in implementation. Use torchattacks only as a
        # cross-check on success rate.
        with torch.no_grad():
            adv_ta = atk(x, y)
            pred_ta = model(adv_ta).argmax(1)
        ta_flip = (pred_ta != y).float().mean().item()
        print(f"  torchattacks SquareAttack flip rate = {ta_flip:.3f}")
    except Exception as e:
        print(f"  torchattacks unavailable or failed ({type(e).__name__}: {e})")

    print(f"  running built-in Square Attack: eps={EPS:.4f}, n_queries={N_QUERIES}")
    t0 = time.time()
    adv, q_to_flip, success = square_attack_linf(model, x, y)
    print(f"    done in {time.time() - t0:.1f}s   flip rate = {success.float().mean().item():.3f}")
    return adv, q_to_flip, success


# -----------------------------------------------------------------------------
# FGSM
# -----------------------------------------------------------------------------
def fgsm_flip(model, x, y, eps=EPS):
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
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True,  download=True, transform=tf)
    test_set  = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print("training victim...")
    model = train_victim(train_set)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i + 512]).argmax(1))
        preds = torch.cat(preds)
    correct_idx = (preds == test_y).nonzero(as_tuple=True)[0]
    print(f"victim test acc = {correct_idx.numel() / test_y.numel():.4f}")

    g = torch.Generator().manual_seed(SEED)
    perm = correct_idx[torch.randperm(correct_idx.numel(), generator=g)[:N_SAMPLES]]
    x_eval, y_eval = test_x[perm], test_y[perm]

    print(f"running Square Attack on {N_SAMPLES} samples...")
    _, q_to_flip, sq_flip = run_square_attack(model, x_eval, y_eval)
    q_np = q_to_flip.cpu().numpy().astype(float)
    sq_np = sq_flip.cpu().numpy().astype(int)
    print(f"  Square flip rate = {sq_np.mean():.3f}")
    flipped_only = q_np[sq_np == 1]
    if flipped_only.size:
        print(f"  queries-to-flip (among flipped): median={np.median(flipped_only):.0f}  "
              f"mean={flipped_only.mean():.1f}")

    print("running FGSM (white-box reference)...")
    fg_flip = fgsm_flip(model, x_eval, y_eval).cpu().numpy().astype(int)
    print(f"  FGSM flip rate = {fg_flip.mean():.3f}")

    print("computing features...")
    feats, names = compute_features(model, x_eval, y_eval)

    # ---- AUROC table ----
    print("\n=========================================================")
    print(" Per-feature AUROC (binary targets; reported as max(a, 1-a))")
    print("=========================================================")
    bins = {"flipped_Square": sq_np, "flipped_FGSM": fg_flip}
    header = f"  {'feature':<16}" + "".join(f"{t:>22}" for t in bins)
    print(header)
    for j, n in enumerate(names):
        row = f"  {n:<16}"
        for tname, yb in bins.items():
            if yb.std() == 0:
                row += f"{'N/A':>22}"; continue
            try:
                a = roc_auc_score(yb, feats[:, j])
                row += f"{max(a, 1 - a):>22.4f}"
            except Exception:
                row += f"{'err':>22}"
        print(row)

    # ---- Spearman vs continuous queries_to_flip ----
    print("\n  Spearman correlation with continuous queries_to_flip")
    print("  (lower q = more vulnerable; n_iters+1 if never flipped)")
    for j, n in enumerate(names):
        rho, p = spearmanr(feats[:, j], q_np)
        print(f"    {n:<16}  rho={rho:+.4f}  p={p:.3g}")

    # ---- Gradient-masking probe ----
    print("\n=========================================================")
    print(" Gradient-masking probe: cross-tabulate FGSM vs Square")
    print("=========================================================")
    a = int(((fg_flip == 1) & (sq_np == 1)).sum())
    b = int(((fg_flip == 1) & (sq_np == 0)).sum())
    c = int(((fg_flip == 0) & (sq_np == 1)).sum())
    d = int(((fg_flip == 0) & (sq_np == 0)).sum())
    print(f"                       Square_flip=1   Square_flip=0")
    print(f"     FGSM_flip=1      {a:>10d}      {b:>10d}")
    print(f"     FGSM_flip=0      {c:>10d}      {d:>10d}")
    print(f"\n  Samples robust to FGSM but flipped by Square  = {c}  "
          f"(suggests gradient masking if non-trivial)")
    print(f"  Samples flipped by FGSM but robust to Square  = {b}  "
          f"(usually 0 unless Square budget too small)")
    if fg_flip.std() and sq_np.std():
        print(f"  Pearson(FGSM_flip, Square_flip) = "
              f"{np.corrcoef(fg_flip, sq_np)[0,1]:+.4f}")
        try:
            au = roc_auc_score(sq_np, fg_flip)
            print(f"  AUROC(FGSM_flip -> Square_flip) = {max(au, 1 - au):.4f}")
        except Exception:
            pass

    print("\nDONE.")


if __name__ == "__main__":
    main()
