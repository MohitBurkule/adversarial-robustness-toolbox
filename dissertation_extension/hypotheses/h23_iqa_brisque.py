"""
Hypothesis H23: No-reference image quality assessment (NR-IQA) scores --
BRISQUE, NIQE (and PIQE if available) -- predict adversarial vulnerability.
Low-quality / unnatural-statistics images may be more attackable.

Background
----------
BRISQUE (Mittal, Moorthy & Bovik 2012) and NIQE (Mittal, Soundararajan & Bovik
2013) are model-free perceptual-quality predictors built on natural-scene
statistics. They operate on Mean-Subtracted Contrast-Normalized (MSCN)
coefficients and pairwise products thereof, fitting Generalized-/Asymmetric-
Generalized-Gaussian distributions and summarising the parameters. These
indices have *never* been benchmarked as features for predicting per-sample
adversarial vulnerability, only for predicting human-perceived quality.

This hypothesis asks whether images that look unnatural to a BRISQUE/NIQE
classifier are also easier for an FGSM/PGD attacker to flip.

Pipeline
--------
  1. Train a small CNN victim on Fashion-MNIST (10 epochs Adam) -- same
     architecture as diagnostic_test.py / h14.
  2. For each correctly-classified test sample compute BRISQUE and NIQE.
     We attempt three backends in order:
       (a) pyiqa (preferred; pip install pyiqa)
       (b) imquality / image-quality (pip install image-quality)
       (c) a from-scratch BRISQUE-style descriptor: MSCN coefficients + GGD
           fit + AGGD fits on the four pairwise-product orientations at two
           scales -> 36-dim feature vector + an unsupervised "BRISQUE-like"
           quality score = Mahalanobis distance of that vector to a
           multivariate-Gaussian model fit on the training-set features
           (this is the NIQE construction, applied to the BRISQUE feature
           space because Fashion-MNIST is far from natural images and the
           original NIQE "pristine-patch" model would be meaningless).
     Whichever backends succeed are all included as features; the from-
     scratch descriptor is always computed so the script is fully self-
     contained.
  3. Baseline features: victim_margin, mean_pix, std_pix.
  4. Targets:
       - flipped_FGSM      (eps = 15/255)
       - flipped_PGD       (eps = 15/255, 20 steps)
       - min_eps_FGSM<=q25 (binary search smallest-eps that flips with FGSM)
  5. Univariate AUROC for every feature.
     Multivariate logistic regression for nested feature sets:
       margin / iqa / baseline / margin+iqa / baseline+iqa / margin+baseline+iqa
     The key question: does BRISQUE/NIQE add over victim_margin?

Tools: PyTorch + CUDA, scipy for GGD/AGGD fits, sklearn for
LogisticRegression / AUROC / StandardScaler. Optional: pyiqa, imquality.

Caveats
-------
* BRISQUE / NIQE were designed for natural RGB photographs at >=96x96. We
  feed them 28x28 single-channel Fashion-MNIST upsampled to 96x96 RGB by
  bicubic interpolation + channel replication. That is the same hack the
  reference TID / Kadid benchmarks use for tiny grayscale inputs; the
  absolute scores are NOT meaningful as perceptual quality, but the *rank
  ordering across samples* is still well-defined and is all we need for
  AUROC.
* Both pyiqa and imquality may be unavailable on the cluster; the from-
  scratch implementation is the fallback and will always run.
* PIQE has no pure-python implementation that ships with pyiqa <=0.1.10
  (it is matlab-only); we attempt pyiqa("piqe") and skip silently if the
  metric is not registered.
* All AUROC numbers are in-sample (no train/test split on the analysis
  set); we use max(auc, 1-auc) for univariate scoring so the direction of
  the feature is not pre-specified.
"""
import os
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy import ndimage
from scipy.special import gamma as gamma_fn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0
IQA_INPUT_SIZE = 96  # upsample target for pyiqa/imquality BRISQUE/NIQE


# --- model (mirrors diagnostic_test.py) -------------------------------------
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


def train_victim(seed, train_set):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# --- attacks ----------------------------------------------------------------
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x_orig = x.clone().detach()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    adv = (x_orig + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0].sign()
        adv = adv.detach() + alpha * grad
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    sign = fgsm_grad(model, x, y)
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# --- victim margin ----------------------------------------------------------
def victim_margin(model, x):
    with torch.no_grad():
        out = []
        for i in range(0, x.size(0), 512):
            logits = model(x[i:i+512])
            s, _ = logits.sort(1, descending=True)
            out.append((s[:, 0] - s[:, 1]).cpu())
        return torch.cat(out).numpy()


# --- from-scratch BRISQUE-style descriptor ---------------------------------
# MSCN coefficient: I_hat(i,j) = (I(i,j) - mu(i,j)) / (sigma(i,j) + C)
# mu and sigma computed with a circularly-symmetric Gaussian (7x7, sigma=7/6).
def _gauss_kernel_2d(size=7, sigma=7.0 / 6.0):
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    return k


_GK = _gauss_kernel_2d()


def mscn(img):
    """img: HxW float in [0,1]. Returns MSCN coefficient map, same shape."""
    img = img.astype(np.float64)
    mu = ndimage.convolve(img, _GK, mode="reflect")
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(ndimage.convolve(img * img, _GK, mode="reflect") - mu_sq))
    return (img - mu) / (sigma + 1.0 / 255.0)


def _ggd_fit(x):
    """
    Fit a zero-mean Generalized Gaussian Distribution to samples x.
    Returns (alpha, sigma^2). Uses the moment-matching method from
    Sharifi & Leon-Garcia 1995.
    """
    x = x.ravel()
    sigma_sq = (x * x).mean()
    if sigma_sq < 1e-12:
        return 2.0, 0.0
    E = np.mean(np.abs(x))
    rho = sigma_sq / (E * E + 1e-12)
    # invert rho(alpha) = Gamma(1/a)Gamma(3/a)/Gamma(2/a)^2  on a grid.
    grid = np.arange(0.2, 10.0, 0.001)
    r_grid = (gamma_fn(1.0 / grid) * gamma_fn(3.0 / grid)) / (gamma_fn(2.0 / grid) ** 2)
    alpha = grid[np.argmin(np.abs(r_grid - rho))]
    return float(alpha), float(sigma_sq)


def _aggd_fit(x):
    """
    Fit an Asymmetric GGD. Returns (alpha, sigma_l^2, sigma_r^2, eta).
    Lasmar et al. / Mittal et al. moment estimator.
    """
    x = x.ravel()
    left = x[x < 0]
    right = x[x > 0]
    if left.size < 2 or right.size < 2:
        return 2.0, 0.0, 0.0, 0.0
    sigma_l = np.sqrt(np.mean(left * left))
    sigma_r = np.sqrt(np.mean(right * right))
    gamma_hat = sigma_l / (sigma_r + 1e-12)
    r_hat = (np.mean(np.abs(x)) ** 2) / (np.mean(x * x) + 1e-12)
    R_hat = r_hat * (gamma_hat ** 3 + 1.0) * (gamma_hat + 1.0) / ((gamma_hat ** 2 + 1.0) ** 2)
    grid = np.arange(0.2, 10.0, 0.001)
    rgrid = (gamma_fn(2.0 / grid) ** 2) / (gamma_fn(1.0 / grid) * gamma_fn(3.0 / grid))
    alpha = grid[np.argmin(np.abs(rgrid - R_hat))]
    eta = (sigma_r - sigma_l) * (gamma_fn(2.0 / alpha) / gamma_fn(1.0 / alpha))
    return float(alpha), float(sigma_l * sigma_l), float(sigma_r * sigma_r), float(eta)


def brisque_features_one_scale(img):
    """18-dim BRISQUE descriptor at a single scale.
    GGD(2) on MSCN + AGGD(4) on each of the 4 pairwise products."""
    mscn_map = mscn(img)
    alpha, sigma_sq = _ggd_fit(mscn_map)
    feats = [alpha, sigma_sq]
    # four pairwise products: H, V, D1, D2
    H = mscn_map[:, :-1] * mscn_map[:, 1:]
    V = mscn_map[:-1, :] * mscn_map[1:, :]
    D1 = mscn_map[:-1, :-1] * mscn_map[1:, 1:]
    D2 = mscn_map[:-1, 1:] * mscn_map[1:, :-1]
    for pp in (H, V, D1, D2):
        a, sl, sr, eta = _aggd_fit(pp)
        feats += [eta, a, sl, sr]
    return np.array(feats, dtype=np.float64)


def brisque_features(img):
    """36-dim BRISQUE descriptor across two scales (original + /2)."""
    s1 = brisque_features_one_scale(img)
    img2 = img[::2, ::2]
    if img2.shape[0] < 8:  # too small for stable conv at scale 2
        img2 = img  # fallback: duplicate
    s2 = brisque_features_one_scale(img2)
    return np.concatenate([s1, s2])  # 36-dim


def fit_pristine_model(F_train):
    """Fit a multivariate Gaussian (mean, cov) to training feature vectors."""
    mu = F_train.mean(axis=0)
    Xc = F_train - mu
    cov = (Xc.T @ Xc) / max(1, F_train.shape[0] - 1)
    cov += np.eye(cov.shape[0]) * 1e-4
    inv = np.linalg.pinv(cov)
    return mu, inv


def niqe_like_score(F_test, mu, inv):
    """NIQE-style Mahalanobis distance to the natural-image model."""
    d = F_test - mu
    return np.sqrt(np.einsum("ni,ij,nj->n", d, inv, d))


# --- optional pyiqa / imquality backends ------------------------------------
def _to_3ch_96(x):
    """(N,1,28,28) in [0,1] -> (N,3,96,96) bicubic resized, on DEVICE."""
    x = F.interpolate(x, size=(IQA_INPUT_SIZE, IQA_INPUT_SIZE), mode="bicubic", align_corners=False)
    x = x.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
    return x


def try_pyiqa():
    try:
        import pyiqa  # type: ignore
    except Exception as e:
        print(f"  pyiqa unavailable: {e}")
        return {}
    metrics = {}
    for name in ("brisque", "niqe", "piqe"):
        try:
            metrics[f"pyiqa_{name}"] = pyiqa.create_metric(name, device=DEVICE)
            print(f"  pyiqa: loaded {name}")
        except Exception as e:
            print(f"  pyiqa: {name} unavailable ({e})")
    return metrics


def run_pyiqa(metrics, x_in):
    """Run a dict of pyiqa metric callables over (N,1,28,28) torch input."""
    out = {}
    if not metrics:
        return out
    x_big = _to_3ch_96(x_in)
    for name, fn in metrics.items():
        scores = []
        with torch.no_grad():
            for i in range(0, x_big.size(0), 64):
                try:
                    s = fn(x_big[i:i+64])
                    scores.append(s.detach().cpu().numpy().ravel())
                except Exception as e:
                    print(f"  {name} batch {i} failed: {e}")
                    scores.append(np.full(x_big[i:i+64].size(0), np.nan))
        out[name] = np.concatenate(scores)
    return out


def try_imquality(x_np_28):
    """imquality.brisque on numpy (N,28,28) in [0,1]. Returns 1D array or None."""
    try:
        from imquality import brisque as imq_brisque  # type: ignore
        from PIL import Image
    except Exception as e:
        print(f"  imquality unavailable: {e}")
        return None
    scores = np.full(x_np_28.shape[0], np.nan)
    for i in range(x_np_28.shape[0]):
        try:
            arr = (x_np_28[i] * 255.0).astype(np.uint8)
            pil = Image.fromarray(arr).resize((IQA_INPUT_SIZE, IQA_INPUT_SIZE)).convert("RGB")
            scores[i] = float(imq_brisque.score(pil))
        except Exception:
            pass
    return scores


# --- analysis ---------------------------------------------------------------
def univariate_auroc(X, y, names):
    print("\n  univariate AUROC (max(auc, 1-auc)):")
    rows = []
    for i, n in enumerate(names):
        col = X[:, i]
        mask = np.isfinite(col)
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            print(f"    {n:<28} insufficient finite/positive data")
            continue
        try:
            a = roc_auc_score(y[mask], col[mask])
            a = max(a, 1 - a)
            rows.append((n, a))
            print(f"    {n:<28} {a:.4f}  (n={mask.sum()})")
        except Exception as e:
            print(f"    {n:<28} ERROR {e}")
    return rows


def multivariate_auroc(X_full, y, names, sets):
    print("\n  multivariate logistic-regression AUROC (in-sample):")
    name_to_idx = {n: i for i, n in enumerate(names)}
    for set_name, set_names in sets:
        cols = [name_to_idx[n] for n in set_names if n in name_to_idx]
        if not cols:
            print(f"    {set_name:<36} (no features available)")
            continue
        X = X_full[:, cols]
        mask = np.isfinite(X).all(axis=1)
        if mask.sum() < 50:
            print(f"    {set_name:<36} too few finite rows ({mask.sum()})")
            continue
        Xs = StandardScaler().fit_transform(X[mask])
        ys = y[mask]
        if len(np.unique(ys)) < 2:
            print(f"    {set_name:<36} degenerate target")
            continue
        clf = LogisticRegression(max_iter=2000).fit(Xs, ys)
        a = roc_auc_score(ys, clf.predict_proba(Xs)[:, 1])
        print(f"    {set_name:<36} ({len(cols):>2} feats, n={mask.sum()})  AUROC = {a:.4f}")


# --- main -------------------------------------------------------------------
def main():
    print(f"device = {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print("training victim CNN on Fashion-MNIST...")
    t0 = time.time()
    model = train_victim(0, train_set)
    print(f"trained in {time.time()-t0:.1f}s")

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)
    print(f"test set: N={N}")

    with torch.no_grad():
        preds = []
        for i in range(0, N, 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct = preds == test_y
    print(f"clean accuracy = {correct.float().mean().item():.4f}")
    x = test_x[correct]
    y = test_y[correct]
    Nc = x.size(0)
    print(f"using {Nc} correctly-classified samples")

    # ---- baseline features ----
    print("\ncomputing victim margin...")
    margin = victim_margin(model, x)

    print("computing baseline pixel stats...")
    with torch.no_grad():
        flat0 = x.view(Nc, -1)
        mean_pix = flat0.mean(dim=1).cpu().numpy()
        std_pix = flat0.std(dim=1, unbiased=False).cpu().numpy()

    # ---- IQA features ----
    x_np = x.squeeze(1).cpu().numpy()  # (Nc, 28, 28) in [0,1]

    print("\ncomputing from-scratch BRISQUE-style 36-dim descriptor...")
    t0 = time.time()
    F_all = np.zeros((Nc, 36), dtype=np.float64)
    for i in range(Nc):
        F_all[i] = brisque_features(x_np[i])
        if (i + 1) % 1000 == 0:
            print(f"   {i+1}/{Nc}  ({time.time()-t0:.1f}s)")
    print(f"  done in {time.time()-t0:.1f}s")

    # Take a "pristine" sub-population: the training set's MSCN-feature
    # distribution would be ideal but we want this to be self-contained and
    # fast. Use the FULL Fashion-MNIST training set (60k) is too slow per-
    # image; use a 2000-sample subset.
    print("\nfitting NIQE-like pristine model from 2000 training images...")
    t0 = time.time()
    n_train_sample = 2000
    rng = np.random.RandomState(0)
    idx = rng.choice(len(train_set), n_train_sample, replace=False)
    F_train = np.zeros((n_train_sample, 36), dtype=np.float64)
    for k, j in enumerate(idx):
        img = train_set[j][0].squeeze(0).numpy()
        F_train[k] = brisque_features(img)
    mu_p, inv_p = fit_pristine_model(F_train)
    print(f"  fit in {time.time()-t0:.1f}s")
    niqe_scratch = niqe_like_score(F_all, mu_p, inv_p)
    # A simple BRISQUE-scratch summary: the L2 norm of the AGGD-eta entries
    # (deviation from symmetric natural statistics). This is a single scalar
    # in addition to the full 36-dim vector.
    eta_idx_scale1 = [2, 6, 10, 14]
    eta_idx_scale2 = [2 + 18, 6 + 18, 10 + 18, 14 + 18]
    brisque_scratch_eta = np.linalg.norm(F_all[:, eta_idx_scale1 + eta_idx_scale2], axis=1)

    # Try pyiqa
    print("\nattempting pyiqa backend...")
    pyiqa_metrics = try_pyiqa()
    pyiqa_scores = run_pyiqa(pyiqa_metrics, x)

    # Try imquality (BRISQUE only)
    print("\nattempting imquality backend (BRISQUE)...")
    imq_brisque = try_imquality(x_np)

    # ---- assemble feature matrix ----
    baseline_names = ["victim_margin", "mean_pix", "std_pix"]
    baseline = np.column_stack([margin, mean_pix, std_pix])

    iqa_cols = []
    iqa_names = []

    iqa_cols.append(niqe_scratch.reshape(-1, 1))
    iqa_names.append("niqe_scratch")
    iqa_cols.append(brisque_scratch_eta.reshape(-1, 1))
    iqa_names.append("brisque_scratch_eta")

    for name, vec in pyiqa_scores.items():
        iqa_cols.append(np.asarray(vec).reshape(-1, 1))
        iqa_names.append(name)

    if imq_brisque is not None and np.isfinite(imq_brisque).any():
        iqa_cols.append(imq_brisque.reshape(-1, 1))
        iqa_names.append("imquality_brisque")

    iqa_block = np.concatenate(iqa_cols, axis=1)
    X = np.concatenate([baseline, iqa_block], axis=1)
    feat_names = baseline_names + iqa_names
    print(f"\nfeature matrix: {X.shape}, columns: {feat_names}")

    # ---- targets ----
    print("\ncomputing flipped_FGSM (eps=15/255) ...")
    fgsm_flip = []
    for i in range(0, Nc, 512):
        fgsm_flip.append(fgsm_attack(model, x[i:i+512], y[i:i+512]))
    fgsm_flip = torch.cat(fgsm_flip).cpu().numpy().astype(int)

    print("computing flipped_PGD (eps=15/255, 20 steps) ...")
    pgd_flip = []
    for i in range(0, Nc, 512):
        pgd_flip.append(pgd_attack(model, x[i:i+512], y[i:i+512]))
    pgd_flip = torch.cat(pgd_flip).cpu().numpy().astype(int)

    print("computing min_eps_FGSM (binary search) ...")
    me = []
    for i in range(0, Nc, 512):
        me.append(min_eps_fgsm(model, x[i:i+512], y[i:i+512]))
    min_eps = torch.cat(me).cpu().numpy()
    me_thresh = np.quantile(min_eps, 0.25)
    min_eps_bin = (min_eps <= me_thresh).astype(int)

    targets = {
        "flipped_FGSM (eps=15/255)": fgsm_flip,
        "flipped_PGD (eps=15/255)": pgd_flip,
        f"min_eps_FGSM<=q25 ({me_thresh:.4f})": min_eps_bin,
    }

    sets = [
        ("margin only",                    ["victim_margin"]),
        ("baseline (margin+mean+std)",     baseline_names),
        ("iqa only",                       iqa_names),
        ("margin + iqa",                   ["victim_margin"] + iqa_names),
        ("baseline + iqa",                 baseline_names + iqa_names),
    ]

    for tname, yt in targets.items():
        print(f"\n========== target: {tname}  (pos rate = {yt.mean():.3f}) ==========")
        if yt.std() == 0:
            print("  degenerate target, skipping")
            continue
        univariate_auroc(X, yt, feat_names)
        multivariate_auroc(X, yt, feat_names, sets)

    print("\n========== Pearson corr with min_eps_FGSM (continuous) ==========")
    for i, n in enumerate(feat_names):
        col = X[:, i]
        mask = np.isfinite(col)
        if mask.sum() < 10:
            print(f"   corr({n:<28}, min_eps) = NaN (no finite)")
            continue
        c = np.corrcoef(col[mask], min_eps[mask])[0, 1]
        print(f"   corr({n:<28}, min_eps) = {c:+.4f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
