"""
H106: Pixel-shuffle baseline.

Hypothesis: If image features predict vulnerability *only* through pixel
statistics (not spatial structure), then randomly permuting pixels should
preserve predictor AUROC. If permuted-image features predict vulnerability
as well as original-image features, that is evidence the signal is just
histogram-level (since pixel permutation preserves the intensity histogram
exactly but destroys all spatial structure -> sobel/edges should collapse).

Pipeline:
  1. Train small CNN (matching diagnostic_test.py) on Fashion-MNIST for 10 epochs.
  2. For each test sample, compute 4 image features:
        mean, std, sobel_mean, oti (Otsu threshold index)
     on (a) the original image and (b) a pixel-permuted version (same permutation
     applied per-sample, but using a fixed seed so results are reproducible).
  3. Targets: FGSM self-attack success at eps=15/255 (binary) and min_eps_to_flip
     (continuous, via per-sample binary search like diagnostic_test.py).
  4. Compare univariate + multivariate AUROC of original-feature set vs
     permuted-feature set against the FGSM target.

Write code only -- DO NOT run.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
PERM_SEED = 20260524


# ---- model: identical architecture to diagnostic_test.py ----
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


def train(seed, train_set):
    torch.manual_seed(seed); np.random.seed(seed)
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


# ---- attacks (FGSM, copied to keep self-contained) ----
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def attack_success(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# ---- image features (vectorised, on whatever (N,1,28,28) tensor you pass in) ----
SOBEL_X = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1., -2., -1.],
                        [0., 0., 0.],
                        [1., 2., 1.]]).view(1, 1, 3, 3)


def _otsu_threshold_index(img_flat_256):
    """Otsu threshold index in [0,255] for a single image; img_flat_256 is a
    length-256 histogram (counts). Returns the integer threshold maximising
    inter-class variance.
    """
    total = img_flat_256.sum()
    if total <= 0:
        return 0
    levels = np.arange(256, dtype=np.float64)
    cum = np.cumsum(img_flat_256)
    cum_mu = np.cumsum(img_flat_256 * levels)
    mu_T = cum_mu[-1]
    # weights
    w0 = cum / total
    w1 = 1.0 - w0
    # avoid div-by-zero
    eps = 1e-12
    mu0 = cum_mu / np.maximum(cum, eps)
    mu1 = (mu_T - cum_mu) / np.maximum((total - cum), eps)
    sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2
    # mask out invalid endpoints
    sigma_b2[(w0 <= 0) | (w1 <= 0)] = -1.0
    return int(np.argmax(sigma_b2))


def compute_image_features(x):
    """x: (N,1,28,28) float in [0,1]. Returns (N,4) tensor on CPU:
        [mean, std, sobel_mean, oti]
    Note: oti is computed in [0,255] integer space (standard Otsu).
    """
    N = x.size(0)
    mean = x.mean(dim=(1, 2, 3))
    std = x.std(dim=(1, 2, 3))
    sx = SOBEL_X.to(x.device); sy = SOBEL_Y.to(x.device)
    gx = F.conv2d(x, sx, padding=1)
    gy = F.conv2d(x, sy, padding=1)
    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    sobel_mean = grad_mag.mean(dim=(1, 2, 3))

    # OTI: per-sample Otsu threshold in [0,255]
    x_np = (x.detach().cpu().numpy() * 255.0).astype(np.int64).reshape(N, -1)
    oti = np.zeros(N, dtype=np.float64)
    for i in range(N):
        hist = np.bincount(x_np[i], minlength=256)
        oti[i] = _otsu_threshold_index(hist)
    oti_t = torch.from_numpy(oti).float()
    return torch.stack([mean.cpu(), std.cpu(), sobel_mean.cpu(), oti_t], dim=1)


def pixel_permute(x, seed=PERM_SEED):
    """Apply a fixed per-sample random permutation of the 784 pixels.
    Same permutation per sample (deterministic given seed) so the experiment
    is reproducible. Note: pixel permutation preserves the histogram exactly,
    so mean/std/oti are invariant; only sobel_mean changes meaningfully. We
    still recompute all four features (and report all four) to make the
    invariance explicit.
    """
    N, C, H, W = x.shape
    rng = np.random.default_rng(seed)
    # one shared permutation across all samples is the strongest form of the
    # null: it destroys spatial structure identically for every image.
    perm = rng.permutation(H * W)
    perm_t = torch.from_numpy(perm).long().to(x.device)
    flat = x.view(N, C, H * W)
    permuted = flat[:, :, perm_t].view(N, C, H, W)
    return permuted


def auroc_both_directions(y, score):
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def evaluate_feature_set(feats, y, names, tag):
    print(f"\n--- feature set: {tag} ---")
    Xs = StandardScaler().fit_transform(feats)
    for i, n in enumerate(names):
        a = auroc_both_directions(y, feats[:, i])
        print(f"  univariate AUROC  {n:<20} {a:.4f}")
    lr = LogisticRegression(max_iter=2000).fit(Xs, y)
    auc = roc_auc_score(y, lr.predict_proba(Xs)[:, 1])
    print(f"  multivariate AUROC ({tag}): {auc:.4f}")
    print(f"  standardised coefficients:")
    for n, c in zip(names, lr.coef_.flatten()):
        print(f"    {n:<20} {c:+.4f}")
    return auc


def main():
    print(f"device: {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("training CNN on Fashion-MNIST...")
    t0 = time.time()
    model = train(0, train_set)
    print(f"trained in {time.time()-t0:.1f}s")

    # materialise test set on device
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    # restrict to correctly classified (the population for which adversarial flip is meaningful)
    with torch.no_grad():
        preds = []
        for i in range(0, N, 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct = preds == test_y
    x_c, y_c = test_x[correct], test_y[correct]
    print(f"using {correct.sum().item()}/{N} correctly classified samples")

    # FGSM target
    print("computing FGSM self-attack success at eps=15/255 ...")
    succ = []
    for i in range(0, x_c.size(0), 512):
        succ.append(attack_success(model, x_c[i:i+512], y_c[i:i+512]))
    flipped = torch.cat(succ).cpu().numpy().astype(int)
    print(f"  FGSM flip rate = {flipped.mean():.4f}")

    # also a continuous target for completeness
    print("computing min_eps_to_flip (binary search) ...")
    me = []
    for i in range(0, x_c.size(0), 512):
        me.append(min_eps_to_flip(model, x_c[i:i+512], y_c[i:i+512]))
    min_eps = torch.cat(me).cpu().numpy()
    print(f"  mean min_eps = {min_eps.mean():.4f}")

    # features on ORIGINAL images
    print("computing image features on ORIGINAL images ...")
    feats_orig = compute_image_features(x_c).numpy()

    # features on PIXEL-PERMUTED images
    print("computing image features on PIXEL-PERMUTED images ...")
    x_perm = pixel_permute(x_c, seed=PERM_SEED)
    feats_perm = compute_image_features(x_perm).numpy()

    names = ["mean", "std", "sobel_mean", "oti"]
    names_perm = [n + "_perm" for n in names]

    # sanity: mean/std/oti should be (nearly) identical across orig and perm
    print("\nsanity check (max abs diff per feature, orig vs perm):")
    diff = np.abs(feats_orig - feats_perm).max(axis=0)
    for n, d in zip(names, diff):
        print(f"  {n:<12} max|orig - perm| = {d:.6g}")
    print("  (mean/std/oti should be ~0; sobel_mean should differ)")

    print("\n========== TARGET: FGSM self-flip (binary, eps=15/255) ==========")
    auc_orig = evaluate_feature_set(feats_orig, flipped, names, "ORIGINAL")
    auc_perm = evaluate_feature_set(feats_perm, flipped, names_perm, "PERMUTED")

    # combined: does permuted-sobel add anything beyond original features?
    feats_both = np.concatenate([feats_orig, feats_perm], axis=1)
    auc_both = evaluate_feature_set(feats_both, flipped,
                                    names + names_perm, "ORIGINAL+PERMUTED")

    print("\n===== H106 SUMMARY =====")
    print(f"  multivariate AUROC, ORIGINAL features:          {auc_orig:.4f}")
    print(f"  multivariate AUROC, PERMUTED features:          {auc_perm:.4f}")
    print(f"  multivariate AUROC, ORIGINAL+PERMUTED features: {auc_both:.4f}")
    delta = auc_orig - auc_perm
    print(f"  Delta (orig - perm) = {delta:+.4f}")
    print("\nInterpretation:")
    print("  If delta ~= 0, the predictive signal is histogram-level (H106 supported).")
    print("  If delta is large and positive, spatial structure carries information")
    print("  beyond pixel statistics (H106 refuted).")


if __name__ == "__main__":
    main()
