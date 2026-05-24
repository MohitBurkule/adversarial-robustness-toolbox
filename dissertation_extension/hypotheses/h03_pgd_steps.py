"""
Hypothesis H03: GAIRAT-style PGD-steps-to-flip as per-sample difficulty.

GAIRAT (Zhang et al., ICLR 2021, "Geometry-aware Instance-reweighted Adversarial
Training") defines a per-sample geometry score as the number of PGD steps needed
to flip a sample at fixed epsilon. Samples that flip in few steps are "close to
the decision boundary" and receive higher loss weight during adversarial
training. This script tests whether *image-only* statistics predict this
quantity for a vanilla (non-adversarial) CNN on Fashion-MNIST.

Pipeline:
  1. Train a 5-layer CNN on Fashion-MNIST (10 epochs, Adam) -- same architecture
     as diagnostic_test.py.
  2. For each correctly-classified test sample, run PGD with eps=15/255,
     alpha=eps/4, K=20 steps. Record k* = first step at which prediction flips,
     or K+1 (=21) if never flipped.
  3. Per-sample image features (no model knowledge except victim_margin):
       victim_margin, mean_pix, std_pix, sobel_mean, edge_density,
       jpeg_q75_size, entropy_pix, fourier_high_freq_ratio.
  4. Univariate AUROC predicting (k* <= 5) and Spearman/Pearson correlation
     with log(k*+1).
  5. Multivariate logistic regression / OLS: do image stats add over margin?
  6. Compare against the secondary target "min_eps to flip via FGSM binary
     search" -- which difficulty target is best predicted by image stats?

Run:  python h03_pgd_steps.py
"""
import io
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPS = 15.0 / 255.0
ALPHA = EPS / 4.0
K_PGD = 20
EPOCHS = 10
BATCH = 128


# ---------------- model ----------------
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


def train_model(train_set, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ---------------- attacks ----------------
def pgd_steps_to_flip(model, x, y, eps=EPS, alpha=ALPHA, K=K_PGD):
    """Return per-sample k* = first PGD step at which prediction flips,
    or K+1 if never flipped. Untargeted L_inf PGD, no random start
    (deterministic), starts at the clean image."""
    x0 = x.clone().detach()
    adv = x0.clone().detach()
    N = x.size(0)
    k_star = torch.full((N,), K + 1, dtype=torch.long, device=DEVICE)
    done = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    for k in range(1, K + 1):
        adv = adv.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        g = adv.grad.sign().detach()
        adv = (adv.detach() + alpha * g)
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
        with torch.no_grad():
            pred = model(adv).argmax(1)
        newly = (pred != y) & (~done)
        k_star = torch.where(newly, torch.full_like(k_star, k), k_star)
        done = done | newly
        if done.all():
            break
    return k_star


def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    """Binary-search smallest L_inf eps that flips FGSM (single sign)."""
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


# ---------------- features ----------------
SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = SOBEL_X.transpose(2, 3).clone()


def image_features(x):
    """x: (N,1,28,28) in [0,1]. Returns dict of (N,) tensors (CPU numpy)."""
    N = x.size(0)
    x_cpu = x.detach().cpu()
    xn = x_cpu.numpy()  # (N,1,28,28)

    mean_pix = xn.mean(axis=(1, 2, 3))
    std_pix = xn.std(axis=(1, 2, 3))

    sx = SOBEL_X.to(x.device); sy = SOBEL_Y.to(x.device)
    gx = F.conv2d(x, sx, padding=1)
    gy = F.conv2d(x, sy, padding=1)
    gmag = torch.sqrt(gx * gx + gy * gy)
    sobel_mean = gmag.mean(dim=(1, 2, 3)).detach().cpu().numpy()
    edge_density = (gmag > 0.5).float().mean(dim=(1, 2, 3)).detach().cpu().numpy()

    # JPEG-Q75 compressed-size as complexity proxy
    jpeg_size = np.zeros(N, dtype=np.float32)
    for i in range(N):
        img = (xn[i, 0] * 255).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img, mode="L").save(buf, format="JPEG", quality=75)
        jpeg_size[i] = buf.tell()

    # pixel-intensity entropy (8-bin histogram)
    entropy = np.zeros(N, dtype=np.float32)
    for i in range(N):
        hist, _ = np.histogram(xn[i, 0], bins=8, range=(0.0, 1.0))
        p = hist.astype(np.float64) / max(hist.sum(), 1)
        p = p[p > 0]
        entropy[i] = float(-(p * np.log(p)).sum())

    # Fourier high-frequency energy ratio
    fft = np.fft.fftshift(np.fft.fft2(xn[:, 0]), axes=(1, 2))
    pwr = (fft.real ** 2 + fft.imag ** 2).astype(np.float64)
    H, W = 28, 28
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high_mask = (r > 7).astype(np.float64)
    total = pwr.sum(axis=(1, 2)) + 1e-12
    high = (pwr * high_mask[None]).sum(axis=(1, 2))
    high_ratio = (high / total).astype(np.float32)

    return {
        "mean_pix": mean_pix.astype(np.float32),
        "std_pix": std_pix.astype(np.float32),
        "sobel_mean": sobel_mean.astype(np.float32),
        "edge_density": edge_density.astype(np.float32),
        "jpeg_q75": jpeg_size,
        "entropy_pix": entropy,
        "fourier_high_freq_ratio": high_ratio,
    }


def victim_margin(model, x):
    with torch.no_grad():
        logits = []
        for i in range(0, x.size(0), 512):
            logits.append(model(x[i:i+512]))
        logits = torch.cat(logits, 0)
    s, _ = logits.sort(1, descending=True)
    return (s[:, 0] - s[:, 1]).detach().cpu().numpy()


# ---------------- evaluation ----------------
def univariate_auroc(feat_arr, y_bin):
    auc = roc_auc_score(y_bin, feat_arr)
    return max(auc, 1 - auc)


def evaluate(feats_dict, feat_names, k_star_np, min_eps_np):
    print("\n==== TARGETS ====")
    k = k_star_np
    print(f"  PGD k*: mean={k.mean():.2f}, median={np.median(k):.1f}, "
          f"frac k<=5 = {(k<=5).mean():.3f}, frac never-flipped (k=K+1) = "
          f"{(k>=K_PGD+1).mean():.3f}")
    print(f"  min_eps_FGSM: mean={min_eps_np.mean():.4f}, "
          f"median={np.median(min_eps_np):.4f}")

    # binary target: flipped within 5 steps
    y_bin = (k_star_np <= 5).astype(int)
    log_k = np.log(k_star_np + 1.0)

    X = np.stack([feats_dict[n] for n in feat_names], axis=1)
    Xs = StandardScaler().fit_transform(X)

    print("\n==== Univariate AUROC for P(k* <= 5) ====")
    for i, n in enumerate(feat_names):
        print(f"  {n:<28} AUROC={univariate_auroc(X[:, i], y_bin):.4f}")

    print("\n==== Univariate Pearson/Spearman with log(k*+1) ====")
    for i, n in enumerate(feat_names):
        p = float(np.corrcoef(X[:, i], log_k)[0, 1])
        s = float(spearmanr(X[:, i], log_k).correlation)
        print(f"  {n:<28} pearson={p:+.4f}  spearman={s:+.4f}")

    print("\n==== Multivariate models ====")
    # logistic for binary
    lr_full = LogisticRegression(max_iter=2000).fit(Xs, y_bin)
    auc_full = roc_auc_score(y_bin, lr_full.predict_proba(Xs)[:, 1])
    margin_idx = feat_names.index("victim_margin")
    Xs_marg_only = Xs[:, [margin_idx]]
    lr_marg = LogisticRegression(max_iter=2000).fit(Xs_marg_only, y_bin)
    auc_marg = roc_auc_score(y_bin, lr_marg.predict_proba(Xs_marg_only)[:, 1])
    Xs_no_marg = np.delete(Xs, margin_idx, axis=1)
    lr_nomarg = LogisticRegression(max_iter=2000).fit(Xs_no_marg, y_bin)
    auc_nomarg = roc_auc_score(y_bin, lr_nomarg.predict_proba(Xs_no_marg)[:, 1])
    print(f"  P(k*<=5):  full AUROC={auc_full:.4f}  margin-only={auc_marg:.4f} "
          f" image-only(no margin)={auc_nomarg:.4f}  Delta(image|margin)="
          f"{auc_full-auc_marg:+.4f}")
    print(f"  standardised LR coefs:")
    for n, c in zip(feat_names, lr_full.coef_.flatten()):
        print(f"    {n:<28} {c:+.4f}")

    # OLS for log(k)
    ols_full = LinearRegression().fit(Xs, log_k)
    ols_marg = LinearRegression().fit(Xs_marg_only, log_k)
    ols_nomarg = LinearRegression().fit(Xs_no_marg, log_k)
    print(f"  log(k*+1) OLS R^2:  full={ols_full.score(Xs, log_k):.4f}  "
          f"margin-only={ols_marg.score(Xs_marg_only, log_k):.4f}  "
          f"image-only={ols_nomarg.score(Xs_no_marg, log_k):.4f}")

    # OLS for min_eps
    ols_e_full = LinearRegression().fit(Xs, min_eps_np)
    ols_e_marg = LinearRegression().fit(Xs_marg_only, min_eps_np)
    ols_e_nomarg = LinearRegression().fit(Xs_no_marg, min_eps_np)
    print(f"  min_eps   OLS R^2:  full={ols_e_full.score(Xs, min_eps_np):.4f}  "
          f"margin-only={ols_e_marg.score(Xs_marg_only, min_eps_np):.4f}  "
          f"image-only={ols_e_nomarg.score(Xs_no_marg, min_eps_np):.4f}")

    # univariate AUROC of each image feature for both binary targets
    print("\n==== Comparing targets: which is best predicted by IMAGE STATS? ====")
    img_only = [n for n in feat_names if n != "victim_margin"]
    bin_eps = (min_eps_np <= np.median(min_eps_np)).astype(int)
    for tgt_name, tgt in [("k*<=5", y_bin),
                          ("min_eps<=median", bin_eps)]:
        # multivariate image-only AUROC
        Xi = np.stack([feats_dict[n] for n in img_only], axis=1)
        Xis = StandardScaler().fit_transform(Xi)
        try:
            lr = LogisticRegression(max_iter=2000).fit(Xis, tgt)
            auc = roc_auc_score(tgt, lr.predict_proba(Xis)[:, 1])
        except Exception as e:
            auc = float("nan")
        print(f"  image-only multivariate AUROC on {tgt_name}: {auc:.4f} "
              f"(pos rate={tgt.mean():.3f})")


def main():
    print(f"device: {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print("training victim CNN ...")
    model = train_model(train_set, seed=0)

    # stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # restrict to correctly-classified (GAIRAT scoring only meaningful for these)
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct = preds == test_y
    x, y = test_x[correct], test_y[correct]
    print(f"correct test samples: {x.size(0)}/{test_x.size(0)}")

    # PGD steps-to-flip
    print("computing PGD steps-to-flip ...")
    t0 = time.time()
    k_chunks = []
    for i in range(0, x.size(0), 256):
        k_chunks.append(pgd_steps_to_flip(model, x[i:i+256], y[i:i+256]))
    k_star = torch.cat(k_chunks).detach().cpu().numpy().astype(np.float32)
    print(f"  done ({time.time()-t0:.1f}s)")

    # min_eps via FGSM binary search
    print("computing min_eps_FGSM ...")
    t0 = time.time()
    me_chunks = []
    for i in range(0, x.size(0), 512):
        me_chunks.append(min_eps_fgsm(model, x[i:i+512], y[i:i+512]))
    min_eps = torch.cat(me_chunks).detach().cpu().numpy().astype(np.float32)
    print(f"  done ({time.time()-t0:.1f}s)")

    # features
    print("computing image features ...")
    feats = image_features(x)
    feats["victim_margin"] = victim_margin(model, x)
    feat_names = ["victim_margin", "mean_pix", "std_pix", "sobel_mean",
                  "edge_density", "jpeg_q75", "entropy_pix",
                  "fourier_high_freq_ratio"]

    evaluate(feats, feat_names, k_star, min_eps)
    print("\nDONE.")


if __name__ == "__main__":
    main()
