"""
Hypothesis H92: Defense-GAN-style projection distance predicts adversarial
vulnerability.

The original Defense-GAN idea: given a test image x, search the latent space
of a trained generator G for z* = argmin_z ||G(z) - x||^2. The residual
||G(z*) - x|| (the "projection distance" onto the manifold of GAN-generated
images) measures how far x sits from the learned data manifold. Off-manifold
samples are hypothesised to be more adversarially vulnerable: they live in
sparse density regions and so the victim's decision boundary near them is
poorly constrained.

Training a full GAN on Fashion-MNIST plus running per-sample latent-space
optimisation is too expensive for this diagnostic. We use two cheap proxies
that capture the same "distance-to-data-manifold" intuition:

  (a) A small conv autoencoder fit on the training set (NO labels). Per
      test sample:
        - AE_recon_err  = MSE(AE(x), x)        (projection residual proxy)
        - AE_latent_norm = ||enc(x)||_2        (latent magnitude; very
                                                far-from-data samples often
                                                produce unusually large or
                                                unusually small codes)
  (b) sklearn KernelDensity (gaussian kernel) fit on PCA-50 features of the
      training set. Per test sample:
        - kde_logp      = log p(x_pca50)       (manifold likelihood proxy;
                                                higher = closer to data
                                                manifold, lower = further)

Baseline features:
        - victim_margin (top1 - top2 logit gap)
        - mean_pix
        - std_pix

Vulnerability targets:
        - flipped_FGSM    (FGSM at eps=15/255)
        - flipped_PGD     (PGD-10 at eps=15/255, alpha=eps/4)
        - FGSM_min_eps    (per-sample binary search smallest L_inf eps that
                          flips FGSM; continuous, vulnerability = -min_eps)

Univariate AUROC of each feature for each binary target, plus Spearman /
Pearson against min_eps, plus a small multivariate check (do the manifold
proxies add over margin alone?).

Victim CNN mirrors diagnostic_test.py (10 epochs Adam). Self-contained;
only needs torch, torchvision, sklearn, scipy, numpy. Data is downloaded
to /tmp/data. Trains on CUDA.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

DEVICE = torch.device("cuda")
DATA_ROOT = "/tmp/data"
EPS_TEST = 15.0 / 255.0
VICTIM_EPOCHS = 10
AE_EPOCHS = 4
BATCH = 128
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0

PCA_DIM = 50
KDE_BANDWIDTH = 1.0          # scott-ish on PCA-50 of FashionMNIST
KDE_FIT_SUBSET = 10000       # KDE is O(n) per query; cap training points


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    """Victim CNN, mirrors diagnostic_test.py."""

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


class ConvAutoencoder(nn.Module):
    """Small conv autoencoder for 28x28 single-channel images.

    Same shape as h13's AE (28 -> 14 -> 7, bottleneck dense=64). Acts as our
    Defense-GAN proxy: the recon is the closest "generated" image to x, the
    MSE is the projection distance, and the latent code is the analogue of
    the GAN's z*.
    """

    def __init__(self, bottleneck=64):
        super().__init__()
        self.bottleneck = bottleneck
        # encoder
        self.e1 = nn.Conv2d(1, 32, 3, padding=1)
        self.e2 = nn.Conv2d(32, 64, 3, padding=1)
        self.e3 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_fc = nn.Linear(128 * 7 * 7, bottleneck)
        # decoder
        self.dec_fc = nn.Linear(bottleneck, 128 * 7 * 7)
        self.d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.d2 = nn.Conv2d(64, 32, 3, padding=1)
        self.d3 = nn.Conv2d(32, 16, 3, padding=1)
        self.d_out = nn.Conv2d(16, 1, 3, padding=1)

    def encode(self, x):
        x = F.relu(self.e1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.e2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.e3(x))
        x = x.flatten(1)
        return self.enc_fc(x)

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.d1(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.d2(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.d3(x))
        return torch.sigmoid(self.d_out(x))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_victim(train_set, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(VICTIM_EPOCHS):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  victim epoch {ep+1}/{VICTIM_EPOCHS}  loss={tot/n:.4f}  "
              f"({time.time()-t0:.1f}s)")
    model.eval()
    return model


def train_autoencoder(train_set, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    ae = ConvAutoencoder().to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    for ep in range(AE_EPOCHS):
        ae.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, _ in loader:                      # label discarded
            x = x.to(DEVICE)
            opt.zero_grad()
            recon, _ = ae(x)
            loss = F.mse_loss(recon, x)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  AE   epoch {ep+1}/{AE_EPOCHS}    mse={tot/n:.6f}  "
              f"({time.time()-t0:.1f}s)")
    ae.eval()
    return ae


# ---------------------------------------------------------------------------
# Per-sample features
# ---------------------------------------------------------------------------

def victim_margin(model, x, batch=512):
    margins, preds = [], []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            logits = model(x[i:i+batch])
            srt, _ = logits.sort(1, descending=True)
            margins.append(srt[:, 0] - srt[:, 1])
            preds.append(logits.argmax(1))
    return torch.cat(margins), torch.cat(preds)


def ae_features(ae, x, batch=512):
    """Returns (recon_err, latent_norm) per sample."""
    errs, lnorms = [], []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            xb = x[i:i+batch]
            r, z = ae(xb)
            errs.append(((r - xb) ** 2).flatten(1).mean(1))
            lnorms.append(z.norm(dim=1))
    return torch.cat(errs), torch.cat(lnorms)


def pixel_stats(x):
    flat = x.flatten(1)
    return flat.mean(1), flat.std(1)


def fit_pca_kde(train_set, seed=2):
    """Fit PCA(50) + KernelDensity on a subset of the training set."""
    rng = np.random.RandomState(seed)
    n = len(train_set)
    idx = rng.choice(n, size=min(KDE_FIT_SUBSET, n), replace=False)
    X = np.stack([train_set[int(i)][0].numpy().reshape(-1) for i in idx])
    pca = PCA(n_components=PCA_DIM, random_state=seed).fit(X)
    Xp = pca.transform(X)
    kde = KernelDensity(kernel="gaussian", bandwidth=KDE_BANDWIDTH).fit(Xp)
    return pca, kde


def kde_logp(pca, kde, x_tensor, batch=512):
    """log-density of test samples under PCA-50 + Gaussian KDE."""
    x_np = x_tensor.detach().cpu().numpy().reshape(x_tensor.size(0), -1)
    Xp = pca.transform(x_np)
    out = np.empty(Xp.shape[0], dtype=np.float64)
    for i in range(0, Xp.shape[0], batch):
        out[i:i+batch] = kde.score_samples(Xp[i:i+batch])
    return torch.from_numpy(out).float().to(x_tensor.device)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST, batch=512):
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        sign = fgsm_grad(model, xb, yb)
        adv = (xb + eps * sign).clamp(0, 1)
        with torch.no_grad():
            out.append(model(adv).argmax(1) != yb)
    return torch.cat(out)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS,
             batch=256):
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        delta = torch.empty_like(xb).uniform_(-eps, eps)
        adv = (xb + delta).clamp(0, 1).detach()
        for _ in range(steps):
            adv.requires_grad_(True)
            F.cross_entropy(model(adv), yb).backward()
            with torch.no_grad():
                adv = adv + alpha * adv.grad.sign()
                adv = torch.max(torch.min(adv, xb + eps), xb - eps).clamp(0, 1)
            adv = adv.detach()
        with torch.no_grad():
            out.append(model(adv).argmax(1) != yb)
    return torch.cat(out)


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15, batch=512):
    """Binary search per-sample smallest eps that flips FGSM."""
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        sign = fgsm_grad(model, xb, yb)
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xb + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
            with torch.no_grad():
                flipped = model(adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        out.append(hi)
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def auroc_both_dirs(y, score):
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def evaluate(feats, feat_names, targets_bin, target_names, min_eps):
    feats_np = feats.cpu().numpy()
    Xs = StandardScaler().fit_transform(feats_np)

    print("\n--- univariate AUROC ---")
    print(f"{'feature':<18} " + "  ".join(f"{t:>22}" for t in target_names))
    for i, fname in enumerate(feat_names):
        row = f"{fname:<18} "
        for tgt in targets_bin:
            y = tgt.cpu().numpy().astype(int)
            if y.std() == 0:
                row += f"{'(degenerate)':>22}  "
            else:
                row += f"{auroc_both_dirs(y, feats_np[:, i]):>22.4f}  "
        print(row)

    print("\n--- correlation with FGSM_min_eps (vulnerability = -min_eps) ---")
    me = min_eps.cpu().numpy()
    for i, fname in enumerate(feat_names):
        sp, _ = spearmanr(feats_np[:, i], me)
        pe, _ = pearsonr(feats_np[:, i], me)
        print(f"  {fname:<18} spearman={sp:+.4f}   pearson={pe:+.4f}")

    # Multivariate: do manifold proxies add over margin alone?
    margin_idx = feat_names.index("victim_margin")
    proxy_idxs = [feat_names.index(n) for n in
                  ("AE_recon_err", "AE_latent_norm", "kde_logp")]

    print("\n--- multivariate: do GAN-proxy features add over margin? ---")
    print(f"{'target':<22} {'margin':>8} {'+proxies':>10} {'delta':>8}  "
          f"{'all':>8}")
    for tgt, tname in zip(targets_bin, target_names):
        y = tgt.cpu().numpy().astype(int)
        if y.std() == 0:
            continue
        a_m = roc_auc_score(y, LogisticRegression(max_iter=2000)
                            .fit(Xs[:, [margin_idx]], y)
                            .predict_proba(Xs[:, [margin_idx]])[:, 1])
        cols = [margin_idx] + proxy_idxs
        a_mp = roc_auc_score(y, LogisticRegression(max_iter=2000)
                             .fit(Xs[:, cols], y)
                             .predict_proba(Xs[:, cols])[:, 1])
        a_all = roc_auc_score(y, LogisticRegression(max_iter=2000)
                              .fit(Xs, y)
                              .predict_proba(Xs)[:, 1])
        print(f"{tname:<22} {a_m:>8.4f} {a_mp:>10.4f} {a_mp-a_m:>+8.4f}  "
              f"{a_all:>8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== H92: Defense-GAN-style projection distance "
          "(AE + KDE proxies) ===")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True,
                                      transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True,
                                     transform=tf)

    print("\n[1/5] training victim CNN")
    t0 = time.time()
    victim = train_victim(train_set, seed=0)
    print(f"  victim trained in {time.time()-t0:.1f}s")

    print("\n[2/5] training conv autoencoder (GAN-projection proxy, NO LABELS)")
    t0 = time.time()
    ae = train_autoencoder(train_set, seed=1)
    print(f"  AE trained in {time.time()-t0:.1f}s")

    print(f"\n[3/5] fitting PCA-{PCA_DIM} + KernelDensity on training subset "
          f"(n={KDE_FIT_SUBSET})")
    t0 = time.time()
    pca, kde = fit_pca_kde(train_set, seed=2)
    print(f"  PCA+KDE fit in {time.time()-t0:.1f}s")

    # materialise test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n[4/5] computing per-sample features")
    margin, preds = victim_margin(victim, test_x)
    correct = preds == test_y
    print(f"  victim test accuracy = {correct.float().mean().item():.4f}")
    x_c = test_x[correct]
    y_c = test_y[correct]
    margin_c = margin[correct]
    print(f"  using {x_c.size(0)} correctly-classified test samples")

    ae_recon, ae_latent = ae_features(ae, x_c)
    mean_pix, std_pix = pixel_stats(x_c)
    logp = kde_logp(pca, kde, x_c)

    feats = torch.stack([margin_c, mean_pix, std_pix,
                         ae_recon, ae_latent, logp], 1)
    feat_names = ["victim_margin", "mean_pix", "std_pix",
                  "AE_recon_err", "AE_latent_norm", "kde_logp"]

    print(f"  AE_recon_err   mean={ae_recon.mean():.5f} std={ae_recon.std():.5f}")
    print(f"  AE_latent_norm mean={ae_latent.mean():.4f} std={ae_latent.std():.4f}")
    print(f"  kde_logp       mean={logp.mean():.4f} std={logp.std():.4f}")
    print(f"  victim_margin  mean={margin_c.mean():.4f} std={margin_c.std():.4f}")

    print("\n[5/5] running attacks (FGSM, PGD, min-eps FGSM binary search)")
    t0 = time.time()
    flipped_fgsm = fgsm_flip(victim, x_c, y_c, eps=EPS_TEST)
    print(f"  FGSM flip rate eps={EPS_TEST:.4f}: "
          f"{flipped_fgsm.float().mean():.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    flipped_pgd = pgd_flip(victim, x_c, y_c, eps=EPS_TEST,
                           alpha=PGD_ALPHA, steps=PGD_STEPS)
    print(f"  PGD-{PGD_STEPS} flip rate eps={EPS_TEST:.4f}: "
          f"{flipped_pgd.float().mean():.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    min_eps = min_eps_fgsm(victim, x_c, y_c)
    print(f"  min_eps FGSM   mean={min_eps.mean():.4f}  "
          f"median={min_eps.median():.4f}  ({time.time()-t0:.1f}s)")

    targets_bin = [flipped_fgsm, flipped_pgd]
    target_names = ["flipped_FGSM", "flipped_PGD"]

    print("\n========== RESULTS ==========")
    evaluate(feats, feat_names, targets_bin, target_names, min_eps)
    print("\n=== H92 done ===")


if __name__ == "__main__":
    main()
