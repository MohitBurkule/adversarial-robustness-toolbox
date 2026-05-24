"""
Hypothesis H75: VAE marginal likelihood (ELBO) of a sample under a generative
model trained on Fashion-MNIST predicts adversarial vulnerability.

Premise: low-likelihood samples are atypical under the data distribution and
should be more adversarially vulnerable. We approximate log p(x) with the
Evidence Lower BOund (ELBO):

    log p(x) >= E_q(z|x)[ log p(x|z) ] - KL( q(z|x) || p(z) ) = ELBO(x).

Pipeline:
  1. Train a small CNN victim on Fashion-MNIST (10 epochs Adam, matches the
     architecture in diagnostic_test.py).
  2. Separately train a small ConvVAE (4-layer encoder / 4-layer decoder,
     32-d latent) on the same training set for 10 epochs with MSE + KL loss
     (beta-VAE style; sum-MSE reconstruction so the units match KL nats per
     image). No labels are used during VAE training.
  3. Per test sample compute the ELBO components (averaged over a handful of
     posterior samples for stability):
        - log_px_z  = - sum-MSE reconstruction term (Gaussian log-likelihood
                      up to a constant)
        - kl        = KL( q(z|x) || N(0,I) ) per sample
        - elbo      = log_px_z - kl
        - recon     = mean per-pixel MSE (raw reconstruction error)
  4. Baseline features:
        - victim_margin   (top1 - top2 logit gap from the victim)
        - mean_pix
        - std_pix
        - elbo
        - recon
  5. Vulnerability targets:
        - flipped_FGSM    (eps = 15/255)
        - flipped_PGD     (PGD-10, eps = 15/255, alpha = eps/4)
        - FGSM_min_eps    (binary-search smallest L_inf eps that flips FGSM)
  6. Univariate AUROC per (feature, binary target) and Spearman / Pearson
     vs min_eps.

Self-contained. Only needs torch, torchvision, sklearn, scipy, numpy.
Data is downloaded to /tmp/data. Trains on CUDA.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr

DEVICE = torch.device("cuda")
DATA_ROOT = "/tmp/data"
EPS_TEST = 15.0 / 255.0
VICTIM_EPOCHS = 10
VAE_EPOCHS = 10
BATCH = 128
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0
LATENT_DIM = 32
BETA = 1.0
ELBO_SAMPLES = 8  # posterior samples per test image for ELBO estimation


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


class ConvVAE(nn.Module):
    """Small 4-layer-encoder / 4-layer-decoder ConvVAE for 28x28 single-channel
    images. 32-d Gaussian latent. Bernoulli/Gaussian-MSE output via sigmoid.

    Encoder spatial flow:  28 -> 14 -> 7 -> 7 -> 7 (stride pattern 2,2,1,1)
    Decoder spatial flow:  7  -> 7  -> 7 -> 14 -> 28
    """

    def __init__(self, latent=LATENT_DIM):
        super().__init__()
        self.latent = latent
        # 4-layer encoder
        self.e1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)    # 28 -> 14
        self.e2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)   # 14 -> 7
        self.e3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)  # 7  -> 7
        self.e4 = nn.Conv2d(128, 128, 3, stride=1, padding=1) # 7  -> 7
        self.fc_mu = nn.Linear(128 * 7 * 7, latent)
        self.fc_lv = nn.Linear(128 * 7 * 7, latent)

        # 4-layer decoder
        self.fc_z = nn.Linear(latent, 128 * 7 * 7)
        self.d1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)              # 7
        self.d2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)               # 7
        self.d3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)       # 14
        self.d4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)       # 28
        self.d_out = nn.Conv2d(16, 1, 3, padding=1)                        # 28

    def encode(self, x):
        h = F.relu(self.e1(x))
        h = F.relu(self.e2(h))
        h = F.relu(self.e3(h))
        h = F.relu(self.e4(h))
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def reparameterise(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_z(z).view(-1, 128, 7, 7)
        h = F.relu(self.d1(h))
        h = F.relu(self.d2(h))
        h = F.relu(self.d3(h))
        h = F.relu(self.d4(h))
        return torch.sigmoid(self.d_out(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=BETA):
    # sum-MSE reconstruction (proportional to -log p(x|z) under fixed-var
    # Gaussian observation model)
    rec = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return rec + beta * kl, rec, kl


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


def train_vae(train_set, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    vae = ConvVAE().to(DEVICE)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for ep in range(VAE_EPOCHS):
        vae.train()
        t0 = time.time()
        tot, tr, tk, n = 0.0, 0.0, 0.0, 0
        for x, _ in loader:                     # labels discarded
            x = x.to(DEVICE)
            opt.zero_grad()
            recon, mu, logvar = vae(x)
            loss, rec, kl = vae_loss(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            bs = x.size(0)
            tot += loss.item() * bs
            tr += rec.item() * bs
            tk += kl.item() * bs
            n += bs
        print(f"  VAE  epoch {ep+1}/{VAE_EPOCHS}  loss={tot/n:.3f}  "
              f"rec={tr/n:.3f}  kl={tk/n:.3f}  ({time.time()-t0:.1f}s)")
    vae.eval()
    return vae


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


def vae_features(vae, x, n_samples=ELBO_SAMPLES, batch=256):
    """Per-sample ELBO decomposition.

    Returns (log_px_z, kl, elbo, recon_mean) on x.device. log_px_z is the
    negative sum-MSE reconstruction term (up to a constant) averaged across
    posterior samples; kl is closed-form per sample; elbo = log_px_z - kl;
    recon_mean is the mean per-pixel MSE.
    """
    log_px_z_all, kl_all, elbo_all, recon_all = [], [], [], []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            xb = x[i:i+batch]
            mu, logvar = vae.encode(xb)
            # closed-form KL per sample (nats)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)

            # average reconstruction term across posterior samples
            sum_mse_acc = torch.zeros(xb.size(0), device=xb.device)
            mean_mse_acc = torch.zeros(xb.size(0), device=xb.device)
            for _ in range(n_samples):
                z = vae.reparameterise(mu, logvar)
                rec = vae.decode(z)
                diff2 = (rec - xb).pow(2).flatten(1)
                sum_mse_acc += diff2.sum(1)
                mean_mse_acc += diff2.mean(1)
            sum_mse = sum_mse_acc / n_samples
            mean_mse = mean_mse_acc / n_samples

            # log p(x|z) up to additive constant = -0.5 * sum_mse / sigma^2.
            # With sigma=1 this is -0.5 * sum_mse; we keep the proportional
            # quantity -sum_mse (monotone), so higher is better.
            log_px_z = -sum_mse
            elbo = log_px_z - kl

            log_px_z_all.append(log_px_z)
            kl_all.append(kl)
            elbo_all.append(elbo)
            recon_all.append(mean_mse)
    return (torch.cat(log_px_z_all), torch.cat(kl_all),
            torch.cat(elbo_all), torch.cat(recon_all))


def pixel_stats(x):
    flat = x.flatten(1)
    return flat.mean(1), flat.std(1)


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
    """Per-sample binary search for smallest eps that flips FGSM (direction
    fixed at eps=0 gradient sign)."""
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
    print(f"\n--- univariate AUROC ---")
    print(f"{'feature':<15} " + "  ".join(f"{t:>22}" for t in target_names))
    for i, fname in enumerate(feat_names):
        row = f"{fname:<15} "
        for tgt in targets_bin:
            y = tgt.cpu().numpy().astype(int)
            if y.std() == 0:
                row += f"{'(degenerate)':>22}  "
            else:
                row += f"{auroc_both_dirs(y, feats_np[:, i]):>22.4f}  "
        print(row)

    print(f"\n--- correlation with FGSM_min_eps (vulnerability = -min_eps) ---")
    me = min_eps.cpu().numpy()
    for i, fname in enumerate(feat_names):
        sp, _ = spearmanr(feats_np[:, i], me)
        pe, _ = pearsonr(feats_np[:, i], me)
        print(f"  {fname:<15}  spearman={sp:+.4f}   pearson={pe:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== H75: VAE ELBO as vulnerability proxy ===")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True,
                                      transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True,
                                     transform=tf)

    print("\n[1/4] training victim CNN")
    t0 = time.time()
    victim = train_victim(train_set, seed=0)
    print(f"  victim trained in {time.time()-t0:.1f}s")

    print("\n[2/4] training ConvVAE (NO LABELS, MSE + KL)")
    t0 = time.time()
    vae = train_vae(train_set, seed=1)
    print(f"  VAE trained in {time.time()-t0:.1f}s")

    # materialise test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n[3/4] computing per-sample features")
    margin, preds = victim_margin(victim, test_x)
    correct = preds == test_y
    print(f"  victim test accuracy = {correct.float().mean().item():.4f}")
    x_c = test_x[correct]
    y_c = test_y[correct]
    margin_c = margin[correct]
    print(f"  using {x_c.size(0)} correctly-classified test samples")

    log_px_z, kl, elbo, recon = vae_features(vae, x_c)
    mean_pix, std_pix = pixel_stats(x_c)
    feats = torch.stack([margin_c, mean_pix, std_pix, elbo, recon], 1)
    feat_names = ["victim_margin", "mean_pix", "std_pix", "elbo", "recon"]

    print(f"  log_px_z  mean={log_px_z.mean():.3f}  std={log_px_z.std():.3f}")
    print(f"  kl        mean={kl.mean():.3f}        std={kl.std():.3f}")
    print(f"  elbo      mean={elbo.mean():.3f}      std={elbo.std():.3f}")
    print(f"  recon     mean={recon.mean():.5f}     std={recon.std():.5f}")
    print(f"  margin    mean={margin_c.mean():.4f}  std={margin_c.std():.4f}")

    print("\n[4/4] running attacks (FGSM, PGD, min-eps FGSM binary search)")
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
    print("\n=== H75 done ===")


if __name__ == "__main__":
    main()
