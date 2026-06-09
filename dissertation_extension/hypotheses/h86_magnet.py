"""
Hypothesis H86: MagNet-style defence (Meng & Chen, CCS 2017) as a per-sample
adversarial detector.

MagNet pairs an autoencoder reconstruction-error "detector" with the
autoencoder output as a "reformer": clean inputs reconstruct well and the
victim's softmax distribution does not change much when the input is replaced
by ae(x); adversarial inputs reconstruct worse, and feeding ae(x_adv) through
the victim should yield a softmax distribution that disagrees with
softmax(model(x_adv)).

We combine both signals into a single per-sample score:

    magnet_score(x) = || x - ae(x) ||^2
                    + JS( softmax(model(x))  ||  softmax(model(ae(x))) )

H86 asks: does this score separate clean vs FGSM-adversarial samples on
Fashion-MNIST, and how does its per-sample detection AUROC compare to the
victim's logit margin alone?

Pipeline:
  1. Train a small CNN victim (architecture mirrors diagnostic_test.py) on
     Fashion-MNIST for 10 epochs.
  2. Train a small convolutional autoencoder on the same training set (no
     labels) for a handful of epochs with MSE loss.
  3. Build a balanced 50/50 detection set:
        - "clean" samples = correctly classified test samples
        - "adv"   samples = FGSM perturbations of those same samples at
          eps = 15/255 that actually flip the prediction (true adversarials)
       Take the min count and subsample to 50/50.
  4. For every sample (clean or adv) compute two features:
        - magnet_score  = recon-MSE + JS-divergence of softmax(model(x))
                          against softmax(model(ae(x)))
        - margin        = top1 - top2 logit gap of the victim on the sample
                          (a strong single-feature baseline)
  5. Univariate detection AUROC of each feature against the binary
     is_adversarial target (50/50 base rate, so chance = 0.5).
  6. Also report the additive decomposition: AUROC of recon-MSE only, of the
     JS-divergence term only, and of margin only -- so the reader can see
     where the magnet_score signal is coming from.

Self-contained; uses torch, torchvision, sklearn, numpy. Data is downloaded
to /tmp/data. Trains on CUDA.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda")
DATA_ROOT = "/tmp/data"
EPS_TEST = 15.0 / 255.0
VICTIM_EPOCHS = 10
AE_EPOCHS = 4
BATCH = 128
SEED = 0


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
    """Small conv autoencoder for 28x28 grayscale images."""

    def __init__(self, bottleneck=64):
        super().__init__()
        self.e1 = nn.Conv2d(1, 32, 3, padding=1)
        self.e2 = nn.Conv2d(32, 64, 3, padding=1)
        self.e3 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_fc = nn.Linear(128 * 7 * 7, bottleneck)
        self.dec_fc = nn.Linear(bottleneck, 128 * 7 * 7)
        self.d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.d2 = nn.Conv2d(64, 32, 3, padding=1)
        self.d3 = nn.Conv2d(32, 16, 3, padding=1)
        self.d_out = nn.Conv2d(16, 1, 3, padding=1)

    def encode(self, x):
        x = F.relu(self.e1(x))
        x = F.max_pool2d(x, 2)             # -> 14
        x = F.relu(self.e2(x))
        x = F.max_pool2d(x, 2)             # -> 7
        x = F.relu(self.e3(x))
        x = x.flatten(1)
        return self.enc_fc(x)

    def decode(self, z):
        x = self.dec_fc(z).view(-1, 128, 7, 7)
        x = F.relu(self.d1(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")    # 14
        x = F.relu(self.d2(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")    # 28
        x = F.relu(self.d3(x))
        return torch.sigmoid(self.d_out(x))

    def forward(self, x):
        return self.decode(self.encode(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_victim(train_set, seed=SEED):
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
        print(f"  victim epoch {ep+1}/{VICTIM_EPOCHS}  "
              f"loss={tot/n:.4f}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


def train_autoencoder(train_set, seed=SEED + 1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    ae = ConvAutoencoder().to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    for ep in range(AE_EPOCHS):
        ae.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, _ in loader:                # label discarded
            x = x.to(DEVICE)
            opt.zero_grad()
            recon = ae(x)
            loss = F.mse_loss(recon, x)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  AE     epoch {ep+1}/{AE_EPOCHS}  "
              f"mse={tot/n:.6f}  ({time.time()-t0:.1f}s)")
    ae.eval()
    return ae


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS_TEST, batch=512):
    """Return adv tensor (same shape as x) and a bool flip mask."""
    advs, flips = [], []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        sign = fgsm_grad(model, xb, yb)
        adv = (xb + eps * sign).clamp(0, 1).detach()
        with torch.no_grad():
            flip = model(adv).argmax(1) != yb
        advs.append(adv)
        flips.append(flip)
    return torch.cat(advs), torch.cat(flips)


# ---------------------------------------------------------------------------
# MagNet score
# ---------------------------------------------------------------------------

def victim_logits(model, x, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            out.append(model(x[i:i+batch]))
    return torch.cat(out)


def recon_mse(ae, x, batch=512):
    """Per-sample reconstruction MSE and also the reconstructed tensor."""
    errs, recons = [], []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            xb = x[i:i+batch]
            r = ae(xb)
            e = ((r - xb) ** 2).flatten(1).mean(1)
            errs.append(e)
            recons.append(r)
    return torch.cat(errs), torch.cat(recons)


def js_divergence(p, q, eps=1e-12):
    """Per-sample Jensen-Shannon divergence between two prob distributions.

    JS(p||q) = 0.5 KL(p||m) + 0.5 KL(q||m),    m = 0.5(p+q)
    p, q: (N, K) probability tensors.
    Returns (N,) tensor.
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(1)
    kl_qm = (q * (q.log() - m.log())).sum(1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def magnet_features(model, ae, x, batch=512):
    """Compute (magnet_score, margin, recon_mse_only, js_only) per sample."""
    mse, recon = recon_mse(ae, x, batch=batch)
    # softmax(model(x)) and softmax(model(ae(x)))
    logits_x = victim_logits(model, x, batch=batch)
    logits_r = victim_logits(model, recon, batch=batch)
    p = F.softmax(logits_x, dim=1)
    q = F.softmax(logits_r, dim=1)
    js = js_divergence(p, q)
    score = mse + js
    sorted_l, _ = logits_x.sort(1, descending=True)
    margin = sorted_l[:, 0] - sorted_l[:, 1]
    return score, margin, mse, js


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def auroc_both_dirs(y, score):
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def report(y, feats, names):
    print(f"\n--- univariate detection AUROC  (target: is_adversarial, "
          f"base rate {y.mean():.3f}) ---")
    for name, f in zip(names, feats):
        a = auroc_both_dirs(y, f)
        print(f"  {name:<20}  AUROC = {a:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== H86: MagNet-style detector (recon error + JS reformer divergence) ===")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True,
                                      transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True,
                                     transform=tf)

    print("\n[1/4] training victim CNN")
    t0 = time.time()
    victim = train_victim(train_set, seed=SEED)
    print(f"  victim trained in {time.time()-t0:.1f}s")

    print("\n[2/4] training convolutional autoencoder (NO LABELS)")
    t0 = time.time()
    ae = train_autoencoder(train_set, seed=SEED + 1)
    print(f"  AE trained in {time.time()-t0:.1f}s")

    # materialise test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n[3/4] selecting clean correctly-classified samples and crafting FGSM")
    # Predictions for selecting correctly-classified subset
    with torch.no_grad():
        preds = victim_logits(victim, test_x).argmax(1)
    correct = preds == test_y
    print(f"  victim test accuracy = {correct.float().mean().item():.4f}")
    x_c = test_x[correct]
    y_c = test_y[correct]
    print(f"  correctly classified: {x_c.size(0)} samples")

    adv_x, flip_mask = fgsm_attack(victim, x_c, y_c, eps=EPS_TEST)
    n_flipped = int(flip_mask.sum().item())
    print(f"  FGSM eps={EPS_TEST:.4f}  successful flips = {n_flipped} / "
          f"{x_c.size(0)}  ({flip_mask.float().mean():.4f})")

    # True adversarials only (model flipped)
    adv_x_succ = adv_x[flip_mask]
    # Build balanced 50/50 set: same N of clean and adv
    n = min(adv_x_succ.size(0), x_c.size(0))
    g = torch.Generator(device="cpu").manual_seed(SEED + 17)
    clean_idx = torch.randperm(x_c.size(0), generator=g)[:n]
    adv_idx = torch.randperm(adv_x_succ.size(0), generator=g)[:n]
    clean_pool = x_c[clean_idx.to(DEVICE)]
    adv_pool = adv_x_succ[adv_idx.to(DEVICE)]
    print(f"  detection set: {n} clean + {n} adversarial (50/50)")

    print("\n[4/4] computing MagNet features and detection AUROC")
    score_clean, margin_clean, mse_clean, js_clean = magnet_features(
        victim, ae, clean_pool)
    score_adv, margin_adv, mse_adv, js_adv = magnet_features(
        victim, ae, adv_pool)

    # Quick sanity summary
    print(f"  recon MSE   clean mean={mse_clean.mean():.5f}  "
          f"adv mean={mse_adv.mean():.5f}")
    print(f"  JS-div      clean mean={js_clean.mean():.5f}  "
          f"adv mean={js_adv.mean():.5f}")
    print(f"  magnet      clean mean={score_clean.mean():.5f}  "
          f"adv mean={score_adv.mean():.5f}")
    print(f"  margin      clean mean={margin_clean.mean():.4f}  "
          f"adv mean={margin_adv.mean():.4f}")

    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    feats = {
        "magnet_score": torch.cat([score_clean, score_adv]).cpu().numpy(),
        "recon_mse":    torch.cat([mse_clean, mse_adv]).cpu().numpy(),
        "js_divergence": torch.cat([js_clean, js_adv]).cpu().numpy(),
        "margin":       torch.cat([margin_clean, margin_adv]).cpu().numpy(),
    }

    print("\n========== RESULTS ==========")
    report(y, list(feats.values()), list(feats.keys()))
    print("\n=== H86 done ===")


if __name__ == "__main__":
    main()
