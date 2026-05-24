"""
Hypothesis H85: PixelDefend-style purification defence (Song et al., ICLR
2018) — purify adversarial inputs by projecting them onto the data manifold
via a maximum-likelihood generative model, then classify the purified image.

The original PixelDefend uses a PixelCNN to do a constrained greedy MAP-style
purification. Here, as a tractable proxy, we use a small convolutional
autoencoder trained on the clean data as the "manifold projection": passing
an adversarial example through the AE pulls it back toward a clean-looking
reconstruction. The victim then classifies the AE output.

We measure per-sample whether the defence recovers the original (clean)
prediction on FGSM-perturbed inputs at eps=15/255, and ask whether simple
features predict recovery (univariate AUROC):
   - victim_margin on the *adv* input
   - mean_pix      of the *adv* input
   - std_pix       of the *adv* input

Pipeline:
  1. Train victim CNN on Fashion-MNIST, 10 epochs (mirrors diagnostic_test.py).
  2. Train small conv autoencoder on Fashion-MNIST (no labels, 4 epochs).
  3. Restrict to correctly-classified test samples.
  4. Generate FGSM adversarial examples at eps=15/255; keep those that flip
     the victim (i.e. the defence has something to recover from).
  5. Defence: classify ae(adv). Per-sample recovered = (defended pred == y).
  6. Features per sample: margin/mean_pix/std_pix on the adversarial input.
  7. Univariate AUROC of each feature for predicting recovery.

Self-contained; uses torch, torchvision, sklearn, numpy. Data goes to
/tmp/data. Trains on CUDA. Write only — do not run.
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
    """Small conv autoencoder used as a manifold-projection proxy for
    PixelDefend's MAP-style purification."""

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
        return self.decode(self.encode(x))


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
        for x, _ in loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            recon = ae(x)
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
# Features / attack / defence
# ---------------------------------------------------------------------------

def predict(model, x, batch=512):
    margins, preds = [], []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            logits = model(x[i:i+batch])
            srt, _ = logits.sort(1, descending=True)
            margins.append(srt[:, 0] - srt[:, 1])
            preds.append(logits.argmax(1))
    return torch.cat(margins), torch.cat(preds)


def pixel_stats(x):
    flat = x.flatten(1)
    return flat.mean(1), flat.std(1)


def fgsm(model, x, y, eps=EPS_TEST, batch=256):
    advs = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        xb_ = xb.clone().detach().requires_grad_(True)
        F.cross_entropy(model(xb_), yb).backward()
        adv = (xb_ + eps * xb_.grad.sign()).clamp(0, 1).detach()
        advs.append(adv)
    return torch.cat(advs)


def purify(ae, x, batch=512):
    outs = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            outs.append(ae(x[i:i+batch]))
    return torch.cat(outs)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def auroc_both_dirs(y, score):
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== H85: PixelDefend-style AE purification defence ===")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True,
                                      transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True,
                                     transform=tf)

    print("\n[1/5] training victim CNN")
    t0 = time.time()
    victim = train_victim(train_set, seed=0)
    print(f"  victim trained in {time.time()-t0:.1f}s")

    print("\n[2/5] training convolutional autoencoder (NO LABELS)")
    t0 = time.time()
    ae = train_autoencoder(train_set, seed=1)
    print(f"  AE trained in {time.time()-t0:.1f}s")

    # materialise test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n[3/5] selecting correctly-classified test samples")
    _, clean_preds = predict(victim, test_x)
    correct = clean_preds == test_y
    acc = correct.float().mean().item()
    print(f"  victim test accuracy = {acc:.4f}")
    x_c = test_x[correct]
    y_c = test_y[correct]
    print(f"  using {x_c.size(0)} correctly-classified samples")

    print(f"\n[4/5] generating FGSM adv at eps={EPS_TEST:.4f}")
    t0 = time.time()
    adv = fgsm(victim, x_c, y_c, eps=EPS_TEST)
    adv_margin, adv_preds = predict(victim, adv)
    flipped = adv_preds != y_c
    print(f"  FGSM flip rate = {flipped.float().mean():.4f}  "
          f"({time.time()-t0:.1f}s)")

    # restrict to samples actually flipped by FGSM — those are the ones the
    # defence has to recover.
    x_adv = adv[flipped]
    y_adv = y_c[flipped]
    margin_adv = adv_margin[flipped]
    n_flip = x_adv.size(0)
    print(f"  flipped samples (defence target set) = {n_flip}")

    if n_flip == 0:
        print("  no flipped samples; aborting analysis")
        return

    print("\n[5/5] applying PixelDefend-style purification (AE projection)")
    t0 = time.time()
    purified = purify(ae, x_adv)
    _, def_preds = predict(victim, purified)
    recovered = (def_preds == y_adv)
    rec_rate = recovered.float().mean().item()
    print(f"  defence recovery rate = {rec_rate:.4f}  "
          f"({time.time()-t0:.1f}s)")

    # Per-sample features on the adversarial input
    mean_pix, std_pix = pixel_stats(x_adv)
    feats = torch.stack([margin_adv, mean_pix, std_pix], 1).cpu().numpy()
    feat_names = ["margin", "mean_pix", "std_pix"]
    y_rec = recovered.cpu().numpy().astype(int)

    print("\n========== RESULTS ==========")
    print(f"  defence recovery rate (on flipped FGSM) : {rec_rate:.4f}")
    print(f"  feature summary (on adversarial inputs):")
    for i, fname in enumerate(feat_names):
        col = feats[:, i]
        print(f"    {fname:<10}  mean={col.mean():+.4f}  std={col.std():.4f}  "
              f"min={col.min():+.4f}  max={col.max():+.4f}")

    print("\n--- univariate AUROC for predicting recovery ---")
    if y_rec.std() == 0:
        print("  (degenerate: all samples have same recovery outcome)")
    else:
        for i, fname in enumerate(feat_names):
            print(f"  {fname:<10}  AUROC = "
                  f"{auroc_both_dirs(y_rec, feats[:, i]):.4f}")

    print("\n=== H85 done ===")


if __name__ == "__main__":
    main()
