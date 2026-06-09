"""
H73: Gabor filter-bank response statistics predict adversarial vulnerability.

Hypothesis
----------
Per-sample Gabor energy statistics (top-3 most-energetic filter responses + total
energy) carry univariate predictive signal for adversarial vulnerability against
FGSM/PGD/min_eps targets, beyond the standard margin/mean-pixel/std-pixel
baselines.

Pipeline
--------
1. Train a small CNN (architecture matched to diagnostic_test.py) on
   Fashion-MNIST for 10 epochs.
2. For every test sample, convolve with a 16-filter Gabor bank
   (4 frequencies x 4 orientations) using skimage.filters.gabor.
   For each filter compute sum(|response|) -> 16 numbers per sample.
   Reduce to 4 features: top-3 most-energetic responses + total energy.
3. Baseline features: final-model margin, mean pixel, std pixel.
4. Targets:
       - flipped_by_FGSM      (eps = 15/255)
       - flipped_by_PGD       (eps = 15/255, 10 steps)
       - min_eps_to_flip      (binary-search continuous target)
5. Univariate AUROC of every feature against every target (continuous target
   discretised at its median for AUROC).
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage.filters import gabor
from sklearn.metrics import roc_auc_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0


# --- CNN matched to diagnostic_test.py ---------------------------------------
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
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    model.eval()
    return model


# --- attacks -----------------------------------------------------------------
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
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


# --- Gabor features ----------------------------------------------------------
GABOR_FREQS = [0.1, 0.2, 0.3, 0.4]
GABOR_ORIENTS = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def gabor_energies(img_2d):
    """Return 16-vector of sum(|gabor_response|) for one HxW image in [0,1]."""
    energies = np.zeros(len(GABOR_FREQS) * len(GABOR_ORIENTS), dtype=np.float64)
    k = 0
    for f in GABOR_FREQS:
        for theta in GABOR_ORIENTS:
            real, imag = gabor(img_2d, frequency=f, theta=theta)
            energies[k] = np.abs(real).sum() + np.abs(imag).sum()
            k += 1
    return energies


def gabor_features(images_np):
    """images_np: (N, H, W) in [0,1] -> (N, 4) feature matrix:
         [top1_energy, top2_energy, top3_energy, total_energy]."""
    N = images_np.shape[0]
    out = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        e = gabor_energies(images_np[i])
        e_sorted = np.sort(e)[::-1]
        out[i, 0] = e_sorted[0]
        out[i, 1] = e_sorted[1]
        out[i, 2] = e_sorted[2]
        out[i, 3] = e.sum()
        if (i + 1) % 500 == 0:
            print(f"   gabor: {i+1}/{N}")
    return out


# --- baseline features -------------------------------------------------------
def final_margin(model, x):
    with torch.no_grad():
        logits = []
        for i in range(0, x.size(0), 512):
            logits.append(model(x[i:i+512]))
        logits = torch.cat(logits, 0)
    sorted_logits, _ = logits.sort(1, descending=True)
    margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).cpu().numpy()
    pred = logits.argmax(1)
    return margin, pred


def pixel_stats(images_np):
    flat = images_np.reshape(images_np.shape[0], -1)
    return flat.mean(1), flat.std(1)


# --- evaluation --------------------------------------------------------------
def univariate_auroc(feat_matrix, feat_names, targets, target_names):
    print("\n===== Univariate AUROC =====")
    for t_name, y in zip(target_names, targets):
        y = np.asarray(y).astype(int)
        if y.std() == 0:
            print(f"\n target {t_name}: degenerate (pos rate {y.mean():.3f}), skipping")
            continue
        print(f"\n target: {t_name}  (positive rate = {y.mean():.3f})")
        for i, n in enumerate(feat_names):
            x_i = feat_matrix[:, i]
            try:
                a = roc_auc_score(y, x_i)
            except Exception as e:
                print(f"   {n:<22} ERROR {e}")
                continue
            a = max(a, 1 - a)
            print(f"   {n:<22} AUROC = {a:.4f}")


def main():
    print("Loading Fashion-MNIST...")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(f"Training CNN ({EPOCHS} epochs)...")
    t0 = time.time()
    model = train(0, train_set)
    print(f"  done ({time.time()-t0:.1f}s)")

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # restrict to correctly classified test points
    margin_all, pred_all = final_margin(model, test_x)
    correct = (pred_all == test_y).cpu().numpy().astype(bool)
    print(f"Using {int(correct.sum())} correctly-classified test points")

    x_c = test_x[torch.as_tensor(correct, device=DEVICE)]
    y_c = test_y[torch.as_tensor(correct, device=DEVICE)]
    margin = margin_all[correct]

    images_np = x_c.squeeze(1).detach().cpu().numpy().astype(np.float64)
    mean_pix, std_pix = pixel_stats(images_np)

    print("Computing Gabor energies (4 freqs x 4 orientations = 16 filters)...")
    t0 = time.time()
    gabor_feats = gabor_features(images_np)
    print(f"  done ({time.time()-t0:.1f}s)")

    # full feature matrix: [gabor_top1, gabor_top2, gabor_top3, gabor_total,
    #                       margin, mean_pix, std_pix]
    feats = np.column_stack([gabor_feats, margin, mean_pix, std_pix])
    feat_names = ["gabor_top1", "gabor_top2", "gabor_top3", "gabor_total_energy",
                  "final_margin", "mean_pix", "std_pix"]

    # --- targets -----------------------------------------------------------
    print("Computing FGSM flips...")
    fgsm = []
    for i in range(0, x_c.size(0), 512):
        fgsm.append(fgsm_flip(model, x_c[i:i+512], y_c[i:i+512]))
    fgsm = torch.cat(fgsm).cpu().numpy().astype(int)

    print("Computing PGD flips...")
    pgd = []
    for i in range(0, x_c.size(0), 512):
        pgd.append(pgd_flip(model, x_c[i:i+512], y_c[i:i+512]))
    pgd = torch.cat(pgd).cpu().numpy().astype(int)

    print("Computing min_eps_to_flip...")
    me = []
    for i in range(0, x_c.size(0), 512):
        me.append(min_eps_to_flip(model, x_c[i:i+512], y_c[i:i+512]))
    min_eps = torch.cat(me).cpu().numpy()

    # binary version of continuous min_eps: small min_eps == vulnerable
    me_median = np.median(min_eps)
    min_eps_bin = (min_eps <= me_median).astype(int)

    targets = [fgsm, pgd, min_eps_bin]
    target_names = ["FGSM_flip", "PGD_flip", "min_eps<=median (vulnerable)"]

    univariate_auroc(feats, feat_names, targets, target_names)

    # continuous correlations with min_eps
    print("\n===== Correlation with continuous min_eps =====")
    for i, n in enumerate(feat_names):
        cor = np.corrcoef(feats[:, i], min_eps)[0, 1]
        print(f"  corr(min_eps, {n:<22}) = {cor:+.4f}")


if __name__ == "__main__":
    main()
