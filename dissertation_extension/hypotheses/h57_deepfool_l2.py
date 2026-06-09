"""
H57: DeepFool L2 per-sample magnitude.

Hypothesis: The minimum L2 perturbation magnitude required by DeepFool
(Moosavi-Dezfooli et al. 2016) is a continuous proxy for adversarial
vulnerability. Per-sample DeepFool L2 magnitude should correlate with
simple image-level features (logit margin, mean/std pixel intensity,
Sobel edge energy) and predict FGSM flips.

Pipeline:
  1. Train a small CNN (architecture matching diagnostic_test.py) on
     Fashion-MNIST for 10 epochs.
  2. Compute features for every test sample:
        - margin       (final-model logit margin: top1 - top2)
        - mean_pix     (mean pixel intensity)
        - std_pix      (std pixel intensity)
        - sobel_mean   (mean Sobel-filter magnitude)
  3. Compute per-sample DeepFool L2 magnitude (Moosavi 2016):
        iterative linearisation; closed-form step
            r_l = ((f_k - f_y) / ||w_k - w_y||_2^2) * (w_k - w_y)
        with k = argmin_k |f_k - f_y| / ||w_k - w_y||_2.
  4. Compute flipped_FGSM at eps=15/255 (L_inf) as a binary target.
  5. Report univariate AUROC of each feature against flipped_FGSM and
     Spearman rank correlation of each feature with deepfool_L2_magnitude.

Self-contained: only depends on torch, torchvision, numpy, sklearn, scipy.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
N_CLASSES = 10
EPS_FGSM = 15.0 / 255.0
DEEPFOOL_MAX_ITER = 50
DEEPFOOL_OVERSHOOT = 0.02


class CNN(nn.Module):
    """Matches dissertation_extension/diagnostic_test.py."""
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


def train_model(train_set, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def sobel_mean(x):
    """Mean Sobel-magnitude per image. x: (N,1,H,W)."""
    kx = _SOBEL_X.to(x.device)
    ky = _SOBEL_Y.to(x.device)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag.mean(dim=(1, 2, 3))


def final_margin(model, x, batch=512):
    margins = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            logits = model(x[i:i+batch])
            top2 = logits.topk(2, dim=1).values
            margins.append(top2[:, 0] - top2[:, 1])
    return torch.cat(margins)


# ---------------------------------------------------------------------------
# DeepFool (L2, Moosavi-Dezfooli et al. 2016)
# ---------------------------------------------------------------------------

def deepfool_l2_single(model, x0, num_classes=N_CLASSES,
                       max_iter=DEEPFOOL_MAX_ITER,
                       overshoot=DEEPFOOL_OVERSHOOT):
    """
    Compute per-sample DeepFool L2 perturbation magnitude.

    x0 : (1, 1, H, W) tensor (single image)
    Returns (||r_total||_2, num_iters, flipped_bool)
    """
    x = x0.clone().detach().to(DEVICE)
    x.requires_grad_(True)

    logits = model(x)
    y0 = int(logits.argmax(1).item())

    r_total = torch.zeros_like(x0).to(DEVICE)
    x_i = x0.clone().detach().to(DEVICE)

    for it in range(max_iter):
        x_i_var = x_i.clone().detach().requires_grad_(True)
        logits = model(x_i_var)
        cur_pred = int(logits.argmax(1).item())
        if cur_pred != y0:
            break

        # gradient of f_y wrt x
        f_y = logits[0, y0]
        grad_y = torch.autograd.grad(f_y, x_i_var, retain_graph=True)[0].detach()

        min_pert = None
        min_dist = float("inf")
        min_w = None
        # iterate over classes k != y0; find closest hyperplane
        for k in range(num_classes):
            if k == y0:
                continue
            f_k = logits[0, k]
            grad_k = torch.autograd.grad(f_k, x_i_var, retain_graph=True)[0].detach()
            w_k = grad_k - grad_y
            f_diff = (f_k - f_y).detach()
            w_norm = w_k.norm().item()
            if w_norm < 1e-12:
                continue
            dist_k = abs(f_diff.item()) / w_norm
            if dist_k < min_dist:
                min_dist = dist_k
                # closed-form step: r = (|f_k - f_y| / ||w_k||_2^2) * w_k
                min_pert = (abs(f_diff.item()) / (w_norm ** 2)) * w_k
                min_w = w_k

        if min_pert is None:
            break

        # small numerical bump so we cross the hyperplane
        r_i = (min_dist + 1e-4) * (min_w / (min_w.norm() + 1e-12))
        r_total = r_total + r_i
        x_i = (x0.to(DEVICE) + (1 + overshoot) * r_total).detach()

    flipped = cur_pred != y0
    return r_total.norm().item(), it + 1, flipped


def deepfool_l2_batch(model, x_batch, num_classes=N_CLASSES):
    """Loop wrapper - DeepFool is naturally per-sample."""
    out = []
    for i in range(x_batch.size(0)):
        mag, _, _ = deepfool_l2_single(model, x_batch[i:i+1], num_classes)
        out.append(mag)
    return torch.tensor(out, device=x_batch.device)


# ---------------------------------------------------------------------------
# FGSM target
# ---------------------------------------------------------------------------

def fgsm_flipped(model, x, y, eps=EPS_FGSM, batch=512):
    out = []
    for i in range(0, x.size(0), batch):
        xb = x[i:i+batch].clone().detach().requires_grad_(True)
        yb = y[i:i+batch]
        F.cross_entropy(model(xb), yb).backward()
        sign = xb.grad.sign().detach()
        adv = (xb.detach() + eps * sign).clamp(0, 1)
        with torch.no_grad():
            pred = model(adv).argmax(1)
        out.append(pred != yb)
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[h57] DeepFool L2 per-sample magnitude on Fashion-MNIST")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(" training small CNN (10 epochs)...")
    t0 = time.time()
    model = train_model(train_set, seed=0)
    print(f"  trained ({time.time()-t0:.1f}s)")

    # tensorise the test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    # restrict to correctly classified samples
    with torch.no_grad():
        preds = []
        for i in range(0, N, 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct_mask = preds == test_y
    x_c = test_x[correct_mask]
    y_c = test_y[correct_mask]
    print(f" using {x_c.size(0)} correctly classified samples (of {N})")

    # ----------------- features -----------------
    print(" computing image features (margin, mean_pix, std_pix, sobel_mean)...")
    margin = final_margin(model, x_c).cpu().numpy()
    mean_pix = x_c.mean(dim=(1, 2, 3)).cpu().numpy()
    std_pix = x_c.std(dim=(1, 2, 3)).cpu().numpy()
    sobel = sobel_mean(x_c).cpu().numpy()

    # ----------------- targets -----------------
    print(" computing DeepFool L2 magnitudes (per-sample loop)...")
    t0 = time.time()
    df_mags = deepfool_l2_batch(model, x_c).cpu().numpy()
    print(f"  done ({time.time()-t0:.1f}s)  "
          f"mean L2={df_mags.mean():.4f}  median={np.median(df_mags):.4f}")

    print(" computing FGSM flips at eps=15/255...")
    fgsm_flip = fgsm_flipped(model, x_c, y_c).cpu().numpy().astype(int)
    print(f"  positive rate = {fgsm_flip.mean():.3f}")

    # ----------------- analysis -----------------
    feat_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]
    feats = {"margin": margin, "mean_pix": mean_pix,
             "std_pix": std_pix, "sobel_mean": sobel}

    print("\n--- univariate AUROC against flipped_FGSM ---")
    for n in feat_names:
        v = feats[n]
        if np.std(v) == 0 or np.std(fgsm_flip) == 0:
            print(f"  {n:<12} degenerate")
            continue
        a = roc_auc_score(fgsm_flip, v)
        a = max(a, 1 - a)
        print(f"  {n:<12} AUROC = {a:.4f}")

    # DeepFool magnitude itself as a predictor of FGSM flips
    if np.std(df_mags) > 0 and np.std(fgsm_flip) > 0:
        a = roc_auc_score(fgsm_flip, df_mags)
        a = max(a, 1 - a)
        print(f"  {'deepfool_L2':<12} AUROC = {a:.4f}")

    print("\n--- Spearman rank corr with deepfool_L2_magnitude ---")
    for n in feat_names:
        rho, p = spearmanr(feats[n], df_mags)
        print(f"  {n:<12} rho = {rho:+.4f}   p = {p:.2e}")

    print("\n--- Spearman rank corr with flipped_FGSM ---")
    for n in feat_names:
        rho, p = spearmanr(feats[n], fgsm_flip)
        print(f"  {n:<12} rho = {rho:+.4f}   p = {p:.2e}")
    rho, p = spearmanr(df_mags, fgsm_flip)
    print(f"  {'deepfool_L2':<12} rho = {rho:+.4f}   p = {p:.2e}")


if __name__ == "__main__":
    main()
