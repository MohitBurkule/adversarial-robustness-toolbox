"""
H60: Adversarial patch (Brown et al. 2017) — per-sample vulnerability.

Pipeline:
  1. Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
  2. Optimize a single fixed-size 8x8 adversarial patch on the training set such
     that, when placed at a random location on each input, the model's
     cross-entropy loss against the true labels is MAXIMIZED (untargeted patch).
  3. Per test sample:
       - apply the patch at each of 9 grid positions
       - record (a) flipped_at_any_position (binary), and
         (b) min L2 perturbation magnitude over positions that flip the prediction
           (NaN-like sentinel for never-flipped, replaced by a large value).
  4. Compute features margin, mean_pix, std_pix, sobel_mean.
  5. Report univariate AUROC of each feature against (flipped_at_any_position).
  6. Compare per-sample ranking with min-eps FGSM (binary search on L_inf).

Self-contained. Write code only — DO NOT run.
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
PATCH_SIZE = 8
IMG_SIZE = 28
SEED = 0

# the 9 grid positions for evaluation (top-left corners of an 8x8 patch on 28x28)
GRID_POS = [(r, c) for r in (0, 10, 20) for c in (0, 10, 20)]


# ---------- model (matches diagnostic_test.py) ----------
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


def train_model(train_set):
    torch.manual_seed(SEED); np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ---------- patch application ----------
def apply_patch(x, patch, r, c):
    """Place patch (1,P,P) at (r,c) on every image in batch x (B,1,H,W). Returns new tensor."""
    out = x.clone()
    out[:, :, r:r + PATCH_SIZE, c:c + PATCH_SIZE] = patch
    return out


def apply_patch_random(x, patch):
    """Place patch at independent random positions for each image in the batch."""
    B = x.size(0)
    out = x.clone()
    rs = torch.randint(0, IMG_SIZE - PATCH_SIZE + 1, (B,))
    cs = torch.randint(0, IMG_SIZE - PATCH_SIZE + 1, (B,))
    for i in range(B):
        out[i, :, rs[i]:rs[i] + PATCH_SIZE, cs[i]:cs[i] + PATCH_SIZE] = patch
    return out


# ---------- patch optimization (Brown et al. 2017) ----------
def train_patch(model, train_set, n_iters=2000, lr=0.05, batch=128):
    """Optimize an 8x8 patch to MAXIMIZE cross-entropy at a random location."""
    torch.manual_seed(SEED + 1)
    patch = torch.rand(1, PATCH_SIZE, PATCH_SIZE, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([patch], lr=lr)

    loader = DataLoader(train_set, batch, shuffle=True, num_workers=2, drop_last=True)
    it = iter(loader)
    model.eval()
    for step in range(n_iters):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        adv = apply_patch_random(x, patch.clamp(0, 1))
        loss = -F.cross_entropy(model(adv), y)  # maximize CE => minimize -CE
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            patch.clamp_(0, 1)
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                pred = model(adv).argmax(1)
                acc = (pred == y).float().mean().item()
            print(f"    patch step {step+1}/{n_iters}  -CE={loss.item():+.4f}  acc_under_patch={acc:.3f}")
    return patch.detach().clamp(0, 1)


# ---------- per-sample patch evaluation over the 9-grid ----------
@torch.no_grad()
def eval_patch_grid(model, x, y, patch, batch=256):
    """For each sample return:
         flipped_any (bool):   patch flipped prediction at ANY of the 9 positions
         min_l2 (float):       smallest L2 norm of perturbation (image_with_patch - image)
                               over positions that flipped the prediction
                               (sentinel value = large number if never flipped).
    """
    N = x.size(0)
    flipped_any = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    min_l2 = torch.full((N,), float("inf"), device=DEVICE)
    SENTINEL = 999.0

    for r, c in GRID_POS:
        for i in range(0, N, batch):
            xb = x[i:i + batch]; yb = y[i:i + batch]
            adv = apply_patch(xb, patch, r, c)
            pred = model(adv).argmax(1)
            flipped = pred != yb
            # L2 of (adv - xb) per sample
            l2 = (adv - xb).flatten(1).norm(dim=1)
            flipped_any[i:i + batch] |= flipped
            # update min_l2 only where flipped this position
            cur = min_l2[i:i + batch]
            new = torch.where(flipped, torch.minimum(cur, l2), cur)
            min_l2[i:i + batch] = new

    # replace +inf with sentinel for unflipped samples (so AUROC / spearman work)
    min_l2 = torch.where(torch.isinf(min_l2), torch.full_like(min_l2, SENTINEL), min_l2)
    return flipped_any, min_l2


# ---------- FGSM min-eps (binary search), used as a ranking comparison ----------
def fgsm_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=15, batch=512):
    """Per-sample binary-search smallest L_inf eps that flips FGSM."""
    N = x.size(0)
    out = torch.zeros(N, device=DEVICE)
    for i in range(0, N, batch):
        xb = x[i:i + batch]; yb = y[i:i + batch]
        sign = fgsm_sign(model, xb, yb)
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xb + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
            with torch.no_grad():
                flipped = model(adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        out[i:i + batch] = hi
    return out


# ---------- per-sample features ----------
@torch.no_grad()
def compute_features(model, x, y, batch=512):
    """Return numpy array (N, 4): margin, mean_pix, std_pix, sobel_mean."""
    N = x.size(0)
    margin = torch.zeros(N, device=DEVICE)
    for i in range(0, N, batch):
        logits = model(x[i:i + batch])
        srt, _ = logits.sort(1, descending=True)
        margin[i:i + batch] = srt[:, 0] - srt[:, 1]

    mean_pix = x.flatten(1).mean(1)
    std_pix = x.flatten(1).std(1)

    # Sobel mean magnitude
    sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=DEVICE).view(1, 1, 3, 3)
    sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=DEVICE).view(1, 1, 3, 3)
    sobel_mean = torch.zeros(N, device=DEVICE)
    for i in range(0, N, batch):
        xb = x[i:i + batch]
        gx = F.conv2d(xb, sx, padding=1)
        gy = F.conv2d(xb, sy, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        sobel_mean[i:i + batch] = mag.flatten(1).mean(1)

    feats = torch.stack([margin, mean_pix, std_pix, sobel_mean], 1).cpu().numpy()
    return feats


# ---------- AUROC helper ----------
def uni_auroc(y_bin, x):
    if y_bin.std() == 0:
        return float("nan")
    a = roc_auc_score(y_bin, x)
    return max(a, 1 - a)


# ---------- main ----------
def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("Training CNN on Fashion-MNIST...")
    model = train_model(train_set)

    # materialize entire test set on device
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # restrict to samples correctly classified by clean model
    with torch.no_grad():
        clean_preds = []
        for i in range(0, test_x.size(0), 512):
            clean_preds.append(model(test_x[i:i + 512]).argmax(1))
        clean_preds = torch.cat(clean_preds)
    correct = clean_preds == test_y
    x_c = test_x[correct]; y_c = test_y[correct]
    print(f"Using {correct.sum().item()} correctly classified test samples")

    print("Optimizing 8x8 adversarial patch on training set...")
    patch = train_patch(model, train_set, n_iters=2000, lr=0.05, batch=128)

    print("Evaluating patch over 9-grid on test samples...")
    flipped_any, min_l2 = eval_patch_grid(model, x_c, y_c, patch)
    print(f"  patch flipped_any rate = {flipped_any.float().mean().item():.4f}")
    print(f"  mean min_l2 (among flipped) = "
          f"{min_l2[flipped_any].mean().item() if flipped_any.any() else float('nan'):.4f}")

    print("Computing FGSM min-eps for comparison...")
    fgsm_eps = fgsm_min_eps(model, x_c, y_c)

    print("Computing features (margin, mean_pix, std_pix, sobel_mean)...")
    feats = compute_features(model, x_c, y_c)
    feat_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    # ---------- univariate AUROC vs flipped_any ----------
    y_bin = flipped_any.cpu().numpy().astype(int)
    min_l2_np = min_l2.cpu().numpy()
    fgsm_eps_np = fgsm_eps.cpu().numpy()

    print("\n==== Univariate AUROC vs patch flipped_at_any_position ====")
    print(f" positive rate = {y_bin.mean():.4f}")
    for i, name in enumerate(feat_names):
        a = uni_auroc(y_bin, feats[:, i])
        print(f"  {name:<12} AUROC = {a:.4f}")

    # ---------- Spearman correlation of per-sample rankings ----------
    print("\n==== Per-sample ranking comparison: patch min_l2 vs FGSM min_eps ====")
    # high min_l2 / high fgsm_eps both mean "more robust"
    rho, p = spearmanr(min_l2_np, fgsm_eps_np)
    print(f"  Spearman rho(patch_min_l2, fgsm_min_eps) = {rho:+.4f}  (p={p:.2e})")

    # also Spearman on the binary patch outcome vs fgsm_eps
    rho2, p2 = spearmanr(y_bin, fgsm_eps_np)
    print(f"  Spearman rho(flipped_any, fgsm_min_eps)  = {rho2:+.4f}  (p={p2:.2e})")

    # ---------- features vs fgsm_eps (continuous) ----------
    print("\n==== Pearson correlation of features with FGSM min-eps ====")
    for i, name in enumerate(feat_names):
        cor = np.corrcoef(feats[:, i], fgsm_eps_np)[0, 1]
        print(f"  corr(fgsm_eps, {name:<12}) = {cor:+.4f}")

    # ---------- features vs patch min_l2 (continuous, treating unflipped as robust) ----------
    print("\n==== Pearson correlation of features with patch min_l2 ====")
    for i, name in enumerate(feat_names):
        cor = np.corrcoef(feats[:, i], min_l2_np)[0, 1]
        print(f"  corr(patch_min_l2, {name:<12}) = {cor:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
