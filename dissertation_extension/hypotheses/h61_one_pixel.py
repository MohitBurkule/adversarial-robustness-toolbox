"""
H61: One-pixel attack (Su et al. 2019) per-sample vulnerability.

Hypothesis: whether a single-pixel modification can flip the model's prediction
is a meaningful per-sample vulnerability target, and simple per-sample features
(final margin, mean/std of pixel intensities, max input-gradient saliency)
have predictive power for it.

Pipeline:
  1. Train a small CNN (architecture matching diagnostic_test.py) on
     Fashion-MNIST for 10 epochs.
  2. Subsample 500 correctly-classified test points.
  3. For each sample run a simplified one-pixel attack: exhaustively try each
     of the 784 pixels with 5 candidate replacement values in {0, 0.25, 0.5,
     0.75, 1.0}. Record:
        - flipped_1pixel               (binary: did any single-pixel change
                                        flip the prediction?)
        - min_pixel_value_change_to_flip (minimum |new - old| over all
                                          successful triplets; NaN if none)
  4. Compute four features per sample:
        - final_margin   (top1 - top2 logit on clean input)
        - mean_pix       (mean of pixel values)
        - std_pix        (std of pixel values)
        - saliency_max   (max absolute input-gradient of CE loss wrt input)
  5. Report univariate AUROC for the binary target (flipped_1pixel) and
     Spearman correlations for the continuous min_pixel_value_change_to_flip
     target.

Self-contained: writes nothing, just prints results. DO NOT RUN here.
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
N_SUB = 500
CANDIDATE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 0


class CNN(nn.Module):
    """Matches diagnostic_test.py's CNN exactly."""
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


def train_model(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
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
        print(f"  epoch {ep+1}/{EPOCHS} done ({time.time()-t0:.1f}s)")
    model.eval()
    return model


def get_correct_subsample(model, test_set, n_sub):
    """Return n_sub test samples that the model classifies correctly."""
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct = (preds == test_y).nonzero(as_tuple=True)[0]
    rng = np.random.RandomState(SEED)
    idx = rng.choice(correct.cpu().numpy(), size=n_sub, replace=False)
    idx = torch.as_tensor(np.sort(idx), device=DEVICE)
    return test_x[idx], test_y[idx]


def compute_features(model, x, y):
    """final_margin, mean_pix, std_pix, saliency_max."""
    N = x.size(0)
    # final margin (top1 - top2)
    with torch.no_grad():
        logits = model(x)
    sorted_logits, _ = logits.sort(1, descending=True)
    final_margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).cpu().numpy()

    flat = x.view(N, -1)
    mean_pix = flat.mean(1).cpu().numpy()
    std_pix = flat.std(1).cpu().numpy()

    # saliency: max |dL/dx| per sample
    x_req = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_req), y, reduction="sum")
    loss.backward()
    grad = x_req.grad.detach().abs().view(N, -1)
    saliency_max = grad.max(1).values.cpu().numpy()

    return np.stack([final_margin, mean_pix, std_pix, saliency_max], axis=1)


def one_pixel_attack(model, x, y):
    """Exhaustive one-pixel attack over 784 pixels x 5 candidate values.

    Returns:
      flipped (bool):           any single-pixel change flips prediction
      min_change (float):       minimum |new - old| over successful triplets
                                (np.nan if none)
    """
    # x: (1,1,28,28); y: scalar long
    H, W = 28, 28
    P = H * W
    K = len(CANDIDATE_VALUES)
    cand = torch.tensor(CANDIDATE_VALUES, device=DEVICE, dtype=x.dtype)  # (K,)

    flat = x.view(-1).clone()  # (784,)
    # Build all P*K perturbed images as a single batch via expand/scatter.
    # Memory: P*K = 3920 images of size 28*28 ~ 3.07M floats ~ 12 MB.
    base = flat.unsqueeze(0).expand(P * K, -1).clone()  # (P*K, 784)
    pix_idx = torch.arange(P, device=DEVICE).repeat_interleave(K)        # (P*K,)
    val_idx = cand.repeat(P)                                             # (P*K,)
    base[torch.arange(P * K, device=DEVICE), pix_idx] = val_idx

    imgs = base.view(P * K, 1, H, W)
    # forward in chunks to be safe
    preds = []
    with torch.no_grad():
        for i in range(0, imgs.size(0), 1024):
            preds.append(model(imgs[i:i+1024]).argmax(1))
    preds = torch.cat(preds)  # (P*K,)

    flipped_mask = preds != y  # (P*K,)
    if not flipped_mask.any():
        return False, float("nan")

    orig_vals = flat[pix_idx]                       # (P*K,)
    deltas = (val_idx - orig_vals).abs()            # (P*K,)
    # only consider successful flips
    deltas_succ = deltas[flipped_mask]
    return True, float(deltas_succ.min().item())


def main():
    print("##### H61: One-pixel attack (Fashion-MNIST) #####")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(" training small CNN (10 epochs)...")
    t0 = time.time()
    model = train_model(train_set)
    print(f"  trained in {time.time()-t0:.1f}s")

    print(f" subsampling {N_SUB} correctly classified test points...")
    x_sub, y_sub = get_correct_subsample(model, test_set, N_SUB)

    print(" computing features (margin, mean_pix, std_pix, saliency_max)...")
    feats = compute_features(model, x_sub, y_sub)
    feat_names = ["final_margin", "mean_pix", "std_pix", "saliency_max"]

    print(f" running exhaustive one-pixel attack (784 pix x {len(CANDIDATE_VALUES)} vals)...")
    t0 = time.time()
    flipped = np.zeros(N_SUB, dtype=bool)
    min_change = np.full(N_SUB, np.nan, dtype=np.float64)
    for i in range(N_SUB):
        fl, mc = one_pixel_attack(model, x_sub[i:i+1], y_sub[i])
        flipped[i] = fl
        min_change[i] = mc
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{N_SUB}   flipped so far = {flipped[:i+1].sum()}")
    print(f"  done ({time.time()-t0:.1f}s)")

    print(f"\n flip rate (any 1-pixel flip succeeds): {flipped.mean():.3f}")
    print(f" mean min_pixel_value_change_to_flip (over flipped): "
          f"{np.nanmean(min_change):.4f}")

    # --- univariate AUROC on flipped_1pixel ---
    print("\n--- target: flipped_1pixel (binary) ---")
    y_bin = flipped.astype(int)
    if y_bin.std() == 0:
        print("  degenerate target, skipping AUROC")
    else:
        print(f"  positive rate = {y_bin.mean():.3f}")
        for i, n in enumerate(feat_names):
            a = roc_auc_score(y_bin, feats[:, i])
            a = max(a, 1 - a)
            print(f"  univariate AUROC  {n:<16} {a:.4f}")

    # --- continuous target: min_pixel_value_change_to_flip ---
    print("\n--- target: min_pixel_value_change_to_flip (continuous, "
          "over samples with at least one successful flip) ---")
    mask = ~np.isnan(min_change)
    if mask.sum() < 5:
        print("  too few flipped samples for correlation")
    else:
        print(f"  n = {int(mask.sum())}")
        for i, n in enumerate(feat_names):
            rho, p = spearmanr(feats[mask, i], min_change[mask])
            print(f"  Spearman  {n:<16} rho={rho:+.4f}  p={p:.3g}")


if __name__ == "__main__":
    main()
