"""
H84: Random resize + padding defence (Xie et al., ICLR 2018).

Hypothesis: per-sample recovery rate under random resize+pad defence is
predictable from simple image statistics (margin, mean pixel, std pixel,
sobel mean).

Pipeline:
  1. Train small CNN (same architecture as diagnostic_test.py) on
     Fashion-MNIST for 10 epochs.
  2. Generate FGSM adversarial examples at eps = 15/255 on the test set
     (restricted to samples the clean model classifies correctly).
  3. For each adversarial sample, apply K=16 random (scale, offset)
     transforms:
         - scale s ~ Uniform{24, 25, 26, 27, 28}
         - resize adv image (28x28) to s x s (bilinear)
         - pad to 28x28 at random offset (ox, oy) ~ Uniform[0, 28 - s]
     and take the majority vote over the K transformed predictions.
  4. Per-sample binary target: recovered? = (majority_vote == true_label)
  5. Per-sample features:
         - margin (final-model logit margin on the *adversarial* input)
         - mean_pix
         - std_pix
         - sobel_mean   (mean magnitude of Sobel gradient)
  6. Univariate AUROC for predicting recovery from each feature.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
K = 16
SCALES = [24, 25, 26, 27, 28]   # random scale set
IMG = 28


# --- model: matches diagnostic_test.py --------------------------------------
class CNN(nn.Module):
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


# --- training ---------------------------------------------------------------
def train(seed, train_set):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done")
    model.eval()
    return model


# --- FGSM -------------------------------------------------------------------
def fgsm(model, x, y, eps=EPS):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    sign = x.grad.sign().detach()
    return (x.detach() + eps * sign).clamp(0, 1)


# --- random resize + pad defence -------------------------------------------
def random_resize_pad(x, scale, ox, oy):
    """Resize x (B,1,28,28) to (B,1,scale,scale) bilinear, then pad to 28x28
    at offset (ox, oy). ox, oy are scalars; pad with zeros."""
    B = x.size(0)
    resized = F.interpolate(x, size=(scale, scale), mode="bilinear",
                            align_corners=False)
    out = torch.zeros(B, 1, IMG, IMG, device=x.device, dtype=x.dtype)
    out[:, :, oy:oy + scale, ox:ox + scale] = resized
    return out


def defended_majority_vote(model, x_adv, K=K, seed=0):
    """Apply K random (scale, ox, oy) transforms; majority vote of preds.

    Returns (vote_pred: LongTensor[B], all_preds: LongTensor[K, B]).
    Same K transforms are applied to the whole batch (i.i.d. across the K
    draws); each sample's vote is over those K transformed copies.
    """
    rng = np.random.RandomState(seed)
    B = x_adv.size(0)
    all_preds = torch.zeros(K, B, dtype=torch.long, device=x_adv.device)
    with torch.no_grad():
        for k in range(K):
            s = int(rng.choice(SCALES))
            ox = int(rng.randint(0, IMG - s + 1))
            oy = int(rng.randint(0, IMG - s + 1))
            xt = random_resize_pad(x_adv, s, ox, oy)
            all_preds[k] = model(xt).argmax(1)
    # majority vote per sample
    n_classes = 10
    votes = torch.zeros(B, n_classes, device=x_adv.device)
    for k in range(K):
        votes.scatter_add_(1, all_preds[k].unsqueeze(1),
                           torch.ones(B, 1, device=x_adv.device))
    return votes.argmax(1), all_preds


# --- per-sample features ----------------------------------------------------
SOBEL_X = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1., -2., -1.],
                        [0., 0., 0.],
                        [1., 2., 1.]]).view(1, 1, 3, 3)


def per_sample_features(model, x_adv):
    """margin (on adv), mean_pix, std_pix, sobel_mean."""
    with torch.no_grad():
        logits = model(x_adv)
        sorted_l, _ = logits.sort(1, descending=True)
        margin = (sorted_l[:, 0] - sorted_l[:, 1]).cpu().numpy()
        mean_pix = x_adv.mean(dim=(1, 2, 3)).cpu().numpy()
        std_pix = x_adv.std(dim=(1, 2, 3)).cpu().numpy()
        sx = F.conv2d(x_adv, SOBEL_X.to(x_adv.device), padding=1)
        sy = F.conv2d(x_adv, SOBEL_Y.to(x_adv.device), padding=1)
        mag = torch.sqrt(sx ** 2 + sy ** 2)
        sobel_mean = mag.mean(dim=(1, 2, 3)).cpu().numpy()
    return np.stack([margin, mean_pix, std_pix, sobel_mean], axis=1)


# --- main -------------------------------------------------------------------
def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True,
                                      transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True,
                                     transform=tf)

    print("Training CNN on Fashion-MNIST (10 epochs)...")
    t0 = time.time()
    model = train(0, train_set)
    print(f"  done ({time.time()-t0:.1f}s)")

    # full test tensor
    N = len(test_set)
    test_x = torch.stack([test_set[i][0] for i in range(N)]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(N)]).to(DEVICE)

    # restrict to samples model classifies correctly
    with torch.no_grad():
        clean_pred = []
        for i in range(0, N, 512):
            clean_pred.append(model(test_x[i:i+512]).argmax(1))
        clean_pred = torch.cat(clean_pred)
    correct = clean_pred == test_y
    x_c = test_x[correct]
    y_c = test_y[correct]
    print(f"Using {x_c.size(0)} correctly-classified test samples")

    # Generate FGSM adversarial examples
    print("Generating FGSM adv at eps=15/255 ...")
    adv_list = []
    for i in range(0, x_c.size(0), 256):
        adv_list.append(fgsm(model, x_c[i:i+256], y_c[i:i+256], eps=EPS))
    x_adv = torch.cat(adv_list)

    # Verify attack success (only those flipped are interesting for "recovery")
    with torch.no_grad():
        adv_pred = []
        for i in range(0, x_adv.size(0), 512):
            adv_pred.append(model(x_adv[i:i+512]).argmax(1))
        adv_pred = torch.cat(adv_pred)
    flipped = adv_pred != y_c
    print(f"FGSM success rate (undefended): {flipped.float().mean().item():.3f}")

    # Apply defence: K random (scale, offset) -> majority vote
    print(f"Applying random resize+pad defence (K={K}) ...")
    vote_pred_list = []
    for i in range(0, x_adv.size(0), 512):
        vp, _ = defended_majority_vote(model, x_adv[i:i+512], K=K,
                                       seed=12345 + i)
        vote_pred_list.append(vp)
    vote_pred = torch.cat(vote_pred_list)
    recovered = (vote_pred == y_c)
    print(f"Defence recovery rate (all correctly-clean samples): "
          f"{recovered.float().mean().item():.3f}")
    # recovery rate among those that were actually flipped
    if flipped.any():
        rec_on_flipped = recovered[flipped].float().mean().item()
        print(f"Defence recovery rate (among adv-flipped samples): "
              f"{rec_on_flipped:.3f}")

    # Compute features on the adversarial input
    print("Computing per-sample features ...")
    feats = per_sample_features(model, x_adv)
    feat_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    # ---- Univariate AUROC for predicting recovery -------------------------
    y_rec = recovered.cpu().numpy().astype(int)
    print("\n=== Univariate AUROC: predicting recovery (all correct-clean) ===")
    print(f"  positive rate = {y_rec.mean():.3f}, n = {len(y_rec)}")
    if y_rec.std() > 0:
        for i, n in enumerate(feat_names):
            a = roc_auc_score(y_rec, feats[:, i])
            a = max(a, 1 - a)
            print(f"  {n:<14} AUROC = {a:.4f}")
    else:
        print("  degenerate target (no variance)")

    # Focus: among samples adversarial attack actually flipped
    print("\n=== Univariate AUROC: predicting recovery | adv-flipped ===")
    mask = flipped.cpu().numpy().astype(bool)
    y_rec_f = y_rec[mask]
    print(f"  positive rate = {y_rec_f.mean():.3f}, n = {len(y_rec_f)}")
    if y_rec_f.std() > 0:
        for i, n in enumerate(feat_names):
            a = roc_auc_score(y_rec_f, feats[mask, i])
            a = max(a, 1 - a)
            print(f"  {n:<14} AUROC = {a:.4f}")
    else:
        print("  degenerate target (no variance)")


if __name__ == "__main__":
    main()
