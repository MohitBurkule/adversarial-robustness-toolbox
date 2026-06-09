"""
H74: PCA reconstruction error on pixel space predicts adversarial vulnerability.

Hypothesis: Atypical pixel patterns (those poorly reconstructed by a low-rank PCA
fit on training pixels) are more attackable. We fit a sklearn PCA with K=50
components on Fashion-MNIST training pixels, project test samples into the
subspace, reconstruct them, and measure per-sample L2 reconstruction error.

Pipeline:
  1. Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
  2. Fit sklearn PCA (K=50) on flattened training pixels.
  3. For each test sample compute:
        - margin       (final-model top1 - top2 logit)
        - mean_pix     (mean pixel intensity)
        - std_pix      (std of pixel intensity)
        - pca_recon_error  (L2 distance between sample and its PCA reconstruction)
  4. Compute vulnerability targets per (correctly-classified) test sample:
        - flipped_by_FGSM   (FGSM at eps=15/255)
        - flipped_by_PGD    (PGD at eps=15/255, 10 steps)
        - min_eps_FGSM      (binary search for smallest L_inf eps that flips)
  5. Report univariate AUROC of each feature against the binary targets, and
     Pearson correlation against min_eps.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
PCA_K = 50
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0


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


def train_model(seed, train_set):
    torch.manual_seed(seed); np.random.seed(seed)
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


def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x_orig = x.clone().detach()
    adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.min(torch.max(adv, x_orig - eps), x_orig + eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
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


def compute_margin(model, x):
    with torch.no_grad():
        logits = []
        for i in range(0, x.size(0), 512):
            logits.append(model(x[i:i+512]))
        logits = torch.cat(logits, 0)
    sorted_logits, _ = logits.sort(1, descending=True)
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]
    pred = logits.argmax(1)
    return margin, pred


def main():
    print("##### H74: PCA reconstruction error vs adversarial vulnerability #####")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(" training CNN on Fashion-MNIST (10 epochs)...")
    model = train_model(0, train_set)

    # gather test tensors
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # ---- fit PCA on training pixels ----
    print(f" fitting sklearn PCA with K={PCA_K} on training pixels ...")
    train_flat = np.stack([train_set[i][0].numpy().reshape(-1)
                           for i in range(len(train_set))], 0).astype(np.float32)
    t0 = time.time()
    pca = PCA(n_components=PCA_K, svd_solver="randomized", random_state=0)
    pca.fit(train_flat)
    print(f"  done ({time.time()-t0:.1f}s)  explained_variance_ratio_sum="
          f"{pca.explained_variance_ratio_.sum():.4f}")

    # ---- test features ----
    test_flat = test_x.detach().cpu().numpy().reshape(test_x.size(0), -1).astype(np.float32)
    z = pca.transform(test_flat)
    recon = pca.inverse_transform(z)
    pca_recon_err = np.linalg.norm(test_flat - recon, axis=1)

    mean_pix = test_flat.mean(axis=1)
    std_pix = test_flat.std(axis=1)

    margin, pred = compute_margin(model, test_x)
    margin_np = margin.detach().cpu().numpy()

    # restrict analysis to correctly classified samples
    correct = (pred == test_y)
    idx = correct.nonzero(as_tuple=True)[0]
    print(f" using {idx.numel()} correctly classified samples")
    x_c = test_x[idx]; y_c = test_y[idx]
    feats = np.stack([
        margin_np[idx.cpu().numpy()],
        mean_pix[idx.cpu().numpy()],
        std_pix[idx.cpu().numpy()],
        pca_recon_err[idx.cpu().numpy()],
    ], axis=1)
    feat_names = ["margin", "mean_pix", "std_pix", "pca_recon_error"]

    # ---- compute targets ----
    print(" computing FGSM attack success ...")
    out = []
    for i in range(0, x_c.size(0), 512):
        out.append(fgsm_attack(model, x_c[i:i+512], y_c[i:i+512]))
    fgsm_flip = torch.cat(out).cpu().numpy().astype(int)

    print(" computing PGD attack success ...")
    out = []
    for i in range(0, x_c.size(0), 512):
        out.append(pgd_attack(model, x_c[i:i+512], y_c[i:i+512]))
    pgd_flip = torch.cat(out).cpu().numpy().astype(int)

    print(" computing min_eps_to_flip (FGSM binary search) ...")
    out = []
    for i in range(0, x_c.size(0), 512):
        out.append(min_eps_to_flip(model, x_c[i:i+512], y_c[i:i+512]))
    min_eps = torch.cat(out).cpu().numpy()

    targets_bin = [("FGSM_flip", fgsm_flip), ("PGD_flip", pgd_flip)]

    # ---- univariate AUROC on binary targets ----
    print("\n===== Univariate AUROC (binary targets) =====")
    for t_name, y_t in targets_bin:
        if y_t.std() == 0:
            print(f" target {t_name}: degenerate (pos rate {y_t.mean():.3f})")
            continue
        print(f"\n  target = {t_name}  (positive rate = {y_t.mean():.3f})")
        for i, n in enumerate(feat_names):
            a = roc_auc_score(y_t, feats[:, i])
            a = max(a, 1 - a)
            print(f"    {n:<18} AUROC = {a:.4f}")

    # ---- correlation with min_eps (continuous) =====
    print("\n===== Correlation with min_eps_FGSM =====")
    print(f"  mean min_eps = {min_eps.mean():.4f}, std = {min_eps.std():.4f}")
    for i, n in enumerate(feat_names):
        c = np.corrcoef(feats[:, i], min_eps)[0, 1]
        print(f"    corr(min_eps, {n:<18}) = {c:+.4f}")

    # also AUROC of features against "min_eps below median" as a binary
    med = np.median(min_eps)
    vulnerable = (min_eps <= med).astype(int)
    print(f"\n  target = min_eps <= median ({med:.4f})  (positive rate = {vulnerable.mean():.3f})")
    for i, n in enumerate(feat_names):
        a = roc_auc_score(vulnerable, feats[:, i])
        a = max(a, 1 - a)
        print(f"    {n:<18} AUROC = {a:.4f}")


if __name__ == "__main__":
    main()
