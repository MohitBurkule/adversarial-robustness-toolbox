"""
H05: Local Binary Patterns (LBP) histogram features predict adversarial vulnerability.

Hypothesis: A classical texture descriptor — the uniform-LBP histogram — captures
per-sample texture complexity that correlates with adversarial vulnerability on
Fashion-MNIST, independent of (and complementary to) the victim margin.

Prior context (dissertation extension benchmark of model-free adversarial vulnerability
predictors on Fashion-MNIST):
  - OTI / JPEG / Fourier features:   AUROC ~ 0.55 - 0.70
  - Simple image statistics:         AUROC ~ 0.70
  - margin alone (victim-conditioned)  is typically the strongest predictor.
The question is whether a richer texture descriptor than Sobel edge-mean (i.e. LBP
histograms) adds predictive value, either univariately or in combination with margin.

Pipeline:
  1. Train CNN victim (model A) on Fashion-MNIST for 10 epochs (Adam, lr=1e-3),
     same arch as diagnostic_test.py.
  2. Train CNN surrogate (model B) with a different seed for FGSM transfer attacks.
  3. Per test sample compute:
       - LBP histogram (P=8, R=1, method='uniform') -> 10-dim feature.
       - LBP entropy and LBP variance (scalar summaries of the LBP code map).
       - Baseline features: victim_margin, mean_pix, std_pix, sobel_mean.
  4. Compute targets per sample (restricted to victim-correct samples):
       - flipped_FGSM     (model A, eps = 15/255)
       - flipped_PGD      (model A, eps = 15/255, 10 iters, step 2/255)
       - FGSM_transfer    (perturbation crafted on model B, evaluated on model A)
  5. Univariate per-feature AUROC vs each binary target.
  6. Multivariate logistic regression: how does {margin + LBP histogram + LBP scalars}
     compare to {margin alone}, and does the LBP block add Delta-AUROC?

Tools: PyTorch (CUDA) + scikit-image. Run with the venv at
  /mnt/1tbone/trashy/adversarial-robustness-toolbox/.venv
Data cache: /tmp/data.

Self-contained: run with `python h05_lbp.py`. Prints all results to stdout.
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from skimage.feature import local_binary_pattern
from scipy.ndimage import sobel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# -------------------- config --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
PGD_ITERS = 10
PGD_STEP = 2.0 / 255.0
LBP_P = 8
LBP_R = 1
LBP_N_BINS = LBP_P + 2          # 10 bins for uniform LBP with P=8
SEED_A = 0
SEED_B = 1


# -------------------- model --------------------
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


def train_model(seed, train_set):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time()
        total = 0.0
        n = 0
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        print(f"   seed={seed} epoch {ep+1}/{EPOCHS}  loss={total/n:.4f}  "
              f"({time.time()-t0:.1f}s)")
    model.eval()
    return model


# -------------------- attacks --------------------
def fgsm_grad_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flipped(model, x, y, eps=EPS):
    sign = fgsm_grad_sign(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flipped(model, x, y, eps=EPS, step=PGD_STEP, iters=PGD_ITERS):
    x_orig = x.clone().detach()
    # random start within eps ball
    delta = (torch.rand_like(x) * 2 - 1) * eps
    adv = (x_orig + delta).clamp(0, 1).detach()
    for _ in range(iters):
        adv = adv.detach().requires_grad_(True)
        F.cross_entropy(model(adv), y).backward()
        with torch.no_grad():
            adv = adv + step * adv.grad.sign()
            adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps)
            adv = adv.clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def transfer_flipped(target, surrogate, x, y, eps=EPS):
    sign = fgsm_grad_sign(surrogate, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (target(adv).argmax(1) != y)


def batched(fn, x, y, bs=512, **kw):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(fn(x[i:i+bs], y[i:i+bs], **kw))
    return torch.cat(out)


# -------------------- features --------------------
def compute_lbp_features(images_np):
    """
    images_np: (N, H, W) float32 in [0, 1].
    Returns:
       hist:    (N, LBP_N_BINS)  normalized uniform-LBP histogram
       entropy: (N,)
       var:     (N,)
    """
    N = images_np.shape[0]
    hist = np.zeros((N, LBP_N_BINS), dtype=np.float32)
    ent = np.zeros(N, dtype=np.float32)
    var = np.zeros(N, dtype=np.float32)
    bins = np.arange(LBP_N_BINS + 1) - 0.5    # bin edges: -0.5, 0.5, ..., P+1.5
    for i in range(N):
        img = images_np[i]
        lbp = local_binary_pattern(img, P=LBP_P, R=LBP_R, method='uniform')
        h, _ = np.histogram(lbp.ravel(), bins=bins, density=False)
        h = h.astype(np.float32)
        s = h.sum()
        if s > 0:
            h /= s
        hist[i] = h
        # entropy of normalized hist
        nz = h[h > 0]
        ent[i] = float(-(nz * np.log(nz)).sum()) if nz.size else 0.0
        var[i] = float(lbp.var())
    return hist, ent, var


def compute_baseline_features(images_np, model, x_t, y_t):
    """
    mean_pix, std_pix, sobel_mean (per sample), and victim_margin (final-model).
    """
    N = images_np.shape[0]
    mean_pix = images_np.reshape(N, -1).mean(axis=1)
    std_pix = images_np.reshape(N, -1).std(axis=1)
    sobel_mean = np.zeros(N, dtype=np.float32)
    for i in range(N):
        gx = sobel(images_np[i], axis=0)
        gy = sobel(images_np[i], axis=1)
        sobel_mean[i] = float(np.sqrt(gx * gx + gy * gy).mean())

    # victim margin
    margins = []
    model.eval()
    with torch.no_grad():
        for i in range(0, x_t.size(0), 512):
            logits = model(x_t[i:i+512])
            s, _ = logits.sort(1, descending=True)
            margins.append((s[:, 0] - s[:, 1]).cpu().numpy())
    margin = np.concatenate(margins).astype(np.float32)
    return margin, mean_pix.astype(np.float32), std_pix.astype(np.float32), sobel_mean


# -------------------- evaluation --------------------
def univariate_auroc(X, y, names):
    print("   univariate AUROC (auto-oriented = max(a, 1-a)):")
    for i, n in enumerate(names):
        col = X[:, i]
        if np.std(col) == 0:
            print(f"     {n:<22} (constant - skipped)")
            continue
        a = roc_auc_score(y, col)
        a = max(a, 1 - a)
        print(f"     {n:<22} {a:.4f}")


def multivariate_auroc(X, y, label):
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=5000).fit(Xs, y)
    auc = roc_auc_score(y, lr.predict_proba(Xs)[:, 1])
    print(f"   multivariate AUROC ({label}, d={X.shape[1]}): {auc:.4f}")
    return auc


def evaluate(target_name, y_bin, feat_blocks):
    """
    feat_blocks: dict label -> (X, names)
    Reports univariate + multivariate AUROC for each block, plus comparisons.
    """
    print(f"\n========== target: {target_name}  (pos rate = {y_bin.mean():.3f}, "
          f"n = {len(y_bin)}) ==========")
    if y_bin.std() == 0:
        print("   degenerate target - skipping")
        return {}

    # univariate over ALL columns in any block (de-duped by name)
    seen = set()
    all_cols = []
    all_names = []
    for X, names in feat_blocks.values():
        for j, n in enumerate(names):
            if n in seen:
                continue
            seen.add(n)
            all_cols.append(X[:, j])
            all_names.append(n)
    X_all = np.stack(all_cols, axis=1)
    univariate_auroc(X_all, y_bin, all_names)

    print("   ---- multivariate models ----")
    aucs = {}
    for label, (X, _) in feat_blocks.items():
        aucs[label] = multivariate_auroc(X, y_bin, label)

    # explicit deltas of interest
    if "margin" in aucs and "margin + LBP_hist" in aucs:
        d = aucs["margin + LBP_hist"] - aucs["margin"]
        print(f"   Delta AUROC: (margin + LBP_hist) - margin       = {d:+.4f}")
    if "margin" in aucs and "margin + LBP_hist + LBP_scalars" in aucs:
        d = aucs["margin + LBP_hist + LBP_scalars"] - aucs["margin"]
        print(f"   Delta AUROC: (margin + LBP all) - margin        = {d:+.4f}")
    if "baseline" in aucs and "baseline + LBP_hist + LBP_scalars" in aucs:
        d = aucs["baseline + LBP_hist + LBP_scalars"] - aucs["baseline"]
        print(f"   Delta AUROC: (baseline + LBP all) - baseline    = {d:+.4f}")
    return aucs


def main():
    print(f"device = {DEVICE}")
    os.makedirs(DATA_DIR, exist_ok=True)
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)

    # materialise test tensor
    x_test = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n[1/4] training victim model A ...")
    t0 = time.time()
    model_A = train_model(SEED_A, train_set)
    print(f"   done in {time.time()-t0:.1f}s")

    print("\n[2/4] training surrogate model B (for FGSM transfer) ...")
    t0 = time.time()
    model_B = train_model(SEED_B, train_set)
    print(f"   done in {time.time()-t0:.1f}s")

    # victim-correct mask
    with torch.no_grad():
        preds_A = []
        for i in range(0, x_test.size(0), 512):
            preds_A.append(model_A(x_test[i:i+512]).argmax(1))
        preds_A = torch.cat(preds_A)
    correct = (preds_A == y_test)
    print(f"\n   victim A test accuracy = {correct.float().mean().item():.4f}  "
          f"({int(correct.sum())} / {len(y_test)})")

    x_c = x_test[correct]
    y_c = y_test[correct]
    images_np = x_c.squeeze(1).cpu().numpy().astype(np.float32)

    # ---------------- features ----------------
    print("\n[3/4] computing features ...")
    t0 = time.time()
    margin, mean_pix, std_pix, sobel_mean = compute_baseline_features(
        images_np, model_A, x_c, y_c)
    print(f"   baseline features: {time.time()-t0:.1f}s")
    t0 = time.time()
    lbp_hist, lbp_ent, lbp_var = compute_lbp_features(images_np)
    print(f"   LBP features:      {time.time()-t0:.1f}s")

    # ---------------- targets ----------------
    print("\n[4/4] computing attack-success targets ...")
    t0 = time.time()
    flipped_fgsm = batched(lambda x, y: fgsm_flipped(model_A, x, y),
                           x_c, y_c).cpu().numpy().astype(int)
    print(f"   FGSM self:     {time.time()-t0:.1f}s  pos rate={flipped_fgsm.mean():.3f}")
    t0 = time.time()
    flipped_pgd = batched(lambda x, y: pgd_flipped(model_A, x, y),
                          x_c, y_c).cpu().numpy().astype(int)
    print(f"   PGD  self:     {time.time()-t0:.1f}s  pos rate={flipped_pgd.mean():.3f}")
    t0 = time.time()
    flipped_xfer = batched(lambda x, y: transfer_flipped(model_A, model_B, x, y),
                           x_c, y_c).cpu().numpy().astype(int)
    print(f"   FGSM transfer: {time.time()-t0:.1f}s  pos rate={flipped_xfer.mean():.3f}")

    # ---------------- feature blocks ----------------
    lbp_hist_names = [f"lbp_h{i}" for i in range(LBP_N_BINS)]
    base_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]
    scalar_lbp_names = ["lbp_entropy", "lbp_var"]

    X_margin = margin.reshape(-1, 1)
    X_base = np.stack([margin, mean_pix, std_pix, sobel_mean], axis=1)
    X_lbp_hist = lbp_hist                                  # (N, 10)
    X_lbp_scalars = np.stack([lbp_ent, lbp_var], axis=1)
    X_margin_hist = np.concatenate([X_margin, X_lbp_hist], axis=1)
    X_margin_all = np.concatenate([X_margin, X_lbp_hist, X_lbp_scalars], axis=1)
    X_base_all = np.concatenate([X_base, X_lbp_hist, X_lbp_scalars], axis=1)
    X_lbp_only = np.concatenate([X_lbp_hist, X_lbp_scalars], axis=1)

    blocks = {
        "margin":                              (X_margin,        ["margin"]),
        "baseline":                            (X_base,          base_names),
        "LBP_hist_only":                       (X_lbp_hist,      lbp_hist_names),
        "LBP_hist + LBP_scalars":              (X_lbp_only,      lbp_hist_names + scalar_lbp_names),
        "margin + LBP_hist":                   (X_margin_hist,   ["margin"] + lbp_hist_names),
        "margin + LBP_hist + LBP_scalars":     (X_margin_all,    ["margin"] + lbp_hist_names + scalar_lbp_names),
        "baseline + LBP_hist + LBP_scalars":   (X_base_all,      base_names + lbp_hist_names + scalar_lbp_names),
    }

    summary = {}
    for tname, y_bin in [("flipped_FGSM", flipped_fgsm),
                         ("flipped_PGD", flipped_pgd),
                         ("FGSM_transfer", flipped_xfer)]:
        aucs = evaluate(tname, y_bin, blocks)
        summary[tname] = aucs

    print("\n===== SUMMARY (multivariate AUROC by target/block) =====")
    cols = list(blocks.keys())
    head = "target".ljust(18) + "  " + "  ".join(c[:32].ljust(32) for c in cols)
    print(head)
    for tname, aucs in summary.items():
        row = tname.ljust(18) + "  " + "  ".join(
            f"{aucs.get(c, float('nan')):.4f}".ljust(32) for c in cols)
        print(row)

    # also dump JSON for downstream collation
    out_json = "/tmp/data/h05_lbp_summary.json"
    try:
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"\nwrote summary -> {out_json}")
    except Exception as e:
        print(f"could not write summary json: {e}")


if __name__ == "__main__":
    main()
