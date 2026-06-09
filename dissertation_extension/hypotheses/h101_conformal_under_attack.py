"""
H101: Conformal prediction set size UNDER ATTACK predicts post-attack accuracy.

Hypothesis:
  The change in LAC conformal set size between clean and FGSM-perturbed inputs
  ('set_size_delta') is a better predictor of whether a sample's prediction
  flips under FGSM than the clean conformal set size alone.

Pipeline:
  1. Train a small CNN (architecture matched to diagnostic_test.py) on
     Fashion-MNIST for 10 epochs. A 5k-sample calibration split is held out
     from training and used to fit the LAC conformal threshold at alpha=0.1.
  2. Compute LAC conformal sets on the clean test set; record per-sample
     clean_set_size.
  3. Apply FGSM at eps = 15/255 to every test input; recompute LAC sets on
     the adversarial inputs using the SAME qhat (i.e. the clean-calibrated
     threshold), giving adv_set_size and set_size_delta = adv - clean.
  4. Per-sample features: clean_set_size, adv_set_size, set_size_delta,
     softmax margin (clean), mean_pix, std_pix.
  5. Target: adv_flipped (FGSM prediction != clean prediction).
  6. Compare:
       - univariate AUROC of each feature against adv_flipped,
       - multivariate AUROC of (margin + clean_set_size) vs
         (margin + clean_set_size + set_size_delta),
     to test whether set-size GROWTH adds predictive value over clean set
     size + margin.

Self-contained: trains victim from scratch and runs FGSM via raw torch grads.
DO NOT RUN AT WRITE TIME — caller will execute separately.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ config
DATA_ROOT = "/tmp/data"
OUT_DIR = "/mnt/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h101_out"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

EPOCHS = 10
BATCH = 128
LR = 1e-3

N_CALIB = 5000          # held out from training for conformal calibration
ALPHA = 0.1             # conformal miscoverage
EPS_FGSM = 15.0 / 255.0


# ------------------------------------------------------------------ model
# Architecture matches diagnostic_test.py:CNN exactly.
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


# ------------------------------------------------------------------ data
def get_data():
    tfm = transforms.ToTensor()
    train_full = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    test_full = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tfm)

    n = len(train_full)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    calib_idx = idx[:N_CALIB]
    train_idx = idx[N_CALIB:]

    train_ds = Subset(train_full, train_idx.tolist())
    calib_ds = Subset(train_full, calib_idx.tolist())
    return train_ds, calib_ds, test_full


# ------------------------------------------------------------------ train
def train_victim(train_ds):
    model = CNN().to(DEVICE)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        t0 = time.time()
        model.train()
        tot, correct, losssum = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            losssum += loss.item() * xb.size(0)
            tot += xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        print(f"[h101][train] ep {ep+1}/{EPOCHS} "
              f"loss={losssum/tot:.4f} acc={correct/tot:.4f} t={time.time()-t0:.1f}s")
    model.eval()
    return model


# ------------------------------------------------------------------ inference
@torch.no_grad()
def softmax_on_loader(model, ds, batch=256):
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        p = F.softmax(model(xb), dim=1).cpu().numpy()
        probs.append(p)
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


@torch.no_grad()
def softmax_on_tensor(model, x, batch=256):
    probs = []
    for i in range(0, x.size(0), batch):
        p = F.softmax(model(x[i:i+batch]), dim=1).cpu().numpy()
        probs.append(p)
    return np.concatenate(probs)


# ------------------------------------------------------------------ FGSM
def fgsm_batch(model, x, y, eps):
    x = x.clone().detach().to(DEVICE).requires_grad_(True)
    y = y.to(DEVICE)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = (x + eps * grad.sign()).clamp(0.0, 1.0).detach()
    return x_adv


def fgsm_all(model, ds, eps, batch=128):
    """Run FGSM on the entire dataset; return adv tensor on DEVICE plus labels."""
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    advs, ys = [], []
    for xb, yb in loader:
        x_adv = fgsm_batch(model, xb, yb, eps)
        advs.append(x_adv.cpu())
        ys.append(yb)
    return torch.cat(advs, 0), torch.cat(ys, 0)


# ------------------------------------------------------------------ LAC conformal
def lac_scores(probs, labels):
    """LAC non-conformity score: 1 - p_y."""
    return 1.0 - probs[np.arange(len(labels)), labels]


def lac_qhat(scores, alpha):
    """Finite-sample LAC threshold: ceil((n+1)(1-alpha))/n quantile (higher)."""
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def lac_set_sizes(probs, qhat):
    """Set = {y : 1 - p_y <= qhat}  <=>  p_y >= 1 - qhat."""
    return (probs >= (1.0 - qhat)).sum(axis=1)


def victim_margin(probs):
    s = np.sort(probs, axis=1)
    return s[:, -1] - s[:, -2]


def pixel_stats_ds(ds):
    means, stds = [], []
    for i in range(len(ds)):
        x, _ = ds[i]
        arr = x.numpy().ravel()
        means.append(arr.mean())
        stds.append(arr.std())
    return np.asarray(means), np.asarray(stds)


# ------------------------------------------------------------------ stats helpers
def safe_auroc(y, score):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def multivariate_auroc(features, y):
    """Fit standardised logistic regression on the columns of `features`
    (list of 1-D arrays) and return in-sample AUROC."""
    if len(np.unique(y)) < 2:
        return float("nan")
    X = np.column_stack(features)
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=2000).fit(Xs, y)
    p = lr.predict_proba(Xs)[:, 1]
    return safe_auroc(y, p)


# ------------------------------------------------------------------ main
def main():
    print(f"[h101] device={DEVICE}")
    train_ds, calib_ds, test_ds = get_data()
    print(f"[h101] train={len(train_ds)} calib={len(calib_ds)} test={len(test_ds)}")

    model = train_victim(train_ds)

    # ---- calibration: LAC qhat from clean calibration set
    print("[h101] computing calibration scores")
    calib_probs, calib_labels = softmax_on_loader(model, calib_ds)
    s_calib = lac_scores(calib_probs, calib_labels)
    qhat = lac_qhat(s_calib, ALPHA)
    print(f"[h101] LAC qhat (alpha={ALPHA}): {qhat:.4f}")

    # ---- clean test set: probs, set sizes, predictions
    print("[h101] computing clean test softmax / set sizes")
    clean_probs, test_labels = softmax_on_loader(model, test_ds)
    clean_pred = clean_probs.argmax(1)
    clean_set_size = lac_set_sizes(clean_probs, qhat).astype(np.float64)
    margin = victim_margin(clean_probs)

    # sanity: empirical coverage on clean test
    cov = float(np.mean(clean_probs[np.arange(len(test_labels)), test_labels] >= (1.0 - qhat)))
    print(f"[h101] clean mean LAC set size = {clean_set_size.mean():.3f} "
          f"(empirical coverage = {cov:.3f}, target {1-ALPHA:.2f})")

    # ---- FGSM on all test inputs, then recompute LAC sets with SAME qhat
    print(f"[h101] running FGSM (eps={EPS_FGSM:.4f}) on all {len(test_ds)} test inputs")
    adv_x, adv_y = fgsm_all(model, test_ds, EPS_FGSM)
    adv_x = adv_x.to(DEVICE)
    adv_probs = softmax_on_tensor(model, adv_x)
    adv_pred = adv_probs.argmax(1)
    adv_set_size = lac_set_sizes(adv_probs, qhat).astype(np.float64)
    set_size_delta = adv_set_size - clean_set_size

    adv_flipped = (adv_pred != clean_pred).astype(np.int32)
    print(f"[h101] adv mean LAC set size = {adv_set_size.mean():.3f}  "
          f"mean delta = {set_size_delta.mean():.3f}  "
          f"FGSM flip rate = {adv_flipped.mean():.3f}")

    # ---- per-sample features
    mean_pix, std_pix = pixel_stats_ds(test_ds)

    features = {
        "clean_set_size": clean_set_size,
        "adv_set_size": adv_set_size,
        "set_size_delta": set_size_delta,
        "neg_margin": -margin,
        "mean_pix": mean_pix,
        "std_pix": std_pix,
    }

    # ---- univariate AUROC against adv_flipped
    uni = {}
    for fname, score in features.items():
        a = safe_auroc(adv_flipped, score)
        # report direction-corrected AUROC for fair comparison
        a_corr = max(a, 1.0 - a) if not np.isnan(a) else a
        uni[fname] = {"auroc_raw": a, "auroc_dir_corrected": a_corr}
        print(f"[h101] univariate AUROC {fname:<18} "
              f"raw={a:.4f}  dir-corrected={a_corr:.4f}")

    # ---- multivariate: does set_size_delta add over (margin + clean_set_size)?
    print("[h101] multivariate AUROCs")
    auc_clean_only = safe_auroc(adv_flipped, clean_set_size)
    auc_delta_only = safe_auroc(adv_flipped, set_size_delta)
    auc_margin_only = safe_auroc(adv_flipped, -margin)
    auc_m_c = multivariate_auroc([-margin, clean_set_size], adv_flipped)
    auc_m_c_d = multivariate_auroc([-margin, clean_set_size, set_size_delta], adv_flipped)
    auc_m_d = multivariate_auroc([-margin, set_size_delta], adv_flipped)
    auc_c_d = multivariate_auroc([clean_set_size, set_size_delta], adv_flipped)

    multivar = {
        "clean_set_size_only": auc_clean_only,
        "set_size_delta_only": auc_delta_only,
        "neg_margin_only": auc_margin_only,
        "margin+clean": auc_m_c,
        "margin+delta": auc_m_d,
        "clean+delta": auc_c_d,
        "margin+clean+delta": auc_m_c_d,
        "delta_added_over_margin+clean": auc_m_c_d - auc_m_c,
        "delta_added_over_clean": auc_c_d - auc_clean_only,
        "delta_vs_clean_univariate": auc_delta_only - auc_clean_only,
    }
    for k, v in multivar.items():
        print(f"[h101]   {k:<32} {v:+.4f}")

    # ---- bundle results
    results = {
        "meta": {
            "n_train": len(train_ds),
            "n_calib": len(calib_ds),
            "n_test": len(test_ds),
            "epochs": EPOCHS,
            "alpha": ALPHA,
            "eps_fgsm": EPS_FGSM,
            "qhat_lac": qhat,
            "clean_empirical_coverage": cov,
            "mean_clean_set_size": float(clean_set_size.mean()),
            "mean_adv_set_size": float(adv_set_size.mean()),
            "mean_set_size_delta": float(set_size_delta.mean()),
            "fgsm_flip_rate": float(adv_flipped.mean()),
        },
        "univariate_auroc_vs_adv_flipped": uni,
        "multivariate_auroc_vs_adv_flipped": multivar,
    }

    out_path = os.path.join(OUT_DIR, "h101_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[h101] wrote {out_path}")

    np.savez(
        os.path.join(OUT_DIR, "h101_arrays.npz"),
        clean_set_size=clean_set_size,
        adv_set_size=adv_set_size,
        set_size_delta=set_size_delta,
        margin=margin,
        mean_pix=mean_pix,
        std_pix=std_pix,
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        adv_flipped=adv_flipped,
        labels=test_labels,
    )
    print("[h101] DONE")
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
