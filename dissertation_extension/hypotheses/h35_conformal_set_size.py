"""
H35: Conformal prediction set size predicts adversarial vulnerability.

Tests three conformal non-conformity scores on a CNN trained on Fashion-MNIST:
  - LAC / THR (Least Ambiguous Classifier): s(x,y) = 1 - softmax_y(x)
  - APS (Adaptive Prediction Sets, Romano et al. 2020):
        sort softmax descending, s(x,y) = cumulative sum up to and including class y
  - RAPS (Regularised APS, Angelopoulos et al. 2021):
        APS score plus a regularisation term penalising large set size
        s(x,y) = sum_{i<=rank(y)} pi_(i) + lambda * max(0, rank(y) - k_reg)

For each test sample we record the per-sample prediction set size at alpha=0.1
and ask: does set size predict whether the sample flips under FGSM/PGD, and
does it add information over the simple softmax margin?

References:
  Romano, Sesia, Candes (2020) "Classification with Valid and Adaptive Coverage"
  Angelopoulos, Bates, Malik, Jordan (2021) "Uncertainty Sets for Image
    Classifiers using Conformal Prediction" (RAPS)

Self-contained. Trains victim from scratch; runs FGSM/PGD via raw PyTorch grads.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

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
OUT_DIR = "/mnt/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h35_out"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

EPOCHS = 10
BATCH = 128
LR = 1e-3

N_CALIB = 5000          # held out from training
N_TEST_EVAL = 2000      # adversarial eval set (smaller, since PGD is costly)
ALPHA = 0.1             # conformal miscoverage

EPS_FGSM = 15.0 / 255.0
EPS_PGD = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = EPS_PGD / 8.0

# RAPS hyperparams (Angelopoulos defaults)
RAPS_LAMBDA = 0.01
RAPS_KREG = 1

# minimum-eps FGSM search grid
MIN_EPS_GRID = np.linspace(0.0, 60.0 / 255.0, 25)


# ------------------------------------------------------------------ model
class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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
    model = SmallCNN().to(DEVICE)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time()
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
        print(f"[train] ep {ep+1}/{EPOCHS} loss={losssum/tot:.4f} acc={correct/tot:.4f} t={time.time()-t0:.1f}s")
    model.eval()
    return model


# ------------------------------------------------------------------ softmax probs
@torch.no_grad()
def get_softmax(model, ds, batch=256):
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        p = F.softmax(model(xb), dim=1).cpu().numpy()
        probs.append(p)
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


# ------------------------------------------------------------------ conformal scores
def lac_scores(probs, labels):
    """LAC / THR non-conformity: 1 - p_y."""
    return 1.0 - probs[np.arange(len(labels)), labels]


def aps_score_for_class(probs_row, y):
    """APS score for a single (probs, y) pair.

    Sort descending; cumulative sum of probs up to and INCLUDING class y.
    Romano et al. use a randomised tiebreak; we use deterministic inclusion
    (slightly conservative on coverage but reproducible).
    """
    order = np.argsort(-probs_row)
    rank = int(np.where(order == y)[0][0])
    return float(probs_row[order[: rank + 1]].sum())


def aps_scores_all(probs, labels):
    out = np.empty(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        out[i] = aps_score_for_class(probs[i], labels[i])
    return out


def raps_score_for_class(probs_row, y, lam=RAPS_LAMBDA, kreg=RAPS_KREG):
    order = np.argsort(-probs_row)
    rank = int(np.where(order == y)[0][0])  # 0-indexed
    cum = float(probs_row[order[: rank + 1]].sum())
    reg = lam * max(0, (rank + 1) - kreg)
    return cum + reg


def raps_scores_all(probs, labels, lam=RAPS_LAMBDA, kreg=RAPS_KREG):
    out = np.empty(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        out[i] = raps_score_for_class(probs[i], labels[i], lam, kreg)
    return out


# ------------------------------------------------------------------ set sizes
def lac_set_sizes(probs, qhat):
    """Set = {y : 1 - p_y <= qhat}  <=>  p_y >= 1 - qhat."""
    return (probs >= (1.0 - qhat)).sum(axis=1)


def aps_set_sizes(probs, qhat):
    """For each row, walk sorted descending until cumulative sum > qhat; include
    the class that crosses the threshold (deterministic inclusion)."""
    n, C = probs.shape
    sizes = np.empty(n, dtype=np.int64)
    for i in range(n):
        order = np.argsort(-probs[i])
        cum = np.cumsum(probs[i][order])
        # smallest k such that cum[k-1] >= qhat; include that class
        k = int(np.searchsorted(cum, qhat, side="left")) + 1
        sizes[i] = min(k, C)
    return sizes


def raps_set_sizes(probs, qhat, lam=RAPS_LAMBDA, kreg=RAPS_KREG):
    n, C = probs.shape
    sizes = np.empty(n, dtype=np.int64)
    for i in range(n):
        order = np.argsort(-probs[i])
        cum = np.cumsum(probs[i][order])
        reg = lam * np.maximum(0, np.arange(1, C + 1) - kreg)
        score_curve = cum + reg  # score if class at rank j is the true one
        # Find smallest j (1..C) with score_curve[j-1] >= qhat; include j.
        k = int(np.searchsorted(score_curve, qhat, side="left")) + 1
        sizes[i] = min(k, C)
    return sizes


# ------------------------------------------------------------------ attacks
def fgsm(model, x, y, eps):
    x = x.clone().detach().to(DEVICE).requires_grad_(True)
    y = y.to(DEVICE)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = (x + eps * grad.sign()).clamp(0.0, 1.0).detach()
    return x_adv


def pgd(model, x, y, eps, alpha, steps):
    x_orig = x.clone().detach().to(DEVICE)
    y = y.to(DEVICE)
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = (x_adv + alpha * grad.sign()).detach()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps).clamp(0.0, 1.0)
    return x_adv


@torch.no_grad()
def predict(model, x):
    return model(x).argmax(1).cpu().numpy()


def compute_flips(model, ds, eps_fgsm, eps_pgd, n_eval):
    """Return arrays of (flipped_fgsm, flipped_pgd, fgsm_min_eps, indices, labels, clean_preds)."""
    idx = np.arange(min(n_eval, len(ds)))
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=128, shuffle=False)
    flipped_fgsm, flipped_pgd, min_eps = [], [], []
    clean_correct, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb_dev = yb.to(DEVICE)
        clean_pred = predict(model, xb)
        all_labels.append(yb.numpy())
        clean_correct.append((clean_pred == yb.numpy()).astype(np.int32))

        x_fgsm = fgsm(model, xb, yb_dev, eps_fgsm)
        p_fgsm = predict(model, x_fgsm)
        flipped_fgsm.append((p_fgsm != clean_pred).astype(np.int32))

        x_pgd = pgd(model, xb, yb_dev, eps_pgd, PGD_ALPHA, PGD_STEPS)
        p_pgd = predict(model, x_pgd)
        flipped_pgd.append((p_pgd != clean_pred).astype(np.int32))

        # min-eps FGSM search
        me = np.full(xb.size(0), np.nan, dtype=np.float64)
        unresolved = np.arange(xb.size(0))
        for eps in MIN_EPS_GRID[1:]:
            if len(unresolved) == 0:
                break
            sub = xb[unresolved]
            sub_y = yb_dev[unresolved]
            sub_clean = clean_pred[unresolved]
            x_adv = fgsm(model, sub, sub_y, eps)
            p = predict(model, x_adv)
            flipped = p != sub_clean
            me[unresolved[flipped]] = eps
            unresolved = unresolved[~flipped]
        # those never flipped: assign max eps + 1 unit so AUROC treats as "very robust"
        me[np.isnan(me)] = MIN_EPS_GRID[-1] + (MIN_EPS_GRID[-1] - MIN_EPS_GRID[-2])
        min_eps.append(me)

    return {
        "flipped_fgsm": np.concatenate(flipped_fgsm),
        "flipped_pgd": np.concatenate(flipped_pgd),
        "fgsm_min_eps": np.concatenate(min_eps),
        "labels": np.concatenate(all_labels),
        "clean_correct": np.concatenate(clean_correct),
        "indices": idx,
    }


# ------------------------------------------------------------------ baselines
def victim_margin(probs):
    """Top1 - Top2 softmax."""
    s = np.sort(probs, axis=1)
    return s[:, -1] - s[:, -2]


def pixel_stats(ds, idx):
    means, stds = [], []
    for i in idx:
        x, _ = ds[int(i)]
        arr = x.numpy().ravel()
        means.append(arr.mean())
        stds.append(arr.std())
    return np.asarray(means), np.asarray(stds)


# ------------------------------------------------------------------ AUROC helpers
def safe_auroc(y, score):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def auroc_continuous(target, score):
    """For continuous targets like fgsm_min_eps, AUROC is ill-defined; we
    binarise at the median so small-eps (vulnerable) is the positive class."""
    med = np.median(target)
    y = (target <= med).astype(np.int32)
    return safe_auroc(y, score)


# ------------------------------------------------------------------ multivariate
def added_value_logreg(margin, conformal_size, y):
    """Does conformal set size add over margin? Compare AUROCs of:
       (a) margin only, (b) margin + set size, via logistic regression."""
    if len(np.unique(y)) < 2:
        return {"auc_margin": float("nan"), "auc_joint": float("nan"), "delta": float("nan")}
    X1 = margin.reshape(-1, 1)
    X2 = np.column_stack([margin, conformal_size])
    sc = StandardScaler()
    X1s = sc.fit_transform(X1)
    X2s = StandardScaler().fit_transform(X2)
    lr1 = LogisticRegression(max_iter=1000).fit(X1s, y)
    lr2 = LogisticRegression(max_iter=1000).fit(X2s, y)
    p1 = lr1.predict_proba(X1s)[:, 1]
    p2 = lr2.predict_proba(X2s)[:, 1]
    a1 = safe_auroc(y, p1)
    a2 = safe_auroc(y, p2)
    return {"auc_margin": a1, "auc_joint": a2, "delta": a2 - a1}


# ------------------------------------------------------------------ main
def main():
    print(f"[h35] device={DEVICE}")
    train_ds, calib_ds, test_ds = get_data()
    print(f"[h35] train={len(train_ds)} calib={len(calib_ds)} test={len(test_ds)}")

    model = train_victim(train_ds)

    # ---- calibration scores
    print("[h35] computing calibration softmax / scores")
    calib_probs, calib_labels = get_softmax(model, calib_ds)
    s_lac = lac_scores(calib_probs, calib_labels)
    s_aps = aps_scores_all(calib_probs, calib_labels)
    s_raps = raps_scores_all(calib_probs, calib_labels)

    n_c = len(calib_labels)
    # conformal quantile level: ceil((n+1)(1-alpha))/n
    q_level = np.ceil((n_c + 1) * (1 - ALPHA)) / n_c
    q_level = min(q_level, 1.0)
    qhat_lac = float(np.quantile(s_lac, q_level, method="higher"))
    qhat_aps = float(np.quantile(s_aps, q_level, method="higher"))
    qhat_raps = float(np.quantile(s_raps, q_level, method="higher"))
    print(f"[h35] qhat LAC={qhat_lac:.4f} APS={qhat_aps:.4f} RAPS={qhat_raps:.4f}")

    # ---- test softmax + set sizes on full test set
    print("[h35] computing test softmax / set sizes")
    test_probs, test_labels = get_softmax(model, test_ds)
    sizes_lac_all = lac_set_sizes(test_probs, qhat_lac)
    sizes_aps_all = aps_set_sizes(test_probs, qhat_aps)
    sizes_raps_all = raps_set_sizes(test_probs, qhat_raps)
    print(f"[h35] mean set size LAC={sizes_lac_all.mean():.3f} "
          f"APS={sizes_aps_all.mean():.3f} RAPS={sizes_raps_all.mean():.3f}")

    # empirical coverage on test (sanity)
    cov_lac = float(np.mean(test_probs[np.arange(len(test_labels)), test_labels] >= (1 - qhat_lac)))
    print(f"[h35] empirical LAC coverage (should be ~{1-ALPHA:.2f}): {cov_lac:.3f}")

    # ---- adversarial flips on first N_TEST_EVAL test samples
    print(f"[h35] running attacks on first {N_TEST_EVAL} test samples")
    atk = compute_flips(model, test_ds, EPS_FGSM, EPS_PGD, N_TEST_EVAL)
    idx = atk["indices"]

    # align features to evaluated samples
    probs_eval = test_probs[idx]
    sizes_lac = sizes_lac_all[idx].astype(np.float64)
    sizes_aps = sizes_aps_all[idx].astype(np.float64)
    sizes_raps = sizes_raps_all[idx].astype(np.float64)
    margin = victim_margin(probs_eval)
    mean_pix, std_pix = pixel_stats(test_ds, idx)

    # vulnerability scores: higher = more vulnerable. Set sizes already
    # higher-when-uncertain. Margin is opposite, so use -margin as "vuln score".
    features = {
        "neg_margin": -margin,
        "mean_pix": mean_pix,
        "std_pix": std_pix,
        "set_size_LAC": sizes_lac,
        "set_size_APS": sizes_aps,
        "set_size_RAPS": sizes_raps,
    }

    # target arrays
    targets = {
        "flipped_FGSM_15_255": atk["flipped_fgsm"],
        "flipped_PGD_15_255": atk["flipped_pgd"],
    }
    # min-eps: small eps => vulnerable, so vuln_score should correlate
    # negatively with min-eps. Use binarised target (vulnerable=1 if min_eps<=median).
    med = float(np.median(atk["fgsm_min_eps"]))
    targets["FGSM_min_eps_below_median"] = (atk["fgsm_min_eps"] <= med).astype(np.int32)

    # ---- univariate AUROC
    results = {"meta": {
        "n_eval": int(N_TEST_EVAL),
        "alpha": ALPHA,
        "qhat_lac": qhat_lac, "qhat_aps": qhat_aps, "qhat_raps": qhat_raps,
        "mean_set_size_lac": float(sizes_lac_all.mean()),
        "mean_set_size_aps": float(sizes_aps_all.mean()),
        "mean_set_size_raps": float(sizes_raps_all.mean()),
        "empirical_coverage_lac": cov_lac,
        "min_eps_median": med,
    }, "univariate_auroc": {}, "multivariate": {}}

    for tname, y in targets.items():
        row = {}
        for fname, score in features.items():
            row[fname] = safe_auroc(y, score)
        results["univariate_auroc"][tname] = row
        # also report fgsm_min_eps as continuous Spearman-ish via binarisation above
    # additional: AUROC against negative min_eps (continuous) — use neg so larger=>more vulnerable
    results["univariate_auroc_continuous_minEps"] = {
        fname: auroc_continuous(atk["fgsm_min_eps"], score)
        for fname, score in features.items()
    }

    # ---- multivariate: does conformal add over margin?
    for tname, y in targets.items():
        sub = {}
        for cname, csize in [("LAC", sizes_lac), ("APS", sizes_aps), ("RAPS", sizes_raps)]:
            sub[cname] = added_value_logreg(-margin, csize, y)
        results["multivariate"][tname] = sub

    # ---- save
    out_path = os.path.join(OUT_DIR, "h35_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[h35] wrote {out_path}")

    # also save raw arrays for downstream inspection
    np.savez(os.path.join(OUT_DIR, "h35_arrays.npz"),
             indices=idx,
             margin=margin,
             mean_pix=mean_pix, std_pix=std_pix,
             set_size_LAC=sizes_lac, set_size_APS=sizes_aps, set_size_RAPS=sizes_raps,
             flipped_FGSM=atk["flipped_fgsm"],
             flipped_PGD=atk["flipped_pgd"],
             fgsm_min_eps=atk["fgsm_min_eps"],
             labels=atk["labels"], clean_correct=atk["clean_correct"])

    print("[h35] DONE")
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
