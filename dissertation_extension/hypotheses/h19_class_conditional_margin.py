"""
H19: Class-conditional rank of a sample's margin is a better vulnerability
predictor than raw margin.

Raw victim margin (top1 - top2 logit) conflates:
  (a) class-level effect: some classes are intrinsically easier and tend to have
      higher margins overall.
  (b) sample-level effect: within a given (predicted) class, where does this
      sample sit relative to its peers.

If (b) is what drives vulnerability, then class-conditional transforms of the
margin (within-class percentile rank, or margin minus class mean) should beat
raw margin for predicting flips.

Features (per test sample):
  - raw_margin                 : top1 - top2 logit on victim
  - within_class_rank_margin   : percentile rank of raw_margin among test samples
                                 whose victim-prediction == this sample's true class.
                                 (We rank within the *predicted-class* group so the
                                 group is well-defined regardless of correctness;
                                 we restrict the group definition to samples whose
                                 victim-prediction equals THIS sample's TRUE class
                                 -- giving a "compared to peers I should be similar
                                 to" rank.)
  - margin_minus_class_mean    : raw_margin - mean(raw_margin) within the same
                                 reference group as above.

Targets:
  - flipped_FGSM  (eps = 15/255)
  - flipped_PGD   (eps = 15/255, 10 steps, alpha = eps/4)
  - FGSM_min_eps_binary_search (smallest eps in [0, 0.5] that flips; treated
    as continuous regression target and as binary at threshold median).

Analysis:
  - Univariate AUROC of each feature against each binary target.
  - Spearman correlation against min-eps.
  - Multivariate logistic regression with the three features (standardised);
    report coefficients and AUROC of the joint model with leave-one-out
    ablations.

Self-contained. Run on a CUDA box. Writes JSON results next to this file.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
SEED = 0
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0
BS_LO, BS_HI, BS_ITERS = 0.0, 0.5, 12  # binary search bounds for min-eps FGSM

OUT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "h19_class_conditional_margin_results.json",
)


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


def get_data():
    tfm = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    return train_set, test_set


def train_victim(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = CNN(10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=2)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"[victim] epoch {ep + 1}/{EPOCHS} time={time.time() - t0:.1f}s", flush=True)
    model.eval()
    return model


@torch.no_grad()
def logits_and_pred(model, x, bs=512):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(model(x[i:i + bs]).detach())
    return torch.cat(out, 0)


def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    adv = (x + eps * x.grad.sign()).clamp(0, 1).detach()
    return adv


def pgd(model, x, y, eps, alpha, steps):
    x0 = x.clone().detach()
    adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
    return adv.detach()


def batched_attack(model, X, Y, attack_fn, bs=256):
    """Run attack_fn(model, x_b, y_b) -> adv_b on the whole set, return preds on adv."""
    preds = []
    for i in range(0, X.size(0), bs):
        x_b = X[i:i + bs]
        y_b = Y[i:i + bs]
        adv = attack_fn(model, x_b, y_b)
        with torch.no_grad():
            preds.append(model(adv).argmax(1))
    return torch.cat(preds, 0)


def fgsm_min_eps_bs(model, X, Y, lo=BS_LO, hi=BS_HI, iters=BS_ITERS, bs=256):
    """Per-sample binary search for the smallest eps in [lo, hi] that flips
    the model's prediction (relative to its CLEAN prediction). Samples that
    don't flip at hi get min_eps = hi (capped). Samples that already differ
    from their label even at eps=0 -- we use clean prediction as the reference
    so eps=0 never flips; min-eps reflects perturbation needed away from clean.
    """
    N = X.size(0)
    with torch.no_grad():
        clean_pred = logits_and_pred(model, X).argmax(1)
    lo_t = torch.full((N,), lo, device=DEVICE)
    hi_t = torch.full((N,), hi, device=DEVICE)
    # First check that hi flips; samples that don't flip at hi get hi as their min-eps
    # (we will mark them but still return hi).
    # Binary search: invariant -- lo does NOT flip, hi DOES flip.
    # Initialise lo as "doesn't flip" trivially (eps=0).
    for _ in range(iters):
        mid = (lo_t + hi_t) / 2.0
        # run FGSM at per-sample eps -- need to do per-sample, so loop in chunks
        flipped = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        for i in range(0, N, bs):
            x_b = X[i:i + bs]
            y_b = Y[i:i + bs]
            eps_b = mid[i:i + bs].view(-1, 1, 1, 1)
            x_req = x_b.clone().detach().requires_grad_(True)
            loss = F.cross_entropy(model(x_req), y_b)
            grad = torch.autograd.grad(loss, x_req)[0]
            adv = (x_b + eps_b * grad.sign()).clamp(0, 1).detach()
            with torch.no_grad():
                p = model(adv).argmax(1)
            flipped[i:i + bs] = (p != clean_pred[i:i + bs])
        lo_t = torch.where(flipped, lo_t, mid)
        hi_t = torch.where(flipped, mid, hi_t)
    min_eps = hi_t  # smallest known-flipping eps
    return min_eps.cpu().numpy(), clean_pred.cpu().numpy()


def compute_margin_features(logits_np, victim_pred_np, true_y_np):
    """Return raw_margin, within_class_rank_margin, margin_minus_class_mean.

    Reference group for class-conditional features:
      samples whose victim_pred == this_sample.true_y
    (i.e. compare each sample to the population of samples the model thinks
    belong to that sample's true class).
    """
    # raw margin: top1 - top2 logit
    sorted_logits = np.sort(logits_np, axis=1)
    raw_margin = sorted_logits[:, -1] - sorted_logits[:, -2]

    N = logits_np.shape[0]
    rank_feat = np.zeros(N, dtype=np.float64)
    centered_feat = np.zeros(N, dtype=np.float64)
    for c in range(int(true_y_np.max()) + 1):
        # reference group: victim predicted this class
        ref_idx = np.where(victim_pred_np == c)[0]
        # samples to assign features to: samples whose TRUE class is c
        tgt_idx = np.where(true_y_np == c)[0]
        if len(ref_idx) == 0:
            rank_feat[tgt_idx] = 0.5
            centered_feat[tgt_idx] = raw_margin[tgt_idx]
            continue
        ref_margins = raw_margin[ref_idx]
        ref_mean = ref_margins.mean()
        # for each target sample, compute its percentile within ref group
        # (fraction of ref group with margin < this sample's margin)
        ref_sorted = np.sort(ref_margins)
        tgt_m = raw_margin[tgt_idx]
        # searchsorted gives count of ref < tgt_m; divide by ref size for pct
        pct = np.searchsorted(ref_sorted, tgt_m, side="left") / max(len(ref_sorted), 1)
        rank_feat[tgt_idx] = pct
        centered_feat[tgt_idx] = tgt_m - ref_mean
    return raw_margin, rank_feat, centered_feat


def univariate_auroc(feature, binary_target):
    # Higher feature should mean LESS vulnerable for margin-like quantities,
    # so we score with -feature so AUROC > 0.5 means feature predicts vulnerability
    # when low (the standard reading). We report AUROC of (-feature) -> flip=1.
    try:
        return roc_auc_score(binary_target, -feature)
    except ValueError:
        return float("nan")


def main():
    print(f"Device: {DEVICE}", flush=True)
    train_set, test_set = get_data()

    # Stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)
    print(f"Test set: {N} samples", flush=True)

    model = train_victim(train_set)

    # Clean logits / predictions / margins
    logits = logits_and_pred(model, test_x).cpu().numpy()
    victim_pred = logits.argmax(1)
    true_y_np = test_y.cpu().numpy()
    clean_acc = (victim_pred == true_y_np).mean()
    print(f"Clean test accuracy: {clean_acc:.4f}", flush=True)

    raw_margin, rank_feat, centered_feat = compute_margin_features(
        logits, victim_pred, true_y_np
    )

    # Targets
    print("Running FGSM eps=15/255 ...", flush=True)
    fgsm_pred = batched_attack(
        model, test_x, test_y, lambda m, x, y: fgsm(m, x, y, EPS_TEST)
    ).cpu().numpy()
    flipped_fgsm = (fgsm_pred != true_y_np).astype(np.int32)
    # "Flipped" from the H19 standpoint -- relative to clean victim prediction
    flipped_fgsm_vs_clean = (fgsm_pred != victim_pred).astype(np.int32)

    print("Running PGD eps=15/255 ...", flush=True)
    pgd_pred = batched_attack(
        model, test_x, test_y,
        lambda m, x, y: pgd(m, x, y, EPS_TEST, PGD_ALPHA, PGD_STEPS),
    ).cpu().numpy()
    flipped_pgd = (pgd_pred != true_y_np).astype(np.int32)
    flipped_pgd_vs_clean = (pgd_pred != victim_pred).astype(np.int32)

    print("Running FGSM min-eps binary search ...", flush=True)
    min_eps, _ = fgsm_min_eps_bs(model, test_x, test_y)
    # binary version: median split
    min_eps_bin = (min_eps < np.median(min_eps)).astype(np.int32)

    # ----------------- Analysis -----------------
    feats = {
        "raw_margin": raw_margin,
        "within_class_rank_margin": rank_feat,
        "margin_minus_class_mean": centered_feat,
    }
    targets_bin = {
        "flipped_FGSM": flipped_fgsm,
        "flipped_FGSM_vs_clean": flipped_fgsm_vs_clean,
        "flipped_PGD": flipped_pgd,
        "flipped_PGD_vs_clean": flipped_pgd_vs_clean,
        "FGSM_min_eps_binary": min_eps_bin,
    }

    results = {
        "config": {
            "epochs": EPOCHS,
            "batch": BATCH,
            "seed": SEED,
            "eps_test": EPS_TEST,
            "pgd_steps": PGD_STEPS,
            "pgd_alpha": PGD_ALPHA,
            "bs_iters": BS_ITERS,
            "bs_range": [BS_LO, BS_HI],
            "device": str(DEVICE),
            "clean_acc": float(clean_acc),
            "n_test": int(N),
        },
        "univariate_auroc": {},
        "spearman_vs_min_eps": {},
        "target_base_rates": {k: float(v.mean()) for k, v in targets_bin.items()},
        "min_eps_stats": {
            "mean": float(min_eps.mean()),
            "median": float(np.median(min_eps)),
            "std": float(min_eps.std()),
        },
        "multivariate": {},
    }

    # Univariate AUROC
    for tname, tvec in targets_bin.items():
        results["univariate_auroc"][tname] = {
            fname: float(univariate_auroc(fvec, tvec)) for fname, fvec in feats.items()
        }

    # Spearman vs continuous min-eps (lower min-eps == more vulnerable;
    # higher margin should correlate with HIGHER min-eps)
    for fname, fvec in feats.items():
        rho, p = spearmanr(fvec, min_eps)
        results["spearman_vs_min_eps"][fname] = {"rho": float(rho), "p": float(p)}

    # Multivariate logistic regression (standardised)
    X_full = np.stack(
        [feats["raw_margin"], feats["within_class_rank_margin"], feats["margin_minus_class_mean"]],
        axis=1,
    )
    scaler = StandardScaler()
    X_full_s = scaler.fit_transform(X_full)

    for tname, tvec in targets_bin.items():
        if len(np.unique(tvec)) < 2:
            continue
        entry = {}
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_full_s, tvec)
        entry["full_auroc"] = float(
            roc_auc_score(tvec, clf.predict_proba(X_full_s)[:, 1])
        )
        entry["coefficients"] = {
            "raw_margin": float(clf.coef_[0][0]),
            "within_class_rank_margin": float(clf.coef_[0][1]),
            "margin_minus_class_mean": float(clf.coef_[0][2]),
        }
        # Leave-one-out (single-feature) AUROCs already in univariate; also do
        # leave-one-OUT (two-feature) AUROCs.
        names = ["raw_margin", "within_class_rank_margin", "margin_minus_class_mean"]
        loo = {}
        for i, drop in enumerate(names):
            keep = [j for j in range(3) if j != i]
            clf2 = LogisticRegression(max_iter=2000)
            clf2.fit(X_full_s[:, keep], tvec)
            loo[f"drop_{drop}"] = float(
                roc_auc_score(tvec, clf2.predict_proba(X_full_s[:, keep])[:, 1])
            )
        entry["leave_one_out_auroc"] = loo
        results["multivariate"][tname] = entry

    # Verdict per target: which univariate feature wins?
    winners = {}
    for tname, aurocs in results["univariate_auroc"].items():
        # rank by distance from 0.5 (predictive power in either direction)
        best = max(aurocs.items(), key=lambda kv: abs(kv[1] - 0.5))
        winners[tname] = {"feature": best[0], "auroc": best[1]}
    results["univariate_winner_per_target"] = winners

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
