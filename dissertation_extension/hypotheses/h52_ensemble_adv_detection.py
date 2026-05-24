"""
H52: K-model ensemble disagreement as an adversarial DETECTOR (not vulnerability predictor).

We previously found ensemble disagreement was a *weak* signal for predicting which
clean samples a single victim would later be fooled on (vulnerability prediction).
Detection is a different problem: given a sample at inference time, decide if it
is clean or adversarial. Smith & Gal (2018) (MC-dropout) and Carrara (2017)
(ensemble distances) both suggest disagreement features should be much stronger
for the detection problem.

Pipeline:
    1. Train K=4 small CNN victims on Fashion-MNIST with different seeds (10 ep).
    2. Pick model 0 as the deployed victim. Generate FGSM adv at eps=15/255 on its
       correctly-classified test samples.
    3. Build a 50/50 clean+adv test pool. Score every sample with all 4 models.
    4. Detection features:
         - vote_disagree        : 1 - fraction agreeing with model-0's argmax
         - softmax_variance_max : max over classes of var of softmax across models
         - mean_pairwise_l2     : mean L2 between every pair of softmaxes
         - fs_l1                : Feature Squeezing-style L1 between victim softmax
                                  on x vs victim softmax on bit-depth-reduced x (4 bits)
    5. Targets: is_adversarial in {0,1} (50/50).
    6. Univariate AUROC each. Multivariate logistic-regression AUROC for combos.
"""

import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
K = 4
EPOCHS = 10
BATCH = 256
EPS = 15.0 / 255.0
SEEDS = [0, 1, 2, 3]
RNG = np.random.RandomState(0)


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


def train_one(seed, train_set):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        print(f"  seed={seed} epoch={ep+1}/{EPOCHS}")
    model.eval()
    return model


@torch.no_grad()
def batched_softmax(model, x, bs=512):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(F.softmax(model(x[i:i + bs]), dim=1))
    return torch.cat(out, 0)


def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    g = x.grad.sign().detach()
    return (x.detach() + eps * g).clamp(0, 1)


def bit_depth_reduce(x, bits=4):
    levels = 2 ** bits - 1
    return torch.round(x * levels) / levels


def auroc(y, s):
    return roc_auc_score(y, s)


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    tfm = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tfm)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print(f"[H52] Training K={K} CNNs on Fashion-MNIST ...")
    models = [train_one(s, train_set) for s in SEEDS]
    victim = models[0]

    # Filter to victim-correct clean samples to ensure adv is real
    with torch.no_grad():
        clean_pred = batched_softmax(victim, test_x).argmax(1)
    correct_mask = (clean_pred == test_y)
    correct_idx = torch.where(correct_mask)[0]
    print(f"[H52] Victim correct on {correct_idx.numel()}/{test_x.size(0)} test samples")

    # Generate FGSM adv only on correctly classified
    cx = test_x[correct_idx]
    cy = test_y[correct_idx]
    adv = torch.zeros_like(cx)
    for i in range(0, cx.size(0), 256):
        adv[i:i + 256] = fgsm(victim, cx[i:i + 256], cy[i:i + 256], EPS)

    # Keep only successful adv (victim flipped)
    with torch.no_grad():
        adv_pred = batched_softmax(victim, adv).argmax(1)
    flipped = (adv_pred != cy)
    adv = adv[flipped]
    cy_adv = cy[flipped]
    print(f"[H52] FGSM success: {adv.size(0)}/{cx.size(0)}")

    # Balanced clean pool: pick same count from clean correct
    n = adv.size(0)
    perm = torch.randperm(cx.size(0), device=DEVICE)[:n]
    clean = cx[perm]
    cy_clean = cy[perm]

    X = torch.cat([clean, adv], 0)
    y_true_label = torch.cat([cy_clean, cy_adv], 0)
    is_adv = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)

    # Softmaxes from every model
    print("[H52] Scoring with all models ...")
    softmaxes = torch.stack([batched_softmax(m, X) for m in models], 0)  # (K, N, C)
    preds = softmaxes.argmax(2)  # (K, N)

    # Feature 1: vote disagreement vs victim
    victim_pred = preds[0]
    agree = (preds == victim_pred.unsqueeze(0)).float().mean(0)  # (N,)
    vote_disagree = (1.0 - agree).cpu().numpy()

    # Feature 2: max over classes of variance of softmax across K
    softmax_var_max = softmaxes.var(0).max(dim=1).values.cpu().numpy()

    # Feature 3: mean pairwise L2 between softmaxes
    pair_l2 = []
    for i, j in itertools.combinations(range(K), 2):
        pair_l2.append((softmaxes[i] - softmaxes[j]).pow(2).sum(1).sqrt())
    mean_pairwise_l2 = torch.stack(pair_l2, 0).mean(0).cpu().numpy()

    # Feature 4: FS-style L1 between victim(x) and victim(bit-depth-reduced(x))
    X_bdr = bit_depth_reduce(X, bits=4)
    sm_v = softmaxes[0]
    sm_v_bdr = batched_softmax(victim, X_bdr)
    fs_l1 = (sm_v - sm_v_bdr).abs().sum(1).cpu().numpy()

    feats = {
        "vote_disagree": vote_disagree,
        "softmax_var_max": softmax_var_max,
        "mean_pairwise_l2": mean_pairwise_l2,
        "fs_l1": fs_l1,
    }

    print("\n[H52] === Univariate AUROC (detect adv vs clean) ===")
    uni_aurocs = {}
    for k, v in feats.items():
        a = auroc(is_adv, v)
        uni_aurocs[k] = a
        print(f"  {k:22s} AUROC = {a:.4f}")

    print("\n[H52] === Multivariate logistic regression (5-fold CV AUROC) ===")
    names = list(feats.keys())
    F_mat = np.stack([feats[k] for k in names], 1)

    def cv_auroc(cols):
        Xm = F_mat[:, cols]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        oof = np.zeros(len(is_adv))
        for tr, te in skf.split(Xm, is_adv):
            sc = StandardScaler().fit(Xm[tr])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xm[tr]), is_adv[tr])
            oof[te] = clf.predict_proba(sc.transform(Xm[te]))[:, 1]
        return roc_auc_score(is_adv, oof)

    results = {}
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(range(len(names)), r):
            label = "+".join(names[i] for i in combo)
            results[label] = cv_auroc(list(combo))
            print(f"  {label:60s} AUROC = {results[label]:.4f}")

    best = max(results.items(), key=lambda kv: kv[1])
    print(f"\n[H52] Best combination: {best[0]} (AUROC = {best[1]:.4f})")

    # Compare ensemble-only vs FS-only
    ens_only = cv_auroc([names.index("vote_disagree"),
                         names.index("softmax_var_max"),
                         names.index("mean_pairwise_l2")])
    fs_only = uni_aurocs["fs_l1"]
    print(f"\n[H52] Ensemble-only multivariate AUROC: {ens_only:.4f}")
    print(f"[H52] FS-only univariate AUROC:        {fs_only:.4f}")
    print(f"[H52] Best ensemble-univariate AUROC:  "
          f"{max(uni_aurocs[k] for k in ['vote_disagree','softmax_var_max','mean_pairwise_l2']):.4f}")

    print("\n[H52] HYPOTHESIS VERDICT:")
    best_ens_uni = max(uni_aurocs[k] for k in ["vote_disagree", "softmax_var_max", "mean_pairwise_l2"])
    if best_ens_uni > 0.75:
        print("  SUPPORTED: ensemble disagreement is a strong adversarial DETECTOR "
              "(>0.75 AUROC), unlike its weak performance for vulnerability PREDICTION.")
    elif best_ens_uni > 0.6:
        print("  PARTIALLY SUPPORTED: ensemble disagreement gives moderate detection signal.")
    else:
        print("  NOT SUPPORTED: ensemble disagreement is weak for detection too.")


if __name__ == "__main__":
    main()
