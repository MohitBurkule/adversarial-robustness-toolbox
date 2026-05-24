"""
Hypothesis H49: Spectral signature (Tran, Li, Madry — NeurIPS 2018) as an
adversarial-vulnerability predictor for *clean* test samples.

Tran et al. ("Spectral Signatures in Backdoor Attacks", NeurIPS 2018,
arXiv:1811.00636) showed that the top singular vector of the *class-mean-centered*
penultimate-feature covariance separates backdoored from clean samples: a sample
with a high |<f(x) - mu_c, v_c>| score is anomalous within its class.

Here we ask the analogous *clean-data* question: are per-class spectral outliers
(samples with large projection onto the top singular vector of their class's
centered features) systematically more *adversarially fragile* than typical
in-class samples? And does the score add information over the model margin?

Pipeline:
  1. Train a small CNN on Fashion-MNIST for 10 epochs (architecture matches
     diagnostic_test.py).
  2. For each class c, collect penultimate features of training samples whose
     label is c; center on the per-class mean mu_c, run numpy SVD to get the
     top-1 right singular vector v_c.
  3. Per correctly classified test sample (true class c): spectral score =
     |<f(x) - mu_c, v_c>|.
  4. Other features: victim_margin (top1-top2 logit), mean_pix, std_pix.
  5. Vulnerability targets:
        - flipped_FGSM      at eps = 15/255
        - flipped_PGD       at eps = 15/255 (40 iters)
        - FGSM_min_eps      per-sample binary search
  6. Univariate AUROC of every feature against every binary target;
     correlation with the continuous target. Multivariate AUROC:
     (margin) vs (margin + spectral_signature_score) to test additive value.

Self-contained; data cached at /tmp/data.  Code only — DO NOT execute.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_ITERS = 40
PGD_ALPHA = 2.0 / 255.0
N_CLASSES = 10
SEED = 0


# ---------------------------------------------------------------------------
# Model — matches diagnostic_test.py. Penultimate features = post-relu fc1 out.
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def features(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        h = self.features(x)
        return self.fc2(self.do2(h))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_victim(train_set):
    torch.manual_seed(SEED); np.random.seed(SEED)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        print(f"  epoch {ep+1:2d}/{EPOCHS}  loss={loss_sum/total:.4f}  "
              f"acc={correct/total:.4f}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, x, batch=512):
    feats = []
    for i in range(0, x.size(0), batch):
        feats.append(model.features(x[i:i+batch]).cpu())
    return torch.cat(feats, 0).numpy()


@torch.no_grad()
def extract_logits(model, x, batch=512):
    out = []
    for i in range(0, x.size(0), batch):
        out.append(model(x[i:i+batch]).cpu())
    return torch.cat(out, 0).numpy()


# ---------------------------------------------------------------------------
# Spectral signature: per-class top right singular vector of centered features.
# Following Tran et al. (NeurIPS 2018): for each class c, form matrix
#   M_c = F_c - mu_c   (rows = samples)
# Compute thin SVD; take v_c = first right singular vector (length d).
# Score(x) = |<f(x) - mu_c, v_c>|.
# ---------------------------------------------------------------------------
def compute_class_spectra(train_feats, train_labels):
    d = train_feats.shape[1]
    mus = np.zeros((N_CLASSES, d), dtype=np.float64)
    vs = np.zeros((N_CLASSES, d), dtype=np.float64)
    for c in range(N_CLASSES):
        mask = train_labels == c
        F_c = train_feats[mask].astype(np.float64)
        mu_c = F_c.mean(0)
        M_c = F_c - mu_c
        # economy SVD: M_c = U S Vt; v_c = Vt[0]
        _, _, Vt = np.linalg.svd(M_c, full_matrices=False)
        v_c = Vt[0]
        mus[c] = mu_c
        vs[c] = v_c
        print(f"  class {c}: n={mask.sum()}  feat_dim={d}  "
              f"top_sv_norm={np.linalg.norm(v_c):.3f}")
    return mus, vs


def spectral_score(feats, labels, mus, vs):
    out = np.zeros(feats.shape[0], dtype=np.float64)
    for c in range(N_CLASSES):
        mask = labels == c
        if not mask.any():
            continue
        centered = feats[mask].astype(np.float64) - mus[c]
        out[mask] = np.abs(centered @ vs[c])
    return out


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def fgsm_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS_TEST):
    s = fgsm_sign(model, x, y)
    return (x + eps * s).clamp(0, 1)


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, iters=PGD_ITERS):
    x_orig = x.clone().detach()
    delta = (torch.rand_like(x) * 2 - 1) * eps
    adv = (x_orig + delta).clamp(0, 1).detach()
    for _ in range(iters):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    return adv.detach()


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_sign(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def batched_attack_flip(model, attack_fn, x, y, batch=256):
    out = []
    for i in range(0, x.size(0), batch):
        adv = attack_fn(model, x[i:i+batch], y[i:i+batch])
        with torch.no_grad():
            flipped = (model(adv).argmax(1) != y[i:i+batch])
        out.append(flipped.cpu())
    return torch.cat(out).numpy().astype(int)


def batched_min_eps(model, x, y, batch=256):
    out = []
    for i in range(0, x.size(0), batch):
        out.append(min_eps_fgsm(model, x[i:i+batch], y[i:i+batch]).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------
def uni_auroc(score, y_bin):
    if y_bin.std() == 0:
        return float("nan")
    a = roc_auc_score(y_bin, score)
    return max(a, 1 - a)


def multivariate_auroc(X, y):
    if y.std() == 0:
        return float("nan")
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000).fit(Xs, y)
    return roc_auc_score(y, clf.predict_proba(Xs)[:, 1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"device = {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True,  download=True, transform=tf)
    test_set  = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    # tensors
    train_x = torch.stack([train_set[i][0] for i in range(len(train_set))])
    train_y = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    test_x  = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y  = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n[1] training victim CNN ...")
    model = train_victim(train_set)

    print("\n[2] extracting penultimate features (train) ...")
    train_x_dev = train_x.to(DEVICE)
    train_feats = extract_features(model, train_x_dev)
    train_labels_np = train_y.numpy()
    del train_x_dev

    print("\n[3] per-class SVD ...")
    mus, vs = compute_class_spectra(train_feats, train_labels_np)

    print("\n[4] extracting penultimate features (test) ...")
    test_feats = extract_features(model, test_x)
    test_logits = extract_logits(model, test_x)
    test_y_np = test_y.cpu().numpy()
    test_pred = test_logits.argmax(1)

    correct_mask = test_pred == test_y_np
    print(f"  test acc = {correct_mask.mean():.4f}")

    # features for correctly classified samples
    x_c = test_x[correct_mask]
    y_c = test_y[correct_mask]
    feats_c = test_feats[correct_mask]
    labels_c = test_y_np[correct_mask]
    logits_c = test_logits[correct_mask]

    print("\n[5] computing per-sample features ...")
    spec = spectral_score(feats_c, labels_c, mus, vs)
    sorted_logits = np.sort(logits_c, axis=1)
    margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    pix = x_c.cpu().numpy().reshape(x_c.size(0), -1)
    mean_pix = pix.mean(1)
    std_pix = pix.std(1)

    print("\n[6] computing adversarial targets ...")
    t0 = time.time()
    flipped_fgsm = batched_attack_flip(model, fgsm_attack, x_c, y_c)
    print(f"  FGSM flip rate    = {flipped_fgsm.mean():.4f}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    flipped_pgd  = batched_attack_flip(model, pgd_attack, x_c, y_c)
    print(f"  PGD flip rate     = {flipped_pgd.mean():.4f}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    min_eps = batched_min_eps(model, x_c, y_c)
    print(f"  FGSM_min_eps mean = {min_eps.mean():.4f}  ({time.time()-t0:.1f}s)")

    feat_names = ["spectral_signature_score", "victim_margin", "mean_pix", "std_pix"]
    F_mat = np.stack([spec, margin, mean_pix, std_pix], axis=1)

    # ----- univariate AUROC -----
    print("\n========== UNIVARIATE AUROC ==========")
    targets_bin = {
        "flipped_FGSM_eps15": flipped_fgsm,
        "flipped_PGD_eps15":  flipped_pgd,
        "low_min_eps_q50":    (min_eps <= np.median(min_eps)).astype(int),
    }
    for tname, y_bin in targets_bin.items():
        print(f"\n--- target: {tname}  (pos rate = {y_bin.mean():.3f}) ---")
        for i, n in enumerate(feat_names):
            print(f"  {n:<28} AUROC = {uni_auroc(F_mat[:, i], y_bin):.4f}")

    # ----- continuous correlations with min_eps -----
    print("\n========== Spearman/Pearson with FGSM_min_eps ==========")
    for i, n in enumerate(feat_names):
        pear = np.corrcoef(F_mat[:, i], min_eps)[0, 1]
        # Spearman via rank
        rx = np.argsort(np.argsort(F_mat[:, i]))
        ry = np.argsort(np.argsort(min_eps))
        spear = np.corrcoef(rx, ry)[0, 1]
        print(f"  {n:<28} pearson={pear:+.4f}  spearman={spear:+.4f}")

    # ----- multivariate: does spectral add over margin? -----
    print("\n========== MULTIVARIATE: spectral additive over margin ==========")
    idx_spec = feat_names.index("spectral_signature_score")
    idx_marg = feat_names.index("victim_margin")
    for tname, y_bin in targets_bin.items():
        print(f"\n--- target: {tname} ---")
        a_margin       = multivariate_auroc(F_mat[:, [idx_marg]], y_bin)
        a_spec         = multivariate_auroc(F_mat[:, [idx_spec]], y_bin)
        a_both         = multivariate_auroc(F_mat[:, [idx_marg, idx_spec]], y_bin)
        a_all          = multivariate_auroc(F_mat, y_bin)
        a_all_no_spec  = multivariate_auroc(
            np.delete(F_mat, idx_spec, axis=1), y_bin)
        print(f"  margin only             AUROC = {a_margin:.4f}")
        print(f"  spectral only           AUROC = {a_spec:.4f}")
        print(f"  margin + spectral       AUROC = {a_both:.4f}")
        print(f"  all four features       AUROC = {a_all:.4f}")
        print(f"  all-minus-spectral      AUROC = {a_all_no_spec:.4f}")
        print(f"  delta (spectral adds)         = {a_all - a_all_no_spec:+.4f}")

    # ----- conditional-on-margin-quartile analysis -----
    print("\n========== conditional on margin quartile ==========")
    q = np.quantile(margin, [0.0, 0.25, 0.5, 0.75, 1.0])
    for qi in range(4):
        lo, hi = q[qi], q[qi+1]
        if qi < 3:
            m = (margin >= lo) & (margin < hi)
        else:
            m = (margin >= lo) & (margin <= hi)
        if m.sum() < 50:
            continue
        print(f"\n  margin Q{qi+1} [{lo:.3f},{hi:.3f}]  n={int(m.sum())}")
        for tname, y_bin in targets_bin.items():
            yb = y_bin[m]
            if yb.std() == 0:
                print(f"    {tname:<22} degenerate (pos rate={yb.mean():.3f})")
                continue
            a = uni_auroc(spec[m], yb)
            print(f"    {tname:<22} uni AUROC(spectral) = {a:.4f}  "
                  f"pos_rate={yb.mean():.3f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
