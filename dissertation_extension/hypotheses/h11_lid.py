"""
Hypothesis H11: Local Intrinsic Dimensionality (LID) computed from a trained
model's penultimate-layer features predicts adversarial vulnerability of clean
samples.

Ma et al. (ICLR 2018, "Characterizing Adversarial Subspaces Using Local
Intrinsic Dimensionality") used LID to *detect* adversarial samples by
contrasting LID of adversarial vs clean inputs in deep feature space.
Here we test the *inverse* direction: does the LID of a clean sample (in
penultimate-layer feature space of the victim model) predict whether that
clean sample is adversarially vulnerable?

LID MLE estimator (Amsaleg et al. / Levina-Bickel):
    LID_hat(x) = - 1 / mean_{i=1..K-1} log( d_i(x) / d_K(x) )
where d_i is the distance from x to its i-th nearest neighbour (i=1..K)
within a reference subsample, and d_K is the distance to the K-th NN.

Pipeline
--------
1. Train a small CNN (same architecture as diagnostic_test.py) for 10 epochs on
   Fashion-MNIST.
2. Extract 128-d penultimate-layer features for every train and test sample.
3. Sample a reference set of 2000 train features. For every test sample,
   compute LID with K=20 nearest neighbours in *feature space*. Also compute
   LID with K=20 in *raw pixel space* (784-d) using a separate 2000-sample
   reference set of train pixels.
4. Baseline features per test sample: victim_margin (logit gap top1-top2),
   mean_pix, std_pix.
5. Targets:
     - flipped_FGSM at eps = 15/255
     - flipped_PGD  (10 steps, eps = 15/255, alpha = eps/4)
     - FGSM_min_eps_binary_search  (continuous; smallest L_inf eps that flips)
6. Analysis (restricted to test samples initially classified correctly):
     - Univariate AUROC of each feature vs each binary target.
     - Spearman/Pearson correlation with min_eps (continuous).
     - Multivariate logistic regression: does LID_feat add over margin alone?
       Reports full AUROC, margin-only AUROC, and delta.
     - Same comparison for LID_pixel to test whether the deep representation
       matters.

Caveats
-------
- Sub-sampling the reference set is stochastic; one fixed seed is used. LID
  values can be noisy at K=20 in 128-d.
- "Vulnerability" is operationalised via FGSM/PGD against the same trained
  model. This is a self-attack setting (white-box, no transfer).
- Pixel-space LID for Fashion-MNIST is mostly driven by global brightness /
  fashion-class manifold density and is a weak baseline by design.
- We use Euclidean distances throughout. Cosine could give different numbers.
- min_eps_to_flip uses the FGSM gradient direction only (binary search over
  magnitude); not a proper minimal-L_inf attack like C&W or DeepFool.
- No multiple-comparison correction across the many AUROC tests reported.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0
K_LID = 20
REF_SUBSAMPLE = 2000
SEED = 0


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
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
        """Penultimate-layer 128-d features (after fc1 ReLU, before dropout/fc2)."""
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def train_model(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
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


# --------------------------------------------------------------------------- #
# Feature extraction                                                          #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(model, x_all, batch=512):
    feats = []
    for i in range(0, x_all.size(0), batch):
        feats.append(model.features(x_all[i:i+batch]).cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def extract_logits(model, x_all, batch=512):
    out = []
    for i in range(0, x_all.size(0), batch):
        out.append(model(x_all[i:i+batch]).cpu())
    return torch.cat(out, 0)


# --------------------------------------------------------------------------- #
# LID MLE estimator                                                           #
# --------------------------------------------------------------------------- #
def compute_lid(query, reference, k=K_LID):
    """
    LID_hat(x) = -1 / mean_{i=1..k} log( d_i / d_k )

    Args:
        query: (N, D) ndarray of points whose LID we want.
        reference: (M, D) ndarray of reference points (the sub-sample).
        k: number of nearest neighbours.

    Returns:
        (N,) ndarray of LID estimates.
    """
    nn_ = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(reference)
    dists, _ = nn_.kneighbors(query)        # (N, k), sorted ascending
    # Replace zero distances with a tiny epsilon so log is finite. Distance to
    # the k-th NN is dists[:, -1].
    eps = 1e-12
    dists = np.maximum(dists, eps)
    d_k = dists[:, -1:]                     # (N, 1)
    ratios = dists / d_k                    # (N, k); last col == 1, log = 0
    # Use indices 0..k-2 (the first k-1 neighbours strictly closer than d_k);
    # this is the standard Amsaleg/Ma form -1 / mean(log r_i / r_k) for i<k.
    log_r = np.log(ratios[:, :-1])          # (N, k-1)
    mean_log = log_r.mean(axis=1)
    # Guard against zero mean (all distances equal -> infinite LID); clamp.
    mean_log = np.where(mean_log == 0, -eps, mean_log)
    lid = -1.0 / mean_log
    return lid


# --------------------------------------------------------------------------- #
# Attacks                                                                     #
# --------------------------------------------------------------------------- #
def fgsm_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign = fgsm_sign(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    # random start within eps-ball
    delta = (torch.rand_like(x0) * 2 - 1) * eps
    adv = (x0 + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
    sign = fgsm_sign(model, x, y)
    lo = torch.zeros(x.size(0), device=x.device)
    hi = torch.full((x.size(0),), eps_max, device=x.device)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def run_attack_batched(fn, model, x_all, y_all, batch=256, **kw):
    out = []
    for i in range(0, x_all.size(0), batch):
        out.append(fn(model, x_all[i:i+batch], y_all[i:i+batch], **kw))
    return torch.cat(out, 0)


# --------------------------------------------------------------------------- #
# Analysis                                                                    #
# --------------------------------------------------------------------------- #
def auroc(y_true, score):
    # Try both directions; AUROC is symmetric so report the >=0.5 version.
    a = roc_auc_score(y_true, score)
    return max(a, 1 - a)


def analyse(features, names, targets, target_names, min_eps):
    print("\n=========== Univariate AUROC ===========")
    print(f"{'feature':<18}" + "".join(f"{t:>22}" for t in target_names))
    for j, n in enumerate(names):
        row = [f"{n:<18}"]
        for t_idx in range(len(target_names)):
            y = targets[t_idx]
            if y.std() == 0:
                row.append(f"{'NA':>22}")
            else:
                row.append(f"{auroc(y, features[:, j]):>22.4f}")
        print("".join(row))

    print("\n=========== Correlation with min_eps_to_flip ===========")
    print(f"{'feature':<18}{'Pearson':>12}{'Spearman':>12}")
    for j, n in enumerate(names):
        pe = pearsonr(features[:, j], min_eps)[0]
        sp = spearmanr(features[:, j], min_eps)[0]
        print(f"{n:<18}{pe:>+12.4f}{sp:>+12.4f}")

    print("\n=========== Multivariate: does LID_feat add over margin? ===========")
    Xs = StandardScaler().fit_transform(features)
    j_margin = names.index("victim_margin")
    j_lid_f = names.index("LID_feat")
    j_lid_p = names.index("LID_pixel")

    for t_idx, t_name in enumerate(target_names):
        y = targets[t_idx].astype(int)
        if y.std() == 0:
            print(f"  {t_name}: degenerate")
            continue
        # full (all features)
        full = LogisticRegression(max_iter=2000).fit(Xs, y)
        full_auc = roc_auc_score(y, full.predict_proba(Xs)[:, 1])
        # margin only
        m_only = LogisticRegression(max_iter=2000).fit(Xs[:, [j_margin]], y)
        m_only_auc = roc_auc_score(y, m_only.predict_proba(Xs[:, [j_margin]])[:, 1])
        # margin + LID_feat
        mf = LogisticRegression(max_iter=2000).fit(Xs[:, [j_margin, j_lid_f]], y)
        mf_auc = roc_auc_score(y, mf.predict_proba(Xs[:, [j_margin, j_lid_f]])[:, 1])
        # margin + LID_pixel
        mp = LogisticRegression(max_iter=2000).fit(Xs[:, [j_margin, j_lid_p]], y)
        mp_auc = roc_auc_score(y, mp.predict_proba(Xs[:, [j_margin, j_lid_p]])[:, 1])
        # all except LID_feat
        keep_no_lid_f = [j for j in range(Xs.shape[1]) if j != j_lid_f]
        no_lid_f = LogisticRegression(max_iter=2000).fit(Xs[:, keep_no_lid_f], y)
        no_lid_f_auc = roc_auc_score(
            y, no_lid_f.predict_proba(Xs[:, keep_no_lid_f])[:, 1])

        print(f"\n  target: {t_name}  (pos_rate={y.mean():.3f})")
        print(f"    margin only                : {m_only_auc:.4f}")
        print(f"    margin + LID_feat          : {mf_auc:.4f}   (delta over margin {mf_auc - m_only_auc:+.4f})")
        print(f"    margin + LID_pixel         : {mp_auc:.4f}   (delta over margin {mp_auc - m_only_auc:+.4f})")
        print(f"    all features               : {full_auc:.4f}")
        print(f"    all except LID_feat        : {no_lid_f_auc:.4f}   (drop {full_auc - no_lid_f_auc:+.4f})")
        print(f"    standardised coefficients (all-feature model):")
        for n, c in zip(names, full.coef_.flatten()):
            print(f"      {n:<18}{c:+.4f}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    print(f"device: {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)

    print("Training victim CNN ...")
    t0 = time.time()
    model = train_model(train_set)
    print(f"  trained in {time.time()-t0:.1f}s")

    # Stack all data tensors.
    train_x = torch.stack([train_set[i][0] for i in range(len(train_set))])
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # --- penultimate-layer features for train (host) and test ---
    print("Extracting penultimate-layer features ...")
    train_feat = extract_features(model, train_x.to(DEVICE))    # (60000, 128)
    test_feat = extract_features(model, test_x)                 # (10000, 128)

    # --- reference subsample (fixed seed) for LID computation ---
    rng = np.random.default_rng(SEED)
    ref_idx = rng.choice(train_feat.shape[0], size=REF_SUBSAMPLE, replace=False)
    ref_feat = train_feat[ref_idx]                              # (2000, 128)
    train_pix_flat = train_x.view(train_x.size(0), -1).cpu().numpy()
    ref_pix = train_pix_flat[ref_idx]                           # (2000, 784)
    test_pix_flat = test_x.view(test_x.size(0), -1).cpu().numpy()

    print(f"Computing LID (K={K_LID}, |ref|={REF_SUBSAMPLE}) ...")
    lid_feat = compute_lid(test_feat, ref_feat, k=K_LID)
    lid_pixel = compute_lid(test_pix_flat, ref_pix, k=K_LID)
    print(f"  LID_feat:  mean={lid_feat.mean():.2f} median={np.median(lid_feat):.2f} "
          f"min={lid_feat.min():.2f} max={lid_feat.max():.2f}")
    print(f"  LID_pixel: mean={lid_pixel.mean():.2f} median={np.median(lid_pixel):.2f} "
          f"min={lid_pixel.min():.2f} max={lid_pixel.max():.2f}")

    # --- baseline features ---
    logits = extract_logits(model, test_x)
    sorted_logits, _ = logits.sort(1, descending=True)
    victim_margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).numpy()
    final_pred = logits.argmax(1).to(DEVICE)
    mean_pix = test_pix_flat.mean(axis=1)
    std_pix = test_pix_flat.std(axis=1)

    # --- attacks (only for correctly classified samples) ---
    correct_mask = (final_pred == test_y).cpu().numpy().astype(bool)
    print(f"Correctly classified by victim: {correct_mask.sum()} / {len(correct_mask)}")
    idx_correct = np.where(correct_mask)[0]
    x_c = test_x[correct_mask]
    y_c = test_y[correct_mask]

    print("FGSM eps=15/255 ...")
    fgsm_flipped = run_attack_batched(fgsm_flip, model, x_c, y_c).cpu().numpy().astype(int)
    print(f"  flip rate {fgsm_flipped.mean():.3f}")

    print("PGD eps=15/255 ...")
    pgd_flipped = run_attack_batched(pgd_flip, model, x_c, y_c).cpu().numpy().astype(int)
    print(f"  flip rate {pgd_flipped.mean():.3f}")

    print("FGSM min_eps binary search ...")
    min_eps = run_attack_batched(fgsm_min_eps, model, x_c, y_c).cpu().numpy()
    print(f"  mean min_eps {min_eps.mean():.4f}, median {np.median(min_eps):.4f}")

    # --- assemble feature matrix on correct subset ---
    feat_names = ["victim_margin", "mean_pix", "std_pix", "LID_feat", "LID_pixel"]
    F_full = np.stack([victim_margin, mean_pix, std_pix, lid_feat, lid_pixel], axis=1)
    F_c = F_full[idx_correct]

    targets = [fgsm_flipped, pgd_flipped]
    target_names = ["flipped_FGSM_15/255", "flipped_PGD_15/255"]

    analyse(F_c, feat_names, targets, target_names, min_eps)

    # Quick descriptive: LID by flipped vs not-flipped (FGSM).
    print("\n=========== LID descriptive (FGSM) ===========")
    for name, arr in [("LID_feat", lid_feat[idx_correct]), ("LID_pixel", lid_pixel[idx_correct])]:
        f_yes = arr[fgsm_flipped == 1]
        f_no = arr[fgsm_flipped == 0]
        print(f"  {name}: flipped mean={f_yes.mean():.3f}  not-flipped mean={f_no.mean():.3f}  "
              f"diff={f_yes.mean() - f_no.mean():+.3f}")


if __name__ == "__main__":
    main()
