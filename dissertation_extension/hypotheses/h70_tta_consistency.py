"""
H70: Test-time augmentation (TTA) consistency predicts adversarial vulnerability.

Hypothesis
----------
For each test sample we generate K=32 augmented copies using a mix of
  - small rotations
  - small translations
  - brightness shifts
  - additive gaussian noise
  - colour-jitter-style contrast scaling
  - small scale (zoom) perturbations
and ask the trained model to classify all copies. The intuition: a sample
that lives close to a decision boundary will see large disagreement across
these label-preserving augmentations, and such samples should also be the
ones most easily flipped by adversarial attacks.

We extract three TTA-derived features:
  tta_vote_agreement     fraction of K copies that vote for the model's
                         clean-prediction (higher = more consistent).
  tta_softmax_l2_variance mean L2-variance of the per-augmentation softmax
                         vectors around the clean softmax (higher = noisier).
  tta_max_loss_increase   max over augmentations of (CE_loss_aug - CE_loss_clean).

Baselines: margin (clean-logit top1 - top2), mean_pix, std_pix.
Targets:   FGSM-flip @ eps=15/255, PGD-flip @ eps=15/255 (10 step), min_eps
           (per-sample binary search smallest L_inf eps that flips with FGSM).

Analysis:
  * univariate AUROC of every feature against every binary target,
    Spearman correlation against the continuous min_eps target;
  * multivariate logistic regression: does adding TTA features improve over
    margin alone? (Delta AUROC, ablation by removing TTA block).

Self-contained: trains the same small CNN architecture used in
dissertation_extension/diagnostic_test.py on Fashion-MNIST for 10 epochs.
DO NOT run from this script automatically - just defines and calls main()
at __main__ as usual.
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
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4
K_TTA = 32
SEED = 0


# ---------------------------------------------------------------------------
# Model (matches diagnostic_test.py)
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

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(train_set):
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
        print(f"  epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Augmentation operators (all differentiable-free; vectorised over a batch)
# Inputs: x  shape (B, 1, 28, 28) in [0, 1]
# ---------------------------------------------------------------------------
def _rotate(x, deg):
    """Rotate by a per-sample angle in degrees using affine_grid."""
    B = x.size(0)
    theta = torch.deg2rad(deg)
    cos, sin = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(cos)
    mat = torch.stack([cos, -sin, zero, sin, cos, zero], dim=1).view(B, 2, 3)
    grid = F.affine_grid(mat, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def _translate(x, tx, ty):
    """Translate by tx, ty in pixels."""
    B, _, H, W = x.size()
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = -2 * tx / W
    theta[:, 1, 2] = -2 * ty / H
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def _scale(x, s):
    """Zoom by per-sample scale factor s (>1 = zoom in)."""
    B = x.size(0)
    theta = torch.zeros(B, 2, 3, device=x.device)
    inv = 1.0 / s
    theta[:, 0, 0] = inv
    theta[:, 1, 1] = inv
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def _brightness(x, db):
    return (x + db.view(-1, 1, 1, 1)).clamp(0, 1)


def _gaussian_noise(x, sigma):
    return (x + sigma.view(-1, 1, 1, 1) * torch.randn_like(x)).clamp(0, 1)


def _contrast(x, c):
    """Colour-jitter-style contrast: pull around mean by factor c."""
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * c.view(-1, 1, 1, 1) + m).clamp(0, 1)


def make_augmented_copies(x, k, rng):
    """
    Generate k augmented copies of every sample in x.

    Returns shape (k, B, 1, 28, 28). Augmentation k=0 is the identity; the
    remaining k-1 copies are random combinations of the six operators with
    randomly sampled per-copy strengths.
    """
    B = x.size(0)
    out = [x]
    for j in range(1, k):
        # sample strengths
        deg = torch.empty(B, device=x.device).uniform_(-10.0, 10.0, generator=None)
        tx = torch.empty(B, device=x.device).uniform_(-2.0, 2.0)
        ty = torch.empty(B, device=x.device).uniform_(-2.0, 2.0)
        s = torch.empty(B, device=x.device).uniform_(0.9, 1.1)
        db = torch.empty(B, device=x.device).uniform_(-0.1, 0.1)
        sigma = torch.empty(B, device=x.device).uniform_(0.0, 0.05)
        c = torch.empty(B, device=x.device).uniform_(0.85, 1.15)

        # generate via deterministic-per-copy RNG-free ops
        a = _rotate(x, deg)
        a = _translate(a, tx, ty)
        a = _scale(a, s)
        a = _brightness(a, db)
        a = _contrast(a, c)
        a = _gaussian_noise(a, sigma)
        out.append(a)
    return torch.stack(out, 0)  # (k, B, 1, 28, 28)


# ---------------------------------------------------------------------------
# TTA features
# ---------------------------------------------------------------------------
def compute_tta_features(model, x, y, k=K_TTA, batch=128):
    """
    Returns dict of three tensors (each shape (N,)):
        tta_vote_agreement
        tta_softmax_l2_variance
        tta_max_loss_increase
    """
    model.eval()
    N = x.size(0)
    vote = torch.empty(N, device=DEVICE)
    sv = torch.empty(N, device=DEVICE)
    mli = torch.empty(N, device=DEVICE)
    rng = np.random.RandomState(123)
    with torch.no_grad():
        for i in range(0, N, batch):
            xb = x[i:i + batch]
            yb = y[i:i + batch]
            B = xb.size(0)
            aug = make_augmented_copies(xb, k, rng)            # (k, B, 1, 28, 28)
            flat = aug.view(k * B, 1, 28, 28)
            logits = model(flat).view(k, B, 10)
            sm = F.softmax(logits, -1)
            preds = logits.argmax(-1)                          # (k, B)
            clean_pred = preds[0]                              # (B,)
            agree = (preds == clean_pred.unsqueeze(0)).float().mean(0)
            mean_sm = sm.mean(0, keepdim=True)
            var_l2 = ((sm - mean_sm) ** 2).sum(-1).mean(0)     # (B,)
            ce_each = F.cross_entropy(
                logits.view(k * B, 10),
                yb.repeat(k),
                reduction="none").view(k, B)
            loss_inc = (ce_each - ce_each[0:1]).max(0).values
            vote[i:i + batch] = agree
            sv[i:i + batch] = var_l2
            mli[i:i + batch] = loss_inc
    # higher tta_vote_agreement -> more robust, so flip sign so all three
    # features are oriented "high = more vulnerable" for sanity, but we keep
    # raw values and let AUROC pick direction.
    return {
        "tta_vote_agreement": vote,
        "tta_softmax_l2_variance": sv,
        "tta_max_loss_increase": mli,
    }


# ---------------------------------------------------------------------------
# Baseline features
# ---------------------------------------------------------------------------
def compute_baselines(model, x, y):
    N = x.size(0)
    with torch.no_grad():
        logits_list = []
        for i in range(0, N, 512):
            logits_list.append(model(x[i:i + 512]))
        logits = torch.cat(logits_list, 0)
    sorted_lg, _ = logits.sort(1, descending=True)
    margin = sorted_lg[:, 0] - sorted_lg[:, 1]
    mean_pix = x.view(N, -1).mean(1)
    std_pix = x.view(N, -1).std(1)
    final_pred = logits.argmax(1)
    return {
        "margin": margin,
        "mean_pix": mean_pix,
        "std_pix": std_pix,
    }, final_pred


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def fgsm_grad_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST, batch=256):
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        sign = fgsm_grad_sign(model, xb, yb)
        adv = (xb + eps * sign).clamp(0, 1)
        with torch.no_grad():
            out.append(model(adv).argmax(1) != yb)
    return torch.cat(out)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA,
             steps=PGD_STEPS, batch=256):
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        adv = xb.clone().detach()
        adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
        adv = adv.clamp(0, 1).detach()
        for _ in range(steps):
            adv.requires_grad_(True)
            loss = F.cross_entropy(model(adv), yb)
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + alpha * grad.sign()
            adv = torch.max(torch.min(adv, xb + eps), xb - eps).clamp(0, 1)
        with torch.no_grad():
            out.append(model(adv).argmax(1) != yb)
    return torch.cat(out)


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15, batch=256):
    """Per-sample binary search on FGSM eps for the smallest flip."""
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        sign = fgsm_grad_sign(model, xb, yb)
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xb + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
            with torch.no_grad():
                flipped = model(adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        out.append(hi)
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def auc_both(y, score):
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def evaluate(features, feat_names, targets_bin, target_names, min_eps):
    F_np = np.stack([features[n].cpu().numpy() for n in feat_names], 1)
    Xs = StandardScaler().fit_transform(F_np)

    print("\n----- univariate AUROC -----")
    print(f"{'feature':<30}" + "".join(f"{t:>20}" for t in target_names))
    for i, n in enumerate(feat_names):
        row = f"{n:<30}"
        for tname in target_names:
            y = targets_bin[tname].cpu().numpy().astype(int)
            if y.std() == 0:
                row += f"{'n/a':>20}"
            else:
                row += f"{auc_both(y, F_np[:, i]):>20.4f}"
        print(row)

    print("\n----- Spearman correlation with min_eps -----")
    me = min_eps.cpu().numpy()
    for i, n in enumerate(feat_names):
        rho, _ = spearmanr(F_np[:, i], me)
        print(f"  {n:<30} rho = {rho:+.4f}")

    # Multivariate: does TTA add to margin?
    tta_names = ["tta_vote_agreement",
                 "tta_softmax_l2_variance",
                 "tta_max_loss_increase"]
    base_names = ["margin", "mean_pix", "std_pix"]
    margin_only_idx = [feat_names.index("margin")]
    base_idx = [feat_names.index(n) for n in base_names]
    all_idx = list(range(len(feat_names)))
    base_plus_tta_idx = all_idx
    tta_idx = [feat_names.index(n) for n in tta_names]

    def fit_auc(cols, y):
        X = Xs[:, cols]
        clf = LogisticRegression(max_iter=2000).fit(X, y)
        return roc_auc_score(y, clf.predict_proba(X)[:, 1]), clf

    print("\n----- multivariate AUROC -----")
    print(f"{'target':<22}{'margin':>10}{'baselines':>12}"
          f"{'+TTA':>10}{'all':>10}{'dTTA(over margin)':>22}"
          f"{'dTTA(over baselines)':>24}")
    for tname in target_names:
        y = targets_bin[tname].cpu().numpy().astype(int)
        if y.std() == 0:
            print(f"  {tname}: degenerate")
            continue
        auc_m, _ = fit_auc(margin_only_idx, y)
        auc_b, _ = fit_auc(base_idx, y)
        auc_mt, _ = fit_auc(margin_only_idx + tta_idx, y)
        auc_all, clf_all = fit_auc(base_plus_tta_idx, y)
        print(f"{tname:<22}{auc_m:>10.4f}{auc_b:>12.4f}"
              f"{auc_mt:>10.4f}{auc_all:>10.4f}"
              f"{auc_mt - auc_m:>+22.4f}"
              f"{auc_all - auc_b:>+24.4f}")
        print("    standardised coefficients (full model):")
        for n, c in zip(feat_names, clf_all.coef_.flatten()):
            print(f"      {n:<30} {c:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tf = transforms.ToTensor()
    print("loading Fashion-MNIST ...")
    train_set = datasets.FashionMNIST("./data", train=True,
                                      download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False,
                                     download=True, transform=tf)

    print("training CNN for", EPOCHS, "epochs ...")
    t0 = time.time()
    model = train(train_set)
    print(f"  done ({time.time() - t0:.1f}s)")

    # pull entire test set onto device
    x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # baselines + correct mask
    base, final_pred = compute_baselines(model, x, y)
    correct = final_pred == y
    print(f" keeping {correct.sum().item()} / {len(y)} correctly classified samples")
    x_c = x[correct]
    y_c = y[correct]
    base_c = {k: v[correct] for k, v in base.items()}

    print("computing TTA features (K=%d) ..." % K_TTA)
    t0 = time.time()
    tta = compute_tta_features(model, x_c, y_c, k=K_TTA)
    print(f"  done ({time.time() - t0:.1f}s)")

    features = {**base_c, **tta}
    feat_names = ["margin", "mean_pix", "std_pix",
                  "tta_vote_agreement",
                  "tta_softmax_l2_variance",
                  "tta_max_loss_increase"]

    # targets
    print("computing FGSM flips @ eps=%.4f ..." % EPS_TEST)
    fgsm_b = fgsm_flip(model, x_c, y_c)
    print(f"  FGSM flip rate = {fgsm_b.float().mean().item():.3f}")

    print("computing PGD-%d flips @ eps=%.4f ..." % (PGD_STEPS, EPS_TEST))
    pgd_b = pgd_flip(model, x_c, y_c)
    print(f"  PGD  flip rate = {pgd_b.float().mean().item():.3f}")

    print("computing min_eps_to_flip (FGSM binary search) ...")
    me = min_eps_fgsm(model, x_c, y_c)
    print(f"  mean min_eps = {me.mean().item():.4f}")

    # Binary form of min_eps: flipped below the median
    med = me.median().item()
    me_bin = (me < med).long()

    targets_bin = {
        "FGSM_flip":         fgsm_b,
        "PGD_flip":          pgd_b,
        "low_min_eps":       me_bin,
    }
    target_names = list(targets_bin.keys())

    evaluate(features, feat_names, targets_bin, target_names, me)
    print("\ndone.")


if __name__ == "__main__":
    main()
