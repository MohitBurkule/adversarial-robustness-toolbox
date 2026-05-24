"""
H64: Random-forest-on-raw-pixels confidence as a predictor of CNN adversarial vulnerability.

Hypothesis
----------
A low-capacity, non-differentiable model (Random Forest on raw flattened pixels) has
its own per-sample confidence/margin. Because it carves the input space very
differently from a CNN, samples near its decision boundary (low RF margin / high RF
entropy) or where the RF disagrees with the CNN may correspond to genuinely ambiguous
inputs --- i.e., the CNN's adversarial vulnerability hot-spots.

Pipeline
--------
  1. Train a small CNN matching ``diagnostic_test.py``'s architecture on Fashion-MNIST
     for 10 epochs.
  2. Train ``sklearn.ensemble.RandomForestClassifier(n_estimators=100)`` on flattened
     raw pixels (no normalization beyond [0,1] scaling).
  3. Per test sample features:
        - rf_margin              : RF top-1 probability minus top-2 probability
        - rf_entropy             : entropy of RF class probability vector
        - rf_agreement_with_cnn  : 1 if RF argmax == CNN argmax else 0
        - cnn_margin             : CNN top-1 logit minus top-2 logit
        - mean_pix               : mean of raw pixel values for the sample
        - std_pix                : std of raw pixel values for the sample
  4. Targets:
        - flipped_FGSM_eps15     : CNN FGSM at eps = 15/255 flips the prediction
        - flipped_PGD            : CNN PGD (10 steps, eps = 15/255) flips the prediction
  5. Univariate AUROC for each feature against each binary target.
  6. Multivariate logistic regression: does RF disagreement add predictive signal for
     adversarial vulnerability beyond CNN margin and simple pixel stats? Compare full
     model AUROC against model with RF features removed.

Run on samples the CNN classifies correctly (the only ones the attack targets care about).
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS / 4.0
N_TREES = 100
SEED = 0


# ---------------------------------------------------------------------------
# CNN (matches diagnostic_test.py exactly)
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
# Training / attacks
# ---------------------------------------------------------------------------
def train_cnn(train_set):
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
        print(f"  epoch {ep+1}/{EPOCHS} done ({time.time()-t0:.1f}s)")
    model.eval()
    return model


def fgsm_attack(model, x, y, eps=EPS):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    adv = (x_adv + eps * x_adv.grad.sign()).clamp(0, 1).detach()
    return adv


def pgd_attack(model, x, y, eps=EPS, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x_orig = x.clone().detach()
    # random start within eps ball
    adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    return adv.detach()


def batched_flip(model, x, y, attack_fn, batch=256):
    out = []
    for i in range(0, x.size(0), batch):
        xb = x[i:i+batch]
        yb = y[i:i+batch]
        adv = attack_fn(model, xb, yb)
        with torch.no_grad():
            pred = model(adv).argmax(1)
        out.append((pred != yb).cpu())
    return torch.cat(out).numpy().astype(int)


def cnn_predictions(model, x, batch=512):
    preds, margins = [], []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            logits = model(x[i:i+batch])
            sorted_logits, _ = logits.sort(1, descending=True)
            margins.append((sorted_logits[:, 0] - sorted_logits[:, 1]).cpu())
            preds.append(logits.argmax(1).cpu())
    return torch.cat(preds).numpy(), torch.cat(margins).numpy()


# ---------------------------------------------------------------------------
# RF features
# ---------------------------------------------------------------------------
def rf_features(rf, X_flat, cnn_pred):
    probs = rf.predict_proba(X_flat)  # (N, 10)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    rf_margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    eps_e = 1e-12
    rf_entropy = -(probs * np.log(probs + eps_e)).sum(axis=1)
    rf_pred = probs.argmax(axis=1)
    rf_agreement = (rf_pred == cnn_pred).astype(int)
    return rf_margin, rf_entropy, rf_agreement, rf_pred


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def univariate_auroc(feats, names, targets, target_names):
    print("\n--- Univariate AUROC ---")
    print(f"{'feature':<26}" + "".join(f"{t:>22}" for t in target_names))
    for i, n in enumerate(names):
        x_i = feats[:, i]
        row = f"{n:<26}"
        for t_idx, _ in enumerate(target_names):
            y = targets[t_idx]
            if y.std() == 0 or np.unique(y).size < 2:
                row += f"{'n/a':>22}"
                continue
            a = roc_auc_score(y, x_i)
            a = max(a, 1 - a)
            row += f"{a:>22.4f}"
        print(row)


def multivariate(feats, names, targets, target_names):
    print("\n--- Multivariate logistic regression ---")
    Xs = StandardScaler().fit_transform(feats)
    rf_idx = [i for i, n in enumerate(names)
              if n in ("rf_margin", "rf_entropy", "rf_agreement_with_cnn")]
    for t_idx, t_name in enumerate(target_names):
        y = targets[t_idx]
        if y.std() == 0 or np.unique(y).size < 2:
            print(f"\n  target {t_name}: degenerate, skipping")
            continue
        print(f"\n  target: {t_name}  (positive rate = {y.mean():.3f})")
        lr_full = LogisticRegression(max_iter=2000).fit(Xs, y)
        full_auc = roc_auc_score(y, lr_full.predict_proba(Xs)[:, 1])
        Xs_no_rf = np.delete(Xs, rf_idx, axis=1)
        lr_no_rf = LogisticRegression(max_iter=2000).fit(Xs_no_rf, y)
        no_rf_auc = roc_auc_score(y, lr_no_rf.predict_proba(Xs_no_rf)[:, 1])
        print(f"    multivariate AUROC (all features):     {full_auc:.4f}")
        print(f"    multivariate AUROC (RF features out):  {no_rf_auc:.4f}")
        print(f"    Delta AUROC from RF features:          {full_auc - no_rf_auc:+.4f}")
        print(f"    standardised coefficients:")
        for n, c in zip(names, lr_full.coef_.flatten()):
            tag = "  [RF]" if n in ("rf_margin", "rf_entropy",
                                    "rf_agreement_with_cnn") else ""
            print(f"      {n:<26} {c:+.4f}{tag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("H64: RandomForest-on-raw-pixels confidence vs CNN adversarial vulnerability")
    print("=" * 70)

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # tensors for CNN (GPU)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)
    print(f"\nTest set: N={N}")

    # ---- Train CNN ----
    print("\n[1/4] Training CNN (10 epochs)...")
    t0 = time.time()
    cnn = train_cnn(train_set)
    print(f"  CNN trained ({time.time()-t0:.1f}s)")

    # ---- Train Random Forest on flattened raw pixels ----
    print(f"\n[2/4] Training RandomForestClassifier(n_estimators={N_TREES}) on raw pixels...")
    t0 = time.time()
    X_tr = np.stack([train_set[i][0].numpy().reshape(-1)
                     for i in range(len(train_set))]).astype(np.float32)
    y_tr = np.array([train_set[i][1] for i in range(len(train_set))])
    X_te = test_x.cpu().numpy().reshape(N, -1).astype(np.float32)
    y_te = test_y.cpu().numpy()
    rf = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=SEED)
    rf.fit(X_tr, y_tr)
    print(f"  RF trained ({time.time()-t0:.1f}s)  test acc={rf.score(X_te, y_te):.4f}")

    # ---- Per-sample features ----
    print("\n[3/4] Computing per-sample features...")
    cnn_pred, cnn_margin = cnn_predictions(cnn, test_x)
    rf_m, rf_e, rf_agree, rf_pred = rf_features(rf, X_te, cnn_pred)
    mean_pix = X_te.mean(axis=1)
    std_pix = X_te.std(axis=1)

    # ---- Adversarial targets ----
    print("\n[4/4] Generating adversarial examples (FGSM eps=15/255, PGD)...")
    # restrict to CNN-correct samples (adversarial flipping only meaningful there)
    correct_mask = (cnn_pred == y_te)
    print(f"  CNN clean accuracy = {correct_mask.mean():.4f} "
          f"({correct_mask.sum()} correct samples used)")

    x_c = test_x[torch.as_tensor(correct_mask, device=DEVICE)]
    y_c = test_y[torch.as_tensor(correct_mask, device=DEVICE)]

    t0 = time.time()
    fgsm_flip = batched_flip(cnn, x_c, y_c, lambda m, a, b: fgsm_attack(m, a, b, EPS))
    print(f"  FGSM eps=15/255 flip rate = {fgsm_flip.mean():.4f} "
          f"({time.time()-t0:.1f}s)")
    t0 = time.time()
    pgd_flip = batched_flip(
        cnn, x_c, y_c,
        lambda m, a, b: pgd_attack(m, a, b, EPS, PGD_ALPHA, PGD_STEPS))
    print(f"  PGD eps=15/255 ({PGD_STEPS} steps) flip rate = {pgd_flip.mean():.4f} "
          f"({time.time()-t0:.1f}s)")

    # subset features to CNN-correct samples
    feats = np.stack([
        rf_m[correct_mask],
        rf_e[correct_mask],
        rf_agree[correct_mask],
        cnn_margin[correct_mask],
        mean_pix[correct_mask],
        std_pix[correct_mask],
    ], axis=1).astype(np.float64)
    names = ["rf_margin", "rf_entropy", "rf_agreement_with_cnn",
             "cnn_margin", "mean_pix", "std_pix"]
    targets = [fgsm_flip, pgd_flip]
    target_names = ["flipped_FGSM_eps15", "flipped_PGD"]

    # ---- Analyses ----
    univariate_auroc(feats, names, targets, target_names)
    multivariate(feats, names, targets, target_names)

    # ---- Direct RF-disagreement probe ----
    print("\n--- RF disagreement vs adversarial vulnerability (contingency) ---")
    disagree = (1 - feats[:, names.index("rf_agreement_with_cnn")]).astype(int)
    for t_idx, t_name in enumerate(target_names):
        y = targets[t_idx]
        if np.unique(y).size < 2:
            continue
        p_flip_dis = y[disagree == 1].mean() if (disagree == 1).any() else float("nan")
        p_flip_agr = y[disagree == 0].mean() if (disagree == 0).any() else float("nan")
        print(f"  {t_name}: P(flip | RF disagrees) = {p_flip_dis:.4f}   "
              f"P(flip | RF agrees) = {p_flip_agr:.4f}   "
              f"lift = {p_flip_dis - p_flip_agr:+.4f}")


if __name__ == "__main__":
    main()
