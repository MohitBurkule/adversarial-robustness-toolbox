"""
H105: Cross-dataset transfer of vulnerability predictors.

Hypothesis: Features trained to predict adversarial vulnerability on MNIST
transfer to Fashion-MNIST (model-agnostic, dataset-free predictors).

Pipeline:
  1. Train a CNN (matching diagnostic_test.py's architecture) on MNIST for
     10 epochs, logging per-sample per-epoch argmax and true-class softmax
     probability on the MNIST test set.
  2. Compute per-sample features on MNIST:
        final_margin, conf_mean, conf_var, learning_epoch,
        forgetting_events, S_x.
  3. Compute FGSM-flip target on MNIST (binary: flipped at eps=15/255).
  4. Fit a logistic regression on standardised MNIST features -> MNIST FGSM
     flip target.
  5. Train a *separate* victim CNN on Fashion-MNIST (10 epochs) with its own
     per-epoch logs; compute Fashion-MNIST per-sample features and Fashion
     FGSM-flip target using the Fashion victim.
  6. Apply the MNIST-trained logistic regression (with MNIST scaler) directly
     to Fashion features. Report cross-dataset AUROC and compare to:
       - in-domain MNIST AUROC (train/test on MNIST)
       - oracle Fashion AUROC (logistic regression trained on Fashion)
     If cross-dataset AUROC >> 0.5 and is close to oracle Fashion AUROC, the
     vulnerability predictor is dataset-free.

Run-only specification: code is written, not executed.
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
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
FEAT_NAMES = ["final_margin", "conf_mean", "conf_var",
              "learning_epoch", "forgetting_events", "S_x"]


# -------------------- model (matches diagnostic_test.py) --------------------
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


# -------------------- training + per-epoch logging --------------------
def train_and_log(seed, train_set, test_set, n_classes=10, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    model = CNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    per_epoch_argmax = torch.zeros(epochs, N, dtype=torch.long, device=DEVICE)
    per_epoch_trueprob = torch.zeros(epochs, N, device=DEVICE)

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds, probs = [], []
            for i in range(0, N, 512):
                logits = model(test_x[i:i + 512])
                preds.append(logits.argmax(1))
                p = F.softmax(logits, 1)
                probs.append(p[torch.arange(p.size(0)), test_y[i:i + 512]])
            per_epoch_argmax[ep] = torch.cat(preds)
            per_epoch_trueprob[ep] = torch.cat(probs)
    return model, test_x, test_y, per_epoch_argmax, per_epoch_trueprob


# -------------------- feature computation --------------------
def compute_features(test_x, test_y, per_epoch_argmax, per_epoch_trueprob,
                     model, n_classes=10, epochs=EPOCHS):
    N = test_x.size(0)
    with torch.no_grad():
        logits = []
        for i in range(0, N, 512):
            logits.append(model(test_x[i:i + 512]))
        logits = torch.cat(logits, 0)
    sorted_logits, _ = logits.sort(1, descending=True)
    final_margin = sorted_logits[:, 0] - sorted_logits[:, 1]
    final_2nd = logits.masked_fill(
        F.one_hot(test_y, n_classes).bool(), -1e9).argmax(1)
    final_pred = logits.argmax(1)

    conf_mean = per_epoch_trueprob.mean(0)
    conf_var = per_epoch_trueprob.var(0)

    is_right = (per_epoch_argmax == test_y.unsqueeze(0))
    first_right = is_right.float().argmax(0)
    never_right = ~is_right.any(0)
    learning_ep = torch.where(never_right,
                              torch.full_like(first_right, epochs),
                              first_right)
    transitions = (is_right[:-1] & ~is_right[1:]).sum(0)

    confusion = torch.zeros(N, n_classes, device=DEVICE)
    for ep in range(epochs):
        wrong = per_epoch_argmax[ep] != test_y
        if wrong.any():
            idx_w = torch.arange(N, device=DEVICE)[wrong]
            confusion[idx_w, per_epoch_argmax[ep][wrong]] += 1
    no_confusion = confusion.sum(1) == 0
    top_conf = confusion.argmax(1)
    S_x = ((top_conf == final_2nd) & (~no_confusion)).long()

    feats = torch.stack([final_margin, conf_mean, conf_var,
                         learning_ep.float(), transitions.float(),
                         S_x.float()], 1)
    return feats, final_pred


# -------------------- FGSM attack --------------------
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def attack_success(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return model(adv).argmax(1) != y


def fgsm_flip_target(model, x, y, batch=512):
    out = []
    for i in range(0, x.size(0), batch):
        out.append(attack_success(model, x[i:i + batch], y[i:i + batch]))
    return torch.cat(out)


# -------------------- per-dataset pipeline --------------------
def build_dataset(name, dataset_cls, seed):
    print(f"\n##### dataset: {name} (seed={seed}) #####")
    tf = transforms.ToTensor()
    train_set = dataset_cls("./data", train=True, download=True, transform=tf)
    test_set = dataset_cls("./data", train=False, download=True, transform=tf)

    t0 = time.time()
    model, x, y, pe_arg, pe_pr = train_and_log(seed, train_set, test_set)
    print(f"  train done ({time.time() - t0:.1f}s)")

    feats, final_pred = compute_features(x, y, pe_arg, pe_pr, model)
    correct = final_pred == y
    print(f"  correctly classified: {correct.sum().item()}/{correct.numel()}")
    x_c, y_c, feats_c = x[correct], y[correct], feats[correct]

    t0 = time.time()
    flip = fgsm_flip_target(model, x_c, y_c)
    print(f"  FGSM flip rate: {flip.float().mean():.4f} "
          f"({time.time() - t0:.1f}s)")
    return feats_c.detach().cpu().numpy(), flip.detach().cpu().numpy().astype(int)


# -------------------- main: cross-dataset transfer --------------------
def main():
    # 1. MNIST source
    X_mn, y_mn = build_dataset("MNIST", datasets.MNIST, seed=0)
    # 2. Fashion-MNIST target (separate victim model)
    X_fa, y_fa = build_dataset("FashionMNIST", datasets.FashionMNIST, seed=0)

    # Sanity checks
    print("\n===== cross-dataset transfer =====")
    print(f"  MNIST   n={len(y_mn)}  pos_rate={y_mn.mean():.4f}")
    print(f"  Fashion n={len(y_fa)}  pos_rate={y_fa.mean():.4f}")
    if y_mn.std() == 0 or y_fa.std() == 0:
        print("  degenerate target on one dataset; aborting.")
        return

    # Scaler fit on MNIST features (the predictor only saw MNIST)
    scaler_mn = StandardScaler().fit(X_mn)
    X_mn_s = scaler_mn.transform(X_mn)
    X_fa_s_under_mn = scaler_mn.transform(X_fa)  # apply MNIST scaler to Fashion

    # Train logistic regression on MNIST
    clf_mn = LogisticRegression(max_iter=2000).fit(X_mn_s, y_mn)

    # In-domain MNIST AUROC (train=test on MNIST, just as a reference)
    in_dom_mn = roc_auc_score(y_mn, clf_mn.predict_proba(X_mn_s)[:, 1])

    # Cross-dataset: MNIST-trained classifier applied to Fashion features
    cross_auc = roc_auc_score(y_fa,
                              clf_mn.predict_proba(X_fa_s_under_mn)[:, 1])

    # Oracle: train logistic regression on Fashion to bound the achievable
    scaler_fa = StandardScaler().fit(X_fa)
    X_fa_s = scaler_fa.transform(X_fa)
    clf_fa = LogisticRegression(max_iter=2000).fit(X_fa_s, y_fa)
    oracle_fa = roc_auc_score(y_fa, clf_fa.predict_proba(X_fa_s)[:, 1])

    # Univariate transfer baselines (each MNIST-z-scored feature applied to
    # Fashion z-scored under MNIST stats)
    print("\n  --- univariate AUROC on Fashion-MNIST (no retraining) ---")
    for i, n in enumerate(FEAT_NAMES):
        a = roc_auc_score(y_fa, X_fa_s_under_mn[:, i])
        a = max(a, 1 - a)
        print(f"    {n:<20} {a:.4f}")

    print("\n  --- multivariate logistic regression ---")
    print(f"    in-domain MNIST  (train=MNIST,   test=MNIST):   {in_dom_mn:.4f}")
    print(f"    CROSS-DATASET    (train=MNIST,   test=Fashion): {cross_auc:.4f}")
    print(f"    oracle Fashion   (train=Fashion, test=Fashion): {oracle_fa:.4f}")
    print(f"    transfer gap (oracle - cross):                  "
          f"{oracle_fa - cross_auc:+.4f}")

    print("\n  MNIST-trained logistic regression coefficients (standardised):")
    for n, c in zip(FEAT_NAMES, clf_mn.coef_.flatten()):
        print(f"    {n:<20} {c:+.4f}")
    print(f"    intercept           {clf_mn.intercept_[0]:+.4f}")

    # Verdict
    print("\n===== verdict =====")
    if cross_auc > 0.5 + 0.02 and (oracle_fa - cross_auc) < 0.05:
        print("  Cross-dataset predictor WORKS: AUROC is well above chance "
              "and within 0.05 of the oracle.")
    elif cross_auc > 0.5 + 0.02:
        print("  Cross-dataset predictor partially transfers: above chance "
              "but with a noticeable gap vs. the oracle.")
    else:
        print("  Cross-dataset predictor does NOT transfer (AUROC at chance).")


if __name__ == "__main__":
    main()
