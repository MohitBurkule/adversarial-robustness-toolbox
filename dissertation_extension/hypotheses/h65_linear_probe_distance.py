"""
H65: Linear probe distance vs. logit margin for predicting adversarial vulnerability.

Hypothesis:
    A linear classifier on penultimate (128-d) features predicts adversarial
    vulnerability by the signed margin-to-its-own-decision-boundary, which may
    be a cleaner geometric signal than the full-model logit margin (which is
    distorted by the final non-linear layer's interaction with the rest of the
    network).

Pipeline:
    1. Train a small CNN matching the architecture in diagnostic_test.py on
       Fashion-MNIST for 10 epochs.
    2. Extract penultimate (128-d) features for the test set.
    3. Fit an sklearn LogisticRegression (multinomial) on those features using
       the true labels, giving a "linear probe" head.
    4. For each test sample compute:
         - lp_margin   : signed distance to the linear probe's decision
                         boundary, computed as the margin between the probe's
                         decision function for the model's predicted class and
                         the runner-up class (in the probe's geometry).
         - cnn_margin  : full-model logit margin (top1 - top2).
         - mean_pix    : per-image mean pixel intensity.
         - std_pix     : per-image pixel std.
    5. Generate FGSM and PGD adversarial examples; record per-sample flip
       indicators (flipped_FGSM, flipped_PGD) as binary targets.
    6. Report univariate AUROC of each feature against each target.

Run notes:
    This script is self-contained. It DOES NOT execute attacks via ART; FGSM
    and PGD are implemented inline against the trained CNN for simplicity and
    speed. It restricts the analysis to test samples that the CNN classifies
    correctly (vulnerability is only well-defined there).
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = EPS / 8.0
SEED = 0


class CNN(nn.Module):
    """Matches diagnostic_test.py CNN exactly; exposes penultimate features."""
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
        x = F.relu(self.fc1(x))  # 128-d penultimate
        return x

    def forward(self, x):
        h = self.features(x)
        h = self.do2(h)
        return self.fc2(h)


def train_model(train_set):
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


def collect_test_tensors(test_set):
    xs = torch.stack([test_set[i][0] for i in range(len(test_set))])
    ys = torch.tensor([test_set[i][1] for i in range(len(test_set))])
    return xs.to(DEVICE), ys.to(DEVICE)


@torch.no_grad()
def batched_forward(model, x, fn, batch=512):
    outs = []
    for i in range(0, x.size(0), batch):
        outs.append(fn(x[i:i + batch]))
    return torch.cat(outs, 0)


def fgsm_attack(model, x, y, eps=EPS):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    sign = x_adv.grad.sign().detach()
    return (x + eps * sign).clamp(0, 1).detach()


def pgd_attack(model, x, y, eps=EPS, alpha=PGD_ALPHA, steps=PGD_STEPS):
    # random start within L_inf ball
    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = (x + delta).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # project to L_inf eps-ball around x
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)
    return x_adv.detach()


def attack_flip(model, x, y, attack_fn, batch=256):
    flips = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        adv = attack_fn(model, xb, yb)
        with torch.no_grad():
            pred = model(adv).argmax(1)
        flips.append((pred != yb).cpu())
    return torch.cat(flips, 0).numpy().astype(int)


def cnn_margin_topk(model, x, batch=512):
    """Return logit-margin top1-top2 and the predicted class."""
    with torch.no_grad():
        logits = batched_forward(model, x, lambda b: model(b), batch=batch)
    sorted_l, _ = logits.sort(1, descending=True)
    margin = (sorted_l[:, 0] - sorted_l[:, 1]).cpu().numpy()
    pred = logits.argmax(1).cpu().numpy()
    return margin, pred


def extract_features(model, x, batch=512):
    with torch.no_grad():
        feats = batched_forward(model, x, lambda b: model.features(b), batch=batch)
    return feats.cpu().numpy()


def linear_probe_margins(probe, feats_np, pred):
    """Signed distance to the probe's decision boundary, defined as the gap
    between the probe's decision_function for the CNN's predicted class and
    its runner-up class in the probe's geometry.

    decision_function returns w_k . phi + b_k for each class k (up to a shared
    softmax normalisation). The difference (top - runner-up) is proportional
    to the signed perpendicular distance to the boundary that separates those
    two classes in the 128-d penultimate space (modulo ||w_top - w_runner||).
    """
    df = probe.decision_function(feats_np)  # (N, 10)
    N = df.shape[0]
    # margin between CNN-predicted class and runner-up in probe's scores
    own_score = df[np.arange(N), pred]
    df_masked = df.copy()
    df_masked[np.arange(N), pred] = -np.inf
    runner = df_masked.max(1)
    # divide by ||w_pred - w_runner|| to convert to true geometric distance
    runner_idx = df_masked.argmax(1)
    W = probe.coef_  # (10, 128)
    diffs = W[pred] - W[runner_idx]  # (N, 128)
    norms = np.linalg.norm(diffs, axis=1) + 1e-12
    return (own_score - runner) / norms


def univariate_auroc(score, target):
    if target.std() == 0:
        return float("nan")
    a = roc_auc_score(target, score)
    return max(a, 1 - a)


def main():
    print(f"device = {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("training CNN ...")
    model = train_model(train_set)

    x_test, y_test = collect_test_tensors(test_set)
    print(f"test set: {x_test.size(0)} samples")

    # CNN margin + predictions
    cnn_margin, pred = cnn_margin_topk(model, x_test)
    y_np = y_test.cpu().numpy()
    correct = (pred == y_np)
    print(f"clean test accuracy = {correct.mean():.4f}")

    # restrict to correctly-classified samples
    idx_correct = np.where(correct)[0]
    x_c = x_test[idx_correct]
    y_c = y_test[idx_correct]

    # penultimate features
    print("extracting penultimate features ...")
    feats_train = []
    feats_train_y = []
    # for the linear probe we train on the TRAIN-set penultimate features
    train_loader = DataLoader(train_set, 512, shuffle=False, num_workers=2)
    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            feats_train.append(model.features(xb).cpu().numpy())
            feats_train_y.append(yb.numpy())
    feats_train = np.concatenate(feats_train, 0)
    feats_train_y = np.concatenate(feats_train_y, 0)
    feats_test = extract_features(model, x_test)

    print("fitting LogisticRegression probe on 128-d penultimate features ...")
    probe = LogisticRegression(max_iter=2000, multi_class="multinomial", n_jobs=-1)
    probe.fit(feats_train, feats_train_y)
    print(f"  probe train acc = {probe.score(feats_train, feats_train_y):.4f}")
    print(f"  probe test  acc = {probe.score(feats_test, y_np):.4f}")

    # linear probe margin (signed geometric distance to its boundary, using
    # CNN's predicted class as the "own" side)
    lp_margin_all = linear_probe_margins(probe, feats_test, pred)

    # restrict to correct samples
    lp_margin = lp_margin_all[idx_correct]
    cnn_margin_c = cnn_margin[idx_correct]
    x_c_np = x_c.cpu().numpy().reshape(x_c.size(0), -1)
    mean_pix = x_c_np.mean(1)
    std_pix = x_c_np.std(1)

    # adversarial targets
    print("running FGSM ...")
    flipped_FGSM = attack_flip(model, x_c, y_c, fgsm_attack)
    print(f"  FGSM flip rate = {flipped_FGSM.mean():.4f}")
    print("running PGD ...")
    flipped_PGD = attack_flip(model, x_c, y_c, pgd_attack)
    print(f"  PGD flip rate  = {flipped_PGD.mean():.4f}")

    # univariate AUROC
    feature_dict = {
        "lp_margin":  lp_margin,
        "cnn_margin": cnn_margin_c,
        "mean_pix":   mean_pix,
        "std_pix":    std_pix,
    }
    targets = {
        "flipped_FGSM": flipped_FGSM,
        "flipped_PGD":  flipped_PGD,
    }
    print("\n===== Univariate AUROC =====")
    print(f"{'feature':<12} " + " ".join(f"{t:>16}" for t in targets))
    for fname, fvals in feature_dict.items():
        cells = []
        for tname, tvals in targets.items():
            a = univariate_auroc(fvals, tvals)
            cells.append(f"{a:>16.4f}")
        print(f"{fname:<12} " + " ".join(cells))

    # Pearson correlation between lp_margin and cnn_margin for context
    if lp_margin.std() > 0 and cnn_margin_c.std() > 0:
        r = np.corrcoef(lp_margin, cnn_margin_c)[0, 1]
        print(f"\ncorr(lp_margin, cnn_margin) = {r:+.4f}")


if __name__ == "__main__":
    main()
