"""
H182 - Learned AUROC ceiling (gradient-boosted meta-predictor) + bootstrap CIs + calibration (ECE).

Motivation (advisor critique, Paper 1 statistical rigour):
  Paper 1 reports differences like "0.9743 vs 0.9723" (margin vs gradient norm) as if meaningful,
  with no confidence intervals or significance test, and never establishes the AUROC ceiling that a
  learned combination of all features achieves. Three fixes:
    (1) Bootstrap 95% CIs on every univariate AUROC, and a paired-bootstrap test of whether margin
        is significantly better than the next-best feature.
    (2) A gradient-boosted meta-predictor over ALL features (5-fold CV) = the learned AUROC ceiling;
        how much head-room is there above the single best feature?
    (3) ECE (Expected Calibration Error) of the meta-predictor, since downstream uses (selective
        prediction) need calibrated scores, not just ranking.

Features (per finally-correct eval sample): margin, input_grad_norm, smoothgrad_norm,
softmax_confidence, logit_entropy, min_eps. Label = PGD-10 flip.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10
EVAL_N = 2000
SEED = 0


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


def train(train_set, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return model


def pgd_flip(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=10):
    model.eval()
    adv = (x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        with torch.no_grad():
            adv = (adv + alpha * adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def compute_features(model, x, y):
    """Return dict of per-sample feature arrays."""
    model.eval()
    # margin, confidence, entropy
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, 1)
        s, _ = logits.sort(1, descending=True)
        margin = (s[:, 0] - s[:, 1]).cpu().numpy()
        conf = probs.max(1).values.cpu().numpy()
        ent = (-(probs * (probs.clamp_min(1e-12)).log()).sum(1)).cpu().numpy()
    # input grad norm
    xr = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xr), y)
    g, = torch.autograd.grad(loss, xr)
    gnorm = g.flatten(1).norm(dim=1).detach().cpu().numpy()
    # smoothgrad norm (K=25, sigma=0.1)
    acc = torch.zeros_like(x)
    for _ in range(25):
        xn = (x + torch.randn_like(x) * 0.1).clamp(0, 1).detach().requires_grad_(True)
        l = F.cross_entropy(model(xn), y)
        gg, = torch.autograd.grad(l, xn)
        acc += gg.detach()
    sg = (acc / 25).flatten(1).norm(dim=1).cpu().numpy()
    # min eps (FGSM direction binary search)
    sign = g.sign().detach()
    lo = torch.zeros(x.size(0), device=DEVICE); hi = torch.full((x.size(0),), 0.3, device=DEVICE)
    for _ in range(12):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            fl = (model(adv).argmax(1) != y)
        hi = torch.where(fl, mid, hi); lo = torch.where(fl, lo, mid)
    min_eps = hi.cpu().numpy()
    return {
        "margin": margin, "input_grad_norm": gnorm, "smoothgrad_norm": sg,
        "softmax_conf": conf, "logit_entropy": ent, "min_eps": min_eps,
    }


def auroc_dir(label, score):
    """AUROC taking the better orientation; return (auroc, sign) where sign=+1 means high->flip."""
    a = roc_auc_score(label, score)
    return (a, +1) if a >= 0.5 else (1 - a, -1)


def bootstrap_ci(label, score, sign, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    N = len(label)
    vals = []
    s = score * sign
    for _ in range(n_boot):
        i = rng.randint(0, N, N)
        if label[i].std() == 0:
            continue
        vals.append(roc_auc_score(label[i], s[i]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def paired_bootstrap_test(label, s1, sign1, s2, sign2, n_boot=2000, seed=0):
    """P(AUROC(s1) <= AUROC(s2)) under bootstrap -> one-sided p that margin is NOT better."""
    rng = np.random.RandomState(seed)
    N = len(label)
    a, b = s1 * sign1, s2 * sign2
    diffs = []
    for _ in range(n_boot):
        i = rng.randint(0, N, N)
        if label[i].std() == 0:
            continue
        diffs.append(roc_auc_score(label[i], a[i]) - roc_auc_score(label[i], b[i]))
    diffs = np.array(diffs)
    return float((diffs <= 0).mean()), float(diffs.mean())


def expected_calibration_error(probs, label, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for k in range(n_bins):
        m = (probs >= bins[k]) & (probs < bins[k + 1] if k < n_bins - 1 else probs <= bins[k + 1])
        if m.sum() == 0:
            continue
        acc = label[m].mean()
        conf = probs[m].mean()
        ece += (m.sum() / len(probs)) * abs(acc - conf)
    return ece


def main():
    print("=" * 74)
    print("H182 - Learned AUROC ceiling (GBM) + bootstrap CIs + ECE")
    print("=" * 74)
    print(f"Device={DEVICE}  EVAL_N={EVAL_N}  EPS={EPS:.4f}")

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n--- Training vanilla CNN ---"); t0 = time.time()
    model = train(train_set); print(f"  {time.time()-t0:.1f}s")
    model.eval()
    with torch.no_grad():
        correct = (model(test_x).argmax(1) == test_y)
    idx = correct.nonzero(as_tuple=True)[0][:EVAL_N]
    x, y = test_x[idx], test_y[idx]

    feats = {}
    for i in range(0, x.size(0), 256):
        fb = compute_features(model, x[i:i+256], y[i:i+256])
        for k, v in fb.items():
            feats.setdefault(k, []).append(v)
    feats = {k: np.concatenate(v) for k, v in feats.items()}

    flips = []
    for i in range(0, x.size(0), 256):
        flips.append(pgd_flip(model, x[i:i+256], y[i:i+256]).cpu())
    label = torch.cat(flips).numpy().astype(int)
    print(f"  eval n={len(label)}  PGD ASR={label.mean():.4f}")

    if label.std() == 0:
        print("  Label saturated; AUROC undefined. (Use H175 matched-ASR conditions.)")
        return

    names = list(feats.keys())
    print("\n--- Univariate AUROC with bootstrap 95% CI ---")
    print(f"  {'feature':<18} {'AUROC':>7} {'95% CI':>20}")
    stats = {}
    for n in names:
        a, sgn = auroc_dir(label, feats[n])
        lo, hi = bootstrap_ci(label, feats[n], sgn)
        stats[n] = (a, sgn)
        print(f"  {n:<18} {a:>7.4f}   [{lo:.4f}, {hi:.4f}]")

    # margin vs next-best
    ranked = sorted(names, key=lambda n: stats[n][0], reverse=True)
    best, second = ranked[0], ranked[1]
    p, mdiff = paired_bootstrap_test(label, feats[best], stats[best][1],
                                     feats[second], stats[second][1])
    print(f"\n  best={best} ({stats[best][0]:.4f})  vs  second={second} ({stats[second][0]:.4f})")
    print(f"  paired-bootstrap mean AUROC diff = {mdiff:+.4f};  "
          f"P(best not better) = {p:.3f}  "
          f"-> {'INDISTINGUISHABLE' if p > 0.05 else 'significant'}")

    # GBM ceiling (5-fold CV)
    X = np.column_stack([feats[n] for n in names])
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=SEED)
    oof = cross_val_predict(gbm, X, label, cv=skf, method="predict_proba")[:, 1]
    ceiling = roc_auc_score(label, oof)
    print(f"\n--- Learned ceiling (GBM over all {len(names)} features, 5-fold CV) ---")
    print(f"  GBM CV-AUROC = {ceiling:.4f}")
    print(f"  head-room over best single feature ({best}) = {ceiling - stats[best][0]:+.4f}")
    print(f"  ECE of GBM probabilities = {expected_calibration_error(oof, label):.4f}")

    print("\n" + "=" * 74)
    print("Takeaways for Paper 1: report CIs (many features are statistically indistinguishable),")
    print("the GBM head-room quantifies how much a learned combo beats the best single predictor,")
    print("and ECE shows whether the scores are usable for selective prediction.")
    print("=" * 74)


if __name__ == "__main__":
    main()
