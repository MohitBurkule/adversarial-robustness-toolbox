"""
H90: Energy score (Liu et al., NeurIPS 2020) as an OOD / atypicality detector
that may also predict adversarial vulnerability.

Energy score:  E(x) = -logsumexp_j z_j(x)
where z(x) is the pre-softmax logit vector. Liu et al. show that E(x) is a
better OOD detector than softmax max-prob (Hendrycks & Gimpel). The hypothesis
here is that atypical / near-distribution samples (high E, i.e. low logsumexp)
should also be more susceptible to L_inf adversarial perturbation, because the
classifier has lower overall confidence "mass" at those inputs.

Pipeline:
  1. Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
  2. For every correctly-classified test sample compute:
        - energy_score  = -logsumexp(logits)
        - softmax_max   = max softmax probability (baseline OOD score)
        - margin        = top1 - top2 logit
        - mean_pix      = pixel intensity mean
        - std_pix       = pixel intensity std
  3. Vulnerability targets per sample:
        - FGSM flip @ eps=15/255
        - PGD flip  @ eps=15/255
        - min_eps to flip (FGSM L_inf binary search)
  4. Univariate AUROC for each feature vs each binary target; Pearson with min_eps.
  5. Multivariate ablation: logistic regression with all features vs. an ablated
     model with energy_score removed (delta-AUROC quantifies its incremental signal
     beyond margin / softmax_max / pixel stats).

Self-contained: run with `python h90_energy_score.py`. Writes nothing to disk.
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
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0
SEED = 0


# ----- model (matches diagnostic_test.py) -----
class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2); x = self.do1(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.do2(x); return self.fc2(x)


def train_model(train_set, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done")
    model.eval()
    return model


def collect_logits(model, x):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), 512):
            out.append(model(x[i:i+512]))
    return torch.cat(out, 0)


# ----- attacks -----
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x_orig = x.clone().detach()
    adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def batched_attack(fn, model, x, y, batch=512):
    out = []
    for i in range(0, x.size(0), batch):
        out.append(fn(model, x[i:i+batch], y[i:i+batch]))
    return torch.cat(out)


# ----- main -----
def main():
    print("loading Fashion-MNIST...")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print(f"training CNN for {EPOCHS} epochs...")
    t0 = time.time()
    model = train_model(train_set)
    print(f"  train time: {time.time()-t0:.1f}s")

    print("collecting test logits...")
    logits = collect_logits(model, test_x)
    preds = logits.argmax(1)
    correct = preds == test_y
    print(f"clean test accuracy: {correct.float().mean().item():.4f}")

    x_c = test_x[correct]
    y_c = test_y[correct]
    logits_c = logits[correct]

    # ----- features -----
    # Energy score, Liu et al. NeurIPS 2020:  E(x) = -logsumexp(z)
    energy_score = -torch.logsumexp(logits_c, dim=1)
    softmax_max = F.softmax(logits_c, dim=1).max(1).values
    logit_sorted, _ = logits_c.sort(1, descending=True)
    margin = logit_sorted[:, 0] - logit_sorted[:, 1]
    flat = x_c.view(x_c.size(0), -1)
    mean_pix = flat.mean(1)
    std_pix = flat.std(1)

    feats = {
        "energy_score": energy_score,
        "softmax_max":  softmax_max,
        "margin":       margin,
        "mean_pix":     mean_pix,
        "std_pix":      std_pix,
    }

    # ----- targets -----
    print("computing FGSM flips...")
    fgsm_flip = batched_attack(fgsm_attack, model, x_c, y_c)
    print(f"  FGSM flip rate: {fgsm_flip.float().mean().item():.3f}")

    print("computing PGD flips...")
    pgd_flip = batched_attack(pgd_attack, model, x_c, y_c)
    print(f"  PGD flip rate: {pgd_flip.float().mean().item():.3f}")

    print("computing min_eps via FGSM binary search...")
    me = []
    for i in range(0, x_c.size(0), 512):
        me.append(min_eps_to_flip(model, x_c[i:i+512], y_c[i:i+512]))
    min_eps = torch.cat(me)
    print(f"  mean min_eps: {min_eps.mean().item():.4f}")

    binary_targets = {
        "FGSM_flip": fgsm_flip.detach().cpu().numpy().astype(int),
        "PGD_flip":  pgd_flip.detach().cpu().numpy().astype(int),
    }
    min_eps_np = min_eps.detach().cpu().numpy()

    # ----- univariate AUROC + Pearson with min_eps -----
    print("\n========== H90: univariate AUROC / Pearson ==========")
    print(f"{'feature':<16} " +
          " ".join(f"{tn:>12}" for tn in binary_targets) +
          f" {'corr(min_eps)':>16}")
    feat_np = {}
    for fname, fval in feats.items():
        fnp = fval.detach().cpu().numpy()
        feat_np[fname] = fnp
        row = [f"{fname:<16}"]
        for tn, y in binary_targets.items():
            if y.std() == 0:
                row.append(f"{'n/a':>12}")
                continue
            a = roc_auc_score(y, fnp)
            a = max(a, 1 - a)
            row.append(f"{a:>12.4f}")
        if min_eps_np.std() > 0 and fnp.std() > 0:
            cor = float(np.corrcoef(fnp, min_eps_np)[0, 1])
        else:
            cor = float("nan")
        row.append(f"{cor:>+16.4f}")
        print(" ".join(row))

    # ----- multivariate ablation -----
    print("\n========== H90: multivariate ablation (logistic regression) ==========")
    feat_names = list(feats.keys())
    X = np.stack([feat_np[n] for n in feat_names], axis=1)
    Xs = StandardScaler().fit_transform(X)
    energy_idx = feat_names.index("energy_score")
    Xs_no_energy = np.delete(Xs, energy_idx, axis=1)

    print(f"{'target':<12} {'full':>8} {'no_energy':>10} {'delta':>10}")
    for tn, y in binary_targets.items():
        if y.std() == 0:
            print(f"{tn:<12} degenerate target")
            continue
        full = LogisticRegression(max_iter=2000).fit(Xs, y)
        full_auc = roc_auc_score(y, full.predict_proba(Xs)[:, 1])
        red = LogisticRegression(max_iter=2000).fit(Xs_no_energy, y)
        red_auc = roc_auc_score(y, red.predict_proba(Xs_no_energy)[:, 1])
        print(f"{tn:<12} {full_auc:>8.4f} {red_auc:>10.4f} {full_auc-red_auc:>+10.4f}")
        print(f"  standardised coefficients (full model):")
        for n, c in zip(feat_names, full.coef_.flatten()):
            print(f"    {n:<16} {c:+.4f}")

    print("\n========== summary ==========")
    print(f"correct test samples used: {int(correct.sum())}")
    print(f"FGSM flip rate: {fgsm_flip.float().mean().item():.3f}")
    print(f"PGD flip rate:  {pgd_flip.float().mean().item():.3f}")
    print(f"mean min_eps:   {min_eps.mean().item():.4f}")
    print(f"energy_score range: [{energy_score.min().item():.3f}, "
          f"{energy_score.max().item():.3f}]")


if __name__ == "__main__":
    main()
