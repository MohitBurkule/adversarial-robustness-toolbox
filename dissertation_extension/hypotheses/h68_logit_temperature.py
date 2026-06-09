"""
H68: Logit temperature calibration as a vulnerability predictor.

Hypothesis: temperature-scaled softmax confidence (Platt-style calibration via
optimal temperature T learned by NLL minimization on a held-out calibration
split) yields a per-sample uncertainty estimate that may predict adversarial
vulnerability better than raw margin.

Pipeline:
  1. Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
  2. Split test set into calibration / evaluation halves. Find optimal T on the
     calibration split by minimizing NLL of softmax(logits / T).
  3. For each evaluation sample, compute:
        - calibrated_softmax_max
        - calibrated_margin              (top1 - top2 of calibrated probs)
        - calibration_correction         (orig softmax max - calibrated softmax max)
        - raw margin (logit top1 - top2)
        - mean pixel intensity
        - std pixel intensity
  4. Compute three vulnerability targets: FGSM flip, PGD flip, min_eps to flip (FGSM bsearch).
  5. Univariate AUROC for each feature against each binary target; Pearson with min_eps.

Self-contained: run with `python h68_logit_temperature.py`. Writes nothing to disk.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

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


# ----- temperature calibration: minimize NLL via LBFGS over scalar T -----
def fit_temperature(logits, labels, max_iter=200):
    T = torch.ones(1, device=logits.device, requires_grad=True)
    optim = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optim.zero_grad()
        loss = nll(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss
    optim.step(closure)
    return T.detach().clamp(min=1e-3).item()


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
    N = test_x.size(0)

    # calibration / evaluation split (deterministic)
    g = torch.Generator().manual_seed(123)
    perm = torch.randperm(N, generator=g)
    n_cal = N // 2
    cal_idx = perm[:n_cal].to(DEVICE)
    eval_idx = perm[n_cal:].to(DEVICE)

    print(f"training CNN for {EPOCHS} epochs...")
    t0 = time.time()
    model = train_model(train_set)
    print(f"  train time: {time.time()-t0:.1f}s")

    # collect logits on full test set
    print("collecting logits...")
    logits_all = collect_logits(model, test_x)

    cal_logits = logits_all[cal_idx]
    cal_labels = test_y[cal_idx]
    eval_logits = logits_all[eval_idx]
    eval_labels = test_y[eval_idx]
    eval_x = test_x[eval_idx]

    # fit T
    print("fitting temperature via NLL minimization on calibration split...")
    T = fit_temperature(cal_logits, cal_labels)
    print(f"  optimal T = {T:.4f}")

    # original & calibrated softmax stats on eval split
    orig_probs = F.softmax(eval_logits, dim=1)
    cal_probs = F.softmax(eval_logits / T, dim=1)

    orig_max = orig_probs.max(1).values
    cal_sorted, _ = cal_probs.sort(1, descending=True)
    cal_max = cal_sorted[:, 0]
    cal_margin_prob = cal_sorted[:, 0] - cal_sorted[:, 1]
    calibration_correction = orig_max - cal_max

    # raw logit margin
    logit_sorted, _ = eval_logits.sort(1, descending=True)
    logit_margin = logit_sorted[:, 0] - logit_sorted[:, 1]

    # pixel statistics
    flat = eval_x.view(eval_x.size(0), -1)
    mean_pix = flat.mean(1)
    std_pix = flat.std(1)

    # restrict to correctly-classified eval samples
    eval_preds = eval_logits.argmax(1)
    correct = eval_preds == eval_labels
    print(f"using {int(correct.sum())} correctly-classified eval samples "
          f"out of {eval_x.size(0)}")

    x_c = eval_x[correct]
    y_c = eval_labels[correct]

    feats = {
        "calibrated_softmax_max": cal_max[correct],
        "calibrated_margin_prob": cal_margin_prob[correct],
        "calibration_correction": calibration_correction[correct],
        "raw_logit_margin": logit_margin[correct],
        "orig_softmax_max": orig_max[correct],
        "mean_pix": mean_pix[correct],
        "std_pix": std_pix[correct],
    }

    # targets
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

    # univariate AUROC against binary targets, Pearson against min_eps
    binary_targets = {
        "FGSM_flip": fgsm_flip.detach().cpu().numpy().astype(int),
        "PGD_flip": pgd_flip.detach().cpu().numpy().astype(int),
    }
    min_eps_np = min_eps.detach().cpu().numpy()

    print("\n========== H68: univariate AUROC ==========")
    print(f"{'feature':<28} " +
          " ".join(f"{tn:>12}" for tn in binary_targets) +
          f" {'corr(min_eps)':>14}")
    for fname, fval in feats.items():
        fnp = fval.detach().cpu().numpy()
        row = [f"{fname:<28}"]
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
        row.append(f"{cor:>+14.4f}")
        print(" ".join(row))

    print("\n========== summary ==========")
    print(f"optimal T = {T:.4f}")
    print(f"eval samples (correctly classified): {int(correct.sum())}")
    print(f"FGSM flip rate: {fgsm_flip.float().mean().item():.3f}")
    print(f"PGD flip rate:  {pgd_flip.float().mean().item():.3f}")
    print(f"mean min_eps:   {min_eps.mean().item():.4f}")


if __name__ == "__main__":
    main()
