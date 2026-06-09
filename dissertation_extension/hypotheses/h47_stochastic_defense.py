"""
Hypothesis H47: Stochastic input perturbation defence — at inference, add small
Gaussian noise to input and average K predictions (similar to randomized
smoothing; Liu et al. 2018, Cohen et al. 2019). We test whether this stochastic
defence recovers a DIFFERENT subset of adversarial samples than deterministic
defences, and whether per-sample recovery is predictable from simple features.

Pipeline:
  1. Train CNN victim on Fashion-MNIST (10 epochs).
  2. Generate FGSM adversarial samples at eps=15/255.
  3. Defence: for each adv sample, draw K=32 Gaussian noise samples for each
     sigma in {0.05, 0.10, 0.25}, average softmax, take argmax. Record per-sample
     whether the defence recovers the true label.
  4. Compute per-sample features: victim_margin (on clean), mean_pix, std_pix,
     sobel_mean.
  5. Univariate AUROC of each feature for predicting recovery, per sigma.
  6. Optimal-sigma-per-sample analysis: for each sample, which sigma recovers it?
     Correlate optimal-sigma with features.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

DEVICE = torch.device("cuda")
DATA_ROOT = "/tmp/data"
EPS = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
K = 32
SIGMAS = [0.05, 0.10, 0.25]


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


def train_victim(train_set):
    torch.manual_seed(0); np.random.seed(0)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done ({time.time()-t0:.1f}s)")
    model.eval()
    return model


def fgsm(model, x, y, eps=EPS):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()


def victim_margin(model, x):
    with torch.no_grad():
        logits = model(x)
    sorted_l, _ = logits.sort(1, descending=True)
    return (sorted_l[:, 0] - sorted_l[:, 1]).cpu().numpy()


def sobel_mean(x):
    # x: (N,1,28,28) tensor in [0,1]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return g.mean(dim=(1, 2, 3)).cpu().numpy()


def stochastic_defence(model, x_adv, sigma, K_samples=K, batch_max=256):
    """For each adv sample, average softmax over K noisy copies; return argmax."""
    N = x_adv.size(0)
    preds = torch.zeros(N, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for i in range(0, N, batch_max):
            xb = x_adv[i:i + batch_max]
            bn = xb.size(0)
            # repeat K times along batch dim
            xb_rep = xb.unsqueeze(1).expand(bn, K_samples, *xb.shape[1:]).reshape(bn * K_samples, *xb.shape[1:])
            noise = torch.randn_like(xb_rep) * sigma
            xn = (xb_rep + noise).clamp(0, 1)
            logits = model(xn)
            probs = F.softmax(logits, 1).view(bn, K_samples, -1).mean(dim=1)
            preds[i:i + batch_max] = probs.argmax(1)
    return preds


def auroc(score, y):
    if len(np.unique(y)) < 2:
        return float('nan')
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def main():
    print("##### H47: stochastic input perturbation defence #####")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print("Training victim CNN ...")
    t0 = time.time()
    model = train_victim(train_set)
    print(f"  total train time {time.time()-t0:.1f}s")

    # Load test set into device
    x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = x.size(0)
    print(f"Test set size: {N}")

    # Restrict to correctly classified samples
    with torch.no_grad():
        clean_pred = []
        for i in range(0, N, 512):
            clean_pred.append(model(x[i:i+512]).argmax(1))
        clean_pred = torch.cat(clean_pred)
    correct = clean_pred == y
    x_c, y_c = x[correct], y[correct]
    print(f"Correctly classified: {x_c.size(0)}")

    # Compute victim margin on clean (a deterministic-defence-style feature)
    print("Computing victim_margin (clean) ...")
    margin = []
    for i in range(0, x_c.size(0), 512):
        margin.append(victim_margin(model, x_c[i:i+512]))
    margin = np.concatenate(margin)

    # Generate FGSM adversarials
    print(f"Generating FGSM adversarials at eps={EPS:.4f} ...")
    adv_list = []
    for i in range(0, x_c.size(0), 512):
        adv_list.append(fgsm(model, x_c[i:i+512], y_c[i:i+512]))
    x_adv = torch.cat(adv_list, 0)

    # Confirm attack success
    with torch.no_grad():
        adv_pred = []
        for i in range(0, x_adv.size(0), 512):
            adv_pred.append(model(x_adv[i:i+512]).argmax(1))
        adv_pred = torch.cat(adv_pred)
    flipped = (adv_pred != y_c)
    print(f"FGSM flipped: {flipped.float().mean().item():.4f}")

    # We test defence per-sample on ALL adv samples (including unflipped). The
    # "recovery" target is: defence prediction == true label.
    # Per-sample features on the adversarial input pixels
    print("Computing per-sample features on adv inputs (mean_pix, std_pix, sobel_mean) ...")
    mean_pix = x_adv.mean(dim=(1, 2, 3)).cpu().numpy()
    std_pix = x_adv.std(dim=(1, 2, 3)).cpu().numpy()
    sobel = []
    for i in range(0, x_adv.size(0), 512):
        sobel.append(sobel_mean(x_adv[i:i+512]))
    sobel = np.concatenate(sobel)

    # Stochastic defence per sigma
    y_np = y_c.cpu().numpy()
    recoveries = {}  # sigma -> bool array
    print("Running stochastic defence ...")
    torch.manual_seed(123)
    for sigma in SIGMAS:
        t0 = time.time()
        pred = stochastic_defence(model, x_adv, sigma)
        rec = (pred == y_c).cpu().numpy().astype(int)
        recoveries[sigma] = rec
        print(f"  sigma={sigma}:  recovery rate = {rec.mean():.4f}   "
              f"({time.time()-t0:.1f}s)")

    # Also baseline: deterministic defence = no defence (raw adv prediction).
    # Per spec, we contrast stochastic recoveries against deterministic.
    det_correct = (adv_pred == y_c).cpu().numpy().astype(int)
    print(f"  no-defence baseline correct rate = {det_correct.mean():.4f}")

    # Subset comparison: which adv samples does stochastic recover that
    # deterministic does not, and vice versa?
    print("\n--- subset overlap (deterministic vs stochastic recovery) ---")
    for sigma in SIGMAS:
        rec = recoveries[sigma]
        only_stoch = ((rec == 1) & (det_correct == 0)).sum()
        only_det = ((rec == 0) & (det_correct == 1)).sum()
        both = ((rec == 1) & (det_correct == 1)).sum()
        neither = ((rec == 0) & (det_correct == 0)).sum()
        print(f"  sigma={sigma}: both={both}  only_stoch={only_stoch}  "
              f"only_det={only_det}  neither={neither}")

    # Best sigma overall
    best_sigma = max(SIGMAS, key=lambda s: recoveries[s].mean())
    print(f"\nBest single sigma overall: {best_sigma}  "
          f"(recovery rate {recoveries[best_sigma].mean():.4f})")

    # Univariate AUROC for predicting recovery, per sigma
    feat_names = ["victim_margin", "mean_pix", "std_pix", "sobel_mean"]
    feats = np.stack([margin, mean_pix, std_pix, sobel], axis=1)
    print("\n--- Univariate AUROC: predicting per-sample stochastic recovery ---")
    print(f"{'sigma':>6}  " + "  ".join(f"{n:>14}" for n in feat_names))
    for sigma in SIGMAS:
        rec = recoveries[sigma]
        aurocs = [auroc(feats[:, i], rec) for i in range(feats.shape[1])]
        print(f"{sigma:>6}  " + "  ".join(f"{a:>14.4f}" for a in aurocs))

    # Per-sample optimal sigma: among the samples recovered by at least one
    # sigma, which sigma recovered it (lowest sigma that succeeded). For samples
    # recovered by none, mark as -1 (excluded from correlation).
    rec_matrix = np.stack([recoveries[s] for s in SIGMAS], axis=1)  # (N, 3)
    any_rec = rec_matrix.any(axis=1)
    # lowest-sigma-that-works
    first_works = np.argmax(rec_matrix, axis=1)  # gives 0 if none, hence guard
    opt_sigma = np.where(any_rec, np.array(SIGMAS)[first_works], np.nan)
    n_any = int(any_rec.sum())
    print(f"\nAny-sigma recovery: {n_any} / {len(any_rec)} = "
          f"{n_any / len(any_rec):.4f}")
    # Spearman correlation between optimal sigma and each feature
    print("\n--- Spearman correlation of optimal_sigma vs feature "
          "(restricted to recoverable samples) ---")
    mask = any_rec
    for i, n in enumerate(feat_names):
        rho, p = spearmanr(opt_sigma[mask], feats[mask, i])
        print(f"  {n:<16} rho={rho:+.4f}  p={p:.3e}")

    # Also: feature means by optimal sigma bucket
    print("\n--- Feature means grouped by optimal sigma ---")
    print(f"{'opt_sigma':>10}  {'n':>6}  " +
          "  ".join(f"{n:>14}" for n in feat_names))
    for s in SIGMAS:
        m = mask & (opt_sigma == s)
        if m.sum() == 0:
            continue
        means = feats[m].mean(axis=0)
        print(f"{s:>10}  {int(m.sum()):>6}  " +
              "  ".join(f"{v:>14.4f}" for v in means))

    # And: recovery-rate-per-sigma stratified by victim_margin quartile to show
    # whether stochastic defence helps low-margin or high-margin samples more.
    print("\n--- Recovery rate per sigma, stratified by victim_margin quartile ---")
    qs = np.quantile(margin, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"{'quartile':>10}  {'n':>6}  " +
          "  ".join(f"sigma={s:<5}" for s in SIGMAS) + "    det")
    for qi in range(4):
        lo, hi = qs[qi], qs[qi + 1]
        if qi < 3:
            qmask = (margin >= lo) & (margin < hi)
        else:
            qmask = (margin >= lo) & (margin <= hi)
        if qmask.sum() == 0:
            continue
        rates = [recoveries[s][qmask].mean() for s in SIGMAS]
        det_rate = det_correct[qmask].mean()
        print(f"   Q{qi+1}  [{lo:.2f},{hi:.2f}]  {int(qmask.sum()):>6}  " +
              "  ".join(f"{r:>11.4f}" for r in rates) +
              f"    {det_rate:.4f}")

    print("\n##### H47 complete #####")


if __name__ == "__main__":
    main()
