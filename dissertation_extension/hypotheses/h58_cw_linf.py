"""
H58: CW L_inf attack (untargeted) per-sample vulnerability.

Hypothesis:
  Per-sample susceptibility to the Carlini-Wagner attack under an L_inf
  constraint is predictable from cheap input/model statistics, and the
  ranking it induces overlaps -- but is not identical to -- the ranking
  induced by a one-step FGSM attack.

Pipeline:
  1. Train a small CNN (matching diagnostic_test.py's architecture) on
     Fashion-MNIST for 10 epochs.
  2. Implement CW L_inf (untargeted) using the tanh reparameterisation
     of the input and a binary search over the trade-off constant c.
     The L_inf budget is approximated through a tanh-mapped variable
     w whose deviation from atanh(2x - 1) is L_inf-clipped at each step.
  3. Targets:
       * cw_linf_at_eps15      -- binary; 1 if CW finds an adversarial
                                  example with L_inf <= 15/255 that flips
                                  the prediction.
       * cw_linf_magnitude     -- continuous; the L_inf magnitude of the
                                  smallest adversarial perturbation CW
                                  finds (or eps_max if it fails).
  4. Features:
       * margin                -- final-model logit margin.
       * mean_pix              -- per-sample mean pixel value.
       * std_pix               -- per-sample pixel standard deviation.
       * sobel_mean            -- mean Sobel gradient magnitude (edge mass).
       * jpeg_q75              -- L2 distance between the image and its
                                  JPEG-q75 round-tripped version (a cheap
                                  compressibility proxy).
  5. Cross-compare with FGSM: also run FGSM at eps=15/255 and compute
     univariate AUROCs of each feature against the FGSM-flip target,
     plus per-sample agreement (Jaccard / Cohen-kappa style) between
     CW and FGSM success sets.
  6. Univariate AUROC for each feature against each target.

This script is self-contained; run with `python h58_cw_linf.py`.
"""
import io
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128


# -----------------------------------------------------------------------------
# Model -- matches diagnostic_test.py exactly.
# -----------------------------------------------------------------------------
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


def train_model(train_set, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
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


# -----------------------------------------------------------------------------
# CW L_inf (untargeted) with tanh reparam + binary search over c.
# -----------------------------------------------------------------------------
def _atanh(x):
    # input in (0,1); map to R via 2x-1 -> atanh
    x = x.clamp(1e-6, 1 - 1e-6)
    u = 2 * x - 1
    return 0.5 * torch.log((1 + u) / (1 - u))


def cw_linf_attack(model, x, y, eps_max=EPS_TEST,
                   c_steps=5, c_lo=1e-2, c_hi=1e2,
                   iters=100, lr=5e-3, kappa=0.0):
    """Untargeted CW L_inf attack.

    Uses the tanh reparameterisation:
        x_adv = 0.5 * (tanh(w) + 1)
    and projects the perturbation x_adv - x to [-eps_max, eps_max] at each
    step. Binary search over the trade-off constant c balances the
    classification loss and the L_inf penalty (approximated by a soft
    hinge on max(|delta|) above an internal slack tau, which is shrunk
    each binary-search round).

    Returns
    -------
    best_adv : (B,1,28,28) tensor of best adversarial examples found
               within eps_max (or the original x if none was found).
    best_linf : (B,) tensor of the L_inf magnitudes (eps_max if failed).
    success  : (B,) bool, True if an adv example with L_inf <= eps_max
               that flips the prediction was found.
    """
    B = x.size(0)
    x = x.detach().to(DEVICE)
    y = y.detach().to(DEVICE)

    best_linf = torch.full((B,), float(eps_max), device=DEVICE)
    best_adv = x.clone()
    success = torch.zeros(B, dtype=torch.bool, device=DEVICE)

    # per-sample c, plus binary-search bounds
    c_lo_v = torch.full((B,), float(c_lo), device=DEVICE)
    c_hi_v = torch.full((B,), float(c_hi), device=DEVICE)
    c = torch.full((B,), float((c_lo * c_hi) ** 0.5), device=DEVICE)

    one_hot = F.one_hot(y, 10).float()

    for cs in range(c_steps):
        w = _atanh(x).clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)

        round_best_linf = torch.full((B,), float("inf"), device=DEVICE)
        round_best_adv = x.clone()
        round_success = torch.zeros(B, dtype=torch.bool, device=DEVICE)

        for it in range(iters):
            x_adv_raw = 0.5 * (torch.tanh(w) + 1.0)
            delta = (x_adv_raw - x).clamp(-eps_max, eps_max)
            x_adv = (x + delta).clamp(0.0, 1.0)

            logits = model(x_adv)
            true_l = (logits * one_hot).sum(1)
            other_l = (logits - 1e9 * one_hot).max(1).values
            # untargeted CW f(x): max(true - other, -kappa)
            f_loss = torch.clamp(true_l - other_l, min=-kappa)

            linf = delta.abs().flatten(1).max(1).values
            # soft L_inf penalty; encourages small perturbations.
            loss = (c * f_loss + linf).sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(1)
                flipped = pred != y
                better = flipped & (linf < round_best_linf)
                round_best_linf = torch.where(better, linf, round_best_linf)
                round_best_adv = torch.where(
                    better.view(-1, 1, 1, 1), x_adv.detach(), round_best_adv
                )
                round_success = round_success | flipped

        # update overall best
        with torch.no_grad():
            improved = round_success & (round_best_linf < best_linf)
            best_linf = torch.where(improved, round_best_linf, best_linf)
            best_adv = torch.where(
                improved.view(-1, 1, 1, 1), round_best_adv, best_adv
            )
            success = success | round_success

            # binary search on c: succeeded -> shrink c; failed -> grow c.
            c_hi_v = torch.where(round_success, torch.minimum(c_hi_v, c), c_hi_v)
            c_lo_v = torch.where(~round_success, torch.maximum(c_lo_v, c), c_lo_v)
            # geometric mean update
            c = torch.where(
                c_hi_v < 1e9,
                torch.sqrt(c_lo_v * c_hi_v),
                c * 10.0,
            )

    # samples that never succeeded: report eps_max as a censoring value
    best_linf = torch.where(success, best_linf, torch.full_like(best_linf, eps_max))
    return best_adv, best_linf, success


# -----------------------------------------------------------------------------
# FGSM (single-step) for cross-comparison.
# -----------------------------------------------------------------------------
def fgsm_flip(model, x, y, eps=EPS_TEST):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    sign = x.grad.sign().detach()
    adv = (x + eps * sign).clamp(0, 1).detach()
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


# -----------------------------------------------------------------------------
# Features.
# -----------------------------------------------------------------------------
_SOBEL_X = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]])
_SOBEL_Y = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]])


def sobel_mean(x):
    kx = _SOBEL_X.to(x.device)
    ky = _SOBEL_Y.to(x.device)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag.flatten(1).mean(1)


def jpeg_q75_dist(x):
    """L2 distance between x and its JPEG-q75 round-tripped reconstruction."""
    out = torch.zeros(x.size(0), device=x.device)
    x_cpu = (x.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    for i in range(x_cpu.shape[0]):
        img = Image.fromarray(x_cpu[i, 0], mode="L")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        rec = np.asarray(Image.open(buf), dtype=np.float32) / 255.0
        diff = rec - x_cpu[i, 0].astype(np.float32) / 255.0
        out[i] = float(np.sqrt((diff * diff).sum()))
    return out


def compute_features(model, x, y):
    N = x.size(0)
    margins = torch.zeros(N, device=DEVICE)
    means = torch.zeros(N, device=DEVICE)
    stds = torch.zeros(N, device=DEVICE)
    sobels = torch.zeros(N, device=DEVICE)
    jpegs = torch.zeros(N, device=DEVICE)
    with torch.no_grad():
        for i in range(0, N, 512):
            xb = x[i:i+512]
            yb = y[i:i+512]
            logits = model(xb)
            sorted_l, _ = logits.sort(1, descending=True)
            true_l = logits.gather(1, yb.view(-1, 1)).squeeze(1)
            # margin = true_class logit - best non-true logit
            other = logits.masked_fill(
                F.one_hot(yb, 10).bool(), -1e9
            ).max(1).values
            margins[i:i+512] = true_l - other
            flat = xb.flatten(1)
            means[i:i+512] = flat.mean(1)
            stds[i:i+512] = flat.std(1)
            sobels[i:i+512] = sobel_mean(xb)
            jpegs[i:i+512] = jpeg_q75_dist(xb)
    return torch.stack([margins, means, stds, sobels, jpegs], dim=1)


# -----------------------------------------------------------------------------
# Evaluation.
# -----------------------------------------------------------------------------
def univariate_aurocs(feats_np, y, names):
    print(f"  positive rate = {y.mean():.3f}  (n={len(y)})")
    for i, n in enumerate(names):
        xi = feats_np[:, i]
        if np.std(xi) == 0:
            print(f"    {n:<12} AUROC  (degenerate)")
            continue
        a = roc_auc_score(y, xi)
        a = max(a, 1 - a)
        print(f"    {n:<12} AUROC  {a:.4f}")


def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("Training small CNN on Fashion-MNIST (10 epochs)...")
    t0 = time.time()
    model = train_model(train_set, seed=0)
    print(f"  trained in {time.time()-t0:.1f}s")

    # Stack the test set as a single tensor.
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # Restrict to samples the model classifies correctly.
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds, 0)
    correct = preds == test_y
    x_c = test_x[correct]
    y_c = test_y[correct]
    print(f"Using {correct.sum().item()} correctly-classified samples.")

    print("Running CW L_inf attack ...")
    t0 = time.time()
    linf_all = []
    succ_all = []
    for i in range(0, x_c.size(0), 256):
        _, linf, succ = cw_linf_attack(
            model, x_c[i:i+256], y_c[i:i+256], eps_max=EPS_TEST,
        )
        linf_all.append(linf)
        succ_all.append(succ)
        if (i // 256) % 4 == 0:
            print(f"  CW batch {i//256+1}/{(x_c.size(0)+255)//256}  "
                  f"({time.time()-t0:.1f}s)")
    cw_linf_mag = torch.cat(linf_all)
    cw_at_eps15 = torch.cat(succ_all)
    print(f"  CW done in {time.time()-t0:.1f}s   "
          f"flip-rate@eps={EPS_TEST:.3f}: {cw_at_eps15.float().mean().item():.3f}")

    print("Running FGSM cross-comparison at eps=15/255 ...")
    t0 = time.time()
    fgsm_all = []
    for i in range(0, x_c.size(0), 512):
        fgsm_all.append(fgsm_flip(model, x_c[i:i+512], y_c[i:i+512], eps=EPS_TEST))
    fgsm_flip_v = torch.cat(fgsm_all)
    print(f"  FGSM done in {time.time()-t0:.1f}s   "
          f"flip-rate: {fgsm_flip_v.float().mean().item():.3f}")

    print("Computing features ...")
    t0 = time.time()
    feats = compute_features(model, x_c, y_c)
    print(f"  features done in {time.time()-t0:.1f}s")

    feat_names = ["margin", "mean_pix", "std_pix", "sobel_mean", "jpeg_q75"]
    feats_np = feats.detach().cpu().numpy()

    print("\n== Univariate AUROC: CW L_inf flip @ eps=15/255 ==")
    y_bin = cw_at_eps15.detach().cpu().numpy().astype(int)
    univariate_aurocs(feats_np, y_bin, feat_names)

    print("\n== Univariate AUROC: CW L_inf magnitude (continuous, AUROC vs median split) ==")
    mag = cw_linf_mag.detach().cpu().numpy()
    med = np.median(mag)
    # Use "low magnitude" (= more vulnerable) as positive class.
    y_mag = (mag <= med).astype(int)
    univariate_aurocs(feats_np, y_mag, feat_names)
    print("  (Pearson correlations with continuous magnitude:)")
    for i, n in enumerate(feat_names):
        if np.std(feats_np[:, i]) == 0:
            print(f"    {n:<12} corr  (degenerate)")
            continue
        cor = float(np.corrcoef(feats_np[:, i], mag)[0, 1])
        print(f"    {n:<12} corr  {cor:+.4f}")

    print("\n== Univariate AUROC: FGSM flip @ eps=15/255 (cross-comparison) ==")
    y_fgsm = fgsm_flip_v.detach().cpu().numpy().astype(int)
    univariate_aurocs(feats_np, y_fgsm, feat_names)

    print("\n== CW vs FGSM agreement ==")
    a = y_bin.astype(bool)
    b = y_fgsm.astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    jacc = inter / union if union > 0 else float("nan")
    # Cohen's kappa
    n = len(a)
    po = ((a == b).sum()) / n
    pe = (a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean()))
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else float("nan")
    print(f"  CW@eps   flip rate: {a.mean():.3f}")
    print(f"  FGSM@eps flip rate: {b.mean():.3f}")
    print(f"  intersection: {inter}   union: {union}   Jaccard: {jacc:.4f}")
    print(f"  Cohen kappa: {kappa:.4f}")
    # AUROC of FGSM-flip indicator against CW-flip (sanity check overlap).
    if y_bin.std() > 0:
        auc_cross = roc_auc_score(y_bin, y_fgsm.astype(float))
        print(f"  FGSM-as-predictor AUROC for CW@eps: {auc_cross:.4f}")


if __name__ == "__main__":
    main()
