"""
H72: Histogram of Oriented Gradients (HOG) summary statistics predict adversarial
vulnerability.

Hypothesis: HOG is a classical, model-free descriptor of local-gradient structure.
Per-sample HOG summary statistics (total energy, orientation-bin entropy, max
orientation fraction) may capture per-sample texture/edge complexity that correlates
with adversarial vulnerability on Fashion-MNIST.

Pipeline:
  1. Train CNN victim (model A) on Fashion-MNIST for 10 epochs (Adam, lr=1e-3),
     same architecture as diagnostic_test.py.
  2. Per test sample compute HOG via skimage.feature.hog with
     pixels_per_cell=(4, 4), cells_per_block=(2, 2), orientations=9.
  3. Reduce HOG to three scalar summaries per sample:
       - hog_energy:   sum of HOG-vector squared magnitudes (total energy)
       - hog_entropy:  entropy of the (sum-over-spatial) 9-bin orientation
                       distribution
       - hog_max_frac: maximum orientation-bin fraction in the same 9-bin
                       orientation distribution (concentration measure)
  4. Baseline features per sample: margin (victim, final-model), mean_pix, std_pix.
  5. Attack targets per sample (restricted to victim-correct samples):
       - flipped_FGSM (eps = 15/255)
       - flipped_PGD  (eps = 15/255, 10 iters, alpha = 2/255)
       - min_eps      (smallest FGSM L_inf eps that flips, binary search)
  6. Univariate AUROC per feature vs each binary target. For continuous min_eps,
     univariate AUROC is reported by binarising on the median.

Self-contained: run with `python h72_hog.py`. Prints all results to stdout.
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from skimage.feature import hog
from sklearn.metrics import roc_auc_score


# -------------------- config --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
PGD_ITERS = 10
PGD_STEP = 2.0 / 255.0
SEED_A = 0

HOG_ORIENTATIONS = 9
HOG_PIX_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)


# -------------------- model --------------------
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


def train_model(seed, train_set):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time()
        total = 0.0
        n = 0
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        print(f"   seed={seed} epoch {ep+1}/{EPOCHS}  loss={total/n:.4f}  "
              f"({time.time()-t0:.1f}s)")
    model.eval()
    return model


# -------------------- attacks --------------------
def fgsm_grad_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flipped(model, x, y, eps=EPS):
    sign = fgsm_grad_sign(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flipped(model, x, y, eps=EPS, step=PGD_STEP, iters=PGD_ITERS):
    x_orig = x.clone().detach()
    delta = (torch.rand_like(x) * 2 - 1) * eps
    adv = (x_orig + delta).clamp(0, 1).detach()
    for _ in range(iters):
        adv = adv.detach().requires_grad_(True)
        F.cross_entropy(model(adv), y).backward()
        with torch.no_grad():
            adv = adv + step * adv.grad.sign()
            adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps)
            adv = adv.clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf FGSM eps that flips."""
    sign = fgsm_grad_sign(model, x, y)
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


def batched(fn, x, y, bs=512, **kw):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(fn(x[i:i+bs], y[i:i+bs], **kw))
    return torch.cat(out)


# -------------------- features --------------------
def compute_hog_features(images_np):
    """
    images_np: (N, H, W) float32 in [0, 1].
    Returns dict with:
      - hog_energy:   (N,)
      - hog_entropy:  (N,) entropy of the 9-bin global orientation distribution
      - hog_max_frac: (N,) max orientation-bin fraction
    """
    N = images_np.shape[0]
    energy = np.zeros(N, dtype=np.float32)
    entropy = np.zeros(N, dtype=np.float32)
    max_frac = np.zeros(N, dtype=np.float32)
    for i in range(N):
        img = images_np[i]
        v = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIX_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm="L2-Hys",
            transform_sqrt=False,
            feature_vector=True,
        ).astype(np.float32)
        # total HOG energy
        energy[i] = float((v * v).sum())
        # collapse to 9-bin orientation distribution by summing every orientation
        # bin position across all (block, cell) entries.
        if v.size % HOG_ORIENTATIONS == 0 and v.size > 0:
            ori = v.reshape(-1, HOG_ORIENTATIONS).sum(axis=0)
        else:
            ori = np.zeros(HOG_ORIENTATIONS, dtype=np.float32)
        s = ori.sum()
        if s > 0:
            p = ori / s
        else:
            p = np.zeros_like(ori)
        nz = p[p > 0]
        entropy[i] = float(-(nz * np.log(nz)).sum()) if nz.size else 0.0
        max_frac[i] = float(p.max()) if p.size else 0.0
    return {
        "hog_energy": energy,
        "hog_entropy": entropy,
        "hog_max_frac": max_frac,
    }


def compute_baseline_features(images_np, model, x_t, y_t):
    N = images_np.shape[0]
    mean_pix = images_np.reshape(N, -1).mean(axis=1).astype(np.float32)
    std_pix = images_np.reshape(N, -1).std(axis=1).astype(np.float32)
    margins = []
    model.eval()
    with torch.no_grad():
        for i in range(0, x_t.size(0), 512):
            logits = model(x_t[i:i+512])
            s, _ = logits.sort(1, descending=True)
            margins.append((s[:, 0] - s[:, 1]).cpu().numpy())
    margin = np.concatenate(margins).astype(np.float32)
    return margin, mean_pix, std_pix


# -------------------- evaluation --------------------
def auroc_auto(y_true, score):
    if np.std(score) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    a = roc_auc_score(y_true, score)
    return max(a, 1 - a)


def univariate_table(feature_dict, target_dict):
    feat_names = list(feature_dict.keys())
    tgt_names = list(target_dict.keys())
    print(f"\n{'feature':<18}" + "".join(f"{t:>18}" for t in tgt_names))
    rows = {}
    for fn in feat_names:
        col = feature_dict[fn]
        row_vals = {}
        out = [f"{fn:<18}"]
        for tn in tgt_names:
            y = target_dict[tn]
            a = auroc_auto(y, col)
            row_vals[tn] = a
            out.append(f"{a:>18.4f}" if not np.isnan(a) else f"{'NA':>18}")
        rows[fn] = row_vals
        print("".join(out))
    return rows


def main():
    print(f"device = {DEVICE}")
    os.makedirs(DATA_DIR, exist_ok=True)
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)

    x_test = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n[1/4] training victim model A ...")
    t0 = time.time()
    model_A = train_model(SEED_A, train_set)
    print(f"   done in {time.time()-t0:.1f}s")

    # victim-correct mask
    with torch.no_grad():
        preds_A = []
        for i in range(0, x_test.size(0), 512):
            preds_A.append(model_A(x_test[i:i+512]).argmax(1))
        preds_A = torch.cat(preds_A)
    correct = (preds_A == y_test)
    print(f"\n   victim A test accuracy = {correct.float().mean().item():.4f}  "
          f"({int(correct.sum())} / {len(y_test)})")

    x_c = x_test[correct]
    y_c = y_test[correct]
    images_np = x_c.squeeze(1).cpu().numpy().astype(np.float32)

    print("\n[2/4] computing features ...")
    t0 = time.time()
    margin, mean_pix, std_pix = compute_baseline_features(images_np, model_A, x_c, y_c)
    print(f"   baseline features: {time.time()-t0:.1f}s")
    t0 = time.time()
    hog_feats = compute_hog_features(images_np)
    print(f"   HOG features:      {time.time()-t0:.1f}s")

    print("\n[3/4] computing attack-success targets ...")
    t0 = time.time()
    flipped_fgsm = batched(lambda x, y: fgsm_flipped(model_A, x, y),
                           x_c, y_c).cpu().numpy().astype(int)
    print(f"   FGSM:    {time.time()-t0:.1f}s  pos rate={flipped_fgsm.mean():.3f}")
    t0 = time.time()
    flipped_pgd = batched(lambda x, y: pgd_flipped(model_A, x, y),
                          x_c, y_c).cpu().numpy().astype(int)
    print(f"   PGD:     {time.time()-t0:.1f}s  pos rate={flipped_pgd.mean():.3f}")
    t0 = time.time()
    min_eps = batched(lambda x, y: fgsm_min_eps(model_A, x, y),
                      x_c, y_c).cpu().numpy().astype(np.float32)
    print(f"   min_eps: {time.time()-t0:.1f}s  mean={min_eps.mean():.4f} "
          f"median={np.median(min_eps):.4f}")

    print("\n[4/4] univariate AUROC ...")
    feature_dict = {
        "margin":        margin,
        "mean_pix":      mean_pix,
        "std_pix":       std_pix,
        "hog_energy":    hog_feats["hog_energy"],
        "hog_entropy":   hog_feats["hog_entropy"],
        "hog_max_frac":  hog_feats["hog_max_frac"],
    }
    # Binarise min_eps via median split (low min_eps -> more vulnerable).
    med = float(np.median(min_eps))
    min_eps_bin = (min_eps <= med).astype(int)
    target_dict = {
        "flipped_FGSM":     flipped_fgsm,
        "flipped_PGD":      flipped_pgd,
        "min_eps_lo_half":  min_eps_bin,
    }
    rows = univariate_table(feature_dict, target_dict)

    summary = {
        "univariate_auroc": rows,
        "n_correct": int(correct.sum().item()),
        "pos_rate": {
            "flipped_FGSM": float(flipped_fgsm.mean()),
            "flipped_PGD":  float(flipped_pgd.mean()),
            "min_eps_lo_half": float(min_eps_bin.mean()),
        },
        "min_eps_stats": {
            "mean":   float(min_eps.mean()),
            "median": med,
            "std":    float(min_eps.std()),
        },
    }
    out_json = "/tmp/data/h72_hog_summary.json"
    try:
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"\nwrote summary -> {out_json}")
    except Exception as e:
        print(f"could not write summary json: {e}")


if __name__ == "__main__":
    main()
