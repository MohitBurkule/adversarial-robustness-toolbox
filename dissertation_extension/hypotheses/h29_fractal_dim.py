"""
H29: Per-sample fractal dimension predicts adversarial vulnerability.

Hypothesis: The box-counting (Minkowski-Bouligand) fractal dimension of a sample's
binarised image and of its Sobel-edge map measures the spatial complexity of the
pattern. A higher fractal dimension reflects richer/finer-scale structure, which
plausibly translates to a more entangled local decision-surface neighbourhood and
hence higher robustness (lower adversarial vulnerability). Fractal dimension is a
classical complexity measure but has never (to our knowledge) been benchmarked as
an adversarial-vulnerability predictor on a standard image classifier.

Pipeline:
  1. Train a small CNN victim on Fashion-MNIST (10 epochs Adam, lr=1e-3), same
     architecture as diagnostic_test.py / h05_lbp.py.
  2. Per (victim-correct) test sample compute:
       - fd_bin:   box-counting fractal dim of the image binarised at its median
       - fd_edge:  box-counting fractal dim of the Sobel-edge magnitude binarised
                   at the per-sample edge mean
     Box-counting uses scales s in {1, 2, 4, 7, 14} on the 28x28 image; the
     fractal dimension is -slope of  log(N_boxes_occupied)  vs  log(s).
  3. Baseline features per sample: victim_margin, mean_pix, std_pix.
  4. Targets (binary, restricted to victim-correct samples):
       - flipped_FGSM   (eps = 15/255)
       - flipped_PGD    (eps = 15/255, 10 iters, step = 2/255, random start)
       - FGSM_min_eps   (binary-search smallest L_inf eps that flips FGSM;
                        binarised at the bottom-quartile = most vulnerable)
  5. Univariate per-feature AUROC (auto-oriented = max(a, 1-a)).
  6. Multivariate logistic regression: does adding fractal-dim features improve
     AUROC over {margin alone} and over {baseline = margin + mean_pix + std_pix}?

Tools: PyTorch (CUDA) + numpy + scipy.ndimage (Sobel). Box-counting is
implemented from scratch with numpy (no `python-fractal` dependency required).
Data cache: /tmp/data. Self-contained: `python h29_fractal_dim.py`.
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

from scipy.ndimage import sobel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# -------------------- config --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
PGD_ITERS = 10
PGD_STEP = 2.0 / 255.0
SEED = 0

# box-counting scales (divisors of/near 28). 28 = 4*7 = 2*14.
BOX_SCALES = (1, 2, 4, 7, 14)
LOG_SCALES = np.log(np.array(BOX_SCALES, dtype=np.float64))


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
        total, n = 0.0, 0
        for x, y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
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


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    """Binary-search smallest L_inf eps that flips FGSM. Returns eps per sample."""
    sign = fgsm_grad_sign(model, x, y)
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
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


# -------------------- box-counting fractal dimension --------------------
def _count_occupied_boxes(binary, scale):
    """
    Count number of scale x scale boxes that contain at least one True pixel.
    Image is right/bottom-padded so its dims are divisible by `scale`.
    """
    H, W = binary.shape
    pad_h = (-H) % scale
    pad_w = (-W) % scale
    if pad_h or pad_w:
        binary = np.pad(binary, ((0, pad_h), (0, pad_w)), mode='constant')
    nH = binary.shape[0] // scale
    nW = binary.shape[1] // scale
    # reshape -> (nH, scale, nW, scale)
    blocks = binary.reshape(nH, scale, nW, scale)
    # a box is occupied if any pixel in it is True
    occ = blocks.any(axis=(1, 3))
    return int(occ.sum())


def box_counting_fd(binary, scales=BOX_SCALES, log_scales=LOG_SCALES):
    """
    Box-counting fractal dimension. Counts at each scale, then fits
        log N(s) = -D * log(s) + c
    so D = -slope. If the binary image is empty, returns 0.0; if it is full,
    returns the topological-dimension upper bound (2.0).

    Robustness: if at any scale the count is 0 (shouldn't happen for non-empty
    binary), we drop that point before the fit. Need >= 2 surviving scales.
    """
    if not binary.any():
        return 0.0
    counts = np.array([_count_occupied_boxes(binary, s) for s in scales],
                      dtype=np.float64)
    mask = counts > 0
    if mask.sum() < 2:
        return 0.0
    log_n = np.log(counts[mask])
    ls = log_scales[mask]
    # slope of log_n vs ls via least squares
    slope, _ = np.polyfit(ls, log_n, 1)
    return float(-slope)


def compute_fractal_features(images_np):
    """
    images_np: (N, H, W) float32 in [0, 1].
    Returns:
      fd_bin   (N,)  - FD of (img > median(img))
      fd_edge  (N,)  - FD of (sobel_mag > mean(sobel_mag))
    """
    N = images_np.shape[0]
    fd_bin = np.zeros(N, dtype=np.float32)
    fd_edge = np.zeros(N, dtype=np.float32)
    for i in range(N):
        img = images_np[i]
        thr = float(np.median(img))
        bin_img = img > thr
        fd_bin[i] = box_counting_fd(bin_img)

        gx = sobel(img, axis=0)
        gy = sobel(img, axis=1)
        mag = np.sqrt(gx * gx + gy * gy)
        em = float(mag.mean())
        # if edge map is degenerate (constant), fall back to >0
        if mag.std() < 1e-8:
            edge_bin = mag > 0
        else:
            edge_bin = mag > em
        fd_edge[i] = box_counting_fd(edge_bin)
    return fd_bin, fd_edge


# -------------------- baseline features --------------------
def compute_baseline_features(images_np, model, x_t):
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
def univariate_auroc(X, y, names):
    print("   univariate AUROC (auto-oriented = max(a, 1-a)):")
    results = {}
    for i, n in enumerate(names):
        col = X[:, i]
        if np.std(col) == 0:
            print(f"     {n:<20} (constant - skipped)")
            continue
        a = roc_auc_score(y, col)
        a_signed = a
        a = max(a, 1 - a)
        results[n] = (a, a_signed)
        # sign tells us direction: a_signed > 0.5 => higher feature => more vulnerable
        direction = "+" if a_signed >= 0.5 else "-"
        print(f"     {n:<20} {a:.4f}   (raw={a_signed:.4f}, sign={direction})")
    return results


def multivariate_auroc(X, y, label):
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=5000).fit(Xs, y)
    auc = roc_auc_score(y, lr.predict_proba(Xs)[:, 1])
    print(f"   multivariate AUROC ({label}, d={X.shape[1]}): {auc:.4f}")
    return auc


def evaluate(target_name, y_bin, feat_blocks, all_names_for_uni, X_all_for_uni):
    print(f"\n========== target: {target_name}  (pos rate = {y_bin.mean():.3f}, "
          f"n = {len(y_bin)}) ==========")
    if y_bin.std() == 0:
        print("   degenerate target - skipping")
        return {}
    univariate_auroc(X_all_for_uni, y_bin, all_names_for_uni)
    print("   ---- multivariate models ----")
    aucs = {}
    for label, (X, _) in feat_blocks.items():
        aucs[label] = multivariate_auroc(X, y_bin, label)
    # explicit deltas
    if "margin" in aucs and "margin + fractal" in aucs:
        d = aucs["margin + fractal"] - aucs["margin"]
        print(f"   Delta AUROC: (margin + fractal) - margin       = {d:+.4f}")
    if "baseline" in aucs and "baseline + fractal" in aucs:
        d = aucs["baseline + fractal"] - aucs["baseline"]
        print(f"   Delta AUROC: (baseline + fractal) - baseline   = {d:+.4f}")
    return aucs


# -------------------- main --------------------
def main():
    print(f"device = {DEVICE}")
    os.makedirs(DATA_DIR, exist_ok=True)
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)

    x_test = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n[1/4] training victim CNN ...")
    t0 = time.time()
    model = train_model(SEED, train_set)
    print(f"   done in {time.time()-t0:.1f}s")

    with torch.no_grad():
        preds = []
        for i in range(0, x_test.size(0), 512):
            preds.append(model(x_test[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct = (preds == y_test)
    print(f"\n   victim test accuracy = {correct.float().mean().item():.4f}  "
          f"({int(correct.sum())} / {len(y_test)})")

    x_c = x_test[correct]
    y_c = y_test[correct]
    images_np = x_c.squeeze(1).cpu().numpy().astype(np.float32)

    # ---------------- features ----------------
    print("\n[2/4] computing baseline features ...")
    t0 = time.time()
    margin, mean_pix, std_pix = compute_baseline_features(images_np, model, x_c)
    print(f"   baseline:        {time.time()-t0:.1f}s")

    print("\n[3/4] computing fractal-dimension features ...")
    t0 = time.time()
    fd_bin, fd_edge = compute_fractal_features(images_np)
    print(f"   fractal-dim:     {time.time()-t0:.1f}s")
    print(f"   fd_bin  : mean={fd_bin.mean():.4f}  std={fd_bin.std():.4f}  "
          f"min={fd_bin.min():.4f}  max={fd_bin.max():.4f}")
    print(f"   fd_edge : mean={fd_edge.mean():.4f}  std={fd_edge.std():.4f}  "
          f"min={fd_edge.min():.4f}  max={fd_edge.max():.4f}")

    # ---------------- targets ----------------
    print("\n[4/4] computing attack-success targets ...")
    t0 = time.time()
    flipped_fgsm = batched(lambda x, y: fgsm_flipped(model, x, y),
                           x_c, y_c).cpu().numpy().astype(int)
    print(f"   FGSM   :   {time.time()-t0:.1f}s  pos rate={flipped_fgsm.mean():.3f}")
    t0 = time.time()
    flipped_pgd = batched(lambda x, y: pgd_flipped(model, x, y),
                          x_c, y_c).cpu().numpy().astype(int)
    print(f"   PGD    :   {time.time()-t0:.1f}s  pos rate={flipped_pgd.mean():.3f}")
    t0 = time.time()
    me_chunks = []
    for i in range(0, x_c.size(0), 512):
        me_chunks.append(min_eps_fgsm(model, x_c[i:i+512], y_c[i:i+512]))
    min_eps = torch.cat(me_chunks).cpu().numpy()
    print(f"   min_eps:   {time.time()-t0:.1f}s  mean={min_eps.mean():.4f}  "
          f"median={np.median(min_eps):.4f}")
    me_thresh = float(np.quantile(min_eps, 0.25))
    fgsm_min_eps_bin = (min_eps <= me_thresh).astype(int)
    print(f"   FGSM_min_eps<=q25 ({me_thresh:.4f}): pos rate={fgsm_min_eps_bin.mean():.3f}")

    # ---------------- feature blocks ----------------
    base_names = ["margin", "mean_pix", "std_pix"]
    fractal_names = ["fd_bin", "fd_edge"]

    X_margin = margin.reshape(-1, 1)
    X_base = np.stack([margin, mean_pix, std_pix], axis=1)
    X_fractal = np.stack([fd_bin, fd_edge], axis=1)
    X_margin_fractal = np.concatenate([X_margin, X_fractal], axis=1)
    X_base_fractal = np.concatenate([X_base, X_fractal], axis=1)

    blocks = {
        "margin":              (X_margin,         ["margin"]),
        "baseline":            (X_base,           base_names),
        "fractal_only":        (X_fractal,        fractal_names),
        "margin + fractal":    (X_margin_fractal, ["margin"] + fractal_names),
        "baseline + fractal":  (X_base_fractal,   base_names + fractal_names),
    }

    # build a deduped univariate matrix
    all_names = base_names + fractal_names
    X_all_uni = np.stack([margin, mean_pix, std_pix, fd_bin, fd_edge], axis=1)

    summary = {}
    for tname, y_bin in [
        ("flipped_FGSM",  flipped_fgsm),
        ("flipped_PGD",   flipped_pgd),
        ("FGSM_min_eps",  fgsm_min_eps_bin),
    ]:
        aucs = evaluate(tname, y_bin, blocks, all_names, X_all_uni)
        summary[tname] = aucs

    print("\n===== SUMMARY (multivariate AUROC by target/block) =====")
    cols = list(blocks.keys())
    head = "target".ljust(18) + "  " + "  ".join(c[:22].ljust(22) for c in cols)
    print(head)
    for tname, aucs in summary.items():
        row = tname.ljust(18) + "  " + "  ".join(
            f"{aucs.get(c, float('nan')):.4f}".ljust(22) for c in cols)
        print(row)

    out_json = "/tmp/data/h29_fractal_dim_summary.json"
    try:
        with open(out_json, "w") as f:
            json.dump({
                "summary_multivariate": summary,
                "min_eps_q25_threshold": me_thresh,
                "fd_bin_stats": {
                    "mean": float(fd_bin.mean()),
                    "std":  float(fd_bin.std()),
                    "min":  float(fd_bin.min()),
                    "max":  float(fd_bin.max()),
                },
                "fd_edge_stats": {
                    "mean": float(fd_edge.mean()),
                    "std":  float(fd_edge.std()),
                    "min":  float(fd_edge.min()),
                    "max":  float(fd_edge.max()),
                },
                "box_scales": list(BOX_SCALES),
            }, f, indent=2, default=float)
        print(f"\nwrote summary -> {out_json}")
    except Exception as e:
        print(f"could not write summary json: {e}")


if __name__ == "__main__":
    main()
