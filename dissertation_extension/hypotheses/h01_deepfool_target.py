"""
Hypothesis H01: DeepFool-derived min-eps is a more accurate per-sample vulnerability
label than binary-search FGSM-min-eps. Image statistics may predict DeepFool-min-eps
better/worse than they predict FGSM-min-eps.

Pipeline:
  1. Train small CNN victim on Fashion-MNIST (10 epochs, Adam).
  2. Compute per-sample DeepFool L_inf minimum perturbation (from scratch, Moosavi-
     Dezfooli et al. 2016; falls back to torchattacks DeepFool L2 if available).
  3. Compute model-free image features (mean_pix, std_pix, sobel_mean, edge_density,
     jpeg_q75 byte size) plus the model-dependent victim_margin.
  4. Targets:
        - flipped_by_FGSM_eps15  (binary)  -- self FGSM attack at eps=15/255
        - deepfool_mag           (continuous, L_inf magnitude of DeepFool perturbation)
        - fgsm_min_eps           (continuous, binary-search FGSM min eps)
  5. Univariate AUROC / Pearson / Spearman of each feature against each target.
  6. Multivariate logistic regression (binary) and OLS / R^2 (continuous).

Run: python h01_deepfool_target.py
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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
DEEPFOOL_MAX_ITERS = 50
DEEPFOOL_OVERSHOOT = 0.02
NUM_CLASSES_CANDIDATES = 10  # consider all 10 logits for DeepFool


# -------------------------------------------------------------- model
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
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time()
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  epoch {ep+1}/{EPOCHS}  loss={tot/n:.4f}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# -------------------------------------------------------------- DeepFool L_inf
def deepfool_linf_batch(model, x, y_true, num_classes=10,
                        max_iters=DEEPFOOL_MAX_ITERS,
                        overshoot=DEEPFOOL_OVERSHOOT):
    """
    DeepFool with L_inf minimisation, per-sample, batched. Implements the
    closed-form linearised step from Moosavi-Dezfooli et al. (2016) where the
    L_inf perturbation toward decision-boundary k is:
        r_k = |f_k - f_y| / ||w_k||_1  * sign(w_k)
    Returns: linf_mag (B,), L_inf magnitude of total perturbation.

    For samples already misclassified, returns 0 perturbation.
    """
    model.eval()
    x = x.clone().detach()
    B = x.size(0)
    pert = torch.zeros_like(x)
    x_adv = x.clone()
    active = torch.ones(B, dtype=torch.bool, device=x.device)
    # initial preds
    with torch.no_grad():
        init_pred = model(x).argmax(1)
    active &= (init_pred == y_true)

    for it in range(max_iters):
        if not active.any():
            break
        idx = torch.where(active)[0]
        xi = x_adv[idx].clone().detach().requires_grad_(True)
        logits = model(xi)
        cur_pred = logits.argmax(1)
        # stop if no longer predicting y_true
        still = cur_pred == y_true[idx]
        if not still.any():
            active[idx[~still]] = False
            continue
        # process only those still on original class
        idx2 = idx[still]
        xi2 = x_adv[idx2].clone().detach().requires_grad_(True)
        logits2 = model(xi2)
        B2 = xi2.size(0)
        y2 = y_true[idx2]

        # gradient of f_y wrt xi2
        grads_y = torch.autograd.grad(
            logits2.gather(1, y2.unsqueeze(1)).sum(), xi2,
            retain_graph=True, create_graph=False)[0]
        # for each non-true class k, compute (f_k - f_y) and grad(f_k) - grad(f_y),
        # then r_k_linf = |f_k - f_y| / ||w_k||_1
        f_y = logits2.gather(1, y2.unsqueeze(1)).squeeze(1)
        best_ratio = torch.full((B2,), float("inf"), device=x.device)
        best_w = torch.zeros_like(xi2)
        best_f = torch.zeros(B2, device=x.device)
        for k in range(num_classes):
            mask_k = (torch.full((B2,), k, device=x.device) != y2)
            if not mask_k.any():
                continue
            f_k = logits2[:, k]
            grads_k = torch.autograd.grad(
                f_k.sum(), xi2, retain_graph=True, create_graph=False)[0]
            w_k = grads_k - grads_y
            f_diff = f_k - f_y  # negative for non-winning classes
            w_l1 = w_k.flatten(1).abs().sum(1) + 1e-12
            ratio = f_diff.abs() / w_l1
            # only consider classes k != y_true and where ratio is smaller than current best
            update = mask_k & (ratio < best_ratio)
            best_ratio = torch.where(update, ratio, best_ratio)
            # update best_w / best_f selectively
            if update.any():
                u_idx = torch.where(update)[0]
                best_w[u_idx] = w_k[u_idx]
                best_f[u_idx] = f_diff[u_idx]
        # compute L_inf-minimising step:
        # r = (|f_diff| / ||w||_1) * sign(w)
        r_step = best_ratio.view(-1, 1, 1, 1) * torch.sign(best_w)
        # apply with overshoot
        pert[idx2] = pert[idx2] + (1 + overshoot) * r_step
        x_adv[idx2] = (x[idx2] + pert[idx2]).clamp(0, 1)
        # update pert to reflect clamped state (so that further iters chain on x_adv)
        pert[idx2] = x_adv[idx2] - x[idx2]

        # recheck predictions
        with torch.no_grad():
            new_pred = model(x_adv[idx2]).argmax(1)
        flipped = new_pred != y2
        active[idx2[flipped]] = False

    linf_mag = pert.flatten(1).abs().max(1).values
    # for samples that never flipped, mark as max value seen (proxy)
    return linf_mag.detach(), (~active | (init_pred != y_true)), x_adv.detach()


# -------------------------------------------------------------- FGSM helpers
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
    lo = torch.zeros(x.size(0), device=x.device)
    hi = torch.full((x.size(0),), eps_max, device=x.device)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def fgsm_flip_at(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


# -------------------------------------------------------------- image features
def sobel_filters(device):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    return kx.view(1, 1, 3, 3).to(device), ky.view(1, 1, 3, 3).to(device)


def compute_image_features(x):
    """x is (B,1,28,28) in [0,1]. Returns dict of (B,) tensors on CPU numpy."""
    B = x.size(0)
    mean_pix = x.flatten(1).mean(1)
    std_pix = x.flatten(1).std(1)
    kx, ky = sobel_filters(x.device)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    grad_mag = (gx * gx + gy * gy).sqrt()
    sobel_mean = grad_mag.flatten(1).mean(1)
    # edge density: fraction of pixels with grad_mag > 0.1
    edge_density = (grad_mag.flatten(1) > 0.1).float().mean(1)

    # JPEG q75 byte size — must do on CPU per-sample via PIL
    jpeg_size = torch.zeros(B, device=x.device)
    x_cpu = (x.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    for i in range(B):
        img = Image.fromarray(x_cpu[i, 0], mode="L")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        jpeg_size[i] = buf.tell()
    return {
        "mean_pix": mean_pix.detach().cpu().numpy(),
        "std_pix": std_pix.detach().cpu().numpy(),
        "sobel_mean": sobel_mean.detach().cpu().numpy(),
        "edge_density": edge_density.detach().cpu().numpy(),
        "jpeg_q75": jpeg_size.detach().cpu().numpy(),
    }


def compute_victim_margin(model, x):
    with torch.no_grad():
        logits = []
        for i in range(0, x.size(0), 512):
            logits.append(model(x[i:i+512]))
        logits = torch.cat(logits, 0)
    sorted_logits, _ = logits.sort(1, descending=True)
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]
    pred = logits.argmax(1)
    return margin.detach().cpu().numpy(), pred


# -------------------------------------------------------------- analysis
def univariate_auroc(features_np, names, y_bin):
    print(f"  univariate AUROC (positive rate = {y_bin.mean():.3f})")
    for i, n in enumerate(names):
        a = roc_auc_score(y_bin, features_np[:, i])
        a = max(a, 1 - a)
        print(f"    {n:<18} AUROC = {a:.4f}")


def univariate_continuous(features_np, names, y_cont):
    print(f"  univariate Pearson / Spearman (target mean={y_cont.mean():.4f})")
    for i, n in enumerate(names):
        p = pearsonr(features_np[:, i], y_cont)[0]
        s = spearmanr(features_np[:, i], y_cont)[0]
        print(f"    {n:<18} pearson={p:+.4f}  spearman={s:+.4f}")


def multivariate_binary(features_np, names, y_bin, tag):
    Xs = StandardScaler().fit_transform(features_np)
    lr = LogisticRegression(max_iter=2000).fit(Xs, y_bin)
    auc = roc_auc_score(y_bin, lr.predict_proba(Xs)[:, 1])
    print(f"  [{tag}] multivariate logistic AUROC = {auc:.4f}")
    print("    standardised coefficients:")
    for n, c in zip(names, lr.coef_.flatten()):
        print(f"      {n:<18} {c:+.4f}")
    return auc


def multivariate_continuous(features_np, names, y_cont, tag):
    Xs = StandardScaler().fit_transform(features_np)
    ols = LinearRegression().fit(Xs, y_cont)
    r2 = ols.score(Xs, y_cont)
    print(f"  [{tag}] OLS R^2 = {r2:.4f}")
    print("    standardised coefficients:")
    for n, c in zip(names, ols.coef_):
        print(f"      {n:<18} {c:+.6f}")
    return r2


# -------------------------------------------------------------- main
def main():
    print(f"device = {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print("Training victim CNN on Fashion-MNIST...")
    t0 = time.time()
    model = train_model(train_set, seed=0)
    print(f"  trained in {time.time()-t0:.1f}s")

    # Stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)
    print(f"test set: {N} samples")

    # Margin + predictions
    margin_np, pred = compute_victim_margin(model, test_x)
    correct = (pred == test_y)
    print(f"victim test acc = {correct.float().mean().item():.4f}")

    # restrict to correctly-classified samples (vulnerability is only well-defined here)
    x_c = test_x[correct]
    y_c = test_y[correct]
    margin_c = margin_np[correct.detach().cpu().numpy()]
    print(f"  using {x_c.size(0)} correctly-classified test samples")

    # --- DeepFool ---
    print("Computing DeepFool L_inf magnitudes ...")
    t0 = time.time()
    df_mags = []
    df_flipped = []
    for i in range(0, x_c.size(0), 256):
        mag, flag, _ = deepfool_linf_batch(
            model, x_c[i:i+256], y_c[i:i+256],
            num_classes=NUM_CLASSES_CANDIDATES)
        df_mags.append(mag)
        df_flipped.append(flag)
    deepfool_mag = torch.cat(df_mags).detach().cpu().numpy()
    deepfool_success = torch.cat(df_flipped).detach().cpu().numpy()
    print(f"  done in {time.time()-t0:.1f}s; "
          f"success rate={deepfool_success.mean():.4f}; "
          f"mean L_inf={deepfool_mag.mean():.4f}; "
          f"median={np.median(deepfool_mag):.4f}")

    # --- FGSM min-eps (binary search) ---
    print("Computing FGSM binary-search min-eps ...")
    t0 = time.time()
    me = []
    for i in range(0, x_c.size(0), 512):
        me.append(fgsm_min_eps(model, x_c[i:i+512], y_c[i:i+512]))
    fgsm_eps = torch.cat(me).detach().cpu().numpy()
    print(f"  done in {time.time()-t0:.1f}s; mean min-eps={fgsm_eps.mean():.4f}")

    # --- FGSM flip @ eps=15/255 (binary) ---
    print(f"Computing FGSM flip-rate @ eps={EPS_TEST:.4f} ...")
    flips = []
    for i in range(0, x_c.size(0), 512):
        flips.append(fgsm_flip_at(model, x_c[i:i+512], y_c[i:i+512], EPS_TEST))
    fgsm_flip = torch.cat(flips).detach().cpu().numpy().astype(int)
    print(f"  flip rate = {fgsm_flip.mean():.4f}")

    # --- image features ---
    print("Computing image-level features ...")
    t0 = time.time()
    img_feats_chunks = {k: [] for k in
                        ["mean_pix", "std_pix", "sobel_mean", "edge_density", "jpeg_q75"]}
    for i in range(0, x_c.size(0), 512):
        f = compute_image_features(x_c[i:i+512])
        for k in img_feats_chunks:
            img_feats_chunks[k].append(f[k])
    img_feats = {k: np.concatenate(v) for k, v in img_feats_chunks.items()}
    print(f"  done in {time.time()-t0:.1f}s")

    # combine into feature matrix
    feat_names = ["victim_margin", "mean_pix", "std_pix",
                  "sobel_mean", "edge_density", "jpeg_q75"]
    feats = np.stack([
        margin_c,
        img_feats["mean_pix"],
        img_feats["std_pix"],
        img_feats["sobel_mean"],
        img_feats["edge_density"],
        img_feats["jpeg_q75"],
    ], axis=1).astype(np.float64)

    image_only_names = ["mean_pix", "std_pix", "sobel_mean", "edge_density", "jpeg_q75"]
    image_only = feats[:, 1:]

    # ============================================================== analysis
    print("\n" + "=" * 70)
    print("TARGET 1: flipped_by_FGSM_eps15  (binary)")
    print("=" * 70)
    univariate_auroc(feats, feat_names, fgsm_flip)
    auc_full = multivariate_binary(feats, feat_names, fgsm_flip, "full (incl margin)")
    auc_img = multivariate_binary(image_only, image_only_names, fgsm_flip,
                                  "image-stats only")

    print("\n" + "=" * 70)
    print("TARGET 2: DeepFool L_inf magnitude  (continuous)")
    print("=" * 70)
    # restrict to samples DeepFool actually flipped
    mask = deepfool_success.astype(bool)
    if mask.sum() < 100:
        print("  WARNING: too few DeepFool successes for reliable regression")
    print(f"  using {int(mask.sum())} successful DeepFool samples")
    univariate_continuous(feats[mask], feat_names, deepfool_mag[mask])
    r2_df_full = multivariate_continuous(feats[mask], feat_names,
                                         deepfool_mag[mask], "full")
    r2_df_img = multivariate_continuous(image_only[mask], image_only_names,
                                        deepfool_mag[mask], "image-stats only")

    print("\n" + "=" * 70)
    print("TARGET 3: FGSM binary-search min-eps  (continuous)")
    print("=" * 70)
    univariate_continuous(feats, feat_names, fgsm_eps)
    r2_fg_full = multivariate_continuous(feats, feat_names, fgsm_eps, "full")
    r2_fg_img = multivariate_continuous(image_only, image_only_names, fgsm_eps,
                                        "image-stats only")

    # ============================================================== relationships
    print("\n" + "=" * 70)
    print("AGREEMENT BETWEEN TARGETS")
    print("=" * 70)
    p = pearsonr(deepfool_mag[mask], fgsm_eps[mask])[0]
    s = spearmanr(deepfool_mag[mask], fgsm_eps[mask])[0]
    print(f"  pearson  (deepfool_mag, fgsm_min_eps) = {p:+.4f}")
    print(f"  spearman (deepfool_mag, fgsm_min_eps) = {s:+.4f}")

    # ============================================================== summary
    print("\n" + "=" * 70)
    print("SUMMARY: how well do image-only stats predict each vulnerability target?")
    print("=" * 70)
    print(f"  FGSM-flip@eps15  multivariate AUROC : image-only={auc_img:.4f}  "
          f"full(+margin)={auc_full:.4f}")
    print(f"  DeepFool-mag     multivariate R^2   : image-only={r2_df_img:.4f}  "
          f"full(+margin)={r2_df_full:.4f}")
    print(f"  FGSM-min-eps     multivariate R^2   : image-only={r2_fg_img:.4f}  "
          f"full(+margin)={r2_fg_full:.4f}")
    print("\nInterpretation:")
    print("  If image-only R^2 is HIGHER for DeepFool than for FGSM-min-eps, then the")
    print("  cleaner DeepFool label is MORE predictable from image statistics alone.")
    print("  If LOWER, then FGSM-min-eps was easier to predict because of its coarser")
    print("  step-quantised structure rather than genuine vulnerability signal.")


if __name__ == "__main__":
    main()
