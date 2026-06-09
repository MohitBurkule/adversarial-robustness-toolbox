"""
Rich image-statistics baseline vs OTI / JPEG / Fourier on natural images.

The point: a properly-engineered SET of cheap pixel/edge statistics may beat
OTI alone. Compute ~20 model-free image features and group them so we can
ablate cleanly:

  GROUP A — pure_pixel_stats: mean, std, skew, kurtosis, MAD, range, IQR,
            percentiles {p10,p25,p50,p75,p90}, per-channel mean+std (RGB only).
  GROUP B — edge/texture (model-free): Sobel mean, Sobel std,
            Laplacian abs mean, gradient histogram entropy.
  GROUP C — OTI / OAR (semi model-free; OTI_simple uses luminance mask only).
  GROUP D — Fourier: high-frequency energy at 25%/50% cutoffs.
  GROUP E — Compression: JPEG byte size at quality 75 and 50.

Then ablate:
  - pure_stats (A only)
  - pure_stats + edge (A+B)
  - pure_stats + edge + fourier (A+B+D)
  - pure_stats + edge + fourier + jpeg (A+B+D+E)   <-- "all-cheap-stats"
  - OTI alone (C)
  - JPEG alone (E)
  - all model-free (A+B+C+D+E)
  - margin (the model-known reference)

Datasets: CIFAR-10 and Imagenette-160. Reuses the same training pipeline as
model_free_natural.py.
"""
import io, time, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda")
DATA_ROOT = "/tmp/data"


def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def pgd(model, x, y, eps, steps=10):
    alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        F.cross_entropy(model(x_adv), y).backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)
    return x_adv.detach()


def chunked(fn, *args, bs=128):
    out = []
    for i in range(0, args[0].size(0), bs):
        out.append(fn(*[a[i:i+bs] for a in args]))
    return torch.cat(out)


# ---------- feature computation ----------
SOBEL_KX = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
LAPLACIAN = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)


def per_channel_filter(x, kernel):
    """Apply a 3x3 kernel per channel."""
    C = x.size(1)
    k = kernel.to(x.device).expand(C, 1, 3, 3)
    return F.conv2d(x, k, padding=1, groups=C)


def sobel_magnitude(x):
    kx = SOBEL_KX.to(x.device)
    ky = kx.transpose(2, 3)
    C = x.size(1)
    kx_g = kx.expand(C, 1, 3, 3); ky_g = ky.expand(C, 1, 3, 3)
    gx = F.conv2d(x, kx_g, padding=1, groups=C)
    gy = F.conv2d(x, ky_g, padding=1, groups=C)
    return (gx ** 2 + gy ** 2).sqrt().mean(1, keepdim=True)  # collapse channels


def laplacian_abs(x):
    C = x.size(1)
    k = LAPLACIAN.to(x.device).expand(C, 1, 3, 3)
    return F.conv2d(x, k, padding=1, groups=C).abs().mean(1, keepdim=True)


def fourier_hf_ratio(x, cutoff_frac):
    N, C, H, W = x.shape
    X = torch.fft.fft2(x)
    mag2 = (X.real ** 2 + X.imag ** 2)
    fy = torch.fft.fftfreq(H, device=x.device).abs()
    fx = torch.fft.fftfreq(W, device=x.device).abs()
    grid = (fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2).sqrt()
    high_mask = (grid > cutoff_frac * 0.5).float().unsqueeze(0).unsqueeze(0)
    total = mag2.flatten(1).sum(1)
    high = (mag2 * high_mask).flatten(1).sum(1)
    return high / total.clamp_min(1e-9)


def jpeg_bytesize(x, quality=75):
    arr = (x.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
    sizes = []
    N, C, H, W = arr.shape
    for i in range(N):
        if C == 1:
            img = Image.fromarray(arr[i, 0])
        else:
            img = Image.fromarray(arr[i].transpose(1, 2, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        sizes.append(buf.tell())
    return torch.tensor(sizes, device=x.device, dtype=torch.float32)


def compute_all_features(x):
    """x: [N,C,H,W] in [0,1]. Returns dict of per-sample features (each [N])."""
    N, C, H, W = x.shape
    pix = x.view(N, -1)
    feats = {}

    # GROUP A: pure pixel stats
    mean = pix.mean(1); std = pix.std(1)
    feats["mean_pix"] = mean
    feats["std_pix"] = std
    # variance redundant w/ std but keep for explicitness — skip
    # higher moments
    centered = pix - mean.unsqueeze(1)
    sd = std.clamp_min(1e-6)
    feats["skew_pix"] = ((centered / sd.unsqueeze(1)) ** 3).mean(1)
    feats["kurt_pix"] = ((centered / sd.unsqueeze(1)) ** 4).mean(1) - 3.0
    feats["MAD_pix"] = centered.abs().mean(1)
    # percentiles via quantile
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=x.device)
    quants = torch.quantile(pix, qs, dim=1)   # [5, N]
    feats["p10_pix"] = quants[0]
    feats["p25_pix"] = quants[1]
    feats["p50_pix"] = quants[2]
    feats["p75_pix"] = quants[3]
    feats["p90_pix"] = quants[4]
    feats["range_pix"] = pix.max(1).values - pix.min(1).values
    feats["IQR_pix"] = quants[3] - quants[1]
    # per-channel means and stds (for RGB)
    if C == 3:
        for ci in range(3):
            feats[f"mean_ch{ci}"] = x[:, ci].flatten(1).mean(1)
            feats[f"std_ch{ci}"] = x[:, ci].flatten(1).std(1)

    # histogram entropy
    bins = 16
    edges = torch.linspace(0, 1, bins + 1, device=x.device)
    h = torch.zeros(N, bins, device=x.device)
    for b in range(bins):
        upper = pix < edges[b + 1] if b < bins - 1 else pix <= edges[b + 1]
        m = (pix >= edges[b]) & upper
        h[:, b] = m.float().sum(1)
    p = h / h.sum(1, keepdim=True).clamp_min(1)
    feats["entropy_pix"] = -(p * torch.log(p.clamp_min(1e-9))).sum(1)

    # GROUP B: edge / texture (model-free)
    sob = sobel_magnitude(x)                              # [N,1,H,W]
    feats["sobel_mean"] = sob.flatten(1).mean(1)
    feats["sobel_std"] = sob.flatten(1).std(1)
    feats["sobel_p90"] = torch.quantile(sob.flatten(1), 0.9, dim=1)
    feats["laplacian_abs_mean"] = laplacian_abs(x).flatten(1).mean(1)

    # GROUP C: OTI/OAR (luminance-threshold mask, fully model-free)
    lum = x.mean(1, keepdim=True)
    thr = lum.flatten(1).median(1).values.view(-1, 1, 1, 1)
    fg_mask = (lum > thr).float()
    feats["OTI_simple"] = (fg_mask * sob).flatten(1).mean(1)
    feats["OAR_simple"] = fg_mask.flatten(1).mean(1)

    # GROUP D: Fourier
    feats["fourier_hf_25%"] = fourier_hf_ratio(x, 0.25)
    feats["fourier_hf_50%"] = fourier_hf_ratio(x, 0.50)

    # GROUP E: compression
    feats["jpeg_q75"] = torch.cat([jpeg_bytesize(x[i:i+256], 75)
                                   for i in range(0, x.size(0), 256)])
    feats["jpeg_q50"] = torch.cat([jpeg_bytesize(x[i:i+256], 50)
                                   for i in range(0, x.size(0), 256)])

    return feats, sob, fg_mask


# ---------- benchmarking ----------
def evaluate(name, feats_dict, margin, victim, surrogate, x, y, eps):
    """Run the multivariate ablation comparing feature sets."""
    feat_names = list(feats_dict.keys())
    feats = torch.stack([feats_dict[n] for n in feat_names], 1).cpu().numpy()
    Xs = StandardScaler().fit_transform(feats)

    # margin treated as a separate column
    margin_np = margin.cpu().numpy().reshape(-1, 1)
    margin_s = StandardScaler().fit_transform(margin_np)
    Xs_all = np.concatenate([margin_s, Xs], axis=1)
    all_names = ["victim_margin"] + feat_names

    # define groups by name pattern
    group_pure_stats = [n for n in feat_names if n in [
        "mean_pix", "std_pix", "skew_pix", "kurt_pix", "MAD_pix",
        "p10_pix", "p25_pix", "p50_pix", "p75_pix", "p90_pix",
        "range_pix", "IQR_pix", "entropy_pix",
        "mean_ch0", "mean_ch1", "mean_ch2", "std_ch0", "std_ch1", "std_ch2"]]
    group_edge = ["sobel_mean", "sobel_std", "sobel_p90", "laplacian_abs_mean"]
    group_oti = ["OTI_simple", "OAR_simple"]
    group_fourier = ["fourier_hf_25%", "fourier_hf_50%"]
    group_jpeg = ["jpeg_q75", "jpeg_q50"]

    def idx(group, with_margin=False):
        out = [all_names.index(n) for n in group if n in all_names]
        if with_margin: out = [all_names.index("victim_margin")] + out
        return out

    # ---- targets ----
    def _fgsm(xs, ys):
        s = fgsm_grad(victim, xs, ys)
        adv = (xs + eps * s).clamp(0, 1)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    yF = chunked(_fgsm, x, y, bs=128).cpu().numpy().astype(int)

    def _pgd(xs, ys):
        adv = pgd(victim, xs, ys, eps=eps)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    yP = chunked(_pgd, x, y, bs=64).cpu().numpy().astype(int)

    def _tx(xs, ys):
        s = fgsm_grad(surrogate, xs, ys)
        adv = (xs + eps * s).clamp(0, 1)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    yT = chunked(_tx, x, y, bs=128).cpu().numpy().astype(int)

    def _meps(xs, ys, eps_max=4 * eps, iters=12):
        s = fgsm_grad(victim, xs, ys)
        lo = torch.zeros(xs.size(0), device=DEVICE)
        hi = torch.full((xs.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xs + mid.view(-1, 1, 1, 1) * s).clamp(0, 1)
            with torch.no_grad():
                f = victim(adv).argmax(1) != ys
            hi = torch.where(f, mid, hi); lo = torch.where(f, lo, mid)
        return hi
    me_np = chunked(_meps, x, y, bs=128).cpu().numpy()

    sets = {
        "pure_stats (A)": idx(group_pure_stats),
        "stats + edge (A+B)": idx(group_pure_stats + group_edge),
        "stats + edge + fourier (A+B+D)": idx(group_pure_stats + group_edge + group_fourier),
        "stats + edge + fourier + jpeg (A+B+D+E) all-cheap-stats": idx(group_pure_stats + group_edge + group_fourier + group_jpeg),
        "OTI alone (C)": idx(group_oti),
        "JPEG alone (E)": idx(group_jpeg),
        "fourier alone (D)": idx(group_fourier),
        "edge alone (B)": idx(group_edge),
        "all_model_free (A+B+C+D+E)": idx(group_pure_stats + group_edge + group_oti + group_fourier + group_jpeg),
        "margin_alone": [all_names.index("victim_margin")],
        "margin + all_model_free": idx(group_pure_stats + group_edge + group_oti + group_fourier + group_jpeg, with_margin=True),
    }

    print(f"\n=== {name}: positive rates  FGSM={yF.mean():.3f}  PGD={yP.mean():.3f}  Transfer={yT.mean():.3f} ===")

    def safe_auc(yy, scores):
        if yy.std() == 0: return float("nan")
        return roc_auc_score(yy, scores)

    print(f"\n  {'set':<58} {'FGSM':>8} {'PGD':>8} {'Transfer':>10} {'min_eps R2':>11}")
    for nm, ix in sets.items():
        if len(ix) == 0: continue
        # logistic for each target
        lr_F = LogisticRegression(max_iter=2000).fit(Xs_all[:, ix], yF)
        a_F = safe_auc(yF, lr_F.predict_proba(Xs_all[:, ix])[:, 1])
        a_P = float("nan")
        if yP.std() > 0:
            lr_P = LogisticRegression(max_iter=2000).fit(Xs_all[:, ix], yP)
            a_P = safe_auc(yP, lr_P.predict_proba(Xs_all[:, ix])[:, 1])
        lr_T = LogisticRegression(max_iter=2000).fit(Xs_all[:, ix], yT)
        a_T = safe_auc(yT, lr_T.predict_proba(Xs_all[:, ix])[:, 1])
        r2 = LinearRegression().fit(Xs_all[:, ix], me_np).score(Xs_all[:, ix], me_np)
        print(f"  {nm:<58} {a_F:.4f}  {a_P:.4f}  {a_T:.4f}    {r2:.4f}")


# ---------- CIFAR-10 ----------
class CifarCNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1); self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*4*4, 128); self.fc2 = nn.Linear(128, n)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


def run_cifar10():
    tf = transforms.ToTensor()
    train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tf)
    def train(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        m = CifarCNN().to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        loader = DataLoader(train_set, 128, shuffle=True, num_workers=2)
        for _ in range(3):
            m.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); F.cross_entropy(m(xb), yb).backward(); opt.step()
        m.eval(); return m

    print("CIFAR-10: training victim+surrogate...")
    t0 = time.time()
    victim = train(0); print(f"  victim {time.time()-t0:.1f}s")
    surrogate = train(1); print(f"  surrogate {time.time()-t0:.1f}s")

    x_all = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y_all = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    with torch.no_grad():
        v_logits = torch.cat([victim(x_all[i:i+512]) for i in range(0, x_all.size(0), 512)])
    keep = v_logits.argmax(1) == y_all
    print(f"  victim acc {keep.float().mean().item():.4f}")
    idx = torch.where(keep)[0]
    x = x_all[idx]; y = y_all[idx]
    sorted_l, _ = v_logits[idx].sort(1, descending=True)
    margin = sorted_l[:, 0] - sorted_l[:, 1]

    print("  computing rich features...")
    feats_dict, _, _ = compute_all_features(x)
    evaluate("CIFAR-10", feats_dict, margin, victim, surrogate, x, y, eps=8/255)


# ---------- Imagenette ----------
def run_imagenette():
    root = os.path.join(DATA_ROOT, "imagenette2-160")
    tf = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(160),
                             transforms.ToTensor()])
    train_set = datasets.ImageFolder(os.path.join(root, "train"), transform=tf)
    val_set = datasets.ImageFolder(os.path.join(root, "val"), transform=tf)

    def build(seed, arch):
        torch.manual_seed(seed); np.random.seed(seed)
        m = (models.resnet18(weights=models.ResNet18_Weights.DEFAULT) if arch == "r18"
             else models.resnet34(weights=models.ResNet34_Weights.DEFAULT))
        m.fc = nn.Linear(m.fc.in_features, 10); m = m.to(DEVICE)
        for p in m.parameters(): p.requires_grad = False
        for p in m.fc.parameters(): p.requires_grad = True
        opt = torch.optim.Adam(m.fc.parameters(), lr=1e-3)
        loader = DataLoader(train_set, 64, shuffle=True, num_workers=2)
        m.train()
        for _ in range(2):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); F.cross_entropy(m(xb), yb).backward(); opt.step()
        for p in m.parameters(): p.requires_grad = True
        m.eval(); return m

    print("Imagenette-160: building victim+surrogate...")
    t0 = time.time()
    victim = build(0, "r18"); print(f"  victim {time.time()-t0:.1f}s")
    surrogate = build(1, "r34"); print(f"  surrogate {time.time()-t0:.1f}s")

    loader = DataLoader(val_set, 64, shuffle=False, num_workers=2)
    xs, ys = [], []
    for xb, yb in loader: xs.append(xb); ys.append(yb)
    x_all = torch.cat(xs).to(DEVICE); y_all = torch.cat(ys).to(DEVICE)
    if x_all.size(0) > 2000:
        sel = torch.randperm(x_all.size(0))[:2000]
        x_all, y_all = x_all[sel], y_all[sel]

    with torch.no_grad():
        v_logits = torch.cat([victim(x_all[i:i+64]) for i in range(0, x_all.size(0), 64)])
    keep = v_logits.argmax(1) == y_all
    print(f"  victim acc {keep.float().mean().item():.4f}")
    idx = torch.where(keep)[0]
    x = x_all[idx]; y = y_all[idx]
    sorted_l, _ = v_logits[idx].sort(1, descending=True)
    margin = sorted_l[:, 0] - sorted_l[:, 1]

    print("  computing rich features...")
    feats_dict, _, _ = compute_all_features(x)
    evaluate("Imagenette-160", feats_dict, margin, victim, surrogate, x, y, eps=8/255)


if __name__ == "__main__":
    t0 = time.time()
    run_cifar10()
    run_imagenette()
    print(f"\nTotal: {time.time()-t0:.1f}s")
