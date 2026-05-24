"""
Benchmark model-free image-attackability measures vs our existing baselines.

Implements:
  - OTI (Object Texture Intensity) — Liang et al., AAAI 2026
      OTI(x) = mean over pixels of  object_mask(x) * |Sobel(x)|
  - OAR (Object Area Ratio)             = mean(object_mask)
  - ITI (Image Texture Intensity)       = mean(|Sobel(x)|)   (== our edge_density)
  - Multiple object_mask thresholds (sensitivity)
  - Fourier high-frequency energy ratio
  - JPEG compressibility (file size at fixed quality)
  - Pixel std, mean, entropy, ink_area (already from image_stats_analysis)

Targets on a trained victim:
  - flipped_FGSM, flipped_PGD, flipped_FGSM_transfer, min_eps_to_flip

For each feature: univariate AUROC + correlation with min_eps.
Multivariate: how do OTI / Fourier / JPEG stack against margin alone, image_stats alone,
margin+image_stats, margin+stats+OTI+Fourier+JPEG.
"""
import io, time, json
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

DEVICE = torch.device("cuda")
EPS_TEST = 15.0 / 255.0
EPOCHS_VICTIM = 10
BATCH = 128


class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*12*12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2); x = self.do1(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.do2(x); return self.fc2(x)


def train_one(seed, train_set, epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    m = CNN().to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(epochs):
        m.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    m.eval(); return m


def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def pgd(model, x, y, eps=EPS_TEST, steps=10):
    alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        F.cross_entropy(model(x_adv), y).backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)
    return x_adv.detach()


def sobel_magnitude(x):
    """x: [N,1,H,W] in [0,1]. Return per-pixel Sobel magnitude [N,1,H,W]."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return (gx ** 2 + gy ** 2).sqrt()


def fourier_high_freq_ratio(x, cutoff_frac=0.25):
    """Fraction of total spectral energy above cutoff_frac of Nyquist."""
    N, C, H, W = x.shape
    X = torch.fft.fft2(x)
    mag2 = (X.real ** 2 + X.imag ** 2)  # power spectrum
    # frequency grid (use rfft style: low freq at center after fftshift)
    fy = torch.fft.fftfreq(H, device=x.device).abs()
    fx = torch.fft.fftfreq(W, device=x.device).abs()
    grid = (fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2).sqrt()  # H,W
    high_mask = (grid > cutoff_frac * 0.5).float().unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    total = mag2.flatten(1).sum(1)
    high = (mag2 * high_mask).flatten(1).sum(1)
    return high / total.clamp_min(1e-9)


def jpeg_bytesize(x, quality=75):
    """x: [N,1,H,W] in [0,1]. Returns JPEG-encoded byte length per image."""
    # numpy on CPU
    arr = (x.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
    sizes = []
    for i in range(arr.shape[0]):
        img = Image.fromarray(arr[i, 0])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        sizes.append(buf.tell())
    return torch.tensor(sizes, device=x.device, dtype=torch.float32)


def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    x_test = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = x_test.size(0)

    print("Training victim + surrogate...")
    t0 = time.time()
    victim = train_one(0, train_set, EPOCHS_VICTIM)
    surrogate = train_one(1, train_set, EPOCHS_VICTIM)
    print(f"  done ({time.time()-t0:.1f}s)")

    # restrict to victim-correct samples
    with torch.no_grad():
        v_logits = []
        for i in range(0, N, 512):
            v_logits.append(victim(x_test[i:i+512]))
        v_logits = torch.cat(v_logits)
    keep = v_logits.argmax(1) == y_test
    print(f"Victim acc {keep.float().mean().item():.4f}, n={keep.sum().item()}")
    idx = torch.where(keep)[0]
    x_c = x_test[idx]; y_c = y_test[idx]
    sorted_l, _ = v_logits[idx].sort(1, descending=True)
    margin_c = sorted_l[:, 0] - sorted_l[:, 1]
    Nc = x_c.size(0)

    print("Computing model-free features...")
    sobel = sobel_magnitude(x_c)                                    # [Nc,1,H,W]
    iti = sobel.flatten(1).mean(1)                                  # mean |Sobel|

    # OTI with several object-mask thresholds (Fashion-MNIST has black background)
    masks = {}
    for thr in [0.05, 0.10, 0.20, 0.40]:
        masks[thr] = (x_c > thr).float()
    oti = {}
    oar = {}
    for thr, m in masks.items():
        oti[thr] = (m * sobel).flatten(1).mean(1)
        oar[thr] = m.flatten(1).mean(1)

    # pixel stats
    pix = x_c.view(Nc, -1)
    mean_pix = pix.mean(1); std_pix = pix.std(1)
    bins = 10
    edges = torch.linspace(0, 1, bins + 1, device=DEVICE)
    h = torch.zeros(Nc, bins, device=DEVICE)
    for b in range(bins):
        h[:, b] = ((pix >= edges[b]) & (pix < edges[b+1] if b < bins-1 else pix <= edges[b+1])).float().sum(1)
    p = h / h.sum(1, keepdim=True).clamp_min(1)
    entropy_pix = -(p * torch.log(p.clamp_min(1e-9))).sum(1)

    # Fourier
    fhf25 = fourier_high_freq_ratio(x_c, cutoff_frac=0.25)
    fhf50 = fourier_high_freq_ratio(x_c, cutoff_frac=0.50)

    # JPEG size (CPU)
    print("  computing JPEG byte sizes...")
    jpeg = []
    for i in range(0, Nc, 1024):
        jpeg.append(jpeg_bytesize(x_c[i:i+1024], quality=75))
    jpeg_size = torch.cat(jpeg)

    feats_dict = {
        "victim_margin": margin_c,
        "ITI_sobel_mean": iti,
        "OTI_thr0.05": oti[0.05],
        "OTI_thr0.10": oti[0.10],
        "OTI_thr0.20": oti[0.20],
        "OTI_thr0.40": oti[0.40],
        "OAR_thr0.10": oar[0.10],
        "fourier_hf_25%": fhf25,
        "fourier_hf_50%": fhf50,
        "jpeg_size_q75": jpeg_size,
        "mean_pix": mean_pix,
        "std_pix": std_pix,
        "entropy_pix": entropy_pix,
    }
    feat_names = list(feats_dict.keys())
    feats = torch.stack([feats_dict[n] for n in feat_names], 1).cpu().numpy()
    Xs = StandardScaler().fit_transform(feats)

    print("Computing attack targets on victim...")
    def chunked(fn, *args, bs=256):
        out = []
        for i in range(0, args[0].size(0), bs):
            out.append(fn(*[a[i:i+bs] for a in args]))
        return torch.cat(out)

    def _fgsm(xs, ys):
        s = fgsm_grad(victim, xs, ys)
        adv = (xs + EPS_TEST * s).clamp(0, 1)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    flipped_FGSM = chunked(_fgsm, x_c, y_c, bs=512)

    def _pgd(xs, ys):
        adv = pgd(victim, xs, ys)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    flipped_PGD = chunked(_pgd, x_c, y_c, bs=256)

    def _tx(xs, ys):
        s = fgsm_grad(surrogate, xs, ys)
        adv = (xs + EPS_TEST * s).clamp(0, 1)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    flipped_transfer = chunked(_tx, x_c, y_c, bs=512)

    def _meps(xs, ys, eps_max=0.3, iters=15):
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
    min_eps_arr = chunked(_meps, x_c, y_c, bs=512)
    me_np = min_eps_arr.cpu().numpy()

    print("\n=== Univariate analysis (per-feature) ===")
    print(f"{'feature':<22} {'corr(me)':>10} {'AUC_FGSM':>10} {'AUC_PGD':>10} {'AUC_tx':>10}")
    for i, n in enumerate(feat_names):
        cor = np.corrcoef(feats[:, i], me_np)[0, 1]
        a_f = max(roc_auc_score(flipped_FGSM.cpu().numpy(), feats[:, i]),
                  1 - roc_auc_score(flipped_FGSM.cpu().numpy(), feats[:, i]))
        a_p = max(roc_auc_score(flipped_PGD.cpu().numpy(), feats[:, i]),
                  1 - roc_auc_score(flipped_PGD.cpu().numpy(), feats[:, i]))
        a_t = max(roc_auc_score(flipped_transfer.cpu().numpy(), feats[:, i]),
                  1 - roc_auc_score(flipped_transfer.cpu().numpy(), feats[:, i]))
        print(f"  {n:<22} {cor:+.3f}     {a_f:.4f}    {a_p:.4f}    {a_t:.4f}")

    print("\n=== Multivariate ablation ===")
    margin_idx = [feat_names.index("victim_margin")]
    oti_idx = [feat_names.index(n) for n in feat_names if n.startswith("OTI_") or n.startswith("OAR_") or n.startswith("ITI_")]
    fourier_idx = [feat_names.index(n) for n in feat_names if n.startswith("fourier_")]
    jpeg_idx = [feat_names.index("jpeg_size_q75")]
    stats_idx = [feat_names.index(n) for n in ["mean_pix", "std_pix", "entropy_pix"]]
    model_free_idx = oti_idx + fourier_idx + jpeg_idx + stats_idx

    sets = {
        "margin_alone": margin_idx,
        "OTI_alone (3-thr)": [feat_names.index(n) for n in ["OTI_thr0.05", "OTI_thr0.10", "OTI_thr0.20"]],
        "fourier_alone": fourier_idx,
        "jpeg_alone": jpeg_idx,
        "ITI_alone": [feat_names.index("ITI_sobel_mean")],
        "all_model_free": model_free_idx,
        "margin + OTI": margin_idx + [feat_names.index("OTI_thr0.10")],
        "margin + image_stats": margin_idx + stats_idx + [feat_names.index("ITI_sobel_mean"), feat_names.index("OAR_thr0.10")],
        "margin + all_model_free": margin_idx + model_free_idx,
    }

    for tname, tgt in [("FGSM_self", flipped_FGSM),
                       ("PGD_self", flipped_PGD),
                       ("FGSM_transfer", flipped_transfer)]:
        y_arr = tgt.cpu().numpy().astype(int)
        if y_arr.std() == 0:
            print(f"  {tname}: degenerate"); continue
        print(f"\n  --- {tname}  (pos rate {y_arr.mean():.3f}) ---")
        for nm, ix in sets.items():
            lr = LogisticRegression(max_iter=2000).fit(Xs[:, ix], y_arr)
            auc = roc_auc_score(y_arr, lr.predict_proba(Xs[:, ix])[:, 1])
            print(f"    {nm:<28}: {auc:.4f}")

    print("\n=== OLS on min_eps ===")
    for nm, ix in sets.items():
        r2 = LinearRegression().fit(Xs[:, ix], me_np).score(Xs[:, ix], me_np)
        print(f"  {nm:<28}: R^2 = {r2:.4f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.1f}s")
