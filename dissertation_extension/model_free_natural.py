"""
Model-free attackability benchmark on natural-image datasets.

Datasets:
  - CIFAR-10 (32x32 natural images, 10 classes) — same data R&G used
  - Imagenette-160 (160x160 natural images, 10 ImageNet classes) — proper test for OTI
                                                                    where object/background matters

Features tested (per image):
  - victim_margin              (top1-top2 logit; requires trained model)
  - ITI                        = mean |Sobel|
  - OTI_simple                 = mean (foreground_threshold_mask * |Sobel|)   model-free mask
  - OTI_gradcam                = mean (gradcam_mask * |Sobel|)                model-based mask
  - OAR_simple                 = fraction of foreground pixels
  - fourier_hf_{25,50}%        = energy fraction above cutoff
  - jpeg_size_q75              = JPEG byte length at quality 75
  - mean_pix, std_pix, entropy_pix

Targets: flipped_FGSM, flipped_PGD, flipped_FGSM_transfer, min_eps_to_flip.

For CIFAR-10:
  - Train a small CNN victim (3 epochs) and surrogate (different seed).
  - No GradCAM (skip — too small) — use simple foreground mask only.

For Imagenette:
  - Use torchvision pretrained ResNet18 / ResNet34, fine-tune the FC head on
    imagenette's 10 classes (~30 seconds each).
  - Compute GradCAM mask from victim's layer4.

Reports: univariate AUROC + multivariate ablation per dataset.
"""
import io, time, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda")
DATA_ROOT = "/tmp/data"


# ---------------- attack utilities ----------------
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


# ---------------- feature utilities ----------------
def sobel_magnitude(x):
    """x: [N,C,H,W] in [0,1]; returns per-pixel Sobel magnitude summed across C [N,1,H,W]."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    C = x.size(1)
    # apply per channel then sum magnitudes
    mags = []
    for c in range(C):
        ch = x[:, c:c+1]
        gx = F.conv2d(ch, kx, padding=1)
        gy = F.conv2d(ch, ky, padding=1)
        mags.append((gx ** 2 + gy ** 2).sqrt())
    return torch.stack(mags, 0).mean(0)


def fourier_high_freq_ratio(x, cutoff_frac=0.25):
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


def gradcam_mask(model, x, target_layer, target_class=None):
    """GradCAM-style spatial attention map, normalised to [0,1] per image."""
    model.eval()
    feats, grads = [], []
    def fwd_hook(_m, _i, o): feats.append(o)
    def bwd_hook(_m, _gi, go): grads.append(go[0])
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    if target_class is None:
        target_class = logits.argmax(1)
    score = logits.gather(1, target_class.view(-1, 1)).squeeze(1).sum()
    model.zero_grad()
    score.backward()
    h1.remove(); h2.remove()
    A = feats[0]                              # [N, C, h, w]
    G = grads[0]                              # [N, C, h, w]
    weights = G.mean(dim=(2, 3), keepdim=True)
    cam = (weights * A).sum(1, keepdim=True).clamp_min(0)  # [N,1,h,w]
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    # normalise per image to [0,1]
    cam_min = cam.flatten(1).min(1).values.view(-1, 1, 1, 1)
    cam_max = cam.flatten(1).max(1).values.view(-1, 1, 1, 1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-9)
    return cam.detach()


# ---------------- CIFAR-10 ----------------
class CifarCNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, n)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.max_pool2d(x, 2)  # 16
        x = F.relu(self.c2(x)); x = F.max_pool2d(x, 2)  # 8
        x = F.relu(self.c3(x)); x = F.max_pool2d(x, 2)  # 4
        x = x.flatten(1)
        x = F.relu(self.fc1(x)); return self.fc2(x)


def run_cifar10(eps_test=8.0/255.0, epochs=3):
    print("\n========== CIFAR-10 ==========")
    tf = transforms.ToTensor()
    train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tf)

    def train_cifar(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        loader = DataLoader(train_set, 128, shuffle=True, num_workers=2)
        m = CifarCNN().to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(epochs):
            m.train()
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
        m.eval(); return m

    t0 = time.time()
    print(" training victim..."); victim = train_cifar(0)
    print(f"   ({time.time()-t0:.1f}s)")
    print(" training surrogate..."); surrogate = train_cifar(1)
    print(f"   ({time.time()-t0:.1f}s)")

    x_all = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y_all = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    with torch.no_grad():
        v_logits = torch.cat([victim(x_all[i:i+512]) for i in range(0, x_all.size(0), 512)])
    keep = v_logits.argmax(1) == y_all
    print(f"  victim acc {keep.float().mean().item():.4f}, n={keep.sum().item()}")
    idx = torch.where(keep)[0]
    x = x_all[idx]; y = y_all[idx]
    sorted_l, _ = v_logits[idx].sort(1, descending=True)
    margin = sorted_l[:, 0] - sorted_l[:, 1]

    print(" computing features...")
    sobel = sobel_magnitude(x)
    iti = sobel.flatten(1).mean(1)
    # simple foreground mask = (luminance > median(luminance) per image)
    lum = x.mean(1, keepdim=True)
    thr = lum.flatten(1).median(1).values.view(-1, 1, 1, 1)
    fg_mask = (lum > thr).float()
    oti_simple = (fg_mask * sobel).flatten(1).mean(1)
    oar = fg_mask.flatten(1).mean(1)
    fhf25 = fourier_high_freq_ratio(x, cutoff_frac=0.25)
    fhf50 = fourier_high_freq_ratio(x, cutoff_frac=0.50)
    print("   computing JPEG byte sizes...")
    jpeg = torch.cat([jpeg_bytesize(x[i:i+1024]) for i in range(0, x.size(0), 1024)])
    pix = x.view(x.size(0), -1)
    mean_pix = pix.mean(1); std_pix = pix.std(1)

    return _benchmark("CIFAR-10", victim, surrogate, x, y, eps_test,
                      margin, iti, oti_simple, None, oar, fhf25, fhf50, jpeg,
                      mean_pix, std_pix)


# ---------------- Imagenette ----------------
def run_imagenette(eps_test=8.0/255.0):
    print("\n========== Imagenette-160 ==========")
    root = os.path.join(DATA_ROOT, "imagenette2-160")
    # imagenette classes ordered alphabetically by folder name
    tf = transforms.Compose([
        transforms.Resize(160), transforms.CenterCrop(160),
        transforms.ToTensor(),
    ])
    train_set = datasets.ImageFolder(os.path.join(root, "train"), transform=tf)
    val_set = datasets.ImageFolder(os.path.join(root, "val"), transform=tf)
    print(f"  train {len(train_set)}, val {len(val_set)}")

    def build_model(seed, arch="resnet18", epochs=2):
        torch.manual_seed(seed); np.random.seed(seed)
        if arch == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 10)
        m = m.to(DEVICE)
        # freeze backbone, train head only
        for p in m.parameters(): p.requires_grad = False
        for p in m.fc.parameters(): p.requires_grad = True
        opt = torch.optim.Adam(m.fc.parameters(), lr=1e-3)
        loader = DataLoader(train_set, 64, shuffle=True, num_workers=2)
        m.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); F.cross_entropy(m(xb), yb).backward(); opt.step()
        # unfreeze (for gradient-based attacks below to work end-to-end)
        for p in m.parameters(): p.requires_grad = True
        m.eval(); return m

    t0 = time.time()
    print(" building victim (ResNet18, head-only finetune)...")
    victim = build_model(0, "resnet18", epochs=2)
    print(f"   ({time.time()-t0:.1f}s)")
    print(" building surrogate (ResNet34, head-only finetune)...")
    surrogate = build_model(1, "resnet34", epochs=2)
    print(f"   ({time.time()-t0:.1f}s)")

    # gather val tensors (subsample to keep it fast)
    val_loader = DataLoader(val_set, 64, shuffle=False, num_workers=2)
    x_chunks, y_chunks = [], []
    for xb, yb in val_loader:
        x_chunks.append(xb); y_chunks.append(yb)
    x_all = torch.cat(x_chunks).to(DEVICE)
    y_all = torch.cat(y_chunks).to(DEVICE)
    # subsample 2000 for speed
    if x_all.size(0) > 2000:
        sel = torch.randperm(x_all.size(0))[:2000]
        x_all, y_all = x_all[sel], y_all[sel]

    with torch.no_grad():
        v_logits = torch.cat([victim(x_all[i:i+64]) for i in range(0, x_all.size(0), 64)])
    keep = v_logits.argmax(1) == y_all
    print(f"  victim val acc {keep.float().mean().item():.4f}, n_correct={keep.sum().item()}")
    idx = torch.where(keep)[0]
    x = x_all[idx]; y = y_all[idx]
    sorted_l, _ = v_logits[idx].sort(1, descending=True)
    margin = sorted_l[:, 0] - sorted_l[:, 1]

    print(" computing features (Sobel/Fourier/JPEG/stats)...")
    sobel = sobel_magnitude(x)
    iti = sobel.flatten(1).mean(1)
    lum = x.mean(1, keepdim=True)
    thr = lum.flatten(1).median(1).values.view(-1, 1, 1, 1)
    fg_mask = (lum > thr).float()
    oti_simple = (fg_mask * sobel).flatten(1).mean(1)
    oar = fg_mask.flatten(1).mean(1)
    fhf25 = fourier_high_freq_ratio(x, cutoff_frac=0.25)
    fhf50 = fourier_high_freq_ratio(x, cutoff_frac=0.50)
    print("   computing JPEG byte sizes...")
    jpeg = torch.cat([jpeg_bytesize(x[i:i+256]) for i in range(0, x.size(0), 256)])
    pix = x.view(x.size(0), -1)
    mean_pix = pix.mean(1); std_pix = pix.std(1)

    print(" computing GradCAM-mask OTI...")
    gradcam_chunks = []
    for i in range(0, x.size(0), 32):
        gradcam_chunks.append(gradcam_mask(victim, x[i:i+32], victim.layer4))
    cam = torch.cat(gradcam_chunks, 0)
    oti_gradcam = (cam * sobel).flatten(1).mean(1)

    return _benchmark("Imagenette-160", victim, surrogate, x, y, eps_test,
                      margin, iti, oti_simple, oti_gradcam, oar, fhf25, fhf50, jpeg,
                      mean_pix, std_pix)


def _benchmark(name, victim, surrogate, x, y, eps,
               margin, iti, oti_simple, oti_gradcam, oar, fhf25, fhf50, jpeg,
               mean_pix, std_pix):
    Nc = x.size(0)

    def _fgsm(xs, ys):
        s = fgsm_grad(victim, xs, ys)
        adv = (xs + eps * s).clamp(0, 1)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    flipped_FGSM = chunked(_fgsm, x, y, bs=128)

    def _pgd(xs, ys):
        adv = pgd(victim, xs, ys, eps=eps)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    flipped_PGD = chunked(_pgd, x, y, bs=64)

    def _tx(xs, ys):
        s = fgsm_grad(surrogate, xs, ys)
        adv = (xs + eps * s).clamp(0, 1)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    flipped_transfer = chunked(_tx, x, y, bs=128)

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
    min_eps_arr = chunked(_meps, x, y, bs=128)

    features = {
        "victim_margin": margin,
        "ITI_sobel_mean": iti,
        "OTI_simple_mask": oti_simple,
        "OAR_simple": oar,
        "fourier_hf_25%": fhf25,
        "fourier_hf_50%": fhf50,
        "jpeg_size_q75": jpeg,
        "mean_pix": mean_pix,
        "std_pix": std_pix,
    }
    if oti_gradcam is not None:
        features["OTI_gradcam_mask"] = oti_gradcam
    feat_names = list(features.keys())
    feats = torch.stack([features[n] for n in feat_names], 1).cpu().numpy()
    me_np = min_eps_arr.cpu().numpy()
    Xs = StandardScaler().fit_transform(feats)
    yF = flipped_FGSM.cpu().numpy().astype(int)
    yP = flipped_PGD.cpu().numpy().astype(int)
    yT = flipped_transfer.cpu().numpy().astype(int)

    print(f"\n=== Univariate ({name}) ===")
    print(f"  pos rates: FGSM={yF.mean():.3f} PGD={yP.mean():.3f} Transfer={yT.mean():.3f}")
    print(f"  {'feature':<22} {'corr(me)':>10} {'AUC_FGSM':>10} {'AUC_PGD':>10} {'AUC_tx':>10}")
    for i, n in enumerate(feat_names):
        cor = np.corrcoef(feats[:, i], me_np)[0, 1]
        def safe_auc(yy):
            if yy.std() == 0: return float("nan")
            a = roc_auc_score(yy, feats[:, i])
            return max(a, 1 - a)
        print(f"  {n:<22} {cor:+.3f}     {safe_auc(yF):.4f}    {safe_auc(yP):.4f}    {safe_auc(yT):.4f}")

    print(f"\n=== Multivariate ablation ({name}) ===")
    def idx_of(*names): return [feat_names.index(n) for n in names if n in feat_names]
    margin_i = idx_of("victim_margin")
    oti_i = idx_of("OTI_simple_mask", "OTI_gradcam_mask", "OAR_simple")
    iti_i = idx_of("ITI_sobel_mean")
    fourier_i = idx_of("fourier_hf_25%", "fourier_hf_50%")
    jpeg_i = idx_of("jpeg_size_q75")
    stats_i = idx_of("mean_pix", "std_pix")
    model_free = oti_i + iti_i + fourier_i + jpeg_i + stats_i
    sets = {
        "margin_alone": margin_i,
        "OTI_alone": oti_i,
        "ITI_alone": iti_i,
        "fourier_alone": fourier_i,
        "jpeg_alone": jpeg_i,
        "all_model_free": model_free,
        "margin + OTI": margin_i + oti_i,
        "margin + image_stats": margin_i + stats_i + iti_i + idx_of("OAR_simple"),
        "margin + all_model_free": margin_i + model_free,
    }
    for tname, y_arr in [("FGSM_self", yF), ("PGD_self", yP), ("FGSM_transfer", yT)]:
        if y_arr.std() == 0:
            print(f"  --- {tname}: degenerate (pos rate {y_arr.mean():.3f})"); continue
        print(f"\n  --- {tname}  (pos rate {y_arr.mean():.3f}) ---")
        for nm, ix in sets.items():
            if len(ix) == 0: continue
            lr = LogisticRegression(max_iter=2000).fit(Xs[:, ix], y_arr)
            auc = roc_auc_score(y_arr, lr.predict_proba(Xs[:, ix])[:, 1])
            print(f"    {nm:<28}: {auc:.4f}")

    print(f"\n=== OLS on min_eps ({name}) ===")
    for nm, ix in sets.items():
        if len(ix) == 0: continue
        r2 = LinearRegression().fit(Xs[:, ix], me_np).score(Xs[:, ix], me_np)
        print(f"  {nm:<28}: R^2 = {r2:.4f}")

    return dict(dataset=name)


if __name__ == "__main__":
    t0 = time.time()
    run_cifar10()
    print(f"\nCIFAR-10 done in {time.time()-t0:.1f}s")
    t1 = time.time()
    run_imagenette()
    print(f"\nImagenette done in {time.time()-t1:.1f}s")
    print(f"\nTotal: {time.time()-t0:.1f}s")
