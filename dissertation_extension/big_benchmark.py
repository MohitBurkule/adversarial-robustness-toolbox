"""
Comprehensive multi-dataset / multi-architecture benchmark of model-free
attackability predictors vs trained-model margin.

Datasets (5):
  - CIFAR-10           (32x32, custom CNN, 3 epochs)
  - CIFAR-100          (32x32, custom CNN, 5 epochs)
  - STL-10             (96x96, pretrained ResNet18 + head finetune)
  - Imagenette-160     (160x160, pretrained models, no finetune — imagenet subset)
  - Imagewoof-160      (160x160, pretrained models, no finetune — imagenet subset, harder dog breeds)

Architectures evaluated as victim (and surrogate from a different one):
  - On imagenette/imagewoof: ResNet18, ResNet34, ResNet50, DenseNet-121,
    MobileNetV3-small, EfficientNet-B0  (all pretrained on ImageNet)
  - On CIFAR: own CifarCNN (different seeds = different "architectures")
  - On STL-10: ResNet18 fine-tuned head, ResNet34 surrogate

Per (dataset, architecture) pair, we compute:
  - 13–16 pure pixel statistics (mean, std, skew, kurtosis, MAD, percentiles,
    range, IQR, per-channel stats, entropy)
  - Edge stats (Sobel mean/std/p90, Laplacian abs mean)
  - OTI_simple (luminance-mask × Sobel)
  - Fourier high-freq energy (25%, 50%)
  - JPEG byte size (q75, q50)
  - victim_margin (top1-top2 logit)

Targets: flipped_FGSM, flipped_PGD, flipped_transfer, min_eps_to_flip.

Reports a single big table at the end:
  dataset | arch | FGSM AUROC for {margin, pure_stats, all_cheap, all_model_free}
"""
import io, time, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda")
DATA_ROOT = "/tmp/data"


# ============== attack utilities ==============
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def pgd_attack(model, x, y, eps, steps=10):
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


# ============== features ==============
SOBEL_KX = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
LAPLACIAN = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)


def sobel_magnitude(x):
    kx = SOBEL_KX.to(x.device); ky = kx.transpose(2, 3)
    C = x.size(1)
    kx_g = kx.expand(C, 1, 3, 3); ky_g = ky.expand(C, 1, 3, 3)
    gx = F.conv2d(x, kx_g, padding=1, groups=C)
    gy = F.conv2d(x, ky_g, padding=1, groups=C)
    return (gx ** 2 + gy ** 2).sqrt().mean(1, keepdim=True)


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


def compute_features(x):
    N, C, H, W = x.shape
    pix = x.view(N, -1)
    feats = {}
    mean = pix.mean(1); std = pix.std(1); sd = std.clamp_min(1e-6)
    feats["mean_pix"] = mean; feats["std_pix"] = std
    centered = pix - mean.unsqueeze(1)
    feats["skew_pix"] = ((centered / sd.unsqueeze(1)) ** 3).mean(1)
    feats["kurt_pix"] = ((centered / sd.unsqueeze(1)) ** 4).mean(1) - 3.0
    feats["MAD_pix"] = centered.abs().mean(1)
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=x.device)
    quants = torch.quantile(pix, qs, dim=1)
    for qi, qn in zip(range(5), ["p10", "p25", "p50", "p75", "p90"]):
        feats[f"{qn}_pix"] = quants[qi]
    feats["range_pix"] = pix.max(1).values - pix.min(1).values
    feats["IQR_pix"] = quants[3] - quants[1]
    if C == 3:
        for ci in range(3):
            feats[f"mean_ch{ci}"] = x[:, ci].flatten(1).mean(1)
            feats[f"std_ch{ci}"] = x[:, ci].flatten(1).std(1)
    # entropy
    bins = 16
    edges = torch.linspace(0, 1, bins + 1, device=x.device)
    h = torch.zeros(N, bins, device=x.device)
    for b in range(bins):
        upper = pix < edges[b + 1] if b < bins - 1 else pix <= edges[b + 1]
        m = (pix >= edges[b]) & upper
        h[:, b] = m.float().sum(1)
    p = h / h.sum(1, keepdim=True).clamp_min(1)
    feats["entropy_pix"] = -(p * torch.log(p.clamp_min(1e-9))).sum(1)
    # edges
    sob = sobel_magnitude(x)
    feats["sobel_mean"] = sob.flatten(1).mean(1)
    feats["sobel_std"] = sob.flatten(1).std(1)
    feats["sobel_p90"] = torch.quantile(sob.flatten(1), 0.9, dim=1)
    feats["laplacian_abs_mean"] = laplacian_abs(x).flatten(1).mean(1)
    # OTI
    lum = x.mean(1, keepdim=True)
    thr = lum.flatten(1).median(1).values.view(-1, 1, 1, 1)
    fg_mask = (lum > thr).float()
    feats["OTI_simple"] = (fg_mask * sob).flatten(1).mean(1)
    feats["OAR_simple"] = fg_mask.flatten(1).mean(1)
    # fourier
    feats["fourier_hf_25"] = fourier_hf_ratio(x, 0.25)
    feats["fourier_hf_50"] = fourier_hf_ratio(x, 0.50)
    # jpeg
    feats["jpeg_q75"] = torch.cat([jpeg_bytesize(x[i:i+256], 75) for i in range(0, N, 256)])
    feats["jpeg_q50"] = torch.cat([jpeg_bytesize(x[i:i+256], 50) for i in range(0, N, 256)])
    return feats


PURE_STATS = ["mean_pix", "std_pix", "skew_pix", "kurt_pix", "MAD_pix",
              "p10_pix", "p25_pix", "p50_pix", "p75_pix", "p90_pix",
              "range_pix", "IQR_pix", "entropy_pix",
              "mean_ch0", "mean_ch1", "mean_ch2", "std_ch0", "std_ch1", "std_ch2"]
EDGE = ["sobel_mean", "sobel_std", "sobel_p90", "laplacian_abs_mean"]
OTI_GROUP = ["OTI_simple", "OAR_simple"]
FOURIER = ["fourier_hf_25", "fourier_hf_50"]
JPEG = ["jpeg_q75", "jpeg_q50"]


def evaluate_block(dataset_name, arch_name, victim, surrogate, x_c, y_c, margin, eps,
                   summary_rows):
    print(f"\n----- {dataset_name} / {arch_name}  (n={x_c.size(0)}) -----")
    feats = compute_features(x_c)
    feat_names = list(feats.keys())
    F_np = torch.stack([feats[n] for n in feat_names], 1).cpu().numpy()
    margin_np = margin.cpu().numpy().reshape(-1, 1)
    Xs = StandardScaler().fit_transform(F_np)
    margin_s = StandardScaler().fit_transform(margin_np)
    Xs_all = np.concatenate([margin_s, Xs], axis=1)
    all_names = ["victim_margin"] + feat_names

    # targets
    def _fgsm(xs, ys):
        s = fgsm_grad(victim, xs, ys); adv = (xs + eps * s).clamp(0, 1)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    yF = chunked(_fgsm, x_c, y_c, bs=64).cpu().numpy().astype(int)

    def _pgd(xs, ys):
        adv = pgd_attack(victim, xs, ys, eps=eps)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    yP = chunked(_pgd, x_c, y_c, bs=32).cpu().numpy().astype(int)

    def _tx(xs, ys):
        s = fgsm_grad(surrogate, xs, ys); adv = (xs + eps * s).clamp(0, 1)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    yT = chunked(_tx, x_c, y_c, bs=64).cpu().numpy().astype(int)

    def idx(*groups):
        names = sum([g for g in groups], [])
        return [all_names.index(n) for n in names if n in all_names]

    sets = {
        "margin": [all_names.index("victim_margin")],
        "OTI_alone": idx(OTI_GROUP),
        "JPEG_alone": idx(JPEG),
        "pure_stats": idx(PURE_STATS),
        "stats+edge+fourier+jpeg": idx(PURE_STATS, EDGE, FOURIER, JPEG),
        "all_model_free": idx(PURE_STATS, EDGE, OTI_GROUP, FOURIER, JPEG),
        "margin+all_model_free": [all_names.index("victim_margin")] + idx(PURE_STATS, EDGE, OTI_GROUP, FOURIER, JPEG),
    }

    def safe_auc(yy, scores):
        if yy.std() == 0: return float("nan")
        return roc_auc_score(yy, scores)

    print(f"  pos rates: FGSM={yF.mean():.3f}  PGD={yP.mean():.3f}  Transfer={yT.mean():.3f}")
    for nm, ix in sets.items():
        if not ix: continue
        lr_F = LogisticRegression(max_iter=2000).fit(Xs_all[:, ix], yF)
        a_F = safe_auc(yF, lr_F.predict_proba(Xs_all[:, ix])[:, 1])
        a_P = float("nan")
        if yP.std() > 0:
            lr_P = LogisticRegression(max_iter=2000).fit(Xs_all[:, ix], yP)
            a_P = safe_auc(yP, lr_P.predict_proba(Xs_all[:, ix])[:, 1])
        lr_T = LogisticRegression(max_iter=2000).fit(Xs_all[:, ix], yT)
        a_T = safe_auc(yT, lr_T.predict_proba(Xs_all[:, ix])[:, 1])
        print(f"  {nm:<26} FGSM={a_F:.4f} PGD={a_P:.4f} Tx={a_T:.4f}")
        summary_rows.append({
            "dataset": dataset_name, "arch": arch_name, "feature_set": nm,
            "FGSM": a_F, "PGD": a_P, "Transfer": a_T,
            "pos_rate_FGSM": float(yF.mean()), "pos_rate_PGD": float(yP.mean()),
            "pos_rate_Tx": float(yT.mean()), "n": int(x_c.size(0)),
        })


# ============== CIFAR datasets ==============
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


class CifarCNNwide(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, padding=1); self.c2 = nn.Conv2d(64, 128, 3, padding=1)
        self.c3 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc1 = nn.Linear(128*4*4, 256); self.fc2 = nn.Linear(256, n)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


def train_cifar_cnn(seed, train_set, n_classes, epochs, cls):
    torch.manual_seed(seed); np.random.seed(seed)
    m = cls(n_classes).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loader = DataLoader(train_set, 128, shuffle=True, num_workers=2)
    for _ in range(epochs):
        m.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    m.eval(); return m


def run_cifar(name, ds_cls, n_classes, epochs, summary_rows):
    tf = transforms.ToTensor()
    train_set = ds_cls(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = ds_cls(DATA_ROOT, train=False, download=True, transform=tf)
    x_all = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y_all = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    for arch_name, cls in [("CifarCNN", CifarCNN), ("CifarCNNwide", CifarCNNwide)]:
        print(f"\n=== {name} / {arch_name} ===")
        t0 = time.time()
        victim = train_cifar_cnn(0, train_set, n_classes, epochs, cls)
        surrogate = train_cifar_cnn(1, train_set, n_classes, epochs,
                                    CifarCNNwide if cls is CifarCNN else CifarCNN)
        print(f"  trained in {time.time()-t0:.1f}s")
        with torch.no_grad():
            v_logits = torch.cat([victim(x_all[i:i+512]) for i in range(0, x_all.size(0), 512)])
        keep = v_logits.argmax(1) == y_all
        print(f"  victim acc {keep.float().mean().item():.4f}")
        idx = torch.where(keep)[0]
        if idx.numel() > 3000:
            idx = idx[torch.randperm(idx.numel())[:3000]]
        x = x_all[idx]; y = y_all[idx]
        sorted_l, _ = v_logits[idx].sort(1, descending=True)
        margin = sorted_l[:, 0] - sorted_l[:, 1]
        evaluate_block(name, arch_name, victim, surrogate, x, y, margin,
                       eps=8/255, summary_rows=summary_rows)


# ============== STL-10 ==============
def run_stl10(summary_rows):
    tf = transforms.Compose([transforms.ToTensor()])
    try:
        train_set = datasets.STL10(DATA_ROOT, split="train", download=True, transform=tf)
        test_set = datasets.STL10(DATA_ROOT, split="test", download=True, transform=tf)
    except Exception as e:
        print(f"STL-10 download failed: {e}"); return

    def build(seed, arch):
        torch.manual_seed(seed); np.random.seed(seed)
        if arch == "r18": m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif arch == "r34": m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else: m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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

    for arch_name, arch in [("ResNet18", "r18"), ("ResNet34", "r34")]:
        print(f"\n=== STL-10 / {arch_name} ===")
        t0 = time.time()
        victim = build(0, arch)
        surrogate_arch = "r34" if arch == "r18" else "r18"
        surrogate = build(1, surrogate_arch)
        print(f"  built in {time.time()-t0:.1f}s")
        loader = DataLoader(test_set, 64, shuffle=False, num_workers=2)
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
        if idx.numel() > 1500:
            idx = idx[torch.randperm(idx.numel())[:1500]]
        x = x_all[idx]; y = y_all[idx]
        sorted_l, _ = v_logits[idx].sort(1, descending=True)
        margin = sorted_l[:, 0] - sorted_l[:, 1]
        evaluate_block("STL-10", arch_name, victim, surrogate, x, y, margin,
                       eps=8/255, summary_rows=summary_rows)


# ============== Imagenette / Imagewoof ==============
# Class index mapping from folder name (which is imagenet class) to imagenet-1k label index
IMAGENETTE_CLASSES = ["n01440764", "n02102040", "n02979186", "n03000684",
                      "n03028079", "n03394916", "n03417042", "n03425413",
                      "n03445777", "n03888257"]
# Need imagenet class indices for these names
def load_imagenet_class_index():
    # torchvision has the mapping via ImageNet meta, but it's not built in. Use known map.
    # These are well-known imagenet-1k indices for imagenette/imagewoof.
    imagenette_idx = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    imagewoof_idx = [155, 159, 162, 167, 182, 193, 207, 229, 258, 273]
    return imagenette_idx, imagewoof_idx


class SubsetClassifier(nn.Module):
    """Wrap a pretrained ImageNet model, output only logits for selected class indices,
    plus imagenet normalisation since pretrained models expect it."""
    def __init__(self, base, class_indices):
        super().__init__()
        self.base = base
        self.register_buffer("class_indices", torch.tensor(class_indices))
        # imagenet normalisation
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):
        x = (x - self.mean) / self.std
        logits = self.base(x)
        return logits[:, self.class_indices]


def build_pretrained(arch):
    if arch == "ResNet18": return models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    if arch == "ResNet34": return models.resnet34(weights=models.ResNet34_Weights.DEFAULT).eval()
    if arch == "ResNet50": return models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
    if arch == "DenseNet121": return models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).eval()
    if arch == "MobileNetV3s": return models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).eval()
    if arch == "EfficientNetB0": return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).eval()
    if arch == "VGG11": return models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT).eval()
    raise ValueError(arch)


def run_imagenet_subset(name, dirname, class_idx, summary_rows,
                        archs=("ResNet18", "ResNet34", "ResNet50",
                               "DenseNet121", "MobileNetV3s", "EfficientNetB0")):
    root = os.path.join(DATA_ROOT, dirname)
    tf = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(160),
                             transforms.ToTensor()])
    val_set = datasets.ImageFolder(os.path.join(root, "val"), transform=tf)
    loader = DataLoader(val_set, 64, shuffle=False, num_workers=2)
    xs, ys = [], []
    for xb, yb in loader: xs.append(xb); ys.append(yb)
    x_all = torch.cat(xs).to(DEVICE); y_all = torch.cat(ys).to(DEVICE)
    if x_all.size(0) > 1500:
        sel = torch.randperm(x_all.size(0))[:1500]
        x_all, y_all = x_all[sel], y_all[sel]

    print(f"\n=== {name} ===  {len(archs)} architectures, loading on demand")
    x_cpu = x_all.cpu(); y_cpu = y_all.cpu()
    del x_all, y_all
    torch.cuda.empty_cache()

    def load(arch):
        return SubsetClassifier(build_pretrained(arch).to(DEVICE), class_idx).to(DEVICE).eval()

    for i, victim_arch in enumerate(archs):
        surrogate_arch = archs[(i + 1) % len(archs)]
        print(f"\n  >> {name} / victim={victim_arch}, surrogate={surrogate_arch}")
        try:
            victim = load(victim_arch); surrogate = load(surrogate_arch)
        except Exception as e:
            print(f"    load failed: {e}"); torch.cuda.empty_cache(); continue
        x_g = x_cpu.to(DEVICE); y_g = y_cpu.to(DEVICE)
        with torch.no_grad():
            v_logits = torch.cat([victim(x_g[k:k+32]) for k in range(0, x_g.size(0), 32)])
        keep = v_logits.argmax(1) == y_g
        acc = keep.float().mean().item()
        print(f"    victim acc {acc:.4f}")
        idx = torch.where(keep)[0]
        if idx.numel() > 800:
            idx = idx[torch.randperm(idx.numel())[:800]]
        if idx.numel() < 50:
            print("    too few correct, skipping")
            del victim, surrogate, x_g, y_g, v_logits
            torch.cuda.empty_cache(); continue
        x = x_g[idx].clone(); y = y_g[idx].clone()
        sorted_l, _ = v_logits[idx].sort(1, descending=True)
        margin = sorted_l[:, 0] - sorted_l[:, 1]
        del v_logits, x_g, y_g
        torch.cuda.empty_cache()
        try:
            evaluate_block(name, victim_arch, victim, surrogate, x, y, margin,
                           eps=4/255, summary_rows=summary_rows)
        except torch.cuda.OutOfMemoryError as e:
            print(f"    OOM during eval: {e}")
        del victim, surrogate, x, y, margin
        torch.cuda.empty_cache()


# ============== main ==============
def main():
    summary_rows = []
    t0 = time.time()

    # CIFAR-10
    print("\n\n############## CIFAR-10 ##############")
    run_cifar("CIFAR-10", datasets.CIFAR10, n_classes=10, epochs=3, summary_rows=summary_rows)

    # CIFAR-100
    print("\n\n############## CIFAR-100 ##############")
    run_cifar("CIFAR-100", datasets.CIFAR100, n_classes=100, epochs=5, summary_rows=summary_rows)

    # STL-10
    print("\n\n############## STL-10 ##############")
    run_stl10(summary_rows)

    # Imagenette / Imagewoof
    imagenette_idx, imagewoof_idx = load_imagenet_class_index()
    print("\n\n############## Imagenette-160 ##############")
    run_imagenet_subset("Imagenette-160", "imagenette2-160", imagenette_idx, summary_rows)
    print("\n\n############## Imagewoof-160 ##############")
    run_imagenet_subset("Imagewoof-160", "imagewoof2-160", imagewoof_idx, summary_rows)

    print(f"\n\n############## SUMMARY (total {time.time()-t0:.1f}s) ##############")
    print(f"{'dataset':<15} {'arch':<18} {'feature_set':<27} {'FGSM':>8} {'PGD':>8} {'Tx':>8}")
    for r in summary_rows:
        a_F = "nan" if np.isnan(r['FGSM']) else f"{r['FGSM']:.4f}"
        a_P = "nan" if np.isnan(r['PGD']) else f"{r['PGD']:.4f}"
        a_T = "nan" if np.isnan(r['Transfer']) else f"{r['Transfer']:.4f}"
        print(f"  {r['dataset']:<13} {r['arch']:<18} {r['feature_set']:<27} {a_F:>8} {a_P:>8} {a_T:>8}")

    with open("/tmp/big_benchmark_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2, default=float)
    print("\nSaved /tmp/big_benchmark_summary.json")


if __name__ == "__main__":
    main()
