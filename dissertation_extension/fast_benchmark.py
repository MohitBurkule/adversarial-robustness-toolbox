"""
Fast multi-architecture benchmark on imagenette + imagewoof.
Skips datasets that require download (CIFAR-100, STL-10) — uses only the
pretrained-model + already-downloaded-data path.

Datasets:
  - Imagenette-160 (already downloaded)
  - Imagewoof-160 (already downloaded)

Architectures (8 pretrained, no training needed):
  - ResNet18, ResNet34, ResNet50, ResNet101
  - DenseNet121
  - VGG11-bn
  - MobileNetV3-small
  - EfficientNet-B0

For each dataset, each victim arch is paired cyclically with the next as surrogate.
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


# attack utilities
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


def chunked(fn, *args, bs=64):
    out = []
    for i in range(0, args[0].size(0), bs):
        out.append(fn(*[a[i:i+bs] for a in args]))
    return torch.cat(out)


# features (reused from big_benchmark)
SOBEL_KX = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
LAPLACIAN = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)


def sobel_magnitude(x):
    kx = SOBEL_KX.to(x.device); ky = kx.transpose(2, 3)
    C = x.size(1)
    gx = F.conv2d(x, kx.expand(C, 1, 3, 3), padding=1, groups=C)
    gy = F.conv2d(x, ky.expand(C, 1, 3, 3), padding=1, groups=C)
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
    for i in range(arr.shape[0]):
        img = Image.fromarray(arr[i].transpose(1, 2, 0)) if arr.shape[1] == 3 else Image.fromarray(arr[i, 0])
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality)
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
    bins = 16
    edges = torch.linspace(0, 1, bins + 1, device=x.device)
    h = torch.zeros(N, bins, device=x.device)
    for b in range(bins):
        upper = pix < edges[b + 1] if b < bins - 1 else pix <= edges[b + 1]
        m = (pix >= edges[b]) & upper
        h[:, b] = m.float().sum(1)
    p = h / h.sum(1, keepdim=True).clamp_min(1)
    feats["entropy_pix"] = -(p * torch.log(p.clamp_min(1e-9))).sum(1)
    sob = sobel_magnitude(x)
    feats["sobel_mean"] = sob.flatten(1).mean(1)
    feats["sobel_std"] = sob.flatten(1).std(1)
    feats["sobel_p90"] = torch.quantile(sob.flatten(1), 0.9, dim=1)
    feats["laplacian_abs_mean"] = laplacian_abs(x).flatten(1).mean(1)
    lum = x.mean(1, keepdim=True)
    thr = lum.flatten(1).median(1).values.view(-1, 1, 1, 1)
    fg_mask = (lum > thr).float()
    feats["OTI_simple"] = (fg_mask * sob).flatten(1).mean(1)
    feats["OAR_simple"] = fg_mask.flatten(1).mean(1)
    feats["fourier_hf_25"] = fourier_hf_ratio(x, 0.25)
    feats["fourier_hf_50"] = fourier_hf_ratio(x, 0.50)
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


class SubsetClassifier(nn.Module):
    def __init__(self, base, class_indices):
        super().__init__()
        self.base = base
        self.register_buffer("class_indices", torch.tensor(class_indices))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.base(x)[:, self.class_indices]


def build_pretrained(arch):
    weights_map = {
        "ResNet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "ResNet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "ResNet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "ResNet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
        "DenseNet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
        "VGG11bn": (models.vgg11_bn, models.VGG11_BN_Weights.DEFAULT),
        "MobileNetV3s": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
        "EfficientNetB0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    }
    fn, w = weights_map[arch]
    return fn(weights=w).eval()


def evaluate_block(dataset_name, arch_name, victim, surrogate, x_c, y_c, margin, eps,
                   summary_rows):
    feats = compute_features(x_c)
    feat_names = list(feats.keys())
    F_np = torch.stack([feats[n] for n in feat_names], 1).cpu().numpy()
    margin_np = margin.cpu().numpy().reshape(-1, 1)
    Xs = StandardScaler().fit_transform(F_np)
    margin_s = StandardScaler().fit_transform(margin_np)
    Xs_all = np.concatenate([margin_s, Xs], axis=1)
    all_names = ["victim_margin"] + feat_names

    def _fgsm(xs, ys):
        s = fgsm_grad(victim, xs, ys); adv = (xs + eps * s).clamp(0, 1)
        with torch.no_grad(): return victim(adv).argmax(1) != ys
    yF = chunked(_fgsm, x_c, y_c, bs=32).cpu().numpy().astype(int)

    def _pgd(xs, ys):
        adv = pgd_attack(victim, xs, ys, eps=eps)
        with torch.no_grad(): return victim(adv).argmax(1) != ys
    yP = chunked(_pgd, x_c, y_c, bs=16).cpu().numpy().astype(int)

    def _tx(xs, ys):
        s = fgsm_grad(surrogate, xs, ys); adv = (xs + eps * s).clamp(0, 1)
        with torch.no_grad(): return victim(adv).argmax(1) != ys
    yT = chunked(_tx, x_c, y_c, bs=32).cpu().numpy().astype(int)

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

    print(f"  pos rates: FGSM={yF.mean():.3f} PGD={yP.mean():.3f} Tx={yT.mean():.3f}")
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
        print(f"    {nm:<26} FGSM={a_F:.4f} PGD={a_P:.4f} Tx={a_T:.4f}")
        summary_rows.append({
            "dataset": dataset_name, "arch": arch_name, "feature_set": nm,
            "FGSM": float(a_F), "PGD": float(a_P) if not np.isnan(a_P) else None,
            "Transfer": float(a_T),
            "pos_FGSM": float(yF.mean()), "pos_PGD": float(yP.mean()),
            "pos_Tx": float(yT.mean()), "n": int(x_c.size(0)),
        })


def run_imagenet_subset(name, dirname, class_idx, summary_rows, archs):
    root = os.path.join(DATA_ROOT, dirname)
    if not os.path.isdir(root):
        print(f"  skip {name}: not at {root}"); return
    tf = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(160),
                             transforms.ToTensor()])
    val_set = datasets.ImageFolder(os.path.join(root, "val"), transform=tf)
    loader = DataLoader(val_set, 64, shuffle=False, num_workers=2)
    xs, ys = [], []
    for xb, yb in loader: xs.append(xb); ys.append(yb)
    x_cpu = torch.cat(xs)        # keep on CPU
    y_cpu = torch.cat(ys)
    if x_cpu.size(0) > 1200:
        sel = torch.randperm(x_cpu.size(0))[:1200]
        x_cpu, y_cpu = x_cpu[sel], y_cpu[sel]

    print(f"\n=== {name} ===  {len(archs)} architectures, loading on demand")

    def load(arch):
        return SubsetClassifier(build_pretrained(arch).to(DEVICE), class_idx).to(DEVICE).eval()

    for i, victim_arch in enumerate(archs):
        surrogate_arch = archs[(i + 1) % len(archs)]
        print(f"\n  >> {name} / victim={victim_arch}, surrogate={surrogate_arch}")
        try:
            victim = load(victim_arch)
            surrogate = load(surrogate_arch)
        except Exception as e:
            print(f"    load failed: {e}"); torch.cuda.empty_cache(); continue
        # move data to GPU now that models are loaded
        x_all = x_cpu.to(DEVICE); y_all = y_cpu.to(DEVICE)
        with torch.no_grad():
            v_logits = torch.cat([victim(x_all[k:k+32]) for k in range(0, x_all.size(0), 32)])
        keep = v_logits.argmax(1) == y_all
        acc = keep.float().mean().item()
        print(f"     victim acc {acc:.4f}")
        idx = torch.where(keep)[0]
        if idx.numel() > 800:
            idx = idx[torch.randperm(idx.numel())[:800]]
        if idx.numel() < 50:
            print("    too few correct, skipping")
            del victim, surrogate, x_all, y_all, v_logits
            torch.cuda.empty_cache(); continue
        x = x_all[idx].clone(); y = y_all[idx].clone()
        sorted_l, _ = v_logits[idx].sort(1, descending=True)
        margin = sorted_l[:, 0] - sorted_l[:, 1]
        del v_logits, x_all, y_all
        torch.cuda.empty_cache()
        try:
            evaluate_block(name, victim_arch, victim, surrogate, x, y, margin,
                           eps=4/255, summary_rows=summary_rows)
        except torch.cuda.OutOfMemoryError as e:
            print(f"    OOM during evaluation: {e}")
        del victim, surrogate, x, y, margin
        torch.cuda.empty_cache()


def main():
    imagenette_idx = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    imagewoof_idx = [155, 159, 162, 167, 182, 193, 207, 229, 258, 273]
    archs = ["ResNet18", "ResNet34", "ResNet50", "DenseNet121",
             "MobileNetV3s", "EfficientNetB0", "VGG11bn", "ResNet101"]
    summary_rows = []
    t0 = time.time()
    run_imagenet_subset("Imagenette-160", "imagenette2-160", imagenette_idx, summary_rows, archs)
    run_imagenet_subset("Imagewoof-160", "imagewoof2-160", imagewoof_idx, summary_rows, archs)

    print(f"\n\n############## SUMMARY (total {time.time()-t0:.1f}s) ##############")
    print(f"{'dataset':<16} {'arch':<16} {'feature_set':<27} {'FGSM':>8} {'PGD':>8} {'Tx':>8}")
    for r in summary_rows:
        p = "nan" if r['PGD'] is None else f"{r['PGD']:.4f}"
        print(f"  {r['dataset']:<14} {r['arch']:<16} {r['feature_set']:<27} "
              f"{r['FGSM']:.4f} {p:>8} {r['Transfer']:.4f}")
    with open("/tmp/fast_bench_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2, default=float)
    print("Saved /tmp/fast_bench_summary.json")


if __name__ == "__main__":
    main()
