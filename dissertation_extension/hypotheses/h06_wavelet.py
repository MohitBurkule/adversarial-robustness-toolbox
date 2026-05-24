"""
Hypothesis H06: Wavelet decomposition energy at multiple spatial scales is a
stronger model-free vulnerability predictor than Fourier high-frequency ratio.

Prior tests showed Fourier high-freq ratio AUROC ~0.55-0.65 on Fashion-MNIST.
Wavelets give multi-scale information (orientation x scale subbands) and should
be strictly more informative than a single scalar FFT high-freq ratio.

Pipeline:
  1. Train a small CNN victim on Fashion-MNIST (10 epochs), matching the
     architecture used in ../diagnostic_test.py. Train a second CNN (seed 1)
     as a surrogate for FGSM transfer.
  2. For every test sample compute:
       - 2D wavelet decomposition (Haar) at 3 levels with pywt.wavedec2.
         Per-subband sum-of-absolute-coefficients gives 3 levels * 3 subbands
         (LH, HL, HH) = 9 wavelet-energy features.
       - Baselines: victim_margin, mean_pix, std_pix, sobel_mean.
       - Fourier high-frequency energy ratio (energy outside a central
         low-frequency disc) for direct comparison.
  3. Adversarial targets on samples the victim classifies correctly:
       - flipped_FGSM at eps = 15/255 (victim model)
       - flipped_PGD  at eps = 15/255 (victim model, 10 PGD steps)
       - FGSM_transfer at eps = 15/255 (crafted on surrogate, eval on victim)
  4. Report univariate AUROC for every feature.
  5. Multivariate: does margin + 9 wavelet energies beat margin alone?
     Does it beat margin + Fourier high-freq ratio?

Run with the venv that has pywavelets installed:
    /mnt/1tbone/trashy/adversarial-robustness-toolbox/.venv/bin/python \
        /mnt/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h06_wavelet.py

If pywavelets is not present:
    /mnt/1tbone/trashy/adversarial-robustness-toolbox/.venv/bin/pip install pywavelets
"""
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import pywt
except ImportError as e:
    sys.stderr.write(
        "PyWavelets not installed. Install with:\n"
        "  /mnt/1tbone/trashy/adversarial-robustness-toolbox/.venv/bin/pip install pywavelets\n"
    )
    raise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
WAVELET = "haar"
N_LEVELS = 3
PGD_STEPS = 10
PGD_ALPHA = 2.0 / 255.0


# ---- victim model (same architecture as diagnostic_test.py) ----
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
        print(f"  seed={seed} epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ---- attacks ----
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def attack_fgsm(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def attack_pgd(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x_orig = x.detach()
    # small random start within the L_inf ball
    delta = (torch.empty_like(x).uniform_(-eps, eps))
    adv = (x_orig + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def attack_fgsm_transfer(target, surrogate, x, y, eps=EPS_TEST):
    sign = fgsm_grad(surrogate, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (target(adv).argmax(1) != y)


# ---- features ----
def victim_margin(model, x):
    """Top-1 minus top-2 logit margin from the victim, per sample."""
    margins = []
    with torch.no_grad():
        for i in range(0, x.size(0), 512):
            logits = model(x[i:i+512])
            s, _ = logits.sort(1, descending=True)
            margins.append(s[:, 0] - s[:, 1])
    return torch.cat(margins)


def sobel_mean(x_np):
    """Mean absolute Sobel gradient magnitude per image.
       x_np: (N, 28, 28) float in [0,1]"""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    N, H, W = x_np.shape
    out = np.zeros(N, dtype=np.float32)
    # use torch conv2d for speed
    t = torch.from_numpy(x_np).unsqueeze(1).to(DEVICE)
    kx_t = torch.from_numpy(kx).view(1, 1, 3, 3).to(DEVICE)
    ky_t = torch.from_numpy(ky).view(1, 1, 3, 3).to(DEVICE)
    with torch.no_grad():
        gx = F.conv2d(t, kx_t, padding=1)
        gy = F.conv2d(t, ky_t, padding=1)
        mag = (gx.pow(2) + gy.pow(2)).sqrt()
        out = mag.mean(dim=(1, 2, 3)).cpu().numpy()
    return out


def fourier_high_freq_ratio(x_np, cutoff_frac=0.25):
    """Energy outside a central disc of radius cutoff_frac * (N/2),
       divided by total spectral energy. One scalar per image."""
    N, H, W = x_np.shape
    fy = np.fft.fftshift(np.fft.fftfreq(H))
    fx = np.fft.fftshift(np.fft.fftfreq(W))
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)
    mask_high = (R > cutoff_frac * 0.5).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)
    for i in range(N):
        F2 = np.fft.fftshift(np.fft.fft2(x_np[i]))
        power = np.abs(F2) ** 2
        total = power.sum()
        if total > 0:
            out[i] = (power * mask_high).sum() / total
    return out


def wavelet_energies(x_np, wavelet=WAVELET, levels=N_LEVELS):
    """Per sample: sum-of-absolute-coefficients for each detail subband
       at each level. Order of features: (cH_L1, cV_L1, cD_L1,
       cH_L2, cV_L2, cD_L2, cH_L3, cV_L3, cD_L3) — 3*levels features.

       pywt.wavedec2 returns [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
       where cH=LH (horizontal detail / vertical edges-ish),
             cV=HL (vertical detail), cD=HH (diagonal).
    """
    N = x_np.shape[0]
    feats = np.zeros((N, 3 * levels), dtype=np.float32)
    names = []
    for lev in range(1, levels + 1):
        for sb in ("LH", "HL", "HH"):
            names.append(f"wav_{sb}_L{lev}")
    for i in range(N):
        coeffs = pywt.wavedec2(x_np[i], wavelet=wavelet, level=levels)
        # coeffs[0] is approx; coeffs[1..levels] are detail tuples from coarse->fine.
        # We want output ordered fine->coarse i.e. L1, L2, L3.
        # Index lev in the loop goes 1..levels meaning "L_lev"; the corresponding
        # detail tuple in the list is coeffs[levels - lev + 1].
        for lev in range(1, levels + 1):
            cH, cV, cD = coeffs[levels - lev + 1]
            base = (lev - 1) * 3
            feats[i, base + 0] = np.abs(cH).sum()
            feats[i, base + 1] = np.abs(cV).sum()
            feats[i, base + 2] = np.abs(cD).sum()
    return feats, names


# ---- evaluation helpers ----
def uni_auroc(feature, y):
    a = roc_auc_score(y, feature)
    return max(a, 1 - a)


def multi_auroc(X, y):
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=2000).fit(Xs, y)
    return roc_auc_score(y, lr.predict_proba(Xs)[:, 1])


def main():
    print(f"device = {DEVICE}")
    print(f"wavelet = {WAVELET}  levels = {N_LEVELS}")

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    print("training victim model (seed=0)...")
    victim = train_model(0, train_set)
    print("training surrogate model (seed=1)...")
    surrogate = train_model(1, train_set)

    # materialise test set on device
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)
    print(f"test set: {N} samples")

    # restrict to victim-correct samples
    with torch.no_grad():
        preds = []
        for i in range(0, N, 512):
            preds.append(victim(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct_mask = (preds == test_y)
    x = test_x[correct_mask]
    y = test_y[correct_mask]
    n = x.size(0)
    print(f"victim-correct samples: {n}")

    # ---- compute adversarial targets ----
    print("computing FGSM (self)...")
    t0 = time.time()
    fgsm = []
    for i in range(0, n, 512):
        fgsm.append(attack_fgsm(victim, x[i:i+512], y[i:i+512]))
    fgsm_flip = torch.cat(fgsm).cpu().numpy().astype(int)
    print(f"  done ({time.time()-t0:.1f}s)  pos rate = {fgsm_flip.mean():.3f}")

    print("computing PGD (self)...")
    t0 = time.time()
    pgd = []
    for i in range(0, n, 512):
        pgd.append(attack_pgd(victim, x[i:i+512], y[i:i+512]))
    pgd_flip = torch.cat(pgd).cpu().numpy().astype(int)
    print(f"  done ({time.time()-t0:.1f}s)  pos rate = {pgd_flip.mean():.3f}")

    print("computing FGSM (transfer from surrogate)...")
    t0 = time.time()
    tr = []
    for i in range(0, n, 512):
        tr.append(attack_fgsm_transfer(victim, surrogate, x[i:i+512], y[i:i+512]))
    transfer_flip = torch.cat(tr).cpu().numpy().astype(int)
    print(f"  done ({time.time()-t0:.1f}s)  pos rate = {transfer_flip.mean():.3f}")

    # ---- compute features ----
    print("computing victim margin ...")
    margin = victim_margin(victim, x).cpu().numpy()

    x_np = x.squeeze(1).cpu().numpy().astype(np.float32)  # (n, 28, 28)
    print("computing image-stat baselines ...")
    mean_pix = x_np.reshape(n, -1).mean(1)
    std_pix = x_np.reshape(n, -1).std(1)
    sob = sobel_mean(x_np)

    print("computing Fourier high-freq ratio ...")
    t0 = time.time()
    fft_hf = fourier_high_freq_ratio(x_np, cutoff_frac=0.25)
    print(f"  done ({time.time()-t0:.1f}s)")

    print(f"computing wavelet energies ({WAVELET}, {N_LEVELS} levels) ...")
    t0 = time.time()
    wav_feats, wav_names = wavelet_energies(x_np, WAVELET, N_LEVELS)
    print(f"  done ({time.time()-t0:.1f}s)  shape={wav_feats.shape}")

    baseline_names = ["victim_margin", "mean_pix", "std_pix", "sobel_mean", "fft_hf_ratio"]
    baseline_feats = np.stack([margin, mean_pix, std_pix, sob, fft_hf], axis=1)
    all_feats = np.concatenate([baseline_feats, wav_feats], axis=1)
    all_names = baseline_names + wav_names

    targets = {
        "flipped_FGSM": fgsm_flip,
        "flipped_PGD": pgd_flip,
        "FGSM_transfer": transfer_flip,
    }

    # ---- univariate AUROC ----
    print("\n========== Univariate AUROC ==========")
    header = f"{'feature':<18}" + "".join(f"{k:>16}" for k in targets)
    print(header)
    for j, fname in enumerate(all_names):
        row = f"{fname:<18}"
        for tname, tvec in targets.items():
            if tvec.std() == 0:
                row += f"{'n/a':>16}"
            else:
                a = uni_auroc(all_feats[:, j], tvec)
                row += f"{a:>16.4f}"
        print(row)

    # ---- multivariate comparisons ----
    print("\n========== Multivariate AUROC ==========")
    margin_idx = all_names.index("victim_margin")
    fft_idx = all_names.index("fft_hf_ratio")
    wav_idx = [all_names.index(n) for n in wav_names]

    combos = {
        "margin_only":             [margin_idx],
        "margin+fft_hf":           [margin_idx, fft_idx],
        "margin+wavelets":         [margin_idx] + wav_idx,
        "margin+fft+wavelets":     [margin_idx, fft_idx] + wav_idx,
        "wavelets_only":           wav_idx,
        "fft_hf_only":             [fft_idx],
        "all_features":            list(range(len(all_names))),
    }
    print(f"{'feature set':<24}" + "".join(f"{k:>16}" for k in targets))
    rows_summary = []
    for cname, idx_list in combos.items():
        Xc = all_feats[:, idx_list]
        row = f"{cname:<24}"
        for tname, tvec in targets.items():
            if tvec.std() == 0:
                row += f"{'n/a':>16}"
                continue
            if Xc.shape[1] == 1:
                a = uni_auroc(Xc[:, 0], tvec)
            else:
                a = multi_auroc(Xc, tvec)
            row += f"{a:>16.4f}"
            rows_summary.append((cname, tname, a))
        print(row)

    # ---- head-to-head: does adding wavelets help over margin (+fft)? ----
    print("\n========== H06 verdict per target ==========")
    print("Delta AUROC (margin+wavelets) - (margin)  AND  (margin+wavelets) - (margin+fft_hf)")
    print(f"{'target':<18}{'m':>10}{'m+fft':>10}{'m+wav':>10}{'d_vs_m':>10}{'d_vs_m+fft':>14}")
    for tname, tvec in targets.items():
        if tvec.std() == 0:
            continue
        a_m = uni_auroc(all_feats[:, margin_idx], tvec)
        a_mf = multi_auroc(all_feats[:, [margin_idx, fft_idx]], tvec)
        a_mw = multi_auroc(all_feats[:, [margin_idx] + wav_idx], tvec)
        print(f"{tname:<18}{a_m:>10.4f}{a_mf:>10.4f}{a_mw:>10.4f}"
              f"{a_mw - a_m:>+10.4f}{a_mw - a_mf:>+14.4f}")

    print("\nH06 supported if (margin+wavelets) > (margin+fft_hf) on most targets,")
    print("and the gap is non-trivial (e.g. > +0.01 AUROC).")
    print("\ndone.")


if __name__ == "__main__":
    main()
