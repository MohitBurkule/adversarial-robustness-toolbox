"""
H15: After FGSM adversarial training (defended model), the predictive value of
margin and image statistics changes. Specifically, AT flattens the loss surface,
making margins more uniform across samples. Hypothesis: when margin is squashed
by AT, image statistics (texture, frequency content, complexity) become
relatively more important predictors of which samples remain vulnerable.

Pipeline:
  1. Train TWO small CNNs on Fashion-MNIST (10 epochs, Adam, lr=1e-3):
       - vanilla:  standard cross-entropy training
       - adv:      FGSM adversarial training, 50% adv ratio per batch, eps=15/255
  2. For EACH victim, compute six per-sample features on the test set:
       victim_margin, mean_pix, std_pix, sobel_mean, jpeg_byte_size,
       fourier_high_freq_ratio.
  3. For EACH victim, compute three binary vulnerability targets:
       flipped_FGSM (eps=15/255), flipped_PGD (eps=15/255, 10 steps),
       FGSM_min_eps_binary_search (binarised at median).
       Also keep the continuous min_eps for OLS.
  4. Per-feature univariate AUROC for each victim, printed side by side.
  5. Multivariate logistic regression: margin alone vs margin + image_stats,
     for each victim. Compare delta-AUROC across vanilla vs AT — if image
     stats matter MORE under AT (larger delta), H15 is supported.

Self-contained. Requires: torch (cuda), torchvision, sklearn, numpy, Pillow.
Data cached at /tmp/data.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPS = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS / 4.0
EPOCHS = 10
BATCH = 128
SEED = 0


# -------------------- model --------------------
class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2); x = self.do1(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.do2(x); return self.fc2(x)


# -------------------- attacks --------------------
def fgsm_grad_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS):
    sign = fgsm_grad_sign(model, x, y)
    return (x + eps * sign).clamp(0, 1).detach()


def pgd_attack(model, x, y, eps=EPS, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    # random start within ball
    adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        F.cross_entropy(model(adv), y).backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
    return adv.detach()


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad_sign(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# -------------------- training --------------------
def train_vanilla(train_set, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  [vanilla] epoch {ep+1}/{EPOCHS} done")
    return model


def train_fgsm_adv(train_set, seed=SEED, adv_ratio=0.5, eps=EPS):
    """FGSM adversarial training: for each batch, replace `adv_ratio` of the
    samples with their FGSM perturbations crafted on the current model."""
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            n_adv = int(adv_ratio * x.size(0))
            if n_adv > 0:
                # craft FGSM on a sub-batch using current model (eval mode for stable grads)
                model.eval()
                x_sub = x[:n_adv]
                y_sub = y[:n_adv]
                x_sub_adv = fgsm_attack(model, x_sub, y_sub, eps=eps)
                model.train()
                x_mixed = torch.cat([x_sub_adv, x[n_adv:]], 0)
                y_mixed = torch.cat([y_sub, y[n_adv:]], 0)
            else:
                x_mixed, y_mixed = x, y
            opt.zero_grad()
            F.cross_entropy(model(x_mixed), y_mixed).backward()
            opt.step()
        print(f"  [adv]     epoch {ep+1}/{EPOCHS} done")
    return model


# -------------------- features --------------------
def compute_margin(model, x):
    """victim_margin = top1_logit - top2_logit."""
    N = x.size(0)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, N, 512):
            logits = model(x[i:i+512])
            sl, _ = logits.sort(1, descending=True)
            out.append(sl[:, 0] - sl[:, 1])
    return torch.cat(out)


def image_stats(x_np):
    """x_np: (N,1,28,28) numpy float in [0,1].
    Returns dict of (N,) numpy arrays for the five image stats."""
    N = x_np.shape[0]
    flat = x_np.reshape(N, -1)
    mean_pix = flat.mean(1)
    std_pix = flat.std(1)

    # Sobel mean magnitude (3x3 Sobel via simple conv)
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sy = sx.T
    imgs = x_np[:, 0]  # (N,28,28)
    # zero-pad reflect-ish
    p = np.pad(imgs, ((0, 0), (1, 1), (1, 1)), mode="edge")
    # vectorised conv via stride tricks
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(p, (3, 3), axis=(1, 2))  # (N,28,28,3,3)
    gx = (win * sx).sum(axis=(-1, -2))
    gy = (win * sy).sum(axis=(-1, -2))
    sobel_mean = np.sqrt(gx * gx + gy * gy).reshape(N, -1).mean(1)

    # JPEG byte size — proxy for compressibility/complexity
    jpeg = np.zeros(N, dtype=np.float32)
    for i in range(N):
        img = (imgs[i] * 255.0).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img, mode="L").save(buf, format="JPEG", quality=75)
        jpeg[i] = buf.tell()

    # 2D FFT high-frequency ratio: energy outside central radius / total energy
    F2 = np.fft.fftshift(np.fft.fft2(imgs), axes=(1, 2))
    P = np.abs(F2) ** 2
    H, W = 28, 28
    cy, cx = H / 2, W / 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high_mask = r > (min(H, W) / 4.0)  # outside radius 7 ≈ "high freq"
    total = P.reshape(N, -1).sum(1) + 1e-12
    high = (P * high_mask).reshape(N, -1).sum(1)
    fourier_hf = high / total

    return {
        "mean_pix": mean_pix.astype(np.float32),
        "std_pix": std_pix.astype(np.float32),
        "sobel_mean": sobel_mean.astype(np.float32),
        "jpeg_byte_size": jpeg.astype(np.float32),
        "fourier_high_freq_ratio": fourier_hf.astype(np.float32),
    }


# -------------------- evaluation --------------------
def batched_flip(model, x, y, attack_fn):
    out = []
    for i in range(0, x.size(0), 512):
        adv = attack_fn(model, x[i:i+512], y[i:i+512])
        with torch.no_grad():
            out.append(model(adv).argmax(1) != y[i:i+512])
    return torch.cat(out)


def safe_auc(y, s):
    if y.std() == 0:
        return float("nan")
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def evaluate_victim(name, model, x, y, img_feats):
    print(f"\n========== victim: {name} ==========")
    # 1) base correctness — restrict to samples model gets right
    model.eval()
    with torch.no_grad():
        pred = []
        for i in range(0, x.size(0), 512):
            pred.append(model(x[i:i+512]).argmax(1))
        pred = torch.cat(pred)
    correct = pred == y
    n_correct = int(correct.sum().item())
    print(f"  clean acc = {n_correct / x.size(0):.4f}  (using {n_correct} correct samples)")

    x_c = x[correct]; y_c = y[correct]
    correct_np = correct.cpu().numpy()
    img_feats_c = {k: v[correct_np] for k, v in img_feats.items()}

    # 2) features
    margin = compute_margin(model, x_c).cpu().numpy()
    feats = {"victim_margin": margin, **img_feats_c}
    feat_names = ["victim_margin", "mean_pix", "std_pix", "sobel_mean",
                  "jpeg_byte_size", "fourier_high_freq_ratio"]

    # 3) targets
    print("  computing FGSM flip ...")
    fgsm_flip = batched_flip(model, x_c, y_c, lambda m, a, b: fgsm_attack(m, a, b, EPS))
    print(f"    FGSM flip rate = {fgsm_flip.float().mean().item():.4f}")
    print("  computing PGD flip ...")
    pgd_flip = batched_flip(model, x_c, y_c, lambda m, a, b: pgd_attack(m, a, b, EPS))
    print(f"    PGD  flip rate = {pgd_flip.float().mean().item():.4f}")
    print("  computing FGSM min_eps binary search ...")
    me = []
    for i in range(0, x_c.size(0), 512):
        me.append(min_eps_fgsm(model, x_c[i:i+512], y_c[i:i+512]))
    min_eps = torch.cat(me).cpu().numpy()
    # binarise at median: low min_eps  ==>  vulnerable
    me_median = float(np.median(min_eps))
    min_eps_bin = (min_eps < me_median).astype(int)
    print(f"    min_eps median = {me_median:.4f}")

    targets = {
        "flipped_FGSM": fgsm_flip.cpu().numpy().astype(int),
        "flipped_PGD":  pgd_flip.cpu().numpy().astype(int),
        "min_eps_vulnerable_bin": min_eps_bin,
    }

    # 4) univariate AUROC
    print("\n  --- univariate AUROC ---")
    header = "    feature".ljust(32) + "".join(f"{t:>22}" for t in targets)
    print(header)
    uni_table = {}
    for fn in feat_names:
        row = f"    {fn:<28}"
        uni_table[fn] = {}
        for t_name, y_arr in targets.items():
            a = safe_auc(y_arr, feats[fn])
            uni_table[fn][t_name] = a
            row += f"{a:>22.4f}"
        print(row)

    # 5) multivariate: margin alone vs margin + image_stats
    print("\n  --- multivariate AUROC (margin alone vs margin+image_stats) ---")
    margin_only = feats["victim_margin"].reshape(-1, 1)
    image_feat_names = [fn for fn in feat_names if fn != "victim_margin"]
    margin_plus = np.column_stack([feats[fn] for fn in feat_names])

    multi_rows = []
    for t_name, y_arr in targets.items():
        if y_arr.std() == 0:
            print(f"   {t_name}: degenerate")
            continue
        Xm = StandardScaler().fit_transform(margin_only)
        Xa = StandardScaler().fit_transform(margin_plus)
        lr_m = LogisticRegression(max_iter=2000).fit(Xm, y_arr)
        lr_a = LogisticRegression(max_iter=2000).fit(Xa, y_arr)
        auc_m = roc_auc_score(y_arr, lr_m.predict_proba(Xm)[:, 1])
        auc_a = roc_auc_score(y_arr, lr_a.predict_proba(Xa)[:, 1])
        # also: image stats alone (no margin)
        Xi = StandardScaler().fit_transform(np.column_stack([feats[fn] for fn in image_feat_names]))
        lr_i = LogisticRegression(max_iter=2000).fit(Xi, y_arr)
        auc_i = roc_auc_score(y_arr, lr_i.predict_proba(Xi)[:, 1])
        delta = auc_a - auc_m
        print(f"   {t_name:<25}  margin_only={auc_m:.4f}  "
              f"img_only={auc_i:.4f}  margin+img={auc_a:.4f}  "
              f"delta={delta:+.4f}")
        print("     standardised coefs (margin+img):")
        for fn, c in zip(feat_names, lr_a.coef_.flatten()):
            print(f"       {fn:<28} {c:+.4f}")
        multi_rows.append((t_name, auc_m, auc_i, auc_a, delta))

    return {
        "victim": name,
        "univariate": uni_table,
        "multivariate": multi_rows,
        "margin_stats": (float(margin.mean()), float(margin.std())),
    }


# -------------------- main --------------------
def main():
    print(f"device = {DEVICE}")
    tf = transforms.ToTensor()
    print("loading Fashion-MNIST ...")
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    # materialise full test set on device
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    print(f"  test set: x={tuple(test_x.shape)}, y={tuple(test_y.shape)}")

    # train both victims
    print("\n>>> training vanilla victim ...")
    t0 = time.time()
    model_vanilla = train_vanilla(train_set, seed=SEED)
    print(f"  done in {time.time()-t0:.1f}s")
    print("\n>>> training FGSM-adv-trained victim ...")
    t0 = time.time()
    model_adv = train_fgsm_adv(train_set, seed=SEED, adv_ratio=0.5, eps=EPS)
    print(f"  done in {time.time()-t0:.1f}s")

    # image stats — computed once on the test set (same images for both victims)
    print("\ncomputing image statistics (mean_pix, std_pix, sobel, jpeg, fourier) ...")
    t0 = time.time()
    test_x_np = test_x.cpu().numpy()
    img_feats = image_stats(test_x_np)
    print(f"  done in {time.time()-t0:.1f}s")

    results = []
    results.append(evaluate_victim("vanilla", model_vanilla, test_x, test_y, img_feats))
    results.append(evaluate_victim("fgsm_adv_trained", model_adv, test_x, test_y, img_feats))

    # -------------------- summary --------------------
    print("\n\n===================== H15 SUMMARY =====================")
    print("Margin distribution by victim (mean, std):")
    for r in results:
        m, s = r["margin_stats"]
        print(f"  {r['victim']:<22}  mean={m:.4f}  std={s:.4f}")
    print("\nUnivariate AUROC, side by side (each target):")
    feat_names = ["victim_margin", "mean_pix", "std_pix", "sobel_mean",
                  "jpeg_byte_size", "fourier_high_freq_ratio"]
    target_names = ["flipped_FGSM", "flipped_PGD", "min_eps_vulnerable_bin"]
    for t in target_names:
        print(f"\n  target = {t}")
        print(f"    {'feature':<30}{'vanilla':>12}{'adv':>12}{'delta(adv-van)':>18}")
        for fn in feat_names:
            v = results[0]["univariate"][fn][t]
            a = results[1]["univariate"][fn][t]
            print(f"    {fn:<30}{v:>12.4f}{a:>12.4f}{a-v:>+18.4f}")

    print("\nMultivariate AUROC delta (margin+img vs margin_only) -- "
          "larger delta under AT supports H15:")
    print(f"  {'target':<28}{'van_margin':>12}{'van_full':>12}{'van_d':>10}"
          f"{'adv_margin':>12}{'adv_full':>12}{'adv_d':>10}")
    # rebuild from multi_rows
    van = {r[0]: r for r in results[0]["multivariate"]}
    adv = {r[0]: r for r in results[1]["multivariate"]}
    for t in target_names:
        if t not in van or t not in adv:
            continue
        _, vm, _vi, va, vd = van[t]
        _, am, _ai, aa, ad = adv[t]
        print(f"  {t:<28}{vm:>12.4f}{va:>12.4f}{vd:>+10.4f}"
              f"{am:>12.4f}{aa:>12.4f}{ad:>+10.4f}")

    print("\nInterpretation key:")
    print("  - If adv-trained victim has SMALLER margin std (flatter), AND")
    print("    image-stat AUROCs rise while margin AUROC falls, AND")
    print("    delta(margin+img - margin_only) is LARGER under AT than vanilla,")
    print("    then H15 is supported: AT shifts predictive weight onto image stats.")
    print("  - If margin remains the dominant predictor with similar delta, H15")
    print("    is not supported -- AT does not redistribute predictive value.")


if __name__ == "__main__":
    main()
