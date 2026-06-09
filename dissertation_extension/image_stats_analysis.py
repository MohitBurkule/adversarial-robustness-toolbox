"""
What is `frac_correct_untrained` actually measuring?

The untrained-ensemble experiment showed that the fraction of K random-init CNNs
that "happen" to predict the correct class for a sample is correlated with the
sample's adversarial robustness (AUROC ~0.73 for FGSM/transfer).

Hypothesis: this signal is dominated by simple image statistics — random conv
filters respond to edges/contrast/brightness, and visually-distinctive images
(strong edges, high contrast) produce more class-aligned random activations,
which is also what makes them adversarially robust.

If true, we should be able to replace `frac_correct_untrained` with cheap
hand-computed image stats and get the same predictive power. If not, the
untrained ensemble captures something image-stats cannot.

Pipeline:
  1. Train victim + surrogate.
  2. Build K=32 untrained CNNs, compute frac_correct_untrained per sample.
  3. Compute per-sample image stats: mean, std, edge density (Sobel), entropy,
     input-gradient-norm of victim loss (a known vulnerability predictor).
  4. Correlation matrix between (frac_correct_untrained, image stats, victim_margin,
     min_eps_to_flip).
  5. Multivariate ablation:
        - margin
        - margin + image_stats
        - margin + frac_correct_untrained
        - margin + image_stats + frac_correct_untrained
     If frac_correct_untrained adds nothing on top of image_stats, the
     "architecture geometry" story collapses into a plain image-statistics story.
"""
import time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda")
EPS_TEST = 15.0 / 255.0
K_UNTRAINED = 32
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


def init_untrained(seed):
    torch.manual_seed(seed); return CNN().to(DEVICE).eval()


def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def input_grad_norm(model, x, y):
    """L2 norm of grad of loss w.r.t. input — classic vulnerability proxy."""
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.flatten(1).norm(dim=1).detach()


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


def image_stats(x):
    """Per-sample image statistics. x: [N,1,H,W] on device."""
    N = x.size(0)
    pix = x.view(N, -1)
    mean_pix = pix.mean(1)
    std_pix = pix.std(1)
    # Sobel edge density
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    edge_mag = (gx ** 2 + gy ** 2).sqrt()
    edge_density = edge_mag.view(N, -1).mean(1)
    # histogram entropy (10 bins)
    bins = 10
    bin_edges = torch.linspace(0, 1, bins + 1, device=x.device)
    h = torch.zeros(N, bins, device=x.device)
    for b in range(bins):
        m = (pix >= bin_edges[b]) & (pix < bin_edges[b+1] if b < bins-1 else pix <= bin_edges[b+1])
        h[:, b] = m.float().sum(1)
    p = h / h.sum(1, keepdim=True).clamp_min(1)
    entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(1)
    # "ink area" — fraction of non-background pixels
    ink = (pix > 0.1).float().mean(1)
    return torch.stack([mean_pix, std_pix, edge_density, entropy, ink], 1)


def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    x_test = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N, n_classes = x_test.size(0), 10

    print("Training victim + surrogate...")
    t0 = time.time()
    victim = train_one(0, train_set, EPOCHS_VICTIM)
    surrogate = train_one(1, train_set, EPOCHS_VICTIM)
    print(f"  done ({time.time()-t0:.1f}s)")

    print(f"Building {K_UNTRAINED} untrained CNNs...")
    frac_correct_untrained = torch.zeros(N, device=DEVICE)
    for k in range(K_UNTRAINED):
        m = init_untrained(1000 + k)
        with torch.no_grad():
            for i in range(0, N, 512):
                pred = m(x_test[i:i+512]).argmax(1)
                frac_correct_untrained[i:i+512] += (pred == y_test[i:i+512]).float()
        del m
    frac_correct_untrained /= K_UNTRAINED
    torch.cuda.empty_cache()

    # restrict to victim-correct samples
    with torch.no_grad():
        v_logits = []
        for i in range(0, N, 512):
            v_logits.append(victim(x_test[i:i+512]))
        v_logits = torch.cat(v_logits)
    keep = v_logits.argmax(1) == y_test
    print(f"Victim acc {keep.float().mean().item():.4f}, n_kept={keep.sum().item()}")
    idx = torch.where(keep)[0]
    x_c, y_c = x_test[idx], y_test[idx]
    fcu_c = frac_correct_untrained[idx]
    sorted_l, _ = v_logits[idx].sort(1, descending=True)
    margin_c = sorted_l[:, 0] - sorted_l[:, 1]
    Nc = x_c.size(0)

    # image stats
    print("Computing image statistics...")
    stats_c = image_stats(x_c)
    stat_names = ["mean_pix", "std_pix", "edge_density", "entropy_pix", "ink_area"]

    # input gradient norm on victim (uses backprop, not strictly an image stat but model-free-ish)
    print("Computing victim input-gradient-norm...")
    ign_chunks = []
    for i in range(0, Nc, 512):
        ign_chunks.append(input_grad_norm(victim, x_c[i:i+512], y_c[i:i+512]))
    ign = torch.cat(ign_chunks)

    # adversarial targets
    print("Computing attack targets...")
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

    # build feature matrix:  margin, stats..., ign, fcu
    feats = torch.cat([margin_c.unsqueeze(1), stats_c, ign.unsqueeze(1),
                       fcu_c.unsqueeze(1)], 1)
    feat_names = ["victim_margin"] + stat_names + ["input_grad_norm", "frac_correct_untrained"]
    feats_np = feats.cpu().numpy()
    me_np = min_eps_arr.cpu().numpy()
    Xs = StandardScaler().fit_transform(feats_np)

    print("\n=== Per-feature univariate analysis ===")
    for i, n in enumerate(feat_names):
        cor = np.corrcoef(feats_np[:, i], me_np)[0, 1]
        a_f = roc_auc_score(flipped_FGSM.cpu().numpy(), feats_np[:, i])
        a_f = max(a_f, 1 - a_f)
        a_p = roc_auc_score(flipped_PGD.cpu().numpy(), feats_np[:, i])
        a_p = max(a_p, 1 - a_p)
        a_t = roc_auc_score(flipped_transfer.cpu().numpy(), feats_np[:, i])
        a_t = max(a_t, 1 - a_t)
        print(f"  {n:<25} corr(min_eps)={cor:+.3f}  FGSM={a_f:.4f}  "
              f"PGD={a_p:.4f}  Transfer={a_t:.4f}")

    # correlation between frac_correct_untrained and image stats
    print("\n=== Correlation of frac_correct_untrained with other features ===")
    fcu_arr = feats_np[:, feat_names.index("frac_correct_untrained")]
    for i, n in enumerate(feat_names):
        if n == "frac_correct_untrained": continue
        cor = np.corrcoef(fcu_arr, feats_np[:, i])[0, 1]
        print(f"  corr(fcu, {n:<25}) = {cor:+.3f}")

    # multivariate ablation
    print("\n=== Multivariate ablation ===")
    margin_idx = [feat_names.index("victim_margin")]
    stat_idx = [feat_names.index(n) for n in stat_names]
    fcu_idx = [feat_names.index("frac_correct_untrained")]
    ign_idx = [feat_names.index("input_grad_norm")]

    sets = {
        "margin": margin_idx,
        "margin+stats": margin_idx + stat_idx,
        "margin+fcu": margin_idx + fcu_idx,
        "margin+stats+fcu": margin_idx + stat_idx + fcu_idx,
        "margin+stats+ign": margin_idx + stat_idx + ign_idx,
        "margin+ign+fcu": margin_idx + ign_idx + fcu_idx,
        "all": list(range(len(feat_names))),
    }
    for tname, tgt in [("FGSM_self", flipped_FGSM),
                       ("PGD_self", flipped_PGD),
                       ("FGSM_transfer", flipped_transfer)]:
        y_arr = tgt.cpu().numpy().astype(int)
        if y_arr.std() == 0: continue
        print(f"\n  --- {tname}  (pos rate {y_arr.mean():.3f}) ---")
        for nm, ix in sets.items():
            lr = LogisticRegression(max_iter=2000).fit(Xs[:, ix], y_arr)
            auc = roc_auc_score(y_arr, lr.predict_proba(Xs[:, ix])[:, 1])
            print(f"    {nm:<22}: {auc:.4f}")

    # continuous
    print("\n=== OLS on min_eps ===")
    for nm, ix in sets.items():
        ols = LinearRegression().fit(Xs[:, ix], me_np)
        r2 = ols.score(Xs[:, ix], me_np)
        print(f"  {nm:<22}: R^2 = {r2:.4f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.1f}s")
