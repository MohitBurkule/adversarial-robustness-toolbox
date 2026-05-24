"""
H80: Cutmix training (Yun et al. 2019) changes vulnerability predictor ranking.

Replace a random rectangular patch from one training image with a patch from
another image (label mixed proportionally to area). Train two Fashion-MNIST
victims (small CNN matching diagnostic_test.py), 10 epochs each:
  - vanilla   (no augmentation)
  - cutmix    (alpha = 1.0)

For each victim, on the test set compute per-sample features:
  - margin      (final-model logit margin, top1 - top2)
  - mean_pix    (mean pixel intensity of the input)
  - std_pix     (std pixel intensity of the input)
  - sobel_mean  (mean magnitude of the Sobel filter response)

Adversarial vulnerability targets (per victim, restricted to correctly-classified):
  - FGSM    (binary: flipped at eps=15/255)
  - PGD     (binary: flipped by 10-step PGD, eps=15/255, alpha=eps/4)
  - min_eps (continuous: smallest L_inf eps that flips FGSM, binary searched)

Report per-feature univariate AUROC (binary targets) and |Spearman corr| for
min_eps, side by side for vanilla vs cutmix, to test whether training-data
augmentation reranks the predictors.

This is a code-only artefact: NOT executed here.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0
CUTMIX_ALPHA = 1.0


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


# ---------------- cutmix ----------------
def rand_bbox(size, lam):
    """Yun et al. cutmix bbox: cut area proportional to (1 - lam)."""
    _, _, H, W = size
    cut_rat = float(np.sqrt(1.0 - lam))
    cut_h = int(H * cut_rat); cut_w = int(W * cut_rat)
    cy = np.random.randint(H); cx = np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H); y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W); x2 = np.clip(cx + cut_w // 2, 0, W)
    return y1, y2, x1, x2


def cutmix_batch(x, y, alpha=CUTMIX_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    y1, y2, x1, x2 = rand_bbox(x.size(), lam)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    # adjust lam to true cut-area ratio (after clipping)
    lam_adj = 1.0 - ((y2 - y1) * (x2 - x1) / float(x.size(-1) * x.size(-2)))
    return x_mix, y, y[perm], lam_adj


# ---------------- training ----------------
def train_victim(seed, train_set, mode, n_classes=10):
    """mode in {'vanilla', 'cutmix'}."""
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if mode == "cutmix":
                x_mix, ya, yb, lam = cutmix_batch(x, y, CUTMIX_ALPHA)
                logits = model(x_mix)
                loss = lam * F.cross_entropy(logits, ya) + (1.0 - lam) * F.cross_entropy(logits, yb)
            else:
                loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step()
    model.eval()
    return model


# ---------------- features ----------------
def sobel_mean(x):
    """Mean magnitude of Sobel response per image. x: (N,1,H,W) on DEVICE."""
    kx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]],
                      device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag.mean(dim=(1, 2, 3))


def compute_features(model, x):
    """Return (N, 4) tensor: margin, mean_pix, std_pix, sobel_mean."""
    N = x.size(0)
    margins = []
    with torch.no_grad():
        for i in range(0, N, 512):
            logits = model(x[i:i+512])
            sl, _ = logits.sort(1, descending=True)
            margins.append(sl[:, 0] - sl[:, 1])
    margin = torch.cat(margins)
    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))
    sob = sobel_mean(x)
    return torch.stack([margin, mean_pix, std_pix, sob], dim=1)


# ---------------- attacks ----------------
def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    adv = adv.clamp(0, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        g = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * g.sign()
        adv = torch.max(torch.min(adv, x + eps), x - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def batched(fn, x, y, bs=512, **kw):
    outs = []
    for i in range(0, x.size(0), bs):
        outs.append(fn(x[i:i+bs], y[i:i+bs], **kw))
    return torch.cat(outs)


# ---------------- evaluation ----------------
def evaluate_victim(model, test_x, test_y, label):
    print(f"\n===== victim: {label} =====")
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        pred = torch.cat(preds)
    clean_acc = (pred == test_y).float().mean().item()
    print(f"  clean accuracy: {clean_acc:.4f}")

    correct = pred == test_y
    x_c, y_c = test_x[correct], test_y[correct]
    print(f"  using {int(correct.sum().item())} correctly-classified samples")

    feats = compute_features(model, x_c)

    print("  computing FGSM ...")
    fg = batched(lambda x, y: fgsm_flip(model, x, y), x_c, y_c)
    print(f"    FGSM flip rate: {fg.float().mean().item():.4f}")
    print("  computing PGD ...")
    pg = batched(lambda x, y: pgd_flip(model, x, y), x_c, y_c)
    print(f"    PGD  flip rate: {pg.float().mean().item():.4f}")
    print("  computing min_eps ...")
    me = batched(lambda x, y: min_eps_to_flip(model, x, y), x_c, y_c)
    print(f"    mean min_eps:   {me.mean().item():.4f}")

    return feats.cpu().numpy(), fg.cpu().numpy().astype(int), pg.cpu().numpy().astype(int), me.cpu().numpy()


def report(feats, fg, pg, me, feat_names, label):
    print(f"\n--- per-feature scores  (victim = {label}) ---")
    print(f"{'feature':<12} {'AUROC_FGSM':>11} {'AUROC_PGD':>11} {'|rho|_min_eps':>15}")
    table = {}
    for i, n in enumerate(feat_names):
        col = feats[:, i]
        a_fg = roc_auc_score(fg, col) if fg.std() > 0 else float("nan")
        a_fg = max(a_fg, 1 - a_fg) if not np.isnan(a_fg) else a_fg
        a_pg = roc_auc_score(pg, col) if pg.std() > 0 else float("nan")
        a_pg = max(a_pg, 1 - a_pg) if not np.isnan(a_pg) else a_pg
        rho, _ = spearmanr(col, me)
        rho_a = abs(rho)
        table[n] = (a_fg, a_pg, rho_a)
        print(f"{n:<12} {a_fg:>11.4f} {a_pg:>11.4f} {rho_a:>15.4f}")
    return table


def rank(d, idx):
    """Rank features (1 = best) by score at index idx (higher = better)."""
    items = sorted(d.items(), key=lambda kv: -kv[1][idx])
    return {k: r + 1 for r, (k, _) in enumerate(items)}


# ---------------- main ----------------
def main():
    print(f"device: {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    feat_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    results = {}
    for mode in ["vanilla", "cutmix"]:
        print(f"\n##### training victim: {mode} #####")
        t0 = time.time()
        model = train_victim(seed=0, train_set=train_set, mode=mode)
        print(f"  trained in {time.time()-t0:.1f}s")
        feats, fg, pg, me = evaluate_victim(model, test_x, test_y, mode)
        table = report(feats, fg, pg, me, feat_names, mode)
        results[mode] = table

    # ---- ranking comparison ----
    print("\n===== RANKING COMPARISON (1 = strongest predictor) =====")
    for label, idx in [("AUROC_FGSM", 0), ("AUROC_PGD", 1), ("|rho| min_eps", 2)]:
        r_v = rank(results["vanilla"], idx)
        r_c = rank(results["cutmix"], idx)
        print(f"\n  target = {label}")
        print(f"    {'feature':<12} {'vanilla':>8} {'cutmix':>8}  changed?")
        for n in feat_names:
            ch = "*" if r_v[n] != r_c[n] else ""
            print(f"    {n:<12} {r_v[n]:>8d} {r_c[n]:>8d}   {ch}")

    # ---- side-by-side raw scores ----
    print("\n===== SIDE-BY-SIDE SCORES =====")
    print(f"{'feature':<12} {'target':<14} {'vanilla':>9} {'cutmix':>9} {'delta':>9}")
    score_labels = ["AUROC_FGSM", "AUROC_PGD", "|rho|min_eps"]
    for n in feat_names:
        for i, sl in enumerate(score_labels):
            v = results["vanilla"][n][i]; c = results["cutmix"][n][i]
            print(f"{n:<12} {sl:<14} {v:>9.4f} {c:>9.4f} {c - v:>+9.4f}")


if __name__ == "__main__":
    main()
