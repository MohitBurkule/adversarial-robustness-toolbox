"""
Hypothesis H44: TRADES adversarial training changes which per-sample features
predict adversarial vulnerability.

Compare FOUR victim CNNs trained on Fashion-MNIST:
    1. vanilla        - standard cross-entropy
    2. FGSM-AT        - 50% of each batch replaced with FGSM adv examples (eps=15/255)
    3. PGD-AT         - Madry-style: replace batch entirely with 10-step PGD adv (eps=15/255)
    4. TRADES         - Zhang, Yu, Jiao, Xing, Ghaoui, Jordan, ICML 2019.
                        Loss = CE(f(x), y) + beta * KL( softmax(f(x)) || softmax(f(x_adv)) )
                        where x_adv is found by 10-step PGD maximising the KL term.
                        beta=6.0 (paper's default for CIFAR-10/Fashion-MNIST).

For each trained victim we measure four per-sample features:
    - margin           (top1 - top2 final logits)
    - mean_pix         (average pixel value)
    - std_pix          (std of pixels)
    - sobel_mean       (mean Sobel-edge magnitude; proxy for texture complexity)
    - jpeg_q75         (JPEG file size at quality 75; another texture proxy)

and three vulnerability targets:
    - flipped_FGSM     (FGSM at eps=15/255 flips label)
    - flipped_PGD      (10-step PGD at eps=15/255 flips label)
    - FGSM_min_eps     (binary-searched smallest eps to flip with FGSM)

Outputs:
    1. Per-victim univariate AUROC table for each feature/target combination.
    2. Margin distribution summary across the four victims (mean, std, quantiles).
       Hypothesis: TRADES should produce the most uniform / lowest-variance margin
       distribution (because it explicitly trades natural accuracy for robustness
       and tends to push samples toward the decision boundary more uniformly).
    3. Comparison: does margin's dominance hold after TRADES?  Do image-statistic
       features matter MORE on robust models?

Web-search references (Zhang et al. 2019 TRADES):
    https://github.com/yaodongyu/TRADES   (official PyTorch reference)
    Loss formulation (trades_loss):
        x_adv = x + 0.001 * randn
        for _ in range(K):
            x_adv.requires_grad_(True)
            loss_kl = KL(softmax(f(x_adv)), softmax(f(x)))
            grad = autograd.grad(loss_kl, x_adv)
            x_adv = x_adv.detach() + step_size * sign(grad)
            x_adv = clamp(x_adv, x-eps, x+eps).clamp(0, 1)
        loss_natural = CE(f(x), y)
        loss_robust  = (1/n) * KL(softmax(f(x_adv)), softmax(f(x)))
        loss = loss_natural + beta * loss_robust

Self-contained.  Tools: PyTorch + CUDA, dataset cached in /tmp/data.
DO NOT auto-run — invoke manually.
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
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPS = 15.0 / 255.0
PGD_STEPS = 10
PGD_STEP = 2.0 / 255.0
EPOCHS = 10
BATCH = 128
BETA_TRADES = 6.0


# ----------------------------- model -----------------------------
class CNN(nn.Module):
    """Matches diagnostic_test.py CNN."""
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


# ----------------------------- attacks -----------------------------
def fgsm_attack(model, x, y, eps=EPS):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return (x_adv + eps * grad.sign()).clamp(0, 1).detach()


def pgd_attack(model, x, y, eps=EPS, steps=PGD_STEPS, step_size=PGD_STEP):
    model.eval()
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)
    return x_adv.detach()


def trades_inner_adv(model, x, eps=EPS, steps=PGD_STEPS, step_size=PGD_STEP):
    """Find x_adv that maximises KL( p(x_adv) || p(x) )."""
    model.eval()
    with torch.no_grad():
        p_nat = F.softmax(model(x), dim=1)
    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logp_adv = F.log_softmax(model(x_adv), dim=1)
        loss_kl = F.kl_div(logp_adv, p_nat, reduction="batchmean")
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)
    return x_adv.detach()


# ----------------------------- training -----------------------------
def train_victim(mode, train_loader, seed=0):
    """mode in {'vanilla','fgsm','pgd','trades'}"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if mode == "vanilla":
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
            elif mode == "fgsm":
                # 50% adv, 50% clean
                half = x.size(0) // 2
                x_adv = fgsm_attack(model, x[:half], y[:half], eps=EPS)
                model.train()
                x_in = torch.cat([x_adv, x[half:]], dim=0)
                y_in = torch.cat([y[:half], y[half:]], dim=0)
                opt.zero_grad()
                loss = F.cross_entropy(model(x_in), y_in)
                loss.backward()
                opt.step()
            elif mode == "pgd":
                x_adv = pgd_attack(model, x, y, eps=EPS,
                                   steps=PGD_STEPS, step_size=PGD_STEP)
                model.train()
                opt.zero_grad()
                loss = F.cross_entropy(model(x_adv), y)
                loss.backward()
                opt.step()
            elif mode == "trades":
                x_adv = trades_inner_adv(model, x, eps=EPS,
                                         steps=PGD_STEPS, step_size=PGD_STEP)
                model.train()
                opt.zero_grad()
                logits_nat = model(x)
                logits_adv = model(x_adv)
                loss_natural = F.cross_entropy(logits_nat, y)
                # KL( softmax(logits_adv) || softmax(logits_nat) ) as in TRADES code
                loss_robust = F.kl_div(
                    F.log_softmax(logits_adv, dim=1),
                    F.softmax(logits_nat, dim=1),
                    reduction="batchmean",
                )
                loss = loss_natural + BETA_TRADES * loss_robust
                loss.backward()
                opt.step()
            else:
                raise ValueError(mode)
        print(f"   [{mode}] epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)")
    return model


# ----------------------------- features -----------------------------
SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def image_stats(x):
    """x: (N,1,28,28) in [0,1]. Returns dict of per-sample features."""
    N = x.size(0)
    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))
    sx = SOBEL_X.to(x.device)
    sy = SOBEL_Y.to(x.device)
    gx = F.conv2d(x, sx, padding=1)
    gy = F.conv2d(x, sy, padding=1)
    sobel_mag = torch.sqrt(gx * gx + gy * gy)
    sobel_mean = sobel_mag.mean(dim=(1, 2, 3))
    # JPEG file size at quality 75 (PIL)
    jpeg_q75 = torch.zeros(N)
    x_cpu = (x.detach().cpu().numpy() * 255).astype(np.uint8)
    for i in range(N):
        img = Image.fromarray(x_cpu[i, 0], mode="L")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        jpeg_q75[i] = buf.tell()
    return {
        "mean_pix": mean_pix.cpu(),
        "std_pix": std_pix.cpu(),
        "sobel_mean": sobel_mean.cpu(),
        "jpeg_q75": jpeg_q75,
    }


def compute_margin(model, x, batch=512):
    model.eval()
    margins = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            logits = model(x[i:i+batch])
            sorted_l, _ = logits.sort(1, descending=True)
            margins.append((sorted_l[:, 0] - sorted_l[:, 1]).cpu())
    return torch.cat(margins)


def compute_flips(model, x, y, attack_fn, batch=256):
    model.eval()
    flips = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        x_adv = attack_fn(model, xb, yb)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        flips.append((pred != yb).cpu())
    return torch.cat(flips)


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=15, batch=256):
    """Binary-search smallest FGSM eps that flips each sample."""
    model.eval()
    out = []
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        # gradient sign (single shot — eps only scales magnitude)
        xg = xb.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(xg), yb)
        grad = torch.autograd.grad(loss, xg)[0]
        sign = grad.sign().detach()
        lo = torch.zeros(xb.size(0), device=xb.device)
        hi = torch.full((xb.size(0),), eps_max, device=xb.device)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xb + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
            with torch.no_grad():
                flipped = model(adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        out.append(hi.cpu())
    return torch.cat(out)


# ----------------------------- AUROC helpers -----------------------------
def safe_auroc(score, target):
    target = np.asarray(target).astype(int)
    if target.std() == 0:
        return float("nan")
    a = roc_auc_score(target, np.asarray(score))
    return max(a, 1 - a)


# ----------------------------- driver -----------------------------
def main():
    print(f"device = {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)

    # cache full test set on device
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    print(f"test set: {test_x.shape}")

    # image stats are model-independent — compute once
    print("computing image stats (model-independent)...")
    stats = image_stats(test_x)

    victims = {}
    for mode in ["vanilla", "fgsm", "pgd", "trades"]:
        print(f"\n==== training victim: {mode} ====")
        victims[mode] = train_victim(mode, train_loader, seed=0)

    # per-victim feature + target tables
    results = {}  # mode -> dict
    margins_all = {}
    for mode, model in victims.items():
        print(f"\n==== evaluating victim: {mode} ====")
        margin = compute_margin(model, test_x)
        margins_all[mode] = margin

        # restrict to correctly-classified samples (canonical for vulnerability)
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, test_x.size(0), 512):
                preds.append(model(test_x[i:i+512]).argmax(1))
            preds = torch.cat(preds)
        correct = (preds == test_y)
        n_correct = int(correct.sum().item())
        print(f"  clean acc = {n_correct / test_y.size(0):.4f}  "
              f"(using {n_correct} correctly-classified samples)")

        idx = correct.nonzero(as_tuple=True)[0]
        x_c = test_x[idx]
        y_c = test_y[idx]
        margin_c = margin[idx.cpu()]
        feats = {
            "margin": margin_c.numpy(),
            "mean_pix": stats["mean_pix"][idx.cpu()].numpy(),
            "std_pix": stats["std_pix"][idx.cpu()].numpy(),
            "sobel_mean": stats["sobel_mean"][idx.cpu()].numpy(),
            "jpeg_q75": stats["jpeg_q75"][idx.cpu()].numpy(),
        }

        print("  computing flipped_FGSM ...")
        t0 = time.time()
        flipped_fgsm = compute_flips(model, x_c, y_c,
                                     lambda m, xx, yy: fgsm_attack(m, xx, yy, eps=EPS))
        print(f"    pos rate = {flipped_fgsm.float().mean():.3f} ({time.time()-t0:.1f}s)")

        print("  computing flipped_PGD ...")
        t0 = time.time()
        flipped_pgd = compute_flips(model, x_c, y_c,
                                    lambda m, xx, yy: pgd_attack(m, xx, yy, eps=EPS))
        print(f"    pos rate = {flipped_pgd.float().mean():.3f} ({time.time()-t0:.1f}s)")

        print("  computing FGSM_min_eps ...")
        t0 = time.time()
        min_eps = fgsm_min_eps(model, x_c, y_c)
        print(f"    mean min_eps = {min_eps.mean():.4f} ({time.time()-t0:.1f}s)")

        # For min_eps treat as continuous target: AUROC against binarisation
        # at the median (top-half = harder-to-flip).
        med = float(min_eps.median())
        min_eps_bin = (min_eps > med).numpy().astype(int)

        targets = {
            "flipped_FGSM": flipped_fgsm.numpy().astype(int),
            "flipped_PGD": flipped_pgd.numpy().astype(int),
            "FGSM_min_eps>median": min_eps_bin,
        }
        # store raw min_eps too for Spearman
        results[mode] = {
            "feats": feats,
            "targets": targets,
            "min_eps_raw": min_eps.numpy(),
            "margin_full": margin.numpy(),
            "clean_acc": n_correct / test_y.size(0),
        }

    # -------- per-victim univariate AUROC tables --------
    feat_names = ["margin", "mean_pix", "std_pix", "sobel_mean", "jpeg_q75"]
    target_names = ["flipped_FGSM", "flipped_PGD", "FGSM_min_eps>median"]

    print("\n\n================ PER-VICTIM UNIVARIATE AUROC ================")
    header = f"{'victim':<10} {'target':<22} " + " ".join(f"{n:>11}" for n in feat_names)
    print(header)
    print("-" * len(header))
    for mode in ["vanilla", "fgsm", "pgd", "trades"]:
        r = results[mode]
        for tname in target_names:
            row = [f"{mode:<10}", f"{tname:<22}"]
            y = r["targets"][tname]
            for fn in feat_names:
                a = safe_auroc(r["feats"][fn], y)
                row.append(f"{a:>11.4f}")
            print(" ".join(row))
        print()

    # -------- Spearman-ish correlations with min_eps (continuous) --------
    print("\n================ Pearson corr( feature, min_eps ) per victim ================")
    print(f"{'victim':<10} " + " ".join(f"{n:>11}" for n in feat_names))
    for mode in ["vanilla", "fgsm", "pgd", "trades"]:
        r = results[mode]
        row = [f"{mode:<10}"]
        for fn in feat_names:
            c = np.corrcoef(r["feats"][fn], r["min_eps_raw"])[0, 1]
            row.append(f"{c:>+11.4f}")
        print(" ".join(row))

    # -------- margin distribution comparison --------
    print("\n\n================ MARGIN DISTRIBUTION across victims ================")
    print(f"{'victim':<10} {'mean':>9} {'std':>9} {'q10':>9} {'q25':>9} {'q50':>9} "
          f"{'q75':>9} {'q90':>9} {'CV':>9}")
    for mode in ["vanilla", "fgsm", "pgd", "trades"]:
        m = margins_all[mode].numpy()
        q = np.quantile(m, [0.10, 0.25, 0.50, 0.75, 0.90])
        cv = m.std() / (abs(m.mean()) + 1e-9)
        print(f"{mode:<10} {m.mean():>9.3f} {m.std():>9.3f} "
              f"{q[0]:>9.3f} {q[1]:>9.3f} {q[2]:>9.3f} {q[3]:>9.3f} {q[4]:>9.3f} "
              f"{cv:>9.3f}")
    print("\n(CV = coefficient of variation = std/|mean|.  Lower CV / lower std means "
          "margins are more uniform across the test set.  H44 predicts TRADES gives "
          "the most uniform margin distribution.)")

    # -------- comparison summary --------
    print("\n================ SUMMARY: does margin's dominance hold? ================")
    print(f"{'victim':<10} {'best_feat (flipped_PGD)':<28} {'margin_AUROC':>14} "
          f"{'best_AUROC':>12} {'best_name':<14}")
    for mode in ["vanilla", "fgsm", "pgd", "trades"]:
        r = results[mode]
        y = r["targets"]["flipped_PGD"]
        aurocs = {fn: safe_auroc(r["feats"][fn], y) for fn in feat_names}
        margin_auc = aurocs["margin"]
        best = max(aurocs.items(), key=lambda kv: (kv[1] if not np.isnan(kv[1]) else 0))
        print(f"{mode:<10} {'':<28} {margin_auc:>14.4f} {best[1]:>12.4f} {best[0]:<14}")

    print("\nDone.")


if __name__ == "__main__":
    main()
