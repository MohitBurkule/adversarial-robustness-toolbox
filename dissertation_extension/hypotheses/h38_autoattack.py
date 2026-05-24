"""
Hypothesis H38: AutoAttack (Croce & Hein 2020) is the gold-standard robustness
benchmark. Per-sample AutoAttack success vs FGSM gives a different sample-difficulty
ranking; check whether image stats / margin predict AutoAttack equally well.

Pipeline:
  1. Train a small CNN on Fashion-MNIST (10 epochs, Adam, same arch as
     diagnostic_test.py).
  2. Run AutoAttack at eps=15/255 (L_inf) on the test set. Record per-sample
     success flag `flipped_AutoAttack`. Fall back to a minimal APGD-CE
     implementation if neither `autoattack` nor `torchattacks` is installed.
  3. Compute per-sample features: victim_margin, mean_pix, std_pix, sobel_mean,
     jpeg_q75 (size in bytes at quality 75).
  4. Targets: flipped_AutoAttack, flipped_FGSM (eps=15/255). Cross-attack agreement.
  5. Univariate AUROC for each feature on each target. Compare margin's AUROC
     against AutoAttack vs FGSM.

Self-contained. Designed to run with the venv at
  /mnt/1tbone/trashy/adversarial-robustness-toolbox/.venv
Do NOT run from here; the parent agent will run it.
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
EPS = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
SEED = 0


# ---------------------------------------------------------------------------
# Model (matches diagnostic_test.py)
# ---------------------------------------------------------------------------
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


def train(model, train_set):
    model.train()
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        t0 = time.time()
        n_seen = 0
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            n_seen += x.size(0)
        print(f"  epoch {ep+1:2d}/{EPOCHS}  loss={loss_sum/n_seen:.4f}  "
              f"({time.time()-t0:.1f}s)")
    model.eval()


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------
def fgsm_attack(model, x, y, eps=EPS):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    adv = (x + eps * x.grad.sign()).clamp(0, 1).detach()
    with torch.no_grad():
        flipped = model(adv).argmax(1) != y
    return adv, flipped


# ---------------------------------------------------------------------------
# AutoAttack (with fall-backs)
# ---------------------------------------------------------------------------
def run_autoattack(model, x, y, eps=EPS):
    """Try autoattack package, then torchattacks, then a basic APGD-CE."""
    # 1) Croce & Hein's autoattack package
    try:
        from autoattack import AutoAttack
        print("  using `autoattack` package (full ensemble)")
        aa = AutoAttack(model, norm="Linf", eps=eps, version="standard",
                        device=DEVICE, verbose=False)
        adv = aa.run_standard_evaluation(x, y, bs=256)
        with torch.no_grad():
            preds = []
            for i in range(0, adv.size(0), 256):
                preds.append(model(adv[i:i+256]).argmax(1))
            preds = torch.cat(preds)
        return adv, preds != y, "autoattack-standard"
    except ImportError:
        pass
    except Exception as e:
        print(f"  autoattack package failed: {e}; trying torchattacks")

    # 2) torchattacks
    try:
        import torchattacks
        print("  using `torchattacks.AutoAttack`")
        atk = torchattacks.AutoAttack(model, norm="Linf", eps=eps,
                                       version="standard", n_classes=10, seed=SEED)
        advs = []
        for i in range(0, x.size(0), 256):
            advs.append(atk(x[i:i+256], y[i:i+256]))
        adv = torch.cat(advs, 0)
        with torch.no_grad():
            preds = []
            for i in range(0, adv.size(0), 256):
                preds.append(model(adv[i:i+256]).argmax(1))
            preds = torch.cat(preds)
        return adv, preds != y, "torchattacks-AutoAttack"
    except ImportError:
        pass
    except Exception as e:
        print(f"  torchattacks failed: {e}; falling back to manual APGD-CE")

    # 3) Manual APGD-CE (per arXiv:2003.01690).
    print("  using manual APGD-CE (single attack only)")
    adv, flipped = apgd_ce(model, x, y, eps=eps, n_iter=50)
    return adv, flipped, "manual-APGD-CE"


def apgd_ce(model, x, y, eps, n_iter=50, rho=0.75):
    """Minimal APGD-CE.

    Implements the auto-PGD-with-momentum + adaptive-step-size scheme from
    Croce & Hein (arXiv:2003.01690), Section 3.1.
    """
    model.eval()
    x = x.detach()
    y = y.detach()

    # init with random uniform perturbation
    x_adv = x + (2 * torch.rand_like(x) - 1) * eps
    x_adv = x_adv.clamp(0, 1).detach()

    step = 2 * eps
    x_best = x_adv.clone()

    def loss_per_sample(adv):
        logits = model(adv)
        return F.cross_entropy(logits, y, reduction="none"), logits

    with torch.enable_grad():
        x_adv.requires_grad_(True)
        loss, _ = loss_per_sample(x_adv)
        loss.sum().backward()
        grad = x_adv.grad.detach()
    loss_best = loss.detach().clone()
    x_adv = x_adv.detach()

    x_prev = x_adv.clone()
    # checkpoint schedule (paper formula simplified)
    checkpoints = [int(p * n_iter) for p in
                   [0.22, 0.41, 0.58, 0.72, 0.84, 0.93, 1.00]]
    step_t = torch.full((x.size(0),), step, device=x.device).view(-1, 1, 1, 1)
    n_improved_since_last = torch.zeros(x.size(0), device=x.device)
    loss_at_last_ckpt = loss_best.clone()
    last_ckpt = 0

    for k in range(1, n_iter + 1):
        # momentum step
        z = x_adv + step_t * grad.sign()
        z = torch.max(torch.min(z, x + eps), x - eps).clamp(0, 1)
        alpha = 0.75
        x_new = x_adv + alpha * (z - x_adv) + (1 - alpha) * (x_adv - x_prev)
        x_new = torch.max(torch.min(x_new, x + eps), x - eps).clamp(0, 1).detach()

        x_new.requires_grad_(True)
        loss, _ = loss_per_sample(x_new)
        loss.sum().backward()
        grad = x_new.grad.detach()
        loss_v = loss.detach()
        x_new = x_new.detach()

        improved = loss_v > loss_best
        n_improved_since_last += improved.float()
        x_best = torch.where(improved.view(-1, 1, 1, 1), x_new, x_best)
        loss_best = torch.where(improved, loss_v, loss_best)

        x_prev = x_adv
        x_adv = x_new

        if k in checkpoints:
            period = k - last_ckpt
            # condition 1: less than rho fraction of steps improved
            cond1 = n_improved_since_last < rho * period
            # condition 2: step didn't shrink AND loss did not improve since
            #              last checkpoint
            cond2 = (step_t.view(-1) >= step_t.view(-1)) & (loss_best <= loss_at_last_ckpt)
            shrink = cond1 | cond2
            step_t = torch.where(shrink.view(-1, 1, 1, 1),
                                  step_t / 2.0, step_t)
            # restart from current best when shrinking
            x_adv = torch.where(shrink.view(-1, 1, 1, 1), x_best, x_adv)
            n_improved_since_last.zero_()
            loss_at_last_ckpt = loss_best.clone()
            last_ckpt = k

    with torch.no_grad():
        preds = []
        for i in range(0, x_best.size(0), 256):
            preds.append(model(x_best[i:i+256]).argmax(1))
        preds = torch.cat(preds)
    return x_best, preds != y


# ---------------------------------------------------------------------------
# Image-stat features
# ---------------------------------------------------------------------------
SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
                        ).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
                        ).view(1, 1, 3, 3)


def compute_image_stats(x):
    """Return mean_pix, std_pix, sobel_mean for each image (CPU tensors)."""
    x = x.detach().cpu()
    mean_pix = x.mean(dim=(1, 2, 3))
    std_pix = x.std(dim=(1, 2, 3))
    gx = F.conv2d(x, SOBEL_X, padding=1)
    gy = F.conv2d(x, SOBEL_Y, padding=1)
    sobel_mag = torch.sqrt(gx ** 2 + gy ** 2)
    sobel_mean = sobel_mag.mean(dim=(1, 2, 3))
    return mean_pix.numpy(), std_pix.numpy(), sobel_mean.numpy()


def jpeg_q75_bytes(x):
    """Per-sample JPEG-quality-75 byte size."""
    x_np = (x.detach().cpu().numpy() * 255.0).astype(np.uint8)
    sizes = np.zeros(x_np.shape[0], dtype=np.float32)
    for i in range(x_np.shape[0]):
        img = Image.fromarray(x_np[i, 0], mode="L")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        sizes[i] = buf.tell()
    return sizes


def compute_margin(model, x):
    margins = []
    with torch.no_grad():
        for i in range(0, x.size(0), 512):
            logits = model(x[i:i+512])
            top2, _ = logits.topk(2, dim=1)
            margins.append((top2[:, 0] - top2[:, 1]).cpu())
    return torch.cat(margins).numpy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def two_sided_auroc(score, y):
    if y.std() == 0:
        return float("nan")
    a = roc_auc_score(y, score)
    return max(a, 1 - a), a  # also return raw direction


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"device = {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True,
                                      transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True,
                                     transform=tf)

    print("training CNN victim...")
    model = CNN(10).to(DEVICE)
    train(model, train_set)

    # stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]
                          ).to(DEVICE)

    # clean accuracy
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    clean_correct = (preds == test_y)
    print(f"clean test accuracy = {clean_correct.float().mean().item():.4f}")

    # restrict to samples correctly classified clean — attacks only meaningful
    # on those.
    x = test_x[clean_correct]
    y = test_y[clean_correct]
    print(f"running attacks on {x.size(0)} clean-correct samples")

    # -------- FGSM ----------
    print("\nrunning FGSM...")
    t0 = time.time()
    fgsm_flipped = []
    for i in range(0, x.size(0), 256):
        _, f = fgsm_attack(model, x[i:i+256], y[i:i+256], eps=EPS)
        fgsm_flipped.append(f.cpu())
    fgsm_flipped = torch.cat(fgsm_flipped).numpy().astype(int)
    print(f"  done ({time.time()-t0:.1f}s)  "
          f"FGSM success rate = {fgsm_flipped.mean():.4f}")

    # -------- AutoAttack ----------
    print("\nrunning AutoAttack...")
    t0 = time.time()
    _, aa_flipped, aa_name = run_autoattack(model, x, y, eps=EPS)
    aa_flipped = aa_flipped.detach().cpu().numpy().astype(int)
    print(f"  done ({time.time()-t0:.1f}s)  attack = {aa_name}  "
          f"success rate = {aa_flipped.mean():.4f}")

    # -------- features ----------
    print("\ncomputing features...")
    margin = compute_margin(model, x)
    mean_pix, std_pix, sobel_mean = compute_image_stats(x)
    jpeg_q75 = jpeg_q75_bytes(x)
    features = {
        "victim_margin": margin,
        "mean_pix": mean_pix,
        "std_pix": std_pix,
        "sobel_mean": sobel_mean,
        "jpeg_q75": jpeg_q75,
    }

    # -------- cross-attack agreement ----------
    both = fgsm_flipped & aa_flipped
    either = fgsm_flipped | aa_flipped
    aa_only = aa_flipped & (~fgsm_flipped.astype(bool)).astype(int)
    fgsm_only = fgsm_flipped & (~aa_flipped.astype(bool)).astype(int)
    iou = both.sum() / max(either.sum(), 1)
    print("\n=== cross-attack agreement ===")
    print(f"  FGSM flips        : {fgsm_flipped.sum()} "
          f"({fgsm_flipped.mean():.4f})")
    print(f"  AutoAttack flips  : {aa_flipped.sum()} "
          f"({aa_flipped.mean():.4f})")
    print(f"  both              : {both.sum()}")
    print(f"  AutoAttack only   : {aa_only.sum()}")
    print(f"  FGSM only         : {fgsm_only.sum()}")
    print(f"  IoU (AA ∩ FGSM)/(AA ∪ FGSM) = {iou:.4f}")
    # Pearson correlation between binary success vectors
    pearson = np.corrcoef(fgsm_flipped, aa_flipped)[0, 1]
    print(f"  Pearson(success_FGSM, success_AA) = {pearson:+.4f}")

    # -------- univariate AUROC ----------
    targets = {
        "flipped_FGSM": fgsm_flipped,
        "flipped_AutoAttack": aa_flipped,
        "agreement_AA_and_FGSM": both,
    }

    print("\n=== univariate AUROC per feature per target ===")
    print(f"{'feature':<16}  " +
          "  ".join(f"{t:>22}" for t in targets))
    aurocs = {}
    for fname, fvals in features.items():
        row = []
        for tname, tvals in targets.items():
            if tvals.std() == 0:
                row.append(float("nan"))
                continue
            a, raw = two_sided_auroc(fvals, tvals)
            aurocs[(fname, tname)] = (a, raw)
            row.append(a)
        print(f"{fname:<16}  " +
              "  ".join(f"{v:>22.4f}" for v in row))

    # -------- direct margin comparison ----------
    print("\n=== does margin predict AutoAttack as well as FGSM? ===")
    a_fgsm, _ = aurocs[("victim_margin", "flipped_FGSM")]
    a_aa, _ = aurocs[("victim_margin", "flipped_AutoAttack")]
    print(f"  AUROC(margin -> flipped_FGSM      ) = {a_fgsm:.4f}")
    print(f"  AUROC(margin -> flipped_AutoAttack) = {a_aa:.4f}")
    print(f"  delta (AA - FGSM)                  = {a_aa - a_fgsm:+.4f}")
    if a_aa > a_fgsm:
        verdict = "margin predicts AutoAttack BETTER than FGSM"
    elif a_aa < a_fgsm:
        verdict = "margin predicts AutoAttack WORSE than FGSM"
    else:
        verdict = "margin predicts both equally"
    print(f"  verdict: {verdict}")

    # -------- per-feature delta summary ----------
    print("\n=== AUROC delta (AutoAttack - FGSM) per feature ===")
    for fname in features:
        a_aa, _ = aurocs[(fname, "flipped_AutoAttack")]
        a_fg, _ = aurocs[(fname, "flipped_FGSM")]
        print(f"  {fname:<16}  AA={a_aa:.4f}  FGSM={a_fg:.4f}  "
              f"delta={a_aa - a_fg:+.4f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
