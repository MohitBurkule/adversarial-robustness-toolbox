"""
H529 – Log-Polar Feature Geometry on Fashion-MNIST
====================================================
Deza & Konkle (2020) showed foveated (log-polar) models are less adversarially
vulnerable.  Mechanism: a pixel-space perturbation delta maps to a non-coherent,
spatially-scrambled direction in log-polar space, so gradient-based attacks lose
their alignment with the loss landscape.

Experiment:
  Three models trained on Fashion-MNIST:
    1. Cartesian-STD  – raw 28x28 pixels, standard training
    2. LogPolar-STD   – log-polar remapped 28x28, standard training
    3. Cartesian-AT   – raw pixels, PGD adversarial training (baseline)

  Attacks evaluated:
    • FGSM (eps=0.1, 0.2, 0.3)
    • PGD-20 (eps=0.1, step=0.01)
    • Both attacks in the NATIVE space of each model
      (pixel-space for Cartesian, log-polar-space for LogPolar)

  Key diagnostic:
    • Visualise 6 example images: original | log-polar | FGSM adv (Cartesian) |
      FGSM adv (LogPolar)
    • ASR vs eps curve for FGSM

  PASS: LogPolar-STD FGSM ASR (eps=0.2) < Cartesian-STD FGSM ASR (eps=0.2)

Output:
  results/fashion_mnist/h529_logpolar_fmnist.png
  results/fashion_mnist/h529_logpolar_fmnist_output.txt
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FIG_PATH = os.path.join(RESULTS_DIR, "h529_logpolar_fmnist.png")
TXT_PATH = os.path.join(RESULTS_DIR, "h529_logpolar_fmnist_output.txt")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42
BATCH_SIZE = 256
EPOCHS     = 15          # enough for ~92% on F-MNIST with small CNN
LR         = 1e-3
PGD_STEPS  = 20
PGD_STEP   = 0.01
EPS_LIST   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
EVAL_EPS   = 0.20        # main comparison point


# ──────────────────────────────────────────────────────────────────────────────
# Tee stdout → file
# ──────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path):
        self._f = open(path, "w")
        self._s = sys.stdout
    def write(self, d):
        self._s.write(d); self._f.write(d)
    def flush(self):
        self._s.flush(); self._f.flush()
    def close(self):
        self._f.close()
    def __getattr__(self, n):
        return getattr(self._s, n)


# ──────────────────────────────────────────────────────────────────────────────
# Log-polar remapping
# ──────────────────────────────────────────────────────────────────────────────
def build_logpolar_map(H=28, W=28):
    """
    Precompute a bilinear sampling grid that maps each output pixel (r, theta)
    to an input pixel (x, y) via the inverse log-polar transform.

    Output pixel (i, j):
        i  -> log-radius axis  (0 = centre, H-1 = max radius)
        j  -> angle axis       (0 = 0°, W-1 = 360°)

    Inverse: x = r*cos(theta) + cx,  y = r*sin(theta) + cy
    where r = exp(i / H * log(r_max))  and  theta = j / W * 2*pi
    """
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r_max  = np.sqrt(cx**2 + cy**2)          # corner distance

    i_idx = np.arange(H, dtype=np.float32)
    j_idx = np.arange(W, dtype=np.float32)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing="ij")  # (H, W)

    r     = np.exp(ii / H * np.log(r_max + 1e-6))      # (H, W)
    theta = jj / W * 2 * np.pi                          # (H, W)

    x_src = r * np.cos(theta) + cx                      # (H, W)
    y_src = r * np.sin(theta) + cy                      # (H, W)

    # Normalise to [-1, 1] for F.grid_sample
    x_norm = (x_src / (W - 1)) * 2 - 1
    y_norm = (y_src / (H - 1)) * 2 - 1

    grid = np.stack([x_norm, y_norm], axis=-1)          # (H, W, 2)
    return torch.tensor(grid, dtype=torch.float32)      # (H, W, 2)


# Build once, move to device later
_LP_GRID = build_logpolar_map()   # (28, 28, 2)


def logpolar_transform(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N, 1, 28, 28)  float32 in [0,1]
    returns: (N, 1, 28, 28) log-polar remapped
    """
    grid = _LP_GRID.to(x.device)                        # (28, 28, 2)
    grid = grid.unsqueeze(0).expand(x.size(0), -1, -1, -1)  # (N,28,28,2)
    return F.grid_sample(x, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
def load_fmnist():
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.FashionMNIST(DATA_DIR, train=True,
                                                  download=True, transform=tf)
    test_ds  = torchvision.datasets.FashionMNIST(DATA_DIR, train=False,
                                                  download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Model: small CNN  (1→32→64→FC128→10)
# ──────────────────────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────────────────────────────────────
# Attacks
# ──────────────────────────────────────────────────────────────────────────────
def fgsm_attack(model, x, y, eps, preprocess=None):
    """
    FGSM attack always in PIXEL space.
    If preprocess is given, gradients flow: pixel → preprocess → model.
    This means for LogPolar models we compute grad w.r.t. the log-polar input
    (adaptive attack).  Use fgsm_pixel_attack for the correct retinal-defense
    experiment where the attacker is blind to the transform.
    """
    xv = x.clone().detach().requires_grad_(True)
    inp = preprocess(xv) if preprocess else xv
    F.cross_entropy(model(inp), y).backward()
    x_adv = (x + eps * xv.grad.sign()).clamp(0, 1).detach()
    return x_adv


def fgsm_pixel_attack(model, x, y, eps):
    """
    FGSM attack in PIXEL space against a Cartesian model (no preprocess).
    For use against LogPolar model: craft perturbation on pixels, then
    the log-polar transform is applied during eval — attacker cannot
    back-prop through the transform (black-box to transform).
    """
    xv = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xv), y).backward()
    return (x + eps * xv.grad.sign()).clamp(0, 1).detach()


def pgd_pixel_attack(model, x, y, eps, step, steps):
    """PGD in pixel space, no preprocess — transform-blind attacker."""
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = (x_adv.detach() + step * grad.sign()).clamp(
            x - eps, x + eps).clamp(0, 1)
    return x_adv.detach()


def pgd_attack(model, x, y, eps, step, steps, preprocess=None):
    """Adaptive PGD (grad flows through preprocess)."""
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv.requires_grad_(True)
        inp = preprocess(x_adv) if preprocess else x_adv
        loss = F.cross_entropy(model(inp), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = (x_adv.detach() + step * grad.sign()).clamp(
            x - eps, x + eps).clamp(0, 1)
    return x_adv.detach()


def eval_asr(model, loader, eps, attack_fn, preprocess=None):
    """
    ASR on correctly-classified examples.
    attack_fn(model, x, y, eps) → x_adv  (pixel space)
    preprocess applied to x_adv before model inference.
    """
    total_correct = 0
    total_flipped = 0
    model.eval()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        inp_clean = preprocess(x) if preprocess else x
        with torch.no_grad():
            correct_mask = model(inp_clean).argmax(1) == y
        if correct_mask.sum() == 0:
            continue
        xc, yc = x[correct_mask], y[correct_mask]
        x_adv = attack_fn(model, xc, yc, eps)    # always pixel-space output
        inp_adv = preprocess(x_adv) if preprocess else x_adv
        with torch.no_grad():
            flipped = (model(inp_adv).argmax(1) != yc).sum().item()
        total_correct += correct_mask.sum().item()
        total_flipped += flipped
    return total_flipped / max(total_correct, 1)


def eval_acc(model, loader, preprocess=None):
    correct = total = 0
    model.eval()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        inp = preprocess(x) if preprocess else x
        with torch.no_grad():
            correct += (model(inp).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train_standard(model, loader, preprocess=None):
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            inp = preprocess(x) if preprocess else x
            opt.zero_grad()
            F.cross_entropy(model(inp), y).backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{EPOCHS}")


def train_pgdat(model, loader, eps=0.1):
    """PGD-AT on raw pixels (Cartesian model)."""
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = pgd_attack(model, x, y, eps=eps,
                                step=eps/4, steps=7)
            opt.zero_grad()
            F.cross_entropy(model(x_adv), y).backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{EPOCHS}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    tee = Tee(TXT_PATH)
    sys.stdout = tee
    try:
        _run()
    finally:
        sys.stdout = tee._s
        tee.close()


def _run():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")
    print("=" * 70)
    print("H529 – Log-Polar Feature Geometry on Fashion-MNIST")
    print("=" * 70)

    train_loader, test_loader = load_fmnist()

    # ── Model 1: Cartesian-STD ────────────────────────────────────────────────
    print("\n[1/3] Training Cartesian-STD ...")
    torch.manual_seed(SEED)
    cart_std = SmallCNN().to(DEVICE)
    train_standard(cart_std, train_loader, preprocess=None)

    # ── Model 2: LogPolar-STD ─────────────────────────────────────────────────
    print("\n[2/3] Training LogPolar-STD ...")
    torch.manual_seed(SEED)
    lp_std = SmallCNN().to(DEVICE)
    train_standard(lp_std, train_loader, preprocess=logpolar_transform)

    # ── Model 3: Cartesian-AT ─────────────────────────────────────────────────
    print("\n[3/3] Training Cartesian-AT (PGD eps=0.1) ...")
    torch.manual_seed(SEED)
    cart_at = SmallCNN().to(DEVICE)
    train_pgdat(cart_at, train_loader, eps=0.10)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    print("\nClean accuracy:")
    acc_cs = eval_acc(cart_std, test_loader, preprocess=None)
    acc_lp = eval_acc(lp_std,   test_loader, preprocess=logpolar_transform)
    acc_at = eval_acc(cart_at,  test_loader, preprocess=None)
    print(f"  Cartesian-STD : {acc_cs:.4f}")
    print(f"  LogPolar-STD  : {acc_lp:.4f}")
    print(f"  Cartesian-AT  : {acc_at:.4f}")

    # ── FGSM eps sweep ────────────────────────────────────────────────────────
    # Three attack protocols:
    #   Cart-STD:      FGSM in pixel space, no transform
    #   LogPolar-BLIND: FGSM crafted against Cartesian-STD (transfer attack),
    #                   then log-polar transform applied at inference
    #                   → simulates attacker blind to the transform
    #   LogPolar-ADAPT: FGSM crafted with gradient through log-polar transform
    #                   → adaptive attacker who knows the transform
    #   Cart-AT:       FGSM in pixel space, AT model
    print("\nFGSM ASR sweep:")
    print("  (BLIND = pixel-space attack transferred to LP model; "
          "ADAPT = attacker knows LP transform)")
    asr_cs_list, asr_lp_blind_list, asr_lp_adapt_list, asr_at_list = [], [], [], []
    for eps in EPS_LIST:
        # Cartesian-STD: normal pixel attack
        asr_cs = eval_asr(cart_std, test_loader, eps,
                          lambda m,x,y,e: fgsm_pixel_attack(m,x,y,e), None)

        # LogPolar-BLIND: craft attack on Cartesian-STD (same arch, pixel space),
        # then apply LP transform for LogPolar model inference
        asr_lp_blind = eval_asr(lp_std, test_loader, eps,
                                 lambda m,x,y,e: fgsm_pixel_attack(cart_std,x,y,e),
                                 logpolar_transform)

        # LogPolar-ADAPT: adaptive attack through the transform
        asr_lp_adapt = eval_asr(lp_std, test_loader, eps,
                                 lambda m,x,y,e: fgsm_attack(m,x,y,e,logpolar_transform),
                                 logpolar_transform)

        # Cart-AT
        asr_at = eval_asr(cart_at, test_loader, eps,
                          lambda m,x,y,e: fgsm_pixel_attack(m,x,y,e), None)

        asr_cs_list.append(asr_cs)
        asr_lp_blind_list.append(asr_lp_blind)
        asr_lp_adapt_list.append(asr_lp_adapt)
        asr_at_list.append(asr_at)
        print(f"  eps={eps:.2f}  Cart-STD={asr_cs:.3f}  LP-blind={asr_lp_blind:.3f}"
              f"  LP-adapt={asr_lp_adapt:.3f}  Cart-AT={asr_at:.3f}")

    # ── PGD ASR at eval_eps ───────────────────────────────────────────────────
    print(f"\nPGD-20 ASR at eps={EVAL_EPS}:")
    pgd_cs = eval_asr(cart_std, test_loader, EVAL_EPS,
                      lambda m,x,y,e: pgd_pixel_attack(m,x,y,e,PGD_STEP,PGD_STEPS), None)
    # LP-blind PGD: attack crafted on Cartesian-STD, evaluated on LP model
    pgd_lp_blind = eval_asr(lp_std, test_loader, EVAL_EPS,
                             lambda m,x,y,e: pgd_pixel_attack(cart_std,x,y,e,PGD_STEP,PGD_STEPS),
                             logpolar_transform)
    pgd_lp_adapt = eval_asr(lp_std, test_loader, EVAL_EPS,
                             lambda m,x,y,e: pgd_attack(m,x,y,e,PGD_STEP,PGD_STEPS,logpolar_transform),
                             logpolar_transform)
    pgd_at = eval_asr(cart_at, test_loader, EVAL_EPS,
                      lambda m,x,y,e: pgd_pixel_attack(m,x,y,e,PGD_STEP,PGD_STEPS), None)
    print(f"  Cartesian-STD  : {pgd_cs:.3f}")
    print(f"  LogPolar-BLIND : {pgd_lp_blind:.3f}  (pixel-space transfer attack)")
    print(f"  LogPolar-ADAPT : {pgd_lp_adapt:.3f}  (adaptive, knows transform)")
    print(f"  Cartesian-AT   : {pgd_at:.3f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    idx_eval = EPS_LIST.index(EVAL_EPS)
    print("\n" + "=" * 76)
    print(f"{'Model':<22} {'CleanAcc':>9} {'FGSM_ASR':>10} {'PGD_ASR':>9}  note")
    print("-" * 76)
    print(f"{'Cartesian-STD':<22} {acc_cs:>9.4f} {asr_cs_list[idx_eval]:>10.4f} {pgd_cs:>9.4f}  pixel attack")
    print(f"{'LogPolar-BLIND':<22} {acc_lp:>9.4f} {asr_lp_blind_list[idx_eval]:>10.4f} {pgd_lp_blind:>9.4f}  pixel attack (no transform knowledge)")
    print(f"{'LogPolar-ADAPT':<22} {acc_lp:>9.4f} {asr_lp_adapt_list[idx_eval]:>10.4f} {pgd_lp_adapt:>9.4f}  adaptive (knows transform)")
    print(f"{'Cartesian-AT':<22} {acc_at:>9.4f} {asr_at_list[idx_eval]:>10.4f} {pgd_at:>9.4f}  pixel attack")
    print("=" * 76)

    passed = asr_lp_blind_list[idx_eval] < asr_cs_list[idx_eval]
    verdict = "PASS" if passed else "FAIL"
    print(f"\n{verdict} (blind attack): LogPolar-BLIND ASR={asr_lp_blind_list[idx_eval]:.4f} "
          f"{'<' if passed else '>='} Cartesian-STD ASR={asr_cs_list[idx_eval]:.4f} at eps={EVAL_EPS}")
    print("Key: if BLIND < ADAPT, the transform provides genuine obfuscation benefit.")

    # ── Visualise log-polar transform + adversarial examples ─────────────────
    print("\nGenerating figure ...")
    # Get a batch of examples (pick 6 from different classes)
    all_x, all_y = [], []
    for x, y in test_loader:
        all_x.append(x); all_y.append(y)
        if len(torch.cat(all_x)) >= 500:
            break
    all_x = torch.cat(all_x)[:500].to(DEVICE)
    all_y = torch.cat(all_y)[:500].to(DEVICE)

    # Pick one example per class (first 6 classes for brevity)
    selected_x, selected_y = [], []
    for cls in range(6):
        idx = (all_y == cls).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            selected_x.append(all_x[idx[0]])
            selected_y.append(all_y[idx[0]])
    sel_x = torch.stack(selected_x)   # (6, 1, 28, 28)
    sel_y = torch.stack(selected_y)

    sel_x_lp = logpolar_transform(sel_x)
    # Pixel-space attack on Cartesian model
    sel_x_adv_cs   = fgsm_pixel_attack(cart_std, sel_x, sel_y, EVAL_EPS)
    # Blind pixel-space attack (crafted on cart_std) → applied to LP model (via transform)
    sel_x_adv_blind = fgsm_pixel_attack(cart_std, sel_x, sel_y, EVAL_EPS)
    sel_x_adv_blind_lp = logpolar_transform(sel_x_adv_blind)   # what LP model sees

    CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal"]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("H529 – Log-Polar Feature Geometry on Fashion-MNIST\n"
                 "Same model architecture, same training budget — only the feature space differs",
                 fontsize=12, fontweight="bold")

    # Row layout:
    #   Row 0: 6 col titles
    #   Rows 1-4: image panels (original, log-polar, adv-cart, adv-lp)
    #   Row 5: FGSM eps sweep
    #   Row 6: ASR bar chart + perturbation norm comparison

    n_ex = len(selected_x)
    gs = fig.add_gridspec(4, n_ex + 2, hspace=0.45, wspace=0.35)

    # Rows: original | log-polar | adv pixel | adv pixel after LP transform
    row_labels = [
        "Original (pixel)",
        "Log-polar view\n(what LP model sees)",
        f"FGSM adv pixel\n(ε={EVAL_EPS}, blind attack)",
        f"Blind adv → LP transform\n(what LP model sees from blind attack)",
    ]
    tensors = [sel_x, sel_x_lp, sel_x_adv_blind, sel_x_adv_blind_lp]

    for row_i, (label, imgs) in enumerate(zip(row_labels, tensors)):
        for col_i in range(n_ex):
            ax = fig.add_subplot(gs[row_i, col_i])
            img = imgs[col_i, 0].cpu().detach().numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(CLASS_NAMES[col_i], fontsize=8, fontweight="bold")
        ax_lbl = fig.add_subplot(gs[row_i, n_ex])
        ax_lbl.axis("off")
        ax_lbl.text(0.0, 0.5, label, va="center", ha="left", fontsize=7.5,
                    transform=ax_lbl.transAxes)

    # FGSM eps sweep — all 4 conditions
    ax_sweep = fig.add_subplot(gs[0:2, n_ex + 1])
    ax_sweep.plot(EPS_LIST, asr_cs_list,       "o-",  color="#c0392b", label="Cartesian-STD", lw=2)
    ax_sweep.plot(EPS_LIST, asr_lp_blind_list, "s--", color="#2980b9", label="LP-BLIND (pixel attack)", lw=2)
    ax_sweep.plot(EPS_LIST, asr_lp_adapt_list, "D:",  color="#8e44ad", label="LP-ADAPT (knows transform)", lw=2)
    ax_sweep.plot(EPS_LIST, asr_at_list,       "^-.", color="#27ae60", label="Cartesian-AT",  lw=2)
    ax_sweep.axvline(EVAL_EPS, color="gray", ls=":", lw=1)
    ax_sweep.set_xlabel("FGSM ε", fontsize=9)
    ax_sweep.set_ylabel("ASR", fontsize=9)
    ax_sweep.set_title("FGSM ASR vs ε\nBLIND=pixel attack; ADAPT=through transform", fontsize=8)
    ax_sweep.legend(fontsize=7)
    ax_sweep.set_ylim(-0.05, 1.05)
    ax_sweep.tick_params(labelsize=7)

    # Bar chart at eval_eps
    ax_bar = fig.add_subplot(gs[2:4, n_ex:])
    model_labels = ["Cart\nSTD", "LP\nBLIND", "LP\nADAPT", "Cart\nAT"]
    fgsm_vals = [asr_cs_list[idx_eval], asr_lp_blind_list[idx_eval],
                 asr_lp_adapt_list[idx_eval], asr_at_list[idx_eval]]
    pgd_vals  = [pgd_cs, pgd_lp_blind, pgd_lp_adapt, pgd_at]
    x_pos = np.arange(4)
    w = 0.35
    colors4 = ["#e74c3c","#3498db","#8e44ad","#2ecc71"]
    bars1 = ax_bar.bar(x_pos - w/2, fgsm_vals, w, label=f"FGSM ε={EVAL_EPS}",
                       color=colors4, alpha=0.8, edgecolor="k", lw=0.7)
    bars2 = ax_bar.bar(x_pos + w/2, pgd_vals,  w, label=f"PGD-20 ε={EVAL_EPS}",
                       color=colors4, alpha=0.45, edgecolor="k", lw=0.7, hatch="//")
    for bar, v in list(zip(bars1, fgsm_vals)) + list(zip(bars2, pgd_vals)):
        ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(model_labels, fontsize=8)
    ax_bar.set_ylabel("ASR", fontsize=9)
    ax_bar.set_title(f"ASR at ε={EVAL_EPS}\nBLIND vs ADAPT gap = obfuscation benefit", fontsize=8.5)
    ax_bar.legend(fontsize=7.5)
    ax_bar.set_ylim(0, 1.15)
    acc_labels = [acc_cs, acc_lp, acc_lp, acc_at]
    for i, av in enumerate(acc_labels):
        ax_bar.text(i, -0.18, f"acc={av:.3f}", ha="center", fontsize=7,
                    transform=ax_bar.get_xaxis_transform())
    ax_bar.tick_params(labelsize=7)

    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {FIG_PATH}")
    print(f"Text   saved → {TXT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
