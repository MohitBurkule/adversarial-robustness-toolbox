"""
Hypothesis H42: Per-sample vulnerability to spatial attacks (rotation+translation,
Engstrom et al. 2019) is orthogonal to L_inf vulnerability. Image statistics may
predict spatial vulnerability differently than they predict L_inf vulnerability.

Context: spatial robustness has been shown to be largely orthogonal to L_inf
robustness (Tramer & Boneh 2019, Engstrom et al. 2019). Open question: do the
*same* per-sample features predict both kinds of vulnerability, or do different
features dominate?

Pipeline:
  1. Train a small CNN victim on Fashion-MNIST (10 epochs, architecture matches
     diagnostic_test.py).
  2. For each test sample, grid-search over
        rotations    in {-30, -20, -10, 0, +10, +20, +30} degrees
        translations in {-3, -2, -1, 0, +1, +2, +3} pixels (each axis)
     Find the smallest (|rotation|, |tx|, |ty|) tuple that flips the prediction.
     Record success at fixed budget (|rotation| <= 10 deg, max(|tx|,|ty|) <= 2 px).
  3. Compute features: victim_margin, mean_pix, std_pix, sobel_mean.
  4. Targets:
        - flipped_spatial            (binary, budget-restricted spatial flip)
        - spatial_min_perturbation   (continuous: combined magnitude of smallest
                                      flipping transform; large value if not flipped)
        - flipped_FGSM_eps15         (binary, FGSM at eps=15/255)
  5. Univariate AUROC of each feature against each binary target; Pearson
     correlation against the continuous target.
  6. Cross-target analysis: Pearson correlation between spatial-vulnerability and
     FGSM-vulnerability; 2x2 confusion table; report samples that are spatially
     fragile but FGSM-robust (and vice-versa).
"""
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda")
EPOCHS = 10
BATCH = 128
EPS_FGSM = 15.0 / 255.0
DATA_ROOT = "/tmp/data"

# Spatial attack grid (Engstrom et al. 2019 style)
ROTATIONS = [-30, -20, -10, 0, 10, 20, 30]
TRANSLATIONS = [-3, -2, -1, 0, 1, 2, 3]

# Fixed-budget thresholds for the binary spatial flip target
BUDGET_ROT = 10.0
BUDGET_TRANS = 2


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


def train_victim(train_set):
    torch.manual_seed(0)
    np.random.seed(0)
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
        print(f"  epoch {ep+1}/{EPOCHS} ({time.time()-t0:.1f}s)")
    model.eval()
    return model


def batched_predict(model, x, batch=512):
    preds, margins = [], []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            logits = model(x[i:i+batch])
            sorted_l, _ = logits.sort(1, descending=True)
            margins.append(sorted_l[:, 0] - sorted_l[:, 1])
            preds.append(logits.argmax(1))
    return torch.cat(preds), torch.cat(margins)


def apply_affine(x, angle, tx, ty):
    """Apply (rotation, translation) to a batch of images via TF.affine.
    angle in degrees; tx,ty in pixels."""
    # TF.affine works on tensors of shape (C,H,W) or (B,C,H,W)
    return TF.affine(x, angle=float(angle), translate=[int(tx), int(ty)],
                     scale=1.0, shear=[0.0, 0.0])


def spatial_attack(model, x, y, batch=256):
    """For each sample in (x,y), grid-search rotation x translation_x x translation_y.

    Returns:
      flipped_budget : bool tensor [N] -- any (rot,tx,ty) within budget flips pred
      min_pert       : float tensor [N] -- magnitude of the smallest flipping
                        transform (sqrt(rot^2/30^2 + tx^2/3^2 + ty^2/3^2)); set
                        to +inf (encoded as a large number) if no flip found.
      any_flipped    : bool tensor [N] -- any grid cell flips
    """
    N = x.size(0)
    flipped_budget = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    any_flipped = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    # store minimum normalised magnitude (inf encoded as 10.0)
    BIG = 10.0
    min_pert = torch.full((N,), BIG, device=DEVICE)

    combos = list(itertools.product(ROTATIONS, TRANSLATIONS, TRANSLATIONS))
    for (rot, tx, ty) in combos:
        mag = float(np.sqrt((rot/30.0)**2 + (tx/3.0)**2 + (ty/3.0)**2))
        in_budget = (abs(rot) <= BUDGET_ROT and
                     max(abs(tx), abs(ty)) <= BUDGET_TRANS)
        # apply transform in batches
        preds_all = []
        with torch.no_grad():
            for i in range(0, N, batch):
                xb = x[i:i+batch]
                xt = apply_affine(xb, rot, tx, ty)
                preds_all.append(model(xt).argmax(1))
        preds = torch.cat(preds_all)
        flipped = preds != y
        any_flipped |= flipped
        if in_budget:
            flipped_budget |= flipped
        # update min_pert for samples flipped at this (rot,tx,ty) where mag<current
        update = flipped & (mag < min_pert)
        min_pert[update] = mag
    return flipped_budget, min_pert, any_flipped


def fgsm_attack(model, x, y, eps=EPS_FGSM, batch=256):
    """Standard untargeted FGSM at fixed eps. Returns bool tensor [N]."""
    flipped = torch.zeros(x.size(0), dtype=torch.bool, device=DEVICE)
    for i in range(0, x.size(0), batch):
        xb = x[i:i+batch].clone().detach().requires_grad_(True)
        yb = y[i:i+batch]
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        sign = xb.grad.sign().detach()
        adv = (xb.detach() + eps * sign).clamp(0, 1)
        with torch.no_grad():
            flipped[i:i+xb.size(0)] = model(adv).argmax(1) != yb
    return flipped


def sobel_mean(x):
    """Mean absolute Sobel gradient magnitude per image. x: [N,1,H,W]."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag.mean(dim=[1, 2, 3])


def univariate_auroc(score, y_binary):
    s = score.detach().cpu().numpy()
    yy = y_binary.detach().cpu().numpy().astype(int)
    if yy.std() == 0:
        return float("nan")
    a = roc_auc_score(yy, s)
    return max(a, 1 - a)


def pearson(a, b):
    a = a.detach().cpu().numpy().astype(float)
    b = b.detach().cpu().numpy().astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print("##### H42: spatial vs L_inf vulnerability on Fashion-MNIST #####")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True,
                                      transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True,
                                     transform=tf)

    print("\n[1] training victim CNN (10 epochs) ...")
    model = train_victim(train_set)

    print("\n[2] loading test set to GPU ...")
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    pred, margin = batched_predict(model, test_x)
    correct = pred == test_y
    print(f"  victim clean accuracy = {correct.float().mean().item():.4f}")
    x_c = test_x[correct]
    y_c = test_y[correct]
    margin_c = margin[correct]
    print(f"  restricting to {x_c.size(0)} correctly classified test samples")

    print("\n[3] running spatial grid attack "
          f"({len(ROTATIONS)} rot x {len(TRANSLATIONS)}^2 trans = "
          f"{len(ROTATIONS)*len(TRANSLATIONS)**2} cells) ...")
    t0 = time.time()
    flipped_spatial, spatial_min_pert, any_flipped = spatial_attack(model, x_c, y_c)
    print(f"  done ({time.time()-t0:.1f}s)")
    print(f"  flipped at budget (rot<={BUDGET_ROT}, trans<={BUDGET_TRANS}): "
          f"{flipped_spatial.float().mean().item():.4f}")
    print(f"  flipped anywhere in grid:                                    "
          f"{any_flipped.float().mean().item():.4f}")
    print(f"  mean spatial_min_perturbation (10 = not flipped):            "
          f"{spatial_min_pert.mean().item():.4f}")

    print("\n[4] running FGSM at eps=15/255 ...")
    flipped_fgsm = fgsm_attack(model, x_c, y_c)
    print(f"  FGSM flip rate: {flipped_fgsm.float().mean().item():.4f}")

    print("\n[5] computing per-sample features ...")
    mean_pix = x_c.mean(dim=[1, 2, 3])
    std_pix = x_c.std(dim=[1, 2, 3])
    sob = sobel_mean(x_c)
    feat_names = ["victim_margin", "mean_pix", "std_pix", "sobel_mean"]
    feats = {"victim_margin": margin_c,
             "mean_pix": mean_pix,
             "std_pix": std_pix,
             "sobel_mean": sob}

    print("\n[6] univariate AUROC per feature per binary target")
    bin_targets = {"flipped_spatial": flipped_spatial,
                   "flipped_FGSM_eps15": flipped_fgsm}
    for tname, tgt in bin_targets.items():
        print(f"\n  target = {tname} (positive rate = "
              f"{tgt.float().mean().item():.4f})")
        for fn in feat_names:
            a = univariate_auroc(feats[fn], tgt)
            print(f"    AUROC  {fn:<16} {a:.4f}")

    print("\n[7] Pearson correlation between features and spatial_min_perturbation"
          " (lower = more fragile)")
    for fn in feat_names:
        r = pearson(feats[fn], spatial_min_pert)
        print(f"    r({fn:<16}, spatial_min_pert) = {r:+.4f}")

    print("\n[8] cross-target: spatial vs L_inf vulnerability")
    r_bin = pearson(flipped_spatial.float(), flipped_fgsm.float())
    print(f"  Pearson(flipped_spatial, flipped_FGSM)        = {r_bin:+.4f}")
    # correlation of continuous spatial fragility vs FGSM-flip binary
    r_cont = pearson(spatial_min_pert, flipped_fgsm.float())
    print(f"  Pearson(spatial_min_pert, flipped_FGSM)       = {r_cont:+.4f}")

    s = flipped_spatial.cpu().numpy().astype(int)
    f = flipped_fgsm.cpu().numpy().astype(int)
    both = int(((s == 1) & (f == 1)).sum())
    only_s = int(((s == 1) & (f == 0)).sum())
    only_f = int(((s == 0) & (f == 1)).sum())
    neither = int(((s == 0) & (f == 0)).sum())
    n = len(s)
    print("\n  2x2 confusion table (rows=spatial, cols=FGSM):")
    print(f"                 FGSM=0   FGSM=1")
    print(f"    spatial=0    {neither:>6}   {only_f:>6}")
    print(f"    spatial=1    {only_s:>6}   {both:>6}")
    print(f"    N = {n}")
    print(f"    spatially-fragile & FGSM-robust:  "
          f"{only_s} ({100*only_s/n:.2f}%)")
    print(f"    spatially-robust  & FGSM-fragile: "
          f"{only_f} ({100*only_f/n:.2f}%)")
    # phi coefficient (same as Pearson for binary)
    print(f"    phi-coefficient = Pearson r        = {r_bin:+.4f}")

    print("\n[9] feature comparison: do same features predict both?")
    print(f"  {'feature':<16} {'AUROC_spatial':>14} {'AUROC_FGSM':>12} "
          f"{'difference':>11}")
    for fn in feat_names:
        a_s = univariate_auroc(feats[fn], flipped_spatial)
        a_f = univariate_auroc(feats[fn], flipped_fgsm)
        print(f"  {fn:<16} {a_s:>14.4f} {a_f:>12.4f} {a_s-a_f:>+11.4f}")

    print("\nH42 evaluation complete.")


if __name__ == "__main__":
    main()
