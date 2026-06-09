"""
H62: StAdv (Xiao et al. 2018, arXiv:1801.02612) spatially-transformed adversarial
examples via a pixel-wise flow field. Instead of perturbing pixel intensities
(as in FGSM/PGD), we optimise a per-pixel 2-D displacement f = (f_u, f_v) and
resample the image bilinearly at the displaced coordinates so that the model's
prediction flips. We then ask: do simple per-image features (margin, mean/std
intensity, mean Sobel response) predict (a) the magnitude of the flow required
to flip, and (b) whether StAdv succeeds at all under a small flow budget?
Univariate AUROC, compared head-to-head with classical FGSM (L_inf, eps=15/255).

Reference: Xiao, Zhu, Li, He, Liu, Song. "Spatially Transformed Adversarial
Examples", ICLR 2018.

Pipeline:
  1. Train a small CNN (architecture mirrors diagnostic_test.py) on
     Fashion-MNIST for 10 epochs.
  2. For correctly classified test samples, run StAdv:
        - learnable flow field f in R^{2 x H x W}
        - bilinear resample x_adv(u,v) = x(u + f_u, v + f_v)
        - loss = CW-style margin loss pushing true class down + small
          smoothness regulariser on the flow
        - Adam optimisation, 80 steps
  3. Targets per sample:
        - stadv_flow_magnitude  (mean L2 of final flow vectors; smaller = easier)
        - flipped_stadv         (1 if flow_budget version flipped, else 0)
  4. Features per (clean) sample:
        - margin   = (top logit - 2nd-best logit) on clean x
        - mean_pix = x.mean()
        - std_pix  = x.std()
        - sobel_mean = mean magnitude of Sobel gradient of x
  5. Univariate AUROC of each feature vs flipped_stadv, and Spearman-ish
     |corr| against stadv_flow_magnitude.
  6. Same analysis using FGSM (eps = 15/255) as the comparison target.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS_FGSM = 15.0 / 255.0

# StAdv hyperparams
STADV_STEPS = 80
STADV_LR = 0.05
STADV_TAU = 0.05          # smoothness regulariser weight
STADV_KAPPA = 5.0         # CW-style margin
STADV_FLOW_BUDGET = 0.5   # mean-L2 budget (in pixel units) for "flipped_stadv"
N_EVAL = 1000             # number of test samples to run StAdv on (it's slow)


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


def train(model, train_set):
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done")
    model.eval()
    return model


# ---------------- StAdv ----------------

def bilinear_resample(x, flow):
    """Resample x at displaced coordinates given by flow.

    x:    (B, C, H, W)  in [0,1]
    flow: (B, 2, H, W)  -- (f_u, f_v) pixel displacements
    returns adversarial image of same shape, bilinearly interpolated.
    """
    B, C, H, W = x.shape
    # build base grid in pixel coords
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=x.dtype),
        torch.arange(W, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    # sample at (col + f_u, row + f_v)
    src_x = xx.unsqueeze(0) + flow[:, 0]   # (B, H, W)
    src_y = yy.unsqueeze(0) + flow[:, 1]
    # normalise to [-1, 1] for grid_sample
    norm_x = 2.0 * src_x / max(W - 1, 1) - 1.0
    norm_y = 2.0 * src_y / max(H - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)   # (B, H, W, 2)
    return F.grid_sample(x, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def flow_smoothness(flow):
    """Total-variation-style smoothness penalty on the flow field.
    Encourages neighbouring pixels to move similarly (the paper's L_flow)."""
    du_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    du_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return (du_dx.pow(2).sum(dim=(1, 2, 3))
            + du_dy.pow(2).sum(dim=(1, 2, 3))).sqrt()


def cw_margin_loss(logits, y, kappa=STADV_KAPPA):
    """Push true-class logit below max other-class logit by at least kappa."""
    n_classes = logits.size(1)
    mask = F.one_hot(y, n_classes).bool()
    true_logit = logits.masked_select(mask)
    other_max = logits.masked_fill(mask, -1e9).max(dim=1).values
    # we want other_max > true_logit + kappa, so minimise max(0, true-other+kappa)
    return torch.clamp(true_logit - other_max + kappa, min=0.0)


def stadv_attack(model, x, y, steps=STADV_STEPS, lr=STADV_LR, tau=STADV_TAU):
    """Returns (x_adv, flow, flow_mag_per_sample, flipped_mask).

    flow_mag_per_sample is the mean L2 over pixels of the final flow vector
    (units: pixels) — the StAdv-equivalent of L_p magnitude.
    """
    B, C, H, W = x.shape
    flow = torch.zeros(B, 2, H, W, device=x.device, requires_grad=True)
    opt = torch.optim.Adam([flow], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        x_adv = bilinear_resample(x, flow)
        logits = model(x_adv)
        l_adv = cw_margin_loss(logits, y).sum()
        l_smooth = flow_smoothness(flow).sum()
        (l_adv + tau * l_smooth).backward()
        opt.step()
    with torch.no_grad():
        x_adv = bilinear_resample(x, flow).clamp(0, 1)
        flipped = model(x_adv).argmax(1) != y
        # mean L2 of per-pixel displacement (B,)
        flow_mag = flow.detach().pow(2).sum(dim=1).sqrt().mean(dim=(1, 2))
    return x_adv.detach(), flow.detach(), flow_mag, flipped


# ---------------- Baseline FGSM ----------------

def fgsm(model, x, y, eps=EPS_FGSM):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    x_adv = (x + eps * x.grad.sign()).clamp(0, 1).detach()
    with torch.no_grad():
        flipped = model(x_adv).argmax(1) != y
    return flipped


# ---------------- Features ----------------

SOBEL_X = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1., -2., -1.],
                        [0., 0., 0.],
                        [1., 2., 1.]]).view(1, 1, 3, 3)


def compute_features(model, x, y):
    """Return (N, 4) feature tensor: margin, mean_pix, std_pix, sobel_mean."""
    with torch.no_grad():
        logits = model(x)
        sorted_logits, _ = logits.sort(1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        mean_pix = x.mean(dim=(1, 2, 3))
        std_pix = x.std(dim=(1, 2, 3))
        kx = SOBEL_X.to(x.device); ky = SOBEL_Y.to(x.device)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        sobel_mean = (gx.pow(2) + gy.pow(2)).sqrt().mean(dim=(1, 2, 3))
    return torch.stack([margin, mean_pix, std_pix, sobel_mean], dim=1)


# ---------------- Driver ----------------

def main():
    torch.manual_seed(0); np.random.seed(0)
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("Training CNN on Fashion-MNIST...")
    model = CNN().to(DEVICE)
    t0 = time.time()
    train(model, train_set)
    print(f"  trained in {time.time()-t0:.1f}s")

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # restrict to correctly classified samples
    with torch.no_grad():
        pred = []
        for i in range(0, test_x.size(0), 512):
            pred.append(model(test_x[i:i+512]).argmax(1))
        pred = torch.cat(pred)
    correct_idx = (pred == test_y).nonzero(as_tuple=True)[0]
    correct_idx = correct_idx[:N_EVAL]
    x = test_x[correct_idx]; y = test_y[correct_idx]
    print(f"Evaluating on {x.size(0)} correctly-classified test samples")

    # features
    feats = compute_features(model, x, y).cpu().numpy()
    feat_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    # StAdv attack (batched)
    print("Running StAdv attack...")
    t0 = time.time()
    flow_mags = []; flipped_st = []
    for i in range(0, x.size(0), 64):
        _, _, fm, fl = stadv_attack(model, x[i:i+64], y[i:i+64])
        flow_mags.append(fm.cpu()); flipped_st.append(fl.cpu())
    flow_mag = torch.cat(flow_mags).numpy()
    flipped_stadv_full = torch.cat(flipped_st).numpy().astype(int)
    print(f"  StAdv done in {time.time()-t0:.1f}s "
          f"(mean flow_mag={flow_mag.mean():.3f}, "
          f"flip-rate-unbudgeted={flipped_stadv_full.mean():.3f})")

    # "flipped under flow budget" target: success AND mean-L2 flow <= budget
    flipped_stadv = ((flipped_stadv_full == 1) &
                     (flow_mag <= STADV_FLOW_BUDGET)).astype(int)

    # FGSM baseline
    print("Running FGSM baseline (eps=15/255)...")
    fg = []
    for i in range(0, x.size(0), 512):
        fg.append(fgsm(model, x[i:i+512], y[i:i+512]).cpu())
    flipped_fgsm = torch.cat(fg).numpy().astype(int)
    print(f"  FGSM flip-rate={flipped_fgsm.mean():.3f}")

    # ---------------- evaluation ----------------
    def uni_auc(feat, tgt):
        if tgt.std() == 0: return float("nan")
        a = roc_auc_score(tgt, feat)
        return max(a, 1 - a)

    def report_binary(target, name):
        print(f"\n-- univariate AUROC vs {name} (pos rate={target.mean():.3f}) --")
        for i, n in enumerate(feat_names):
            print(f"   {n:<12} AUROC = {uni_auc(feats[:, i], target):.4f}")

    print("\n========== StAdv flow magnitude (continuous) ==========")
    print(f"  budget for flipped_stadv = {STADV_FLOW_BUDGET}")
    for i, n in enumerate(feat_names):
        rho, _ = spearmanr(feats[:, i], flow_mag)
        print(f"   Spearman |corr|({n}, stadv_flow_magnitude) = {abs(rho):.4f}")

    report_binary(flipped_stadv, "flipped_stadv (budgeted)")
    report_binary(flipped_fgsm, "flipped_FGSM (eps=15/255)")

    print("\n========== head-to-head: StAdv vs FGSM ==========")
    print(f"{'feature':<12} {'AUROC_stadv':>12} {'AUROC_fgsm':>12} {'delta':>10}")
    for i, n in enumerate(feat_names):
        a_s = uni_auc(feats[:, i], flipped_stadv)
        a_f = uni_auc(feats[:, i], flipped_fgsm)
        print(f"{n:<12} {a_s:>12.4f} {a_f:>12.4f} {a_s - a_f:>+10.4f}")

    # cross-target agreement: are the same samples vulnerable to both?
    if flipped_stadv.std() and flipped_fgsm.std():
        from sklearn.metrics import matthews_corrcoef
        mcc = matthews_corrcoef(flipped_stadv, flipped_fgsm)
        print(f"\nMCC(flipped_stadv, flipped_fgsm) = {mcc:.4f}")


if __name__ == "__main__":
    main()
