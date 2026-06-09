"""
H98: Variance of fc1 activations across K=20 MC-dropout forward passes (with
dropout enabled at inference) is a per-sample feature that predicts adversarial
vulnerability beyond final-output MC-dropout variance.

Pipeline:
  1. Train small CNN (matching diagnostic_test.py architecture) on Fashion-MNIST
     for 10 epochs.
  2. For each test sample: run K=20 forward passes with dropout enabled,
     capturing fc1 (128-d) activations and final logits.
     - fc1 std across K -> per-unit std (128-d) -> reduce to mean, max,
       sum-l2-norm scalars.
     - final-output MC-dropout variance (mean over classes of softmax var) as
       baseline.
  3. Add baseline features: margin, mean_pix, std_pix.
  4. Targets: FGSM flip @ eps=15/255, PGD flip @ eps=15/255, min_eps (binary
     search on FGSM).
  5. Univariate AUROC for each feature against each binary target.

Run-time only; code not executed here per instructions.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
K_MC = 20
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0


class CNN(nn.Module):
    """Matches diagnostic_test.py architecture exactly."""
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x, return_fc1=False):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        h = F.relu(self.fc1(x))
        h_drop = self.do2(h)
        logits = self.fc2(h_drop)
        if return_fc1:
            # Return post-dropout fc1 activations (these vary across MC passes)
            return logits, h_drop
        return logits


def train_model(seed, train_set, n_classes=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done")
    return model


def mc_dropout_features(model, x, K=K_MC, batch=256):
    """For each sample, run K forward passes with dropout enabled.

    Returns dict of per-sample features:
      - fc1_std_mean   : mean over 128 units of std-across-K
      - fc1_std_max    : max  over 128 units of std-across-K
      - fc1_std_l2     : L2 norm over 128 units of std-across-K
      - out_var        : mean over classes of softmax variance across K
    """
    N = x.size(0)
    fc1_std_mean = torch.zeros(N)
    fc1_std_max = torch.zeros(N)
    fc1_std_l2 = torch.zeros(N)
    out_var = torch.zeros(N)

    model.train()  # enable dropout
    with torch.no_grad():
        for i in range(0, N, batch):
            xb = x[i:i+batch].to(DEVICE)
            B = xb.size(0)
            fc1_stack = torch.zeros(K, B, 128, device=DEVICE)
            soft_stack = torch.zeros(K, B, 10, device=DEVICE)
            for k in range(K):
                logits, h = model(xb, return_fc1=True)
                fc1_stack[k] = h
                soft_stack[k] = F.softmax(logits, dim=1)
            fc1_std = fc1_stack.std(dim=0)            # (B, 128)
            soft_var = soft_stack.var(dim=0)          # (B, 10)
            fc1_std_mean[i:i+B] = fc1_std.mean(dim=1).cpu()
            fc1_std_max[i:i+B] = fc1_std.max(dim=1).values.cpu()
            fc1_std_l2[i:i+B] = fc1_std.norm(dim=1).cpu()
            out_var[i:i+B] = soft_var.mean(dim=1).cpu()
    model.eval()
    return {
        "fc1_std_mean": fc1_std_mean,
        "fc1_std_max": fc1_std_max,
        "fc1_std_l2": fc1_std_l2,
        "mc_out_var": out_var,
    }


def baseline_features(model, x, y):
    """margin (final-model), mean_pix, std_pix."""
    N = x.size(0)
    model.eval()
    margin = torch.zeros(N)
    with torch.no_grad():
        for i in range(0, N, 512):
            xb = x[i:i+512].to(DEVICE)
            logits = model(xb)
            sl, _ = logits.sort(1, descending=True)
            margin[i:i+512] = (sl[:, 0] - sl[:, 1]).cpu()
    mean_pix = x.view(N, -1).mean(dim=1)
    std_pix = x.view(N, -1).std(dim=1)
    return {"margin": margin, "mean_pix": mean_pix, "std_pix": std_pix}


def fgsm_grad(model, x, y):
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST, batch=256):
    N = x.size(0)
    out = torch.zeros(N, dtype=torch.bool)
    for i in range(0, N, batch):
        xb = x[i:i+batch].to(DEVICE)
        yb = y[i:i+batch].to(DEVICE)
        sign = fgsm_grad(model, xb, yb)
        adv = (xb + eps * sign).clamp(0, 1)
        with torch.no_grad():
            out[i:i+batch] = (model(adv).argmax(1) != yb).cpu()
    return out


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS, batch=256):
    N = x.size(0)
    out = torch.zeros(N, dtype=torch.bool)
    model.eval()
    for i in range(0, N, batch):
        xb = x[i:i+batch].to(DEVICE)
        yb = y[i:i+batch].to(DEVICE)
        adv = xb.clone().detach()
        adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
        adv = adv.clamp(0, 1)
        for _ in range(steps):
            adv.requires_grad_(True)
            loss = F.cross_entropy(model(adv), yb)
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + alpha * grad.sign()
            adv = torch.max(torch.min(adv, xb + eps), xb - eps).clamp(0, 1)
        with torch.no_grad():
            out[i:i+batch] = (model(adv).argmax(1) != yb).cpu()
    return out


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15, batch=256):
    N = x.size(0)
    out = torch.zeros(N)
    for i in range(0, N, batch):
        xb = x[i:i+batch].to(DEVICE)
        yb = y[i:i+batch].to(DEVICE)
        sign = fgsm_grad(model, xb, yb)
        lo = torch.zeros(xb.size(0), device=DEVICE)
        hi = torch.full((xb.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xb + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
            with torch.no_grad():
                flipped = model(adv).argmax(1) != yb
            hi = torch.where(flipped, mid, hi)
            lo = torch.where(flipped, lo, mid)
        out[i:i+batch] = hi.cpu()
    return out


def univariate_auroc(feature_vals, target_vals):
    """Direction-agnostic AUROC."""
    y = np.asarray(target_vals).astype(int)
    x = np.asarray(feature_vals).astype(float)
    if y.std() == 0:
        return float("nan")
    a = roc_auc_score(y, x)
    return max(a, 1 - a)


def main():
    print("##### H98: MC-dropout fc1-variance features #####")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(" training model (10 epochs)...")
    t0 = time.time()
    model = train_model(0, train_set)
    print(f"  trained in {time.time()-t0:.1f}s")

    # assemble full test tensor on CPU; move per-batch later
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))])

    # restrict to correctly classified
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, test_x.size(0), 512):
            preds.append(model(test_x[i:i+512].to(DEVICE)).argmax(1).cpu())
        preds = torch.cat(preds)
    correct = preds == test_y
    x_c, y_c = test_x[correct], test_y[correct]
    print(f" using {correct.sum().item()} correctly-classified samples")

    print(" computing MC-dropout features (K=20)...")
    t0 = time.time()
    mc_feats = mc_dropout_features(model, x_c, K=K_MC)
    print(f"  done ({time.time()-t0:.1f}s)")

    print(" computing baseline features (margin/mean_pix/std_pix)...")
    base_feats = baseline_features(model, x_c, y_c)

    print(" computing FGSM flip target...")
    fgsm_t = fgsm_flip(model, x_c, y_c)
    print(f"  FGSM positive rate = {fgsm_t.float().mean():.3f}")

    print(" computing PGD flip target...")
    pgd_t = pgd_flip(model, x_c, y_c)
    print(f"  PGD positive rate = {pgd_t.float().mean():.3f}")

    print(" computing min_eps_to_flip target (continuous)...")
    me_t = min_eps_to_flip(model, x_c, y_c)
    print(f"  mean min_eps = {me_t.mean():.4f}")
    # binarise min_eps at median for AUROC
    median_eps = me_t.median().item()
    me_bin = (me_t <= median_eps)

    all_feats = {**mc_feats, **base_feats}
    feat_names = ["fc1_std_mean", "fc1_std_max", "fc1_std_l2", "mc_out_var",
                  "margin", "mean_pix", "std_pix"]
    targets = {
        "FGSM_flip": fgsm_t.numpy(),
        "PGD_flip": pgd_t.numpy(),
        "min_eps_below_median": me_bin.numpy(),
    }

    print("\n========== Univariate AUROC ==========")
    header = f"{'feature':<18} " + " ".join(f"{t:>22}" for t in targets)
    print(header)
    for fname in feat_names:
        fv = all_feats[fname].numpy()
        row = f"{fname:<18} "
        for tname, tv in targets.items():
            a = univariate_auroc(fv, tv)
            row += f"{a:>22.4f}"
        print(row)

    # spearman-style linear corr to continuous min_eps too
    print("\n========== corr with continuous min_eps ==========")
    me_np = me_t.numpy()
    for fname in feat_names:
        fv = all_feats[fname].numpy()
        c = np.corrcoef(fv, me_np)[0, 1]
        print(f"  corr(min_eps, {fname:<18}) = {c:+.4f}")


if __name__ == "__main__":
    main()
