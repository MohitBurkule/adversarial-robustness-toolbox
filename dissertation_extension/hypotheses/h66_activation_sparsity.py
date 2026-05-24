"""
H66: Activation sparsity predicts adversarial vulnerability.

Hypothesis:
  Per-sample fraction of active (>0) neurons at each ReLU layer (activation
  sparsity) carries signal about how vulnerable that sample is to adversarial
  perturbation. Samples with denser activation patterns (more active neurons)
  may either be more robust (more redundancy) or more vulnerable (more
  gradient signal). We test it.

Pipeline:
  1. Train small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
  2. For each test sample extract activations from three layers:
        relu1       : ReLU(c1(x))
        relu_conv2  : ReLU(c2(.)) after pooling pipeline (post-max_pool2d)
        relu_fc1    : ReLU(fc1(.))
  3. Per-sample features:
        - fraction-active (>0) per layer  -> 3 numbers
        - mean activation magnitude per layer -> 3 numbers
        - final-model margin
        - mean pixel intensity
        - std pixel intensity
     => 9 features total.
  4. Targets:
        - FGSM flip at eps=15/255
        - PGD flip at eps=15/255
        - min_eps (binary-search smallest L_inf eps that flips with FGSM)
  5. Univariate AUROC per feature vs each (binarised) target.
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
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0
N_CLASSES = 10
SEED = 0


class CNN(nn.Module):
    """Matches the architecture in diagnostic_test.py."""

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

    def forward_with_activations(self, x):
        """Return logits and the three activation tensors of interest."""
        a1 = F.relu(self.c1(x))                  # relu1
        a2 = F.relu(self.c2(a1))                 # post-relu of conv2
        a2p = F.max_pool2d(a2, 2)                # downsampled
        # interpret "relu_conv2" as the post-relu conv2 activations (a2);
        # we use a2 for sparsity stats (richer per-neuron info than the pooled
        # version).
        flat = a2p.flatten(1)
        a3 = F.relu(self.fc1(flat))              # relu_fc1
        logits = self.fc2(self.do2(a3) if self.training else a3)
        return logits, a1, a2, a3


def train_model(train_set, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done ({time.time()-t0:.1f}s)")
    return model


def extract_activation_features(model, x):
    """For a batch of inputs, return:
       fraction_active (B,3), mean_abs (B,3)
       layer order: relu1, relu_conv2, relu_fc1
    """
    model.eval()
    with torch.no_grad():
        _, a1, a2, a3 = model.forward_with_activations(x)
        feats_frac = []
        feats_mean = []
        for a in (a1, a2, a3):
            flat = a.flatten(1)
            feats_frac.append((flat > 0).float().mean(1))
            feats_mean.append(flat.abs().mean(1))
        return torch.stack(feats_frac, 1), torch.stack(feats_mean, 1)


def final_margin(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(x)
    sorted_logits, _ = logits.sort(1, descending=True)
    return sorted_logits[:, 0] - sorted_logits[:, 1], logits.argmax(1)


def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack_success(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack_success(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x_orig = x.clone().detach()
    # random start within eps-ball
    delta = (torch.rand_like(x) * 2 - 1) * eps
    adv = (x_orig + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=x.device)
    hi = torch.full((x.size(0),), eps_max, device=x.device)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def batched(fn, x, y, chunk=256):
    out = []
    for i in range(0, x.size(0), chunk):
        out.append(fn(x[i:i+chunk], y[i:i+chunk]))
    return torch.cat(out)


def auroc(scores, targets):
    """Direction-agnostic AUROC: report max(AUC, 1-AUC)."""
    s = np.asarray(scores)
    t = np.asarray(targets).astype(int)
    if t.std() == 0:
        return float("nan")
    a = roc_auc_score(t, s)
    return max(a, 1 - a)


def main():
    print(f"[H66] device={DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("[H66] training small CNN on Fashion-MNIST ...")
    t0 = time.time()
    model = train_model(train_set)
    print(f"[H66] training done in {time.time()-t0:.1f}s")

    # Stack the full test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)
    print(f"[H66] test set: {N} samples")

    # Extract activation features (batched)
    print("[H66] extracting activation features ...")
    fracs, mags = [], []
    for i in range(0, N, 512):
        f, m = extract_activation_features(model, test_x[i:i+512])
        fracs.append(f)
        mags.append(m)
    fracs = torch.cat(fracs, 0)            # (N, 3)
    mags = torch.cat(mags, 0)              # (N, 3)

    # Final-model margin + predictions
    print("[H66] computing margins ...")
    margins, preds = [], []
    for i in range(0, N, 512):
        mg, pr = final_margin(model, test_x[i:i+512])
        margins.append(mg)
        preds.append(pr)
    margins = torch.cat(margins, 0)
    preds = torch.cat(preds, 0)

    # Pixel features
    mean_pix = test_x.flatten(1).mean(1)
    std_pix = test_x.flatten(1).std(1)

    # Restrict to correctly-classified samples (consistent with diagnostic_test.py)
    correct = (preds == test_y)
    idx = correct.nonzero(as_tuple=True)[0]
    print(f"[H66] using {idx.numel()} correctly classified samples")
    x_c = test_x[idx]
    y_c = test_y[idx]
    fracs_c = fracs[idx]
    mags_c = mags[idx]
    margins_c = margins[idx]
    meanp_c = mean_pix[idx]
    stdp_c = std_pix[idx]

    # Targets
    print("[H66] FGSM attack ...")
    fgsm_flip = batched(lambda a, b: fgsm_attack_success(model, a, b), x_c, y_c)
    print(f"   FGSM positive rate: {fgsm_flip.float().mean().item():.3f}")

    print("[H66] PGD attack ...")
    pgd_flip = batched(lambda a, b: pgd_attack_success(model, a, b), x_c, y_c)
    print(f"   PGD positive rate:  {pgd_flip.float().mean().item():.3f}")

    print("[H66] min-eps binary search ...")
    min_eps = batched(lambda a, b: min_eps_to_flip(model, a, b), x_c, y_c)
    print(f"   mean min_eps:       {min_eps.mean().item():.4f}")

    # Assemble feature matrix
    feat_names = [
        "frac_active_relu1",
        "frac_active_relu_conv2",
        "frac_active_relu_fc1",
        "mean_abs_relu1",
        "mean_abs_relu_conv2",
        "mean_abs_relu_fc1",
        "final_margin",
        "mean_pix",
        "std_pix",
    ]
    feats = torch.cat([
        fracs_c,
        mags_c,
        margins_c.unsqueeze(1),
        meanp_c.unsqueeze(1),
        stdp_c.unsqueeze(1),
    ], dim=1).cpu().numpy()

    # Binarise min_eps at its median for AUROC purposes (small min_eps -> vulnerable)
    me_np = min_eps.cpu().numpy()
    me_bin = (me_np <= np.median(me_np)).astype(int)

    target_specs = [
        ("FGSM_flip",   fgsm_flip.cpu().numpy().astype(int)),
        ("PGD_flip",    pgd_flip.cpu().numpy().astype(int)),
        ("min_eps_lo",  me_bin),
    ]

    print("\n===== H66: Univariate AUROC =====")
    header = f"{'feature':<26}" + "".join(f"{n:>14}" for n, _ in target_specs)
    print(header)
    print("-" * len(header))
    for i, fname in enumerate(feat_names):
        row = f"{fname:<26}"
        for _, tvec in target_specs:
            row += f"{auroc(feats[:, i], tvec):>14.4f}"
        print(row)

    # Also report Spearman-like Pearson correlation against continuous min_eps
    print("\n===== H66: Pearson correlation with min_eps (continuous) =====")
    for i, fname in enumerate(feat_names):
        cor = np.corrcoef(feats[:, i], me_np)[0, 1]
        print(f"  corr(min_eps, {fname:<26}) = {cor:+.4f}")


if __name__ == "__main__":
    main()
