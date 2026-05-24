"""
Hypothesis H08: Features from a random-initialised CNN (no training) used as
inputs to a logistic regression predict adversarial vulnerability.

Background: We previously found that the fraction-correct of K untrained CNNs
weakly predicts adversarial vulnerability (AUROC ~0.72 on Imagenette), but the
signal was largely image statistics. Here we replace the argmax-fraction signal
with the actual penultimate-layer activations of ONE random CNN (PCA-reduced),
and ask: do random conv features add over (a) margin, (b) hand-crafted image
stats?

Pipeline:
  1. Train victim CNN on Fashion-MNIST (10 epochs, same small CNN as
     dissertation_extension/diagnostic_test.py).
  2. Build ONE untrained random CNN with the same architecture, different seed.
     Extract 128-d penultimate activations on test samples; PCA to 16-d.
  3. Compute hand-crafted scalars: victim_margin, mean_pix, std_pix, sobel_mean.
  4. Targets:
        - flipped_FGSM  (eps = 15/255, victim)
        - flipped_PGD   (eps = 15/255, 10 iters, victim)
        - flipped_FGSM_transfer (surrogate trained with seed=2)
  5. Multivariate AUROC for:
        (a) margin alone
        (b) random_conv_features alone (16-d logistic regression)
        (c) margin + random_conv_features
        (d) margin + image_stats
        (e) margin + random_conv_features + image_stats
  6. Print headline question: does random_conv_features add over hand-crafted
     stats? Over margin?

Run:
  python h08_random_conv_features.py
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
EPS = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
PCA_DIM = 16
PGD_STEPS = 10
PGD_ALPHA = EPS / 4


class CNN(nn.Module):
    """Small CNN matching diagnostic_test.py (penultimate layer = 128-d fc1)."""
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def features(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        h = self.features(x)
        h = self.do2(h)
        return self.fc2(h)


def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_attack(model, x, y, eps=EPS):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x0 = x.clone().detach()
    adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def transfer_fgsm(target_model, surrogate_model, x, y, eps=EPS):
    sign = fgsm_grad(surrogate_model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (target_model(adv).argmax(1) != y)


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
    model.eval()
    return model


def batched(fn, x, *args, bs=512):
    out = []
    for i in range(0, x.size(0), bs):
        out.append(fn(x[i:i+bs], *(a[i:i+bs] for a in args)))
    return torch.cat(out)


def extract_features(model, x, bs=512):
    feats = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            feats.append(model.features(x[i:i+bs]).cpu())
    return torch.cat(feats).numpy()


def get_logits(model, x, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i+bs]))
    return torch.cat(out)


def sobel_mean(x):
    """Mean absolute Sobel response per image; x: (N,1,H,W) on DEVICE."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    with torch.no_grad():
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        mag = (gx ** 2 + gy ** 2).sqrt()
    return mag.mean(dim=(1, 2, 3))


def auc_logreg(X, y):
    """Fit logistic regression, return in-sample AUROC."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=2000).fit(Xs, y)
    return roc_auc_score(y, lr.predict_proba(Xs)[:, 1])


def main():
    print(f"device: {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True,  download=True, transform=tf)
    test_set  = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("training victim (seed=0)...")
    t0 = time.time()
    victim = train_model(seed=0, train_set=train_set)
    print(f"  done in {time.time()-t0:.1f}s")

    print("training surrogate for transfer attack (seed=2)...")
    t0 = time.time()
    surrogate = train_model(seed=2, train_set=train_set)
    print(f"  done in {time.time()-t0:.1f}s")

    print("building random-init CNN (seed=999, untrained)...")
    torch.manual_seed(999)
    random_cnn = CNN(10).to(DEVICE)
    random_cnn.eval()

    # ---------- features ----------
    print("extracting random-CNN penultimate features...")
    rcf_full = extract_features(random_cnn, test_x)  # (N, 128)
    print(f"  shape {rcf_full.shape}")

    # victim logits/margin/pred
    victim_logits = get_logits(victim, test_x)
    sorted_l, _ = victim_logits.sort(1, descending=True)
    margin = (sorted_l[:, 0] - sorted_l[:, 1]).detach().cpu().numpy()
    victim_pred = victim_logits.argmax(1)

    # image stats
    mean_pix = test_x.mean(dim=(1, 2, 3)).cpu().numpy()
    std_pix = test_x.std(dim=(1, 2, 3)).cpu().numpy()
    sob = sobel_mean(test_x).cpu().numpy()

    # ---------- targets (restrict to correctly classified) ----------
    correct = (victim_pred == test_y)
    idx = correct.nonzero(as_tuple=True)[0]
    print(f"victim accuracy on test: {correct.float().mean().item():.4f}  "
          f"using {idx.numel()} correctly classified samples")

    xc = test_x[idx]
    yc = test_y[idx]

    print("attack: FGSM (victim)...")
    flip_fgsm = batched(lambda a, b: fgsm_attack(victim, a, b), xc, yc).cpu().numpy().astype(int)
    print(f"  flip rate: {flip_fgsm.mean():.4f}")

    print("attack: PGD (victim)...")
    flip_pgd = batched(lambda a, b: pgd_attack(victim, a, b), xc, yc).cpu().numpy().astype(int)
    print(f"  flip rate: {flip_pgd.mean():.4f}")

    print("attack: FGSM transfer (surrogate->victim)...")
    flip_tr = batched(lambda a, b: transfer_fgsm(victim, surrogate, a, b),
                      xc, yc).cpu().numpy().astype(int)
    print(f"  flip rate: {flip_tr.mean():.4f}")

    # restrict features to correct subset
    idx_np = idx.cpu().numpy()
    rcf = rcf_full[idx_np]
    margin_c = margin[idx_np]
    mean_c = mean_pix[idx_np]
    std_c = std_pix[idx_np]
    sob_c = sob[idx_np]

    # PCA on random conv features
    print(f"PCA: 128 -> {PCA_DIM}")
    pca = PCA(n_components=PCA_DIM, random_state=0)
    rcf_pca = pca.fit_transform(rcf)
    evr = pca.explained_variance_ratio_.sum()
    print(f"  cumulative explained variance: {evr:.4f}")

    # ---------- feature blocks ----------
    F_margin = margin_c.reshape(-1, 1)
    F_rcf = rcf_pca
    F_stats = np.stack([mean_c, std_c, sob_c], axis=1)
    F_m_rcf = np.concatenate([F_margin, F_rcf], axis=1)
    F_m_stats = np.concatenate([F_margin, F_stats], axis=1)
    F_all = np.concatenate([F_margin, F_rcf, F_stats], axis=1)

    blocks = [
        ("margin",                    F_margin),
        ("random_conv_features",      F_rcf),
        ("margin + rcf",              F_m_rcf),
        ("margin + image_stats",      F_m_stats),
        ("margin + rcf + img_stats",  F_all),
    ]

    targets = [
        ("flipped_FGSM",          flip_fgsm),
        ("flipped_PGD",           flip_pgd),
        ("flipped_FGSM_transfer", flip_tr),
    ]

    results = {}
    for t_name, y in targets:
        print(f"\n========== target: {t_name}  (pos rate = {y.mean():.3f}) ==========")
        if y.std() == 0:
            print("  degenerate; skipping")
            continue
        results[t_name] = {}
        for b_name, X in blocks:
            try:
                auc = auc_logreg(X, y)
            except Exception as e:
                auc = float("nan")
                print(f"  {b_name}: error {e}")
                continue
            results[t_name][b_name] = auc
            print(f"  AUROC  {b_name:<28}  {auc:.4f}")

    # ---------- headline deltas ----------
    print("\n=========== HEADLINE: does random_conv_features add? ===========")
    print(f"{'target':<24} {'+rcf over margin':>18} {'+rcf over m+stats':>20}")
    for t_name, _ in targets:
        if t_name not in results:
            continue
        r = results[t_name]
        d1 = r.get("margin + rcf", float('nan')) - r.get("margin", float('nan'))
        d2 = (r.get("margin + rcf + img_stats", float('nan'))
              - r.get("margin + image_stats", float('nan')))
        print(f"{t_name:<24} {d1:>+18.4f} {d2:>+20.4f}")

    print("\nInterpretation:")
    print(" - column 1: gain from adding 16-d random CNN features to margin baseline.")
    print(" - column 2: gain from adding random CNN features on top of margin AND")
    print("   hand-engineered image stats (mean/std/sobel). If this is ~0, then")
    print("   random conv features carry no information beyond crude image stats.")
    print(" - Positive deltas support H08; near-zero / negative reject it.")

    print("\nCAVEATS:")
    print(" - In-sample AUROC (no held-out split). 16-d features on ~9k samples is")
    print("   not heavily over-parameterised but small positive deltas could still")
    print("   be optimism; for a publication report cross-validated AUROC.")
    print(" - Single random seed for the untrained CNN; results may vary.")
    print(" - PCA fit on test features (in-sample); for a held-out evaluation, PCA")
    print("   should be fit on a training split.")
    print(" - Fashion-MNIST only (28x28, 1 channel) — generalisation to Imagenette")
    print("   not tested here.")
    print(" - Only 'correctly classified' samples are scored; biases by accuracy.")


if __name__ == "__main__":
    main()
