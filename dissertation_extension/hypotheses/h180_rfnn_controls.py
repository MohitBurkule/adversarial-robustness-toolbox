"""
H180 - Random-feature controls: is per-sample vulnerability "input geometry" or "architectural prior"?

Motivation (advisor critique, Paper 6):
  Paper 6 claims adversarial vulnerability is "primarily encoded in input-space geometry" because a
  Random-Feature Neural Net (frozen random CNN + linear readout) ranks samples almost as well as a
  fully trained CNN. The reviewer's central objection: a random CNN's frozen features still carry
  strong ARCHITECTURAL priors (convolution, locality, ReLU, pooling). "RFNN ~ CNN" therefore shows
  "input geometry + conv prior is sufficient", NOT "input geometry alone". The decisive missing
  controls are a random *MLP* feature network (no conv prior) and a raw Gaussian random projection
  (no architecture at all).

This script trains a reference CNN, then builds four predictors and measures how well each ranks
the reference CNN's PGD vulnerability (AUROC). Each predictor's score is its own logit margin,
obtained by fitting only a linear readout (logistic regression) on frozen features:

    1. Trained-CNN margin           : the reference upper bound.
    2. Random-CNN RFNN margin       : frozen random conv stack + linear readout  (conv prior).
    3. Random-MLP RFNN margin       : frozen random fully-connected stack + linear readout
                                      (NO conv prior -- the key control).
    4. Gaussian-projection margin   : fixed random Gaussian matrix on raw pixels + linear readout
                                      (NO architecture at all -- pure input geometry).
    5. Raw-pixel logistic margin    : linear readout directly on pixels (weakest baseline).

If vulnerability were truly "input geometry alone", controls 3-5 would match control 2. If the
conv prior is doing the work, control 2 >> controls 3-5.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10
EVAL_N = 1000
FEAT_TRAIN_N = 8000   # samples used to fit the linear readouts
SEED = 0


class CNN(nn.Module):
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
        x = x.flatten(1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


class RandomMLP(nn.Module):
    """Frozen random fully-connected feature extractor (no conv prior)."""
    def __init__(self, in_dim=784, hidden=2048, out=512):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, out)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = x.flatten(1)
        return F.relu(self.l2(F.relu(self.l1(x))))


def pgd_flip(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=10):
    model.eval()
    adv = (x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        with torch.no_grad():
            adv = (adv + alpha * adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def margin_from_proba(proba):
    """Top1 - Top2 of predicted class probabilities, per sample."""
    s = np.sort(proba, axis=1)[:, ::-1]
    return s[:, 0] - s[:, 1]


def fit_readout_and_margin(feat_train, y_train, feat_eval):
    """Fit multinomial logistic regression on frozen features; return eval-set margins."""
    sc = StandardScaler().fit(feat_train)
    clf = LogisticRegression(max_iter=500, C=1.0).fit(sc.transform(feat_train), y_train)
    proba = clf.predict_proba(sc.transform(feat_eval))
    return margin_from_proba(proba)


def safe_auroc(label, score):
    label = np.asarray(label).astype(int)
    if label.std() == 0:
        return float("nan")
    a = roc_auc_score(label, np.asarray(score))
    return max(a, 1 - a)


@torch.no_grad()
def extract(fn, x, batch=512):
    out = []
    for i in range(0, x.size(0), batch):
        out.append(fn(x[i:i+batch]).detach().cpu().numpy())
    return np.concatenate(out)


def main():
    print("=" * 74)
    print("H180 - Random-feature controls (conv prior vs input geometry)")
    print("=" * 74)
    print(f"Device={DEVICE}  EVAL_N={EVAL_N}  FEAT_TRAIN_N={FEAT_TRAIN_N}  EPS={EPS:.4f}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)

    # reference trained CNN
    print("\n--- Training reference CNN ---")
    t0 = time.time()
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    print(f"  {time.time()-t0:.1f}s")

    # feature-fit set (train) and eval set (correctly-classified test)
    rng = np.random.RandomState(SEED)
    ft_idx = rng.choice(len(train_set), FEAT_TRAIN_N, replace=False)
    ft_x = torch.stack([train_set[i][0] for i in ft_idx]).to(DEVICE)
    ft_y = np.array([train_set[i][1] for i in ft_idx])

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    model.eval()
    with torch.no_grad():
        correct = (model(test_x).argmax(1) == test_y)
    idx = correct.nonzero(as_tuple=True)[0][:EVAL_N]
    ex, ey = test_x[idx], test_y[idx]

    # reference vulnerability label (PGD on trained CNN)
    flips = []
    for i in range(0, ex.size(0), 256):
        flips.append(pgd_flip(model, ex[i:i+256], ey[i:i+256]).cpu())
    flip = torch.cat(flips).numpy().astype(int)
    print(f"  reference PGD ASR = {flip.mean():.4f}  (eval n={ex.size(0)})")

    # reference trained-CNN margin
    with torch.no_grad():
        s, _ = model(ex).sort(1, descending=True)
        trained_margin = (s[:, 0] - s[:, 1]).cpu().numpy()

    # frozen random feature extractors
    rand_cnn = CNN(N_CLASSES).to(DEVICE).eval()   # random init, use .features()
    rand_mlp = RandomMLP().to(DEVICE).eval()
    # Gaussian projection matrix on raw pixels
    G = torch.randn(784, 512, device=DEVICE) / np.sqrt(784)

    extractors = {
        "RandomCNN (conv prior)": lambda x: rand_cnn.features(x),
        "RandomMLP (no conv)":    lambda x: rand_mlp(x),
        "GaussProj (pure input)": lambda x: F.relu(x.flatten(1) @ G),
        "RawPixels":              lambda x: x.flatten(1),
    }

    print("\n--- AUROC of each predictor vs reference-CNN PGD vulnerability ---")
    print(f"  {'predictor':<26} {'AUROC':>8}")
    print(f"  {'TrainedCNN margin':<26} {safe_auroc(flip, -trained_margin):>8.4f}   (reference)")
    for name, fn in extractors.items():
        ftr = extract(fn, ft_x)
        fev = extract(fn, ex)
        m = fit_readout_and_margin(ftr, ft_y, fev)
        print(f"  {name:<26} {safe_auroc(flip, -m):>8.4f}")

    print("\n" + "=" * 74)
    print("If RandomMLP / GaussProj match RandomCNN -> vulnerability is input geometry.")
    print("If RandomCNN >> RandomMLP/GaussProj -> the conv ARCHITECTURAL prior does the work,")
    print("refuting the 'input geometry alone' framing (advisor's central objection to Paper 6).")
    print("=" * 74)


if __name__ == "__main__":
    main()
