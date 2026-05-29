"""
Shared library for the autonomous research campaign (500+ hypotheses).

Themes:
  A. Adversarial robustness x machine unlearning.
  B. Learning rules vs gradient descent (is vulnerability a GD artefact or data geometry?).
  C. Optical illusions: is the human "adversarial mode" the same as the machine's, and
     does training human-like confusion buy robustness?

Design goals:
  * Pure torch + sklearn, no ART.
  * All data / caches on NewVolume1 (the OS disk is ~99% full).
  * Small, fast nets so 500+ runs finish within ~2 days on one RTX 4090.
  * Everything self-contained and deterministic per seed.
"""
import os
import sys

# ---- keep ALL caches off the (full) OS disk -------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dissertation_extension/
_VOL = os.path.dirname(os.path.dirname(_REPO))                       # NewVolume1/adversarial-robustness-toolbox
_CACHE = os.path.join(_REPO, ".cache")
os.makedirs(_CACHE, exist_ok=True)
os.environ.setdefault("TORCH_HOME", os.path.join(_CACHE, "torch"))
os.environ.setdefault("HF_HOME", os.path.join(_CACHE, "hf"))
os.environ.setdefault("XDG_CACHE_HOME", _CACHE)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE, "mpl"))

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join(_REPO, "data")

# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# datasets  -> returns (Xtr, Ytr, Xte, Yte) float tensors in [0,1] on DEVICE
# ---------------------------------------------------------------------------
_DATASET_SPECS = {
    "fashion_mnist": (datasets.FashionMNIST, 1, 28, 10),
    "kmnist": (datasets.KMNIST, 1, 28, 10),
    "mnist": (datasets.MNIST, 1, 28, 10),
    "cifar10": (datasets.CIFAR10, 3, 32, 10),
    "svhn": (datasets.SVHN, 3, 32, 10),
}

_CACHE_TENSORS = {}


def dataset_meta(name):
    _, ch, sz, ncls = _DATASET_SPECS[name]
    return {"channels": ch, "size": sz, "n_classes": ncls}


def _to_tensor(ds, ch, sz):
    # ds yields PIL images; stack into a uint8 then float tensor.
    n = len(ds)
    X = torch.empty(n, ch, sz, sz, dtype=torch.float32)
    Y = torch.empty(n, dtype=torch.long)
    tf = transforms.ToTensor()  # -> [0,1], CxHxW
    for i in range(n):
        img, lab = ds[i]
        X[i] = tf(img)
        Y[i] = int(lab)
    return X, Y


def load_dataset(name, n_train=None, n_eval=2000, seed=0):
    """Return (Xtr,Ytr,Xte,Yte) on DEVICE in [0,1]. Cached in-process."""
    key = name
    if key not in _CACHE_TENSORS:
        cls, ch, sz, ncls = _DATASET_SPECS[name]
        os.makedirs(DATA_ROOT, exist_ok=True)
        if name == "svhn":
            tr = cls(DATA_ROOT, split="train", download=True)
            te = cls(DATA_ROOT, split="test", download=True)
        else:
            tr = cls(DATA_ROOT, train=True, download=True)
            te = cls(DATA_ROOT, train=False, download=True)
        Xtr, Ytr = _to_tensor(tr, ch, sz)
        Xte, Yte = _to_tensor(te, ch, sz)
        _CACHE_TENSORS[key] = (Xtr, Ytr, Xte, Yte)
    Xtr, Ytr, Xte, Yte = _CACHE_TENSORS[key]

    g = torch.Generator().manual_seed(seed)
    if n_train is not None and n_train < Xtr.size(0):
        idx = torch.randperm(Xtr.size(0), generator=g)[:n_train]
        Xtr, Ytr = Xtr[idx], Ytr[idx]
    if n_eval is not None and n_eval < Xte.size(0):
        idx = torch.randperm(Xte.size(0), generator=g)[:n_eval]
        Xte, Yte = Xte[idx], Yte[idx]
    return (Xtr.to(DEVICE), Ytr.to(DEVICE), Xte.to(DEVICE), Yte.to(DEVICE))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "elu": nn.ELU,
        "sigmoid": nn.Sigmoid, "softplus": nn.Softplus}


class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, size=28, n_classes=10, width=32, act="relu", bn=True):
        super().__init__()
        A = ACTS[act]
        def block(i, o):
            layers = [nn.Conv2d(i, o, 3, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(o))
            layers += [A(), nn.MaxPool2d(2)]
            return layers
        self.features = nn.Sequential(
            *block(in_ch, width), *block(width, width * 2), *block(width * 2, width * 4))
        feat = size // 8
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width * 4 * feat * feat, 256), A(),
                                  nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class MLP(nn.Module):
    def __init__(self, in_ch=1, size=28, n_classes=10, width=512, depth=3, act="relu"):
        super().__init__()
        A = ACTS[act]
        d = in_ch * size * size
        layers = [nn.Flatten()]
        for _ in range(depth):
            layers += [nn.Linear(d, width), A()]
            d = width
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RandomFeatureNet(nn.Module):
    """Frozen random conv features + trained linear readout (no GD through features)."""
    def __init__(self, in_ch=1, size=28, n_classes=10, width=64, act="relu", seed=0):
        super().__init__()
        A = ACTS[act]
        g = torch.Generator().manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), A(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), A(), nn.MaxPool2d(2))
        for p in self.features.parameters():
            p.requires_grad_(False)
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
        feat = size // 4
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width * 2 * feat * feat, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(arch, meta, **kw):
    ch, sz, ncls = meta["channels"], meta["size"], meta["n_classes"]
    if arch == "cnn":
        return SmallCNN(ch, sz, ncls, width=kw.get("width", 32), act=kw.get("act", "relu"),
                        bn=kw.get("bn", True)).to(DEVICE)
    if arch == "mlp":
        return MLP(ch, sz, ncls, width=kw.get("width", 512), depth=kw.get("depth", 3),
                   act=kw.get("act", "relu")).to(DEVICE)
    if arch == "rfnn":
        return RandomFeatureNet(ch, sz, ncls, width=kw.get("width", 64),
                                act=kw.get("act", "relu"), seed=kw.get("seed", 0)).to(DEVICE)
    raise ValueError(arch)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def make_optimizer(model, name, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    if name == "sgd_highwd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-3)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    raise ValueError(name)


def _mixup(x, y, ncls, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    xm = lam * x + (1 - lam) * x[perm]
    return xm, y, y[perm], lam


def train_model(model, Xtr, Ytr, epochs=8, batch=128, opt="sgd", lr=0.05,
                label_smooth=0.0, mixup=False, sample_weights=None, ncls=10,
                adv_train=False, adv_eps=0.1, adv_steps=7, verbose=False):
    opt_ = make_optimizer(model, opt, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            if adv_train:
                xb = pgd(model, xb, yb, eps=adv_eps, steps=adv_steps, alpha=2.5 * adv_eps / adv_steps)
            opt_.zero_grad()
            if mixup:
                xm, ya, yb2, lam = _mixup(xb, yb, ncls)
                out = model(xm)
                loss = lam * F.cross_entropy(out, ya, label_smoothing=label_smooth) \
                    + (1 - lam) * F.cross_entropy(out, yb2, label_smoothing=label_smooth)
            else:
                out = model(xb)
                if sample_weights is not None:
                    w = sample_weights[idx]
                    loss = (F.cross_entropy(out, yb, reduction="none", label_smoothing=label_smooth) * w).mean()
                else:
                    loss = F.cross_entropy(out, yb, label_smoothing=label_smooth)
            loss.backward()
            opt_.step()
        sched.step()
        if verbose:
            print(f"    epoch {ep+1}/{epochs} loss={loss.item():.3f}")
    model.eval()
    return model


def forward_forward_train(meta, Xtr, Ytr, Xte, Yte, epochs=8, width=512, lr=0.03):
    """A compact Forward-Forward (Hinton 2022) MLP: no backprop / no end-to-end GD.

    Each layer trained locally to push goodness (sum of squares) high on
    correctly-labelled inputs and low on wrong-labelled inputs. Label is
    overlaid on the first few input pixels. Returns a predict() closure.
    """
    ch, sz, ncls = meta["channels"], meta["size"], meta["n_classes"]
    d = ch * sz * sz

    def overlay(x, y):
        xf = x.flatten(1).clone()
        xf[:, :ncls] = 0.0
        xf[torch.arange(xf.size(0)), y] = xf.max().item()
        return xf

    class Layer(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.lin = nn.Linear(i, o)
            self.opt = torch.optim.Adam(self.parameters(), lr=lr)

        def forward(self, x):
            xn = x / (x.norm(dim=1, keepdim=True) + 1e-4)
            return F.relu(self.lin(xn))

        def train_step(self, xpos, xneg, thresh=2.0):
            gp = self.forward(xpos).pow(2).mean(1)
            gn = self.forward(xneg).pow(2).mean(1)
            loss = F.softplus(torch.cat([-gp + thresh, gn - thresh])).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            return self.forward(xpos).detach(), self.forward(xneg).detach()

    L1 = Layer(d, width).to(DEVICE)
    L2 = Layer(width, width).to(DEVICE)
    layers = [L1, L2]
    n = Xtr.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            x, y = Xtr[idx], Ytr[idx]
            yneg = (y + torch.randint(1, ncls, y.shape, device=DEVICE)) % ncls
            hpos, hneg = overlay(x, y), overlay(x, yneg)
            for L in layers:
                hpos, hneg = L.train_step(hpos, hneg)

    def goodness_for_label(x, lab):
        h = overlay(x, torch.full((x.size(0),), lab, device=DEVICE, dtype=torch.long))
        g = 0.0
        for L in layers:
            h = L.forward(h)
            g = g + h.pow(2).mean(1)
        return g

    def predict_logits(x):
        gs = torch.stack([goodness_for_label(x, c) for c in range(ncls)], dim=1)
        return gs  # use goodness as logits
    return predict_logits


# ---------------------------------------------------------------------------
# attacks  (operate in [0,1]; clamp)
# ---------------------------------------------------------------------------
def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    g, = torch.autograd.grad(loss, x)
    return (x + eps * g.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps, steps=10, alpha=None, random_start=True):
    if alpha is None:
        alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        xa = xa + torch.empty_like(xa).uniform_(-eps, eps)
        xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def attack_success(model, X, Y, attack="pgd", eps=0.1, steps=10, batch=256):
    """Fraction of *originally-correct* samples flipped, plus per-sample flip vector
    (over originally-correct samples) and their indices."""
    model.eval()
    flips, corr_mask = [], []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        with torch.no_grad():
            correct = model(x).argmax(1) == y
        if attack == "fgsm":
            xa = fgsm(model, x, y, eps)
        else:
            xa = pgd(model, x, y, eps, steps)
        with torch.no_grad():
            flipped = model(xa).argmax(1) != y
        flips.append(flipped.cpu())
        corr_mask.append(correct.cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr_mask).numpy().astype(bool)
    asr = float(flips[corr].mean()) if corr.sum() > 0 else float("nan")
    return {"asr": asr, "flips": flips, "correct": corr}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def logits_and_acc(model, X, Y, batch=512):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(model(X[i:i + batch]).cpu())
    logits = torch.cat(outs)
    acc = float((logits.argmax(1) == Y.cpu()).float().mean())
    return logits, acc


def margin_of(logits, Y):
    """Correct-class logit minus max other-class logit."""
    Y = Y.cpu()
    correct = logits.gather(1, Y[:, None]).squeeze(1)
    tmp = logits.clone()
    tmp[torch.arange(tmp.size(0)), Y] = -1e9
    other = tmp.max(1).values
    return (correct - other).numpy()


def safe_auroc(label, score):
    label = np.asarray(label).astype(int)
    if roc_auc_score is None or label.min() == label.max():
        return float("nan")
    try:
        return float(roc_auc_score(label, score))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# machine unlearning methods
# ---------------------------------------------------------------------------
def unlearn(model, Xr, Yr, Xf, Yf, method, meta, epochs=3, lr=0.01):
    """Apply an unlearning method in-place; returns the (same) model.

    Xr/Yr = retain set, Xf/Yf = forget set.
    """
    ncls = meta["n_classes"]
    model.train()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9)
    n = Xr.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xr.device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xr, yr = Xr[idx], Yr[idx]
            opt.zero_grad()
            if method == "finetune_retain":
                loss = F.cross_entropy(model(xr), yr)
            elif method == "neggrad":
                # gradient ascent on forget only
                fi = torch.randint(0, Xf.size(0), (min(128, Xf.size(0)),), device=Xf.device)
                loss = -F.cross_entropy(model(Xf[fi]), Yf[fi])
            elif method == "neggrad_plus":
                fi = torch.randint(0, Xf.size(0), (min(128, Xf.size(0)),), device=Xf.device)
                loss = F.cross_entropy(model(xr), yr) - 0.5 * F.cross_entropy(model(Xf[fi]), Yf[fi])
            elif method == "random_relabel":
                fi = torch.randint(0, Xf.size(0), (min(128, Xf.size(0)),), device=Xf.device)
                rnd = torch.randint(0, ncls, (fi.size(0),), device=Xf.device)
                loss = F.cross_entropy(model(xr), yr) + F.cross_entropy(model(Xf[fi]), rnd)
            else:
                raise ValueError(method)
            loss.backward()
            opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# optical illusions  (procedural; pure torch; grayscale by default)
# ---------------------------------------------------------------------------
def _canvas(sz, val=0.5):
    return torch.full((sz, sz), float(val))


def _disk(img, cy, cx, r, val):
    sz = img.size(0)
    yy, xx = torch.meshgrid(torch.arange(sz), torch.arange(sz), indexing="ij")
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    img[mask] = val
    return img


def _rect(img, y0, x0, y1, x1, val):
    img[y0:y1, x0:x1] = val
    return img


def gen_illusion(kind, label_scheme="physical", sz=32, seed=0, strength=None):
    """Return (image[1,sz,sz] in [0,1], physical_label, human_label).

    label_scheme controls which label gets returned as the *training* target.
    For each illusion the binary task is set up so that the physically-correct
    answer can diverge from the human-perceived answer.
    """
    rng = np.random.RandomState(seed)
    if strength is None:
        strength = rng.uniform(0.4, 1.0)

    img = _canvas(sz, 0.5)
    physical = 0
    human = 0

    if kind == "ebbinghaus":
        # two equal central disks; one surrounded by big, one by small inducers.
        # physical: equal (label by tiny random jitter); human: left looks bigger/smaller.
        r = sz // 10
        jitter = rng.choice([-1, 0, 1])
        rl, rr = r + jitter, r - jitter
        _disk(img, sz // 2, sz // 4, rl, 0.15)
        _disk(img, sz // 2, 3 * sz // 4, rr, 0.15)
        big = int(round(strength * sz / 14)) + 2
        small = 2
        for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            _disk(img, int(sz // 2 + 2.3 * rl * np.sin(ang)), int(sz // 4 + 2.3 * rl * np.cos(ang)), small, 0.15)
        for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            _disk(img, int(sz // 2 + 2.6 * rr * np.sin(ang)), int(3 * sz // 4 + 2.6 * rr * np.cos(ang)), big, 0.15)
        physical = int(rl >= rr)            # is left disk physically >= right
        human = 1                            # small-inducer (left) looks bigger to humans

    elif kind == "muller_lyer":
        # two equal horizontal lines; arrowheads in vs out. human: one looks longer.
        y0, y1 = sz // 3, 2 * sz // 3
        Lp = sz // 2
        jitter = rng.choice([-1, 0, 1])
        l_left = Lp + jitter
        x0 = (sz - l_left) // 2
        _rect(img, y0, x0, y0 + 1, x0 + l_left, 0.1)
        _rect(img, y1, (sz - Lp) // 2, y1 + 1, (sz - Lp) // 2 + Lp, 0.1)
        # arrows: top fins-out (looks longer), bottom fins-in
        a = max(2, int(strength * sz / 8))
        for xx, yy, d in [(x0, y0, 1), (x0 + l_left, y0, -1)]:
            for k in range(a):
                if 0 <= yy - k < sz and 0 <= xx + d * k < sz:
                    img[yy - k, xx + d * k] = 0.1
                if 0 <= yy + k < sz and 0 <= xx + d * k < sz:
                    img[yy + k, xx + d * k] = 0.1
        physical = int(l_left >= Lp)
        human = 1

    elif kind == "ponzo":
        # two equal horizontal bars between converging rails; upper looks longer.
        for t in range(sz):
            w = int((sz // 2) * (1 - 0.6 * t / sz))
            c = sz // 2
            if 0 <= c - w < sz:
                img[t, max(0, c - w)] = 0.1
            if 0 <= c + w < sz:
                img[t, min(sz - 1, c + w)] = 0.1
        bar = sz // 3
        jitter = rng.choice([-1, 0, 1])
        _rect(img, sz // 4, (sz - bar) // 2, sz // 4 + 1, (sz - bar) // 2 + bar + jitter, 0.05)
        _rect(img, 3 * sz // 4, (sz - bar) // 2, 3 * sz // 4 + 1, (sz - bar) // 2 + bar, 0.05)
        physical = int((bar + jitter) >= bar)   # is top bar physically >= bottom
        human = 1                                # top looks longer

    elif kind == "brightness_contrast":
        # two equal-grey patches on dark vs light surrounds; one looks brighter.
        left_bg, right_bg = 0.15, 0.85
        _rect(img, 0, 0, sz, sz // 2, left_bg)
        _rect(img, 0, sz // 2, sz, sz, right_bg)
        patch = 0.5 + rng.choice([-1, 0, 1]) * 0.02
        ps = sz // 6
        _rect(img, sz // 2 - ps, sz // 4 - ps, sz // 2 + ps, sz // 4 + ps, patch)
        _rect(img, sz // 2 - ps, 3 * sz // 4 - ps, sz // 2 + ps, 3 * sz // 4 + ps, 0.5)
        physical = int(patch >= 0.5)        # is left patch physically brighter
        human = 1                            # patch on dark bg looks brighter

    elif kind == "delboeuf":
        r = sz // 8
        jitter = rng.choice([-1, 0, 1])
        _disk(img, sz // 2, sz // 4, r + jitter, 0.15)
        _disk(img, sz // 2, 3 * sz // 4, r, 0.15)
        # left tight ring (looks bigger), right far ring (looks smaller)
        _ring(img, sz // 2, sz // 4, int(r * 1.3))
        _ring(img, sz // 2, 3 * sz // 4, int(r * 2.2))
        physical = int((r + jitter) >= r)
        human = 1

    elif kind == "vertical_horizontal":
        # an inverted-T: vertical and horizontal equal lines; vertical looks longer.
        Lh = sz // 2
        jitter = rng.choice([-1, 0, 1])
        Lv = Lh + jitter
        cx = sz // 2
        _rect(img, 3 * sz // 4, (sz - Lh) // 2, 3 * sz // 4 + 1, (sz - Lh) // 2 + Lh, 0.1)
        img[3 * sz // 4 - Lv:3 * sz // 4, cx] = 0.1
        physical = int(Lv >= Lh)
        human = 1                            # vertical looks longer

    else:
        raise ValueError(kind)

    lab = physical if label_scheme == "physical" else human
    out = img.clamp(0, 1).unsqueeze(0)
    return out, physical, human, lab


def _ring(img, cy, cx, r, val=0.15, thick=1):
    sz = img.size(0)
    yy, xx = torch.meshgrid(torch.arange(sz), torch.arange(sz), indexing="ij")
    d = torch.sqrt(((yy - cy) ** 2 + (xx - cx) ** 2).float())
    mask = (d >= r - thick) & (d <= r + thick)
    img[mask] = val
    return img


ILLUSION_KINDS = ["ebbinghaus", "muller_lyer", "ponzo", "brightness_contrast",
                  "delboeuf", "vertical_horizontal"]


def make_illusion_dataset(kind, n=3000, label_scheme="physical", sz=32, seed=0):
    X = torch.empty(n, 1, sz, sz)
    PHY = torch.empty(n, dtype=torch.long)
    HUM = torch.empty(n, dtype=torch.long)
    LAB = torch.empty(n, dtype=torch.long)
    for i in range(n):
        img, phy, hum, lab = gen_illusion(kind, label_scheme, sz, seed=seed * 100000 + i)
        X[i], PHY[i], HUM[i], LAB[i] = img, phy, hum, lab
    return X.to(DEVICE), PHY.to(DEVICE), HUM.to(DEVICE), LAB.to(DEVICE)
