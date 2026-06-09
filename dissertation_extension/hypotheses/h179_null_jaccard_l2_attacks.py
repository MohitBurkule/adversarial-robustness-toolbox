"""
H179 - Null-Jaccard baseline + cross-norm / decision-based attack invariance.

Motivation (advisor critique, Paper 2):
  Paper 2 claims adversarial vulnerability is "structural and attack-invariant" from a high Jaccard
  overlap (~0.99) between L-inf attacks. Two objections:
    (1) Tautology: when two attacks both reach ASR ~95% on the same model, a high Jaccard is forced
        by the pigeonhole principle. The paper never reports the NULL Jaccard expected from two
        random success sets of matched size. Without it, "0.99" is uninterpretable.
    (2) L-inf only: the "attack-invariant" thesis is tested only within the L-inf white-box family.
        A genuine test needs a different norm (L2) and a different threat model (decision-based,
        labels only).

This script trains a vanilla CNN and runs four attacks spanning norms and threat models:
    - PGD-Linf   : L-inf white-box (the reference family).
    - DeepFool   : L2 white-box minimal-perturbation (Moosavi-Dezfooli 2016).
    - CW-L2      : L2 white-box, margin loss optimised in tanh space (Carlini & Wagner 2017).
    - Boundary   : DECISION-BASED black-box (labels only), random-walk toward the original from an
                   adversarial start (Brendel et al. 2018, budgeted).
For each attack pair it reports the OBSERVED Jaccard, the analytic NULL Jaccard under independent
random success sets of matched ASR, and a permutation estimate of the null. The "excess over null"
is the real evidence for structural attack-invariance.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0       # L-inf budget for PGD
L2_EPS = 2.0             # L2 budget used to threshold DeepFool / CW / Boundary "success"
N_CLASSES = 10
EVAL_N = 1000
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

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


def train(train_set, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return model


# ---- attacks: each returns boolean flip tensor (within its budget) ----

def pgd_linf_flip(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=20):
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


def deepfool_l2(model, x, y, max_iter=30, overshoot=0.02, l2_budget=L2_EPS):
    """Multiclass DeepFool (L2). Returns (flip_within_budget, l2_perturbation_norm)."""
    model.eval()
    N = x.size(0)
    x_adv = x.clone().detach()
    done = torch.zeros(N, dtype=torch.bool, device=x.device)
    for _ in range(max_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        pred = logits.argmax(1)
        flipped = pred != y
        done = done | flipped
        if done.all():
            break
        # gradient of true-class logit
        f_true = logits[torch.arange(N), y]
        g_true, = torch.autograd.grad(f_true.sum(), x_adv, retain_graph=True)
        # find closest other class (cheap: use top non-true class by current logit)
        masked = logits.detach().clone()
        masked[torch.arange(N), y] = float("-inf")
        k = masked.argmax(1)
        f_k = logits[torch.arange(N), k]
        g_k, = torch.autograd.grad(f_k.sum(), x_adv)
        with torch.no_grad():
            w = (g_k - g_true).flatten(1)
            fdiff = (f_k - f_true)
            wn = w.norm(dim=1).clamp_min(1e-8)
            r = (fdiff.abs() / (wn ** 2)).view(-1, 1) * w
            r = r.view_as(x_adv) * (1 + overshoot)
            upd = (~done).view(-1, 1, 1, 1).float()
            x_adv = (x_adv + r * upd).clamp(0, 1)
    with torch.no_grad():
        l2 = (x_adv.detach() - x).flatten(1).norm(dim=1)
        final_flip = (model(x_adv.detach()).argmax(1) != y)
        return (final_flip & (l2 <= l2_budget)), l2


def cw_l2(model, x, y, steps=80, lr=0.05, c=1.0, kappa=0.0, l2_budget=L2_EPS):
    """Carlini-Wagner L2 (tanh space, margin loss). Returns (flip_within_budget, l2_norm)."""
    model.eval()
    N = x.size(0)
    x_c = x.clamp(1e-6, 1 - 1e-6)
    w = torch.atanh(2 * x_c - 1).detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    onehot = F.one_hot(y, N_CLASSES).float()
    for _ in range(steps):
        adv = 0.5 * (torch.tanh(w) + 1)
        logits = model(adv)
        real = (onehot * logits).sum(1)
        other = ((1 - onehot) * logits - onehot * 1e4).max(1).values
        f = torch.clamp(real - other + kappa, min=0)
        l2sq = ((adv - x) ** 2).flatten(1).sum(1)
        loss = (l2sq + c * f).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        adv = 0.5 * (torch.tanh(w) + 1)
        l2 = (adv - x).flatten(1).norm(dim=1)
        flip = (model(adv).argmax(1) != y)
        return (flip & (l2 <= l2_budget)), l2


def boundary_attack(model, x, y, steps=60, l2_budget=L2_EPS, seed=0):
    """Decision-based (labels only) random-walk boundary attack, budgeted.
    Start from random noise that is misclassified, then walk toward x keeping adversarial."""
    model.eval()
    g = torch.Generator(device=x.device); g.manual_seed(seed)
    N = x.size(0)

    @torch.no_grad()
    def is_adv(z):
        return model(z).argmax(1) != y

    # init: find a misclassified start via uniform noise (few tries)
    adv = torch.rand(x.shape, generator=g, device=x.device)
    for _ in range(20):
        bad = ~is_adv(adv)
        if not bad.any():
            break
        adv = torch.where(bad.view(-1, 1, 1, 1), torch.rand(x.shape, generator=g, device=x.device), adv)
    init_adv = is_adv(adv)

    step = 0.1 * torch.ones(N, device=x.device)
    for _ in range(steps):
        with torch.no_grad():
            # move toward original
            toward = adv + 0.3 * (x - adv)
            toward = toward.clamp(0, 1)
            ok = is_adv(toward)
            adv = torch.where(ok.view(-1, 1, 1, 1), toward, adv)
            # small orthogonal random perturbation
            noise = torch.randn(x.shape, generator=g, device=x.device)
            cand = (adv + step.view(-1, 1, 1, 1) * noise).clamp(0, 1)
            ok2 = is_adv(cand)
            adv = torch.where(ok2.view(-1, 1, 1, 1), cand, adv)
    with torch.no_grad():
        l2 = (adv - x).flatten(1).norm(dim=1)
        flip = is_adv(adv) & init_adv
        return (flip & (l2 <= l2_budget)), l2


def batched_attack(fn, x, y, B=200, returns_tuple=False):
    outs = []
    for i in range(0, x.size(0), B):
        r = fn(x[i:i+B], y[i:i+B])
        outs.append((r[0] if returns_tuple else r).cpu())
    return torch.cat(outs).numpy().astype(int)


def jaccard(a, b):
    inter = ((a == 1) & (b == 1)).sum()
    union = ((a == 1) | (b == 1)).sum()
    return inter / union if union > 0 else float("nan")


def null_jaccard_analytic(pa, pb):
    inter = pa * pb
    union = pa + pb - inter
    return inter / union if union > 0 else float("nan")


def null_jaccard_perm(a, b, trials=200, seed=0):
    rng = np.random.RandomState(seed)
    na, nb, N = int(a.sum()), int(b.sum()), len(a)
    js = []
    for _ in range(trials):
        sa = np.zeros(N, int); sa[rng.choice(N, na, replace=False)] = 1
        sb = np.zeros(N, int); sb[rng.choice(N, nb, replace=False)] = 1
        js.append(jaccard(sa, sb))
    return float(np.mean(js)), float(np.std(js))


def main():
    print("=" * 74)
    print("H179 - Null-Jaccard baseline + cross-norm / decision-based attack invariance")
    print("=" * 74)
    print(f"Device={DEVICE}  Linf-eps={EPS:.4f}  L2-budget={L2_EPS}  EVAL_N={EVAL_N}")

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n--- Training vanilla CNN ---"); t0 = time.time()
    model = train(train_set); print(f"  {time.time()-t0:.1f}s")
    model.eval()
    with torch.no_grad():
        correct = (model(test_x).argmax(1) == test_y)
    idx = correct.nonzero(as_tuple=True)[0][:EVAL_N]
    x, y = test_x[idx], test_y[idx]
    print(f"  eval n={x.size(0)}")

    print("\n--- Running attacks ---")
    t0 = time.time()
    flips = {
        "PGD-Linf":  batched_attack(lambda a, b: pgd_linf_flip(model, a, b), x, y),
        "DeepFool":  batched_attack(lambda a, b: deepfool_l2(model, a, b), x, y, returns_tuple=True),
        "CW-L2":     batched_attack(lambda a, b: cw_l2(model, a, b), x, y, returns_tuple=True),
        "Boundary":  batched_attack(lambda a, b: boundary_attack(model, a, b), x, y, returns_tuple=True),
    }
    print(f"  attacks done in {time.time()-t0:.1f}s")
    for k, v in flips.items():
        print(f"    {k:<10} ASR (within budget) = {v.mean():.4f}")

    print("\n--- Pairwise Jaccard: observed vs null ---")
    print(f"  {'pair':<22} {'observed':>9} {'null(analytic)':>15} {'null(perm)':>16} {'excess':>9}")
    names = list(flips.keys())
    N = len(y)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = flips[names[i]], flips[names[j]]
            obs = jaccard(a, b)
            na = null_jaccard_analytic(a.mean(), b.mean())
            pm, ps = null_jaccard_perm(a, b)
            excess = obs - na if not (np.isnan(obs) or np.isnan(na)) else float("nan")
            print(f"  {names[i]+'-'+names[j]:<22} {obs:>9.4f} {na:>15.4f} "
                  f"{pm:>10.4f}+/-{ps:<5.4f} {excess:>+9.4f}")

    print("\n" + "=" * 74)
    print("If observed Jaccard >> null across the L-inf vs L2 vs decision-based pairs, vulnerability")
    print("is genuinely structural/attack-invariant. If observed ~ null, the original 0.99 was the")
    print("pigeonhole tautology the advisor warned about.")
    print("=" * 74)


if __name__ == "__main__":
    main()
