"""
H174 - AutoAttack-style gold-standard vulnerability labels.

Motivation (advisor meta-review, the single most important missing experiment):
  Every robustness/vulnerability claim in the dissertation extension rests on PGD-10/PGD-20 with
  the same hyperparameters used during training -- exactly the configuration Carlini & Wagner and
  Croce & Hein (AutoAttack) warn against. The whole "margin dominates per-sample vulnerability"
  story must be re-checked against a strong, parameter-free ensemble attack.

We implement, in pure PyTorch (matching the rest of the codebase, which does not depend on ART),
an AutoAttack-style ensemble of three complementary attacks evaluated at fixed L-inf eps:
    - APGD-CE   : Auto-PGD with cross-entropy loss, momentum + adaptive step-size halving
                  with checkpoint conditions (Croce & Hein, ICML 2020).
    - APGD-DLR  : Auto-PGD with the Difference-of-Logits-Ratio loss (scale-invariant, defeats
                  gradient masking / loss saturation that CE suffers from).
    - Square    : a query-limited random-search black-box attack (no gradients), to catch
                  samples where the white-box gradient is masked.
A sample counts as flipped under AutoAttack if ANY of the three attacks flips it (the standard
AA "robust accuracy = survives all attacks" definition).

We then compare, on EVAL_N correctly-classified test samples:
    - per-attack ASR and the ensemble (AA) ASR;
    - margin AUROC computed against the PGD-10 label vs against the AA label;
    - the over-estimate of robustness by PGD-10 alone (samples PGD misses but AA flips).

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def pgd_train_attack(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=7):
    model.eval()
    adv = (x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        with torch.no_grad():
            adv = (adv + alpha * adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return adv.detach()


def train(train_set, adversarial, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if adversarial:
                x = pgd_train_attack(model, x, y); model.train()
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return model


# ---------------------------------------------------------------------------
# Attacks for evaluation (return boolean flip tensors)
# ---------------------------------------------------------------------------

def pgd10_flip(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=10):
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


def dlr_loss(logits, y):
    """Difference of Logits Ratio loss (Croce & Hein 2020). Higher = more adversarial."""
    s, idx = logits.sort(dim=1, descending=True)
    N = logits.size(0)
    correct = logits[torch.arange(N), y]
    top1, top2, top3 = s[:, 0], s[:, 1], s[:, 2]
    # if correct class is the top, use top2 as the competitor; else top1
    competitor = torch.where(idx[:, 0] == y, top2, top1)
    denom = (top1 - top3).clamp_min(1e-12)
    return -(correct - competitor) / denom  # maximise -> push correct below competitor


def apgd(model, x, y, loss_type, eps=EPS, n_iter=50):
    """Auto-PGD (Croce & Hein 2020): momentum step + adaptive step-size halving on
    checkpointed success conditions. Returns boolean flip tensor.

    loss_type in {"ce","dlr"}. Maximises the chosen loss within the L-inf eps ball.
    """
    model.eval()
    N = x.size(0)
    x = x.detach()
    # checkpoints (fractions of n_iter) per the paper
    cps = [0]
    p = [0.0, 0.22]
    while p[-1] <= 1.0:
        p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
    checkpoints = sorted(set(int(round(pj * n_iter)) for pj in p if pj <= 1.0))

    def loss_fn(logits):
        if loss_type == "ce":
            return F.cross_entropy(logits, y, reduction="none")
        return dlr_loss(logits, y)

    step = 2.0 * eps * torch.ones(N, device=x.device)  # per-sample step size
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    x_adv = x_adv.detach().requires_grad_(True)
    with torch.enable_grad():
        l = loss_fn(model(x_adv)).sum()
    grad, = torch.autograd.grad(l, x_adv)

    x_best = x_adv.detach().clone()
    with torch.no_grad():
        loss_indiv = loss_fn(model(x_best))
    loss_best = loss_indiv.detach().clone()

    x_prev = x_adv.detach().clone()
    succ_since = torch.zeros(N, device=x.device)   # count of improved steps since last checkpoint
    step_at_cp = step.clone()
    loss_best_at_cp = loss_best.clone()
    alpha_mom = 0.75
    next_cp = 1

    for i in range(1, n_iter + 1):
        with torch.no_grad():
            g = grad.detach()
            x_new = x_adv.detach() + step.view(-1, 1, 1, 1) * g.sign()
            x_new = x_new.clamp(x - eps, x + eps).clamp(0, 1)
            # momentum term
            z = x_adv.detach() + alpha_mom * (x_new - x_adv.detach()) + (1 - alpha_mom) * (x_adv.detach() - x_prev)
            z = z.clamp(x - eps, x + eps).clamp(0, 1)
            x_prev = x_adv.detach().clone()
            x_adv = z.detach()
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            li = loss_fn(model(x_adv))
            l = li.sum()
        grad, = torch.autograd.grad(l, x_adv)
        with torch.no_grad():
            improved = li.detach() > loss_best
            succ_since += improved.float()
            better = li.detach() > loss_best
            x_best = torch.where(better.view(-1, 1, 1, 1), x_adv.detach(), x_best)
            loss_best = torch.where(better, li.detach(), loss_best)

        # checkpoint: halve step where progress stalled
        if next_cp < len(checkpoints) and i == checkpoints[next_cp]:
            with torch.no_grad():
                period = checkpoints[next_cp] - checkpoints[next_cp - 1]
                cond1 = succ_since < (0.75 * period)
                cond2 = (step_at_cp == step) & (loss_best_at_cp >= loss_best)
                halve = cond1 | cond2
                step = torch.where(halve, step / 2.0, step)
                x_adv = x_best.clone().detach()
                step_at_cp = step.clone()
                loss_best_at_cp = loss_best.clone()
                succ_since = torch.zeros(N, device=x.device)
            next_cp += 1

    with torch.no_grad():
        return (model(x_best).argmax(1) != y)


def square_attack(model, x, y, eps=EPS, n_queries=200, p_init=0.3):
    """Square Attack (Andriushchenko et al. 2020), L-inf, simplified square-shaped random search.
    Black-box (logit-margin loss, no gradients). Returns boolean flip tensor."""
    model.eval()
    N, C, H, W = x.shape
    x = x.detach()

    def margin_loss(imgs):
        with torch.no_grad():
            logits = model(imgs)
        N_ = imgs.size(0)
        correct = logits[torch.arange(N_), y]
        masked = logits.clone()
        masked[torch.arange(N_), y] = float("-inf")
        other = masked.max(1).values
        return correct - other  # minimise; <0 means flipped

    # init: vertical-stripe +-eps perturbation
    delta = (torch.randint(0, 2, (N, C, 1, W), device=x.device).float() * 2 - 1) * eps
    x_adv = (x + delta).clamp(0, 1)
    loss = margin_loss(x_adv)

    for q in range(n_queries):
        p = p_init * max(1.0 - q / n_queries, 0.1)
        s = max(int(round((p * H * W) ** 0.5)), 1)
        s = min(s, H - 1) if H > 1 else 1
        new = x_adv.clone()
        for b in range(N):
            if loss[b] < 0:  # already flipped, skip
                continue
            r = np.random.randint(0, H - s + 1)
            c = np.random.randint(0, W - s + 1)
            ch = np.random.randint(0, C)
            sign = (torch.randint(0, 2, (1,)).item() * 2 - 1)
            patch = x[b:b+1, ch, r:r+s, c:c+s] + sign * eps
            new[b, ch, r:r+s, c:c+s] = patch.clamp(x[b, ch, r:r+s, c:c+s] - eps,
                                                    x[b, ch, r:r+s, c:c+s] + eps)
        new = new.clamp(0, 1)
        new_loss = margin_loss(new)
        improve = new_loss < loss
        x_adv = torch.where(improve.view(-1, 1, 1, 1), new, x_adv)
        loss = torch.where(improve, new_loss, loss)
        if (loss < 0).all():
            break
    with torch.no_grad():
        return (model(x_adv).argmax(1) != y)


def safe_auroc(label, score):
    label = np.asarray(label).astype(int)
    if label.std() == 0:
        return float("nan")
    a = roc_auc_score(label, np.asarray(score))
    return max(a, 1 - a)


def eval_model(name, model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        correct = (model(test_x).argmax(1) == test_y)
    idx = correct.nonzero(as_tuple=True)[0][:EVAL_N]
    x, y = test_x[idx], test_y[idx]
    N = x.size(0)
    with torch.no_grad():
        logits = model(x)
        s, _ = logits.sort(1, descending=True)
        margin = (s[:, 0] - s[:, 1]).cpu().numpy()

    pgd, ace, adlr, sq = [], [], [], []
    B = 250
    for i in range(0, N, B):
        bx, by = x[i:i+B], y[i:i+B]
        pgd.append(pgd10_flip(model, bx, by).cpu())
        ace.append(apgd(model, bx, by, "ce").cpu())
        adlr.append(apgd(model, bx, by, "dlr").cpu())
        sq.append(square_attack(model, bx, by).cpu())
    pgd = torch.cat(pgd).numpy().astype(int)
    ace = torch.cat(ace).numpy().astype(int)
    adlr = torch.cat(adlr).numpy().astype(int)
    sq = torch.cat(sq).numpy().astype(int)
    aa = ((ace + adlr + sq) > 0).astype(int)  # AutoAttack ensemble

    print(f"\n=== {name} (n={N}) ===")
    print(f"  ASR  PGD10={pgd.mean():.4f}  APGD-CE={ace.mean():.4f}  "
          f"APGD-DLR={adlr.mean():.4f}  Square={sq.mean():.4f}  AA-ensemble={aa.mean():.4f}")
    pgd_misses_aa_flips = int(((pgd == 0) & (aa == 1)).sum())
    print(f"  samples PGD10 calls robust but AA flips: {pgd_misses_aa_flips} "
          f"({100*pgd_misses_aa_flips/max(N,1):.2f}% of eval)")
    print(f"  margin AUROC vs PGD10 label = {safe_auroc(pgd, -margin):.4f}")
    print(f"  margin AUROC vs AA    label = {safe_auroc(aa, -margin):.4f}")
    return {"pgd": pgd, "aa": aa, "margin": margin}


def main():
    print("=" * 74)
    print("H174 - AutoAttack-style ensemble (APGD-CE + APGD-DLR + Square) vulnerability labels")
    print("=" * 74)
    print(f"Device={DEVICE}  EPS={EPS:.4f}  EVAL_N={EVAL_N}")

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n--- Training Vanilla ---"); t0 = time.time()
    vanilla = train(train_set, adversarial=False); print(f"  {time.time()-t0:.1f}s")
    print("--- Training PGD-AT ---"); t0 = time.time()
    at = train(train_set, adversarial=True); print(f"  {time.time()-t0:.1f}s")

    eval_model("Vanilla", vanilla, test_x, test_y)
    eval_model("PGD-AT", at, test_x, test_y)

    print("\n" + "=" * 74)
    print("Key check: does margin AUROC survive the gold-standard AA ensemble, and how badly")
    print("does PGD-10 alone over-estimate robustness (the 'PGD robust but AA flips' count)?")
    print("=" * 74)


if __name__ == "__main__":
    main()
