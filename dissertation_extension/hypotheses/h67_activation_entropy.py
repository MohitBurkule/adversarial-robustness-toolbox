"""
Hypothesis H67: Per-layer activation distribution entropy predicts per-sample
adversarial vulnerability.

Idea:
  For each test sample x, run a forward pass and capture the post-activation
  tensors at four layers of the victim CNN:
      L1: relu(c1(x))             (32 x 26 x 26)
      L2: relu(c2(x))             (64 x 24 x 24)  [pre-pool, post-relu]
      L3: relu(fc1(...))          (128,)
      L4: fc2(...)  (logits)      (10,)
  At each layer we flatten, take absolute value (to map to a non-negative
  signal), normalize to a probability distribution by sum, and compute the
  Shannon entropy H_l = -sum_i p_i log p_i (nats).

  Low entropy at a layer means a small number of units carry almost all of
  the activation mass -> a tightly-tuned, brittle representation along a
  small subspace -> hypothesised more vulnerable.
  High entropy means activation is diffuse -> potentially more robust per
  unit L_inf budget.

Features:
  entropy_L1, entropy_L2, entropy_L3, entropy_L4   (per-layer activation entropy)
  victim_margin                                    (logit top1 - top2)
  mean_pix, std_pix                                (image stats)
Targets:
  flipped_FGSM_eps15     (FGSM at eps=15/255 flips prediction)
  flipped_PGD_eps15      (PGD-10 at eps=15/255 flips prediction)
  min_eps_FGSM           (smallest eps that flips, binary search) -- continuous

Restricted to test samples the victim classifies correctly.

Architecture matches diagnostic_test.py exactly (CNN with c1, c2, fc1, fc2,
dropouts 0.25/0.5). Trained on Fashion-MNIST for 10 epochs.

Stats: univariate AUROC per feature against each binary target.
For min_eps we also report Spearman rho.

Self-contained. Run with cuda. Data cached in /tmp/data.
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

DEVICE = torch.device("cuda")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4
SEED = 0


# -------------------- model (matches diagnostic_test.py) --------------------
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

    def forward_with_activations(self, x):
        """Return (logits, [a1, a2, a3, a4]) where a* are post-activation tensors.

        a1: relu(c1(x))         shape (N, 32, 26, 26)
        a2: relu(c2(a1))        shape (N, 64, 24, 24)   (pre-pool, post-relu)
        a3: relu(fc1(...))      shape (N, 128)
        a4: fc2(...) = logits   shape (N, 10)
        Dropout is not applied (eval mode); we still call the same operations
        as forward(), minus dropout, so the captured activations correspond
        to what the network actually computes at test time.
        """
        a1 = F.relu(self.c1(x))
        a2 = F.relu(self.c2(a1))
        h = F.max_pool2d(a2, 2)
        h = h.flatten(1)
        a3 = F.relu(self.fc1(h))
        a4 = self.fc2(a3)
        return a4, [a1, a2, a3, a4]


def train(model, train_loader, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        print(f"  epoch {ep+1:2d}/{epochs}  loss={loss_sum/total:.4f}  "
              f"train_acc={correct/total:.4f}  ({time.time()-t0:.1f}s)")


# -------------------- attacks --------------------
def fgsm_grad_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad_sign(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    x_orig = x.clone().detach()
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


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad_sign(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# -------------------- features --------------------
def shannon_entropy_of_activations(a):
    """
    a: (N, ...) tensor of activations. Flatten per-sample, take absolute value
    (so post-relu activations stay non-negative; logits get mapped to magnitudes),
    normalize to a probability distribution, return Shannon entropy in nats.
    """
    flat = a.detach().flatten(1).abs()
    s = flat.sum(dim=1, keepdim=True).clamp_min(1e-12)
    p = flat / s
    logp = torch.where(p > 0, p.log(), torch.zeros_like(p))
    return -(p * logp).sum(dim=1)


def activation_entropies(model, x):
    """Return tensor (N, 4) of per-layer Shannon entropies."""
    with torch.no_grad():
        _, acts = model.forward_with_activations(x)
    ents = [shannon_entropy_of_activations(a) for a in acts]
    return torch.stack(ents, dim=1)


def victim_margin(model, x):
    with torch.no_grad():
        logits = model(x)
    sorted_logits, _ = logits.sort(1, descending=True)
    return (sorted_logits[:, 0] - sorted_logits[:, 1]), logits.argmax(1)


# -------------------- driver --------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)

    print("training victim CNN on Fashion-MNIST ...")
    model = CNN(10).to(DEVICE)
    train(model, train_loader, EPOCHS)
    model.eval()

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    print("computing victim margin & predictions ...")
    margins, preds = [], []
    BS = 512
    for i in range(0, N, BS):
        m, p = victim_margin(model, test_x[i:i+BS])
        margins.append(m); preds.append(p)
    margin = torch.cat(margins)
    pred = torch.cat(preds)

    correct = pred == test_y
    print(f"victim test acc = {correct.float().mean().item():.4f}")
    x_c = test_x[correct]
    y_c = test_y[correct]
    margin_c = margin[correct]
    Nc = x_c.size(0)
    print(f"keeping {Nc} correctly-classified test samples")

    flat = x_c.flatten(1)
    mean_pix = flat.mean(dim=1)
    std_pix = flat.std(dim=1)

    print("computing per-layer activation entropies ...")
    ents = []
    for i in range(0, Nc, BS):
        ents.append(activation_entropies(model, x_c[i:i+BS]))
    ent_all = torch.cat(ents, dim=0)  # (Nc, 4)
    entropy_L1 = ent_all[:, 0]
    entropy_L2 = ent_all[:, 1]
    entropy_L3 = ent_all[:, 2]
    entropy_L4 = ent_all[:, 3]

    # targets
    print("computing FGSM eps=15/255 attack ...")
    fgsm_l = []
    for i in range(0, Nc, BS):
        fgsm_l.append(fgsm_flip(model, x_c[i:i+BS], y_c[i:i+BS], EPS_TEST))
    fgsm_target = torch.cat(fgsm_l)

    print("computing PGD-10 eps=15/255 attack ...")
    pgd_l = []
    for i in range(0, Nc, BS):
        pgd_l.append(pgd_flip(model, x_c[i:i+BS], y_c[i:i+BS], EPS_TEST, PGD_ALPHA, PGD_STEPS))
    pgd_target = torch.cat(pgd_l)

    print("computing min_eps FGSM (binary search) ...")
    me_l = []
    for i in range(0, Nc, BS):
        me_l.append(min_eps_fgsm(model, x_c[i:i+BS], y_c[i:i+BS]))
    min_eps = torch.cat(me_l)

    feat_names = [
        "entropy_L1", "entropy_L2", "entropy_L3", "entropy_L4",
        "victim_margin", "mean_pix", "std_pix",
    ]
    feats = torch.stack([
        entropy_L1, entropy_L2, entropy_L3, entropy_L4,
        margin_c, mean_pix, std_pix,
    ], dim=1).cpu().numpy()

    targets_bin = {
        "flipped_FGSM_eps15": fgsm_target.cpu().numpy().astype(int),
        "flipped_PGD_eps15":  pgd_target.cpu().numpy().astype(int),
    }
    target_cont = min_eps.cpu().numpy()

    print("\n========== H67 results ==========")
    print(f"N = {Nc} correctly-classified samples")

    # summary stats of the entropy features
    print("\nfeature summary (mean +/- std):")
    for i, n in enumerate(feat_names):
        v = feats[:, i]
        print(f"  {n:<18} mean={v.mean():+.4f}  std={v.std():.4f}  "
              f"min={v.min():+.4f}  max={v.max():+.4f}")

    for tname, y in targets_bin.items():
        if y.std() == 0:
            print(f"\ntarget {tname}: degenerate (rate={y.mean():.3f}), skipping")
            continue
        print(f"\n--- target: {tname}   positive rate = {y.mean():.4f} ---")
        for i, n in enumerate(feat_names):
            a = roc_auc_score(y, feats[:, i])
            a_dir = max(a, 1 - a)
            print(f"  univariate AUROC  {n:<18} {a_dir:.4f}  (raw={a:.4f})")

    print(f"\n--- target: min_eps_FGSM   mean={target_cont.mean():.4f}  "
          f"std={target_cont.std():.4f} ---")
    for i, n in enumerate(feat_names):
        rho, p = spearmanr(feats[:, i], target_cont)
        print(f"  Spearman rho  {n:<18} {rho:+.4f}  (p={p:.2e})")
        # also report AUROC of feature against "low min_eps" (median split)
        thresh = np.median(target_cont)
        y_bin = (target_cont <= thresh).astype(int)
        if y_bin.std() > 0:
            a = roc_auc_score(y_bin, feats[:, i])
            a_dir = max(a, 1 - a)
            print(f"  univariate AUROC (min_eps<=median)  {n:<18} {a_dir:.4f}  (raw={a:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
