"""
Hypothesis H12: Per-sample saliency-map spatial entropy is a per-sample
adversarial vulnerability predictor.

Idea:
  saliency(x) = |d L(model(x), y_true) / d x|             (pixel-wise abs gradient)
  p_i = saliency_i / sum_j saliency_j                     (probability distribution)
  H(x) = -sum_i p_i * log(p_i)                            (Shannon entropy, in nats)

Prediction: low entropy (saliency concentrated on few pixels) => few-pixel
attacks succeed cheaply => sample more vulnerable.
High entropy (diffuse) => attacker must perturb many pixels => more robust
per unit L_inf budget (or rather, low-entropy samples have lower min_eps).

Note: the relationship of entropy to L_inf vulnerability is non-trivial.
Concentrated saliency means a single-pixel gradient is large, so a small
L_inf step in that pixel produces a big logit change. So we still expect
low entropy -> more vulnerable to FGSM/PGD at fixed eps. We test it.

Features evaluated:
  saliency_entropy       (H, nats)
  saliency_max           (max p_i)
  saliency_top10_frac    (sum of top-10 p_i)
Baselines:
  victim_margin          (logit margin top1 - top2 of victim)
  mean_pix, std_pix      (image statistics)
  input_grad_l2_norm     (||d L / d x||_2)
Targets:
  flipped_FGSM_eps15     (FGSM at eps=15/255 flips prediction)
  flipped_PGD            (PGD-10 at eps=15/255 flips prediction)
  min_eps_FGSM           (smallest eps that flips, binary search)  -- continuous

Restricted to test samples the victim classifies correctly (the only samples
where "flip" is well-defined).

Stats: univariate AUROC per feature, multivariate logistic adding entropy
on top of {input_grad_l2_norm, victim_margin} baseline. For the continuous
target (min_eps) we use Spearman correlation and OLS R^2 ablation.

Self-contained. Run with cuda. Data cached in /tmp/data.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

DEVICE = torch.device("cuda")
DATA_ROOT = "/tmp/data"
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4
SEED = 0


# -------------------- model --------------------
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
    # random start within eps ball
    delta = (torch.rand_like(x) * 2 - 1) * eps
    adv = (x_orig + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        # project to eps ball
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_fgsm(model, x, y, eps_max=0.3, iters=15):
    """Binary search for smallest L_inf eps that flips via FGSM (sign-step)."""
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
def per_sample_input_grad(model, x, y):
    """Returns the raw input gradient d L / d x  (no sign, no abs), shape (N,1,28,28)."""
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y, reduction="sum")
    grad = torch.autograd.grad(loss, x)[0]
    return grad.detach()


def saliency_features(grads):
    """
    grads: (N, 1, H, W) raw gradients
    Returns: entropy (N,), max_frac (N,), top10_frac (N,), grad_l2_norm (N,)
    """
    N = grads.size(0)
    abs_g = grads.abs().flatten(1)                              # (N, P)
    l2 = abs_g.norm(dim=1)                                      # ||grad||_2
    s = abs_g.sum(dim=1, keepdim=True).clamp_min(1e-12)
    p = abs_g / s                                               # prob dist
    # Shannon entropy (nats); 0 * log 0 := 0
    logp = torch.where(p > 0, p.log(), torch.zeros_like(p))
    entropy = -(p * logp).sum(dim=1)
    max_frac = p.max(dim=1).values
    top10 = p.topk(10, dim=1).values.sum(dim=1)
    return entropy, max_frac, top10, l2


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

    # full test set tensors
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    # -------- compute baseline features in batches --------
    print("computing victim margin & predictions ...")
    margins, preds = [], []
    BS = 512
    for i in range(0, N, BS):
        m, p = victim_margin(model, test_x[i:i+BS])
        margins.append(m); preds.append(p)
    margin = torch.cat(margins)
    pred = torch.cat(preds)

    # restrict to correctly classified samples
    correct = pred == test_y
    print(f"victim test acc = {correct.float().mean().item():.4f}")
    x_c = test_x[correct]
    y_c = test_y[correct]
    margin_c = margin[correct]
    Nc = x_c.size(0)
    print(f"keeping {Nc} correctly-classified test samples")

    # image statistics
    flat = x_c.flatten(1)
    mean_pix = flat.mean(dim=1)
    std_pix = flat.std(dim=1)

    # -------- saliency features in batches (need grad) --------
    print("computing saliency features ...")
    ent_l, max_l, top10_l, l2_l = [], [], [], []
    for i in range(0, Nc, BS):
        g = per_sample_input_grad(model, x_c[i:i+BS], y_c[i:i+BS])
        e, m, t10, l2 = saliency_features(g)
        ent_l.append(e); max_l.append(m); top10_l.append(t10); l2_l.append(l2)
    sal_entropy = torch.cat(ent_l)
    sal_max = torch.cat(max_l)
    sal_top10 = torch.cat(top10_l)
    grad_l2 = torch.cat(l2_l)

    # -------- targets --------
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

    # -------- stack & analyse --------
    feat_names = [
        "saliency_entropy", "saliency_max", "saliency_top10_frac",
        "victim_margin", "mean_pix", "std_pix", "input_grad_l2_norm",
    ]
    feats = torch.stack([
        sal_entropy, sal_max, sal_top10,
        margin_c, mean_pix, std_pix, grad_l2,
    ], dim=1).cpu().numpy()

    targets_bin = {
        "flipped_FGSM_eps15": fgsm_target.cpu().numpy().astype(int),
        "flipped_PGD_eps15":  pgd_target.cpu().numpy().astype(int),
    }
    target_cont = min_eps.cpu().numpy()

    print("\n========== H12 results ==========")
    print(f"N = {Nc} correctly-classified samples")
    for tname, y in targets_bin.items():
        if y.std() == 0:
            print(f"target {tname}: degenerate (rate={y.mean():.3f}), skipping")
            continue
        print(f"\n--- target: {tname}   positive rate = {y.mean():.4f} ---")
        for i, n in enumerate(feat_names):
            a = roc_auc_score(y, feats[:, i])
            a_dir = max(a, 1 - a)
            print(f"  univariate AUROC  {n:<22} {a_dir:.4f}  (raw={a:.4f})")

        Xs = StandardScaler().fit_transform(feats)

        # baseline: input_grad_l2_norm + victim_margin
        base_idx = [feat_names.index("input_grad_l2_norm"),
                    feat_names.index("victim_margin")]
        ent_idx = feat_names.index("saliency_entropy")
        Xs_base = Xs[:, base_idx]
        Xs_base_ent = Xs[:, base_idx + [ent_idx]]
        Xs_full = Xs

        def auc_for(X):
            lr = LogisticRegression(max_iter=2000).fit(X, y)
            return roc_auc_score(y, lr.predict_proba(X)[:, 1])

        auc_base = auc_for(Xs_base)
        auc_base_ent = auc_for(Xs_base_ent)
        auc_full = auc_for(Xs_full)
        print(f"  multivariate AUROC  baseline {{grad_l2, margin}}            = {auc_base:.4f}")
        print(f"  multivariate AUROC  baseline + saliency_entropy            = {auc_base_ent:.4f}")
        print(f"  Delta AUROC from adding entropy                            = {auc_base_ent-auc_base:+.4f}")
        print(f"  multivariate AUROC  all features                           = {auc_full:.4f}")

        lr_full = LogisticRegression(max_iter=2000).fit(Xs_full, y)
        print("  standardised coefficients (all features):")
        for n, c in zip(feat_names, lr_full.coef_.flatten()):
            print(f"    {n:<22} {c:+.4f}")

    # continuous min_eps target
    print(f"\n--- target: min_eps_FGSM   mean={target_cont.mean():.4f}  std={target_cont.std():.4f} ---")
    for i, n in enumerate(feat_names):
        rho, p = spearmanr(feats[:, i], target_cont)
        print(f"  Spearman rho  {n:<22} {rho:+.4f}  (p={p:.2e})")
    Xs = StandardScaler().fit_transform(feats)
    base_idx = [feat_names.index("input_grad_l2_norm"),
                feat_names.index("victim_margin")]
    ent_idx = feat_names.index("saliency_entropy")
    r2_base = LinearRegression().fit(Xs[:, base_idx], target_cont).score(Xs[:, base_idx], target_cont)
    r2_base_ent = LinearRegression().fit(Xs[:, base_idx + [ent_idx]], target_cont).score(
        Xs[:, base_idx + [ent_idx]], target_cont)
    r2_full = LinearRegression().fit(Xs, target_cont).score(Xs, target_cont)
    print(f"  OLS R^2  baseline {{grad_l2, margin}}            = {r2_base:.4f}")
    print(f"  OLS R^2  baseline + saliency_entropy            = {r2_base_ent:.4f}")
    print(f"  Delta R^2 from adding entropy                   = {r2_base_ent-r2_base:+.4f}")
    print(f"  OLS R^2  all features                           = {r2_full:.4f}")
    ols_full = LinearRegression().fit(Xs, target_cont)
    print("  standardised coefficients (all features):")
    for n, c in zip(feat_names, ols_full.coef_):
        print(f"    {n:<22} {c:+.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
