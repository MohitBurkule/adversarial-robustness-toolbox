"""
H91: Softmax entropy is the dominant uncertainty score.

Hypothesis
----------
The softmax entropy H(p) = -sum_i p_i log p_i of the victim's prediction is the
dominant uncertainty score for predicting adversarial vulnerability. We compare
it head-to-head against the logit margin and a few siblings of entropy that are
all functions of the softmax vector. The question we want to settle is whether
entropy adds anything over margin, or whether it is essentially a non-linear
transformation of it on the data manifold the victim sees.

Pipeline
--------
1. Train a small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
2. For every test sample, compute the victim's softmax vector p and derive:
       softmax_entropy = -sum p_i log p_i
       softmax_gini    = 1 - sum p_i^2
       softmax_top1    = max_i p_i
       softmax_l2      = ||p||_2
   Plus the auxiliary features:
       margin    (final logit margin: top1 - top2)
       mean_pix  (mean pixel value of x)
       std_pix   (std  pixel value of x)
3. Targets per sample (binary except min_eps which is continuous):
       FGSM   flipped at eps=15/255
       PGD    flipped at eps=15/255, 10 iters
       min_eps  smallest L_inf eps that flips with FGSM (binary-search)
4. Univariate AUROC per (feature, target). For min_eps we use Spearman rank
   correlation. We also Spearman-correlate softmax_entropy against margin (and
   the other softmax features) directly to quantify how much of entropy's
   signal is "just margin in disguise".
5. Standalone logistic regression of {entropy} vs {margin} vs {entropy, margin}
   to test whether entropy adds incremental AUROC over margin.

Write code only --- do NOT run.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_ITERS = 10
PGD_ALPHA = EPS_TEST / 4.0
EVAL_BATCH = 512


# ---- model: identical to diagnostic_test.py ----
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


def train_victim(train_set, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
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


# ---- attacks ----
def fgsm_sign(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    sign = fgsm_sign(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, iters=PGD_ITERS):
    x_orig = x.clone().detach()
    # random start within eps-ball
    delta = (torch.rand_like(x_orig) * 2 - 1) * eps
    adv = (x_orig + delta).clamp(0, 1).detach()
    for _ in range(iters):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_sign(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# ---- features ----
def compute_features(model, test_x, test_y):
    """Return (feats_dict, final_pred) on the test set."""
    N = test_x.size(0)
    all_logits = []
    with torch.no_grad():
        for i in range(0, N, EVAL_BATCH):
            all_logits.append(model(test_x[i:i + EVAL_BATCH]))
    logits = torch.cat(all_logits, 0)
    probs = F.softmax(logits, dim=1)

    # softmax-derived
    eps = 1e-12
    entropy = -(probs * (probs + eps).log()).sum(1)
    gini = 1.0 - (probs.pow(2)).sum(1)
    top1 = probs.max(1).values
    l2 = probs.pow(2).sum(1).sqrt()

    # margin
    sorted_logits, _ = logits.sort(1, descending=True)
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]

    # pixel stats
    flat = test_x.flatten(1)
    mean_pix = flat.mean(1)
    std_pix = flat.std(1)

    final_pred = logits.argmax(1)

    feats = {
        "softmax_entropy": entropy,
        "softmax_gini": gini,
        "softmax_top1": top1,
        "softmax_l2": l2,
        "margin": margin,
        "mean_pix": mean_pix,
        "std_pix": std_pix,
    }
    return feats, final_pred


# ---- evaluation ----
def directional_auroc(feature, y_bin):
    """AUROC robust to sign: report max(a, 1-a) and the direction."""
    a = roc_auc_score(y_bin, feature)
    if a >= 0.5:
        return a, "+"
    return 1.0 - a, "-"


def main():
    print("loading Fashion-MNIST ...")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print(f"training CNN for {EPOCHS} epochs on {DEVICE} ...")
    t0 = time.time()
    model = train_victim(train_set, seed=0)
    print(f"  done ({time.time() - t0:.1f}s)")

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("computing features ...")
    feats, final_pred = compute_features(model, test_x, test_y)

    # restrict to samples the victim classifies correctly
    correct = final_pred == test_y
    keep = correct.nonzero(as_tuple=True)[0]
    x_c = test_x[keep]
    y_c = test_y[keep]
    feats_c = {k: v[keep] for k, v in feats.items()}
    print(f" using {keep.numel()} correctly-classified samples (of {test_x.size(0)})")

    # targets
    print("computing FGSM flips ...")
    fgsm_flips = []
    for i in range(0, x_c.size(0), EVAL_BATCH):
        fgsm_flips.append(fgsm_flip(model, x_c[i:i + EVAL_BATCH], y_c[i:i + EVAL_BATCH]))
    fgsm_flips = torch.cat(fgsm_flips)

    print("computing PGD flips ...")
    pgd_flips = []
    for i in range(0, x_c.size(0), EVAL_BATCH):
        pgd_flips.append(pgd_flip(model, x_c[i:i + EVAL_BATCH], y_c[i:i + EVAL_BATCH]))
    pgd_flips = torch.cat(pgd_flips)

    print("computing min_eps ...")
    me_chunks = []
    for i in range(0, x_c.size(0), EVAL_BATCH):
        me_chunks.append(min_eps_to_flip(model, x_c[i:i + EVAL_BATCH], y_c[i:i + EVAL_BATCH]))
    min_eps = torch.cat(me_chunks)

    # ---- analysis ----
    feat_names = ["softmax_entropy", "softmax_gini", "softmax_top1", "softmax_l2",
                  "margin", "mean_pix", "std_pix"]
    feats_np = {k: feats_c[k].detach().cpu().numpy() for k in feat_names}

    y_fgsm = fgsm_flips.detach().cpu().numpy().astype(int)
    y_pgd = pgd_flips.detach().cpu().numpy().astype(int)
    y_min_eps = min_eps.detach().cpu().numpy()

    print("\n========== Univariate AUROC (binary targets) ==========")
    print(f"  FGSM positive rate: {y_fgsm.mean():.3f}")
    print(f"  PGD  positive rate: {y_pgd.mean():.3f}")
    print(f"\n  {'feature':<20} {'AUROC_FGSM':>12} {'dir':>4} "
          f"{'AUROC_PGD':>12} {'dir':>4}")
    for n in feat_names:
        a_f, d_f = directional_auroc(feats_np[n], y_fgsm)
        a_p, d_p = directional_auroc(feats_np[n], y_pgd)
        print(f"  {n:<20} {a_f:>12.4f} {d_f:>4} {a_p:>12.4f} {d_p:>4}")

    print("\n========== Spearman correlation with min_eps ==========")
    print("  (negative correlation => higher feature => more vulnerable)")
    for n in feat_names:
        rho, _ = spearmanr(feats_np[n], y_min_eps)
        print(f"  rho(min_eps, {n:<20}) = {rho:+.4f}")

    print("\n========== Entropy vs margin: are they redundant? ==========")
    print("  Spearman correlation among softmax-derived features and margin:")
    keys_corr = ["softmax_entropy", "softmax_gini", "softmax_top1", "softmax_l2", "margin"]
    for i, a in enumerate(keys_corr):
        for b in keys_corr[i + 1:]:
            rho, _ = spearmanr(feats_np[a], feats_np[b])
            print(f"    rho({a:<18}, {b:<18}) = {rho:+.4f}")

    print("\n========== Head-to-head: entropy vs margin (logistic regression) ==========")
    for target_name, y_bin in [("FGSM", y_fgsm), ("PGD", y_pgd)]:
        if y_bin.std() == 0:
            print(f"  {target_name}: degenerate target, skipping")
            continue
        print(f"\n  --- target: {target_name} ---")
        configs = {
            "entropy_only":         ["softmax_entropy"],
            "margin_only":          ["margin"],
            "entropy+margin":       ["softmax_entropy", "margin"],
            "all_softmax":          ["softmax_entropy", "softmax_gini",
                                     "softmax_top1", "softmax_l2"],
            "all_softmax+margin":   ["softmax_entropy", "softmax_gini",
                                     "softmax_top1", "softmax_l2", "margin"],
            "all_features":         feat_names,
        }
        for cfg_name, cols in configs.items():
            X = np.stack([feats_np[c] for c in cols], axis=1)
            Xs = StandardScaler().fit_transform(X)
            lr = LogisticRegression(max_iter=2000).fit(Xs, y_bin)
            p = lr.predict_proba(Xs)[:, 1]
            auc = roc_auc_score(y_bin, p)
            coefs = ", ".join(f"{c}={w:+.3f}" for c, w in zip(cols, lr.coef_.flatten()))
            print(f"    {cfg_name:<22} AUROC={auc:.4f}   coefs: {coefs}")

    print("\n========== Verdict heuristic ==========")
    a_ent_f, _ = directional_auroc(feats_np["softmax_entropy"], y_fgsm)
    a_mar_f, _ = directional_auroc(feats_np["margin"], y_fgsm)
    a_ent_p, _ = directional_auroc(feats_np["softmax_entropy"], y_pgd)
    a_mar_p, _ = directional_auroc(feats_np["margin"], y_pgd)
    rho_em, _ = spearmanr(feats_np["softmax_entropy"], feats_np["margin"])
    print(f"  entropy AUROC (FGSM)= {a_ent_f:.4f}   margin AUROC (FGSM)= {a_mar_f:.4f}")
    print(f"  entropy AUROC (PGD) = {a_ent_p:.4f}   margin AUROC (PGD) = {a_mar_p:.4f}")
    print(f"  |Spearman(entropy, margin)| = {abs(rho_em):.4f}")
    print("  Interpretation: if |rho| ~ 1, entropy is a monotone transform of margin")
    print("  on this data; if entropy AUROC > margin AUROC AND adding entropy to a")
    print("  margin-only model raises AUROC noticeably, entropy carries unique signal.")


if __name__ == "__main__":
    main()
