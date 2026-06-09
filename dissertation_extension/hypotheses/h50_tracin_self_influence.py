"""
Hypothesis H50: TracIn self-influence (Pruthi et al., NeurIPS 2020) predicts
adversarial vulnerability.

Reference: Pruthi, Liu, Sundararajan, Kale. "Estimating Training Data Influence
by Tracing Gradient Descent." NeurIPS 2020. (arXiv:2002.08484)

TracIn self-influence (TracInCP estimator) for a sample z = (x, y) is:
    I(z, z) = sum_{c in checkpoints} eta_c * <grad_theta L(z; theta_c),
                                             grad_theta L(z; theta_c)>
            = sum_c eta_c * ||grad_theta L(z; theta_c)||^2.

We use a unit learning-rate proxy and AVERAGE across checkpoints (so the value
is on a stable scale), matching the description "averaged over training
checkpoints". High self-influence => atypical / mislabeled / "often-flipping"
during training. Hypothesis: high TracIn self-influence => more attackable.

Pipeline:
  1. Train a small CNN on Fashion-MNIST for 10 epochs (Adam). Save checkpoints
     at epochs {2,4,6,8,10}.
  2. For each test sample, compute ||grad_theta L(x,y;theta_c)||^2 at every
     checkpoint and average across the 5 checkpoints => tracin_self_influence.
  3. Compute auxiliary features: victim_margin (final model), mean_pix, std_pix.
  4. Targets:
       flipped_FGSM   at eps = 15/255
       flipped_PGD    at eps = 15/255 (10-step, alpha = eps/4)
       FGSM_min_eps   per-sample binary-search L_inf eps that flips FGSM
  5. Univariate AUROC for binary targets, Spearman/Pearson for the continuous
     target. Multivariate logistic regression: does TracIn add over margin?

Self-contained; uses PyTorch + CUDA. Data cached under /tmp/data.
"""
import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr


# -----------------------------  config  -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/tmp/data"
CKPT_DIR = "/tmp/data/h50_ckpts"
EPOCHS = 10
CKPT_EPOCHS = [2, 4, 6, 8, 10]
BATCH = 128
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = EPS_TEST / 4.0
SEED = 0
N_TEST = 10000          # use full Fashion-MNIST test split
GRAD_BATCH = 1          # we need per-sample gradients => batch size 1
                         # (Fashion-MNIST CNN is tiny so this is fine on GPU)


# -----------------------------  model  ------------------------------
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


# -----------------------------  train  ------------------------------
def train_with_checkpoints(train_set):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_paths = {}
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep}/{EPOCHS}  ({time.time()-t0:.1f}s)")
        if ep in CKPT_EPOCHS:
            p = os.path.join(CKPT_DIR, f"ckpt_ep{ep}.pt")
            torch.save(model.state_dict(), p)
            ckpt_paths[ep] = p
            print(f"    saved checkpoint {p}")
    return model, ckpt_paths


# -----------------------------  tracin  -----------------------------
def per_sample_grad_norm_sq(model, x, y):
    """
    Per-sample squared gradient L2 norm of cross-entropy w.r.t. all model params.

    We loop per-sample (small CNN, manageable cost). Returns a 1D tensor of
    length x.size(0).
    """
    model.eval()  # disable dropout so the grad is deterministic given theta_c
    out = torch.zeros(x.size(0), device=DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    for i in range(x.size(0)):
        model.zero_grad(set_to_none=True)
        xi = x[i:i+1]
        yi = y[i:i+1]
        loss = F.cross_entropy(model(xi), yi)
        grads = torch.autograd.grad(loss, params, retain_graph=False,
                                    create_graph=False)
        s = 0.0
        for g in grads:
            s = s + g.detach().pow(2).sum()
        out[i] = s
    return out


def compute_tracin_self_influence(test_x, test_y, ckpt_paths):
    """
    Per-sample TracInCP self-influence proxy:
        avg_c ||grad_theta L(x,y; theta_c)||^2
    using one term per saved checkpoint.
    """
    N = test_x.size(0)
    accum = torch.zeros(N, device=DEVICE)
    model = CNN().to(DEVICE)
    for ep, path in ckpt_paths.items():
        print(f"  TracIn term @ epoch {ep}: loading {path}")
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        t0 = time.time()
        # process in chunks to bound memory of any intermediate storage
        for i in range(0, N, 256):
            xb = test_x[i:i+256]
            yb = test_y[i:i+256]
            accum[i:i+xb.size(0)] += per_sample_grad_norm_sq(model, xb, yb)
            if i % (256 * 10) == 0:
                print(f"    sample {i}/{N}  ({time.time()-t0:.1f}s)")
        print(f"  done epoch {ep} in {time.time()-t0:.1f}s")
    accum /= float(len(ckpt_paths))
    return accum


# -----------------------------  attacks  ----------------------------
def fgsm_sign(model, x, y):
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def fgsm_flip(model, x, y, eps=EPS_TEST):
    s = fgsm_sign(model, x, y)
    adv = (x + eps * s).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_flip(model, x, y, eps=EPS_TEST, alpha=PGD_ALPHA, steps=PGD_STEPS):
    model.eval()
    x0 = x.clone().detach()
    adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        g = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * g.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def fgsm_min_eps(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary-search for smallest L_inf eps that flips FGSM.
    Returns eps in [0, eps_max] (eps_max if never flips)."""
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


# -----------------------------  features  ---------------------------
def compute_aux_features(model, test_x, test_y):
    """Returns victim_margin, mean_pix, std_pix, final_pred."""
    model.eval()
    N = test_x.size(0)
    with torch.no_grad():
        logits = []
        for i in range(0, N, 512):
            logits.append(model(test_x[i:i+512]))
        logits = torch.cat(logits, 0)
    sorted_logits, _ = logits.sort(1, descending=True)
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]
    pred = logits.argmax(1)
    mean_pix = test_x.view(N, -1).mean(1)
    std_pix = test_x.view(N, -1).std(1)
    return margin, mean_pix, std_pix, pred


# -----------------------------  analysis  ---------------------------
def auroc_safe(y, x):
    if y.std() == 0:
        return float("nan")
    a = roc_auc_score(y, x)
    return max(a, 1 - a)


def evaluate(feats_np, feat_names, targets, target_names):
    print("\n========== univariate AUROC ==========")
    rows = []
    for ti, tname in enumerate(target_names):
        y = targets[ti]
        if y.dtype != np.int64:
            print(f"\n[continuous target] {tname}")
            for fi, fn in enumerate(feat_names):
                rho, _ = spearmanr(feats_np[:, fi], y)
                rp, _ = pearsonr(feats_np[:, fi], y)
                print(f"  {fn:<25} spearman={rho:+.4f}  pearson={rp:+.4f}")
            continue
        print(f"\n[binary target] {tname}  (positive rate = {y.mean():.3f})")
        if y.std() == 0:
            print("  degenerate, skipping")
            continue
        for fi, fn in enumerate(feat_names):
            a = auroc_safe(y, feats_np[:, fi])
            print(f"  univariate AUROC  {fn:<25} {a:.4f}")
        Xs = StandardScaler().fit_transform(feats_np)
        # full multivariate
        lr = LogisticRegression(max_iter=2000).fit(Xs, y)
        full_auc = roc_auc_score(y, lr.predict_proba(Xs)[:, 1])
        # margin-only
        mi = feat_names.index("victim_margin")
        lr_m = LogisticRegression(max_iter=2000).fit(Xs[:, [mi]], y)
        margin_only_auc = roc_auc_score(y, lr_m.predict_proba(Xs[:, [mi]])[:, 1])
        # margin + tracin
        ti2 = feat_names.index("tracin_self_influence")
        idx_mt = [mi, ti2]
        lr_mt = LogisticRegression(max_iter=2000).fit(Xs[:, idx_mt], y)
        mt_auc = roc_auc_score(y, lr_mt.predict_proba(Xs[:, idx_mt])[:, 1])
        # tracin-only
        lr_t = LogisticRegression(max_iter=2000).fit(Xs[:, [ti2]], y)
        tracin_only_auc = roc_auc_score(y,
                                        lr_t.predict_proba(Xs[:, [ti2]])[:, 1])
        # full minus tracin
        idx_no_tracin = [i for i in range(len(feat_names)) if i != ti2]
        lr_nt = LogisticRegression(max_iter=2000).fit(Xs[:, idx_no_tracin], y)
        no_tracin_auc = roc_auc_score(
            y, lr_nt.predict_proba(Xs[:, idx_no_tracin])[:, 1])

        print(f"  multivariate FULL                 {full_auc:.4f}")
        print(f"  multivariate margin only          {margin_only_auc:.4f}")
        print(f"  multivariate margin + tracin      {mt_auc:.4f}")
        print(f"  multivariate tracin only          {tracin_only_auc:.4f}")
        print(f"  multivariate FULL - tracin        {no_tracin_auc:.4f}")
        print(f"  delta(full - no_tracin)          {full_auc - no_tracin_auc:+.4f}")
        print(f"  delta(margin+tracin - margin)    {mt_auc - margin_only_auc:+.4f}")
        print(f"  standardised coefficients (full):")
        for n, c in zip(feat_names, lr.coef_.flatten()):
            print(f"    {n:<25} {c:+.4f}")
        rows.append({
            "target": tname,
            "univariate_tracin": auroc_safe(y, feats_np[:, ti2]),
            "univariate_margin": auroc_safe(y, feats_np[:, mi]),
            "full_auc": full_auc,
            "margin_only": margin_only_auc,
            "margin_plus_tracin": mt_auc,
            "tracin_only": tracin_only_auc,
            "full_minus_tracin": no_tracin_auc,
            "delta_full_no_tracin": full_auc - no_tracin_auc,
            "delta_mt_m": mt_auc - margin_only_auc,
        })
    return rows


# -----------------------------  main  -------------------------------
def main():
    print(f"device: {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True,
                                      transform=tf)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True,
                                     transform=tf)

    print("Training victim CNN on Fashion-MNIST ...")
    model, ckpt_paths = train_with_checkpoints(train_set)

    # materialise test set on GPU
    test_x = torch.stack([test_set[i][0] for i in range(N_TEST)]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(N_TEST)]).to(DEVICE)

    print("\nComputing TracIn self-influence (avg over checkpoints) ...")
    t0 = time.time()
    tracin = compute_tracin_self_influence(test_x, test_y, ckpt_paths)
    print(f"  TracIn done in {time.time()-t0:.1f}s")

    print("\nComputing victim margin / pixel stats ...")
    margin, mean_pix, std_pix, pred = compute_aux_features(model, test_x, test_y)

    # Use only samples the final model gets right (those are the ones an attacker
    # would flip; matches the convention in diagnostic_test.py).
    correct = pred == test_y
    print(f"  {correct.sum().item()}/{N_TEST} samples correctly classified")
    idx = torch.nonzero(correct, as_tuple=True)[0]
    x_c = test_x[idx]
    y_c = test_y[idx]
    tracin_c = tracin[idx]
    margin_c = margin[idx]
    mean_c = mean_pix[idx]
    std_c = std_pix[idx]

    print("\nComputing targets ...")
    # FGSM flip
    t0 = time.time()
    fgsm_res = []
    for i in range(0, x_c.size(0), 256):
        fgsm_res.append(fgsm_flip(model, x_c[i:i+256], y_c[i:i+256]))
    flipped_FGSM = torch.cat(fgsm_res)
    print(f"  FGSM done in {time.time()-t0:.1f}s  rate={flipped_FGSM.float().mean():.3f}")

    # PGD flip
    t0 = time.time()
    pgd_res = []
    for i in range(0, x_c.size(0), 256):
        pgd_res.append(pgd_flip(model, x_c[i:i+256], y_c[i:i+256]))
    flipped_PGD = torch.cat(pgd_res)
    print(f"  PGD done in {time.time()-t0:.1f}s  rate={flipped_PGD.float().mean():.3f}")

    # FGSM_min_eps
    t0 = time.time()
    me = []
    for i in range(0, x_c.size(0), 256):
        me.append(fgsm_min_eps(model, x_c[i:i+256], y_c[i:i+256]))
    FGSM_min_eps_vec = torch.cat(me)
    print(f"  min_eps done in {time.time()-t0:.1f}s  mean={FGSM_min_eps_vec.mean():.4f}")

    # ----- assemble features -----
    feat_names = ["tracin_self_influence", "victim_margin", "mean_pix", "std_pix"]
    feats = torch.stack([tracin_c, margin_c, mean_c, std_c], dim=1).cpu().numpy()

    targets = [
        flipped_FGSM.cpu().numpy().astype(np.int64),
        flipped_PGD.cpu().numpy().astype(np.int64),
        FGSM_min_eps_vec.cpu().numpy().astype(np.float64),
    ]
    target_names = ["flipped_FGSM", "flipped_PGD", "FGSM_min_eps"]

    # ----- describe TracIn distribution -----
    tr = feats[:, 0]
    print("\n========== TracIn self-influence distribution ==========")
    print(f"  n={len(tr)}  mean={tr.mean():.6g}  std={tr.std():.6g}")
    print(f"  min={tr.min():.6g}  max={tr.max():.6g}")
    qs = np.quantile(tr, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"  quantiles 5/25/50/75/95: " + ", ".join(f"{q:.6g}" for q in qs))
    print(f"  spearman(tracin, victim_margin) = "
          f"{spearmanr(tr, feats[:, 1]).correlation:+.4f}")

    rows = evaluate(feats, feat_names, targets, target_names)

    # ----- summary table -----
    print("\n========== SUMMARY ==========")
    print(f"{'target':<15} {'uni_tracin':>10} {'uni_margin':>10} "
          f"{'mar_only':>9} {'mar+tr':>9} {'full':>9} "
          f"{'d(mt-m)':>9} {'d(full-no_tr)':>14}")
    for r in rows:
        print(f"{r['target']:<15} {r['univariate_tracin']:>10.4f} "
              f"{r['univariate_margin']:>10.4f} {r['margin_only']:>9.4f} "
              f"{r['margin_plus_tracin']:>9.4f} {r['full_auc']:>9.4f} "
              f"{r['delta_mt_m']:>+9.4f} {r['delta_full_no_tracin']:>+14.4f}")

    # save numeric outputs to disk for downstream analysis
    out_path = "/tmp/data/h50_tracin_results.npz"
    np.savez(out_path,
             tracin=feats[:, 0], margin=feats[:, 1],
             mean_pix=feats[:, 2], std_pix=feats[:, 3],
             flipped_FGSM=targets[0], flipped_PGD=targets[1],
             FGSM_min_eps=targets[2])
    print(f"\nSaved features+targets to {out_path}")
    print("Saved checkpoints in:", CKPT_DIR)


if __name__ == "__main__":
    main()
