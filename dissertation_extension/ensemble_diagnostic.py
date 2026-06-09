"""
Ensemble-disagreement diagnostic for adversarial vulnerability.

Question: train K models that differ only in random initialisation. For each test
sample, measure how much the K models disagree (label vote, softmax variance,
margin of average softmax, etc). Are samples with high disagreement also the
ones easily flipped by adversarial attacks?  AND does this disagreement add
predictive power beyond a single model's final-logit margin?

Pipeline:
  1. Train K=6 CNNs on Fashion-MNIST with seeds 0..K-1 (identical arch / data /
     hyperparameters; only initialisation differs).
  2. Pick model 0 as "victim", model K-1 as "surrogate" (for transfer attacks).
  3. Compute per-sample features from the K models:
        - victim_margin               (final-logit margin of model 0 alone)
        - victim_conf                 (softmax prob on true class, model 0)
        - mean_conf_K                 (mean true-class softmax across K models)
        - var_conf_K                  (variance of same across K models)
        - mean_margin_K               (mean of per-model logit margin)
        - vote_agree                  (fraction of K voting for majority class)
        - vote_entropy                (Shannon entropy of K-vote distribution)
        - n_distinct_preds            (# distinct argmax classes across K models)
        - mean_softmax_l2_to_truth    (mean L2 distance from softmax to one-hot truth)
        - var_softmax_l2_total        (variance of softmax vectors across K, summed)
  4. Targets on model 0:
        - flipped_by_FGSM
        - flipped_by_PGD
        - flipped_by_FGSM_transfer (perturbations from model K-1)
        - min_eps_to_flip (continuous)
  5. Univariate AUROC, multivariate logistic regression, and ablations:
        - full vs (drop only ensemble features)
        - full vs (drop only victim_margin)
        - The Delta tells us whether ensemble info adds anything beyond single-model
          margin, and whether margin alone is enough.

This is a clean test of: "are samples that wobble across random inits the same
samples that are adversarially fragile?"
"""
import time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda")
EPS_TEST = 15.0 / 255.0
K = 6
EPOCHS = 10
BATCH = 128


class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*12*12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2); x = self.do1(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.do2(x); return self.fc2(x)


def fgsm_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad.sign().detach()


def pgd(model, x, y, eps=EPS_TEST, steps=10):
    alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        F.cross_entropy(model(x_adv), y).backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)
    return x_adv.detach()


def train_one(seed, train_set):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    m = CNN().to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        m.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    m.eval(); return m


def batched_logits(m, x, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(m(x[i:i+bs]))
    return torch.cat(out, 0)


def min_eps(m, x, y, eps_max=0.3, iters=15):
    sign = fgsm_grad(m, x, y)
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            f = m(adv).argmax(1) != y
        hi = torch.where(f, mid, hi); lo = torch.where(f, lo, mid)
    return hi


def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N, n_classes = x.size(0), 10

    print(f"Training {K} models with seeds 0..{K-1} ...")
    t0 = time.time()
    models = []
    for k in range(K):
        m = train_one(k, train_set)
        with torch.no_grad():
            acc = (batched_logits(m, x).argmax(1) == y).float().mean().item()
        models.append(m)
        print(f"  seed {k}: clean acc = {acc:.4f}  ({time.time()-t0:.1f}s)")

    # Per-model softmax and argmax on test set
    print("\nCollecting per-model softmax tensors...")
    all_softmax = torch.zeros(K, N, n_classes, device=DEVICE)
    all_argmax = torch.zeros(K, N, dtype=torch.long, device=DEVICE)
    for k, m in enumerate(models):
        logits = batched_logits(m, x)
        all_softmax[k] = F.softmax(logits, 1)
        all_argmax[k] = logits.argmax(1)

    # Pick model 0 as victim, model K-1 as surrogate for transfer attacks
    victim = models[0]
    surrogate = models[K-1]

    # Restrict to samples victim classifies correctly clean
    correct = all_argmax[0] == y
    print(f"Victim clean accuracy: {correct.float().mean().item():.4f}, "
          f"keeping {correct.sum().item()} samples for analysis")
    idx = torch.where(correct)[0]
    x_c, y_c = x[idx], y[idx]
    softmax_c = all_softmax[:, idx, :]              # [K, N_c, C]
    argmax_c = all_argmax[:, idx]                    # [K, N_c]

    # ---- features ----
    Nc = x_c.size(0)
    print("\nComputing features...")
    with torch.no_grad():
        # single-victim features
        victim_logits = batched_logits(victim, x_c)
        sorted_l, _ = victim_logits.sort(1, descending=True)
        victim_margin = (sorted_l[:, 0] - sorted_l[:, 1])
        victim_softmax = F.softmax(victim_logits, 1)
        victim_conf = victim_softmax[torch.arange(Nc, device=DEVICE), y_c]

        # ensemble features
        true_softmax = softmax_c[:, torch.arange(Nc, device=DEVICE), y_c]   # [K, Nc]
        mean_conf_K = true_softmax.mean(0)
        var_conf_K = true_softmax.var(0)

        # per-model logit margin then average
        per_model_logits = []
        for m in models:
            l = batched_logits(m, x_c)
            sl, _ = l.sort(1, descending=True)
            per_model_logits.append(sl[:, 0] - sl[:, 1])
        mean_margin_K = torch.stack(per_model_logits).mean(0)

        # vote-agreement
        votes = argmax_c  # [K, Nc]
        # majority count per sample
        majority_count = torch.zeros(Nc, device=DEVICE)
        for c in range(n_classes):
            majority_count = torch.maximum(majority_count, (votes == c).sum(0).float())
        vote_agree = majority_count / K

        # vote entropy
        vote_hist = torch.zeros(Nc, n_classes, device=DEVICE)
        for c in range(n_classes):
            vote_hist[:, c] = (votes == c).sum(0)
        p = vote_hist / K
        vote_entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(1)

        # number of distinct prediction classes
        n_distinct = torch.zeros(Nc, device=DEVICE)
        for c in range(n_classes):
            n_distinct += ((votes == c).any(0)).float()

        # mean softmax distance to one-hot truth, averaged across K models
        onehot = F.one_hot(y_c, n_classes).float()
        l2_per_model = ((softmax_c - onehot.unsqueeze(0)) ** 2).sum(2).sqrt()  # [K, Nc]
        mean_l2_to_truth = l2_per_model.mean(0)

        # variance of softmax vectors across K (total, summed over classes)
        var_softmax_total = softmax_c.var(0).sum(1)

    feats = torch.stack([victim_margin, victim_conf, mean_conf_K, var_conf_K,
                         mean_margin_K, vote_agree, vote_entropy, n_distinct,
                         mean_l2_to_truth, var_softmax_total], 1)
    feat_names = ["victim_margin", "victim_conf", "mean_conf_K", "var_conf_K",
                  "mean_margin_K", "vote_agree", "vote_entropy", "n_distinct_preds",
                  "mean_l2_to_truth", "var_softmax_total"]

    # ---- targets ----
    print("Computing attack targets on victim...")
    # free models we don't need anymore to make room for PGD
    for k in range(K):
        if k not in (0, K-1):
            models[k] = None
    del all_softmax, all_argmax, softmax_c, argmax_c
    torch.cuda.empty_cache()

    def chunked(fn, *args, bs=256):
        out = []
        for i in range(0, args[0].size(0), bs):
            sub_args = [a[i:i+bs] for a in args]
            out.append(fn(*sub_args))
        return torch.cat(out, 0)

    def _self_fgsm(xs, ys):
        s = fgsm_grad(victim, xs, ys)
        adv = (xs + EPS_TEST * s).clamp(0, 1)
        with torch.no_grad():
            return (victim(adv).argmax(1) != ys)
    flipped_FGSM = chunked(_self_fgsm, x_c, y_c, bs=512)

    def _self_pgd(xs, ys):
        adv = pgd(victim, xs, ys)
        with torch.no_grad():
            return (victim(adv).argmax(1) != ys)
    flipped_PGD = chunked(_self_pgd, x_c, y_c, bs=256)

    def _transfer(xs, ys):
        s = fgsm_grad(surrogate, xs, ys)
        adv = (xs + EPS_TEST * s).clamp(0, 1)
        with torch.no_grad():
            return (victim(adv).argmax(1) != ys)
    flipped_transfer = chunked(_transfer, x_c, y_c, bs=512)
    # min eps to flip
    me_chunks = []
    for i in range(0, Nc, 512):
        me_chunks.append(min_eps(victim, x_c[i:i+512], y_c[i:i+512]))
    min_eps_arr = torch.cat(me_chunks)

    print("\n=== Univariate Spearman / AUROC ===")
    feats_np = feats.cpu().numpy()
    me_np = min_eps_arr.cpu().numpy()
    Xs = StandardScaler().fit_transform(feats_np)

    for i, n in enumerate(feat_names):
        cor = np.corrcoef(feats_np[:, i], me_np)[0, 1]
        auc_self = roc_auc_score(flipped_FGSM.cpu().numpy(), feats_np[:, i])
        auc_self = max(auc_self, 1 - auc_self)
        auc_tx = roc_auc_score(flipped_transfer.cpu().numpy(), feats_np[:, i])
        auc_tx = max(auc_tx, 1 - auc_tx)
        auc_pgd = roc_auc_score(flipped_PGD.cpu().numpy(), feats_np[:, i])
        auc_pgd = max(auc_pgd, 1 - auc_pgd)
        print(f"  {n:<22}  corr(min_eps)={cor:+.3f}  AUROC_self={auc_self:.4f}  "
              f"AUROC_PGD={auc_pgd:.4f}  AUROC_transfer={auc_tx:.4f}")

    ensemble_idx = [feat_names.index(n) for n in
                    ["mean_conf_K", "var_conf_K", "mean_margin_K", "vote_agree",
                     "vote_entropy", "n_distinct_preds", "mean_l2_to_truth",
                     "var_softmax_total"]]
    victim_only_idx = [feat_names.index("victim_margin"), feat_names.index("victim_conf")]

    print("\n=== Multivariate AUROC ablations ===")
    for tname, tgt in [("FGSM_self", flipped_FGSM),
                       ("PGD_self", flipped_PGD),
                       ("FGSM_transfer", flipped_transfer)]:
        y_arr = tgt.cpu().numpy().astype(int)
        if y_arr.std() == 0:
            print(f"  {tname}: degenerate"); continue
        # full
        lr_full = LogisticRegression(max_iter=2000).fit(Xs, y_arr)
        full_auc = roc_auc_score(y_arr, lr_full.predict_proba(Xs)[:, 1])
        # victim only (drop all ensemble features)
        Xs_vonly = Xs[:, victim_only_idx]
        lr_v = LogisticRegression(max_iter=2000).fit(Xs_vonly, y_arr)
        v_auc = roc_auc_score(y_arr, lr_v.predict_proba(Xs_vonly)[:, 1])
        # ensemble only (drop the two victim-only features)
        Xs_eonly = Xs[:, ensemble_idx]
        lr_e = LogisticRegression(max_iter=2000).fit(Xs_eonly, y_arr)
        e_auc = roc_auc_score(y_arr, lr_e.predict_proba(Xs_eonly)[:, 1])
        # margin alone
        Xs_marg = Xs[:, [feat_names.index("victim_margin")]]
        lr_m = LogisticRegression(max_iter=2000).fit(Xs_marg, y_arr)
        m_auc = roc_auc_score(y_arr, lr_m.predict_proba(Xs_marg)[:, 1])
        # full minus ensemble (==victim_margin+victim_conf)
        # full minus victim_margin
        Xs_nomarg = np.delete(Xs, feat_names.index("victim_margin"), axis=1)
        lr_nm = LogisticRegression(max_iter=2000).fit(Xs_nomarg, y_arr)
        nm_auc = roc_auc_score(y_arr, lr_nm.predict_proba(Xs_nomarg)[:, 1])

        print(f"\n  --- {tname}  (pos rate {y_arr.mean():.3f}) ---")
        print(f"    margin-only            : {m_auc:.4f}")
        print(f"    victim-only (margin+conf): {v_auc:.4f}")
        print(f"    ensemble-only          : {e_auc:.4f}")
        print(f"    full (victim+ensemble) : {full_auc:.4f}")
        print(f"    Delta(full - victim)   : {full_auc - v_auc:+.4f}   <- ensemble adds")
        print(f"    Delta(full - ensemble) : {full_auc - e_auc:+.4f}   <- victim adds")
        print(f"    full - margin          : {full_auc - m_auc:+.4f}   <- everything beyond margin")

    # continuous OLS
    print("\n=== OLS on min_eps ===")
    ols_full = LinearRegression().fit(Xs, me_np); r2_full = ols_full.score(Xs, me_np)
    ols_v = LinearRegression().fit(Xs[:, victim_only_idx], me_np); r2_v = ols_v.score(Xs[:, victim_only_idx], me_np)
    ols_e = LinearRegression().fit(Xs[:, ensemble_idx], me_np); r2_e = ols_e.score(Xs[:, ensemble_idx], me_np)
    ols_m = LinearRegression().fit(Xs[:, [feat_names.index("victim_margin")]], me_np)
    r2_m = ols_m.score(Xs[:, [feat_names.index("victim_margin")]], me_np)
    print(f"  margin-only         R^2 = {r2_m:.4f}")
    print(f"  victim-only         R^2 = {r2_v:.4f}")
    print(f"  ensemble-only       R^2 = {r2_e:.4f}")
    print(f"  full                R^2 = {r2_full:.4f}")
    print(f"  Delta(full-victim)      = {r2_full - r2_v:+.4f}")
    print(f"  Delta(full-ensemble)    = {r2_full - r2_e:+.4f}")

    # Spearman-like correlation of every ensemble feature with min_eps (already printed)
    out = {
        "K": K, "EPOCHS": EPOCHS, "N_samples": Nc,
        "feat_names": feat_names,
    }
    with open("ensemble_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.1f}s")
