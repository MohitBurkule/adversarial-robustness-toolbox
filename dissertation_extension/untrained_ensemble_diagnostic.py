"""
Untrained-ensemble disagreement as adversarial-vulnerability proxy.

User's idea: K CNNs with the same architecture but DIFFERENT random
initialisations, NO TRAINING. For each test input we record the K-vote
distribution. If random initialisations of the same architecture disagree a
lot for a given input, that's a property of the (architecture, input) pair
alone — independent of any learned weights. Question: does that disagreement
predict adversarial vulnerability of a *separately trained* victim model on
the same inputs?

Pipeline:
  1. Train ONE victim CNN (and one surrogate for transfer attacks).
  2. Spawn K=32 untrained CNNs with different random inits — NO training.
  3. Compute per-sample untrained-ensemble features:
        - vote_agree_untrained, vote_entropy_untrained, n_distinct_untrained
        - var_softmax_untrained_total, mean_l2_to_truth_untrained
  4. Targets on victim: flipped_FGSM, flipped_PGD, flipped_transfer, min_eps.
  5. Compare per-feature AUROC and ablation vs victim's final-margin baseline.

Notes:
  - Without training, the softmax of each random model is approximately uniform
    on average; differences between inits depend on how random projections land
    for each input. The signal will be very noisy. We test whether ANY signal
    survives.
  - We also include a positive control: a separately-trained K-ensemble run is
    in `ensemble_diagnostic.py`; results can be compared.
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
K_UNTRAINED = 32        # can afford many because no training
EPOCHS_VICTIM = 10
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


def train_one(seed, train_set, epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    m = CNN().to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(epochs):
        m.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    m.eval(); return m


def init_untrained(seed):
    torch.manual_seed(seed)
    m = CNN().to(DEVICE).eval()
    return m


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


def main():
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    x_test = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N, n_classes = x_test.size(0), 10

    print("Training victim and surrogate (the only trained models)...")
    t0 = time.time()
    victim = train_one(0, train_set, EPOCHS_VICTIM)
    surrogate = train_one(1, train_set, EPOCHS_VICTIM)
    print(f"  done ({time.time()-t0:.1f}s)")

    # untrained ensemble: spawn K random CNNs, never train
    print(f"Building {K_UNTRAINED} untrained random-init CNNs...")
    # collect their softmax/argmax on test set
    untrained_softmax = torch.zeros(K_UNTRAINED, N, n_classes, device=DEVICE)
    untrained_argmax = torch.zeros(K_UNTRAINED, N, dtype=torch.long, device=DEVICE)
    for k in range(K_UNTRAINED):
        m = init_untrained(1000 + k)        # seed offset so disjoint from victim
        with torch.no_grad():
            for i in range(0, N, 512):
                logits = m(x_test[i:i+512])
                untrained_softmax[k, i:i+512] = F.softmax(logits, 1)
                untrained_argmax[k, i:i+512] = logits.argmax(1)
        del m
    torch.cuda.empty_cache()

    # restrict to samples victim classifies correctly
    with torch.no_grad():
        v_logits = []
        for i in range(0, N, 512):
            v_logits.append(victim(x_test[i:i+512]))
        v_logits = torch.cat(v_logits)
    v_pred = v_logits.argmax(1)
    keep = v_pred == y_test
    print(f"Victim clean acc {keep.float().mean().item():.4f}; analysing {keep.sum().item()} samples")
    idx = torch.where(keep)[0]
    x_c = x_test[idx]; y_c = y_test[idx]
    softmax_c = untrained_softmax[:, idx, :]
    argmax_c = untrained_argmax[:, idx]
    Nc = x_c.size(0)

    # victim margin
    sorted_l, _ = v_logits[idx].sort(1, descending=True)
    victim_margin = sorted_l[:, 0] - sorted_l[:, 1]

    # ---- untrained-ensemble features ----
    with torch.no_grad():
        votes = argmax_c
        majority = torch.zeros(Nc, device=DEVICE)
        for c in range(n_classes):
            majority = torch.maximum(majority, (votes == c).sum(0).float())
        vote_agree_u = majority / K_UNTRAINED

        vote_hist = torch.zeros(Nc, n_classes, device=DEVICE)
        for c in range(n_classes):
            vote_hist[:, c] = (votes == c).sum(0)
        p = vote_hist / K_UNTRAINED
        vote_entropy_u = -(p * torch.log(p.clamp_min(1e-9))).sum(1)

        n_distinct_u = torch.zeros(Nc, device=DEVICE)
        for c in range(n_classes):
            n_distinct_u += ((votes == c).any(0)).float()

        onehot = F.one_hot(y_c, n_classes).float()
        l2_per_model = ((softmax_c - onehot.unsqueeze(0)) ** 2).sum(2).sqrt()
        mean_l2_truth_u = l2_per_model.mean(0)

        var_softmax_u = softmax_c.var(0).sum(1)

        # fraction of K untrained models that happened to predict the TRUE class
        # (essentially noise, but maybe biased by input statistics)
        frac_correct_untrained = (votes == y_c.unsqueeze(0)).float().mean(0)

    feats = torch.stack([victim_margin, vote_agree_u, vote_entropy_u, n_distinct_u,
                         mean_l2_truth_u, var_softmax_u, frac_correct_untrained], 1)
    feat_names = ["victim_margin", "vote_agree_u", "vote_entropy_u",
                  "n_distinct_u", "mean_l2_truth_u", "var_softmax_u",
                  "frac_correct_untrained"]

    # ---- attack targets ----
    print("Computing attack targets on victim...")
    def chunked(fn, *args, bs=256):
        out = []
        for i in range(0, args[0].size(0), bs):
            sub = [a[i:i+bs] for a in args]
            out.append(fn(*sub))
        return torch.cat(out)

    def _fgsm(xs, ys):
        s = fgsm_grad(victim, xs, ys)
        adv = (xs + EPS_TEST * s).clamp(0, 1)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    flipped_FGSM = chunked(_fgsm, x_c, y_c, bs=512)

    def _pgd(xs, ys):
        adv = pgd(victim, xs, ys)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    flipped_PGD = chunked(_pgd, x_c, y_c, bs=256)

    def _tx(xs, ys):
        s = fgsm_grad(surrogate, xs, ys)
        adv = (xs + EPS_TEST * s).clamp(0, 1)
        with torch.no_grad():
            return victim(adv).argmax(1) != ys
    flipped_transfer = chunked(_tx, x_c, y_c, bs=512)

    def _meps(xs, ys, eps_max=0.3, iters=15):
        s = fgsm_grad(victim, xs, ys)
        lo = torch.zeros(xs.size(0), device=DEVICE)
        hi = torch.full((xs.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xs + mid.view(-1, 1, 1, 1) * s).clamp(0, 1)
            with torch.no_grad():
                f = victim(adv).argmax(1) != ys
            hi = torch.where(f, mid, hi); lo = torch.where(f, lo, mid)
        return hi
    min_eps_arr = chunked(_meps, x_c, y_c, bs=512)

    print("\n=== Per-feature univariate analysis ===")
    feats_np = feats.cpu().numpy()
    me_np = min_eps_arr.cpu().numpy()
    Xs = StandardScaler().fit_transform(feats_np)
    for i, n in enumerate(feat_names):
        cor = np.corrcoef(feats_np[:, i], me_np)[0, 1]
        a_self = roc_auc_score(flipped_FGSM.cpu().numpy(), feats_np[:, i])
        a_self = max(a_self, 1 - a_self)
        a_pgd = roc_auc_score(flipped_PGD.cpu().numpy(), feats_np[:, i])
        a_pgd = max(a_pgd, 1 - a_pgd)
        a_tx = roc_auc_score(flipped_transfer.cpu().numpy(), feats_np[:, i])
        a_tx = max(a_tx, 1 - a_tx)
        print(f"  {n:<25} corr(min_eps)={cor:+.3f}  AUROC_FGSM={a_self:.4f}  "
              f"AUROC_PGD={a_pgd:.4f}  AUROC_transfer={a_tx:.4f}")

    print("\n=== Multivariate: does untrained-ensemble add to victim_margin alone? ===")
    untrained_idx = [feat_names.index(n) for n in feat_names if n != "victim_margin"]
    margin_idx = [feat_names.index("victim_margin")]
    for tname, tgt in [("FGSM_self", flipped_FGSM),
                       ("PGD_self", flipped_PGD),
                       ("FGSM_transfer", flipped_transfer)]:
        y_arr = tgt.cpu().numpy().astype(int)
        if y_arr.std() == 0:
            print(f"  {tname}: degenerate"); continue
        # margin only
        lr_m = LogisticRegression(max_iter=2000).fit(Xs[:, margin_idx], y_arr)
        m_auc = roc_auc_score(y_arr, lr_m.predict_proba(Xs[:, margin_idx])[:, 1])
        # untrained-ensemble only
        lr_u = LogisticRegression(max_iter=2000).fit(Xs[:, untrained_idx], y_arr)
        u_auc = roc_auc_score(y_arr, lr_u.predict_proba(Xs[:, untrained_idx])[:, 1])
        # full
        lr_f = LogisticRegression(max_iter=2000).fit(Xs, y_arr)
        f_auc = roc_auc_score(y_arr, lr_f.predict_proba(Xs)[:, 1])
        print(f"\n  --- {tname}  (pos rate {y_arr.mean():.3f}) ---")
        print(f"    margin-only            : {m_auc:.4f}")
        print(f"    untrained-ensemble only: {u_auc:.4f}")
        print(f"    full                   : {f_auc:.4f}")
        print(f"    Delta(full - margin)   : {f_auc - m_auc:+.4f}")

    out = {"K_untrained": K_UNTRAINED,
           "victim_acc": keep.float().mean().item(),
           "mean_vote_agree_untrained": vote_agree_u.mean().item(),
           "mean_n_distinct_untrained": n_distinct_u.mean().item()}
    with open("untrained_ensemble_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.1f}s")
