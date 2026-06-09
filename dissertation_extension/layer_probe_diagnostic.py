"""
Layer-probe "prediction depth" diagnostic for adversarial vulnerability.

Idea (yours):
  Train one CNN normally. Then attach a linear classifier ("probe") at every
  intermediate layer of the *frozen* backbone. Train each probe.
  For each test sample, define prediction_depth(x) = earliest probe that puts
  x in its correct class. Easy samples are decodable from early layers; hard
  samples need deeper representations.

Question:
  Are samples with deeper prediction_depth (decodable only by late layers)
  the same samples that are easy to attack adversarially?
  And does prediction_depth add value beyond final-logit margin?

This is the linear-probe variant of Baldock et al. 2021 ("Deep Learning Through
the Lens of Example Difficulty", NeurIPS), which used kNN probes. Linear is
cheaper and matches our small CNN.

Pipeline:
  1. Train backbone CNN normally on Fashion-MNIST.
  2. Hook intermediate activations at L probing points (after each conv/relu,
     after pool, after fc1).
  3. Train a linear probe on each hook output, with backbone frozen.
  4. Compute per-sample prediction_depth.
  5. Compute adversarial-vulnerability targets (FGSM/PGD/transfer/min_eps).
  6. Univariate AUROC + multivariate ablation vs final-margin alone.
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
EPOCHS = 10
PROBE_EPOCHS = 5
BATCH = 128
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*12*12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)

    def forward_with_features(self, x):
        """Return (final_logits, dict of intermediate activations)."""
        a = {}
        h = F.relu(self.c1(x)); a["after_conv1"] = h
        h = F.relu(self.c2(h));  a["after_conv2"] = h
        h = F.max_pool2d(h, 2); a["after_pool"] = h
        h = self.do1(h)
        h = h.flatten(1)
        h = F.relu(self.fc1(h)); a["after_fc1"] = h
        h = self.do2(h)
        logits = self.fc2(h)
        return logits, a

    def forward(self, x):
        return self.forward_with_features(x)[0]


def train_backbone(train_set):
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    m = CNN().to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        m.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    m.eval()
    return m


def train_probes(backbone, train_set, probe_names, n_classes=10):
    """Train one linear probe per intermediate layer. Backbone frozen."""
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    # discover probe input dims with one forward pass
    with torch.no_grad():
        sample = train_set[0][0].unsqueeze(0).to(DEVICE)
        _, acts = backbone.forward_with_features(sample)
    probes = {}
    opts = {}
    for name in probe_names:
        feat = acts[name].flatten(1)
        d = feat.size(1)
        probes[name] = nn.Linear(d, n_classes).to(DEVICE)
        opts[name] = torch.optim.Adam(probes[name].parameters(), lr=1e-3)
    for ep in range(PROBE_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                _, acts = backbone.forward_with_features(x)
            for name in probe_names:
                feat = acts[name].flatten(1).detach()
                opts[name].zero_grad()
                logits = probes[name](feat)
                F.cross_entropy(logits, y).backward()
                opts[name].step()
    for p in probes.values():
        p.eval()
    return probes


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

    print("Training backbone (one CNN)...")
    t0 = time.time()
    backbone = train_backbone(train_set)
    print(f"  done ({time.time()-t0:.1f}s)")
    # also surrogate for transfer attack
    torch.manual_seed(1)
    surrogate = train_backbone(train_set)
    print(f"  surrogate done ({time.time()-t0:.1f}s)")

    probe_names = ["after_conv1", "after_conv2", "after_pool", "after_fc1"]
    print(f"Training {len(probe_names)} linear probes (5 epochs each)...")
    probes = train_probes(backbone, train_set, probe_names)

    # ---- probe predictions on test set ----
    print("Computing per-layer probe predictions...")
    probe_argmax = {}
    probe_correct = {}
    with torch.no_grad():
        for i in range(0, x_test.size(0), 512):
            xb = x_test[i:i+512]; yb = y_test[i:i+512]
            _, acts = backbone.forward_with_features(xb)
            for name in probe_names:
                feat = acts[name].flatten(1)
                pred = probes[name](feat).argmax(1)
                probe_argmax.setdefault(name, []).append(pred)
                probe_correct.setdefault(name, []).append(pred == yb)
    for n in probe_names:
        probe_correct[n] = torch.cat(probe_correct[n])
        probe_argmax[n] = torch.cat(probe_argmax[n])

    # also collect final-model correctness
    with torch.no_grad():
        final_logits = []
        for i in range(0, x_test.size(0), 512):
            final_logits.append(backbone(x_test[i:i+512]))
        final_logits = torch.cat(final_logits)
    final_pred = final_logits.argmax(1)
    sorted_l, _ = final_logits.sort(1, descending=True)
    final_margin = sorted_l[:, 0] - sorted_l[:, 1]
    final_correct = (final_pred == y_test)

    # ---- prediction_depth: earliest probe (1..L+1) that gets correct ----
    L = len(probe_names)
    is_right_per_layer = torch.stack([probe_correct[n] for n in probe_names] +
                                     [final_correct])  # [L+1, N]
    # first index that is True; if none, depth = L+1
    first_right = is_right_per_layer.float().argmax(0)
    never_right = ~is_right_per_layer.any(0)
    prediction_depth = torch.where(never_right,
                                   torch.full_like(first_right, L+1),
                                   first_right)
    # Also: how many of the L+1 layers got it right (count)
    n_layers_correct = is_right_per_layer.sum(0).float()
    # Layer-flip events: count of (right->wrong->right->wrong...) transitions
    flips_layer = (is_right_per_layer[:-1] != is_right_per_layer[1:]).sum(0).float()

    print("\nPer-layer accuracy on test set:")
    for n in probe_names:
        print(f"  {n:<14}: {probe_correct[n].float().mean().item():.4f}")
    print(f"  final          : {final_correct.float().mean().item():.4f}")

    # restrict analysis to samples final model gets right
    keep = final_correct
    x_c = x_test[keep]; y_c = y_test[keep]
    margin_c = final_margin[keep]
    depth_c = prediction_depth[keep].float()
    nlc_c = n_layers_correct[keep]
    flips_c = flips_layer[keep]

    # ---- attack targets ----
    print("Computing attack targets...")
    def chunked(fn, *args, bs=256):
        out = []
        for i in range(0, args[0].size(0), bs):
            sub = [a[i:i+bs] for a in args]
            out.append(fn(*sub))
        return torch.cat(out)

    def _fgsm(xs, ys):
        s = fgsm_grad(backbone, xs, ys)
        adv = (xs + EPS_TEST * s).clamp(0, 1)
        with torch.no_grad():
            return backbone(adv).argmax(1) != ys
    flipped_FGSM = chunked(_fgsm, x_c, y_c, bs=512)

    def _pgd(xs, ys):
        adv = pgd(backbone, xs, ys)
        with torch.no_grad():
            return backbone(adv).argmax(1) != ys
    flipped_PGD = chunked(_pgd, x_c, y_c, bs=256)

    def _transfer(xs, ys):
        s = fgsm_grad(surrogate, xs, ys)
        adv = (xs + EPS_TEST * s).clamp(0, 1)
        with torch.no_grad():
            return backbone(adv).argmax(1) != ys
    flipped_transfer = chunked(_transfer, x_c, y_c, bs=512)

    # min eps to flip with FGSM
    def _min_eps(xs, ys, eps_max=0.3, iters=15):
        s = fgsm_grad(backbone, xs, ys)
        lo = torch.zeros(xs.size(0), device=DEVICE)
        hi = torch.full((xs.size(0),), eps_max, device=DEVICE)
        for _ in range(iters):
            mid = (lo + hi) / 2
            adv = (xs + mid.view(-1, 1, 1, 1) * s).clamp(0, 1)
            with torch.no_grad():
                f = backbone(adv).argmax(1) != ys
            hi = torch.where(f, mid, hi); lo = torch.where(f, lo, mid)
        return hi
    min_eps_arr = chunked(_min_eps, x_c, y_c, bs=512)

    # ---- features ----
    feats = torch.stack([margin_c, depth_c, nlc_c, flips_c], 1).cpu().numpy()
    feat_names = ["final_margin", "prediction_depth", "n_layers_correct", "flips_across_layers"]
    Xs = StandardScaler().fit_transform(feats)

    print("\n=== Univariate analysis ===")
    me_np = min_eps_arr.cpu().numpy()
    for i, n in enumerate(feat_names):
        cor = np.corrcoef(feats[:, i], me_np)[0, 1]
        auc_self = max(roc_auc_score(flipped_FGSM.cpu().numpy(), feats[:, i]),
                       1 - roc_auc_score(flipped_FGSM.cpu().numpy(), feats[:, i]))
        auc_pgd = max(roc_auc_score(flipped_PGD.cpu().numpy(), feats[:, i]),
                      1 - roc_auc_score(flipped_PGD.cpu().numpy(), feats[:, i]))
        auc_tx = max(roc_auc_score(flipped_transfer.cpu().numpy(), feats[:, i]),
                     1 - roc_auc_score(flipped_transfer.cpu().numpy(), feats[:, i]))
        print(f"  {n:<22}  corr(min_eps)={cor:+.3f}  AUROC_FGSM={auc_self:.4f}  "
              f"AUROC_PGD={auc_pgd:.4f}  AUROC_transfer={auc_tx:.4f}")

    print("\n=== Multivariate ablation: does prediction_depth add over margin? ===")
    for tname, tgt in [("FGSM_self", flipped_FGSM),
                       ("PGD_self", flipped_PGD),
                       ("FGSM_transfer", flipped_transfer)]:
        y_arr = tgt.cpu().numpy().astype(int)
        if y_arr.std() == 0:
            print(f"  {tname}: degenerate"); continue
        # full
        lr_full = LogisticRegression(max_iter=2000).fit(Xs, y_arr)
        full_auc = roc_auc_score(y_arr, lr_full.predict_proba(Xs)[:, 1])
        # margin only
        Xs_m = Xs[:, [0]]
        lr_m = LogisticRegression(max_iter=2000).fit(Xs_m, y_arr)
        m_auc = roc_auc_score(y_arr, lr_m.predict_proba(Xs_m)[:, 1])
        # depth only
        Xs_d = Xs[:, [1]]
        lr_d = LogisticRegression(max_iter=2000).fit(Xs_d, y_arr)
        d_auc = roc_auc_score(y_arr, lr_d.predict_proba(Xs_d)[:, 1])
        # full minus depth
        Xs_no_d = np.delete(Xs, 1, axis=1)
        lr_nd = LogisticRegression(max_iter=2000).fit(Xs_no_d, y_arr)
        nd_auc = roc_auc_score(y_arr, lr_nd.predict_proba(Xs_no_d)[:, 1])
        print(f"\n  --- {tname}  (pos rate {y_arr.mean():.3f}) ---")
        print(f"    margin-only             : {m_auc:.4f}")
        print(f"    depth-only              : {d_auc:.4f}")
        print(f"    full (margin+depth+others): {full_auc:.4f}")
        print(f"    full without depth      : {nd_auc:.4f}")
        print(f"    Delta(full - margin)    : {full_auc - m_auc:+.4f}")
        print(f"    Delta(full - no_depth)  : {full_auc - nd_auc:+.4f}  <- value of depth")

    # continuous
    print("\n=== OLS on min_eps ===")
    ols = LinearRegression().fit(Xs, me_np); r2 = ols.score(Xs, me_np)
    ols_m = LinearRegression().fit(Xs[:, [0]], me_np); r2_m = ols_m.score(Xs[:, [0]], me_np)
    ols_d = LinearRegression().fit(Xs[:, [1]], me_np); r2_d = ols_d.score(Xs[:, [1]], me_np)
    print(f"  margin-only         R^2 = {r2_m:.4f}")
    print(f"  depth-only          R^2 = {r2_d:.4f}")
    print(f"  full                R^2 = {r2:.4f}")
    print(f"  Delta(full-margin)  = {r2 - r2_m:+.4f}")

    out = {"per_layer_acc": {n: probe_correct[n].float().mean().item() for n in probe_names},
           "final_acc": final_correct.float().mean().item()}
    with open("layer_probe_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.1f}s")
