"""
Concrete test of the proposed diagnostic S(x):

  S(x) = 1[ argmax_train_confusion_class(x)  ==  argmax_final_2nd_best_logit(x) ]

Question: does S(x) carry predictive signal for adversarial vulnerability after
controlling for known training-dynamics features?

Pipeline:
  1. Train Model A on Fashion-MNIST (and MNIST), logging per-sample per-epoch argmax
     on the test set. Train a second Model B (different seed) for TRANSFER attacks.
  2. Compute six per-sample features from Model A's training history:
        - final_margin                       (final-model only — strong baseline)
        - confidence_mean (Cartography)      (mean true-class softmax across epochs)
        - confidence_var  (Cartography)      (variance of the same)
        - learning_epoch                     (first epoch correctly classified)
        - forgetting_events                  (# correct->wrong flips)
        - S_x                                (our diagnostic)
  3. Compute three vulnerability targets per sample:
        - min_eps_FGSM           (binary-search smallest eps to flip with FGSM)
        - flipped_by_FGSM_self   (Model A's own FGSM at eps=15/255)
        - flipped_by_FGSM_transfer (FGSM perturbations crafted on Model B,
                                    evaluated on Model A — black-box transfer)
  4. Univariate AUROC of each feature against the binary targets.
  5. Multivariate logistic regression and OLS with all 6 features; report
     standardised coefficients and ablation R^2 (remove S_x and see drop).
  6. Optional: do the same on MNIST.
"""
import time, json, sys
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
EPOCHS = 15
BATCH = 128
N_FEATURES = 6


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


def train_and_log(seed, train_set, test_set, n_classes=10):
    """Train; record per-epoch argmax + softmax-prob-on-true for the entire test set."""
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    model = CNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    per_epoch_argmax = torch.zeros(EPOCHS, N, dtype=torch.long, device=DEVICE)
    per_epoch_trueprob = torch.zeros(EPOCHS, N, device=DEVICE)

    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            # batched evaluation to fit in memory
            preds, probs = [], []
            for i in range(0, N, 512):
                logits = model(test_x[i:i+512])
                preds.append(logits.argmax(1))
                p = F.softmax(logits, 1)
                probs.append(p[torch.arange(p.size(0)), test_y[i:i+512]])
            per_epoch_argmax[ep] = torch.cat(preds)
            per_epoch_trueprob[ep] = torch.cat(probs)
    return model, test_x, test_y, per_epoch_argmax, per_epoch_trueprob


def compute_features(test_x, test_y, per_epoch_argmax, per_epoch_trueprob, model, n_classes=10):
    N = test_x.size(0)
    # final-model logits & margin & 2nd-best class
    with torch.no_grad():
        logits = []
        for i in range(0, N, 512):
            logits.append(model(test_x[i:i+512]))
        logits = torch.cat(logits, 0)
    sorted_logits, _ = logits.sort(1, descending=True)
    final_margin = (sorted_logits[:, 0] - sorted_logits[:, 1])
    final_2nd = logits.masked_fill(F.one_hot(test_y, n_classes).bool(), -1e9).argmax(1)
    final_pred = logits.argmax(1)

    # cartography
    conf_mean = per_epoch_trueprob.mean(0)
    conf_var = per_epoch_trueprob.var(0)
    # learning_epoch
    is_right = (per_epoch_argmax == test_y.unsqueeze(0))
    first_right = is_right.float().argmax(0)
    never_right = ~is_right.any(0)
    learning_ep = torch.where(never_right, torch.full_like(first_right, EPOCHS), first_right)
    # forgetting events: count correct->wrong transitions
    transitions = (is_right[:-1] & ~is_right[1:]).sum(0)
    # top-confusion class & S_x
    confusion = torch.zeros(N, n_classes, device=DEVICE)
    for ep in range(EPOCHS):
        wrong = per_epoch_argmax[ep] != test_y
        if wrong.any():
            idx_w = torch.arange(N, device=DEVICE)[wrong]
            confusion[idx_w, per_epoch_argmax[ep][wrong]] += 1
    no_confusion = confusion.sum(1) == 0
    top_conf = confusion.argmax(1)
    S_x = ((top_conf == final_2nd) & (~no_confusion)).long()

    feats = torch.stack([final_margin, conf_mean, conf_var,
                         learning_ep.float(), transitions.float(), S_x.float()], 1)
    return feats, final_pred, final_2nd


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    # use single gradient sign — eps scaling only changes magnitude, not direction
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def attack_success(model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def transfer_attack_success(target_model, surrogate_model, x, y, eps=EPS_TEST):
    sign = fgsm_grad(surrogate_model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (target_model(adv).argmax(1) != y)


def evaluate_features(feats, targets, names, target_names, dataset_name):
    print(f"\n========== {dataset_name} ==========")
    feats_np = feats.detach().cpu().numpy()
    Xs = StandardScaler().fit_transform(feats_np)

    rows = []
    for t_idx, t_name in enumerate(target_names):
        y = targets[t_idx].detach().cpu().numpy().astype(int)
        if y.std() == 0:
            print(f"  target {t_name} has no variance, skipping")
            continue
        print(f"\n--- target: {t_name}  (positive rate = {y.mean():.3f}) ---")
        # univariate AUROC
        for i, n in enumerate(names):
            x_i = feats_np[:, i]
            # try both directions; pick the better-AUROC
            a = roc_auc_score(y, x_i)
            a = max(a, 1 - a)
            print(f"   univariate AUROC  {n:<20} {a:.4f}")

        # full multivariate model + ablation removing S_x
        lr_full = LogisticRegression(max_iter=2000).fit(Xs, y)
        full_auc = roc_auc_score(y, lr_full.predict_proba(Xs)[:, 1])
        Xs_noS = np.delete(Xs, names.index("S_x"), axis=1)
        lr_noS = LogisticRegression(max_iter=2000).fit(Xs_noS, y)
        noS_auc = roc_auc_score(y, lr_noS.predict_proba(Xs_noS)[:, 1])
        delta = full_auc - noS_auc
        print(f"   multivariate AUROC (all 6 features):  {full_auc:.4f}")
        print(f"   multivariate AUROC (S_x removed):     {noS_auc:.4f}")
        print(f"   Delta AUROC from S_x:                 {delta:+.4f}")
        print(f"   standardised coefficients:")
        for n, c in zip(names, lr_full.coef_.flatten()):
            print(f"     {n:<20} {c:+.4f}")
        rows.append((dataset_name, t_name, full_auc, noS_auc, delta))

    # OLS on min_eps (continuous)
    return rows


def run_dataset(name, dataset_cls):
    print(f"\n##### dataset: {name} #####")
    tf = transforms.ToTensor()
    train_set = dataset_cls("./data", train=True, download=True, transform=tf)
    test_set = dataset_cls("./data", train=False, download=True, transform=tf)
    n_classes = 10

    print(" training model A (seed 0)...")
    t0 = time.time()
    model_A, x, y, pe_arg, pe_pr = train_and_log(0, train_set, test_set, n_classes)
    print(f"  done ({time.time()-t0:.1f}s)")
    print(" training model B (seed 1, surrogate for transfer attack)...")
    t0 = time.time()
    model_B, _, _, _, _ = train_and_log(1, train_set, test_set, n_classes)
    print(f"  done ({time.time()-t0:.1f}s)")

    feats, final_pred, _ = compute_features(x, y, pe_arg, pe_pr, model_A, n_classes)
    feat_names = ["final_margin", "conf_mean", "conf_var",
                  "learning_epoch", "forgetting_events", "S_x"]

    # restrict to samples Model A predicts correctly (those are the ones AT cares about)
    correct = final_pred == y
    x_c, y_c, feats_c = x[correct], y[correct], feats[correct]
    print(f" using {correct.sum().item()} samples that Model A classifies correctly")

    # targets
    print(" computing min_eps_to_flip ...")
    t0 = time.time()
    # batch for memory
    me = []
    for i in range(0, x_c.size(0), 512):
        me.append(min_eps_to_flip(model_A, x_c[i:i+512], y_c[i:i+512]))
    min_eps = torch.cat(me)
    print(f"  done ({time.time()-t0:.1f}s)  mean min_eps={min_eps.mean():.4f}")

    print(" computing self FGSM attack success ...")
    asuccess = []
    for i in range(0, x_c.size(0), 512):
        asuccess.append(attack_success(model_A, x_c[i:i+512], y_c[i:i+512]))
    self_flip = torch.cat(asuccess)

    print(" computing transfer FGSM attack success ...")
    tsuccess = []
    for i in range(0, x_c.size(0), 512):
        tsuccess.append(transfer_attack_success(model_A, model_B, x_c[i:i+512], y_c[i:i+512]))
    transfer_flip = torch.cat(tsuccess)

    targets = [self_flip, transfer_flip]
    target_names = ["FGSM_self_flip", "FGSM_transfer_flip"]
    rows = evaluate_features(feats_c, targets, feat_names, target_names, name)

    # ---- conditional analysis: restrict by margin quartile ----
    print(f"\n========== {name}: conditional on margin quartile ==========")
    margin = feats_c[:, feat_names.index("final_margin")].cpu().numpy()
    q = np.quantile(margin, [0.0, 0.25, 0.50, 0.75, 1.0])
    for qi in range(4):
        lo, hi = q[qi], q[qi+1]
        if qi < 3:
            mask = (margin >= lo) & (margin < hi)
        else:
            mask = (margin >= lo) & (margin <= hi)
        if mask.sum() < 50: continue
        print(f"\n  margin quartile {qi+1}  [{lo:.2f}, {hi:.2f}]  n={int(mask.sum())}")
        feats_q = feats_c[torch.as_tensor(mask, device=feats_c.device)]
        Xs = StandardScaler().fit_transform(feats_q.cpu().numpy())
        for t_idx, t_name in enumerate(target_names):
            y_arr = targets[t_idx].cpu().numpy().astype(int)[mask]
            if y_arr.std() == 0:
                print(f"    {t_name}: degenerate (pos rate = {y_arr.mean():.3f})")
                continue
            try:
                full = LogisticRegression(max_iter=2000).fit(Xs, y_arr)
                full_auc = roc_auc_score(y_arr, full.predict_proba(Xs)[:, 1])
                Xs_noS = np.delete(Xs, feat_names.index("S_x"), axis=1)
                noS = LogisticRegression(max_iter=2000).fit(Xs_noS, y_arr)
                noS_auc = roc_auc_score(y_arr, noS.predict_proba(Xs_noS)[:, 1])
                uni_S = roc_auc_score(y_arr, Xs[:, feat_names.index("S_x")])
                uni_S = max(uni_S, 1 - uni_S)
                print(f"    {t_name}: full={full_auc:.4f}  no_Sx={noS_auc:.4f}  "
                      f"delta={full_auc-noS_auc:+.4f}  uni_Sx={uni_S:.4f}  "
                      f"pos_rate={y_arr.mean():.3f}")
            except Exception as e:
                print(f"    {t_name}: error {e}")

    # also a continuous OLS on min_eps
    print(f"\n--- target: min_eps_to_flip  (Spearman-style) ---")
    feats_np = feats_c.detach().cpu().numpy()
    Xs = StandardScaler().fit_transform(feats_np)
    y_cont = min_eps.detach().cpu().numpy()
    for i, n in enumerate(feat_names):
        cor = np.corrcoef(feats_np[:, i], y_cont)[0, 1]
        print(f"   corr(min_eps, {n:<20}) = {cor:+.4f}")
    ols_full = LinearRegression().fit(Xs, y_cont)
    Xs_noS = np.delete(Xs, feat_names.index("S_x"), axis=1)
    ols_noS = LinearRegression().fit(Xs_noS, y_cont)
    r2_full = ols_full.score(Xs, y_cont)
    r2_noS = ols_noS.score(Xs_noS, y_cont)
    print(f"  OLS R^2 with all features:        {r2_full:.4f}")
    print(f"  OLS R^2 with S_x removed:         {r2_noS:.4f}")
    print(f"  Delta R^2 attributable to S_x:    {r2_full - r2_noS:+.4f}")
    print(f"  standardised coefficients:")
    for n, c in zip(feat_names, ols_full.coef_):
        print(f"   {n:<20} {c:+.6f}")
    return rows


if __name__ == "__main__":
    all_rows = []
    for name, ds in [("FashionMNIST", datasets.FashionMNIST),
                     ("MNIST", datasets.MNIST)]:
        all_rows += run_dataset(name, ds)
    print("\n===== SUMMARY (multivariate AUROC: full vs S_x-removed) =====")
    print(f"{'dataset':<14} {'target':<22} {'full':>7} {'no_Sx':>7} {'delta':>8}")
    for d, t, f, n, delta in all_rows:
        print(f"{d:<14} {t:<22} {f:>7.4f} {n:>7.4f} {delta:>+8.4f}")
