"""
H87: Defensive distillation (Papernot et al. 2016).

Train a teacher CNN (architecture matching diagnostic_test.py) on Fashion-MNIST
for 10 epochs.  Train a student with the same architecture using KL-divergence
loss against teacher softmax(z/T=20) (i.e. soft labels at high temperature).
At inference both teacher and student logits are evaluated at T=1.

For each victim model (vanilla teacher + defensively distilled student),
compute three per-sample predictors:
   - margin            (logit gap top1 - top2 at T=1)
   - mean_pix          (mean pixel intensity of the input)
   - std_pix           (std of pixel intensity of the input)

For three vulnerability targets:
   - FGSM_flip   at eps=15/255
   - PGD_flip    eps=15/255, 20 steps, alpha=eps/8
   - min_eps     binary-search smallest L_inf eps that flips FGSM

Report per-feature univariate AUROC against each binary target, and
Spearman/Pearson correlation against min_eps, for both victims so we can see
whether defensive distillation changes feature importance.

CW comparison: Carlini-Wagner is famously known to break defensive
distillation.  We include a small-batch CW L2 attack (a lightweight in-file
implementation, no ART dependency) and report mean L2 perturbation needed to
flip each victim, plus the AUROC of the three predictors against
"CW_flip_within_budget" (budget = L2 norm 2.0).
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
TEMPERATURE = 20.0
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = EPS_TEST / 8.0
MIN_EPS_MAX = 0.3
MIN_EPS_ITERS = 15

# CW config (kept modest because CW is expensive)
CW_N = 1000           # number of samples to attack with CW
CW_STEPS = 200
CW_LR = 0.01
CW_C = 1.0
CW_KAPPA = 0.0
CW_L2_BUDGET = 2.0


class CNN(nn.Module):
    """Matches diagnostic_test.py."""
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


# ---------------------------------------------------------------- training
def train_teacher(train_loader, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  teacher epoch {ep+1}/{EPOCHS} done")
    model.eval()
    return model


def train_student_distilled(teacher, train_loader, seed=1, T=TEMPERATURE):
    """
    KL(student_softmax(z_s/T) || teacher_softmax(z_t/T))
    Student is trained at temperature T; at evaluation we use T=1.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    student = CNN().to(DEVICE)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    teacher.eval()
    for ep in range(EPOCHS):
        student.train()
        for x, _ in train_loader:
            x = x.to(DEVICE)
            with torch.no_grad():
                t_logits = teacher(x)
                t_soft = F.softmax(t_logits / T, dim=1)
            s_logits = student(x)
            s_logsoft = F.log_softmax(s_logits / T, dim=1)
            # Standard distillation loss scaling: multiply by T^2 so that
            # gradient magnitudes are comparable to plain CE.
            loss = F.kl_div(s_logsoft, t_soft, reduction="batchmean") * (T * T)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  student epoch {ep+1}/{EPOCHS} done")
    student.eval()
    return student


# ---------------------------------------------------------------- features
def per_sample_features(model, x):
    """margin (T=1), mean_pix, std_pix"""
    with torch.no_grad():
        logits_list = []
        for i in range(0, x.size(0), 512):
            logits_list.append(model(x[i:i+512]))
        logits = torch.cat(logits_list, 0)
    sorted_l, _ = logits.sort(1, descending=True)
    margin = (sorted_l[:, 0] - sorted_l[:, 1]).cpu().numpy()
    flat = x.view(x.size(0), -1)
    mean_pix = flat.mean(1).cpu().numpy()
    std_pix = flat.std(1).cpu().numpy()
    pred = logits.argmax(1)
    return {"margin": margin, "mean_pix": mean_pix, "std_pix": std_pix}, pred


# ---------------------------------------------------------------- attacks
def fgsm(model, x, y, eps=EPS_TEST):
    x_adv = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_adv), y).backward()
    sign = x_adv.grad.sign().detach()
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        flipped = model(adv).argmax(1) != y
    return flipped


def pgd(model, x, y, eps=EPS_TEST, steps=PGD_STEPS, alpha=PGD_ALPHA):
    # random start within eps-ball
    delta = torch.empty_like(x).uniform_(-eps, eps)
    adv = (x + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        F.cross_entropy(model(adv), y).backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            adv = torch.max(torch.min(adv, x + eps), x - eps).clamp(0, 1).detach()
    with torch.no_grad():
        flipped = model(adv).argmax(1) != y
    return flipped


def min_eps_to_flip(model, x, y, eps_max=MIN_EPS_MAX, iters=MIN_EPS_ITERS):
    """Per-sample binary search for smallest L_inf eps that flips FGSM."""
    x_g = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_g), y).backward()
    sign = x_g.grad.sign().detach()
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def cw_l2(model, x, y, n_classes=10,
          steps=CW_STEPS, lr=CW_LR, c=CW_C, kappa=CW_KAPPA):
    """
    Minimal Carlini-Wagner L2 untargeted attack.
    Returns: (final L2 perturbation per sample, flipped boolean per sample).
    Uses the tanh box-constraint trick.  Single value of c (no outer search)
    because we only need a rough comparison.
    """
    x = x.detach()
    # invert clamp(0,1) into tanh-space: w = atanh(2x-1)
    x_safe = x.clamp(1e-6, 1 - 1e-6)
    w = torch.atanh(2 * x_safe - 1).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    one_hot_y = F.one_hot(y, n_classes).bool()

    best_l2 = torch.full((x.size(0),), float("inf"), device=DEVICE)
    best_adv = x.clone()

    for _ in range(steps):
        adv = 0.5 * (torch.tanh(w) + 1)
        logits = model(adv)
        true_logit = logits[one_hot_y]
        other_logit = logits.masked_fill(one_hot_y, -1e9).max(1).values
        # untargeted: drive true logit below the best other
        f_term = torch.clamp(true_logit - other_logit + kappa, min=0)
        l2 = ((adv - x) ** 2).flatten(1).sum(1)
        loss = (l2 + c * f_term).sum()
        opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            pred = logits.argmax(1)
            succ = (pred != y) & (l2.sqrt() < best_l2)
            best_l2 = torch.where(succ, l2.sqrt(), best_l2)
            best_adv = torch.where(
                succ.view(-1, 1, 1, 1).expand_as(adv), adv, best_adv)
    with torch.no_grad():
        flipped = model(best_adv).argmax(1) != y
    return best_l2.detach().cpu().numpy(), flipped.detach().cpu().numpy()


# ---------------------------------------------------------------- reporting
def auroc_safe(y, score):
    if y.std() == 0:
        return float("nan")
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def evaluate_victim(name, model, x, y, feat_names, do_cw=True):
    print(f"\n========== victim: {name} ==========")
    feats, pred = per_sample_features(model, x)
    correct = (pred == y)
    print(f"  clean accuracy = {correct.float().mean().item():.4f}  "
          f"({correct.sum().item()} / {x.size(0)})")
    x_c = x[correct]; y_c = y[correct]
    feats_c = {k: v[correct.cpu().numpy()] for k, v in feats.items()}

    # FGSM / PGD / min_eps targets (batched)
    fgsm_flip, pgd_flip, min_eps_list = [], [], []
    for i in range(0, x_c.size(0), 256):
        xb = x_c[i:i+256]; yb = y_c[i:i+256]
        fgsm_flip.append(fgsm(model, xb, yb).cpu().numpy())
        pgd_flip.append(pgd(model, xb, yb).cpu().numpy())
        min_eps_list.append(min_eps_to_flip(model, xb, yb).cpu().numpy())
    fgsm_flip = np.concatenate(fgsm_flip).astype(int)
    pgd_flip = np.concatenate(pgd_flip).astype(int)
    min_eps_arr = np.concatenate(min_eps_list)

    print(f"  FGSM flip rate = {fgsm_flip.mean():.4f}")
    print(f"  PGD  flip rate = {pgd_flip.mean():.4f}")
    print(f"  mean min_eps   = {min_eps_arr.mean():.4f}")

    print(f"\n  ---- per-feature AUROC ----")
    print(f"  {'feature':<12} {'FGSM':>8} {'PGD':>8} {'corr_minE':>10} {'sp_minE':>9}")
    binary_targets = {"FGSM": fgsm_flip, "PGD": pgd_flip}
    results = {}
    for f in feat_names:
        v = feats_c[f]
        row = []
        for tname in ("FGSM", "PGD"):
            row.append(auroc_safe(binary_targets[tname], v))
        pear = np.corrcoef(v, min_eps_arr)[0, 1]
        sp = spearmanr(v, min_eps_arr).correlation
        results[f] = {"FGSM_AUROC": row[0], "PGD_AUROC": row[1],
                      "pearson_minE": pear, "spearman_minE": sp}
        print(f"  {f:<12} {row[0]:>8.4f} {row[1]:>8.4f} {pear:>+10.4f} {sp:>+9.4f}")

    # ---------- CW comparison ----------
    if do_cw:
        n_cw = min(CW_N, x_c.size(0))
        # deterministic subset (first n_cw correctly-classified samples)
        x_cw = x_c[:n_cw]; y_cw = y_c[:n_cw]
        print(f"\n  ---- CW-L2 on first {n_cw} correctly-classified samples ----")
        cw_chunks_l2, cw_chunks_flip = [], []
        for i in range(0, n_cw, 256):
            l2, fl = cw_l2(model, x_cw[i:i+256], y_cw[i:i+256])
            cw_chunks_l2.append(l2); cw_chunks_flip.append(fl)
        cw_l2_arr = np.concatenate(cw_chunks_l2)
        cw_flipped = np.concatenate(cw_chunks_flip).astype(int)
        cw_l2_finite = cw_l2_arr[np.isfinite(cw_l2_arr)]
        cw_within = ((cw_l2_arr < CW_L2_BUDGET) & (cw_flipped == 1)).astype(int)
        print(f"  CW overall flip rate = {cw_flipped.mean():.4f}")
        if cw_l2_finite.size:
            print(f"  mean L2 (successful) = {cw_l2_finite.mean():.4f}")
        print(f"  CW within-budget (L2<{CW_L2_BUDGET}) flip rate = {cw_within.mean():.4f}")

        print(f"\n  ---- per-feature AUROC vs CW_within_budget ----")
        for f in feat_names:
            v = feats_c[f][:n_cw]
            a = auroc_safe(cw_within, v)
            # correlation with continuous L2 (use only finite values)
            mask = np.isfinite(cw_l2_arr)
            if mask.sum() > 1:
                pear = np.corrcoef(v[mask], cw_l2_arr[mask])[0, 1]
            else:
                pear = float("nan")
            results[f]["CW_within_AUROC"] = a
            results[f]["pearson_CW_L2"] = pear
            print(f"  {f:<12} AUROC={a:.4f}  corr(L2)={pear:+.4f}")

    return results


def main():
    print(f"device: {DEVICE}")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True,  download=True, transform=tf)
    test_set  = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)

    # build a single test tensor on device
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("\n--- training teacher CNN (vanilla) ---")
    t0 = time.time()
    teacher = train_teacher(train_loader, seed=0)
    print(f"  teacher trained in {time.time()-t0:.1f}s")

    print(f"\n--- training student via defensive distillation (T={TEMPERATURE}) ---")
    t0 = time.time()
    student = train_student_distilled(teacher, train_loader, seed=1, T=TEMPERATURE)
    print(f"  student trained in {time.time()-t0:.1f}s")

    feat_names = ["margin", "mean_pix", "std_pix"]
    res_teacher = evaluate_victim("vanilla_teacher",  teacher, test_x, test_y, feat_names, do_cw=True)
    res_student = evaluate_victim("distilled_student", student, test_x, test_y, feat_names, do_cw=True)

    # -------- comparative summary --------
    print("\n========== H87 SUMMARY ==========")
    keys = list(next(iter(res_teacher.values())).keys())
    header = f"{'feature':<12} {'metric':<18} {'teacher':>10} {'student':>10} {'delta':>10}"
    print(header)
    for f in feat_names:
        for k in keys:
            tv = res_teacher[f].get(k, float("nan"))
            sv = res_student[f].get(k, float("nan"))
            d  = sv - tv if (np.isfinite(tv) and np.isfinite(sv)) else float("nan")
            print(f"{f:<12} {k:<18} {tv:>10.4f} {sv:>10.4f} {d:>+10.4f}")

    print("\nInterpretation hint: defensive distillation typically inflates the "
          "logit `margin` (vanishing-gradient masking), so margin-AUROC against "
          "FGSM/PGD may become misleading.  CW-L2 ignores logit scale and is "
          "expected to still flip the student at modest L2; whether `margin` "
          "predicts CW-within-budget vulnerability is the key comparison.")


if __name__ == "__main__":
    main()
