"""
H178 - Area Under the Margin (AUM) as a per-sample vulnerability predictor.

Motivation (advisor critique, Paper 4):
  Paper 4 ("training difficulty is orthogonal to adversarial vulnerability") shows that forgetting
  events and an approximate C-score predict adversarial vulnerability at near chance. The reviewer
  explicitly flags a closely-related training-dynamics measure that may NOT be at chance:
  AUM -- Area Under the Margin (Pleiss et al., NeurIPS 2020) -- the margin trajectory integrated
  over training. Two questions:
    (1) Does AUM predict per-sample adversarial vulnerability better than chance?
    (2) Does AUM carry INCREMENTAL signal beyond the final-epoch logit margin (the dominant
        predictor), or is it -- like forgetting events -- redundant with margin?

Protocol:
  - Train a vanilla CNN; at the end of every epoch record each EVAL sample's logit margin
    (correct-class logit minus the largest other-class logit) and whether it is misclassified.
  - AUM            = mean over epochs of the (signed) margin trajectory.
  - final_margin   = last-epoch margin (the standard dominant predictor).
  - forget_count   = number of epochs the sample is misclassified (a forgetting-style control).
  - Vulnerability label = PGD-10 flip at the final model, restricted to finally-correct samples.
  - Univariate AUROC for AUM, final_margin, forget_count.
  - Incremental test: 5-fold CV logistic regression, [final_margin] vs [final_margin, AUM];
    report mean held-out AUROC and the delta. Also Spearman corr(AUM, final_margin).

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10
EVAL_N = 2000
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


def pgd_attack(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=10):
    model.eval()
    adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    adv = adv.clamp(0, 1)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        with torch.no_grad():
            adv = (adv + alpha * adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return adv.detach()


@torch.no_grad()
def signed_margin_and_wrong(model, x, y):
    """Return (signed margin = correct_logit - max_other_logit, is_misclassified) per sample."""
    model.eval()
    logits = model(x)
    N = x.size(0)
    correct_logit = logits[torch.arange(N), y]
    masked = logits.clone()
    masked[torch.arange(N), y] = float("-inf")
    max_other = masked.max(1).values
    margin = (correct_logit - max_other)
    wrong = (logits.argmax(1) != y)
    return margin, wrong


def safe_auroc(label, score):
    label = np.asarray(label).astype(int)
    if label.std() == 0:
        return float("nan")
    a = roc_auc_score(label, np.asarray(score))
    return max(a, 1 - a)


def cv_auroc(X, y, seed=0):
    """Mean 5-fold held-out AUROC for a standardised logistic-regression predictor."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        a = roc_auc_score(y[te], p)
        aucs.append(max(a, 1 - a))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    print("=" * 74)
    print("H178 - AUM (Area Under the Margin) as a vulnerability predictor")
    print("=" * 74)
    print(f"Device={DEVICE}  EPOCHS={EPOCHS}  EVAL_N={EVAL_N}  EPS={EPS:.4f}")

    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    rng = np.random.RandomState(SEED)
    eval_idx = rng.choice(len(test_set), EVAL_N, replace=False)
    eval_x = torch.stack([test_set[i][0] for i in eval_idx]).to(DEVICE)
    eval_y = torch.tensor([test_set[i][1] for i in eval_idx]).to(DEVICE)

    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    margin_traj = np.zeros((EPOCHS, EVAL_N), dtype=np.float64)
    wrong_traj = np.zeros((EPOCHS, EVAL_N), dtype=np.int32)

    print("\n--- Training + recording per-epoch eval margins ---")
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        # record eval-set margins this epoch
        m_all, w_all = [], []
        for i in range(0, EVAL_N, 512):
            m, w = signed_margin_and_wrong(model, eval_x[i:i+512], eval_y[i:i+512])
            m_all.append(m.cpu().numpy()); w_all.append(w.cpu().numpy())
        margin_traj[ep] = np.concatenate(m_all)
        wrong_traj[ep] = np.concatenate(w_all).astype(int)
    print(f"  trained in {time.time()-t0:.1f}s")

    aum = margin_traj.mean(axis=0)              # area under the margin
    final_margin = margin_traj[-1]              # last-epoch margin
    forget_count = wrong_traj.sum(axis=0)       # epochs misclassified (forgetting-style)

    # vulnerability label on finally-correct samples
    finally_correct = wrong_traj[-1] == 0
    fc_idx = np.where(finally_correct)[0]
    print(f"\n  finally-correct eval samples: {len(fc_idx)} / {EVAL_N}")

    fx = eval_x[torch.tensor(fc_idx, device=DEVICE)]
    fy = eval_y[torch.tensor(fc_idx, device=DEVICE)]
    flips = []
    for i in range(0, fx.size(0), 256):
        xa = pgd_attack(model, fx[i:i+256], fy[i:i+256])
        with torch.no_grad():
            flips.append((model(xa).argmax(1) != fy[i:i+256]).cpu())
    flip = torch.cat(flips).numpy().astype(int)
    print(f"  PGD ASR on finally-correct: {flip.mean():.4f}")

    aum_fc = aum[fc_idx]
    fm_fc = final_margin[fc_idx]
    forget_fc = forget_count[fc_idx]

    print("\n--- Univariate AUROC (predicting PGD flip) ---")
    print(f"  {'predictor':<16} {'AUROC':>8}")
    print(f"  {'AUM':<16} {safe_auroc(flip, aum_fc):>8.4f}")
    print(f"  {'final_margin':<16} {safe_auroc(flip, fm_fc):>8.4f}")
    print(f"  {'forget_count':<16} {safe_auroc(flip, forget_fc):>8.4f}")

    rho, _ = spearmanr(aum_fc, fm_fc)
    print(f"\n  Spearman corr(AUM, final_margin) = {rho:.4f}")

    print("\n--- Incremental signal: 5-fold CV logistic regression ---")
    if flip.std() > 0:
        a_m, s_m = cv_auroc(fm_fc.reshape(-1, 1), flip)
        a_ma, s_ma = cv_auroc(np.column_stack([fm_fc, aum_fc]), flip)
        print(f"  [final_margin]          CV-AUROC = {a_m:.4f} +/- {s_m:.4f}")
        print(f"  [final_margin + AUM]    CV-AUROC = {a_ma:.4f} +/- {s_ma:.4f}")
        print(f"  delta AUROC (AUM adds)  = {a_ma - a_m:+.4f}")
        print("\n  Interpretation: delta ~ 0 => AUM is redundant with final margin (like forgetting);")
        print("  delta > a few sigma => AUM carries genuinely incremental vulnerability signal.")
    else:
        print("  PGD flip label is constant; AUROC undefined (saturation).")
    print("=" * 74)


if __name__ == "__main__":
    main()
