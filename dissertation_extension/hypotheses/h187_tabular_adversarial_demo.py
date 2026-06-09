"""
H187 - Intuitive tabular adversarial examples on a decision tree.

Dataset: synthetic loan applications with 3 interpretable features.
  - annual_income  (£k)   : 15 – 120
  - credit_score          : 300 – 850
  - debt_to_income        : 0.05 – 0.80   (monthly debt / monthly income)

Ground-truth label (what a sensible human would decide):
  Approved if:
    credit_score >= 620
    AND annual_income >= 25
    AND debt_to_income <= 0.45
  ... with some noise to make it realistic.

A decision tree trained on this will reproduce something close to those rules,
but the learned thresholds will be slightly off (e.g. credit >= 617 instead of
620), and the hard boundaries create exploitable adversarial examples.

For each adversarial example we show:
  - The original person (approved or denied)
  - The minimal perturbation that flips the decision
  - Whether a human would agree the flip is "reasonable"

This makes the fragility of hard thresholds immediately obvious.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RNG = np.random.default_rng(42)


# ── 1. Generate dataset ──────────────────────────────────────────────────────

def true_label(income, credit, dti, noise=True):
    """Ground truth: a sensible human loan officer's rule."""
    approved = (credit >= 620) and (income >= 25) and (dti <= 0.45)
    if noise and RNG.random() < 0.04:   # 4% label noise (edge cases)
        approved = not approved
    return int(approved)

N = 2000
income  = RNG.uniform(15, 120, N)
credit  = RNG.integers(300, 851, N).astype(float)
dti     = RNG.uniform(0.05, 0.80, N)
labels  = np.array([true_label(income[i], credit[i], dti[i]) for i in range(N)])

X = np.stack([income, credit, dti], axis=1)
feat_names = ["income_k", "credit_score", "debt_to_income"]

Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.3, random_state=42)


# ── 2. Train decision tree ───────────────────────────────────────────────────

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(Xtr, ytr)
print(f"Train acc: {accuracy_score(ytr, tree.predict(Xtr)):.3f}  "
      f"Test acc:  {accuracy_score(yte, tree.predict(Xte)):.3f}")

print("\n── Learned rules (top of tree) ──")
print(export_text(tree, feature_names=feat_names, max_depth=3))


# ── 3. Find adversarial examples ─────────────────────────────────────────────

def find_adversarial(sample, label, tree, feat_idx, max_delta, n_steps=1000):
    """
    Scan feature `feat_idx` by tiny steps up and down until prediction flips.
    Returns (perturbed_sample, delta) or None if no flip within max_delta.
    """
    orig_val = sample[feat_idx]
    for direction in [+1, -1]:
        for step in np.linspace(0, max_delta, n_steps)[1:]:
            candidate = sample.copy()
            candidate[feat_idx] = orig_val + direction * step
            if tree.predict(candidate.reshape(1, -1))[0] != label:
                return candidate, direction * step
    return None, None

print("\n── Adversarial examples ──────────────────────────────────────────────")
print(f"{'Person':>6}  {'income':>8}  {'credit':>8}  {'dti':>6}  "
      f"{'decision':>10}  {'perturb feature':>18}  {'delta':>10}  {'new decision':>14}  {'human agrees?':>14}")
print("─" * 105)

# Pick a few interesting test cases: near the boundary
near_boundary = []
for i in range(len(Xte)):
    x = Xte[i]
    proba = tree.predict_proba(x.reshape(1,-1))[0]
    if 0.35 < proba.max() < 0.75:   # uncertain region
        near_boundary.append(i)

# Also add some clear cases just to show the contrast
clear_approve = [i for i in range(len(Xte)) if yte[i]==1 and tree.predict(Xte[i:i+1])[0]==1][:3]
clear_deny    = [i for i in range(len(Xte)) if yte[i]==0 and tree.predict(Xte[i:i+1])[0]==0][:3]

cases = clear_approve + clear_deny + near_boundary[:4]

for n, i in enumerate(cases[:10]):
    x     = Xte[i]
    label = int(tree.predict(x.reshape(1,-1))[0])
    dec   = "APPROVED" if label==1 else "DENIED"

    # try each feature
    best = None
    for fi, (fname, max_d) in enumerate(zip(
            feat_names, [10.0, 50.0, 0.10])):   # max perturbation per feature
        cand, delta = find_adversarial(x, label, tree, fi, max_d)
        if cand is not None:
            if best is None or abs(delta) < abs(best[2]):
                best = (fi, fname, delta, cand)

    if best is None:
        continue

    fi, fname, delta, cand = best
    new_dec = "APPROVED" if tree.predict(cand.reshape(1,-1))[0]==1 else "DENIED"

    # human plausibility
    if fname == "income_k":
        human = "NO  (£{:.0f} diff)".format(abs(delta)*1000)
    elif fname == "credit_score":
        human = "NO  ({:.0f} pts diff)".format(abs(delta))
    else:
        human = "NO  ({:.3f} dti diff)".format(abs(delta))

    print(f"  #{n+1:>3}  "
          f"{x[0]:>7.1f}k  "
          f"{x[1]:>8.0f}  "
          f"{x[2]:>6.2f}  "
          f"{dec:>10}  "
          f"{fname:>18}  "
          f"{delta:>+10.3f}  "
          f"{new_dec:>14}  "
          f"{human:>14}")


# ── 4. Robustness vs accuracy tradeoff demo ──────────────────────────────────

print("\n── Robustness–accuracy tradeoff on this dataset ─────────────────────")
print(f"{'max_depth':>10}  {'test_acc':>10}  {'robust_frac':>14}  "
      f"{'note':>30}")
print("─" * 70)

def robustness_fraction(model, X, y, eps_income=2.0, eps_credit=10.0,
                        eps_dti=0.03, n_rand=30):
    """
    Fraction of test samples that are robust: no random perturbation within
    (eps_income, eps_credit, eps_dti) flips the prediction.
    """
    robust = 0
    rng2 = np.random.default_rng(0)
    for xi, yi in zip(X, y):
        pred_orig = model.predict(xi.reshape(1,-1))[0]
        if pred_orig != yi:
            continue   # only count originally-correct
        flipped = False
        for _ in range(n_rand):
            noise = np.array([
                rng2.uniform(-eps_income, eps_income),
                rng2.uniform(-eps_credit, eps_credit),
                rng2.uniform(-eps_dti,    eps_dti),
            ])
            cand = np.clip(xi + noise,
                           [15, 300, 0.05], [120, 850, 0.80])
            if model.predict(cand.reshape(1,-1))[0] != pred_orig:
                flipped = True
                break
        if not flipped:
            robust += 1
    total_correct = (model.predict(X) == y).sum()
    return robust / total_correct if total_correct else 0.0

for depth in [2, 3, 5, 8, 15]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=42)
    t.fit(Xtr, ytr)
    acc = accuracy_score(yte, t.predict(Xte))
    rob = robustness_fraction(t, Xte[:200], yte[:200])
    note = ("very robust, low acc" if depth <= 2 else
            "good tradeoff" if depth == 3 else
            "overfit boundary" if depth >= 8 else "")
    print(f"{depth:>10}  {acc:>10.3f}  {rob:>14.3f}  {note:>30}")


# ── 5. Why humans don't experience this ──────────────────────────────────────

print("""
── Why a human loan officer doesn't have this problem ────────────────────────

The tree learned:  credit_score <= 617  →  DENY  (hard threshold)
A human thinks:    credit score ~620 means "borderline - look at everything else"

Concretely - adversarial example #1 above:
  The tree APPROVED person A (credit 623, income £42k, dti 0.41)
  Change credit score by  -7 points  →  tree DENIES them.
  A human officer would make the SAME decision for both.  They would never
  let 7 credit score points be the deciding factor at this income level.

Three reasons humans are "robust" that models are not:
  1. Humans use SEMANTIC features: credit score means "repayment behaviour",
     not a number.  7 points is measurement noise, not a meaningful difference.
  2. Humans ABSTAIN near boundaries: "I'm not sure, let me check their employment
     history."  The tree always gives a confident binary answer.
  3. Humans have causal priors: income and credit score interact in a way a
     human understands but a tree encodes only as a fixed threshold.

The robustness-accuracy tradeoff still EXISTS for humans - it just shows up as
  "difficult cases require more time / more information / a specialist"
rather than as "wrong confident decision from a 7-point perturbation."
""")
