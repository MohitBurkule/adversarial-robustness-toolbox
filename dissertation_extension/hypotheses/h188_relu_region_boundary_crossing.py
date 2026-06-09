"""
H188 - ReLU linear region boundary crossing.

Hypothesis: adversarial examples cross more ReLU activation pattern boundaries
than clean perturbations of the same L-inf magnitude.

Methodology:
  - Train a small MLP (784->256->128->10) on Fashion-MNIST (6k train).
  - For N=500 test samples:
    - Record activation pattern at clean input (binary: 1 if pre-activation > 0).
    - Generate PGD-10 adversarial example (eps=0.3, step=0.03).
    - Generate random perturbation of same L-inf magnitude 0.3.
    - Count Hamming distance (neurons that changed activation) for AE vs random.
  - Report: mean Hamming dist (AE vs random), ratio, Wilcoxon signed-rank p-value.
  - Per logit-margin quartile: mean Hamming distance.
  - AUROC: predict "is adversarial?" using Hamming distance as feature.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import campaign.common as C

# ---- config ----------------------------------------------------------------
SEED = 42
N_TRAIN = 6000
N_EVAL = 1000
N_SAMPLES = 500
EPS = 0.3
PGD_STEPS = 10
PGD_ALPHA = 0.03
EPOCHS = 12
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
OUT_FILE = os.path.join(RESULTS_DIR, "h188_relu_region_boundary_crossing_output.txt")


# ---- MLP with hook-able pre-activations -----------------------------------
class MLPWithPreAct(nn.Module):
    """784->256->128->10 ReLU MLP that can return pre-activation values."""
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def pre_activations(self, x):
        """Return list of pre-activation tensors for hidden layers."""
        x = self.flat(x)
        z1 = self.fc1(x)
        h1 = F.relu(z1)
        z2 = self.fc2(h1)
        return [z1, z2]


def activation_pattern(model, x):
    """Binary activation pattern: 1 where pre-activation > 0, concatenated across layers."""
    pres = model.pre_activations(x)
    bits = torch.cat([(z > 0).float() for z in pres], dim=1)  # (B, 256+128)
    return bits


def hamming_distance(pat_a, pat_b):
    """Per-sample Hamming distance between two binary patterns."""
    return (pat_a != pat_b).float().sum(dim=1)  # (B,)


# ---- main ------------------------------------------------------------------
def main():
    t0 = time.time()
    C.set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = ["=" * 70, "H188 - ReLU Linear Region Boundary Crossing", "=" * 70, ""]

    # load data
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")

    # build and train MLP
    model = MLPWithPreAct().to(C.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(Xtr.size(0), device=Xtr.device)
        for i in range(0, Xtr.size(0), 128):
            idx = perm[i:i+128]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()

    # evaluate clean accuracy
    with torch.no_grad():
        logits_all = model(Xte)
        preds = logits_all.argmax(1)
        clean_acc = (preds == Yte).float().mean().item()
    lines.append(f"Clean accuracy: {clean_acc:.4f}")

    # select N_SAMPLES correctly classified samples
    correct_mask = (preds == Yte)
    correct_idx = correct_mask.nonzero(as_tuple=True)[0]
    if correct_idx.size(0) > N_SAMPLES:
        correct_idx = correct_idx[:N_SAMPLES]
    X_sub = Xte[correct_idx]
    Y_sub = Yte[correct_idx]
    n = X_sub.size(0)
    lines.append(f"Samples used: {n}")
    lines.append("")

    # clean activation pattern
    with torch.no_grad():
        pat_clean = activation_pattern(model, X_sub)

    # adversarial examples (PGD-10)
    X_adv = C.pgd(model, X_sub, Y_sub, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    with torch.no_grad():
        pat_adv = activation_pattern(model, X_adv)

    # random perturbation of same L-inf magnitude
    torch.manual_seed(SEED + 1)
    noise = torch.empty_like(X_sub).uniform_(-EPS, EPS)
    X_rand = (X_sub + noise).clamp(0, 1)
    with torch.no_grad():
        pat_rand = activation_pattern(model, X_rand)

    # Hamming distances
    ham_adv = hamming_distance(pat_clean, pat_adv).cpu().numpy()
    ham_rand = hamming_distance(pat_clean, pat_rand).cpu().numpy()

    lines.append("--- Hamming Distance (neurons that changed activation) ---")
    lines.append(f"  Mean Hamming dist (adversarial): {ham_adv.mean():.2f} +/- {ham_adv.std():.2f}")
    lines.append(f"  Mean Hamming dist (random):      {ham_rand.mean():.2f} +/- {ham_rand.std():.2f}")
    ratio = ham_adv.mean() / max(ham_rand.mean(), 1e-9)
    lines.append(f"  Ratio (adv / random):            {ratio:.3f}")
    lines.append("")

    # Wilcoxon signed-rank test
    try:
        stat, pval = wilcoxon(ham_adv, ham_rand, alternative="greater")
        lines.append(f"Wilcoxon signed-rank (adv > random): stat={stat:.1f}, p={pval:.2e}")
    except Exception as e:
        lines.append(f"Wilcoxon test failed: {e}")
    lines.append("")

    # per-quartile of logit margin
    with torch.no_grad():
        logits_sub = model(X_sub).cpu()
    margins = C.margin_of(logits_sub, Y_sub.cpu())
    quartiles = np.percentile(margins, [25, 50, 75])
    q_labels = [
        (margins <= quartiles[0], "Q1 (lowest margin)"),
        ((margins > quartiles[0]) & (margins <= quartiles[1]), "Q2"),
        ((margins > quartiles[1]) & (margins <= quartiles[2]), "Q3"),
        (margins > quartiles[2], "Q4 (highest margin)"),
    ]
    lines.append("--- Hamming distance by logit-margin quartile ---")
    lines.append(f"{'Quartile':<22} {'N':>5} {'Ham_adv':>10} {'Ham_rand':>10} {'Margin_mean':>12}")
    for mask, label in q_labels:
        if mask.sum() == 0:
            continue
        lines.append(f"{label:<22} {mask.sum():>5} {ham_adv[mask].mean():>10.2f} "
                     f"{ham_rand[mask].mean():>10.2f} {margins[mask].mean():>12.3f}")
    lines.append("")

    # AUROC: predict "is adversarial?" using Hamming distance
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    scores = np.concatenate([ham_adv, ham_rand])
    auroc = C.safe_auroc(labels, scores)
    lines.append(f"AUROC (Hamming dist -> is_adversarial?): {auroc:.4f}")
    lines.append("")

    # adversarial success rate
    with torch.no_grad():
        adv_preds = model(X_adv).argmax(1)
        asr = (adv_preds != Y_sub).float().mean().item()
    lines.append(f"PGD-10 ASR on these samples: {asr:.4f}")

    elapsed = time.time() - t0
    lines.append(f"\nElapsed: {elapsed:.1f}s")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    with open(OUT_FILE, "w") as f:
        f.write(report)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
