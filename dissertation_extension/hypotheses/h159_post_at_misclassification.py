"""
H159 - Post-AT Misclassification Prediction

Hypothesis: After adversarial training (AT), some samples that were correctly
classified by the vanilla model become misclassified. The vanilla model's own
features (margin, gradient norm, min-eps) can predict *before AT* which samples
will be hurt by adversarial training.

Procedure:
1. Train a vanilla CNN (10 epochs, seed=0) on Fashion-MNIST.
2. Evaluate on 2000 test samples; record S_vanilla (correctly classified).
3. Train a PGD-AT CNN (same seed/arch, 10 epochs, PGD-7, alpha=2/255).
4. Evaluate the same 2000 test samples on the AT model; record S_at.
5. Partition S_vanilla into:
   - Preserved   : S_vanilla ∩ S_at
   - Newly wrong : S_vanilla \\ S_at
   - Newly right : S_at \\ S_vanilla  (bonus)
6. For samples in S_vanilla compute vanilla features:
   - margin        (top1 - top2 logit)
   - top1_prob     (softmax max)
   - min_eps       (binary search, 8 iters, FGSM direction)
   - grad_l2_norm  (L2 of input gradient wrt cross-entropy loss)
7. Compare features (mean ± std) between Preserved vs Newly Wrong groups.
8. Compute AUROC of each feature for predicting "newly wrong" membership.

Key question: Can the vanilla model's margin predict which samples AT will break?
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10
N_TEST = 2000
SEED = 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def fgsm_direction(model, x, y):
    """Return the sign of the gradient of the loss wrt x (FGSM direction)."""
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return x.grad.detach().sign()


def pgd_attack(model, x, y, eps, alpha, steps):
    """PGD L-inf attack."""
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x, -eps, eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)
    return x_adv.detach()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_vanilla(train_loader):
    set_seed(SEED)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        avg = total_loss / len(train_loader.dataset)
        print(f"  [Vanilla] Epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}")
    return model


def train_at(train_loader):
    set_seed(SEED)
    model = CNN(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = pgd_attack(model, x, y, eps=EPS, alpha=2.0 / 255.0, steps=7)
            opt.zero_grad()
            loss = F.cross_entropy(model(x_adv), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        avg = total_loss / len(train_loader.dataset)
        print(f"  [AT]      Epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictions(model, loader):
    """Return (predictions, labels) arrays over the loader."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds.append(logits.argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


# ---------------------------------------------------------------------------
# Feature computation for samples in S_vanilla
# ---------------------------------------------------------------------------

def compute_vanilla_features(model, loader_items):
    """
    Compute per-sample features using the vanilla model.
    loader_items: list of (x_single, y_single) tensors (already on CPU).
    Returns dict of lists: margin, top1_prob, min_eps, grad_l2_norm.
    """
    model.eval()
    margins = []
    top1_probs = []
    min_epses = []
    grad_norms = []

    for x, y in loader_items:
        x = x.unsqueeze(0).to(DEVICE)   # (1,1,28,28)
        y_t = y.unsqueeze(0).to(DEVICE)  # (1,)

        # --- margin and top1_prob ---
        x_in = x.clone().detach().requires_grad_(True)
        logits = model(x_in)
        probs = torch.softmax(logits, dim=1).squeeze()
        sorted_logits = logits.squeeze().sort(descending=True).values
        margin = (sorted_logits[0] - sorted_logits[1]).item()
        top1_prob = probs.max().item()

        # --- grad_l2_norm ---
        loss = F.cross_entropy(logits, y_t)
        model.zero_grad()
        loss.backward()
        grad = x_in.grad.detach()
        grad_norm = grad.norm(2).item()

        # --- min_eps via binary search (8 iters, FGSM direction) ---
        lo, hi = 0.0, 1.0
        with torch.no_grad():
            sign_dir = grad.sign()  # already computed above
        for _ in range(8):
            mid = (lo + hi) / 2.0
            x_pert = torch.clamp(x + mid * sign_dir, 0.0, 1.0)
            with torch.no_grad():
                pred = model(x_pert).argmax(1).item()
            if pred != y.item():
                hi = mid  # adversarial, reduce eps
            else:
                lo = mid  # still correct, increase eps

        margins.append(margin)
        top1_probs.append(top1_prob)
        min_epses.append(hi)
        grad_norms.append(grad_norm)

    return {
        "margin": np.array(margins),
        "top1_prob": np.array(top1_probs),
        "min_eps": np.array(min_epses),
        "grad_l2_norm": np.array(grad_norms),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("H159 - Post-AT Misclassification Prediction")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"EPS={EPS:.4f}  EPOCHS={EPOCHS}  N_TEST={N_TEST}  SEED={SEED}")

    # --- Data ---
    tf = transforms.ToTensor()
    train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_ds  = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Fix 2000 test indices deterministically
    rng = np.random.RandomState(SEED)
    test_indices = rng.choice(len(test_ds), N_TEST, replace=False)
    test_subset = Subset(test_ds, test_indices)

    train_loader = DataLoader(train_ds,   batch_size=BATCH, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_subset, batch_size=BATCH, shuffle=False, num_workers=2)

    # --- Train vanilla ---
    print("\n--- Training Vanilla CNN ---")
    vanilla_model = train_vanilla(train_loader)

    # --- Train AT ---
    print("\n--- Training PGD-AT CNN ---")
    at_model = train_at(train_loader)

    # --- Evaluate both models on 2000 test samples ---
    print("\n--- Evaluating on test subset ---")
    vanilla_preds, labels = get_predictions(vanilla_model, test_loader)
    at_preds, _           = get_predictions(at_model, test_loader)

    correct_vanilla = (vanilla_preds == labels)   # bool array length N_TEST
    correct_at      = (at_preds == labels)

    s_vanilla = set(np.where(correct_vanilla)[0])
    s_at      = set(np.where(correct_at)[0])

    preserved   = sorted(s_vanilla & s_at)
    newly_wrong = sorted(s_vanilla - s_at)
    newly_right = sorted(s_at - s_vanilla)

    vanilla_acc = correct_vanilla.mean() * 100
    at_acc      = correct_at.mean() * 100

    print(f"\nVanilla model accuracy on {N_TEST} samples: {vanilla_acc:.2f}%")
    print(f"AT model accuracy on {N_TEST} samples:      {at_acc:.2f}%")
    print(f"\n|S_vanilla|  (correct by vanilla): {len(s_vanilla)}")
    print(f"|S_at|       (correct by AT):       {len(s_at)}")
    print(f"\nPreserved   (S_vanilla ∩ S_at):  {len(preserved)}")
    print(f"Newly wrong (S_vanilla \\ S_at):  {len(newly_wrong)}")
    print(f"Newly right (S_at \\ S_vanilla):  {len(newly_right)}")

    # --- Compute vanilla features for S_vanilla samples ---
    print("\n--- Computing vanilla features for S_vanilla samples ---")
    s_vanilla_list = sorted(s_vanilla)
    # Build per-sample items for feature computation
    items = [(test_subset[i][0], torch.tensor(test_subset[i][1])) for i in s_vanilla_list]
    features = compute_vanilla_features(vanilla_model, items)

    # Map from global index in s_vanilla_list to position
    idx_map = {idx: pos for pos, idx in enumerate(s_vanilla_list)}
    pres_pos  = np.array([idx_map[i] for i in preserved])
    nwrg_pos  = np.array([idx_map[i] for i in newly_wrong])

    # --- Compare features ---
    print("\n--- Feature Comparison: Preserved vs Newly Wrong ---")
    print(f"{'Feature':<16}  {'Preserved (mean±std)':<26}  {'Newly Wrong (mean±std)':<26}")
    print("-" * 72)

    feature_names = ["margin", "top1_prob", "min_eps", "grad_l2_norm"]
    for feat in feature_names:
        vals = features[feat]
        if len(pres_pos) > 0:
            pv = vals[pres_pos]
            ps = f"{pv.mean():.4f} ± {pv.std():.4f}"
        else:
            ps = "N/A"
        if len(nwrg_pos) > 0:
            nv = vals[nwrg_pos]
            ns = f"{nv.mean():.4f} ± {nv.std():.4f}"
        else:
            ns = "N/A"
        print(f"{feat:<16}  {ps:<26}  {ns:<26}")

    # --- AUROC: can vanilla features predict newly_wrong? ---
    print("\n--- AUROC: predicting 'newly wrong' membership (higher = better predictor) ---")
    # Binary target: 1 = newly wrong, 0 = preserved
    if len(pres_pos) > 0 and len(nwrg_pos) > 0:
        y_bin = np.zeros(len(s_vanilla_list), dtype=int)
        y_bin[nwrg_pos] = 1
        # Restrict to preserved ∪ newly_wrong
        mask = np.zeros(len(s_vanilla_list), dtype=bool)
        mask[pres_pos] = True
        mask[nwrg_pos] = True

        y_bin_masked = y_bin[mask]
        print(f"{'Feature':<16}  {'AUROC':<10}  {'Note'}")
        print("-" * 55)
        for feat in feature_names:
            vals = features[feat][mask]
            # For margin, top1_prob, min_eps: lower value -> more likely newly_wrong
            # So we score with negative of the feature to get AUROC > 0.5
            try:
                auroc_pos = roc_auc_score(y_bin_masked, vals)
                auroc_neg = roc_auc_score(y_bin_masked, -vals)
                auroc = max(auroc_pos, auroc_neg)
                direction = "lower -> hurt" if auroc_neg > auroc_pos else "higher -> hurt"
                print(f"{feat:<16}  {auroc:.4f}      {direction}")
            except ValueError as e:
                print(f"{feat:<16}  ERROR: {e}")
    else:
        print("  Not enough samples in one of the groups for AUROC computation.")

    # --- Key finding summary ---
    print("\n--- Key Finding ---")
    if len(pres_pos) > 0 and len(nwrg_pos) > 0:
        m_vals = features["margin"]
        e_vals = features["min_eps"]

        margin_pres = m_vals[pres_pos].mean()
        margin_nwrg = m_vals[nwrg_pos].mean()
        eps_pres    = e_vals[pres_pos].mean()
        eps_nwrg    = e_vals[nwrg_pos].mean()

        margin_lower = margin_nwrg < margin_pres
        eps_lower    = eps_nwrg    < eps_pres

        print(f"Newly wrong samples have lower vanilla margin:  "
              f"{margin_nwrg:.4f} vs {margin_pres:.4f}  -> {margin_lower}")
        print(f"Newly wrong samples have lower min_eps:        "
              f"{eps_nwrg:.4f} vs {eps_pres:.4f}  -> {eps_lower}")

        if margin_lower and eps_lower:
            print("\nCONFIRMED: Samples that AT breaks have lower pre-AT margin AND")
            print("           lower min_eps. The vanilla model's confidence predicts")
            print("           which samples adversarial training will hurt.")
        elif margin_lower or eps_lower:
            print("\nPARTIAL: Only one of the two indicators (margin / min_eps)")
            print("          is consistently lower in the newly-wrong group.")
        else:
            print("\nNOT CONFIRMED: Neither margin nor min_eps is reliably lower")
            print("               in newly-wrong samples. AT damage may be random.")
    else:
        print("  Insufficient group sizes to draw conclusions.")

    print("\nDone.")


if __name__ == "__main__":
    main()
