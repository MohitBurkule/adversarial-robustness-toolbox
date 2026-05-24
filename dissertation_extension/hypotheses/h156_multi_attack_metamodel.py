"""
Hypothesis H156: Multi-Task Meta Learner Across Attacks.

Train small CNNs on Fashion-MNIST matching diagnostic_test.py for 10 epochs.
Train 3 separate models (Model 0, Model 1, Model 2) to extract ensemble features.
Evaluate on the first 1000 correctly classified test samples from Model 0.

Extract exactly 20 diverse features per sample:
1. margin: logits top1 - top2 difference
2. mean_pix: mean of input pixels
3. std_pix: standard deviation of input pixels
4. sobel_mean: mean magnitude of Sobel spatial gradient filters
5. max_pix: maximum input pixel value
6. min_pix: minimum input pixel value
7. entropy_pix: histogram-based Shannon entropy of pixel values
8. top1_prob: predicted class softmax probability
9. top2_prob: second-highest class softmax probability
10. confusion_ratio: top2_prob / (top1_prob + 1e-9)
11. mc_entropy: MC-Dropout softmax prediction entropy (10 runs)
12. mc_var: MC-Dropout softmax prediction variance (10 runs)
13. ens_disagreement: Softmax prediction variance across the 3 models
14. grad_l2_norm: L2 norm of the loss gradient w.r.t the input
15. grad_mean: mean of the loss gradient w.r.t the input
16. grad_std: standard deviation of the loss gradient w.r.t the input
17. act_norm_layer1: activation L2 norm of Conv1 ReLU
18. act_norm_layer2: activation L2 norm of Conv2 ReLU
19. act_norm_layer3: activation L2 norm of FC1 ReLU
20. act_norm_layer4: activation L2 norm of FC2 (logits)

Generate 4 adversarial attacks at eps=15/255 on Model 0:
- FGSM (untargeted)
- BIM (10 steps)
- PGD (10 steps)
- MIM (10 steps)

Train a multi-output meta-learner (XGBoost Classifier with MultiOutputClassifier,
falling back to RandomForestClassifier if xgboost is not installed) to predict vulnerability to all 4 attacks.
Evaluate feature importances, compute cross-attack importance correlations, and analyze sample vulnerability overlaps.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Handle XGBoost imports with a robust fallback to RandomForest
try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    USE_XGB = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10


class CNN(nn.Module):
    """Small CNN matching diagnostic_test.py architecture with internal activation access."""
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

    def forward_with_internals(self, x):
        """Returns logits and activations from the 4 primary layers during eval mode."""
        is_training = self.training
        self.eval()
        with torch.no_grad():
            a1 = F.relu(self.c1(x))
            a2 = F.relu(self.c2(a1))
            x_pool = F.max_pool2d(a2, 2)
            x_flat = x_pool.flatten(1)
            a3 = F.relu(self.fc1(x_flat))
            a4 = self.fc2(a3)
        if is_training:
            self.train()
        return a4, (a1, a2, a3, a4)


def train_model(train_set, seed):
    """Train a CNN model with a specific seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

    print(f"  Model with seed {seed} trained for {EPOCHS} epochs.")
    return model


# --- Feature Extraction Functions ---

def compute_sobel_mean(x):
    """Apply Sobel filter to find mean magnitude of spatial edges."""
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=DEVICE).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=DEVICE).view(1, 1, 3, 3)
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-9)
    return magnitude.mean(dim=(1, 2, 3))


def compute_pixel_entropy(x):
    """Compute Shannon entropy of the pixel intensity distribution using a 10-bin histogram."""
    N = x.size(0)
    flat = x.view(N, -1)
    # Map to 10 bins
    bins = 10
    counts = []
    for b in range(bins):
        low = b / bins
        high = (b + 1) / bins
        counts.append(((flat >= low) & (flat < high)).float().sum(dim=1))
    counts = torch.stack(counts, dim=1)  # (N, 10)
    probs = counts / flat.size(1) + 1e-9
    entropy = - (probs * torch.log(probs)).sum(dim=1)
    return entropy


def extract_20_features(x, y, model0, model1, model2):
    """Extract exactly 20 features as described in the requirements."""
    N = x.size(0)

    # Put models in eval mode
    model0.eval()
    model1.eval()
    model2.eval()

    # 1-3. Image stats
    flat_x = x.view(N, -1)
    mean_pix = flat_x.mean(dim=1)
    std_pix = flat_x.std(dim=1)
    max_pix, _ = flat_x.max(dim=1)
    min_pix, _ = flat_x.min(dim=1)

    # 4. Sobel mean
    sobel_mean = compute_sobel_mean(x)

    # 5. Pixel entropy
    entropy_pix = compute_pixel_entropy(x)

    # Run Model 0 internals
    logits0, (a1, a2, a3, a4) = model0.forward_with_internals(x)
    probs0 = F.softmax(logits0, dim=1)
    sorted_probs, _ = probs0.sort(dim=1, descending=True)
    sorted_logits, _ = logits0.sort(dim=1, descending=True)

    # 6-8. Probabilities & Margin
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]
    top1_prob = sorted_probs[:, 0]
    top2_prob = sorted_probs[:, 1]
    confusion_ratio = top2_prob / (top1_prob + 1e-9)

    # 9-10. MC-Dropout features on Model 0
    model0.train()  # Turn on dropout
    mc_probs = []
    with torch.no_grad():
        for _ in range(10):
            mc_probs.append(F.softmax(model0(x), dim=1))
    mc_probs = torch.stack(mc_probs)  # (10, N, 10)
    mean_mc_prob = mc_probs.mean(dim=0)  # (N, 10)
    mc_entropy = - (mean_mc_prob * torch.log(mean_mc_prob + 1e-9)).sum(dim=1)
    mc_var = mc_probs.var(dim=0).mean(dim=1)  # Mean class variance across runs
    model0.eval()  # Restore eval mode

    # 11. Ensemble disagreement
    with torch.no_grad():
        probs1 = F.softmax(model1(x), dim=1)
        probs2 = F.softmax(model2(x), dim=1)
    stacked_p = torch.stack([probs0, probs1, probs2])  # (3, N, 10)
    ens_disagreement = stacked_p.var(dim=0).mean(dim=1)

    # 12-14. Gradient w.r.t input features
    x_grad = x.clone().detach().requires_grad_(True)
    out_g = model0(x_grad)
    loss_g = F.cross_entropy(out_g, y)
    model0.zero_grad()
    loss_g.backward()
    grad = x_grad.grad.detach()

    grad_flat = grad.flatten(1)
    grad_l2 = torch.norm(grad_flat, p=2, dim=1)
    grad_mean = grad_flat.mean(dim=1)
    grad_std = grad_flat.std(dim=1)

    # 15-18. Activation norms at 4 layers
    norm_layer1 = torch.norm(a1.flatten(1), p=2, dim=1)
    norm_layer2 = torch.norm(a2.flatten(1), p=2, dim=1)
    norm_layer3 = torch.norm(a3.flatten(1), p=2, dim=1)
    norm_layer4 = torch.norm(a4.flatten(1), p=2, dim=1)

    # Stack features to form a tensor: shape (N, 20)
    features = torch.stack([
        margin,
        mean_pix,
        std_pix,
        sobel_mean,
        max_pix,
        min_pix,
        entropy_pix,
        top1_prob,
        top2_prob,
        confusion_ratio,
        mc_entropy,
        mc_var,
        ens_disagreement,
        grad_l2,
        grad_mean,
        grad_std,
        norm_layer1,
        norm_layer2,
        norm_layer3,
        norm_layer4
    ], dim=1)

    return features.detach().cpu().numpy()


# --- Attack Implementations ---

def attack_fgsm(model, x, y, eps=EPS):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    grad = x_adv.grad.sign().detach()
    adv = (x + eps * grad).clamp(0, 1)
    with torch.no_grad():
        return model(adv).argmax(1) != y


def attack_bim(model, x, y, eps=EPS, steps=10):
    model.eval()
    alpha = eps / 5.0
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.sign().detach()
        x_adv = torch.clamp(x_adv + alpha * grad, x - eps, x + eps).clamp(0, 1).detach()
    with torch.no_grad():
        return model(x_adv).argmax(1) != y


def attack_pgd(model, x, y, eps=EPS, steps=10):
    model.eval()
    alpha = eps / 5.0
    x_adv = x.clone().detach() + (torch.rand_like(x) * 2 - 1) * eps
    x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.sign().detach()
        x_adv = torch.clamp(x_adv + alpha * grad, x - eps, x + eps).clamp(0, 1).detach()
    with torch.no_grad():
        return model(x_adv).argmax(1) != y


def attack_mim(model, x, y, eps=EPS, steps=10, decay=1.0):
    model.eval()
    alpha = eps / 5.0
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.detach()
        grad_norm = torch.norm(grad.flatten(1), p=1, dim=1).view(-1, 1, 1, 1) + 1e-9
        momentum = decay * momentum + grad / grad_norm
        x_adv = torch.clamp(x_adv + alpha * momentum.sign(), x - eps, x + eps).clamp(0, 1).detach()
    with torch.no_grad():
        return model(x_adv).argmax(1) != y


def main():
    print("=" * 80)
    print("Hypothesis H156: Multi-Attack Meta-Learner (FGSM, BIM, PGD, MIM)")
    print("=" * 80)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # 1. Train 3 models
    print("\nTraining 3 models for ensemble features...")
    t0 = time.time()
    model0 = train_model(train_set, seed=0)
    model1 = train_model(train_set, seed=1)
    model2 = train_model(train_set, seed=2)
    print(f"All models trained in {time.time() - t0:.1f}s")

    # Evaluate test set on Model 0 to find correct samples
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    model0.eval()
    with torch.no_grad():
        logits0 = model0(test_x)
        pred0 = logits0.argmax(1)
        correct = (pred0 == test_y)

    x_c = test_x[correct][:1000]
    y_c = test_y[correct][:1000]
    N = x_c.size(0)
    print(f"\nUsing {N} correctly classified samples from Model 0.")

    # 2. Extract the 20 features
    print("\nExtracting exactly 20 features...")
    features = extract_20_features(x_c, y_c, model0, model1, model2)

    # 3. Generate 4 attacks
    print("\nGenerating attack outcomes...")
    batch_size = 256
    fgsm_success = []
    bim_success = []
    pgd_success = []
    mim_success = []

    for i in range(0, N, batch_size):
        xb = x_c[i:i+batch_size]
        yb = y_c[i:i+batch_size]

        fgsm_success.append(attack_fgsm(model0, xb, yb))
        bim_success.append(attack_bim(model0, xb, yb))
        pgd_success.append(attack_pgd(model0, xb, yb))
        mim_success.append(attack_mim(model0, xb, yb))

    fgsm_success = torch.cat(fgsm_success).cpu().numpy().astype(int)
    bim_success = torch.cat(bim_success).cpu().numpy().astype(int)
    pgd_success = torch.cat(pgd_success).cpu().numpy().astype(int)
    mim_success = torch.cat(mim_success).cpu().numpy().astype(int)

    # Stack targets: shape (1000, 4)
    Y = np.stack([fgsm_success, bim_success, pgd_success, mim_success], axis=1)
    attack_names = ["FGSM", "BIM", "PGD", "MIM"]

    # 4. Multi-attack overlap statistics
    vulnerability_sum = Y.sum(axis=1)  # number of attacks that flipped the sample (0 to 4)
    print(f"\nMulti-Attack Vulnerability Overlap:")
    for count in range(5):
        pct = (vulnerability_sum == count).mean() * 100
        print(f"  Vulnerable to exactly {count} attacks: {pct:.1f}%")

    print(f"  Vulnerable to AT LEAST ONE attack: {(vulnerability_sum > 0).mean() * 100:.1f}%")
    print(f"  Vulnerable to ALL FOUR attacks: {(vulnerability_sum == 4).mean() * 100:.1f}%")

    # 5. Fit the Multi-Task Meta-Learner
    print("\nFitting Multi-Output Meta-Learner...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    if USE_XGB:
        print("  Using XGBoost Classifier as the base meta-learner.")
        base = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric='logloss')
    else:
        print("  Using RandomForestClassifier as the fallback base meta-learner.")
        base = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

    metamodel = MultiOutputClassifier(base)
    metamodel.fit(X_scaled, Y)

    # 6. Extract Feature Importances
    # Extract importances for each individual attack model in the multi-output wrapper
    feature_names = [
        "margin", "mean_pix", "std_pix", "sobel_mean", "max_pix",
        "min_pix", "entropy_pix", "top1_prob", "top2_prob", "confusion_ratio",
        "mc_entropy", "mc_var", "ens_disagreement", "grad_l2_norm", "grad_mean",
        "grad_std", "act_norm_layer1", "act_norm_layer2", "act_norm_layer3", "act_norm_layer4"
    ]

    importances = np.zeros((20, 4))
    for idx, est in enumerate(metamodel.estimators_):
        importances[:, idx] = est.feature_importances_

    print("\n" + "=" * 60)
    print("Top 5 Most Important Features per Attack")
    print("=" * 60)
    for idx, att_name in enumerate(attack_names):
        print(f"\n--- Attack: {att_name} ---")
        best_indices = importances[:, idx].argsort()[::-1][:5]
        for b_idx in best_indices:
            print(f"  {feature_names[b_idx]:<25}: importance = {importances[b_idx, idx]:.4f}")

    # 7. Cross-Attack Feature Importance Correlation
    print("\n" + "=" * 60)
    print("Cross-Attack Feature Importance Correlation Matrix")
    print("=" * 60)
    corr_matrix = np.corrcoef(importances, rowvar=False)

    print(f"      {'      '.join([f'{n:<6}' for n in attack_names])}")
    for i in range(4):
        row_str = "      ".join([f"{corr_matrix[i, j]:.4f}" for j in range(4)])
        print(f"{attack_names[i]:<5}: {row_str}")

    # Check if FGSM and PGD are predicted similarly
    fgsm_pgd_corr = corr_matrix[0, 2]
    print(f"\nCorrelation between FGSM-flip and PGD-flip feature importances: {fgsm_pgd_corr:.4f}")


if __name__ == "__main__":
    main()
