"""
Hypothesis H155: Ensembled Meta-Detector for Adversarial Inputs.

Train small CNNs on Fashion-MNIST matching diagnostic_test.py for 10 epochs.
To compute ensemble-based features, we train 3 separate models (Model 0, Model 1, Model 2) with different seeds.

Evaluation dataset:
- Take 1000 correctly classified test samples from Model 0.
- Create a 50/50 mix:
  - 500 clean samples (is_adversarial = 0)
  - 500 adversarial samples generated via FGSM on Model 0 at eps=15/255 (is_adversarial = 1)

For each sample in the 50/50 mix, compute 5 advanced features:
1. Feature-Squeezing L1: L1 difference between logits of original and 2-bit squeezed input.
2. MC-Dropout Entropy: Entropy of averaged softmax predictions over 10 forward passes with dropout active.
3. Ensemble Disagreement: Mean variance of softmax predictions across the 3 trained models.
4. KDE-Density: Kernel Density Estimation log-likelihood in FC1 representation space, fitted on clean training representations.
5. Mahalanobis Distance: Distance in FC1 representation space to the closest class-conditional training representation mean.

Train a Logistic Regression meta-detector on these 5 features.
Evaluate using 5-fold cross-validated AUROC, and compare it with the individual AUROCs of each detector.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

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

    def forward_representation(self, x):
        """Returns logits and FC1 activations (representations)."""
        # Save dropout state, set dropout to identity for feature extraction
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
        return a4, a3


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


def attack_fgsm(model, x, y, eps=EPS):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    grad = x_adv.grad.sign().detach()
    adv = (x + eps * grad).clamp(0, 1)
    return adv


def main():
    print("=" * 70)
    print("Hypothesis H155: Ensembled Meta-Detector for Adversaries")
    print("=" * 70)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # 1. Train 3 models for ensemble statistics
    print("\nTraining 3 ensemble models...")
    t0 = time.time()
    model0 = train_model(train_set, seed=0)
    model1 = train_model(train_set, seed=1)
    model2 = train_model(train_set, seed=2)
    print(f"All models trained in {time.time() - t0:.1f}s")

    # Evaluate test set with Model 0 to find correct samples
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    model0.eval()
    with torch.no_grad():
        logits0 = model0(test_x)
        pred0 = logits0.argmax(1)
        correct = (pred0 == test_y)

    x_c = test_x[correct][:1000]
    y_c = test_y[correct][:1000]
    print(f"\nUsing 1000 correctly classified samples from Model 0.")

    # 2. Build 50/50 clean / adversarial dataset
    # First 500 remain clean, last 500 are perturbed via FGSM
    print("\nGenerating 50/50 clean/adversarial evaluation mix...")
    x_eval = x_c.clone()
    y_eval = y_c.clone()

    # Apply FGSM to the last 500 samples
    x_eval[500:] = attack_fgsm(model0, x_eval[500:], y_eval[500:], eps=EPS)

    # Meta-learning target: is_adversarial (0 for clean, 1 for adversarial)
    is_adv = np.array([0] * 500 + [1] * 500)

    # 3. Fit KDE and Mahalanobis parameters on training representations
    print("\nExtracting reference representations from training set...")
    train_loader_ref = DataLoader(train_set, batch_size=1000, shuffle=False)
    x_train_ref, y_train_ref = next(iter(train_loader_ref))
    x_train_ref, y_train_ref = x_train_ref.to(DEVICE), y_train_ref.to(DEVICE)

    _, z_train = model0.forward_representation(x_train_ref)
    z_train_np = z_train.cpu().numpy()
    y_train_np = y_train_ref.cpu().numpy()

    # Fit KDE
    print("  Fitting Kernel Density Estimator...")
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(z_train_np)

    # Fit Class-conditional Means and Covariance for Mahalanobis
    print("  Calculating class-conditional covariance for Mahalanobis...")
    class_means = {}
    for c in range(N_CLASSES):
        class_means[c] = z_train_np[y_train_np == c].mean(axis=0)

    # Pool covariance: centering representations class-wise
    z_centered = np.zeros_like(z_train_np)
    for c in range(N_CLASSES):
        indices = (y_train_np == c)
        z_centered[indices] = z_train_np[indices] - class_means[c]

    cov = np.cov(z_centered, rowvar=False)
    # Add a small regularizer to make it invertible
    cov_reg = cov + np.eye(cov.shape[0]) * 1e-4
    cov_inv = np.linalg.inv(cov_reg)

    # 4. Compute 5 detection features for evaluation mix
    print("\nExtracting 5 detection features on the 50/50 mix...")

    # Initialize feature arrays
    feat_squeezing = []
    feat_mc_entropy = []
    feat_ens_disagree = []
    feat_kde = []
    feat_mahalanobis = []

    # Batch process to compute features
    batch_size = 100
    for i in range(0, 1000, batch_size):
        xb = x_eval[i:i+batch_size]
        yb = y_eval[i:i+batch_size]

        # Feature 1: Feature Squeezing L1
        # Convert to 2-bit (3 levels: 0.0, 0.5, 1.0)
        xb_squeezed = torch.round(xb * 2.0) / 2.0
        with torch.no_grad():
            log_orig = model0(xb)
            log_sq = model0(xb_squeezed)
            l1_diff = F.l1_loss(log_orig, log_sq, reduction='none').sum(dim=1)
            feat_squeezing.append(l1_diff.cpu())

        # Feature 2: MC-Dropout Entropy
        model0.train()  # Active dropout
        mc_probs = []
        with torch.no_grad():
            for _ in range(10):
                mc_probs.append(F.softmax(model0(xb), dim=1))
        mean_mc_prob = torch.stack(mc_probs).mean(0)
        mc_ent = - (mean_mc_prob * torch.log(mean_mc_prob + 1e-9)).sum(dim=1)
        feat_mc_entropy.append(mc_ent.cpu())
        model0.eval()  # Restore eval

        # Feature 3: Ensemble Disagreement
        with torch.no_grad():
            p0 = F.softmax(model0(xb), dim=1)
            p1 = F.softmax(model1(xb), dim=1)
            p2 = F.softmax(model2(xb), dim=1)
        stacked_p = torch.stack([p0, p1, p2])  # (3, batch, 10)
        disagree = stacked_p.var(dim=0).mean(dim=1)  # average class variance across models
        feat_ens_disagree.append(disagree.cpu())

        # Feature 4 & 5: Representation Space features (KDE & Mahalanobis)
        _, zb = model0.forward_representation(xb)
        zb_np = zb.cpu().numpy()

        # KDE score
        kde_scores = kde.score_samples(zb_np)
        feat_kde.extend(kde_scores)

        # Mahalanobis distance to nearest class mean
        with torch.no_grad():
            preds = model0(xb).argmax(dim=1).cpu().numpy()

        m_dists = []
        for j in range(xb.size(0)):
            c_pred = preds[j]
            z_j = zb_np[j]
            mean_j = class_means[c_pred]
            diff = z_j - mean_j
            m_dist = diff.dot(cov_inv).dot(diff)
            m_dists.append(m_dist)
        feat_mahalanobis.extend(m_dists)

    # Concatenate and normalize features
    feat_squeezing = torch.cat(feat_squeezing).numpy()
    feat_mc_entropy = torch.cat(feat_mc_entropy).numpy()
    feat_ens_disagree = torch.cat(feat_ens_disagree).numpy()
    feat_kde = np.array(feat_kde)
    feat_mahalanobis = np.array(feat_mahalanobis)

    # Stack features into matrix: shape (1000, 5)
    X = np.stack([
        feat_squeezing,
        feat_mc_entropy,
        feat_ens_disagree,
        feat_kde,
        feat_mahalanobis
    ], axis=1)

    feature_names = [
        "Feature-Squeezing L1",
        "MC-Dropout Entropy",
        "Ensemble Disagreement",
        "KDE-Density Log-Likelihood",
        "Mahalanobis Distance"
    ]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Univariate AUROC analysis for each feature
    print("\n" + "=" * 60)
    print("Individual Detector AUROCs")
    print("=" * 60)
    individual_aurocs = {}
    for idx, fname in enumerate(feature_names):
        a = roc_auc_score(is_adv, X_scaled[:, idx])
        a_best = max(a, 1 - a)
        direction = "Adversarial has HIGHER values" if a >= 0.5 else "Adversarial has LOWER values"
        print(f"  {fname:<30}: AUROC = {a_best:.4f} ({direction})")
        individual_aurocs[fname] = a_best

    # 6. Train Meta-Classifier and perform K-Fold Cross-Validation
    print("\n" + "=" * 60)
    print("Meta-Detector Logistic Regression Analysis")
    print("=" * 60)
    lr = LogisticRegression(random_state=42)

    # Evaluate via 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr, X_scaled, is_adv, cv=kf, scoring='roc_auc')

    print(f"  5-Fold Cross-Validated AUROC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Train final model on all data to inspect coefficients
    lr.fit(X_scaled, is_adv)
    print("\n  Logistic Regression Standardized Coefficients:")
    for fname, coef in zip(feature_names, lr.coef_[0]):
        print(f"    {fname:<30}: {coef:+.4f}")

    print("\nComparison Summary:")
    print(f"  Best Individual Detector: {max(individual_aurocs, key=individual_aurocs.get)} ({max(individual_aurocs.values()):.4f})")
    print(f"  Meta-Detector (Ensemble): {cv_scores.mean():.4f}")


if __name__ == "__main__":
    main()
