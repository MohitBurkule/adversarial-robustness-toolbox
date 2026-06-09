"""
Hypothesis H107: Targeted FGSM vulnerability analysis on Fashion-MNIST.

Train small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
For each test sample run targeted FGSM toward each of 9 wrong classes at eps=15/255.

Record:
  - easiest_target_class: which wrong class had lowest loss at epsilon (easiest to flip to)
  - hardest_target_class: which wrong class had highest loss (hardest to flip to)
  - mean_loss_at_target: average loss across all targets

Features (same as diagnostic_test.py):
  - margin: logit margin (top - 2nd logit)
  - mean_pix: mean pixel value per sample (lightness)
  - std_pix: standard deviation of pixels per sample (contrast)

Targets:
  - flipped_to_any_target: binary (1 if sample successfully flipped to at least one target)
  - mean_targeted_success_rate: proportion of 9 successful targets

Compare with untargeted FGSM (single-step attack at same epsilon).

Univariate AUROC for each feature-target pair.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
N_CLASSES = 10


class CNN(nn.Module):
    """Small CNN matching diagnostic_test.py architecture."""
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


def train_model(train_set, test_set, seed=0):
    """Train CNN on Fashion-MNIST for specified epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

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
        print(f"  Epoch {epoch+1}/{EPOCHS} completed")

    return model, test_x, test_y


def compute_features(test_x, model):
    """Compute features for test samples: margin, mean_pix, std_pix."""
    N = test_x.size(0)

    with torch.no_grad():
        logits = []
        for i in range(0, N, 512):
            logits.append(model(test_x[i:i+512]))
        logits = torch.cat(logits, 0)

    sorted_logits, _ = logits.sort(1, descending=True)
    margin = (sorted_logits[:, 0] - sorted_logits[:, 1])

    flat_x = test_x.view(N, -1)
    mean_pix = flat_x.mean(1)
    std_pix = flat_x.std(1)

    features = torch.stack([margin, mean_pix, std_pix], 1)
    return features, model


def fgsm_grad_sign(model, x, y, eps=EPS):
    """Compute gradient sign for FGSM."""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    return x_adv.grad.sign().detach()


def evaluate_targeted_attacks(model, x_c, y_c, eps=EPS):
    """
    For each test sample, run targeted FGSM toward all 9 wrong classes.

    Returns:
        - easiest_target_class: which wrong class had lowest loss (highest success)
        - hardest_target_class: which wrong class had highest loss (lowest success)
        - mean_loss_at_target: average loss across all 9 targets
        - success_matrix: (N, 9) boolean matrix of successful flips
    """
    N = x_c.size(0)

    easiest_target = torch.zeros(N, dtype=torch.long, device=DEVICE)
    hardest_target = torch.zeros(N, dtype=torch.long, device=DEVICE)
    mean_loss_at_target = torch.zeros(N, device=DEVICE)
    success_matrix = torch.zeros(N, 9, dtype=torch.bool, device=DEVICE)

    model.eval()

    losses_all = torch.zeros(N, N_CLASSES, device=DEVICE)
    success_all = torch.zeros(N, N_CLASSES, dtype=torch.bool, device=DEVICE)

    for target_class in range(N_CLASSES):
        # We only attack samples where y_c != target_class
        mask = (y_c != target_class)
        if not mask.any():
            continue

        x_batch = x_c[mask]

        # Run targeted FGSM: minimize loss for target_class
        x_adv = x_batch.clone().detach().requires_grad_(True)
        target_tensor = torch.full((x_batch.size(0),), target_class, dtype=torch.long, device=DEVICE)
        loss = F.cross_entropy(model(x_adv), target_tensor)
        loss.backward()
        grad = x_adv.grad.sign().detach()
        x_pert = (x_batch - eps * grad).clamp(0, 1)

        with torch.no_grad():
            outputs = model(x_pert)
            new_pred = outputs.argmax(dim=1)
            success = (new_pred == target_class)
            loss_vals = F.cross_entropy(outputs, target_tensor, reduction='none')

        losses_all[mask, target_class] = loss_vals
        success_all[mask, target_class] = success

    # Now, extract the 9 wrong targets for each sample
    for i in range(N):
        true_class = y_c[i].item()
        wrong_indices = [c for c in range(N_CLASSES) if c != true_class]

        sample_losses = losses_all[i, wrong_indices]
        sample_success = success_all[i, wrong_indices]

        success_matrix[i] = sample_success
        mean_loss_at_target[i] = sample_losses.mean()

        easiest_idx = sample_losses.argmin().item()
        easiest_target[i] = wrong_indices[easiest_idx]

        hardest_idx = sample_losses.argmax().item()
        hardest_target[i] = wrong_indices[hardest_idx]

    return easiest_target, hardest_target, mean_loss_at_target, success_matrix


def compute_untargeted_fgsm(model, x_c, y_c, eps=EPS):
    """Untargeted FGSM baseline for comparison."""
    sign = fgsm_grad_sign(model, x_c, y_c, eps)
    adv = (x_c + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y_c)


def main():
    print("=" * 60)
    print("Hypothesis H107: Targeted FGSM Vulnerability Analysis")
    print("=" * 60)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Train model
    print("\nTraining model for 10 epochs...")
    t0 = time.time()
    model, test_x, test_y = train_model(train_set, test_set)
    print(f"Training completed in {time.time() - t0:.1f}s")

    # Compute features
    print("\nComputing features...")
    features, _ = compute_features(test_x, model)

    # Restrict to correctly classified samples
    with torch.no_grad():
        logits = model(test_x)
        pred = logits.argmax(1)
        correct = (pred == test_y)
    x_c, y_c = test_x[correct], test_y[correct]
    features_c = features[correct]
    N_correct = correct.sum().item()
    print(f"Using {N_correct}/{test_x.size(0)} correctly classified samples")

    # Run targeted FGSM attacks
    print("\nRunning targeted FGSM attacks (9 target classes per sample)...")
    t0 = time.time()

    easiest_target, hardest_target, mean_loss_at_target, success_matrix = evaluate_targeted_attacks(
        model, x_c, y_c, EPS
    )

    print(f"Completed in {time.time() - t0:.1f}s")

    # Compute per-sample metrics
    success_per_sample = success_matrix.float().sum(1)  # Number of successful target attacks

    # Compute targets
    flipped_to_any_target = (success_per_sample > 0).long()
    mean_targeted_success_rate = success_per_sample / 9.0

    print(f"\nTargeted attack statistics:")
    print(f"  Flipped to at least one target: {flipped_to_any_target.float().mean():.3f}")
    print(f"  Mean success rate across 9 targets: {mean_targeted_success_rate.float().mean():.3f}")

    # Compute untargeted FGSM baseline
    print(f"\nComparing with untargeted FGSM (single step)...")
    untargeted_flip = compute_untargeted_fgsm(model, x_c, y_c, EPS)

    print(f"Untargeted attack success rate: {untargeted_flip.float().mean():.3f}")

    # Stack features for analysis
    all_feats = features_c
    feature_names = ["margin", "mean_pix", "std_pix"]

    # Univariate AUROC for targeted features
    print("\n" + "=" * 60)
    print("TARGETED FGSM: Univariate AUROC")
    print("=" * 60)
    for i, fname in enumerate(feature_names):
        y = flipped_to_any_target.cpu().numpy()
        if y.std() == 0:
            continue
        x_i = all_feats[:, i].cpu().numpy()
        a = roc_auc_score(y, x_i)
        a = max(a, 1 - a)
        print(f"  {fname:<15} AUROC = {a:.4f} (raw = {roc_auc_score(y, x_i):.4f})")

    # Univariate AUROC for untargeted FGSM for comparison
    print("\n" + "=" * 60)
    print("UNTARGETED FGSM: Univariate AUROC (baseline)")
    print("=" * 60)
    y_untarget = untargeted_flip.cpu().numpy().astype(int)
    if y_untarget.std() == 0:
        print("  Untargeted FGSM: all samples flip or none flip")
    else:
        print(f"Untargeted FGSM: {y_untarget.mean():.3f} flip rate")
        print(f"  {'feature':<15} {'AUROC':>8} {'Direction':<10}")
        for i, fname in enumerate(feature_names):
            x_i = all_feats[:, i].cpu().numpy()
            a = roc_auc_score(y_untarget, x_i)
            a = max(a, 1 - a)
            print(f"  {fname:<15} AUROC = {a:.4f} (raw = {roc_auc_score(y_untarget, x_i):.4f})")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Number of correctly classified samples: {N_correct}")
    print(f"Targeted FGSM - flipped to at least one target: {flipped_to_any_target.float().mean():.3f}")
    print(f"Targeted FGSM - high success rate (>median): {(mean_targeted_success_rate > mean_targeted_success_rate.median()).float().mean():.3f}")
    print(f"Untargeted FGSM (baseline): {y_untarget.mean():.3f}")

    print("\nKey finding:")
    print(f"  Success rate distribution:")
    values, counts = torch.unique(success_per_sample, return_counts=True)
    for val, cnt in zip(values, counts):
        print(f"    {int(val)} targets flipped: {cnt} ({cnt/N_correct*100:.1f}%)")


if __name__ == "__main__":
    main()
