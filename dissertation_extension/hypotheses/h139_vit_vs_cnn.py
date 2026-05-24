"""
Hypothesis H139: Attention vs CNN adversarial vulnerability comparison on Fashion-MNIST.

Train a CNN (matching diagnostic_test.py) and a small Vision Transformer (ViT, 4 blocks, 8 heads, patch 4)
on Fashion-MNIST for 10 epochs each.
Evaluate both architectures on the same correctly classified test samples.

Features:
  - margin (from victim model logits)
  - mean_pix
  - std_pix
  - sobel_mean

Targets (per victim):
  - flipped_FGSM
  - flipped_PGD
  - min_eps (min epsilon to flip via binary search)

Analysis:
  - Univariate AUROC for both models.
  - Cross-correlation of FGSM vulnerability (binary flip status) between CNN and ViT.
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

# ---- Sobel Filters ----
SOBEL_X = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1., -2., -1.],
                        [0., 0., 0.],
                        [1., 2., 1.]]).view(1, 1, 3, 3)


# ---- CNN Model ----
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


# ---- ViT Model ----
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)  # (B, NumPatches, d)


class ViTBlock(nn.Module):
    def __init__(self, dim=64, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    """Small ViT: 4 blocks, 8 heads, patch 4."""
    def __init__(self, patch_size=4, in_chans=1, num_classes=10, embed_dim=64, depth=4, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 28x28 image has 7x7 = 49 patches. Plus cls_token is 50.
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, embed_dim))
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Init pos_embed and cls_token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


# ---- Training Loops ----
def train_cnn(train_set, test_set, seed=0):
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
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
    return model


def train_vit(train_set, test_set, seed=0):
    torch.manual_seed(seed + 1)
    np.random.seed(seed + 1)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = ViT(patch_size=4, in_chans=1, num_classes=N_CLASSES, embed_dim=64, depth=4, num_heads=8).to(DEVICE)
    # ViT usually benefits from slightly lower learning rates and warmups, but for 10 epochs 5e-4 is standard.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
    return model


# ---- Features & Attacks ----
def compute_image_stats(x):
    N = x.size(0)
    flat_x = x.view(N, -1)
    mean_pix = flat_x.mean(1).cpu()
    std_pix = flat_x.std(1).cpu()

    # Sobel
    sx = SOBEL_X.to(x.device)
    sy = SOBEL_Y.to(x.device)
    gx = F.conv2d(x, sx, padding=1)
    gy = F.conv2d(x, sy, padding=1)
    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    sobel_mean = grad_mag.mean(dim=(1, 2, 3)).cpu()

    return mean_pix, std_pix, sobel_mean


def fgsm_attack(model, x, y, eps=EPS):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    sign = x_adv.grad.sign().detach()
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def pgd_attack(model, x, y, eps=EPS, alpha=2.0/255.0, steps=10):
    adv = x.clone().detach().requires_grad_(True)
    adv = adv + torch.FloatTensor(*adv.shape).uniform_(-eps, eps).to(DEVICE)
    adv = adv.clamp(0, 1)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        loss.backward()
        adv = (adv + alpha * adv.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    N = x.size(0)
    lo = torch.zeros(N, device=DEVICE)
    hi = torch.full((N,), eps_max, device=DEVICE)

    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    sign = x_adv.grad.sign().detach()

    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = (model(adv).argmax(1) != y)
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def main():
    print("=" * 60)
    print("Hypothesis H139: Attention (ViT) vs CNN Adversarial Vulnerability")
    print("=" * 60)

    # Load Fashion-MNIST
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    # Train CNN
    print("\nTraining CNN model...")
    t0 = time.time()
    cnn_model = train_cnn(train_set, test_set)
    print(f"CNN trained in {time.time() - t0:.1f}s")

    # Train ViT
    print("\nTraining ViT model...")
    t0 = time.time()
    vit_model = train_vit(train_set, test_set)
    print(f"ViT trained in {time.time() - t0:.1f}s")

    # Stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    # Get predictions
    cnn_model.eval()
    vit_model.eval()
    with torch.no_grad():
        cnn_pred = cnn_model(test_x).argmax(1)
        vit_pred = vit_model(test_x).argmax(1)

        # Restrict to samples correctly classified by BOTH
        correct_both = (cnn_pred == test_y) & (vit_pred == test_y)

    x_c = test_x[correct_both]
    y_c = test_y[correct_both]
    N_correct = correct_both.sum().item()
    print(f"\nUsing {N_correct}/{test_x.size(0)} samples correctly classified by both models")

    # Image-level features
    mean_pix, std_pix, sobel_mean = compute_image_stats(x_c)

    # CNN specific features and targets
    print("\nRunning attacks on CNN...")
    with torch.no_grad():
        cnn_logits = cnn_model(x_c)
        sorted_cnn_logits, _ = cnn_logits.sort(1, descending=True)
        cnn_margin = (sorted_cnn_logits[:, 0] - sorted_cnn_logits[:, 1]).cpu()

    cnn_flipped_fgsm = []
    cnn_flipped_pgd = []
    cnn_min_eps = []
    for i in range(0, N_correct, 512):
        bx = x_c[i:i+512]
        by = y_c[i:i+512]
        cnn_flipped_fgsm.append(fgsm_attack(cnn_model, bx, by))
        cnn_flipped_pgd.append(pgd_attack(cnn_model, bx, by))
        cnn_min_eps.append(min_eps_to_flip(cnn_model, bx, by))

    cnn_flipped_fgsm = torch.cat(cnn_flipped_fgsm).cpu().numpy().astype(int)
    cnn_flipped_pgd = torch.cat(cnn_flipped_pgd).cpu().numpy().astype(int)
    cnn_min_eps = torch.cat(cnn_min_eps).cpu().numpy()

    # ViT specific features and targets
    print("Running attacks on ViT...")
    with torch.no_grad():
        vit_logits = vit_model(x_c)
        sorted_vit_logits, _ = vit_logits.sort(1, descending=True)
        vit_margin = (sorted_vit_logits[:, 0] - sorted_vit_logits[:, 1]).cpu()

    vit_flipped_fgsm = []
    vit_flipped_pgd = []
    vit_min_eps = []
    for i in range(0, N_correct, 512):
        bx = x_c[i:i+512]
        by = y_c[i:i+512]
        vit_flipped_fgsm.append(fgsm_attack(vit_model, bx, by))
        vit_flipped_pgd.append(pgd_attack(vit_model, bx, by))
        vit_min_eps.append(min_eps_to_flip(vit_model, bx, by))

    vit_flipped_fgsm = torch.cat(vit_flipped_fgsm).cpu().numpy().astype(int)
    vit_flipped_pgd = torch.cat(vit_flipped_pgd).cpu().numpy().astype(int)
    vit_min_eps = torch.cat(vit_min_eps).cpu().numpy()

    # Print vulnerability statistics
    print(f"\nCNN: FGSM={cnn_flipped_fgsm.mean():.3f}, PGD-10={cnn_flipped_pgd.mean():.3f}, mean min_eps={cnn_min_eps.mean():.3f}")
    print(f"ViT: FGSM={vit_flipped_fgsm.mean():.3f}, PGD-10={vit_flipped_pgd.mean():.3f}, mean min_eps={vit_min_eps.mean():.3f}")

    # Compute Cross-Correlation of FGSM flip
    cross_corr = np.corrcoef(cnn_flipped_fgsm, vit_flipped_fgsm)[0, 1]
    print(f"\nCross-correlation of FGSM vulnerability (flip status): {cross_corr:.4f}")

    # Univariate AUROCs
    feature_names = ["margin", "mean_pix", "std_pix", "sobel_mean"]

    # CNN Analysis
    print("\n" + "=" * 60)
    print("CNN VULNERABILITY ANALYSIS (Univariate AUROC)")
    print("=" * 60)
    cnn_feats = torch.stack([cnn_margin, mean_pix, std_pix, sobel_mean], 1).numpy()
    for target_name, y in [("flipped_FGSM", cnn_flipped_fgsm), ("flipped_PGD", cnn_flipped_pgd)]:
        print(f"\nTarget: {target_name}")
        if y.std() == 0:
            print("  Constant target, skipping.")
            continue
        for i, fname in enumerate(feature_names):
            a = roc_auc_score(y, cnn_feats[:, i])
            a = max(a, 1 - a)
            print(f"  {fname:<15} AUROC = {a:.4f}")

    # ViT Analysis
    print("\n" + "=" * 60)
    print("ViT VULNERABILITY ANALYSIS (Univariate AUROC)")
    print("=" * 60)
    vit_feats = torch.stack([vit_margin, mean_pix, std_pix, sobel_mean], 1).numpy()
    for target_name, y in [("flipped_FGSM", vit_flipped_fgsm), ("flipped_PGD", vit_flipped_pgd)]:
        print(f"\nTarget: {target_name}")
        if y.std() == 0:
            print("  Constant target, skipping.")
            continue
        for i, fname in enumerate(feature_names):
            a = roc_auc_score(y, vit_feats[:, i])
            a = max(a, 1 - a)
            print(f"  {fname:<15} AUROC = {a:.4f}")


if __name__ == "__main__":
    main()
