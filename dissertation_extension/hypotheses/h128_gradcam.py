"""
Hypothesis H128: GradCAM attribution on the last conv layer predicts adversarial vulnerability.

Train small CNN matching diagnostic_test.py on Fashion-MNIST for 10 epochs.
Implement GradCAM (Selvaraju 2017) attribution on the last conv layer (c2).
Per sample compute features:
  - gradcam_entropy: entropy of normalized GradCAM heatmap
  - gradcam_max: maximum value in the GradCAM heatmap
  - gradcam_l2_to_image_center: L2 distance from GradCAM center of mass to the image center
  - gradcam_concentration: area of the top-50%-mass region in the heatmap
Plus baseline features:
  - margin: logit margin (top - 2nd logit)
  - mean_pix: mean pixel value
  - std_pix: standard deviation of pixel values

Targets:
  - flipped_fgsm: binary indicator of whether FGSM flips the prediction
  - flipped_pgd: binary indicator of whether PGD flips the prediction
  - min_eps: minimum epsilon to flip via FGSM binary search

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


class CNN(nn.Module):
    """Small CNN matching diagnostic_test.py with activation capture for GradCAM."""
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)
        self.c2_activation = None

    def forward(self, x):
        x = F.relu(self.c1(x))
        self.c2_activation = F.relu(self.c2(x))
        x = F.max_pool2d(self.c2_activation, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


def train_model(train_set, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN(10).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} done")
    model.eval()
    return model


def fgsm_grad(model, x, y):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return grad.sign().detach()


def attack_fgsm(model, x, y, eps=EPS):
    sign = fgsm_grad(model, x, y)
    adv = (x + eps * sign).clamp(0, 1)
    with torch.no_grad():
        return (model(adv).argmax(1) != y)


def attack_pgd(model, x, y, eps=EPS, alpha=2.0/255.0, steps=10):
    x_adv = x.clone().detach().requires_grad_(True)
    for _ in range(steps):
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = x_adv.clamp(0.0, 1.0).detach().requires_grad_(True)
    with torch.no_grad():
        return (model(x_adv).argmax(1) != y)


def min_eps_to_flip(model, x, y, eps_max=0.3, iters=15):
    lo = torch.zeros(x.size(0), device=DEVICE)
    hi = torch.full((x.size(0),), eps_max, device=DEVICE)
    sign = fgsm_grad(model, x, y)
    for _ in range(iters):
        mid = (lo + hi) / 2
        adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0, 1)
        with torch.no_grad():
            flipped = model(adv).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


def compute_gradcam(model, x, y):
    """Computes GradCAM heatmap on c2 layer of model."""
    model.eval()
    x_var = x.clone().detach()
    logits = model(x_var)
    act = model.c2_activation
    score = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    grads = torch.autograd.grad(score.sum(), act, retain_graph=False)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(28, 28), mode='bilinear', align_corners=False)
    return cam.detach()


def auroc_both_directions(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.5
    a = roc_auc_score(y_true, y_score)
    return max(a, 1.0 - a)


def main():
    print("Loading Fashion-MNIST dataset...")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)

    print("Training CNN model...")
    model = train_model(train_set)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
    N = test_x.size(0)

    # Filter to correctly predicted test samples
    with torch.no_grad():
        preds = []
        for i in range(0, N, 512):
            preds.append(model(test_x[i:i+512]).argmax(1))
        preds = torch.cat(preds)
    correct = (preds == test_y)
    x_c = test_x[correct]
    y_c = test_y[correct]
    N_c = x_c.size(0)
    print(f"Correctly classified samples: {N_c}/{N}")

    # Compute targets in batches
    print("Computing attack targets...")
    flipped_fgsm_list = []
    flipped_pgd_list = []
    min_eps_list = []
    for i in range(0, N_c, 512):
        xb = x_c[i:i+512]
        yb = y_c[i:i+512]
        flipped_fgsm_list.append(attack_fgsm(model, xb, yb))
        flipped_pgd_list.append(attack_pgd(model, xb, yb))
        min_eps_list.append(min_eps_to_flip(model, xb, yb))

    y_fgsm = torch.cat(flipped_fgsm_list).cpu().numpy().astype(int)
    y_pgd = torch.cat(flipped_pgd_list).cpu().numpy().astype(int)
    y_mineps = torch.cat(min_eps_list).cpu().numpy()

    # Compute features in batches
    print("Computing features...")
    gradcam_entropy = []
    gradcam_max = []
    gradcam_l2 = []
    gradcam_concentration = []
    margin_list = []
    mean_pix_list = []
    std_pix_list = []

    grid_y, grid_x = torch.meshgrid(torch.arange(28, device=DEVICE), torch.arange(28, device=DEVICE), indexing='ij')
    grid_x = grid_x.float()
    grid_y = grid_y.float()

    for i in range(0, N_c, 512):
        xb = x_c[i:i+512]
        yb = y_c[i:i+512]
        
        # GradCAM
        cam = compute_gradcam(model, xb, yb) # (B, 1, 28, 28)
        B_curr = cam.size(0)
        
        # entropy
        p = cam / (cam.sum(dim=(2, 3), keepdim=True) + 1e-12)
        ent = -(p * torch.log(p + 1e-12)).sum(dim=(2, 3)).squeeze(1)
        gradcam_entropy.append(ent)
        
        # max
        mx = cam.view(B_curr, -1).max(dim=1).values
        gradcam_max.append(mx)
        
        # l2 distance to center
        x_com = (p.squeeze(1) * grid_x).sum(dim=(1, 2))
        y_com = (p.squeeze(1) * grid_y).sum(dim=(1, 2))
        dist = torch.sqrt((x_com - 13.5)**2 + (y_com - 13.5)**2)
        gradcam_l2.append(dist)
        
        # concentration (fraction of area of top-50%-mass region)
        flat_cam = cam.view(B_curr, -1)
        sorted_cam, _ = flat_cam.sort(dim=1, descending=True)
        total_mass = sorted_cam.sum(dim=1, keepdim=True)
        cum_mass = sorted_cam.cumsum(dim=1)
        is_under = cum_mass < 0.5 * total_mass
        num_pixels = is_under.sum(dim=1) + 1
        gradcam_concentration.append(num_pixels.float() / 784.0)

        # Baseline features
        with torch.no_grad():
            logits = model(xb)
            sorted_logits, _ = logits.sort(dim=1, descending=True)
            margin = sorted_logits[:, 0] - sorted_logits[:, 1]
            margin_list.append(margin)
            
            flat_x = xb.view(B_curr, -1)
            mean_pix_list.append(flat_x.mean(dim=1))
            std_pix_list.append(flat_x.std(dim=1))

    feats = {
        "gradcam_entropy": torch.cat(gradcam_entropy).cpu().numpy(),
        "gradcam_max": torch.cat(gradcam_max).cpu().numpy(),
        "gradcam_l2_to_image_center": torch.cat(gradcam_l2).cpu().numpy(),
        "gradcam_concentration": torch.cat(gradcam_concentration).cpu().numpy(),
        "margin": torch.cat(margin_list).cpu().numpy(),
        "mean_pix": torch.cat(mean_pix_list).cpu().numpy(),
        "std_pix": torch.cat(std_pix_list).cpu().numpy(),
    }

    # Evaluate Univariate AUROC
    print("\n" + "="*50)
    print("UNIVARIATE AUROC RESULTS")
    print("="*50)
    
    for feat_name, feat_val in feats.items():
        print(f"\nFeature: {feat_name}")
        auc_fgsm = auroc_both_directions(y_fgsm, feat_val)
        auc_pgd = auroc_both_directions(y_pgd, feat_val)
        # For continuous target min_eps, we partition at the median to compute AUROC
        median_eps = np.median(y_mineps)
        y_mineps_bin = (y_mineps <= median_eps).astype(int)
        auc_mineps = auroc_both_directions(y_mineps_bin, feat_val)
        
        print(f"  Target: flipped_fgsm  AUROC = {auc_fgsm:.4f}")
        print(f"  Target: flipped_pgd   AUROC = {auc_pgd:.4f}")
        print(f"  Target: min_eps       AUROC = {auc_mineps:.4f}")


if __name__ == "__main__":
    main()
