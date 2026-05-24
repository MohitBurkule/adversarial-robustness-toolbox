"""
H83: Thermometer-encoded input defence (Buckman et al., ICLR 2018).

Hypothesis: discretising each pixel into L=16 thermometer levels (cumulative-sum
one-hot encoding) recovers a per-sample-different set of adversarial examples
back to the correct label. We test whether per-sample features
(final_margin, mean_pix, std_pix) predict which adversarial samples are
"recovered" (i.e. correctly classified after the defence is applied).

Pipeline:
  1. Train small CNN on Fashion-MNIST for 10 epochs (architecture matches
     diagnostic_test.py).
  2. Train a second CNN that takes thermometer-encoded inputs (L=16 channels).
  3. Generate FGSM adversarial examples on the standard CNN at eps=15/255.
  4. For each sample originally correctly classified by the standard CNN AND
     successfully flipped by FGSM: apply thermometer discretisation and pass
     the encoded adv example to the thermometer CNN. "Recovered" = thermometer
     CNN classifies it correctly.
  5. Compute three per-sample features:
        - margin   : final-model logit margin on the clean input
        - mean_pix : mean pixel intensity of the clean image
        - std_pix  : std pixel intensity of the clean image
  6. Univariate AUROC for predicting recovery from each feature.
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
EPS_TEST = 15.0 / 255.0
EPOCHS = 10
BATCH = 128
L = 16  # thermometer levels


# ----------------------- models -----------------------
class CNN(nn.Module):
    """Standard 1-channel CNN matching diagnostic_test.py."""
    def __init__(self, in_ch=1, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 32, 3)
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


# ----------------------- thermometer encoding -----------------------
def thermometer_encode(x, levels=L):
    """Cumulative-sum thermometer encoding.

    For each pixel value v in [0,1], discretise to level k = floor(v * levels),
    then emit a length-`levels` vector where the first (k+1) entries are 1 and
    the rest are 0. Equivalently: channel c is 1 iff floor(v*L) >= c.

    Input x:  (N, 1, H, W)  in [0,1]
    Output:   (N, L, H, W)  in {0,1}
    """
    # discretise to integer level in [0, levels-1]
    k = torch.clamp((x * levels).floor().long(), 0, levels - 1)  # (N,1,H,W)
    # build (N, L, H, W) where channel c is 1 iff k >= c
    c = torch.arange(levels, device=x.device).view(1, levels, 1, 1)
    enc = (k >= c).float()
    return enc


# ----------------------- training -----------------------
def train_model(model, train_loader, epochs=EPOCHS, encode=False):
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if encode:
                x = thermometer_encode(x, L)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{epochs} ({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ----------------------- FGSM on standard CNN -----------------------
def fgsm_attack(model, x, y, eps=EPS_TEST):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    sign = x.grad.sign().detach()
    adv = (x.detach() + eps * sign).clamp(0, 1)
    return adv


# ----------------------- evaluation helpers -----------------------
def predict(model, x, batch=512, encode=False):
    preds = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            xb = x[i:i+batch]
            if encode:
                xb = thermometer_encode(xb, L)
            preds.append(model(xb).argmax(1))
    return torch.cat(preds)


def logits_on(model, x, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            out.append(model(x[i:i+batch]))
    return torch.cat(out, 0)


# ----------------------- main -----------------------
def main():
    print("Loading Fashion-MNIST ...")
    tf = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)

    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)

    print("Training standard CNN (1 channel) ...")
    torch.manual_seed(0); np.random.seed(0)
    model_std = CNN(in_ch=1, n=10)
    train_model(model_std, train_loader, EPOCHS, encode=False)

    print("Training thermometer CNN (L=16 channels) ...")
    torch.manual_seed(1); np.random.seed(1)
    model_therm = CNN(in_ch=L, n=10)
    train_model(model_therm, train_loader, EPOCHS, encode=True)

    # ---- clean accuracy
    clean_pred = predict(model_std, test_x, encode=False)
    therm_clean_pred = predict(model_therm, test_x, encode=True)
    print(f"Standard clean acc:    {(clean_pred == test_y).float().mean().item():.4f}")
    print(f"Thermometer clean acc: {(therm_clean_pred == test_y).float().mean().item():.4f}")

    # ---- restrict to samples standard model gets right
    correct = clean_pred == test_y
    x_c, y_c = test_x[correct], test_y[correct]
    print(f"Using {x_c.size(0)} samples standard CNN classifies correctly")

    # ---- generate FGSM adv on standard CNN at eps=15/255
    print("Generating FGSM adv at eps=15/255 ...")
    adv = []
    for i in range(0, x_c.size(0), 512):
        adv.append(fgsm_attack(model_std, x_c[i:i+512], y_c[i:i+512], EPS_TEST))
    x_adv = torch.cat(adv, 0)

    # adversarially fooled the standard model?
    adv_pred_std = predict(model_std, x_adv, encode=False)
    fooled = adv_pred_std != y_c
    print(f"FGSM success rate on standard CNN: {fooled.float().mean().item():.4f}")

    # restrict to samples that WERE actually flipped — these are the ones the
    # defence has a chance to "recover"
    x_adv_f = x_adv[fooled]
    x_clean_f = x_c[fooled]
    y_f = y_c[fooled]
    print(f"Restricting to {x_adv_f.size(0)} successfully flipped adv samples")

    # ---- apply thermometer defence: classify encoded adv with thermometer CNN
    therm_pred_adv = predict(model_therm, x_adv_f, encode=True)
    recovered = (therm_pred_adv == y_f).cpu().numpy().astype(int)
    print(f"Recovery rate via thermometer defence: {recovered.mean():.4f}")

    if recovered.std() == 0:
        print("Recovery is degenerate (all 0 or all 1); univariate AUROC undefined.")
        return

    # ---- per-sample features on the CLEAN image
    # margin on standard model
    logits_clean = logits_on(model_std, x_clean_f)
    sorted_l, _ = logits_clean.sort(1, descending=True)
    margin = (sorted_l[:, 0] - sorted_l[:, 1]).cpu().numpy()
    mean_pix = x_clean_f.view(x_clean_f.size(0), -1).mean(1).cpu().numpy()
    std_pix = x_clean_f.view(x_clean_f.size(0), -1).std(1).cpu().numpy()

    feats = {"margin": margin, "mean_pix": mean_pix, "std_pix": std_pix}

    print(f"\nUnivariate AUROC for predicting per-sample recovery "
          f"(pos rate = {recovered.mean():.3f}, n={len(recovered)}):")
    for name, f in feats.items():
        auc = roc_auc_score(recovered, f)
        auc = max(auc, 1 - auc)
        print(f"  {name:<10} AUROC = {auc:.4f}")


if __name__ == "__main__":
    main()
