"""
Where does untargeted FGSM actually send each sample?
For every sample in a fixed eval batch we track during training:
  - per-step predicted class (only at end of each epoch to limit cost)
  - first epoch the sample was predicted correctly (learning_epoch)
  - most-frequent wrong class during training
At the end we run untargeted FGSM on the final model and ask:
  - which class does FGSM land on?
  - how often does that class == 2nd-best final logit / first-epoch wrong pred /
    most-frequent training confusion?
  - does the answer depend on learning_epoch (i.e. sample difficulty)?
"""
import time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda")
EPS = 15.0 / 255.0
BATCH = 128
EPOCHS = 8                  # longer so learning-speed differences exist
N_FIXED = 2000              # bigger eval batch for statistics
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*12*12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2); x = self.do1(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.do2(x); return self.fc2(x)


def main():
    tf = transforms.ToTensor()
    train = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train, BATCH, shuffle=True, num_workers=2)

    idx = torch.randperm(len(test))[:N_FIXED]
    fixed_x = torch.stack([test[i][0] for i in idx]).to(DEVICE)
    fixed_y = torch.tensor([test[i][1] for i in idx]).to(DEVICE)
    n_classes = 10

    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    per_epoch_pred = []          # [E, N]
    confusion_hist = torch.zeros(N_FIXED, n_classes, device=DEVICE)
    learning_epoch = torch.full((N_FIXED,), -1, device=DEVICE, dtype=torch.long)

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        # end-of-epoch snapshot on fixed batch
        model.eval()
        with torch.no_grad():
            pred = model(fixed_x).argmax(1)
        per_epoch_pred.append(pred.clone())
        wrong = pred != fixed_y
        if wrong.any():
            confusion_hist[torch.arange(N_FIXED, device=DEVICE)[wrong], pred[wrong]] += 1
        # first epoch model got each sample right
        right_now = (pred == fixed_y) & (learning_epoch == -1)
        learning_epoch[right_now] = ep
        print(f"  epoch {ep+1}/{EPOCHS} ({time.time()-t0:.1f}s)")

    model.eval()
    per_epoch_pred = torch.stack(per_epoch_pred)  # [E, N]

    # final model: logits, 2nd-best class, FGSM result class
    with torch.no_grad():
        final_logits = model(fixed_x)
        final_pred = final_logits.argmax(1)
        final_2nd = final_logits.masked_fill(
            F.one_hot(fixed_y, n_classes).bool(), -1e9).argmax(1)

    xa = fixed_x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xa), fixed_y).backward()
    x_adv = (fixed_x + EPS * xa.grad.sign()).clamp(0, 1).detach()
    with torch.no_grad():
        fgsm_pred = model(x_adv).argmax(1)

    fooled = (fgsm_pred != fixed_y) & (final_pred == fixed_y)  # restrict to samples we'd care about
    # "first-epoch wrong prediction" - earliest epoch the sample was predicted incorrectly
    first_wrong = torch.full((N_FIXED,), -1, device=DEVICE, dtype=torch.long)
    first_wrong_class = torch.full((N_FIXED,), -1, device=DEVICE, dtype=torch.long)
    for e in range(EPOCHS):
        miss = (per_epoch_pred[e] != fixed_y) & (first_wrong == -1)
        first_wrong[miss] = e
        first_wrong_class[miss] = per_epoch_pred[e][miss]
    # most-frequent confusion class (set to -1 if never confused)
    top_conf_class = confusion_hist.argmax(1)
    top_conf_class[confusion_hist.sum(1) == 0] = -1

    def frac(cond):
        return cond.float().mean().item()

    overall = {
        "n_samples": N_FIXED,
        "n_correctly_classified_clean": int((final_pred == fixed_y).sum()),
        "n_fooled_by_fgsm": int(fooled.sum()),
        "fgsm_lands_on_final_2nd_best": frac(fgsm_pred[fooled] == final_2nd[fooled]),
        "fgsm_lands_on_first_wrong_class": frac(
            (first_wrong_class[fooled] >= 0) & (fgsm_pred[fooled] == first_wrong_class[fooled])
        ),
        "fgsm_lands_on_top_training_confusion": frac(
            (top_conf_class[fooled] >= 0) & (fgsm_pred[fooled] == top_conf_class[fooled])
        ),
        "first_wrong_class_matches_final_2nd_best": frac(
            (first_wrong_class >= 0) & (first_wrong_class == final_2nd)
        ),
        "top_training_confusion_matches_final_2nd_best": frac(
            (top_conf_class >= 0) & (top_conf_class == final_2nd)
        ),
    }

    # bucket by learning_epoch on fooled samples only
    print("\n== Bucketed by learning_epoch (when model first got sample right) ==")
    print("learning_epoch | n | %FGSM->2ndbest | %FGSM->firstwrong | %FGSM->topconfusion | clean_acc")
    bucket_rows = []
    for le in range(-1, EPOCHS):
        mask = (learning_epoch == le)
        n = int(mask.sum())
        if n == 0: continue
        m_fooled = mask & fooled
        nf = int(m_fooled.sum())
        if nf == 0:
            row = (le, n, None, None, None, frac(final_pred[mask] == fixed_y[mask]))
        else:
            row = (le, n,
                   frac(fgsm_pred[m_fooled] == final_2nd[m_fooled]),
                   frac((first_wrong_class[m_fooled] >= 0) &
                        (fgsm_pred[m_fooled] == first_wrong_class[m_fooled])),
                   frac((top_conf_class[m_fooled] >= 0) &
                        (fgsm_pred[m_fooled] == top_conf_class[m_fooled])),
                   frac(final_pred[mask] == fixed_y[mask]))
        bucket_rows.append(row)
        s = lambda v: f"{v:.2%}" if v is not None else "n/a"
        label = "never" if le == -1 else f"ep {le}"
        print(f"  {label:>8} | {n:4d} | {s(row[2]):>12} | {s(row[3]):>15} | {s(row[4]):>17} | {row[5]:.2%}")

    print("\n== Overall ==")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    out = {"overall": overall, "buckets": bucket_rows}
    with open("analysis_targets.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
