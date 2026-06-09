"""
Replication of MSc thesis (Burkule 2022): FGSM adversarial training generalises to unseen PGD/BIM.
Extension: training-time FGSM attacks vs. final-model FGSM.

Dataset: Fashion-MNIST (drop-in for MNIST arch).
Two novel attacks:
  V1 (AccumSign): accumulate sign(grad_x L) on a fixed eval batch across every training step,
                  final perturbation = epsilon * sign(running_sum).
  V2 (TrajEnsemble): save per-epoch FGSM perturbation against the in-training model,
                  pick per-sample the perturbation that maximises loss on the FINAL model.
Baseline attack: vanilla FGSM on the final trained model.
"""
import copy, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 15.0 / 255.0           # match thesis epsilon=15 on [0,255], here normalised to [0,1]
BIM_STEP = 0.15 / 255.0 * 10 # small step ~ thesis 0.15-style; we use eps/10
PGD_STEPS = 10
BIM_STEPS = 10
BATCH = 128
EPOCHS = 5                    # Fashion-MNIST converges fast; keep small for runtime
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


class CNN(nn.Module):
    """5-layer CNN matching thesis MNIST architecture."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n_classes)
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


def get_loaders():
    tf = transforms.ToTensor()
    train = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=2),
            DataLoader(test, batch_size=256, shuffle=False, num_workers=2),
            train, test)


# ---------- attacks ----------
def fgsm(model, x, y, eps=EPS):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    g = torch.autograd.grad(loss, x)[0]
    return (x + eps * g.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=None, random_start=True):
    alpha = alpha or (2.5 * eps / steps)
    x0 = x.clone().detach()
    if random_start:
        x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    else:
        x_adv = x0.clone()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha * g.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)
    return x_adv.detach()


def bim(model, x, y, eps=EPS, steps=BIM_STEPS):
    return pgd(model, x, y, eps=eps, steps=steps, random_start=False)


# ---------- training ----------
def train_model(model, train_loader, adv_train=False, fixed_eval_x=None, fixed_eval_y=None):
    """Train; if fixed_eval_(x,y) given, also collect:
       - per-step sign(grad_x L) on that batch (for V1 AccumSign)
       - per-epoch FGSM perturbation on that batch (for V2 TrajEnsemble)
    """
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    accum_sign = None
    traj_perts = []  # list of (eval_batch,) eps perturbations, one per epoch
    n_steps = 0
    for ep in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if adv_train:
                model.eval()
                x_adv = fgsm(model, x, y)
                model.train()
                x = torch.cat([x, x_adv]); y = torch.cat([y, y])
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

            # collect trajectory signal on fixed eval batch
            if fixed_eval_x is not None:
                model.eval()
                xe = fixed_eval_x.clone().detach().requires_grad_(True)
                le = F.cross_entropy(model(xe), fixed_eval_y)
                g = torch.autograd.grad(le, xe)[0]
                s = g.sign().detach()
                accum_sign = s if accum_sign is None else accum_sign + s
                n_steps += 1
                model.train()
        # end of epoch: save FGSM perturbation
        if fixed_eval_x is not None:
            model.eval()
            x_adv = fgsm(model, fixed_eval_x, fixed_eval_y)
            traj_perts.append((x_adv - fixed_eval_x).detach())
            model.train()
        print(f"  epoch {ep+1}/{EPOCHS} done")
    model.eval()
    return accum_sign, traj_perts


# ---------- evaluation ----------
@torch.no_grad()
def accuracy(model, loader):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def adv_accuracy(model, loader, attack_fn):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = attack_fn(model, x, y)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    train_loader, test_loader, train_set, test_set = get_loaders()

    # fixed eval batch used to harvest training-time signal
    idx = torch.randperm(len(test_set))[:512]
    fixed_x = torch.stack([test_set[i][0] for i in idx]).to(DEVICE)
    fixed_y = torch.tensor([test_set[i][1] for i in idx]).to(DEVICE)

    print("[1/3] Training baseline...")
    base = CNN().to(DEVICE)
    accum_sign, traj_perts = train_model(base, train_loader,
                                         adv_train=False,
                                         fixed_eval_x=fixed_x, fixed_eval_y=fixed_y)

    print("[2/3] Training FGSM adversarially-trained model...")
    defended = CNN().to(DEVICE)
    train_model(defended, train_loader, adv_train=True)

    results = {}

    print("[3/3] Evaluating...")
    for name, model in [("baseline", base), ("fgsm_defended", defended)]:
        clean = accuracy(model, test_loader)
        a_fgsm = adv_accuracy(model, test_loader, lambda m, x, y: fgsm(m, x, y))
        a_pgd = adv_accuracy(model, test_loader, lambda m, x, y: pgd(m, x, y))
        a_bim = adv_accuracy(model, test_loader, lambda m, x, y: bim(m, x, y))
        results[name] = dict(clean=clean, fgsm=a_fgsm, pgd=a_pgd, bim=a_bim)
        print(f"  {name}: clean={clean:.4f} fgsm={a_fgsm:.4f} pgd={a_pgd:.4f} bim={a_bim:.4f}")

    # ----- novel attacks on baseline -----
    print("\n[novel] V1 AccumSign attack (training-trajectory accumulated sign)")
    v1_pert = EPS * accum_sign.sign()
    x_adv_v1 = (fixed_x + v1_pert).clamp(0, 1)
    x_adv_fgsm = fgsm(base, fixed_x, fixed_y)  # needs grad
    with torch.no_grad():
        v1_acc = (base(x_adv_v1).argmax(1) == fixed_y).float().mean().item()
        fgsm_acc_fixed = (base(x_adv_fgsm).argmax(1) == fixed_y).float().mean().item()
        clean_fixed = (base(fixed_x).argmax(1) == fixed_y).float().mean().item()

    print(f"  clean (fixed batch)        : {clean_fixed:.4f}")
    print(f"  final-model FGSM (baseline): {fgsm_acc_fixed:.4f}  <-- baseline attack")
    print(f"  V1 AccumSign               : {v1_acc:.4f}")

    print("\n[novel] V2 TrajEnsemble attack (per-sample best epoch perturbation)")
    # For each sample, evaluate every epoch's perturbation against final model, pick worst-for-model
    best_loss = torch.full((fixed_x.size(0),), -1e9, device=DEVICE)
    best_adv = fixed_x.clone()
    for p in traj_perts:
        cand = (fixed_x + p).clamp(0, 1)
        with torch.no_grad():
            logits = base(cand)
            losses = F.cross_entropy(logits, fixed_y, reduction="none")
        mask = losses > best_loss
        best_loss = torch.where(mask, losses, best_loss)
        best_adv[mask] = cand[mask]
    with torch.no_grad():
        v2_acc = (base(best_adv).argmax(1) == fixed_y).float().mean().item()
    print(f"  V2 TrajEnsemble            : {v2_acc:.4f}")

    # also: V2-mean (average traj perturbations, project to eps ball via sign)
    mean_pert = torch.stack(traj_perts).mean(0)
    x_adv_mean = (fixed_x + EPS * mean_pert.sign()).clamp(0, 1)
    with torch.no_grad():
        v2m_acc = (base(x_adv_mean).argmax(1) == fixed_y).float().mean().item()
    print(f"  V2 TrajEnsemble (sign-mean): {v2m_acc:.4f}")

    results["novel_on_baseline_fixed_batch"] = dict(
        clean=clean_fixed,
        fgsm_final_model=fgsm_acc_fixed,
        v1_accum_sign=v1_acc,
        v2_traj_per_sample_best=v2_acc,
        v2_traj_sign_mean=v2m_acc,
    )

    # also evaluate the novel attacks on FGSM-defended model
    print("\n[novel] transfer to FGSM-defended model")
    x_adv_d_fgsm = fgsm(defended, fixed_x, fixed_y)
    with torch.no_grad():
        clean_d = (defended(fixed_x).argmax(1) == fixed_y).float().mean().item()
        fgsm_d = (defended(x_adv_d_fgsm).argmax(1) == fixed_y).float().mean().item()
        v1_d = (defended(x_adv_v1).argmax(1) == fixed_y).float().mean().item()
        v2_d = (defended(best_adv).argmax(1) == fixed_y).float().mean().item()
    print(f"  defended clean             : {clean_d:.4f}")
    print(f"  defended FGSM (own grad)   : {fgsm_d:.4f}")
    print(f"  defended vs V1 (transfer)  : {v1_d:.4f}")
    print(f"  defended vs V2 (transfer)  : {v2_d:.4f}")
    results["novel_on_defended_fixed_batch"] = dict(
        clean=clean_d, fgsm=fgsm_d, v1=v1_d, v2=v2_d)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total: {time.time()-t0:.1f}s on {DEVICE}")
