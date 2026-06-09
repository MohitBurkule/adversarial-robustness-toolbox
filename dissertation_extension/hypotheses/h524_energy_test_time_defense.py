"""
H524 - Energy-Guided Test-Time Transformation Defense (ET3-style)

Paper: "A Provable Energy-Guided Test-Time Defense Boosting Adversarial
       Robustness of Large Vision-Language Models"
       Mirza et al., arXiv 2603.26984, March 2026.

Core insight: ET3 minimises the free-energy of an input at test time via
gradient-based input optimisation, pushing adversarial examples back onto the
clean data manifold WITHOUT retraining the model.  The energy is defined as
the negative log-sum-exp of logits (Helmholtz free energy / energy-based model
interpretation of the softmax classifier).

Experiment (Fashion-MNIST):
  1. Train a standard CNN.
  2. Generate PGD-10 adversarial examples.
  3. Apply energy-minimisation purification: iteratively update x to minimise
     E(x) = -log(sum(exp(f(x)))) with step size eta for T steps, keeping x
     in [0,1].
  4. Sweep T in {5, 10, 20, 50} and eta in {0.5/255, 1/255, 2/255}.
  5. PASS if any (T, eta) recovers >= 8 pp accuracy over unpurified adversarial
     accuracy, validating the energy-minimisation defense principle.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign import common as C

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = 2.5 / 255.0
N_TRAIN = 6000
N_EVAL = 1000
SEED = 42
ET3_STEPS_LIST = [5, 10, 20, 50]
ET3_ETA_LIST = [0.5 / 255.0, 1.0 / 255.0, 2.0 / 255.0]


class CNN(nn.Module):
    def __init__(self, ch=1, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(ch, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def pgd_attack(model, x, y, eps, alpha, steps):
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1).detach()
    return x_adv


def energy_purify(model, x, eta, steps):
    """
    ET3-style energy minimisation: iteratively update x to minimise the
    Helmholtz free energy E(x) = -log(sum(exp(f_k(x)))).
    Lower energy => higher total logit mass => model is more 'confident'.
    """
    x_pur = x.clone().detach()
    model.eval()
    for _ in range(steps):
        x_pur.requires_grad_(True)
        logits = model(x_pur)
        energy = -torch.logsumexp(logits, dim=1).sum()  # minimise
        grad = torch.autograd.grad(energy, x_pur)[0]
        x_pur = x_pur.detach() - eta * grad.sign()  # gradient descent
        x_pur = x_pur.clamp(0, 1).detach()
    return x_pur


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(1)
    return (preds == y).float().mean().item()


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("H524 - Energy-Guided Test-Time Transformation Defense (ET3)")
    log("=" * 60)

    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    Xtr = torch.tensor(Xtr, dtype=torch.float32).to(DEVICE)
    Ytr = torch.tensor(Ytr, dtype=torch.long).to(DEVICE)
    Xte = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
    Yte = torch.tensor(Yte, dtype=torch.long).to(DEVICE)
    if Xtr.dim() == 3:
        Xtr = Xtr.unsqueeze(1)
        Xte = Xte.unsqueeze(1)

    # Train
    model = CNN(ch=Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(Xtr), BATCH):
            b = idx[i:i + BATCH]
            loss = F.cross_entropy(model(Xtr[b]), Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    clean_acc = evaluate(model, Xte, Yte)
    log(f"Clean accuracy: {clean_acc:.4f}")

    # Adversarial examples
    model.eval()
    Xadv = []
    for i in range(0, len(Xte), BATCH):
        Xadv.append(pgd_attack(model, Xte[i:i+BATCH], Yte[i:i+BATCH], EPS, PGD_ALPHA, PGD_STEPS))
    Xadv = torch.cat(Xadv)
    adv_acc = evaluate(model, Xadv, Yte)
    log(f"Adversarial accuracy (unpurified): {adv_acc:.4f}")

    # ET3 sweep
    log(f"\nEnergy purification sweep:")
    log(f"{'Steps':>6}  {'Eta':>12}  {'Purified Acc':>12}  {'Recovery pp':>12}")
    best_recovery = 0.0
    for T in ET3_STEPS_LIST:
        for eta in ET3_ETA_LIST:
            # Process in batches to save memory
            purified = []
            for i in range(0, len(Xadv), BATCH):
                purified.append(energy_purify(model, Xadv[i:i+BATCH], eta, T))
            Xpur = torch.cat(purified)
            pur_acc = evaluate(model, Xpur, Yte)
            recovery = (pur_acc - adv_acc) * 100
            best_recovery = max(best_recovery, recovery)
            log(f"{T:>6}  {eta:>12.6f}  {pur_acc:>12.4f}  {recovery:>+12.1f}")

    log(f"\nBest recovery: {best_recovery:+.1f} pp")
    verdict = "PASS" if best_recovery >= 8.0 else "FAIL"
    log(f"\nVerdict: {verdict} (need >= 8 pp recovery)")
    log(f"Time: {time.time() - t0:.1f}s")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "h524_energy_test_time_defense_output.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
