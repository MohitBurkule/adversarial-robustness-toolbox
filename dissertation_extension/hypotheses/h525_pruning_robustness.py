"""
H525 - Adversarial Pruning: Does Sparsity Help Robustness?

Paper: "Adversarial Pruning: A Survey and Benchmark of Pruning Methods for
       Adversarial Robustness" (arXiv 2409.01249, Sept 2024 / updated 2025).
       Also: "Over-parameterization and Adversarial Robustness in Neural
       Networks: An Overview and Empirical Analysis" (arXiv 2406.10090, 2024).

Core insight: the survey benchmarks magnitude pruning, lottery-ticket style
iterative pruning, and adversarial pruning.  Theory suggests weight sparsity
can *improve* robustness by removing redundant parameters that adversaries
exploit, but empirical results are mixed and dataset/architecture dependent.

Experiment (Fashion-MNIST):
  1. Train a CNN with standard training.
  2. Adversarially train the same CNN architecture (PGD-AT).
  3. For both models, apply global unstructured magnitude pruning at
     sparsity levels {0, 30, 50, 70, 85, 95}%.
  4. Evaluate clean accuracy and PGD-10 robust accuracy at each level.
  5. PASS if for AT model, there exists a sparsity > 0 where robust accuracy
     is within 2pp of the dense (0%) model, confirming pruning does NOT
     catastrophically hurt robustness (the "free lunch" finding).

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign import common as C

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH = 128
EPS = 15.0 / 255.0
PGD_STEPS = 10
PGD_ALPHA = 2.5 / 255.0
AT_STEPS = 7
AT_ALPHA = 2.0 / 255.0
N_TRAIN = 6000
N_EVAL = 1000
SEED = 42
SPARSITIES = [0, 30, 50, 70, 85, 95]


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


def train_model(Xtr, Ytr, ch, adversarial=False):
    model = CNN(ch=ch).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(Xtr), BATCH):
            b = idx[i:i + BATCH]
            xb, yb = Xtr[b], Ytr[b]
            if adversarial:
                xb = pgd_attack(model, xb, yb, EPS, AT_ALPHA, AT_STEPS)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def apply_pruning(model, sparsity_pct):
    """Apply global unstructured L1 magnitude pruning."""
    if sparsity_pct == 0:
        return model
    m = copy.deepcopy(model)
    params_to_prune = []
    for name, module in m.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params_to_prune.append((module, "weight"))
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity_pct / 100.0,
    )
    # Make pruning permanent
    for module, _ in params_to_prune:
        prune.remove(module, "weight")
    return m


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def robust_accuracy(model, Xte, Yte):
    model.eval()
    Xadv = []
    for i in range(0, len(Xte), BATCH):
        Xadv.append(pgd_attack(model, Xte[i:i+BATCH], Yte[i:i+BATCH], EPS, PGD_ALPHA, PGD_STEPS))
    Xadv = torch.cat(Xadv)
    return evaluate(model, Xadv, Yte)


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("H525 - Adversarial Pruning: Sparsity vs Robustness")
    log("=" * 60)

    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    Xtr = torch.tensor(Xtr, dtype=torch.float32).to(DEVICE)
    Ytr = torch.tensor(Ytr, dtype=torch.long).to(DEVICE)
    Xte = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
    Yte = torch.tensor(Yte, dtype=torch.long).to(DEVICE)
    if Xtr.dim() == 3:
        Xtr = Xtr.unsqueeze(1)
        Xte = Xte.unsqueeze(1)

    ch = Xtr.shape[1]

    # Train both models
    log("Training standard model...")
    std_model = train_model(Xtr, Ytr, ch, adversarial=False)
    log("Training adversarially-trained model...")
    at_model = train_model(Xtr, Ytr, ch, adversarial=True)

    # Evaluate across sparsity levels
    for label, base_model in [("Standard", std_model), ("Adv-Trained", at_model)]:
        log(f"\n--- {label} Model ---")
        log(f"{'Sparsity%':>10}  {'Clean Acc':>10}  {'Robust Acc':>10}")
        dense_robust = None
        free_lunch = False
        for sp in SPARSITIES:
            pruned = apply_pruning(base_model, sp)
            cacc = evaluate(pruned, Xte, Yte)
            racc = robust_accuracy(pruned, Xte, Yte)
            log(f"{sp:>10}  {cacc:>10.4f}  {racc:>10.4f}")
            if sp == 0:
                dense_robust = racc
            elif label == "Adv-Trained" and dense_robust is not None:
                if (dense_robust - racc) <= 0.02:
                    free_lunch = True

        if label == "Adv-Trained":
            log(f"\nDense robust acc: {dense_robust:.4f}")
            log(f"Free-lunch (sparse within 2pp of dense): {free_lunch}")

    # Verdict based on AT model
    verdict = "PASS" if free_lunch else "FAIL"
    log(f"\nVerdict: {verdict}")
    log(f"Time: {time.time() - t0:.1f}s")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "h525_pruning_robustness_output.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
