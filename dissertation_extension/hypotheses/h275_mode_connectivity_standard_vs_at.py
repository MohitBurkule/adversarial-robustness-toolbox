"""
H275 - Linear mode connectivity between standard-trained and AT-trained models.

Interpolate weights between a standard-trained model and a PGD-AT model:
  theta(alpha) = (1 - alpha) * theta_standard + alpha * theta_at
for alpha in {0.0, 0.1, 0.2, ..., 1.0}.

At each interpolation point, update BatchNorm stats and measure:
  clean_acc, FGSM_ASR, PGD_ASR, mean_margin, train_loss.

Questions:
  1. Is there a loss barrier mid-interpolation? (spike in train_loss = different basins)
  2. Is robustness monotone in alpha? Or is there a sweet spot with both high
     clean accuracy AND lower adversarial susceptibility?
  3. Do the two models lie in the same loss basin (connected) or different ones?

PGD-AT is implemented with a manual adversarial training loop (10 epochs, PGD-7).
N_train=6000 for speed.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS         = "fashion_mnist"
SEED       = 0
N_TRAIN    = 6000
EPOCHS     = 10
LR_STD     = 0.05
LR_AT      = 0.01
MOMENTUM   = 0.9
BATCH      = 128
EPS        = 0.1
PGD_STEPS  = 10
PGD_ALPHA  = 0.01
AT_STEPS   = 7
AT_ALPHA   = 0.02
ALPHAS     = [round(a * 0.1, 1) for a in range(11)]  # 0.0, 0.1, ..., 1.0
OUT_FILE   = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h275_mode_connectivity_standard_vs_at_output.txt"
)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_standard(model, X, Y):
    opt = torch.optim.SGD(model.parameters(), lr=LR_STD, momentum=MOMENTUM,
                          weight_decay=5e-4)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = X[idx].to(C.DEVICE), Y[idx].to(C.DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"    std epoch {ep+1}/{EPOCHS}  loss={total_loss:.3f}")


def train_pgd_at(model, X, Y):
    """PGD-7 adversarial training, 10 epochs."""
    opt = torch.optim.SGD(model.parameters(), lr=LR_AT, momentum=MOMENTUM,
                          weight_decay=5e-4)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = X[idx].to(C.DEVICE), Y[idx].to(C.DEVICE)
            # PGD inner loop
            xadv = xb.clone().detach() + torch.zeros_like(xb).uniform_(-EPS, EPS)
            xadv = xadv.clamp(0, 1).detach()
            for _ in range(AT_STEPS):
                xadv.requires_grad_(True)
                loss_adv = F.cross_entropy(model(xadv), yb)
                loss_adv.backward()
                xadv = xadv.detach() + AT_ALPHA * xadv.grad.sign()
                xadv = torch.max(torch.min(xadv, xb + EPS), xb - EPS).clamp(0, 1).detach()
            opt.zero_grad()
            loss = F.cross_entropy(model(xadv), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"    AT epoch {ep+1}/{EPOCHS}  loss={total_loss:.3f}")


def interpolate_params(sd_a, sd_b, alpha):
    """Return state_dict = (1-alpha)*sd_a + alpha*sd_b for float tensors."""
    import copy
    sd_new = copy.deepcopy(sd_a)
    for key in sd_new:
        if sd_new[key].is_floating_point():
            sd_new[key] = (1.0 - alpha) * sd_a[key].float() + alpha * sd_b[key].float()
    return sd_new


def update_bn_stats(model, X):
    model.train()
    with torch.no_grad():
        for i in range(0, len(X), 256):
            model(X[i:i+256].to(C.DEVICE))
    model.eval()


def measure_train_loss(model, X, Y):
    model.eval()
    total, nb = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb, yb = X[i:i+256].to(C.DEVICE), Y[i:i+256].to(C.DEVICE)
            total += F.cross_entropy(model(xb), yb).item()
            nb += 1
    return total / nb


def evaluate_point(model, Xtr, Ytr, Xte, Yte, alpha):
    update_bn_stats(model, Xtr)
    train_loss = measure_train_loss(model, Xtr, Ytr)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, fgsm_acc = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, pgd_acc = C.logits_and_acc(model, Xpgd, Yte)
    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))
    fgsm_asr, pgd_asr = 1.0 - fgsm_acc, 1.0 - pgd_acc

    print(f"  alpha={alpha:.1f}: train_loss={train_loss:.4f}  clean={clean_acc:.4f}  "
          f"FGSM_ASR={fgsm_asr:.4f}  PGD_ASR={pgd_asr:.4f}  margin={mean_margin:.4f}")
    return dict(alpha=alpha, train_loss=train_loss, clean_acc=clean_acc,
                fgsm_asr=fgsm_asr, pgd_asr=pgd_asr, mean_margin=mean_margin)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    C.set_seed(SEED)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    print("Loading data...")
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    idx = torch.randperm(len(Xtr))[:N_TRAIN]
    Xtr, Ytr = Xtr[idx], Ytr[idx]
    Xte, Yte = Xte[:500], Yte[:500]
    print(f"  train={len(Xtr)}  test={len(Xte)}")

    # Train standard model
    print("\n--- Training standard model ---")
    C.set_seed(SEED)
    model_std = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                              width=32, seed=SEED)
    model_std.to(C.DEVICE)
    train_standard(model_std, Xtr, Ytr)
    sd_std = {k: v.clone() for k, v in model_std.state_dict().items()}

    # Train AT model (same architecture, same seed init then AT)
    print("\n--- Training PGD-AT model ---")
    C.set_seed(SEED)
    model_at = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                             width=32, seed=SEED)
    model_at.to(C.DEVICE)
    train_pgd_at(model_at, Xtr, Ytr)
    sd_at = {k: v.clone() for k, v in model_at.state_dict().items()}

    # Interpolation sweep
    print("\n--- Interpolation sweep ---")
    records = []
    interp_model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                                 width=32, seed=SEED)
    interp_model.to(C.DEVICE)

    for alpha in ALPHAS:
        sd_interp = interpolate_params(sd_std, sd_at, alpha)
        interp_model.load_state_dict(sd_interp)
        r = evaluate_point(interp_model, Xtr, Ytr, Xte, Yte, alpha)
        records.append(r)

    elapsed = time.time() - t0

    # Detect loss barrier (max train_loss mid-interpolation)
    mid_losses = [r["train_loss"] for r in records if 0.0 < r["alpha"] < 1.0]
    end_losses  = [records[0]["train_loss"], records[-1]["train_loss"]]
    barrier = max(mid_losses) - max(end_losses) if mid_losses else 0.0
    best_pgd = min(records, key=lambda r: r["pgd_asr"])

    lines = [
        "H275 - Mode Connectivity: Standard vs AT\n",
        "=" * 70 + "\n\n",
        f"N_train={N_TRAIN}  epochs={EPOCHS}  eps={EPS}  at_steps={AT_STEPS}\n",
        f"Interpolation alphas: {ALPHAS}\n\n",
        f"{'alpha':>6} {'train_loss':>12} {'CleanAcc':>10} {'FGSM_ASR':>10} "
        f"{'PGD_ASR':>10} {'Margin':>10}\n",
        "-" * 60 + "\n",
    ]
    for r in records:
        lines.append(
            f"{r['alpha']:>6.1f} {r['train_loss']:>12.4f} {r['clean_acc']:>10.4f} "
            f"{r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f}\n"
        )
    lines.append(f"\nElapsed: {elapsed:.1f}s\n\n")
    lines.append("ANALYSIS\n--------\n")
    lines.append(f"Loss barrier (max mid-loss - max endpoint loss): {barrier:+.4f}\n")
    lines.append(f"  -> {'significant barrier detected (different basins)' if barrier > 0.05 else 'no significant barrier (linearly connected)'}\n")
    lines.append(f"Best PGD_ASR={best_pgd['pgd_asr']:.4f} at alpha={best_pgd['alpha']:.1f}  "
                 f"(clean_acc={best_pgd['clean_acc']:.4f})\n")
    is_monotone = all(records[i]["pgd_asr"] <= records[i+1]["pgd_asr"]
                      for i in range(len(records)-1))
    lines.append(f"Robustness monotone in alpha: {is_monotone}\n")
    if not is_monotone:
        lines.append("  -> Sweet spot exists: some interpolation ratio gives better "
                     "robustness than either endpoint.\n")

    with open(OUT_FILE, "w") as f:
        f.writelines(lines)
    print(f"\nResults written to {OUT_FILE}")
