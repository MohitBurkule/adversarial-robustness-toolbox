"""
H523 - Frequency-Domain Adversarial Purification

Paper: "Diffusion-based Adversarial Purification from the Perspective of the
       Frequency Domain" (arXiv 2505.01267, May 2025)

Core insight: adversarial perturbations concentrate in high-frequency components
of the image spectrum.  The paper decomposes images into amplitude and phase via
FFT and finds that damage from adversarial perturbations increases monotonically
with frequency.

Experiment (Fashion-MNIST, no diffusion model needed):
  1. Train a standard CNN on clean data.
  2. Generate PGD-10 adversarial examples (L-inf eps=15/255).
  3. Apply a simple frequency-domain purification: FFT the image, zero out the
     top-k% highest-frequency components, inverse-FFT back to pixel space.
  4. Measure recovered accuracy across k in {5, 10, 20, 30, 50, 70}%.
  5. PASS if any k recovers >= 10 pp accuracy over the unpurified adv accuracy,
     confirming the high-frequency concentration of perturbations.

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
FREQ_CUTOFFS = [5, 10, 20, 30, 50, 70]  # percent of highest freq to zero


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
    """PGD-k L-inf attack."""
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1).detach()
    return x_adv


def freq_purify(images, cutoff_pct):
    """Zero out the top cutoff_pct% highest-frequency 2D-FFT components."""
    # images: (B, C, H, W)
    fft = torch.fft.fft2(images, norm="ortho")
    B, Ch, H, W = images.shape
    # Build a mask that keeps low frequencies (center of shifted spectrum)
    freq_y = torch.fft.fftfreq(H, device=images.device).abs()
    freq_x = torch.fft.fftfreq(W, device=images.device).abs()
    freq_grid = torch.sqrt(freq_y[:, None] ** 2 + freq_x[None, :] ** 2)
    threshold = np.percentile(freq_grid.cpu().numpy(), 100 - cutoff_pct)
    mask = (freq_grid <= threshold).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    fft_filtered = fft * mask
    purified = torch.fft.ifft2(fft_filtered, norm="ortho").real
    return purified.clamp(0, 1)


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

    log("H523 - Frequency-Domain Adversarial Purification")
    log("=" * 60)

    # Load data
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    Xtr = torch.tensor(Xtr, dtype=torch.float32).to(DEVICE)
    Ytr = torch.tensor(Ytr, dtype=torch.long).to(DEVICE)
    Xte = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
    Yte = torch.tensor(Yte, dtype=torch.long).to(DEVICE)
    if Xtr.dim() == 3:
        Xtr = Xtr.unsqueeze(1)
        Xte = Xte.unsqueeze(1)

    # Train model
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

    # Generate adversarial examples
    model.eval()
    Xadv = []
    for i in range(0, len(Xte), BATCH):
        xb = Xte[i:i + BATCH]
        yb = Yte[i:i + BATCH]
        Xadv.append(pgd_attack(model, xb, yb, EPS, PGD_ALPHA, PGD_STEPS))
    Xadv = torch.cat(Xadv)
    adv_acc = evaluate(model, Xadv, Yte)
    log(f"Adversarial accuracy (no purification): {adv_acc:.4f}")

    # Frequency purification sweep
    log(f"\nFrequency cutoff sweep (zeroing top k% of spectrum):")
    log(f"{'Cutoff%':>8}  {'Purified Adv Acc':>16}  {'Recovery (pp)':>14}")
    best_recovery = 0.0
    for k in FREQ_CUTOFFS:
        Xpur = freq_purify(Xadv, k)
        pur_acc = evaluate(model, Xpur, Yte)
        recovery = (pur_acc - adv_acc) * 100
        best_recovery = max(best_recovery, recovery)
        log(f"{k:>8}  {pur_acc:>16.4f}  {recovery:>+14.1f}")
        # Also check clean-image degradation
        Xclean_pur = freq_purify(Xte, k)
        clean_pur_acc = evaluate(model, Xclean_pur, Yte)
        log(f"          (clean after purification: {clean_pur_acc:.4f})")

    log(f"\nBest recovery: {best_recovery:+.1f} pp")
    verdict = "PASS" if best_recovery >= 10.0 else "FAIL"
    log(f"\nVerdict: {verdict} (need >= 10 pp recovery)")
    log(f"Time: {time.time() - t0:.1f}s")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "h523_frequency_purification_output.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
