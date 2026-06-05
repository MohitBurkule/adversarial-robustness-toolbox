"""
H232 - Random noise padding: does attacker waste budget on padding pixels?

Pad test images with B pixels of random noise on all sides.
B in [0, 2, 4, 8].  For B>0: padded shape = (1, 28+2B, 28+2B).
Padding filled with uniform U(0,1) noise, re-randomised each forward pass.

Train CNN on randomly-padded training images.
Test: PGD-10 over the full padded image (all pixels perturb-able).

Measure:
  core_delta_energy  = ||delta[B:28+B, B:28+B]||_F^2 / ||delta||_F^2
  padding_delta_energy = 1 - core_delta_energy
  ASR vs baseline (no padding)
  clean_acc on no-padding test images (model run through zero-padding)

Print table: B -> core_energy, padding_energy, ASR, clean_acc
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_EVAL = 300
SEED = 0
EPS = 0.1
PGD_STEPS = 10
EPOCHS = 10
B_VALUES = [0, 2, 4, 8]

os.makedirs("results/fashion_mnist", exist_ok=True)


def add_noise_padding(x, B):
    """Add B pixels of random U(0,1) noise padding on all sides."""
    if B == 0:
        return x
    N, C, H, W = x.shape
    # create padded tensor with random noise
    xp = torch.rand(N, C, H + 2 * B, W + 2 * B, device=x.device)
    xp[:, :, B:B + H, B:B + W] = x
    return xp


def add_zero_padding(x, B):
    """Add B pixels of zero padding (for clean-acc evaluation)."""
    if B == 0:
        return x
    return F.pad(x, (B, B, B, B), value=0.0)


class PaddedCNN(nn.Module):
    """CNN that wraps a standard SmallCNN but adds noise padding on each forward pass."""

    def __init__(self, inner_model, B):
        super().__init__()
        self.inner = inner_model
        self.B = B

    def forward(self, x):
        if self.B == 0:
            return self.inner(x)
        # x is already padded; just forward through inner
        return self.inner(x)


def train_with_padding(B, Xtr, Ytr, meta, seed):
    """Build a CNN for padded image size and train with re-randomised padding."""
    C.set_seed(seed)
    size_padded = meta["size"] + 2 * B
    meta_padded = {"channels": meta["channels"], "size": size_padded, "n_classes": meta["n_classes"]}
    model = C.build_model("cnn", meta_padded, width=32, seed=seed)

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_padded = add_noise_padding(xb, B)
            opt.zero_grad()
            out = model(xb_padded)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def pgd_full(model, x, y, eps, steps, alpha=None):
    """PGD over the full (padded) image."""
    if alpha is None:
        alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def evaluate(model, B, Xte, Yte, n_eval=N_EVAL, eps=EPS, steps=PGD_STEPS):
    """Evaluate ASR and core/padding energy for given B."""
    model.eval()
    X = Xte[:n_eval]
    Y = Yte[:n_eval]

    # build padded test inputs with noise
    Xp = add_noise_padding(X, B)

    # clean accuracy on padded inputs
    with torch.no_grad():
        logits = model(Xp)
        clean_correct = (logits.argmax(1) == Y).float()
    clean_acc = float(clean_correct.mean())

    # PGD attack on padded images
    # only on correctly-classified samples
    corr_mask = clean_correct.bool()
    if corr_mask.sum() == 0:
        return clean_acc, float("nan"), float("nan"), float("nan")

    Xc = Xp[corr_mask]
    Yc = Y[corr_mask]

    Xa = pgd_full(model, Xc, Yc, eps=eps, steps=steps)

    with torch.no_grad():
        flipped = (model(Xa).argmax(1) != Yc).float()
    asr = float(flipped.mean())

    # compute delta energy split
    delta = (Xa - Xc).cpu()  # shape (N, C, H+2B, W+2B)
    if B > 0:
        H = 28
        delta_core = delta[:, :, B:B + H, B:B + H]
        delta_pad = delta.clone()
        delta_pad[:, :, B:B + H, B:B + H] = 0.0
        core_energy = float((delta_core ** 2).sum()) / (float((delta ** 2).sum()) + 1e-12)
        pad_energy = float((delta_pad ** 2).sum()) / (float((delta ** 2).sum()) + 1e-12)
    else:
        core_energy = 1.0
        pad_energy = 0.0

    # clean accuracy without padding (zero-pad for correct size)
    Xzp = add_zero_padding(X, B)
    with torch.no_grad():
        logits_zp = model(Xzp)
        clean_acc_nop = float((logits_zp.argmax(1) == Y).float().mean())

    return clean_acc, clean_acc_nop, asr, core_energy, pad_energy


def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_EVAL, seed=SEED)

    print("H232 - Random Noise Padding Budget")
    print("=" * 70)
    print(f"Dataset: {DS}, N_EVAL={N_EVAL}, EPS={EPS}, PGD_STEPS={PGD_STEPS}")
    print(f"{'B':>4}  {'clean_acc':>10}  {'clean_acc_nop':>14}  {'ASR':>8}  {'core_energy':>12}  {'pad_energy':>11}")
    print("-" * 70)

    results = []
    for B in B_VALUES:
        print(f"  Training CNN with B={B} ...")
        model = train_with_padding(B, Xtr, Ytr, meta, SEED)
        clean_acc, clean_acc_nop, asr, core_energy, pad_energy = evaluate(
            model, B, Xte, Yte)
        results.append((B, clean_acc, clean_acc_nop, asr, core_energy, pad_energy))
        print(f"{B:>4}  {clean_acc:>10.4f}  {clean_acc_nop:>14.4f}  {asr:>8.4f}  {core_energy:>12.4f}  {pad_energy:>11.4f}")

    print("\nSummary Table:")
    print(f"{'B':>4}  {'clean_acc':>10}  {'clean_acc_nop':>14}  {'ASR':>8}  {'core_energy':>12}  {'pad_energy':>11}")
    print("-" * 70)
    baseline_asr = results[0][3]
    for B, ca, ca_nop, asr, ce, pe in results:
        rel = f"({asr / baseline_asr:.2f}x)" if baseline_asr > 0 and B > 0 else "(baseline)"
        print(f"{B:>4}  {ca:>10.4f}  {ca_nop:>14.4f}  {asr:>8.4f} {rel:>10}  {ce:>12.4f}  {pe:>11.4f}")

    print("\nInterpretation:")
    print("  core_energy: fraction of adversarial perturbation energy on 28x28 core")
    print("  pad_energy:  fraction on padding pixels (wasted budget)")
    print("  Hypothesis: larger B -> lower core_energy -> attacker wastes budget -> lower ASR")
    if len(results) > 1:
        asrs = [r[3] for r in results]
        cores = [r[4] for r in results]
        print(f"\n  B=0 ASR={asrs[0]:.4f}, B={B_VALUES[-1]} ASR={asrs[-1]:.4f}")
        print(f"  B=0 core_energy={cores[0]:.4f}, B={B_VALUES[-1]} core_energy={cores[-1]:.4f}")
        if asrs[-1] < asrs[0]:
            print("  -> CONFIRMED: larger padding -> lower ASR")
        else:
            print("  -> NOT CONFIRMED: padding did not reduce ASR")
        if cores[-1] < cores[0]:
            print("  -> CONFIRMED: attacker wastes energy on padding pixels")
        else:
            print("  -> NOT CONFIRMED: attacker focuses on core despite padding")


if __name__ == "__main__":
    main()
