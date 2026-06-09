"""
H271 - Does learned (trainable) noise scale per layer outperform fixed Gaussian
noise, even without an adversarial training inner loop?

Four model variants:
  a) baseline         : standard CE training, no noise
  b) fixed_noise      : Gaussian σ=0.05 added after each feature block (hook),
                        only during training
  c) learned_noise    : per-block trainable σ (nn.Parameter, initialised at
                        0.05), noise = randn_like(act) * σ.abs().
                        Trained with standard CE loss.
  d) learned_noise_at : same as (c) but each batch mixes clean + FGSM 50/50
                        (cheap Parametric Noise Injection style).

After training we log the learned σ values for variants (c) and (d).

Metrics: clean_acc, FGSM_ASR, PGD_ASR, mean_margin

Key question: is learning σ necessary, or does fixed noise work just as well?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign import common as C


FIXED_SIGMA  = 0.05
EPOCHS       = 10
BATCH_SIZE   = 128
FGSM_EPS     = 0.1
RESULTS_DIR  = os.path.join(os.path.dirname(__file__),
                            "..", "results", "fashion_mnist")


# ---------------------------------------------------------------------------
# Hook helper for fixed noise
# ---------------------------------------------------------------------------

def make_fixed_noise_hook(sigma: float, model_ref):
    def hook(module, input, output):
        if model_ref[0].training:
            return output + torch.randn_like(output) * sigma
        return output
    return hook


def attach_fixed_noise_hooks(model, sigma=FIXED_SIGMA):
    model_ref = [model]
    handles = []
    n_layers = len(model.features)
    layers_per_block = n_layers // 3  # 3 blocks
    # Attach hook to the last layer of each block (MaxPool2d)
    for b in range(3):
        last_layer_idx = (b + 1) * layers_per_block - 1
        h = model.features[last_layer_idx].register_forward_hook(
            make_fixed_noise_hook(sigma, model_ref))
        handles.append(h)
    return handles


# ---------------------------------------------------------------------------
# Learned-noise wrapper
# ---------------------------------------------------------------------------

class LearnedNoiseModel(nn.Module):
    """
    Wraps a SmallCNN and injects per-block learnable Gaussian noise.
    noise_i = randn_like(act_i) * |sigma_i|
    Only injected during training (self.training is True).
    """
    def __init__(self, base_model, n_blocks=3, init_sigma=FIXED_SIGMA):
        super().__init__()
        self.features = base_model.features
        self.head     = base_model.head
        self.sigmas   = nn.ParameterList([
            nn.Parameter(torch.tensor(init_sigma)) for _ in range(n_blocks)
        ])

    def forward(self, x):
        out = x
        # features is a flat Sequential; each "block" is 4 layers
        # (Conv2d, BN, ReLU, MaxPool2d). Inject noise after each block.
        layers_per_block = len(self.features) // len(self.sigmas)
        for i, layer in enumerate(self.features):
            out = layer(out)
            block_idx = i // layers_per_block
            if (i + 1) % layers_per_block == 0 and block_idx < len(self.sigmas):
                if self.training:
                    out = out + torch.randn_like(out) * self.sigmas[block_idx].abs()
        out = self.head(out)
        return out

    def get_sigmas(self):
        return [float(s.abs().item()) for s in self.sigmas]


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def train_baseline(model, Xtr, Ytr, epochs=EPOCHS):
    model.to(C.DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl  = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()


def train_fixed_noise(model, Xtr, Ytr, epochs=EPOCHS, sigma=FIXED_SIGMA):
    handles = attach_fixed_noise_hooks(model, sigma)
    train_baseline(model, Xtr, Ytr, epochs)
    for h in handles:
        h.remove()


def train_learned_noise(model, Xtr, Ytr, epochs=EPOCHS):
    """Standard CE with learned noise parameters."""
    model.to(C.DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl  = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()


def train_learned_noise_at(model, Xtr, Ytr, epochs=EPOCHS, fgsm_eps=FGSM_EPS):
    """Learned noise + 50/50 clean-FGSM mixing."""
    model.to(C.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl  = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)

            # Generate FGSM perturbation
            model.eval()
            xb_in = xb.detach().requires_grad_(True)
            loss0  = F.cross_entropy(model(xb_in), yb)
            loss0.backward()
            xb_fgsm = (xb + fgsm_eps * xb_in.grad.sign()).clamp(0, 1).detach()
            model.train()

            # Mix 50/50
            half = xb.size(0) // 2
            mixed = torch.cat([xb[:half], xb_fgsm[half:]], dim=0)
            labels = torch.cat([yb[:half], yb[half:]], dim=0)

            opt.zero_grad()
            F.cross_entropy(model(mixed), labels).backward()
            opt.step()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_asr(model, X, Y, Xadv):
    model.eval()
    with torch.no_grad():
        clean_pred = model(X.to(C.DEVICE)).argmax(1).cpu().numpy()
        adv_pred   = model(Xadv.to(C.DEVICE)).argmax(1).cpu().numpy()
    y_np = Y.cpu().numpy()
    correct_clean = clean_pred == y_np
    fooled = correct_clean & (adv_pred != y_np)
    if correct_clean.sum() == 0:
        return 0.0
    return float(fooled.sum() / correct_clean.sum())


def evaluate_model(model, Xte, Yte, tag=""):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    _, clean_acc = C.logits_and_acc(model, Xte, Yte)

    Xfgsm = C.fgsm(model, Xte, Yte, eps=0.1)
    fgsm_asr = compute_asr(model, Xte, Yte, Xfgsm)

    Xpgd = C.pgd(model, Xte, Yte, eps=0.1, steps=10, alpha=0.01)
    pgd_asr = compute_asr(model, Xte, Yte, Xpgd)

    margins = C.margin(model, Xte, Yte)
    mean_mg = float(np.mean(margins))

    print(f"  [{tag}]  clean_acc={clean_acc:.4f}  fgsm_asr={fgsm_asr:.4f}  "
          f"pgd_asr={pgd_asr:.4f}  mean_margin={mean_mg:.4f}")
    return dict(tag=tag, clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                pgd_asr=pgd_asr, mean_margin=mean_mg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            "h271_learned_vs_fixed_hidden_noise_output.txt")

    C.set_seed(0)
    print("Loading Fashion-MNIST …")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")

    results       = []
    learned_sigmas = {}   # tag -> list of sigma values

    # ---- (a) baseline ----
    print("\n=== Training: baseline ===")
    C.set_seed(0)
    m_base = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                           width=32, seed=0)
    t0 = time.time()
    train_baseline(m_base, Xtr, Ytr)
    results.append({**evaluate_model(m_base, Xte, Yte, "baseline"),
                    "train_time_s": time.time() - t0})

    # ---- (b) fixed noise ----
    print("\n=== Training: fixed_noise ===")
    C.set_seed(0)
    m_fixed = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                            width=32, seed=0)
    t0 = time.time()
    train_fixed_noise(m_fixed, Xtr, Ytr)
    results.append({**evaluate_model(m_fixed, Xte, Yte, "fixed_noise"),
                    "train_time_s": time.time() - t0})

    # ---- (c) learned noise ----
    print("\n=== Training: learned_noise ===")
    C.set_seed(0)
    base_model_c = C.build_model("cnn",
                                 {"channels": 1, "size": 28, "n_classes": 10},
                                 width=32, seed=0)
    m_learned = LearnedNoiseModel(base_model_c)
    t0 = time.time()
    train_learned_noise(m_learned, Xtr, Ytr)
    sigs_c = m_learned.get_sigmas()
    learned_sigmas["learned_noise"] = sigs_c
    print(f"  Learned σ per block: {[f'{s:.4f}' for s in sigs_c]}")
    results.append({**evaluate_model(m_learned, Xte, Yte, "learned_noise"),
                    "train_time_s": time.time() - t0})

    # ---- (d) learned noise + AT ----
    print("\n=== Training: learned_noise_at ===")
    C.set_seed(0)
    base_model_d = C.build_model("cnn",
                                 {"channels": 1, "size": 28, "n_classes": 10},
                                 width=32, seed=0)
    m_learned_at = LearnedNoiseModel(base_model_d)
    t0 = time.time()
    train_learned_noise_at(m_learned_at, Xtr, Ytr)
    sigs_d = m_learned_at.get_sigmas()
    learned_sigmas["learned_noise_at"] = sigs_d
    print(f"  Learned σ per block: {[f'{s:.4f}' for s in sigs_d]}")
    results.append({**evaluate_model(m_learned_at, Xte, Yte, "learned_noise_at"),
                    "train_time_s": time.time() - t0})

    # Summary table
    col = 18
    header = (f"{'Variant':<{col}} {'CleanAcc':>9} {'FGSM_ASR':>9} "
              f"{'PGD_ASR':>8} {'Margin':>8}")
    sep = "-" * len(header)
    rows = [f"{r['tag']:<{col}} {r['clean_acc']:>9.4f} {r['fgsm_asr']:>9.4f} "
            f"{r['pgd_asr']:>8.4f} {r['mean_margin']:>8.4f}"
            for r in results]
    table = "\n".join([header, sep] + rows)
    print("\n\n" + table)

    base_r = results[0]
    finding_lines = [table, "\n\nLearned sigma values:"]
    for tag, sigs in learned_sigmas.items():
        finding_lines.append(
            f"  {tag}: block1={sigs[0]:.4f}  block2={sigs[1]:.4f}  block3={sigs[2]:.4f}"
        )

    finding_lines.append("\nKey Findings (delta vs baseline):")
    for r in results[1:]:
        finding_lines.append(
            f"  {r['tag']}: ΔFGSM_ASR={r['fgsm_asr']-base_r['fgsm_asr']:+.4f}  "
            f"ΔPGD_ASR={r['pgd_asr']-base_r['pgd_asr']:+.4f}  "
            f"ΔMargin={r['mean_margin']-base_r['mean_margin']:+.4f}"
        )

    output = "\n".join(finding_lines)
    print(output)

    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
