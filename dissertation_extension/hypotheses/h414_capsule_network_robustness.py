"""
H414 - Capsule Network Robustness: Dynamic Routing vs SmallCNN Baseline

Reference:
    Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic Routing Between
    Capsules. NeurIPS 2017. https://arxiv.org/abs/1710.09829

Hypothesis:
    Capsule routing's vote-agreement creates implicit manifold projection that
    may reduce adversarial susceptibility — i.e., the iterative routing-by-
    agreement mechanism forces activations onto a lower-dimensional equivariant
    manifold, potentially acting as an implicit input purifier. However, prior
    adaptive-attack work (Kanbak et al. 2018; Michels et al. 2019) shows
    CapsNets are not truly robust under white-box attacks that account for the
    routing iterations; apparent robustness is largely gradient masking arising
    from the non-differentiable argmax-like routing softmax and the iterative
    fixed-point loop.

Expected outcome:
    PARTIAL — CapsNet will likely show lower PGD ASR than SmallCNN under
    naive PGD (gradient masking artifact), but EOT-PGD (averaging gradients
    over N routing iterations with different random starts) will close or
    eliminate that gap, confirming the robustness is illusory.

Masking check (EOT-PGD):
    We run PGD with gradient averaging over EOT_SAMPLES restarts to bypass
    gradient obfuscation. If (EOT-PGD ASR) >> (vanilla PGD ASR) for CapsNet
    but not for CNN, that is evidence of gradient masking in the CapsNet.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9,wd=5e-4),
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config ---------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
ROUTING_ITERS = 3          # dynamic routing iterations (Sabour 2017 default)
EOT_SAMPLES = 20           # gradient averages for EOT-PGD masking check
EOT_PGD_STEPS = 20         # more steps for EOT-PGD to be thorough

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h414_capsule_network_robustness_output.txt"
)


# ---------------------------------------------------------------------------
# CapsNet (Sabour-Hinton 2017) for Fashion-MNIST
# ---------------------------------------------------------------------------

def squash(s, dim=-1):
    """Non-linear squashing activation (Sabour 2017, eq. 1)."""
    sq = (s * s).sum(dim=dim, keepdim=True)
    scale = sq / (1.0 + sq)
    unit = s / (sq.sqrt() + 1e-8)
    return scale * unit


class PrimaryCaps(nn.Module):
    """
    Conv layer -> reshape into capsules.
    Produces (B, n_caps, caps_dim) = (B, 1152, 8) in the Sabour paper
    for 28x28 input; we use a scaled-down version for speed.
    """
    def __init__(self, in_ch=1, n_caps_channels=32, caps_dim=8, kernel=9, stride=2):
        super().__init__()
        # initial conv to extract features
        self.conv1 = nn.Conv2d(in_ch, 256, kernel_size=9, stride=1, padding=0)
        # primary caps convolution
        self.caps_conv = nn.Conv2d(256, n_caps_channels * caps_dim,
                                   kernel_size=kernel, stride=stride, padding=0)
        self.n_caps_channels = n_caps_channels
        self.caps_dim = caps_dim

    def forward(self, x):
        h = F.relu(self.conv1(x))               # (B, 256, H1, W1)
        h = self.caps_conv(h)                   # (B, caps_ch*caps_dim, H2, W2)
        B, _, H, W = h.shape
        # reshape to (B, n_caps, caps_dim)
        h = h.view(B, self.n_caps_channels, self.caps_dim, H, W)
        h = h.permute(0, 1, 3, 4, 2).contiguous()  # (B, caps_ch, H, W, caps_dim)
        h = h.view(B, -1, self.caps_dim)            # (B, n_primary_caps, caps_dim)
        return squash(h)


class DigitCaps(nn.Module):
    """
    Dynamic-routing digit capsule layer (Sabour 2017, algorithm 1).
    Input:  (B, n_primary, primary_dim)
    Output: (B, n_classes, class_dim)  after squash
    """
    def __init__(self, n_primary, primary_dim, n_classes=10, class_dim=16,
                 routing_iters=3):
        super().__init__()
        self.n_primary = n_primary
        self.n_classes = n_classes
        self.class_dim = class_dim
        self.routing_iters = routing_iters
        # weight matrix W: (n_primary, n_classes, class_dim, primary_dim)
        self.W = nn.Parameter(
            torch.randn(1, n_primary, n_classes, class_dim, primary_dim) * 0.1
        )

    def forward(self, u):
        """
        u: (B, n_primary, primary_dim)
        returns: (B, n_classes, class_dim)
        """
        B = u.size(0)
        # u_hat: predictions (B, n_primary, n_classes, class_dim)
        u_ = u[:, :, None, :, None]               # (B, n_p, 1, p_dim, 1)
        W = self.W.expand(B, -1, -1, -1, -1)      # (B, n_p, n_cls, c_dim, p_dim)
        u_hat = torch.matmul(W, u_).squeeze(-1)   # (B, n_p, n_cls, c_dim)

        # routing logits: (B, n_primary, n_classes), detached for routing updates
        b = torch.zeros(B, self.n_primary, self.n_classes,
                        device=u.device, dtype=u.dtype)

        v = None
        for r in range(self.routing_iters):
            c = F.softmax(b, dim=2)                       # (B, n_p, n_cls)
            # s: weighted sum of predictions
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)     # (B, n_cls, c_dim)
            v = squash(s, dim=-1)                          # (B, n_cls, c_dim)
            if r < self.routing_iters - 1:
                # agreement: (B, n_p, n_cls)
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)
                b = b + agreement
        return v  # (B, n_classes, class_dim)


class CapsNet(nn.Module):
    """
    Full CapsNet classifier for Fashion-MNIST (1x28x28 -> 10 classes).

    Architecture follows Sabour 2017 but scaled to work with 28x28 input
    and be trainable in ~10 epochs.

    Classification: use norm of each class capsule as class probability.
    Returns logits as norms (no margin loss; use cross-entropy for simplicity
    and comparability with baseline SmallCNN).
    """
    def __init__(self, in_ch=1, n_classes=10, routing_iters=3,
                 n_caps_channels=16, caps_dim=8, class_dim=16):
        super().__init__()
        self.primary = PrimaryCaps(
            in_ch=in_ch,
            n_caps_channels=n_caps_channels,
            caps_dim=caps_dim,
            kernel=5,     # smaller kernel so 28x28 input works
            stride=2
        )
        # Compute n_primary dynamically via a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 28, 28)
            p_out = self.primary(dummy)
            n_primary = p_out.size(1)

        self.digit_caps = DigitCaps(
            n_primary=n_primary,
            primary_dim=caps_dim,
            n_classes=n_classes,
            class_dim=class_dim,
            routing_iters=routing_iters
        )

    def forward(self, x):
        u = self.primary(x)          # (B, n_primary, caps_dim)
        v = self.digit_caps(u)       # (B, n_classes, class_dim)
        # logits = L2 norm of each class capsule
        logits = v.norm(dim=-1)      # (B, n_classes)
        return logits


# ---------------------------------------------------------------------------
# training helpers
# ---------------------------------------------------------------------------

def _make_sgd(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def train_model(model, Xtr, Ytr, epochs, batch, lr, seed):
    C.set_seed(seed)
    opt = _make_sgd(model, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# EOT-PGD: expectation over transformations for gradient masking check
# ---------------------------------------------------------------------------

def eot_pgd(model, x, y, eps, steps, alpha, eot_samples):
    """
    PGD where each gradient step uses the average gradient over `eot_samples`
    independent forward passes (with fresh routing iterations, same input).
    This bypasses gradient masking from non-differentiable or stochastic
    components (here: the iterative routing fixed-point loop).

    For a deterministic model, EOT simply averages identical gradients
    (redundant). For a model with gradient masking, EOT reveals true gradients
    that vanilla PGD misses.
    """
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)

    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        # average gradient over eot_samples (each is an independent graph)
        avg_grad = torch.zeros_like(xa)
        for _ in range(eot_samples):
            loss = F.cross_entropy(model(xa), y)
            g, = torch.autograd.grad(loss, xa, retain_graph=False)
            avg_grad = avg_grad + g.detach()
        avg_grad = avg_grad / eot_samples
        xa = xa.detach() + alpha * avg_grad.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


@torch.no_grad()
def clean_acc(model, X, Y, batch=256):
    correct = 0
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i+batch], Y[i:i+batch]
        correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / X.size(0)


def pgd_asr(model, X, Y, eps, steps, alpha, batch=128):
    """Attack success rate (fraction of correctly classified samples flipped)."""
    model.eval()
    flips, corrects = [], []
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i+batch], Y[i:i+batch]
        with torch.no_grad():
            correct = model(xb).argmax(1) == yb
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps, alpha=alpha)
        with torch.no_grad():
            flipped = model(xa).argmax(1) != yb
        flips.append(flipped.cpu())
        corrects.append(correct.cpu())
    flips = torch.cat(flips).numpy()
    corrects = torch.cat(corrects).numpy().astype(bool)
    return float(flips[corrects].mean()) if corrects.sum() > 0 else float("nan")


def eot_pgd_asr(model, X, Y, eps, steps, alpha, eot_samples, batch=64):
    """ASR under EOT-PGD (gradient masking check)."""
    model.eval()
    flips, corrects = [], []
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i+batch], Y[i:i+batch]
        with torch.no_grad():
            correct = model(xb).argmax(1) == yb
        xa = eot_pgd(model, xb, yb, eps=eps, steps=steps, alpha=alpha,
                     eot_samples=eot_samples)
        with torch.no_grad():
            flipped = model(xa).argmax(1) != yb
        flips.append(flipped.cpu())
        corrects.append(correct.cpu())
    flips = torch.cat(flips).numpy()
    corrects = torch.cat(corrects).numpy().astype(bool)
    return float(flips[corrects].mean()) if corrects.sum() > 0 else float("nan")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H414  Capsule Network Robustness: Dynamic Routing vs SmallCNN Baseline")
    out("=" * 80)
    out("Ref: Sabour, Frosst, Hinton (2017). Dynamic Routing Between Capsules.")
    out("Hypothesis: capsule routing-by-agreement creates implicit manifold")
    out("  projection that may reduce adversarial susceptibility, but prior")
    out("  adaptive-attack work suggests this is largely gradient masking.")
    out("Masking check: EOT-PGD with gradient averaging over routing passes.")
    out("")
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        ROUTING_ITERS={ROUTING_ITERS}  EOT_SAMPLES={EOT_SAMPLES} "
        f"EOT_PGD_STEPS={EOT_PGD_STEPS}")
    out(f"        device={C.DEVICE}")
    out("")

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    # ---- SmallCNN baseline ----
    out("\n[1] Training SmallCNN baseline...")
    C.set_seed(SEED)
    cnn = C.build_model("cnn", META, width=32).to(C.DEVICE)
    cnn = train_model(cnn, Xtr, Ytr, EPOCHS, BATCH, LR, SEED)
    cnn_clean = clean_acc(cnn, Xte, Yte)
    out(f"    SmallCNN clean acc = {cnn_clean:.4f}")

    out("    [1a] SmallCNN: vanilla PGD ASR...")
    cnn_pgd = pgd_asr(cnn, Xte, Yte, EPS, PGD_STEPS, PGD_ALPHA)
    out(f"    SmallCNN PGD ASR = {cnn_pgd:.4f}  ({time.time()-t0:.0f}s)")

    out("    [1b] SmallCNN: FGSM ASR...")
    cnn_fgsm = pgd_asr(cnn, Xte, Yte, EPS, steps=1, alpha=EPS)
    out(f"    SmallCNN FGSM ASR = {cnn_fgsm:.4f}  ({time.time()-t0:.0f}s)")

    out("    [1c] SmallCNN: EOT-PGD ASR (masking check)...")
    cnn_eot = eot_pgd_asr(cnn, Xte, Yte, EPS, EOT_PGD_STEPS, PGD_ALPHA, EOT_SAMPLES)
    out(f"    SmallCNN EOT-PGD ASR = {cnn_eot:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- CapsNet ----
    out("\n[2] Building and training CapsNet (Sabour 2017 dynamic routing)...")
    C.set_seed(SEED)
    caps = CapsNet(
        in_ch=1, n_classes=10, routing_iters=ROUTING_ITERS,
        n_caps_channels=16, caps_dim=8, class_dim=16
    ).to(C.DEVICE)
    n_params_caps = sum(p.numel() for p in caps.parameters())
    n_params_cnn = sum(p.numel() for p in cnn.parameters())
    out(f"    CapsNet params: {n_params_caps:,}  |  SmallCNN params: {n_params_cnn:,}")
    caps = train_model(caps, Xtr, Ytr, EPOCHS, BATCH, LR, SEED)
    caps_clean = clean_acc(caps, Xte, Yte)
    out(f"    CapsNet clean acc = {caps_clean:.4f}  ({time.time()-t0:.0f}s)")

    out("    [2a] CapsNet: vanilla PGD ASR...")
    caps_pgd = pgd_asr(caps, Xte, Yte, EPS, PGD_STEPS, PGD_ALPHA)
    out(f"    CapsNet PGD ASR = {caps_pgd:.4f}  ({time.time()-t0:.0f}s)")

    out("    [2b] CapsNet: FGSM ASR...")
    caps_fgsm = pgd_asr(caps, Xte, Yte, EPS, steps=1, alpha=EPS)
    out(f"    CapsNet FGSM ASR = {caps_fgsm:.4f}  ({time.time()-t0:.0f}s)")

    out("    [2c] CapsNet: EOT-PGD ASR (gradient masking check)...")
    caps_eot = eot_pgd_asr(caps, Xte, Yte, EPS, EOT_PGD_STEPS, PGD_ALPHA, EOT_SAMPLES)
    out(f"    CapsNet EOT-PGD ASR = {caps_eot:.4f}  ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- margin analysis ----
    out("\n[3] Margin analysis (clean inputs)...")
    cnn_margins = C.margin(cnn, Xte, Yte)
    caps_margins = C.margin(caps, Xte, Yte)
    out(f"    SmallCNN margins: mean={cnn_margins.mean():.4f} "
        f"std={cnn_margins.std():.4f} min={cnn_margins.min():.4f}")
    out(f"    CapsNet  margins: mean={caps_margins.mean():.4f} "
        f"std={caps_margins.std():.4f} min={caps_margins.min():.4f}")

    # ---- summary table ----
    out("\n" + "=" * 80)
    out("[4] SUMMARY TABLE")
    out("=" * 80)
    hdr = "{:<14} {:>10} {:>12} {:>12} {:>14} {:>14}".format(
        "model", "clean_acc", "FGSM_ASR", "PGD_ASR",
        "EOT-PGD_ASR", "mask_ratio")
    out(hdr)
    out("-" * len(hdr))
    # mask_ratio = EOT-PGD ASR / vanilla PGD ASR  (>1 => masking suspected)
    cnn_mask = cnn_eot / cnn_pgd if cnn_pgd > 0 else float("nan")
    caps_mask = caps_eot / caps_pgd if caps_pgd > 0 else float("nan")
    out("{:<14} {:>10.4f} {:>12.4f} {:>12.4f} {:>14.4f} {:>14.2f}x".format(
        "SmallCNN", cnn_clean, cnn_fgsm, cnn_pgd, cnn_eot, cnn_mask))
    out("{:<14} {:>10.4f} {:>12.4f} {:>12.4f} {:>14.4f} {:>14.2f}x".format(
        "CapsNet", caps_clean, caps_fgsm, caps_pgd, caps_eot, caps_mask))
    out("-" * len(hdr))

    # ---- verdict ----
    out("\n" + "=" * 80)
    out("[5] VERDICT")
    out("=" * 80)
    pgd_delta = caps_pgd - cnn_pgd          # negative => CapsNet more robust
    eot_delta = caps_eot - cnn_eot          # negative => CapsNet more robust under EOT
    masking_gap = caps_eot - caps_pgd       # how much EOT increases ASR vs vanilla PGD

    out(f"  CapsNet  vanilla PGD ASR: {caps_pgd:.4f}  (SmallCNN: {cnn_pgd:.4f}  "
        f"delta={pgd_delta:+.4f})")
    out(f"  CapsNet  EOT-PGD ASR:     {caps_eot:.4f}  (SmallCNN: {cnn_eot:.4f}  "
        f"delta={eot_delta:+.4f})")
    out(f"  CapsNet masking gap (EOT-PGD minus vanilla PGD): {masking_gap:+.4f}  "
        f"(>0 => masking detected)")
    out(f"  CapsNet mask_ratio: {caps_mask:.2f}x  SmallCNN mask_ratio: {cnn_mask:.2f}x")
    out("")

    # classification logic
    capsnet_seems_robust = pgd_delta < -0.03        # >=3pp more robust by vanilla PGD
    masking_detected = masking_gap > 0.05 and caps_mask > cnn_mask * 1.2
    eot_erases_gap = eot_delta > -0.01              # EOT removes CapsNet advantage

    if capsnet_seems_robust and masking_detected and eot_erases_gap:
        verdict = ("GRADIENT_MASKING: CapsNet appears more robust under vanilla PGD, "
                   "but EOT-PGD erases the advantage — confirming the robustness is "
                   "an artifact of gradient masking in the routing-by-agreement loop.")
    elif capsnet_seems_robust and not masking_detected:
        verdict = ("GENUINE_PARTIAL_ROBUSTNESS: CapsNet shows lower PGD ASR than CNN, "
                   "and EOT-PGD does not significantly close the gap — suggesting some "
                   "genuine robustness from the routing manifold projection, not just "
                   "gradient masking.")
    elif not capsnet_seems_robust and masking_detected:
        verdict = ("MASKING_WITHOUT_ROBUSTNESS: CapsNet does not appear more robust "
                   "under vanilla PGD, but the EOT gap is still larger than CNN, "
                   "suggesting the routing gradients are partially masked without "
                   "providing actual adversarial benefit.")
    else:
        verdict = ("NO_ROBUSTNESS: CapsNet shows no meaningful robustness advantage "
                   "over SmallCNN under either vanilla PGD or EOT-PGD. Routing-by-"
                   "agreement alone is insufficient for adversarial robustness.")

    out(f"  VERDICT: {verdict}")
    out("")
    out(f"  Hypothesis: CAPSNET ROUTING MAY CREATE IMPLICIT MANIFOLD PROJECTION — "
        f"{'SUPPORTED (partially)' if capsnet_seems_robust else 'NOT SUPPORTED'}")
    out(f"  Masking artifact detected: {'YES' if masking_detected else 'NO'}")

    out("")
    out(f"done in {time.time()-t0:.1f}s")

    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
