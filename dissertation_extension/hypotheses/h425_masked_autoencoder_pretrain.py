"""
H425 - Masked Autoencoder pretraining yields more adversarially robust features.

Hypothesis: reconstructing an image from 75% masked patches forces the encoder to
build holistic, distributed representations that are less reliant on the local
high-frequency texture cues adversarial attackers exploit.  A classifier fine-tuned
on top of MAE features should therefore require larger perturbations to flip its
predictions than a purely supervised baseline trained from scratch.

Reference: He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners"
(CVPR 2022).  Patch-masking ratio 75% following the paper's ablation sweet-spot.

Three conditions evaluated on Fashion-MNIST:
  1. supervised   -- SmallCNN trained end-to-end with cross-entropy (baseline).
  2. mae_linear   -- MAE encoder (pretrained) + frozen; only linear head trained.
  3. mae_finetune -- MAE encoder pretrained, then full network fine-tuned with CE.

Metrics per condition (evaluated on correctly-classified test samples):
  * clean accuracy
  * FGSM attack-success rate  (eps=8/255)
  * PGD-10 attack-success rate (eps=8/255)
  * mean min-eps to flip (binary-search, 8 iters)
  * univariate AUROC: decision margin vs PGD-flip label
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DS          = "fashion_mnist"
SEEDS       = [0, 1, 2]
PRETRAIN_EP = 10          # MAE reconstruction pre-training epochs
FINETUNE_EP = 10          # Classifier fine-tuning / supervised training epochs
PATCH       = 4           # patch size (px); 28/4 = 7 patches per side -> 49 patches
MASK_RATIO  = 0.75        # fraction of patches masked during MAE pre-train
EPS         = 8.0 / 255   # L-inf attack budget
ALPHA       = 2.0 / 255   # PGD step size
PGD_STEPS   = 10
N_TRAIN     = 6000
N_EVAL      = 2000

# ---------------------------------------------------------------------------
# Patch utilities
# ---------------------------------------------------------------------------
IMG_PATCHES = (28 // PATCH) ** 2   # 49


def patchify(x: torch.Tensor) -> torch.Tensor:
    """(B,1,28,28) -> (B, N_patches, patch_dim)."""
    B = x.size(0)
    p = PATCH
    x = x.view(B, 1, 28 // p, p, 28 // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()   # (B, h, w, 1, p, p)
    return x.view(B, IMG_PATCHES, p * p)             # (B, 49, 16)


def unpatchify(patches: torch.Tensor) -> torch.Tensor:
    """(B, N_patches, patch_dim) -> (B,1,28,28)."""
    B = patches.size(0)
    p = PATCH
    n = 28 // p
    x = patches.view(B, n, n, 1, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, 1, 28, 28)


def random_mask(B: int, device: torch.device):
    """Return (B, N_patches) bool mask where True = keep, False = mask."""
    n_keep = int(IMG_PATCHES * (1 - MASK_RATIO))
    noise  = torch.rand(B, IMG_PATCHES, device=device)
    ids    = noise.argsort(dim=1)
    mask   = torch.zeros(B, IMG_PATCHES, dtype=torch.bool, device=device)
    mask.scatter_(1, ids[:, :n_keep], True)
    return mask


# ---------------------------------------------------------------------------
# MAE architecture: tiny encoder + decoder for Fashion-MNIST patches
# ---------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    """Lightweight patch-level encoder: linear projection -> small transformer-lite MLP."""
    def __init__(self, patch_dim: int = PATCH * PATCH, embed_dim: int = 128):
        super().__init__()
        self.proj  = nn.Linear(patch_dim, embed_dim)
        self.norm  = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)
        # positional embedding (learned)
        self.pos_emb = nn.Parameter(torch.randn(1, IMG_PATCHES, embed_dim) * 0.02)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, N, patch_dim) -> (B, N, embed_dim)."""
        x = self.proj(patches) + self.pos_emb
        x = self.norm(x + self.ff(x))
        return x


class MAEModel(nn.Module):
    """Encoder + pixel-space decoder for masked patch reconstruction."""
    def __init__(self, embed_dim: int = 128, patch_dim: int = PATCH * PATCH):
        super().__init__()
        self.encoder  = PatchEncoder(patch_dim, embed_dim)
        # mask token
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # decoder: simple MLP per patch position
        self.decoder  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Linear(embed_dim * 2, patch_dim))

    def forward(self, x: torch.Tensor):
        """Full forward for pre-training: returns (pred_patches, target_patches, mask)."""
        B = x.size(0)
        patches = patchify(x)                          # (B, N, pd)
        mask    = random_mask(B, x.device)             # (B, N) bool, True=keep
        # encode visible patches
        enc_in  = patches.clone()
        enc_in[~mask] = 0.0                            # zero out masked (encoder sees 0)
        enc_out = self.encoder(enc_in)                 # (B, N, E)
        # replace masked positions with mask token
        full    = enc_out.clone()
        full[~mask] = self.mask_tok.expand(B, -1, -1).view(-1, enc_out.size(-1))[
            (~mask).view(-1)]
        # decode all positions
        pred    = self.decoder(full)                   # (B, N, pd)
        return pred, patches, mask

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled feature vector for a full image (no masking)."""
        patches = patchify(x)
        enc_out = self.encoder(patches)                # (B, N, E)
        return enc_out.mean(1)                         # (B, E)  global average pool


class MAEClassifier(nn.Module):
    """MAE encoder (optionally frozen) + linear/MLP classifier head."""
    def __init__(self, mae: MAEModel, n_classes: int = 10, embed_dim: int = 128,
                 frozen_encoder: bool = False):
        super().__init__()
        self.encoder = mae.encoder
        if frozen_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.pool = lambda e: e.mean(1)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = patchify(x)
        enc     = self.encoder(patches)
        return self.head(enc.mean(1))


# ---------------------------------------------------------------------------
# Supervised baseline: SmallCNN from common.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Attack helpers
# ---------------------------------------------------------------------------

def fgsm(model, x, y, eps=EPS):
    xr = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xr), y).backward()
    return (xr + eps * xr.grad.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps=EPS, alpha=ALPHA, steps=PGD_STEPS):
    xa = x.clone().detach()
    for _ in range(steps):
        xa.requires_grad_(True)
        F.cross_entropy(model(xa), y).backward()
        with torch.no_grad():
            xa = (xa + alpha * xa.grad.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return xa.detach()


def min_eps_flip(model, x, y, eps_max=0.3, iters=8):
    lo = torch.zeros(x.size(0), device=x.device)
    hi = torch.full_like(lo, eps_max)
    for _ in range(iters):
        mid = (lo + hi) / 2
        xa  = x.clone().detach()
        for _ in range(5):
            xa.requires_grad_(True)
            F.cross_entropy(model(xa), y).backward()
            with torch.no_grad():
                a = (mid / 4).view(-1, 1, 1, 1)
                xa = (xa + a * xa.grad.sign()).clamp(
                    x - mid.view(-1,1,1,1), x + mid.view(-1,1,1,1)).clamp(0, 1)
        with torch.no_grad():
            flipped = model(xa).argmax(1) != y
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    return hi


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def _train_loader(X, Y, batch=128):
    ds  = torch.utils.data.TensorDataset(X, Y)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)


def pretrain_mae(mae: MAEModel, X: torch.Tensor, epochs: int = PRETRAIN_EP):
    opt = torch.optim.Adam(mae.parameters(), lr=3e-4)
    mae.train()
    for ep in range(epochs):
        total_loss = 0.0
        for (xb,) in torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X), batch_size=128, shuffle=True):
            opt.zero_grad()
            pred, target, mask = mae(xb)
            # loss only on masked patches (He et al. 2022)
            loss = F.mse_loss(pred[~mask], target[~mask])
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"   [MAE pretrain] ep {ep+1}/{epochs}  recon_loss={total_loss:.4f}")


def finetune_classifier(clf: MAEClassifier, X: torch.Tensor, Y: torch.Tensor,
                         epochs: int = FINETUNE_EP):
    opt = torch.optim.Adam(
        [p for p in clf.parameters() if p.requires_grad], lr=1e-3)
    clf.train()
    for ep in range(epochs):
        for xb, yb in _train_loader(X, Y):
            opt.zero_grad()
            F.cross_entropy(clf(xb), yb).backward()
            opt.step()
        print(f"   [classifier]   ep {ep+1}/{epochs} done")


def train_supervised(meta, X, Y, epochs=FINETUNE_EP, seed=0):
    model = C.build_model("cnn", meta, seed=seed)
    C.train_model(model, X, Y, epochs=epochs, opt="adam", lr=1e-3,
                  ncls=meta["n_classes"])
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, Xte, Yte, label: str):
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1)
        correct = preds == Yte
    Xc, Yc = Xte[correct], Yte[correct]
    acc = correct.float().mean().item()

    # margin feature
    with torch.no_grad():
        lg = model(Xc)
        top2, _ = lg.sort(1, descending=True)
        margin  = (top2[:, 0] - top2[:, 1]).cpu().numpy()

    # attacks
    xf  = fgsm(model, Xc, Yc)
    xp  = pgd(model,  Xc, Yc)
    with torch.no_grad():
        fgsm_sr = (model(xf).argmax(1) != Yc).float().mean().item()
        pgd_sr  = (model(xp).argmax(1) != Yc).float().mean().item()
    me  = min_eps_flip(model, Xc, Yc).cpu().numpy()

    # AUROC margin vs pgd-flip
    with torch.no_grad():
        pgd_flip = (model(xp).argmax(1) != Yc).cpu().numpy().astype(int)
    auroc = 0.5
    if C.roc_auc_score is not None and pgd_flip.std() > 0:
        a = C.roc_auc_score(pgd_flip, -margin)   # higher margin -> less likely flip
        auroc = max(a, 1 - a)

    print(f"\n  [{label}]")
    print(f"    clean acc       : {acc:.4f}  (on {Xc.size(0)} correct samples)")
    print(f"    FGSM succ rate  : {fgsm_sr:.4f}")
    print(f"    PGD-10 succ rate: {pgd_sr:.4f}")
    print(f"    mean min_eps    : {me.mean():.4f}")
    print(f"    AUROC(margin,pgd): {auroc:.4f}")

    return {
        "label": label,
        "acc": acc, "n_correct": Xc.size(0),
        "fgsm_sr": fgsm_sr, "pgd_sr": pgd_sr,
        "mean_min_eps": float(me.mean()),
        "auroc_margin_pgd": auroc,
    }


# ---------------------------------------------------------------------------
# Per-seed experiment
# ---------------------------------------------------------------------------

def run_seed(seed: int, meta: dict, Xtr, Ytr, Xte, Yte):
    C.set_seed(seed)
    results = {}

    # --- Condition 1: supervised baseline ---
    print(f"\n  [seed {seed}] Training supervised baseline ({FINETUNE_EP} epochs)...")
    sup = train_supervised(meta, Xtr, Ytr, epochs=FINETUNE_EP, seed=seed)
    results["supervised"] = evaluate(sup, Xte, Yte, label="supervised")

    # --- MAE pretraining (shared for both MAE conditions) ---
    print(f"\n  [seed {seed}] MAE pre-training ({PRETRAIN_EP} epochs)...")
    mae = MAEModel(embed_dim=128).to(C.DEVICE)
    pretrain_mae(mae, Xtr, epochs=PRETRAIN_EP)

    # --- Condition 2: MAE + linear probe (encoder frozen) ---
    print(f"\n  [seed {seed}] MAE linear probe ({FINETUNE_EP} epochs, encoder frozen)...")
    clf_linear = MAEClassifier(mae, n_classes=meta["n_classes"],
                               frozen_encoder=True).to(C.DEVICE)
    finetune_classifier(clf_linear, Xtr, Ytr, epochs=FINETUNE_EP)
    results["mae_linear"] = evaluate(clf_linear, Xte, Yte, label="mae_linear")

    # --- Condition 3: MAE + full fine-tune ---
    print(f"\n  [seed {seed}] MAE full fine-tune ({FINETUNE_EP} epochs)...")
    # re-load fresh copy so linear and finetune share identical pretrained weights
    mae2 = MAEModel(embed_dim=128).to(C.DEVICE)
    mae2.load_state_dict(mae.state_dict())
    clf_ft = MAEClassifier(mae2, n_classes=meta["n_classes"],
                           frozen_encoder=False).to(C.DEVICE)
    finetune_classifier(clf_ft, Xtr, Ytr, epochs=FINETUNE_EP)
    results["mae_finetune"] = evaluate(clf_ft, Xte, Yte, label="mae_finetune")

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    OUT_FILE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "fashion_mnist", "h425_masked_autoencoder_pretrain_output.txt")
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    import io, contextlib
    buf = io.StringIO()

    def _run():
        print("=" * 74)
        print("H425 - Masked Autoencoder Pretraining vs Supervised Baseline")
        print("He et al. 2022 MAE  |  Fashion-MNIST  |  mask_ratio=75%")
        print("=" * 74)
        print(f"Device={C.DEVICE}  pretrain_ep={PRETRAIN_EP}  finetune_ep={FINETUNE_EP}")
        print(f"eps={EPS:.5f}  patch={PATCH}px  patches={IMG_PATCHES}")

        meta = C.dataset_meta(DS)
        Xtr, Ytr, Xte, Yte = C.load_dataset(
            DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=0)

        conditions  = ["supervised", "mae_linear", "mae_finetune"]
        agg: dict[str, list] = {c: [] for c in conditions}

        for seed in SEEDS:
            t0 = time.time()
            print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
            res = run_seed(seed, meta, Xtr, Ytr, Xte, Yte)
            for c in conditions:
                agg[c].append(res[c])
            print(f"\n  seed {seed} runtime: {time.time()-t0:.1f}s")

        # Summary
        print("\n" + "=" * 74)
        print("SUMMARY  (mean across seeds)")
        print(f"{'Condition':<20} {'CleanAcc':>9} {'FGSM_SR':>9} "
              f"{'PGD_SR':>9} {'MinEps':>9} {'AUROC':>7}")
        print("-" * 74)
        for c in conditions:
            rows = agg[c]
            def m(k): return sum(r[k] for r in rows) / len(rows)
            print(f"{c:<20} {m('acc'):>9.4f} {m('fgsm_sr'):>9.4f} "
                  f"{m('pgd_sr'):>9.4f} {m('mean_min_eps'):>9.4f} "
                  f"{m('auroc_margin_pgd'):>7.4f}")
        print("=" * 74)
        print("Interpretation:")
        print("  If mae_finetune has lower PGD_SR and/or higher MinEps than")
        print("  supervised, reconstructing from masked patches has induced")
        print("  holistic features that are harder for local-gradient attacks")
        print("  to exploit (He et al. 2022 MAE hypothesis confirmed).")
        print("  mae_linear isolates the encoder quality from fine-tuning bias.")
        print("=" * 74)

    with contextlib.redirect_stdout(io.StringIO()) as s:
        pass  # warm-up redirect import

    # Tee to stdout + file
    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, d):
            for s in self.streams: s.write(d)
        def flush(self):
            for s in self.streams: s.flush()

    import sys as _sys
    orig_stdout = _sys.stdout
    tee = Tee(orig_stdout, buf)
    _sys.stdout = tee
    try:
        _run()
    finally:
        _sys.stdout = orig_stdout

    with open(OUT_FILE, "w") as f:
        f.write(buf.getvalue())
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
