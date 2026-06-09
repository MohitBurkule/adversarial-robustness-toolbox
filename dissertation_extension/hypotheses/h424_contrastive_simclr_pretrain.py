"""
H424 - SimCLR-style contrastive self-supervised pretraining robustness.

Hypothesis: augmentation-invariant features learned via SimCLR contrastive
pretraining (Chen et al. 2020, ICML) are inherently more robust than supervised
features, because the encoder is explicitly trained to be invariant to aggressive
augmentations (random crop, horizontal flip, color jitter). Hendrycks et al. (2019,
NeurIPS) showed SSL pretraining improves corruption robustness; we test whether
this generalises to l-inf adversarial robustness. The invariance enforced by the
NT-Xent objective forces the encoder to ignore image-level perturbations, which may
reduce sensitivity to adversarial perturbations as a beneficial side-effect.

Design (10 pretrain + 10 finetune = 20 epochs total; matches campaign standard):
  - Augmentation pair: random resized crop (0.6-1.0, scale), random horizontal flip,
    color jitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1) applied as
    a random additive grayscale shift (no torchvision dependency). Each sample produces
    two views; the NT-Xent loss (temperature=0.5) maximises agreement between the pair.
  - Backbone: shared CNN encoder (same width=32 as campaign standard); projection head
    is a 2-layer MLP (enc_dim -> 128 -> 64).
  - Three conditions (all same total compute budget):
      A. Supervised baseline: 20 epochs supervised CE on Xtr (standard, no SSL).
      B. SSL + linear probe: 10 epochs SimCLR pretrain, then freeze encoder and train
         a linear classifier for 10 epochs.
      C. SSL + fine-tune: 10 epochs SimCLR pretrain, then unfreeze all and fine-tune
         with CE + low LR for 10 epochs.
  - Evaluation: clean acc, FGSM ASR, PGD ASR (eps=0.1, 10 steps).

Config: N_TRAIN=6000, N_EVAL=2000, PRETRAIN_EPOCHS=10, FINETUNE_EPOCHS=10,
LR_PRETRAIN=0.05, LR_LINEAR=0.1, LR_FINETUNE=0.01, BATCH=128, TEMP=0.5,
EPS=0.1, PGD_STEPS=10, SEED=0, CNN width=32.

References:
  Chen et al. 2020: "A Simple Framework for Contrastive Learning of Visual
    Representations." ICML 2020. https://arxiv.org/abs/2002.05709
  Hendrycks et al. 2019: "Using Self-Supervised Learning Can Improve Model
    Robustness and Uncertainty." NeurIPS 2019. https://arxiv.org/abs/1906.12340
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
PRETRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 10
LR_PRETRAIN = 0.05
LR_LINEAR = 0.1
LR_FINETUNE = 0.01
BATCH = 128
TEMP = 0.5          # NT-Xent temperature (SimCLR default)
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
SEED = 0
CNN_WIDTH = 32

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h424_contrastive_simclr_pretrain_output.txt",
)


# ---- augmentation ---------------------------------------------------------

def augment_pair(X, seed_offset=0):
    """Return two augmented views of X (B,1,H,W) -> (Xa, Xb) each (B,1,H,W).

    Augmentations (grayscale-safe, no torchvision):
      1. Random resized crop: scale in [0.6, 1.0], then resize back to 28x28
         via nearest-neighbour slicing (approx).
      2. Random horizontal flip with p=0.5.
      3. Colour jitter: random brightness shift in [-0.4, 0.4] clamped to [0,1].
    All operations are differentiable-agnostic; applied on CPU tensors.
    """
    B, C_, H, W = X.shape
    device = X.device
    Xc = X.detach().cpu()

    def _aug(Xc, g):
        out = []
        for i in range(B):
            img = Xc[i]  # (1,H,W)
            # random resized crop
            scale = (torch.rand(1, generator=g).item() * 0.4 + 0.6)  # [0.6,1.0]
            ch = int(round(H * scale))
            cw = int(round(W * scale))
            ch = max(ch, 4); cw = max(cw, 4)
            top = int(torch.randint(0, H - ch + 1, (1,), generator=g).item())
            left = int(torch.randint(0, W - cw + 1, (1,), generator=g).item())
            crop = img[:, top:top + ch, left:left + cw]  # (1,ch,cw)
            # nearest-neighbour resize back to HxW
            resized = F.interpolate(
                crop.unsqueeze(0), size=(H, W), mode="nearest"
            ).squeeze(0)
            # random horizontal flip
            if torch.rand(1, generator=g).item() < 0.5:
                resized = torch.flip(resized, dims=[2])
            # brightness jitter
            shift = (torch.rand(1, generator=g).item() - 0.5) * 0.8  # [-0.4,0.4]
            resized = (resized + shift).clamp(0, 1)
            out.append(resized)
        return torch.stack(out, dim=0)

    ga = torch.Generator(); ga.manual_seed(SEED + seed_offset)
    gb = torch.Generator(); gb.manual_seed(SEED + seed_offset + 10**7)
    Xa = _aug(Xc, ga).to(device)
    Xb = _aug(Xc, gb).to(device)
    return Xa, Xb


# ---- projection head ------------------------------------------------------

class ProjectionHead(nn.Module):
    """2-layer MLP projection head (SimCLR standard)."""

    def __init__(self, in_dim, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---- NT-Xent loss ---------------------------------------------------------

def nt_xent_loss(za, zb, temp):
    """NT-Xent contrastive loss (Chen et al. 2020).

    za, zb: (B, D) L2-normalised projections of the two views.
    Returns scalar loss.
    """
    B = za.size(0)
    z = torch.cat([za, zb], dim=0)  # (2B, D)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temp  # (2B, 2B)
    # mask out self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    # positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])
    loss = F.cross_entropy(sim, labels)
    return loss


# ---- encoder extraction ---------------------------------------------------

def get_encoder_features(model, X, batch=256):
    """Extract penultimate (pre-logit) features from a C.build_model CNN.

    Assumes model has a .features() or we hook the last linear layer.
    We forward-hook the layer just before the final linear to get the flat vector.
    """
    # Identify the last Linear layer
    last_lin = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_lin = m
    assert last_lin is not None, "No Linear layer found in model"

    feats_list = []
    handle_input = []

    def hook(mod, inp, out):
        feats_list.append(inp[0].detach())

    h = last_lin.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            _ = model(X[i:i + batch])
    h.remove()
    return torch.cat(feats_list, dim=0)


def enc_dim(model, X):
    """Infer encoder output dimension from one forward pass."""
    f = get_encoder_features(model, X[:2])
    return f.size(1)


# ---- training routines ----------------------------------------------------

def train_supervised_baseline(Xtr, Ytr, seed):
    """Condition A: 20 epochs supervised CE (pretrain + finetune epochs combined)."""
    total_epochs = PRETRAIN_EPOCHS + FINETUNE_EPOCHS
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=CNN_WIDTH, seed=seed)
    opt = torch.optim.SGD(model.parameters(), lr=LR_PRETRAIN,
                          momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(total_epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def pretrain_simclr(Xtr, seed, out_fn):
    """SimCLR contrastive pretraining for PRETRAIN_EPOCHS epochs.

    Returns (backbone, proj_head, feat_dim).
    """
    C.set_seed(seed)
    backbone = C.build_model("cnn", META, width=CNN_WIDTH, seed=seed)
    fdim = enc_dim(backbone, Xtr[:2])
    out_fn(f"    encoder feature dim = {fdim}")
    proj = ProjectionHead(fdim, hidden=128, out_dim=64).to(C.DEVICE)

    params = list(backbone.parameters()) + list(proj.parameters())
    opt = torch.optim.SGD(params, lr=LR_PRETRAIN, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)

    n = Xtr.size(0)
    backbone.train(); proj.train()
    for ep in range(PRETRAIN_EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        ep_loss = 0.0
        n_batch = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            Xb = Xtr[idx]
            Xa_v, Xb_v = augment_pair(Xb, seed_offset=ep * 10000 + i)
            # get backbone features for both views
            def _fwd(Xv):
                feats_list = []
                last_lin = None
                for m in backbone.modules():
                    if isinstance(m, nn.Linear):
                        last_lin = m
                h = last_lin.register_forward_hook(
                    lambda mod, inp, out: feats_list.append(inp[0])
                )
                backbone(Xv)
                h.remove()
                return feats_list[-1]

            fa = _fwd(Xa_v)
            fb = _fwd(Xb_v)
            za = proj(fa)
            zb = proj(fb)
            loss = nt_xent_loss(za, zb, TEMP)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batch += 1
        sched.step()
        if ep == 0 or (ep + 1) % 5 == 0:
            out_fn(f"    pretrain ep {ep+1}/{PRETRAIN_EPOCHS}  "
                   f"NT-Xent={ep_loss/n_batch:.4f}")

    backbone.eval(); proj.eval()
    return backbone, proj, fdim


def train_linear_probe(backbone, Xtr, Ytr, fdim, seed):
    """Condition B: freeze backbone, train linear head for FINETUNE_EPOCHS epochs."""
    C.set_seed(seed + 1)
    for p in backbone.parameters():
        p.requires_grad = False
    # extract features once (frozen encoder)
    with torch.no_grad():
        Ftr = get_encoder_features(backbone, Xtr)
    head = nn.Linear(fdim, META["n_classes"]).to(C.DEVICE)
    opt = torch.optim.SGD(head.parameters(), lr=LR_LINEAR,
                          momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS)
    n = Ftr.size(0)
    head.train()
    for ep in range(FINETUNE_EPOCHS):
        perm = torch.randperm(n, device=Ftr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            F.cross_entropy(head(Ftr[idx]), Ytr[idx]).backward()
            opt.step()
        sched.step()
    head.eval()

    # wrap backbone + head into a single nn.Module for eval_robustness
    class LinearProbeModel(nn.Module):
        def __init__(self, enc, lin):
            super().__init__()
            self.enc = enc
            self.lin = lin

        def forward(self, x):
            # route through backbone up to penultimate, then linear head
            feats_list = []
            last_lin_enc = None
            for m in self.enc.modules():
                if isinstance(m, nn.Linear):
                    last_lin_enc = m
            h = last_lin_enc.register_forward_hook(
                lambda mod, inp, out: feats_list.append(inp[0])
            )
            self.enc(x)
            h.remove()
            return self.lin(feats_list[-1])

    model = LinearProbeModel(backbone, head).to(C.DEVICE)
    model.eval()
    return model


def train_ssl_finetune(backbone, Xtr, Ytr, fdim, seed):
    """Condition C: unfreeze backbone + new linear head, fine-tune for FINETUNE_EPOCHS."""
    C.set_seed(seed + 2)
    for p in backbone.parameters():
        p.requires_grad = True
    # replace final linear layer in backbone with a new one
    last_lin_ref = [None]
    last_lin_name = [None]
    for name, m in backbone.named_modules():
        if isinstance(m, nn.Linear):
            last_lin_ref[0] = m
            last_lin_name[0] = name

    # Attach new classification head in place
    new_head = nn.Linear(fdim, META["n_classes"]).to(C.DEVICE)
    # find parent and replace
    parts = last_lin_name[0].split(".")
    parent = backbone
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_head)

    opt = torch.optim.SGD(backbone.parameters(), lr=LR_FINETUNE,
                          momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS)
    n = Xtr.size(0)
    backbone.train()
    for ep in range(FINETUNE_EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            F.cross_entropy(backbone(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        sched.step()
    backbone.eval()
    return backbone


def eval_robustness(model, X, Y):
    """clean acc, FGSM ASR, PGD ASR."""
    _, acc = C.logits_and_acc(model, X, Y)
    fg = C.attack_success(model, X, Y, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


# ---- main -----------------------------------------------------------------

def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H424  SimCLR contrastive self-supervised pretraining robustness")
    out("=" * 80)
    out("Hypothesis: SSL augmentation-invariant features are inherently more robust")
    out("  than supervised features (Chen et al. 2020; Hendrycks et al. 2019).")
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} PRETRAIN={PRETRAIN_EPOCHS}ep "
        f"FINETUNE={FINETUNE_EPOCHS}ep (total 20ep)")
    out(f"        LR_pretrain={LR_PRETRAIN} LR_linear={LR_LINEAR} "
        f"LR_finetune={LR_FINETUNE} BATCH={BATCH} TEMP={TEMP}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} CNN_WIDTH={CNN_WIDTH} SEED={SEED}")
    out(f"        device={C.DEVICE}")
    out("")

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    rows = []

    # ---- Condition A: Supervised baseline ----
    out("\n" + "=" * 80)
    out("[A] Supervised baseline (20 epochs CE, no SSL)")
    out("=" * 80)
    sup = train_supervised_baseline(Xtr, Ytr, SEED)
    acc_a, fgsm_a, pgd_a = eval_robustness(sup, Xte, Yte)
    out(f"    clean_acc={acc_a:.4f}  FGSM_ASR={fgsm_a:.4f}  PGD_ASR={pgd_a:.4f}")
    rows.append({"cond": "A: supervised", "acc": acc_a, "fgsm": fgsm_a, "pgd": pgd_a})
    flush_file()

    # ---- SimCLR pretraining (shared for B and C) ----
    out("\n" + "=" * 80)
    out("[pretrain] SimCLR NT-Xent pretraining (10 epochs)")
    out("=" * 80)
    backbone_b, proj_b, fdim = pretrain_simclr(Xtr, SEED, out)
    out(f"    pretraining done ({time.time()-t0:.0f}s)")
    flush_file()

    # Need a second pretrained backbone for condition C (so C doesn't clobber B)
    out("\n[pretrain-C] second SimCLR pretraining for condition C...")
    backbone_c, _, _ = pretrain_simclr(Xtr, SEED, out)
    out(f"    second pretraining done ({time.time()-t0:.0f}s)")
    flush_file()

    # ---- Condition B: SSL + linear probe ----
    out("\n" + "=" * 80)
    out("[B] SSL + linear probe (frozen encoder, 10 epochs linear)")
    out("=" * 80)
    model_b = train_linear_probe(backbone_b, Xtr, Ytr, fdim, SEED)
    acc_b, fgsm_b, pgd_b = eval_robustness(model_b, Xte, Yte)
    out(f"    clean_acc={acc_b:.4f}  FGSM_ASR={fgsm_b:.4f}  PGD_ASR={pgd_b:.4f}")
    rows.append({"cond": "B: SSL+linear-probe", "acc": acc_b,
                 "fgsm": fgsm_b, "pgd": pgd_b})
    flush_file()

    # ---- Condition C: SSL + fine-tune ----
    out("\n" + "=" * 80)
    out("[C] SSL + fine-tune (full network, 10 epochs LR=0.01)")
    out("=" * 80)
    model_c = train_ssl_finetune(backbone_c, Xtr, Ytr, fdim, SEED)
    acc_c, fgsm_c, pgd_c = eval_robustness(model_c, Xte, Yte)
    out(f"    clean_acc={acc_c:.4f}  FGSM_ASR={fgsm_c:.4f}  PGD_ASR={pgd_c:.4f}")
    rows.append({"cond": "C: SSL+finetune", "acc": acc_c,
                 "fgsm": fgsm_c, "pgd": pgd_c})
    flush_file()

    # ---- MAIN TABLE ----
    out("\n" + "=" * 80)
    out("[TABLE] Summary")
    out("=" * 80)
    hdr = "{:<22} {:>10} {:>10} {:>10}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:<22} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["cond"], r["acc"], r["fgsm"], r["pgd"]))
    out("-" * len(hdr))

    # ---- VERDICT ----
    out("\n" + "=" * 80)
    out("[VERDICT]")
    out("=" * 80)
    base = rows[0]
    for r in rows[1:]:
        d_pgd = base["pgd"] - r["pgd"]   # positive => more robust
        d_acc = r["acc"] - base["acc"]
        out(f"  {r['cond']}: PGD_ASR {base['pgd']:.4f}->{r['pgd']:.4f} "
            f"(delta={-d_pgd:+.4f}); clean_acc {base['acc']:.4f}->{r['acc']:.4f} "
            f"(delta={d_acc:+.4f})")

    best_ssl = min(rows[1:], key=lambda r: r["pgd"])
    pgd_gain = base["pgd"] - best_ssl["pgd"]
    acc_ok = best_ssl["acc"] >= base["acc"] - 0.02

    if pgd_gain > 0.03 and acc_ok:
        verdict = ("SUPPORTED: SSL pretraining yields meaningfully more robust "
                   "features (PGD_ASR reduced >3pp) without significant accuracy cost, "
                   "consistent with the augmentation-invariance hypothesis "
                   "(Chen et al. 2020; Hendrycks et al. 2019).")
    elif pgd_gain > 0.01 and acc_ok:
        verdict = ("WEAKLY SUPPORTED: SSL pretraining gives modest robustness "
                   "benefit (1-3pp PGD_ASR reduction) at acceptable accuracy cost.")
    elif pgd_gain > 0.03 and not acc_ok:
        verdict = ("PARTIAL: SSL robustness gain is real but comes with >2pp "
                   "clean-accuracy cost; trade-off may not be favourable.")
    else:
        verdict = ("NOT SUPPORTED: SSL pretraining does not reduce PGD_ASR "
                   "meaningfully; augmentation-invariance alone does not confer "
                   "l-inf adversarial robustness on this scale.")

    out(f"\n  best SSL condition: {best_ssl['cond']}")
    out(f"  PGD_ASR gain over supervised: {pgd_gain:+.4f}")
    out(f"  clean_acc delta: {best_ssl['acc']-base['acc']:+.4f}")
    out(f"\n  ONE-LINE VERDICT: {verdict}")
    out("")
    out(f"done in {time.time()-t0:.1f}s")

    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
