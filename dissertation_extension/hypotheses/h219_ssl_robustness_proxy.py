"""
H219 - Self-supervised vs supervised: do SSL-like features produce more robust
representations?

Train 3 models on Fashion-MNIST:
  1. Supervised CNN (cross-entropy)
  2. SimCLR-lite: NT-Xent contrastive pretraining, then frozen encoder + linear head
  3. Rotation SSL: predict rotation angle {0,90,180,270°} as pretext, then
     frozen encoder + linear head for classification

Compare clean accuracy, FGSM ASR, PGD ASR, margin distribution.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS           = "fashion_mnist"
N_EVAL       = 300
SEED         = 0
EPS          = 0.1
EPOCHS_SUP   = 10
EPOCHS_SSL   = 20   # pretraining epochs
EPOCHS_HEAD  = 10   # linear head finetuning
BATCH        = 128
TEMP         = 0.5   # NT-Xent temperature
PROJ_DIM     = 128


# ---------------------------------------------------------------------------
# Encoder shared by SimCLR and Rotation-SSL
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """CNN encoder that outputs a flat feature vector."""
    def __init__(self, in_ch=1, width=32):
        super().__init__()
        def blk(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.net = nn.Sequential(
            *blk(in_ch, width),
            *blk(width, width * 2),
            *blk(width * 2, width * 4),
        )
        self.feat_dim = width * 4 * (28 // 8) * (28 // 8)  # = 128 * 9 = 1152 for width=8
        # but we want a compact embedding; add flatten
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(self.net(x))


class SimCLRModel(nn.Module):
    def __init__(self, encoder, feat_dim, proj_dim=PROJ_DIM):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

    def encode(self, x):
        return self.encoder(x)


class LinearClassifier(nn.Module):
    def __init__(self, feat_dim, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class FullClassifier(nn.Module):
    """Frozen encoder + linear head."""
    def __init__(self, encoder, n_classes=10):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.head = nn.Linear(encoder.feat_dim, n_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)


# ---------------------------------------------------------------------------
# Augmentations for SimCLR
# ---------------------------------------------------------------------------
def simclr_aug(x):
    """Random crop+hflip for a batch tensor (N,C,H,W)."""
    # random hflip
    flip = torch.rand(x.size(0)) < 0.5
    xf = x.clone()
    xf[flip] = xf[flip].flip(-1)
    # random crop (pad=2)
    xp = F.pad(xf, [2, 2, 2, 2], mode="reflect")
    out = []
    for i in range(xp.size(0)):
        r = torch.randint(0, 5, (1,)).item()
        c = torch.randint(0, 5, (1,)).item()
        out.append(xp[i:i+1, :, r:r+28, c:c+28])
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------
def nt_xent_loss(z1, z2, temperature=TEMP):
    """NT-Xent contrastive loss for batch of (z1, z2) positive pairs."""
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)   # (2N, D)
    sim = torch.mm(z, z.t()) / temperature   # (2N, 2N)

    # mask out self-similarities
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # positive pair indices: (i, i+N) and (i+N, i)
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    loss = F.cross_entropy(sim, labels)
    return loss


# ---------------------------------------------------------------------------
# Rotation SSL helpers
# ---------------------------------------------------------------------------
def make_rotation_batch(X, Y):
    """Rotate each image by k*90° and return (rotated_X, rotation_labels)."""
    N = X.size(0)
    ks = torch.randint(0, 4, (N,))
    out = []
    for i in range(N):
        out.append(torch.rot90(X[i], int(ks[i].item()), dims=[1, 2]).unsqueeze(0))
    return torch.cat(out, dim=0), ks.to(X.device)


class RotationModel(nn.Module):
    """Encoder + 4-class rotation head."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.feat_dim, 4)

    def forward(self, x):
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------
def train_supervised(model, Xtr, Ytr, epochs=EPOCHS_SUP, lr=0.05):
    C.train_model(model, Xtr, Ytr, epochs=epochs, opt="sgd", lr=lr, ncls=10)


def train_simclr(simclr_model, Xtr, epochs=EPOCHS_SSL, lr=1e-3):
    opt = torch.optim.Adam(simclr_model.parameters(), lr=lr)
    n = Xtr.size(0)
    simclr_model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        total_loss = 0.0
        nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = Xtr[idx]
            x1 = simclr_aug(xb)
            x2 = simclr_aug(xb)
            z1 = simclr_model(x1)
            z2 = simclr_model(x2)
            loss = nt_xent_loss(z1, z2)
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            total_loss += loss.item()
            nb += 1
        if (ep + 1) % 5 == 0:
            print(f"    SimCLR epoch {ep+1}/{epochs}  loss={total_loss/nb:.3f}")
    simclr_model.eval()


def train_rotation(rot_model, Xtr, epochs=EPOCHS_SSL, lr=0.01):
    opt = torch.optim.SGD([p for p in rot_model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    rot_model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        total_loss = 0.0; nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = Xtr[idx]
            xr, rot_labels = make_rotation_batch(xb, None)
            opt.zero_grad()
            F.cross_entropy(rot_model(xr), rot_labels).backward()
            opt.step()
            nb += 1
        sched.step()
        if (ep + 1) % 5 == 0:
            print(f"    RotSSL epoch {ep+1}/{epochs}")
    rot_model.eval()


def train_linear_head(full_clf, Xtr, Ytr, epochs=EPOCHS_HEAD, lr=0.05):
    """Train only the head (encoder frozen)."""
    opt = torch.optim.SGD([p for p in full_clf.head.parameters()],
                          lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    full_clf.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            with torch.no_grad():
                h = full_clf.encoder(xb)
            out = full_clf.head(h)
            opt.zero_grad()
            F.cross_entropy(out, yb).backward()
            opt.step()
        sched.step()
    full_clf.eval()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("=== H219: SSL vs Supervised — robustness of representations ===")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  N_EVAL={N_EVAL}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_EVAL, seed=SEED)

    ENC_WIDTH = 8   # encoder width (keeps param count tractable)
    # feat_dim = ENC_WIDTH*4 * (28//8)^2 = 32 * 9 = 288

    results = {}

    # -----------------------------------------------------------------------
    # 1. Supervised CNN
    # -----------------------------------------------------------------------
    print("\n--- 1. Supervised CNN ---")
    t0 = time.time()
    C.set_seed(SEED)
    model_sup = C.build_model("cnn", meta, width=32, seed=SEED)
    C.train_model(model_sup, Xtr, Ytr, epochs=EPOCHS_SUP, opt="sgd", lr=0.05,
                  ncls=meta["n_classes"])

    _, clean_acc = C.logits_and_acc(model_sup, Xte, Yte)
    # Unfreeze for gradient-based attacks
    for p in model_sup.parameters(): p.requires_grad_(True)
    X_fgsm = C.fgsm(model_sup, Xte, Yte, eps=EPS)
    X_pgd  = C.pgd(model_sup,  Xte, Yte, eps=EPS, steps=10, alpha=0.01)
    with torch.no_grad():
        fgsm_asr = (model_sup(X_fgsm).argmax(1) != Yte).float().mean().item()
        pgd_asr  = (model_sup(X_pgd ).argmax(1) != Yte).float().mean().item()
    mgn = C.margin(model_sup, Xte, Yte)
    results["supervised"] = dict(clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                                  pgd_asr=pgd_asr,
                                  margin_mean=float(mgn.mean()), margin_std=float(mgn.std()))
    print(f"  clean={clean_acc:.3f}  FGSM={fgsm_asr:.3f}  "
          f"PGD={pgd_asr:.3f}  mgn={mgn.mean():.3f}±{mgn.std():.3f}  "
          f"({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # 2. SimCLR-lite
    # -----------------------------------------------------------------------
    print("\n--- 2. SimCLR-lite pretraining ---")
    t0 = time.time()
    C.set_seed(SEED)
    enc_simclr = Encoder(in_ch=1, width=ENC_WIDTH).to(C.DEVICE)
    simclr_m = SimCLRModel(enc_simclr, enc_simclr.feat_dim, proj_dim=PROJ_DIM).to(C.DEVICE)
    train_simclr(simclr_m, Xtr, epochs=EPOCHS_SSL)

    print("  Training linear head...")
    full_simclr = FullClassifier(enc_simclr, n_classes=10).to(C.DEVICE)
    train_linear_head(full_simclr, Xtr, Ytr, epochs=EPOCHS_HEAD)

    _, clean_acc = C.logits_and_acc(full_simclr, Xte, Yte)
    # Unfreeze for gradient-based attacks
    for p in full_simclr.parameters(): p.requires_grad_(True)
    X_fgsm = C.fgsm(full_simclr, Xte, Yte, eps=EPS)
    X_pgd  = C.pgd(full_simclr,  Xte, Yte, eps=EPS, steps=10, alpha=0.01)
    with torch.no_grad():
        fgsm_asr = (full_simclr(X_fgsm).argmax(1) != Yte).float().mean().item()
        pgd_asr  = (full_simclr(X_pgd ).argmax(1) != Yte).float().mean().item()
    mgn = C.margin(full_simclr, Xte, Yte)
    results["simclr_lite"] = dict(clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                                   pgd_asr=pgd_asr,
                                   margin_mean=float(mgn.mean()), margin_std=float(mgn.std()))
    print(f"  clean={clean_acc:.3f}  FGSM={fgsm_asr:.3f}  "
          f"PGD={pgd_asr:.3f}  mgn={mgn.mean():.3f}±{mgn.std():.3f}  "
          f"({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # 3. Rotation-prediction SSL
    # -----------------------------------------------------------------------
    print("\n--- 3. Rotation-prediction SSL ---")
    t0 = time.time()
    C.set_seed(SEED)
    enc_rot = Encoder(in_ch=1, width=ENC_WIDTH).to(C.DEVICE)
    rot_m = RotationModel(enc_rot).to(C.DEVICE)
    train_rotation(rot_m, Xtr, epochs=EPOCHS_SSL)

    print("  Training linear head...")
    full_rot = FullClassifier(enc_rot, n_classes=10).to(C.DEVICE)
    train_linear_head(full_rot, Xtr, Ytr, epochs=EPOCHS_HEAD)

    _, clean_acc = C.logits_and_acc(full_rot, Xte, Yte)
    # Unfreeze for gradient-based attacks
    for p in full_rot.parameters(): p.requires_grad_(True)
    X_fgsm = C.fgsm(full_rot, Xte, Yte, eps=EPS)
    X_pgd  = C.pgd(full_rot,  Xte, Yte, eps=EPS, steps=10, alpha=0.01)
    with torch.no_grad():
        fgsm_asr = (full_rot(X_fgsm).argmax(1) != Yte).float().mean().item()
        pgd_asr  = (full_rot(X_pgd ).argmax(1) != Yte).float().mean().item()
    mgn = C.margin(full_rot, Xte, Yte)
    results["rotation_ssl"] = dict(clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                                    pgd_asr=pgd_asr,
                                    margin_mean=float(mgn.mean()), margin_std=float(mgn.std()))
    print(f"  clean={clean_acc:.3f}  FGSM={fgsm_asr:.3f}  "
          f"PGD={pgd_asr:.3f}  mgn={mgn.mean():.3f}±{mgn.std():.3f}  "
          f"({time.time()-t0:.1f}s)")

    # --- Summary ---
    print("\n" + "=" * 74)
    print("--- Summary ---")
    print(f"{'Model':<14} {'CleanAcc':>9} {'FGSM_ASR':>9} {'PGD_ASR':>8} "
          f"{'Mgn_mean':>9} {'Mgn_std':>8}")
    for model_name, r in results.items():
        print(f"{model_name:<14} {r['clean_acc']:>9.3f} {r['fgsm_asr']:>9.3f} "
              f"{r['pgd_asr']:>8.3f} {r['margin_mean']:>9.3f} {r['margin_std']:>8.3f}")
    print("=" * 74)
    print("Interpretation: if SSL-pretrained models (SimCLR/Rotation) show lower")
    print("PGD ASR or higher margins than the supervised model — despite no label")
    print("information during pretraining — this suggests that label-free feature")
    print("learning can incidentally produce more robust representations.")
    print("Note: SSL models here use a smaller encoder (width=8) vs supervised (32)")
    print("to keep compute similar; capacity differences should be considered.")
    print("=" * 74)


if __name__ == "__main__":
    main()
