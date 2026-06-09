"""
H209 - Supervised contrastive pre-training improves adversarial robustness.

Hypothesis: supervised contrastive pre-training (SupCon) achieves PGD-10 ASR
4-7pp lower than standard CE, and 2-4pp lower than unsupervised SimCLR + fine-tune,
because label-aware clustering creates tighter class manifolds.

Grounded in: arXiv:2412.19747 (SupCon + adversarial robustness, Dec 2024).

Three conditions:
  1. Baseline CE: standard CNN, cross-entropy, 20 epochs.
  2. SimCLR + FT: 10 epochs unsupervised contrastive, 10 epochs linear fine-tune.
  3. SupCon + FT: 10 epochs supervised contrastive, 10 epochs linear fine-tune.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign.common import (
    DEVICE, set_seed, load_dataset, dataset_meta, build_model,
    train_model, attack_success, logits_and_acc,
)

DATASET = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 500
PRETRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 10
CE_EPOCHS = 20
ADV_EPS = 0.1
PGD_STEPS = 10
BATCH = 128
PROJ_DIM = 128
TEMPERATURE = 0.07
SEED = 42


# ---------------------------------------------------------------------------
# Encoder: reuse SmallCNN backbone but split head off
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """SmallCNN feature extractor (no classification head)."""
    def __init__(self, in_ch=1, size=28, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.BatchNorm2d(width * 4),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        feat = size // 8
        self.feat_dim = width * 4 * feat * feat
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(self.features(x))


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# Augmentations (simple, for 28x28 grayscale)
# ---------------------------------------------------------------------------
def augment_batch(x):
    """Random crop (pad 4) + horizontal flip + brightness jitter."""
    B, C, H, W = x.shape
    # pad and random crop
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    crops = torch.empty_like(x)
    for i in range(B):
        top = torch.randint(0, 9, (1,)).item()
        left = torch.randint(0, 9, (1,)).item()
        crops[i] = padded[i, :, top:top + H, left:left + W]
    # random horizontal flip
    mask = torch.rand(B, 1, 1, 1, device=x.device) > 0.5
    crops = torch.where(mask, crops.flip(-1), crops)
    # brightness jitter
    jitter = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.4
    crops = (crops * jitter).clamp(0, 1)
    return crops


# ---------------------------------------------------------------------------
# Contrastive losses
# ---------------------------------------------------------------------------
def simclr_loss(z1, z2, temperature=TEMPERATURE):
    """NT-Xent: attract (z1_i, z2_i) pairs, repel all others."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = z @ z.T / temperature  # (2B, 2B)
    # mask out self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    # positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


def supcon_loss(z1, z2, labels, temperature=TEMPERATURE):
    """Supervised contrastive: attract all same-class pairs, repel different-class."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    y = torch.cat([labels, labels])  # (2B,)
    sim = z @ z.T / temperature  # (2B, 2B)

    # mask: same class (excluding self)
    self_mask = torch.eye(2 * B, device=z.device).bool()
    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & (~self_mask)

    # for numerical stability
    logits_max = sim.max(dim=1, keepdim=True).values.detach()
    sim = sim - logits_max
    sim.masked_fill_(self_mask, -1e9)

    # log-sum-exp over all non-self
    exp_sim = sim.exp()
    exp_sim.masked_fill_(self_mask, 0)
    log_sum_exp = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12).log()

    # mean of log-prob over positives
    log_prob = sim - log_sum_exp
    # average over positive pairs per anchor
    n_pos = pos_mask.float().sum(dim=1).clamp_min(1)
    loss = -(pos_mask.float() * log_prob).sum(dim=1) / n_pos
    return loss.mean()


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------
def pretrain_contrastive(encoder, proj_head, Xtr, Ytr, mode="simclr"):
    """Pre-train encoder + projection head with contrastive loss."""
    encoder.train(); proj_head.train()
    params = list(encoder.parameters()) + list(proj_head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    n = Xtr.size(0)

    for ep in range(PRETRAIN_EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            x, y = Xtr[idx], Ytr[idx]
            x1 = augment_batch(x)
            x2 = augment_batch(x)
            z1 = proj_head(encoder(x1))
            z2 = proj_head(encoder(x2))

            if mode == "simclr":
                loss = simclr_loss(z1, z2)
            else:  # supcon
                loss = supcon_loss(z1, z2, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

    encoder.eval(); proj_head.eval()
    return encoder


def finetune_linear(encoder, Xtr, Ytr, n_classes=10):
    """Freeze encoder, train linear classifier."""
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    classifier = LinearClassifier(encoder.feat_dim, n_classes).to(DEVICE)
    opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    n = Xtr.size(0)

    for ep in range(FINETUNE_EPOCHS):
        classifier.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            x, y = Xtr[idx], Ytr[idx]
            with torch.no_grad():
                feat = encoder(x)
            logits = classifier(feat)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # re-enable grads for attack
    for p in encoder.parameters():
        p.requires_grad_(True)

    classifier.eval()
    return classifier


class FullModel(nn.Module):
    """Encoder + linear classifier wrapped as one module for attack_success."""
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.encoder(x))


def main():
    print("=" * 74)
    print("H209 - Supervised contrastive pre-training vs adversarial robustness")
    print("=" * 74)
    t0 = time.time()
    set_seed(SEED)

    meta = dataset_meta(DATASET)
    ch, sz, ncls = meta["channels"], meta["size"], meta["n_classes"]
    Xtr, Ytr, Xte, Yte = load_dataset(DATASET, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    print(f"Dataset: {DATASET}  n_train={Xtr.size(0)}  n_eval={Xte.size(0)}  device={DEVICE}")

    results = {}

    # --- Condition 1: Baseline CE ---
    print("\n--- Condition 1: Baseline CE (20 epochs) ---")
    set_seed(SEED)
    model_ce = build_model("cnn", meta, width=32)
    train_model(model_ce, Xtr, Ytr, epochs=CE_EPOCHS, batch=BATCH,
                opt="adam", lr=1e-3, ncls=ncls)
    _, clean_ce = logits_and_acc(model_ce, Xte, Yte)
    fgsm_ce = attack_success(model_ce, Xte, Yte, attack="fgsm", eps=ADV_EPS, batch=BATCH)
    pgd_ce = attack_success(model_ce, Xte, Yte, attack="pgd", eps=ADV_EPS, steps=PGD_STEPS, batch=BATCH)
    results["Baseline CE"] = (clean_ce, fgsm_ce["asr"], pgd_ce["asr"])
    print(f"  clean={clean_ce:.4f}  fgsm={fgsm_ce['asr']:.4f}  pgd={pgd_ce['asr']:.4f}")

    # --- Condition 2: SimCLR + FT ---
    print("\n--- Condition 2: SimCLR + linear fine-tune ---")
    set_seed(SEED)
    enc_sim = Encoder(ch, sz, width=32).to(DEVICE)
    proj_sim = ProjectionHead(enc_sim.feat_dim, PROJ_DIM).to(DEVICE)
    pretrain_contrastive(enc_sim, proj_sim, Xtr, Ytr, mode="simclr")
    clf_sim = finetune_linear(enc_sim, Xtr, Ytr, ncls)
    model_simclr = FullModel(enc_sim, clf_sim)
    _, clean_sim = logits_and_acc(model_simclr, Xte, Yte)
    fgsm_sim = attack_success(model_simclr, Xte, Yte, attack="fgsm", eps=ADV_EPS, batch=BATCH)
    pgd_sim = attack_success(model_simclr, Xte, Yte, attack="pgd", eps=ADV_EPS, steps=PGD_STEPS, batch=BATCH)
    results["SimCLR + FT"] = (clean_sim, fgsm_sim["asr"], pgd_sim["asr"])
    print(f"  clean={clean_sim:.4f}  fgsm={fgsm_sim['asr']:.4f}  pgd={pgd_sim['asr']:.4f}")

    # --- Condition 3: SupCon + FT ---
    print("\n--- Condition 3: SupCon + linear fine-tune ---")
    set_seed(SEED)
    enc_sup = Encoder(ch, sz, width=32).to(DEVICE)
    proj_sup = ProjectionHead(enc_sup.feat_dim, PROJ_DIM).to(DEVICE)
    pretrain_contrastive(enc_sup, proj_sup, Xtr, Ytr, mode="supcon")
    clf_sup = finetune_linear(enc_sup, Xtr, Ytr, ncls)
    model_supcon = FullModel(enc_sup, clf_sup)
    _, clean_sup = logits_and_acc(model_supcon, Xte, Yte)
    fgsm_sup = attack_success(model_supcon, Xte, Yte, attack="fgsm", eps=ADV_EPS, batch=BATCH)
    pgd_sup = attack_success(model_supcon, Xte, Yte, attack="pgd", eps=ADV_EPS, steps=PGD_STEPS, batch=BATCH)
    results["SupCon + FT"] = (clean_sup, fgsm_sup["asr"], pgd_sup["asr"])
    print(f"  clean={clean_sup:.4f}  fgsm={fgsm_sup['asr']:.4f}  pgd={pgd_sup['asr']:.4f}")

    # --- Summary ---
    elapsed = time.time() - t0
    print(f"\n{'=' * 74}")
    print("RESULTS")
    print(f"{'=' * 74}")
    print(f"  {'Condition':<18} {'Clean Acc':>10} {'FGSM ASR':>10} {'PGD-10 ASR':>12}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*12}")
    for cond, (ca, fa, pa) in results.items():
        print(f"  {cond:<18} {ca:>10.4f} {fa:>10.4f} {pa:>12.4f}")

    delta_ce = pgd_ce["asr"] - pgd_sup["asr"]
    delta_sim = pgd_sim["asr"] - pgd_sup["asr"]
    print(f"\n  SupCon PGD ASR reduction vs CE:     {delta_ce:+.4f} ({delta_ce*100:+.1f}pp)")
    print(f"  SupCon PGD ASR reduction vs SimCLR:  {delta_sim:+.4f} ({delta_sim*100:+.1f}pp)")
    h1 = pgd_sup["asr"] < pgd_ce["asr"]
    h2 = pgd_sup["asr"] < pgd_sim["asr"]
    print(f"  SupCon < CE?     {'YES' if h1 else 'NO'}")
    print(f"  SupCon < SimCLR? {'YES' if h2 else 'NO'}")
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
