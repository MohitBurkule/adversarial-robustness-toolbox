"""
H428 - Triplet margin loss: does metric-learning's explicit embedding margin
translate to a larger input-space adversarial margin?

Hypothesis: cross-entropy training has no explicit constraint on inter-class
separation in embedding space, so decision boundaries can be arbitrarily thin.
Triplet loss (Schroff et al., FaceNet, CVPR 2015) directly maximises
d(anchor, negative) - d(anchor, positive) >= m in L2 embedding space.
We test whether this explicit margin in embedding space produces a different
(ideally larger) adversarial margin in input space, measured by FGSM/PGD ASR.

Three conditions (Fashion-MNIST, N=6000):
  A. CE-baseline  - standard CNN trained with cross-entropy; classify by argmax logit.
  B. Triplet+1NN  - CNN backbone trained with triplet loss; classify by 1-nearest-
                    neighbour over per-class training embeddings at eval time.
  C. Triplet+Centroid - same backbone; classify by nearest class centroid (mean
                    embedding per class). Centroids recomputed on full training set
                    after training.
  D. Triplet+Centroid+Jacobian - condition C + Jacobian-norm penalty on embedding
                    to encourage smooth embedding w.r.t. input.

Triplet mining: semi-hard negatives (d_an > d_ap, d_an < d_ap + m) sampled
within each mini-batch.  Margin m = 1.0 (L2, features L2-normalised to unit sphere).
The backbone is the SmallCNN feature extractor (before the classification head);
the 256-d penultimate activation is taken as the embedding.

References:
  Schroff F., Kalenichenko D., Philbin J. (2015). FaceNet: A Unified Embedding
  for Face Recognition and Clustering. CVPR 2015. arXiv:1503.03832.

Config: N_TRAIN=6000, N_EVAL=2000, EPOCHS=15, LR=0.01, BATCH=128,
        SGD mom=0.9 wd=5e-4, SEED=0, EPS=0.1, PGD_STEPS=10,
        TRIPLET_MARGIN=1.0, EMBED_DIM=256, JAC_LAMBDA=0.01.
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

# ---- config ------------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 15
LR = 0.01
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
TRIPLET_MARGIN = 1.0
EMBED_DIM = 256        # penultimate width (matches SmallCNN head linear input)
JAC_LAMBDA = 0.01      # Jacobian penalty weight for condition D
WIDTH = 32             # CNN conv width
META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h428_triplet_margin_loss_output.txt"
)


# ---- embedding backbone ------------------------------------------------------

class EmbedCNN(nn.Module):
    """SmallCNN with width=32; forward() returns L2-normalised 256-d embedding."""
    def __init__(self):
        super().__init__()
        w = WIDTH
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(
            *block(1, w), *block(w, w * 2), *block(w * 2, w * 4))
        feat = 28 // 8   # = 3
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 4 * feat * feat, EMBED_DIM),
            nn.ReLU())

    def forward(self, x):
        h = self.embed(self.features(x))
        return F.normalize(h, dim=1)   # unit sphere


class CEModel(nn.Module):
    """Standard CNN with cross-entropy head (condition A)."""
    def __init__(self):
        super().__init__()
        w = WIDTH
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(
            *block(1, w), *block(w, w * 2), *block(w * 2, w * 4))
        feat = 28 // 8
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 4 * feat * feat, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, 10))

    def forward(self, x):
        return self.head(self.features(x))


# ---- triplet mining ----------------------------------------------------------

def semi_hard_triplet_loss(embeds, labels, margin):
    """
    Semi-hard negative mining within the batch (Schroff 2015).
    embeds: (B, D) L2-normalised. labels: (B,) long.
    Returns scalar loss (mean over valid anchors).
    """
    sq_dist = torch.cdist(embeds, embeds, p=2).pow(2)   # (B, B)
    loss_total = torch.tensor(0.0, device=embeds.device, requires_grad=True)
    count = 0
    B = embeds.size(0)
    for i in range(B):
        pos_mask = (labels == labels[i]) & (torch.arange(B, device=embeds.device) != i)
        neg_mask = labels != labels[i]
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        d_ap = sq_dist[i][pos_mask].max()   # hardest positive
        d_an_all = sq_dist[i][neg_mask]
        # semi-hard: d_an > d_ap  AND  d_an < d_ap + margin^2
        semi = d_an_all[(d_an_all > d_ap) & (d_an_all < d_ap + margin ** 2)]
        if semi.numel() == 0:
            semi = d_an_all   # fall back to all negatives
        d_an = semi.min()
        loss_total = loss_total + F.relu(d_ap - d_an + margin)
        count += 1
    return loss_total / max(count, 1)


# ---- training ----------------------------------------------------------------

def _make_sgd(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)


def train_ce(Xtr, Ytr, seed):
    """Condition A: standard cross-entropy."""
    C.set_seed(seed)
    model = CEModel().to(C.DEVICE)
    opt = _make_sgd(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_triplet(Xtr, Ytr, seed, jac_penalty=False):
    """Conditions B/C/D: train EmbedCNN with triplet loss."""
    C.set_seed(seed)
    model = EmbedCNN().to(C.DEVICE)
    opt = _make_sgd(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            if jac_penalty:
                xb.requires_grad_(True)
            embeds = model(xb)
            loss = semi_hard_triplet_loss(embeds, yb, TRIPLET_MARGIN)
            if jac_penalty and xb.requires_grad:
                # Frobenius norm of Jacobian: E[||de/dx||_F^2] via random projection
                v = torch.randn_like(embeds)
                v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
                jac_vec = torch.autograd.grad(
                    (embeds * v).sum(), xb,
                    create_graph=True)[0]
                jac_reg = jac_vec.pow(2).sum(dim=(1, 2, 3)).mean()
                loss = loss + JAC_LAMBDA * jac_reg
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- nearest-centroid / 1-NN classifier wrappers ----------------------------

class CentroidClassifier:
    """Wraps EmbedCNN; classifies by nearest L2 centroid. Supports model(x) API."""
    def __init__(self, backbone, centroids):
        self.backbone = backbone   # EmbedCNN (eval mode)
        # centroids: (n_classes, EMBED_DIM) on DEVICE
        self.centroids = centroids

    def __call__(self, x):
        with torch.no_grad() if not x.requires_grad else torch.enable_grad():
            e = self.backbone(x)                     # (B, D) normalised
        # return negative-distance as "logits" so argmax = nearest centroid
        dists = torch.cdist(e, self.centroids, p=2) # (B, n_classes)
        return -dists

    def parameters(self):
        return self.backbone.parameters()

    def eval(self):
        self.backbone.eval()
        return self

    def train(self):
        self.backbone.train()
        return self


class KNNClassifier:
    """1-NN over stored training embeddings."""
    def __init__(self, backbone, Xtr_embed, Ytr):
        self.backbone = backbone
        self.Xtr_embed = Xtr_embed   # (N, D) normalised, on DEVICE
        self.Ytr = Ytr               # (N,) long, on DEVICE
        self.n_classes = int(Ytr.max().item()) + 1

    def __call__(self, x):
        e = self.backbone(x)                          # (B, D)
        dists = torch.cdist(e, self.Xtr_embed, p=2)  # (B, N)
        nn_idx = dists.argmin(dim=1)                  # (B,)
        # convert to one-hot "logits" (just need argmax to be correct class)
        logits = torch.full((x.size(0), self.n_classes), -1e9, device=x.device)
        nn_labels = self.Ytr[nn_idx]
        logits.scatter_(1, nn_labels.unsqueeze(1), 0.0)
        return logits

    def parameters(self):
        return self.backbone.parameters()

    def eval(self):
        self.backbone.eval()
        return self

    def train(self):
        self.backbone.train()
        return self


def compute_centroids(backbone, X, Y, n_classes=10):
    """Return (n_classes, EMBED_DIM) centroid tensor on DEVICE."""
    backbone.eval()
    with torch.no_grad():
        all_e = []
        for i in range(0, X.size(0), BATCH):
            all_e.append(backbone(X[i:i + BATCH]))
        all_e = torch.cat(all_e, dim=0)
    centroids = torch.zeros(n_classes, all_e.size(1), device=C.DEVICE)
    counts = torch.zeros(n_classes, device=C.DEVICE)
    for c in range(n_classes):
        mask = Y == c
        if mask.sum() > 0:
            centroids[c] = all_e[mask].mean(dim=0)
            counts[c] = mask.sum().float()
    centroids = F.normalize(centroids, dim=1)
    return centroids


def compute_train_embeddings(backbone, X):
    backbone.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, X.size(0), BATCH):
            parts.append(backbone(X[i:i + BATCH]))
    return torch.cat(parts, dim=0)


# ---- evaluation --------------------------------------------------------------

def eval_model(model, Xte, Yte, label):
    """Clean acc, FGSM ASR, PGD ASR."""
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    return {"label": label, "acc": acc, "fgsm": fg["asr"], "pgd": pg["asr"]}


# ---- main --------------------------------------------------------------------

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

    out("=" * 78)
    out("H428  Triplet Margin Loss vs Cross-Entropy: adversarial profile comparison")
    out("=" * 78)
    out("Hypothesis: metric-learning's explicit margin in embedding space may")
    out("translate to input-space margin, reducing adversarial vulnerability.")
    out("Reference: Schroff et al., FaceNet, CVPR 2015 (arXiv:1503.03832).")
    out("")
    out(f"config: DS={DS} N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} "
        f"LR={LR} BATCH={BATCH}")
    out(f"        SGD(mom=0.9,wd=5e-4) SEED={SEED} EPS={EPS} "
        f"PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        TRIPLET_MARGIN={TRIPLET_MARGIN} EMBED_DIM={EMBED_DIM} "
        f"JAC_LAMBDA={JAC_LAMBDA} WIDTH={WIDTH}")
    out(f"        device={C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    results = []

    # -- Condition A: CE baseline ----------------------------------------------
    out("[A] Training CE baseline (cross-entropy, standard CNN)...")
    ce_model = train_ce(Xtr, Ytr, SEED)
    r = eval_model(ce_model, Xte, Yte, "A:CE-baseline")
    results.append(r)
    out(f"    clean={r['acc']:.4f}  FGSM_ASR={r['fgsm']:.4f}  PGD_ASR={r['pgd']:.4f}")
    flush_file()

    # -- Condition B: Triplet + 1-NN -------------------------------------------
    out("\n[B] Training EmbedCNN with triplet loss (no Jacobian penalty)...")
    tri_model_b = train_triplet(Xtr, Ytr, SEED, jac_penalty=False)
    out("    Computing training embeddings for 1-NN...")
    Xtr_embed_b = compute_train_embeddings(tri_model_b, Xtr)
    knn_clf = KNNClassifier(tri_model_b, Xtr_embed_b, Ytr)
    r = eval_model(knn_clf, Xte, Yte, "B:Triplet+1NN")
    results.append(r)
    out(f"    clean={r['acc']:.4f}  FGSM_ASR={r['fgsm']:.4f}  PGD_ASR={r['pgd']:.4f}")
    flush_file()

    # -- Condition C: Triplet + Centroid ----------------------------------------
    out("\n[C] Triplet+Centroid (same backbone as B, centroid classifier)...")
    centroids_c = compute_centroids(tri_model_b, Xtr, Ytr, n_classes=10)
    cent_clf = CentroidClassifier(tri_model_b, centroids_c)
    r = eval_model(cent_clf, Xte, Yte, "C:Triplet+Centroid")
    results.append(r)
    out(f"    clean={r['acc']:.4f}  FGSM_ASR={r['fgsm']:.4f}  PGD_ASR={r['pgd']:.4f}")
    flush_file()

    # -- Condition D: Triplet + Centroid + Jacobian penalty --------------------
    out("\n[D] Training EmbedCNN with triplet loss + Jacobian penalty...")
    tri_model_d = train_triplet(Xtr, Ytr, SEED + 1, jac_penalty=True)
    out("    Computing centroids for condition D...")
    centroids_d = compute_centroids(tri_model_d, Xtr, Ytr, n_classes=10)
    cent_jac_clf = CentroidClassifier(tri_model_d, centroids_d)
    r = eval_model(cent_jac_clf, Xte, Yte, "D:Triplet+Centroid+Jacobian")
    results.append(r)
    out(f"    clean={r['acc']:.4f}  FGSM_ASR={r['fgsm']:.4f}  PGD_ASR={r['pgd']:.4f}")
    flush_file()

    # -- Summary table ---------------------------------------------------------
    out("")
    out("=" * 78)
    out("SUMMARY TABLE")
    out("=" * 78)
    hdr = f"{'Condition':<35} {'CleanAcc':>9} {'FGSM_ASR':>9} {'PGD_ASR':>9}"
    out(hdr)
    out("-" * 65)
    for r in results:
        out(f"{r['label']:<35} {r['acc']:>9.4f} {r['fgsm']:>9.4f} {r['pgd']:>9.4f}")
    out("")

    # delta relative to CE baseline
    base_fgsm = results[0]["fgsm"]
    base_pgd  = results[0]["pgd"]
    out("Delta vs CE baseline (negative = fewer attacks succeed):")
    for r in results[1:]:
        dfg = r["fgsm"] - base_fgsm
        dpg = r["pgd"]  - base_pgd
        out(f"  {r['label']:<35}  dFGSM={dfg:+.4f}  dPGD={dpg:+.4f}")

    out("")
    out("Interpretation notes:")
    out("  If triplet conditions show lower ASR -> embedding margin transfers to input.")
    out("  If ASR is similar or higher -> adversarial vulnerability is not explained")
    out("  by lack of embedding-space margin; geometry is set in input space, not")
    out("  embedding space. Jacobian penalty probes whether smoothness of the map")
    out("  independently reduces vulnerability.")
    out("")
    elapsed = time.time() - t0
    out(f"Total elapsed: {elapsed:.1f}s")
    out("=" * 78)
    flush_file()
    print(f"\nOutput written to {OUT_FILE}")


if __name__ == "__main__":
    main()
