"""
H445 - Deep k-NN classifier head (Papernot & McDaniel 2018, "Deep k-Nearest
Neighbors", arXiv:1803.04765) on Fashion-MNIST.

Gap addressed: G2 (defence families barely touched) in CAMPAIGN_GAP_MAP.md
section 5. The campaign has never installed a non-parametric (retrieval) head
on top of a learned feature extractor and tested whether removing the linear
softmax boundary buys input-space robustness.

Hypothesis
----------
Adversarial vulnerability is amplified by the deep network's softmax/linear
classification head, whose locally linear decision boundary is exactly what
PGD exploits (FGSM = first-order linearisation of CE around x). Replacing
that head with a k-NN look-up over the penultimate features stored at train
time should:
  (a) remove the exploitable linear boundary,
  (b) force any successful attack to move x in INPUT space to a point whose
      penultimate feature is closer to a different class's neighbours,
  (c) therefore raise the attack budget required to flip a sample.

If true, DkNN should reduce PGD ASR relative to softmax under a fair
white-box attack. If the only "robustness" we see comes from gradient
masking (k-NN argmin is non-differentiable, so naive PGD through the head is
near-useless), then the BPDA / feature-space attack should re-attack the
defence to near-baseline ASR.

Critique / risks
----------------
1. **Gradient masking.** k-NN classification is piecewise-constant. White-box
   PGD with CE on the *re-classified* output (one-hot logits) has zero
   useful gradient. Sitawarin & Wagner (arXiv:1903.08333, 2019) showed naive
   PGD on DkNN gives apparent robustness which collapses under heuristic /
   gradient-approximation attacks (AdvKNN, Li et al. arXiv:1911.06591). We
   therefore evaluate THREE attack flavours:
     (i)   Naive-PGD on k-NN one-hot logits         <- expected to fail
           (this is the "obfuscated gradients" trap),
     (ii)  Surrogate-PGD: PGD against a softmax head (CE-trained) on the
           SAME backbone and transfer to k-NN (transfer attack),
     (iii) Feature-PGD (BPDA-style): PGD in *input* space whose loss is
           the negative L2 distance to the nearest correct-class feature
           PLUS positive L2 distance to the nearest incorrect-class
           feature - this is differentiable through the backbone and is
           the strongest adaptive attack of the three. This is the
           "deep k-NN aware" adaptive attack recommended by Sitawarin 2019
           and is the BPDA-style linear surrogate for the k-NN head.
2. **Backbone confound.** A k-NN head over CE-trained features is not the
   same defence as a k-NN head over AT-trained features. We evaluate BOTH.
   If only AT features help, the k-NN head is doing nothing; if CE+k-NN
   also helps under (iii), the head matters.
3. **k sweep.** k in {1, 5, 25, 100}. Papernot 2018 used k=75 with
   credibility scores; here we use plain majority vote at multiple k.
4. **Soft-kNN ablation.** A differentiable softmax-weighted kNN classifier
   (temperature T) is included as a sanity check: it should match
   hard-kNN clean acc and is by construction NOT masked (white-box PGD
   has gradient through it).

Controls (six conditions)
-------------------------
  A. CE-softmax baseline (canonical campaign baseline).
  B. k-NN head over CE-trained backbone, k in {1, 5, 25, 100}.
  C. k-NN head over PGD-AT-trained backbone, k in {1, 5, 25, 100}.
  D. Soft-kNN head (differentiable, T=1.0) over CE backbone, k=25.
  E. Transfer attack: PGD adv generated on the A baseline, evaluated on
     every B/C/D classifier.
  F. Feature-PGD (BPDA-style adaptive attack) on every B/C/D classifier.

References
----------
  Papernot, N. & McDaniel, P. (2018). Deep k-Nearest Neighbors: Towards
    Confident, Interpretable and Robust Deep Learning. arXiv:1803.04765.
  Sitawarin, C. & Wagner, D. (2019). On the Robustness of Deep K-Nearest
    Neighbors. IEEE S&P Deep Learning and Security Workshop.
    arXiv:1903.08333.
  Li, X. et al. (2019). AdvKnn: Adversarial Attacks on K-Nearest Neighbor
    Classifiers with Approximate Gradients. arXiv:1911.06591.
  Athalye, A., Carlini, N. & Wagner, D. (2018). Obfuscated Gradients Give
    a False Sense of Security: Circumventing Defenses to Adversarial
    Examples. ICML 2018.

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

# ---- config ------------------------------------------------------------------
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
K_VALUES = [1, 5, 25, 100]
SOFT_KNN_T = 1.0           # soft-kNN temperature
SOFT_KNN_K = 25
AT_STEPS = 7               # PGD-AT inner steps for the AT backbone
WIDTH = 32
EMBED_DIM = 256
META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h445_deep_knn_head_output.txt",
)


# ---- backbone with penultimate feature exposure ------------------------------

class FeatCNN(nn.Module):
    """SmallCNN matching the campaign default; exposes penultimate features.

    forward(x) returns logits (10-d). features(x) returns the EMBED_DIM-d
    penultimate activation (post-ReLU of the first head linear). The k-NN
    classifier uses features(x); the CE-softmax baseline uses forward(x).
    """
    def __init__(self):
        super().__init__()
        w = WIDTH

        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(), nn.MaxPool2d(2)]

        self.conv = nn.Sequential(*block(1, w), *block(w, w * 2),
                                  *block(w * 2, w * 4))
        feat = 28 // 8   # 3
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(w * 4 * feat * feat, EMBED_DIM)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(EMBED_DIM, 10)

    def features(self, x):
        h = self.conv(x)
        h = self.flat(h)
        h = self.act(self.fc1(h))
        return h

    def forward(self, x):
        return self.fc2(self.features(x))


# ---- training ----------------------------------------------------------------

def _make_sgd(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)


def train_ce(Xtr, Ytr, seed):
    """Standard cross-entropy training."""
    C.set_seed(seed)
    model = FeatCNN().to(C.DEVICE)
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


def train_pgd_at(Xtr, Ytr, seed):
    """PGD adversarial training (Madry 2018) on the same backbone."""
    C.set_seed(seed)
    model = FeatCNN().to(C.DEVICE)
    opt = _make_sgd(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    alpha = 2.5 * EPS / AT_STEPS
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            model.eval()
            xa = C.pgd(model, xb, yb, eps=EPS, steps=AT_STEPS, alpha=alpha)
            model.train()
            opt.zero_grad()
            F.cross_entropy(model(xa), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- k-NN classifiers (hard and soft) ---------------------------------------

@torch.no_grad()
def _stack_features(backbone, X, batch=256):
    parts = []
    for i in range(0, X.size(0), batch):
        parts.append(backbone.features(X[i:i + batch]))
    return torch.cat(parts, dim=0)


class HardKNNHead:
    """Majority-vote k-NN over stored training features. Returns one-hot
    "logits" (correct class = 0, others = -1e9) so argmax = k-NN label.
    NOT differentiable; useful for clean acc / transfer / feature-PGD eval."""
    def __init__(self, backbone, Xtr_feat, Ytr, k, n_classes=10):
        self.backbone = backbone
        self.Xtr_feat = Xtr_feat   # (N, D)
        self.Ytr = Ytr             # (N,)
        self.k = k
        self.n_classes = n_classes

    def __call__(self, x):
        # Forward through backbone (with grad if x requires it, else no_grad).
        if x.requires_grad:
            e = self.backbone.features(x)
        else:
            with torch.no_grad():
                e = self.backbone.features(x)
        with torch.no_grad():
            # batched cdist to keep memory bounded
            B = e.size(0)
            logits = torch.full((B, self.n_classes), -1e9, device=x.device)
            CHUNK = 256
            for i in range(0, B, CHUNK):
                ei = e[i:i + CHUNK].detach()
                d = torch.cdist(ei, self.Xtr_feat, p=2)   # (b, N)
                idx = d.topk(self.k, largest=False).indices  # (b, k)
                nn_labels = self.Ytr[idx]                    # (b, k)
                # majority vote
                votes = torch.zeros(idx.size(0), self.n_classes,
                                    device=x.device)
                votes.scatter_add_(
                    1, nn_labels,
                    torch.ones_like(nn_labels, dtype=torch.float))
                logits[i:i + CHUNK] = torch.where(
                    votes > 0, votes.log(), torch.full_like(votes, -1e9))
            return logits

    def parameters(self):
        return self.backbone.parameters()

    def eval(self):
        self.backbone.eval(); return self

    def train(self):
        self.backbone.train(); return self


class SoftKNNHead:
    """Differentiable softmax-weighted k-NN (Goldberger 2005 NCA-flavour).

    logits[c] = logsumexp over the k nearest neighbours of class c of
               (-||e - e_n||^2 / T). White-box PGD can flow through this.
    """
    def __init__(self, backbone, Xtr_feat, Ytr, k, T=1.0, n_classes=10):
        self.backbone = backbone
        self.Xtr_feat = Xtr_feat
        self.Ytr = Ytr
        self.k = k
        self.T = T
        self.n_classes = n_classes

    def __call__(self, x):
        e = self.backbone.features(x)        # (B, D); grad iff x has grad
        # distances to ALL training features (we want differentiability)
        d = torch.cdist(e, self.Xtr_feat, p=2).pow(2)     # (B, N)
        # take top-k nearest (smallest distance) per sample
        top_d, top_i = d.topk(self.k, largest=False)       # (B, k)
        top_lab = self.Ytr[top_i]                           # (B, k)
        # weights via softmax(-d/T) over the k
        w = F.softmax(-top_d / self.T, dim=1)              # (B, k)
        logits = torch.zeros(x.size(0), self.n_classes, device=x.device)
        logits.scatter_add_(1, top_lab, w)
        # avoid log(0)
        return (logits + 1e-12).log()

    def parameters(self):
        return self.backbone.parameters()

    def eval(self):
        self.backbone.eval(); return self

    def train(self):
        self.backbone.train(); return self


# ---- feature-PGD (BPDA-style adaptive attack on k-NN head) -------------------

def feature_pgd(backbone, Xtr_feat, Ytr, x, y, eps, steps=PGD_STEPS,
                alpha=PGD_ALPHA, k=25):
    """Sitawarin-2019-style adaptive attack on the k-NN head.

    Loss in input space:
        L(x) = - mean_dist(e(x), k nearest correct-class train feats)
               + mean_dist(e(x), k nearest incorrect-class train feats)
    Ascend on L to push the embedding away from its own class and toward
    other classes. Differentiable through the backbone, so this is BPDA on
    the (non-differentiable) k-NN argmin: linear surrogate = mean k-NN
    distance.
    """
    backbone.eval()
    n_classes = int(Ytr.max().item()) + 1
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    # Pre-split train features by class.
    class_feats = [Xtr_feat[Ytr == c] for c in range(n_classes)]
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        e = backbone.features(xa)                        # (B, D)
        loss = 0.0
        # mean over batch of (-d_correct_k + d_other_k)
        d_corr = torch.zeros(xa.size(0), device=xa.device)
        d_othr = torch.zeros(xa.size(0), device=xa.device)
        for c in range(n_classes):
            mask = (y == c)
            if mask.sum() == 0:
                continue
            ec = e[mask]
            # correct-class distance (smaller k nearest of same class)
            d_cc = torch.cdist(ec, class_feats[c], p=2)
            d_corr[mask] = d_cc.topk(k, largest=False).values.mean(dim=1)
            # other-class distance: nearest across all OTHER classes
            other = torch.cat([class_feats[j] for j in range(n_classes)
                               if j != c], dim=0)
            d_oo = torch.cdist(ec, other, p=2)
            d_othr[mask] = d_oo.topk(k, largest=False).values.mean(dim=1)
        # We want to increase d_corr (push away from correct) and
        # decrease d_othr (pull toward wrong). Ascent on this loss:
        loss = (d_corr - d_othr).mean()
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def eval_clean_and_attack(clf, Xte, Yte, label):
    """clean acc + naive-PGD ASR + naive-FGSM ASR (gradient through clf head)."""
    _, acc = C.logits_and_acc(clf, Xte, Yte)
    fg = C.attack_success(clf, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(clf, Xte, Yte, attack="pgd",  eps=EPS,
                          steps=PGD_STEPS)
    return {"label": label, "acc": acc, "fgsm_naive": fg["asr"],
            "pgd_naive": pg["asr"]}


def transfer_asr(clf, Xte_adv, Yte):
    """Evaluate ASR of pre-computed adv inputs Xte_adv on clf,
    restricted to samples clf classifies correctly on clean Xte."""
    # we need a 'clean correct mask' from the matching clean Xte; but the
    # caller passes Yte and adv x in the same order so we just evaluate raw
    # ASR over originally-correct samples (computed here):
    return None  # implemented inline in main


# ---- main --------------------------------------------------------------------

def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 78)
    out("H445  Deep k-NN classifier head (Papernot & McDaniel 2018)")
    out("=" * 78)
    out("Hypothesis: replacing softmax head with k-NN over penultimate features")
    out("removes the linear adversarial boundary; risk = gradient masking.")
    out("Adaptive attack: feature-PGD (Sitawarin 2019, arXiv:1903.08333) +")
    out("transfer attack from CE-softmax baseline.")
    out("")
    out(f"config: DS={DS} N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS}")
    out(f"        LR={LR} BATCH={BATCH} SEED={SEED} EPS={EPS} "
        f"PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        K_VALUES={K_VALUES} SOFT_KNN_K={SOFT_KNN_K} "
        f"SOFT_KNN_T={SOFT_KNN_T} AT_STEPS={AT_STEPS}")
    out(f"        WIDTH={WIDTH} EMBED_DIM={EMBED_DIM} device={C.DEVICE}")
    out("")
    flush()

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                        seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    # =================================================================
    # [A] CE-softmax baseline
    # =================================================================
    out("[A] Training CE-softmax baseline...")
    ce_model = train_ce(Xtr, Ytr, SEED)
    r_a = eval_clean_and_attack(ce_model, Xte, Yte, "A:CE-softmax")
    out(f"    clean={r_a['acc']:.4f}  FGSM_ASR={r_a['fgsm_naive']:.4f}  "
        f"PGD_ASR(white)={r_a['pgd_naive']:.4f}")
    flush()

    # generate transfer adversarials from A (used to attack B/C/D)
    out("    Generating transfer-PGD examples from baseline A...")
    Xte_transfer = []
    for i in range(0, Xte.size(0), 256):
        xb = Xte[i:i + 256]; yb = Yte[i:i + 256]
        Xte_transfer.append(C.pgd(ce_model, xb, yb,
                                  eps=EPS, steps=PGD_STEPS,
                                  alpha=PGD_ALPHA))
    Xte_transfer = torch.cat(Xte_transfer, dim=0)

    # =================================================================
    # [B] k-NN head over CE backbone (k in K_VALUES)
    # =================================================================
    out("\n[B] k-NN head over CE backbone:")
    ce_feat_tr = _stack_features(ce_model, Xtr)
    out(f"    train features stacked: shape={tuple(ce_feat_tr.shape)}")

    results_b = []
    for k in K_VALUES:
        clf = HardKNNHead(ce_model, ce_feat_tr, Ytr, k=k)
        r = eval_clean_and_attack(clf, Xte, Yte, f"B:kNN(CE)-k={k}")
        # Transfer attack from A
        with torch.no_grad():
            # restrict to originally-correct samples
            clean_pred = []
            for i in range(0, Xte.size(0), 256):
                clean_pred.append(clf(Xte[i:i + 256]).argmax(1).cpu())
            clean_pred = torch.cat(clean_pred)
            corr_mask = (clean_pred == Yte.cpu()).numpy().astype(bool)
            tr_pred = []
            for i in range(0, Xte_transfer.size(0), 256):
                tr_pred.append(clf(Xte_transfer[i:i + 256]).argmax(1).cpu())
            tr_pred = torch.cat(tr_pred)
            flipped = (tr_pred != Yte.cpu()).numpy()
        transfer_asr_val = float(flipped[corr_mask].mean()) \
            if corr_mask.sum() > 0 else float("nan")
        # Feature-PGD adaptive attack
        adv_chunks = []
        for i in range(0, Xte.size(0), 256):
            adv_chunks.append(
                feature_pgd(ce_model, ce_feat_tr, Ytr,
                            Xte[i:i + 256], Yte[i:i + 256],
                            eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA,
                            k=min(k, SOFT_KNN_K)))
        Xte_fp = torch.cat(adv_chunks, dim=0)
        with torch.no_grad():
            fp_pred = []
            for i in range(0, Xte_fp.size(0), 256):
                fp_pred.append(clf(Xte_fp[i:i + 256]).argmax(1).cpu())
            fp_pred = torch.cat(fp_pred)
            fp_flip = (fp_pred != Yte.cpu()).numpy()
        feat_pgd_asr = float(fp_flip[corr_mask].mean()) \
            if corr_mask.sum() > 0 else float("nan")
        r["transfer_asr"] = transfer_asr_val
        r["feat_pgd_asr"] = feat_pgd_asr
        results_b.append(r)
        out(f"    k={k:>3}  clean={r['acc']:.4f}  "
            f"PGD_ASR(naive)={r['pgd_naive']:.4f}  "
            f"transfer_ASR={transfer_asr_val:.4f}  "
            f"feat_PGD_ASR={feat_pgd_asr:.4f}")
        flush()

    # =================================================================
    # [C] k-NN head over PGD-AT backbone
    # =================================================================
    out("\n[C] Training PGD-AT backbone for k-NN head...")
    at_model = train_pgd_at(Xtr, Ytr, SEED)
    r_at = eval_clean_and_attack(at_model, Xte, Yte, "AT-backbone(softmax)")
    out(f"    AT softmax baseline: clean={r_at['acc']:.4f}  "
        f"PGD_ASR={r_at['pgd_naive']:.4f}")
    at_feat_tr = _stack_features(at_model, Xtr)
    out(f"    AT-train features stacked: shape={tuple(at_feat_tr.shape)}")
    flush()

    results_c = []
    for k in K_VALUES:
        clf = HardKNNHead(at_model, at_feat_tr, Ytr, k=k)
        r = eval_clean_and_attack(clf, Xte, Yte, f"C:kNN(AT)-k={k}")
        # Transfer attack from CE baseline (cross-model transfer)
        with torch.no_grad():
            clean_pred = []
            for i in range(0, Xte.size(0), 256):
                clean_pred.append(clf(Xte[i:i + 256]).argmax(1).cpu())
            clean_pred = torch.cat(clean_pred)
            corr_mask = (clean_pred == Yte.cpu()).numpy().astype(bool)
            tr_pred = []
            for i in range(0, Xte_transfer.size(0), 256):
                tr_pred.append(clf(Xte_transfer[i:i + 256]).argmax(1).cpu())
            tr_pred = torch.cat(tr_pred)
            flipped = (tr_pred != Yte.cpu()).numpy()
        transfer_asr_val = float(flipped[corr_mask].mean()) \
            if corr_mask.sum() > 0 else float("nan")
        # Feature-PGD adaptive on the AT backbone
        adv_chunks = []
        for i in range(0, Xte.size(0), 256):
            adv_chunks.append(
                feature_pgd(at_model, at_feat_tr, Ytr,
                            Xte[i:i + 256], Yte[i:i + 256],
                            eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA,
                            k=min(k, SOFT_KNN_K)))
        Xte_fp = torch.cat(adv_chunks, dim=0)
        with torch.no_grad():
            fp_pred = []
            for i in range(0, Xte_fp.size(0), 256):
                fp_pred.append(clf(Xte_fp[i:i + 256]).argmax(1).cpu())
            fp_pred = torch.cat(fp_pred)
            fp_flip = (fp_pred != Yte.cpu()).numpy()
        feat_pgd_asr = float(fp_flip[corr_mask].mean()) \
            if corr_mask.sum() > 0 else float("nan")
        r["transfer_asr"] = transfer_asr_val
        r["feat_pgd_asr"] = feat_pgd_asr
        results_c.append(r)
        out(f"    k={k:>3}  clean={r['acc']:.4f}  "
            f"PGD_ASR(naive)={r['pgd_naive']:.4f}  "
            f"transfer_ASR={transfer_asr_val:.4f}  "
            f"feat_PGD_ASR={feat_pgd_asr:.4f}")
        flush()

    # =================================================================
    # [D] Soft-kNN head (differentiable, white-box PGD)
    # =================================================================
    out("\n[D] Soft-kNN head over CE backbone (differentiable)...")
    soft_clf = SoftKNNHead(ce_model, ce_feat_tr, Ytr,
                           k=SOFT_KNN_K, T=SOFT_KNN_T)
    r_d = eval_clean_and_attack(soft_clf, Xte, Yte,
                                f"D:SoftKNN(CE)-k={SOFT_KNN_K}")
    # transfer
    with torch.no_grad():
        clean_pred = []
        for i in range(0, Xte.size(0), 256):
            clean_pred.append(soft_clf(Xte[i:i + 256]).argmax(1).cpu())
        clean_pred = torch.cat(clean_pred)
        corr_mask_d = (clean_pred == Yte.cpu()).numpy().astype(bool)
        tr_pred = []
        for i in range(0, Xte_transfer.size(0), 256):
            tr_pred.append(soft_clf(Xte_transfer[i:i + 256]).argmax(1).cpu())
        tr_pred = torch.cat(tr_pred)
        flipped = (tr_pred != Yte.cpu()).numpy()
    transfer_d = float(flipped[corr_mask_d].mean()) \
        if corr_mask_d.sum() > 0 else float("nan")
    # feature-PGD on D (same surrogate; D shares CE backbone)
    adv_chunks = []
    for i in range(0, Xte.size(0), 256):
        adv_chunks.append(
            feature_pgd(ce_model, ce_feat_tr, Ytr,
                        Xte[i:i + 256], Yte[i:i + 256],
                        eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA,
                        k=SOFT_KNN_K))
    Xte_fp = torch.cat(adv_chunks, dim=0)
    with torch.no_grad():
        fp_pred = []
        for i in range(0, Xte_fp.size(0), 256):
            fp_pred.append(soft_clf(Xte_fp[i:i + 256]).argmax(1).cpu())
        fp_pred = torch.cat(fp_pred)
        fp_flip = (fp_pred != Yte.cpu()).numpy()
    feat_pgd_d = float(fp_flip[corr_mask_d].mean()) \
        if corr_mask_d.sum() > 0 else float("nan")
    r_d["transfer_asr"] = transfer_d
    r_d["feat_pgd_asr"] = feat_pgd_d
    out(f"    clean={r_d['acc']:.4f}  "
        f"PGD_ASR(white,diffble)={r_d['pgd_naive']:.4f}  "
        f"transfer_ASR={transfer_d:.4f}  feat_PGD_ASR={feat_pgd_d:.4f}")
    flush()

    # =================================================================
    # Summary
    # =================================================================
    out("")
    out("=" * 78)
    out("SUMMARY TABLE")
    out("=" * 78)
    hdr = (f"{'Condition':<26} {'Clean':>7} {'PGDnaive':>9} "
           f"{'Transfer':>9} {'FeatPGD':>8}")
    out(hdr)
    out("-" * 64)
    # baseline A (no transfer / feat-pgd: it IS the source)
    out(f"{r_a['label']:<26} {r_a['acc']:>7.4f} "
        f"{r_a['pgd_naive']:>9.4f} {'-':>9} {'-':>8}")
    out(f"{'AT-backbone(softmax)':<26} {r_at['acc']:>7.4f} "
        f"{r_at['pgd_naive']:>9.4f} {'-':>9} {'-':>8}")
    for r in results_b + results_c + [r_d]:
        out(f"{r['label']:<26} {r['acc']:>7.4f} "
            f"{r['pgd_naive']:>9.4f} "
            f"{r['transfer_asr']:>9.4f} {r['feat_pgd_asr']:>8.4f}")
    out("")

    # Verdict logic
    out("=" * 78)
    out("VERDICT")
    out("=" * 78)
    base_pgd = r_a["pgd_naive"]
    best_ce_knn = min(results_b, key=lambda r: r["feat_pgd_asr"])
    best_at_knn = min(results_c, key=lambda r: r["feat_pgd_asr"])
    out(f"Baseline (A) PGD ASR              : {base_pgd:.4f}")
    out(f"Best CE-kNN feature-PGD ASR       : "
        f"{best_ce_knn['feat_pgd_asr']:.4f} ({best_ce_knn['label']})")
    out(f"Best AT-kNN feature-PGD ASR       : "
        f"{best_at_knn['feat_pgd_asr']:.4f} ({best_at_knn['label']})")
    out(f"AT-softmax PGD ASR (reference)    : {r_at['pgd_naive']:.4f}")
    out(f"Soft-kNN(CE) white-box PGD ASR    : {r_d['pgd_naive']:.4f}")
    out("")

    # Naive PGD on hard-kNN should look "robust" (masking) but feature-PGD
    # should re-attack. Diagnose the masking:
    masking_b = any(
        (r["pgd_naive"] < 0.5 * base_pgd) and (r["feat_pgd_asr"] > 0.8 * base_pgd)
        for r in results_b)
    real_gain_ce_knn = (best_ce_knn["feat_pgd_asr"] < base_pgd - 0.05)
    real_gain_at_knn = (best_at_knn["feat_pgd_asr"] < r_at["pgd_naive"] - 0.05)

    if masking_b:
        out("DIAGNOSIS: naive PGD on hard-kNN(CE) appears 'robust' but "
            "feature-PGD re-attacks the head -> the naive metric was "
            "gradient masking (Athalye 2018 / Sitawarin 2019).")
    else:
        out("DIAGNOSIS: no large gap between naive-PGD and feature-PGD on "
            "CE-kNN; the head is not masking by this measure.")
    out("")
    if real_gain_ce_knn:
        out("CE-kNN head: SUPPORTED - real reduction in PGD ASR vs baseline.")
    else:
        out("CE-kNN head: NOT SUPPORTED - feature-PGD ASR ties or exceeds "
            "the CE-softmax baseline. The non-parametric head alone does "
            "not buy input-space robustness.")
    if real_gain_at_knn:
        out("AT-kNN head: SUPPORTED - kNN head on AT features improves over "
            "AT-softmax under adaptive attack.")
    else:
        out("AT-kNN head: NOT SUPPORTED - kNN head adds no robustness on top "
            "of PGD-AT under feature-PGD; the AT backbone is the load-bearer.")
    out("")
    elapsed = time.time() - t0
    out(f"Total elapsed: {elapsed:.1f}s")
    out("=" * 78)
    flush()
    print(f"\nOutput written to {OUT_FILE}")


if __name__ == "__main__":
    main()
