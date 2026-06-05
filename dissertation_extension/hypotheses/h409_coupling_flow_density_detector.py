"""
H409 - Coupling-flow (RealNVP/Glow-style) exact-likelihood adversarial detector.

Topic: exactly-invertible nets for adversarial robustness, detection angle.
A normalizing flow built from affine coupling layers (Dinh et al. 2017 RealNVP
arXiv:1605.08803; Kingma & Dhariwal 2018 Glow arXiv:1807.03039) is an EXACTLY
invertible bijection x <-> z whose change-of-variables formula gives the exact
log-density:
        log p(x) = log p(z) + sum_layers log|det J_layer|.
The detection hypothesis (cf. Jacobsen et al. "Excessive Invariance Causes
Adversarial Vulnerability", arXiv:1811.00401, and flow-based OOD detection):
adversarial inputs are pushed off the learned data manifold, so they should sit
in LOWER-density regions -> separable from clean inputs by exact log p(x).

We:
  1. Train a victim SmallCNN on Fashion-MNIST.
  2. Train a small affine-coupling flow on the SAME clean training images
     (no labels). EXACT invertibility verified: x -> z -> x_rec, assert ~0.
  3. Build PGD adversarial copies of correctly-classified test images.
  4. Compute exact log p(x) for clean vs adversarial inputs and report the
     detection AUROC (clean=0 vs adv=1; we use -log p as the adv score, since
     adv is hypothesised to be lower-density). Also report a margin/entropy
     baseline detector and whether the flow adds AUROC over it.

Flow design (pure PyTorch): logit-dequantisation preprocessing + N affine
coupling layers over the flattened 784-dim input with alternating masks, each
with a small MLP s/t net (stable s = tanh). Standard-normal base.

Config: N_TRAIN=6000, EPOCHS=10 (victim), FLOW_EPOCHS=8, SEED=0, EPS=0.1,
PGD_STEPS=10. (Smoke config via env SMOKE=1.)
"""
import os
import sys
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config --------------------------------------------------------------
DS = "fashion_mnist"
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_TRAIN = 2000 if SMOKE else 6000
N_EVAL = 1000 if SMOKE else 2000
EPOCHS = 2 if SMOKE else 10
FLOW_EPOCHS = 2 if SMOKE else 8
LR = 0.05
FLOW_LR = 5e-4
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
IMG_DIM = 28 * 28
N_COUPLING = 6 if SMOKE else 8
HIDDEN = 256
LOGIT_ALPHA = 0.05
META = {"channels": 1, "size": 28, "n_classes": 10}


# ---------------------------------------------------------------------------
# affine-coupling flow (exact bijection)
# ---------------------------------------------------------------------------
class STNet(nn.Module):
    def __init__(self, dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        log_s, t = self.net(x).chunk(2, dim=1)
        return torch.tanh(log_s), t


class AffineCoupling(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.st = STNet(mask.numel())

    def forward(self, x):
        passive = x * self.mask
        log_s, t = self.st(passive)
        active = 1.0 - self.mask
        log_s = log_s * active
        t = t * active
        z = passive + active * (x * torch.exp(log_s) + t)
        return z, log_s.sum(dim=1)

    def inverse(self, z):
        passive = z * self.mask
        log_s, t = self.st(passive)
        active = 1.0 - self.mask
        log_s = log_s * active
        t = t * active
        x = passive + active * ((z - t) * torch.exp(-log_s))
        return x


def checkerboard(dim, parity):
    m = torch.arange(dim) % 2
    if parity == 0:
        m = 1 - m
    return m.float()


class RealNVP(nn.Module):
    def __init__(self, dim=IMG_DIM, n_coupling=N_COUPLING, logit_alpha=LOGIT_ALPHA):
        super().__init__()
        self.dim = dim
        self.logit_alpha = logit_alpha
        self.layers = nn.ModuleList(
            [AffineCoupling(checkerboard(dim, i % 2)) for i in range(n_coupling)])

    def logit_transform(self, x):
        a = self.logit_alpha
        u = a + (1 - 2 * a) * x
        y = torch.log(u) - torch.log1p(-u)
        log_det = (math.log(1 - 2 * a) - torch.log(u) - torch.log1p(-u)).sum(dim=1)
        return y, log_det

    def inv_logit_transform(self, y):
        a = self.logit_alpha
        u = torch.sigmoid(y)
        x = (u - a) / (1 - 2 * a)
        return x

    def forward(self, x_img):
        B = x_img.size(0)
        x = x_img.view(B, -1)
        z, ld = self.logit_transform(x)
        log_det = ld
        for layer in self.layers:
            z, d = layer(z)
            log_det = log_det + d
        return z, log_det

    def log_prob(self, x_img):
        z, log_det = self.forward(x_img)
        log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.dim * math.log(2 * math.pi)
        return log_pz + log_det

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        x = self.inv_logit_transform(z)
        return x


def dequantise(x):
    return (x * 255.0 + torch.rand_like(x)) / 256.0


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_victim(Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, seed=SEED)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_flow(Xtr, out):
    C.set_seed(SEED + 2)
    flow = RealNVP().to(C.DEVICE)
    opt = torch.optim.Adam(flow.parameters(), lr=FLOW_LR)
    n = Xtr.size(0)
    for ep in range(FLOW_EPOCHS):
        flow.train()
        perm = torch.randperm(n, device=Xtr.device)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = dequantise(Xtr[idx])
            opt.zero_grad()
            loss = -flow.log_prob(xb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 50.0)
            opt.step()
            tot += loss.item() * idx.numel()
        out(f"    flow epoch {ep+1}/{FLOW_EPOCHS}  -logp={tot/n:.2f} "
            f"({(tot/n)/IMG_DIM:.4f} nats/dim)")
    flow.eval()
    return flow


@torch.no_grad()
def flow_logp(flow, x, batch=256, n_draws=4):
    accum = torch.zeros(x.size(0), device=x.device)
    for _ in range(n_draws):
        for i in range(0, x.size(0), batch):
            accum[i:i + batch] += flow.log_prob(dequantise(x[i:i + batch]))
    return accum / n_draws


def margin_entropy(model, x, batch=512):
    margins, ents = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            logits = model(x[i:i + batch])
            srt, _ = logits.sort(1, descending=True)
            margins.append((srt[:, 0] - srt[:, 1]).cpu())
            p = F.softmax(logits, 1)
            ents.append((-(p * (p + 1e-12).log()).sum(1)).cpu())
    return torch.cat(margins).numpy(), torch.cat(ents).numpy()


def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h409_coupling_flow_density_detector_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H409  Coupling-flow exact-likelihood adversarial detector (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: SMOKE={SMOKE} N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} "
        f"FLOW_EPOCHS={FLOW_EPOCHS} N_COUPLING={N_COUPLING} SEED={SEED} EPS={EPS} "
        f"PGD_STEPS={PGD_STEPS} device={C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    out("\n[1] training victim SmallCNN...")
    victim = train_victim(Xtr, Ytr)
    _, acc = C.logits_and_acc(victim, Xte, Yte)
    out(f"    victim clean_acc={acc:.4f}")

    out("\n[2] training affine-coupling flow on clean images (NO labels)...")
    flow = train_flow(Xtr, out)

    # ---- exact-invertibility check ----
    xb = dequantise(Xte[:64])
    with torch.no_grad():
        z, _ = flow(xb)
        x_rec = flow.inverse(z)
    rec_err = float((xb.view(64, -1) - x_rec).abs().max())
    out(f"\n[invertibility] flow x->z->x_rec max|err| = {rec_err:.3e}")
    assert rec_err < 1e-3, f"flow not invertible (err={rec_err})"

    # ---- correctly-classified clean subset & PGD copies ----
    out("\n[3] building clean / PGD-adversarial test pairs...")
    with torch.no_grad():
        correct = victim(Xte).argmax(1) == Yte
    Xc, Yc = Xte[correct], Yte[correct]
    out(f"    correctly-classified clean test samples: {Xc.size(0)}")
    Xadv = C.pgd(victim, Xc, Yc, eps=EPS, steps=PGD_STEPS)
    with torch.no_grad():
        flipped = victim(Xadv).argmax(1) != Yc
    out(f"    PGD flip rate on these = {flipped.float().mean().item():.4f}")

    # ---- exact log p(x) ----
    out("\n[4] computing exact flow log p(x) for clean vs adversarial...")
    logp_clean = flow_logp(flow, Xc).cpu().numpy()
    logp_adv = flow_logp(flow, Xadv).cpu().numpy()
    out(f"    log p(clean): mean={logp_clean.mean():.1f} std={logp_clean.std():.1f}")
    out(f"    log p(adv)  : mean={logp_adv.mean():.1f} std={logp_adv.std():.1f}")
    out(f"    mean shift (clean-adv) = {logp_clean.mean()-logp_adv.mean():+.1f} "
        f"(positive => adv lower density, as hypothesised)")

    # ---- detection AUROC ----
    labels = np.concatenate([np.zeros(len(logp_clean)), np.ones(len(logp_adv))])
    flow_score = np.concatenate([-logp_clean, -logp_adv])   # adv hypothesised high
    auc_flow = C.safe_auroc(labels, flow_score)

    m_c, e_c = margin_entropy(victim, Xc)
    m_a, e_a = margin_entropy(victim, Xadv)
    margin_score = np.concatenate([-m_c, -m_a])             # adv lower margin
    ent_score = np.concatenate([e_c, e_a])                  # adv higher entropy
    auc_margin = C.safe_auroc(labels, margin_score)
    auc_ent = C.safe_auroc(labels, ent_score)

    # combined flow+margin logistic detector
    auc_combo = float("nan")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        feats = np.stack([flow_score, margin_score, ent_score], 1)
        Xs = StandardScaler().fit_transform(feats)
        lr = LogisticRegression(max_iter=2000).fit(Xs, labels)
        auc_combo = C.safe_auroc(labels, lr.predict_proba(Xs)[:, 1])
    except Exception as e:
        out(f"    (combo detector skipped: {e})")

    # ---- table ----
    out("\n" + "=" * 80)
    out("[5] DETECTION AUROC (clean vs PGD-adversarial)")
    out("=" * 80)
    out("{:<34} {:>10}".format("detector", "AUROC"))
    out("-" * 46)
    out("{:<34} {:>10.4f}".format("flow exact -log p(x)", auc_flow))
    out("{:<34} {:>10.4f}".format("victim margin (baseline)", auc_margin))
    out("{:<34} {:>10.4f}".format("victim softmax entropy (base)", auc_ent))
    out("{:<34} {:>10.4f}".format("flow+margin+entropy (logreg)", auc_combo))
    out("-" * 46)

    # ---- verdict ----
    out("\n" + "=" * 80)
    out("[6] VERDICT")
    out("=" * 80)
    best_base = max(auc_margin, auc_ent)
    adds = (auc_combo - best_base) if auc_combo == auc_combo else float("nan")
    out(f"  flow exact-likelihood AUROC = {auc_flow:.4f}")
    out(f"  best classifier-only baseline AUROC = {best_base:.4f}")
    out(f"  combo over best baseline = {adds:+.4f}")
    out(f"  flow invertibility recon error = {rec_err:.2e} (EXACT bijection)")
    if auc_flow > 0.7 and auc_flow > best_base + 0.02:
        verdict = ("YES: exact flow density separates clean from PGD-adversarial "
                   "inputs and beats the classifier-confidence baseline.")
    elif auc_flow > 0.7:
        verdict = ("PARTIAL: flow density detects adversarial inputs but does not "
                   "beat the cheaper classifier-confidence baseline.")
    elif auc_flow > 0.55:
        verdict = ("WEAK: flow density gives only modest separation -- PGD inputs "
                   "are not strongly off-manifold under the flow.")
    else:
        verdict = ("NO: exact flow log-likelihood does NOT separate clean from "
                   "PGD-adversarial inputs (consistent with flows assigning high "
                   "density to off-distribution inputs).")
    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
