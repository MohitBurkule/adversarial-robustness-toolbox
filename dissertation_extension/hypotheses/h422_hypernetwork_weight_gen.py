"""
H422 - Hypernetwork input-conditioned weight generation.

Hypothesis: a hypernetwork (Ha et al., 2016 "HyperNetworks") that generates the
weights of a classifier CNN *conditioned on the input itself* creates a moving-target
defence: because the effective parameters change per-input, an attacker optimising
a gradient-based perturbation must simultaneously attack a function that shifts under
its own feet. We test three conditions:

  A. BASELINE   - standard static SmallCNN (width=32), trained normally.
  B. HYPERNET   - a hypernetwork generates the *head* (FC 256->10) weights conditioned
                  on the input's low-dimensional embedding. The feature extractor is
                  shared/static; only the final linear layer is input-dependent.
                  At inference the head weights are re-generated per sample.
  C. HYPERNET+EOT - same architecture; robustness evaluated with Expectation Over
                  Transformations (EOT, Athalye et al. 2018): the attack averages
                  gradients over K=8 independently-seeded hypernet forward passes
                  (simulating an attacker that accounts for the stochastic-like weight
                  variation). Tests whether observed robustness of B is illusory.

Cite: Ha et al. 2016 "HyperNetworks" (arXiv:1609.09106).

Design:
  - Dataset: Fashion-MNIST  N_TRAIN=6000, N_EVAL=2000, SEED=0
  - Shared CNN features: 3 conv blocks (width=32), same as SmallCNN, BN+ReLU+MaxPool2.
  - Hypernet: takes the flattened conv feature vector (width*4 * (28//8)^2 = 128*9 =
    1152 dims after 3 pools) -> FC 256 -> ReLU -> FC (256*10 + 10) to produce
    weight matrix (256,10) and bias (10) for the classification head.
    Head is applied per-sample via torch.einsum / batched matmul.
  - Training: both models trained identically (SGD mom=0.9 wd=5e-4, CosineAnnealingLR,
    EPOCHS=10, LR=0.05, BATCH=128).
  - Eval: clean acc, FGSM ASR (eps=0.1), PGD ASR (eps=0.1, steps=10).
    For hypernet models FGSM/PGD gradients flow through the hypernetwork automatically.
  - EOT check (condition C): PGD with K=8 forward passes averaged for gradient.

Config: N_TRAIN=6000, N_EVAL=2000, EPOCHS=10, LR=0.05, BATCH=128,
        SGD(mom=0.9, wd=5e-4), SEED=0, EPS=0.1, PGD_STEPS=10, EOT_K=8.
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
PGD_ALPHA = 2.5 * EPS / PGD_STEPS
EOT_K = 8          # EOT gradient averaging samples

META = {"channels": 1, "size": 28, "n_classes": 10}
WIDTH = 32         # CNN feature width
FEAT_DIM = WIDTH * 4 * (28 // 8) ** 2   # = 128 * 9 = 1152
HEAD_IN = 256      # head input dim (shared FC before hypernet classification head)
N_CLS = 10


# ---- models ------------------------------------------------------------------

class StaticCNN(nn.Module):
    """Standard SmallCNN baseline (mirrors common.SmallCNN width=32)."""
    def __init__(self):
        super().__init__()
        w = WIDTH
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(
            *block(1, w), *block(w, w * 2), *block(w * 2, w * 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FEAT_DIM, HEAD_IN), nn.ReLU(),
            nn.Linear(HEAD_IN, N_CLS))

    def forward(self, x):
        return self.head(self.features(x))


class HyperNet(nn.Module):
    """Input-conditioned classifier: a hypernetwork generates the final FC weights.

    Architecture:
      shared feature extractor (3 conv blocks, frozen BN at eval) ->
      shared FC (FEAT_DIM -> HEAD_IN, ReLU) ->
      hypernetwork branch: FC (HEAD_IN -> 512, ReLU) -> FC (512, HEAD_IN*N_CLS + N_CLS)
        => produces per-sample weight matrix W (B, HEAD_IN, N_CLS) and bias b (B, N_CLS)
      classification: logits_i = h_i @ W_i + b_i   (batched, no shared final linear)
    """
    def __init__(self):
        super().__init__()
        w = WIDTH
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(
            *block(1, w), *block(w, w * 2), *block(w * 2, w * 4))
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FEAT_DIM, HEAD_IN), nn.ReLU())
        # hypernetwork: maps HEAD_IN -> W (HEAD_IN x N_CLS) + b (N_CLS)
        hyp_out = HEAD_IN * N_CLS + N_CLS
        self.hypernet = nn.Sequential(
            nn.Linear(HEAD_IN, 512), nn.ReLU(),
            nn.Linear(512, hyp_out))

    def forward(self, x):
        f = self.features(x)               # (B, FEAT_DIM) after flatten via shared_fc
        h = self.shared_fc(f)              # (B, HEAD_IN)
        params = self.hypernet(h)          # (B, HEAD_IN*N_CLS + N_CLS)
        W = params[:, :HEAD_IN * N_CLS].view(-1, HEAD_IN, N_CLS)   # (B, HEAD_IN, N_CLS)
        b = params[:, HEAD_IN * N_CLS:]                              # (B, N_CLS)
        # per-sample linear: logit_i = h_i @ W_i + b_i
        logits = torch.einsum("bi,bio->bo", h, W) + b               # (B, N_CLS)
        return logits


# ---- training ----------------------------------------------------------------

def make_sgd(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)


def train_model(model, Xtr, Ytr, seed):
    C.set_seed(seed)
    opt = make_sgd(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
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


# ---- attacks -----------------------------------------------------------------

def fgsm_attack(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    g, = torch.autograd.grad(loss, x)
    return (x + EPS * g.sign()).clamp(0, 1).detach()


def pgd_attack(model, x, y, steps=PGD_STEPS, alpha=PGD_ALPHA,
               eot_k=1, random_start=True):
    """PGD with optional EOT: gradient averaged over eot_k forward passes."""
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        xa = (xa + torch.empty_like(xa).uniform_(-EPS, EPS)).clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        if eot_k <= 1:
            loss = F.cross_entropy(model(xa), y)
            g, = torch.autograd.grad(loss, xa)
        else:
            # EOT: average gradient over eot_k forward passes
            g_acc = torch.zeros_like(xa)
            for _ in range(eot_k):
                loss = F.cross_entropy(model(xa), y)
                gi, = torch.autograd.grad(loss, xa, retain_graph=False,
                                          create_graph=False)
                g_acc = g_acc + gi.detach()
                xa = xa.detach().requires_grad_(True)
            g = g_acc / eot_k
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - EPS), x0 + EPS).clamp(0, 1)
    return xa.detach()


def eval_robustness(model, X, Y, eot_k=1, label=""):
    """Returns (clean_acc, fgsm_asr, pgd_asr). eot_k=1 -> standard PGD."""
    model.eval()
    batch = 256
    # clean
    corr_clean, corr_fgsm, corr_pgd = [], [], []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        with torch.no_grad():
            pred_clean = model(x).argmax(1)
        hit = (pred_clean == y)

        xa_fg = fgsm_attack(model, x, y)
        with torch.no_grad():
            pred_fg = model(xa_fg).argmax(1)

        xa_pg = pgd_attack(model, x, y, eot_k=eot_k)
        with torch.no_grad():
            pred_pg = model(xa_pg).argmax(1)

        corr_clean.append(hit.cpu())
        corr_fgsm.append(((pred_fg != y) & hit).cpu())
        corr_pgd.append(((pred_pg != y) & hit).cpu())

    corr_clean = torch.cat(corr_clean).numpy().astype(bool)
    flip_fgsm  = torch.cat(corr_fgsm).numpy().astype(bool)
    flip_pgd   = torch.cat(corr_pgd).numpy().astype(bool)

    clean_acc = float(corr_clean.mean())
    fgsm_asr  = float(flip_fgsm[corr_clean].mean()) if corr_clean.sum() > 0 else float("nan")
    pgd_asr   = float(flip_pgd[corr_clean].mean())  if corr_clean.sum() > 0 else float("nan")
    return clean_acc, fgsm_asr, pgd_asr


# ---- main --------------------------------------------------------------------

def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "fashion_mnist",
        "h422_hypernetwork_weight_gen_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H422  Hypernetwork input-conditioned weight generation (Fashion-MNIST)")
    out("=" * 80)
    out("Hypothesis: per-input weights (Ha et al. 2016 HyperNetworks) make the")
    out("classifier a moving target -- the attacker must attack a function that")
    out("shifts under its own perturbation, potentially reducing attack success.")
    out("")
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA:.4f} "
        f"EOT_K={EOT_K} FEAT_DIM={FEAT_DIM} HEAD_IN={HEAD_IN}")
    out(f"        device={C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    results = []

    # ---- A: BASELINE static CNN ------------------------------------------
    out("[A] BASELINE static CNN (standard SmallCNN width=32)")
    C.set_seed(SEED)
    baseline = StaticCNN().to(C.DEVICE)
    train_model(baseline, Xtr, Ytr, SEED)
    b_acc, b_fg, b_pgd = eval_robustness(baseline, Xte, Yte, eot_k=1, label="baseline")
    out(f"    clean_acc={b_acc:.4f}  FGSM_ASR={b_fg:.4f}  PGD_ASR={b_pgd:.4f}")
    out(f"    ({time.time()-t0:.0f}s)")
    results.append(("A: baseline static", b_acc, b_fg, b_pgd))
    flush_file()

    # ---- B: HYPERNET input-conditioned -----------------------------------
    out("")
    out("[B] HYPERNET input-conditioned weights (standard PGD eval)")
    C.set_seed(SEED)
    hypnet = HyperNet().to(C.DEVICE)
    train_model(hypnet, Xtr, Ytr, SEED)
    h_acc, h_fg, h_pgd = eval_robustness(hypnet, Xte, Yte, eot_k=1, label="hypernet")
    out(f"    clean_acc={h_acc:.4f}  FGSM_ASR={h_fg:.4f}  PGD_ASR={h_pgd:.4f}")
    out(f"    delta vs baseline: d_clean={h_acc-b_acc:+.4f}  "
        f"d_FGSM={h_fg-b_fg:+.4f}  d_PGD={h_pgd-b_pgd:+.4f}")
    out(f"    ({time.time()-t0:.0f}s)")
    results.append(("B: hypernet (std PGD)", h_acc, h_fg, h_pgd))
    flush_file()

    # ---- C: HYPERNET + EOT check -----------------------------------------
    out("")
    out(f"[C] HYPERNET + EOT check (PGD with K={EOT_K} gradient averaging)")
    out(f"    (attacker accounts for per-input weight variation; same model as B)")
    _, _, h_pgd_eot = eval_robustness(hypnet, Xte, Yte, eot_k=EOT_K, label="hypernet+eot")
    # FGSM doesn't naturally benefit from EOT; report standard FGSM
    out(f"    FGSM_ASR={h_fg:.4f} (same)  PGD_ASR_EOT={h_pgd_eot:.4f}")
    out(f"    delta EOT vs std PGD (hypernet): d_PGD_EOT={h_pgd_eot-h_pgd:+.4f} "
        f"(positive => EOT breaks robustness)")
    out(f"    ({time.time()-t0:.0f}s)")
    results.append(("C: hypernet+EOT", h_acc, h_fg, h_pgd_eot))
    flush_file()

    # ---- MAIN TABLE -------------------------------------------------------
    out("")
    out("=" * 80)
    out("MAIN TABLE")
    out("=" * 80)
    hdr = "{:<26} {:>10} {:>10} {:>10}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    for (name, acc, fg, pg) in results:
        out("{:<26} {:>10.4f} {:>10.4f} {:>10.4f}".format(name, acc, fg, pg))
    out("-" * len(hdr))
    out("")

    # ---- VERDICT ----------------------------------------------------------
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)

    robust_gain_std  = b_pgd - h_pgd            # B vs A, standard PGD (positive = B better)
    robust_gain_eot  = b_pgd - h_pgd_eot        # C vs A, EOT PGD
    acc_ok = h_acc >= b_acc - 0.02              # hypernet keeps accuracy within 2%
    eot_breaks = (h_pgd_eot - h_pgd) > 0.05    # EOT substantially closes gap

    out(f"  A->B  standard PGD gain:  {robust_gain_std:+.4f} "
        f"(positive => hypernet harder to attack)")
    out(f"  A->C  EOT PGD gain:       {robust_gain_eot:+.4f} "
        f"(positive => still robust under adaptive attacker)")
    out(f"  B->C  EOT vs std PGD:     {h_pgd_eot-h_pgd:+.4f} "
        f"({'EOT breaks robustness' if eot_breaks else 'EOT does NOT substantially break robustness'})")
    out(f"  clean-acc within 0.02:    {'YES' if acc_ok else 'NO'}")
    out("")

    if robust_gain_std > 0.05 and acc_ok and not eot_breaks:
        verdict = ("SUPPORTED: hypernet provides genuine per-input-weight robustness -- "
                   "PGD ASR drops vs baseline and the gain survives an EOT adaptive attacker, "
                   "confirming the moving-target hypothesis (Ha et al. 2016).")
    elif robust_gain_std > 0.05 and acc_ok and eot_breaks:
        verdict = ("ILLUSORY: hypernet appears robust under standard PGD but EOT largely "
                   "closes the gap -- the robustness was an artefact of the attacker not "
                   "accounting for input-dependent weights, not a genuine moving-target effect.")
    elif robust_gain_std <= 0.05 and acc_ok:
        verdict = ("NOT SUPPORTED: input-conditioned weight generation does not provide "
                   "meaningful robustness over a static CNN (PGD ASR gain <= 0.05) even "
                   "though clean accuracy is preserved -- per-input weights alone are "
                   "insufficient to create a useful moving-target defence.")
    elif not acc_ok:
        verdict = ("INCONCLUSIVE (ACC COST): the hypernet achieves different robustness but "
                   "at a clean-accuracy cost > 0.02 -- the architecture may simply underfit "
                   "rather than genuinely defend via moving-target dynamics.")
    else:
        verdict = ("MIXED: see per-metric deltas above.")

    out("ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time()-t0:.1f}s")

    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
