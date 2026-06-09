"""H471: L0 attack JSMA / Pixle on Fashion-MNIST.

Gap filled (CAMPAIGN_GAP_MAP G1):
  The entire H173-H413 campaign never evaluates an L0 / sparse-pixel
  threat model on any defended model. The campaign collapses to Linf
  eps=0.1 PGD-10 (M3). H471 closes that hole by running two
  L0 attacks at three pixel budgets against four canonical defences.

Threat model:
  L0 = change at most k pixels by ANY magnitude (clamped to [0,1]).
  Distinct from Linf (small change, every pixel) and L2 (small total
  energy). L0 robustness is a separate axis: a Linf-AT model can be
  brittle to one well-placed pixel flip and vice versa (cf. Modas 2019,
  Su 2019 one-pixel attack).

Attacks implemented (pure torch, no ART):

  1. JSMA (Papernot et al. 2016, "The Limitations of Deep Learning in
     Adversarial Settings", EuroS&P). Jacobian Saliency-Map Attack.
     White-box, gradient-based. Targeted variant used here (Papernot's
     canonical formulation). For each pixel i and target class t,
     saliency S(i) = dF_t/dx_i * |sum_{j!=t} dF_j/dx_i|, kept only when
     dF_t/dx_i > 0 and sum_{j!=t} dF_j/dx_i < 0. We pick the single best
     pixel each step (instead of Papernot's pair, which is for
     bidirectional flips on RGB; greyscale Fashion-MNIST + clipping
     means single-pixel works identically). Pixel is set to 1.0 (the
     extreme). Budget = max changed pixels k. Target = next-most-likely
     class on the clean image (least-likely target is harder to drive
     in 10-way classification at k <= 200; next-most is the standard
     "most threatening" target).

  2. Pixle (Pomponi, Scardapane, Uncini 2022, IJCNN). Black-box,
     gradient-free, random search. Each iteration: pick a random source
     pixel inside the image and a random destination pixel; copy
     source value into destination (overwrite). If the model's loss on
     the true label INCREASED (i.e., we made the model more wrong),
     keep the swap; else revert. This is the simplest "1-pixel single
     swap" variant of the Pixle family (the paper also defines patch
     variants). Budget = max changed destination pixels k.
     We track the set of modified destinations; once |set| == k we
     stop. We allow up to 10*k trial swaps to fill the budget so
     reverted no-op trials do not exhaust the run.

Models compared (all standard 6k / 10 epoch / SmallCNN / SGD 0.05):
  - CE         standard cross-entropy baseline
  - Linf-AT    PGD-AT at eps=0.1, 7 steps (campaign's strongest lever)
  - TRADES     beta=6 (H304 canonical setting)
  - Jacobian-F K=5 random-projection Jacobian Frobenius (H323 winner)

Pixel budgets k in {10, 50, 200} (out of 784 = 1.3%, 6.4%, 25.5%).

Hypothesis (a priori): Linf-AT was trained with EVERY pixel allowed to
move by 0.1; it has no incentive to be robust to a SINGLE pixel moving
by 1.0. We expect L0-ASR roughly comparable across the four models, or
even WORSE for AT/TRADES at small k because their decision boundaries
are pushed in Linf-direction. Jacobian-Frobenius (which smooths the
input-output map isotropically) should generalise better to L0.

VERDICT logic (printed at end):
  - SUPPORTED if AT/TRADES are NOT meaningfully more L0-robust than CE
    (ASR delta < 0.05 in JSMA@k=50, the mid-budget).
  - NOT SUPPORTED if AT/TRADES clearly beat CE on L0 too.
  - Either way, JSMA is white-box (gradient-based, like PGD) while
    Pixle is black-box random-search; a defence that helps on JSMA
    but not Pixle is suspected of gradient masking (Athalye 2018) in
    the L0 regime. We log both.

Compute: 4 models trained sequentially; at k=200 JSMA does 200
gradient-of-target-logit calls per sample on 1000 eval images, ~ a few
minutes per cell on RTX 4090. Total under ~30 minutes.

References:
  - Papernot et al. 2016, EuroS&P, "The Limitations of Deep Learning
    in Adversarial Settings" (JSMA).
  - Pomponi, Scardapane, Uncini 2022, IJCNN, "Pixle: a fast and
    effective black-box attack based on rearranging pixels"
    (arXiv:2202.02236).
  - Modas, Moosavi-Dezfooli, Frossard 2019, CVPR, "SparseFool: a few
    pixels make a big difference" - additional sparse-attack prior
    art (we do not implement SparseFool here as JSMA + Pixle already
    cover white-box and black-box L0 corners; SparseFool is a third
    point on the geometric-linearisation axis).
  - Su, Vargas, Sakurai 2019, "One Pixel Attack for Fooling Deep
    Neural Networks", TEVC - the extreme k=1 case.
  - Croce & Hein 2019, "Sparse and imperceptible adversarial attacks"
    (PGD_0) - more recent sparse-Linf hybrid.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0

EPS_LINF = 0.1
PGD_STEPS_AT = 7
PGD_ALPHA_AT = 2.5 * EPS_LINF / PGD_STEPS_AT
EPS_PGD_EVAL = 0.1
PGD_STEPS_EVAL = 10
PGD_ALPHA_EVAL = 0.01

TRADES_BETA = 6.0
JACOBIAN_LAM = 0.001
JACOBIAN_K = 5

# L0 attack budgets
PIXEL_BUDGETS = [10, 50, 200]

# Eval set size (kept modest because JSMA is O(k * forward+backward) per sample)
N_EVAL_L0 = 500

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h471_jsma_pixle_l0_attack_output.txt"
)


# ---------------------------------------------------------------------------
# Training routines for the four defences
# ---------------------------------------------------------------------------
def train_ce(model, Xtr, Ytr):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()


def train_at(model, Xtr, Ytr):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.pgd(model, xb, yb, eps=EPS_LINF, steps=PGD_STEPS_AT,
                       alpha=PGD_ALPHA_AT)
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(xa), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()


def _pgd_on_kl(model, x, steps, eps, alpha):
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)
    x_adv = (x.detach() + 0.001 * torch.randn_like(x)).clamp(0, 1)
    x0 = x.detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        kl = F.kl_div(F.log_softmax(model(x_adv), dim=1), p_clean,
                      reduction='batchmean')
        g, = torch.autograd.grad(kl, x_adv)
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1)
    return x_adv.detach()


def train_trades(model, Xtr, Ytr):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            x_adv = _pgd_on_kl(model, xb, steps=PGD_STEPS_AT,
                               eps=EPS_LINF, alpha=PGD_ALPHA_AT)
            model.train()
            opt.zero_grad()
            out_c = model(xb)
            out_a = model(x_adv)
            loss_ce = F.cross_entropy(out_c, yb)
            loss_kl = F.kl_div(F.log_softmax(out_a, dim=1),
                               F.softmax(out_c, dim=1),
                               reduction='batchmean')
            loss = loss_ce + TRADES_BETA * loss_kl
            loss.backward()
            opt.step()
        sched.step()
    model.eval()


def _jacobian_frob_penalty(model, xb, K):
    penalties = []
    for _ in range(K):
        v = torch.randn(xb.size(0), 10, device=xb.device)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        xg = xb.clone().detach().requires_grad_(True)
        out = model(xg)
        proj = (out * v).sum()
        g, = torch.autograd.grad(proj, xg, create_graph=True)
        penalties.append((g ** 2).sum(dim=(1, 2, 3)))
    return torch.stack(penalties, dim=0).mean(0).mean()


def train_jacobian(model, Xtr, Ytr):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb)
            jp = _jacobian_frob_penalty(model, xb, K=JACOBIAN_K)
            loss = ce + JACOBIAN_LAM * jp
            loss.backward()
            opt.step()
        sched.step()
    model.eval()


# ---------------------------------------------------------------------------
# JSMA (Jacobian Saliency Map Attack, Papernot 2016)
# Single-pixel variant (greyscale + clipping; equivalent to original on
# Fashion-MNIST). Targeted at next-most-likely class.
# ---------------------------------------------------------------------------
def _jacobian_per_class(model, x):
    """Compute dF_c/dx_i for every class c, per single sample x[1,1,28,28].
    Returns J of shape [n_classes, H*W]."""
    ncls = 10
    H = x.size(-2)
    W = x.size(-1)
    J = torch.zeros(ncls, H * W, device=x.device)
    for c in range(ncls):
        xg = x.clone().detach().requires_grad_(True)
        out = model(xg)
        # scalar per-class logit
        s = out[0, c]
        g, = torch.autograd.grad(s, xg, retain_graph=False)
        J[c] = g.view(-1)
    return J


def jsma_attack(model, x, y, k_budget):
    """Targeted JSMA on a single sample.

    x: tensor [1,1,28,28] in [0,1].
    y: int (true label).
    k_budget: max pixels to flip.

    Strategy:
      - target t = second-highest clean-logit class (most-threatening).
      - at each step, build saliency S(i) over pixels not yet modified
        AND not already saturated at 1.0, set pixel i with highest S to
        1.0 (increase target prob, decrease others).
      - if model already misclassifies, stop early.
    """
    model.eval()
    x = x.clone().detach()
    H, W = x.size(-2), x.size(-1)
    flat_n = H * W
    with torch.no_grad():
        logits = model(x)
    if logits.argmax(1).item() != y:
        # already wrong; no work needed (skip from attack stats)
        return x, 0, True  # adv, pixels_used, was_already_wrong

    # target = second-most-likely class on clean
    sl = logits[0].clone()
    sl[y] = -1e9
    target = int(sl.argmax().item())

    modified = torch.zeros(flat_n, dtype=torch.bool, device=x.device)
    xa = x.clone()
    for step in range(k_budget):
        # check current prediction
        with torch.no_grad():
            pred = model(xa).argmax(1).item()
        if pred != y:
            return xa, int(modified.sum().item()), False

        J = _jacobian_per_class(model, xa)  # [10, 784]
        dF_t = J[target]
        # sum over non-target classes
        mask = torch.ones(10, dtype=torch.bool, device=x.device)
        mask[target] = False
        dF_other = J[mask].sum(0)  # [784]

        # saliency (positive form): increase pixel if dF_t > 0 AND dF_other < 0
        sal = -dF_t * dF_other  # high when dF_t>0 and dF_other<0
        # zero out invalid:
        invalid = (dF_t <= 0) | (dF_other >= 0)
        sal = sal.masked_fill(invalid, 0.0)
        # zero out already-modified or saturated pixels
        flat_x = xa.view(-1)
        sat_high = flat_x >= 1.0 - 1e-6
        sal = sal.masked_fill(modified | sat_high, 0.0)

        if (sal > 0).sum() == 0:
            # No saliency direction available; bail.
            return xa, int(modified.sum().item()), False

        i_pix = int(sal.argmax().item())
        flat_x[i_pix] = 1.0
        xa = flat_x.view_as(xa)
        modified[i_pix] = True

    with torch.no_grad():
        pred = model(xa).argmax(1).item()
    return xa, int(modified.sum().item()), False


def jsma_eval(model, X, Y, k_budget):
    """Returns ASR on originally-correct samples, plus mean pixels used."""
    model.eval()
    flips = 0
    correct = 0
    px_used = []
    for i in range(X.size(0)):
        x = X[i:i + 1]
        y = int(Y[i].item())
        with torch.no_grad():
            pred = model(x).argmax(1).item()
        if pred != y:
            continue  # only attack originally-correct
        correct += 1
        xa, used, _ = jsma_attack(model, x, y, k_budget)
        with torch.no_grad():
            pa = model(xa).argmax(1).item()
        if pa != y:
            flips += 1
            px_used.append(used)
    asr = flips / max(1, correct)
    mean_px = float(np.mean(px_used)) if px_used else 0.0
    return asr, mean_px, correct


# ---------------------------------------------------------------------------
# Pixle (Pomponi 2022): random source/destination swap, accept if loss up
# ---------------------------------------------------------------------------
def pixle_attack(model, x, y, k_budget, max_trials_mult=10, seed=0):
    """Single-sample Pixle, simplest single-pixel-swap variant.

    Keeps a set of modified destination pixels; budget = k unique destinations.
    Loss = cross-entropy on the true label; accept swap if loss increases.
    """
    model.eval()
    g = torch.Generator(device=x.device).manual_seed(seed)
    H, W = x.size(-2), x.size(-1)
    flat_n = H * W
    xa = x.clone()
    with torch.no_grad():
        loss_cur = F.cross_entropy(model(xa), torch.tensor([y], device=x.device)).item()
        if model(xa).argmax(1).item() != y:
            return xa, 0, True
    modified = set()
    trials = 0
    max_trials = max_trials_mult * k_budget
    while len(modified) < k_budget and trials < max_trials:
        trials += 1
        src = int(torch.randint(0, flat_n, (1,), generator=g, device=x.device).item())
        dst = int(torch.randint(0, flat_n, (1,), generator=g, device=x.device).item())
        if src == dst:
            continue
        flat = xa.view(-1).clone()
        old = float(flat[dst].item())
        new = float(flat[src].item())
        if abs(old - new) < 1e-6:
            continue  # no-op swap
        flat[dst] = new
        xb = flat.view_as(xa)
        with torch.no_grad():
            loss_new = F.cross_entropy(model(xb), torch.tensor([y], device=x.device)).item()
        if loss_new > loss_cur:
            xa = xb
            loss_cur = loss_new
            modified.add(dst)
            with torch.no_grad():
                if model(xa).argmax(1).item() != y:
                    return xa, len(modified), False
    return xa, len(modified), False


def pixle_eval(model, X, Y, k_budget, seed=0):
    model.eval()
    flips = 0
    correct = 0
    px_used = []
    for i in range(X.size(0)):
        x = X[i:i + 1]
        y = int(Y[i].item())
        with torch.no_grad():
            pred = model(x).argmax(1).item()
        if pred != y:
            continue
        correct += 1
        xa, used, _ = pixle_attack(model, x, y, k_budget, seed=seed * 10000 + i)
        with torch.no_grad():
            pa = model(xa).argmax(1).item()
        if pa != y:
            flips += 1
            px_used.append(used)
    asr = flips / max(1, correct)
    mean_px = float(np.mean(px_used)) if px_used else 0.0
    return asr, mean_px, correct


# ---------------------------------------------------------------------------
# Standard eval (clean / FGSM / Linf-PGD) for reference
# ---------------------------------------------------------------------------
def standard_eval(model, Xte, Yte):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean = C.logits_and_acc(model, Xte, Yte)
    Xf = C.fgsm(model, Xte, Yte, eps=EPS_PGD_EVAL)
    _, accf = C.logits_and_acc(model, Xf, Yte)
    Xp = C.pgd(model, Xte, Yte, eps=EPS_PGD_EVAL, steps=PGD_STEPS_EVAL,
               alpha=PGD_ALPHA_EVAL)
    _, accp = C.logits_and_acc(model, Xp, Yte)
    return dict(clean=float(clean), fgsm_asr=1 - float(accf),
                pgd_asr=1 - float(accp))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
DEFENCES = [
    ("CE",         train_ce),
    ("Linf-AT",    train_at),
    ("TRADES_b6",  train_trades),
    ("Jacobian-F", train_jacobian),
]


def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=2000, seed=SEED)
    meta = C.dataset_meta(DS)

    # L0 eval subset (fixed across defences for paired comparison)
    idx_l0 = torch.randperm(Xte.size(0))[:N_EVAL_L0]
    XL0, YL0 = Xte[idx_l0], Yte[idx_l0]

    all_results = []
    for name, train_fn in DEFENCES:
        print(f"\n=== Training defence: {name} ===")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        model.to(C.DEVICE)
        t0 = time.time()
        train_fn(model, Xtr, Ytr)
        t_train = time.time() - t0

        std = standard_eval(model, Xte, Yte)
        print(f"  trained in {t_train:.1f}s | clean={std['clean']:.4f} "
              f"fgsm_asr={std['fgsm_asr']:.4f} pgd_asr={std['pgd_asr']:.4f}")

        per_k = {}
        for k in PIXEL_BUDGETS:
            tj = time.time()
            jsma_asr, jsma_px, jsma_corr = jsma_eval(model, XL0, YL0, k_budget=k)
            tj = time.time() - tj
            tp = time.time()
            pix_asr, pix_px, pix_corr = pixle_eval(model, XL0, YL0,
                                                    k_budget=k, seed=SEED)
            tp = time.time() - tp
            print(f"  k={k:>3}: JSMA ASR={jsma_asr:.4f} "
                  f"(mean_px={jsma_px:.1f}, n_corr={jsma_corr}, t={tj:.1f}s) | "
                  f"Pixle ASR={pix_asr:.4f} (mean_px={pix_px:.1f}, "
                  f"n_corr={pix_corr}, t={tp:.1f}s)")
            per_k[k] = dict(jsma_asr=jsma_asr, jsma_px=jsma_px,
                            jsma_corr=jsma_corr, jsma_t=tj,
                            pix_asr=pix_asr, pix_px=pix_px,
                            pix_corr=pix_corr, pix_t=tp)

        all_results.append(dict(name=name, std=std, per_k=per_k,
                                t_train=t_train))

    # ---- report ----
    lines = []
    lines.append("H471 - L0 Attacks (JSMA / Pixle) on 4 Defences")
    lines.append("=" * 78)
    lines.append(f"Dataset: {DS}  N_train={N_TRAIN}  Epochs={EPOCHS}  "
                 f"Seed={SEED}  N_eval_L0={N_EVAL_L0}")
    lines.append(f"Linf-AT/TRADES: eps={EPS_LINF}, {PGD_STEPS_AT}-step PGD inner. "
                 f"TRADES beta={TRADES_BETA}. Jacobian-F lam={JACOBIAN_LAM} K={JACOBIAN_K}.")
    lines.append(f"Pixel budgets k = {PIXEL_BUDGETS} (out of 784, "
                 f"i.e. {[round(100*k/784,2) for k in PIXEL_BUDGETS]}%).")
    lines.append("")

    lines.append("--- Standard reference eval (Linf eps=0.1) ---")
    hdr = f"{'Defence':<12}  {'clean':>7}  {'fgsm_asr':>9}  {'pgd_asr':>8}  {'train_s':>8}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in all_results:
        s = r['std']
        lines.append(f"{r['name']:<12}  {s['clean']:>7.4f}  "
                     f"{s['fgsm_asr']:>9.4f}  {s['pgd_asr']:>8.4f}  "
                     f"{r['t_train']:>8.1f}")
    lines.append("")

    for atk_key, atk_label, asr_k, px_k in [
        ("jsma_asr", "JSMA (white-box, gradient saliency, targeted=2nd-class)",
         "jsma_asr", "jsma_px"),
        ("pix_asr",  "Pixle (black-box, random pixel swap, accept-if-loss-up)",
         "pix_asr",  "pix_px"),
    ]:
        lines.append(f"--- {atk_label} ---")
        hdr2 = f"{'Defence':<12}  " + "  ".join(
            [f"k={k}_asr  k={k}_mean_px" for k in PIXEL_BUDGETS])
        lines.append(hdr2)
        lines.append("-" * len(hdr2))
        for r in all_results:
            row = [f"{r['name']:<12}"]
            for k in PIXEL_BUDGETS:
                d = r['per_k'][k]
                row.append(f"{d[asr_k]:>7.4f}  {d[px_k]:>10.1f}")
            lines.append("  ".join(row))
        lines.append("")

    # ---- Verdict logic ----
    lines.append("--- Analysis ---")
    base = next(r for r in all_results if r['name'] == 'CE')
    at = next(r for r in all_results if r['name'] == 'Linf-AT')
    tr = next(r for r in all_results if r['name'] == 'TRADES_b6')
    jf = next(r for r in all_results if r['name'] == 'Jacobian-F')

    jsma50_ce = base['per_k'][50]['jsma_asr']
    jsma50_at = at['per_k'][50]['jsma_asr']
    jsma50_tr = tr['per_k'][50]['jsma_asr']
    jsma50_jf = jf['per_k'][50]['jsma_asr']
    pix50_ce  = base['per_k'][50]['pix_asr']
    pix50_at  = at['per_k'][50]['pix_asr']

    lines.append(f"JSMA@k=50  CE={jsma50_ce:.3f}  AT={jsma50_at:.3f}  "
                 f"TRADES={jsma50_tr:.3f}  Jac-F={jsma50_jf:.3f}")
    lines.append(f"Pixle@k=50 CE={pix50_ce:.3f}  AT={pix50_at:.3f}")

    delta_at_jsma = jsma50_ce - jsma50_at
    delta_tr_jsma = jsma50_ce - jsma50_tr
    delta_jf_jsma = jsma50_ce - jsma50_jf

    lines.append(f"Delta vs CE (JSMA@k=50): AT={delta_at_jsma:+.3f}  "
                 f"TRADES={delta_tr_jsma:+.3f}  Jac-F={delta_jf_jsma:+.3f}")

    # Masking probe: a defence that helps JSMA much more than Pixle is
    # gradient-masking-suspect, since Pixle is gradient-free.
    if delta_at_jsma > 0.10:
        delta_at_pix = pix50_ce - pix50_at
        masking_ratio = delta_at_jsma / max(1e-6, delta_at_pix)
        lines.append(f"Linf-AT JSMA-vs-Pixle improvement ratio: "
                     f"{masking_ratio:.2f}  "
                     f"(>>1 suggests gradient-masking artefact)")

    verdict = "INCONCLUSIVE"
    if abs(delta_at_jsma) < 0.05 and abs(delta_tr_jsma) < 0.05:
        verdict = ("SUPPORTED: Linf AT/TRADES does NOT meaningfully transfer "
                   "to L0 robustness (JSMA@k=50 within 0.05 of CE). The "
                   "Linf eps=0.1 threat model under-specifies sparse attacks.")
    elif delta_at_jsma >= 0.05 or delta_tr_jsma >= 0.05:
        verdict = ("PARTIAL: Linf AT/TRADES does provide some L0 transfer "
                   "(JSMA@k=50 ASR drops by >=0.05). Hypothesis - that "
                   "L0 is orthogonal to Linf - is NOT fully supported.")
    elif delta_at_jsma <= -0.05 or delta_tr_jsma <= -0.05:
        verdict = ("STRONG SUPPORT: AT/TRADES is actually WORSE than CE "
                   "under L0 (negative delta). Defences trained on Linf "
                   "pay an L0 cost.")

    lines.append("")
    lines.append(f"VERDICT: {verdict}")
    lines.append("")
    lines.append("Caveats:")
    lines.append("- Single seed (campaign default; cf. M1).")
    lines.append("- N_EVAL_L0=500 is modest; expect +/-0.02 noise.")
    lines.append("- JSMA targeted at 2nd-most-likely class; LL target may give")
    lines.append("  lower ASR but is the conventional 'hardest' setting.")
    lines.append("- Pixle is one-pixel-swap; the paper also defines patch-swap")
    lines.append("  variants that are strictly stronger at fixed k.")
    lines.append("- L0 budget k=200 is 25.5% of 784 pixels and almost always")
    lines.append("  achieves ~100% ASR on any model; reported mainly to show")
    lines.append("  the saturation point.")

    report = "\n".join(lines) + "\n"
    print("\n" + report)
    with open(OUT_FILE, "w") as f:
        f.write(report)
        f.flush()
        os.fsync(f.fileno())
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
