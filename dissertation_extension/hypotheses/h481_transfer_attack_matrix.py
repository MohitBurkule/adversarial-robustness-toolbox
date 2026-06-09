"""
H481 - Transfer-attack matrix as a gradient-masking detector.

Seed (Athalye-Carlini-Wagner 2018 "Obfuscated Gradients", section 5; Carlini et
al. 2019 "On Evaluating Adversarial Robustness", M3):
    A defense that suppresses *white-box* PGD ASR via gradient masking will fail
    to suppress black-box / transfer attacks crafted on a substitute model.
    Concretely: transfer-ASR > white-box-ASR is a smoking-gun signature of
    masking. Genuine robustness lowers BOTH.

Critique seed (Papernot et al. 2017 "Practical Black-box Attacks Against
Machine Learning"): Papernot's substitute-model attack used Jacobian-based
dataset augmentation to train the substitute. We deliberately simplify here
and use a plain STD model on the same clean training data as the substitute -
this is the weakest possible substitute, so any transferability we *do* see is
a lower bound on what a Jacobian-augmented or query-based substitute would
achieve. We then strengthen the attack itself with MI-FGSM (Dong et al. 2018
"Boosting Adversarial Attacks with Momentum") which is the canonical
transfer-friendly variant. Tramer et al. 2018 ("Ensemble Adversarial
Training") provides the matrix design template - source != target attack
crafting.

Extra-paper coverage (>=2):
  * Dong et al. CVPR 2018 - MI-FGSM (momentum iterative FGSM): include as a
    transfer-friendly attack and compare against vanilla PGD.
  * Xie et al. CVPR 2019 - DI-FGSM (input diversity / random resize-pad):
    cited; we use the random-resize-pad transform on the source-model input
    inside MI-FGSM to get a DI+MI hybrid step.
  * Tramer et al. ICLR 2018 - Ensemble Adversarial Training: motivates the
    NxN source-vs-target matrix as the diagnostic object.
  * Papernot et al. AsiaCCS 2017 - Practical Black-box Attacks: substitute
    training template.
  * Athalye-Carlini-Wagner ICML 2018 - Obfuscated Gradients: the rule
    "transfer-ASR > white-box-ASR => masking".

Design:
  N=6 source/target models trained on Fashion-MNIST SmallCNN (n=6000):
    M0  STD          - vanilla cross-entropy (acts as substitute too).
    M1  PGD-AT       - Madry adversarial training (genuine robustness control).
    M2  TRADES       - KL-coupled AT (genuine robustness control).
    M3  DEF-DISTILL  - high-temperature defensive distillation (Papernot 2016)
                       - classic masking suspect.
    M4  RFNN         - frozen random conv features + linear head (robust-feature
                       control: limited capacity, no end-to-end GD pathway).
    M5  RAND-NOISE   - STD model wrapped with input gaussian-noise smoothing
                       (Cohen et al. 2019-ish; stochastic gradients = masking-
                       suspect under fixed-seed PGD).

Controls / outputs:
  (1) Full NxN ASR matrix for vanilla PGD AND for MI-FGSM.
  (2) Per-target "masking score" = max_src (transfer-ASR) / white-box-ASR.
      Score > 1 (with a small additive floor to dodge div-by-zero) flags
      the target as gradient-masking.
  (3) Per-class transfer-ASR breakdown for each target's worst class.
  (4) MI-FGSM vs vanilla PGD: mean off-diagonal lift in transferability.
  (5) HEADLINE verdict per defense.

Verification of disk state: no checkpoints are saved under
`results/fashion_mnist/*.pt` (verified pre-write); training is inline and fast
at n_train=6000.

Output: results/fashion_mnist/h481_transfer_attack_matrix_output.txt
DO NOT execute from this writer pass; the harness will run it later.
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


# -------- experiment config -------------------------------------------------
DS         = "fashion_mnist"
N_TRAIN    = 6000
N_EVAL     = 2000
SEED       = 0
EPS        = 0.1
PGD_STEPS  = 20
PGD_ALPHA  = 2.5 * EPS / PGD_STEPS
EPOCHS_STD = 8
EPOCHS_AT  = 10        # AT defenses train a touch longer
DD_TEMP    = 20.0      # defensive-distillation temperature
NOISE_SIGMA = 0.25     # input gaussian-noise sigma for RAND-NOISE wrapper
DI_PROB    = 0.5       # DI-FGSM probability of applying input diversity
MI_MU      = 1.0       # MI-FGSM momentum decay

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h481_transfer_attack_matrix_output.txt",
)

META = {"channels": 1, "size": 28, "n_classes": 10}
NCLS = META["n_classes"]
MODEL_NAMES = ["STD", "PGD-AT", "TRADES", "DEF-DISTILL", "RFNN", "RAND-NOISE"]
N_MODELS = len(MODEL_NAMES)


# -------- training routines (inline; no checkpoints expected) ---------------
def _opt(model, lr=0.05):
    return torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9, weight_decay=5e-4,
    )


def train_std(Xtr, Ytr, seed=SEED, epochs=EPOCHS_STD):
    C.set_seed(seed)
    m = C.build_model("cnn", META, width=32, seed=seed)
    opt = _opt(m, lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            F.cross_entropy(m(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        sched.step()
    m.eval()
    return m


def train_pgd_at(Xtr, Ytr, seed=SEED, epochs=EPOCHS_AT):
    C.set_seed(seed)
    m = C.build_model("cnn", META, width=32, seed=seed)
    opt = _opt(m, lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = Xtr[idx], Ytr[idx]
            m.eval()
            xa = C.pgd(m, xb, yb, eps=EPS, steps=10, alpha=PGD_ALPHA)
            m.train()
            opt.zero_grad()
            F.cross_entropy(m(xa), yb).backward()
            opt.step()
        sched.step()
    m.eval()
    return m


def _pgd_kl(model, x, eps, steps, alpha):
    """TRADES inner attack: PGD maximising KL(f(x) || f(x+delta))."""
    x0 = x.clone().detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1).detach()
    with torch.no_grad():
        p_clean = F.softmax(model(x0), dim=1)
    for _ in range(steps):
        xa.requires_grad_(True)
        log_p_adv = F.log_softmax(model(xa), dim=1)
        kl = F.kl_div(log_p_adv, p_clean, reduction="batchmean")
        g, = torch.autograd.grad(kl, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def train_trades(Xtr, Ytr, seed=SEED, epochs=EPOCHS_AT, beta=6.0):
    C.set_seed(seed)
    m = C.build_model("cnn", META, width=32, seed=seed)
    opt = _opt(m, lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = Xtr[idx], Ytr[idx]
            m.eval()
            xa = _pgd_kl(m, xb, EPS, 10, PGD_ALPHA)
            m.train()
            opt.zero_grad()
            out_c = m(xb)
            log_p_clean = F.log_softmax(out_c, dim=1)
            log_p_adv = F.log_softmax(m(xa), dim=1)
            p_clean = log_p_clean.exp().detach()
            kl_per = (p_clean * (log_p_clean.detach() - log_p_adv)).sum(1).mean()
            loss = F.cross_entropy(out_c, yb) + beta * kl_per
            loss.backward()
            opt.step()
        sched.step()
    m.eval()
    return m


def train_def_distill(Xtr, Ytr, seed=SEED, epochs=EPOCHS_STD, T=DD_TEMP):
    """Defensive distillation (Papernot et al. 2016 IEEE S&P).

    Step 1: train a teacher at temperature T with hard labels.
    Step 2: train a student at temperature T using the teacher's softened
            probabilities as soft targets.
    At eval the student is evaluated at T=1 (logits / 1 = logits) - this is
    where the gradient-vanishing trick kicks in for white-box gradient attacks
    that backprop through the temperature-1 softmax.
    """
    # ---- teacher (at temperature T)
    C.set_seed(seed)
    teacher = C.build_model("cnn", META, width=32, seed=seed)
    opt_t = _opt(teacher, lr=0.05)
    sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=epochs)
    n = Xtr.size(0)
    for _ in range(epochs):
        teacher.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt_t.zero_grad()
            logits = teacher(Xtr[idx]) / T
            F.cross_entropy(logits, Ytr[idx]).backward()
            opt_t.step()
        sched_t.step()
    teacher.eval()
    # soft targets from teacher (at temperature T)
    with torch.no_grad():
        soft = []
        for i in range(0, n, 512):
            soft.append(F.softmax(teacher(Xtr[i:i + 512]) / T, dim=1))
        soft = torch.cat(soft, dim=0).detach()

    # ---- student (trained on soft targets at temperature T)
    C.set_seed(seed + 1)
    student = C.build_model("cnn", META, width=32, seed=seed + 1)
    opt_s = _opt(student, lr=0.05)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=epochs)
    for _ in range(epochs):
        student.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt_s.zero_grad()
            log_p = F.log_softmax(student(Xtr[idx]) / T, dim=1)
            loss = -(soft[idx] * log_p).sum(1).mean()
            loss.backward()
            opt_s.step()
        sched_s.step()
    student.eval()
    # student is used at T=1 at eval (i.e. plain forward). This is the
    # configuration in which defensive distillation is famous for masking.
    return student


def train_rfnn(Xtr, Ytr, seed=SEED, epochs=EPOCHS_STD):
    """Random-feature CNN: frozen random conv stack, trained linear head.
    Robust-feature control - no end-to-end GD path to the input."""
    C.set_seed(seed)
    m = C.build_model("rfnn", META, width=64, seed=seed)
    # only the linear head is trainable
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                          lr=0.05, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            F.cross_entropy(m(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        sched.step()
    m.eval()
    return m


class NoisyWrap(nn.Module):
    """STD model wrapped with input gaussian noise.

    On every forward we inject N(0, sigma) noise then clamp to [0,1]. This is
    a deliberately stochastic defense (cf. Cohen et al. 2019 "Randomized
    Smoothing"); white-box single-shot gradient attacks against it become
    noisy and tend to under-estimate true robustness - a classic case the
    transfer attack diagnostic is designed to flag.
    """
    def __init__(self, inner, sigma=NOISE_SIGMA):
        super().__init__()
        self.inner = inner
        self.sigma = sigma

    def forward(self, x):
        noise = torch.randn_like(x) * self.sigma
        return self.inner((x + noise).clamp(0, 1))


def train_rand_noise(Xtr, Ytr, seed=SEED, epochs=EPOCHS_STD, sigma=NOISE_SIGMA):
    """Train STD then wrap with input-noise injection at forward time."""
    inner = train_std(Xtr, Ytr, seed=seed, epochs=epochs)
    return NoisyWrap(inner, sigma=sigma).to(C.DEVICE).eval()


# -------- attacks: vanilla PGD, MI-FGSM (+ DI input-diversity step) ---------
def _input_diversity(x, p=DI_PROB, max_pad=2):
    """DI-FGSM (Xie 2019): with probability p, randomly resize then pad x
    back to original size. For 28x28 we use a small max_pad to stay close to
    the L-inf budget intuition (the resize-pad is geometric, not L-inf, so
    this is only used DURING source-side attack crafting)."""
    if torch.rand(1).item() > p:
        return x
    sz = x.size(-1)
    # random rescale by 1..max_pad fewer pixels, then pad back
    rsz = sz - int(torch.randint(0, max_pad + 1, (1,)).item())
    if rsz < sz:
        x_r = F.interpolate(x, size=(rsz, rsz), mode="bilinear", align_corners=False)
        pad_l = int(torch.randint(0, sz - rsz + 1, (1,)).item())
        pad_t = int(torch.randint(0, sz - rsz + 1, (1,)).item())
        x_r = F.pad(x_r, [pad_l, sz - rsz - pad_l, pad_t, sz - rsz - pad_t], value=0.0)
        return x_r
    return x


def mi_fgsm(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA, mu=MI_MU,
            use_di=True):
    """MI-FGSM (Dong et al. 2018) optionally with DI input diversity step.

    Crafted in the source model's domain only - we never apply DI when
    *evaluating* on the target.
    """
    x0 = x.clone().detach()
    xa = x0.clone() + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1).detach()
    g_prev = torch.zeros_like(x0)
    for _ in range(steps):
        xa.requires_grad_(True)
        x_in = _input_diversity(xa) if use_di else xa
        loss = F.cross_entropy(model(x_in), y)
        g, = torch.autograd.grad(loss, xa)
        # L1-normalise grad then add momentum (Dong et al. eq. 6).
        g_norm = g / (g.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        g_prev = mu * g_prev + g_norm
        xa = xa.detach() + alpha * g_prev.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


# -------- eval helpers ------------------------------------------------------
@torch.no_grad()
def clean_pred(model, X, batch=512):
    out = []
    for i in range(0, X.size(0), batch):
        out.append(model(X[i:i + batch]).argmax(1).cpu())
    return torch.cat(out)


def craft_adv(src, X, Y, attack, eps=EPS, steps=PGD_STEPS, batch=256):
    """Craft adversarials on `src` against labels Y."""
    xs = []
    src.eval()
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        if attack == "pgd":
            xa = C.pgd(src, xb, yb, eps=eps, steps=steps, alpha=PGD_ALPHA)
        elif attack == "mifgsm":
            xa = mi_fgsm(src, xb, yb, eps=eps, steps=steps, alpha=PGD_ALPHA,
                         mu=MI_MU, use_di=True)
        else:
            raise ValueError(attack)
        xs.append(xa.detach())
    return torch.cat(xs, dim=0)


@torch.no_grad()
def asr_on_target(tgt, X_adv, Y_ref, correct_mask, batch=512):
    """ASR = fraction of originally-correct (per Y_ref) samples whose target
    prediction now differs from Y_ref. Y_ref is the ground-truth label."""
    pred = []
    for i in range(0, X_adv.size(0), batch):
        pred.append(tgt(X_adv[i:i + batch]).argmax(1).cpu())
    pred = torch.cat(pred)
    flipped = (pred != Y_ref.cpu()).numpy()
    cm = correct_mask.cpu().numpy().astype(bool)
    if cm.sum() == 0:
        return float("nan"), flipped, cm
    return float(flipped[cm].mean()), flipped, cm


@torch.no_grad()
def per_class_asr(pred_np, Y_ref_np, correct_mask, ncls=NCLS):
    """Per-true-class ASR among originally-correct samples."""
    out = np.full(ncls, np.nan)
    for c in range(ncls):
        m = correct_mask & (Y_ref_np == c)
        if m.sum() > 0:
            out[c] = float((pred_np[m] != c).mean())
    return out


def fmt_row(name, vals, fmt="{:.3f}"):
    parts = [name.ljust(12)] + [fmt.format(v) if v == v else "  nan".rjust(6)
                                for v in vals]
    return "  ".join(parts)


# -------- main --------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    lines = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    log("=" * 78)
    log("H481 - Transfer-attack matrix as a gradient-masking detector")
    log("=" * 78)
    log(f"Device={C.DEVICE}  dataset={DS}  n_train={N_TRAIN}  n_eval={N_EVAL}")
    log(f"eps={EPS}  pgd_steps={PGD_STEPS}  mi_mu={MI_MU}  di_prob={DI_PROB}  "
        f"dd_T={DD_TEMP}  noise_sigma={NOISE_SIGMA}")
    log(f"Models (N={N_MODELS}): {MODEL_NAMES}")

    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # ---- train N=6 models inline ------------------------------------------
    trainers = [
        ("STD",          lambda: train_std(Xtr, Ytr, seed=SEED)),
        ("PGD-AT",       lambda: train_pgd_at(Xtr, Ytr, seed=SEED)),
        ("TRADES",       lambda: train_trades(Xtr, Ytr, seed=SEED)),
        ("DEF-DISTILL",  lambda: train_def_distill(Xtr, Ytr, seed=SEED)),
        ("RFNN",         lambda: train_rfnn(Xtr, Ytr, seed=SEED)),
        ("RAND-NOISE",   lambda: train_rand_noise(Xtr, Ytr, seed=SEED)),
    ]
    models = []
    log("\n-- Training --")
    for name, fn in trainers:
        t0 = time.time()
        m = fn()
        models.append(m)
        log(f"  {name:<12} trained in {time.time() - t0:5.1f}s")

    # ---- clean accuracy ----------------------------------------------------
    log("\n-- Clean accuracy --")
    clean_acc = []
    for name, m in zip(MODEL_NAMES, models):
        p = clean_pred(m, Xte)
        acc = float((p == Yte.cpu()).float().mean())
        clean_acc.append(acc)
        log(f"  {name:<12}  clean_acc = {acc:.3f}")

    # samples each target classifies correctly (per-target ASR denominator)
    correct_masks = [(clean_pred(m, Xte) == Yte.cpu()) for m in models]

    # ---- build the N x N matrices for both attacks ------------------------
    Yte_cpu = Yte.cpu()
    Yte_np = Yte_cpu.numpy()

    results = {}   # results[attack] = dict with 'matrix', 'preds'
    for attack in ("pgd", "mifgsm"):
        log(f"\n-- Crafting adversarials with {attack.upper()} on each source --")
        # craft once per source; reuse against all targets
        adv_per_src = []
        for s, name in enumerate(MODEL_NAMES):
            t0 = time.time()
            xa = craft_adv(models[s], Xte, Yte, attack=attack)
            adv_per_src.append(xa)
            log(f"  src={name:<12} crafted in {time.time() - t0:5.1f}s")

        matrix = np.full((N_MODELS, N_MODELS), np.nan)
        preds  = [[None] * N_MODELS for _ in range(N_MODELS)]   # per-target prediction arrays
        for s in range(N_MODELS):
            for t in range(N_MODELS):
                asr, fl, cm = asr_on_target(models[t], adv_per_src[s], Yte_cpu,
                                            correct_masks[t])
                matrix[s, t] = asr
                # store per-sample predictions for per-class breakdown
                preds[s][t] = (fl, cm)
        results[attack] = {"matrix": matrix, "preds": preds, "adv": adv_per_src}

    # ---- print NxN matrices ----------------------------------------------
    for attack in ("pgd", "mifgsm"):
        mat = results[attack]["matrix"]
        log("\n" + "=" * 78)
        log(f"ASR matrix ({attack.upper()}) - rows=SOURCE, cols=TARGET")
        log("=" * 78)
        header = "src \\ tgt   " + "  ".join(f"{n:>6}" for n in MODEL_NAMES)
        log(header)
        for s in range(N_MODELS):
            log(fmt_row(MODEL_NAMES[s], mat[s].tolist(), fmt="{:>6.3f}"))

    # ---- masking score per target ----------------------------------------
    # Score = max_{src != tgt} transfer-ASR / max(white-box ASR, floor)
    FLOOR = 0.02
    log("\n" + "=" * 78)
    log("Transferability / masking score per TARGET")
    log("  score = max_src!=tgt (transfer-ASR) / max(white-box-ASR, floor)")
    log(f"  floor = {FLOOR}   (avoid div-by-zero when white-box=0; if white-box")
    log("  is at the floor or below, ANY non-trivial transfer ASR flags masking)")
    log("=" * 78)
    log(f"{'TARGET':<12}  {'WB-PGD':>7}  {'tr-PGD':>7}  {'score-PGD':>10}  "
        f"{'WB-MI':>7}  {'tr-MI':>7}  {'score-MI':>10}  FLAG")
    flags = {}
    for t in range(N_MODELS):
        row = []
        for attack in ("pgd", "mifgsm"):
            mat = results[attack]["matrix"]
            wb = mat[t, t]
            transfers = [mat[s, t] for s in range(N_MODELS) if s != t]
            tr = max(transfers) if transfers else float("nan")
            denom = max(wb, FLOOR)
            score = tr / denom if denom > 0 else float("nan")
            row.append((wb, tr, score))
        # masking flag = either attack gives score > 1 AND transfer > floor
        flag = ((row[0][2] > 1.0 and row[0][1] > FLOOR) or
                (row[1][2] > 1.0 and row[1][1] > FLOOR))
        flags[MODEL_NAMES[t]] = (flag, row)
        log(f"{MODEL_NAMES[t]:<12}  "
            f"{row[0][0]:7.3f}  {row[0][1]:7.3f}  {row[0][2]:10.2f}  "
            f"{row[1][0]:7.3f}  {row[1][1]:7.3f}  {row[1][2]:10.2f}  "
            f"{'MASKING' if flag else 'ok':>7}")

    # ---- per-class transfer-ASR for worst class (under MI-FGSM) ----------
    log("\n" + "=" * 78)
    log("Worst-class per-target transfer-ASR (MI-FGSM, max source != target)")
    log("=" * 78)
    mi_preds = results["mifgsm"]["preds"]
    for t in range(N_MODELS):
        # take the source != target that gave the highest transfer-ASR
        best_s, best_asr = None, -1.0
        for s in range(N_MODELS):
            if s == t:
                continue
            a = results["mifgsm"]["matrix"][s, t]
            if a == a and a > best_asr:
                best_asr, best_s = a, s
        if best_s is None:
            log(f"  {MODEL_NAMES[t]:<12} no transfer source")
            continue
        fl, cm = mi_preds[best_s][t]
        pc = per_class_asr(fl, Yte_np, cm)
        worst_c = int(np.nanargmax(pc))
        log(f"  tgt={MODEL_NAMES[t]:<12} best_src={MODEL_NAMES[best_s]:<12}  "
            f"worst_class={worst_c} (tr-ASR={pc[worst_c]:.3f})   "
            f"per-class=[" + ", ".join(f"{v:.2f}" for v in pc) + "]")

    # ---- MI-FGSM vs PGD lift (off-diagonal mean) -------------------------
    pgd_mat = results["pgd"]["matrix"]
    mi_mat  = results["mifgsm"]["matrix"]
    mask_off = ~np.eye(N_MODELS, dtype=bool)
    pgd_off  = np.nanmean(pgd_mat[mask_off])
    mi_off   = np.nanmean(mi_mat[mask_off])
    pgd_diag = np.nanmean(np.diag(pgd_mat))
    mi_diag  = np.nanmean(np.diag(mi_mat))
    log("\n" + "=" * 78)
    log("MI-FGSM vs vanilla PGD transferability comparison")
    log("=" * 78)
    log(f"  white-box (diagonal) mean ASR    PGD={pgd_diag:.3f}   MI-FGSM={mi_diag:.3f}")
    log(f"  transfer (off-diag)  mean ASR    PGD={pgd_off:.3f}    MI-FGSM={mi_off:.3f}")
    log(f"  MI-FGSM transfer LIFT over PGD: {mi_off - pgd_off:+.3f}")
    log("  (positive lift confirms Dong et al. 2018: momentum boosts transferability.)")

    # ---- HEADLINE verdict ------------------------------------------------
    log("\n" + "=" * 78)
    log("HEADLINE")
    log("=" * 78)
    masked = [n for n, (f, _) in flags.items() if f]
    clean_ = [n for n, (f, _) in flags.items() if not f]
    log(f"  Flagged as GRADIENT MASKING (transfer > white-box): {masked}")
    log(f"  Pass the transfer test (no masking signal):         {clean_}")
    log("")
    log("  Expected pattern (Athalye-Carlini-Wagner 2018, Carlini 2019 M3):")
    log("    * STD: white-box ASR is already high; no masking signal expected.")
    log("    * PGD-AT / TRADES: genuinely robust; both white-box and transfer ASR")
    log("      should be moderate, with transfer-ASR <= white-box-ASR.")
    log("    * DEF-DISTILL: classic masking suspect; white-box ASR collapses due")
    log("      to vanishing temperature-1 gradients, but transfer ASR from the")
    log("      STD substitute should land far above it -> FLAGGED.")
    log("    * RFNN: limited capacity, no end-to-end GD; white-box and transfer")
    log("      should track each other (no masking signal).")
    log("    * RAND-NOISE: stochastic forward; single-shot white-box PGD is noisy")
    log("      and tends to under-estimate vulnerability -> transfer is likely to")
    log("      flag this as masking, validating the diagnostic on a known case.")
    log("=" * 78)
    log("References:")
    log("  Athalye, Carlini, Wagner (ICML 2018) - Obfuscated Gradients, sec 5.")
    log("  Carlini et al. (2019) - On Evaluating Adversarial Robustness, M3.")
    log("  Dong et al. (CVPR 2018)  - MI-FGSM (momentum iterative).")
    log("  Xie et al.  (CVPR 2019)  - DI-FGSM (input diversity).")
    log("  Tramer et al. (ICLR 2018) - Ensemble Adversarial Training.")
    log("  Papernot et al. (AsiaCCS 2017) - Practical Black-box Attacks.")
    log("  Papernot et al. (IEEE S&P 2016) - Defensive Distillation.")
    log("  Cohen et al. (ICML 2019) - Randomized Smoothing.")
    log("=" * 78)

    with open(OUT_FILE, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {OUT_FILE}")


if __name__ == "__main__":
    main()
