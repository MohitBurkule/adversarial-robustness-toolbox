"""
H466 - Adaptive attack on H372 per-layer Jacobian penalty model.

Hypothesis seed:
  "h466: Adaptive attack on H372 per-layer Jacobian - gap M4 - anchor
   tramer-2020-adaptive - design loss using per-layer norms"

Critique of H372:
  H372 trained a SmallCNNBlocks with a Hutchinson-estimated per-layer
  Jacobian Frobenius penalty applied at every block (block0/block1/block2).
  Reported PGD-10 ASR ~ 0.76 (vs ~0.92 baseline). H391's masking battery
  audited an *input-only* Jacobian-penalty model and labelled it GENUINE,
  but H372's per-layer variant was NOT explicitly adapt-attacked.

  Per Tramer et al. 2020 ("On Adaptive Attacks to Adversarial Example
  Defenses"), an adaptive attacker uses domain knowledge of the defence.
  A per-layer Jacobian penalty drives down ||J_l||_F at every block, which
  smooths the local loss surface and can MASK gradients in the directions
  the penalty actually shrinks. The natural adaptive losses are:

    L_adaptive_low  = CE - lambda * sum_l ||J_l(h)||_F
        (push attack into LOW-Jacobian directions where the penalty bites,
         the model still classifies confidently but its gradient is small
         and PGD-CE would have stalled.)

    L_adaptive_high = CE + lambda * sum_l ||J_l(h)||_F
        (push attack into HIGH-Jacobian directions left ungarded by the
         penalty - input slack the defence never constrained.)

  We test both, plus standard PGD-CE at 10 and 50 steps, plus a transfer
  attack from a standard baseline (M5 control). H372 is GENUINE iff the
  ADAPTIVE-WORST ASR plateaus near the strong standard PGD-50 ASR.

Extra references (background, confirmed via prior literature):
  - Croce & Hein 2020 (AutoAttack)      - parameter-free strong PGD ladder.
  - Carlini et al. 2019 ("On Evaluating Adversarial Robustness")
                                        - mandates adaptive + transfer +
                                          step-increase eval before claiming
                                          robustness.
  - Athalye, Carlini, Wagner 2018       - obfuscated-gradient signature
                                          (steep step-curve, transfer >
                                          white-box) we look for here.
  - Tramer et al. 2020                  - explicit anchor.

Protocol (Fashion-MNIST, SmallCNN/SmallCNNBlocks, pure-torch, no ART):
  N_TRAIN = 6000, EPOCHS = 10, LR = 0.05, BATCH = 128,
  SGD(mom=0.9, wd=5e-4), SEED = 0, EPS = 0.1.

  Trains three models:
    (M_base) standard CE-only SmallCNN  (transfer source).
    (M_j372) H372 per-layer Jacobian SmallCNNBlocks (weighted config).
    (the H372 'baseline' arm is the same arch with lambdas=0; trained for
     reference clean acc.)

  Attacks vs M_j372 (over originally-correct samples):
    (a) standard PGD-CE-10.
    (b) standard PGD-CE-50.
    (c1) adaptive PGD: CE - lambda * sum_l ||J_l||_F_hat   (low-Jacobian).
    (c2) adaptive PGD: CE + lambda * sum_l ||J_l||_F_hat   (high-Jacobian).
    (d) transfer: PGD-CE-50 advs crafted on M_base, evaluated on M_j372.

  Verdict (GENUINE / MASKING):
    GENUINE if    max(a, b, c1, c2, d)  -  b  <= 0.05
              AND  d  <=  b + 0.05
    MASKING if any adaptive variant lifts ASR by > 0.05 over standard
    PGD-50, or transfer ASR exceeds white-box PGD-50 by > 0.05.

ASCII output only.  Output flushed.  DO NOT execute.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS_STD = 10
PGD_STEPS_STRONG = 50
K_HUTCH_TRAIN = 5    # Hutchinson vectors during training (matches H372)
K_HUTCH_ATTACK = 2   # Hutchinson vectors during adaptive attack (cheaper)
LAM_J372 = dict(weighted=[1e-4, 5e-5, 2e-5])  # H372 'weighted' config
ADAPT_LAM = 1.0      # weight on the Jacobian-norm steer in the attack loss

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "fashion_mnist",
                   "h466_adaptive_attack_per_layer_jacobian_output.txt")

_LINES = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LINES.append(s)


# ---------------------------------------------------------------------------
# H372 architecture (re-declared locally so this file is self-contained).
# ---------------------------------------------------------------------------
class SmallCNNBlocks(nn.Module):
    def __init__(self, in_ch=1, size=28, n_classes=10, width=32):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                nn.ReLU(), nn.MaxPool2d(2))
        self.block0 = block(in_ch, width)
        self.block1 = block(width, width * 2)
        self.block2 = block(width * 2, width * 4)
        feat = size // 8
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 4 * feat * feat, 256),
            nn.ReLU(), nn.Linear(256, n_classes))

    def forward(self, x):
        h = self.block0(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.head(h)

    def blocks(self):
        return [self.block0, self.block1, self.block2]


def estimate_jac_frob_sq_per_block(model, x, k=K_HUTCH_ATTACK, create_graph=True):
    """Sum over blocks of Hutchinson estimate of ||J_l||_F^2.

    Re-runs the block forward on a leaf copy of its input to get a graph.
    Returns a SCALAR tensor on x.device, suitable for autograd wrt x.
    """
    h = x
    total = 0.0
    for blk in model.blocks():
        h_in = h.detach().requires_grad_(True) if not h.requires_grad else h
        h_out = blk(h_in)
        block_pen = 0.0
        for _ in range(k):
            v = torch.randn_like(h_out)
            JTv = autograd.grad((h_out * v).sum(), h_in,
                                create_graph=create_graph,
                                retain_graph=True)[0]
            block_pen = block_pen + (JTv ** 2).sum() / x.size(0)
        total = total + block_pen / k
        h = h_out  # propagate forward
    return total


# ---------------------------------------------------------------------------
# Training: H372 per-layer Jacobian model + a standard CE-only transfer source
# ---------------------------------------------------------------------------
def train_h372_jacobian(meta, Xtr, Ytr, lambdas):
    C.set_seed(SEED)
    model = SmallCNNBlocks(meta["channels"], meta["size"], meta["n_classes"]).to(C.DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    blocks = model.blocks()
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            # Forward block-by-block with detached, grad-tracked block inputs
            # so we can estimate ||J_l||_F^2 per block via Hutchinson (K=5).
            h = xb
            h_inputs = []
            for blk in blocks:
                h_det = h.detach().requires_grad_(True)
                h_inputs.append(h_det)
                h = blk(h_det)
            logits = model.head(h)
            ce = F.cross_entropy(logits, yb)
            jac_loss = 0.0
            for lam, blk, h_in in zip(lambdas, blocks, h_inputs):
                if lam <= 0:
                    continue
                h_out = blk(h_in)
                block_pen = 0.0
                for _ in range(K_HUTCH_TRAIN):
                    v = torch.randn_like(h_out)
                    JTv = autograd.grad((h_out * v).sum(), h_in,
                                        create_graph=True, retain_graph=True)[0]
                    block_pen = block_pen + (JTv ** 2).sum() / xb.size(0)
                jac_loss = jac_loss + lam * (block_pen / K_HUTCH_TRAIN)
            loss = ce + jac_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_baseline_cnn(meta, Xtr, Ytr):
    """Plain CE-only SmallCNN, used as transfer source (Tramer M5 control)."""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=0)
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                         opt="sgd", lr=LR)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def _project(xa, x0, eps):
    return torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)


def pgd_ce(model, x, y, eps, steps, random_start=True):
    """Standard PGD with cross-entropy loss (sign step)."""
    alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        xa = (xa + torch.empty_like(xa).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = autograd.grad(loss, xa)
        xa = _project(xa.detach() + alpha * g.sign(), x0, eps)
    return xa.detach()


def pgd_adaptive_jacobian(model, x, y, eps, steps, sign=-1.0,
                          lam=ADAPT_LAM, k=K_HUTCH_ATTACK,
                          random_start=True):
    """Adaptive PGD targeting H372's per-layer Jacobian penalty.

    Loss = CE + sign * lam * sum_l ||J_l||_F   (estimated via Hutchinson).

    sign = -1 (LOW-Jacobian): steer the attack TOWARD regions the penalty
                              has flattened (gradient masking would hide
                              there).
    sign = +1 (HIGH-Jacobian): steer the attack TOWARD high-Jacobian slack
                               the penalty failed to flatten.

    We use sum_l ||J_l||_F (not ^2) so the gradient does not blow up where
    the penalty has already pushed ||J_l|| toward zero.
    """
    alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        xa = (xa + torch.empty_like(xa).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        logits = model(xa)
        ce = F.cross_entropy(logits, y)
        jac_sq = estimate_jac_frob_sq_per_block(model, xa, k=k, create_graph=True)
        jac_norm = torch.sqrt(jac_sq + 1e-12)
        loss = ce + sign * lam * jac_norm
        g, = autograd.grad(loss, xa)
        xa = _project(xa.detach() + alpha * g.sign(), x0, eps)
    return xa.detach()


def batched(fn, X, Y, batch=128, **kw):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(fn(X[i:i + batch], Y[i:i + batch], **kw))
    return torch.cat(outs)


@torch.no_grad()
def correct_mask(model, X, Y, batch=256):
    parts = []
    for i in range(0, X.size(0), batch):
        parts.append((model(X[i:i + batch]).argmax(1) == Y[i:i + batch]).cpu())
    return torch.cat(parts)


def asr_from_advx(model, Xadv, Y, corr, batch=256):
    flips = []
    with torch.no_grad():
        for i in range(0, Xadv.size(0), batch):
            pred = model(Xadv[i:i + batch]).argmax(1)
            flips.append((pred != Y[i:i + batch]).cpu())
    flips = torch.cat(flips).numpy()
    corr_np = corr.numpy().astype(bool)
    return float(flips[corr_np].mean()) if corr_np.sum() > 0 else float("nan")


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log("=" * 78)
    log("H466  Adaptive attack on H372 per-layer Jacobian (Fashion-MNIST)")
    log(f"  N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} EPS={EPS} SEED={SEED}")
    log(f"  PGD steps std={PGD_STEPS_STD}  strong={PGD_STEPS_STRONG}  "
        f"adapt lam={ADAPT_LAM}  K_hutch_attack={K_HUTCH_ATTACK}  device={C.DEVICE}")
    log("=" * 78)

    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # ---- train H372 model (weighted config: lam = [1e-4, 5e-5, 2e-5]) ----
    log("\n[1/3] training H372 per-layer Jacobian (weighted) ...")
    ts = time.time()
    lambdas = LAM_J372["weighted"]
    m_j372 = train_h372_jacobian(meta, Xtr, Ytr, lambdas)
    _, acc_j372 = C.logits_and_acc(m_j372, Xte, Yte)
    log(f"  H372 weighted: clean_acc={acc_j372:.4f}  ({time.time()-ts:.1f}s)")

    # ---- train standard CE baseline (transfer source) ----
    log("\n[2/3] training CE-only baseline SmallCNN (transfer source) ...")
    ts = time.time()
    m_base = train_baseline_cnn(meta, Xtr, Ytr)
    _, acc_base = C.logits_and_acc(m_base, Xte, Yte)
    log(f"  baseline: clean_acc={acc_base:.4f}  ({time.time()-ts:.1f}s)")

    # ---- attack ladder vs H372 ----
    log("\n[3/3] attack ladder vs H372 (originally-correct samples only) ...")
    corr = correct_mask(m_j372, Xte, Yte)
    log(f"  originally-correct on H372: {int(corr.sum())} / {Xte.size(0)}")

    log("\n  (a) standard PGD-CE-10 ...")
    ts = time.time()
    adv_a = batched(lambda x, y: pgd_ce(m_j372, x, y, EPS, PGD_STEPS_STD), Xte, Yte)
    asr_a = asr_from_advx(m_j372, adv_a, Yte, corr)
    log(f"      ASR={asr_a:.3f}  ({time.time()-ts:.1f}s)")

    log("\n  (b) standard PGD-CE-50 ...")
    ts = time.time()
    adv_b = batched(lambda x, y: pgd_ce(m_j372, x, y, EPS, PGD_STEPS_STRONG), Xte, Yte)
    asr_b = asr_from_advx(m_j372, adv_b, Yte, corr)
    log(f"      ASR={asr_b:.3f}  ({time.time()-ts:.1f}s)")

    log("\n  (c1) adaptive PGD with -lam * sum_l ||J_l||_F (LOW-Jacobian steer) ...")
    ts = time.time()
    adv_c1 = batched(
        lambda x, y: pgd_adaptive_jacobian(
            m_j372, x, y, EPS, PGD_STEPS_STRONG, sign=-1.0, lam=ADAPT_LAM),
        Xte, Yte)
    asr_c1 = asr_from_advx(m_j372, adv_c1, Yte, corr)
    log(f"      ASR={asr_c1:.3f}  ({time.time()-ts:.1f}s)")

    log("\n  (c2) adaptive PGD with +lam * sum_l ||J_l||_F (HIGH-Jacobian steer) ...")
    ts = time.time()
    adv_c2 = batched(
        lambda x, y: pgd_adaptive_jacobian(
            m_j372, x, y, EPS, PGD_STEPS_STRONG, sign=+1.0, lam=ADAPT_LAM),
        Xte, Yte)
    asr_c2 = asr_from_advx(m_j372, adv_c2, Yte, corr)
    log(f"      ASR={asr_c2:.3f}  ({time.time()-ts:.1f}s)")

    log("\n  (d)  transfer from CE-only baseline (PGD-CE-50 on m_base, eval on m_j372) ...")
    ts = time.time()
    corr_base = correct_mask(m_base, Xte, Yte)
    adv_d = batched(lambda x, y: pgd_ce(m_base, x, y, EPS, PGD_STEPS_STRONG), Xte, Yte)
    # measured over samples originally-correct on the TARGET (j372)
    asr_d = asr_from_advx(m_j372, adv_d, Yte, corr)
    log(f"      ASR={asr_d:.3f}  (baseline corr={int(corr_base.sum())})  "
        f"({time.time()-ts:.1f}s)")

    # ---- summary table ----
    log("\n" + "-" * 78)
    hdr = f"{'attack':<46} {'steps':>5} {'ASR':>7}"
    log(hdr)
    log("-" * 78)
    rows = [
        ("(a)  standard PGD-CE-10",                        PGD_STEPS_STD,    asr_a),
        ("(b)  standard PGD-CE-50",                        PGD_STEPS_STRONG, asr_b),
        ("(c1) adaptive PGD  -lam * sum||J_l||_F  (low)",  PGD_STEPS_STRONG, asr_c1),
        ("(c2) adaptive PGD  +lam * sum||J_l||_F  (high)", PGD_STEPS_STRONG, asr_c2),
        ("(d)  transfer PGD-CE-50 from CE baseline",       PGD_STEPS_STRONG, asr_d),
    ]
    for name, st, val in rows:
        log(f"{name:<46} {st:>5d} {val:>7.3f}")
    log("-" * 78)

    # ---- verdict ----
    worst = max(asr_a, asr_b, asr_c1, asr_c2, asr_d)
    adapt_lift = max(asr_c1, asr_c2) - asr_b
    transfer_lift = asr_d - asr_b
    log(f"\nclean_acc(H372)           = {acc_j372:.4f}")
    log(f"clean_acc(baseline)       = {acc_base:.4f}")
    log(f"std PGD-50 ASR (anchor)   = {asr_b:.3f}")
    log(f"worst attack ASR          = {worst:.3f}")
    log(f"adaptive lift (vs std 50) = {adapt_lift:+.3f}")
    log(f"transfer lift (vs std 50) = {transfer_lift:+.3f}")

    masking = (adapt_lift > 0.05) or (transfer_lift > 0.05)
    genuine = not masking and (worst - asr_b <= 0.05)
    if genuine:
        verdict = ("GENUINE  -  per-layer Jacobian penalty survives adaptive "
                   "PGD and transfer; ASR plateaus near std PGD-50.")
    elif masking:
        verdict = ("MASKING  -  adaptive (Jacobian-aware) or transfer attack "
                   "exceeds std PGD-50 by > 0.05; H372's reported PGD-10 ASR "
                   "is an under-estimate.")
    else:
        verdict = ("INCONCLUSIVE  -  worst attack lifts ASR by > 0.05 over "
                   "std PGD-50 but neither adaptive nor transfer is the "
                   "dominant contributor; rerun with stronger restarts.")
    log("\nVERDICT: " + verdict)

    log(f"\ntotal runtime {time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(_LINES) + "\n")
        f.flush()
    log(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
