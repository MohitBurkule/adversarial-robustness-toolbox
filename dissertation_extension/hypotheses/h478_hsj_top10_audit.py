"""
H478 - HopSkipJump audit of the campaign's top-10 PGD defences.

Seed paper (sec.5): Chen, Jordan & Wainwright (2020) "HopSkipJumpAttack: A
Query-Efficient Decision-Based Attack" (S&P 2020). HSJ is a HARD-LABEL,
decision-based attack: it never reads gradients, only the model's argmax label.
That makes it the canonical probe for gradient-masking - if a defence's white-box
PGD ASR is far below its hard-label HSJ ASR, the defence is hiding rather than
removing adversarial examples.

Extra prior art we are aware of and cite:
  * Athalye, Carlini & Wagner (2018) "Obfuscated Gradients Give a False Sense of
    Security: Circumventing Defences to Adversarial Examples" (ICML).  This is
    the paper that motivates the whole audit: 7 of 9 ICLR'18 defences were
    broken by BPDA/EOT once obfuscation was bypassed.  We use HSJ rather than
    BPDA because BPDA requires defence-specific surrogate gradients while HSJ
    is defence-agnostic.
  * Tramer, Carlini, Brendel & Madry (2020) "On Adaptive Attacks to Adversarial
    Example Defences" (NeurIPS).  Argues that score-based + decision-based
    attacks must BOTH be reported alongside PGD; we follow their checklist by
    adding Square-Attack (Andriushchenko et al. 2020) as a score-based control.

Hypothesis:
  Applying HSJ to the 10 strongest PGD defences in the campaign will reveal that
  >= 3 of them have HSJ-ASR significantly higher than PGD-ASR (absolute gap >
  0.10), flagging gradient masking in line with Athalye et al.

Top-10 selection (lowest PGD-ASR at eps=0.1 on Fashion-MNIST in RESULTS_SUMMARY,
hard-coded here so the script is self-contained; TODO if SUMMARY moves):
  H371_pgd_at         (PGD-AT, RESULTS_SUMMARY PGD_ASR = 0.196)
  H371_fgsm_at        (FGSM-AT, PGD_ASR = 0.217)
  H351_awp            (AWP gamma=0.001 on FGSM-AT, PGD_ASR = 0.321)
  H363_revkl_trades   (Reverse-KL TRADES beta=1, PGD_ASR = 0.324)
  H348_fgsm_at_ctrl   (FGSM-AT control in SupCon ablation, PGD_ASR = 0.324)
  H347_jsd_trades     (JSD-TRADES beta=6, PGD_ASR = 0.326)
  H316_fgsm_at        (FGSM-AT in AWP ablation, PGD_ASR = 0.328)
  H326_fgsm_at_gp     (FGSM-AT + GP(0.01), PGD_ASR = 0.334)
  H346_trades_b1      (TRADES beta=1, PGD_ASR = 0.334)
  H303_alp_l1         (Adversarial Logit Pairing lam=1.0, PGD_ASR = 0.336)

Constraints (per critique):
  * HSJ here is L-inf (not the L2 default) since the campaign's PGD baseline is
    L-inf eps=0.1.
  * Hard-label only: queries are model(x).argmax(1).
  * Compute budget: 100 audit samples per defence x 1000 queries each, so the
    total HSJ cost is ~10 defences * 1e5 queries = 1e6 model passes.
  * Square-Attack ("simple" sub-1000 query variant) is also capped at 1000 queries
    per sample as the score-based companion.

This script DOES NOT RUN by default - it is provided for review.  Execute with
the project venv:
    .venv/bin/python dissertation_extension/hypotheses/h478_hsj_top10_audit.py
Output:
    results/fashion_mnist/h478_hsj_top10_audit_output.txt
"""
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
DS = "fashion_mnist"
EPS = 0.1
N_AUDIT = 100         # samples per defence  (critique: keep it small, HSJ is expensive)
HSJ_QUERY_BUDGET = 1000
SQUARE_QUERY_BUDGET = 1000
PGD_STEPS = 20
SEED = 0
OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", DS, "h478_hsj_top10_audit_output.txt",
)

# Gradient-masking flag threshold (HSJ-ASR - PGD-ASR > FLAG_DELTA absolute)
FLAG_DELTA = 0.10

META = C.dataset_meta(DS)


# ---------------------------------------------------------------------------
# defence builders
# ---------------------------------------------------------------------------
# All defences below are reproduced from the campaign at a SMALL training
# budget so the audit fits in one GPU-hour.  They are *not* expected to match
# the original campaign numbers exactly, only to be representative trained
# instances of the same defence class.
def _base_train(model, Xtr, Ytr, **kw):
    return C.train_model(model, Xtr, Ytr, ncls=META["n_classes"], **kw)


def build_fgsm_at(Xtr, Ytr, seed):
    C.set_seed(seed)
    m = C.build_model("cnn", META, width=32)
    return _base_train(m, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05,
                       adv_train=True, adv_eps=EPS, adv_steps=1)


def build_pgd_at(Xtr, Ytr, seed):
    C.set_seed(seed)
    m = C.build_model("cnn", META, width=32)
    return _base_train(m, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05,
                       adv_train=True, adv_eps=EPS, adv_steps=7)


def _trades_train(beta, Xtr, Ytr, seed, kind="kl"):
    """Lightweight TRADES variant: clean CE + beta * KL(clean || adv).
    `kind` is one of 'kl' (forward KL), 'revkl', 'jsd'."""
    C.set_seed(seed)
    m = C.build_model("cnn", META, width=32).to(C.DEVICE)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                          lr=0.05, momentum=0.9, weight_decay=1e-4)
    n = Xtr.size(0)
    for ep in range(6):
        perm = torch.randperm(n, device=Xtr.device)
        m.train()
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.pgd(m, xb, yb, eps=EPS, steps=5)
            opt.zero_grad()
            log_clean = F.log_softmax(m(xb), dim=1)
            log_adv = F.log_softmax(m(xa), dim=1)
            p_clean = log_clean.exp()
            p_adv = log_adv.exp()
            ce = F.cross_entropy(m(xb), yb)
            if kind == "kl":
                div = F.kl_div(log_adv, p_clean, reduction="batchmean")
            elif kind == "revkl":
                div = F.kl_div(log_clean, p_adv, reduction="batchmean")
            elif kind == "jsd":
                m_mix = 0.5 * (p_clean + p_adv)
                log_m = (m_mix + 1e-12).log()
                div = 0.5 * (F.kl_div(log_m, p_clean, reduction="batchmean")
                             + F.kl_div(log_m, p_adv, reduction="batchmean"))
            else:
                raise ValueError(kind)
            (ce + beta * div).backward()
            opt.step()
    m.eval()
    return m


def build_trades_b1(Xtr, Ytr, seed):
    return _trades_train(1.0, Xtr, Ytr, seed, "kl")


def build_revkl_trades(Xtr, Ytr, seed):
    return _trades_train(1.0, Xtr, Ytr, seed, "revkl")


def build_jsd_trades(Xtr, Ytr, seed):
    return _trades_train(6.0, Xtr, Ytr, seed, "jsd")


def build_alp(Xtr, Ytr, seed, lam=1.0):
    """Adversarial Logit Pairing (Kannan et al.): CE(clean) + lam * MSE(clean_logits, adv_logits)."""
    C.set_seed(seed)
    m = C.build_model("cnn", META, width=32).to(C.DEVICE)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                          lr=0.05, momentum=0.9, weight_decay=1e-4)
    n = Xtr.size(0)
    for ep in range(6):
        perm = torch.randperm(n, device=Xtr.device)
        m.train()
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.fgsm(m, xb, yb, eps=EPS)
            opt.zero_grad()
            lc, la = m(xb), m(xa)
            loss = F.cross_entropy(lc, yb) + lam * F.mse_loss(lc, la)
            loss.backward()
            opt.step()
    m.eval()
    return m


def build_awp_fgsm_at(Xtr, Ytr, seed, gamma=0.001):
    """FGSM-AT + Adversarial Weight Perturbation (Wu et al.).  AWP step uses a
    single-shot sign-gradient weight bump scaled by `gamma`."""
    C.set_seed(seed)
    m = C.build_model("cnn", META, width=32).to(C.DEVICE)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                          lr=0.05, momentum=0.9, weight_decay=1e-4)
    n = Xtr.size(0)
    for ep in range(6):
        perm = torch.randperm(n, device=Xtr.device)
        m.train()
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.fgsm(m, xb, yb, eps=EPS)
            # AWP: compute weight-grad on adv loss, bump weights sign-wise, then revert
            opt.zero_grad()
            adv_loss = F.cross_entropy(m(xa), yb)
            adv_loss.backward()
            backups = []
            with torch.no_grad():
                for p in m.parameters():
                    if p.grad is None:
                        backups.append(None); continue
                    bump = gamma * p.grad.sign() * (p.detach().abs() + 1e-6)
                    backups.append(bump.clone())
                    p.add_(bump)
            opt.zero_grad()
            loss = F.cross_entropy(m(xa), yb)
            loss.backward()
            with torch.no_grad():
                for p, bump in zip(m.parameters(), backups):
                    if bump is not None:
                        p.sub_(bump)
            opt.step()
    m.eval()
    return m


def build_fgsm_at_gp(Xtr, Ytr, seed, lam=0.01):
    """FGSM-AT + input-gradient penalty (H326-style)."""
    C.set_seed(seed)
    m = C.build_model("cnn", META, width=32).to(C.DEVICE)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                          lr=0.05, momentum=0.9, weight_decay=1e-4)
    n = Xtr.size(0)
    for ep in range(6):
        perm = torch.randperm(n, device=Xtr.device)
        m.train()
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.fgsm(m, xb, yb, eps=EPS)
            xb_g = xb.clone().detach().requires_grad_(True)
            ce_clean = F.cross_entropy(m(xb_g), yb)
            g_in, = torch.autograd.grad(ce_clean, xb_g, create_graph=True)
            gp = g_in.flatten(1).pow(2).sum(1).mean()
            opt.zero_grad()
            loss = F.cross_entropy(m(xa), yb) + lam * gp
            loss.backward()
            opt.step()
    m.eval()
    return m


# Order matters: name -> (builder fn, reference PGD-ASR from RESULTS_SUMMARY)
DEFENCES = [
    ("H371_pgd_at",       build_pgd_at,           0.196),
    ("H371_fgsm_at",      build_fgsm_at,          0.217),
    ("H351_awp",          build_awp_fgsm_at,      0.321),
    ("H363_revkl_trades", build_revkl_trades,     0.324),
    ("H348_fgsm_at_ctrl", build_fgsm_at,          0.324),
    ("H347_jsd_trades",   build_jsd_trades,       0.326),
    ("H316_fgsm_at",      build_fgsm_at,          0.328),
    ("H326_fgsm_at_gp",   build_fgsm_at_gp,       0.334),
    ("H346_trades_b1",    build_trades_b1,        0.334),
    ("H303_alp_l1",       build_alp,              0.336),
]


# ---------------------------------------------------------------------------
# pure-torch HopSkipJump (L-inf), hard-label only
# ---------------------------------------------------------------------------
@torch.no_grad()
def _is_adv(model, x, y_true):
    """Decision-based oracle: 1 if argmax label != y_true (untargeted)."""
    return (model(x).argmax(1) != y_true)


def _project_linf(x_orig, x_adv, eps):
    return torch.min(torch.max(x_adv, x_orig - eps), x_orig + eps).clamp(0, 1)


def _binary_search(model, x_orig, x_adv, y_true, eps, n_steps=10):
    """L-inf binary search along the (x_orig, x_adv) line, returning the
    closest-to-original adversarial that still satisfies the L-inf cap."""
    lo = torch.zeros(x_orig.size(0), device=x_orig.device)
    hi = torch.ones(x_orig.size(0), device=x_orig.device)
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        cand = x_orig + mid.view(-1, 1, 1, 1) * (x_adv - x_orig)
        cand = _project_linf(x_orig, cand, eps)
        adv = _is_adv(model, cand, y_true)
        hi = torch.where(adv, mid, hi)
        lo = torch.where(adv, lo, mid)
    out = x_orig + hi.view(-1, 1, 1, 1) * (x_adv - x_orig)
    return _project_linf(x_orig, out, eps)


@torch.no_grad()
def hsj_linf(model, x, y_true, eps, query_budget=1000, init_tries=50, mb_size=32, seed=0):
    """Pure-torch HopSkipJump (Chen et al. 2020) in L-inf mode.

    Steps (one outer iter = init binary search + gradient-estimate + geometric step):
      1. Random-uniform init in [0,1] until an adversarial init is found.
      2. Binary-search along the line x -> x_init to bring the boundary close.
      3. Estimate the gradient direction via Monte-Carlo Bernoulli sign-flip count.
      4. Move along sign(grad-direction) with geometric step search.
      5. Binary-search again.  Repeat until query_budget is exhausted.

    Returns (x_adv, queries_used, success_mask).  `success_mask` is the per-sample
    flag of whether the returned x_adv is adversarial and satisfies the L-inf
    eps constraint.
    """
    device = x.device
    B = x.size(0)
    g = torch.Generator(device=device).manual_seed(seed)

    # ---- step 1: random init ------------------------------------------------
    x_adv = x.clone()
    found = torch.zeros(B, dtype=torch.bool, device=device)
    queries = torch.zeros(B, dtype=torch.long, device=device)
    for _ in range(init_tries):
        cand = torch.rand(x.shape, generator=g, device=device)
        adv = _is_adv(model, cand, y_true)
        queries += 1
        new = adv & ~found
        x_adv = torch.where(new.view(-1, 1, 1, 1), cand, x_adv)
        found = found | adv
        if found.all():
            break
    # samples that never found an init are marked failed and skipped below.

    # ---- step 2..N: HSJ outer iterations -----------------------------------
    queries_per_iter = mb_size + 20    # MC + binary search overhead
    max_iters = max(1, (query_budget - int(queries.max().item())) // queries_per_iter)
    delta_step = eps * 0.5            # geometric step length (L-inf)

    for it in range(max_iters):
        # binary-search refine
        x_adv = _binary_search(model, x, x_adv, y_true, eps, n_steps=8)
        queries += 8

        # MC gradient-direction estimate (Bernoulli sign decision)
        u = torch.randn((mb_size,) + x.shape[1:], generator=g, device=device)
        u = u / (u.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
        # rescale per outer-iter; classical HSJ uses delta proportional to dist to original
        delta_probe = 0.01
        # per-sample probe: average sign(is_adv(x_adv + delta * u_k)) * u_k
        grad_est = torch.zeros_like(x_adv)
        for k in range(mb_size):
            uk = u[k].unsqueeze(0).expand_as(x_adv)
            probe = (x_adv + delta_probe * uk).clamp(0, 1)
            phi = _is_adv(model, probe, y_true).float() * 2 - 1   # +/-1
            grad_est = grad_est + phi.view(-1, 1, 1, 1) * uk
        queries += mb_size
        grad_est = grad_est / mb_size
        direction = grad_est.sign()                # L-inf step uses sign

        # geometric step search along `direction`
        step = delta_step
        for _ in range(10):
            cand = _project_linf(x, x_adv + step * direction, eps)
            adv = _is_adv(model, cand, y_true)
            queries += 1
            x_adv = torch.where(adv.view(-1, 1, 1, 1), cand, x_adv)
            if adv.all():
                break
            step = step * 0.5
        delta_step = max(delta_step * 0.9, eps * 0.05)

        if (queries.min() >= query_budget):
            break

    # final mask: adversarial AND within L-inf ball
    x_adv = _project_linf(x, x_adv, eps)
    success = _is_adv(model, x_adv, y_true) & found
    return x_adv.detach(), queries.detach(), success.detach()


# ---------------------------------------------------------------------------
# Square Attack (score-based, L-inf) -- Andriushchenko et al. 2020
# ---------------------------------------------------------------------------
@torch.no_grad()
def square_attack(model, x, y_true, eps, query_budget=1000, p_init=0.1, seed=0):
    """Minimal pure-torch L-inf Square Attack.  Score-based: uses the margin
    between true-class logit and max-other logit as the score to minimise."""
    device = x.device
    g = torch.Generator(device=device).manual_seed(seed)
    B, ch, H, W = x.shape

    # initialisation: vertical-stripe pattern of width 1 (paper's standard init)
    init = torch.empty_like(x).uniform_(-eps, eps, generator=g)
    x_adv = (x + init).clamp(0, 1)
    queries = torch.zeros(B, dtype=torch.long, device=device)

    def _margin(xx):
        z = model(xx)
        zy = z.gather(1, y_true.view(-1, 1)).squeeze(1)
        z_mask = z.clone()
        z_mask.scatter_(1, y_true.view(-1, 1), float("-inf"))
        zo = z_mask.max(1).values
        return zy - zo   # smaller (more negative) is more adversarial

    cur_score = _margin(x_adv)
    queries += 1
    success = cur_score < 0
    p = p_init
    for it in range(query_budget):
        if queries.min().item() >= query_budget:
            break
        # choose square side length
        s = max(1, int(round(math.sqrt(p * H * W))))
        s = min(s, H - 1, W - 1)
        i = torch.randint(0, H - s + 1, (B,), generator=g, device=device)
        j = torch.randint(0, W - s + 1, (B,), generator=g, device=device)
        delta = (torch.randint(0, 2, (B, ch, 1, 1), generator=g, device=device).float() * 2 - 1) * eps
        cand = x_adv.clone()
        for b in range(B):
            cand[b, :, i[b]:i[b] + s, j[b]:j[b] + s] = (
                x[b, :, i[b]:i[b] + s, j[b]:j[b] + s] + delta[b]
            ).clamp(0, 1)
        cand = _project_linf(x, cand, eps)
        new_score = _margin(cand)
        queries += 1
        improve = new_score < cur_score
        x_adv = torch.where(improve.view(-1, 1, 1, 1), cand, x_adv)
        cur_score = torch.where(improve, new_score, cur_score)
        success = success | (cur_score < 0)
        # decay p (paper's schedule, simplified)
        if it == query_budget // 4:
            p = p * 0.5
        if it == query_budget // 2:
            p = p * 0.5
    return x_adv.detach(), queries.detach(), success.detach()


# ---------------------------------------------------------------------------
# audit driver
# ---------------------------------------------------------------------------
def audit_defence(name, builder, ref_pgd, Xtr, Ytr, Xte, Yte):
    print(f"\n--- {name} ---")
    t0 = time.time()
    model = builder(Xtr, Ytr, seed=SEED)
    train_s = time.time() - t0

    # restrict audit to originally-correct samples (HSJ on already-wrong samples
    # is meaningless: argmax already != y_true).
    with torch.no_grad():
        pred = model(Xte).argmax(1)
        correct = (pred == Yte).nonzero(as_tuple=True)[0]
    if correct.numel() == 0:
        return {"name": name, "skipped": True}
    idx = correct[:N_AUDIT]
    Xa, Ya = Xte[idx], Yte[idx]

    # ---- PGD baseline (white-box) ---------------------------------------
    t0 = time.time()
    xa_pgd = C.pgd(model, Xa, Ya, eps=EPS, steps=PGD_STEPS)
    with torch.no_grad():
        pgd_asr = float((model(xa_pgd).argmax(1) != Ya).float().mean())
    pgd_s = time.time() - t0

    # ---- HSJ (decision-based) ------------------------------------------
    t0 = time.time()
    xa_hsj, q_hsj, ok_hsj = hsj_linf(model, Xa, Ya, eps=EPS,
                                     query_budget=HSJ_QUERY_BUDGET, seed=SEED)
    hsj_asr = float(ok_hsj.float().mean())
    hsj_mean_q = float(q_hsj.float().mean())
    hsj_s = time.time() - t0

    # ---- Square Attack (score-based) -----------------------------------
    t0 = time.time()
    xa_sq, q_sq, ok_sq = square_attack(model, Xa, Ya, eps=EPS,
                                       query_budget=SQUARE_QUERY_BUDGET, seed=SEED)
    sq_asr = float(ok_sq.float().mean())
    sq_mean_q = float(q_sq.float().mean())
    sq_s = time.time() - t0

    gap_hsj = hsj_asr - pgd_asr
    gap_sq = sq_asr - pgd_asr
    flagged = (gap_hsj > FLAG_DELTA) or (gap_sq > FLAG_DELTA)

    return {
        "name": name,
        "ref_pgd_asr": ref_pgd,
        "pgd_asr": pgd_asr,
        "hsj_asr": hsj_asr,
        "hsj_mean_queries": hsj_mean_q,
        "square_asr": sq_asr,
        "square_mean_queries": sq_mean_q,
        "gap_hsj_minus_pgd": gap_hsj,
        "gap_square_minus_pgd": gap_sq,
        "flagged": flagged,
        "train_s": train_s,
        "pgd_s": pgd_s,
        "hsj_s": hsj_s,
        "square_s": sq_s,
    }


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    log_lines = []

    def _log(s=""):
        print(s)
        log_lines.append(s)

    _log("=" * 78)
    _log("H478 - HopSkipJump audit of the top-10 PGD defences (Fashion-MNIST)")
    _log("=" * 78)
    _log(f"device={C.DEVICE}  eps={EPS}  N_AUDIT={N_AUDIT}  "
         f"HSJ_budget={HSJ_QUERY_BUDGET}  Square_budget={SQUARE_QUERY_BUDGET}")
    _log("Seed paper: Chen, Jordan & Wainwright (2020) HopSkipJumpAttack [S&P].")
    _log("Cited extras: Athalye-Carlini-Wagner 2018 (Obfuscated Gradients);")
    _log("             Tramer et al. 2020 (On Adaptive Attacks).")
    _log("")

    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=8000, n_eval=2000, seed=SEED)

    rows = []
    for (name, builder, ref) in DEFENCES:
        try:
            r = audit_defence(name, builder, ref, Xtr, Ytr, Xte, Yte)
        except Exception as e:  # do not let one defence kill the audit
            r = {"name": name, "error": str(e)}
        rows.append(r)
        if "error" in r:
            _log(f"  {name}: ERROR {r['error']}")
        elif r.get("skipped"):
            _log(f"  {name}: SKIPPED (no correct samples)")
        else:
            _log(f"  {name}: ref_PGD={r['ref_pgd_asr']:.3f}  "
                 f"PGD={r['pgd_asr']:.3f}  HSJ={r['hsj_asr']:.3f}  "
                 f"Square={r['square_asr']:.3f}  "
                 f"gapHSJ={r['gap_hsj_minus_pgd']:+.3f}  "
                 f"gapSq={r['gap_square_minus_pgd']:+.3f}  "
                 f"qHSJ={r['hsj_mean_queries']:.0f}  qSq={r['square_mean_queries']:.0f}  "
                 f"FLAG={'YES' if r['flagged'] else 'no'}  "
                 f"t(train/pgd/hsj/sq)="
                 f"{r['train_s']:.0f}/{r['pgd_s']:.0f}/{r['hsj_s']:.0f}/{r['sq_s']:.0f}s")

    # ---------- summary --------------------------------------------------
    _log("")
    _log("-" * 78)
    _log("PER-DEFENCE TABLE (PGD < HSJ by >0.10 = gradient-masking flag)")
    _log("-" * 78)
    _log(f"{'defence':22s} {'PGD':>6s} {'HSJ':>6s} {'Sq':>6s} "
         f"{'dHSJ':>7s} {'dSq':>7s} {'qHSJ':>6s} {'flag':>5s}")
    n_flagged = 0
    n_valid = 0
    for r in rows:
        if "error" in r or r.get("skipped"):
            _log(f"{r['name']:22s}  -- skipped/error --")
            continue
        n_valid += 1
        if r["flagged"]:
            n_flagged += 1
        _log(f"{r['name']:22s} {r['pgd_asr']:6.3f} {r['hsj_asr']:6.3f} "
             f"{r['square_asr']:6.3f} {r['gap_hsj_minus_pgd']:+7.3f} "
             f"{r['gap_square_minus_pgd']:+7.3f} {r['hsj_mean_queries']:6.0f} "
             f"{'YES' if r['flagged'] else 'no':>5s}")

    _log("")
    _log("Compute-equalisation note: PGD-AT trains with 7 inner PGD steps per")
    _log("batch (~7x the gradient cost of FGSM-AT); HSJ uses ~1000 forward passes")
    _log("per sample; Square uses ~1000 forward passes per sample.  ASR numbers")
    _log("are therefore NOT compute-equalised - HSJ/Square see far more model")
    _log("evaluations than the 20-step PGD baseline (which sees 20).  This biases")
    _log("the audit IN FAVOUR of finding masking, which is the conservative")
    _log("choice per Athalye et al. and Tramer et al.")

    _log("")
    _log("=" * 78)
    if n_valid == 0:
        verdict = "INCONCLUSIVE (no defences trained successfully)"
    elif n_flagged >= 3:
        verdict = (f"SUPPORTED: {n_flagged}/{n_valid} top-PGD defences show "
                   f"HSJ-ASR or Square-ASR > PGD-ASR + {FLAG_DELTA:.2f}, "
                   f"flagging gradient masking per Athalye et al. 2018.")
    else:
        verdict = (f"NOT SUPPORTED: only {n_flagged}/{n_valid} top defences "
                   f"flagged; the strongest PGD defences appear to be genuinely "
                   f"robust under hard-label and score-based audit.")
    _log(f"HEADLINE VERDICT: {verdict}")
    _log("=" * 78)

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
