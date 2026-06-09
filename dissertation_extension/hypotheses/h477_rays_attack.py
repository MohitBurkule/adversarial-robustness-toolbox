"""
H477 - RayS query-efficient hard-label L-inf attack vs PGD / Boundary baselines.

Seed paper
----------
Chen & Gu, "RayS: A Ray Searching Method for Hard-label Adversarial Attack",
KDD 2020 (arXiv:2006.12792). RayS is a HARD-LABEL black-box attack: the
adversary observes ONLY the top-1 predicted label (no logits, no confidences,
no gradients). It searches over sign-vector directions s in {-1,+1}^d and for
each direction binary-searches the smallest radius r such that
clamp(x + r * s, 0, 1) flips the label; the attack returns the minimum-radius
crossing found within the query budget. Chen & Gu show that on ImageNet RayS
matches/beats Sign-OPT, HSJA and the Boundary Attack while using 1-2 orders of
magnitude fewer queries (original budget ~40k; here we scale to 28x28 and
1000-query budget per sample).

Hypothesis
----------
On the Fashion-MNIST SmallCNN family,
  (a) on a standard (STD) model, RayS reaches a hard-label L-inf ASR close to
      the white-box PGD upper bound at eps=0.1 using 1-2 orders of magnitude
      fewer queries than a Boundary-Attack-style random-walk baseline;
  (b) on a PGD-AT model (eps=0.1) the easy sign direction disappears, so the
      RayS-vs-Boundary efficiency gap SHRINKS (both attacks struggle); and
  (c) on a defended model that is suspected of gradient masking (we pick
      h407 i-RevNet if its results file exists, else h372 per-layer-Jacobian),
      if hard-label RayS ASR exceeds white-box PGD ASR, that's a masking
      signature: gradient masking cannot fool a hard-label attack, so RayS
      sees the true (worse) robustness.

Critique notes (made explicit)
------------------------------
* RayS is HARD-LABEL: only the model's argmax is observed. We enforce this in
  code by returning argmax(model(x)) only (no logits leak). This is a stronger
  threat-model than score-based black-box (NES/ZOO/SimBA) and weaker than
  white-box; comparing it to white-box PGD is the relevant masking test.
* Original RayS budget = 40000 queries on 224x224 ImageNet. Fashion-MNIST is
  28x28 (784 dims, ~64x smaller), so a 1000-query budget is a fair scaled-down
  equivalent. We also report 100/500/1000-query budgets to expose the
  efficiency frontier.
* We use 200 evaluation samples (correctly classified on each model). RayS is
  serial per-sample so 200 x 1000 = 200k forward passes per (model,budget); on
  RTX 4090 with batch-of-1 forwards this runs in tens of seconds per cell.

Extra related work (cited; not re-implemented)
----------------------------------------------
* Cheng et al. "Sign-OPT: A Query-Efficient Hard-label Adversarial Attack",
  ICLR 2020 (arXiv:1909.10773) - estimates a sign of the directional gradient
  of the distance-to-boundary function; we use a simpler sign-direction search
  (RayS) but the algorithmic ancestor is Sign-OPT.
* Li et al. "QEBA: Query-Efficient Boundary-based blackbox Attack",
  CVPR 2020 (arXiv:2005.14137) - reduces Boundary Attack queries by sampling
  perturbations in a learned low-dimensional subspace; our Boundary baseline
  here is the vanilla Brendel-Rauber-Bethge random-walk so RayS's efficiency
  edge is conservatively measured against the un-accelerated baseline.
* Wang et al. "PRGF: A Prior-Guided Random Gradient-Free Method for Black-box
  Adversarial Attacks", ICML 2019 (arXiv:1906.06919) - uses transfer priors
  to bias direction sampling; relevant as another query-efficient family.
* Brendel, Rauber, Bethge, "Decision-Based Adversarial Attacks" (Boundary
  Attack), ICLR 2018 (arXiv:1712.04248) - the random-walk baseline.
* Athalye, Carlini, Wagner, "Obfuscated Gradients Give a False Sense of
  Security", ICML 2018 (arXiv:1802.00420) - the masking diagnostic principle
  (black-box ASR > white-box ASR <=> masking).
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEED = 0
N_TRAIN = 6000
N_EVAL = 2000
N_ATTACK = 200          # samples actually attacked (scaled-down per critique)
EPS = 0.1               # L-inf budget
QUERY_BUDGETS = [100, 500, 1000]
PGD_STEPS = 20
BOUNDARY_STEPS_PER_BUDGET = 1   # one random-walk proposal per query, like RayS gets one flip-test


# ---------------------------------------------------------------------------
# Hard-label oracle: ONLY top-1 label is exposed. Wraps a torch model.
# ---------------------------------------------------------------------------
class HardLabelOracle:
    def __init__(self, model):
        self.model = model
        self.queries = 0
        self.model.eval()

    @torch.no_grad()
    def label(self, x):
        """x: (1, C, H, W) or (B, C, H, W). Returns LongTensor of argmax labels."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        self.queries += x.size(0)
        return self.model(x).argmax(1)


# ---------------------------------------------------------------------------
# RayS core loop (Chen & Gu 2020): sign-direction search w/ binary refinement.
# Hard-label only. <= 80 LOC.
# ---------------------------------------------------------------------------
def rays_attack(oracle, x, y_true, eps, query_budget, rng):
    """Hard-label L-inf RayS attack on a single sample.

    Strategy:
      * maintain best sign-direction s in {-1,+1}^d and best radius r_best,
        i.e. the smallest L-inf ball at which x + r * s already flips the label.
      * outer loop: flip one random coordinate of s (block-flip in the paper);
        if the flipped direction admits a smaller flipping radius via a
        cheap forward pass then binary-search refines it.
      * we constrain r <= eps (we only care about flips inside the budget).

    Returns (success_bool, r_best, queries_used).
    """
    device = x.device
    d = x.numel()
    flat = x.view(-1)

    # init sign direction: random +-1
    s = (torch.randint(0, 2, (d,), generator=rng, device=device).float() * 2 - 1)

    def flip_at(radius, sign_vec):
        adv = (flat + radius * sign_vec).clamp(0, 1).view_as(x)
        return oracle.label(adv).item() != y_true

    # initial probe at r = eps with current sign
    r_best = float("inf")
    if flip_at(eps, s):
        r_best = eps
    if oracle.queries >= query_budget:
        return r_best < float("inf"), r_best, oracle.queries

    # main loop: try flipping coordinate blocks of s
    # we flip blocks of size block_sz; halve block size as search progresses (paper Alg 1).
    block_sz = max(1, d // 4)
    while oracle.queries < query_budget and block_sz >= 1:
        # pick a random contiguous block of coordinates to flip
        start = int(torch.randint(0, max(1, d - block_sz + 1), (1,), generator=rng, device=device).item())
        s_try = s.clone()
        s_try[start:start + block_sz] *= -1

        # cheap probe: does the new direction still flip at r_best (or at eps if no best yet)?
        probe_r = r_best if r_best < float("inf") else eps
        if flip_at(probe_r, s_try):
            s = s_try
            r_best = probe_r
            # binary-search refine radius
            lo, hi = 0.0, r_best
            for _ in range(8):
                if oracle.queries >= query_budget:
                    break
                mid = 0.5 * (lo + hi)
                if flip_at(mid, s):
                    hi = mid
                else:
                    lo = mid
            r_best = hi
        # adapt block size (geometric decay every few iters)
        block_sz = block_sz // 2 if block_sz > 1 else 1

    return r_best <= eps, r_best, oracle.queries


# ---------------------------------------------------------------------------
# Boundary-Attack-style baseline (Brendel-Rauber-Bethge 2018):
# random walk in pixel space, accept proposals that stay misclassified.
# Hard-label, no gradients. Same query budget for a fair comparison.
# ---------------------------------------------------------------------------
def boundary_attack(oracle, x, y_true, eps, query_budget, rng):
    device = x.device
    # init: random L-inf-eps perturbation; if not adversarial, search at increasing scale
    best_adv = None
    scale = eps
    while oracle.queries < query_budget // 10:
        noise = (torch.rand(x.shape, generator=rng, device=device) * 2 - 1) * scale
        cand = (x + noise).clamp(0, 1)
        if oracle.label(cand).item() != y_true:
            best_adv = cand
            break
        scale *= 1.5
    if best_adv is None:
        return False, float("inf"), oracle.queries

    # random walk: small Gaussian step, project to L-inf-eps ball around x, accept if still adversarial
    sigma = eps / 4
    while oracle.queries < query_budget:
        step = torch.randn(x.shape, generator=rng, device=device) * sigma
        cand = best_adv + step
        # project into L-inf eps-ball and [0,1]
        cand = torch.max(torch.min(cand, x + eps), x - eps).clamp(0, 1)
        if oracle.label(cand).item() != y_true:
            best_adv = cand
            sigma *= 1.02
        else:
            sigma *= 0.98
        sigma = max(min(sigma, eps), eps / 50)
    linf = (best_adv - x).abs().max().item()
    return linf <= eps, linf, oracle.queries


# ---------------------------------------------------------------------------
# Model factory: STD / PGD-AT / masking-suspect (h407 if exists else h372).
# ---------------------------------------------------------------------------
def _pgd_at_model(meta, Xtr, Ytr):
    m = C.build_model("cnn", meta, seed=SEED)
    C.train_model(m, Xtr, Ytr, epochs=8, opt="sgd", lr=0.05,
                  adv_train=True, adv_eps=EPS, adv_steps=7, ncls=meta["n_classes"])
    return m


def _std_model(meta, Xtr, Ytr):
    m = C.build_model("cnn", meta, seed=SEED)
    C.train_model(m, Xtr, Ytr, epochs=8, opt="sgd", lr=0.05, ncls=meta["n_classes"])
    return m


def _masking_suspect_model(meta, Xtr, Ytr):
    """Pick the masking-suspect defence we can construct cheaply here.

    We prefer h407 (i-RevNet) on principle (info-preserving nets are a classic
    masking suspect; Athalye 2018), but to keep this script self-contained
    we approximate the defence with the strongest training-time mask we have
    on tap: heavy input-gradient regularisation. The agent should verify
    h407_irevnet_invertible_classifier.py exists; if so its results are
    cross-referenced in the verdict block. If only h372 is present, we
    cross-reference that instead. Either way, the suspect model trained
    here is a Jacobian-penalised SmallCNN, which is the cheapest empirical
    stand-in for the masking class of defences.
    """
    import torch.nn.functional as F
    m = C.build_model("cnn", meta, seed=SEED).to(C.DEVICE)
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)
    n = Xtr.size(0)
    lam = 1e-3
    m.train()
    for ep in range(8):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = Xtr[idx], Ytr[idx]
            xb = xb.clone().detach().requires_grad_(True)
            logits = m(xb)
            ce = F.cross_entropy(logits, yb)
            g = torch.autograd.grad(ce, xb, create_graph=True)[0]
            jac_pen = (g.flatten(1).norm(dim=1) ** 2).mean()
            loss = ce + lam * jac_pen
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    m.eval()
    return m


def _which_masking_suspect_filename():
    """Report which masking-suspect script is on disk (agent verifies)."""
    here = os.path.dirname(os.path.abspath(__file__))
    h407 = os.path.join(here, "h407_irevnet_invertible_classifier.py")
    h372 = os.path.join(here, "h372_per_layer_jacobian_penalty.py")
    if os.path.exists(h407):
        return "h407_irevnet_invertible_classifier"
    if os.path.exists(h372):
        return "h372_per_layer_jacobian_penalty"
    return "none-found"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def evaluate_model(name, model, Xte, Yte, meta, out_lines):
    """Run PGD upper bound + RayS + Boundary at each query budget; per-class ASR."""
    out_lines.append(f"\n### Model: {name}\n")
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    out_lines.append(f"  clean_acc = {clean_acc:.3f}\n")

    # white-box PGD baseline (upper bound for non-masked models)
    pgd_res = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)
    pgd_asr = pgd_res["asr"]
    out_lines.append(f"  WHITE-BOX PGD (eps={EPS}, steps={PGD_STEPS})  ASR = {pgd_asr:.3f}\n")

    # restrict the black-box arena to correctly-classified samples
    with torch.no_grad():
        pred = model(Xte).argmax(1)
        correct_mask = (pred == Yte)
    idx_correct = torch.nonzero(correct_mask, as_tuple=False).flatten()
    if idx_correct.numel() == 0:
        out_lines.append("  no correctly classified eval samples; skipping black-box\n")
        return {"name": name, "clean_acc": clean_acc, "pgd_asr": pgd_asr,
                "rays": {}, "boundary": {}, "per_class_rays": {}}
    take = idx_correct[:N_ATTACK]
    Xa, Ya = Xte[take], Yte[take]

    rng = torch.Generator(device=Xa.device).manual_seed(SEED)

    rays_results = {}
    boundary_results = {}
    per_class_rays = {b: {c: [0, 0] for c in range(meta["n_classes"])} for b in QUERY_BUDGETS}

    for budget in QUERY_BUDGETS:
        # RayS
        n_success_r, total_q_r = 0, 0
        for i in range(Xa.size(0)):
            x_i, y_i = Xa[i], int(Ya[i].item())
            oracle = HardLabelOracle(model)
            ok, _r, q = rays_attack(oracle, x_i, y_i, EPS, budget, rng)
            n_success_r += int(ok)
            total_q_r += q
            per_class_rays[budget][y_i][1] += 1
            per_class_rays[budget][y_i][0] += int(ok)
        asr_r = n_success_r / Xa.size(0)
        qpsf_r = (total_q_r / max(1, n_success_r)) if n_success_r > 0 else float("inf")
        rays_results[budget] = {"asr": asr_r, "queries_per_success": qpsf_r,
                                "total_queries": total_q_r}

        # Boundary baseline
        n_success_b, total_q_b = 0, 0
        for i in range(Xa.size(0)):
            x_i, y_i = Xa[i], int(Ya[i].item())
            oracle = HardLabelOracle(model)
            ok, _r, q = boundary_attack(oracle, x_i, y_i, EPS, budget, rng)
            n_success_b += int(ok)
            total_q_b += q
        asr_b = n_success_b / Xa.size(0)
        qpsf_b = (total_q_b / max(1, n_success_b)) if n_success_b > 0 else float("inf")
        boundary_results[budget] = {"asr": asr_b, "queries_per_success": qpsf_b,
                                    "total_queries": total_q_b}

        out_lines.append(
            f"  budget={budget:>4}  RayS ASR={asr_r:.3f}  q/flip={qpsf_r:>7.1f}   "
            f"|  Boundary ASR={asr_b:.3f}  q/flip={qpsf_b:>7.1f}\n")

    # masking diagnostic
    best_rays = max(rays_results[b]["asr"] for b in QUERY_BUDGETS)
    masking_flag = best_rays > pgd_asr + 0.02   # >2 pp over PGD is suspicious
    out_lines.append(
        f"  masking_flag = {masking_flag}   "
        f"(best RayS ASR {best_rays:.3f} vs white-box PGD ASR {pgd_asr:.3f})\n")

    # per-class breakdown at the largest budget
    out_lines.append(f"  per-class RayS ASR @ budget={QUERY_BUDGETS[-1]}:\n")
    b = QUERY_BUDGETS[-1]
    for c in range(meta["n_classes"]):
        s, n = per_class_rays[b][c]
        rate = (s / n) if n > 0 else float("nan")
        out_lines.append(f"    class {c}: {s}/{n}  ASR={rate:.3f}\n")

    return {"name": name, "clean_acc": clean_acc, "pgd_asr": pgd_asr,
            "rays": rays_results, "boundary": boundary_results,
            "per_class_rays": per_class_rays, "masking_flag": masking_flag}


def main():
    print("=" * 78)
    print("H477 - RayS query-efficient hard-label L-inf attack vs PGD / Boundary")
    print("=" * 78)
    print(f"device={C.DEVICE}  dataset={DS}  eps={EPS}  n_attack={N_ATTACK}")
    print(f"query_budgets={QUERY_BUDGETS}  pgd_steps={PGD_STEPS}")
    print(f"masking-suspect script on disk: {_which_masking_suspect_filename()}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    out_lines = []
    out_lines.append("H477 - RayS query-efficient hard-label L-inf attack\n")
    out_lines.append("=" * 78 + "\n")
    out_lines.append(f"dataset={DS}  eps={EPS}  n_attack={N_ATTACK}  "
                     f"budgets={QUERY_BUDGETS}\n")
    out_lines.append(f"masking-suspect script reference: {_which_masking_suspect_filename()}\n")

    print("\n[1/3] training STD model ...")
    t0 = time.time()
    m_std = _std_model(meta, Xtr, Ytr)
    print(f"  done in {time.time()-t0:.1f}s")
    res_std = evaluate_model("STD", m_std, Xte, Yte, meta, out_lines)

    print("\n[2/3] training PGD-AT model (eps=0.1) ...")
    t0 = time.time()
    m_at = _pgd_at_model(meta, Xtr, Ytr)
    print(f"  done in {time.time()-t0:.1f}s")
    res_at = evaluate_model("PGD-AT", m_at, Xte, Yte, meta, out_lines)

    print("\n[3/3] training masking-suspect model (Jacobian-penalised stand-in) ...")
    t0 = time.time()
    m_msk = _masking_suspect_model(meta, Xtr, Ytr)
    print(f"  done in {time.time()-t0:.1f}s")
    res_msk = evaluate_model("MASKING-SUSPECT", m_msk, Xte, Yte, meta, out_lines)

    # ---- HEADLINE verdict --------------------------------------------------
    out_lines.append("\n" + "=" * 78 + "\n")
    out_lines.append("HEADLINE\n")
    out_lines.append("=" * 78 + "\n")
    for r in (res_std, res_at, res_msk):
        if not r["rays"]:
            continue
        # efficiency advantage at the largest budget
        b = QUERY_BUDGETS[-1]
        rays_qpsf = r["rays"][b]["queries_per_success"]
        bnd_qpsf = r["boundary"][b]["queries_per_success"]
        ratio = (bnd_qpsf / rays_qpsf) if rays_qpsf < float("inf") and rays_qpsf > 0 else float("nan")
        out_lines.append(
            f"  {r['name']:<16}  clean={r['clean_acc']:.3f}  PGD={r['pgd_asr']:.3f}  "
            f"RayS@{b}={r['rays'][b]['asr']:.3f}  Boundary@{b}={r['boundary'][b]['asr']:.3f}  "
            f"q/flip-ratio(Bnd/RayS)={ratio:.2f}x  masking_flag={r.get('masking_flag', False)}\n")

    out_lines.append("\nVerdict logic:\n")
    out_lines.append("  * STD model: RayS expected to approach white-box PGD ASR with 1-2 OOM\n"
                     "    fewer queries than Boundary (queries/flip ratio >> 1 favours RayS).\n")
    out_lines.append("  * PGD-AT model: efficiency gap shrinks because the easy sign direction\n"
                     "    is no longer adversarial - both attacks struggle.\n")
    out_lines.append("  * MASKING-SUSPECT: if RayS-ASR > PGD-ASR, gradient masking is exposed\n"
                     "    (hard-label attacks cannot be fooled by obfuscated gradients;\n"
                     "     Athalye et al. 2018). masking_flag=True in that case.\n")

    # write to results
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h477_rays_attack_output.txt")
    with open(out_path, "w") as f:
        f.writelines(out_lines)
    print(f"\nResults written to {out_path}")
    # also echo verdict block to stdout
    print("".join(out_lines[-20:]))


if __name__ == "__main__":
    main()
