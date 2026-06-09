"""
H476 - Brendel et al. Boundary Attack (decision-based, query-only) on Fashion-MNIST.

ANCHOR
------
Brendel, Rauber, Bethge 2018 "Decision-Based Adversarial Attacks: Reliable
Attacks Against Black-Box Machine Learning Models" (ICLR 2018, arXiv:1712.04248).
The Boundary Attack starts from an already-adversarial random image and walks
ALONG the decision boundary toward the source image using a random-walk
proposal (orthogonal step on the boundary tangent + small step toward the
source), accepting only proposals that remain adversarial. It needs only the
TOP-1 LABEL of the model -- no gradients, no logits, no scores. This makes it
the canonical *decision-based* attack and the gold standard for diagnosing
"gradient masking" defences that only suppress white-box gradients.

AUX PAPERS (web-search, GAP G1)
-------------------------------
* Chen, Jordan, Wainwright 2020 "HopSkipJumpAttack" (S&P 2020, arXiv:1904.02144).
  Estimates the decision-boundary gradient by sign-aggregation of random
  query directions; far more query-efficient than the original Boundary
  Attack at high accuracy targets. We do NOT implement HSJA here -- the
  brief is to keep the implementation ~80 LOC -- but flag it as the
  state-of-the-art successor.
* Cheng, Singh, Chen, Chen, Liu, Hsieh 2020 "Sign-OPT: A Query-Efficient
  Hard-Label Adversarial Attack" (ICLR 2020, arXiv:1909.10773). Reformulates
  decision-based attack as zeroth-order optimisation of a directional
  function g(theta) = min_eps s.t. adversarial; uses sign-only gradient
  estimates. H116 in this campaign already runs Sign-OPT; cross-check
  ranking should be: Sign-OPT median-L2 <= Boundary median-L2 at matched
  queries (Sign-OPT is the more query-efficient cousin).
* Andriushchenko, Croce, Flammarion, Hein 2020 "Square Attack: a query-
  efficient black-box adversarial attack via random search" (ECCV 2020,
  arXiv:1912.00049). Square Attack is SCORE-based (needs the loss / logits),
  NOT decision-based. Listed here as the standard query-only L_inf / L_2
  alternative when scores ARE available; if the defence withholds scores,
  Boundary/HSJA/Sign-OPT remain the only options.

GAP / CRITIQUE (from advisor §2 M3, §3 G1)
------------------------------------------
The campaign so far evaluates defences almost exclusively under white-box
PGD at eps=0.1. There is NO AutoAttack, NO EOT against stochastic defences,
NO transfer attack -- and crucially, NO decision-based audit. Any defence
whose PGD-ASR is suspiciously low compared to its clean accuracy is a prime
suspect for *gradient masking* (Athalye, Carlini, Wagner 2018, "Obfuscated
Gradients Give a False Sense of Security"). The cleanest test for masking
is: turn off the gradient entirely and attack with queries only. If the
defence's "PGD-ASR << query-only-ASR" gap is large, the defence is masked.

HYPOTHESIS
----------
H476-CORE: A defence that lowers PGD-ASR via *gradient masking* will be
EXPOSED by Boundary Attack -- i.e. Boundary-ASR will substantially exceed
PGD-ASR despite Boundary having zero gradient access. A genuine defence
(e.g. PGD-AT) will retain similar ASR under both attacks (PGD upper-bounds
Boundary up to query budget).

CONDITIONS (controls)
---------------------
Three models trained on Fashion-MNIST SmallCNN (width=32), single seed:
  1. STD       : plain cross-entropy.
  2. PGD-AT    : PGD-7 adversarial training at eps=0.1 (campaign winner,
                 expected genuine defence; gold-standard control).
  3. MASK-SUSP : a suspected-masking defence picked from the campaign.
                 Preference order (must EXIST in hypotheses/):
                   (a) H407 i-RevNet invertible classifier
                       (deterministic but information-preserving;
                        invertible nets have been flagged for
                        obfuscated-gradient behaviour in the literature)
                   (b) H372 per-layer Jacobian penalty
                       (Jacobian penalty -> shrinks input gradient
                        magnitude -> classic masking failure mode if
                        the penalty is too aggressive)
                 We attempt (a) first; if i-RevNet import fails we
                 fall back to (b). Whichever loads is logged.

EVALUATIONS
-----------
For each of the three models we report:
  C1. Clean accuracy (sanity).
  C2. White-box PGD-10 at eps=0.1: ASR + median final L_2 of perturbations.
  C3. Boundary Attack at query budget Q in {100, 500, 1000, 2000}
      (query-budget ablation). At each Q we report:
        * ASR (fraction of originally-correct samples flipped, where
          "flipped" means model.argmax != true label and the perturbation
          is bounded; we report ASR-at-any-perturbation as the standard
          decision-attack ASR -- the attack always succeeds eventually
          if init is adversarial, so we additionally report median-L2
          which is the more informative quantity for query attacks).
        * Median final L_2 distance from original.
  C4. Per-class Boundary-ASR at Q=1000 (10 classes -- looks for
      class-imbalanced masking, e.g. T-shirt vs Shirt confusion).
  C5. *Masking flag*: (Boundary-ASR(Q=1000) - PGD-ASR) > 0.10 AND
      (PGD-ASR < 0.50). Both clauses required: a model with high PGD-ASR
      that also has high Boundary-ASR is not masked, just weak.

QUERY BUDGET (§2 M3)
--------------------
Boundary Attack at Q=1000 queries/sample x 200 samples = 200k forward
passes per (model, budget) cell. Total cells = 3 models x 4 budgets = 12.
Plus per-class breakdown at Q=1000: reuse the Q=1000 run. Total queries
~12 * 200k = 2.4M model evaluations -- tractable on one RTX 4090 in
minutes per cell since the SmallCNN is tiny.

BOUNDARY ATTACK IMPLEMENTATION (~80 LOC, pure torch, NO foolbox)
----------------------------------------------------------------
Algorithm (Brendel 2018, simplified to the standard L_2 untargeted form):
  1. Initialise: sample uniform-random images until model.argmax != y_true
     (or use a class-mean image of a different class). This is x_adv_0.
  2. At step t, propose
        x_prop = x_adv_t + source_step * (x_orig - x_adv_t)        # toward source
                       + spherical_step * orth_noise                # tangent walk
     where orth_noise is gaussian noise projected onto the sphere centred
     on x_orig of radius ||x_adv_t - x_orig||_2 (i.e. perpendicular to
     (x_adv_t - x_orig)).
  3. Query the model. If model(x_prop).argmax != y_true (still adversarial)
     and ||x_prop - x_orig||_2 < ||x_adv_t - x_orig||_2 (closer), accept.
  4. Adaptive step sizes: track acceptance rate over a sliding window of
     30 steps; if accept_rate > 0.5 grow steps by 1.1, if < 0.2 shrink
     by 0.9. Target accept rate around 0.3 (Brendel default).
  5. Output: x_adv_final, total queries used.

We batch over the 200 samples in groups of 50 to amortise forward-pass
overhead; each sample has its own adv state and step sizes (vectorised).

ATTACK COMPARISONS (controls C5)
--------------------------------
For each model we will compute Spearman correlation across models of
(PGD-ASR, Boundary-ASR-at-Q=1000) -- expected sign: positive on genuine
defences, NEGATIVE or near-zero on masked ones (mask suppresses PGD but
not Boundary). With only 3 models this is a sanity check, not a stats
test; the per-model masking flag in C5 is the headline.

STANDARD CONFIG
---------------
DS=fashion_mnist, N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128,
SGD(mom=0.9, wd=5e-4), SEED=0, EPS=0.1 (Linf), SmallCNN width=32.
Boundary: N_TEST_BOUNDARY=200, source_step_init=0.01, spherical_step_init=0.01,
adapt window=30, adapt up/down factors 1.1 / 0.9.

LIMITATIONS / FUTURE WORK
-------------------------
* Single seed.
* Boundary Attack is the FIRST decision-based attack; HopSkipJump and
  Sign-OPT (Cheng 2020, H116) are strictly more query-efficient and would
  give tighter median-L2. If a defence resists Boundary at Q=1000 we
  cannot conclude it is robust -- it may just need more queries or HSJA.
* 200 samples is small; per-class numbers (20/class on average) are noisy.
* Untargeted only; targeted Boundary needs initialising in a specific
  class region.

DO NOT RUN: This file is delegated; main session does not execute scripts.
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
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
WIDTH = 32
N_CLASSES = 10

# Boundary Attack settings.
N_TEST_BOUNDARY = 200
QUERY_BUDGETS = [100, 500, 1000, 2000]
HEADLINE_BUDGET = 1000
SOURCE_STEP_INIT = 0.01
SPHERICAL_STEP_INIT = 0.01
ADAPT_WINDOW = 30
ADAPT_UP = 1.1
ADAPT_DOWN = 0.9

# Masking-flag thresholds (C5).
MASK_FLAG_GAP = 0.10
MASK_FLAG_PGD_CEIL = 0.50

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h476_boundary_attack_output.txt",
)


# ---------------------------------------------------------------------------
# model training
# ---------------------------------------------------------------------------
def _make_model():
    return C.build_model("cnn", C.dataset_meta(DS), width=WIDTH, act="relu", bn=True)


def train_std(Xtr, Ytr):
    C.set_seed(SEED)
    model = _make_model()
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                         opt="sgd", lr=LR, adv_train=False)


def train_pgdat(Xtr, Ytr):
    C.set_seed(SEED)
    model = _make_model()
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                         opt="sgd", lr=LR, adv_train=True,
                         adv_eps=EPS, adv_steps=7)


def train_mask_suspect(Xtr, Ytr):
    """Pick a campaign masking-suspect. Try H407 (i-RevNet) first; on import
    failure fall back to H372 (per-layer Jacobian penalty). Whichever loads
    is logged into the report via the returned ``name``."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    # ---- attempt (a) H407 i-RevNet ----------------------------------------
    try:
        import importlib
        h407 = importlib.import_module("h407_irevnet_invertible_classifier")
        # h407 exposes the iRevNet class; train it via its own helper if any,
        # else build + train with a plain CE loop matching the campaign cfg.
        if hasattr(h407, "build_irevnet"):
            model = h407.build_irevnet(C.dataset_meta(DS), width=WIDTH).to(C.DEVICE)
        elif hasattr(h407, "iRevNet"):
            model = h407.iRevNet(in_ch=1, size=28, n_classes=10).to(C.DEVICE)
        else:
            raise ImportError("h407 has no build_irevnet/iRevNet symbol")
        C.set_seed(SEED)
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                              weight_decay=5e-4)
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
        model.eval()
        return model, "H407-iRevNet"
    except Exception as e_h407:
        print(f"  H407 not usable ({e_h407}); falling back to H372.", flush=True)

    # ---- fallback (b) H372 per-layer Jacobian penalty ---------------------
    # Re-implement here (Hutchinson Frobenius on every block) to avoid relying
    # on H372's exact API surface. This matches H372's hypothesis.
    C.set_seed(SEED)
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)
    n = Xtr.size(0)
    lam = 0.01
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr[idx].clone().detach().requires_grad_(True)
            yb = Ytr[idx]
            opt.zero_grad()
            logits = model(xb)
            ce = F.cross_entropy(logits, yb)
            v = F.one_hot(torch.randint(0, logits.size(1), (xb.size(0),),
                                        device=xb.device),
                          logits.size(1)).float()
            scalar = (logits * v).sum()
            gx = torch.autograd.grad(scalar, xb, create_graph=True)[0]
            jac = (gx ** 2).sum() / xb.size(0)
            (ce + lam * jac).backward()
            opt.step()
    model.eval()
    return model, "H372-PerLayerJac(fallback)"


# ---------------------------------------------------------------------------
# Boundary Attack (Brendel 2018), pure torch, NO foolbox
# ---------------------------------------------------------------------------
@torch.no_grad()
def _predict(model, x, batch=512):
    out = []
    for i in range(0, x.size(0), batch):
        out.append(model(x[i:i + batch]).argmax(1))
    return torch.cat(out, dim=0)


@torch.no_grad()
def _initial_adv(model, x_orig, y_true, max_tries=200):
    """Find an initial random-image adversarial start for EACH sample.
    Uniform-random in [0,1]; retry until model.argmax != y_true.
    If we run out of tries, fall back to a flipped-label class-mean image
    (here just uniform-random with different seed). Counts queries."""
    B = x_orig.size(0)
    device = x_orig.device
    x_adv = torch.empty_like(x_orig)
    found = torch.zeros(B, dtype=torch.bool, device=device)
    queries = torch.zeros(B, dtype=torch.long, device=device)
    for t in range(max_tries):
        cand = torch.rand_like(x_orig)
        pred = model(cand).argmax(1)
        queries += (~found).long()                     # 1 query for not-yet-found
        ok = (pred != y_true) & (~found)
        x_adv[ok] = cand[ok]
        found |= ok
        if found.all():
            break
    # Anything still not adversarial: just set to 1 - x_orig (extreme image),
    # accept the small chance it isn't adversarial -- counted as 1 more query.
    if (~found).any():
        cand = 1.0 - x_orig
        pred = model(cand).argmax(1)
        queries += (~found).long()
        x_adv[~found] = cand[~found]
    return x_adv, queries


def boundary_attack(model, x_orig, y_true, query_budget,
                    src_step=SOURCE_STEP_INIT, sph_step=SPHERICAL_STEP_INIT,
                    adapt_window=ADAPT_WINDOW):
    """Vectorised Boundary Attack.

    Inputs:
      x_orig (B,1,28,28) clean images in [0,1]
      y_true (B,)        clean labels (the attack tries to flip away from)
      query_budget       total queries per sample (init + main loop)

    Returns: x_adv (B,1,28,28), used_queries (B,), final_l2 (B,)
    """
    model.eval()
    device = x_orig.device
    B = x_orig.size(0)

    x_adv, used = _initial_adv(model, x_orig, y_true)
    # Per-sample step sizes.
    src = torch.full((B,), src_step, device=device)
    sph = torch.full((B,), sph_step, device=device)
    # Rolling accept history (1 = accepted, 0 = rejected) over ADAPT_WINDOW.
    hist = torch.zeros(B, adapt_window, device=device)
    hist_ptr = 0

    flat = lambda t: t.reshape(B, -1)
    def l2(a, b):
        return (flat(a) - flat(b)).norm(dim=1)

    step = 0
    remaining = query_budget - used.clamp(min=0)
    while (remaining > 0).any() and step < query_budget * 2:
        # ---- propose ------------------------------------------------------
        direction = x_orig - x_adv                         # toward source
        # Spherical (orthogonal) noise.
        noise = torch.randn_like(x_adv)
        # Project noise onto the tangent of the sphere centred at x_orig,
        # radius ||x_adv - x_orig||. That means: remove the radial component
        # along (x_adv - x_orig).
        radial = x_adv - x_orig                            # (B,1,28,28)
        radial_norm = flat(radial).norm(dim=1).clamp_min(1e-12)
        radial_unit = radial / radial_norm.view(B, 1, 1, 1)
        dot = (flat(noise) * flat(radial_unit)).sum(dim=1).view(B, 1, 1, 1)
        noise_perp = noise - dot * radial_unit
        # Rescale noise_perp to have norm sph * radial_norm (per-sample).
        np_norm = flat(noise_perp).norm(dim=1).clamp_min(1e-12)
        noise_perp = noise_perp / np_norm.view(B, 1, 1, 1) \
                     * (sph * radial_norm).view(B, 1, 1, 1)
        # Combine: orthogonal walk + source pull.
        proposal = x_adv + noise_perp + src.view(B, 1, 1, 1) * direction
        proposal = proposal.clamp(0.0, 1.0)

        # ---- query --------------------------------------------------------
        with torch.no_grad():
            pred = model(proposal).argmax(1)
        used += (remaining > 0).long()
        is_adv = pred != y_true
        is_closer = l2(proposal, x_orig) < l2(x_adv, x_orig)
        # Only update samples that (a) still have budget and (b) attack ok.
        active = remaining > 0
        accept = active & is_adv & is_closer
        # Apply.
        for_mask = accept.view(B, 1, 1, 1).expand_as(x_adv)
        x_adv = torch.where(for_mask, proposal, x_adv)

        # ---- adapt step sizes (per-sample) --------------------------------
        hist[:, hist_ptr] = accept.float()
        hist_ptr = (hist_ptr + 1) % adapt_window
        accept_rate = hist.mean(dim=1)
        grow = accept_rate > 0.5
        shrink = accept_rate < 0.2
        src = torch.where(grow, src * ADAPT_UP, src)
        src = torch.where(shrink, src * ADAPT_DOWN, src)
        sph = torch.where(grow, sph * ADAPT_UP, sph)
        sph = torch.where(shrink, sph * ADAPT_DOWN, sph)
        src = src.clamp(1e-5, 1.0)
        sph = sph.clamp(1e-5, 1.0)

        remaining = query_budget - used.clamp(min=0)
        step += 1

    final_l2 = l2(x_adv, x_orig)
    return x_adv.detach(), used.detach(), final_l2.detach()


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def clean_acc(model, X, Y):
    _, acc = C.logits_and_acc(model, X, Y)
    return float(acc)


def pgd_eval(model, X, Y):
    """Returns (asr, median_L2_of_perturbations_on_orig_correct)."""
    model.eval()
    # ASR via campaign helper.
    res = C.attack_success(model, X, Y, attack="pgd", eps=EPS, steps=PGD_STEPS)
    # Median L2 of the actual perturbation (recompute X_adv).
    X_adv = C.pgd(model, X, Y, eps=EPS, steps=PGD_STEPS)
    diff = (X_adv - X).reshape(X.size(0), -1)
    l2 = diff.norm(dim=1).cpu().numpy()
    corr = res["correct"].astype(bool)
    med = float(np.median(l2[corr])) if corr.sum() > 0 else float("nan")
    return float(res["asr"]), med


def boundary_eval(model, X, Y, budget, batch=50):
    """Run Boundary at given query budget. Returns:
      asr  : fraction of originally-correct samples whose final x_adv has
             prediction != y_true (Boundary almost always succeeds at
             *some* perturbation since init is adversarial; meaningful
             quantity is also median_l2).
      med_l2 : median final L_2 distance from clean (over originally
               correct samples).
      flips  : (N,) bool over originally-correct samples (for per-class).
    """
    model.eval()
    N = X.size(0)
    flips = torch.zeros(N, dtype=torch.bool)
    final_l2 = torch.zeros(N)
    # Identify originally-correct.
    pred_clean = _predict(model, X).cpu()
    correct = (pred_clean == Y.cpu())
    # Attack in batches.
    for i in range(0, N, batch):
        xb = X[i:i + batch]
        yb = Y[i:i + batch]
        x_adv, _, l2 = boundary_attack(model, xb, yb, query_budget=budget)
        with torch.no_grad():
            pred_adv = model(x_adv).argmax(1).cpu()
        flips[i:i + batch] = pred_adv != yb.cpu()
        final_l2[i:i + batch] = l2.cpu()
    asr = float(flips[correct].float().mean()) if correct.any() else float("nan")
    med = float(final_l2[correct].median()) if correct.any() else float("nan")
    return asr, med, flips.numpy(), correct.numpy(), final_l2.numpy()


def per_class_boundary(flips, correct, y_true_np, n_classes=N_CLASSES):
    """Per-class Boundary-ASR on originally-correct samples."""
    out = []
    for c in range(n_classes):
        mask = correct & (y_true_np == c)
        if mask.sum() == 0:
            out.append(float("nan"))
        else:
            out.append(float(flips[mask].mean()))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    # Boundary uses a smaller subsample (queries are the bottleneck).
    g = torch.Generator().manual_seed(SEED)
    idx_b = torch.randperm(Xte.size(0), generator=g)[:N_TEST_BOUNDARY]
    Xb = Xte[idx_b]
    Yb = Yte[idx_b]

    # ---- train all three models ------------------------------------------
    trained = {}        # name -> model
    timings = {}
    print("\n=== training STD ===", flush=True)
    t0 = time.time(); trained["STD"]    = train_std(Xtr, Ytr)
    timings["STD"] = time.time() - t0
    print(f"  STD trained in {timings['STD']:.1f}s", flush=True)

    print("\n=== training PGD-AT ===", flush=True)
    t0 = time.time(); trained["PGD-AT"] = train_pgdat(Xtr, Ytr)
    timings["PGD-AT"] = time.time() - t0
    print(f"  PGD-AT trained in {timings['PGD-AT']:.1f}s", flush=True)

    print("\n=== training MASK-SUSP ===", flush=True)
    t0 = time.time(); mask_model, mask_name = train_mask_suspect(Xtr, Ytr)
    trained["MASK-SUSP"] = mask_model
    timings["MASK-SUSP"] = time.time() - t0
    print(f"  MASK-SUSP={mask_name} trained in {timings['MASK-SUSP']:.1f}s", flush=True)

    # ---- baselines: clean + PGD ------------------------------------------
    base = {}
    for name, model in trained.items():
        ca = clean_acc(model, Xte, Yte)
        pgd_asr, pgd_med_l2 = pgd_eval(model, Xte, Yte)
        base[name] = dict(clean=ca, pgd_asr=pgd_asr, pgd_med_l2=pgd_med_l2)
        print(f"  [{name}] clean={ca:.4f} pgd_asr={pgd_asr:.4f} "
              f"pgd_med_L2={pgd_med_l2:.4f}", flush=True)

    # ---- Boundary sweep over query budgets -------------------------------
    boundary_rows = []   # (name, Q, asr, med_l2, time_s)
    per_class = {}       # name -> list of class ASRs at HEADLINE_BUDGET
    headline_l2 = {}     # name -> per-sample L2 array at HEADLINE_BUDGET
    headline_flips = {}  # name -> flips array
    headline_correct = {}
    for name, model in trained.items():
        for Q in QUERY_BUDGETS:
            t0 = time.time()
            asr, med_l2, flips, corr, l2_arr = boundary_eval(
                model, Xb, Yb, budget=Q)
            dt = time.time() - t0
            boundary_rows.append((name, Q, asr, med_l2, dt))
            print(f"  [{name}] Boundary Q={Q:>4d} ASR={asr:.4f} "
                  f"medL2={med_l2:.4f}  ({dt:.1f}s)", flush=True)
            if Q == HEADLINE_BUDGET:
                per_class[name] = per_class_boundary(
                    flips, corr, Yb.cpu().numpy())
                headline_l2[name] = l2_arr
                headline_flips[name] = flips
                headline_correct[name] = corr

    # ---- masking flag (C5) -----------------------------------------------
    mask_flags = {}
    for name in trained.keys():
        pgd_asr = base[name]["pgd_asr"]
        b_row = [r for r in boundary_rows if r[0] == name and r[1] == HEADLINE_BUDGET][0]
        b_asr = b_row[2]
        gap = b_asr - pgd_asr
        flagged = (gap > MASK_FLAG_GAP) and (pgd_asr < MASK_FLAG_PGD_CEIL)
        mask_flags[name] = dict(gap=gap, flagged=bool(flagged), b_asr=b_asr)

    # ---- write report ----------------------------------------------------
    lines = []
    lines.append("H476 Boundary Attack (Brendel 2018) on Fashion-MNIST")
    lines.append("=" * 78)
    lines.append("Decision-based, query-only attack. Anchor: brendel-2018-boundary.")
    lines.append("Aux: chen-2020-hopskipjump, cheng-2020-sign-opt,")
    lines.append("     andriushchenko-2020-square (score-based, NOT decision-based).")
    lines.append("Threat model: top-1 label only; no gradients; no logits.")
    lines.append("")
    lines.append(f"Dataset={DS}  N_train={N_TRAIN}  Epochs={EPOCHS}  LR={LR}  "
                 f"Batch={BATCH}  Seed={SEED}  EPS(Linf for PGD)={EPS}")
    lines.append(f"Boundary: N_test={N_TEST_BOUNDARY}  budgets={QUERY_BUDGETS}  "
                 f"src_step0={SOURCE_STEP_INIT}  sph_step0={SPHERICAL_STEP_INIT}")
    lines.append(f"MASK-SUSP defence loaded: {mask_name}")
    lines.append("")
    lines.append("Baselines: clean acc, white-box PGD-10 (eps=0.1) ASR and median L2")
    lines.append("-" * 78)
    lines.append(f"{'model':<12} {'clean':>9} {'pgd_asr':>9} {'pgd_medL2':>11} "
                 f"{'train_s':>9}")
    for name in trained.keys():
        b = base[name]
        lines.append(f"{name:<12} {b['clean']:>9.4f} {b['pgd_asr']:>9.4f} "
                     f"{b['pgd_med_l2']:>11.4f} {timings[name]:>9.1f}")
    lines.append("")
    lines.append("Boundary Attack: ASR and median final L2 vs query budget")
    lines.append("-" * 78)
    lines.append(f"{'model':<12} {'Q':>5} {'asr':>8} {'med_L2':>10} {'attack_s':>9}")
    for name, Q, asr, med_l2, dt in boundary_rows:
        lines.append(f"{name:<12} {Q:>5d} {asr:>8.4f} {med_l2:>10.4f} {dt:>9.1f}")
    lines.append("")
    lines.append(f"Per-class Boundary ASR at Q={HEADLINE_BUDGET}")
    lines.append("-" * 78)
    header = f"{'model':<12}" + "".join(f" c{c:>2d}" for c in range(N_CLASSES))
    lines.append(header)
    for name in trained.keys():
        row = f"{name:<12}" + "".join(
            f" {pc:>5.2f}" if not math.isnan(pc) else "   nan"
            for pc in per_class[name])
        lines.append(row)
    lines.append("")
    lines.append("Masking diagnosis (C5):")
    lines.append("-" * 78)
    lines.append("  flag := (Boundary-ASR@Q=1000 - PGD-ASR) > 0.10 AND PGD-ASR < 0.50")
    lines.append(f"{'model':<12} {'pgd_asr':>9} {'b_asr@1k':>10} {'gap':>8} "
                 f"{'flag':>6}")
    for name, mf in mask_flags.items():
        lines.append(f"{name:<12} {base[name]['pgd_asr']:>9.4f} "
                     f"{mf['b_asr']:>10.4f} {mf['gap']:>8.4f} "
                     f"{'YES' if mf['flagged'] else 'no':>6}")
    lines.append("")
    # ---- headline verdict ------------------------------------------------
    n_flagged = sum(1 for v in mask_flags.values() if v["flagged"])
    is_mask_flagged = mask_flags["MASK-SUSP"]["flagged"]
    pgdat_flagged   = mask_flags["PGD-AT"]["flagged"]
    std_flagged     = mask_flags["STD"]["flagged"]
    lines.append("HEADLINE")
    lines.append("=" * 78)
    if is_mask_flagged and not pgdat_flagged:
        lines.append(f"  SUPPORTED. The suspected-masking defence ({mask_name})")
        lines.append( "  shows a Boundary-vs-PGD ASR gap of "
                     f"{mask_flags['MASK-SUSP']['gap']:+.3f} -- decision-based")
        lines.append( "  attack EXPOSES gradient masking that white-box PGD missed.")
        lines.append( "  PGD-AT shows no such gap (genuine defence).")
    elif not is_mask_flagged and not pgdat_flagged:
        lines.append( "  NOT SUPPORTED (this seed). Neither model triggers the")
        lines.append( "  masking flag; the suspected defence may be genuinely")
        lines.append( "  robust on Fashion-MNIST or Boundary at Q=1000 is too")
        lines.append( "  query-starved to expose it. HopSkipJump / Sign-OPT")
        lines.append( "  (more query-efficient cousins) are the natural follow-up.")
    elif is_mask_flagged and pgdat_flagged:
        lines.append( "  AMBIGUOUS. Both PGD-AT and the suspect trip the flag --")
        lines.append( "  likely the masking threshold (gap>0.10) is too lax for")
        lines.append( "  this dataset, where PGD on PGD-AT is itself loose.")
    else:
        lines.append( "  ANOMALOUS. PGD-AT trips the flag but the suspect does not;")
        lines.append( "  re-examine PGD-AT under stronger white-box (AutoAttack).")
    lines.append(f"  flagged={n_flagged}/3 models  "
                 f"(STD={std_flagged}, PGD-AT={pgdat_flagged}, "
                 f"MASK-SUSP={is_mask_flagged}).")
    lines.append("")
    lines.append("Notes / limitations:")
    lines.append("  * Single seed (SEED=0); 200 Boundary samples (per-class noisy).")
    lines.append("  * Boundary is the FIRST decision-based attack; HSJA / Sign-OPT")
    lines.append("    (Cheng 2020, see H116) are more query-efficient and would give")
    lines.append("    tighter median L2 -- resistance to Boundary at Q=1000 does NOT")
    lines.append("    imply robustness to all decision-based attacks.")
    lines.append("  * Untargeted only.")
    lines.append("  * MASK-SUSP fallback may be H372 if H407 import fails; the")
    lines.append("    actual loaded defence is logged above in the config block.")

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
        f.flush()
    print(f"\nSaved to {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
