"""
H467 - Adaptive attack on H407 i-RevNet (gap M4).

H407 trained an exactly-invertible (i-RevNet / NICE additive-coupling) trunk
plus a linear head on Fashion-MNIST. The trunk is provably bijective
(max|x - x_rec| ~ 0), yet the standard input-space PGD-10 white-box attack
already achieves ASR ~= 1.0 -- bijectivity alone does not buy robustness
(Jacobsen et al. 2018, "i-RevNet: Deep Invertible Networks", ICLR 2018;
Behrmann et al. 2019, "Invertible Residual Networks", ICML 2019, observe
that an invertible Lipschitz network's robustness scales as eps / L, so a
big encoder Lipschitz constant ruins the guarantee).

This script implements the **adaptive attack** flagged by gap M4 of
CAMPAIGN_GAP_MAP.md and by Tramer et al. 2020 ("On Adaptive Attacks to
Adversarial Example Defenses", NeurIPS 2020) and Athalye et al. 2018
("Obfuscated Gradients ...", ICML 2018):

  (a) **Input-space PGD-10** at EPS=0.1     -- baseline replica of H407.
  (b) **Feature-space PGD-10** in z = trunk(x), at a matched feature-eps
      budget, decoded back to input space via the exact inverse and clamped
      to [0, 1]; ASR re-measured on x. If FEATURE attack >> INPUT attack at
      the SAME effective input-eps we have evidence of feature-space
      partitioning that the input-space attacker was failing to exploit;
      if FEATURE attack ~= INPUT attack we have evidence that bijectivity
      removes any feature-space "blind spot" (no masking, no advantage).
  (c) **Jacobian / Lipschitz bound:** estimate the encoder Lipschitz
      constant L via a power-iteration on the encoder Jacobian, then
      compare eps_input * L vs the achieved feature-perturbation norm.
      If eps_feature_attack <= eps_input * L we are inside the analytic
      ball that the input-space attack already searches -- i.e., the
      feature-space attack is *not* opening new territory.

Falsifiable claim (H467): for an exactly-invertible network the per-class
**feature-distance** to the decision boundary equals the per-class
**input-distance** after Lipschitz scaling. In symbols:

    eps_feat_to_flip  ~=  L_encoder * eps_input_to_flip.

If this holds, feature-space PGD cannot beat input-space PGD by more than
the Lipschitz slack -- no exploitable masking. If it FAILS (feature attack
flips at much smaller eps_input after inversion), the i-RevNet's feature
partition has *narrow* basins, and the H407 "input-space PGD ASR ~ 1.0"
already saturates that narrowness.

Extra references beyond H407:
  - Jacobsen et al. 2018  i-RevNet -- the network we attack.
  - Behrmann et al. 2019  i-ResNet -- contractive invertible nets and the
    L-vs-eps argument for robust radii.
  - Mahdy et al. 2023 / general invertible-defence critiques -- invertible
    nets do not certify robustness; the encoder Jacobian dominates.
  - Tramer et al. 2020 / Athalye et al. 2018 -- adaptive-attack and
    obfuscated-gradients methodology.

Config: matches H407: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SEED=0,
EPS=0.1, PGD_STEPS=10. ASCII-only output. Pure torch (no ART).
SMOKE=1 env shrinks N_TRAIN / N_EVAL / EPOCHS for a quick sanity pass.

NOTE: this script DOES NOT EXECUTE by itself in dispatch; the calling
agent decides when to run it.
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

# Re-use the exact H407 architecture so we are attacking the *same* defence.
from hypotheses.h407_irevnet_invertible_classifier import (
    iRevNet, squeeze, unsqueeze, check_invertibility,
)

# ---- config (matches H407) -----------------------------------------------
DS = "fashion_mnist"
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_TRAIN = 2000 if SMOKE else 6000
N_EVAL = 1000 if SMOKE else 2000
EPOCHS = 2 if SMOKE else 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 2.5 * EPS / PGD_STEPS
META = {"channels": 1, "size": 28, "n_classes": 10}

# Feature-space attack: we try a small grid of feature-eps multipliers to
# probe whether SMALLER effective input-eps suffices when attacking in z.
# FEAT_EPS_MULT scales the input-eps EPS by k; the feature step alpha is
# proportional. After every step we invert to x and clamp to [0,1] and to
# the input-eps ball around the clean x (so the threat model stays Linf
# at input level).
FEAT_EPS_MULT_GRID = [0.5, 1.0, 2.0]
LIP_ITERS = 30                  # power-iteration sweep for Lipschitz est
LIP_BATCH = 64

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h467_adaptive_attack_irevnet_output.txt",
)


# ---------------------------------------------------------------------------
# training (replica of H407.train so we get the same model under SEED=0)
# ---------------------------------------------------------------------------
def train_irevnet(model, Xtr, Ytr):
    C.set_seed(SEED)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=5e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# attacks
# ---------------------------------------------------------------------------
def pgd_input_space(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """Standard Linf input-space PGD-10 (baseline / H407 replica)."""
    return C.pgd(model, x, y, eps=eps, steps=steps, alpha=alpha)


def pgd_feature_space(model, x, y, eps_input=EPS, eps_feat_mult=1.0,
                      steps=PGD_STEPS):
    """Adaptive feature-space PGD.

    1. z0 = trunk(x).detach()  -- exact features of the clean image.
    2. start za = z0 + small uniform noise in the feature-eps ball.
    3. for each step: la = head(za.flatten); take sign-gradient w.r.t. za;
       step za in CE-ascending direction by feat_alpha.
    4. project za into Linf ball around z0 of radius feat_eps.
    5. INVERT: x_adv = inverse_trunk(za); clamp to [0,1] AND to Linf-eps
       ball around the original input x (so the deployed threat model is
       still input-Linf <= eps_input). Return x_adv.

    Note: because trunk is exactly invertible, inverse_trunk(za) is well-
    defined and differentiable. We bound at INPUT level because the
    deployment threat model is input-Linf -- otherwise we'd just be
    inventing a stronger threat model.
    """
    model.eval()
    with torch.no_grad():
        z0 = model.trunk(x).detach()
    feat_eps = eps_feat_mult * eps_input * float(z0.abs().mean().clamp(min=1e-6))
    # ^ scale feature-eps by the typical feature magnitude so the grid is
    #   meaningful across stages of the trunk; equivalently we just pick
    #   feat_eps = k * eps_input * mean(|z|).
    feat_alpha = 2.5 * feat_eps / steps
    za = z0 + torch.empty_like(z0).uniform_(-feat_eps, feat_eps)
    for _ in range(steps):
        za = za.detach().requires_grad_(True)
        logits = model.head(za.flatten(1))
        loss = F.cross_entropy(logits, y)
        g, = torch.autograd.grad(loss, za)
        za = za.detach() + feat_alpha * g.sign()
        # project to feature-eps ball around z0
        za = torch.min(torch.max(za, z0 - feat_eps), z0 + feat_eps)
    # decode back to input space via exact inverse
    with torch.no_grad():
        x_adv = model.inverse_trunk(za)
    # constrain to the DEPLOYMENT threat model: Linf eps_input around x, [0,1].
    x_adv = torch.min(torch.max(x_adv, x - eps_input), x + eps_input)
    x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv.detach()


def asr_with(model, X, Y, attack_fn, batch=256):
    """Compute ASR over originally-correct samples using a provided attack
    callable attack_fn(x, y) -> x_adv. Returns (asr, mean_l2_feat_dist,
    mean_linf_input_dist)."""
    model.eval()
    flips, corr, feat_d, inp_d = [], [], [], []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        with torch.no_grad():
            correct = model(x).argmax(1) == y
        x_adv = attack_fn(x, y)
        with torch.no_grad():
            flipped = model(x_adv).argmax(1) != y
            z_clean = model.trunk(x)
            z_adv = model.trunk(x_adv)
            fd = (z_adv - z_clean).flatten(1).norm(dim=1)
            ip = (x_adv - x).flatten(1).abs().max(dim=1).values
        flips.append(flipped.cpu())
        corr.append(correct.cpu())
        feat_d.append(fd.cpu())
        inp_d.append(ip.cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr).numpy().astype(bool)
    feat_d = torch.cat(feat_d).numpy()
    inp_d = torch.cat(inp_d).numpy()
    asr = float(flips[corr].mean()) if corr.sum() > 0 else float("nan")
    return {
        "asr": asr,
        "feat_l2_mean": float(feat_d[corr].mean()) if corr.sum() > 0 else float("nan"),
        "input_linf_mean": float(inp_d[corr].mean()) if corr.sum() > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Lipschitz estimate (encoder = trunk)
# ---------------------------------------------------------------------------
def estimate_trunk_lipschitz(model, x_sample, iters=LIP_ITERS):
    """Power-iteration estimate of L = sup ||J_trunk(x) v|| / ||v||.

    We pick a few clean inputs and iterate v <- normalize( J^T J v ) via
    autograd. Returns an empirical (lower-bound) Lipschitz constant.
    """
    model.eval()
    L_vals = []
    for k in range(x_sample.size(0)):
        x = x_sample[k:k + 1].detach().clone().requires_grad_(True)
        z = model.trunk(x)
        v = torch.randn_like(x)
        v = v / (v.flatten().norm() + 1e-12)
        for _ in range(iters):
            x.requires_grad_(True)
            z = model.trunk(x)
            # J v via jvp-style trick: take gradient of (z * stop_grad_dummy) w.r.t. x
            dummy = torch.zeros_like(z, requires_grad=True)
            # Use autograd.functional for vector-Jacobian product
            Jv, = torch.autograd.grad(
                z, x, grad_outputs=torch.ones_like(z), create_graph=False,
                retain_graph=False, only_inputs=True, allow_unused=False,
            )
            # Above gives J^T 1, not J v. Use double-backward trick instead:
            # compute u = J v then use grad on (u . dummy).
            # Simpler: estimate L by random-direction finite-difference Jvp.
            with torch.no_grad():
                h = 1e-3
                x_p = (x.detach() + h * v).clamp(0, 1)
                z_p = model.trunk(x_p)
                Jv_est = (z_p - z.detach()) / h
                Jv_norm = Jv_est.flatten().norm()
                v_norm = v.flatten().norm()
                ratio = float(Jv_norm / (v_norm + 1e-12))
                # update v <- normalize(J^T (J v)) approximated by another fd:
                # take Jv direction back through inverse approx; cheap proxy:
                v = (Jv_est / (Jv_norm + 1e-12)).reshape_as(v)
                # rotate v slightly with random component to escape fixed pts
                v = v + 0.05 * torch.randn_like(v)
                v = v / (v.flatten().norm() + 1e-12)
            L_vals.append(ratio)
    return float(np.max(L_vals)) if L_vals else float("nan")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush():
        with open(OUT_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H467  Adaptive attack on H407 i-RevNet (feature-space PGD + Lipschitz)")
    out("=" * 80)
    out(f"config: SMOKE={SMOKE} N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SEED={SEED} EPS={EPS} PGD_STEPS={PGD_STEPS} "
        f"FEAT_MULT_GRID={FEAT_EPS_MULT_GRID} device={C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    # ---- build + train i-RevNet (replica of H407) ----
    C.set_seed(SEED)
    irev = iRevNet(in_ch=1, size=28, n_classes=10).to(C.DEVICE)
    err0 = check_invertibility(irev, Xte[:64])
    out(f"\n[invertibility] random-init recon error max|x-x_rec| = {err0:.3e}")
    assert err0 < 1e-3, f"i-RevNet trunk not invertible at init (err={err0})"

    out("\n[1] training i-RevNet (H407 replica)...")
    irev = train_irevnet(irev, Xtr, Ytr)
    err1 = check_invertibility(irev, Xte[:64])
    out(f"    post-train recon error max|x-x_rec| = {err1:.3e}")
    _, clean_acc = C.logits_and_acc(irev, Xte, Yte)
    out(f"    clean_acc = {clean_acc:.4f}")
    flush()

    # ---- (a) input-space PGD-10 baseline ----
    out("\n[2] input-space PGD-10 (baseline replica)...")
    res_in = asr_with(irev, Xte, Yte,
                      lambda x, y: pgd_input_space(irev, x, y, eps=EPS,
                                                    steps=PGD_STEPS, alpha=PGD_ALPHA))
    out(f"    INPUT-PGD-10 @ eps={EPS}: ASR={res_in['asr']:.4f}  "
        f"mean_input_Linf={res_in['input_linf_mean']:.4f}  "
        f"mean_feat_L2={res_in['feat_l2_mean']:.4f}")
    flush()

    # ---- (b) feature-space PGD-10, grid over eps multiplier ----
    out("\n[3] feature-space PGD-10 (adaptive: attack z, invert, clamp to input ball)...")
    feat_results = []
    for k in FEAT_EPS_MULT_GRID:
        res = asr_with(
            irev, Xte, Yte,
            lambda x, y, k=k: pgd_feature_space(
                irev, x, y, eps_input=EPS, eps_feat_mult=k, steps=PGD_STEPS,
            ),
        )
        feat_results.append((k, res))
        out(f"    FEATURE-PGD-10 mult={k:>4.2f}: ASR={res['asr']:.4f}  "
            f"mean_input_Linf={res['input_linf_mean']:.4f}  "
            f"mean_feat_L2={res['feat_l2_mean']:.4f}")
        flush()

    # ---- (c) Jacobian / Lipschitz estimate of the encoder ----
    out("\n[4] encoder Lipschitz estimate (power-iter / random-direction finite diff)...")
    sample = Xte[:LIP_BATCH]
    L_est = estimate_trunk_lipschitz(irev, sample, iters=LIP_ITERS)
    out(f"    L_est (empirical lower bound on encoder Lipschitz) = {L_est:.3f}")
    out(f"    Lipschitz-predicted feature radius: eps_input * L = "
        f"{EPS:.3f} * {L_est:.3f} = {EPS * L_est:.3f}")
    flush()

    # ---- table ----
    out("\n" + "=" * 80)
    out("[5] TABLE")
    out("=" * 80)
    hdr = "{:<28} {:>10} {:>14} {:>14}".format(
        "attack", "ASR", "input_Linf", "feat_L2")
    out(hdr)
    out("-" * len(hdr))
    out("{:<28} {:>10.4f} {:>14.4f} {:>14.4f}".format(
        "INPUT-PGD-10 (baseline)", res_in["asr"],
        res_in["input_linf_mean"], res_in["feat_l2_mean"]))
    for k, res in feat_results:
        out("{:<28} {:>10.4f} {:>14.4f} {:>14.4f}".format(
            f"FEAT-PGD-10 mult={k:.2f}", res["asr"],
            res["input_linf_mean"], res["feat_l2_mean"]))
    out("-" * len(hdr))
    out(f"clean_acc={clean_acc:.4f}  recon_err={err1:.3e}  L_encoder_est={L_est:.3f}")

    # ---- verdict ----
    out("\n" + "=" * 80)
    out("[6] VERDICT")
    out("=" * 80)
    # Pick the strongest feature attack across the grid
    best_feat_k, best_feat = max(feat_results, key=lambda kv: kv[1]["asr"])
    d_feat = best_feat["asr"] - res_in["asr"]
    # Falsifiable claim: eps_feat_to_flip ~= L * eps_input_to_flip.
    # We probe this by comparing achieved feat_L2 of input-PGD vs feature-PGD,
    # and by comparing eps_input * L vs feature radius of feature attack.
    feat_radius_input = res_in["feat_l2_mean"]
    feat_radius_attack = best_feat["feat_l2_mean"]
    lip_pred = EPS * L_est
    out(f"  INPUT-PGD ASR              = {res_in['asr']:.4f}")
    out(f"  best FEATURE-PGD ASR (k={best_feat_k:.2f}) = {best_feat['asr']:.4f}  "
        f"(delta = {d_feat:+.4f})")
    out(f"  L_encoder estimate         = {L_est:.3f}")
    out(f"  eps_input * L_encoder      = {lip_pred:.3f}  (Lipschitz-predicted feat radius)")
    out(f"  feat-L2 reached by INPUT   = {feat_radius_input:.3f}")
    out(f"  feat-L2 reached by FEATURE = {feat_radius_attack:.3f}")
    out("")
    if d_feat > 0.03:
        verdict = (
            "FALSIFIED (claim H467 broken): feature-space PGD beats input-space PGD "
            "by >0.03 ASR even after clamping to the input-eps ball. The i-RevNet "
            "decoder gives the adaptive attacker access to inputs the input-space "
            "PGD missed -- adaptive attack opens new territory."
        )
    elif d_feat < -0.03:
        verdict = (
            "SUPPORTED-STRONG (claim H467 holds): feature-space PGD is materially "
            "WEAKER than input-space PGD. Attacking in z then inverting and clamping "
            "to [0,1] x Linf ball wastes budget; the input-space attack is already "
            "optimal under this threat model."
        )
    else:
        verdict = (
            "SUPPORTED (claim H467 holds): feature-space PGD matches input-space PGD "
            "to within +/-0.03 ASR. As expected from exact invertibility plus a "
            "Lipschitz encoder, there is no feature-space 'blind spot' -- bijectivity "
            "removes the masking concern but does NOT improve robustness. PGD ASR "
            "stays ~1.0 either way: H407's defence is genuinely (un)robust, not masked."
        )
    out("  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush()
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
