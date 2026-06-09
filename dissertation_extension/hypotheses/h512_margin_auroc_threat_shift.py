"""
H512 — Margin-AUROC degradation under threat-model shift (§5 / §2 M12).

Hypothesis
----------
For an Linf-PGD-AT (eps=0.1) SmallCNN on Fashion-MNIST, the per-sample
Linf-PGD margin (computed under the *training* threat model, Linf eps=0.1)
strongly discriminates "will-flip vs won't-flip" *within that same threat*
(AUROC > 0.8), but its discrimination collapses (AUROC < 0.6) when the
target threat is shifted to L2-PGD (eps=2.0), StAdv spatial (tau=0.1), or
FGSM (eps=0.05).  I.e., margin under one threat does NOT generalise as a
flip-predictor across threats.

Critique (advisor's seed): same-threat margin-AUROC is well-studied
(Madry et al. 2018 implicitly; Carlini & Wagner 2017 explicitly use margin
as a confidence proxy); the *cross-threat* generalisation is the gap this
hypothesis targets.

Literature anchors
------------------
* Croce & Hein, "Reliable Evaluation of Adversarial Robustness with an
  Ensemble of Diverse Parameter-free Attacks" (AutoAttack), ICML 2020 —
  per-threat evaluation matters; a model robust on one threat may be far
  weaker on another.
* Mao et al., "Metric Learning for Adversarial Robustness", NeurIPS 2019 —
  uses margins/feature-space distances as a robustness signal; shows that
  the geometry differs across attack types.
* Kang et al., "Testing Robustness Against Unforeseen Adversaries", 2019
  (arXiv:1908.08016) — robustness to one threat model does NOT transfer
  reliably to unseen threats (Lp -> non-Lp gap).  Direct motivation for
  expecting margin-AUROC to drop on StAdv.

Design
------
1. Train PGD-AT SmallCNN on F-MNIST (Linf eps=0.1, ~7 PGD steps).
2. Compute per-sample Linf-PGD margin on 1000 test samples (the "predictor").
3. For each target threat T in {Linf eps=0.1, L2 eps=2.0, StAdv tau=0.1,
   FGSM eps=0.05}, compute flip-status under T and report:
     - AUROC(-margin -> flip)            (use -margin so larger score
                                          means "more likely to flip")
     - per-class AUROC
4. HEADLINE verdict:
     SUPPORTED   if  AUROC_Linf > 0.8 AND AUROC_L2,StAdv each < 0.6
     PARTIAL     if  AUROC_Linf > 0.8 AND at least one cross-threat < 0.6
     REFUTED     otherwise
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from campaign.common import (
    DEVICE, set_seed, load_dataset, dataset_meta, build_model,
    train_model, pgd, fgsm, logits_and_acc, margin_of, safe_auroc,
)


# ---------------------------------------------------------------------------
# attacks beyond Linf-PGD / FGSM (already in common.py)
# ---------------------------------------------------------------------------
def pgd_l2(model, x, y, eps=2.0, steps=20, alpha=None, random_start=True):
    """L2-PGD on inputs in [0,1].  eps is the L2 ball radius across the
    whole image (not per pixel)."""
    if alpha is None:
        alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        noise = torch.randn_like(xa)
        nflat = noise.flatten(1)
        nnorm = nflat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        u = torch.rand(xa.size(0), 1, device=xa.device).pow(
            1.0 / float(xa[0].numel())
        )
        nflat = nflat / nnorm * eps * u
        noise = nflat.view_as(xa)
        xa = (xa + noise).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        gflat = g.flatten(1)
        gnorm = gflat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        step = (gflat / gnorm).view_as(g) * alpha
        xa = xa.detach() + step
        # project to L2 ball around x0
        delta = (xa - x0).flatten(1)
        dnorm = delta.norm(dim=1, keepdim=True)
        factor = torch.clamp(eps / dnorm.clamp_min(1e-12), max=1.0)
        delta = (delta * factor).view_as(x0)
        xa = (x0 + delta).clamp(0, 1)
    return xa.detach()


def stadv_attack(model, x, y, tau=0.1, steps=20, lr=0.05):
    """Spatially-transformed adversarial example (Xiao et al. 2018).

    Optimise a per-pixel flow field f in R^{H,W,2} that samples the
    original image at (i + f_y, j + f_x).  Regularise the flow's total
    variation; bound it by tau (Linf on the flow).  We use a simple
    Adam-style update + projection.  Returns the warped image.
    """
    B, C, H, W = x.shape
    # base coordinate grid in [-1,1] (align_corners=True convention)
    ys = torch.linspace(-1, 1, H, device=x.device)
    xs = torch.linspace(-1, 1, W, device=x.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], dim=-1)[None].expand(B, -1, -1, -1).contiguous()
    flow = torch.zeros(B, H, W, 2, device=x.device, requires_grad=True)
    opt = torch.optim.Adam([flow], lr=lr)
    for _ in range(steps):
        # convert flow (in pixels) to grid offsets in [-1,1] units
        # 2 / (H-1) pixels per grid unit
        gx_off = flow[..., 0] * (2.0 / max(W - 1, 1))
        gy_off = flow[..., 1] * (2.0 / max(H - 1, 1))
        grid = base.clone()
        grid[..., 0] = grid[..., 0] + gx_off
        grid[..., 1] = grid[..., 1] + gy_off
        xa = F.grid_sample(x, grid, mode="bilinear",
                           padding_mode="border", align_corners=True)
        loss = -F.cross_entropy(model(xa), y)
        # small TV penalty to keep the flow smooth
        tv = (flow[:, 1:, :, :] - flow[:, :-1, :, :]).abs().mean() \
             + (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
        loss = loss + 1e-3 * tv
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            flow.clamp_(-tau, tau)
    with torch.no_grad():
        gx_off = flow[..., 0] * (2.0 / max(W - 1, 1))
        gy_off = flow[..., 1] * (2.0 / max(H - 1, 1))
        grid = base.clone()
        grid[..., 0] = grid[..., 0] + gx_off
        grid[..., 1] = grid[..., 1] + gy_off
        xa = F.grid_sample(x, grid, mode="bilinear",
                           padding_mode="border", align_corners=True)
    return xa.detach()


# ---------------------------------------------------------------------------
# per-sample helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, X, batch=256):
    parts = []
    for i in range(0, X.size(0), batch):
        parts.append(model(X[i:i + batch]).argmax(1).cpu())
    return torch.cat(parts)


def flip_vector(model, X, Y, attack_fn, batch=128):
    """Return numpy bool array: did adv example flip the prediction
    (away from the true label)?"""
    out = []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        xa = attack_fn(model, x, y)
        with torch.no_grad():
            pred = model(xa).argmax(1)
        out.append((pred != y).cpu().numpy())
    return np.concatenate(out).astype(bool)


def linf_pgd_margin(model, X, Y, eps=0.1, steps=10, batch=128):
    """Per-sample margin AT the adversarial point under Linf-PGD."""
    margins = []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        xa = pgd(model, x, y, eps=eps, steps=steps)
        with torch.no_grad():
            logits = model(xa).cpu()
        margins.append(margin_of(logits, y.cpu()))
    return np.concatenate(margins)


# ---------------------------------------------------------------------------
# main experiment
# ---------------------------------------------------------------------------
def main():
    out_dir = os.path.join(_ROOT, "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h512_margin_auroc_threat_shift_output.txt")
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(str(s))

    SEED = 0
    set_seed(SEED)
    t0 = time.time()

    log("=" * 78)
    log("H512  Margin-AUROC under threat-model shift  (Linf-AT SmallCNN on F-MNIST)")
    log("=" * 78)
    log("Hypothesis: Linf-PGD margin AUROC > 0.8 for predicting Linf-flip but")
    log("            < 0.6 for L2 / StAdv / FGSM flip on the same model.")
    log("Refs: Croce&Hein 2020 (AutoAttack); Mao et al. 2019 (metric AT);")
    log("      Kang et al. 2019 (unforeseen adversaries).")
    log("")

    # ----- data + model ----------------------------------------------------
    meta = dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = load_dataset("fashion_mnist", n_train=8000,
                                       n_eval=2000, seed=SEED)
    log(f"[data] train={tuple(Xtr.shape)}  test={tuple(Xte.shape)}")

    model = build_model("cnn", meta, width=32, act="relu", bn=True)
    log("[train] PGD-AT  Linf eps=0.1  steps=7  epochs=8  opt=sgd lr=0.05")
    train_model(model, Xtr, Ytr, epochs=8, batch=128, opt="sgd", lr=0.05,
                adv_train=True, adv_eps=0.1, adv_steps=7)

    # clean acc + Linf-PGD acc as sanity checks
    _, clean_acc = logits_and_acc(model, Xte, Yte)
    log(f"[sanity] clean test acc = {clean_acc:.4f}")

    # ----- pick 1000 test samples, restrict to originally-correct ones -----
    N = 1000
    idx = torch.randperm(Xte.size(0), device=Xte.device,
                         generator=torch.Generator(device=Xte.device).manual_seed(SEED))[:N]
    X, Y = Xte[idx], Yte[idx]
    pred_clean = predict(model, X)
    correct_mask = (pred_clean == Y.cpu()).numpy().astype(bool)
    log(f"[eval] using N={N} test samples, originally correct = {int(correct_mask.sum())}")

    # ----- predictor: Linf-PGD margin (under the *training* threat) --------
    log("[predictor] computing Linf-PGD margin  (eps=0.1, steps=10)")
    margins = linf_pgd_margin(model, X, Y, eps=0.1, steps=10)
    # score = -margin so that HIGH score means "expected to flip"
    score = -margins

    # ----- flip-labels under each target threat ----------------------------
    threats = {
        "Linf_eps0.1": (lambda m, x, y: pgd(m, x, y, eps=0.1, steps=20),
                        "Linf-PGD eps=0.1 (same as training)"),
        "L2_eps2.0":   (lambda m, x, y: pgd_l2(m, x, y, eps=2.0, steps=20),
                        "L2-PGD eps=2.0 (cross-threat)"),
        "StAdv_tau0.1": (lambda m, x, y: stadv_attack(m, x, y, tau=0.1, steps=20),
                         "StAdv spatial tau=0.1 (cross-threat / non-Lp)"),
        "FGSM_eps0.05": (lambda m, x, y: fgsm(m, x, y, eps=0.05),
                         "FGSM eps=0.05 (weaker Linf, single-step)"),
    }

    results = {}
    log("")
    log("[threats] computing flip-status per target threat ...")
    for name, (atk, desc) in threats.items():
        flips = flip_vector(model, X, Y, atk)
        # restrict AUROC computation to originally-correct samples
        mask = correct_mask
        asr = float(flips[mask].mean()) if mask.sum() > 0 else float("nan")
        auc = safe_auroc(flips[mask], score[mask])

        # per-class AUROC (restrict to correct mask AND class c)
        Y_np = Y.cpu().numpy()
        per_class = {}
        for c in range(meta["n_classes"]):
            m_c = mask & (Y_np == c)
            if m_c.sum() >= 10 and flips[m_c].min() != flips[m_c].max():
                per_class[c] = safe_auroc(flips[m_c], score[m_c])
            else:
                per_class[c] = float("nan")

        results[name] = {"desc": desc, "asr": asr, "auroc": auc,
                         "per_class": per_class}
        log(f"  - {name:14s} ASR={asr:.3f}  AUROC={auc:.3f}   ({desc})")

    # ----- per-class table -------------------------------------------------
    log("")
    log("[per-class AUROC]   (NaN = degenerate label split or n<10)")
    header = "  cls | " + " | ".join(f"{n:>14s}" for n in threats.keys())
    log(header)
    log("  " + "-" * (len(header) - 2))
    for c in range(meta["n_classes"]):
        row = f"  {c:>3d} | " + " | ".join(
            f"{results[n]['per_class'][c]:>14.3f}" if not np.isnan(results[n]['per_class'][c]) else f"{'NaN':>14s}"
            for n in threats.keys()
        )
        log(row)

    # ----- HEADLINE verdict ------------------------------------------------
    auc_linf = results["Linf_eps0.1"]["auroc"]
    auc_l2   = results["L2_eps2.0"]["auroc"]
    auc_sta  = results["StAdv_tau0.1"]["auroc"]
    auc_fgs  = results["FGSM_eps0.05"]["auroc"]

    same_high = (not np.isnan(auc_linf)) and auc_linf > 0.8
    cross_low_l2 = (not np.isnan(auc_l2)) and auc_l2 < 0.6
    cross_low_sta = (not np.isnan(auc_sta)) and auc_sta < 0.6

    if same_high and cross_low_l2 and cross_low_sta:
        verdict = "SUPPORTED"
    elif same_high and (cross_low_l2 or cross_low_sta):
        verdict = "PARTIAL"
    else:
        verdict = "REFUTED"

    log("")
    log("=" * 78)
    log(f"HEADLINE verdict: {verdict}")
    log(f"  AUROC(Linf same-threat)       = {auc_linf:.3f}   (claim: > 0.8)")
    log(f"  AUROC(L2  cross-threat)       = {auc_l2:.3f}   (claim: < 0.6)")
    log(f"  AUROC(StAdv cross-threat)     = {auc_sta:.3f}   (claim: < 0.6)")
    log(f"  AUROC(FGSM weaker Linf)       = {auc_fgs:.3f}   (no a-priori claim)")
    log("=" * 78)
    log(f"[time] {time.time() - t0:.1f}s")

    with open(out_path, "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
