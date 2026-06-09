"""
H472 - Spatial / StAdv adversarial attack: flow-budget sweep across defences.

Gap: G1 (threat-model coverage). Anchor paper: Xiao et al. 2018 (ICLR),
"Spatially Transformed Adversarial Examples" (arXiv:1801.02612).
Extra references:
  - Engstrom et al. 2019 (ICML), "Exploring the Landscape of Spatial
    Robustness" (arXiv:1712.02779) - shows first-order methods are weak
    on spatial perturbations; rotation+translation alone fools many CNNs;
    spatial-AT does not transfer to Linf-AT and vice versa.
  - Yang et al. 2020 / Kang et al. 2019 "Testing Robustness Against
    Unforeseen Adversaries" - cross-norm transfer is weak; Linf-AT models
    are not robust under spatial/StAdv attacks.
  - Athalye et al. 2018 (Obfuscated Gradients) - reminds us white-box
    attacks must directly target the defence; bilinear-resample StAdv is
    fully differentiable, so no BPDA tricks are needed here.

Critique / design rationale:
  StAdv perturbs an image by learning a per-pixel 2-D displacement field
  and resampling bilinearly. The threat model is *not* Linf-bounded in
  pixel-intensity space: a small flow can produce a large Linf distance
  while remaining perceptually realistic. The campaign H173-H413 has
  evaluated >100 defences under Linf-eps=0.1 PGD-10 only; H62 implemented
  StAdv as a *vulnerability probe* but never as a cross-defence eval.
  Concretely we expect:
    - Linf-AT (eps=0.1 and large-eps eps=0.2) defenses are tuned to
      pixel-intensity perturbations. There is no a-priori reason they
      generalise to spatial warps; in CIFAR-10 literature (Engstrom 2019,
      Kang 2019) cross-norm/cross-threat transfer is poor.
    - FGSM-AT and PGD-AT should give modest spatial robustness because
      bilinear interpolation produces some pixel-intensity perturbation
      as a side effect, but the per-pixel flow degree of freedom is much
      richer than an Linf ball.
    - TRADES has the same pixel-Linf inner attack as PGD-AT so should
      behave similarly under StAdv.
    - A meaningful "flow-budget" sweep (tau in {0.05, 0.10, 0.20}) lets
      us see whether spatial robustness is monotonic in attack budget.
  Hypothesis: at every flow budget tau, ALL Linf-AT-trained defences
  show StAdv ASR within +/- 5 pp of the undefended (CE) baseline. That
  would falsify any claim that "AT gives general adversarial robustness".

Protocol (campaign standard):
  Dataset       : Fashion-MNIST
  N_train       : 6000
  Epochs        : 10
  Optimizer     : SGD, lr=0.05, momentum=0.9, weight_decay=5e-4
  Batch         : 128
  Seed          : 0
  Linf eps      : 0.1 (for FGSM-AT, PGD-AT, TRADES); 0.2 for large-eps AT.
  PGD inner     : 10 steps, alpha=0.01 (training); same for white-box PGD eval.

Defences trained (five):
  CE            : standard cross-entropy, no AT.
  FGSM-AT       : single-step FGSM perturbations at eps=0.1.
  PGD-AT        : 10-step PGD perturbations at eps=0.1.
  TRADES        : beta=6 KL-trades inner step at eps=0.1.
  Linf-AT-large : 10-step PGD perturbations at eps=0.2.

Evaluation suite per defence:
  - Clean acc
  - FGSM ASR (eps=0.1)
  - PGD-10 ASR (eps=0.1)             [validates AT defences work as advertised]
  - StAdv ASR at tau in {0.05, 0.10, 0.20}
  - Mean flow magnitude on flipped samples
  - Mean Linf and L2 of (x_adv - x) after StAdv resampling (informational)

StAdv attack details (pure torch, no ART):
  - Learnable flow field f in R^{N, 2, H, W} (initialised to 0).
  - Bilinear resample via F.grid_sample (align_corners=True), padding=border.
  - Adam(lr=0.05) on the flow, 100 iterations.
  - Inner objective: CW-margin loss (push true logit below max-other by kappa)
                    + tau_smooth * total-variation on the flow field.
  - PGD-style projection: after each Adam step, clamp per-pixel flow magnitude
    to <= tau (in pixel units). This is the flow-budget knob.
  - Final adversarial image clamped to [0,1] (no out-of-range pixels).

Output: results/fashion_mnist/h472_stadv_spatial_attack_output.txt
Verdict written into the output file.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C


# ---- standard config -------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
EPS_LARGE = 0.2
PGD_STEPS = 10
PGD_ALPHA = 0.01
N_EVAL = 2000           # campaign default (matches load_dataset n_eval)

# ---- StAdv config ----------------------------------------------------------
STADV_STEPS = 100
STADV_LR = 0.05
STADV_SMOOTH_LAMBDA = 0.001   # weight on flow total-variation regulariser
STADV_KAPPA = 5.0             # CW margin
FLOW_BUDGETS = [0.05, 0.10, 0.20]   # tau, in normalised-coordinate units
STADV_BATCH = 256             # mini-batch for the attack loop (memory)

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h472_stadv_spatial_attack_output.txt",
)


# ---------------------------------------------------------------------------
# StAdv attack (pure torch, bilinear-grid_sample, flow-budget projection)
# ---------------------------------------------------------------------------
def _base_grid(B, H, W, device, dtype):
    """Identity sampling grid in normalised coordinates for F.grid_sample."""
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1)             # (H, W, 2)
    return grid.unsqueeze(0).expand(B, H, W, 2).contiguous()


def stadv_resample(x, flow):
    """x: (B,C,H,W); flow: (B,2,H,W) per-pixel displacement in normalised coords."""
    B, C, H, W = x.shape
    base = _base_grid(B, H, W, x.device, x.dtype)
    # flow channel 0 -> dx, channel 1 -> dy
    disp = flow.permute(0, 2, 3, 1)                  # (B, H, W, 2)
    grid = base + disp
    return F.grid_sample(x, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def flow_tv(flow):
    """Total-variation smoothness on the flow field (paper's L_flow)."""
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()


def cw_margin(logits, y, kappa=STADV_KAPPA):
    n = logits.size(1)
    mask = F.one_hot(y, n).bool()
    true_l = logits.masked_select(mask)
    other_max = logits.masked_fill(mask, -1e9).max(dim=1).values
    return torch.clamp(true_l - other_max + kappa, min=0.0)


def project_flow(flow, tau):
    """Clamp per-pixel L2 magnitude of the flow to tau (PGD-style projection)."""
    mag = flow.pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-12)
    factor = torch.clamp(tau / mag, max=1.0)
    return flow * factor


def stadv_attack(model, x, y, tau,
                 steps=STADV_STEPS, lr=STADV_LR,
                 lam_smooth=STADV_SMOOTH_LAMBDA):
    """Optimise a per-pixel flow field via Adam under per-pixel flow budget tau.

    Returns (x_adv, flipped_mask, flow_mag_mean_per_sample).
    flow is in normalised-grid units; per-pixel L2 magnitude is bounded by tau.
    """
    model.eval()
    B, C, H, W = x.shape
    flow = torch.zeros(B, 2, H, W, device=x.device, requires_grad=True)
    opt = torch.optim.Adam([flow], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        x_adv = stadv_resample(x, flow).clamp(0.0, 1.0)
        logits = model(x_adv)
        loss = cw_margin(logits, y).mean() + lam_smooth * flow_tv(flow)
        loss.backward()
        opt.step()
        with torch.no_grad():
            flow.data = project_flow(flow.data, tau)
    with torch.no_grad():
        x_adv = stadv_resample(x, flow).clamp(0.0, 1.0)
        flipped = model(x_adv).argmax(1) != y
        mag = flow.pow(2).sum(dim=1).sqrt().mean(dim=(1, 2))
    return x_adv.detach(), flipped.detach(), mag.detach()


def stadv_asr_full(model, X, Y, tau, batch=STADV_BATCH):
    """Run StAdv on the full eval set; return ASR over originally-correct samples
    plus mean flow magnitude, mean Linf, mean L2 of (x_adv - x)."""
    model.eval()
    flips, corr_mask, mags, linfs, l2s = [], [], [], [], []
    for i in range(0, X.size(0), batch):
        x = X[i:i + batch]
        y = Y[i:i + batch]
        with torch.no_grad():
            correct = model(x).argmax(1) == y
        x_adv, flipped, mag = stadv_attack(model, x, y, tau=tau)
        delta = (x_adv - x).flatten(1)
        flips.append(flipped.cpu())
        corr_mask.append(correct.cpu())
        mags.append(mag.cpu())
        linfs.append(delta.abs().max(dim=1).values.cpu())
        l2s.append(delta.norm(dim=1).cpu())
    flips = torch.cat(flips).numpy()
    corr = torch.cat(corr_mask).numpy().astype(bool)
    mags = torch.cat(mags).numpy()
    linfs = torch.cat(linfs).numpy()
    l2s = torch.cat(l2s).numpy()
    if corr.sum() > 0:
        asr = float(flips[corr].mean())
        mag_flipped = float(mags[corr & (flips == 1)].mean()) if (corr & (flips == 1)).any() else float("nan")
    else:
        asr = float("nan")
        mag_flipped = float("nan")
    return {
        "stadv_asr": asr,
        "stadv_flow_mag_flipped": mag_flipped,
        "stadv_linf_mean": float(linfs.mean()),
        "stadv_l2_mean": float(l2s.mean()),
    }


# ---------------------------------------------------------------------------
# Trainers (one per defence)
# ---------------------------------------------------------------------------
def _new_model(meta):
    C.set_seed(SEED)
    return C.build_model("cnn", meta, width=32, act="relu", bn=True)


def _opt(model):
    return torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)


def train_ce(meta, Xtr, Ytr):
    model = _new_model(meta)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
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
    return model


def train_fgsm_at(meta, Xtr, Ytr, eps=EPS):
    model = _new_model(meta)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_adv = C.fgsm(model, xb, yb, eps=eps)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_pgd_at(meta, Xtr, Ytr, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    model = _new_model(meta)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_adv = C.pgd(model, xb, yb, eps=eps, steps=steps, alpha=alpha)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def _pgd_on_kl(model, x, steps=PGD_STEPS, eps=EPS, alpha=PGD_ALPHA):
    """TRADES inner attack: PGD maximising KL from clean logits."""
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)
    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(0, 1)
    x0 = x.detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        kl = F.kl_div(F.log_softmax(model(x_adv), dim=1), p_clean, reduction="batchmean")
        g, = torch.autograd.grad(kl, x_adv)
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1)
    return x_adv.detach()


def train_trades(meta, Xtr, Ytr, beta=6.0):
    model = _new_model(meta)
    opt = _opt(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            x_adv = _pgd_on_kl(model, xb)
            model.train()
            out_clean = model(xb)
            out_adv = model(x_adv)
            loss = F.cross_entropy(out_clean, yb) + beta * F.kl_div(
                F.log_softmax(out_adv, dim=1),
                F.softmax(out_clean, dim=1),
                reduction="batchmean",
            )
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_linf(model, Xte, Yte):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return {
        "clean_acc": float(clean_acc),
        "fgsm_asr": 1.0 - float(acc_fgsm),
        "pgd_asr": 1.0 - float(acc_pgd),
    }


def eval_defence(name, model, Xte, Yte):
    print(f"  [{name}] evaluating Linf...")
    metrics = eval_linf(model, Xte, Yte)
    for tau in FLOW_BUDGETS:
        print(f"  [{name}] evaluating StAdv tau={tau:.2f}...")
        t0 = time.time()
        sm = stadv_asr_full(model, Xte, Yte, tau=tau)
        dt = time.time() - t0
        metrics[f"stadv_asr_tau{tau:.2f}"] = sm["stadv_asr"]
        metrics[f"stadv_flowmag_tau{tau:.2f}"] = sm["stadv_flow_mag_flipped"]
        metrics[f"stadv_linf_tau{tau:.2f}"] = sm["stadv_linf_mean"]
        metrics[f"stadv_l2_tau{tau:.2f}"] = sm["stadv_l2_mean"]
        metrics[f"stadv_time_tau{tau:.2f}"] = dt
    return metrics


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
DEFENCES = [
    ("CE",            lambda meta, Xtr, Ytr: train_ce(meta, Xtr, Ytr)),
    ("FGSM-AT",       lambda meta, Xtr, Ytr: train_fgsm_at(meta, Xtr, Ytr, eps=EPS)),
    ("PGD-AT",        lambda meta, Xtr, Ytr: train_pgd_at(meta, Xtr, Ytr, eps=EPS)),
    ("TRADES(beta=6)", lambda meta, Xtr, Ytr: train_trades(meta, Xtr, Ytr, beta=6.0)),
    ("PGD-AT-large(eps=0.2)",
                       lambda meta, Xtr, Ytr: train_pgd_at(meta, Xtr, Ytr, eps=EPS_LARGE)),
]


def fmt(v, w=8, p=4):
    if isinstance(v, float):
        if np.isnan(v):
            return f"{'nan':>{w}}"
        return f"{v:>{w}.{p}f}"
    return f"{str(v):>{w}}"


def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    meta = C.dataset_meta(DS)

    print(f"Device: {C.DEVICE}")
    print(f"N_train={Xtr.size(0)}  N_eval={Xte.size(0)}  EPS={EPS}  EPS_LARGE={EPS_LARGE}")
    print(f"Flow budgets: {FLOW_BUDGETS}  StAdv steps={STADV_STEPS}")

    results = []
    for name, trainer in DEFENCES:
        print(f"\n=== Training {name} ===")
        t0 = time.time()
        model = trainer(meta, Xtr, Ytr)
        train_t = time.time() - t0
        print(f"  trained in {train_t:.1f}s")
        m = eval_defence(name, model, Xte, Yte)
        m["name"] = name
        m["train_time_s"] = train_t
        results.append(m)
        print(f"  clean={m['clean_acc']:.4f}  fgsm={m['fgsm_asr']:.4f}  "
              f"pgd={m['pgd_asr']:.4f}  "
              f"stadv@0.05={m['stadv_asr_tau0.05']:.4f}  "
              f"stadv@0.10={m['stadv_asr_tau0.10']:.4f}  "
              f"stadv@0.20={m['stadv_asr_tau0.20']:.4f}")

    # --- write report -----------------------------------------------------
    lines = []
    lines.append("H472 Spatial / StAdv Attack -- Flow-Budget Sweep Across Defences")
    lines.append("=" * 78)
    lines.append(f"Dataset      : {DS}")
    lines.append(f"N_train      : {N_TRAIN}   N_eval: {Xte.size(0)}")
    lines.append(f"Epochs       : {EPOCHS}   Optimizer: SGD lr={LR} mom=0.9 wd=5e-4")
    lines.append(f"Batch        : {BATCH}   Seed: {SEED}")
    lines.append(f"Linf eps     : {EPS}   Linf eps (large): {EPS_LARGE}")
    lines.append(f"PGD inner    : {PGD_STEPS} steps, alpha={PGD_ALPHA}")
    lines.append(f"StAdv        : {STADV_STEPS} Adam steps lr={STADV_LR}, "
                 f"smooth_lambda={STADV_SMOOTH_LAMBDA}, kappa={STADV_KAPPA}")
    lines.append(f"Flow budgets : {FLOW_BUDGETS} (per-pixel L2, normalised-grid units)")
    lines.append("")
    lines.append("Anchor: Xiao et al. 2018, 'Spatially Transformed Adversarial Examples',")
    lines.append("        ICLR 2018 (arXiv:1801.02612).")
    lines.append("Refs  : Engstrom et al. 2019 (ICML) 'Exploring the Landscape of Spatial")
    lines.append("        Robustness' (arXiv:1712.02779); Kang et al. 2019 'Unforeseen")
    lines.append("        Adversaries'; Athalye et al. 2018 'Obfuscated Gradients'.")
    lines.append("")
    header = (
        f"{'defence':<24} {'clean':>7} {'fgsm':>7} {'pgd':>7} "
        f"{'stadv05':>8} {'stadv10':>8} {'stadv20':>8} "
        f"{'flowmag10':>10} {'linf10':>8} {'l2_10':>8} {'train_s':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"{r['name']:<24} "
            f"{r['clean_acc']:>7.4f} {r['fgsm_asr']:>7.4f} {r['pgd_asr']:>7.4f} "
            f"{r['stadv_asr_tau0.05']:>8.4f} {r['stadv_asr_tau0.10']:>8.4f} "
            f"{r['stadv_asr_tau0.20']:>8.4f} "
            f"{r['stadv_flowmag_tau0.10']:>10.4f} "
            f"{r['stadv_linf_tau0.10']:>8.4f} {r['stadv_l2_tau0.10']:>8.4f} "
            f"{r['train_time_s']:>8.1f}"
        )

    # ------ analysis & verdict --------------------------------------------
    lines.append("")
    lines.append("Analysis:")
    lines.append("- 'stadv05/10/20' = StAdv ASR at per-pixel flow budgets 0.05 / 0.10 / 0.20.")
    lines.append("- 'flowmag10' = mean per-pixel L2 of the learned flow on FLIPPED samples")
    lines.append("                at tau=0.10 (saturates near tau when attack succeeds).")
    lines.append("- 'linf10', 'l2_10' = Linf and L2 norms of the resulting pixel-space delta")
    lines.append("                at tau=0.10 (informational; StAdv is NOT Linf-bounded).")
    lines.append("")
    lines.append("Verdict heuristic:")
    lines.append("- If StAdv ASR for every AT defence stays within +/- 5pp of CE at every")
    lines.append("  tau, the H472 finding is: Linf-AT does not transfer to spatial threats,")
    lines.append("  confirming Engstrom 2019 / Kang 2019 cross-norm transfer claim.")
    lines.append("- If PGD-AT-large(eps=0.2) drops StAdv ASR by > 10pp vs CE, then larger")
    lines.append("  Linf-AT does buy partial spatial robustness (interpretation: large-eps")
    lines.append("  Linf adv examples already include some local warping signal).")
    lines.append("- TRADES is expected to match PGD-AT (same inner Linf attack).")
    lines.append("")
    # Compute the verdict programmatically.
    ce = next(r for r in results if r["name"] == "CE")
    deltas = {}
    for r in results:
        if r["name"] == "CE":
            continue
        d = {tau: r[f"stadv_asr_tau{tau:.2f}"] - ce[f"stadv_asr_tau{tau:.2f}"]
             for tau in FLOW_BUDGETS}
        deltas[r["name"]] = d
    lines.append("StAdv ASR delta vs CE baseline (negative = defence helps):")
    for nm, d in deltas.items():
        s = "  ".join(f"tau={t:.2f}:{d[t]:+.4f}" for t in FLOW_BUDGETS)
        lines.append(f"  {nm:<24}  {s}")
    lines.append("")
    worst_at_tau010 = max(deltas.values(), key=lambda d: d[0.10])[0.10]
    best_at_tau010 = min(deltas.values(), key=lambda d: d[0.10])[0.10]
    if best_at_tau010 > -0.05:
        verdict = ("VERDICT: NOT SUPPORTED (transfer hypothesis falsified). "
                   "No Linf-AT variant reduces StAdv ASR by more than 5pp at tau=0.10. "
                   "Linf-AT does NOT transfer to spatial StAdv threats; H472 confirms "
                   "Engstrom-2019 / Kang-2019 cross-norm robustness gap.")
    elif best_at_tau010 < -0.10:
        verdict = ("VERDICT: PARTIAL TRANSFER. Best Linf-AT defence cuts StAdv ASR by "
                   f">{abs(best_at_tau010):.2%} at tau=0.10. Interpretation: large-eps "
                   "Linf-AT incidentally covers some spatial-warp directions.")
    else:
        verdict = ("VERDICT: WEAK TRANSFER. Best Linf-AT defence buys 5-10pp StAdv-ASR "
                   "drop; partial transfer but well below pixel-Linf robustness gains.")
    lines.append(verdict)
    lines.append(f"  worst-AT delta at tau=0.10: {worst_at_tau010:+.4f}")
    lines.append(f"  best-AT  delta at tau=0.10: {best_at_tau010:+.4f}")
    lines.append("")
    lines.append("Caveats:")
    lines.append("- Single seed (SEED=0); cross-seed variance not estimated (M1).")
    lines.append("- N_train=6000 means PGD-AT itself is sub-saturating (M2); large-eps may")
    lines.append("  under-train.")
    lines.append("- StAdv attack is run with 100 Adam steps; a stronger attack (more steps,")
    lines.append("  smaller lr, or restart) might raise ASRs uniformly but should preserve")
    lines.append("  rank ordering. Athalye 2018 not applicable: the attack is fully")
    lines.append("  differentiable through grid_sample, no obfuscated gradients.")
    lines.append("- Per-pixel L2 budget is in normalised-grid units (image goes from -1 to 1)")
    lines.append("  so tau=0.10 corresponds to ~1.35 pixels of warp on a 28x28 image.")

    report = "\n".join(lines)
    print("\n" + report)
    with open(OUT_FILE, "w") as f:
        f.write(report + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
