"""
H450 - Robust EMA / Polyak weight averaging under PGD-AT.

Gap: G5 (optimization side under AT; only base SAM (H273) and SWA (H274)
were tested previously, and H274's SWA was clean-only / FGSM-only).
Paper anchor: extends `wu-2020-awp` (weight-space flatness helps).
Related: Izmailov et al. 2018 (SWA); Gowal et al. 2020 (Uncovering the
Limits of Adversarial Training, NeurIPS-W) which uses EMA with decay
~0.9999; Pang et al. 2022 (Robustness and Accuracy Could Be Reconcilable
by (Proper) Definition, ICML) which also relies on EMA of weights.

Hypothesis: an exponential moving average (Polyak) of the online PGD-AT
weights, evaluated as the deployed model, sits in a flatter region of the
adversarial loss landscape and so achieves lower PGD ASR than the online
PGD-AT weights themselves. EMA is the simplest form of weight-space
flattening (AWP is a per-step adversarial flatten; SWA is a uniform tail
average; EMA is a recursive exponential tail average).

Risk: EMA may simply smooth single-batch attack noise during training
without addressing core robustness - i.e. the gain (if any) might not
survive a transfer-attack control. We therefore include a transfer-attack
masking check: adversarials crafted on the *online* (non-EMA) AT model
are transferred to the EMA model. If the EMA model resists the
white-box PGD attack but is broken by transferred adversarials at a
similar rate, the white-box result is a masking artifact.

Conditions (all PGD-10 AT with the standard config; only the deployed
weights differ):
    A. baseline_clean       - clean SGD (no AT, no EMA), as sanity ref.
    B. pgdat_online         - standard PGD-AT, deploy online weights.
    C. pgdat_ema_0.99       - PGD-AT online + EMA with decay 0.99 deployed.
    D. pgdat_ema_0.999      - PGD-AT online + EMA with decay 0.999 deployed.
    E. pgdat_ema_0.9995     - PGD-AT online + EMA with decay 0.9995 deployed.
    F. pgdat_awp            - PGD-AT + AWP (extends H316; gamma=5e-3).

Each AT condition shares the same trajectory (B reused as the online for
C/D/E - i.e. we collect three EMA streams from the SAME run to avoid 3x
extra compute and to keep the comparison apples-to-apples).

Transfer-attack control: PGD-10 adversarials crafted on B (online AT
weights) are evaluated on C, D, E and F. ASR delta between white-box PGD
on the EMA model and transferred PGD from B tells us whether EMA
white-box robustness is shortcut by gradient masking.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD mom=0.9 wd=5e-4,
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01. CNN width=32.

Output: results/fashion_mnist/h450_ema_polyak_at_output.txt (ASCII).
"""
import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C


# ---- config --------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
EMA_DECAYS = [0.99, 0.999, 0.9995]
AWP_GAMMA = 5e-3

META = {"channels": 1, "size": 28, "n_classes": 10}

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
)
OUT_FILE = os.path.join(RESULTS_DIR, "h450_ema_polyak_at_output.txt")


# ---- helpers -------------------------------------------------------------
def _make_optimizer_sgd(model, lr):
    """SGD mom=0.9 wd=5e-4 (campaign default; common.make_optimizer uses 1e-4)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def _new_model():
    C.set_seed(SEED)
    return C.build_model("cnn", META, width=32)


def _clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _zero_like_state(state):
    out = {}
    for k, v in state.items():
        if v.is_floating_point():
            out[k] = torch.zeros_like(v)
        else:
            out[k] = v.detach().clone()
    return out


def _ema_update(ema_state, online_state, decay):
    """In-place: ema = decay*ema + (1-decay)*online (float tensors only)."""
    for k in ema_state:
        v_ema = ema_state[k]
        v_on = online_state[k]
        if v_ema.is_floating_point():
            v_ema.mul_(decay).add_(v_on.detach(), alpha=(1.0 - decay))
        else:
            # integer running counters etc: copy from online
            v_ema.copy_(v_on)


def _load_state_into_new_model(state):
    m = _new_model()
    m.load_state_dict(state)
    m.to(C.DEVICE)
    return m


def _refresh_bn(model, Xtr):
    """Recompute BN running stats with one pass over Xtr (eval-time fix
    needed after weight averaging - same pattern as H274)."""
    model.train()
    with torch.no_grad():
        for i in range(0, len(Xtr), 256):
            model(Xtr[i:i + 256])
    model.eval()


# ---- evaluation ----------------------------------------------------------
def eval_white_box(model, Xte, Yte):
    """Standard clean/FGSM/PGD eval on the *deployed* model."""
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    margin = float(np.mean(C.margin(model, Xte, Yte)))
    return dict(
        clean_acc=float(clean_acc),
        fgsm_asr=1.0 - float(acc_fgsm),
        pgd_asr=1.0 - float(acc_pgd),
        mean_margin=margin,
    )


def eval_transfer(model_target, Xadv, Yte):
    """Evaluate model on pre-crafted adversarials (transfer attack)."""
    model_target.eval()
    with torch.no_grad():
        _, acc = C.logits_and_acc(model_target, Xadv, Yte)
    return 1.0 - float(acc)


# ---- training routines ---------------------------------------------------
def train_clean(Xtr, Ytr):
    """Standard SGD, no AT, no EMA - sanity baseline."""
    model = _new_model().to(C.DEVICE)
    opt = _make_optimizer_sgd(model, LR)
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


def train_pgdat_with_ema(Xtr, Ytr, decays):
    """PGD-10 adversarial training; maintain one EMA shadow per decay.

    Returns (online_model, {decay: ema_state_dict}).
    """
    model = _new_model().to(C.DEVICE)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # Initialise EMA shadows = exact copy of current online weights.
    ema_states = {d: _clone_state(model) for d in decays}

    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_adv = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            opt.step()
            # Update EMA shadows after the SGD step.
            online = model.state_dict()
            for d in decays:
                _ema_update(ema_states[d], online, d)
        sched.step()
    model.eval()
    return model, ema_states


def train_pgdat_awp(Xtr, Ytr, gamma):
    """PGD-AT + AWP (extends H316): perturb weights by gamma*sign(grad_w L_adv),
    update on perturbed weights, then restore."""
    model = _new_model().to(C.DEVICE)
    opt = _make_optimizer_sgd(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_adv = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            # Compute weight-perturbation direction (grad on adv loss).
            model.zero_grad()
            loss_adv = F.cross_entropy(model(xb_adv), yb)
            loss_adv.backward()
            perts = {}
            if gamma > 0:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            d = gamma * p.grad.sign()
                            perts[name] = d
                            p.add_(d)
            # Train on the perturbed weights.
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            # Restore weights before optimizer step (so opt sees orig + new grad).
            if gamma > 0:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in perts:
                            p.sub_(perts[name])
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- main ----------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    print(f"  train={Xtr.size(0)}  test={Xte.size(0)}")

    rows = []  # ordered list of (label, result_dict, transfer_asr_or_None)

    # ---- A. clean baseline (no AT) --------------------------------------
    print("\n--- A. baseline_clean (no AT, no EMA) ---")
    m_clean = train_clean(Xtr, Ytr)
    r = eval_white_box(m_clean, Xte, Yte)
    print(f"  clean={r['clean_acc']:.4f}  FGSM_ASR={r['fgsm_asr']:.4f}  "
          f"PGD_ASR={r['pgd_asr']:.4f}  margin={r['mean_margin']:.4f}")
    rows.append(("baseline_clean", r, None))

    # ---- B-E. PGD-AT online + EMA shadows -------------------------------
    print("\n--- B. pgdat_online (and EMA shadows) ---")
    m_online, ema_states = train_pgdat_with_ema(Xtr, Ytr, EMA_DECAYS)

    # Craft transfer adversarials on the *online* AT model.
    print("  Crafting PGD adversarials on online AT model for transfer check...")
    for p in m_online.parameters():
        p.requires_grad_(True)
    Xadv_online = C.pgd(m_online, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)

    # Online eval (white-box).
    r_online = eval_white_box(m_online, Xte, Yte)
    print(f"  [online] clean={r_online['clean_acc']:.4f}  "
          f"FGSM_ASR={r_online['fgsm_asr']:.4f}  "
          f"PGD_ASR={r_online['pgd_asr']:.4f}  margin={r_online['mean_margin']:.4f}")
    # Transfer onto self == white-box PGD (sanity).
    rows.append(("pgdat_online", r_online, r_online["pgd_asr"]))

    # For each EMA decay, load EMA state into a fresh model, refresh BN,
    # then eval white-box AND transfer.
    for d in EMA_DECAYS:
        label = f"pgdat_ema_{d}"
        print(f"\n--- {label} ---")
        m_ema = _load_state_into_new_model(ema_states[d])
        _refresh_bn(m_ema, Xtr)
        r = eval_white_box(m_ema, Xte, Yte)
        t_asr = eval_transfer(m_ema, Xadv_online, Yte)
        print(f"  [{label}] clean={r['clean_acc']:.4f}  "
              f"FGSM_ASR={r['fgsm_asr']:.4f}  "
              f"PGD_ASR_wb={r['pgd_asr']:.4f}  "
              f"PGD_ASR_xfer_from_online={t_asr:.4f}  "
              f"margin={r['mean_margin']:.4f}")
        rows.append((label, r, t_asr))

    # ---- F. PGD-AT + AWP -------------------------------------------------
    print("\n--- F. pgdat_awp ---")
    m_awp = train_pgdat_awp(Xtr, Ytr, AWP_GAMMA)
    r_awp = eval_white_box(m_awp, Xte, Yte)
    t_asr_awp = eval_transfer(m_awp, Xadv_online, Yte)
    print(f"  [pgdat_awp] clean={r_awp['clean_acc']:.4f}  "
          f"FGSM_ASR={r_awp['fgsm_asr']:.4f}  "
          f"PGD_ASR_wb={r_awp['pgd_asr']:.4f}  "
          f"PGD_ASR_xfer_from_online={t_asr_awp:.4f}  "
          f"margin={r_awp['mean_margin']:.4f}")
    rows.append(("pgdat_awp", r_awp, t_asr_awp))

    elapsed = time.time() - t0

    # ---- write output ---------------------------------------------------
    lines = []
    lines.append("H450 - Robust EMA / Polyak weight averaging under PGD-AT")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"dataset={DS}  N_train={N_TRAIN}  N_eval={N_EVAL}  "
                 f"epochs={EPOCHS}  lr={LR}  batch={BATCH}")
    lines.append(f"seed={SEED}  eps={EPS}  pgd_steps={PGD_STEPS}  "
                 f"pgd_alpha={PGD_ALPHA}")
    lines.append(f"EMA decays tested: {EMA_DECAYS}  AWP gamma={AWP_GAMMA}")
    lines.append("")
    lines.append("Note: EMA shadows (C/D/E) are tracked from the SAME PGD-AT")
    lines.append("training run as the online model (B); only the deployed")
    lines.append("weights differ. F uses an independent AWP-flavoured run.")
    lines.append("")
    header = (f"{'condition':<22} {'clean':>8} {'fgsm_asr':>10} "
              f"{'pgd_asr_wb':>12} {'pgd_asr_xfer':>14} {'margin':>10}")
    lines.append(header)
    lines.append("-" * len(header))
    for label, r, t_asr in rows:
        t_str = f"{t_asr:.4f}" if t_asr is not None else "   n/a"
        lines.append(
            f"{label:<22} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>10.4f} "
            f"{r['pgd_asr']:>12.4f} {t_str:>14} {r['mean_margin']:>10.4f}"
        )
    lines.append("")
    lines.append(f"Elapsed: {elapsed:.1f}s")
    lines.append("")

    # ---- analysis & verdict --------------------------------------------
    base_pgd = next(r for lab, r, _ in rows if lab == "pgdat_online")["pgd_asr"]
    best_ema_label, best_ema_pgd = None, 1.01
    for lab, r, _ in rows:
        if lab.startswith("pgdat_ema_") and r["pgd_asr"] < best_ema_pgd:
            best_ema_pgd, best_ema_label = r["pgd_asr"], lab
    awp_pgd = next(r for lab, r, _ in rows if lab == "pgdat_awp")["pgd_asr"]

    delta_ema = best_ema_pgd - base_pgd
    delta_awp = awp_pgd - base_pgd

    lines.append("ANALYSIS")
    lines.append("--------")
    lines.append(f"PGD_ASR online PGD-AT baseline: {base_pgd:.4f}")
    lines.append(f"Best EMA condition ({best_ema_label}): PGD_ASR_wb="
                 f"{best_ema_pgd:.4f}  (delta vs online = {delta_ema:+.4f})")
    lines.append(f"AWP condition: PGD_ASR_wb={awp_pgd:.4f}  "
                 f"(delta vs online = {delta_awp:+.4f})")

    # Masking check: for each EMA, compare white-box vs transfer ASR.
    lines.append("")
    lines.append("Masking check (transfer-from-online vs white-box on EMA models):")
    masking_flag = False
    for lab, r, t_asr in rows:
        if lab.startswith("pgdat_ema_") and t_asr is not None:
            gap = t_asr - r["pgd_asr"]
            lines.append(f"  {lab}: wb={r['pgd_asr']:.4f}  xfer={t_asr:.4f}  "
                         f"(xfer-wb={gap:+.4f})")
            # If white-box looks much lower than transfer => masking suspect.
            if gap > 0.05:
                masking_flag = True

    # ---- verdict --------------------------------------------------------
    THRESHOLD = 0.02  # PGD ASR improvement worth calling "supported".
    if delta_ema < -THRESHOLD and not masking_flag:
        verdict = ("SUPPORTED: EMA(decay={}) reduces PGD ASR by {:.3f} over "
                   "online PGD-AT with no transfer-attack masking signature."
                   .format(best_ema_label.split('_')[-1], -delta_ema))
    elif delta_ema < -THRESHOLD and masking_flag:
        verdict = ("PARTIAL/MASKED: EMA lowers white-box PGD ASR by {:.3f} but "
                   "transfer-from-online is markedly higher than white-box "
                   "(masking suspected; gain may be an artifact of gradient "
                   "noise smoothing).".format(-delta_ema))
    elif abs(delta_ema) <= THRESHOLD:
        verdict = ("NOT SUPPORTED: EMA decay sweep ties online PGD-AT within "
                   "+/-{:.2f} PGD ASR; weight-averaging alone does not "
                   "meaningfully flatten the AT loss basin at this scale."
                   .format(THRESHOLD))
    else:
        verdict = ("NEGATIVE: EMA increases PGD ASR over online PGD-AT by "
                   "{:.3f}; the stale shadow lags the AT trajectory.".format(delta_ema))

    lines.append("")
    lines.append("VERDICT")
    lines.append("-------")
    lines.append(verdict)
    lines.append("")
    lines.append("Comparator: AWP delta vs online = {:+.4f} PGD ASR.".format(delta_awp))
    lines.append("")

    out = "\n".join(lines) + "\n"
    with open(OUT_FILE, "w") as f:
        f.write(out)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (AttributeError, OSError):
            pass
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
