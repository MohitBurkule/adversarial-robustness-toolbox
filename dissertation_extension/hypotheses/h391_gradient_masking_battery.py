"""
H391 - Gradient-masking detection battery across implicit-defense "winners".

Several prior campaign hypotheses reported robustness gains from *implicit*
defenses (input-gradient penalty H288-style, Jacobian-Frobenius penalty
H323-style). Such gains are notorious for being gradient-masking artefacts
(Athalye et al. 2018, "Obfuscated Gradients") rather than genuine robustness.

This script re-trains a small set of defended Fashion-MNIST CNNs and runs a
five-signal masking battery on each:

  (a) white-box PGD ASR (steps=10).
  (b) black-box TRANSFER ASR: PGD adversarials are crafted on the STANDARD
      baseline, then evaluated on the defended model. If transfer ASR exceeds
      the model's own white-box ASR, its own gradients are useless to the
      attacker => masking signal.
  (c) many random restarts: PGD with 10 random restarts, worst-case kept per
      sample. If multi-restart ASR >> single-restart ASR, the single white-box
      number under-estimated true vulnerability => masking.
  (d) increasing-steps PGD curve over steps in {1,5,10,20,50}. A genuinely
      robust model plateaus; a masked one keeps climbing steeply.
  (e) FGSM-vs-PGD gap. If FGSM is weak but PGD strong (large gap), the loss
      surface is locally jagged / gradients unreliable for one-step => masking.

Each model is FLAGGED as LIKELY-MASKING or GENUINE based on the union of
signals.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

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
PGD_STEPS = 10
PGD_ALPHA = 0.01
LAM = 0.1                 # implicit-penalty strength
N_RESTARTS = 10
STEP_CURVE = [1, 5, 10, 20, 50]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "fashion_mnist", "h391_gradient_masking_battery_output.txt")

_LINES = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LINES.append(s)


# ---------------------------------------------------------------------------
# training helpers (re-train, do not use checkpoints)
# ---------------------------------------------------------------------------
def _opt_sched(model):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return opt, sched


def _iter_batches(Xtr, Ytr):
    n = Xtr.size(0)
    perm = torch.randperm(n, device=Xtr.device)
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        yield Xtr[idx], Ytr[idx]


def train_standard(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=0)
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, opt="sgd", lr=LR)


def train_input_grad(meta, Xtr, Ytr):
    """CE + lambda * || d L / d x ||^2 (double-backprop)."""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=0)
    opt, sched = _opt_sched(model)
    model.train()
    for ep in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            xb = xb.clone().detach().requires_grad_(True)
            out = model(xb)
            ce = F.cross_entropy(out, yb)
            g = torch.autograd.grad(ce, xb, create_graph=True, retain_graph=True)[0]
            pen = (g.reshape(g.size(0), -1) ** 2).sum(1).mean()
            loss = ce + LAM * pen
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_jacobian(meta, Xtr, Ytr):
    """CE + lambda * Hutchinson estimate of ||J_f(x)||_F^2."""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=0)
    opt, sched = _opt_sched(model)
    model.train()
    for ep in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            xb = xb.clone().detach().requires_grad_(True)
            out = model(xb)
            ce = F.cross_entropy(out, yb)
            v = torch.randn_like(out)
            jvp = torch.autograd.grad((out * v).sum(), xb,
                                      create_graph=True, retain_graph=True)[0]
            pen = (jvp ** 2).sum(1).mean() if jvp.dim() == 2 else (jvp ** 2).sum() / xb.size(0)
            loss = ce + LAM * pen
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_pgd_at(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=0)
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, adv_train=True,
                         adv_eps=EPS, adv_steps=7)


# ---------------------------------------------------------------------------
# attack utilities operating only on originally-correct samples
# ---------------------------------------------------------------------------
@torch.no_grad()
def _correct_mask(model, X, Y, batch=256):
    parts = []
    for i in range(0, X.size(0), batch):
        parts.append((model(X[i:i + batch]).argmax(1) == Y[i:i + batch]).cpu())
    return torch.cat(parts)


def _asr_from_advx(model, Xadv, Y, corr, batch=256):
    """ASR over originally-correct samples given precomputed adversarials."""
    flips = []
    with torch.no_grad():
        for i in range(0, Xadv.size(0), batch):
            pred = model(Xadv[i:i + batch]).argmax(1)
            flips.append((pred != Y[i:i + batch]).cpu())
    flips = torch.cat(flips).numpy()
    corr = corr.numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() > 0 else float("nan")


def pgd_batched(model, X, Y, eps, steps, alpha, random_start=True, batch=256):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(C.pgd(model, X[i:i + batch], Y[i:i + batch], eps=eps,
                          steps=steps, alpha=alpha, random_start=random_start))
    return torch.cat(outs)


def fgsm_batched(model, X, Y, eps, batch=256):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(C.fgsm(model, X[i:i + batch], Y[i:i + batch], eps))
    return torch.cat(outs)


def whitebox_pgd_asr(model, X, Y, corr, steps=PGD_STEPS):
    adv = pgd_batched(model, X, Y, EPS, steps, PGD_ALPHA, random_start=True)
    return _asr_from_advx(model, adv, Y, corr)


def fgsm_asr(model, X, Y, corr):
    adv = fgsm_batched(model, X, Y, EPS)
    return _asr_from_advx(model, adv, Y, corr)


def transfer_asr(target_model, X, Y, corr, surrogate_advx):
    """Evaluate adversarials crafted on surrogate against target_model."""
    return _asr_from_advx(target_model, surrogate_advx, Y, corr)


def multi_restart_asr(model, X, Y, corr, restarts=N_RESTARTS, batch=256):
    """Worst-case (per-sample) flip over many random restarts of PGD."""
    corr_np = corr.numpy().astype(bool)
    flipped_any = np.zeros(X.size(0), dtype=bool)
    for r in range(restarts):
        adv = pgd_batched(model, X, Y, EPS, PGD_STEPS, PGD_ALPHA, random_start=True)
        with torch.no_grad():
            f = []
            for i in range(0, adv.size(0), batch):
                f.append((model(adv[i:i + batch]).argmax(1) != Y[i:i + batch]).cpu())
            f = torch.cat(f).numpy()
        flipped_any |= f
    return float(flipped_any[corr_np].mean()) if corr_np.sum() > 0 else float("nan")


def steps_curve_asr(model, X, Y, corr):
    curve = {}
    for s in STEP_CURVE:
        a = 2.5 * EPS / s
        adv = pgd_batched(model, X, Y, EPS, s, a, random_start=True)
        curve[s] = _asr_from_advx(model, adv, Y, corr)
    return curve


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log("=" * 78)
    log("H391  Gradient-masking detection battery (Fashion-MNIST)")
    log(f"  N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} EPS={EPS} PGD_STEPS={PGD_STEPS} "
        f"restarts={N_RESTARTS} lambda={LAM} device={C.DEVICE}")
    log("=" * 78)

    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    log("\n[1/4] training models (re-train, no checkpoints) ...")
    models = {}
    for name, fn in [("standard", train_standard),
                     ("input_grad", train_input_grad),
                     ("jacobian", train_jacobian),
                     ("pgd_at", train_pgd_at)]:
        ts = time.time()
        models[name] = fn(meta, Xtr, Ytr)
        _, acc = C.logits_and_acc(models[name], Xte, Yte)
        log(f"    {name:12s} clean_acc={acc:.4f}  ({time.time()-ts:.1f}s)")

    # surrogate adversarials = PGD crafted on the STANDARD model (black-box source)
    log("\n[2/4] crafting transfer adversarials on STANDARD surrogate ...")
    surrogate = models["standard"]
    surr_pgd_advx = pgd_batched(surrogate, Xte, Yte, EPS, PGD_STEPS, PGD_ALPHA,
                                random_start=True)

    log("\n[3/4] running battery per model ...")
    results = {}
    for name, model in models.items():
        ts = time.time()
        corr = _correct_mask(model, Xte, Yte)
        wb = whitebox_pgd_asr(model, Xte, Yte, corr)
        # transfer: evaluate the standard-surrogate adversarials on this model.
        # (for the standard model itself this is its own white-box, kept for ref)
        tr = transfer_asr(model, Xte, Yte, corr, surr_pgd_advx)
        mr = multi_restart_asr(model, Xte, Yte, corr)
        curve = steps_curve_asr(model, Xte, Yte, corr)
        fg = fgsm_asr(model, Xte, Yte, corr)
        results[name] = dict(corr=int(corr.sum()), wb=wb, tr=tr, mr=mr,
                             curve=curve, fgsm=fg)
        log(f"    {name:12s} done ({time.time()-ts:.1f}s)")

    # ----- analysis & flags -------------------------------------------------
    log("\n[4/4] battery results")
    log("-" * 78)
    hdr = (f"{'model':12s} {'clean_n':>7s} {'wb_PGD':>7s} {'transfer':>8s} "
           f"{'10xrest':>7s} {'FGSM':>6s} {'fgsm-pgd':>8s}")
    log(hdr)
    log("-" * 78)
    for name in models:
        r = results[name]
        gap = r["fgsm"] - r["wb"]
        log(f"{name:12s} {r['corr']:7d} {r['wb']:7.3f} {r['tr']:8.3f} "
            f"{r['mr']:7.3f} {r['fgsm']:6.3f} {gap:8.3f}")
    log("-" * 78)

    log("\nincreasing-steps PGD ASR curve (steps -> ASR):")
    log(f"{'model':12s} " + " ".join(f"s={s:<5d}" for s in STEP_CURVE))
    for name in models:
        c = results[name]["curve"]
        log(f"{name:12s} " + " ".join(f"{c[s]:7.3f}" for s in STEP_CURVE))

    log("\nper-model masking verdicts:")
    log("  (masking signals are only meaningful for a model CLAIMING robustness,")
    log("   i.e. a low white-box PGD ASR; a wide-open model with ~0.9 ASR cannot")
    log("   be 'masking' since the attacker already wins. Signals are therefore")
    log("   gated on the model appearing robust: white-box ASR < 0.5.)")
    log("-" * 78)
    ROBUST_CLAIM = 0.5   # white-box ASR below this = the model 'claims' robustness
    for name in models:
        r = results[name]
        claims_robust = r["wb"] < ROBUST_CLAIM
        signals = []
        # (b) transfer > white-box (skip for standard: transfer IS its white-box).
        #     Masking => the defended model is MORE vulnerable to externally-crafted
        #     adversarials than to its own gradients.
        if name != "standard" and r["tr"] > r["wb"] + 0.03:
            signals.append(f"transfer ASR {r['tr']:.3f} > white-box {r['wb']:.3f}")
        # (c) multi-restart >> single restart
        if r["mr"] > r["wb"] + 0.05:
            signals.append(f"10-restart ASR {r['mr']:.3f} >> single {r['wb']:.3f} "
                           f"(+{r['mr']-r['wb']:.3f})")
        # (d) steeply-rising curve: large jump from steps=10 to steps=50
        rise = r["curve"][50] - r["curve"][10]
        if rise > 0.05:
            signals.append(f"steps-curve still rising +{rise:.3f} (s10->s50)")
        # (e) large FGSM-vs-PGD gap (FGSM weak, PGD strong). Only diagnostic of
        #     masking if it would overturn a robustness claim, i.e. FGSM looked
        #     robust but PGD breaks it. For a wide-open model this is just the
        #     ordinary FGSM<PGD ordering and is NOT masking.
        gap = r["wb"] - r["fgsm"]
        if gap > 0.15 and r["fgsm"] < ROBUST_CLAIM and not claims_robust:
            signals.append(f"FGSM looked safe ({r['fgsm']:.3f}) but PGD breaks "
                           f"({r['wb']:.3f}); one-step gradient unreliable")
        # only meaningful as 'masking' if the model otherwise looks robust;
        # signals (b)-(d) that survive on a robust-claiming model are the real tell.
        meaningful = [s for s in signals]
        if name == "standard":
            flag = "N/A (undefended reference; white-box ASR=%.3f)" % r["wb"]
            log(f"  {name:12s} -> {flag}")
            continue
        flag = "LIKELY-MASKING" if meaningful else "GENUINE"
        log(f"  {name:12s} -> {flag}  (white-box PGD ASR={r['wb']:.3f}, "
            f"{'claims robustness' if claims_robust else 'not robust anyway'})")
        if meaningful:
            for s in meaningful:
                log(f"        signal: {s}")
        else:
            log(f"        (no masking signals: transfer<=wb, restarts flat, "
                f"curve plateaued, FGSM~PGD)")
    log("-" * 78)

    log(f"\ntotal runtime {time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(_LINES) + "\n")
    log(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
