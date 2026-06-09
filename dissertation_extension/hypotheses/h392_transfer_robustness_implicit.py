"""
H392 - Transfer robustness of implicit-method models (boundary-smoothing test).

Hypothesis: implicit-method models (input-gradient penalty H288-style,
Jacobian-Frobenius H323-style) resist *transferred* adversarials crafted on a
standard surrogate better than they resist their OWN white-box adversarials.
If a model's transfer ASR is far below its white-box ASR, the defense is
smoothing the decision boundary against externally-aimed perturbations while
remaining (partly) breakable under direct white-box optimisation -- the
opposite of gradient masking (where transfer would EXCEED white-box).

Implementation:
  * train a standard surrogate;
  * craft FGSM and PGD adversarials on the surrogate;
  * evaluate transfer ASR of those adversarials on each defended model
    {input-grad-penalty, Jacobian, PGD-AT}, plus the standard model as control;
  * compare each model's transfer ASR against its own white-box ASR (FGSM, PGD).

Report a table of clean acc, white-box FGSM/PGD ASR, and transfer FGSM/PGD ASR,
with the white-box - transfer gap per model.
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
LAM = 0.1

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "fashion_mnist", "h392_transfer_robustness_implicit_output.txt")

_LINES = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LINES.append(s)


# --- training (shared with H391 design; re-train, no checkpoints) ----------
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
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model


def train_jacobian(meta, Xtr, Ytr):
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
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model


def train_pgd_at(meta, Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=0)
    return C.train_model(model, Xtr, Ytr, epochs=EPOCHS, adv_train=True,
                         adv_eps=EPS, adv_steps=7)


# --- attack / eval utilities ----------------------------------------------
@torch.no_grad()
def _correct_mask(model, X, Y, batch=256):
    parts = []
    for i in range(0, X.size(0), batch):
        parts.append((model(X[i:i + batch]).argmax(1) == Y[i:i + batch]).cpu())
    return torch.cat(parts)


def _asr(model, Xadv, Y, corr, batch=256):
    flips = []
    with torch.no_grad():
        for i in range(0, Xadv.size(0), batch):
            flips.append((model(Xadv[i:i + batch]).argmax(1) != Y[i:i + batch]).cpu())
    flips = torch.cat(flips).numpy()
    corr = corr.numpy().astype(bool)
    return float(flips[corr].mean()) if corr.sum() > 0 else float("nan")


def pgd_batched(model, X, Y, batch=256):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(C.pgd(model, X[i:i + batch], Y[i:i + batch], eps=EPS,
                          steps=PGD_STEPS, alpha=PGD_ALPHA, random_start=True))
    return torch.cat(outs)


def fgsm_batched(model, X, Y, batch=256):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(C.fgsm(model, X[i:i + batch], Y[i:i + batch], EPS))
    return torch.cat(outs)


def main():
    t0 = time.time()
    log("=" * 86)
    log("H392  Transfer robustness of implicit-method models (Fashion-MNIST)")
    log(f"  N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} EPS={EPS} PGD_STEPS={PGD_STEPS} "
        f"lambda={LAM} device={C.DEVICE}")
    log("  surrogate for transfer adversarials = STANDARD model")
    log("=" * 86)

    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    log("\n[1/3] training models ...")
    models = {}
    for name, fn in [("standard", train_standard),
                     ("input_grad", train_input_grad),
                     ("jacobian", train_jacobian),
                     ("pgd_at", train_pgd_at)]:
        ts = time.time()
        models[name] = fn(meta, Xtr, Ytr)
        _, acc = C.logits_and_acc(models[name], Xte, Yte)
        log(f"    {name:12s} clean_acc={acc:.4f}  ({time.time()-ts:.1f}s)")

    log("\n[2/3] crafting transfer adversarials on STANDARD surrogate ...")
    surrogate = models["standard"]
    surr_fgsm = fgsm_batched(surrogate, Xte, Yte)
    surr_pgd = pgd_batched(surrogate, Xte, Yte)

    log("\n[3/3] white-box vs transfer ASR per model ...")
    rows = {}
    for name, model in models.items():
        corr = _correct_mask(model, Xte, Yte)
        # white-box (own gradients)
        wb_fgsm = _asr(model, fgsm_batched(model, Xte, Yte), Yte, corr)
        wb_pgd = _asr(model, pgd_batched(model, Xte, Yte), Yte, corr)
        # transfer (adversarials crafted on the standard surrogate)
        tr_fgsm = _asr(model, surr_fgsm, Yte, corr)
        tr_pgd = _asr(model, surr_pgd, Yte, corr)
        _, acc = C.logits_and_acc(model, Xte, Yte)
        rows[name] = dict(acc=acc, corr=int(corr.sum()),
                          wb_fgsm=wb_fgsm, wb_pgd=wb_pgd,
                          tr_fgsm=tr_fgsm, tr_pgd=tr_pgd)
        log(f"    {name:12s} done")

    log("\nResults: white-box (own-grad) vs transfer (standard-surrogate) ASR")
    log("-" * 86)
    log(f"{'model':12s} {'clean':>6s} {'wbFGSM':>7s} {'trFGSM':>7s} {'wbPGD':>7s} "
        f"{'trPGD':>7s} {'PGD wb-tr':>10s}")
    log("-" * 86)
    for name in models:
        r = rows[name]
        log(f"{name:12s} {r['acc']:6.3f} {r['wb_fgsm']:7.3f} {r['tr_fgsm']:7.3f} "
            f"{r['wb_pgd']:7.3f} {r['tr_pgd']:7.3f} {r['wb_pgd']-r['tr_pgd']:10.3f}")
    log("-" * 86)

    log("\ninterpretation (per defended model):")
    log("  boundary-smoothing  => transfer ASR << own white-box ASR (wb-tr > 0)")
    log("  gradient-masking    => transfer ASR >> own white-box ASR (wb-tr < 0)")
    log("-" * 86)
    for name in models:
        if name == "standard":
            continue
        r = rows[name]
        gap = r["wb_pgd"] - r["tr_pgd"]
        if gap > 0.05:
            verdict = ("RESISTS-TRANSFER (transfer << white-box): consistent with "
                       "boundary smoothing, NOT masking")
        elif gap < -0.05:
            verdict = ("TRANSFER-EXCEEDS-WHITEBOX: classic gradient-masking "
                       "signature")
        else:
            verdict = "transfer ~ white-box: no strong smoothing or masking signal"
        log(f"  {name:12s} PGD wb={r['wb_pgd']:.3f} tr={r['tr_pgd']:.3f} "
            f"(gap={gap:+.3f}) -> {verdict}")
    log("-" * 86)

    log(f"\ntotal runtime {time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(_LINES) + "\n")
    log(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
