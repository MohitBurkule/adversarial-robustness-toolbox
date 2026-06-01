"""
H183 - Machine unlearning leaves adversarial residual knowledge / holes.

Motivated by the Theme-A campaign finding that relabel/finetune unlearning of
high-margin (previously robust) samples raises their attack-success rate, and by
the 2025 literature showing unlearning guarantees do not extend to adversarial
proximities ("Residual Knowledge in Machine Unlearning under Perturbed Samples",
arXiv:2601.22359; Unlearning Mapping Attack; "Machine Unlearning Fails to
Remove", ICLR 2025).

We unlearn one whole class from a Fashion-MNIST CNN with the *gentle* method
(finetune-on-retain, which does not collapse the model) and ask three questions:

  (1) Hole:        does clean accuracy on the forgotten class drop while its
                   adversarial-success rate behaves anomalously (margin collapse)?
  (2) Recovery:    of forgotten-class test samples now misclassified on clean
                   input, what fraction can a *small* PGD step push BACK to the
                   forgotten (true) label? High recovery = residual knowledge: the
                   class boundary is still there, just nudged.
  (3) Attractor:   of OTHER-class samples correctly classified by BOTH models,
                   what fraction have an untargeted adversarial example that lands
                   IN the forgotten class? If unlearning truly removed the class,
                   this should fall toward zero; if it stays high the forgotten
                   class is still an adversarial attractor.

A genuine unlearn should remove the class as an adversarial attractor and make it
unrecoverable. Residual knowledge = high recovery / high attractor rate.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
EPS = 0.1                 # main L-inf budget (matches campaign fashion eps)
SMALL_EPS = 0.05          # "small nudge" for the recovery probe
PGD_STEPS = 10


def _pgd_targeted(model, X, target, eps, steps=10):
    """Targeted PGD: push inputs toward `target` label (minimise CE to target)."""
    X = X.clone().detach()
    delta = torch.zeros_like(X).uniform_(-eps, eps)
    delta.requires_grad_(True)
    for _ in range(steps):
        logits = model(torch.clamp(X + delta, 0, 1))
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        with torch.no_grad():
            delta -= (eps / 4) * delta.grad.sign()      # descend toward target
            delta.clamp_(-eps, eps)
        delta.grad.zero_()
    return torch.clamp(X + delta, 0, 1).detach()


def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=2000, seed=seed)
    forget_class = seed % meta["n_classes"]

    model = C.build_model("cnn", meta, seed=seed)
    C.train_model(model, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05, ncls=meta["n_classes"])

    # split train into retain / forget for the unlearning step
    fmask_tr = (Ytr == forget_class)
    Xf, Yf = Xtr[fmask_tr], Ytr[fmask_tr]
    Xr, Yr = Xtr[~fmask_tr], Ytr[~fmask_tr]

    # --- metrics BEFORE unlearning ---
    fte = (Yte == forget_class)
    ote = ~fte

    def class_acc(m, X, Y):
        lg, _ = C.logits_and_acc(m, X, Y)
        return float((lg.argmax(1).cpu() == Y.cpu()).float().mean())

    before_facc = class_acc(model, Xte[fte], Yte[fte])

    # attractor rate BEFORE: of other-class samples correct on the ORIGINAL model,
    # fraction whose untargeted PGD adversarial lands in forget_class
    lg_o, _ = C.logits_and_acc(model, Xte[ote], Yte[ote])
    corr_o = (lg_o.argmax(1).cpu() == Yte[ote].cpu())
    Xo_corr = Xte[ote][corr_o.to(Xte.device)]
    Yo_corr = Yte[ote][corr_o.to(Yte.device)]
    xa = C.pgd(model, Xo_corr, Yo_corr, EPS, PGD_STEPS)
    with torch.no_grad():
        adv_pred = model(xa).argmax(1).cpu()
    before_attractor = float((adv_pred == forget_class).float().mean())

    # --- UNLEARN (finetune on retain only; gentle, non-collapsing) ---
    C.unlearn(model, Xr, Yr, Xf, Yf, method="finetune_retain", meta=meta, epochs=3, lr=0.01)

    # --- metrics AFTER unlearning ---
    after_facc = class_acc(model, Xte[fte], Yte[fte])

    # (2) recovery: forget-class test samples now MISclassified on clean input;
    # can a small targeted PGD push them back to the forgotten label?
    Xf_te, Yf_te = Xte[fte], Yte[fte]
    with torch.no_grad():
        clean_pred = model(Xf_te).argmax(1).cpu()
    now_wrong = (clean_pred != forget_class)
    recovery = float("nan")
    if now_wrong.sum() > 0:
        Xw = Xf_te[now_wrong.to(Xf_te.device)]
        tgt = torch.full((Xw.size(0),), forget_class, device=Xw.device, dtype=torch.long)
        xr = _pgd_targeted(model, Xw, tgt, SMALL_EPS, PGD_STEPS)
        with torch.no_grad():
            rec_pred = model(xr).argmax(1).cpu()
        recovery = float((rec_pred == forget_class).float().mean())

    # (3) attractor rate AFTER (same other-class correct samples as before)
    xa2 = C.pgd(model, Xo_corr, Yo_corr, EPS, PGD_STEPS)
    with torch.no_grad():
        adv_pred2 = model(xa2).argmax(1).cpu()
    after_attractor = float((adv_pred2 == forget_class).float().mean())

    return {
        "seed": seed, "forget_class": forget_class,
        "forget_clean_acc_before": before_facc,
        "forget_clean_acc_after": after_facc,
        "attractor_rate_before": before_attractor,
        "attractor_rate_after": after_attractor,
        "adversarial_recovery_rate": recovery,
        "n_forget_test": int(fte.sum()),
    }


def main():
    print("=" * 74)
    print("H183 - Machine unlearning leaves adversarial residual knowledge / holes")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  small_eps={SMALL_EPS}")
    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"\n[seed {s}] forget_class={r['forget_class']}  ({r['runtime_s']}s)")
        print(f"  forget clean acc   : {r['forget_clean_acc_before']:.3f} -> {r['forget_clean_acc_after']:.3f}")
        print(f"  attractor rate     : {r['attractor_rate_before']:.3f} -> {r['attractor_rate_after']:.3f}"
              f"   (frac of other-class advs landing in forgotten class)")
        print(f"  adv recovery rate  : {r['adversarial_recovery_rate']:.3f}"
              f"   (forgotten samples pushed back to true label by small PGD)")

    def m(k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return sum(v) / len(v) if v else float("nan")

    print("\n" + "=" * 74)
    print("MEANS across seeds")
    print(f"  forget clean acc  : {m('forget_clean_acc_before'):.3f} -> {m('forget_clean_acc_after'):.3f}")
    print(f"  attractor rate    : {m('attractor_rate_before'):.3f} -> {m('attractor_rate_after'):.3f}")
    print(f"  adv recovery rate : {m('adversarial_recovery_rate'):.3f}")
    print("=" * 74)
    print("Interpretation: high adv-recovery and a non-vanishing attractor rate after")
    print("unlearning => the forgotten class survives as an adversarial residual: the")
    print("boundary is still present and reachable with a small perturbation, even")
    print("though clean accuracy on the class has dropped. This is an empirical")
    print("instance of unlearning failing under adversarial proximity (cf. 2601.22359,")
    print("UMA, 'Machine Unlearning Fails to Remove' ICLR 2025).")
    print("=" * 74)


if __name__ == "__main__":
    main()
