"""
Theme A: adversarial robustness x machine unlearning.

Central question: when we make a model FORGET a subset of training samples,
what happens to adversarial vulnerability?  Does unlearning open "adversarial
holes" -- regions that become trivially attackable -- and is the forget set
itself left maximally fragile (low margin, high ASR)?

For each config we:
  1. train a base model on a train subset,
  2. select a forget set (by class / random / low-margin / high-margin),
  3. apply an unlearning method,
  4. measure accuracy + adversarial ASR + margin on retain/forget/test, before
     and after unlearning.
"""
import numpy as np
import torch
from . import common as C

DATASETS = ["fashion_mnist", "kmnist", "cifar10", "svhn"]
FORGET_KINDS = ["class", "random", "lowmargin", "highmargin"]
METHODS = ["finetune_retain", "neggrad", "neggrad_plus", "random_relabel"]
SEEDS = [0, 1, 2]


def build_specs():
    specs = []
    for ds in DATASETS:
        for fk in FORGET_KINDS:
            for m in METHODS:
                for s in SEEDS:
                    specs.append({
                        "theme": "A_unlearning",
                        "id": f"A_{ds}_{fk}_{m}_s{s}",
                        "params": {"dataset": ds, "forget_kind": fk, "method": m,
                                   "seed": s, "forget_frac": 0.1},
                    })
    return specs


def _eps_for(ds):
    return 0.1 if ds in ("fashion_mnist", "kmnist", "mnist") else 0.031


def _select_forget(forget_kind, Xtr, Ytr, base, meta, frac, seed):
    n = Xtr.size(0)
    g = torch.Generator().manual_seed(seed)
    if forget_kind == "class":
        cls = int(torch.randint(0, meta["n_classes"], (1,), generator=g))
        idx = (Ytr == cls).nonzero(as_tuple=True)[0]
        return idx, cls
    if forget_kind == "random":
        idx = torch.randperm(n, generator=g)[:int(frac * n)].to(Xtr.device)
        return idx, None
    # margin-based
    lg, _ = C.logits_and_acc(base, Xtr, Ytr)
    mar = C.margin_of(lg, Ytr)
    order = np.argsort(mar)  # ascending -> low margin first
    k = int(frac * n)
    sel = order[:k] if forget_kind == "lowmargin" else order[-k:]
    return torch.tensor(sel, device=Xtr.device), None


def _split_stats(model, X, Y, eps):
    if X.size(0) == 0:
        return {"acc": float("nan"), "asr": float("nan"), "margin": float("nan")}
    lg, acc = C.logits_and_acc(model, X, Y)
    mar = float(np.mean(C.margin_of(lg, Y)))
    r = C.attack_success(model, X, Y, attack="pgd", eps=eps, steps=7)
    return {"acc": acc, "asr": r["asr"], "margin": mar}


def run(spec):
    p = spec["params"]
    C.set_seed(p["seed"])
    ds = p["dataset"]
    meta = C.dataset_meta(ds)
    eps = _eps_for(ds)
    Xtr, Ytr, Xte, Yte = C.load_dataset(ds, n_train=8000, n_eval=1500, seed=p["seed"])

    width = 24 if meta["channels"] == 1 else 32
    base = C.build_model("cnn", meta, width=width)
    C.train_model(base, Xtr, Ytr, epochs=5, opt="sgd", lr=0.05, ncls=meta["n_classes"])

    fidx, forget_class = _select_forget(p["forget_kind"], Xtr, Ytr, base, meta, p["forget_frac"], p["seed"])
    mask = torch.ones(Xtr.size(0), dtype=torch.bool, device=Xtr.device)
    mask[fidx] = False
    Xr, Yr = Xtr[mask], Ytr[mask]
    Xf, Yf = Xtr[fidx], Ytr[fidx]

    before = {
        "forget": _split_stats(base, Xf, Yf, eps),
        "retain": _split_stats(base, Xr, Yr, eps),
        "test": _split_stats(base, Xte, Yte, eps),
    }

    C.unlearn(base, Xr, Yr, Xf, Yf, p["method"], meta, epochs=3, lr=0.01)

    after = {
        "forget": _split_stats(base, Xf, Yf, eps),
        "retain": _split_stats(base, Xr, Yr, eps),
        "test": _split_stats(base, Xte, Yte, eps),
    }

    metrics = {
        "dataset": ds, "forget_kind": p["forget_kind"], "method": p["method"],
        "seed": p["seed"], "eps": eps, "n_forget": int(Xf.size(0)),
        "forget_class": forget_class,
        "before": before, "after": after,
        "forget_acc_drop": before["forget"]["acc"] - after["forget"]["acc"],
        "retain_acc_drop": before["retain"]["acc"] - after["retain"]["acc"],
        "test_acc_drop": before["test"]["acc"] - after["test"]["acc"],
        "forget_asr_rise": after["forget"]["asr"] - before["forget"]["asr"],
        "test_asr_rise": after["test"]["asr"] - before["test"]["asr"],
        "forget_margin_drop": before["forget"]["margin"] - after["forget"]["margin"],
    }

    lines = []
    lines.append(f"Theme A | unlearning x adversarial | {spec['id']}")
    lines.append(f"dataset={ds} forget_kind={p['forget_kind']} method={p['method']} "
                 f"seed={p['seed']} eps={eps:.3f} n_forget={int(Xf.size(0))} class={forget_class}")
    lines.append("")
    lines.append(f"{'split':<8} {'phase':<7} {'acc':>8} {'pgd_asr':>9} {'margin':>9}")
    for split in ("forget", "retain", "test"):
        for phase, d in (("before", before), ("after", after)):
            s = d[split]
            lines.append(f"{split:<8} {phase:<7} {s['acc']:>8.4f} {s['asr']:>9.4f} {s['margin']:>9.3f}")
    lines.append("")
    lines.append(f"forget_acc_drop = {metrics['forget_acc_drop']:+.4f}  "
                 f"(unlearning effectiveness: higher = more forgotten)")
    lines.append(f"retain_acc_drop = {metrics['retain_acc_drop']:+.4f}  (collateral damage)")
    lines.append(f"test_acc_drop   = {metrics['test_acc_drop']:+.4f}")
    lines.append(f"forget_asr_rise = {metrics['forget_asr_rise']:+.4f}  "
                 f"(adversarial hole on forget set?)")
    lines.append(f"test_asr_rise   = {metrics['test_asr_rise']:+.4f}  (global robustness change)")
    lines.append(f"forget_margin_drop = {metrics['forget_margin_drop']:+.3f}")
    return "\n".join(lines), metrics
