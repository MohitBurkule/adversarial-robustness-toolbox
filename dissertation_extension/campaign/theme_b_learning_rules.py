"""
Theme B: is adversarial vulnerability a property of GRADIENT DESCENT, or of the
data / input geometry?

We hold the data fixed and vary the *learning rule*:
  * GD variants: SGD, Adam, RMSprop, AdamW, high-WD SGD, label-smoothing, mixup,
    adversarial training, different activations / architectures.
  * Non-GD / non-end-to-end learners: random-feature net (frozen random conv +
    trained readout), Forward-Forward (Hinton 2022, local goodness, no backprop),
    k-NN on raw pixels, gradient-boosted trees (HistGBM), logistic regression.

For every learner we measure, under a SHARED black-box random-search L-inf attack
(so the comparison is apples-to-apples and does not advantage differentiable
models), the attack success rate, plus clean accuracy and the AUROC of the
model's confidence margin as a per-sample vulnerability predictor.

If non-GD learners are *also* fragile and *also* margin-predictable, vulnerability
is about the data geometry, not the optimiser.
"""
import numpy as np
import torch
import torch.nn.functional as F
from . import common as C

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    _SK = True
except Exception:  # pragma: no cover
    _SK = False

DATASETS = ["fashion_mnist", "kmnist", "cifar10", "svhn"]
SEEDS = [0, 1, 2]

# (learner_id, kind, kwargs)
LEARNERS = [
    ("sgd_cnn", "torch", {"arch": "cnn", "opt": "sgd"}),
    ("adam_cnn", "torch", {"arch": "cnn", "opt": "adam", "lr": 0.001}),
    ("rmsprop_cnn", "torch", {"arch": "cnn", "opt": "rmsprop", "lr": 0.001}),
    ("adamw_cnn", "torch", {"arch": "cnn", "opt": "adamw", "lr": 0.001}),
    ("sgd_highwd_cnn", "torch", {"arch": "cnn", "opt": "sgd_highwd"}),
    ("labelsmooth_cnn", "torch", {"arch": "cnn", "opt": "sgd", "label_smooth": 0.1}),
    ("mixup_cnn", "torch", {"arch": "cnn", "opt": "sgd", "mixup": True}),
    ("advtrain_cnn", "torch", {"arch": "cnn", "opt": "sgd", "adv_train": True}),
    ("tanh_cnn", "torch", {"arch": "cnn", "opt": "sgd", "act": "tanh"}),
    ("gelu_cnn", "torch", {"arch": "cnn", "opt": "sgd", "act": "gelu"}),
    ("sgd_mlp", "torch", {"arch": "mlp", "opt": "sgd"}),
    ("adam_mlp", "torch", {"arch": "mlp", "opt": "adam", "lr": 0.001}),
    ("rfnn", "torch", {"arch": "rfnn", "opt": "adam", "lr": 0.005}),
    ("forward_forward", "ff", {}),
    ("knn", "sklearn", {"model": "knn"}),
    ("gbm", "sklearn", {"model": "gbm"}),
    ("logreg", "sklearn", {"model": "logreg"}),
]


def build_specs():
    specs = []
    for ds in DATASETS:
        for lid, kind, kw in LEARNERS:
            for s in SEEDS:
                specs.append({
                    "theme": "B_learning_rules",
                    "id": f"B_{ds}_{lid}_s{s}",
                    "params": {"dataset": ds, "learner": lid, "kind": kind,
                               "kw": kw, "seed": s},
                })
    return specs


def _eps_for(ds):
    return 0.1 if ds in ("fashion_mnist", "kmnist", "mnist") else 0.031


def _random_search_asr(predict_label, X, Y, eps, K=30, seed=0):
    """Model-agnostic black-box L-inf attack: K random perturbations, success if
    any flips an originally-correct sample. predict_label: tensor[N,...]->tensor[N] labels."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    with torch.no_grad():
        correct = predict_label(X) == Y
        flipped = torch.zeros_like(correct)
        for _ in range(K):
            noise = (torch.rand(X.shape, generator=g, device=X.device) * 2 - 1) * eps
            xa = (X + noise).clamp(0, 1)
            pred = predict_label(xa)
            flipped |= (pred != Y)
    corr = correct.cpu().numpy().astype(bool)
    fl = (flipped & correct).cpu().numpy()
    asr = float(fl[corr].mean()) if corr.sum() else float("nan")
    return asr, corr, flipped.cpu().numpy()


def run(spec):
    p = spec["params"]
    C.set_seed(p["seed"])
    ds = p["dataset"]
    meta = C.dataset_meta(ds)
    eps = _eps_for(ds)
    Xtr, Ytr, Xte, Yte = C.load_dataset(ds, n_train=6000, n_eval=1200, seed=p["seed"])
    kw = p["kw"]
    kind = p["kind"]

    clean_acc = float("nan")
    wb_asr = float("nan")     # white-box (FGSM/PGD) where available
    margin_score = None       # per-test-sample vulnerability score (higher = more vulnerable)

    if kind == "torch":
        model = C.build_model(kw["arch"], meta, act=kw.get("act", "relu"),
                              width=kw.get("width", 32 if kw["arch"] == "cnn" else 512),
                              seed=p["seed"])
        C.train_model(model, Xtr, Ytr, epochs=6, opt=kw["opt"], lr=kw.get("lr", 0.05),
                      label_smooth=kw.get("label_smooth", 0.0), mixup=kw.get("mixup", False),
                      adv_train=kw.get("adv_train", False), adv_eps=eps, ncls=meta["n_classes"])
        lg, clean_acc = C.logits_and_acc(model, Xte, Yte)
        margin = C.margin_of(lg, Yte)
        margin_score = -margin  # lower margin -> more vulnerable
        r = C.attack_success(model, Xte, Yte, attack="pgd", eps=eps, steps=7)
        wb_asr = r["asr"]

        def predict_label(x):
            with torch.no_grad():
                return model(x).argmax(1)

    elif kind == "ff":
        predict_logits = C.forward_forward_train(meta, Xtr, Ytr, Xte, Yte, epochs=6)
        with torch.no_grad():
            lg = torch.cat([predict_logits(Xte[i:i+256]) for i in range(0, Xte.size(0), 256)]).cpu()
        clean_acc = float((lg.argmax(1) == Yte.cpu()).float().mean())
        margin_score = -C.margin_of(lg, Yte)

        def predict_label(x):
            with torch.no_grad():
                outs = [predict_logits(x[i:i+256]) for i in range(0, x.size(0), 256)]
            return torch.cat(outs).argmax(1)

    elif kind == "sklearn":
        if not _SK:
            return f"sklearn unavailable for {spec['id']}", {"error": "no sklearn"}
        Xtr_np = Xtr.flatten(1).cpu().numpy()
        Ytr_np = Ytr.cpu().numpy()
        Xte_np = Xte.flatten(1).cpu().numpy()
        Yte_np = Yte.cpu().numpy()
        if kw["model"] == "knn":
            clf = KNeighborsClassifier(n_neighbors=5)
        elif kw["model"] == "gbm":
            clf = HistGradientBoostingClassifier(max_iter=120, max_depth=6, random_state=p["seed"])
        else:
            clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(Xtr_np, Ytr_np)
        pred = clf.predict(Xte_np)
        clean_acc = float((pred == Yte_np).mean())
        # confidence margin from predict_proba where available
        try:
            proba = clf.predict_proba(Xte_np)
            sort = np.sort(proba, axis=1)
            conf_margin = sort[:, -1] - sort[:, -2]
            margin_score = -conf_margin
        except Exception:
            margin_score = None

        def predict_label(x):
            xn = x.flatten(1).cpu().numpy()
            return torch.tensor(clf.predict(xn), device=x.device)
    else:
        raise ValueError(kind)

    # shared black-box attack for ALL learners
    bb_asr, corr, flipped = _random_search_asr(predict_label, Xte, Yte, eps, K=30, seed=p["seed"])

    margin_auroc = float("nan")
    if margin_score is not None and corr.sum() > 0:
        margin_auroc = C.safe_auroc(flipped[corr].astype(int), margin_score[corr])

    metrics = {
        "dataset": ds, "learner": p["learner"], "kind": kind, "seed": p["seed"],
        "eps": eps, "clean_acc": clean_acc, "whitebox_pgd_asr": wb_asr,
        "blackbox_rs_asr": bb_asr, "margin_auroc": margin_auroc,
        "uses_gradient_descent": kind in ("torch", "ff") and p["learner"] not in ("rfnn",),
    }

    lines = [
        f"Theme B | learning rule vs gradient descent | {spec['id']}",
        f"dataset={ds} learner={p['learner']} kind={kind} seed={p['seed']} eps={eps:.3f}",
        "",
        f"clean_acc           = {clean_acc:.4f}",
        f"whitebox PGD ASR    = {wb_asr if wb_asr==wb_asr else float('nan'):.4f}  (differentiable only)",
        f"blackbox RS ASR     = {bb_asr:.4f}  (shared random-search L-inf, comparable across all)",
        f"margin/conf AUROC   = {margin_auroc:.4f}  (does confidence predict which flip?)",
        "",
        "Interpretation: if non-GD learners (knn/gbm/rfnn/forward_forward) also show",
        "high blackbox ASR and margin AUROC > 0.5, adversarial vulnerability is a",
        "property of the data geometry, not of gradient descent.",
    ]
    return "\n".join(lines), metrics
