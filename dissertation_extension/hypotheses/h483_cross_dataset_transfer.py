"""
H483 - External (out-of-distribution) transfer of adversarial training:
does Fashion-MNIST PGD-AT buy ANY robustness on MNIST / KMNIST?

Seed (dissertation extension paper, paper #6, sections 5 & 2 M9). We train a
SmallCNN on Fashion-MNIST under (a) standard ERM and (b) PGD-AT at L-inf
eps=0.1. We then evaluate both models OUT-OF-DOMAIN on MNIST and KMNIST -- two
other 28x28 greyscale 10-class datasets -- and ask whether AT's robustness
benefit on F-MNIST transfers to those neighbouring domains.

Critique answered: the F-MNIST classifier has no notion of "correct" MNIST or
KMNIST labels, so we cannot measure ordinary robust accuracy out of domain.
Instead we use TWO label-free stability metrics on the foreign datasets:

  * PGD prediction stability:    fraction of foreign inputs whose argmax
                                 (under the F-MNIST classifier) is unchanged
                                 by an untargeted PGD attack at the same
                                 L-inf eps as training (0.1, 10 steps). The
                                 attack uses the model's own clean argmax as
                                 the "true" label, exactly the standard
                                 white-box setup. High stability = robust
                                 prediction surface on that domain.
  * Gaussian noise stability:    fraction of foreign inputs whose argmax is
                                 unchanged under additive Gaussian noise of
                                 sigma=0.1. A purely random-perturbation
                                 control that is unaffected by gradient
                                 masking.

We also (in domain) report clean accuracy and PGD robust accuracy on
F-MNIST to anchor what we are comparing against.

Hypothesis (M9): AT buys substantial in-domain PGD robust-accuracy gain on
F-MNIST, but its PGD-stability gain on MNIST/KMNIST is far smaller (close to
the STD baseline) -- i.e. AT learns dataset-specific robust features, not a
generic flat surface. Clean accuracy / clean-prediction overlap may transfer
better than robustness, replaying the classical "robust features != useful
features" pattern.

Controls / additional measurements:
  (1) datasets: F-MNIST (train+test, in-domain), MNIST, KMNIST
      (torchvision; download to TORCH_HOME via campaign.common.load_dataset
      which already sets TORCH_HOME/_CACHE).
  (2) models: STD vs PGD-AT (eps=0.1, 7 steps, matching campaign.common
      defaults).
  (3) per source-class breakdown of PGD stability on each foreign dataset
      (bucketed by the F-MNIST model's clean argmax in {0..9}).
  (4) domain-distance probe: mean L2 between F-MNIST and foreign-dataset
      class-conditional means in the penultimate feature space of the
      F-MNIST STD model. This contextualises how "far" each foreign domain
      lives from the source manifold in feature space.
  (5) STD-to-AT transfer comparison (rather than a single number) to
      isolate the AT effect from the architectural baseline.

Literature anchors:
  * Madry et al. 2018, "Towards Deep Learning Models Resistant to
    Adversarial Attacks" -- defines PGD-AT we use.
  * Tsipras et al. 2019, "Robustness May Be at Odds with Accuracy" -- AT
    learns a different (more semantic) feature set; this predicts that
    robust features are dataset-specific.
  * Salman et al. 2020, "Do Adversarially Robust ImageNet Models Transfer
    Better?" -- robust ImageNet features transfer better as *initialisations
    for downstream FINETUNING*. We test the strictly harder zero-shot
    transfer regime: no foreign-domain training labels are used.
  * Utrera et al. 2021, "Adversarially-Trained Deep Nets Transfer Better:
    Illustrating the Catastrophic Forgetting of Robust Features" -- robust
    representations are richer but can lose their robustness off-domain;
    directly motivates our prediction.

Run-only spec: code is written, NOT executed.
Output file (when run): results/fashion_mnist/h483_cross_dataset_transfer_output.txt
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
SRC = "fashion_mnist"
FOREIGN = ["mnist", "kmnist"]
SEEDS = [0, 1, 2]
EPS = 0.1                  # L-inf budget for AT + eval (matches campaign)
PGD_STEPS = 10
AT_STEPS = 7               # campaign default for adv_train
NOISE_SIGMA = 0.1          # Gaussian noise std for stability control
N_TRAIN = 8000             # subset of F-MNIST train (keeps cost low; campaign norm)
N_EVAL = 2000              # per-dataset eval sample (campaign norm)
EPOCHS = 8
NCLS = 10

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
OUT_PATH = os.path.join(OUT_DIR, "h483_cross_dataset_transfer_output.txt")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def train_one(adv, seed):
    """Train a SmallCNN on F-MNIST, optionally with PGD-AT."""
    C.set_seed(seed)
    Xtr, Ytr, Xte, Yte = C.load_dataset(SRC, n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)
    meta = C.dataset_meta(SRC)
    model = C.build_model("cnn", meta, width=32, seed=seed)
    C.train_model(model, Xtr, Ytr, epochs=EPOCHS, opt="adam", lr=2e-3,
                  ncls=NCLS, adv_train=adv, adv_eps=EPS, adv_steps=AT_STEPS)
    return model, Xte, Yte


@torch.no_grad()
def clean_argmax(model, X, batch=512):
    """Argmax of clean logits (used both as labels for PGD stability and as
    'predicted source class' bucket for the per-class breakdown)."""
    parts = []
    for i in range(0, X.size(0), batch):
        parts.append(model(X[i:i + batch]).argmax(1).cpu())
    return torch.cat(parts)


def pgd_stability(model, X, eps=EPS, steps=PGD_STEPS, batch=256):
    """Fraction of inputs whose argmax is unchanged by untargeted PGD that
    treats the clean argmax as the 'true' label. Returns (stability_rate,
    per-sample stable bool vector, clean_argmax vector).

    Note: untargeted attack maximises CE loss against the model's own clean
    argmax. Standard white-box stability metric (does not require ground
    truth, so it works on foreign datasets)."""
    model.eval()
    yhat = clean_argmax(model, X, batch=batch).to(X.device)
    stable = []
    for i in range(0, X.size(0), batch):
        x = X[i:i + batch]
        y = yhat[i:i + batch]
        xa = C.pgd(model, x, y, eps=eps, steps=steps)
        with torch.no_grad():
            yp = model(xa).argmax(1)
        stable.append((yp == y).cpu())
    stable = torch.cat(stable)
    return float(stable.float().mean()), stable.numpy(), yhat.cpu().numpy()


def noise_stability(model, X, sigma=NOISE_SIGMA, batch=512, seed=0):
    """Fraction of inputs whose argmax is unchanged under Gaussian noise of
    std sigma (clamped to [0,1]). Gradient-free control."""
    model.eval()
    yhat = clean_argmax(model, X, batch=batch).to(X.device)
    g = torch.Generator(device=X.device).manual_seed(seed)
    stable = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            x = X[i:i + batch]
            y = yhat[i:i + batch]
            noise = torch.randn(x.shape, generator=g, device=x.device) * sigma
            xa = (x + noise).clamp(0, 1)
            yp = model(xa).argmax(1)
            stable.append((yp == y).cpu())
    stable = torch.cat(stable)
    return float(stable.float().mean())


def per_class_stability(stable_bool, yhat_np, ncls=NCLS):
    """Group stability-under-PGD by the source-model's clean argmax."""
    out = {}
    for c in range(ncls):
        mask = yhat_np == c
        if mask.sum() == 0:
            out[c] = (0, float("nan"))
        else:
            out[c] = (int(mask.sum()), float(stable_bool[mask].mean()))
    return out


@torch.no_grad()
def penult_features(model, X, batch=512):
    """Penultimate-layer features from SmallCNN. We use the post-conv
    flattened activations (just before head.Linear(.,256)) as the 'feature
    space' in which to measure domain distance. SmallCNN.head[0] is Flatten,
    head[1] is Linear(width*4*feat*feat, 256), head[2] is activation,
    head[3] is Linear(256, n_classes). So features after .features() and
    Flatten() are a stable per-input embedding."""
    feats = []
    flatten = torch.nn.Flatten()
    for i in range(0, X.size(0), batch):
        z = model.features(X[i:i + batch])
        feats.append(flatten(z).cpu())
    return torch.cat(feats, 0)


def class_conditional_means(feats, labels, ncls=NCLS):
    """Return a (ncls, D) tensor of per-class feature means; classes that
    don't appear get an NaN row."""
    D = feats.size(1)
    out = torch.full((ncls, D), float("nan"))
    for c in range(ncls):
        mask = labels == c
        if mask.any():
            out[c] = feats[mask].mean(0)
    return out


def domain_distance(model, X_src, Y_src, X_for, Y_for_clean_argmax):
    """Mean L2 distance between per-class means of source (true labels) and
    foreign (model's clean argmax buckets). Foreign 'class' = source-model
    prediction class. Skips classes where either side is empty."""
    f_src = penult_features(model, X_src)
    f_for = penult_features(model, X_for)
    mu_src = class_conditional_means(f_src, Y_src.cpu())
    mu_for = class_conditional_means(f_for, torch.tensor(Y_for_clean_argmax))
    ds = []
    for c in range(NCLS):
        if torch.isnan(mu_src[c]).any() or torch.isnan(mu_for[c]).any():
            continue
        ds.append(float((mu_src[c] - mu_for[c]).norm()))
    if not ds:
        return float("nan")
    return float(np.mean(ds))


@torch.no_grad()
def acc(model, X, Y, batch=512):
    parts = []
    for i in range(0, X.size(0), batch):
        parts.append((model(X[i:i + batch]).argmax(1) == Y[i:i + batch]).cpu())
    return float(torch.cat(parts).float().mean())


def robust_acc(model, X, Y, eps=EPS, steps=PGD_STEPS, batch=256):
    """In-domain PGD robust accuracy (fraction of inputs whose adv argmax
    matches the TRUE label)."""
    model.eval()
    out = []
    for i in range(0, X.size(0), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        xa = C.pgd(model, x, y, eps=eps, steps=steps)
        with torch.no_grad():
            out.append((model(xa).argmax(1) == y).cpu())
    return float(torch.cat(out).float().mean())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_seed(seed):
    res = {"seed": seed}
    for tag, adv in [("STD", False), ("AT", True)]:
        t0 = time.time()
        model, Xte_src, Yte_src = train_one(adv, seed)
        # in-domain anchors
        clean_src = acc(model, Xte_src, Yte_src)
        rob_src = robust_acc(model, Xte_src, Yte_src)
        d = {"train_s": round(time.time() - t0, 1),
             "src_clean_acc": clean_src,
             "src_robust_acc": rob_src,
             "foreign": {}}

        for fds in FOREIGN:
            _, _, Xfe, Yfe = C.load_dataset(fds, n_eval=N_EVAL, seed=seed)
            t1 = time.time()
            pgd_stab, stable_vec, yhat_np = pgd_stability(model, Xfe)
            ns = noise_stability(model, Xfe, seed=seed + 1)
            per_cls = per_class_stability(stable_vec, yhat_np)
            dd = domain_distance(model, Xte_src, Yte_src, Xfe, yhat_np)
            d["foreign"][fds] = {
                "pgd_stability": pgd_stab,
                "noise_stability": ns,
                "per_class_stability": per_cls,
                "domain_distance": dd,
                "eval_s": round(time.time() - t1, 1),
            }
        res[tag] = d
    return res


def fmt_per_class(per_cls):
    parts = []
    for c in range(NCLS):
        n, s = per_cls[c]
        parts.append(f"c{c}(n={n}):{s:.2f}" if n else f"c{c}(n=0):--")
    return " ".join(parts)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    header = (
        "=" * 78 + "\n"
        "H483 - External transfer of PGD-AT: F-MNIST -> MNIST / KMNIST\n"
        + "=" * 78 + "\n"
        f"Device={C.DEVICE}  src={SRC}  foreign={FOREIGN}  seeds={SEEDS}\n"
        f"eps={EPS}  pgd_steps={PGD_STEPS}  at_steps={AT_STEPS}  "
        f"sigma={NOISE_SIGMA}\n"
        f"n_train={N_TRAIN}  n_eval={N_EVAL}  epochs={EPOCHS}\n"
        + "=" * 78 + "\n"
    )
    print(header, end="")

    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["wall_s"] = round(time.time() - t0, 1)
        rows.append(r)
        for tag in ("STD", "AT"):
            d = r[tag]
            print(f"\n[seed={s} {tag}] train={d['train_s']}s  "
                  f"src_clean={d['src_clean_acc']:.3f}  "
                  f"src_robust={d['src_robust_acc']:.3f}")
            for fds in FOREIGN:
                f = d["foreign"][fds]
                print(f"  {fds:<7} PGD-stab={f['pgd_stability']:.3f}  "
                      f"Noise-stab={f['noise_stability']:.3f}  "
                      f"dom-dist={f['domain_distance']:.2f}  "
                      f"(eval={f['eval_s']}s)")
                print(f"    per-class: {fmt_per_class(f['per_class_stability'])}")

    # aggregate across seeds
    def mean(vals):
        v = [x for x in vals if x == x]
        return sum(v) / len(v) if v else float("nan")

    print("\n" + "=" * 78)
    print("MEANS across seeds")
    print("=" * 78)
    for tag in ("STD", "AT"):
        clean = mean([r[tag]["src_clean_acc"] for r in rows])
        rob = mean([r[tag]["src_robust_acc"] for r in rows])
        print(f"\n {tag}:  src_clean={clean:.3f}  src_robust={rob:.3f}")
        for fds in FOREIGN:
            ps = mean([r[tag]["foreign"][fds]["pgd_stability"] for r in rows])
            ns = mean([r[tag]["foreign"][fds]["noise_stability"] for r in rows])
            dd = mean([r[tag]["foreign"][fds]["domain_distance"] for r in rows])
            print(f"   {fds:<7} PGD-stab={ps:.3f}  Noise-stab={ns:.3f}  "
                  f"dom-dist={dd:.2f}")

    # contrasts: AT - STD on each metric
    print("\n" + "=" * 78)
    print("AT MINUS STD  (positive = AT helps)")
    print("=" * 78)
    d_clean = (mean([r["AT"]["src_clean_acc"] for r in rows])
               - mean([r["STD"]["src_clean_acc"] for r in rows]))
    d_rob = (mean([r["AT"]["src_robust_acc"] for r in rows])
             - mean([r["STD"]["src_robust_acc"] for r in rows]))
    print(f"  src clean acc   delta = {d_clean:+.3f}")
    print(f"  src robust acc  delta = {d_rob:+.3f}  <-- AT's in-domain win")
    for fds in FOREIGN:
        d_ps = (mean([r["AT"]["foreign"][fds]["pgd_stability"] for r in rows])
                - mean([r["STD"]["foreign"][fds]["pgd_stability"] for r in rows]))
        d_ns = (mean([r["AT"]["foreign"][fds]["noise_stability"] for r in rows])
                - mean([r["STD"]["foreign"][fds]["noise_stability"] for r in rows]))
        print(f"  {fds:<7} PGD-stab delta   = {d_ps:+.3f}")
        print(f"  {fds:<7} Noise-stab delta = {d_ns:+.3f}")

    # headline verdict
    print("\n" + "=" * 78)
    print("HEADLINE VERDICT")
    print("=" * 78)
    print("If AT's PGD-stability delta on MNIST/KMNIST is << its in-domain robust-")
    print("acc delta on F-MNIST (and noise-stability delta is similarly small),")
    print("PGD-AT learns dataset-specific robust features and does NOT confer")
    print("generic flatness on neighbouring 28x28 greyscale domains. This")
    print("supports the M9 claim (Tsipras 2019 / Utrera 2021 prior art): robust")
    print("features are dataset-specific. If, instead, the foreign PGD-stability")
    print("delta is comparable to the in-domain robustness delta, AT generalises")
    print("its flatness and would refute the seed hypothesis (Salman 2020-style")
    print("'robust features transfer' would hold even zero-shot).")
    print("=" * 78)


if __name__ == "__main__":
    main()
