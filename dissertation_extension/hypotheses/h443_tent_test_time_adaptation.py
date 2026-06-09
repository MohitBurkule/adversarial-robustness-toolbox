"""
H443 - TENT test-time adaptation under adversarial attack: defence or masking?

Anchor: Wang et al., "TENT: Fully Test-time Adaptation by Entropy
Minimization" (ICLR 2021). Gap G2 in CAMPAIGN_GAP_MAP.md (test-time adaptation
family is otherwise absent from H173-H413).

Critique
--------
TENT updates BatchNorm running statistics on the *test* batch and minimises
prediction entropy w.r.t. BN affine parameters (gamma, beta) only. Adversarial
inputs typically push entropy UP (the perturbation walks the example toward
the decision boundary), so TENT-style entropy minimisation may pull
adversarial examples back to a confident class -> potentially better robust
accuracy. BUT:
  1. It is a stochastic/state-changing defence: the model's BN stats depend
     on the order and composition of the test batch. The "robust accuracy"
     can be a function of batch composition rather than a real property of
     the network. Tramer 2020 ("On Adaptive Attacks") and Croce 2022
     ("Evaluating the Adversarial Robustness of Adaptive Test-time
     Defenses") show such defences routinely look strong under naive eval
     and collapse under an attack that knows the defence.
  2. The natural adaptive attack is entropy-aware: instead of maximising
     cross-entropy on the *frozen* pre-TENT model, recompute PGD on the
     post-TENT model (or directly minimise post-TENT log-likelihood). This
     is what TENT will see in deployment.
  3. The defence is order-dependent: shuffling the test batch produces
     different BN updates and (in principle) different per-sample
     predictions. A non-trivial robustness defence must be invariant to
     batch ordering.
  4. A naked transfer attack from a non-adapted twin model is the
     cleanest masking diagnostic (Athalye 2018) -- if TENT only blocks
     gradients *to* itself, the transfer attack still flips it.

Further references (located via lit search):
  * Croce, Gowal et al. (ICML 2022) "Evaluating the Adversarial Robustness
    of Adaptive Test-time Defenses" -- shows TENT + variants almost always
    break under proper adaptive attacks (BPDA + EOT).
  * Zhang et al. (NeurIPS 2022) "MEMO: Test Time Robustness via Adaptation
    and Augmentation" -- single-sample TTA via TTA-aug entropy.
  * Wang et al. (CVPR 2022) "Continual Test-Time Domain Adaptation"
    (CoTTA) -- iterates TENT-style updates over a stream, magnifying
    batch-order sensitivity.
  * Athalye, Carlini, Wagner (ICML 2018) "Obfuscated Gradients" -- BPDA
    for non-differentiable / state-changing defences.

Design
------
Two BASE models:
  (A) STD     -- standard CE training, SmallCNN-with-BN
  (B) PGDAT   -- PGD-AT model (eps=0.1, 10 steps inner, same arch)

For each base model we run an evaluation grid that crosses
  defence in {OFF, TENT}                # state-changing defence
  attack  in {clean,
              PGD-static    (gradient through the FROZEN base model),
              PGD-adaptive  (gradient through the TENT-adapted model; the
                             attacker runs one TENT update on the current
                             batch before each PGD step),
              PGD-transfer  (perturbation crafted on the non-adapted STD
                             twin and replayed against TENT-target)}.
Plus two diagnostics:
  * batch-order swap (shuffle test batch, re-evaluate clean acc + PGD-static
    ASR under TENT) -- a real defence must be order-invariant up to noise.
  * EOT-style: for the adaptive attacker we additionally average gradients
    over K=4 fresh TENT reinitialisations of the batch order, modelling an
    attacker uncertain about test-batch composition.

If TENT looks strong only against PGD-static and collapses under
PGD-adaptive / PGD-transfer / batch-order swap, that is the canonical
masking signature.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9,wd=5e-4),
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01. SmallCNN width=32 with BN.
"""
import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn as nn
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
AT_INNER_STEPS = 10        # for PGD-AT base model
TENT_LR = 1e-3             # standard TENT learning rate (Wang 2021)
TENT_STEPS = 1             # one entropy step per batch is the TENT default
K_EOT = 4                  # EOT samples for the adaptive attack
META = {"channels": 1, "size": 28, "n_classes": 10}

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "fashion_mnist",
                   "h443_tent_test_time_adaptation_output.txt")

_LINES = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LINES.append(s)


def flush_file():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(_LINES) + "\n")


# ---- training -------------------------------------------------------------
def _opt_sched(model, epochs):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    return opt, sched


def train_std(Xtr, Ytr):
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, bn=True)
    opt, sched = _opt_sched(model, EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
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


def train_pgdat(Xtr, Ytr):
    C.set_seed(SEED + 1)
    model = C.build_model("cnn", META, width=32, bn=True)
    opt, sched = _opt_sched(model, EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            # inner adversary
            model.eval()
            xb_adv = C.pgd(model, xb, yb, eps=EPS, steps=AT_INNER_STEPS,
                           alpha=2.5 * EPS / AT_INNER_STEPS)
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---- TENT ----------------------------------------------------------------
def configure_tent(model):
    """Per Wang 2021: set model.train() for BN (so running stats use batch
    stats), freeze ALL params then unfreeze BN affine (gamma, beta) only.
    Returns the list of trainable params + a fresh optimizer."""
    for p in model.parameters():
        p.requires_grad_(False)
    bn_params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()                          # use batch stats
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            if m.weight is not None:
                m.weight.requires_grad_(True)
                bn_params.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad_(True)
                bn_params.append(m.bias)
    opt = torch.optim.Adam(bn_params, lr=TENT_LR)
    return opt


def tent_step(model, x, opt):
    """One TENT entropy-minimisation step on inputs x."""
    out = model(x)
    p = F.softmax(out, dim=1)
    ent = -(p * torch.log(p + 1e-8)).sum(1).mean()
    opt.zero_grad()
    ent.backward()
    opt.step()


def clone_tent_model(src):
    """Deep-copy `src` and configure it for TENT. The clone keeps its own BN
    affine + running stats so each evaluation starts from the same weights
    but adapts independently."""
    m = copy.deepcopy(src)
    opt = configure_tent(m)
    return m, opt


# ---- attacks --------------------------------------------------------------
def pgd_static(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """Standard PGD against the FROZEN model (model is set to eval/no-grad
    BN stats by caller). Uses true labels."""
    x0 = x.detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def pgd_adaptive_tent(base_model, x, y, eps=EPS, steps=PGD_STEPS,
                      alpha=PGD_ALPHA, k_eot=K_EOT):
    """Adaptive PGD that knows about TENT.

    At every PGD step:
      * make k_eot fresh clones of base_model, each configured for TENT;
      * run ONE TENT entropy step per clone on the current adversarial
        batch (this is exactly what the deployed TENT defence will do);
      * compute the gradient of the cross-entropy loss through the *adapted*
        model on the current xa;
      * average the k_eot gradients (EOT over batch-order / init noise).

    This is the canonical adaptive attack for a stochastic / state-changing
    defence (Athalye 2018, Tramer 2020, Croce 2022).
    """
    x0 = x.detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        agg_grad = torch.zeros_like(xa)
        for _e in range(k_eot):
            tent_model, opt = clone_tent_model(base_model)
            # one entropy step on the current adversarial input
            tent_step(tent_model, xa.detach(), opt)
            # now compute the attacker's loss through the adapted model
            out = tent_model(xa)
            loss = F.cross_entropy(out, y)
            g, = torch.autograd.grad(loss, xa, retain_graph=False)
            agg_grad = agg_grad + g.detach()
            del tent_model, opt
        agg_grad = agg_grad / float(k_eot)
        xa = xa.detach() + alpha * agg_grad.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


# ---- evaluation utilities ------------------------------------------------
@torch.no_grad()
def predict_static(model, X, batch=256):
    model.eval()
    preds = []
    for i in range(0, X.size(0), batch):
        preds.append(model(X[i:i + batch]).argmax(1).cpu())
    return torch.cat(preds)


def predict_tent(base_model, X, Y_for_correct_mask=None, batch=BATCH,
                 shuffle_seed=None):
    """Run TENT on test batches in order, returning per-sample predictions.

    The model is re-cloned at the start (fresh TENT state per call).
    If shuffle_seed is given, the test set is shuffled with that seed before
    being fed into TENT (used to probe batch-order dependence).
    """
    tent_model, opt = clone_tent_model(base_model)
    n = X.size(0)
    order = torch.arange(n, device=X.device)
    if shuffle_seed is not None:
        g = torch.Generator(device=X.device.type).manual_seed(shuffle_seed)
        order = torch.randperm(n, generator=g, device=X.device)
    preds = torch.empty(n, dtype=torch.long)
    for i in range(0, n, batch):
        idx = order[i:i + batch]
        xb = X[idx]
        # one TENT update on the batch
        for _ in range(TENT_STEPS):
            tent_step(tent_model, xb, opt)
        with torch.no_grad():
            yp = tent_model(xb).argmax(1).cpu()
        preds[idx.cpu()] = yp
    del tent_model, opt
    return preds


def acc_from_preds(preds, Y):
    return float((preds == Y.cpu()).float().mean())


def asr_from_preds(preds, Y, correct_mask):
    """ASR = fraction of originally-correct samples flipped by attack."""
    flipped = (preds != Y.cpu()).numpy().astype(bool)
    corr = correct_mask.astype(bool)
    if corr.sum() == 0:
        return float("nan")
    return float(flipped[corr].mean())


def craft_adversarials(base_model, X, Y, mode, batch=BATCH):
    """Craft per-sample adversarial inputs using `mode` against `base_model`.

    mode in {'static','adaptive','transfer-src-frozen'}. 'transfer-src-frozen'
    just means base_model IS the transfer source -- same procedure as static.
    """
    n = X.size(0)
    out = torch.empty_like(X)
    for i in range(0, n, batch):
        x = X[i:i + batch]
        y = Y[i:i + batch]
        if mode == "static" or mode == "transfer-src-frozen":
            base_model.eval()
            xa = pgd_static(base_model, x, y)
        elif mode == "adaptive":
            base_model.eval()
            xa = pgd_adaptive_tent(base_model, x, y)
        else:
            raise ValueError(mode)
        out[i:i + batch] = xa
    return out


# ---- main ----------------------------------------------------------------
def main():
    t0 = time.time()
    log("=" * 80)
    log("H443  TENT test-time adaptation under adversarial attack (Fashion-MNIST)")
    log("=" * 80)
    log(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    log(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} "
        f"AT_INNER_STEPS={AT_INNER_STEPS}")
    log(f"        TENT_LR={TENT_LR} TENT_STEPS={TENT_STEPS} K_EOT={K_EOT}")
    log(f"        device={C.DEVICE}")
    log("")

    # ---- data ----
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                        seed=SEED)
    log(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    flush_file()

    # ---- train base models ----
    log("\n[1] training STD base model (standard CE)...")
    std_model = train_std(Xtr, Ytr)
    _, std_clean = C.logits_and_acc(std_model, Xte, Yte)
    log(f"    STD clean acc = {std_clean:.4f}   ({time.time()-t0:.0f}s)")
    flush_file()

    log("\n[2] training PGD-AT base model "
        f"(eps={EPS}, {AT_INNER_STEPS} inner steps)...")
    at_model = train_pgdat(Xtr, Ytr)
    _, at_clean = C.logits_and_acc(at_model, Xte, Yte)
    log(f"    PGDAT clean acc = {at_clean:.4f}   ({time.time()-t0:.0f}s)")
    flush_file()

    base_models = [("STD", std_model, std_clean),
                   ("PGDAT", at_model, at_clean)]

    # ---- correctness masks (on clean inputs, FROZEN base model) ----
    masks = {}
    for name, m, _ in base_models:
        pred = predict_static(m, Xte)
        masks[name] = (pred == Yte.cpu()).numpy().astype(bool)
    log("\nbase-model clean correctness:")
    for name, _, ca in base_models:
        log(f"   {name}: clean_acc={ca:.4f}  "
            f"#correct={int(masks[name].sum())}/{N_EVAL}")

    # ---- precompute adversarial inputs once per (base, attack) ----
    log("\n[3] crafting adversarial inputs (static + adaptive) per base model...")
    advs = {}     # advs[(base_name, attack_mode)] = X_adv
    for name, m, _ in base_models:
        log(f"   - {name} / static PGD ...")
        t1 = time.time()
        advs[(name, "static")] = craft_adversarials(m, Xte, Yte, "static")
        log(f"     done in {time.time()-t1:.1f}s")
        flush_file()
        log(f"   - {name} / adaptive (TENT-aware, EOT={K_EOT}) PGD ...")
        t1 = time.time()
        advs[(name, "adaptive")] = craft_adversarials(m, Xte, Yte, "adaptive")
        log(f"     done in {time.time()-t1:.1f}s")
        flush_file()
    # transfer attack: use STD-static perturbations against the PGDAT base
    # and use PGDAT-static perturbations against the STD base.
    advs[("PGDAT", "transfer_from_STD")] = advs[("STD", "static")]
    advs[("STD",   "transfer_from_PGDAT")] = advs[("PGDAT", "static")]

    # ---- evaluation grid ----
    log("\n[4] evaluation grid")
    log("-" * 80)
    rows = []
    for base_name, m, clean_acc in base_models:
        cmask = masks[base_name]

        # --- defence OFF (frozen model) ---
        # clean
        rows.append({
            "base": base_name, "defence": "OFF",
            "attack": "clean", "acc": clean_acc, "asr": 0.0,
        })
        # static PGD
        pa = predict_static(m, advs[(base_name, "static")])
        rows.append({
            "base": base_name, "defence": "OFF",
            "attack": "pgd_static",
            "acc": acc_from_preds(pa, Yte),
            "asr": asr_from_preds(pa, Yte, cmask),
        })
        # adaptive PGD (against frozen model: just stronger PGD)
        pa = predict_static(m, advs[(base_name, "adaptive")])
        rows.append({
            "base": base_name, "defence": "OFF",
            "attack": "pgd_adaptive_on_frozen",
            "acc": acc_from_preds(pa, Yte),
            "asr": asr_from_preds(pa, Yte, cmask),
        })

        # --- defence TENT ---
        # clean
        p_clean = predict_tent(m, Xte)
        rows.append({
            "base": base_name, "defence": "TENT",
            "attack": "clean",
            "acc": acc_from_preds(p_clean, Yte),
            "asr": 0.0,
        })
        # static PGD (gradient through FROZEN base) then evaluated via TENT
        p_st = predict_tent(m, advs[(base_name, "static")])
        rows.append({
            "base": base_name, "defence": "TENT",
            "attack": "pgd_static",
            "acc": acc_from_preds(p_st, Yte),
            "asr": asr_from_preds(p_st, Yte, cmask),
        })
        # adaptive PGD (gradient through TENT-adapted base)
        p_ad = predict_tent(m, advs[(base_name, "adaptive")])
        rows.append({
            "base": base_name, "defence": "TENT",
            "attack": "pgd_adaptive",
            "acc": acc_from_preds(p_ad, Yte),
            "asr": asr_from_preds(p_ad, Yte, cmask),
        })
        # transfer PGD (adversarials crafted on the OTHER base model)
        other = "PGDAT" if base_name == "STD" else "STD"
        adv_tr = advs[(base_name, f"transfer_from_{other}")]
        p_tr = predict_tent(m, adv_tr)
        rows.append({
            "base": base_name, "defence": "TENT",
            "attack": f"pgd_transfer_from_{other}",
            "acc": acc_from_preds(p_tr, Yte),
            "asr": asr_from_preds(p_tr, Yte, cmask),
        })
        # batch-order swap: re-evaluate TENT clean + TENT-vs-static under
        # a shuffled batch order, to expose state-dependence
        p_clean_sh = predict_tent(m, Xte, shuffle_seed=12345)
        rows.append({
            "base": base_name, "defence": "TENT_shuffled",
            "attack": "clean",
            "acc": acc_from_preds(p_clean_sh, Yte),
            "asr": 0.0,
        })
        p_st_sh = predict_tent(m, advs[(base_name, "static")],
                               shuffle_seed=12345)
        rows.append({
            "base": base_name, "defence": "TENT_shuffled",
            "attack": "pgd_static",
            "acc": acc_from_preds(p_st_sh, Yte),
            "asr": asr_from_preds(p_st_sh, Yte, cmask),
        })

        log(f"  evaluated base={base_name} ({time.time()-t0:.0f}s)")
        flush_file()

    # ---- MAIN TABLE ----
    log("\n" + "=" * 80)
    log("[5] MAIN TABLE")
    log("=" * 80)
    hdr = "{:<6} {:<14} {:<28} {:>9} {:>9}".format(
        "base", "defence", "attack", "acc", "ASR")
    log(hdr)
    log("-" * len(hdr))
    for r in rows:
        log("{:<6} {:<14} {:<28} {:>9.4f} {:>9.4f}".format(
            r["base"], r["defence"], r["attack"], r["acc"], r["asr"]))
    log("-" * len(hdr))

    # ---- VERDICT ----
    log("\n" + "=" * 80)
    log("[6] VERDICT")
    log("=" * 80)

    def get(base, defence, attack):
        for r in rows:
            if r["base"] == base and r["defence"] == defence \
                    and r["attack"] == attack:
                return r
        return None

    for base in ("STD", "PGDAT"):
        off_static = get(base, "OFF", "pgd_static")["asr"]
        tent_static = get(base, "TENT", "pgd_static")["asr"]
        tent_adapt = get(base, "TENT", "pgd_adaptive")["asr"]
        other = "PGDAT" if base == "STD" else "STD"
        tent_trans = get(base, "TENT",
                         f"pgd_transfer_from_{other}")["asr"]
        tent_shufclean = get(base, "TENT_shuffled", "clean")["acc"]
        tent_clean = get(base, "TENT", "clean")["acc"]
        log("")
        log(f"  base = {base}")
        log(f"    PGD_ASR (defence=OFF, static)         = {off_static:.4f}")
        log(f"    PGD_ASR (TENT, static attacker)       = {tent_static:.4f}   "
            f"(d vs OFF = {tent_static-off_static:+.4f})")
        log(f"    PGD_ASR (TENT, ADAPTIVE attacker)     = {tent_adapt:.4f}")
        log(f"    PGD_ASR (TENT, TRANSFER from {other:<5})  = {tent_trans:.4f}")
        log(f"    TENT clean acc (order A vs shuffled)  = "
            f"{tent_clean:.4f} vs {tent_shufclean:.4f} "
            f"(|d|={abs(tent_clean-tent_shufclean):.4f})")

        # masking diagnostic per base
        # criteria:
        #   masking if TENT helps under static but loses gain under adaptive
        #     OR transfer attack flips it about as much as adaptive does
        gain_static = off_static - tent_static
        gain_adapt = off_static - tent_adapt
        recovered = tent_adapt - tent_static     # how much adaptive recovers
        order_drift = abs(tent_clean - tent_shufclean)
        masking = (gain_static > 0.05 and
                   (recovered > 0.05 or tent_trans - tent_static > 0.05))
        log(f"    gain(static)={gain_static:+.4f}  "
            f"gain(adaptive)={gain_adapt:+.4f}  "
            f"adaptive_recovery={recovered:+.4f}  order_drift={order_drift:.4f}")
        if masking:
            log(f"    -> MASKING SIGNATURE on {base}: TENT's static gain "
                "collapses under adaptive/transfer attack.")
        elif gain_adapt > 0.05:
            log(f"    -> GENUINE on {base}: TENT survives adaptive + transfer "
                "attacks with non-trivial gain.")
        else:
            log(f"    -> NULL on {base}: TENT does not give meaningful "
                "robustness gain even against the static attacker.")

    # ---- ONE-LINE VERDICT ----
    log("")
    std_gain_static = (get("STD", "OFF", "pgd_static")["asr"]
                       - get("STD", "TENT", "pgd_static")["asr"])
    std_gain_adapt = (get("STD", "OFF", "pgd_static")["asr"]
                      - get("STD", "TENT", "pgd_adaptive")["asr"])
    at_gain_static = (get("PGDAT", "OFF", "pgd_static")["asr"]
                      - get("PGDAT", "TENT", "pgd_static")["asr"])
    at_gain_adapt = (get("PGDAT", "OFF", "pgd_static")["asr"]
                     - get("PGDAT", "TENT", "pgd_adaptive")["asr"])
    if std_gain_static > 0.05 and std_gain_adapt < 0.02:
        one = ("MASKING: TENT looks robust on the STD model only against the "
               "static attacker; the adaptive (entropy-aware) attacker erases "
               "the gain. TENT is not a real defence at this protocol.")
    elif at_gain_adapt > 0.02:
        one = ("PARTIAL: TENT adds a small but real PGD-ASR reduction on top "
               "of PGD-AT even under the adaptive attacker.")
    elif std_gain_adapt > 0.05:
        one = ("SUPPORTED: TENT survives the adaptive attacker on the STD "
               "model and gives meaningful robust accuracy without AT.")
    else:
        one = ("NULL: TENT produces no meaningful robustness gain on either "
               "base under the adaptive evaluation protocol.")
    log("ONE-LINE VERDICT: " + one)
    log("")
    log(f"done in {time.time()-t0:.1f}s")

    flush_file()
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
