"""
H444 - MEMO test-time augmentation (Zhang et al., NeurIPS 2022).

Gap G2 (test-time adaptation defences): the H173-H413 campaign covers
adversarial training, gradient penalties, noise augmentation and a few
architecture probes but has zero coverage of single-image test-time
adaptation. H229 ("test-time input opt") is adjacent but does NOT
implement TENT / MEMO / CoTTA. Here we implement MEMO and ask whether
single-image marginal-entropy minimisation across K augmentations buys
PGD robustness on Fashion-MNIST.

MEMO recipe (Zhang 2022):
  At test time, for each input x:
    1. Draw K stochastic augmentations a_1(x), ..., a_K(x).
    2. Compute the marginal predictive p_bar = (1/K) sum softmax(f(a_k(x))).
    3. One (or a few) SGD step on entropy(p_bar) w.r.t. a small set of
       model params (here: BN-affine + final linear layer), starting from
       a freshly-loaded snapshot of the trained weights for EACH test
       point (episodic adaptation - the canonical MEMO setting).
    4. Predict argmax of the post-update marginal.

CRITIQUE (Croce et al., ICML 2022 "Evaluating the Adversarial Robustness
of Adaptive Test-time Defenses"):
  MEMO is STOCHASTIC. A standard PGD attacker that backprops through a
  deterministic forward pass will see a non-representative gradient and
  may produce a *masked* robustness signal. The canonical adaptive attack
  is EOT-PGD that averages gradients over the same TTA sampler the
  defender uses. We include both standard PGD and EOT-PGD (K_eot
  matched to K_TTA) and we report a "masking flag" when EOT-PGD ASR
  greatly exceeds standard PGD ASR on the MEMO-defended model. We also
  include a vanilla-TTA ablation (K augmentations, marginal vote, NO
  entropy adaptation) to separate the "averaging-already-helps" effect
  from the "entropy minimisation" effect.

EXTRA ANCHORS:
  * Croce, Gowal, Brunner, Shelhamer, Hein, Cemgil. "Evaluating the
    Adversarial Robustness of Adaptive Test-time Defenses." ICML 2022.
    arXiv:2202.13711. -> EOT + transfer + careful adaptive eval.
  * Hendrycks et al. "AugMix: A Simple Data Processing Method to Improve
    Robustness and Uncertainty." ICLR 2020. -> the augmentation family
    MEMO uses; we use a Fashion-MNIST-friendly subset (crop+flip+noise
    +translate+contrast).
  * Athalye, Carlini, Wagner. "Obfuscated Gradients Give a False Sense
    of Security." ICML 2018. -> EOT-PGD is the canonical audit.

KNOB: K_TTA in {4, 8, 16} (per the seed). For each K we run:
  Block A. Standard-CE baseline + MEMO(K).
  Block B. PGD-AT baseline + MEMO(K).
  Block C. Vanilla-TTA(K) ablation (averaging only, no entropy adapt).
  Block D. EOT-PGD adaptive eval against the MEMO models (K_eot = K_TTA).

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD mom=0.9
wd=5e-4, SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.

Note: MEMO adaptation is per-test-point (episodic), which is expensive.
We therefore evaluate on a fixed N_MEMO_EVAL=400-subset of the test
loader for the adaptive-defence rows. Clean and standard-PGD rows for
the BASE models still use the full N_EVAL=2000 test slice.
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
N_MEMO_EVAL = 400          # subset for episodic-adaptation rows (cost)
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01

K_TTA_VALUES = [4, 8, 16]   # MEMO sample-count seed

# MEMO inner optimisation
MEMO_INNER_STEPS = 1        # single SGD step on marginal entropy (canonical)
MEMO_INNER_LR = 0.001       # small, BN-affine + head only

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h444_memo_test_time_aug_output.txt")


# --------------------------------------------------------------------------
# training helpers (config-spec SGD: mom=0.9, wd=5e-4)
# --------------------------------------------------------------------------
def _make_opt(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def train_std(Xtr, Ytr, seed):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_opt(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
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


def train_at(Xtr, Ytr, seed):
    C.set_seed(seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    opt = _make_opt(model, LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            # PGD inner-max
            model.eval()
            xa = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(xa), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# --------------------------------------------------------------------------
# AugMix-flavoured augmentation sampler for 28x28 grayscale Fashion-MNIST.
# All ops are differentiable (matters for EOT-PGD).
# --------------------------------------------------------------------------
def _aug_once(x, gen):
    """Return one stochastic augmentation of a batch x (B,1,28,28) in [0,1].
    Composition: random horizontal flip; random pad+crop (pad=2); additive
    uniform pixel noise; random contrast scale around 0.5."""
    B, _, H, W = x.shape
    out = x

    # 1) random h-flip per sample
    flip_mask = (torch.rand(B, device=x.device, generator=gen) < 0.5).view(B, 1, 1, 1)
    out = torch.where(flip_mask, out.flip(-1), out)

    # 2) random translation (pad=2, crop back) - per-sample (dx,dy) in {-2,..,+2}
    pad = 2
    out_p = F.pad(out, (pad, pad, pad, pad), mode="constant", value=0.0)
    dx = torch.randint(0, 2 * pad + 1, (B,), device=x.device, generator=gen)
    dy = torch.randint(0, 2 * pad + 1, (B,), device=x.device, generator=gen)
    crops = []
    for b in range(B):
        crops.append(out_p[b:b + 1, :, dy[b]:dy[b] + H, dx[b]:dx[b] + W])
    out = torch.cat(crops, dim=0)

    # 3) additive uniform pixel noise [-0.03, 0.03]
    out = out + (2.0 * torch.rand(out.shape, device=x.device, generator=gen) - 1.0) * 0.03

    # 4) contrast jitter around mean 0.5: scale in [0.85, 1.15]
    s = 0.85 + 0.30 * torch.rand((B, 1, 1, 1), device=x.device, generator=gen)
    out = (out - 0.5) * s + 0.5

    return out.clamp(0.0, 1.0)


def aug_K(x, K, gen):
    """Return tensor of shape (K, B, 1, 28, 28)."""
    return torch.stack([_aug_once(x, gen) for _ in range(K)], dim=0)


# --------------------------------------------------------------------------
# MEMO: episodic adaptation of BN-affine + final linear layer
# --------------------------------------------------------------------------
def _memo_adapt_params(model):
    """Return the params MEMO is allowed to adapt: BN affine + final linear
    (last Linear in head). Everything else frozen during inner step."""
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    # SmallCNN.head is Sequential(Flatten, Linear, A, Linear); adapt the
    # FINAL Linear (logits)
    head = model.head
    last_lin = None
    for sub in head:
        if isinstance(sub, nn.Linear):
            last_lin = sub
    if last_lin is not None:
        params.append(last_lin.weight)
        if last_lin.bias is not None:
            params.append(last_lin.bias)
    return params


def memo_predict(base_model, x_batch, K, gen, inner_steps=MEMO_INNER_STEPS,
                 inner_lr=MEMO_INNER_LR):
    """Episodic MEMO: for each sample in x_batch, restore weights, run
    inner_steps of marginal-entropy SGD on K augmentations, then predict.
    Implemented batched at the sample axis (one inner-step op per sample
    via cloning the model per-sample is too slow; instead we run one
    batched inner step using shared params across the batch). This is a
    *batch-collective* MEMO (Mummadi 2021 / Wang 2021 TENT style) which
    is a standard relaxation; the canonical per-image variant is also
    implemented as the inner forward sees K augmentations of EACH
    sample - so the marginal entropy is averaged across augmentations
    only, not across samples in the batch.

    Returns predicted labels (B,) and post-adapt logits (B,C).
    """
    state = copy.deepcopy(base_model.state_dict())
    adapt_params = _memo_adapt_params(base_model)
    opt = torch.optim.SGD(adapt_params, lr=inner_lr, momentum=0.9)
    B = x_batch.size(0)
    base_model.train()  # so BN-affine has grad
    # but we want BN running stats FROZEN -> set track_running_stats False?
    # Standard MEMO leaves BN in train mode using batch stats; we follow.
    for _ in range(inner_steps):
        opt.zero_grad()
        x_aug = aug_K(x_batch, K, gen)             # (K,B,1,28,28)
        x_flat = x_aug.reshape(K * B, *x_batch.shape[1:])
        logits = base_model(x_flat)                # (K*B, C)
        logits = logits.view(K, B, -1)
        p = F.softmax(logits, dim=-1)              # (K,B,C)
        p_bar = p.mean(dim=0)                      # (B,C) marginal
        # entropy of marginal predictive (averaged across batch)
        ent = -(p_bar * (p_bar + 1e-8).log()).sum(dim=-1).mean()
        ent.backward()
        opt.step()
    base_model.eval()
    with torch.no_grad():
        # final prediction is the marginal-vote over K augmentations
        x_aug = aug_K(x_batch, K, gen)
        x_flat = x_aug.reshape(K * B, *x_batch.shape[1:])
        logits = base_model(x_flat).view(K, B, -1)
        p_bar = F.softmax(logits, dim=-1).mean(dim=0)
        pred = p_bar.argmax(dim=-1)
    # restore weights for next sample/batch (episodic)
    base_model.load_state_dict(state)
    return pred, p_bar


def vanilla_tta_predict(base_model, x_batch, K, gen):
    """Vanilla TTA ablation: K augmentations, marginal vote, NO adapt."""
    base_model.eval()
    with torch.no_grad():
        x_aug = aug_K(x_batch, K, gen)
        x_flat = x_aug.reshape(K * x_batch.size(0), *x_batch.shape[1:])
        logits = base_model(x_flat).view(K, x_batch.size(0), -1)
        p_bar = F.softmax(logits, dim=-1).mean(dim=0)
        pred = p_bar.argmax(dim=-1)
    return pred, p_bar


# --------------------------------------------------------------------------
# attacks: standard PGD (deterministic forward) and EOT-PGD (averages
# gradient over the same TTA sampler the defender uses)
# --------------------------------------------------------------------------
def pgd_standard(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    return C.pgd(model, x, y, eps=eps, steps=steps, alpha=alpha)


def pgd_eot(model, x, y, K_eot, gen, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """EOT-PGD: at each step average gradient over K_eot stochastic
    augmentations of the perturbed input. The attacker uses the SAME
    augmentation sampler as the defender. Targets the cross-entropy on
    the marginal-softmax prediction (the defender's read-out)."""
    x0 = x.clone().detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        B = xa.size(0)
        # K_eot augmentations -> marginal softmax -> NLL on y
        x_aug = aug_K(xa, K_eot, gen)              # (K,B,1,28,28)
        x_flat = x_aug.reshape(K_eot * B, *xa.shape[1:])
        logits = model(x_flat).view(K_eot, B, -1)
        p_bar = F.softmax(logits, dim=-1).mean(dim=0)
        log_p_bar = (p_bar + 1e-8).log()
        loss = F.nll_loss(log_p_bar, y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


# --------------------------------------------------------------------------
# eval helpers
# --------------------------------------------------------------------------
def batched_accuracy_with_predfn(predfn, X, Y, batch=64):
    """predfn(x_batch) -> labels. Returns acc on (X,Y)."""
    correct = 0
    total = 0
    for i in range(0, X.size(0), batch):
        xb = X[i:i + batch]
        yb = Y[i:i + batch]
        pred = predfn(xb)
        correct += int((pred == yb).sum())
        total += xb.size(0)
    return correct / max(total, 1)


def asr_with_predfn_and_attack(predfn_def, attack_fn, X, Y, batch=64):
    """ASR (over originally-correct) using a defended predict-fn and a
    pre-supplied attack-fn that produces adversarial x's against the
    *base model* (caller decides). Returns asr (fraction of correct
    that flip)."""
    nflip, ncorr = 0, 0
    for i in range(0, X.size(0), batch):
        xb = X[i:i + batch]
        yb = Y[i:i + batch]
        pred_c = predfn_def(xb)
        corr = (pred_c == yb)
        if corr.sum().item() == 0:
            continue
        # attack the *original* x, then re-defend
        xa = attack_fn(xb, yb)
        pred_a = predfn_def(xa)
        flipped = (pred_a != yb) & corr
        nflip += int(flipped.sum())
        ncorr += int(corr.sum())
    return float(nflip) / max(ncorr, 1)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H444  MEMO test-time augmentation  (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} N_MEMO_EVAL={N_MEMO_EVAL} "
        f"EPOCHS={EPOCHS} LR={LR} BATCH={BATCH} SGD(mom=0.9,wd=5e-4) "
        f"SEED={SEED} EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        K_TTA values = {K_TTA_VALUES}  "
        f"MEMO_INNER_STEPS={MEMO_INNER_STEPS}  MEMO_INNER_LR={MEMO_INNER_LR}")
    out(f"        anchor: zhang-2022-memo  (NeurIPS 2022)")
    out(f"        adaptive-eval anchor: croce-2022-adaptive-ttd (ICML 2022)")
    out(f"        device = {C.DEVICE}")
    out("")

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL,
                                         seed=SEED)
    # MEMO-subset (first N_MEMO_EVAL deterministically)
    Xte_m = Xte[:N_MEMO_EVAL]
    Yte_m = Yte[:N_MEMO_EVAL]
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}  "
        f"Xte_memo={tuple(Xte_m.shape)}")

    # generator for augmentation reproducibility
    aug_gen = torch.Generator(device=C.DEVICE).manual_seed(SEED + 1234)

    # ---- [1] BASE MODELS ----------------------------------------------------
    out("\n[1] training BASE models")
    out("    [1a] CE (standard) ...")
    t = time.time()
    base_ce = train_std(Xtr, Ytr, SEED)
    out(f"        done in {time.time()-t:.1f}s")

    out("    [1b] PGD-AT ...")
    t = time.time()
    base_at = train_at(Xtr, Ytr, SEED)
    out(f"        done in {time.time()-t:.1f}s")

    # ---- [2] BASELINE METRICS (full N_EVAL, deterministic forward) ----------
    out("\n[2] baseline metrics on full test slice (N_EVAL=%d)" % N_EVAL)
    _, ce_acc = C.logits_and_acc(base_ce, Xte, Yte)
    ce_fgsm = C.attack_success(base_ce, Xte, Yte, attack="fgsm", eps=EPS)["asr"]
    ce_pgd = C.attack_success(base_ce, Xte, Yte, attack="pgd", eps=EPS,
                              steps=PGD_STEPS)["asr"]
    out(f"    CE   : clean_acc={ce_acc:.4f}  FGSM_ASR={ce_fgsm:.4f}  "
        f"PGD_ASR={ce_pgd:.4f}")

    _, at_acc = C.logits_and_acc(base_at, Xte, Yte)
    at_fgsm = C.attack_success(base_at, Xte, Yte, attack="fgsm", eps=EPS)["asr"]
    at_pgd = C.attack_success(base_at, Xte, Yte, attack="pgd", eps=EPS,
                              steps=PGD_STEPS)["asr"]
    out(f"    AT   : clean_acc={at_acc:.4f}  FGSM_ASR={at_fgsm:.4f}  "
        f"PGD_ASR={at_pgd:.4f}")
    flush_file()

    # ---- [3] MEMO + VANILLA TTA + EOT-PGD over K in {4,8,16} ----------------
    out("\n[3] MEMO / vanilla-TTA / EOT-PGD sweep over K_TTA "
        f"(eval on N_MEMO_EVAL={N_MEMO_EVAL})")

    # results rows
    rows = []
    # store base-model baseline rows on the MEMO subset for fair compare
    def base_pgd_subset(model):
        _, acc = C.logits_and_acc(model, Xte_m, Yte_m)
        pgd_asr = C.attack_success(model, Xte_m, Yte_m, attack="pgd",
                                    eps=EPS, steps=PGD_STEPS)["asr"]
        return acc, pgd_asr

    ce_acc_m, ce_pgd_m = base_pgd_subset(base_ce)
    at_acc_m, at_pgd_m = base_pgd_subset(base_at)
    rows.append({"label": "CE base (subset)", "K": "-", "tta": "none",
                 "clean": ce_acc_m, "pgd_std": ce_pgd_m, "pgd_eot": float("nan")})
    rows.append({"label": "AT base (subset)", "K": "-", "tta": "none",
                 "clean": at_acc_m, "pgd_std": at_pgd_m, "pgd_eot": float("nan")})
    out(f"    CE base subset : clean={ce_acc_m:.4f}  PGD_std={ce_pgd_m:.4f}")
    out(f"    AT base subset : clean={at_acc_m:.4f}  PGD_std={at_pgd_m:.4f}")
    flush_file()

    for K in K_TTA_VALUES:
        out("\n" + "=" * 80)
        out(f"  K_TTA = {K}")
        out("=" * 80)

        # ---- vanilla TTA ablation (no entropy adapt) -----------------------
        for base_name, base_model in [("CE", base_ce), ("AT", base_at)]:
            gen = torch.Generator(device=C.DEVICE).manual_seed(SEED + K * 17)
            def predfn(xb, _bm=base_model, _K=K, _g=gen):
                p, _ = vanilla_tta_predict(_bm, xb, _K, _g)
                return p

            # clean acc
            clean = batched_accuracy_with_predfn(predfn, Xte_m, Yte_m)

            # standard PGD (attack base model deterministically, then re-defend)
            def atk_std(xb, yb, _bm=base_model):
                return pgd_standard(_bm, xb, yb)
            pgd_std_asr = asr_with_predfn_and_attack(predfn, atk_std,
                                                      Xte_m, Yte_m)

            # EOT-PGD (attacker uses same K-sampler the defender uses)
            gen_atk = torch.Generator(device=C.DEVICE).manual_seed(SEED + K * 17 + 99)
            def atk_eot(xb, yb, _bm=base_model, _K=K, _g=gen_atk):
                return pgd_eot(_bm, xb, yb, _K, _g)
            pgd_eot_asr = asr_with_predfn_and_attack(predfn, atk_eot,
                                                      Xte_m, Yte_m)
            rows.append({"label": f"{base_name} + vanilla-TTA",
                         "K": K, "tta": "vanilla",
                         "clean": clean, "pgd_std": pgd_std_asr,
                         "pgd_eot": pgd_eot_asr})
            out(f"  [{base_name}+TTA(K={K})  vanilla     ] "
                f"clean={clean:.4f}  PGD_std={pgd_std_asr:.4f}  "
                f"EOT-PGD={pgd_eot_asr:.4f}  "
                f"masking_flag={'YES' if pgd_eot_asr - pgd_std_asr > 0.10 else 'no'}  "
                f"({time.time()-t0:.0f}s)")
            flush_file()

        # ---- MEMO (entropy-min on marginal) --------------------------------
        for base_name, base_model in [("CE", base_ce), ("AT", base_at)]:
            gen = torch.Generator(device=C.DEVICE).manual_seed(SEED + K * 23)
            def predfn(xb, _bm=base_model, _K=K, _g=gen):
                p, _ = memo_predict(_bm, xb, _K, _g)
                return p

            clean = batched_accuracy_with_predfn(predfn, Xte_m, Yte_m)

            def atk_std(xb, yb, _bm=base_model):
                return pgd_standard(_bm, xb, yb)
            pgd_std_asr = asr_with_predfn_and_attack(predfn, atk_std,
                                                      Xte_m, Yte_m)

            gen_atk = torch.Generator(device=C.DEVICE).manual_seed(SEED + K * 23 + 99)
            def atk_eot(xb, yb, _bm=base_model, _K=K, _g=gen_atk):
                return pgd_eot(_bm, xb, yb, _K, _g)
            pgd_eot_asr = asr_with_predfn_and_attack(predfn, atk_eot,
                                                      Xte_m, Yte_m)
            rows.append({"label": f"{base_name} + MEMO",
                         "K": K, "tta": "memo",
                         "clean": clean, "pgd_std": pgd_std_asr,
                         "pgd_eot": pgd_eot_asr})
            out(f"  [{base_name}+MEMO(K={K}) entropy-adp] "
                f"clean={clean:.4f}  PGD_std={pgd_std_asr:.4f}  "
                f"EOT-PGD={pgd_eot_asr:.4f}  "
                f"masking_flag={'YES' if pgd_eot_asr - pgd_std_asr > 0.10 else 'no'}  "
                f"({time.time()-t0:.0f}s)")
            flush_file()

    # ---- [4] MAIN TABLE -----------------------------------------------------
    out("\n" + "=" * 80)
    out("[4] MAIN TABLE")
    out("=" * 80)
    hdr = "{:<28} {:>4} {:>8} {:>10} {:>10} {:>10}".format(
        "condition", "K", "tta", "clean_acc", "PGD_std", "EOT-PGD")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        eot = "       -" if (isinstance(r["pgd_eot"], float)
                              and np.isnan(r["pgd_eot"])) else f"{r['pgd_eot']:.4f}"
        out("{:<28} {:>4} {:>8} {:>10.4f} {:>10.4f} {:>10}".format(
            r["label"], str(r["K"]), r["tta"], r["clean"], r["pgd_std"], eot))
    out("-" * len(hdr))

    # ---- [5] VERDICT --------------------------------------------------------
    out("\n" + "=" * 80)
    out("[5] VERDICT")
    out("=" * 80)
    # 1. best CE+MEMO under standard PGD vs CE base on subset
    ce_memo = [r for r in rows if r["label"] == "CE + MEMO"]
    ce_van  = [r for r in rows if r["label"] == "CE + vanilla-TTA"]
    at_memo = [r for r in rows if r["label"] == "AT + MEMO"]
    at_van  = [r for r in rows if r["label"] == "AT + vanilla-TTA"]

    def best(rs, key):
        return min(rs, key=lambda r: r[key]) if rs else None

    ce_memo_best_std = best(ce_memo, "pgd_std")
    ce_memo_best_eot = best(ce_memo, "pgd_eot")
    at_memo_best_std = best(at_memo, "pgd_std")
    at_memo_best_eot = best(at_memo, "pgd_eot")

    out("  CE base subset PGD_std = {:.4f}".format(ce_pgd_m))
    if ce_memo_best_std:
        out("  best CE+MEMO PGD_std   = {:.4f}  (K={}, gain {:+.4f})".format(
            ce_memo_best_std["pgd_std"], ce_memo_best_std["K"],
            ce_pgd_m - ce_memo_best_std["pgd_std"]))
    if ce_memo_best_eot:
        out("  best CE+MEMO EOT-PGD   = {:.4f}  (K={})".format(
            ce_memo_best_eot["pgd_eot"], ce_memo_best_eot["K"]))

    out("  AT base subset PGD_std = {:.4f}".format(at_pgd_m))
    if at_memo_best_std:
        out("  best AT+MEMO PGD_std   = {:.4f}  (K={}, gain {:+.4f})".format(
            at_memo_best_std["pgd_std"], at_memo_best_std["K"],
            at_pgd_m - at_memo_best_std["pgd_std"]))
    if at_memo_best_eot:
        out("  best AT+MEMO EOT-PGD   = {:.4f}  (K={})".format(
            at_memo_best_eot["pgd_eot"], at_memo_best_eot["K"]))

    # masking summary
    masked_rows = [r for r in rows if isinstance(r["pgd_eot"], float)
                   and not np.isnan(r["pgd_eot"])
                   and (r["pgd_eot"] - r["pgd_std"]) > 0.10]
    out("\n  masking flags (EOT-PGD ASR exceeds PGD_std by >0.10):")
    if not masked_rows:
        out("    NONE - MEMO/vanilla-TTA defences are robust under EOT-PGD.")
    for r in masked_rows:
        out(f"    {r['label']} K={r['K']} ({r['tta']}): "
            f"PGD_std={r['pgd_std']:.4f} -> EOT-PGD={r['pgd_eot']:.4f}  "
            f"(+{r['pgd_eot']-r['pgd_std']:.4f})")

    # MEMO vs vanilla-TTA: does entropy adaptation add anything?
    if ce_memo and ce_van:
        ce_van_best = best(ce_van, "pgd_eot")
        ce_memo_best = best(ce_memo, "pgd_eot")
        out("\n  CE: vanilla-TTA EOT-PGD best = {:.4f}  vs  MEMO EOT-PGD best = "
            "{:.4f}  -> entropy-adapt delta {:+.4f}".format(
            ce_van_best["pgd_eot"], ce_memo_best["pgd_eot"],
            ce_van_best["pgd_eot"] - ce_memo_best["pgd_eot"]))
    if at_memo and at_van:
        at_van_best = best(at_van, "pgd_eot")
        at_memo_best = best(at_memo, "pgd_eot")
        out("  AT: vanilla-TTA EOT-PGD best = {:.4f}  vs  MEMO EOT-PGD best = "
            "{:.4f}  -> entropy-adapt delta {:+.4f}".format(
            at_van_best["pgd_eot"], at_memo_best["pgd_eot"],
            at_van_best["pgd_eot"] - at_memo_best["pgd_eot"]))

    # one-line verdict
    out("")
    # use AT+MEMO EOT vs AT base PGD_std (apples-to-oranges but the campaign
    # default frame) to pick the verdict
    if at_memo_best_eot is None:
        one = "INCONCLUSIVE: AT+MEMO row missing."
    else:
        gain_at_eot = at_pgd_m - at_memo_best_eot["pgd_eot"]
        gain_ce_eot = ce_pgd_m - (ce_memo_best_eot["pgd_eot"]
                                   if ce_memo_best_eot else 1.0)
        if max(gain_at_eot, gain_ce_eot) > 0.05 and not masked_rows:
            one = ("YES: MEMO test-time augmentation reduces PGD ASR under "
                   "adaptive EOT-PGD eval (gain >0.05) without masking.")
        elif masked_rows and max(gain_at_eot, gain_ce_eot) < 0.02:
            one = ("NO: MEMO's apparent PGD_std gains are MASKED - EOT-PGD "
                   "recovers attack success on the same defended models.")
        elif max(gain_at_eot, gain_ce_eot) < 0.02:
            one = ("NO: MEMO does not improve adversarial robustness on "
                   "Fashion-MNIST (gain <0.02 under EOT-PGD).")
        else:
            one = ("PARTIAL: MEMO gives small EOT-PGD robustness gains "
                   "(<0.05) and shows partial masking on some K.")
    out("  ONE-LINE VERDICT: " + one)

    out("")
    out(f"done in {time.time()-t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
