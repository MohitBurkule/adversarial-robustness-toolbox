"""
H397 - EOT-aware attack on a noise-augmentation model.

Hypothesis: the modest robustness of noise-augmentation training is partly an
*evaluation artefact* of attacking through a single noisy forward pass. A model
trained with input Gaussian noise has a stochastic-ish loss surface w.r.t. the
attacker's gradient; a vanilla PGD step follows one noisy gradient and can be
misled. EOT-PGD (Expectation Over Transformation, Athalye et al. 2018) averages
the gradient over K fresh noise draws per step, yielding a far more reliable
ascent direction. If EOT-PGD raises ASR substantially over vanilla PGD on the
noise model (but not on the standard control), the noise robustness was a
masking / evaluation artefact.

NOTE on determinism: this noise model adds noise during TRAINING only; at
EVALUATION the model is a fixed deterministic network (no test-time noise),
so vanilla PGD already sees a clean gradient. The EOT variant here therefore
also injects fresh input noise at attack time to model the realistic
test-time-noise defense scenario, and compares against the matched
single-sample (K=1) stochastic attack. We report both framings:
  - deterministic eval (no test noise): vanilla PGD vs EOT-PGD
  - stochastic eval (test-time noise sigma): single-draw PGD vs EOT-PGD(K)
so the masking question is answered under the regime where it can actually
arise (randomised / test-time-noise inference).

Implementation:
  * train noise-augmentation model: add fresh Gaussian noise (sigma=0.15) to
    every training batch input;
  * train a standard model as control;
  * Attacks at EPS=0.1, PGD_STEPS=10:
      (a) vanilla PGD (single forward/grad per step);
      (b) EOT-PGD: at each step average gradient over K=10 fresh-noise forward
          passes of the input.
  * Evaluate ASR of each attack on the noise model and on the standard control,
    under both deterministic and test-time-noise (randomised-defense) inference.
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
SIGMA = 0.15      # Gaussian noise std (training augmentation & test-time noise)
K_EOT = 10        # EOT samples per PGD step

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "fashion_mnist", "h397_eot_attack_noise_models_output.txt")

_LINES = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LINES.append(s)


# --- training --------------------------------------------------------------
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


def train_noise_aug(meta, Xtr, Ytr):
    """Standard CE training but each batch input gets fresh Gaussian noise."""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta, width=32, seed=0)
    opt, sched = _opt_sched(model)
    model.train()
    for ep in range(EPOCHS):
        for xb, yb in _iter_batches(Xtr, Ytr):
            xn = (xb + SIGMA * torch.randn_like(xb)).clamp(0, 1)
            out = model(xn)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model


# --- inference wrappers ----------------------------------------------------
class NoisyInference(torch.nn.Module):
    """Wrap a model so forward() adds fresh Gaussian noise (test-time-noise
    randomised defense). Used both as the attacked target and to model the
    stochastic-eval regime."""
    def __init__(self, model, sigma):
        super().__init__()
        self.model = model
        self.sigma = sigma

    def forward(self, x):
        return self.model((x + self.sigma * torch.randn_like(x)).clamp(0, 1))


# --- attacks ---------------------------------------------------------------
def pgd_vanilla(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """Standard PGD: one forward/grad per step (matches C.pgd)."""
    return C.pgd(model, x, y, eps=eps, steps=steps, alpha=alpha, random_start=True)


def pgd_eot(model, x, y, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA, K=K_EOT,
            sigma=SIGMA):
    """EOT-PGD: at each step average the loss-gradient over K fresh-noise
    forward passes of the *current* iterate (Expectation Over Transformation).
    """
    x0 = x.clone().detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        grad_acc = torch.zeros_like(xa)
        for _k in range(K):
            xn = (xa + sigma * torch.randn_like(xa)).clamp(0, 1)
            loss = F.cross_entropy(model(xn), y)
            g, = torch.autograd.grad(loss, xa, retain_graph=False)
            grad_acc = grad_acc + g
        g_mean = grad_acc / K
        xa = xa.detach() + alpha * g_mean.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


# --- evaluation ------------------------------------------------------------
@torch.no_grad()
def _eval_acc(eval_model, X, Y, batch=256, reps=1):
    """Accuracy; if reps>1 average over multiple stochastic forward passes."""
    correct = torch.zeros(X.size(0))
    for _ in range(reps):
        parts = []
        for i in range(0, X.size(0), batch):
            parts.append((eval_model(X[i:i + batch]).argmax(1) == Y[i:i + batch]).float().cpu())
        correct += torch.cat(parts)
    return correct / reps  # per-sample mean-correct in [0,1]


def asr_on(attack_target, eval_target, X, Y, attack_fn, batch=256, eval_reps=1):
    """Craft adversarials on attack_target, evaluate on eval_target.
    ASR over originally-correct samples (clean correctness measured on
    eval_target, averaged over eval_reps if stochastic)."""
    # clean correctness on eval target (majority over reps -> treat as correct
    # if mean-correct > 0.5)
    clean_meancorr = _eval_acc(eval_target, X, Y, batch, reps=eval_reps).numpy()
    corr = clean_meancorr > 0.5
    # craft adversarials in batches
    advs = []
    for i in range(0, X.size(0), batch):
        advs.append(attack_fn(attack_target, X[i:i + batch], Y[i:i + batch]))
    Xadv = torch.cat(advs)
    # flip rate on eval target (averaged over reps)
    flip_mean = 1.0 - _eval_acc(eval_target, Xadv, Y, batch, reps=eval_reps).numpy()
    asr = float(flip_mean[corr].mean()) if corr.sum() > 0 else float("nan")
    return asr, int(corr.sum())


def main():
    t0 = time.time()
    log("=" * 86)
    log("H397  EOT-aware attack on noise-augmentation model (Fashion-MNIST)")
    log(f"  N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} EPS={EPS} PGD_STEPS={PGD_STEPS} "
        f"sigma={SIGMA} K_EOT={K_EOT} device={C.DEVICE}")
    log("=" * 86)

    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    log("\n[1/3] training models ...")
    standard = train_standard(meta, Xtr, Ytr)
    _, acc_s = C.logits_and_acc(standard, Xte, Yte)
    log(f"    standard     clean_acc={acc_s:.4f}")
    noise = train_noise_aug(meta, Xtr, Ytr)
    _, acc_n = C.logits_and_acc(noise, Xte, Yte)
    log(f"    noise_aug    clean_acc={acc_n:.4f} (deterministic eval, no test noise)")

    # stochastic (test-time-noise) inference wrappers
    noise_stoch = NoisyInference(noise, SIGMA).to(C.DEVICE).eval()
    std_stoch = NoisyInference(standard, SIGMA).to(C.DEVICE).eval()
    C.set_seed(SEED + 1)
    nn_acc = _eval_acc(noise_stoch, Xte, Yte, reps=5).mean().item()
    log(f"    noise_aug    clean_acc={nn_acc:.4f} (stochastic eval, test noise sigma={SIGMA})")

    # =====================================================================
    # Regime A: DETERMINISTIC eval (no test-time noise). The eval network is
    # fixed; vanilla PGD already sees a clean gradient. EOT here just averages
    # over training-like noise -- should NOT help (control for "EOT is not
    # magically stronger on a deterministic net").
    # =====================================================================
    log("\n[2/3] Regime A: DETERMINISTIC eval (no test-time noise)")
    log("    vanilla-PGD vs EOT-PGD on the fixed deterministic networks")
    rows_A = {}
    for tag, m in [("standard", standard), ("noise_aug", noise)]:
        C.set_seed(SEED + 2)
        a_van, nA = asr_on(m, m, Xte, Yte, pgd_vanilla)
        C.set_seed(SEED + 2)
        a_eot, _ = asr_on(m, m, Xte, Yte,
                          lambda mm, x, y: pgd_eot(mm, x, y))
        rows_A[tag] = dict(van=a_van, eot=a_eot, corr=nA)
        log(f"    {tag:10s} corr={nA:4d}  PGD={a_van:.3f}  EOT-PGD={a_eot:.3f}  "
            f"(delta={a_eot-a_van:+.3f})")

    # =====================================================================
    # Regime B: STOCHASTIC eval (test-time-noise randomised defense). This is
    # where EOT is supposed to matter: the defense adds fresh noise at
    # inference, so a single-draw attack chases a noisy gradient. EOT averages
    # it out. We attack THROUGH the noisy inference wrapper.
    #   single-draw  = vanilla PGD through the noisy wrapper (K=1)
    #   EOT-PGD      = average gradient over K=10 noisy draws
    # Evaluation ASR is averaged over 5 stochastic inference reps for stability.
    # =====================================================================
    log("\n[3/3] Regime B: STOCHASTIC eval (test-time noise sigma -> randomised defense)")
    log("    single-draw PGD (K=1) vs EOT-PGD (K=10), attacking through noisy inference")
    rows_B = {}
    for tag, m, stoch in [("standard", standard, std_stoch),
                          ("noise_aug", noise, noise_stoch)]:
        C.set_seed(SEED + 3)
        a_single, nB = asr_on(stoch, stoch, Xte, Yte,
                              lambda mm, x, y: pgd_vanilla(mm, x, y),
                              eval_reps=5)
        C.set_seed(SEED + 3)
        a_eot, _ = asr_on(stoch, stoch, Xte, Yte,
                          lambda mm, x, y: pgd_eot(mm, x, y),
                          eval_reps=5)
        rows_B[tag] = dict(single=a_single, eot=a_eot, corr=nB)
        log(f"    {tag:10s} corr={nB:4d}  single-draw-PGD={a_single:.3f}  "
            f"EOT-PGD={a_eot:.3f}  (delta={a_eot-a_single:+.3f})")

    # ----- summary & verdict ------------------------------------------------
    log("\n" + "=" * 86)
    log("SUMMARY")
    log("-" * 86)
    log("Regime A (deterministic eval, no test noise):")
    log(f"{'model':10s} {'PGD':>7s} {'EOT-PGD':>8s} {'delta':>7s}")
    for tag in ("standard", "noise_aug"):
        r = rows_A[tag]
        log(f"{tag:10s} {r['van']:7.3f} {r['eot']:8.3f} {r['eot']-r['van']:+7.3f}")
    log("\nRegime B (stochastic test-time-noise defense):")
    log(f"{'model':10s} {'1-draw':>7s} {'EOT-PGD':>8s} {'delta':>7s}")
    for tag in ("standard", "noise_aug"):
        r = rows_B[tag]
        log(f"{tag:10s} {r['single']:7.3f} {r['eot']:8.3f} {r['eot']-r['single']:+7.3f}")
    log("-" * 86)

    dA = rows_A["noise_aug"]["eot"] - rows_A["noise_aug"]["van"]
    dB = rows_B["noise_aug"]["eot"] - rows_B["noise_aug"]["single"]
    dB_std = rows_B["standard"]["eot"] - rows_B["standard"]["single"]
    log("\nverdict:")
    if dB > 0.05 and dB > dB_std + 0.02:
        log(f"  EOT-PGD ERODES the noise model under randomised (test-noise) eval: "
            f"ASR +{dB:.3f} vs single-draw, far above the standard control "
            f"(+{dB_std:.3f}). The test-time-noise robustness is largely an "
            f"evaluation artefact (gradient obfuscation by stochasticity) that EOT "
            f"removes.")
    elif dA > 0.05:
        log(f"  EOT raises ASR on the deterministic noise model by +{dA:.3f}: "
            f"residual masking even without test-time noise.")
    else:
        log(f"  EOT does NOT meaningfully raise ASR on the noise model "
            f"(Regime A delta={dA:+.3f}, Regime B delta={dB:+.3f} vs std "
            f"control {dB_std:+.3f}). Any noise-training robustness here is not a "
            f"single-draw evaluation artefact -- it survives EOT.")
    log("=" * 86)

    log(f"\ntotal runtime {time.time()-t0:.1f}s")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(_LINES) + "\n")
    log(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
