"""
H490 - Linear Mode Connectivity: PGD-AT solutions live in wider, linearly
        connected basins; STD solutions do not.

Seed (Frankle et al. 2020, "Linear Mode Connectivity and the Lottery Ticket
Hypothesis", ICML): two networks trained from the same initialization with
different data orders converge to weights that are linearly connected in loss
landscape - i.e. the straight-line interpolation between them stays at low loss.
The hypothesis here is that adversarial training (PGD-AT), which is known to
produce wider/flatter minima (Wu et al. 2020, "Adversarial Weight Perturbation";
Stutz et al. 2021, "Relating Adversarially Robust Generalization to Flat
Minima"), enforces or amplifies this connectivity, while standard-trained (STD)
solutions at sub-scale may still show a non-trivial loss barrier.

Hypothesis (precise):
  For two SmallCNN F-MNIST networks trained from the SAME init but DIFFERENT
  data-order seeds:
    * PGD-AT pair: max-loss-bump (barrier) along alpha-interpolation <= 0.05
    * STD    pair: max-loss-bump (barrier) along alpha-interpolation  > 0.10

Control:
  * DIFFERENT init pair (both STD): expected disconnected (large barrier),
    confirming that the basin is init-determined (Frankle's main claim).

Caveat (sub-scale critique):
  We train at N=6000, 10 epochs - this is far smaller than Frankle's
  CIFAR/ImageNet regime. At sub-scale the two STD solutions may not yet have
  separated into distinct basins, so the STD barrier may be artificially low and
  the AT-vs-STD gap may shrink. We document this and still report the numbers;
  if the AT pair is *strictly* lower-barrier than the STD pair the effect is
  visible even at sub-scale.

Extra papers cited:
  * Garipov et al. 2018, "Loss Surfaces, Mode Connectivity, and Fast Ensembling
    of DNNs" (NeurIPS) - non-linear curves connect modes; we restrict to the
    LINEAR case here.
  * Frankle et al. 2020, "Linear Mode Connectivity and the Lottery Ticket
    Hypothesis" (ICML) - the seed paper.
  * Wortsman et al. 2022, "Model Soups" (ICML) - linear weight averaging works
    when models are linearly connected, motivating practical interest in LMC.

Protocol:
  (1) Train 2 STD pairs and 2 PGD-AT pairs, each pair sharing init but using
      different data-order seeds (same-init control).
  (2) Train 1 different-init STD pair (different-init control).
  (3) Interpolate weights at alpha in {0, 0.1, ..., 1.0}.
  (4) For each alpha, report clean accuracy, PGD-ASR, and clean loss.
  (5) Barrier = max_alpha(loss(alpha)) - 0.5*(loss(0)+loss(1)).
"""
import os, sys, time, copy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
EPS = 0.1
PGD_STEPS = 10
ALPHAS = [round(a, 2) for a in np.linspace(0.0, 1.0, 11).tolist()]
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10

# Same-init pairs use the same init_seed; the two members differ in data_seed.
# Different-init pair uses two different init_seeds.
SAME_INIT_PAIRS_STD   = [(11, 101, 102), (12, 103, 104)]   # (init, dataA, dataB)
SAME_INIT_PAIRS_AT    = [(21, 201, 202), (22, 203, 204)]
DIFF_INIT_PAIR_STD    = (31, 32, 301)                       # (initA, initB, data)


def _init_model(meta, init_seed):
    """Build a SmallCNN with deterministic init driven by init_seed."""
    C.set_seed(init_seed)
    return C.build_model("cnn", meta, seed=init_seed)


def _train_from_init(init_model, Xtr, Ytr, meta, data_seed, adv_train):
    """Clone the *initialized* model, then train with a given data-order seed."""
    model = copy.deepcopy(init_model)
    C.set_seed(data_seed)
    C.train_model(
        model, Xtr, Ytr,
        epochs=EPOCHS, opt="sgd", lr=0.05, ncls=meta["n_classes"],
        adv_train=adv_train, adv_eps=EPS, adv_steps=7,
    )
    return model


def _interp_state(sd_a, sd_b, alpha):
    """Return (1-alpha)*sd_a + alpha*sd_b for floating-point tensors;
    non-float buffers (e.g. BN num_batches_tracked) are copied from A."""
    out = {}
    for k in sd_a:
        a, b = sd_a[k], sd_b[k]
        if torch.is_floating_point(a):
            out[k] = (1.0 - alpha) * a + alpha * b
        else:
            out[k] = a.clone()
    return out


@torch.no_grad()
def _clean_loss_and_acc(model, X, Y, batch=512):
    model.eval()
    losses, correct, n = 0.0, 0, 0
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i+batch], Y[i:i+batch]
        logits = model(xb)
        losses += float(F.cross_entropy(logits, yb, reduction="sum"))
        correct += int((logits.argmax(1) == yb).sum())
        n += xb.size(0)
    return losses / n, correct / n


def _interpolate_curve(model_a, model_b, Xte, Yte, label):
    """Sweep alpha over ALPHAS; return list of dicts and barrier metrics."""
    sd_a = {k: v.detach().clone() for k, v in model_a.state_dict().items()}
    sd_b = {k: v.detach().clone() for k, v in model_b.state_dict().items()}

    # scratch model to load the interpolated weights into
    scratch = copy.deepcopy(model_a)

    curve = []
    for alpha in ALPHAS:
        scratch.load_state_dict(_interp_state(sd_a, sd_b, alpha))
        scratch.eval()
        loss, acc = _clean_loss_and_acc(scratch, Xte, Yte)
        adv = C.attack_success(scratch, Xte, Yte, attack="pgd",
                               eps=EPS, steps=PGD_STEPS, batch=256)
        curve.append({
            "alpha": alpha,
            "clean_loss": loss,
            "clean_acc": acc,
            "pgd_asr": adv["asr"],
        })

    losses = [p["clean_loss"] for p in curve]
    endpoint_mean = 0.5 * (losses[0] + losses[-1])
    barrier = max(losses) - endpoint_mean
    max_loss_bump = max(losses) - min(losses[0], losses[-1])
    return {
        "label": label,
        "curve": curve,
        "endpoint_loss_mean": endpoint_mean,
        "barrier_max_minus_endpoint_mean": barrier,
        "max_loss_bump_vs_min_endpoint": max_loss_bump,
    }


def _print_curve(block):
    print(f"\n[{block['label']}]  barrier = {block['barrier_max_minus_endpoint_mean']:+.4f}"
          f"   max-loss-bump = {block['max_loss_bump_vs_min_endpoint']:+.4f}")
    print(f"  alpha   clean_loss   clean_acc   pgd_asr")
    for p in block["curve"]:
        print(f"  {p['alpha']:.2f}    {p['clean_loss']:.4f}      "
              f"{p['clean_acc']:.3f}       {p['pgd_asr']:.3f}")


def run_pair(label, init_seed_a, init_seed_b, data_seed_a, data_seed_b,
             adv_train, Xtr, Ytr, Xte, Yte, meta):
    """Train two models (each from its own init/data seed) and interpolate.
    For same-init runs, pass init_seed_a == init_seed_b."""
    init_a = _init_model(meta, init_seed_a)
    if init_seed_a == init_seed_b:
        init_b = copy.deepcopy(init_a)            # identical init
    else:
        init_b = _init_model(meta, init_seed_b)   # different init

    t0 = time.time()
    model_a = _train_from_init(init_a, Xtr, Ytr, meta, data_seed_a, adv_train)
    model_b = _train_from_init(init_b, Xtr, Ytr, meta, data_seed_b, adv_train)
    train_s = round(time.time() - t0, 1)

    block = _interpolate_curve(model_a, model_b, Xte, Yte, label)
    block["train_runtime_s"] = train_s
    return block


def main():
    print("=" * 78)
    print("H490 - Linear Mode Connectivity of STD vs PGD-AT solutions (F-MNIST)")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  alphas={ALPHAS}")
    print(f"N_train={N_TRAIN}  N_eval={N_EVAL}  epochs={EPOCHS}")
    print("Seeds (Frankle 2020 LMC; Garipov 2018 Mode Connectivity; "
          "Wortsman 2022 Model Soups)")

    meta = C.dataset_meta(DS)
    # one fixed eval set so all curves are comparable
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=0)

    blocks = []

    # ---- 2 STD same-init pairs ----
    for k, (init, da, db) in enumerate(SAME_INIT_PAIRS_STD):
        b = run_pair(
            label=f"STD same-init pair #{k+1} (init={init}, data={da}/{db})",
            init_seed_a=init, init_seed_b=init,
            data_seed_a=da, data_seed_b=db,
            adv_train=False,
            Xtr=Xtr, Ytr=Ytr, Xte=Xte, Yte=Yte, meta=meta,
        )
        blocks.append(("std_same_init", b))
        _print_curve(b)

    # ---- 2 PGD-AT same-init pairs ----
    for k, (init, da, db) in enumerate(SAME_INIT_PAIRS_AT):
        b = run_pair(
            label=f"PGD-AT same-init pair #{k+1} (init={init}, data={da}/{db})",
            init_seed_a=init, init_seed_b=init,
            data_seed_a=da, data_seed_b=db,
            adv_train=True,
            Xtr=Xtr, Ytr=Ytr, Xte=Xte, Yte=Yte, meta=meta,
        )
        blocks.append(("at_same_init", b))
        _print_curve(b)

    # ---- different-init STD control (expected disconnected) ----
    initA, initB, data_seed = DIFF_INIT_PAIR_STD
    b = run_pair(
        label=f"STD DIFF-init pair (initA={initA}, initB={initB}, data={data_seed})",
        init_seed_a=initA, init_seed_b=initB,
        data_seed_a=data_seed, data_seed_b=data_seed,
        adv_train=False,
        Xtr=Xtr, Ytr=Ytr, Xte=Xte, Yte=Yte, meta=meta,
    )
    blocks.append(("std_diff_init", b))
    _print_curve(b)

    # ---- aggregate ----
    def mean_barrier(tag):
        vals = [bl["barrier_max_minus_endpoint_mean"] for t, bl in blocks if t == tag]
        return float(np.mean(vals)) if vals else float("nan")

    std_same_bar = mean_barrier("std_same_init")
    at_same_bar  = mean_barrier("at_same_init")
    std_diff_bar = mean_barrier("std_diff_init")

    print("\n" + "=" * 78)
    print("BARRIER SUMMARY (max clean-loss along alpha minus endpoint mean)")
    print("=" * 78)
    print(f"  STD  same-init  mean barrier : {std_same_bar:+.4f}   (expect > 0.10 per H)")
    print(f"  AT   same-init  mean barrier : {at_same_bar:+.4f}    (expect <= 0.05 per H)")
    print(f"  STD  DIFF-init  barrier      : {std_diff_bar:+.4f}   (expect disconnected, large)")

    # ---- HEADLINE verdict ----
    H_AT_LOW   = at_same_bar  <= 0.05
    H_STD_HIGH = std_same_bar  > 0.10
    H_DIFF_BIG = std_diff_bar  > std_same_bar
    H_AT_VS_STD = at_same_bar < std_same_bar

    if H_AT_LOW and H_STD_HIGH:
        verdict = ("SUPPORTED: PGD-AT same-init pairs are linearly connected "
                   "(<=0.05 barrier) while STD same-init pairs are not (>0.10).")
    elif H_AT_VS_STD and at_same_bar < 0.10:
        verdict = ("PARTIALLY SUPPORTED: AT barrier is strictly smaller than STD "
                   "barrier, but the exact 0.05/0.10 thresholds were not both met. "
                   "Sub-scale caveat applies (N=6000, 10 epochs may not have fully "
                   "separated STD basins or fully widened AT basins).")
    elif std_same_bar <= 0.10 and at_same_bar <= 0.05:
        verdict = ("INCONCLUSIVE (sub-scale): both STD and AT same-init pairs are "
                   "linearly connected -- consistent with the critique that at "
                   "N=6000/10ep the STD pair has not yet separated into distinct "
                   "basins, so there is no AT-vs-STD gap to detect.")
    else:
        verdict = ("REFUTED at this scale: STD and AT same-init barriers do not "
                   "follow the predicted pattern (AT lower, STD higher).")

    diff_note = ("DIFF-init control behaves as expected (disconnected: barrier > "
                 "same-init STD)." if H_DIFF_BIG else
                 "DIFF-init control DOES NOT show a larger barrier than same-init "
                 "STD - the basin-by-init claim itself is not visible at this scale.")

    print("\n" + "=" * 78)
    print("HEADLINE VERDICT")
    print("=" * 78)
    print(f"  {verdict}")
    print(f"  {diff_note}")
    print()
    print("Caveat: sub-scale (N=6000, 10 epochs) regime. Frankle 2020 reports LMC")
    print("at CIFAR/ImageNet scale; at our scale STD pairs may already share a")
    print("basin, suppressing the STD>AT barrier gap.  See Garipov 2018 (non-linear")
    print("connectivity), Frankle 2020 (linear connectivity & lottery tickets), and")
    print("Wortsman 2022 (Model Soups) for the practical relevance of LMC.")
    print("=" * 78)


if __name__ == "__main__":
    main()
