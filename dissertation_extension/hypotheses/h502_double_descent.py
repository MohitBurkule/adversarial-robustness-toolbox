"""
H502 - Epoch-wise robust overfitting / double descent under PGD-AT on F-MNIST.

Seed paper (Nakkiran et al. 2020, "Deep Double Descent: Where Bigger Models and
More Data Hurt") frames non-monotonic test behaviour as a function of model size,
data, AND training epochs. The canonical *empirical* observation for the
adversarially-trained regime is Rice, Wong & Kolter 2020 ("Overfitting in
Adversarially Robust Deep Learning"): under PGD adversarial training, test
robust accuracy peaks early, dips, and can subsequently rise/oscillate, while the
TRAIN robust loss is still going down -- "robust overfitting". Yu et al. 2022
("Robust Overfitting May Be Mitigated by Properly Learned Smoothening") confirm
the same phenomenon on CIFAR / SVHN / TinyImageNet and propose smoothening as
mitigation.

Hypothesis: PGD-AT on F-MNIST with a SmallCNN, trained for 30 epochs, exhibits a
clear early-epoch robust-test peak followed by a dip ("robust overfitting
valley") between roughly epoch 5 and 10, AND in some seeds an epoch-wise
double-descent re-rise later in training -- whilst the train robust loss /
train robust acc continues to improve monotonically. Best-epoch (early-stop)
robust accuracy will exceed final-epoch robust accuracy.

Cite (paper-list for the dissertation):
  * Nakkiran et al. 2020 - "Deep Double Descent: Where Bigger Models and More
    Data Hurt", ICLR 2020 (framing).
  * Rice, Wong & Kolter 2020 - "Overfitting in Adversarially Robust Deep
    Learning", ICML 2020 (canonical empirical paper).
  * Yu et al. 2022 - "Robust Overfitting May Be Mitigated by Properly Learned
    Smoothening", ICLR 2022 (replication + mitigation).

Controls (G6, per advisor critique):
  (1) Train PGD-AT for 30 epochs on F-MNIST (extends beyond the standard ~10
      that we use elsewhere in the campaign).
  (2) Record train AND test PGD attack-success rate at EVERY epoch (so both the
      robust-train trajectory and the robust-test trajectory are visible).
  (3) Record best-epoch vs final-epoch test robust accuracy (the early-stopping
      gap is the headline robust-overfitting statistic of Rice 2020).
  (4) Per-epoch margin-distribution shifts on the test set (mean and quartiles
      of correct-class minus best-other-class logit gap; tracks whether the
      decision surface is sharpening on train at the expense of test).
  (5) Compare "early-stopped" model (best test robust acc) against the final
      epoch model (same eps, same eval set), reporting the early-stop benefit.

A clear early-epoch peak with a subsequent dip in test robust acc, with train
robust acc still rising, would replicate robust overfitting on F-MNIST and -
combined with the test-margin distribution narrowing while train margins widen
- match the double-descent framing of Nakkiran.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
EPS = 0.1               # main L-inf budget (matches campaign Fashion-MNIST eps)
PGD_STEPS_TRAIN = 7     # standard AT inner steps
PGD_STEPS_EVAL = 10     # slightly stronger evaluation attack
EPOCHS = 30             # extend WELL past the standard ~10
BATCH = 128
LR = 0.05
N_TRAIN = 6000
N_EVAL = 2000
TRAIN_PROBE = 2000      # subsample of train used for the per-epoch train-PGD probe
EARLY_STOP_DIP_WINDOW = (5, 10)  # epochs where Rice 2020 sees the typical valley


def eval_clean_acc(model, X, Y, batch=512):
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            xb, yb = X[i:i + batch], Y[i:i + batch]
            n_correct += int((model(xb).argmax(1) == yb).sum().item())
    return n_correct / X.size(0)


def eval_pgd_asr(model, X, Y, eps=EPS, steps=PGD_STEPS_EVAL, batch=256):
    """Per-epoch evaluation attack: ASR over ORIGINALLY-correct samples (matches
    campaign.attack_success). Reuses C.pgd so the inner loop is identical to the
    training-time inner loop (apart from steps)."""
    model.eval()
    n_corr = 0
    n_flip = 0
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        with torch.no_grad():
            correct = model(xb).argmax(1) == yb
        if correct.sum() == 0:
            continue
        xc, yc = xb[correct], yb[correct]
        xa = C.pgd(model, xc, yc, eps, steps)
        with torch.no_grad():
            flipped = model(xa).argmax(1) != yc
        n_corr += int(correct.sum().item())
        n_flip += int(flipped.sum().item())
    return (n_flip / n_corr) if n_corr > 0 else float("nan")


def eval_margin_quartiles(model, X, Y, batch=512):
    """Per-epoch margin distribution on CLEAN inputs (correct-class logit minus
    max other-class logit). Returns (mean, q25, q50, q75)."""
    model.eval()
    logits_all = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            logits_all.append(model(X[i:i + batch]).cpu())
    logits = torch.cat(logits_all)
    m = C.margin_of(logits, Y)
    return (float(np.mean(m)), float(np.quantile(m, 0.25)),
            float(np.quantile(m, 0.5)), float(np.quantile(m, 0.75)))


def train_with_per_epoch_probes(model, Xtr, Ytr, Xtr_probe, Ytr_probe, Xte, Yte,
                                 epochs=EPOCHS, batch=BATCH, lr=LR,
                                 eps=EPS, adv_steps=PGD_STEPS_TRAIN, seed=0):
    """PGD-AT training loop with per-epoch metric collection (test robust acc,
    train robust acc on a probe split, clean accs, train robust loss, margin
    quartiles). Mirrors the structure of C.train_model with adv_train=True but
    exposes per-epoch hooks so we can log the trajectories required for G6."""
    opt = C.make_optimizer(model, "sgd", lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    trace = []
    g = torch.Generator(device=Xtr.device).manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g, device=Xtr.device)
        ep_loss_sum, ep_loss_n = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_adv = C.pgd(model, xb, yb, eps=eps, steps=adv_steps,
                            alpha=2.5 * eps / adv_steps)
            opt.zero_grad()
            out = model(xb_adv)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            ep_loss_sum += float(loss.item()) * xb.size(0)
            ep_loss_n += xb.size(0)
        sched.step()
        train_robust_loss = ep_loss_sum / max(1, ep_loss_n)

        # --- per-epoch probes --------------------------------------------------
        clean_test = eval_clean_acc(model, Xte, Yte)
        clean_train = eval_clean_acc(model, Xtr_probe, Ytr_probe)
        asr_test = eval_pgd_asr(model, Xte, Yte, eps=eps, steps=PGD_STEPS_EVAL)
        asr_train = eval_pgd_asr(model, Xtr_probe, Ytr_probe, eps=eps,
                                  steps=PGD_STEPS_EVAL)
        rob_test = (1.0 - asr_test) * clean_test if asr_test == asr_test else float("nan")
        rob_train = (1.0 - asr_train) * clean_train if asr_train == asr_train else float("nan")
        mu, q25, q50, q75 = eval_margin_quartiles(model, Xte, Yte)
        trace.append({
            "epoch": ep + 1,
            "train_robust_loss": train_robust_loss,
            "clean_train": clean_train,
            "clean_test": clean_test,
            "pgd_asr_train": asr_train,
            "pgd_asr_test": asr_test,
            "robust_acc_train": rob_train,
            "robust_acc_test": rob_test,
            "margin_test_mean": mu,
            "margin_test_q25": q25,
            "margin_test_q50": q50,
            "margin_test_q75": q75,
        })
    return trace


def snapshot_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_state(model, state):
    model.load_state_dict({k: v.to(C.DEVICE) for k, v in state.items()})


def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)

    # fixed train-probe split for the per-epoch train-robust-acc measurement
    g = torch.Generator().manual_seed(seed + 4242)
    idx_probe = torch.randperm(Xtr.size(0), generator=g)[:TRAIN_PROBE]
    Xtr_probe, Ytr_probe = Xtr[idx_probe], Ytr[idx_probe]

    model = C.build_model("cnn", meta, seed=seed)

    # train with per-epoch probes AND snapshot weights every epoch so we can
    # restore the best-epoch model and evaluate it under the same final eval
    # protocol (control 5: early-stopped vs final).
    snapshots = []

    def _train_with_snapshots():
        opt = C.make_optimizer(model, "sgd", LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        n = Xtr.size(0)
        trace = []
        gen = torch.Generator(device=Xtr.device).manual_seed(seed)
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(n, generator=gen, device=Xtr.device)
            ep_loss_sum, ep_loss_n = 0.0, 0
            for i in range(0, n, BATCH):
                idx = perm[i:i + BATCH]
                xb, yb = Xtr[idx], Ytr[idx]
                xb_adv = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS_TRAIN,
                                alpha=2.5 * EPS / PGD_STEPS_TRAIN)
                opt.zero_grad()
                loss = F.cross_entropy(model(xb_adv), yb)
                loss.backward()
                opt.step()
                ep_loss_sum += float(loss.item()) * xb.size(0)
                ep_loss_n += xb.size(0)
            sched.step()
            train_robust_loss = ep_loss_sum / max(1, ep_loss_n)
            clean_test = eval_clean_acc(model, Xte, Yte)
            clean_train = eval_clean_acc(model, Xtr_probe, Ytr_probe)
            asr_test = eval_pgd_asr(model, Xte, Yte)
            asr_train = eval_pgd_asr(model, Xtr_probe, Ytr_probe)
            rob_test = (1.0 - asr_test) * clean_test if asr_test == asr_test else float("nan")
            rob_train = (1.0 - asr_train) * clean_train if asr_train == asr_train else float("nan")
            mu, q25, q50, q75 = eval_margin_quartiles(model, Xte, Yte)
            trace.append({
                "epoch": ep + 1,
                "train_robust_loss": train_robust_loss,
                "clean_train": clean_train,
                "clean_test": clean_test,
                "pgd_asr_train": asr_train,
                "pgd_asr_test": asr_test,
                "robust_acc_train": rob_train,
                "robust_acc_test": rob_test,
                "margin_test_mean": mu,
                "margin_test_q25": q25,
                "margin_test_q50": q50,
                "margin_test_q75": q75,
            })
            snapshots.append(snapshot_state(model))
        return trace

    trace = _train_with_snapshots()

    # pick best epoch by test robust acc (control 3 + 5)
    rob_test_vec = [t["robust_acc_test"] for t in trace]
    best_ep = int(np.nanargmax(rob_test_vec)) + 1
    best_rob = float(rob_test_vec[best_ep - 1])
    final_ep = EPOCHS
    final_rob = float(rob_test_vec[-1])

    # valley detector: minimum of robust_acc_test on the EARLY_STOP_DIP_WINDOW range
    lo, hi = EARLY_STOP_DIP_WINDOW
    window = rob_test_vec[lo - 1:hi]
    valley_ep = (lo - 1) + int(np.nanargmin(window)) + 1
    valley_rob = float(window[int(np.nanargmin(window))])
    # peak BEFORE the valley (earliest peak)
    pre_valley = rob_test_vec[:lo]
    peak_pre_ep = int(np.nanargmax(pre_valley)) + 1
    peak_pre_rob = float(pre_valley[peak_pre_ep - 1])
    dip_drop = peak_pre_rob - valley_rob   # >0 means a clear early dip

    # restore best-epoch weights, re-evaluate on the FINAL test attack protocol
    # (already the same protocol, but this gives an explicit "early-stopped"
    # number using the snapshot itself; control 5)
    restore_state(model, snapshots[best_ep - 1])
    es_clean = eval_clean_acc(model, Xte, Yte)
    es_asr = eval_pgd_asr(model, Xte, Yte)
    es_rob = (1.0 - es_asr) * es_clean if es_asr == es_asr else float("nan")

    # also re-evaluate the final-epoch model under the same call (sanity)
    restore_state(model, snapshots[-1])
    fin_clean = eval_clean_acc(model, Xte, Yte)
    fin_asr = eval_pgd_asr(model, Xte, Yte)
    fin_rob = (1.0 - fin_asr) * fin_clean if fin_asr == fin_asr else float("nan")

    return {
        "seed": seed,
        "trace": trace,
        "best_epoch": best_ep,
        "best_robust_acc_test": best_rob,
        "final_epoch": final_ep,
        "final_robust_acc_test": final_rob,
        "early_stop_benefit": best_rob - final_rob,
        "peak_pre_valley_epoch": peak_pre_ep,
        "peak_pre_valley_robust_acc": peak_pre_rob,
        "valley_epoch": valley_ep,
        "valley_robust_acc": valley_rob,
        "valley_dip_size": dip_drop,
        "early_stopped_clean": es_clean,
        "early_stopped_robust": es_rob,
        "final_clean": fin_clean,
        "final_robust": fin_rob,
    }


def main():
    print("=" * 78)
    print("H502 - Epoch-wise robust overfitting / double descent under PGD-AT")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  epochs={EPOCHS}  "
          f"train_pgd_steps={PGD_STEPS_TRAIN}  eval_pgd_steps={PGD_STEPS_EVAL}")
    print(f"n_train={N_TRAIN}  n_eval={N_EVAL}  train_probe={TRAIN_PROBE}  "
          f"dip window epochs={EARLY_STOP_DIP_WINDOW}")
    print(f"Seeds={SEEDS}")
    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"\n[seed {s}]  ({r['runtime_s']}s)")
        print("  epoch | trL  | clTr  clTe | asrTr asrTe | robTr robTe | margMean q25  q50  q75")
        for t in r["trace"]:
            print(f"   {t['epoch']:>2d}  | "
                  f"{t['train_robust_loss']:.3f}| "
                  f"{t['clean_train']:.3f} {t['clean_test']:.3f} | "
                  f"{t['pgd_asr_train']:.3f} {t['pgd_asr_test']:.3f} | "
                  f"{t['robust_acc_train']:.3f} {t['robust_acc_test']:.3f} | "
                  f"{t['margin_test_mean']:+.2f}  {t['margin_test_q25']:+.2f} "
                  f"{t['margin_test_q50']:+.2f} {t['margin_test_q75']:+.2f}")
        print(f"  peak pre-valley : epoch {r['peak_pre_valley_epoch']:>2d}  "
              f"robust={r['peak_pre_valley_robust_acc']:.3f}")
        print(f"  valley          : epoch {r['valley_epoch']:>2d}  "
              f"robust={r['valley_robust_acc']:.3f}  "
              f"(dip from peak = {r['valley_dip_size']:+.3f})")
        print(f"  best epoch      : {r['best_epoch']:>2d}  robust={r['best_robust_acc_test']:.3f}"
              f"  (clean={r['early_stopped_clean']:.3f})")
        print(f"  final epoch     : {r['final_epoch']:>2d}  robust={r['final_robust_acc_test']:.3f}"
              f"  (clean={r['final_clean']:.3f})")
        print(f"  early-stop benefit (best - final robust acc) : "
              f"{r['early_stop_benefit']:+.3f}")

    # ---- aggregate across seeds --------------------------------------------
    def m(k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return sum(v) / len(v) if v else float("nan")

    print("\n" + "=" * 78)
    print("MEANS across seeds")
    print(f"  peak pre-valley robust acc  : {m('peak_pre_valley_robust_acc'):.3f}")
    print(f"  valley robust acc           : {m('valley_robust_acc'):.3f}")
    print(f"  valley dip size (peak-val)  : {m('valley_dip_size'):+.3f}")
    print(f"  best-epoch robust acc       : {m('best_robust_acc_test'):.3f}")
    print(f"  final-epoch robust acc      : {m('final_robust_acc_test'):.3f}")
    print(f"  early-stop benefit          : {m('early_stop_benefit'):+.3f}")
    print(f"  mean best epoch  : {sum(r['best_epoch'] for r in rows)/len(rows):.1f}")
    print(f"  mean valley epoch: {sum(r['valley_epoch'] for r in rows)/len(rows):.1f}")

    # ---- HEADLINE verdict -----------------------------------------------------
    print("\n" + "=" * 78)
    mean_dip = m("valley_dip_size")
    mean_es = m("early_stop_benefit")
    if mean_dip > 0.01 and mean_es > 0.01:
        verdict = ("CONFIRMED. PGD-AT on F-MNIST SmallCNN shows an early-epoch "
                   "robust-test peak, a clear robust-overfitting dip in the "
                   f"epoch-{EARLY_STOP_DIP_WINDOW[0]}-{EARLY_STOP_DIP_WINDOW[1]} "
                   "window, and a positive early-stopping benefit -- "
                   "consistent with Rice et al. 2020 and the epoch-wise "
                   "double-descent framing of Nakkiran et al. 2020.")
    elif mean_es > 0.01:
        verdict = ("PARTIAL. Early stopping clearly beats the final epoch "
                   "(Rice-style robust overfitting), but the peak/valley pattern "
                   "in the 5-10 window is mild on F-MNIST -- the dip is shallow "
                   "or shifted; the canonical Rice valley is dataset-dependent.")
    elif mean_dip > 0.01:
        verdict = ("PARTIAL. There IS a 5-10 epoch dip in robust test acc, but "
                   "later training recovers (double-descent re-rise) so the "
                   "final model is competitive with the early-stop point.")
    else:
        verdict = ("REFUTED on F-MNIST. The robust-test trajectory under "
                   "PGD-AT is approximately monotone on F-MNIST -- the Rice "
                   "2020 robust-overfitting phenomenon does not reproduce at "
                   "this scale/dataset. F-MNIST is easy enough that the "
                   "epoch-wise dip seen on CIFAR-10 does not materialise.")
    print("HEADLINE: " + verdict)
    print("=" * 78)


if __name__ == "__main__":
    main()
