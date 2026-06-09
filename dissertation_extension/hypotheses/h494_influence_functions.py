"""
H494 - Influence functions on robust loss: a small set of training points
explains most of the robust-loss reduction, and they are atypical.

Seed (Sec. 5 / Sec. 3 G8): Koh & Liang, "Understanding Black-box Predictions
via Influence Functions" (ICML 2017). The original framework uses second-order
information (HVPs) to estimate the influence of a training point on a test
loss. Two pragmatic followups motivate our choice of estimator:

  * Pruthi et al. "TracIn: Estimating Training Data Influence by Tracing
    Gradient Descent" (NeurIPS 2020) - a *first-order*, checkpoint-based
    surrogate that avoids HVPs entirely: influence is a sum of dot products
    between train-grad and test-grad evaluated at intermediate checkpoints.
  * Bae et al. "If Influence Functions are the Answer, Then What is the
    Question?" (NeurIPS 2022) - shows that standard influence functions for
    non-convex deep nets actually estimate the *proximal Bregman response
    function*, not leave-one-out retraining; this further justifies a
    checkpoint-based first-order estimator since the Hessian inverse is
    poorly defined here anyway.

Critique-driven design choices (vs. the literal Koh-Liang recipe):
  - We use TracIn-style checkpoint dot products (no HVPs / LiSSA).
  - The "test loss" is the *robust* loss (CE on PGD adversarials).
  - We only score 200 test points (random subset) for tractability.
  - Per-sample train gradients are computed in mini-batches with explicit
    per-example loops over a small final classifier slice (full-net per-sample
    grads are too slow), restricted to the last linear layer's parameters as
    in Pruthi et al.'s "TracInCP last-layer" variant.

Hypothesis:
  H1 (concentration): the top-5% training points by sum_t |grad_train.grad_test|
      account for >40% of the total positive robust-influence mass.
  H2 (atypicality):   top-5% influencers have systematically *lower* clean
      margin and *lower* C-score-proxy (= mean clean-correctness across PGD-AT
      epochs) than the bottom 95%.
  H3 (causal LOO):    retraining without the top-5% influencers degrades
      robust accuracy by >= 2pp on the same 200-sample test split, more than
      removing a random 5% (control).

If H1 + H2 + H3 all hold => robust generalisation is carried by a small,
atypical "support set" of training points, consistent with the memorization
view of robustness (cf. Feldman 2020, Carlini "Distribution Density" 2019).

Output: results/fashion_mnist/h494_influence_functions_output.txt
"""
import os, sys, time, copy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
N_TRAIN = 4000           # tractable budget for per-sample TracIn
N_EVAL = 2000            # held-out test pool
N_TEST_SUBSET = 200      # Koh-Liang style subset (critique: 200 not 2000)
EPS = 0.1                # L-inf budget (matches campaign Fashion default)
PGD_STEPS = 10
AT_EPOCHS = 6            # PGD-AT epochs -> 6 checkpoints for TracIn
LR = 0.05
TOP_FRAC = 0.05          # "top-5% influencers"


# ---------------------------------------------------------------------------
# Per-sample gradient on the last linear layer only (TracInCP last-layer).
# For SmallCNN the last layer is the classifier head (fc on flattened feats).
# We compute grads of CE(model(x), y) wrt that single weight+bias matrix.
# ---------------------------------------------------------------------------
def _last_linear(model):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    return last


def _per_sample_lastlayer_grads(model, X, Y, batch=128):
    """Return tensor (N, P) of per-sample gradients of CE loss w.r.t. the last
    Linear layer's [W;b]. Uses the analytic form for softmax-CE: for a Linear
    layer with input phi and output logits z, dL/dW = (p - one_hot(y)) outer phi
    and dL/db = (p - one_hot(y)). This is O(1) per sample given phi, p.
    """
    model.eval()
    last = _last_linear(model)
    assert last is not None, "model has no Linear layer"
    feats_out = {}

    def hook(_m, inp, _out):
        feats_out["phi"] = inp[0].detach()

    h = last.register_forward_hook(hook)
    grads = []
    try:
        for i in range(0, X.size(0), batch):
            x, y = X[i:i + batch], Y[i:i + batch]
            with torch.no_grad():
                logits = model(x)
                phi = feats_out["phi"]                       # (B, D_in)
                p = F.softmax(logits, dim=1)                 # (B, C)
                onehot = F.one_hot(y, num_classes=p.size(1)).float()
                dz = (p - onehot)                            # (B, C)
                # per-sample dW = dz_i outer phi_i -> flatten to (B, C*D_in)
                B, Cn = dz.shape
                Din = phi.shape[1]
                dW = (dz.unsqueeze(2) * phi.unsqueeze(1)).reshape(B, Cn * Din)
                g = torch.cat([dW, dz], dim=1)               # append bias grads
                grads.append(g.cpu())
    finally:
        h.remove()
    return torch.cat(grads, dim=0)                            # (N, P)


# ---------------------------------------------------------------------------
# Robust gradient on test points: gradient of CE(model(pgd(x)), y).
# ---------------------------------------------------------------------------
def _robust_test_grads(model, Xte, Yte, eps, steps):
    Xa = C.pgd(model, Xte, Yte, eps=eps, steps=steps)
    return _per_sample_lastlayer_grads(model, Xa, Yte)


# ---------------------------------------------------------------------------
# Cscore-proxy: per-epoch clean correctness (1.0 means easy, 0.0 means hard).
# After Jiang et al. "Characterizing Structural Regularities of Labeled Data
# in Overparameterized Models" (ICML 2021); cheap surrogate.
# ---------------------------------------------------------------------------
def _cscore_proxy(snapshot_logits_list, Y):
    """snapshot_logits_list: list of (N, C) cpu tensors across epochs."""
    cor = torch.stack([(lg.argmax(1) == Y.cpu()).float() for lg in snapshot_logits_list], dim=0)
    return cor.mean(0).numpy()  # in [0,1]


# ---------------------------------------------------------------------------
# PGD-AT with explicit per-epoch checkpoints. Returns list of state_dicts
# AND list of per-epoch train-logits-on-clean (for the C-score proxy).
# ---------------------------------------------------------------------------
def _pgd_at_with_checkpoints(model, Xtr, Ytr, epochs, lr, eps, steps, batch=128):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    ckpts = []
    snap_logits = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
            opt.zero_grad()
            loss = F.cross_entropy(model(xa), yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        lg, _ = C.logits_and_acc(model, Xtr, Ytr)
        snap_logits.append(lg)
        ckpts.append({k: v.detach().clone() for k, v in model.state_dict().items()})
    return ckpts, snap_logits


# ---------------------------------------------------------------------------
# TracIn: I(z_test, z_train) = sum_t lr_t * < g_train(z, theta_t), g_test(z_test, theta_t) >
# Returns per-training-point total |influence| over test subset.
# ---------------------------------------------------------------------------
def _tracin_scores(model, ckpts, Xtr, Ytr, Xsub, Ysub, lr, eps, steps):
    N = Xtr.size(0)
    M = Xsub.size(0)
    train_signed = torch.zeros(N, dtype=torch.float64)      # sum over t, sum over test of <g_tr, g_te>
    train_abs = torch.zeros(N, dtype=torch.float64)         # sum over t of |sum_te <g_tr, g_te>|
    test_signed = torch.zeros(M, dtype=torch.float64)
    for t, sd in enumerate(ckpts):
        model.load_state_dict(sd)
        model.eval()
        Gtr = _per_sample_lastlayer_grads(model, Xtr, Ytr).double()      # (N, P)
        Gte = _robust_test_grads(model, Xsub, Ysub, eps, steps).double()  # (M, P)
        # influence matrix piece (N, M)
        contrib = lr * (Gtr @ Gte.T)                                       # (N, M)
        train_signed += contrib.sum(dim=1)
        train_abs += contrib.sum(dim=1).abs()
        test_signed += contrib.sum(dim=0)
    return train_signed.numpy(), train_abs.numpy(), test_signed.numpy()


# ---------------------------------------------------------------------------
# Per-seed pipeline
# ---------------------------------------------------------------------------
def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)

    # robust test-loss target subset
    g = torch.Generator().manual_seed(seed)
    sub_idx = torch.randperm(Xte.size(0), generator=g)[:N_TEST_SUBSET]
    Xsub, Ysub = Xte[sub_idx], Yte[sub_idx]

    # 1) PGD-AT with checkpoints  ------------------------------------------
    model = C.build_model("cnn", meta, seed=seed)
    ckpts, snap_logits = _pgd_at_with_checkpoints(
        model, Xtr, Ytr, epochs=AT_EPOCHS, lr=LR, eps=EPS, steps=PGD_STEPS)
    cscore = _cscore_proxy(snap_logits, Ytr)                  # (N_train,)

    # robust accuracy of the final PGD-AT model on the test subset (baseline)
    model.load_state_dict(ckpts[-1])
    model.eval()
    Xa_sub = C.pgd(model, Xsub, Ysub, eps=EPS, steps=PGD_STEPS)
    with torch.no_grad():
        base_robust_acc = float((model(Xa_sub).argmax(1) == Ysub).float().mean())

    # margin on clean training points (final model)
    margin_tr = C.margin(model, Xtr, Ytr)

    # 2) TracIn influence  --------------------------------------------------
    inf_signed, inf_abs, _ = _tracin_scores(
        model, ckpts, Xtr, Ytr, Xsub, Ysub, lr=LR, eps=EPS, steps=PGD_STEPS)

    # 3) Concentration: what fraction of *positive* robust-influence mass
    #    (the part that reduces robust loss) is carried by the top-5%?
    helpful = -inf_signed                                       # higher = reduces robust loss
    helpful_pos = np.clip(helpful, 0, None)
    total_pos = helpful_pos.sum()
    order = np.argsort(-inf_abs)                                # rank by magnitude
    k = max(1, int(round(TOP_FRAC * inf_abs.size)))
    top_idx = order[:k]
    rest_idx = order[k:]
    top_pos_share = float(helpful_pos[top_idx].sum() / (total_pos + 1e-12))

    # 4) Atypicality
    top_margin = float(margin_tr[top_idx].mean())
    rest_margin = float(margin_tr[rest_idx].mean())
    top_cscore = float(cscore[top_idx].mean())
    rest_cscore = float(cscore[rest_idx].mean())

    # 5) Causal leave-out: retrain without top-k vs random-k
    keep_top = np.ones(inf_abs.size, dtype=bool); keep_top[top_idx] = False
    rng = np.random.RandomState(seed)
    rand_idx = rng.choice(inf_abs.size, size=k, replace=False)
    keep_rand = np.ones(inf_abs.size, dtype=bool); keep_rand[rand_idx] = False

    def _retrain_and_robust_acc(mask):
        m = C.build_model("cnn", meta, seed=seed)
        Xk = Xtr[torch.tensor(mask, device=Xtr.device)]
        Yk = Ytr[torch.tensor(mask, device=Ytr.device)]
        C.train_model(m, Xk, Yk, epochs=AT_EPOCHS, opt="sgd", lr=LR,
                      ncls=meta["n_classes"], adv_train=True,
                      adv_eps=EPS, adv_steps=PGD_STEPS)
        m.eval()
        xa = C.pgd(m, Xsub, Ysub, eps=EPS, steps=PGD_STEPS)
        with torch.no_grad():
            return float((m(xa).argmax(1) == Ysub).float().mean())

    rob_no_top = _retrain_and_robust_acc(keep_top)
    rob_no_rand = _retrain_and_robust_acc(keep_rand)

    return {
        "seed": seed,
        "n_train": int(N_TRAIN),
        "n_test_subset": int(N_TEST_SUBSET),
        "k_top": int(k),
        "base_robust_acc": base_robust_acc,
        "top5_positive_influence_share": top_pos_share,
        "top5_mean_margin": top_margin,
        "rest_mean_margin": rest_margin,
        "top5_mean_cscore": top_cscore,
        "rest_mean_cscore": rest_cscore,
        "robust_acc_drop_no_top5": base_robust_acc - rob_no_top,
        "robust_acc_drop_no_rand5": base_robust_acc - rob_no_rand,
        "robust_acc_no_top5": rob_no_top,
        "robust_acc_no_rand5": rob_no_rand,
    }


def main():
    print("=" * 78)
    print("H494 - Influence functions (TracIn) on robust loss for F-MNIST PGD-AT")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  n_train={N_TRAIN}  n_test_subset={N_TEST_SUBSET}")
    print(f"PGD eps={EPS}  steps={PGD_STEPS}  AT epochs={AT_EPOCHS}  top_frac={TOP_FRAC}")
    print("Seed: Koh & Liang 2017. TracIn: Pruthi et al. 2020 (NeurIPS).")
    print("Caveat re. influence-fn interpretation: Bae et al. 2022 (NeurIPS).")
    print("-" * 78)

    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"\n[seed {s}]  ({r['runtime_s']}s)")
        print(f"  base robust acc (200 test)    : {r['base_robust_acc']:.3f}")
        print(f"  top-5% positive-inf share     : {r['top5_positive_influence_share']:.3f}")
        print(f"  margin   top-5% / rest        : {r['top5_mean_margin']:.3f} / {r['rest_mean_margin']:.3f}")
        print(f"  c-score  top-5% / rest        : {r['top5_mean_cscore']:.3f} / {r['rest_mean_cscore']:.3f}")
        print(f"  robust acc drop  no-top5      : {r['robust_acc_drop_no_top5']:+.3f}"
              f"   (acc -> {r['robust_acc_no_top5']:.3f})")
        print(f"  robust acc drop  no-rand5     : {r['robust_acc_drop_no_rand5']:+.3f}"
              f"   (acc -> {r['robust_acc_no_rand5']:.3f})")

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    print("\n" + "=" * 78)
    print("MEANS across seeds")
    print(f"  top-5% positive-influence share    : {m('top5_positive_influence_share'):.3f}")
    print(f"  margin   top-5% / rest             : {m('top5_mean_margin'):.3f} / {m('rest_mean_margin'):.3f}")
    print(f"  c-score  top-5% / rest             : {m('top5_mean_cscore'):.3f} / {m('rest_mean_cscore'):.3f}")
    print(f"  robust acc drop  no-top5  / no-rand: "
          f"{m('robust_acc_drop_no_top5'):+.3f} / {m('robust_acc_drop_no_rand5'):+.3f}")
    print("=" * 78)

    # ---------- HEADLINE verdict --------------------------------------------
    share_ok = m("top5_positive_influence_share") > 0.40
    atyp_ok = (m("top5_mean_margin") < m("rest_mean_margin")) and \
              (m("top5_mean_cscore") < m("rest_mean_cscore"))
    loo_ok = m("robust_acc_drop_no_top5") >= 0.02 and \
             m("robust_acc_drop_no_top5") > m("robust_acc_drop_no_rand5")
    verdict = (
        "SUPPORTED: robust loss is concentrated in a small atypical subset "
        "and removing it causally hurts robustness."
        if (share_ok and atyp_ok and loo_ok)
        else "NOT (fully) SUPPORTED - see per-criterion bools below."
    )
    print(f"HEADLINE: {verdict}")
    print(f"  concentration>40%        : {share_ok}")
    print(f"  top5 more atypical       : {atyp_ok}")
    print(f"  LOO causal drop >=2pp    : {loo_ok}")
    print("=" * 78)
    print("Refs: Koh & Liang (ICML 2017); Pruthi et al. 'TracIn' (NeurIPS 2020);")
    print("      Bae et al. 'If Influence Functions are the Answer...' (NeurIPS 2022);")
    print("      Feldman 'Does Learning Require Memorization?' (STOC 2020);")
    print("      Jiang et al. C-score (ICML 2021).")
    print("=" * 78)


if __name__ == "__main__":
    main()
