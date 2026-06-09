"""
H489 - Empirical Neural Tangent Kernel (NTK) signature of PGD-AT vs STD.

Seed (campaign §5 / §3 G6 / G8). The empirical NTK
    K(x, x') = J_theta(x) J_theta(x')^T
where J_theta is the parameter-gradient of the (top-class) logit, encodes the
local linearisation that governs SGD dynamics (Jacot et al. 2018, "Neural
Tangent Kernel: Convergence and Generalization in Neural Networks", NeurIPS).
For trained finite networks the NTK is data- and training-dependent (Loo,
Hadji, Lengyel 2022, "Evolution of NTK during training"; Fort et al. 2020).

Hypothesis (anchored to Tsilivis & Kempe 2022, "What Can the Neural Tangent
Kernel Tell Us About Adversarial Robustness?", NeurIPS): PGD adversarial
training reshapes the NTK so that

  (a) sample-level features of same-class points become MORE aligned
      (within-class NTK kernel goes up), and
  (b) sample-level features across classes become MORE orthogonal
      (between-class NTK kernel goes down),

so the "robust NTK signature" ratio
        rho = mean_{y_i = y_j} K(x_i, x_j) / mean_{y_i != y_j} K(x_i, x_j)
should be LARGER for the PGD-AT model than for the STD model. We additionally
report a margin -- NTK-self-norm correlation: Tsilivis & Kempe note that the
NTK eigenstructure aligns with robust directions in adv-trained nets, which
predicts that per-sample sqrt(K(x,x)) tracks the per-sample margin.

Operational details (critique constraints):
  * empirical NTK on N = 200 stratified test samples (memory: |params| can be
    >1e6, so we never materialise the full Jacobian -- we compute one row of
    K at a time via a single backward of model(x_i)_{c_i}, flatten gradient,
    then take dot product with each other gradient row stored on CPU as fp32).
  * use the TOP-CLASS logit only (one scalar per sample), per the §5 seed.
  * controls (1)-(5) are reported as a HEADLINE verdict at the end.

Refs cited in the report block:
  - Jacot, Gabriel, Hongler 2018, NeurIPS, "Neural Tangent Kernel".
  - Tsilivis & Kempe 2022, NeurIPS, "What Can the Neural Tangent Kernel Tell
    Us About Adversarial Robustness?".
  - Loo, Hadji, Lengyel 2022, "Evolution of the NTK under SGD".
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
SEEDS = [0, 1, 2]
EPS = 0.1
PGD_STEPS = 7
N_NTK = 200            # subset size for empirical NTK (critique cap)


# --------------------------------------------------------------------------
# empirical NTK helpers
# --------------------------------------------------------------------------
def _trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _grad_top_logit(model, x_single, params):
    """Flat gradient of the top-class logit at x_single (1, C, H, W).
    Returns a 1-D fp32 tensor on CPU.
    """
    model.zero_grad(set_to_none=True)
    logits = model(x_single)                          # (1, K)
    top = int(logits.argmax(1).item())
    scalar = logits[0, top]
    grads = torch.autograd.grad(scalar, params, retain_graph=False,
                                create_graph=False, allow_unused=True)
    flats = []
    for g, p in zip(grads, params):
        if g is None:
            flats.append(torch.zeros(p.numel(), device="cpu"))
        else:
            flats.append(g.detach().reshape(-1).to("cpu"))
    return torch.cat(flats).float()


def empirical_ntk(model, X, Y):
    """Compute the N x N empirical NTK on the top-class logit.
    Returns (K, self_norms) with K a CPU fp32 tensor and self_norms = diag(K).
    """
    model.eval()
    params = _trainable_params(model)
    N = X.size(0)
    # store Jacobian rows on CPU to keep GPU memory bounded
    J_rows = []
    for i in range(N):
        gi = _grad_top_logit(model, X[i:i + 1], params)
        J_rows.append(gi)
    J = torch.stack(J_rows, dim=0)                    # (N, P) on CPU fp32
    # K = J J^T  (do in chunks to be safe on RAM)
    K = torch.empty(N, N, dtype=torch.float32)
    chunk = 32
    for i in range(0, N, chunk):
        K[i:i + chunk] = J[i:i + chunk] @ J.T
    diag = K.diagonal().clone()
    return K, diag


def within_between(K, Y):
    """Mean kernel value within-class vs between-class (off-diagonal only)."""
    N = K.size(0)
    Y_cpu = Y.cpu()
    same = (Y_cpu[:, None] == Y_cpu[None, :])
    eye = torch.eye(N, dtype=torch.bool)
    same_off = same & ~eye
    diff_off = ~same & ~eye
    w = float(K[same_off].mean().item()) if same_off.any() else float("nan")
    b = float(K[diff_off].mean().item()) if diff_off.any() else float("nan")
    return w, b


def stratified_subset(X, Y, n, n_classes, seed=0):
    """Pick ~n/ncls samples per class to keep within/between estimates stable."""
    g = torch.Generator().manual_seed(seed)
    per = max(1, n // n_classes)
    idx_all = []
    for c in range(n_classes):
        ci = (Y == c).nonzero(as_tuple=True)[0]
        if ci.numel() == 0:
            continue
        perm = ci[torch.randperm(ci.numel(), generator=g)][:per]
        idx_all.append(perm)
    idx = torch.cat(idx_all)
    if idx.numel() > n:
        idx = idx[:n]
    return X[idx], Y[idx]


# --------------------------------------------------------------------------
# per-seed experiment
# --------------------------------------------------------------------------
def run_seed(seed):
    C.set_seed(seed)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=2000, seed=seed)

    # control (1): STD and PGD-AT models, both trained to end-of-training
    std = C.build_model("cnn", meta)
    C.train_model(std, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05,
                  ncls=meta["n_classes"])

    adv = C.build_model("cnn", meta)
    C.train_model(adv, Xtr, Ytr, epochs=6, opt="sgd", lr=0.05,
                  ncls=meta["n_classes"],
                  adv_train=True, adv_eps=EPS, adv_steps=PGD_STEPS)

    # subset for empirical NTK (stratified, N=200 across 10 classes)
    Xs, Ys = stratified_subset(Xte, Yte, N_NTK, meta["n_classes"], seed=seed)

    out = {"seed": seed, "n_ntk": int(Xs.size(0))}
    for name, m in [("std", std), ("adv", adv)]:
        # control (2): J J^T empirical NTK on subset
        K, diag = empirical_ntk(m, Xs, Ys)
        # control (3): within vs between mean kernel
        w, b = within_between(K, Ys)
        # control (4): robust NTK signature
        rho = w / b if (b == b and b != 0) else float("nan")

        # control (5): correlation between sqrt(K(x,x)) and margin
        margins = C.margin(m, Xs, Ys)                                 # (N,)
        snorm = torch.sqrt(diag.clamp_min(0.0)).numpy()
        # Pearson correlation (numpy)
        if np.std(snorm) > 0 and np.std(margins) > 0:
            corr = float(np.corrcoef(snorm, margins)[0, 1])
        else:
            corr = float("nan")

        # also note clean accuracy and PGD robust accuracy for context
        _, clean_acc = C.logits_and_acc(m, Xte, Yte)
        asr = C.attack_success(m, Xte, Yte, "pgd", eps=EPS,
                               steps=PGD_STEPS)["asr"]

        out[name] = {
            "within": w, "between": b, "rho": rho,
            "diag_mean": float(diag.mean().item()),
            "diag_std": float(diag.std().item()),
            "margin_corr": corr,
            "clean_acc": clean_acc,
            "pgd_asr": asr,
        }
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("H489 - Empirical NTK signature: STD vs PGD-AT (Fashion-MNIST, SmallCNN)")
    print("=" * 78)
    print(f"Device={C.DEVICE}  dataset={DS}  eps={EPS}  N_NTK={N_NTK}")
    print("Refs: Jacot et al. 2018 (NTK); Tsilivis & Kempe 2022 (NTK & adv. "
          "robustness); Loo et al. 2022 (NTK evolution).")

    rows = []
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(s)
        r["runtime_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"\n[seed {s}]  ({r['runtime_s']}s, n_ntk={r['n_ntk']})")
        for name in ("std", "adv"):
            d = r[name]
            print(f"  {name.upper():3s}  clean={d['clean_acc']:.3f}  pgd_asr={d['pgd_asr']:.3f}"
                  f"  within={d['within']:.3e}  between={d['between']:.3e}"
                  f"  rho={d['rho']:.3f}  diag_mu={d['diag_mean']:.3e}"
                  f"  margin~||J||_corr={d['margin_corr']:+.3f}")

    def m(key, sub):
        v = [r[sub][key] for r in rows if r[sub][key] == r[sub][key]]
        return sum(v) / len(v) if v else float("nan")

    rho_std = m("rho", "std")
    rho_adv = m("rho", "adv")
    w_std, b_std = m("within", "std"), m("between", "std")
    w_adv, b_adv = m("within", "adv"), m("between", "adv")
    cor_std, cor_adv = m("margin_corr", "std"), m("margin_corr", "adv")
    asr_std, asr_adv = m("pgd_asr", "std"), m("pgd_asr", "adv")

    print("\n" + "=" * 78)
    print("MEANS across seeds")
    print(f"  STD : within={w_std:.3e}  between={b_std:.3e}  rho={rho_std:.3f}"
          f"  margin_corr={cor_std:+.3f}  pgd_asr={asr_std:.3f}")
    print(f"  ADV : within={w_adv:.3e}  between={b_adv:.3e}  rho={rho_adv:.3f}"
          f"  margin_corr={cor_adv:+.3f}  pgd_asr={asr_adv:.3f}")
    print("=" * 78)

    # HEADLINE verdict
    if rho_adv > rho_std and rho_adv == rho_adv and rho_std == rho_std:
        verdict = "CONFIRMED"
        detail = ("PGD-AT NTK has higher within/between ratio than STD: per-sample "
                  "parameter-gradient features are more class-aligned and more "
                  "orthogonal between classes, consistent with Tsilivis & Kempe.")
    elif rho_adv == rho_adv and rho_std == rho_std and rho_adv < rho_std:
        verdict = "REFUTED"
        detail = ("PGD-AT NTK has LOWER within/between ratio than STD: adversarial "
                  "training did not sharpen class-conditional NTK alignment on this "
                  "subset / architecture.")
    else:
        verdict = "INCONCLUSIVE"
        detail = "NaNs in within/between means -- subset too small or degenerate."
    print(f"HEADLINE: {verdict}.  rho_adv={rho_adv:.3f} vs rho_std={rho_std:.3f}.")
    print(detail)
    print("Margin correlation with sqrt(K(x,x)):  STD={:+.3f}  ADV={:+.3f}  "
          "(Tsilivis & Kempe predict |corr| larger / more positive for ADV).".format(
              cor_std, cor_adv))
    print("=" * 78)


if __name__ == "__main__":
    main()
