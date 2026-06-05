"""
H390: Hard weight spectral-normalization vs soft Jacobian penalty.

Hypothesis: architecturally constraining each layer's spectral norm
(Miyato-style) gives equal-or-better robustness than the soft Jacobian
penalty (prior H344 got PGD ASR ~ 0.685) at lower clean-accuracy cost.

Three conditions, all trained from the same seed on Fashion-MNIST:
  (a) baseline            : standard SmallCNN, no constraint.
  (b) spectral_norm       : torch.nn.utils.parametrizations.spectral_norm on
                            every Conv2d and Linear. Trained normally.
  (c) jacobian_penalty    : loss = CE + lambda * ||J(x)||_F^2  (lambda=0.1),
                            Jacobian Frobenius via Hutchinson estimator.

For each: clean_acc, fgsm_asr, pgd_asr, mean_margin, and the achieved global
input-Jacobian Frobenius norm on a held-out batch (Lipschitz proxy).

Run offline (no auto-run). Output -> results/fashion_mnist/h390_*.txt
"""
import os
import sys
import time

# import campaign.common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import campaign.common as C

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

# ---- config ---------------------------------------------------------------
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
JAC_LAMBDA = 0.1
META = {"channels": 1, "size": 28, "n_classes": 10}

DEVICE = C.DEVICE


# ---------------------------------------------------------------------------
# training helpers
# ---------------------------------------------------------------------------
def make_opt(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)


def train_standard(model, Xtr, Ytr):
    """Plain CE training with manual SGD loop (used by baseline + spectral-norm)."""
    opt = make_opt(model)
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
    model.eval()
    return model


def train_jacobian(model, Xtr, Ytr, lam=JAC_LAMBDA):
    """CE + lam * ||J(x)||_F^2 via Hutchinson estimator (single random probe)."""
    opt = make_opt(model)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr[idx].clone().detach().requires_grad_(True)
            yb = Ytr[idx]
            opt.zero_grad()
            logits = model(xb)
            ce = F.cross_entropy(logits, yb)
            # Hutchinson: E_v[ ||J^T v||^2 ] = ||J||_F^2 for v ~ N(0,I)
            v = torch.randn_like(logits)
            jvp = torch.autograd.grad((logits * v).sum(), xb,
                                      create_graph=True, retain_graph=True)[0]
            penalty = (jvp ** 2).sum() / xb.size(0)
            loss = ce + lam * penalty
            loss.backward()
            opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# apply spectral_norm to every Conv2d / Linear
# ---------------------------------------------------------------------------
def apply_spectral_norm(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            spectral_norm(module)
    return model


# ---------------------------------------------------------------------------
# achieved global input-Jacobian Frobenius norm (Hutchinson, held-out batch)
# ---------------------------------------------------------------------------
def jacobian_frob_norm(model, X, n_probes=5, batch=256):
    """Estimate mean over samples of ||J(x)||_F using Hutchinson probes.

    ||J||_F^2 = E_v[ ||J^T v||^2 ], v ~ N(0,I) in logit space.
    Needs grad wrt input, so cannot use no_grad; we detach the result.
    """
    model.eval()
    totals = []
    for i in range(0, X.size(0), batch):
        xb = X[i:i + batch].clone().detach().requires_grad_(True)
        acc = torch.zeros(xb.size(0), device=xb.device)
        for _ in range(n_probes):
            logits = model(xb)
            v = torch.randn_like(logits)
            jvp = torch.autograd.grad((logits * v).sum(), xb,
                                      retain_graph=False, create_graph=False)[0]
            acc = acc + (jvp ** 2).flatten(1).sum(1).detach()
        # average over probes -> estimate of ||J||_F^2 per sample
        frob2 = acc / n_probes
        totals.append(torch.sqrt(frob2.clamp(min=0)).detach().cpu())
    return float(torch.cat(totals).mean())


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate(model, Xte, Yte):
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)

    fgsm = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pgd = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=PGD_STEPS)

    m = C.margin(model, Xte, Yte)
    mean_margin = float(np.mean(m))

    jfn = jacobian_frob_norm(model, Xte[:512])

    return {
        "clean_acc": clean_acc,
        "fgsm_asr": fgsm["asr"],
        "pgd_asr": pgd["asr"],
        "mean_margin": mean_margin,
        "jac_frob": jfn,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 78)
    print("H390: Hard spectral-normalization vs soft Jacobian penalty (Fashion-MNIST)")
    print("=" * 78)
    print(f"DEVICE={DEVICE}  N_TRAIN={N_TRAIN} EPOCHS={EPOCHS} LR={LR} BATCH={BATCH}")
    print(f"EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} JAC_LAMBDA={JAC_LAMBDA}")
    print()

    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN,
                                        n_eval=N_EVAL, seed=SEED)
    print(f"data: Xtr={tuple(Xtr.shape)} Xte={tuple(Xte.shape)}")
    print()

    results = {}

    # (a) baseline
    print("--- (a) baseline: standard training ---")
    C.set_seed(SEED)
    m_base = C.build_model("cnn", META, width=32, seed=0)
    t0 = time.time()
    train_standard(m_base, Xtr, Ytr)
    print(f"    trained in {time.time()-t0:.1f}s")
    results["baseline"] = evaluate(m_base, Xte, Yte)
    print(f"    {results['baseline']}")
    print()

    # (b) spectral-norm
    print("--- (b) spectral_norm: Miyato-style hard constraint on Conv2d/Linear ---")
    C.set_seed(SEED)
    m_sn = C.build_model("cnn", META, width=32, seed=0)
    try:
        apply_spectral_norm(m_sn)
        t0 = time.time()
        train_standard(m_sn, Xtr, Ytr)
        print(f"    trained in {time.time()-t0:.1f}s")
        results["spectral_norm"] = evaluate(m_sn, Xte, Yte)
        print(f"    {results['spectral_norm']}")
    except Exception as e:
        print(f"    spectral_norm parametrization FAILED: {e!r}")
        print("    falling back to manual power-iteration normalization")
        import traceback
        traceback.print_exc()
        results["spectral_norm"] = train_sn_manual(Xtr, Ytr, Xte, Yte)
        print(f"    {results['spectral_norm']}")
    print()

    # (c) jacobian penalty
    print("--- (c) jacobian_penalty: CE + lambda*||J(x)||_F^2 (Hutchinson) ---")
    C.set_seed(SEED)
    m_jac = C.build_model("cnn", META, width=32, seed=0)
    t0 = time.time()
    train_jacobian(m_jac, Xtr, Ytr, lam=JAC_LAMBDA)
    print(f"    trained in {time.time()-t0:.1f}s")
    results["jacobian_penalty"] = evaluate(m_jac, Xte, Yte)
    print(f"    {results['jacobian_penalty']}")
    print()

    # ---- summary table ----
    print("=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    hdr = f"{'condition':<18} {'clean_acc':>10} {'fgsm_asr':>10} {'pgd_asr':>10} {'margin':>10} {'jac_frob':>10}"
    print(hdr)
    print("-" * len(hdr))
    order = ["baseline", "spectral_norm", "jacobian_penalty"]
    for k in order:
        r = results[k]
        print(f"{k:<18} {r['clean_acc']:>10.4f} {r['fgsm_asr']:>10.4f} "
              f"{r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f} {r['jac_frob']:>10.4f}")
    print()

    # ---- verdict ----
    sn = results["spectral_norm"]
    jac = results["jacobian_penalty"]
    base = results["baseline"]
    sn_matches = sn["pgd_asr"] <= jac["pgd_asr"] + 1e-3   # lower ASR = more robust
    acc_cost_sn = base["clean_acc"] - sn["clean_acc"]
    acc_cost_jac = base["clean_acc"] - jac["clean_acc"]
    cheaper = acc_cost_sn <= acc_cost_jac

    print("VERDICT")
    print("-" * 78)
    verdict = (
        f"spectral_norm PGD ASR={sn['pgd_asr']:.4f} vs jacobian_penalty "
        f"PGD ASR={jac['pgd_asr']:.4f} -> hard SN "
        f"{'MATCHES/BEATS' if sn_matches else 'WORSE THAN'} soft penalty on robustness; "
        f"clean-acc cost: SN={acc_cost_sn:+.4f}, Jac={acc_cost_jac:+.4f} "
        f"(SN is {'cheaper/equal' if cheaper else 'more expensive'})."
    )
    print(verdict)
    print()
    print(f"total runtime {time.time()-t_start:.1f}s")


# ---------------------------------------------------------------------------
# fallback: manual power-iteration spectral normalization
# ---------------------------------------------------------------------------
def train_sn_manual(Xtr, Ytr, Xte, Yte):
    """Manual power-iteration spectral normalization of each Conv2d/Linear weight,
    applied each step (renormalize weight to spectral norm 1 in a no-grad block)."""
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, seed=0)
    targets = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    # persistent left/right singular vectors per module
    us, vs = {}, {}
    for mod in targets:
        W = mod.weight.data.reshape(mod.weight.shape[0], -1)
        us[mod] = F.normalize(torch.randn(W.size(0), device=W.device), dim=0)
        vs[mod] = F.normalize(torch.randn(W.size(1), device=W.device), dim=0)

    @torch.no_grad()
    def normalize_weights():
        for mod in targets:
            W = mod.weight.data.reshape(mod.weight.shape[0], -1)
            u, v = us[mod], vs[mod]
            for _ in range(1):
                v = F.normalize(W.t() @ u, dim=0)
                u = F.normalize(W @ v, dim=0)
            sigma = torch.dot(u, W @ v)
            us[mod], vs[mod] = u, v
            if sigma > 1.0:
                mod.weight.data.div_(sigma)

    opt = make_opt(model)
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
            normalize_weights()
    model.eval()
    return evaluate(model, Xte, Yte)


if __name__ == "__main__":
    main()
