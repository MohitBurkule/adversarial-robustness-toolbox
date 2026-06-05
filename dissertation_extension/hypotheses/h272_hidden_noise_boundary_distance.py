"""
H272 - Does hidden layer Gaussian noise training actually increase the
input-space geometric boundary distance (not just reduce PGD ASR)?

This tests the KEY MECHANISTIC GAP in the literature: nobody has directly
measured the input-space L2 margin before vs after hidden-layer noise training.

Three model variants:
  - baseline     : standard CE training
  - input_noise  : C.train_model(noise_std=0.2) — Gaussian noise on raw inputs
  - hidden_noise : Gaussian σ=0.2 added after block2 via forward hook

For each model we measure:
  1. PGD ASR   (standard robustness proxy)
  2. Per-sample geometric boundary distance via binary search:
       find minimum eps in [0.001, 0.5] (8-iteration binary search) where
       C.pgd(model, x.unsqueeze(0), y.unsqueeze(0), eps=eps, steps=20)
       flips the prediction. Call this `min_eps_to_flip`.
     Averaged over N=100 test samples.

We also compute Spearman correlation between clean margin and min_eps_to_flip
to check whether the two boundary measures agree.

Key question: does input noise push min_eps_to_flip higher than hidden noise?
Are they correlated?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from campaign import common as C


SIGMA       = 0.2
EPOCHS      = 10
N_SAMPLES   = 100      # samples for boundary distance estimation
BS_ITERS    = 8        # binary search iterations
EPS_LO      = 0.001
EPS_HI      = 0.5
PGD_STEPS   = 20
RESULTS_DIR = os.path.join(os.path.dirname(__file__),
                           "..", "results", "fashion_mnist")


# ---------------------------------------------------------------------------
# Hook helper
# ---------------------------------------------------------------------------

def make_hidden_noise_hook(sigma: float, model_ref):
    def hook(module, input, output):
        if model_ref[0].training:
            return output + torch.randn_like(output) * sigma
        return output
    return hook


def attach_hidden_noise_hook(model, block_idx=1, sigma=SIGMA):
    """Attach noise hook after the given block (0-indexed).
    features is flat: each block has 4 layers (Conv, BN, ReLU, MaxPool2d)."""
    model_ref = [model]
    n_layers = len(model.features)
    lpb = n_layers // 3  # layers per block
    target_layer_idx = (block_idx + 1) * lpb - 1  # last layer of block
    h = model.features[target_layer_idx].register_forward_hook(
        make_hidden_noise_hook(sigma, model_ref)
    )
    return [h]


# ---------------------------------------------------------------------------
# Geometric boundary distance via binary search
# ---------------------------------------------------------------------------

def prediction(model, x_single):
    """Predict label for a single (1, C, H, W) tensor."""
    model.eval()
    with torch.no_grad():
        return model(x_single.to(C.DEVICE)).argmax(1).item()


def min_eps_to_flip(model, x_single, y_single, n_iters=BS_ITERS,
                    lo=EPS_LO, hi=EPS_HI, pgd_steps=PGD_STEPS):
    """
    Binary search over eps to find the smallest eps where PGD flips the
    prediction of x_single from its clean label y_single.
    Returns hi if no flip is found (sample is highly robust) and
    returns the smallest eps found otherwise.
    """
    for p in model.parameters():
        p.requires_grad_(True)

    orig_pred = prediction(model, x_single)
    if orig_pred != y_single.item():
        # Already misclassified on clean input — boundary distance = 0
        return 0.0

    result_eps = hi   # pessimistic default (sample not flipped)

    lo_cur, hi_cur = lo, hi
    for _ in range(n_iters):
        mid = (lo_cur + hi_cur) / 2.0
        Xadv = C.pgd(model,
                     x_single, y_single,
                     eps=mid, steps=pgd_steps, alpha=mid / 5.0)
        adv_pred = prediction(model, Xadv)
        if adv_pred != orig_pred:
            result_eps = mid
            hi_cur = mid
        else:
            lo_cur = mid

    return result_eps


def geometric_boundary_distances(model, Xte, Yte, n_samples=N_SAMPLES):
    """
    Compute min_eps_to_flip for n_samples randomly selected test examples.
    Returns a numpy array of shape (n_samples,).
    """
    model.eval()
    idx = torch.randperm(len(Xte))[:n_samples]
    distances = []
    for i, j in enumerate(idx):
        x = Xte[j].unsqueeze(0)
        y = Yte[j].unsqueeze(0)
        d = min_eps_to_flip(model, x, y)
        distances.append(d)
        if (i + 1) % 20 == 0:
            print(f"    boundary search: {i+1}/{n_samples} done, "
                  f"current mean={np.mean(distances):.4f}")
    return np.array(distances)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_asr(model, X, Y, Xadv):
    model.eval()
    with torch.no_grad():
        clean_pred = model(X.to(C.DEVICE)).argmax(1).cpu().numpy()
        adv_pred   = model(Xadv.to(C.DEVICE)).argmax(1).cpu().numpy()
    y_np = Y.cpu().numpy()
    correct_clean = clean_pred == y_np
    fooled = correct_clean & (adv_pred != y_np)
    if correct_clean.sum() == 0:
        return 0.0
    return float(fooled.sum() / correct_clean.sum())


def spearman_rho(a, b):
    """Spearman rank correlation between two 1-D arrays."""
    n = len(a)
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    d2 = ((rank_a - rank_b) ** 2).sum()
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def evaluate_model(model, Xte, Yte, tag=""):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    _, clean_acc = C.logits_and_acc(model, Xte, Yte)

    # PGD ASR
    Xpgd = C.pgd(model, Xte, Yte, eps=0.1, steps=10, alpha=0.01)
    pgd_asr = compute_asr(model, Xte, Yte, Xpgd)

    # Geometric boundary distances (slow — binary search per sample)
    print(f"  [{tag}] computing boundary distances …")
    t0 = time.time()
    distances = geometric_boundary_distances(model, Xte, Yte, n_samples=N_SAMPLES)
    bd_elapsed = time.time() - t0
    mean_bd = float(np.mean(distances))

    # Clean margin
    margins = C.margin(model, Xte, Yte)   # numpy array
    # Align margins with the same N_SAMPLES random subset
    # (We recompute margins on the same 100-sample slice via full array)
    idx = torch.randperm(len(Xte))[:N_SAMPLES]
    sample_margins = margins[idx.numpy()]

    rho = spearman_rho(sample_margins, distances)

    print(f"  [{tag}]  clean_acc={clean_acc:.4f}  pgd_asr={pgd_asr:.4f}  "
          f"mean_bd={mean_bd:.4f}  spearman_rho(margin,bd)={rho:.4f}  "
          f"bd_time={bd_elapsed:.1f}s")

    return dict(tag=tag, clean_acc=clean_acc, pgd_asr=pgd_asr,
                mean_bd=mean_bd, spearman_rho=rho,
                distances=distances, margins=sample_margins)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            "h272_hidden_noise_boundary_distance_output.txt")

    C.set_seed(0)
    print("Loading Fashion-MNIST …")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")

    results = []

    # ---- baseline ----
    print("\n=== Training: baseline ===")
    C.set_seed(0)
    m_base = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                           width=32, seed=0)
    t0 = time.time()
    C.train_model(m_base, Xtr, Ytr, epochs=EPOCHS)
    train_t = time.time() - t0
    res = evaluate_model(m_base, Xte, Yte, "baseline")
    res["train_time_s"] = train_t
    results.append(res)

    # ---- input noise ----
    print("\n=== Training: input_noise (noise_std=0.2) ===")
    C.set_seed(0)
    m_in = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                         width=32, seed=0)
    t0 = time.time()
    # Custom training loop with input noise
    from torch.utils.data import DataLoader, TensorDataset
    m_in.to(C.DEVICE).train()
    opt_ = torch.optim.SGD(m_in.parameters(), lr=0.05, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_, T_max=EPOCHS)
    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=128, shuffle=True)
    for _ in range(EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
            xb_noisy = xb + torch.randn_like(xb) * SIGMA
            xb_noisy = xb_noisy.clamp(0, 1)
            opt_.zero_grad()
            nn.CrossEntropyLoss()(m_in(xb_noisy), yb).backward()
            opt_.step()
        sched.step()
    train_t = time.time() - t0
    res = evaluate_model(m_in, Xte, Yte, "input_noise")
    res["train_time_s"] = train_t
    results.append(res)

    # ---- hidden noise (after block2) ----
    print("\n=== Training: hidden_noise (after block2) ===")
    C.set_seed(0)
    m_hid = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                          width=32, seed=0)
    handles = attach_hidden_noise_hook(m_hid, block_idx=1, sigma=SIGMA)
    t0 = time.time()
    C.train_model(m_hid, Xtr, Ytr, epochs=EPOCHS)
    train_t = time.time() - t0
    for h in handles:
        h.remove()
    res = evaluate_model(m_hid, Xte, Yte, "hidden_noise")
    res["train_time_s"] = train_t
    results.append(res)

    # Summary table
    col = 14
    header = (f"{'Variant':<{col}} {'CleanAcc':>9} {'PGD_ASR':>8} "
              f"{'MeanBD':>8} {'SpearmanRho':>12}")
    sep = "-" * len(header)
    rows = [f"{r['tag']:<{col}} {r['clean_acc']:>9.4f} {r['pgd_asr']:>8.4f} "
            f"{r['mean_bd']:>8.4f} {r['spearman_rho']:>12.4f}"
            for r in results]
    table = "\n".join([header, sep] + rows)
    print("\n\n" + table)

    base_r = results[0]
    in_r   = results[1]
    hid_r  = results[2]

    finding_lines = [table, "\n\nKey Findings:"]
    for r in results[1:]:
        finding_lines.append(
            f"  {r['tag']} vs baseline: "
            f"ΔPGD_ASR={r['pgd_asr']-base_r['pgd_asr']:+.4f}  "
            f"ΔMeanBD={r['mean_bd']-base_r['mean_bd']:+.4f}"
        )

    finding_lines.append(
        f"\n  input_noise vs hidden_noise boundary distance: "
        f"ΔMeanBD={in_r['mean_bd']-hid_r['mean_bd']:+.4f} "
        f"({'input_noise further' if in_r['mean_bd'] > hid_r['mean_bd'] else 'hidden_noise further or equal'})"
    )
    finding_lines.append(
        f"  Spearman rho (clean_margin, min_eps_to_flip):"
    )
    for r in results:
        finding_lines.append(
            f"    {r['tag']}: rho={r['spearman_rho']:.4f}"
        )

    output = "\n".join(finding_lines)
    print(output)

    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
