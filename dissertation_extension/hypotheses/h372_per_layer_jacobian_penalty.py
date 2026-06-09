"""
H372 - Per-layer Jacobian penalty for adversarial robustness.

Hypothesis: penalising ||dh_{k+1}/dh_k||_F^2 at every block (not just the
input-output Jacobian) constrains sensitivity at every stage and should
improve robustness beyond a global input-gradient penalty.

Background:
  H288 input grad penalty   -> PGD_ASR 0.514
  H323 full input Jacobian  -> PGD_ASR 0.676
  H295 loss grad all layers -> PGD_ASR 0.913 (FAILED)

We estimate each layer Jacobian's Frobenius norm via Hutchinson random
projections (K=5 vectors per block per batch).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
K_HUTCH = 5  # Hutchinson vectors


class SmallCNNBlocks(nn.Module):
    """SmallCNN with explicit block access for intermediate activations."""
    def __init__(self, in_ch=1, size=28, n_classes=10, width=32):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                nn.ReLU(), nn.MaxPool2d(2))
        self.block0 = block(in_ch, width)
        self.block1 = block(width, width * 2)
        self.block2 = block(width * 2, width * 4)
        feat = size // 8
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 4 * feat * feat, 256),
            nn.ReLU(), nn.Linear(256, n_classes))

    def forward(self, x):
        h = self.block0(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.head(h)

    def forward_with_intermediates(self, x):
        """Return logits and list of (input, output) for each block."""
        pairs = []
        h = x
        for blk in [self.block0, self.block1, self.block2]:
            h_in = h
            h = blk(h)
            pairs.append((h_in, h))
        return self.head(h), pairs


def estimate_jacobian_frob_sq(block, h_in, k=K_HUTCH):
    """Estimate ||J||_F^2 for block via Hutchinson: E[||J^T v||^2].
    h_in must have requires_grad=True and be the input that produced h_out = block(h_in).
    We recompute h_out here to get the graph.
    """
    h_out = block(h_in)
    penalty = 0.0
    for _ in range(k):
        v = torch.randn_like(h_out)
        # J^T v = d(h_out . v) / d(h_in)
        JTv = autograd.grad(
            (h_out * v).sum(), h_in,
            create_graph=True, retain_graph=True)[0]
        penalty = penalty + (JTv ** 2).sum() / h_in.size(0)
    return penalty / k


def train_with_jacobian_penalty(Xtr, Ytr, lambdas, epochs=EPOCHS, lr=LR):
    """Train SmallCNNBlocks with per-layer Jacobian penalties.
    lambdas: list of 3 floats, one per block. 0 = no penalty for that block.
    """
    meta = C.dataset_meta(DS)
    model = SmallCNNBlocks(meta["channels"], meta["size"], meta["n_classes"]).to(C.DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    blocks = [model.block0, model.block1, model.block2]

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            # Forward with intermediates, keeping grads
            h = xb
            h_inputs = []
            for blk in blocks:
                h_det = h.detach().requires_grad_(True)
                h_inputs.append(h_det)
                h = blk(h_det)
            logits = model.head(h)
            ce_loss = F.cross_entropy(logits, yb)

            # Jacobian penalties
            jac_loss = 0.0
            for k_idx, (lam, blk, h_in) in enumerate(zip(lambdas, blocks, h_inputs)):
                if lam > 0:
                    jac_loss = jac_loss + lam * estimate_jacobian_frob_sq(blk, h_in)

            loss = ce_loss + jac_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


CONDITIONS = {
    "baseline":       [0.0,    0.0,    0.0],
    "input_only":     [1e-4,   0.0,    0.0],
    "all_layers":     [1e-4,   1e-4,   1e-4],
    "input_plus_all": [5e-5,   5e-5,   5e-5],
    "weighted":       [1e-4,   5e-5,   2e-5],
}


def run():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    rows = []
    for cond, lambdas in CONDITIONS.items():
        print(f"\n=== {cond}  lambdas={lambdas} ===")
        C.set_seed(SEED)
        t0 = time.time()
        model = train_with_jacobian_penalty(Xtr, Ytr, lambdas)
        elapsed = time.time() - t0

        _, clean_acc = C.logits_and_acc(model, Xte, Yte)
        fgsm_res = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
        pgd_res = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=10)
        m = C.margin(model, Xte, Yte)

        row = dict(cond=cond, lambdas=lambdas, clean_acc=clean_acc,
                   fgsm_asr=fgsm_res["asr"], pgd_asr=pgd_res["asr"],
                   mean_margin=float(np.mean(m)), time_s=elapsed)
        rows.append(row)
        print(f"  clean={clean_acc:.3f}  fgsm={fgsm_res['asr']:.3f}  "
              f"pgd={pgd_res['asr']:.3f}  margin={np.mean(m):.3f}  t={elapsed:.1f}s")

    # Write results
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h372_per_layer_jacobian_penalty_output.txt")
    with open(out_path, "w") as f:
        f.write("H372 - Per-layer Jacobian penalty\n")
        f.write("=" * 60 + "\n\n")
        hdr = f"{'condition':<18} {'lambdas':<22} {'clean':>6} {'fgsm':>6} {'pgd':>6} {'margin':>8} {'time':>6}\n"
        f.write(hdr)
        f.write("-" * 74 + "\n")
        for r in rows:
            f.write(f"{r['cond']:<18} {str(r['lambdas']):<22} {r['clean_acc']:>6.3f} "
                    f"{r['fgsm_asr']:>6.3f} {r['pgd_asr']:>6.3f} {r['mean_margin']:>8.3f} "
                    f"{r['time_s']:>6.1f}\n")
        f.write("\n")
        # Interpretation
        base = [r for r in rows if r["cond"] == "baseline"][0]
        best = min(rows, key=lambda r: r["pgd_asr"])
        f.write(f"Baseline PGD ASR: {base['pgd_asr']:.3f}\n")
        f.write(f"Best PGD ASR:     {best['pgd_asr']:.3f} ({best['cond']})\n")
        if best["pgd_asr"] < base["pgd_asr"] - 0.02:
            f.write("RESULT: SUPPORTED - per-layer Jacobian penalty improves robustness\n")
        else:
            f.write("RESULT: NOT SUPPORTED - per-layer Jacobian penalty does not clearly improve robustness\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()
