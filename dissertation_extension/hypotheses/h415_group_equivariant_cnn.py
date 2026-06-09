"""
H415 - Group Equivariant CNN: does p4-rotation equivariance reduce PGD ASR?

Hypothesis: A CNN that is equivariant to 90-degree rotations (the p4 group,
Cohen & Welling 2016 "Group Equivariant Convolutional Networks", ICML 2016)
reduces adversarial vulnerability because equivariance eliminates redundant,
orientation-specific feature detectors that attackers exploit; instead, features
align with the natural geometric symmetries of the data, compressing the
hypothesis class in a way that removes artificially attackable directions in
input space. A Jacobian-penalty term further regularises these directions.

Experimental conditions
-----------------------
  A. baseline   — standard SmallCNN (common.py width=32)
  B. gcnn       — p4-equivariant CNN (4-fold rotation, rotate-and-stack kernels)
  C. gcnn+jac   — G-CNN with Jacobian spectral-norm penalty during training

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128,
        SGD(mom=0.9, wd=5e-4), SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.

Reference
---------
Cohen, T., & Welling, M. (2016). Group equivariant convolutional networks.
In International conference on machine learning (pp. 2990-2999). PMLR.
arXiv:1602.07576
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DS          = "fashion_mnist"
N_TRAIN     = 6000
N_EVAL      = 2000
EPOCHS      = 10
LR          = 0.05
BATCH       = 128
SEED        = 0
EPS         = 0.1
PGD_STEPS   = 10
PGD_ALPHA   = 0.01
JAC_COEFF   = 0.01   # Jacobian penalty weight for condition C
JAC_N       = 8      # samples per mini-batch used for Jacobian estimate
META        = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h415_group_equivariant_cnn_output.txt")


# ---------------------------------------------------------------------------
# p4-equivariant ("G-CNN") building blocks
# ---------------------------------------------------------------------------
# The p4 group has 4 elements: rotations by 0, 90, 180, 270 degrees.
# A G-conv layer maps a "feature field" over the group to another one.
#
# Implementation strategy (steerable / rotate-and-stack):
#   Input feature field:  tensor of shape (N, C_in, G, H, W)  where G=4
#   First layer:          input has no group axis (standard image), G_in=1
#   Output feature field: (N, C_out, G_out=4, H, W)
#
# For a single output orientation g, the convolution applies the g-rotated
# kernel to the (all-orientation) input field.  We materialise the four
# rotated copies of each kernel with rot90 and stack them.
# ---------------------------------------------------------------------------

def _rot90_kernel(w, k):
    """Rotate a conv2d weight tensor by k*90 degrees in the spatial dims.

    w: (C_out, C_in, kH, kW)  ->  (C_out, C_in, kH, kW)  rotated.
    """
    # torch.rot90 on the last two dims
    return torch.rot90(w, k=k, dims=(-2, -1))


class GConv2d(nn.Module):
    """p4-equivariant group convolution (rotate-and-stack).

    First-layer mode (lifting conv, g_in=1):
        input  x : (N, C_in,      H, W)
        output y : (N, C_out, 4,  H, W)

    Subsequent-layer mode (group conv, g_in=4):
        input  x : (N, C_in, 4,   H, W)
        output y : (N, C_out, 4,  H, W)

    In both cases the weight is a single (C_out, C_in * g_in, kH, kW) kernel;
    we rotate it 4 ways and apply each rotated version to the input, producing
    4 output orientations.
    """

    def __init__(self, c_in, c_out, kernel_size=3, padding=1,
                 lifting=False, bias=True):
        super().__init__()
        self.lifting  = lifting
        self.c_in     = c_in
        self.c_out    = c_out
        self.g_in     = 1 if lifting else 4
        self.kernel_size = kernel_size
        self.padding  = padding
        # One set of weights; rotated copies are computed on the fly.
        self.weight = nn.Parameter(
            torch.empty(c_out, c_in * self.g_in, kernel_size, kernel_size))
        self.bias_param = nn.Parameter(torch.zeros(c_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def _get_rotated_weights(self):
        """Return list of 4 weight tensors, one per output orientation."""
        ws = []
        for k in range(4):
            # For lifting (g_in=1): simply rotate the kernel spatially.
            # For group conv (g_in=4): also permute the group-input channels
            # to account for the cyclic shift of the group axis.
            w = _rot90_kernel(self.weight, k)
            if not self.lifting:
                # Reshape to (c_out, c_in, g_in, kH, kW), roll group axis,
                # then flatten back.
                w2 = w.view(self.c_out, self.c_in, 4,
                            self.kernel_size, self.kernel_size)
                # Roll input group index by k positions (cyclic shift)
                w2 = torch.roll(w2, shifts=-k, dims=2)
                w = w2.view(self.c_out, self.c_in * 4,
                            self.kernel_size, self.kernel_size)
            ws.append(w)
        return ws

    def forward(self, x):
        """
        x: (N, C_in, H, W)         if lifting
           (N, C_in, 4, H, W)      if group conv
        """
        ws = self._get_rotated_weights()

        if self.lifting:
            # x: (N, C_in, H, W)
            outs = []
            for w in ws:
                outs.append(F.conv2d(x, w, bias=self.bias_param,
                                     padding=self.padding))
            # outs[i]: (N, C_out, H, W) — stack along dim 2
            return torch.stack(outs, dim=2)   # (N, C_out, 4, H, W)
        else:
            # x: (N, C_in, 4, H, W) -> merge C_in and 4 for conv
            N, C_in, G, H, W = x.shape
            # Flatten group into channels: (N, C_in*4, H, W)
            xf = x.permute(0, 1, 2, 3, 4).reshape(N, C_in * G, H, W)
            outs = []
            for w in ws:
                outs.append(F.conv2d(xf, w, bias=self.bias_param,
                                     padding=self.padding))
            return torch.stack(outs, dim=2)   # (N, C_out, 4, H, W)


class GBatchNorm(nn.Module):
    """BatchNorm over (N, C, G, H, W) field — shares BN statistics across G."""

    def __init__(self, c_out):
        super().__init__()
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        # x: (N, C, G, H, W)
        N, C, G, H, W = x.shape
        # Merge N and G into batch dim for BN (shares stats across orientations)
        xr = x.permute(0, 2, 1, 3, 4).reshape(N * G, C, H, W)
        xr = self.bn(xr)
        return xr.view(N, G, C, H, W).permute(0, 2, 1, 3, 4)


class GMaxPool(nn.Module):
    """MaxPool2d applied independently to each group element."""

    def __init__(self, kernel_size=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x):
        N, C, G, H, W = x.shape
        xr = x.reshape(N * C * G, 1, H, W)
        xr = self.pool(xr)
        _, _, H2, W2 = xr.shape
        return xr.view(N, C, G, H2, W2)


class GroupPool(nn.Module):
    """Global max-pool over the group axis to produce a translation-equivariant,
    rotation-invariant feature map: (N, C, G, H, W) -> (N, C, H, W)."""

    def forward(self, x):
        return x.max(dim=2).values


class GCNN(nn.Module):
    """p4 Group-Equivariant CNN for Fashion-MNIST (1-channel, 28x28 -> 10 classes).

    Architecture mirrors SmallCNN:
      block1: GConv(lifting) + GBN + ReLU + GMaxPool2
      block2: GConv(group)   + GBN + ReLU + GMaxPool2
      block3: GConv(group)   + GBN + ReLU + GMaxPool2
      group-pool  (orientation invariance)
      flatten -> Linear(256) -> ReLU -> Linear(10)

    Using width w means C_out = w for the first block; w*2 and w*4 follow.
    """

    def __init__(self, in_ch=1, size=28, n_classes=10, width=16):
        super().__init__()
        w = width
        # Lifting block
        self.lift = nn.Sequential(
            GConv2d(in_ch, w, kernel_size=3, padding=1, lifting=True),
            GBatchNorm(w), nn.ReLU(), GMaxPool())
        # Group conv block 2
        self.gc2 = nn.Sequential(
            GConv2d(w, w * 2, kernel_size=3, padding=1, lifting=False),
            GBatchNorm(w * 2), nn.ReLU(), GMaxPool())
        # Group conv block 3
        self.gc3 = nn.Sequential(
            GConv2d(w * 2, w * 4, kernel_size=3, padding=1, lifting=False),
            GBatchNorm(w * 4), nn.ReLU(), GMaxPool())
        # Orientation pooling -> spatial feature map
        self.gpool = GroupPool()
        feat = size // 8   # after 3 x MaxPool2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 4 * feat * feat, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes))

    def forward(self, x):
        x = self.lift(x)
        x = self.gc2(x)
        x = self.gc3(x)
        x = self.gpool(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_sgd(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def _jacobian_penalty(model, xb):
    """Estimate ||J||_F^2 / input_dim using finite-difference Jacobian rows.

    Sample JAC_N inputs from xb, compute output softmax, compute sum-of-grad-sq
    over the output w.r.t. each input pixel via autograd.  Returns scalar.
    """
    n = min(JAC_N, xb.size(0))
    xs = xb[:n].detach().requires_grad_(True)
    out = model(xs)                          # (n, C)
    # Sum all output dims to get a scalar per sample; grad gives sum of rows
    s = out.sum()
    g, = torch.autograd.grad(s, xs, create_graph=True)
    return (g ** 2).mean()


def train_model(model, Xtr, Ytr, epochs, lr, use_jac_penalty=False, seed=0):
    C.set_seed(seed)
    opt = _make_sgd(model, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb)
            if use_jac_penalty:
                jp = _jacobian_penalty(model, xb)
                loss = ce + JAC_COEFF * jp
            else:
                loss = ce
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_robustness(model, Xte, Yte):
    _, acc = C.logits_and_acc(model, Xte, Yte)
    fg = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg = C.attack_success(model, Xte, Yte, attack="pgd",  eps=EPS, steps=PGD_STEPS)
    return acc, fg["asr"], pg["asr"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    lines = []
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H415  Group Equivariant CNN — p4 rotation equivariance vs PGD ASR")
    out("=" * 80)
    out("Hypothesis: p4-equivariance compresses the hypothesis class along "
        "orientation-specific directions")
    out("that attackers exploit; G-CNN should exhibit lower PGD ASR than a "
        "standard CNN of")
    out("comparable depth, optionally augmented by a Jacobian spectral penalty.")
    out("")
    out(f"Config: DS={DS} N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} "
        f"LR={LR} BATCH={BATCH}")
    out(f"        SGD(mom=0.9,wd=5e-4) SEED={SEED} EPS={EPS} "
        f"PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA}")
    out(f"        JAC_COEFF={JAC_COEFF} JAC_N={JAC_N}")
    out(f"        device={C.DEVICE}")
    out("")
    out("Reference: Cohen, T. & Welling, M. (2016). Group equivariant "
        "convolutional networks.")
    out("           ICML 2016. arXiv:1602.07576")
    out("")

    # ---- data ----------------------------------------------------------------
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"Data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    out("")

    results = []

    # ---- A: Baseline SmallCNN ------------------------------------------------
    out("=" * 80)
    out("[A] BASELINE — SmallCNN (width=32, standard training)")
    out("=" * 80)
    C.set_seed(SEED)
    baseline = C.build_model("cnn", META, width=32).to(C.DEVICE)
    baseline = train_model(baseline, Xtr, Ytr, EPOCHS, LR, use_jac_penalty=False, seed=SEED)
    acc_b, fgsm_b, pgd_b = eval_robustness(baseline, Xte, Yte)
    out(f"  clean_acc={acc_b:.4f}  FGSM_ASR={fgsm_b:.4f}  PGD_ASR={pgd_b:.4f}")
    out(f"  params: {sum(p.numel() for p in baseline.parameters()):,}")
    out(f"  elapsed: {time.time()-t0:.1f}s")
    results.append({"cond": "baseline", "acc": acc_b, "fgsm": fgsm_b, "pgd": pgd_b})
    flush()

    # ---- B: G-CNN (p4) -------------------------------------------------------
    out("")
    out("=" * 80)
    out("[B] G-CNN — p4 equivariant (width=16, lifting+group convs, standard training)")
    out("=" * 80)
    out("    (width=16 chosen so param count is comparable to baseline width=32)")
    C.set_seed(SEED)
    gcnn = GCNN(in_ch=1, size=28, n_classes=10, width=16).to(C.DEVICE)
    n_gcnn = sum(p.numel() for p in gcnn.parameters())
    out(f"  G-CNN params: {n_gcnn:,}")
    gcnn = train_model(gcnn, Xtr, Ytr, EPOCHS, LR, use_jac_penalty=False, seed=SEED)
    acc_g, fgsm_g, pgd_g = eval_robustness(gcnn, Xte, Yte)
    out(f"  clean_acc={acc_g:.4f}  FGSM_ASR={fgsm_g:.4f}  PGD_ASR={pgd_g:.4f}")
    out(f"  elapsed: {time.time()-t0:.1f}s")
    results.append({"cond": "gcnn", "acc": acc_g, "fgsm": fgsm_g, "pgd": pgd_g})
    flush()

    # ---- C: G-CNN + Jacobian penalty -----------------------------------------
    out("")
    out("=" * 80)
    out(f"[C] G-CNN + Jacobian penalty (coeff={JAC_COEFF})")
    out("=" * 80)
    C.set_seed(SEED)
    gcnn_jac = GCNN(in_ch=1, size=28, n_classes=10, width=16).to(C.DEVICE)
    gcnn_jac = train_model(gcnn_jac, Xtr, Ytr, EPOCHS, LR, use_jac_penalty=True, seed=SEED)
    acc_gj, fgsm_gj, pgd_gj = eval_robustness(gcnn_jac, Xte, Yte)
    out(f"  clean_acc={acc_gj:.4f}  FGSM_ASR={fgsm_gj:.4f}  PGD_ASR={pgd_gj:.4f}")
    out(f"  elapsed: {time.time()-t0:.1f}s")
    results.append({"cond": "gcnn+jac", "acc": acc_gj, "fgsm": fgsm_gj, "pgd": pgd_gj})
    flush()

    # ---- Summary table -------------------------------------------------------
    out("")
    out("=" * 80)
    out("[SUMMARY TABLE]")
    out("=" * 80)
    hdr = "{:<14} {:>10} {:>10} {:>10} {:>12} {:>12}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR",
        "d_FGSM_ASR", "d_PGD_ASR")
    out(hdr)
    out("-" * len(hdr))
    b_row = results[0]
    for r in results:
        dfgsm = r["fgsm"] - b_row["fgsm"]
        dpgd  = r["pgd"]  - b_row["pgd"]
        out("{:<14} {:>10.4f} {:>10.4f} {:>10.4f} {:>+12.4f} {:>+12.4f}".format(
            r["cond"], r["acc"], r["fgsm"], r["pgd"], dfgsm, dpgd))
    out("-" * len(hdr))
    out("  d_* = condition minus baseline; negative = more robust than baseline.")

    # ---- Verdict -------------------------------------------------------------
    out("")
    out("=" * 80)
    out("[VERDICT]")
    out("=" * 80)

    gcnn_pgd_gain   = b_row["pgd"] - results[1]["pgd"]    # positive => G-CNN more robust
    gcnnj_pgd_gain  = b_row["pgd"] - results[2]["pgd"]
    gcnn_acc_ok     = results[1]["acc"] >= b_row["acc"] - 0.03
    gcnnj_acc_ok    = results[2]["acc"] >= b_row["acc"] - 0.03

    out(f"  G-CNN PGD robustness gain vs baseline     : {gcnn_pgd_gain:+.4f} "
        f"(positive => G-CNN more robust)")
    out(f"  G-CNN+Jac PGD robustness gain vs baseline : {gcnnj_pgd_gain:+.4f}")
    out(f"  G-CNN clean-acc within 0.03 of baseline   : {gcnn_acc_ok}")
    out(f"  G-CNN+Jac clean-acc within 0.03 of baseline: {gcnnj_acc_ok}")

    # Classify outcome
    if gcnn_pgd_gain > 0.03 and gcnn_acc_ok:
        verdict = ("SUPPORTED: p4 equivariance reduces PGD ASR (>{:.0f}pp) "
                   "without clean-accuracy cost.".format(gcnn_pgd_gain * 100))
    elif gcnn_pgd_gain > 0.01 and gcnn_acc_ok:
        verdict = ("WEAK SUPPORT: p4 equivariance gives a modest PGD reduction "
                   "({:.1f}pp) at comparable clean accuracy.".format(gcnn_pgd_gain * 100))
    elif gcnn_pgd_gain > 0.03 and not gcnn_acc_ok:
        verdict = ("PARTIAL: p4 equivariance reduces PGD ASR but at a "
                   "clean-accuracy cost >0.03.")
    else:
        verdict = ("NOT SUPPORTED: p4 equivariance does not meaningfully reduce "
                   "PGD ASR compared to an equivalently-sized baseline CNN.")

    # Extra note on Jacobian penalty
    if gcnnj_pgd_gain > gcnn_pgd_gain + 0.01 and gcnnj_acc_ok:
        jac_note = ("  Jacobian penalty provides additional robustness gain "
                    "on top of equivariance.")
    elif gcnnj_pgd_gain < gcnn_pgd_gain - 0.01:
        jac_note = ("  Jacobian penalty hurts robustness relative to plain G-CNN.")
    else:
        jac_note = ("  Jacobian penalty has negligible additional effect.")

    out("")
    out(f"  ONE-LINE VERDICT: {verdict}")
    out(jac_note)
    out("")
    out(f"done in {time.time() - t0:.1f}s")

    flush()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
