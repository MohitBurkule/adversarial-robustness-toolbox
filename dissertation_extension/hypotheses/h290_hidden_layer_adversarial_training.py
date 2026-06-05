"""
H290 - Hidden-layer adversarial training: does perturbing activations
       during training improve input-space robustness?

We train 5 models:
  (a) Baseline — standard CE
  (b) Input FGSM-AT — standard adversarial training at input (eps=0.1)
  (c) Block0-AT — FGSM-AT in block0 activation space (eps_act=0.5)
  (d) Block1-AT — FGSM-AT in block1 activation space (eps_act=1.0)
  (e) All-layers-AT — sequential FGSM-AT at every block's output

For hidden-layer AT the training step is:
  1. Forward through block_k → h_k
  2. h_k_adv = h_k + eps_k * sign(∇_{h_k} CE(g_k(h_k), y))  (1-step FGSM)
  3. Detach h_k_adv, continue forward through g_k
  4. Compute CE loss on perturbed path; backprop into all parameters

Evaluation (input-space): clean_acc, FGSM_ASR, PGD_ASR, mean_margin.
Key question: does activation-space AT transfer to input-space robustness?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
BATCH = 128
LR = 0.01
EPS_INPUT = 0.1       # input-space FGSM-AT budget
EPS_BLOCK0 = 0.5      # activation-space budget at block0 output
EPS_BLOCK1 = 1.0      # activation-space budget at block1 output
EPS_BLOCK2 = 2.0      # activation-space budget at block2 output
SEED = 0

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
OUT_FILE = os.path.join(OUT_DIR, "h290_hidden_layer_adversarial_training_output.txt")


# ---------------------------------------------------------------------------
# Model splitting helpers (same as H289)
# ---------------------------------------------------------------------------

def get_blocks(model):
    """Return (block0, block1, block2, head) as nn.Sequential objects."""
    feat = list(model.features.children())
    block0 = nn.Sequential(*feat[0:4])
    block1 = nn.Sequential(*feat[4:8])
    block2 = nn.Sequential(*feat[8:12])
    head   = model.head
    return block0, block1, block2, head


# ---------------------------------------------------------------------------
# One-step FGSM in activation space
# ---------------------------------------------------------------------------

def fgsm_activation(g, h, y, eps):
    """
    h: activation tensor (requires_grad will be set here)
    Returns h_adv (detached), same shape as h.
    """
    h_in = h.detach().requires_grad_(True)
    loss = F.cross_entropy(g(h_in), y)
    loss.backward()
    with torch.no_grad():
        h_adv = h_in + eps * h_in.grad.sign()
    return h_adv.detach()


# ---------------------------------------------------------------------------
# Custom training loops
# ---------------------------------------------------------------------------

def train_baseline(model, Xtr, Ytr):
    """Standard CE training."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    ds  = TensorDataset(Xtr, Ytr)
    dl  = DataLoader(ds, batch_size=BATCH, shuffle=True)
    for ep in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()


def train_input_fgsm_at(model, Xtr, Ytr):
    """Standard FGSM adversarial training at input space."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    ds  = TensorDataset(Xtr, Ytr)
    dl  = DataLoader(ds, batch_size=BATCH, shuffle=True)
    for ep in range(EPOCHS):
        for xb, yb in dl:
            # FGSM adversarial example
            for p in model.parameters(): p.requires_grad_(True)
            x_adv = C.fgsm(model, xb, yb, eps=EPS_INPUT)
            opt.zero_grad()
            loss = F.cross_entropy(model(x_adv), yb)
            loss.backward()
            opt.step()


def train_block_at(model, Xtr, Ytr, block_idx, eps_act):
    """
    Activation-space AT at a single block (block_idx ∈ {0, 1, 2}).
    At each step: forward through blocks 0..block_idx, FGSM-perturb that activation,
    then continue forward from there and backprop CE.
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    ds  = TensorDataset(Xtr, Ytr)
    dl  = DataLoader(ds, batch_size=BATCH, shuffle=True)

    for ep in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            # Forward with grad tracking only for suffix
            h = xb
            feat = list(model.features.children())
            # Forward prefix (detach intermediate, only suffix needs grad)
            prefix_mods = feat[:((block_idx + 1) * 4)]
            suffix_mods = feat[((block_idx + 1) * 4):]

            with torch.no_grad():
                for mod in prefix_mods:
                    h = mod(h)

            # FGSM in activation space
            suffix_net = nn.Sequential(*suffix_mods, model.head).to(C.DEVICE)
            h_adv = fgsm_activation(suffix_net, h, yb, eps_act)

            # Forward from perturbed activation through suffix (with grad for backprop)
            logits = suffix_net(h_adv)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()


def train_all_layers_at(model, Xtr, Ytr):
    """
    Sequential activation-space AT at every block.
    After block0: FGSM perturb with EPS_BLOCK0
    After block1: FGSM perturb with EPS_BLOCK1
    After block2: FGSM perturb with EPS_BLOCK2
    Each perturbation is applied in sequence.
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    ds  = TensorDataset(Xtr, Ytr)
    dl  = DataLoader(ds, batch_size=BATCH, shuffle=True)

    eps_list = [EPS_BLOCK0, EPS_BLOCK1, EPS_BLOCK2]

    for ep in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            feat = list(model.features.children())

            h = xb
            for k in range(3):
                block_mods = feat[k*4:(k+1)*4]
                remaining_feat = feat[(k+1)*4:]

                # forward through block k
                with torch.no_grad():
                    for mod in block_mods:
                        h = mod(h)

                # build suffix from remaining feature mods + head
                if remaining_feat:
                    suffix = nn.Sequential(*remaining_feat, model.head).to(C.DEVICE)
                else:
                    suffix = model.head

                # FGSM perturb activation
                h = fgsm_activation(suffix, h, yb, eps_list[k])
                # h is now perturbed and detached; continue loop

            # Final forward through head with last perturbed h
            logits = model.head(h)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, Xte, Yte, label):
    model.eval()
    for p in model.parameters(): p.requires_grad_(True)

    # clean accuracy
    with torch.no_grad():
        logits, clean_acc = C.logits_and_acc(model, Xte, Yte)
    clean_acc = float(clean_acc)

    # FGSM attack success rate
    X_fgsm = C.fgsm(model, Xte, Yte, eps=EPS_INPUT)
    with torch.no_grad():
        fgsm_pred = model(X_fgsm).argmax(1).cpu()
    fgsm_asr = float((fgsm_pred != Yte.cpu()).float().mean())

    # PGD attack success rate
    X_pgd = C.pgd(model, Xte, Yte, eps=EPS_INPUT, steps=10, alpha=0.01)
    with torch.no_grad():
        pgd_pred = model(X_pgd).argmax(1).cpu()
    pgd_asr = float((pgd_pred != Yte.cpu()).float().mean())

    # mean margin
    margins = C.margin(model, Xte, Yte)
    mean_margin = float(np.mean(margins))

    print(f"  [{label}] clean={clean_acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  "
          f"PGD_ASR={pgd_asr:.4f}  margin={mean_margin:.4f}")
    return {
        "clean_acc": clean_acc,
        "fgsm_asr": fgsm_asr,
        "pgd_asr": pgd_asr,
        "mean_margin": mean_margin,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    C.set_seed(SEED)

    meta = {"channels": 1, "size": 28, "n_classes": 10}
    Xtr_full, Ytr_full, Xte, Yte = C.load_dataset(DS)
    Xtr, Ytr = Xtr_full[:N_TRAIN].to(C.DEVICE), Ytr_full[:N_TRAIN].to(C.DEVICE)
    Xte, Yte = Xte.to(C.DEVICE), Yte.to(C.DEVICE)

    configs = [
        ("(a) Baseline",      "baseline"),
        ("(b) Input FGSM-AT", "input_at"),
        ("(c) Block0-AT",     "block0_at"),
        ("(d) Block1-AT",     "block1_at"),
        ("(e) All-layers-AT", "all_at"),
    ]

    results = {}
    t0 = time.time()

    for label, key in configs:
        print(f"\nTraining {label} ...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32, seed=SEED).to(C.DEVICE)

        if key == "baseline":
            train_baseline(model, Xtr, Ytr)
        elif key == "input_at":
            train_input_fgsm_at(model, Xtr, Ytr)
        elif key == "block0_at":
            train_block_at(model, Xtr, Ytr, block_idx=0, eps_act=EPS_BLOCK0)
        elif key == "block1_at":
            train_block_at(model, Xtr, Ytr, block_idx=1, eps_act=EPS_BLOCK1)
        elif key == "all_at":
            train_all_layers_at(model, Xtr, Ytr)

        results[key] = evaluate(model, Xte, Yte, label)
        results[key]["label"] = label

    elapsed = time.time() - t0

    # Report
    lines = []
    lines.append("=" * 75)
    lines.append("H290 — Hidden-Layer Adversarial Training")
    lines.append(f"Dataset: {DS}  N_train={N_TRAIN}  epochs={EPOCHS}  seed={SEED}")
    lines.append(f"eps_input={EPS_INPUT}  eps_block0={EPS_BLOCK0}  eps_block1={EPS_BLOCK1}")
    lines.append(f"Elapsed: {elapsed:.1f}s")
    lines.append("=" * 75)
    lines.append("")
    lines.append(f"{'Model':<22} {'clean_acc':>10} {'FGSM_ASR':>10} {'PGD_ASR':>9} {'margin':>8}")
    lines.append("-" * 63)
    for label, key in configs:
        r = results[key]
        lines.append(f"{label:<22} {r['clean_acc']:>10.4f} {r['fgsm_asr']:>10.4f} "
                     f"{r['pgd_asr']:>9.4f} {r['mean_margin']:>8.4f}")
    lines.append("")
    lines.append("Key question: does activation-space AT transfer to input-space robustness?")
    lines.append("")
    # Simple interpretation
    base_pgd = results["baseline"]["pgd_asr"]
    inp_pgd  = results["input_at"]["pgd_asr"]
    b0_pgd   = results["block0_at"]["pgd_asr"]
    b1_pgd   = results["block1_at"]["pgd_asr"]
    all_pgd  = results["all_at"]["pgd_asr"]
    lines.append(f"  Baseline PGD_ASR:      {base_pgd:.4f}")
    lines.append(f"  Input-AT PGD_ASR:      {inp_pgd:.4f}  (delta={inp_pgd-base_pgd:+.4f})")
    lines.append(f"  Block0-AT PGD_ASR:     {b0_pgd:.4f}  (delta={b0_pgd-base_pgd:+.4f})")
    lines.append(f"  Block1-AT PGD_ASR:     {b1_pgd:.4f}  (delta={b1_pgd-base_pgd:+.4f})")
    lines.append(f"  All-layers-AT PGD_ASR: {all_pgd:.4f}  (delta={all_pgd-base_pgd:+.4f})")
    lines.append("")
    transfer = [k for k, key in [("Block0-AT", "block0_at"),
                                  ("Block1-AT", "block1_at"),
                                  ("All-AT",    "all_at")]
                if results[key]["pgd_asr"] < base_pgd]
    if transfer:
        lines.append(f"  Transfer observed in: {', '.join(transfer)}")
    else:
        lines.append("  No transfer to input-space robustness from activation-space AT.")
    lines.append("=" * 75)

    report = "\n".join(lines)
    print("\n" + report)
    with open(OUT_FILE, "w") as f:
        f.write(report)
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
