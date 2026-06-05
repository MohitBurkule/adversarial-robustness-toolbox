"""
H220 - Part-based / shape-biased training vs adversarial robustness.

Train 4 models (all standard CNN, no AT):
  1. baseline
  2. random_patch_mask: zero out random 8x8 patch per image during training
  3. saliency_dropout: during training, mask top-20% magnitude gradient pixels
  4. spatial_dropout: add nn.Dropout2d(p=0.2) after each conv layer

Key question: does forcing global / shape reasoning reduce adversarial vulnerability?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

os.makedirs("results/fashion_mnist", exist_ok=True)

DS = "fashion_mnist"
N_EVAL = 300
SEED = 0
EPS = 0.1
PGD_STEPS = 10
EPOCHS = 10
BATCH = 128


# ---------------------------------------------------------------------------
# Custom model with spatial dropout
# ---------------------------------------------------------------------------
class SmallCNNSpatialDrop(nn.Module):
    def __init__(self, in_ch=1, size=28, n_classes=10, width=32, drop_p=0.2):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(), nn.Dropout2d(p=drop_p), nn.MaxPool2d(2)]
        self.features = nn.Sequential(
            *block(in_ch, width), *block(width, width * 2), *block(width * 2, width * 4))
        feat = size // 8
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(width * 4 * feat * feat, 256), nn.ReLU(),
                                  nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# Custom training loops
# ---------------------------------------------------------------------------
def train_baseline(model, Xtr, Ytr, epochs=EPOCHS):
    C.train_model(model, Xtr, Ytr, epochs=epochs, opt="sgd", lr=0.05, ncls=10)


def train_patch_mask(model, Xtr, Ytr, epochs=EPOCHS):
    """Zero out a random 8x8 patch per image in each batch."""
    opt = C.make_optimizer(model, "sgd", 0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr[idx].clone()
            yb = Ytr[idx]
            # random 8x8 patch per image
            B = xb.size(0)
            ry = torch.randint(0, 28 - 8, (B,))
            rx = torch.randint(0, 28 - 8, (B,))
            for b in range(B):
                xb[b, :, ry[b]:ry[b]+8, rx[b]:rx[b]+8] = 0.0
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()


def train_saliency_dropout(model, Xtr, Ytr, epochs=EPOCHS):
    """Mask top-20% magnitude gradient pixels per image in each batch."""
    opt = C.make_optimizer(model, "sgd", 0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    top_k_frac = 0.20
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb_clean = Xtr[idx].clone().detach()
            yb = Ytr[idx]

            # Step 1: compute saliency mask (no param update)
            xb_probe = xb_clean.clone().requires_grad_(True)
            out_probe = model(xb_probe)
            loss_probe = F.cross_entropy(out_probe, yb)
            loss_probe.backward()
            grad_mag = xb_probe.grad.detach().abs()  # (B, C, H, W)

            # build mask: flatten per image, zero top-20%
            B = xb_clean.size(0)
            gflat = grad_mag.view(B, -1)
            k = max(1, int(top_k_frac * gflat.size(1)))
            topk_vals, topk_idx = gflat.topk(k, dim=1)
            mask = torch.ones_like(gflat)
            mask.scatter_(1, topk_idx, 0.0)
            mask = mask.view_as(xb_clean)

            # Step 2: real forward with masked input
            xb_masked = (xb_clean * mask).detach()
            opt.zero_grad()
            loss_real = F.cross_entropy(model(xb_masked), yb)
            loss_real.backward()
            opt.step()
        sched.step()
    model.eval()


def train_spatial_dropout(model, Xtr, Ytr, epochs=EPOCHS):
    C.train_model(model, Xtr, Ytr, epochs=epochs, opt="sgd", lr=0.05, ncls=10)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def eval_model(model, Xte, Yte, label):
    # clean acc
    with torch.no_grad():
        logits_all = []
        for i in range(0, Xte.size(0), 256):
            logits_all.append(model(Xte[i:i+256]).cpu())
        logits = torch.cat(logits_all)
    clean_acc = float((logits.argmax(1) == Yte.cpu()).float().mean())

    # FGSM ASR
    res_fgsm = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    # PGD ASR
    res_pgd  = C.attack_success(model, Xte, Yte, attack="pgd",  eps=EPS, steps=PGD_STEPS)

    # margin
    mg = C.margin(model, Xte, Yte)

    return {
        "label": label,
        "clean_acc": clean_acc,
        "fgsm_asr": res_fgsm["asr"],
        "pgd_asr": res_pgd["asr"],
        "margin_mean": float(np.mean(mg)),
        "margin_std": float(np.std(mg)),
    }


def main():
    print("=" * 74)
    print("H220 - Shape-biased training vs adversarial robustness")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  N_EVAL={N_EVAL}  eps={EPS}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_eval=N_EVAL, seed=SEED)

    results = []
    configs = [
        ("baseline",          "standard CNN"),
        ("random_patch_mask", "zero random 8x8 patch per image"),
        ("saliency_dropout",  "mask top-20% gradient pixels"),
        ("spatial_dropout",   "Dropout2d(0.2) after each conv"),
    ]

    for name, desc in configs:
        print(f"\n[{name}] {desc}")
        t0 = time.time()
        C.set_seed(SEED)

        if name == "spatial_dropout":
            model = SmallCNNSpatialDrop(in_ch=meta["channels"], size=meta["size"],
                                        n_classes=meta["n_classes"], width=32, drop_p=0.2)
            model = model.to(C.DEVICE)
            train_spatial_dropout(model, Xtr, Ytr)
        else:
            model = C.build_model("cnn", meta, width=32, seed=SEED)
            if name == "baseline":
                train_baseline(model, Xtr, Ytr)
            elif name == "random_patch_mask":
                train_patch_mask(model, Xtr, Ytr)
            elif name == "saliency_dropout":
                train_saliency_dropout(model, Xtr, Ytr)

        r = eval_model(model, Xte, Yte, name)
        r["runtime_s"] = round(time.time() - t0, 1)
        results.append(r)

        print(f"  clean_acc={r['clean_acc']:.3f}  fgsm_asr={r['fgsm_asr']:.3f}"
              f"  pgd_asr={r['pgd_asr']:.3f}"
              f"  margin={r['margin_mean']:.3f}±{r['margin_std']:.3f}"
              f"  ({r['runtime_s']}s)")

    print("\n" + "=" * 74)
    print(f"{'Model':<22} {'CleanAcc':>9} {'FGSM_ASR':>9} {'PGD_ASR':>9} "
          f"{'Margin_mean':>12} {'Margin_std':>11}")
    print("-" * 74)
    for r in results:
        print(f"{r['label']:<22} {r['clean_acc']:>9.3f} {r['fgsm_asr']:>9.3f}"
              f" {r['pgd_asr']:>9.3f} {r['margin_mean']:>12.3f} {r['margin_std']:>11.3f}")

    baseline = results[0]
    print("\n" + "=" * 74)
    print("Interpretation:")
    for r in results[1:]:
        d_fgsm = r['fgsm_asr'] - baseline['fgsm_asr']
        d_pgd  = r['pgd_asr']  - baseline['pgd_asr']
        print(f"  {r['label']}: ΔFGSM_ASR={d_fgsm:+.3f}  ΔPGD_ASR={d_pgd:+.3f}")
    print("Negative delta = reduced vulnerability (shape-bias helps).")
    print("=" * 74)


if __name__ == "__main__":
    main()
