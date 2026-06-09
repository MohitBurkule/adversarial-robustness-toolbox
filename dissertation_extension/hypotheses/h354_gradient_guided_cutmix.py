"""
H354 - Gradient-Guided CutMix

Place CutMix patch in the region of highest input gradient magnitude.
Compare: baseline, random CutMix (α=1.0), gradient-guided CutMix,
         gradient-guided CutMix + gradient penalty.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
CUTMIX_ALPHA = 1.0
GRAD_PEN_LAM = 0.01

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h354_gradient_guided_cutmix_output.txt")


def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1 - float(acc_fgsm),
                pgd_asr=1 - float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))


def random_bbox(size, lam):
    """Random CutMix bounding box."""
    W = H = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)
    return x1, y1, x2, y2


def gradient_bbox(grad_mag, cut_w, cut_h):
    """Find bounding box of highest integrated gradient magnitude (sliding window)."""
    B, H, W = grad_mag.shape
    # Aggregate over batch
    agg = grad_mag.mean(0)  # H x W
    best_score = -1
    best = (0, 0, cut_h, cut_w)
    for y in range(0, H - cut_h + 1, 2):
        for x in range(0, W - cut_w + 1, 2):
            score = agg[y:y+cut_h, x:x+cut_w].sum().item()
            if score > best_score:
                best_score = score
                best = (x, y, x + cut_w, y + cut_h)
    return best


def apply_cutmix(xb, perm, x1, y1, x2, y2):
    xm = xb.clone()
    xm[:, :, y1:y2, x1:x2] = xb[perm, :, y1:y2, x1:x2]
    lam_actual = 1.0 - (y2 - y1) * (x2 - x1) / (xb.size(2) * xb.size(3))
    return xm, lam_actual


def train_condition(meta, Xtr, Ytr, condition):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    sz = meta["size"]
    ncls = meta["n_classes"]

    for ep in range(EPOCHS):
        model.train()
        perm_data = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm_data[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            perm = torch.randperm(xb.size(0), device=xb.device)

            opt.zero_grad()
            if condition == "baseline":
                loss = F.cross_entropy(model(xb), yb)

            elif condition == "random_cutmix":
                lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
                x1, y1, x2, y2 = random_bbox(sz, lam)
                xm, lam_act = apply_cutmix(xb, perm, x1, y1, x2, y2)
                out = model(xm)
                loss = lam_act * F.cross_entropy(out, yb) + (1 - lam_act) * F.cross_entropy(out, yb[perm])

            elif condition in ("grad_cutmix", "grad_cutmix_pen"):
                # Compute input gradient for bbox selection
                xb_req = xb.clone().detach().requires_grad_(True)
                loss_tmp = F.cross_entropy(model(xb_req), yb)
                grad = torch.autograd.grad(loss_tmp, xb_req)[0].detach()
                grad_mag = grad.abs().mean(dim=1)  # B x H x W

                lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
                cut_rat = np.sqrt(1.0 - lam)
                cut_w = max(1, int(sz * cut_rat))
                cut_h = max(1, int(sz * cut_rat))
                x1, y1, x2, y2 = gradient_bbox(grad_mag.cpu(), cut_w, cut_h)
                xm, lam_act = apply_cutmix(xb, perm, x1, y1, x2, y2)

                out = model(xm)
                loss = lam_act * F.cross_entropy(out, yb) + (1 - lam_act) * F.cross_entropy(out, yb[perm])

                if condition == "grad_cutmix_pen":
                    xb_req2 = xb.clone().detach().requires_grad_(True)
                    out2 = model(xb_req2)
                    loss_ce2 = F.cross_entropy(out2, yb)
                    grad2 = torch.autograd.grad(loss_ce2, xb_req2, create_graph=True)[0]
                    loss = loss + GRAD_PEN_LAM * grad2.flatten(1).norm(dim=1).mean()

            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("H354 - Gradient-Guided CutMix")
    lines.append("=" * 72)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_TRAIN={N_TRAIN}  alpha={CUTMIX_ALPHA}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    conditions = ["baseline", "random_cutmix", "grad_cutmix", "grad_cutmix_pen"]
    for cond in conditions:
        t0 = time.time()
        model = train_condition(meta, Xtr, Ytr, cond)
        metrics = eval_model(model, Xte, Yte)
        metrics["condition"] = cond
        metrics["runtime_s"] = round(time.time() - t0, 1)
        line = (f"  {cond:<20}  clean={metrics['clean_acc']:.3f}  "
                f"fgsm_asr={metrics['fgsm_asr']:.3f}  pgd_asr={metrics['pgd_asr']:.3f}  "
                f"margin={metrics['mean_margin']:.3f}  ({metrics['runtime_s']}s)")
        lines.append(line)
        print(line)

    lines.append("")
    lines.append("Interpretation: Placing CutMix patches in high-gradient regions forces")
    lines.append("the model to attend to alternative features; gradient-guided CutMix")
    lines.append("should degrade fewer gradient-aligned features vs random CutMix.")
    lines.append("=" * 72)

    out = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(out + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
