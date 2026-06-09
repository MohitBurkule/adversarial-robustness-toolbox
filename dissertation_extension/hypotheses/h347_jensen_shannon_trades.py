"""
H347 - Jensen-Shannon TRADES: Replace KL with JSD in TRADES

JSD(p||q) = 0.5*KL(p||M) + 0.5*KL(q||M), M = 0.5*(p+q).
x_adv via PGD maximising JSD(f(x)||f(x+δ)). β grid: [1, 3, 6].
JSD is symmetric and bounded [0, log2].
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
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
BETAS = [1, 3, 6]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h347_jensen_shannon_trades_output.txt")


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


def jsd(p, q):
    """JSD(p||q), p and q are probability distributions (N, C)."""
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.log().clamp(min=-100), p, reduction="batchmean")
    kl_qm = F.kl_div(m.log().clamp(min=-100), q, reduction="batchmean")
    return 0.5 * kl_pm + 0.5 * kl_qm


def pgd_jsd(model, x, eps, steps, alpha):
    """Inner PGD maximising JSD(f(x)||f(x+delta))."""
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0, 1).detach()
    with torch.no_grad():
        p_clean = F.softmax(model(x0), dim=1)
    for _ in range(steps):
        xa.requires_grad_(True)
        p_adv = F.softmax(model(xa), dim=1)
        divergence = jsd(p_clean.detach(), p_adv)
        g, = torch.autograd.grad(divergence, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def train_jsd_trades(meta, Xtr, Ytr, beta):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]

            model.eval()
            x_adv = pgd_jsd(model, xb, EPS, PGD_STEPS, PGD_ALPHA)
            model.train()

            opt.zero_grad()
            out_clean = model(xb)
            loss_ce = F.cross_entropy(out_clean, yb)
            p_clean = F.softmax(out_clean, dim=1).detach()
            p_adv = F.softmax(model(x_adv), dim=1)
            jsd_loss = jsd(p_clean, p_adv)
            loss = loss_ce + beta * jsd_loss
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("H347 - Jensen-Shannon TRADES: Symmetric divergence in TRADES")
    lines.append("=" * 72)
    lines.append(f"Device={C.DEVICE}  dataset={DS}  N_TRAIN={N_TRAIN}  eps={EPS}")
    lines.append(f"beta grid: {BETAS}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, seed=SEED)

    for beta in BETAS:
        t0 = time.time()
        model = train_jsd_trades(meta, Xtr, Ytr, beta)
        metrics = eval_model(model, Xte, Yte)
        metrics["beta"] = beta
        metrics["runtime_s"] = round(time.time() - t0, 1)
        line = (f"  beta={beta:<4}  clean={metrics['clean_acc']:.3f}  "
                f"fgsm_asr={metrics['fgsm_asr']:.3f}  pgd_asr={metrics['pgd_asr']:.3f}  "
                f"margin={metrics['mean_margin']:.3f}  ({metrics['runtime_s']}s)")
        lines.append(line)
        print(line)

    lines.append("")
    lines.append("Interpretation: JSD-TRADES replaces asymmetric KL with bounded symmetric")
    lines.append("JSD. Avoids KL pathologies when distributions diverge; compare with H346.")
    lines.append("=" * 72)

    out = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(out + "\n")
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
