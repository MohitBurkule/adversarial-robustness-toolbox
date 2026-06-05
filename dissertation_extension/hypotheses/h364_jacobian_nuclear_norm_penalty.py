"""
H364: Jacobian Nuclear Norm Penalty.
Estimate nuclear norm via K=10 random vectors: sum_k ||J v_k||.
J v = autograd.grad(sum(f(x)*v), x)[0].
Lambda grid: [0, 0.001, 0.01].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import campaign.common as C

N_TRAIN = 6000
EPOCHS  = 10
LR      = 0.05
BATCH   = 128
SEED    = 0
K_VECS  = 10
LAMBDAS = [0, 0.001, 0.01]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h364_jacobian_nuclear_norm_penalty_output.txt")


def nuclear_norm_approx(model, xb, K=10):
    """Approximate nuclear norm of input Jacobian via K random vectors."""
    xb_req = xb.detach().requires_grad_(True)
    out = model(xb_req)  # (B, C)

    total = 0.0
    for _ in range(K):
        v = torch.randn_like(out)  # (B, C)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        # J^T v = grad of (out * v).sum() wrt x
        scalar = (out * v).sum()
        g, = torch.autograd.grad(scalar, xb_req, create_graph=True, retain_graph=True)
        total = total + g.view(g.size(0), -1).norm(dim=1).mean()

    return total / K


def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1-float(acc_fgsm),
                pgd_asr=1-float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))


def train_jnn(model, Xtr, Ytr, lam):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            out = model(xb)
            ce = F.cross_entropy(out, yb)
            if lam > 0:
                pen = nuclear_norm_approx(model, xb, K=K_VECS)
                loss = ce + lam * pen
            else:
                loss = ce
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = ["H364: Jacobian Nuclear Norm Penalty\n" + "="*50]
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    for lam in LAMBDAS:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        train_jnn(model, Xtr, Ytr, lam)
        res = eval_model(model, Xte, Yte)
        line = (f"lambda={lam:.4f} | clean={res['clean_acc']:.3f} "
                f"fgsm_asr={res['fgsm_asr']:.3f} pgd_asr={res['pgd_asr']:.3f} "
                f"margin={res['mean_margin']:.3f}")
        print(line)
        lines.append(line)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
