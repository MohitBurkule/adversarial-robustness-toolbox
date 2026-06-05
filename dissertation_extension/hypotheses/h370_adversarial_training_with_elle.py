"""
H370: FGSM-AT + ELLE local linearity penalty.
loss = CE(x_adv, y) + λ * |L(x+δ_rand) - [L(x) + ∇_x L · δ_rand]|
δ_rand = random direction * eps_taylor = 0.05. x_adv = FGSM(eps=0.1).
Lambda grid: [0, 0.01, 0.1].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import campaign.common as C

N_TRAIN    = 6000
EPOCHS     = 10
LR         = 0.05
BATCH      = 128
SEED       = 0
EPS        = 0.1
EPS_TAYLOR = 0.05
LAMBDAS    = [0, 0.01, 0.1]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h370_adversarial_training_with_elle_output.txt")


def elle_penalty(model, xb, yb, eps_taylor=0.05):
    """Local linearity penalty: |L(x+δ) - [L(x) + ∇L·δ]|"""
    xb_req = xb.detach().requires_grad_(True)
    out = model(xb_req)
    L_x = F.cross_entropy(out, yb)
    g, = torch.autograd.grad(L_x, xb_req, create_graph=True)

    # random perturbation direction
    delta = torch.randn_like(xb) * eps_taylor
    delta = delta / (delta.norm() + 1e-8) * eps_taylor

    x_perturbed = (xb + delta).clamp(0, 1).detach()
    with torch.no_grad():
        L_xd = F.cross_entropy(model(x_perturbed), yb)

    linear_approx = L_x.detach() + (g.detach() * delta).sum()
    pen = (L_xd - linear_approx).abs()
    return pen


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


def train_at_elle(model, Xtr, Ytr, lam):
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
            # FGSM adversarial example
            x_adv = C.fgsm(model, xb, yb, eps=EPS)
            out_adv = model(x_adv)
            ce = F.cross_entropy(out_adv, yb)
            if lam > 0:
                pen = elle_penalty(model, xb, yb, eps_taylor=EPS_TAYLOR)
                loss = ce + lam * pen
            else:
                loss = ce
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = ["H370: FGSM-AT + ELLE Local Linearity Penalty\n" + "="*50]
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    for lam in LAMBDAS:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        train_at_elle(model, Xtr, Ytr, lam)
        res = eval_model(model, Xte, Yte)
        line = (f"lambda={lam:.3f} | clean={res['clean_acc']:.3f} "
                f"fgsm_asr={res['fgsm_asr']:.3f} pgd_asr={res['pgd_asr']:.3f} "
                f"margin={res['mean_margin']:.3f}")
        print(line)
        lines.append(line)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
