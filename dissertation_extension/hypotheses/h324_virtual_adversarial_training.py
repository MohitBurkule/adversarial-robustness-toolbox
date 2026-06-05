"""H324: Virtual Adversarial Training (VAT) comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

N_TRAIN, EPOCHS, LR, BATCH, SEED = 6000, 10, 0.05, 128, 0
VAT_EPS = 0.1
OUT = "results/fashion_mnist/h324_virtual_adversarial_training_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters(): p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1-float(acc_fgsm), pgd_asr=1-float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))

def vat_perturbation(model, xb, eps=0.1, xi=1e-6):
    """1-step power iteration VAT perturbation."""
    with torch.no_grad():
        p_orig = F.softmax(model(xb), dim=1)
    r = torch.randn_like(xb)
    r_flat = r.flatten(1)
    r = r / (r_flat.norm(dim=1, keepdim=True).view(-1,1,1,1) + 1e-8) * xi
    r.requires_grad_(True)
    p_pert = F.log_softmax(model(xb + r), dim=1)
    kl = F.kl_div(p_pert, p_orig, reduction='batchmean')
    g, = torch.autograd.grad(kl, r)
    g_flat = g.flatten(1)
    r_adv = g / (g_flat.norm(dim=1, keepdim=True).view(-1,1,1,1) + 1e-8) * eps
    return r_adv.detach()

def train_condition(model, Xtr, Ytr, mode="baseline"):
    C.set_seed(SEED)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            if mode == "vat":
                r_adv = vat_perturbation(model, xb, eps=VAT_EPS)
                xb_adv = (xb + r_adv).clamp(0, 1).detach()
                with torch.no_grad():
                    p_orig = F.softmax(model(xb), dim=1)
                p_pert = F.log_softmax(model(xb_adv), dim=1)
                vat_loss = F.kl_div(p_pert, p_orig, reduction='batchmean')
                ce_loss = F.cross_entropy(model(xb), yb)
                loss = ce_loss + vat_loss
            elif mode == "fgsm_at":
                model.eval()
                xb_adv = C.fgsm(model, xb, yb, eps=VAT_EPS)
                model.train()
                loss = F.cross_entropy(model(xb_adv), yb)
            else:
                loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    lines = ["H324: Virtual Adversarial Training\n"]

    conditions = [
        ("baseline",  "baseline"),
        ("VAT(eps=0.1)", "vat"),
        ("FGSM-AT",   "fgsm_at"),
    ]
    for name, mode in conditions:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_condition(model, Xtr, Ytr, mode=mode)
        res = eval_model(model, Xte, Yte)
        line = (f"{name}: clean={res['clean_acc']:.4f} fgsm_asr={res['fgsm_asr']:.4f} "
                f"pgd_asr={res['pgd_asr']:.4f} margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line + "\n")

    with open(OUT, "w") as f: f.writelines(lines)
    print(f"Saved to {OUT}")

if __name__ == "__main__":
    main()
