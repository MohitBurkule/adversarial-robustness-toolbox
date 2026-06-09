"""H316: Adversarial Weight Perturbation (AWP) - perturb weights before computing AT loss."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign import common as C

N_TRAIN = 6000; EPOCHS = 10; LR = 0.05; BATCH = 128; SEED = 0
EPS = 0.1
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)

def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters(): p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1-float(acc_fgsm),
                pgd_asr=1-float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))

def train_baseline(Xtr, Ytr, meta):
    """Standard clean training."""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def train_fgsm_at(Xtr, Ytr, meta):
    """Standard FGSM adversarial training."""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_adv = C.fgsm(model, xb, yb, eps=EPS)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def train_awp(Xtr, Ytr, meta, gamma):
    """FGSM-AT + AWP: perturb weights by gamma * sign(grad_w L_adv)."""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            # Step 1: compute adversarial examples
            xb_adv = C.fgsm(model, xb, yb, eps=EPS)
            # Step 2: compute gradients w.r.t. weights on adv loss
            model.zero_grad()
            loss_adv = F.cross_entropy(model(xb_adv), yb)
            loss_adv.backward()
            # Step 3: perturb weights by gamma * sign(grad)
            if gamma > 0:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.add_(gamma * p.grad.sign())
            # Step 4: forward pass on perturbed weights + update
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            # Step 5: restore weight perturbation
            if gamma > 0:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            # Need to undo perturbation - but grad was zeroed...
                            # Re-compute: use stored sign from before
                            pass
            opt.step()
        sched.step()
    model.eval()
    return model

def train_awp_v2(Xtr, Ytr, meta, gamma):
    """AWP properly: store weight perturbation and restore after update."""
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            # Step 1: compute adv examples
            xb_adv = C.fgsm(model, xb, yb, eps=EPS)
            # Step 2: compute grad_w(L_adv) for perturbation
            model.zero_grad()
            loss_adv = F.cross_entropy(model(xb_adv), yb)
            loss_adv.backward()
            # Save signs and perturb weights
            perturbations = {}
            if gamma > 0:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            delta = gamma * p.grad.sign()
                            perturbations[name] = delta
                            p.add_(delta)
            # Step 3: train on perturbed weights
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            # Step 4: restore weights
            if gamma > 0:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in perturbations:
                            p.sub_(perturbations[name])
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    lines = ["H316: Adversarial Weight Perturbation (AWP) vs Baseline", "="*60]

    # Baseline (clean)
    model = train_baseline(Xtr, Ytr, meta)
    res = eval_model(model, Xte, Yte)
    line = (f"baseline (clean): clean_acc={res['clean_acc']:.4f}, "
            f"fgsm_asr={res['fgsm_asr']:.4f}, pgd_asr={res['pgd_asr']:.4f}, "
            f"mean_margin={res['mean_margin']:.4f}")
    print(line); lines.append(line)

    # Standard FGSM-AT
    model = train_fgsm_at(Xtr, Ytr, meta)
    res = eval_model(model, Xte, Yte)
    line = (f"FGSM-AT (gamma=0): clean_acc={res['clean_acc']:.4f}, "
            f"fgsm_asr={res['fgsm_asr']:.4f}, pgd_asr={res['pgd_asr']:.4f}, "
            f"mean_margin={res['mean_margin']:.4f}")
    print(line); lines.append(line)

    # AWP variants
    for gamma in [0.001, 0.01]:
        model = train_awp_v2(Xtr, Ytr, meta, gamma)
        res = eval_model(model, Xte, Yte)
        line = (f"AWP (gamma={gamma:.3f}): clean_acc={res['clean_acc']:.4f}, "
                f"fgsm_asr={res['fgsm_asr']:.4f}, pgd_asr={res['pgd_asr']:.4f}, "
                f"mean_margin={res['mean_margin']:.4f}")
        print(line); lines.append(line)

    out_path = os.path.join(RESULTS_DIR, "h316_adversarial_weight_perturbation_output.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
