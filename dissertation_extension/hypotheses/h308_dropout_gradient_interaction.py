"""H308: Dropout rate effect on gradient structure and adversarial robustness."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from campaign import common as C

N_TRAIN = 6000; EPOCHS = 10; LR = 0.05; BATCH = 128; SEED = 0
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

def compute_grad_norm(model, Xte, Yte, n_samples=200):
    model.train()
    xb = Xte[:n_samples].requires_grad_(False)
    xb = xb.clone().detach().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(xb), Yte[:n_samples])
    g = torch.autograd.grad(loss, xb)[0]
    model.eval()
    return float(g.flatten(1).norm(dim=1).mean())

def train_with_dropout(dropout_rate, Xtr, Ytr, meta):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    # Insert dropout before the final linear layer in head
    # head = Flatten, Linear(256), ReLU, Linear(10)
    old_head = model.head
    new_head = nn.Sequential(
        old_head[0],  # Flatten
        old_head[1],  # Linear -> 256
        old_head[2],  # ReLU
        nn.Dropout(p=dropout_rate),
        old_head[3],  # Linear -> 10
    )
    model.head = new_head
    model = model.to(C.DEVICE)
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
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    dropout_rates = [0.0, 0.1, 0.3, 0.5]
    lines = ["H308: Dropout Rate vs Gradient Structure and Robustness", "="*60]
    for dr in dropout_rates:
        model = train_with_dropout(dr, Xtr, Ytr, meta)
        res = eval_model(model, Xte, Yte)
        grad_norm = compute_grad_norm(model, Xte, Yte)
        line = (f"dropout={dr:.1f}: clean_acc={res['clean_acc']:.4f}, "
                f"fgsm_asr={res['fgsm_asr']:.4f}, pgd_asr={res['pgd_asr']:.4f}, "
                f"mean_margin={res['mean_margin']:.4f}, grad_norm={grad_norm:.4f}")
        print(line)
        lines.append(line)
    out_path = os.path.join(RESULTS_DIR, "h308_dropout_gradient_interaction_output.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
