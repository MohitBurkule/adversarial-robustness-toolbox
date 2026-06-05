"""H315: Frequency-decomposed gradient penalty (penalise high-freq gradient components)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from campaign import common as C

N_TRAIN = 6000; EPOCHS = 10; LR = 0.05; BATCH = 128; SEED = 0
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)

def high_freq_penalty(g):
    """Penalise high-frequency components of input gradient via 2D FFT."""
    # g: (B, C, H, W)
    B, C, H, W = g.shape
    # FFT
    gf = torch.fft.fft2(g)  # complex (B, C, H, W)
    gf_mag = gf.abs()
    # High-frequency mask: distance from DC > H/4 or W/4
    yy, xx = torch.meshgrid(torch.arange(H, device=g.device),
                             torch.arange(W, device=g.device), indexing="ij")
    # Shift frequencies: DC is at (0,0) in fft2 output
    # For shifted equivalent: freq distance from corner in wrap-around sense
    fy = torch.minimum(yy, H - yy).float()
    fx = torch.minimum(xx, W - xx).float()
    freq_dist = torch.sqrt(fy**2 + fx**2)
    threshold = min(H, W) / 4.0
    high_mask = (freq_dist > threshold).float()  # (H, W)
    high_mask = high_mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    penalty = (gf_mag * high_mask).pow(2).mean()
    return penalty

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

def train_freq_gp(lam, Xtr, Ytr, meta):
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
            xb = Xtr[idx].clone().detach().requires_grad_(True)
            yb = Ytr[idx]
            opt.zero_grad()
            out = model(xb)
            ce_loss = F.cross_entropy(out, yb)
            if lam > 0:
                g = torch.autograd.grad(ce_loss, xb, create_graph=True)[0]
                hf_pen = lam * high_freq_penalty(g)
                loss = ce_loss + hf_pen
            else:
                loss = ce_loss
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    lams = [0, 0.01, 0.1]
    lines = ["H315: High-Frequency Gradient Penalty vs Robustness", "="*60]
    for lam in lams:
        model = train_freq_gp(lam, Xtr, Ytr, meta)
        res = eval_model(model, Xte, Yte)
        line = (f"lambda={lam:.2f}: clean_acc={res['clean_acc']:.4f}, "
                f"fgsm_asr={res['fgsm_asr']:.4f}, pgd_asr={res['pgd_asr']:.4f}, "
                f"mean_margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line)
    out_path = os.path.join(RESULTS_DIR, "h315_gradient_regularisation_frequency_output.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
