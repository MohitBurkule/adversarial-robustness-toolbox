"""
H361: Perceptually Aligned Gradient (PAG) enforcement — Ganz et al. 2022.
Penalise 1 - cosine(∇_x L, sobel(x)).
Lambda grid: [0, 0.01, 0.1].
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
LAMBDAS = [0, 0.01, 0.1]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h361_perceptually_aligned_gradient_output.txt")

# Sobel kernels
SOBEL_X = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
SOBEL_Y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)


def sobel_map(x):
    """x: (B,1,H,W) -> edge map (B,1,H,W)"""
    global SOBEL_X, SOBEL_Y
    SOBEL_X = SOBEL_X.to(x.device)
    SOBEL_Y = SOBEL_Y.to(x.device)
    gx = F.conv2d(x, SOBEL_X, padding=1)
    gy = F.conv2d(x, SOBEL_Y, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-8)


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


def pag_penalty(model, xb, yb):
    xb_req = xb.detach().requires_grad_(True)
    out = model(xb_req)
    ce = F.cross_entropy(out, yb)
    grads, = torch.autograd.grad(ce, xb_req, create_graph=True)
    # Sobel saliency (no grad needed)
    with torch.no_grad():
        saliency = sobel_map(xb.detach())
    # flatten
    g_flat = grads.view(grads.size(0), -1)
    s_flat = saliency.view(saliency.size(0), -1).detach()
    # cosine similarity per sample
    cos = F.cosine_similarity(g_flat, s_flat, dim=1)
    return (1 - cos).mean()


def train_pag(model, Xtr, Ytr, lam):
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
                pen = pag_penalty(model, xb, yb)
                loss = ce + lam * pen
            else:
                loss = ce
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def main():
    lines = ["H361: Perceptually Aligned Gradient (PAG)\n" + "="*50]
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    for lam in LAMBDAS:
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, width=32)
        train_pag(model, Xtr, Ytr, lam)
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
