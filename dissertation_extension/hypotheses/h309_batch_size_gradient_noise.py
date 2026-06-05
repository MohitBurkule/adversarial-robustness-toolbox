"""H309: Batch size effect on gradient noise and adversarial robustness."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from campaign import common as C

N_TRAIN = 6000; LR = 0.05; SEED = 0
BASE_TOTAL_STEPS = (N_TRAIN // 128) * 10  # ~10 epochs with batch=128
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

def train_with_batch(batch, Xtr, Ytr, meta):
    C.set_seed(SEED)
    # Adjust epochs to keep total steps constant
    steps_per_epoch = max(1, N_TRAIN // batch)
    epochs = max(1, BASE_TOTAL_STEPS // steps_per_epoch)
    model = C.build_model("cnn", meta)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model, epochs

def main():
    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    batch_sizes = [16, 32, 64, 128, 256, 512]
    lines = ["H309: Batch Size vs Gradient Noise and Robustness", "="*60]
    for bs in batch_sizes:
        model, epochs = train_with_batch(bs, Xtr, Ytr, meta)
        res = eval_model(model, Xte, Yte)
        line = (f"batch={bs:4d} (epochs={epochs:3d}): clean_acc={res['clean_acc']:.4f}, "
                f"fgsm_asr={res['fgsm_asr']:.4f}, pgd_asr={res['pgd_asr']:.4f}, "
                f"mean_margin={res['mean_margin']:.4f}")
        print(line)
        lines.append(line)
    out_path = os.path.join(RESULTS_DIR, "h309_batch_size_gradient_noise_output.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
