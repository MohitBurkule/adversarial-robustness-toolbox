"""
H362: Gradient Diversity Ensemble (GradDiv, arXiv 2107.02425).
Train 3 small models with diversity penalty = mean pairwise cosine sim of input gradients.
Ensemble prediction = mean softmax. Lambda grid: [0, 0.01, 0.1].
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
N_MODELS = 3
LAMBDAS = [0, 0.01, 0.1]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h362_gradient_diversity_ensemble_output.txt")


def grad_diversity_penalty(models, xb, yb):
    """Mean pairwise cosine similarity of input gradients."""
    grads_list = []
    for model in models:
        xb_req = xb.detach().requires_grad_(True)
        out = model(xb_req)
        ce = F.cross_entropy(out, yb)
        g, = torch.autograd.grad(ce, xb_req, create_graph=True)
        grads_list.append(g.view(g.size(0), -1))  # (B, D)

    n = len(grads_list)
    sim_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            cos = F.cosine_similarity(grads_list[i], grads_list[j], dim=1)
            sim_sum = sim_sum + cos.abs().mean()
            count += 1
    return sim_sum / max(count, 1)


def eval_ensemble(models, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for model in models:
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

    def ensemble_model_fn(x):
        probs = [F.softmax(m(x), dim=1) for m in models]
        return torch.stack(probs).mean(0)

    # Wrapper class for ensemble
    class EnsembleWrapper(torch.nn.Module):
        def __init__(self, ms):
            super().__init__()
            self.ms = torch.nn.ModuleList(ms)
        def forward(self, x):
            return torch.stack([F.softmax(m(x), dim=1) for m in self.ms]).mean(0).log()

    ew = EnsembleWrapper(models)

    def _acc(x, y):
        with torch.no_grad():
            logits = ew(x)
        return float((logits.argmax(1).cpu() == y.cpu()).float().mean())

    clean_acc = _acc(Xte, Yte)

    Xfgsm = C.fgsm(ew, Xte, Yte, eps=eps)
    acc_fgsm = _acc(Xfgsm, Yte)

    Xpgd = C.pgd(ew, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    acc_pgd = _acc(Xpgd, Yte)

    # margin from ensemble logits
    with torch.no_grad():
        logits = ew(Xte)
    margin = C.margin_of(logits.detach().cpu(), Yte.cpu())

    return dict(clean_acc=clean_acc, fgsm_asr=1-acc_fgsm, pgd_asr=1-acc_pgd,
                mean_margin=float(np.mean(margin)))


def train_ensemble(models, Xtr, Ytr, lam):
    opts = [torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                             lr=LR, momentum=0.9, weight_decay=5e-4) for m in models]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=EPOCHS) for o in opts]
    n = Xtr.size(0)
    for m in models:
        m.train()

    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            for o in opts:
                o.zero_grad()

            ce_losses = [F.cross_entropy(m(xb), yb) for m in models]
            total_ce = sum(ce_losses)

            if lam > 0:
                div_pen = grad_diversity_penalty(models, xb, yb)
                loss = total_ce + lam * div_pen
            else:
                loss = total_ce

            loss.backward()
            for o in opts:
                o.step()

        for s in scheds:
            s.step()

    for m in models:
        m.eval()
    return models


def main():
    lines = ["H362: Gradient Diversity Ensemble\n" + "="*50]
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    # Single model baseline
    C.set_seed(SEED)
    single = C.build_model("cnn", meta, width=32)
    import campaign.common as C2
    C2.train_model(single, Xtr, Ytr, epochs=EPOCHS, batch=BATCH, lr=LR,
                   opt="sgd")
    single.eval()
    for p in single.parameters():
        p.requires_grad_(True)
    _, cacc = C.logits_and_acc(single, Xte, Yte)
    Xf = C.fgsm(single, Xte, Yte, eps=0.1)
    _, facc = C.logits_and_acc(single, Xf, Yte)
    Xp = C.pgd(single, Xte, Yte, eps=0.1, steps=10, alpha=0.01)
    _, pacc = C.logits_and_acc(single, Xp, Yte)
    mm = float(np.mean(C.margin(single, Xte, Yte)))
    line = f"single_model | clean={float(cacc):.3f} fgsm_asr={1-float(facc):.3f} pgd_asr={1-float(pacc):.3f} margin={mm:.3f}"
    print(line); lines.append(line)

    for lam in LAMBDAS:
        C.set_seed(SEED)
        models = [C.build_model("cnn", meta, width=32) for _ in range(N_MODELS)]
        train_ensemble(models, Xtr, Ytr, lam)
        res = eval_ensemble(models, Xte, Yte)
        line = (f"lambda={lam:.3f} (ensemble) | clean={res['clean_acc']:.3f} "
                f"fgsm_asr={res['fgsm_asr']:.3f} pgd_asr={res['pgd_asr']:.3f} "
                f"margin={res['mean_margin']:.3f}")
        print(line)
        lines.append(line)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
