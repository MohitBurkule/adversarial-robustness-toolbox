"""H301: Lion optimizer robustness comparison."""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/run/media/mohit/NewVolume1/adversarial-robustness-toolbox/dissertation_extension")
from campaign import common as C

N_TRAIN, EPOCHS, LR, BATCH, SEED = 6000, 10, 0.05, 128, 0
EPS, PGD_STEPS, PGD_ALPHA = 0.1, 10, 0.01

def eval_model(model, Xte, Yte):
    for p in model.parameters(): p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))
    return dict(clean_acc=float(clean_acc), fgsm_asr=1-float(acc_fgsm),
                pgd_asr=1-float(acc_pgd), mean_margin=mean_margin)

class LionOptimizer(torch.optim.Optimizer):
    """Lion: EvoLved Sign Momentum (Chen et al. 2023)."""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99, weight_decay=0.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['beta1'], group['beta2']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                m = state['m']
                # Update: sign(beta1*m + (1-beta1)*g)
                update = (b1 * m + (1 - b1) * g).sign()
                p.data.add_(update, alpha=-lr)
                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                # Update momentum: m = beta2*m + (1-beta2)*g
                m.mul_(b2).add_(g, alpha=1 - b2)
        return loss

def train_with_optimizer(meta, Xtr, Ytr, opt_name, lr=None, weight_decay=0.0):
    C.set_seed(SEED)
    model = C.build_model("cnn", meta)
    params = list(model.parameters())
    if opt_name == "sgd":
        opt = torch.optim.SGD(params, lr=0.05, momentum=0.9, weight_decay=5e-4)
    elif opt_name == "adam":
        opt = torch.optim.Adam(params, lr=0.001)
    elif opt_name == "lion":
        opt = LionOptimizer(params, lr=0.001, beta1=0.9, beta2=0.99, weight_decay=0.0)
    elif opt_name == "lion_wd":
        opt = LionOptimizer(params, lr=0.001, beta1=0.9, beta2=0.99, weight_decay=1e-2)
    else:
        raise ValueError(opt_name)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def main():
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")
    optimizers = ["sgd", "adam", "lion", "lion_wd"]
    results = []
    for opt_name in optimizers:
        t0 = time.time()
        model = train_with_optimizer(meta, Xtr, Ytr, opt_name)
        t1 = time.time()
        r = eval_model(model, Xte, Yte)
        r.update({"opt": opt_name, "time_s": round(t1-t0, 1)})
        results.append(r)
        print(f"opt={opt_name}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
              f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s")

    import os; os.makedirs("results/fashion_mnist", exist_ok=True)
    out = "results/fashion_mnist/h301_lion_optimizer_robustness_output.txt"
    with open(out, "w") as f:
        f.write("H301: Lion Optimizer Robustness\n" + "="*60 + "\n")
        for r in results:
            f.write(f"opt={r['opt']}: clean={r['clean_acc']:.3f} fgsm_asr={r['fgsm_asr']:.3f} "
                    f"pgd_asr={r['pgd_asr']:.3f} margin={r['mean_margin']:.3f} time={r['time_s']}s\n")
    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
