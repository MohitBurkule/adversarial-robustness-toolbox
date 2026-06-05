"""H333: Gradient similarity across classes.

Measure pairwise cosine similarity of input gradients within-class vs across-class.
High within-class similarity → easier to attack (universal perturbation possible).
Compare baseline vs FGSM-AT models.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from campaign import common as C

OUT = "results/fashion_mnist/h333_gradient_similarity_across_classes_output.txt"
os.makedirs("results/fashion_mnist", exist_ok=True)

N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0

def train_model_custom(model, Xtr, Ytr, adv=False, eps=0.1):
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
            if adv:
                xb = C.fgsm(model, xb, yb, eps=eps)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def compute_per_class_gradients(model, X, Y, n_classes=10, n_per_class=50):
    """Compute per-class mean input gradients. Returns dict class -> flat grad vector."""
    model.eval()
    class_grads = {}
    for c in range(n_classes):
        mask = (Y == c)
        Xc = X[mask][:n_per_class]
        if Xc.size(0) == 0:
            continue
        Xc = Xc.clone().requires_grad_(True)
        loss = F.cross_entropy(model(Xc), torch.full((Xc.size(0),), c, dtype=torch.long, device=Xc.device))
        g, = torch.autograd.grad(loss, Xc)
        # Mean gradient across samples, flattened
        class_grads[c] = g.detach().mean(0).flatten()
    return class_grads


def cosine_sim(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def compute_similarity_stats(class_grads, n_classes=10):
    within = []
    across = []
    classes = list(class_grads.keys())
    # Within-class: we only have one mean per class, so compute individual sample grads
    # For cross-class: compare mean grad of different classes
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            if i == j:
                continue
            sim = cosine_sim(class_grads[c1], class_grads[c2])
            across.append(sim)
    return np.nan, np.mean(across), np.std(across)


def compute_within_across_similarity(model, X, Y, n_classes=10, n_per_class=30):
    """More careful: compute per-sample gradients, then within vs across class pairs."""
    model.eval()
    all_grads = []
    all_labels = []
    for c in range(n_classes):
        mask = (Y == c)
        Xc = X[mask][:n_per_class]
        if Xc.size(0) == 0:
            continue
        for k in range(Xc.size(0)):
            xi = Xc[k:k+1].clone().requires_grad_(True)
            loss = F.cross_entropy(model(xi), torch.tensor([c], device=xi.device))
            g, = torch.autograd.grad(loss, xi)
            all_grads.append(g.detach().flatten())
            all_labels.append(c)

    grads = torch.stack(all_grads)  # (N, D)
    labels = np.array(all_labels)

    # Normalize
    norms = grads.norm(dim=1, keepdim=True) + 1e-8
    grads_norm = grads / norms

    # Pairwise cosine similarity matrix
    sim_mat = (grads_norm @ grads_norm.T).cpu().numpy()

    within_sims = []
    across_sims = []
    n = len(labels)
    for i in range(n):
        for j in range(i+1, n):
            if labels[i] == labels[j]:
                within_sims.append(sim_mat[i, j])
            else:
                across_sims.append(sim_mat[i, j])

    return np.mean(within_sims), np.mean(across_sims), np.std(within_sims), np.std(across_sims)


def eval_model(model, Xte, Yte, eps=0.1, pgd_steps=10, pgd_alpha=0.01):
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=eps)
    _, acc_fgsm = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=eps, steps=pgd_steps, alpha=pgd_alpha)
    _, acc_pgd = C.logits_and_acc(model, Xpgd, Yte)
    return dict(clean_acc=float(clean_acc), fgsm_asr=1-float(acc_fgsm), pgd_asr=1-float(acc_pgd),
                mean_margin=float(np.mean(C.margin(model, Xte, Yte))))


def main():
    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, seed=SEED)

    lines = ["H333: Gradient Similarity Across Classes", "="*60]

    conditions = [
        ("baseline", False),
        ("fgsm_at", True),
    ]

    results = {}
    for name, adv in conditions:
        print(f"Training {name}...")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta)
        train_model_custom(model, Xtr, Ytr, adv=adv)

        print(f"  Evaluating robustness...")
        rob = eval_model(model, Xte, Yte)

        print(f"  Computing gradient similarities...")
        w_mean, ac_mean, w_std, ac_std = compute_within_across_similarity(model, Xte, Yte, n_per_class=20)

        results[name] = {**rob, "within_sim_mean": w_mean, "across_sim_mean": ac_mean,
                        "within_sim_std": w_std, "across_sim_std": ac_std,
                        "within_minus_across": w_mean - ac_mean}

        lines.append(f"\nCondition: {name}")
        lines.append(f"  clean_acc={rob['clean_acc']:.4f}  fgsm_asr={rob['fgsm_asr']:.4f}  pgd_asr={rob['pgd_asr']:.4f}  margin={rob['mean_margin']:.4f}")
        lines.append(f"  within_class_grad_sim={w_mean:.4f} ± {w_std:.4f}")
        lines.append(f"  across_class_grad_sim={ac_mean:.4f} ± {ac_std:.4f}")
        lines.append(f"  within_minus_across={w_mean - ac_mean:.4f}")

    # Summary
    lines.append("\n" + "="*60)
    lines.append("SUMMARY: Gradient Similarity")
    lines.append(f"{'Condition':<15} {'Clean':>8} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Within_Sim':>12} {'Across_Sim':>12} {'W-A':>8}")
    for name, r in results.items():
        lines.append(f"{name:<15} {r['clean_acc']:>8.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['within_sim_mean']:>12.4f} {r['across_sim_mean']:>12.4f} {r['within_minus_across']:>8.4f}")

    lines.append("\nConclusion: Higher within-class similarity (W-A > 0) → gradient space exploitable.")
    lines.append("AT training should reduce within-class alignment and make across-class gradients more orthogonal.")

    out_text = "\n".join(lines)
    print(out_text)
    with open(OUT, "w") as f:
        f.write(out_text)
    print(f"\nResults written to {OUT}")

if __name__ == "__main__":
    main()
