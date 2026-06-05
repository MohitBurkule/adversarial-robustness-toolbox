"""
H277 - Dataset Distillation via Clean Gradient Matching (Zhao et al. 2021).

Synthesise K=1 image per class (10 synthetic images total) such that gradients
on the synthetic data match gradients on real data.  A model trained on ONLY
the 10 synthetic images should generalise nearly as well as one trained on the
full dataset if the distillation captures class structure faithfully.

Gradient matching objective (simplified, no second-order):
  For each outer step t:
    1. Sample a fresh random model theta_t.
    2. Do T=1 inner gradient step on synthetic data to get theta_t'.
    3. Compute G_real = grad_{theta_t'} L(theta_t', D_real_batch)
    4. Compute G_syn  = grad_{theta_t'} L(theta_t', D_syn)   (auto-diff through syn)
    5. Match loss = 1 - cosine_similarity(G_real, G_syn)  (averaged over layers)
    6. Backprop match loss w.r.t. synthetic pixels and update them with Adam.

After 200 outer steps:
  - Train a fresh model for 50 epochs on the 10 synthetic images.
  - Evaluate: clean_acc, FGSM_ASR, PGD_ASR on 300 real test samples.
  - Compare to baseline trained on full 60k real images.
  - Report pixel stats (mean, std) for each synthetic image.

N_eval=300 test samples.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS           = "fashion_mnist"
SEED         = 0
N_CLASSES    = 10
K            = 1          # synthetic images per class
OUTER_STEPS  = 200
INNER_STEPS  = 1          # inner gradient steps on synthetic data
LR_SYN       = 0.01       # Adam lr for synthetic image optimisation
LR_INNER     = 0.01       # SGD lr for inner model update
LR_TRAIN     = 0.05       # lr for training on distilled data
TRAIN_EPOCHS = 50         # epochs to train on synthetic data
N_REAL_BATCH = 32         # real samples per class per outer step
EPS          = 0.1
PGD_STEPS    = 10
PGD_ALPHA    = 0.01
N_EVAL       = 300
OUT_FILE     = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h277_dataset_distillation_gradient_matching_output.txt"
)


# ---------------------------------------------------------------------------
# Helpers: per-class real data index
# ---------------------------------------------------------------------------

def class_indices(Y, n_classes=N_CLASSES):
    """Return dict {class: tensor of indices} for fast per-class sampling."""
    return {c: (Y == c).nonzero(as_tuple=True)[0] for c in range(n_classes)}


def sample_real_batch(X, class_idx, n_per_class):
    """Sample n_per_class examples per class, return (X_batch, Y_batch)."""
    xs, ys = [], []
    for c, idx in class_idx.items():
        sel = idx[torch.randperm(len(idx))[:n_per_class]]
        xs.append(X[sel])
        ys.append(torch.full((len(sel),), c, dtype=torch.long))
    return torch.cat(xs).to(C.DEVICE), torch.cat(ys).to(C.DEVICE)


# ---------------------------------------------------------------------------
# Gradient matching distillation
# ---------------------------------------------------------------------------

def flatten_grads(grads):
    """Flatten a list of gradient tensors into a single 1-D tensor."""
    return torch.cat([g.reshape(-1) for g in grads if g is not None])


def cosine_distance(g1, g2):
    """1 - cosine similarity of two flat gradient vectors."""
    eps = 1e-8
    return 1.0 - F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).squeeze()


def build_fresh_model():
    m = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": N_CLASSES},
                      width=32, seed=0)
    m.to(C.DEVICE)
    return m


def distill_clean(Xtr, Ytr, use_adversarial=False):
    """
    Run gradient-matching dataset distillation.

    use_adversarial=False: match gradients of clean loss  (H277)
    use_adversarial=True : match gradients of FGSM loss   (H278)

    Returns: X_syn (N_CLASSES*K x C x H x W), Y_syn (N_CLASSES*K,)
    """
    C.set_seed(SEED)
    class_idx = class_indices(Ytr)

    # Initialise synthetic images as random noise in [0,1]
    X_syn = torch.rand(N_CLASSES * K, 1, 28, 28, device=C.DEVICE, requires_grad=True)
    Y_syn = torch.arange(N_CLASSES, device=C.DEVICE).repeat_interleave(K)

    opt_syn = torch.optim.Adam([X_syn], lr=LR_SYN)

    for step in range(OUTER_STEPS):
        # Fresh model theta_t
        theta = build_fresh_model()
        for p in theta.parameters():
            p.requires_grad_(True)

        # Inner step on synthetic data
        inner_opt = torch.optim.SGD(theta.parameters(), lr=LR_INNER)
        inner_opt.zero_grad()
        x_syn_clamp = X_syn.clamp(0, 1)
        loss_inner = F.cross_entropy(theta(x_syn_clamp), Y_syn)
        loss_inner.backward(create_graph=False)
        inner_opt.step()

        # Compute G_real
        x_real, y_real = sample_real_batch(Xtr, class_idx, N_REAL_BATCH)
        if use_adversarial:
            # FGSM on real data using current theta
            for p in theta.parameters():
                p.requires_grad_(True)
            x_real_adv = x_real.clone().detach().requires_grad_(True)
            loss_r = F.cross_entropy(theta(x_real_adv), y_real)
            loss_r.backward()
            x_real = (x_real + EPS * x_real_adv.grad.sign()).clamp(0, 1).detach()

        for p in theta.parameters():
            p.requires_grad_(True)
        loss_real = F.cross_entropy(theta(x_real), y_real)
        g_real = torch.autograd.grad(loss_real, theta.parameters(),
                                     retain_graph=False, create_graph=False)
        g_real_flat = flatten_grads(g_real).detach()

        # Compute G_syn (with graph through X_syn)
        x_syn_clamp2 = X_syn.clamp(0, 1)
        loss_syn = F.cross_entropy(theta(x_syn_clamp2), Y_syn)
        g_syn = torch.autograd.grad(loss_syn, theta.parameters(),
                                    retain_graph=True, create_graph=True)
        g_syn_flat = flatten_grads(g_syn)

        # Gradient matching loss
        match_loss = cosine_distance(g_real_flat, g_syn_flat)

        opt_syn.zero_grad()
        match_loss.backward()
        opt_syn.step()

        if (step + 1) % 50 == 0:
            print(f"    outer step {step+1}/{OUTER_STEPS}  match_loss={match_loss.item():.4f}")

    return X_syn.clamp(0, 1).detach(), Y_syn


# ---------------------------------------------------------------------------
# Train a model on synthetic data
# ---------------------------------------------------------------------------

def train_on_synthetic(X_syn, Y_syn):
    C.set_seed(SEED)
    model = build_fresh_model()
    opt = torch.optim.SGD(model.parameters(), lr=LR_TRAIN, momentum=0.9,
                          weight_decay=5e-4)
    n = len(X_syn)
    for ep in range(TRAIN_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, max(1, n // 4)):
            idx = perm[i:i + max(1, n // 4)]
            xb = X_syn[idx].to(C.DEVICE)
            yb = Y_syn[idx].to(C.DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                loss = F.cross_entropy(model(X_syn.to(C.DEVICE)), Y_syn.to(C.DEVICE))
            print(f"    train-on-syn epoch {ep+1}/{TRAIN_EPOCHS}  loss={loss.item():.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, Xte, Yte, label):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, fgsm_acc = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, pgd_acc = C.logits_and_acc(model, Xpgd, Yte)
    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))
    fgsm_asr, pgd_asr = 1.0 - fgsm_acc, 1.0 - pgd_acc
    print(f"  [{label}] clean={clean_acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  "
          f"PGD_ASR={pgd_asr:.4f}  margin={mean_margin:.4f}")
    return dict(clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                pgd_asr=pgd_asr, mean_margin=mean_margin)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    C.set_seed(SEED)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    print("Loading data...")
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    Xte_eval, Yte_eval = Xte[:N_EVAL], Yte[:N_EVAL]
    print(f"  train={len(Xtr)}  eval={len(Xte_eval)}")

    results = {}

    # Baseline: train on full real data
    print("\n--- Baseline (full real data, 10 epochs) ---")
    C.set_seed(SEED)
    model_base = build_fresh_model()
    C.train_model(model_base, Xtr, Ytr, epochs=10)
    results["baseline_full"] = evaluate(model_base, Xte_eval, Yte_eval,
                                        label="baseline_full")

    # Distillation with clean gradients
    print("\n--- Distilling dataset (clean gradient matching, 200 steps) ---")
    X_syn_clean, Y_syn_clean = distill_clean(Xtr, Ytr, use_adversarial=False)

    # Synthetic image stats
    syn_stats = []
    for i in range(N_CLASSES * K):
        img = X_syn_clean[i].cpu().numpy().flatten()
        syn_stats.append((float(img.mean()), float(img.std())))
    print("  Synthetic image pixel stats (mean, std per image):")
    for i, (m, s) in enumerate(syn_stats):
        print(f"    class {i}: mean={m:.4f}  std={s:.4f}")

    print("\n--- Training on clean-distilled data (50 epochs) ---")
    model_distilled = train_on_synthetic(X_syn_clean, Y_syn_clean)
    results["clean_distilled"] = evaluate(model_distilled, Xte_eval, Yte_eval,
                                          label="clean_distilled")

    elapsed = time.time() - t0

    lines = [
        "H277 - Dataset Distillation (Clean Gradient Matching)\n",
        "=" * 70 + "\n\n",
        f"K={K} images/class  outer_steps={OUTER_STEPS}  train_epochs={TRAIN_EPOCHS}\n",
        f"N_eval={N_EVAL}  eps={EPS}  pgd_steps={PGD_STEPS}\n\n",
        f"{'Model':<20} {'CleanAcc':>10} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Margin':>10}\n",
        "-" * 60 + "\n",
    ]
    for name, r in results.items():
        lines.append(
            f"{name:<20} {r['clean_acc']:>10.4f} {r['fgsm_asr']:>10.4f} "
            f"{r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f}\n"
        )
    lines.append("\nSynthetic image pixel stats (mean, std per image):\n")
    for i, (m, s) in enumerate(syn_stats):
        lines.append(f"  class {i}: mean={m:.4f}  std={s:.4f}\n")
    lines.append(f"\nElapsed: {elapsed:.1f}s\n\n")
    lines.append("ANALYSIS\n--------\n")
    base = results["baseline_full"]
    dist = results["clean_distilled"]
    lines.append(f"Accuracy retained: {dist['clean_acc']/base['clean_acc']*100:.1f}% "
                 f"({dist['clean_acc']:.4f} vs {base['clean_acc']:.4f})\n")
    lines.append(f"PGD_ASR change (distilled vs baseline): {dist['pgd_asr']-base['pgd_asr']:+.4f}\n")
    verdict = ("Distillation largely preserves accuracy => synthetic images capture class structure."
               if dist["clean_acc"] > 0.5
               else "Distillation fails to preserve accuracy; synthetic images are insufficient.")
    lines.append(f"Verdict: {verdict}\n")

    with open(OUT_FILE, "w") as f:
        f.writelines(lines)
    print(f"\nResults written to {OUT_FILE}")
