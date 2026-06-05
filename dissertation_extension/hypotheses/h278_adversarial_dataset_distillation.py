"""
H278 - Adversarial Dataset Distillation: matching adversarial gradients.

Extension of H277. The key novel idea: instead of matching gradients of the
CLEAN loss on real data, match gradients of the ADVERSARIAL loss (FGSM examples
on real data). The synthetic images should then encode not just what the model
learns on average, but WHERE the decision boundaries lie.

Hypothesis: a model trained on adversarially-distilled synthetic images should
be more adversarially robust than one trained on clean-distilled images, because
the distillation process embeds boundary geometry into the synthetic pixels.

Comparison:
  (a) Baseline: full real data (10 epochs)
  (b) Clean-gradient distilled (H277 method): match grad of clean loss
  (c) Adversarial-gradient distilled (H278): match grad of FGSM loss on real data

K=1 image per class, 200 distillation steps, 50 training epochs on synthetic data.
N_eval=300 test samples.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS           = "fashion_mnist"
SEED         = 0
N_CLASSES    = 10
K            = 1
OUTER_STEPS  = 200
INNER_STEPS  = 1
LR_SYN       = 0.01
LR_INNER     = 0.01
LR_TRAIN     = 0.05
TRAIN_EPOCHS = 50
N_REAL_BATCH = 32
EPS          = 0.1
PGD_STEPS    = 10
PGD_ALPHA    = 0.01
N_EVAL       = 300
OUT_FILE     = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h278_adversarial_dataset_distillation_output.txt"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def class_indices(Y):
    return {c: (Y == c).nonzero(as_tuple=True)[0] for c in range(N_CLASSES)}


def sample_real_batch(X, class_idx, n_per_class):
    xs, ys = [], []
    for c, idx in class_idx.items():
        sel = idx[torch.randperm(len(idx))[:n_per_class]]
        xs.append(X[sel])
        ys.append(torch.full((len(sel),), c, dtype=torch.long))
    return torch.cat(xs).to(C.DEVICE), torch.cat(ys).to(C.DEVICE)


def flatten_grads(grads):
    return torch.cat([g.reshape(-1) for g in grads if g is not None])


def cosine_distance(g1, g2):
    return 1.0 - F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).squeeze()


def build_fresh_model():
    m = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": N_CLASSES},
                      width=32, seed=0)
    m.to(C.DEVICE)
    return m


# ---------------------------------------------------------------------------
# Distillation (parameterised by use_adversarial)
# ---------------------------------------------------------------------------

def distill(Xtr, Ytr, use_adversarial=False, label=""):
    """
    Gradient-matching distillation.
    use_adversarial=False -> H277 (clean loss matching)
    use_adversarial=True  -> H278 (FGSM adversarial loss matching)
    """
    C.set_seed(SEED)
    class_idx = class_indices(Ytr)
    X_syn = torch.rand(N_CLASSES * K, 1, 28, 28, device=C.DEVICE, requires_grad=True)
    Y_syn = torch.arange(N_CLASSES, device=C.DEVICE).repeat_interleave(K)
    opt_syn = torch.optim.Adam([X_syn], lr=LR_SYN)

    for step in range(OUTER_STEPS):
        theta = build_fresh_model()
        for p in theta.parameters():
            p.requires_grad_(True)

        # Inner update on synthetic data
        inner_opt = torch.optim.SGD(theta.parameters(), lr=LR_INNER)
        inner_opt.zero_grad()
        x_syn_clamp = X_syn.clamp(0, 1)
        F.cross_entropy(theta(x_syn_clamp), Y_syn).backward(create_graph=False)
        inner_opt.step()

        # Build real batch (possibly adversarial)
        x_real, y_real = sample_real_batch(Xtr, class_idx, N_REAL_BATCH)

        if use_adversarial:
            # FGSM adversarial perturbation of real data
            x_real = x_real.clone().detach().requires_grad_(True)
            loss_r_fgsm = F.cross_entropy(theta(x_real), y_real)
            loss_r_fgsm.backward()
            x_real = (x_real.detach() + EPS * x_real.grad.sign()).clamp(0, 1).detach()
        else:
            x_real = x_real.detach()

        # G_real: gradient of (adversarial) loss on real data
        for p in theta.parameters():
            p.requires_grad_(True)
        loss_real = F.cross_entropy(theta(x_real), y_real)
        g_real_list = torch.autograd.grad(loss_real, theta.parameters(),
                                          retain_graph=False, create_graph=False)
        g_real_flat = flatten_grads(g_real_list).detach()

        # G_syn: gradient of (adversarial) loss on synthetic data
        x_syn_c2 = X_syn.clamp(0, 1)
        if use_adversarial:
            # FGSM on synthetic images (no graph through perturbation)
            x_syn_c2_adv = x_syn_c2.clone().detach().requires_grad_(True)
            loss_syn_fgsm = F.cross_entropy(theta(x_syn_c2_adv), Y_syn)
            loss_syn_fgsm.backward()
            x_syn_adv = (x_syn_c2_adv.detach() + EPS * x_syn_c2_adv.grad.sign()).clamp(0, 1)
            # Now compute G_syn at adversarial synthetic images; need graph through X_syn
            # Approximation: use clean x_syn for G_syn graph (perturb direction is fixed above)
            loss_syn = F.cross_entropy(theta(x_syn_c2), Y_syn)
        else:
            loss_syn = F.cross_entropy(theta(x_syn_c2), Y_syn)

        g_syn_list = torch.autograd.grad(loss_syn, theta.parameters(),
                                         retain_graph=True, create_graph=True)
        g_syn_flat = flatten_grads(g_syn_list)

        match_loss = cosine_distance(g_real_flat, g_syn_flat)
        opt_syn.zero_grad()
        match_loss.backward()
        opt_syn.step()

        if (step + 1) % 50 == 0:
            print(f"    [{label}] outer step {step+1}/{OUTER_STEPS}  "
                  f"match_loss={match_loss.item():.4f}")

    return X_syn.clamp(0, 1).detach(), Y_syn


# ---------------------------------------------------------------------------
# Train on synthetic data
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
    results["baseline_full"] = evaluate(model_base, Xte_eval, Yte_eval, "baseline_full")

    # Clean-gradient distillation (H277 method, reproduced here for fair comparison)
    print("\n--- Clean-gradient distillation (200 steps) ---")
    X_syn_clean, Y_syn_clean = distill(Xtr, Ytr, use_adversarial=False, label="clean")
    print("\n--- Training on clean-distilled data ---")
    model_clean_dist = train_on_synthetic(X_syn_clean, Y_syn_clean)
    results["clean_distilled"] = evaluate(model_clean_dist, Xte_eval, Yte_eval, "clean_distilled")

    # Adversarial-gradient distillation (H278 novel method)
    print("\n--- Adversarial-gradient distillation (200 steps) ---")
    X_syn_adv, Y_syn_adv = distill(Xtr, Ytr, use_adversarial=True, label="adversarial")
    print("\n--- Training on adversarial-distilled data ---")
    model_adv_dist = train_on_synthetic(X_syn_adv, Y_syn_adv)
    results["adv_distilled"] = evaluate(model_adv_dist, Xte_eval, Yte_eval, "adv_distilled")

    # Pixel stats for both synthetic sets
    def syn_pixel_stats(X_syn):
        stats = []
        for i in range(N_CLASSES * K):
            img = X_syn[i].cpu().numpy().flatten()
            stats.append((float(img.mean()), float(img.std())))
        return stats

    clean_stats = syn_pixel_stats(X_syn_clean)
    adv_stats   = syn_pixel_stats(X_syn_adv)

    elapsed = time.time() - t0

    lines = [
        "H278 - Adversarial Dataset Distillation\n",
        "=" * 70 + "\n\n",
        f"K={K} images/class  outer_steps={OUTER_STEPS}  train_epochs={TRAIN_EPOCHS}\n",
        f"N_eval={N_EVAL}  eps={EPS}  pgd_steps={PGD_STEPS}\n\n",
        f"{'Model':<22} {'CleanAcc':>10} {'FGSM_ASR':>10} {'PGD_ASR':>10} {'Margin':>10}\n",
        "-" * 62 + "\n",
    ]
    for name, r in results.items():
        lines.append(
            f"{name:<22} {r['clean_acc']:>10.4f} {r['fgsm_asr']:>10.4f} "
            f"{r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f}\n"
        )
    lines.append("\nSynthetic pixel stats -- clean distillation:\n")
    for i, (m, s) in enumerate(clean_stats):
        lines.append(f"  class {i}: mean={m:.4f}  std={s:.4f}\n")
    lines.append("\nSynthetic pixel stats -- adversarial distillation:\n")
    for i, (m, s) in enumerate(adv_stats):
        lines.append(f"  class {i}: mean={m:.4f}  std={s:.4f}\n")
    lines.append(f"\nElapsed: {elapsed:.1f}s\n\n")
    lines.append("ANALYSIS\n--------\n")
    base  = results["baseline_full"]
    clean = results["clean_distilled"]
    adv   = results["adv_distilled"]
    lines.append(f"Adv-distilled vs clean-distilled: PGD_ASR delta={adv['pgd_asr']-clean['pgd_asr']:+.4f}\n")
    lines.append(f"Adv-distilled vs baseline:        PGD_ASR delta={adv['pgd_asr']-base['pgd_asr']:+.4f}\n")
    lines.append(f"Clean acc cost of adv distillation: {adv['clean_acc']-clean['clean_acc']:+.4f}\n")
    verdict = ("Adversarial gradient distillation improves robustness vs clean distillation, "
               "supporting the boundary-geometry encoding hypothesis."
               if adv["pgd_asr"] < clean["pgd_asr"]
               else "Adversarial gradient distillation does NOT improve robustness vs clean distillation; "
               "boundary-geometry encoding via gradient matching may require more steps or higher K.")
    lines.append(f"Verdict: {verdict}\n")

    with open(OUT_FILE, "w") as f:
        f.writelines(lines)
    print(f"\nResults written to {OUT_FILE}")
