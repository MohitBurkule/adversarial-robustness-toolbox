"""
H230 - Early vs late epoch adversarial examples — do they transfer to final model?

Train CNN, save checkpoints at epochs [1, 3, 5, 10, 15, 20].
For each source_epoch checkpoint:
  - Generate FGSM and PGD-10 adversarials on Xte[:N_EVAL]
  - Test those adversarials on the FINAL model (epoch 20)
  - Measure ASR(source_epoch_adversarials, final_model)

Also compute: gradient cosine similarity between source_epoch gradients and final_model gradients.
  cos_sim = mean cosine_similarity(∇_x_source, ∇_x_final) across test samples

Print table: source_epoch -> FGSM_transfer_ASR, PGD_transfer_ASR, gradient_cosine_sim
Key finding: does ASR(transfer) and gradient_cos_sim both increase monotonically with source_epoch?
"""
import os, sys, copy, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_EVAL = 300
SEED = 0
EPS = 0.1
CHECKPOINTS = [1, 3, 5, 10, 15, 20]
FINAL_EPOCH = 20

os.makedirs("results/fashion_mnist", exist_ok=True)


def get_input_gradients(model, X, Y):
    """Return per-sample input gradients (loss wrt input), shape (N, C, H, W)."""
    device = next(model.parameters()).device
    X = X.to(device).clone().detach().requires_grad_(True)
    Y = Y.to(device)
    model.eval()
    loss = F.cross_entropy(model(X), Y, reduction="sum")
    loss.backward()
    grads = X.grad.detach().cpu()  # (N, C, H, W)
    return grads


def cosine_sim_batch(g1, g2):
    """Mean cosine similarity between two (N, C, H, W) gradient tensors."""
    g1_flat = g1.flatten(1)  # (N, D)
    g2_flat = g2.flatten(1)
    cos = F.cosine_similarity(g1_flat, g2_flat, dim=1)  # (N,)
    return float(cos.mean())


def attack_success_rate(model, X_adv, Y_true):
    """Fraction of X_adv that fool the model (predicted != Y_true)."""
    device = next(model.parameters()).device
    with torch.no_grad():
        preds = model(X_adv.to(device)).argmax(1).cpu()
    return float((preds != Y_true.cpu()).float().mean())


def train_with_checkpoints(model, Xtr, Ytr, checkpoints, epochs):
    """Train model for `epochs`, returning copies at each checkpoint epoch."""
    device = next(model.parameters()).device
    saved = {}
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()

    dataset = torch.utils.data.TensorDataset(Xtr.cpu(), Ytr.cpu())
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(1, epochs + 1):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        if epoch in checkpoints:
            saved[epoch] = copy.deepcopy(model.state_dict())
            acc = float((model(Xtr[:500].to(device)).argmax(1).cpu() == Ytr[:500].cpu()).float().mean())
            print(f"  Epoch {epoch:>2}: train_acc={acc:.3f}  [checkpoint saved]")
    return saved


def main():
    t0 = time.time()
    C.set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_EVAL, seed=SEED)
    Xte, Yte = Xte[:N_EVAL].to(device), Yte[:N_EVAL].to(device)
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)

    model = C.build_model("cnn", meta, width=32, seed=SEED)
    model = model.to(device)

    print(f"\nTraining with checkpoints at epochs {CHECKPOINTS}...")
    saved_states = train_with_checkpoints(model, Xtr, Ytr, CHECKPOINTS, FINAL_EPOCH)

    # Final model is already trained; load final state
    model.load_state_dict(saved_states[FINAL_EPOCH])
    model.eval()
    final_state = copy.deepcopy(saved_states[FINAL_EPOCH])

    # Evaluate final model baseline
    with torch.no_grad():
        final_clean_acc = float((model(Xte).argmax(1) == Yte).float().mean())
    print(f"\nFinal model clean accuracy: {final_clean_acc:.3f}")

    # Compute final model input gradients (reference)
    grads_final = get_input_gradients(model, Xte, Yte)

    # Rebuild a fresh model for each source epoch
    print(f"\n{'SrcEpoch':>9}  {'FGSM_ASR':>8}  {'PGD_ASR':>8}  {'Grad_CosSim':>12}")
    print("-" * 50)

    results = []
    for epoch in CHECKPOINTS:
        src_model = C.build_model("cnn", meta, width=32, seed=SEED)
        src_model = src_model.to(device)
        src_model.load_state_dict(saved_states[epoch])
        src_model.eval()

        # Generate adversarials from source epoch model
        X_fgsm_src = C.fgsm(src_model, Xte, Yte, eps=EPS)
        X_pgd_src = C.pgd(src_model, Xte, Yte, eps=EPS, steps=10, alpha=0.01)

        # Test on final model
        # Reload final model
        model.load_state_dict(final_state)
        model.eval()

        fgsm_asr = attack_success_rate(model, X_fgsm_src, Yte)
        pgd_asr = attack_success_rate(model, X_pgd_src, Yte)

        # Gradient cosine similarity
        grads_src = get_input_gradients(src_model, Xte, Yte)
        cos_sim = cosine_sim_batch(grads_src, grads_final)

        results.append((epoch, fgsm_asr, pgd_asr, cos_sim))
        print(f"{epoch:>9}  {fgsm_asr:>8.3f}  {pgd_asr:>8.3f}  {cos_sim:>12.4f}")

    # Check monotonicity
    print("\n--- Monotonicity check ---")
    fgsm_asrs = [r[1] for r in results]
    pgd_asrs = [r[2] for r in results]
    cos_sims = [r[3] for r in results]

    def is_monotone_inc(seq):
        return all(b >= a for a, b in zip(seq, seq[1:]))

    print(f"FGSM_ASR monotone increasing: {is_monotone_inc(fgsm_asrs)}")
    print(f"PGD_ASR  monotone increasing: {is_monotone_inc(pgd_asrs)}")
    print(f"Grad_CosSim monotone increasing: {is_monotone_inc(cos_sims)}")

    # White-box ASR on final model as upper bound
    model.load_state_dict(final_state)
    model.eval()
    X_fgsm_final = C.fgsm(model, Xte, Yte, eps=EPS)
    X_pgd_final = C.pgd(model, Xte, Yte, eps=EPS, steps=10, alpha=0.01)
    ub_fgsm = attack_success_rate(model, X_fgsm_final, Yte)
    ub_pgd = attack_success_rate(model, X_pgd_final, Yte)
    print(f"\nUpper bound (white-box on final model): FGSM={ub_fgsm:.3f}  PGD={ub_pgd:.3f}")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
