"""
H202 - AT models are more sensitive to aggressive pruning than NT models.

Hypothesis: at moderate sparsity (70%), adversarially trained models lose < 3 pp
robust accuracy; but at high sparsity (90%), AT models lose more robust accuracy
than NT models lose clean accuracy, with a crossover between 80-85% sparsity.

Grounded in: arXiv:2202.09844 (robust lottery tickets up to ~80% sparsity),
             arXiv:2311.15782 (AT models more sensitive to aggressive pruning).

Protocol:
  - Train two models on Fashion-MNIST (n_train=6000):
      NT: standard cross-entropy (15 epochs)
      AT: PGD-7 adversarial training (eps=0.3, 15 epochs)
  - Apply iterative magnitude pruning at sparsity [0%, 50%, 70%, 80%, 85%, 90%]:
      Prune lowest-magnitude weights globally, fine-tune 3 epochs with mask.
  - Evaluate: clean accuracy (both), PGD-10 robust accuracy (both).
  - Key: crossover sparsity where AT robust_acc_drop > NT clean_acc_drop.

Default dataset Fashion-MNIST; patchable via run_with_patch.py.
"""
import sys, os, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import campaign.common as C

DEVICE = C.DEVICE
SEED = 42
N_TRAIN = 6000
N_EVAL = 500
EPOCHS = 15
FINETUNE_EPOCHS = 3
EPS = 0.3
AT_STEPS = 7
EVAL_PGD_STEPS = 10
BATCH = 128
SPARSITY_LEVELS = [0.0, 0.50, 0.70, 0.80, 0.85, 0.90]


def train_standard(model, Xtr, Ytr, epochs=EPOCHS):
    """Train with standard cross-entropy."""
    return C.train_model(model, Xtr, Ytr, epochs=epochs, batch=BATCH,
                         opt="adam", lr=1e-3, verbose=False)


def train_adversarial(model, Xtr, Ytr, epochs=EPOCHS):
    """Train with PGD-7 adversarial training."""
    return C.train_model(model, Xtr, Ytr, epochs=epochs, batch=BATCH,
                         opt="adam", lr=1e-3, adv_train=True, adv_eps=EPS,
                         adv_steps=AT_STEPS, verbose=False)


def get_prunable_params(model):
    """Return list of (module, 'weight') for all Conv2d and Linear layers."""
    params = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params.append((module, 'weight'))
    return params


def apply_global_pruning(model, sparsity):
    """Apply global L1 unstructured pruning at the given sparsity level."""
    if sparsity <= 0:
        return model
    params = get_prunable_params(model)
    prune.global_unstructured(
        params, pruning_method=prune.L1Unstructured, amount=sparsity
    )
    return model


def remove_pruning_reparametrization(model):
    """Make pruning permanent (remove hooks, apply mask)."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
    return model


def count_sparsity(model):
    """Compute actual global sparsity of the model."""
    total = 0
    zeros = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def finetune_pruned(model, Xtr, Ytr, epochs=FINETUNE_EPOCHS, adversarial=False):
    """Fine-tune a pruned model, zeroing gradients of pruned weights each step."""
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            if adversarial:
                model.eval()
                xb = C.pgd(model, xb, yb, eps=EPS, steps=AT_STEPS,
                           alpha=2.5 * EPS / AT_STEPS)
                model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            # Zero out gradients for pruned (masked) weights
            # The prune hooks keep weight_orig and weight_mask;
            # we zero grad on weight_orig where mask is 0
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if hasattr(module, 'weight_mask'):
                        module.weight_orig.grad.data *= module.weight_mask
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def eval_clean_acc(model, X, Y, batch=256):
    model.eval()
    correct = 0
    total = 0
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total


def eval_robust_acc(model, X, Y, eps=EPS, steps=EVAL_PGD_STEPS, batch=256):
    """Robust accuracy = fraction of samples correctly classified after PGD."""
    model.eval()
    correct = 0
    total = 0
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch], Y[i:i + batch]
        xa = C.pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            pred = model(xa).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total


def main():
    print("=" * 74)
    print("H202 - AT models are more sensitive to aggressive pruning than NT models")
    print("=" * 74)
    print(f"Device={DEVICE}  EPOCHS={EPOCHS}  N_TRAIN={N_TRAIN}  N_EVAL={N_EVAL}")
    print(f"PGD AT: eps={EPS}  steps={AT_STEPS}")
    print(f"PGD eval: eps={EPS}  steps={EVAL_PGD_STEPS}")
    print(f"Sparsity levels: {SPARSITY_LEVELS}")
    print(f"Fine-tune epochs per level: {FINETUNE_EPOCHS}")

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN,
                                          n_eval=N_EVAL, seed=SEED)

    # ---- Train NT model ----
    print("\n--- Training NT (standard) model ---")
    t0 = time.time()
    model_nt = C.build_model("cnn", meta, width=32)
    train_standard(model_nt, Xtr, Ytr)
    print(f"  trained in {time.time() - t0:.1f}s")

    # ---- Train AT model ----
    print("\n--- Training AT (adversarial) model ---")
    t0 = time.time()
    model_at = C.build_model("cnn", meta, width=32)
    train_adversarial(model_at, Xtr, Ytr)
    print(f"  trained in {time.time() - t0:.1f}s")

    # ---- Baseline evaluation ----
    nt_clean_0 = eval_clean_acc(model_nt, Xte, Yte)
    at_clean_0 = eval_clean_acc(model_at, Xte, Yte)
    nt_robust_0 = eval_robust_acc(model_nt, Xte, Yte)
    at_robust_0 = eval_robust_acc(model_at, Xte, Yte)
    print(f"\n  Baseline NT: clean={nt_clean_0:.4f}  robust={nt_robust_0:.4f}")
    print(f"  Baseline AT: clean={at_clean_0:.4f}  robust={at_robust_0:.4f}")

    # Save base models for pruning
    nt_state = copy.deepcopy(model_nt.state_dict())
    at_state = copy.deepcopy(model_at.state_dict())

    # ---- Pruning sweep ----
    print("\n--- Pruning sweep ---")
    results = []

    for sp in SPARSITY_LEVELS:
        print(f"\n  Sparsity = {sp:.0%}")

        # NT model
        m_nt = C.build_model("cnn", meta, width=32)
        m_nt.load_state_dict(copy.deepcopy(nt_state))
        if sp > 0:
            apply_global_pruning(m_nt, sp)
            finetune_pruned(m_nt, Xtr, Ytr, adversarial=False)
        actual_sp_nt = count_sparsity(m_nt)
        nt_clean = eval_clean_acc(m_nt, Xte, Yte)
        nt_robust = eval_robust_acc(m_nt, Xte, Yte)

        # AT model
        m_at = C.build_model("cnn", meta, width=32)
        m_at.load_state_dict(copy.deepcopy(at_state))
        if sp > 0:
            apply_global_pruning(m_at, sp)
            finetune_pruned(m_at, Xtr, Ytr, adversarial=True)
        actual_sp_at = count_sparsity(m_at)
        at_clean = eval_clean_acc(m_at, Xte, Yte)
        at_robust = eval_robust_acc(m_at, Xte, Yte)

        nt_clean_drop = nt_clean_0 - nt_clean
        at_robust_drop = at_robust_0 - at_robust

        results.append({
            "sparsity": sp,
            "nt_clean": nt_clean, "nt_robust": nt_robust,
            "at_clean": at_clean, "at_robust": at_robust,
            "nt_clean_drop": nt_clean_drop,
            "at_robust_drop": at_robust_drop,
            "actual_sp_nt": actual_sp_nt, "actual_sp_at": actual_sp_at,
        })
        print(f"    NT: clean={nt_clean:.4f} robust={nt_robust:.4f} "
              f"clean_drop={nt_clean_drop:+.4f}")
        print(f"    AT: clean={at_clean:.4f} robust={at_robust:.4f} "
              f"robust_drop={at_robust_drop:+.4f}")

    # ---- Summary table ----
    print("\n" + "=" * 74)
    print("--- Summary table ---")
    print(f"  {'sparsity':>8} {'NT_clean':>9} {'AT_clean':>9} {'AT_robust':>10} "
          f"{'NT_c_drop':>10} {'AT_r_drop':>10}")
    print(f"  {'-' * 58}")
    for r in results:
        print(f"  {r['sparsity']:>8.0%} {r['nt_clean']:>9.4f} {r['at_clean']:>9.4f} "
              f"{r['at_robust']:>10.4f} {r['nt_clean_drop']:>+10.4f} "
              f"{r['at_robust_drop']:>+10.4f}")

    # ---- Crossover analysis ----
    print("\n--- Crossover analysis ---")
    print("  (Crossover = sparsity where AT_robust_drop > NT_clean_drop)")
    crossover = None
    for i, r in enumerate(results):
        if r["at_robust_drop"] > r["nt_clean_drop"] and r["sparsity"] > 0:
            crossover = r["sparsity"]
            if i > 0:
                prev = results[i - 1]
                print(f"  Crossover between {prev['sparsity']:.0%} and {r['sparsity']:.0%}")
            else:
                print(f"  Crossover at or before {r['sparsity']:.0%}")
            break

    if crossover is None:
        print("  No crossover found (AT never lost more robust acc than NT lost clean acc)")

    # ---- Hypothesis tests ----
    print("\n--- Hypothesis tests ---")
    # Test 1: at 70%, AT loses < 3pp robust accuracy
    r70 = [r for r in results if abs(r["sparsity"] - 0.70) < 0.01]
    if r70:
        drop_70 = r70[0]["at_robust_drop"]
        test1 = drop_70 < 0.03
        print(f"  [1] AT robust_drop at 70%: {drop_70:.4f} pp "
              f"(< 3pp? {'YES' if test1 else 'NO'})")
    else:
        print("  [1] 70% sparsity not tested")

    # Test 2: crossover between 80-85%
    if crossover is not None:
        test2 = 0.80 <= crossover <= 0.85
        print(f"  [2] Crossover at {crossover:.0%} "
              f"(between 80-85%? {'YES' if test2 else 'NO'})")
    else:
        print("  [2] No crossover found")

    # Test 3: at 90%, AT robust drop > NT clean drop
    r90 = [r for r in results if abs(r["sparsity"] - 0.90) < 0.01]
    if r90:
        test3 = r90[0]["at_robust_drop"] > r90[0]["nt_clean_drop"]
        print(f"  [3] At 90%: AT_robust_drop ({r90[0]['at_robust_drop']:.4f}) > "
              f"NT_clean_drop ({r90[0]['nt_clean_drop']:.4f})? "
              f"{'YES' if test3 else 'NO'}")

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
