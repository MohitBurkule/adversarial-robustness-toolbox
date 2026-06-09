"""
H234 - LSB bit-plane randomisation as implicit adversarial smoothing.

For K in [0, 1, 2, 3, 4]: randomly flip bottom-K bits of each pixel during training.

Implementation:
  x_uint = (x * 255).round().to(torch.int32)
  noise = torch.randint(0, 2**K, x_uint.shape, device=x.device).to(torch.int32)
  x_flipped = ((x_uint ^ noise).clamp(0, 255).to(torch.float32)) / 255.0

Train CNN on LSB-flipped training images (different random flip each batch).
At test time: no flipping (standard test images).

Also test with K-flip at test time: run model 10 times, take majority vote.

Measure: clean_acc, FGSM ASR, PGD ASR, majority_vote_clean_acc, majority_vote_fgsm_asr.

Hypothesis: K=1-2 bit flipping destroys adversarial perturbations in the LSB range.
  eps=0.1 ≈ 25/255 ≈ 5 bits, but perturbation spread across pixels.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_EVAL = 300
SEED = 0
EPS = 0.1
PGD_STEPS = 10
EPOCHS = 10
K_VALUES = [0, 1, 2, 3, 4]
VOTE_RUNS = 10

os.makedirs("results/fashion_mnist", exist_ok=True)


def lsb_flip(x, K):
    """Randomly flip bottom-K bits of each pixel value."""
    if K == 0:
        return x
    x_uint = (x * 255).round().to(torch.int32)
    noise = torch.randint(0, 2 ** K, x_uint.shape, device=x.device, dtype=torch.int32)
    x_flipped = (x_uint ^ noise).clamp(0, 255).to(torch.float32) / 255.0
    return x_flipped


def train_with_lsb(K, Xtr, Ytr, meta, seed):
    """Train CNN with K-bit LSB flipping applied to each batch."""
    C.set_seed(seed)
    model = C.build_model("cnn", meta, width=32, seed=seed)

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = Xtr[idx], Ytr[idx]
            xb_flip = lsb_flip(xb, K)
            opt.zero_grad()
            out = model(xb_flip)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def majority_vote_predict(model, X, K, n_runs=VOTE_RUNS):
    """Run model n_runs times with different K-bit flips, return majority vote predictions."""
    model.eval()
    votes = []
    with torch.no_grad():
        for _ in range(n_runs):
            Xf = lsb_flip(X, K) if K > 0 else X
            logits = model(Xf)
            votes.append(logits.argmax(1))
    # stack and take majority vote
    votes = torch.stack(votes, dim=1)  # (N, n_runs)
    majority = torch.mode(votes, dim=1).values
    return majority


def evaluate_model(model, K, Xte, Yte, n_eval=N_EVAL, eps=EPS, steps=PGD_STEPS):
    model.eval()
    X = Xte[:n_eval]
    Y = Yte[:n_eval]

    # Standard clean accuracy (no flipping at test time)
    with torch.no_grad():
        logits, clean_acc = C.logits_and_acc(model, X, Y)

    # FGSM and PGD ASR (no flipping at test time)
    fgsm_res = C.attack_success(model, X, Y, attack="fgsm", eps=eps)
    pgd_res = C.attack_success(model, X, Y, attack="pgd", eps=eps, steps=steps)

    # Majority vote clean accuracy (K-flip at test time)
    mv_preds = majority_vote_predict(model, X, K, n_runs=VOTE_RUNS)
    mv_clean_acc = float((mv_preds == Y).float().mean())

    # Majority vote FGSM ASR
    # compute FGSM adversarial examples (against standard model, no flip)
    Xa_fgsm = C.fgsm(model, X, Y, eps=eps)
    # get majority vote predictions on adversarial examples
    mv_fgsm_preds = majority_vote_predict(model, Xa_fgsm, K, n_runs=VOTE_RUNS)
    # ASR = fraction of originally correct samples that get flipped
    with torch.no_grad():
        orig_corr = (model(X).argmax(1) == Y)
    if orig_corr.sum() > 0:
        mv_fgsm_asr = float((mv_fgsm_preds[orig_corr] != Y[orig_corr]).float().mean())
    else:
        mv_fgsm_asr = float("nan")

    return {
        "clean_acc": clean_acc,
        "fgsm_asr": fgsm_res["asr"],
        "pgd_asr": pgd_res["asr"],
        "mv_clean_acc": mv_clean_acc,
        "mv_fgsm_asr": mv_fgsm_asr,
    }


def main():
    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=6000, n_eval=N_EVAL, seed=SEED)

    print("H234 - LSB Bit-Plane Randomisation as Adversarial Smoothing")
    print("=" * 80)
    print(f"Dataset: {DS}, N_EVAL={N_EVAL}, EPS={EPS}, PGD_STEPS={PGD_STEPS}, EPOCHS={EPOCHS}")
    print(f"VOTE_RUNS={VOTE_RUNS} (for majority vote test-time defence)")
    print(f"eps={EPS} = {int(EPS*255)}/255 ≈ {int(EPS*255).bit_length()} bits")

    results = []
    for K in K_VALUES:
        print(f"\n  Training CNN with K={K} LSB flipping...")
        model = train_with_lsb(K, Xtr, Ytr, meta, SEED)
        res = evaluate_model(model, K, Xte, Yte)
        results.append((K, res))
        print(f"  K={K}: clean={res['clean_acc']:.4f}, fgsm_asr={res['fgsm_asr']:.4f}, "
              f"pgd_asr={res['pgd_asr']:.4f}, mv_clean={res['mv_clean_acc']:.4f}, "
              f"mv_fgsm_asr={res['mv_fgsm_asr']:.4f}")

    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print(f"{'K':>3}  {'clean_acc':>10}  {'fgsm_asr':>9}  {'pgd_asr':>8}  "
          f"{'mv_clean_acc':>13}  {'mv_fgsm_asr':>12}")
    print("-" * 80)
    for K, res in results:
        print(f"{K:>3}  {res['clean_acc']:>10.4f}  {res['fgsm_asr']:>9.4f}  "
              f"{res['pgd_asr']:>8.4f}  {res['mv_clean_acc']:>13.4f}  {res['mv_fgsm_asr']:>12.4f}")

    print("\nHypothesis check: K=1-2 bit flipping reduces ASR")
    base_fgsm = results[0][1]["fgsm_asr"]
    base_pgd = results[0][1]["pgd_asr"]
    print(f"  Baseline (K=0): FGSM ASR={base_fgsm:.4f}, PGD ASR={base_pgd:.4f}")
    for K, res in results[1:]:
        fgsm_drop = base_fgsm - res["fgsm_asr"]
        pgd_drop = base_pgd - res["pgd_asr"]
        print(f"  K={K}: FGSM drop={fgsm_drop:+.4f}, PGD drop={pgd_drop:+.4f}, "
              f"clean drop={results[0][1]['clean_acc'] - res['clean_acc']:+.4f}")

    print("\nMajority vote effect (test-time K-flip, VOTE_RUNS={})".format(VOTE_RUNS))
    for K, res in results:
        if K > 0:
            clean_delta = res["mv_clean_acc"] - res["clean_acc"]
            fgsm_delta = res["mv_fgsm_asr"] - res["fgsm_asr"]
            print(f"  K={K}: mv_clean_delta={clean_delta:+.4f}, mv_fgsm_asr_delta={fgsm_delta:+.4f}")

    # overall conclusion
    print("\nConclusion:")
    best_pgd_k = min(results, key=lambda r: r[1]["pgd_asr"] if not np.isnan(r[1]["pgd_asr"]) else 1e9)
    print(f"  Best PGD robustness at K={best_pgd_k[0]} (PGD ASR={best_pgd_k[1]['pgd_asr']:.4f})")
    best_clean_k = max(results, key=lambda r: r[1]["clean_acc"])
    print(f"  Best clean acc at K={best_clean_k[0]} (clean={best_clean_k[1]['clean_acc']:.4f})")
    if best_pgd_k[0] in [1, 2]:
        print("  -> CONFIRMED: K=1-2 range provides best robustness tradeoff")
    else:
        print(f"  -> K={best_pgd_k[0]} best, outside K=1-2 prediction")


if __name__ == "__main__":
    main()
