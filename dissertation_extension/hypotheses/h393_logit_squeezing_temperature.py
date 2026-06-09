"""
H393: Logit Squeezing + Temperature Scaling
Hypothesis: explicit logit-L2 penalty ("logit squeezing") plus training-time temperature
reduces FGSM exploitability by shrinking gradient magnitude — but check it's not just masking.

Anti-masking checks:
  - FGSM-vs-PGD gap (large gap = masking)
  - Black-box transfer ASR (PGD from a standard baseline model)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import campaign.common as C

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "h393_logit_squeezing_temperature_output.txt")

# Shared config
META = {"channels": 1, "size": 28, "n_classes": 10}
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01

def make_optimizer(model):
    return torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=5e-4
    )

def train_with_logit_squeeze(Xtr, Ytr, T=1.0, lam=0.0):
    C.set_seed(SEED)
    model = C.build_model("cnn", META, width=32, seed=SEED)
    opt = make_optimizer(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            z = model(xb)  # logits
            ce_loss = F.cross_entropy(z / T, yb)
            sq_loss = lam * z.pow(2).mean()
            loss = ce_loss + sq_loss
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model

def eval_model(model, Xte, Yte, transfer_model=None):
    # Clean accuracy
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)

    # FGSM ASR
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, fgsm_acc = C.logits_and_acc(model, Xfgsm, Yte)
    fgsm_asr = 1 - fgsm_acc

    # PGD ASR
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, pgd_acc = C.logits_and_acc(model, Xpgd, Yte)
    pgd_asr = 1 - pgd_acc

    # Margin
    mg = C.margin(model, Xte, Yte).mean()

    # Transfer ASR: PGD adversarial examples from transfer_model evaluated on this model
    transfer_asr = float('nan')
    if transfer_model is not None:
        Xtransfer = C.pgd(transfer_model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
        _, trans_acc = C.logits_and_acc(model, Xtransfer, Yte)
        transfer_asr = 1 - trans_acc

    fgsm_pgd_gap = fgsm_asr - pgd_asr

    return {
        "clean_acc": clean_acc,
        "fgsm_asr": fgsm_asr,
        "pgd_asr": pgd_asr,
        "margin": mg,
        "transfer_asr": transfer_asr,
        "fgsm_pgd_gap": fgsm_pgd_gap,
    }

def main():
    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 70)
    p("H393: Logit Squeezing + Temperature Scaling")
    p("=" * 70)

    # Load data
    p("\nLoading Fashion-MNIST...")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)
    p(f"Train: {Xtr.shape}, Test: {Xte.shape}")

    # Train standard baseline model (for transfer ASR)
    p("\nTraining standard baseline model (T=1, lambda=0)...")
    baseline_model = train_with_logit_squeeze(Xtr, Ytr, T=1.0, lam=0.0)
    baseline_metrics = eval_model(baseline_model, Xte, Yte, transfer_model=None)
    p(f"Baseline: clean_acc={baseline_metrics['clean_acc']:.3f}, "
      f"fgsm_asr={baseline_metrics['fgsm_asr']:.3f}, "
      f"pgd_asr={baseline_metrics['pgd_asr']:.3f}, "
      f"margin={baseline_metrics['margin']:.3f}")

    # Grid search
    T_vals = [1, 2, 4]
    lam_vals = [0.0, 0.001, 0.01]

    results = {}
    p("\nRunning grid T x lambda...")

    for T in T_vals:
        for lam in lam_vals:
            if T == 1 and lam == 0.0:
                # Already trained as baseline
                m = baseline_model
            else:
                p(f"  Training T={T}, lambda={lam}...")
                m = train_with_logit_squeeze(Xtr, Ytr, T=T, lam=lam)
            metrics = eval_model(m, Xte, Yte, transfer_model=baseline_model)
            results[(T, lam)] = metrics
            p(f"  T={T}, lam={lam}: clean={metrics['clean_acc']:.3f} "
              f"fgsm_asr={metrics['fgsm_asr']:.3f} pgd_asr={metrics['pgd_asr']:.3f} "
              f"margin={metrics['margin']:.3f} transfer_asr={metrics['transfer_asr']:.3f} "
              f"gap={metrics['fgsm_pgd_gap']:.3f}")

    # Print results table
    p("\n" + "=" * 70)
    p("RESULTS TABLE")
    p("=" * 70)
    header = f"{'T':>4} {'lambda':>8} {'clean':>7} {'fgsm_asr':>9} {'pgd_asr':>8} {'margin':>7} {'transfer':>9} {'gap':>6} {'masking?':>9}"
    p(header)
    p("-" * 70)
    for T in T_vals:
        for lam in lam_vals:
            m = results[(T, lam)]
            masking = "YES" if m['fgsm_pgd_gap'] > 0.15 or m['transfer_asr'] >= m['pgd_asr'] - 0.05 else "no"
            p(f"{T:>4} {lam:>8.3f} {m['clean_acc']:>7.3f} {m['fgsm_asr']:>9.3f} "
              f"{m['pgd_asr']:>8.3f} {m['margin']:>7.3f} {m['transfer_asr']:>9.3f} "
              f"{m['fgsm_pgd_gap']:>6.3f} {masking:>9}")

    p("\n" + "=" * 70)
    p("VERDICT")
    p("=" * 70)

    # Find best by PGD ASR (most honest metric)
    best_key = min(results, key=lambda k: results[k]['pgd_asr'])
    best = results[best_key]
    baseline = results[(1, 0.0)]

    pgd_reduction = baseline['pgd_asr'] - best['pgd_asr']
    is_masking = best['fgsm_pgd_gap'] > 0.15 or best['transfer_asr'] >= best['pgd_asr'] - 0.05

    p(f"Best config: T={best_key[0]}, lambda={best_key[1]}")
    p(f"PGD ASR: {baseline['pgd_asr']:.3f} (baseline) -> {best['pgd_asr']:.3f} (best), "
      f"reduction={pgd_reduction:.3f}")
    p(f"FGSM-vs-PGD gap: {best['fgsm_pgd_gap']:.3f} "
      f"(>0.15 suggests masking)")
    p(f"Transfer ASR: {best['transfer_asr']:.3f} vs PGD ASR: {best['pgd_asr']:.3f}")

    if is_masking:
        p("\nVERDICT: MASKING — logit squeezing + temperature appears to obfuscate "
          "gradients rather than provide genuine robustness. "
          "Transfer ASR ≈ white-box PGD ASR or large FGSM-PGD gap detected.")
    elif pgd_reduction > 0.05:
        p("\nVERDICT: GENUINE ROBUSTNESS — PGD ASR reduced meaningfully without "
          "masking indicators. Logit squeezing + temperature provides real benefit.")
    else:
        p("\nVERDICT: MARGINAL EFFECT — neither meaningful robustness gain nor "
          "clear masking. Effect size is small.")

    output = "\n".join(lines)
    with open(OUTFILE, "w") as f:
        f.write(output)
    print(f"\nResults saved to {OUTFILE}")

if __name__ == "__main__":
    main()
