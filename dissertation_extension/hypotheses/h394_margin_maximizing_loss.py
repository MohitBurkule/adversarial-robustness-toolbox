"""
H394: Margin-Maximizing Loss
Hypothesis: replacing cross-entropy with an explicit large-margin loss increases
input-space margin and reduces FGSM ASR.

Conditions:
  (a) Standard CE baseline
  (b) Multiclass hinge loss: mean over wrong classes of relu(m - (z_y - z_j))
  (c) Soft-margin LogSumExp: log(1 + sum_{j!=y} exp(z_j - z_y + m))

Sweep target margin m in {1, 5, 10} for (b) and (c).

Anti-masking checks:
  - FGSM-vs-PGD gap
  - Black-box transfer ASR (PGD from standard CE baseline)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import campaign.common as C

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "h394_margin_maximizing_loss_output.txt")

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


def multiclass_hinge_loss(logits, targets, margin):
    """Crammer-Singer style hinge: mean over wrong classes of relu(m - (z_y - z_j))."""
    n, C = logits.shape
    z_y = logits.gather(1, targets.unsqueeze(1))  # (n, 1)
    # mask out correct class
    wrong_mask = torch.ones_like(logits, dtype=torch.bool)
    wrong_mask.scatter_(1, targets.unsqueeze(1), False)
    diffs = z_y - logits  # (n, C); diff for correct vs each class
    diffs_wrong = diffs[wrong_mask].view(n, C - 1)
    loss = F.relu(margin - diffs_wrong).mean()
    return loss


def soft_margin_lse_loss(logits, targets, margin):
    """Soft-margin LogSumExp: log(1 + sum_{j!=y} exp(z_j - z_y + m))."""
    n, C = logits.shape
    z_y = logits.gather(1, targets.unsqueeze(1))  # (n, 1)
    wrong_mask = torch.ones_like(logits, dtype=torch.bool)
    wrong_mask.scatter_(1, targets.unsqueeze(1), False)
    z_wrong = logits[wrong_mask].view(n, C - 1)
    # exp(z_j - z_y + m) for each wrong class
    exponents = z_wrong - z_y + margin
    loss = torch.log1p(exponents.exp().sum(dim=1)).mean()
    return loss


def make_optimizer(model):
    return torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=5e-4
    )


def train_model(Xtr, Ytr, loss_type="ce", margin=1.0):
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
            z = model(xb)
            if loss_type == "ce":
                loss = F.cross_entropy(z, yb)
            elif loss_type == "hinge":
                loss = multiclass_hinge_loss(z, yb, margin)
            elif loss_type == "lse":
                loss = soft_margin_lse_loss(z, yb, margin)
            else:
                raise ValueError(loss_type)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_model(model, Xte, Yte, transfer_model=None):
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)

    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, fgsm_acc = C.logits_and_acc(model, Xfgsm, Yte)
    fgsm_asr = 1 - fgsm_acc

    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, pgd_acc = C.logits_and_acc(model, Xpgd, Yte)
    pgd_asr = 1 - pgd_acc

    mg = C.margin(model, Xte, Yte).mean()

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

    p("=" * 75)
    p("H394: Margin-Maximizing Loss")
    p("=" * 75)

    p("\nLoading Fashion-MNIST...")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)
    p(f"Train: {Xtr.shape}, Test: {Xte.shape}")

    # Standard CE baseline
    p("\nTraining CE baseline...")
    ce_model = train_model(Xtr, Ytr, loss_type="ce")
    ce_metrics = eval_model(ce_model, Xte, Yte, transfer_model=None)
    p(f"CE baseline: clean={ce_metrics['clean_acc']:.3f} "
      f"fgsm_asr={ce_metrics['fgsm_asr']:.3f} "
      f"pgd_asr={ce_metrics['pgd_asr']:.3f} "
      f"margin={ce_metrics['margin']:.3f}")

    margin_vals = [1, 5, 10]
    loss_types = ["hinge", "lse"]

    results = {"ce_baseline": ce_metrics}

    for loss_type in loss_types:
        for m_val in margin_vals:
            key = f"{loss_type}_m{m_val}"
            p(f"  Training {loss_type} margin={m_val}...")
            model = train_model(Xtr, Ytr, loss_type=loss_type, margin=m_val)
            metrics = eval_model(model, Xte, Yte, transfer_model=ce_model)
            results[key] = metrics
            p(f"  {key}: clean={metrics['clean_acc']:.3f} "
              f"fgsm_asr={metrics['fgsm_asr']:.3f} pgd_asr={metrics['pgd_asr']:.3f} "
              f"margin={metrics['margin']:.3f} transfer_asr={metrics['transfer_asr']:.3f} "
              f"gap={metrics['fgsm_pgd_gap']:.3f}")

    # Print results table
    p("\n" + "=" * 75)
    p("RESULTS TABLE")
    p("=" * 75)
    header = f"{'Condition':<20} {'clean':>7} {'fgsm_asr':>9} {'pgd_asr':>8} {'margin':>7} {'transfer':>9} {'gap':>6} {'masking?':>9}"
    p(header)
    p("-" * 75)

    # CE baseline row (no transfer model available)
    m = ce_metrics
    masking_ce = "YES" if m['fgsm_pgd_gap'] > 0.15 else "no"
    p(f"{'ce_baseline':<20} {m['clean_acc']:>7.3f} {m['fgsm_asr']:>9.3f} "
      f"{m['pgd_asr']:>8.3f} {m['margin']:>7.3f} {'n/a':>9} "
      f"{m['fgsm_pgd_gap']:>6.3f} {masking_ce:>9}")

    for loss_type in loss_types:
        for m_val in margin_vals:
            key = f"{loss_type}_m{m_val}"
            m = results[key]
            masking = "YES" if m['fgsm_pgd_gap'] > 0.15 or m['transfer_asr'] >= m['pgd_asr'] - 0.05 else "no"
            p(f"{key:<20} {m['clean_acc']:>7.3f} {m['fgsm_asr']:>9.3f} "
              f"{m['pgd_asr']:>8.3f} {m['margin']:>7.3f} {m['transfer_asr']:>9.3f} "
              f"{m['fgsm_pgd_gap']:>6.3f} {masking:>9}")

    p("\n" + "=" * 75)
    p("VERDICT")
    p("=" * 75)

    # Best by PGD ASR (excluding CE baseline to compare against it)
    non_ce_keys = [k for k in results if k != "ce_baseline"]
    best_key = min(non_ce_keys, key=lambda k: results[k]['pgd_asr'])
    best = results[best_key]
    baseline = ce_metrics

    pgd_reduction = baseline['pgd_asr'] - best['pgd_asr']
    margin_increase = best['margin'] - baseline['margin']
    is_masking = best['fgsm_pgd_gap'] > 0.15 or best['transfer_asr'] >= best['pgd_asr'] - 0.05

    p(f"Best config: {best_key}")
    p(f"PGD ASR: {baseline['pgd_asr']:.3f} (CE baseline) -> {best['pgd_asr']:.3f}, "
      f"reduction={pgd_reduction:.3f}")
    p(f"Mean margin: {baseline['margin']:.3f} (CE) -> {best['margin']:.3f}, "
      f"increase={margin_increase:.3f}")
    p(f"FGSM-vs-PGD gap (best): {best['fgsm_pgd_gap']:.3f} (>0.15 = masking)")
    p(f"Transfer ASR: {best['transfer_asr']:.3f} vs PGD ASR: {best['pgd_asr']:.3f}")

    if is_masking:
        p("\nVERDICT: MASKING — margin-maximizing loss reduces white-box FGSM/PGD ASR "
          "but transfer ASR stays high or FGSM-PGD gap is large, indicating gradient "
          "obfuscation rather than genuine robustness.")
    elif pgd_reduction > 0.05 and margin_increase > 0:
        p("\nVERDICT: GENUINE ROBUSTNESS — margin-maximizing loss raises measured margin "
          "and meaningfully reduces PGD ASR without masking indicators. "
          "Hypothesis confirmed: larger margin correlates with lower FGSM/PGD vulnerability.")
    elif pgd_reduction > 0.05:
        p("\nVERDICT: PARTIAL SUPPORT — PGD ASR reduced but margin did not increase "
          "as expected. Loss reshaping helps robustness but not via margin mechanism.")
    else:
        p("\nVERDICT: NO MEANINGFUL IMPROVEMENT — margin-maximizing loss does not "
          "significantly reduce adversarial vulnerability vs CE baseline.")

    output = "\n".join(lines)
    with open(OUTFILE, "w") as f:
        f.write(output)
    print(f"\nResults saved to {OUTFILE}")

if __name__ == "__main__":
    main()
