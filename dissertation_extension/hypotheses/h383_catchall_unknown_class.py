"""
H383: Catch-all "unknown" class (class 10) for adversarial robustness.

Hypothesis: Adding an explicit class 10 ("I don't know") gives the model an
escape route during adversarial attacks, raising abstention_rate_adv >>
abstention_rate_clean if the hypothesis holds.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import campaign.common as C

import torch
import torch.nn.functional as F
import math

# ── config ──────────────────────────────────────────────────────────────────
N_TRAIN  = 6000
EPOCHS   = 10
LR       = 0.05
BATCH    = 128
SEED     = 0
EPS      = 0.1
PGD_STEPS = 10
OUT_PATH = os.path.join(os.path.dirname(__file__), "..",
                        "results", "fashion_mnist",
                        "h383_catchall_unknown_class_output.txt")

C.set_seed(SEED)


# ── structured noise: sine waves ────────────────────────────────────────────
def sine_noise(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    imgs = torch.zeros(n, 1, 28, 28)
    for i in range(n):
        fx = torch.randint(1, 8, (1,), generator=g).item()
        fy = torch.randint(1, 8, (1,), generator=g).item()
        xs = torch.linspace(0, 2 * math.pi * fx, 28)
        ys = torch.linspace(0, 2 * math.pi * fy, 28)
        grid = (torch.sin(xs).unsqueeze(0) + torch.sin(ys).unsqueeze(1)) / 2
        imgs[i, 0] = (grid + 1) / 2  # map to [0,1]
    return imgs.clamp(0, 1)


# ── build unknown images ─────────────────────────────────────────────────────
def make_unknown(kind, n, Xtr_cpu, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == "gaussian":
        return torch.randn(n, 1, 28, 28, generator=g).clamp(0, 1)
    elif kind == "uniform":
        return torch.rand(n, 1, 28, 28, generator=g)
    elif kind == "sine":
        return sine_noise(n, seed=seed)
    elif kind == "inverted":
        idx = torch.randint(0, Xtr_cpu.size(0), (n,), generator=g)
        return 1.0 - Xtr_cpu[idx]
    elif kind == "mixed":
        q = n // 4
        parts = [
            torch.randn(q, 1, 28, 28, generator=g).clamp(0, 1),
            torch.rand(q, 1, 28, 28, generator=g),
            sine_noise(q, seed=seed),
            1.0 - Xtr_cpu[torch.randint(0, Xtr_cpu.size(0), (n - 3*q,), generator=g)],
        ]
        return torch.cat(parts, dim=0)
    raise ValueError(kind)


# ── evaluation for 11-class model ───────────────────────────────────────────
@torch.no_grad()
def _preds(model, X, batch=256):
    parts = []
    for i in range(0, X.size(0), batch):
        parts.append(model(X[i:i+batch].to(C.DEVICE)).argmax(1).cpu())
    return torch.cat(parts)


def eval_11class(model, Xte, Yte, eps=0.1):
    model.eval()
    Xte_dev = Xte.to(C.DEVICE)
    Yte_cpu = Yte.cpu()

    # clean
    preds_clean = _preds(model, Xte_dev)
    correct_clean = (preds_clean == Yte_cpu) & (preds_clean < 10)
    clean_acc = correct_clean.float().mean().item()
    abstention_clean = (preds_clean == 10).float().mean().item()

    # FGSM
    Xfgsm = C.fgsm(model, Xte_dev, Yte.to(C.DEVICE), eps=eps)
    preds_fgsm = _preds(model, Xfgsm)
    fgsm_asr = (preds_fgsm != Yte_cpu).float().mean().item()
    abstention_fgsm = (preds_fgsm == 10).float().mean().item()

    # PGD
    Xpgd = C.pgd(model, Xte_dev, Yte.to(C.DEVICE), eps=eps, steps=PGD_STEPS)
    preds_pgd = _preds(model, Xpgd)
    pgd_asr = (preds_pgd != Yte_cpu).float().mean().item()
    abstention_pgd = (preds_pgd == 10).float().mean().item()

    return dict(clean_acc=clean_acc,
                fgsm_asr=fgsm_asr,
                pgd_asr=pgd_asr,
                abstention_clean=abstention_clean,
                abstention_fgsm=abstention_fgsm,
                abstention_pgd=abstention_pgd)


# ── evaluation for 10-class baseline ─────────────────────────────────────────
def eval_10class(model, Xte, Yte, eps=0.1):
    model.eval()
    Xte_dev = Xte.to(C.DEVICE)
    Yte_dev = Yte.to(C.DEVICE)

    with torch.no_grad():
        preds = model(Xte_dev).argmax(1).cpu()
    clean_acc = (preds == Yte.cpu()).float().mean().item()

    res_f = C.attack_success(model, Xte_dev, Yte_dev, attack="fgsm", eps=eps)
    res_p = C.attack_success(model, Xte_dev, Yte_dev, attack="pgd",  eps=eps, steps=PGD_STEPS)

    return dict(clean_acc=clean_acc,
                fgsm_asr=res_f["asr"],
                pgd_asr=res_p["asr"],
                abstention_clean=0.0,
                abstention_fgsm=0.0,
                abstention_pgd=0.0)


# ── run one condition ─────────────────────────────────────────────────────────
def run_condition(name, noise_kind, n_unknown, Xtr, Ytr, Xte, Yte):
    print(f"  [{name}] training ...", flush=True)
    C.set_seed(SEED)

    if noise_kind is None:
        # baseline 10-class
        meta = {"channels": 1, "size": 28, "n_classes": 10}
        model = C.build_model("cnn", meta, width=32, seed=SEED)
        C.train_model(model, Xtr, Ytr, epochs=EPOCHS, batch=BATCH,
                      opt="sgd", lr=LR, ncls=10)
        metrics = eval_10class(model, Xte, Yte, eps=EPS)
    else:
        meta = {"channels": 1, "size": 28, "n_classes": 11}
        model = C.build_model("cnn", meta, width=32, seed=SEED)

        Xtr_cpu = Xtr.cpu()
        Xunk = make_unknown(noise_kind, n_unknown, Xtr_cpu, seed=SEED)
        Yunk = torch.full((n_unknown,), 10, dtype=torch.long)

        Xall = torch.cat([Xtr_cpu, Xunk], dim=0).to(C.DEVICE)
        Yall = torch.cat([Ytr.cpu(), Yunk], dim=0).to(C.DEVICE)

        # shuffle
        perm = torch.randperm(Xall.size(0), generator=torch.Generator().manual_seed(SEED))
        Xall, Yall = Xall[perm], Yall[perm]

        C.train_model(model, Xall, Yall, epochs=EPOCHS, batch=BATCH,
                      opt="sgd", lr=LR, ncls=11)
        metrics = eval_11class(model, Xte, Yte, eps=EPS)

    return metrics


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading Fashion-MNIST ...", flush=True)
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=2000, seed=SEED)

    conditions = [
        ("baseline_10class",       None,         0),
        ("unknown_gaussian_1x",    "gaussian",   N_TRAIN),
        ("unknown_gaussian_ratio10","gaussian",  N_TRAIN // 10),
        ("unknown_uniform_1x",     "uniform",    N_TRAIN),
        ("unknown_inverted_1x",    "inverted",   N_TRAIN),
        ("unknown_mixed_1x",       "mixed",      N_TRAIN),
    ]

    results = {}
    for name, noise_kind, n_unknown in conditions:
        m = run_condition(name, noise_kind, n_unknown, Xtr, Ytr, Xte, Yte)
        results[name] = m
        print(f"  [{name}] clean_acc={m['clean_acc']:.3f} "
              f"fgsm_asr={m['fgsm_asr']:.3f} pgd_asr={m['pgd_asr']:.3f} "
              f"abstain_clean={m['abstention_clean']:.3f} "
              f"abstain_fgsm={m['abstention_fgsm']:.3f} "
              f"abstain_pgd={m['abstention_pgd']:.3f}", flush=True)

    # ── write output ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    lines = []
    lines.append("H383: Catch-all Unknown Class — Results")
    lines.append("=" * 60)
    lines.append(f"N_TRAIN={N_TRAIN}, EPOCHS={EPOCHS}, LR={LR}, EPS={EPS}, PGD_STEPS={PGD_STEPS}")
    lines.append("")
    header = f"{'Condition':<30} {'clean_acc':>9} {'fgsm_asr':>9} {'pgd_asr':>8} {'abs_clean':>10} {'abs_fgsm':>9} {'abs_pgd':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in results.items():
        lines.append(
            f"{name:<30} {m['clean_acc']:>9.3f} {m['fgsm_asr']:>9.3f} "
            f"{m['pgd_asr']:>8.3f} {m['abstention_clean']:>10.3f} "
            f"{m['abstention_fgsm']:>9.3f} {m['abstention_pgd']:>8.3f}"
        )
    lines.append("")
    lines.append("Key question: does abstention_adv >> abstention_clean?")
    lines.append("")
    for name, m in results.items():
        if m['abstention_clean'] == 0 and m['abstention_fgsm'] == 0:
            continue
        fgsm_ratio = (m['abstention_fgsm'] / m['abstention_clean']
                      if m['abstention_clean'] > 1e-6 else float('inf'))
        pgd_ratio  = (m['abstention_pgd']  / m['abstention_clean']
                      if m['abstention_clean'] > 1e-6 else float('inf'))
        lines.append(f"  {name}: fgsm_ratio={fgsm_ratio:.2f}x, pgd_ratio={pgd_ratio:.2f}x")

    report = "\n".join(lines)
    print("\n" + report)
    with open(OUT_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
