"""
H384: Noise Copy Multiplicity — does augmenting training with K noisy copies per clean
image improve adversarial robustness, and where does it saturate?

Fixed sigma=0.15, sweep K in {0, 1, 2, 4, 8, 16}.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import campaign.common as C

SEED      = 0
SIGMA     = 0.15
K_VALUES  = [0, 1, 2, 4, 8, 16]
EPOCHS    = 10
BATCH     = 128
LR        = 0.05
EPS       = 0.1
PGD_STEPS = 10
N_TRAIN   = 6000
N_EVAL    = 2000

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "fashion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "h384_noise_copy_multiplicity_output.txt")


def build_aug_set(Xtr, Ytr, K, sigma):
    """Return (Xaug, Yaug) = clean + K noisy copies each."""
    if K == 0:
        return Xtr, Ytr
    copies = [Xtr, Ytr]
    parts_x = [Xtr]
    parts_y = [Ytr]
    for _ in range(K):
        noise = sigma * torch.randn_like(Xtr)
        parts_x.append((Xtr + noise).clamp(0, 1))
        parts_y.append(Ytr)
    Xaug = torch.cat(parts_x, dim=0)
    Yaug = torch.cat(parts_y, dim=0)
    # shuffle
    perm = torch.randperm(Xaug.size(0), device=Xaug.device,
                          generator=torch.Generator(device=Xaug.device).manual_seed(SEED))
    return Xaug[perm], Yaug[perm]


def main():
    print("Loading Fashion-MNIST...")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    meta = C.dataset_meta("fashion_mnist")

    rows = []
    for K in K_VALUES:
        print(f"\n--- K={K} ---")
        C.set_seed(SEED)

        Xaug, Yaug = build_aug_set(Xtr, Ytr, K, SIGMA)
        train_size = Xaug.size(0)
        print(f"  train_size={train_size}")

        model = C.build_model("cnn", meta, width=32, seed=SEED)

        t0 = time.time()
        C.train_model(model, Xaug, Yaug, epochs=EPOCHS, batch=BATCH,
                      opt="sgd", lr=LR, ncls=10)
        elapsed = time.time() - t0

        _, clean_acc = C.logits_and_acc(model, Xte, Yte)
        fgsm_res = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
        pgd_res  = C.attack_success(model, Xte, Yte, attack="pgd",  eps=EPS, steps=PGD_STEPS)
        mean_margin = float(np.mean(C.margin(model, Xte, Yte)))

        fgsm_asr = fgsm_res["asr"]
        pgd_asr  = pgd_res["asr"]

        print(f"  clean_acc={clean_acc:.4f}  fgsm_asr={fgsm_asr:.4f}  "
              f"pgd_asr={pgd_asr:.4f}  margin={mean_margin:.4f}  time={elapsed:.1f}s")

        rows.append(dict(K=K, train_size=train_size,
                         clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                         pgd_asr=pgd_asr, margin=mean_margin, time_s=elapsed))
    # Saturation detection: find first K (after K=0) where *both* FGSM and PGD
    # marginal improvement vs the previous step are < 2pp, AND it isn't just a
    # transient blip (i.e. the next step also fails to improve by >=2pp).
    # We look at consecutive pairs and flag the first K where improvement stops.
    saturation_k = None
    for i in range(1, len(rows)):
        r_prev = rows[i - 1]
        r_cur  = rows[i]
        d_fgsm = (r_prev["fgsm_asr"] - r_cur["fgsm_asr"]) * 100
        d_pgd  = (r_prev["pgd_asr"]  - r_cur["pgd_asr"])  * 100
        if d_fgsm < 2.0 and d_pgd < 2.0:
            # check whether *any* later step gives >=2pp improvement on either
            future_improves = any(
                (rows[j - 1]["fgsm_asr"] - rows[j]["fgsm_asr"]) * 100 >= 2.0 or
                (rows[j - 1]["pgd_asr"]  - rows[j]["pgd_asr"])  * 100 >= 2.0
                for j in range(i + 1, len(rows))
            )
            if not future_improves:
                saturation_k = r_cur["K"]
                break

    # build report
    lines = []
    lines.append("=" * 70)
    lines.append("H384: Noise Copy Multiplicity — Fashion-MNIST")
    lines.append(f"sigma={SIGMA}, epochs={EPOCHS}, lr={LR}, batch={BATCH}, "
                 f"eps={EPS}, pgd_steps={PGD_STEPS}, seed={SEED}")
    lines.append("=" * 70)
    lines.append("")
    hdr = f"{'K':>4} | {'train_size':>10} | {'clean_acc':>9} | {'fgsm_asr':>8} | {'pgd_asr':>7} | {'margin':>7} | {'time_s':>7}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        lines.append(
            f"{r['K']:>4} | {r['train_size']:>10} | "
            f"{r['clean_acc']:>9.4f} | {r['fgsm_asr']:>8.4f} | "
            f"{r['pgd_asr']:>7.4f} | {r['margin']:>7.4f} | {r['time_s']:>7.1f}")
    lines.append("")
    lines.append("Analysis")
    lines.append("-" * 40)

    # find best K (lowest pgd_asr)
    best = min(rows, key=lambda r: r["pgd_asr"])
    base = rows[0]

    lines.append(f"Baseline (K=0):  fgsm_asr={base['fgsm_asr']:.4f}  pgd_asr={base['pgd_asr']:.4f}")
    lines.append(f"Best (K={best['K']}): fgsm_asr={best['fgsm_asr']:.4f}  pgd_asr={best['pgd_asr']:.4f}")
    lines.append(f"PGD ASR reduction vs baseline: {(base['pgd_asr']-best['pgd_asr'])*100:.1f} pp")

    if saturation_k is not None:
        lines.append(f"Saturation detected at K={saturation_k} "
                     f"(marginal improvement <2pp for both FGSM and PGD ASR)")
    else:
        lines.append("No saturation detected within the swept K range "
                     "(robustness keeps improving through K=16).")

    lines.append("")
    lines.append("Marginal improvements (ASR reduction vs previous K):")
    for i in range(1, len(rows)):
        r_prev = rows[i-1]
        r_cur  = rows[i]
        d_fgsm = (r_prev["fgsm_asr"] - r_cur["fgsm_asr"]) * 100
        d_pgd  = (r_prev["pgd_asr"]  - r_cur["pgd_asr"])  * 100
        lines.append(f"  K={r_cur['K']:>2} vs K={r_prev['K']:>2}: "
                     f"ΔFGSM={d_fgsm:+.2f}pp  ΔPGD={d_pgd:+.2f}pp")

    lines.append("")
    # one-line verdict
    if best["K"] > 0:
        verdict = (f"VERDICT: Noise-copy multiplicity (sigma={SIGMA}) helps — "
                   f"best at K={best['K']} with PGD ASR {base['pgd_asr']*100:.1f}% → "
                   f"{best['pgd_asr']*100:.1f}% ({(base['pgd_asr']-best['pgd_asr'])*100:.1f}pp drop); "
                   f"saturation at K={saturation_k if saturation_k else '>16'}.")
    else:
        verdict = "VERDICT: Noise-copy multiplicity provides no benefit over clean training at sigma=0.15."
    lines.append(verdict)
    lines.append("")

    report = "\n".join(lines)
    print("\n" + report)

    with open(OUT_FILE, "w") as f:
        f.write(report)
    print(f"Report saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
